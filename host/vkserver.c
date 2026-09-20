/* vkserver.c -- persistent Vulkan compute server (resident-buffer aware).
 *
 * Keeps the device, queue, command pool, descriptor pool, fence and the
 * per-kernel pipelines alive across requests.  Adds resident GPU buffers
 * so weights can be uploaded ONCE and reused for every op, instead of
 * re-encoding + re-uploading them per dispatch (the inference bottleneck).
 *
 *   build: cc -O2 vkserver.c -lvulkan -o vkserver
 *   run:   ./vkserver KERNEL_DIR   (loads <dir>/<name>.spv on first use)
 *
 * Every request starts with u32 OPCODE:
 *   OP_RUN=0:
 *     u32 name_len; char name[name_len];
 *     u32 nbuf;
 *     per buffer: u32 size(float count); u32 kind; [u32 handle if kind==1]
 *        kind 0 = inline input (data sent, not returned)
 *        kind 1 = resident    (referenced by handle, not returned)
 *        kind 2 = output      (zeroed, returned)
 *        kind 3 = inout       (data sent, returned)  -- legacy generic run
 *     u32 npush; u32 push[npush];
 *     u32 groups;
 *     f32 inline[sum(size) over kind 0 and 3, in binding order];
 *   Response: u32 status; f32 out[sum(size) over kind 2 and 3, binding order].
 *
 *   OP_UPLOAD=1: u32 nfloats; f32 data[nfloats];
 *   Response: u32 status; u32 handle.
 *
 *   OP_FREE=2: u32 handle;
 *   Response: u32 status.
 *
 *   OP_UPLOAD_FILE=8: u32 pathlen; char path[pathlen]; u32 off_lo; u32 off_hi;
 *                     u32 nfloats;
 *   Response: u32 status; u32 handle.
 *   The bytes never cross the pipe.  Emacs `process-send-string' moves about
 *   3 MB/s to a pipe -- measured against 147 MB/s for the same string written
 *   to a file -- so shipping a 27 GB model through stdin costs hours that have
 *   nothing to do with the GPU.  Here the caller sends a path and the server
 *   reads the region straight into mapped device memory.
 */
#define _FILE_OFFSET_BITS 64
#include <sys/types.h>
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VK_CHECK(x) do{VkResult _r=(x);if(_r!=VK_SUCCESS){\
 fprintf(stderr,"vkserver VK %d %s:%d\n",_r,__FILE__,__LINE__);exit(2);}}while(0)

#define OP_RUN 0
#define OP_UPLOAD 1
#define OP_FREE 2
#define OP_BATCH 3
#define OP_COMPILE 4
#define OP_RUN_COMPILED 5
#define OP_FREE_COMPILED 6
#define OP_WRITE_RESIDENT 7
#define OP_UPLOAD_FILE 8

static VkInstance inst;
static VkPhysicalDevice pd;
static VkDevice dev;
static VkQueue queue;
static uint32_t cq;
static VkCommandPool cmdpool;
static VkCommandBuffer cmd;
static VkDescriptorPool descpool;
static VkFence fence;
static const char *kdir;

typedef struct {
  char name[64];
  VkShaderModule sm;
  VkDescriptorSetLayout dsl;
  VkPipelineLayout pl;
  VkPipeline pipe;
  int nbuf, npush;
} Kernel;
#define MAXKERNEL 256
static Kernel kcache[MAXKERNEL];
static int nkernel = 0;

typedef struct { VkBuffer buf; VkDeviceMemory mem; uint32_t size; int used; } Resident;
#define MAXRES 4096
static Resident res[MAXRES];
static int nres = 0;

/* A compiled batch: buffers + a pre-recorded command buffer that is re-submitted
 * each training step.  The graph (slots + dispatches) is fixed across steps;
 * only the resident weight buffers change in place, so we record the command
 * buffer (with per-step tmp zeroing via vkCmdFillBuffer) ONCE and just submit it,
 * eliminating per-step protocol re-send, buffer alloc and descriptor rebuild. */
#define MAXCOMP 16
typedef struct {
  int used; uint32_t nslot;
  VkBuffer *sbuf; VkDeviceMemory *smem; uint32_t *ssize; int *stmp;
  uint32_t nout; uint32_t *outidx;
  VkCommandBuffer cmd; VkDescriptorPool pool;
} Compiled;
static Compiled comp[MAXCOMP];

static int rd_u32(uint32_t *v){ return fread(v,4,1,stdin)==1; }
static void wr_u32(uint32_t v){ fwrite(&v,4,1,stdout); }

static uint32_t mtype(uint32_t bits, VkMemoryPropertyFlags w){
  VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd,&mp);
  for(uint32_t i=0;i<mp.memoryTypeCount;i++)
    if((bits&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&w)==w) return i;
  fprintf(stderr,"vkserver: no mem type\n"); exit(2);
}

static void make_buffer(uint32_t nfloats, VkBuffer *buf, VkDeviceMemory *mem){
  VkDeviceSize bs=(VkDeviceSize)nfloats*4; if(bs==0)bs=4;
  VkBufferCreateInfo bi={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size=bs;
  bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(dev,&bi,NULL,buf));
  VkMemoryRequirements r; vkGetBufferMemoryRequirements(dev,*buf,&r);
  VkMemoryAllocateInfo ai={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize=r.size;
  ai.memoryTypeIndex=mtype(r.memoryTypeBits,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VK_CHECK(vkAllocateMemory(dev,&ai,NULL,mem)); VK_CHECK(vkBindBufferMemory(dev,*buf,*mem,0));
}

static void write_buffer(VkDeviceMemory mem, uint32_t nfloats, const void *data){
  if(nfloats==0) return;
  void *p; VK_CHECK(vkMapMemory(dev,mem,0,(VkDeviceSize)nfloats*4,0,&p));
  memcpy(p,data,(size_t)nfloats*4); vkUnmapMemory(dev,mem);
}

static void init_vulkan(void){
  VkApplicationInfo app={VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion=VK_API_VERSION_1_1;
  VkInstanceCreateInfo ici={VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&app;
  VK_CHECK(vkCreateInstance(&ici,NULL,&inst));
  uint32_t nd=0; VK_CHECK(vkEnumeratePhysicalDevices(inst,&nd,NULL));
  if(!nd){fprintf(stderr,"vkserver: no device\n");exit(2);}
  VkPhysicalDevice *ds=malloc(nd*sizeof(*ds)); VK_CHECK(vkEnumeratePhysicalDevices(inst,&nd,ds));
  int pick=0; const char*e=getenv("NELISP_GPU_DEVICE");
  if(e) pick=atoi(e); else for(uint32_t i=0;i<nd;i++){VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(ds[i],&p); if(p.deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU){pick=i;break;}}
  if(pick<0||pick>=(int)nd)pick=0; pd=ds[pick];
  VkPhysicalDeviceProperties pr; vkGetPhysicalDeviceProperties(pd,&pr);
  fprintf(stderr,"vkserver: device = %s\n",pr.deviceName);
  uint32_t nq=0; vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,NULL);
  VkQueueFamilyProperties *qf=malloc(nq*sizeof(*qf)); vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,qf);
  cq=0; for(uint32_t i=0;i<nq;i++) if(qf[i].queueFlags&VK_QUEUE_COMPUTE_BIT){cq=i;break;}
  float prio=1; VkDeviceQueueCreateInfo qci={VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex=cq; qci.queueCount=1; qci.pQueuePriorities=&prio;
  VkDeviceCreateInfo dci={VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci;
  VK_CHECK(vkCreateDevice(pd,&dci,NULL,&dev)); vkGetDeviceQueue(dev,cq,0,&queue);
  VkCommandPoolCreateInfo ci={VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  ci.queueFamilyIndex=cq; ci.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VK_CHECK(vkCreateCommandPool(dev,&ci,NULL,&cmdpool));
  VkCommandBufferAllocateInfo ba={VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ba.commandPool=cmdpool; ba.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ba.commandBufferCount=1;
  VK_CHECK(vkAllocateCommandBuffers(dev,&ba,&cmd));
  VkDescriptorPoolSize ps={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,65536};
  VkDescriptorPoolCreateInfo dpi={VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpi.maxSets=8192; dpi.poolSizeCount=1; dpi.pPoolSizes=&ps;
  VK_CHECK(vkCreateDescriptorPool(dev,&dpi,NULL,&descpool));
  VkFenceCreateInfo fi={VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VK_CHECK(vkCreateFence(dev,&fi,NULL,&fence));
}

static Kernel *get_kernel(const char *name, int nbuf, int npush){
  for(int i=0;i<nkernel;i++) if(!strcmp(kcache[i].name,name)) return &kcache[i];
  if(nkernel>=MAXKERNEL){fprintf(stderr,"kernel cache full (%d)\n",MAXKERNEL);exit(2);}
  Kernel *k=&kcache[nkernel++]; strncpy(k->name,name,63); k->nbuf=nbuf; k->npush=npush;
  char path[512]; snprintf(path,sizeof(path),"%s/%s.spv",kdir,name);
  FILE *f=fopen(path,"rb"); if(!f){fprintf(stderr,"vkserver: open %s\n",path);exit(2);}
  fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
  uint32_t *spv=malloc(sz); if(fread(spv,1,sz,f)!=(size_t)sz)exit(2); fclose(f);
  VkShaderModuleCreateInfo smci={VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smci.codeSize=sz; smci.pCode=spv;
  VK_CHECK(vkCreateShaderModule(dev,&smci,NULL,&k->sm)); free(spv);
  VkDescriptorSetLayoutBinding bd[16];
  for(int i=0;i<nbuf;i++){bd[i]=(VkDescriptorSetLayoutBinding){0}; bd[i].binding=i; bd[i].descriptorCount=1;
    bd[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bd[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
  VkDescriptorSetLayoutCreateInfo dl={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dl.bindingCount=nbuf; dl.pBindings=bd; VK_CHECK(vkCreateDescriptorSetLayout(dev,&dl,NULL,&k->dsl));
  VkPushConstantRange pcr={VK_SHADER_STAGE_COMPUTE_BIT,0,(uint32_t)(npush*4)};
  VkPipelineLayoutCreateInfo pli={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pli.setLayoutCount=1; pli.pSetLayouts=&k->dsl;
  if(npush>0){pli.pushConstantRangeCount=1; pli.pPushConstantRanges=&pcr;}
  VK_CHECK(vkCreatePipelineLayout(dev,&pli,NULL,&k->pl));
  VkComputePipelineCreateInfo cp={VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cp.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; cp.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;
  cp.stage.module=k->sm; cp.stage.pName="main"; cp.layout=k->pl;
  VK_CHECK(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&cp,NULL,&k->pipe));
  return k;
}

static void handle_run(void){
  uint32_t name_len; if(!rd_u32(&name_len)) exit(0);
  char name[64]={0}; if(name_len>=64){fprintf(stderr,"name too long\n");exit(2);}
  if(fread(name,1,name_len,stdin)!=name_len) exit(0);
  uint32_t nbuf; if(!rd_u32(&nbuf)) exit(0);
  uint32_t sizes[16], kind[16], handle[16];
  for(uint32_t i=0;i<nbuf;i++){ rd_u32(&sizes[i]); rd_u32(&kind[i]); handle[i]=0;
    if(kind[i]==1) rd_u32(&handle[i]); }
  uint32_t npush; rd_u32(&npush);
  uint32_t push[16]; for(uint32_t i=0;i<npush;i++) rd_u32(&push[i]);
  uint32_t groups; rd_u32(&groups);

  VkBuffer vbuf[16]; VkDeviceMemory vmem[16]; int temp[16];
  for(uint32_t i=0;i<nbuf;i++){
    if(kind[i]==1){ vbuf[i]=res[handle[i]].buf; vmem[i]=res[handle[i]].mem; temp[i]=0; continue; }
    make_buffer(sizes[i],&vbuf[i],&vmem[i]); temp[i]=1;
    if(kind[i]==0 || kind[i]==3){
      float *tmp=malloc((sizes[i]?sizes[i]:1)*4);
      if(sizes[i] && fread(tmp,4,sizes[i],stdin)!=sizes[i]){ free(tmp); exit(0); }
      write_buffer(vmem[i],sizes[i],tmp); free(tmp);
    } else { /* kind 2 output: zero */
      if(sizes[i]){ void*p; VK_CHECK(vkMapMemory(dev,vmem[i],0,(VkDeviceSize)sizes[i]*4,0,&p));
        memset(p,0,(size_t)sizes[i]*4); vkUnmapMemory(dev,vmem[i]); }
    }
  }

  Kernel *k=get_kernel(name,(int)nbuf,(int)npush);
  VK_CHECK(vkResetDescriptorPool(dev,descpool,0));
  VkDescriptorSetAllocateInfo da={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  da.descriptorPool=descpool; da.descriptorSetCount=1; da.pSetLayouts=&k->dsl;
  VkDescriptorSet dset; VK_CHECK(vkAllocateDescriptorSets(dev,&da,&dset));
  VkDescriptorBufferInfo bi2[16]; VkWriteDescriptorSet wr[16];
  for(uint32_t i=0;i<nbuf;i++){bi2[i]=(VkDescriptorBufferInfo){vbuf[i],0,VK_WHOLE_SIZE};
    wr[i]=(VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; wr[i].dstSet=dset;
    wr[i].dstBinding=i; wr[i].descriptorCount=1; wr[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr[i].pBufferInfo=&bi2[i];}
  vkUpdateDescriptorSets(dev,nbuf,wr,0,NULL);

  VK_CHECK(vkResetCommandBuffer(cmd,0));
  VkCommandBufferBeginInfo cb={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  cb.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; VK_CHECK(vkBeginCommandBuffer(cmd,&cb));
  vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,k->pipe);
  vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,k->pl,0,1,&dset,0,NULL);
  if(npush>0) vkCmdPushConstants(cmd,k->pl,VK_SHADER_STAGE_COMPUTE_BIT,0,npush*4,push);
  vkCmdDispatch(cmd,groups,1,1); VK_CHECK(vkEndCommandBuffer(cmd));
  VK_CHECK(vkResetFences(dev,1,&fence));
  VkSubmitInfo si={VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cmd;
  VK_CHECK(vkQueueSubmit(queue,1,&si,fence));
  VK_CHECK(vkWaitForFences(dev,1,&fence,VK_TRUE,(uint64_t)6e10));

  wr_u32(0);
  for(uint32_t i=0;i<nbuf;i++){
    if(kind[i]==2 || kind[i]==3){
      if(sizes[i]){ void*p; VK_CHECK(vkMapMemory(dev,vmem[i],0,(VkDeviceSize)sizes[i]*4,0,&p));
        fwrite(p,4,sizes[i],stdout); vkUnmapMemory(dev,vmem[i]); }
    }
  }
  fflush(stdout);

  for(uint32_t i=0;i<nbuf;i++) if(temp[i]){ vkDestroyBuffer(dev,vbuf[i],NULL); vkFreeMemory(dev,vmem[i],NULL); }
}

static void handle_upload(void){
  uint32_t n; if(!rd_u32(&n)) exit(0);
  float *data=malloc((n?n:1)*4);
  if(n && fread(data,4,n,stdin)!=n){ free(data); exit(0); }
  int h=-1;
  for(int i=0;i<nres;i++) if(!res[i].used){ h=i; break; }   /* reuse a freed slot */
  if(h<0){
    if(nres>=MAXRES){ fprintf(stderr,"vkserver: resident table full\n"); free(data); wr_u32(1); wr_u32(0); fflush(stdout); return; }
    h=nres++;
  }
  make_buffer(n,&res[h].buf,&res[h].mem); res[h].size=n; res[h].used=1;
  write_buffer(res[h].mem,n,data); free(data);
  wr_u32(0); wr_u32((uint32_t)h); fflush(stdout);
}

/* OP_WRITE_RESIDENT: u32 handle; u32 nfloats; f32 data[].  Overwrite an existing
 * resident buffer's contents in place (e.g. a new training window's one-hot
 * inputs) so a compiled batch can be re-run on fresh data without recompiling. */
static void handle_write_resident(void){
  uint32_t h, n; if(!rd_u32(&h)) exit(0); if(!rd_u32(&n)) exit(0);
  float *data=malloc((n?n:1)*4);
  if(n && fread(data,4,n,stdin)!=n){ free(data); exit(0); }
  if(h<MAXRES && res[h].used){ write_buffer(res[h].mem,n,data); wr_u32(0); }
  else { fprintf(stderr,"vkserver: write bad resident %u\n",h); wr_u32(1); }
  free(data); fflush(stdout);
}

/* OP_UPLOAD_FILE: read NFLOATS*4 bytes at OFF of PATH straight into a fresh
 * resident buffer's mapped memory.  No host copy and no pipe traffic. */
static void handle_upload_file(void){
  uint32_t plen, olo, ohi, n;
  if(!rd_u32(&plen)) exit(0);
  if(plen==0||plen>4095){ fprintf(stderr,"vkserver: bad path length %u\n",plen); exit(0); }
  char path[4096];
  if(fread(path,1,plen,stdin)!=plen) exit(0);
  path[plen]=0;
  if(!rd_u32(&olo)) exit(0);
  if(!rd_u32(&ohi)) exit(0);
  if(!rd_u32(&n)) exit(0);
  uint64_t off=((uint64_t)ohi<<32)|olo;
  FILE *f=fopen(path,"rb");
  if(!f){ fprintf(stderr,"vkserver: cannot open %s\n",path); wr_u32(1); wr_u32(0); fflush(stdout); return; }
  if(fseeko(f,(off_t)off,SEEK_SET)!=0){ fprintf(stderr,"vkserver: seek %s\n",path); fclose(f); wr_u32(1); wr_u32(0); fflush(stdout); return; }
  int h=-1;
  for(int i=0;i<nres;i++) if(!res[i].used){ h=i; break; }
  if(h<0){
    if(nres>=MAXRES){ fprintf(stderr,"vkserver: resident table full\n"); fclose(f); wr_u32(1); wr_u32(0); fflush(stdout); return; }
    h=nres++;
  }
  make_buffer(n,&res[h].buf,&res[h].mem); res[h].size=n; res[h].used=1;
  if(n){
    void *p; VK_CHECK(vkMapMemory(dev,res[h].mem,0,(VkDeviceSize)n*4,0,&p));
    size_t want=(size_t)n*4, got=fread(p,1,want,f);
    vkUnmapMemory(dev,res[h].mem);
    if(got!=want){
      fprintf(stderr,"vkserver: %s short read %zu of %zu\n",path,got,want);
      vkDestroyBuffer(dev,res[h].buf,NULL); vkFreeMemory(dev,res[h].mem,NULL);
      res[h].used=0; fclose(f); wr_u32(1); wr_u32(0); fflush(stdout); return;
    }
  }
  fclose(f);
  wr_u32(0); wr_u32((uint32_t)h); fflush(stdout);
}

static void handle_free(void){
  uint32_t h; if(!rd_u32(&h)) exit(0);
  if(h<(uint32_t)nres && res[h].used){
    vkDestroyBuffer(dev,res[h].buf,NULL); vkFreeMemory(dev,res[h].mem,NULL); res[h].used=0;
  }
  wr_u32(0); fflush(stdout);
}

/* OP_BATCH: run a sequence of dispatches in ONE command buffer with a
 * memory barrier between them, so intermediate tensors stay resident on
 * the GPU instead of round-tripping to the host between ops (op fusion /
 * deferred GPU graph).
 *   u32 nslot;
 *     per slot: u32 kind; u32 size; [u32 handle if kind==1]
 *        kind 0=inline input, 1=resident, 2=transient(zeroed), 3=output(returned)
 *   u32 ndisp;
 *     per dispatch: u32 name_len; char name[]; u32 nbuf; u32 slot_idx[nbuf];
 *                   u32 npush; u32 push[npush]; u32 groups;
 *   f32 inline[ for kind 0 slots, in slot order ];
 * Response: u32 status; f32 out[ for kind 3 slots, in slot order ].
 */
typedef struct { char name[64]; uint32_t nbuf, slot[16], npush, push[16], groups; } Disp;

#define MAXSLOT 8192
#define MAXDISP 8192
static void handle_batch(void){
  uint32_t nslot; if(!rd_u32(&nslot)) exit(0);
  if(nslot>MAXSLOT){ fprintf(stderr,"vkserver: nslot %u > %d\n",nslot,MAXSLOT); exit(2); }
  static uint32_t skind[MAXSLOT], ssize[MAXSLOT], shandle[MAXSLOT];
  for(uint32_t i=0;i<nslot;i++){ rd_u32(&skind[i]); rd_u32(&ssize[i]); shandle[i]=0;
    if(skind[i]==1) rd_u32(&shandle[i]); }
  uint32_t ndisp; if(!rd_u32(&ndisp)) exit(0);
  if(ndisp>MAXDISP){ fprintf(stderr,"vkserver: ndisp %u > %d\n",ndisp,MAXDISP); exit(2); }
  static Disp disp[MAXDISP];
  for(uint32_t d=0;d<ndisp;d++){
    uint32_t nl; if(!rd_u32(&nl)||nl>=64) exit(2);
    memset(disp[d].name,0,64); if(fread(disp[d].name,1,nl,stdin)!=nl) exit(0);
    rd_u32(&disp[d].nbuf); for(uint32_t i=0;i<disp[d].nbuf;i++) rd_u32(&disp[d].slot[i]);
    rd_u32(&disp[d].npush); for(uint32_t i=0;i<disp[d].npush;i++) rd_u32(&disp[d].push[i]);
    rd_u32(&disp[d].groups);
  }
  static VkBuffer sbuf[MAXSLOT]; static VkDeviceMemory smem[MAXSLOT]; static int stmp[MAXSLOT];
  for(uint32_t i=0;i<nslot;i++){
    if(skind[i]==1){ sbuf[i]=res[shandle[i]].buf; smem[i]=res[shandle[i]].mem; stmp[i]=0; continue; }
    make_buffer(ssize[i],&sbuf[i],&smem[i]); stmp[i]=1;
    if(skind[i]==0){
      float *tmp=malloc((ssize[i]?ssize[i]:1)*4);
      if(ssize[i] && fread(tmp,4,ssize[i],stdin)!=ssize[i]){ free(tmp); exit(0); }
      write_buffer(smem[i],ssize[i],tmp); free(tmp);
    } else if(ssize[i]){ void*p; VK_CHECK(vkMapMemory(dev,smem[i],0,(VkDeviceSize)ssize[i]*4,0,&p));
      memset(p,0,(size_t)ssize[i]*4); vkUnmapMemory(dev,smem[i]); }
  }

  VK_CHECK(vkResetDescriptorPool(dev,descpool,0));
  VK_CHECK(vkResetCommandBuffer(cmd,0));
  VkCommandBufferBeginInfo cb={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  cb.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; VK_CHECK(vkBeginCommandBuffer(cmd,&cb));
  for(uint32_t d=0;d<ndisp;d++){
    Kernel *k=get_kernel(disp[d].name,(int)disp[d].nbuf,(int)disp[d].npush);
    VkDescriptorSetAllocateInfo da={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    da.descriptorPool=descpool; da.descriptorSetCount=1; da.pSetLayouts=&k->dsl;
    VkDescriptorSet dset; VK_CHECK(vkAllocateDescriptorSets(dev,&da,&dset));
    VkDescriptorBufferInfo bi2[16]; VkWriteDescriptorSet wr[16];
    for(uint32_t i=0;i<disp[d].nbuf;i++){ uint32_t s=disp[d].slot[i];
      bi2[i]=(VkDescriptorBufferInfo){sbuf[s],0,VK_WHOLE_SIZE};
      wr[i]=(VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; wr[i].dstSet=dset;
      wr[i].dstBinding=i; wr[i].descriptorCount=1; wr[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      wr[i].pBufferInfo=&bi2[i]; }
    vkUpdateDescriptorSets(dev,disp[d].nbuf,wr,0,NULL);
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,k->pipe);
    vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,k->pl,0,1,&dset,0,NULL);
    if(disp[d].npush>0) vkCmdPushConstants(cmd,k->pl,VK_SHADER_STAGE_COMPUTE_BIT,0,disp[d].npush*4,disp[d].push);
    vkCmdDispatch(cmd,disp[d].groups,1,1);
    if(d+1<ndisp){
      VkMemoryBarrier mb={VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
      mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0,1,&mb,0,NULL,0,NULL);
    }
  }
  VK_CHECK(vkEndCommandBuffer(cmd));
  VK_CHECK(vkResetFences(dev,1,&fence));
  VkSubmitInfo si={VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cmd;
  VK_CHECK(vkQueueSubmit(queue,1,&si,fence));
  VK_CHECK(vkWaitForFences(dev,1,&fence,VK_TRUE,(uint64_t)6e10));

  wr_u32(0);
  for(uint32_t i=0;i<nslot;i++) if(skind[i]==3 && ssize[i]){
    void*p; VK_CHECK(vkMapMemory(dev,smem[i],0,(VkDeviceSize)ssize[i]*4,0,&p));
    fwrite(p,4,ssize[i],stdout); vkUnmapMemory(dev,smem[i]); }
  fflush(stdout);

  for(uint32_t i=0;i<nslot;i++) if(stmp[i]){ vkDestroyBuffer(dev,sbuf[i],NULL); vkFreeMemory(dev,smem[i],NULL); }
}

/* OP_COMPILE: same wire format as OP_BATCH, but instead of running once we
 * allocate persistent buffers, record the command buffer, and return a handle.
 * Response: u32 status; u32 handle. */
static void handle_compile(void){
  uint32_t nslot; if(!rd_u32(&nslot)) exit(0);
  if(nslot>MAXSLOT){ fprintf(stderr,"vkserver: compile nslot %u\n",nslot); exit(2); }
  static uint32_t skind[MAXSLOT], ssize[MAXSLOT], shandle[MAXSLOT];
  for(uint32_t i=0;i<nslot;i++){ rd_u32(&skind[i]); rd_u32(&ssize[i]); shandle[i]=0;
    if(skind[i]==1) rd_u32(&shandle[i]); }
  uint32_t ndisp; if(!rd_u32(&ndisp)) exit(0);
  if(ndisp>MAXDISP){ fprintf(stderr,"vkserver: compile ndisp %u\n",ndisp); exit(2); }
  static Disp disp[MAXDISP];
  for(uint32_t d=0;d<ndisp;d++){
    uint32_t nl; if(!rd_u32(&nl)||nl>=64) exit(2);
    memset(disp[d].name,0,64); if(fread(disp[d].name,1,nl,stdin)!=nl) exit(0);
    rd_u32(&disp[d].nbuf); for(uint32_t i=0;i<disp[d].nbuf;i++) rd_u32(&disp[d].slot[i]);
    rd_u32(&disp[d].npush); for(uint32_t i=0;i<disp[d].npush;i++) rd_u32(&disp[d].push[i]);
    rd_u32(&disp[d].groups);
  }
  /* find a free compiled slot */
  int c=-1; for(int i=0;i<MAXCOMP;i++) if(!comp[i].used){c=i;break;}
  if(c<0){ fprintf(stderr,"vkserver: compiled table full\n"); wr_u32(1); wr_u32(0); fflush(stdout); return; }
  Compiled *C=&comp[c]; C->used=1; C->nslot=nslot; C->nout=0;
  C->sbuf=calloc(nslot,sizeof(VkBuffer)); C->smem=calloc(nslot,sizeof(VkDeviceMemory));
  C->ssize=calloc(nslot,sizeof(uint32_t)); C->stmp=calloc(nslot,sizeof(int));
  C->outidx=calloc(nslot,sizeof(uint32_t));
  int *fill=calloc(nslot,sizeof(int));
  for(uint32_t i=0;i<nslot;i++){ C->ssize[i]=ssize[i];
    if(skind[i]==1){ C->sbuf[i]=res[shandle[i]].buf; C->smem[i]=res[shandle[i]].mem; C->stmp[i]=0; fill[i]=0; continue; }
    make_buffer(ssize[i],&C->sbuf[i],&C->smem[i]); C->stmp[i]=1;
    if(skind[i]==0){ float *tmp=malloc((ssize[i]?ssize[i]:1)*4);
      if(ssize[i] && fread(tmp,4,ssize[i],stdin)!=ssize[i]){ free(tmp); exit(0); }
      write_buffer(C->smem[i],ssize[i],tmp); free(tmp); fill[i]=0; }
    else { fill[i]=1; if(skind[i]==3) C->outidx[C->nout++]=i; }  /* tmp/out zeroed each run */
  }
  /* per-batch descriptor pool sized to this graph */
  uint32_t totaldesc=0; for(uint32_t d=0;d<ndisp;d++) totaldesc+=disp[d].nbuf;
  VkDescriptorPoolSize ps={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, totaldesc?totaldesc:1};
  VkDescriptorPoolCreateInfo dpi={VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpi.maxSets=ndisp?ndisp:1; dpi.poolSizeCount=1; dpi.pPoolSizes=&ps;
  VK_CHECK(vkCreateDescriptorPool(dev,&dpi,NULL,&C->pool));
  VkCommandBufferAllocateInfo ba={VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ba.commandPool=cmdpool; ba.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ba.commandBufferCount=1;
  VK_CHECK(vkAllocateCommandBuffers(dev,&ba,&C->cmd));
  VkCommandBufferBeginInfo cb={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};  /* re-submittable */
  VK_CHECK(vkBeginCommandBuffer(C->cmd,&cb));
  /* zero all tmp/out buffers on-GPU each run */
  for(uint32_t i=0;i<nslot;i++) if(fill[i] && ssize[i])
    vkCmdFillBuffer(C->cmd,C->sbuf[i],0,(VkDeviceSize)ssize[i]*4,0);
  { VkMemoryBarrier mb={VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(C->cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,1,&mb,0,NULL,0,NULL); }
  for(uint32_t d=0;d<ndisp;d++){
    Kernel *k=get_kernel(disp[d].name,(int)disp[d].nbuf,(int)disp[d].npush);
    VkDescriptorSetAllocateInfo da={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    da.descriptorPool=C->pool; da.descriptorSetCount=1; da.pSetLayouts=&k->dsl;
    VkDescriptorSet dset; VK_CHECK(vkAllocateDescriptorSets(dev,&da,&dset));
    VkDescriptorBufferInfo bi2[16]; VkWriteDescriptorSet wr[16];
    for(uint32_t i=0;i<disp[d].nbuf;i++){ uint32_t s=disp[d].slot[i];
      bi2[i]=(VkDescriptorBufferInfo){C->sbuf[s],0,VK_WHOLE_SIZE};
      wr[i]=(VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; wr[i].dstSet=dset;
      wr[i].dstBinding=i; wr[i].descriptorCount=1; wr[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      wr[i].pBufferInfo=&bi2[i]; }
    vkUpdateDescriptorSets(dev,disp[d].nbuf,wr,0,NULL);
    vkCmdBindPipeline(C->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,k->pipe);
    vkCmdBindDescriptorSets(C->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,k->pl,0,1,&dset,0,NULL);
    if(disp[d].npush>0) vkCmdPushConstants(C->cmd,k->pl,VK_SHADER_STAGE_COMPUTE_BIT,0,disp[d].npush*4,disp[d].push);
    vkCmdDispatch(C->cmd,disp[d].groups,1,1);
    if(d+1<ndisp){ VkMemoryBarrier mb={VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
      mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(C->cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0,1,&mb,0,NULL,0,NULL); }
  }
  VK_CHECK(vkEndCommandBuffer(C->cmd));
  free(fill);
  wr_u32(0); wr_u32((uint32_t)c); fflush(stdout);
}

/* OP_RUN_COMPILED: u32 handle.  Re-submit the pre-recorded command buffer.
 * Response: u32 status; f32 out[ over kind-3 slots, in slot order ]. */
static void handle_run_compiled(void){
  uint32_t h; if(!rd_u32(&h)) exit(0);
  if(h>=MAXCOMP || !comp[h].used){ fprintf(stderr,"vkserver: bad compiled %u\n",h); wr_u32(1); fflush(stdout); return; }
  Compiled *C=&comp[h];
  VK_CHECK(vkResetFences(dev,1,&fence));
  VkSubmitInfo si={VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&C->cmd;
  VK_CHECK(vkQueueSubmit(queue,1,&si,fence));
  VK_CHECK(vkWaitForFences(dev,1,&fence,VK_TRUE,(uint64_t)6e10));
  wr_u32(0);
  for(uint32_t o=0;o<C->nout;o++){ uint32_t i=C->outidx[o]; if(!C->ssize[i]) continue;
    void*p; VK_CHECK(vkMapMemory(dev,C->smem[i],0,(VkDeviceSize)C->ssize[i]*4,0,&p));
    fwrite(p,4,C->ssize[i],stdout); vkUnmapMemory(dev,C->smem[i]); }
  fflush(stdout);
}

static void handle_free_compiled(void){
  uint32_t h; if(!rd_u32(&h)) exit(0);
  if(h<MAXCOMP && comp[h].used){ Compiled *C=&comp[h];
    for(uint32_t i=0;i<C->nslot;i++) if(C->stmp[i]){ vkDestroyBuffer(dev,C->sbuf[i],NULL); vkFreeMemory(dev,C->smem[i],NULL); }
    vkDestroyDescriptorPool(dev,C->pool,NULL);
    vkFreeCommandBuffers(dev,cmdpool,1,&C->cmd);
    free(C->sbuf); free(C->smem); free(C->ssize); free(C->stmp); free(C->outidx);
    C->used=0; }
  wr_u32(0); fflush(stdout);
}

int main(int argc,char**argv){
  if(argc<2){fprintf(stderr,"usage: %s KERNEL_DIR\n",argv[0]);return 1;}
  kdir=argv[1];
  init_vulkan();
  fprintf(stderr,"vkserver: ready\n"); fflush(stderr);
  uint32_t op;
  while(rd_u32(&op)){
    if(op==OP_RUN) handle_run();
    else if(op==OP_UPLOAD) handle_upload();
    else if(op==OP_FREE) handle_free();
    else if(op==OP_BATCH) handle_batch();
    else if(op==OP_COMPILE) handle_compile();
    else if(op==OP_RUN_COMPILED) handle_run_compiled();
    else if(op==OP_FREE_COMPILED) handle_free_compiled();
    else if(op==OP_WRITE_RESIDENT) handle_write_resident();
    else if(op==OP_UPLOAD_FILE) handle_upload_file();
    else { fprintf(stderr,"vkserver: bad opcode %u\n",op); return 2; }
  }
  return 0;
}

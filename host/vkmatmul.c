/* vkmatmul.c -- Vulkan compute reference host for matmul (C = A*B).
 *
 * Reference runtime to verify + time the DSL-compiled matmul SPIR-V
 * kernel on the GPU.  A is M x K, B is K x N, C is M x N (row-major
 * float).  Push constants: uint M, K, N.  Dispatch one thread per
 * output element (local_size_x = 64).
 *
 *   build: cc -O2 vkmatmul.c -lvulkan -o vkmatmul
 *   run:   ./vkmatmul kernels/matmul.spv [M K N]
 */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define VK_CHECK(x) do { VkResult _r=(x); if(_r!=VK_SUCCESS){ \
  fprintf(stderr,"VK error %d at %s:%d\n",_r,__FILE__,__LINE__); exit(2);} } while(0)

static uint32_t *read_spirv(const char *p, size_t *n){
  FILE *f=fopen(p,"rb"); if(!f){fprintf(stderr,"open %s\n",p);exit(2);}
  fseek(f,0,SEEK_END); long s=ftell(f); fseek(f,0,SEEK_SET);
  if(s<=0||s%4){fprintf(stderr,"bad spv\n");exit(2);}
  uint32_t *b=malloc(s); if(fread(b,1,s,f)!=(size_t)s){exit(2);} fclose(f);
  *n=s; return b;
}
static uint32_t mtype(VkPhysicalDevice pd,uint32_t bits,VkMemoryPropertyFlags w){
  VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd,&mp);
  for(uint32_t i=0;i<mp.memoryTypeCount;i++)
    if((bits&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&w)==w) return i;
  fprintf(stderr,"no mem type\n"); exit(2);
}
typedef struct{VkBuffer buf;VkDeviceMemory mem;} Buf;
static Buf mkbuf(VkPhysicalDevice pd,VkDevice d,VkDeviceSize sz){
  Buf b; VkBufferCreateInfo bi={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size=sz; bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(d,&bi,NULL,&b.buf));
  VkMemoryRequirements r; vkGetBufferMemoryRequirements(d,b.buf,&r);
  VkMemoryAllocateInfo ai={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize=r.size;
  ai.memoryTypeIndex=mtype(pd,r.memoryTypeBits,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VK_CHECK(vkAllocateMemory(d,&ai,NULL,&b.mem));
  VK_CHECK(vkBindBufferMemory(d,b.buf,b.mem,0));
  return b;
}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
  return t.tv_sec+t.tv_nsec*1e-9;}

int main(int argc,char**argv){
  if(argc<2){fprintf(stderr,"usage: %s matmul.spv [M K N]\n",argv[0]);return 1;}
  uint32_t M=(argc>=5)?atoi(argv[2]):128;
  uint32_t K=(argc>=5)?atoi(argv[3]):128;
  uint32_t N=(argc>=5)?atoi(argv[4]):128;
  size_t na=(size_t)M*K, nb=(size_t)K*N, nc=(size_t)M*N;

  VkApplicationInfo app={VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.apiVersion=VK_API_VERSION_1_1;
  VkInstanceCreateInfo ici={VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo=&app;
  VkInstance inst; VK_CHECK(vkCreateInstance(&ici,NULL,&inst));
  uint32_t nd=0; VK_CHECK(vkEnumeratePhysicalDevices(inst,&nd,NULL));
  if(!nd){fprintf(stderr,"no device\n");return 2;}
  VkPhysicalDevice*ds=malloc(nd*sizeof(*ds));
  VK_CHECK(vkEnumeratePhysicalDevices(inst,&nd,ds));
  int pick=0; const char*e=getenv("NELISP_GPU_DEVICE");
  if(e) pick=atoi(e); else for(uint32_t i=0;i<nd;i++){
    VkPhysicalDeviceProperties p;vkGetPhysicalDeviceProperties(ds[i],&p);
    if(p.deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU){pick=i;break;}}
  if(pick<0||pick>=(int)nd)pick=0;
  VkPhysicalDevice pd=ds[pick];
  VkPhysicalDeviceProperties pr;vkGetPhysicalDeviceProperties(pd,&pr);
  printf("device = %s\n",pr.deviceName);

  uint32_t nq=0;vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,NULL);
  VkQueueFamilyProperties*qf=malloc(nq*sizeof(*qf));
  vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,qf);
  uint32_t cq=nq; for(uint32_t i=0;i<nq;i++)
    if(qf[i].queueFlags&VK_QUEUE_COMPUTE_BIT){cq=i;break;}
  float prio=1.0f;
  VkDeviceQueueCreateInfo qci={VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex=cq;qci.queueCount=1;qci.pQueuePriorities=&prio;
  VkDeviceCreateInfo dci={VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount=1;dci.pQueueCreateInfos=&qci;
  VkDevice dev;VK_CHECK(vkCreateDevice(pd,&dci,NULL,&dev));
  VkQueue q;vkGetDeviceQueue(dev,cq,0,&q);

  Buf A=mkbuf(pd,dev,na*4),B=mkbuf(pd,dev,nb*4),C=mkbuf(pd,dev,nc*4);
  float*pa,*pb; VK_CHECK(vkMapMemory(dev,A.mem,0,na*4,0,(void**)&pa));
  VK_CHECK(vkMapMemory(dev,B.mem,0,nb*4,0,(void**)&pb));
  for(size_t i=0;i<na;i++) pa[i]=(float)(i%4);
  for(size_t i=0;i<nb;i++) pb[i]=(float)(i%3);
  vkUnmapMemory(dev,A.mem);vkUnmapMemory(dev,B.mem);

  size_t spvn;uint32_t*spv=read_spirv(argv[1],&spvn);
  VkShaderModuleCreateInfo smci={VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize=spvn;smci.pCode=spv;
  VkShaderModule sm;VK_CHECK(vkCreateShaderModule(dev,&smci,NULL,&sm));

  VkDescriptorSetLayoutBinding bd[3];
  for(int i=0;i<3;i++){bd[i]=(VkDescriptorSetLayoutBinding){0};bd[i].binding=i;
    bd[i].descriptorCount=1;bd[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bd[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
  VkDescriptorSetLayoutCreateInfo dl={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dl.bindingCount=3;dl.pBindings=bd;
  VkDescriptorSetLayout dsl;VK_CHECK(vkCreateDescriptorSetLayout(dev,&dl,NULL,&dsl));
  VkPushConstantRange pcr={VK_SHADER_STAGE_COMPUTE_BIT,0,3*sizeof(uint32_t)};
  VkPipelineLayoutCreateInfo pli={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pli.setLayoutCount=1;pli.pSetLayouts=&dsl;
  pli.pushConstantRangeCount=1;pli.pPushConstantRanges=&pcr;
  VkPipelineLayout pl;VK_CHECK(vkCreatePipelineLayout(dev,&pli,NULL,&pl));
  VkComputePipelineCreateInfo cp={VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cp.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cp.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;cp.stage.module=sm;cp.stage.pName="main";
  cp.layout=pl;VkPipeline pipe;
  VK_CHECK(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&cp,NULL,&pipe));

  VkDescriptorPoolSize ps={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3};
  VkDescriptorPoolCreateInfo dpi={VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpi.maxSets=1;dpi.poolSizeCount=1;dpi.pPoolSizes=&ps;
  VkDescriptorPool dp;VK_CHECK(vkCreateDescriptorPool(dev,&dpi,NULL,&dp));
  VkDescriptorSetAllocateInfo da={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  da.descriptorPool=dp;da.descriptorSetCount=1;da.pSetLayouts=&dsl;
  VkDescriptorSet dset;VK_CHECK(vkAllocateDescriptorSets(dev,&da,&dset));
  VkDescriptorBufferInfo bi3[3]={{A.buf,0,VK_WHOLE_SIZE},{B.buf,0,VK_WHOLE_SIZE},{C.buf,0,VK_WHOLE_SIZE}};
  VkWriteDescriptorSet wr[3];
  for(int i=0;i<3;i++){wr[i]=(VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wr[i].dstSet=dset;wr[i].dstBinding=i;wr[i].descriptorCount=1;
    wr[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;wr[i].pBufferInfo=&bi3[i];}
  vkUpdateDescriptorSets(dev,3,wr,0,NULL);

  VkCommandPoolCreateInfo ci={VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  ci.queueFamilyIndex=cq;VkCommandPool cpool;
  VK_CHECK(vkCreateCommandPool(dev,&ci,NULL,&cpool));
  VkCommandBufferAllocateInfo ba={VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ba.commandPool=cpool;ba.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ba.commandBufferCount=1;
  VkCommandBuffer cmd;VK_CHECK(vkAllocateCommandBuffers(dev,&ba,&cmd));
  VkCommandBufferBeginInfo cb={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  cb.flags=0;  /* reusable: warmup + timed submit */
  VK_CHECK(vkBeginCommandBuffer(cmd,&cb));
  vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
  vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&dset,0,NULL);
  uint32_t pc[3]={M,K,N};
  vkCmdPushConstants(cmd,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(pc),pc);
  uint32_t iters=(argc>=7)?(uint32_t)atoi(argv[6]):20;
  {uint32_t groups;
   if(argc>=6&&!strcmp(argv[5],"tiled")) groups=((M+15)/16)*((N+15)/16);
   else if(argc>=6&&!strcmp(argv[5],"reg")) groups=((((M+3)/4)*((N+3)/4))+63)/64;
   else groups=((uint32_t)nc+63)/64;
   for(uint32_t it=0;it<iters;it++){
     vkCmdDispatch(cmd,groups,1,1);
     VkMemoryBarrier mb={VK_STRUCTURE_TYPE_MEMORY_BARRIER};
     mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
     vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,NULL,0,NULL);}}
  VK_CHECK(vkEndCommandBuffer(cmd));

  VkSubmitInfo si={VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount=1;si.pCommandBuffers=&cmd;
  VkFenceCreateInfo fi={VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;VK_CHECK(vkCreateFence(dev,&fi,NULL,&fence));
  VK_CHECK(vkQueueSubmit(q,1,&si,fence));                      /* warmup */
  VK_CHECK(vkWaitForFences(dev,1,&fence,VK_TRUE,(uint64_t)6e10));
  VK_CHECK(vkResetFences(dev,1,&fence));
  double t0=now();
  VK_CHECK(vkQueueSubmit(q,1,&si,fence));                      /* timed */
  VK_CHECK(vkWaitForFences(dev,1,&fence,VK_TRUE,(uint64_t)6e10));
  double dt=(now()-t0)/(double)iters;

  float*pc2;VK_CHECK(vkMapMemory(dev,C.mem,0,nc*4,0,(void**)&pc2));
  /* CPU reference on a few sampled rows for correctness */
  int bad=0; size_t checks=0;
  for(uint32_t r=0;r<M && bad<5; r+=(M/7>0?M/7:1))
    for(uint32_t c=0;c<N && bad<5; c+=(N/7>0?N/7:1)){
      double acc=0; for(uint32_t k=0;k<K;k++) acc+=(double)(((size_t)r*K+k)%4)*(double)(((size_t)k*N+c)%3);
      float got=pc2[(size_t)r*N+c];
      if((double)got!=acc){ if(bad<5) fprintf(stderr,"C[%u,%u]=%g want=%g\n",r,c,got,acc); bad++; }
      checks++;
    }
  double flop=2.0*(double)M*N*K;
  printf("matmul %ux%u x %ux%u  gpu=%.4fs  %.1f GFLOP/s  checks=%zu  %s\n",
         M,K,K,N,dt,flop/dt/1e9,checks,bad?"FAIL":"PASS");
  vkUnmapMemory(dev,C.mem);
  return bad?1:0;
}

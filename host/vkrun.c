/* vkrun.c -- generic Vulkan compute dispatcher.
 *
 * Kernel-agnostic reference host: given a SPIR-V module, a set of float
 * storage buffers (sizes in binding order), uint push constants, and a
 * dispatch group count, it loads input buffer data from a binary file,
 * runs the kernel, and writes ALL buffers back to a binary file.  The
 * caller (elisp) prepares inputs and verifies outputs against the
 * photon-tensor CPU oracle -- so new kernels need no new C host.
 *
 *   build: cc -O2 vkrun.c -lvulkan -o vkrun
 *   run:   ./vkrun SPV IN_FILE OUT_FILE LOCAL GROUPS PUSH_CSV SIZES_CSV
 *          PUSH_CSV = "-" for none, e.g. "128,128,128"
 *          SIZES_CSV = float count per buffer, e.g. "16384,16384,16384"
 */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VK_CHECK(x) do{VkResult _r=(x);if(_r!=VK_SUCCESS){\
 fprintf(stderr,"VK %d %s:%d\n",_r,__FILE__,__LINE__);exit(2);}}while(0)

static int parse_csv(const char*s,uint32_t*out,int max){
 if(!s||!strcmp(s,"-"))return 0; int n=0; char*tmp=strdup(s);
 for(char*t=strtok(tmp,",");t&&n<max;t=strtok(NULL,",")) out[n++]=(uint32_t)strtoul(t,NULL,10);
 free(tmp); return n;}
static uint32_t mt(VkPhysicalDevice pd,uint32_t bits,VkMemoryPropertyFlags w){
 VkPhysicalDeviceMemoryProperties mp;vkGetPhysicalDeviceMemoryProperties(pd,&mp);
 for(uint32_t i=0;i<mp.memoryTypeCount;i++)if((bits&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&w)==w)return i;
 exit(2);}

int main(int argc,char**argv){
 if(argc<8){fprintf(stderr,"usage: %s SPV IN OUT LOCAL GROUPS PUSH SIZES\n",argv[0]);return 1;}
 const char*spvp=argv[1],*inp=argv[2],*outp=argv[3];
 uint32_t groups=(uint32_t)strtoul(argv[5],NULL,10);
 uint32_t push[16]; int npush=parse_csv(argv[6],push,16);
 uint32_t sizes[16]; int nbuf=parse_csv(argv[7],sizes,16);
 if(nbuf<1){fprintf(stderr,"need >=1 buffer\n");return 1;}
 size_t total=0; for(int i=0;i<nbuf;i++) total+=sizes[i];

 /* read input floats */
 float*host=malloc(total*4);
 { FILE*f=fopen(inp,"rb"); if(!f){fprintf(stderr,"open %s\n",inp);return 2;}
   size_t got=fread(host,4,total,f); fclose(f);
   for(size_t i=got;i<total;i++) host[i]=0.0f; }

 VkApplicationInfo app={VK_STRUCTURE_TYPE_APPLICATION_INFO};app.apiVersion=VK_API_VERSION_1_1;
 VkInstanceCreateInfo ici={VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ici.pApplicationInfo=&app;
 VkInstance inst;VK_CHECK(vkCreateInstance(&ici,NULL,&inst));
 uint32_t nd=0;VK_CHECK(vkEnumeratePhysicalDevices(inst,&nd,NULL));if(!nd)return 2;
 VkPhysicalDevice*ds=malloc(nd*sizeof(*ds));VK_CHECK(vkEnumeratePhysicalDevices(inst,&nd,ds));
 int pick=0;const char*e=getenv("NELISP_GPU_DEVICE");
 if(e)pick=atoi(e);else for(uint32_t i=0;i<nd;i++){VkPhysicalDeviceProperties p;
  vkGetPhysicalDeviceProperties(ds[i],&p);if(p.deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU){pick=i;break;}}
 if(pick<0||pick>=(int)nd)pick=0;VkPhysicalDevice pd=ds[pick];
 uint32_t nq=0;vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,NULL);
 VkQueueFamilyProperties*qf=malloc(nq*sizeof(*qf));vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,qf);
 uint32_t cq=0;for(uint32_t i=0;i<nq;i++)if(qf[i].queueFlags&VK_QUEUE_COMPUTE_BIT){cq=i;break;}
 float prio=1;VkDeviceQueueCreateInfo qci={VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
 qci.queueFamilyIndex=cq;qci.queueCount=1;qci.pQueuePriorities=&prio;
 VkDeviceCreateInfo dci={VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};dci.queueCreateInfoCount=1;dci.pQueueCreateInfos=&qci;
 VkDevice dev;VK_CHECK(vkCreateDevice(pd,&dci,NULL,&dev));VkQueue q;vkGetDeviceQueue(dev,cq,0,&q);

 VkBuffer*buf=malloc(nbuf*sizeof(VkBuffer));VkDeviceMemory*mem=malloc(nbuf*sizeof(VkDeviceMemory));
 size_t off=0;
 for(int i=0;i<nbuf;i++){VkDeviceSize bs=(VkDeviceSize)sizes[i]*4;
  VkBufferCreateInfo bi={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=bs;
  bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(dev,&bi,NULL,&buf[i]));
  VkMemoryRequirements r;vkGetBufferMemoryRequirements(dev,buf[i],&r);
  VkMemoryAllocateInfo ai={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=r.size;
  ai.memoryTypeIndex=mt(pd,r.memoryTypeBits,
   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VK_CHECK(vkAllocateMemory(dev,&ai,NULL,&mem[i]));VK_CHECK(vkBindBufferMemory(dev,buf[i],mem[i],0));
  void*p;VK_CHECK(vkMapMemory(dev,mem[i],0,bs,0,&p));memcpy(p,host+off,bs);vkUnmapMemory(dev,mem[i]);
  off+=sizes[i];}

 FILE*sf=fopen(spvp,"rb");if(!sf){fprintf(stderr,"open %s\n",spvp);return 2;}
 fseek(sf,0,SEEK_END);long ssz=ftell(sf);fseek(sf,0,SEEK_SET);
 uint32_t*spv=malloc(ssz);if(fread(spv,1,ssz,sf)!=(size_t)ssz)return 2;fclose(sf);
 VkShaderModuleCreateInfo smci={VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};smci.codeSize=ssz;smci.pCode=spv;
 VkShaderModule sm;VK_CHECK(vkCreateShaderModule(dev,&smci,NULL,&sm));

 VkDescriptorSetLayoutBinding*bd=malloc(nbuf*sizeof(*bd));
 for(int i=0;i<nbuf;i++){bd[i]=(VkDescriptorSetLayoutBinding){0};bd[i].binding=i;bd[i].descriptorCount=1;
  bd[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;bd[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
 VkDescriptorSetLayoutCreateInfo dl={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
 dl.bindingCount=nbuf;dl.pBindings=bd;VkDescriptorSetLayout dsl;VK_CHECK(vkCreateDescriptorSetLayout(dev,&dl,NULL,&dsl));
 VkPushConstantRange pcr={VK_SHADER_STAGE_COMPUTE_BIT,0,(uint32_t)(npush*4)};
 VkPipelineLayoutCreateInfo pli={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
 pli.setLayoutCount=1;pli.pSetLayouts=&dsl;
 if(npush>0){pli.pushConstantRangeCount=1;pli.pPushConstantRanges=&pcr;}
 VkPipelineLayout pl;VK_CHECK(vkCreatePipelineLayout(dev,&pli,NULL,&pl));
 VkComputePipelineCreateInfo cp={VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
 cp.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;cp.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;
 cp.stage.module=sm;cp.stage.pName="main";cp.layout=pl;VkPipeline pipe;
 VK_CHECK(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&cp,NULL,&pipe));

 VkDescriptorPoolSize ps={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,(uint32_t)nbuf};
 VkDescriptorPoolCreateInfo dpi={VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
 dpi.maxSets=1;dpi.poolSizeCount=1;dpi.pPoolSizes=&ps;VkDescriptorPool dp;VK_CHECK(vkCreateDescriptorPool(dev,&dpi,NULL,&dp));
 VkDescriptorSetAllocateInfo da={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
 da.descriptorPool=dp;da.descriptorSetCount=1;da.pSetLayouts=&dsl;VkDescriptorSet dset;VK_CHECK(vkAllocateDescriptorSets(dev,&da,&dset));
 VkDescriptorBufferInfo*bi=malloc(nbuf*sizeof(*bi));VkWriteDescriptorSet*wr=malloc(nbuf*sizeof(*wr));
 for(int i=0;i<nbuf;i++){bi[i]=(VkDescriptorBufferInfo){buf[i],0,VK_WHOLE_SIZE};
  wr[i]=(VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};wr[i].dstSet=dset;wr[i].dstBinding=i;
  wr[i].descriptorCount=1;wr[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;wr[i].pBufferInfo=&bi[i];}
 vkUpdateDescriptorSets(dev,nbuf,wr,0,NULL);

 VkCommandPoolCreateInfo ci={VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};ci.queueFamilyIndex=cq;
 VkCommandPool cpool;VK_CHECK(vkCreateCommandPool(dev,&ci,NULL,&cpool));
 VkCommandBufferAllocateInfo ba={VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
 ba.commandPool=cpool;ba.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ba.commandBufferCount=1;
 VkCommandBuffer cmd;VK_CHECK(vkAllocateCommandBuffers(dev,&ba,&cmd));
 VkCommandBufferBeginInfo cb={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
 cb.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;VK_CHECK(vkBeginCommandBuffer(cmd,&cb));
 vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
 vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&dset,0,NULL);
 if(npush>0)vkCmdPushConstants(cmd,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,npush*4,push);
 vkCmdDispatch(cmd,groups,1,1);VK_CHECK(vkEndCommandBuffer(cmd));
 VkSubmitInfo si={VK_STRUCTURE_TYPE_SUBMIT_INFO};si.commandBufferCount=1;si.pCommandBuffers=&cmd;
 VkFenceCreateInfo fi={VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};VkFence fence;VK_CHECK(vkCreateFence(dev,&fi,NULL,&fence));
 VK_CHECK(vkQueueSubmit(q,1,&si,fence));VK_CHECK(vkWaitForFences(dev,1,&fence,VK_TRUE,(uint64_t)6e10));

 off=0;
 for(int i=0;i<nbuf;i++){void*p;VK_CHECK(vkMapMemory(dev,mem[i],0,(VkDeviceSize)sizes[i]*4,0,&p));
  memcpy(host+off,p,(size_t)sizes[i]*4);vkUnmapMemory(dev,mem[i]);off+=sizes[i];}
 FILE*of=fopen(outp,"wb");if(!of){fprintf(stderr,"open %s\n",outp);return 2;}
 fwrite(host,4,total,of);fclose(of);
 return 0;
}

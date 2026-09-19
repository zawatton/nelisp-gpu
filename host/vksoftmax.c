/* vksoftmax.c -- Vulkan compute reference host for row-wise softmax.
 *
 * Verifies the DSL-compiled softmax kernel: A and C are M x N row-major
 * float; one thread per row computes C[r] = softmax(A[r]).  Push
 * constants: uint M, N.  Compares against a CPU reference with a
 * tolerance (GPU `exp` differs slightly from libm).
 *
 *   build: cc -O2 vksoftmax.c -lvulkan -lm -o vksoftmax
 *   run:   ./vksoftmax kernels/softmax.spv [M N]
 */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define VK_CHECK(x) do{VkResult _r=(x);if(_r!=VK_SUCCESS){\
 fprintf(stderr,"VK %d %s:%d\n",_r,__FILE__,__LINE__);exit(2);}}while(0)

static uint32_t*rd(const char*p,size_t*n){FILE*f=fopen(p,"rb");if(!f){fprintf(stderr,"open %s\n",p);exit(2);}
 fseek(f,0,SEEK_END);long s=ftell(f);fseek(f,0,SEEK_SET);if(s<=0||s%4){exit(2);}
 uint32_t*b=malloc(s);if(fread(b,1,s,f)!=(size_t)s){exit(2);}fclose(f);*n=s;return b;}
static uint32_t mt(VkPhysicalDevice pd,uint32_t bits,VkMemoryPropertyFlags w){
 VkPhysicalDeviceMemoryProperties mp;vkGetPhysicalDeviceMemoryProperties(pd,&mp);
 for(uint32_t i=0;i<mp.memoryTypeCount;i++)if((bits&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&w)==w)return i;
 exit(2);}
typedef struct{VkBuffer b;VkDeviceMemory m;}Buf;
static Buf mk(VkPhysicalDevice pd,VkDevice d,VkDeviceSize sz){Buf x;
 VkBufferCreateInfo bi={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=sz;
 bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
 VK_CHECK(vkCreateBuffer(d,&bi,NULL,&x.b));VkMemoryRequirements r;
 vkGetBufferMemoryRequirements(d,x.b,&r);VkMemoryAllocateInfo ai={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
 ai.allocationSize=r.size;ai.memoryTypeIndex=mt(pd,r.memoryTypeBits,
  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
 VK_CHECK(vkAllocateMemory(d,&ai,NULL,&x.m));VK_CHECK(vkBindBufferMemory(d,x.b,x.m,0));return x;}

int main(int argc,char**argv){
 if(argc<2){fprintf(stderr,"usage: %s softmax.spv [M N]\n",argv[0]);return 1;}
 uint32_t M=(argc>=4)?atoi(argv[2]):64, N=(argc>=4)?atoi(argv[3]):256;
 size_t sz=(size_t)M*N;
 VkApplicationInfo app={VK_STRUCTURE_TYPE_APPLICATION_INFO};app.apiVersion=VK_API_VERSION_1_1;
 VkInstanceCreateInfo ici={VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ici.pApplicationInfo=&app;
 VkInstance inst;VK_CHECK(vkCreateInstance(&ici,NULL,&inst));
 uint32_t nd=0;VK_CHECK(vkEnumeratePhysicalDevices(inst,&nd,NULL));if(!nd)return 2;
 VkPhysicalDevice*ds=malloc(nd*sizeof(*ds));VK_CHECK(vkEnumeratePhysicalDevices(inst,&nd,ds));
 int pick=0;const char*e=getenv("NELISP_GPU_DEVICE");
 if(e)pick=atoi(e);else for(uint32_t i=0;i<nd;i++){VkPhysicalDeviceProperties p;
  vkGetPhysicalDeviceProperties(ds[i],&p);if(p.deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU){pick=i;break;}}
 if(pick<0||pick>=(int)nd)pick=0;VkPhysicalDevice pd=ds[pick];
 VkPhysicalDeviceProperties pr;vkGetPhysicalDeviceProperties(pd,&pr);printf("device = %s\n",pr.deviceName);
 uint32_t nq=0;vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,NULL);
 VkQueueFamilyProperties*qf=malloc(nq*sizeof(*qf));vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,qf);
 uint32_t cq=0;for(uint32_t i=0;i<nq;i++)if(qf[i].queueFlags&VK_QUEUE_COMPUTE_BIT){cq=i;break;}
 float prio=1;VkDeviceQueueCreateInfo qci={VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
 qci.queueFamilyIndex=cq;qci.queueCount=1;qci.pQueuePriorities=&prio;
 VkDeviceCreateInfo dci={VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};dci.queueCreateInfoCount=1;dci.pQueueCreateInfos=&qci;
 VkDevice dev;VK_CHECK(vkCreateDevice(pd,&dci,NULL,&dev));VkQueue q;vkGetDeviceQueue(dev,cq,0,&q);

 Buf A=mk(pd,dev,sz*4),C=mk(pd,dev,sz*4);
 float*pa;VK_CHECK(vkMapMemory(dev,A.m,0,sz*4,0,(void**)&pa));
 for(size_t i=0;i<sz;i++)pa[i]=(float)((i%7))*0.3f;
 vkUnmapMemory(dev,A.m);

 size_t sn;uint32_t*spv=rd(argv[1],&sn);
 VkShaderModuleCreateInfo smci={VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};smci.codeSize=sn;smci.pCode=spv;
 VkShaderModule sm;VK_CHECK(vkCreateShaderModule(dev,&smci,NULL,&sm));
 VkDescriptorSetLayoutBinding bd[2];for(int i=0;i<2;i++){bd[i]=(VkDescriptorSetLayoutBinding){0};
  bd[i].binding=i;bd[i].descriptorCount=1;bd[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bd[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
 VkDescriptorSetLayoutCreateInfo dl={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
 dl.bindingCount=2;dl.pBindings=bd;VkDescriptorSetLayout dsl;VK_CHECK(vkCreateDescriptorSetLayout(dev,&dl,NULL,&dsl));
 VkPushConstantRange pcr={VK_SHADER_STAGE_COMPUTE_BIT,0,2*sizeof(uint32_t)};
 VkPipelineLayoutCreateInfo pli={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
 pli.setLayoutCount=1;pli.pSetLayouts=&dsl;pli.pushConstantRangeCount=1;pli.pPushConstantRanges=&pcr;
 VkPipelineLayout pl;VK_CHECK(vkCreatePipelineLayout(dev,&pli,NULL,&pl));
 VkComputePipelineCreateInfo cp={VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
 cp.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;cp.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;
 cp.stage.module=sm;cp.stage.pName="main";cp.layout=pl;VkPipeline pipe;
 VK_CHECK(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&cp,NULL,&pipe));
 VkDescriptorPoolSize ps={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2};
 VkDescriptorPoolCreateInfo dpi={VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
 dpi.maxSets=1;dpi.poolSizeCount=1;dpi.pPoolSizes=&ps;VkDescriptorPool dp;VK_CHECK(vkCreateDescriptorPool(dev,&dpi,NULL,&dp));
 VkDescriptorSetAllocateInfo da={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
 da.descriptorPool=dp;da.descriptorSetCount=1;da.pSetLayouts=&dsl;VkDescriptorSet dset;
 VK_CHECK(vkAllocateDescriptorSets(dev,&da,&dset));
 VkDescriptorBufferInfo bi2[2]={{A.b,0,VK_WHOLE_SIZE},{C.b,0,VK_WHOLE_SIZE}};
 VkWriteDescriptorSet wr[2];for(int i=0;i<2;i++){wr[i]=(VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  wr[i].dstSet=dset;wr[i].dstBinding=i;wr[i].descriptorCount=1;wr[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  wr[i].pBufferInfo=&bi2[i];}vkUpdateDescriptorSets(dev,2,wr,0,NULL);
 VkCommandPoolCreateInfo ci={VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};ci.queueFamilyIndex=cq;
 VkCommandPool cpool;VK_CHECK(vkCreateCommandPool(dev,&ci,NULL,&cpool));
 VkCommandBufferAllocateInfo ba={VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
 ba.commandPool=cpool;ba.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ba.commandBufferCount=1;
 VkCommandBuffer cmd;VK_CHECK(vkAllocateCommandBuffers(dev,&ba,&cmd));
 VkCommandBufferBeginInfo cb={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
 cb.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;VK_CHECK(vkBeginCommandBuffer(cmd,&cb));
 vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
 vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&dset,0,NULL);
 uint32_t pc[2]={M,N};vkCmdPushConstants(cmd,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(pc),pc);
 vkCmdDispatch(cmd,(M+63)/64,1,1);VK_CHECK(vkEndCommandBuffer(cmd));
 VkSubmitInfo si={VK_STRUCTURE_TYPE_SUBMIT_INFO};si.commandBufferCount=1;si.pCommandBuffers=&cmd;
 VkFenceCreateInfo fi={VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};VkFence fence;VK_CHECK(vkCreateFence(dev,&fi,NULL,&fence));
 VK_CHECK(vkQueueSubmit(q,1,&si,fence));VK_CHECK(vkWaitForFences(dev,1,&fence,VK_TRUE,(uint64_t)6e10));
 float*pc2;VK_CHECK(vkMapMemory(dev,C.m,0,sz*4,0,(void**)&pc2));
 int bad=0;double maxerr=0;
 for(uint32_t r=0;r<M && bad<5; r+=(M/9>0?M/9:1)){
  double mx=-1e30;for(uint32_t j=0;j<N;j++){double v=(double)(((size_t)r*N+j)%7)*0.3;if(v>mx)mx=v;}
  double s=0;for(uint32_t j=0;j<N;j++)s+=exp((double)(((size_t)r*N+j)%7)*0.3-mx);
  for(uint32_t j=0;j<N;j+=(N/9>0?N/9:1)){
   double ref=exp((double)(((size_t)r*N+j)%7)*0.3-mx)/s;double got=pc2[(size_t)r*N+j];
   double err=fabs(got-ref);if(err>maxerr)maxerr=err;
   if(err>1e-3){if(bad<5)fprintf(stderr,"C[%u,%u]=%g ref=%g\n",r,j,got,ref);bad++;}}}
 /* row sum ~ 1.0 check on row 0 */
 double rs=0;for(uint32_t j=0;j<N;j++)rs+=pc2[j];
 printf("softmax %ux%u  row0_sum=%.5f  max_abs_err=%.2e  %s\n",M,N,rs,maxerr,bad?"FAIL":"PASS");
 vkUnmapMemory(dev,C.m);return bad?1:0;
}

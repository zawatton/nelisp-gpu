/* vkcompute.c -- minimal Vulkan compute runner (REFERENCE host).
 *
 * Part of nelisp-gpu.  This C host is the *reference* runtime used to
 * de-risk the GPU path: it loads a SPIR-V compute shader, binds three
 * float storage buffers (a, b, c), dispatches, and verifies c = a + b.
 *
 * The end goal is to drive Vulkan from elisp/nelisp (nelisp-cc extern
 * calls or a thin module).  This C file exists only to prove the
 * hardware + SPIR-V + Vulkan stack end-to-end before the elisp host
 * and the elisp -> SPIR-V kernel compiler are built.
 *
 *   build: cc vkcompute.c -lvulkan -o vkcompute
 *   run:   ./vkcompute kernels/vadd.spv [N]
 *
 * Device selection: env NELISP_GPU_DEVICE = physical device index,
 * otherwise the first VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, else 0.
 */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VK_CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
  fprintf(stderr, "VK error %d at %s:%d\n", _r, __FILE__, __LINE__); \
  exit(2); } } while (0)

static uint32_t *read_spirv(const char *path, size_t *out_bytes) {
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  if (sz <= 0 || (sz % 4) != 0) { fprintf(stderr, "bad spv size %ld\n", sz); exit(2); }
  uint32_t *buf = malloc(sz);
  if (fread(buf, 1, sz, f) != (size_t)sz) { fprintf(stderr, "short read\n"); exit(2); }
  fclose(f);
  *out_bytes = (size_t)sz;
  return buf;
}

static uint32_t find_mem_type(VkPhysicalDevice pd, uint32_t bits,
                              VkMemoryPropertyFlags want) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(pd, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((bits & (1u << i)) &&
        (mp.memoryTypes[i].propertyFlags & want) == want)
      return i;
  fprintf(stderr, "no suitable memory type\n"); exit(2);
}

typedef struct { VkBuffer buf; VkDeviceMemory mem; VkDeviceSize size; } Buffer;

static Buffer make_buffer(VkPhysicalDevice pd, VkDevice dev, VkDeviceSize size) {
  Buffer b; b.size = size;
  VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = size;
  bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(dev, &bi, NULL, &b.buf));
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(dev, b.buf, &req);
  VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = find_mem_type(pd, req.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VK_CHECK(vkAllocateMemory(dev, &ai, NULL, &b.mem));
  VK_CHECK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
  return b;
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s shader.spv [N]\n", argv[0]); return 1; }
  const char *spv_path = argv[1];
  uint32_t N = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 1024;
  size_t bytes = (size_t)N * sizeof(float);

  /* instance */
  VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "nelisp-gpu"; app.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  VkInstance inst;
  VK_CHECK(vkCreateInstance(&ici, NULL, &inst));

  /* physical device */
  uint32_t ndev = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(inst, &ndev, NULL));
  if (ndev == 0) { fprintf(stderr, "no Vulkan devices\n"); return 2; }
  VkPhysicalDevice *devs = malloc(ndev * sizeof(*devs));
  VK_CHECK(vkEnumeratePhysicalDevices(inst, &ndev, devs));
  int pick = -1;
  const char *envd = getenv("NELISP_GPU_DEVICE");
  if (envd) pick = atoi(envd);
  if (pick < 0 || pick >= (int)ndev) {
    pick = 0;
    for (uint32_t i = 0; i < ndev; i++) {
      VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(devs[i], &p);
      if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { pick = (int)i; break; }
    }
  }
  VkPhysicalDevice pd = devs[pick];
  VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props);
  printf("device[%d] = %s (type %d)\n", pick, props.deviceName, props.deviceType);

  /* compute queue family */
  uint32_t nqf = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, NULL);
  VkQueueFamilyProperties *qf = malloc(nqf * sizeof(*qf));
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qf);
  uint32_t cq = nqf;
  for (uint32_t i = 0; i < nqf; i++)
    if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { cq = i; break; }
  if (cq == nqf) { fprintf(stderr, "no compute queue\n"); return 2; }

  /* logical device + queue */
  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = cq; qci.queueCount = 1; qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
  VkDevice dev;
  VK_CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
  VkQueue queue; vkGetDeviceQueue(dev, cq, 0, &queue);

  /* buffers a, b, c */
  Buffer a = make_buffer(pd, dev, bytes);
  Buffer b = make_buffer(pd, dev, bytes);
  Buffer c = make_buffer(pd, dev, bytes);
  float *pa, *pb;
  VK_CHECK(vkMapMemory(dev, a.mem, 0, bytes, 0, (void **)&pa));
  VK_CHECK(vkMapMemory(dev, b.mem, 0, bytes, 0, (void **)&pb));
  for (uint32_t i = 0; i < N; i++) { pa[i] = (float)i; pb[i] = (float)(2 * i); }
  vkUnmapMemory(dev, a.mem); vkUnmapMemory(dev, b.mem);

  /* shader module */
  size_t spv_bytes; uint32_t *spv = read_spirv(spv_path, &spv_bytes);
  VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = spv_bytes; smci.pCode = spv;
  VkShaderModule sm;
  VK_CHECK(vkCreateShaderModule(dev, &smci, NULL, &sm));

  /* descriptor set layout: 3 storage buffers */
  VkDescriptorSetLayoutBinding binds[3];
  for (int i = 0; i < 3; i++) {
    binds[i] = (VkDescriptorSetLayoutBinding){0};
    binds[i].binding = i; binds[i].descriptorCount = 1;
    binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dslci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dslci.bindingCount = 3; dslci.pBindings = binds;
  VkDescriptorSetLayout dsl;
  VK_CHECK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));

  /* pipeline layout with push constant (uint n) */
  VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
  VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
  plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
  VkPipelineLayout pl;
  VK_CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

  /* compute pipeline */
  VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = sm; cpci.stage.pName = "main";
  cpci.layout = pl;
  VkPipeline pipe;
  VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));

  /* descriptor pool + set */
  VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
  VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &ps;
  VkDescriptorPool dp;
  VK_CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
  VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = dp; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
  VkDescriptorSet ds;
  VK_CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));
  VkDescriptorBufferInfo bi3[3] = {
    {a.buf, 0, VK_WHOLE_SIZE}, {b.buf, 0, VK_WHOLE_SIZE}, {c.buf, 0, VK_WHOLE_SIZE}};
  VkWriteDescriptorSet w[3];
  for (int i = 0; i < 3; i++) {
    w[i] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[i].dstSet = ds; w[i].dstBinding = i; w[i].descriptorCount = 1;
    w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[i].pBufferInfo = &bi3[i];
  }
  vkUpdateDescriptorSets(dev, 3, w, 0, NULL);

  /* command buffer */
  VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpi.queueFamilyIndex = cq;
  VkCommandPool cmdpool;
  VK_CHECK(vkCreateCommandPool(dev, &cpi, NULL, &cmdpool));
  VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = cmdpool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd;
  VK_CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));
  VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(cmd, &cbbi));
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &N);
  vkCmdDispatch(cmd, (N + 63) / 64, 1, 1);
  VK_CHECK(vkEndCommandBuffer(cmd));

  /* submit + wait */
  VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
  VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence; VK_CHECK(vkCreateFence(dev, &fci, NULL, &fence));
  VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
  VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, (uint64_t)1e10));

  /* verify c = a + b */
  float *pc;
  VK_CHECK(vkMapMemory(dev, c.mem, 0, bytes, 0, (void **)&pc));
  int bad = 0;
  for (uint32_t i = 0; i < N; i++) {
    float want = (float)i + (float)(2 * i);
    if (pc[i] != want) {
      if (bad < 5) fprintf(stderr, "mismatch i=%u got=%g want=%g\n", i, pc[i], want);
      bad++;
    }
  }
  printf("N=%u  c[0]=%g c[1]=%g c[last]=%g  %s\n",
         N, pc[0], (N > 1 ? pc[1] : 0.0f), pc[N - 1],
         bad ? "FAIL" : "PASS");
  vkUnmapMemory(dev, c.mem);
  return bad ? 1 : 0;
}

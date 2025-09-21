#include <vulkan/vk_platform.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_structs.hpp>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <memory>
#include <map>
#include <algorithm>
#include <cstring>
#include <GLFW/glfw3native.h>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <fstream>
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <array>
#include <chrono>
#include <unordered_map>
#include <optional>
#include <random>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader/tiny_obj_loader.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr uint64_t FenceTimeout = 100000000;
constexpr uint32_t PARTICLE_COUNT = 8192;

constexpr int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct UniformBufferObject {
    float deltaTime = 1.0f;
};

struct Particle {
    glm::vec2 position;
    glm::vec2 velocity;
    glm::vec4 color;

    static vk::VertexInputBindingDescription getBindingDescription() {
        return { 0, sizeof(Particle), vk::VertexInputRate::eVertex };
    }

    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
        return {
            vk::VertexInputAttributeDescription( 0, 0, vk::Format::eR32G32Sfloat, offsetof(Particle, position) ),
            vk::VertexInputAttributeDescription( 1, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Particle, color) ),
        };
    }
};

class ComputeShaderApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window = nullptr;

    vk::raii::Context  context;
    vk::raii::Instance instance = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
    vk::raii::SurfaceKHR surface = nullptr;

    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;

    vk::raii::Queue graphicsQueue = nullptr;
    vk::raii::Queue computeQueue = nullptr;
    vk::raii::Queue presentQueue = nullptr;

    vk::raii::SwapchainKHR swapChain = nullptr;
    std::vector<vk::Image> swapChainImages;
    vk::SurfaceFormatKHR swapChainImageFormat;
    vk::Extent2D swapChainExtent;
    std::vector<vk::raii::ImageView> swapChainImageViews;

    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;

    vk::raii::DescriptorSetLayout computeDescriptorSetLayout = nullptr;
    vk::raii::PipelineLayout computePipelineLayout = nullptr;
    vk::raii::Pipeline computePipeline = nullptr;

    std::vector<vk::raii::Buffer> shaderStorageBuffers;
    std::vector<vk::raii::DeviceMemory> shaderStorageBuffersMemory;

    std::vector<vk::raii::Buffer> uniformBuffers;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> computeDescriptorSets;

    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;
    std::vector<vk::raii::CommandBuffer> computeCommandBuffers;
    uint32_t graphicsAndComputeIndex = 0;

    vk::raii::Semaphore semaphore = nullptr;
    uint64_t timelineValue = 0;
    std::vector<vk::raii::Fence> inFlightFences;
    uint32_t currentFrame = 0;

    double lastFrameTime = 0.0;

    bool framebufferResized = false;

    double lastTime = 0.0f;

    std::vector<const char*> requiredDeviceExtension = {
        vk::KHRSwapchainExtensionName,
        vk::KHRSpirv14ExtensionName,
        vk::KHRSynchronization2ExtensionName,
        vk::KHRCreateRenderpass2ExtensionName
    };

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

        lastTime = glfwGetTime();
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = static_cast<ComputeShaderApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createComputeDescriptorSetLayout();
        createGraphicsPipeline();
        createComputePipeline();
        createCommandPool();
        createShaderStorageBuffers();
        createUniformBuffers();
        createDescriptorPool();
        createComputeDescriptorSets();
        createCommandBuffers();
        createComputeCommandBuffers();
        createSyncObjects();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
            double currentTime = glfwGetTime();
            lastFrameTime = (currentTime - lastTime) * 1000.0;
            lastTime = currentTime;
        }

        device.waitIdle();
    }

    void cleanupSwapChain() {
        swapChainImageViews.clear();
    }

    void cleanup() const {
        glfwDestroyWindow(window);

        glfwTerminate();
    }

    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        device.waitIdle();

        cleanupSwapChain();
        createSwapChain();
        createImageViews();
    }

    void createInstance() {
    vk::ApplicationInfo appInfo{};
    appInfo.pApplicationName = "Compute Shader";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = vk::ApiVersion14;
    
    std::vector<char const*> requiredLayers;
    if (enableValidationLayers) {
        requiredLayers.assign(validationLayers.begin(), validationLayers.end());
    }
    
    auto layerProperties = context.enumerateInstanceLayerProperties();
    for (auto const& requiredLayer : requiredLayers) {
        if (std::none_of(layerProperties.begin(), layerProperties.end(),
            [requiredLayer](auto const& layerProperty)
            { return strcmp(layerProperty.layerName, requiredLayer) == 0; })) {
            throw std::runtime_error("Required layer not supported: " + std::string(requiredLayer));
        }
    }
    
    auto requiredExtensions = getRequiredExtensions();
    auto extensionProperties = context.enumerateInstanceExtensionProperties();
    for (auto const& requiredExtension : requiredExtensions) {
        if (std::none_of(extensionProperties.begin(), extensionProperties.end(),
            [requiredExtension](auto const& extensionProperty)
            { return strcmp(extensionProperty.extensionName, requiredExtension) == 0; })) {
            throw std::runtime_error("Required extension not supported: " + std::string(requiredExtension));
        }
    }
    
    vk::InstanceCreateInfo createInfo{};
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledLayerCount = static_cast<uint32_t>(requiredLayers.size());
    createInfo.ppEnabledLayerNames = requiredLayers.data();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size());
    createInfo.ppEnabledExtensionNames = requiredExtensions.data();
    instance = vk::raii::Instance(context, createInfo);
}

void setupDebugMessenger() {
    if (!enableValidationLayers) return;
    
    vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
    );
    
        vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
        );
    
        vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT;
        debugUtilsMessengerCreateInfoEXT.messageSeverity = severityFlags;
        debugUtilsMessengerCreateInfoEXT.messageType = messageTypeFlags;
        debugUtilsMessengerCreateInfoEXT.pfnUserCallback = &debugCallback;
    
        debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
    }

    void createSurface() {
        VkSurfaceKHR       _surface;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
            throw std::runtime_error("failed to create window surface!");
        }
        surface = vk::raii::SurfaceKHR(instance, _surface);
    }

    void pickPhysicalDevice() {
        std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
    
        auto devIter = std::find_if(devices.begin(), devices.end(),
            [&](auto const& device) {
                // Check if the device supports the Vulkan 1.3 API version
                bool supportsVulkan1_3 = device.getProperties().apiVersion >= VK_API_VERSION_1_3;

                // Check if any of the queue families support graphics operations
                auto queueFamilies = device.getQueueFamilyProperties();
                bool supportsGraphics = std::any_of(queueFamilies.begin(), queueFamilies.end(),
                    [](auto const& qfp) { return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics); });

                // Check if all required device extensions are available
                auto availableDeviceExtensions = device.enumerateDeviceExtensionProperties();
                bool supportsAllRequiredExtensions = std::all_of(requiredDeviceExtension.begin(), requiredDeviceExtension.end(),
                    [&availableDeviceExtensions](auto const& requiredDeviceExtension) {
                        return std::any_of(availableDeviceExtensions.begin(), availableDeviceExtensions.end(),
                            [requiredDeviceExtension](auto const& availableDeviceExtension) {
                                return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0;
                            });
                    });

                auto features = device.template getFeatures2<vk::PhysicalDeviceFeatures2,
                                                         vk::PhysicalDeviceVulkan13Features,
                                                         vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
                                                         vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>();
                bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&
                                            features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
                                            features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState &&
                                            features.template get<vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>().timelineSemaphore;

                return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures;
            });
        
        if (devIter != devices.end()) {
            physicalDevice = *devIter;
        } else {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

void createLogicalDevice() {
    // find the index of the first queue family that supports graphics
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

    // get the first index into queueFamilyProperties which supports graphics and compute
    auto graphicsAndComputeQueueFamilyProperty = std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(),
        [](auto const& qfp) {
            return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) && (qfp.queueFlags & vk::QueueFlagBits::eCompute);
        });

    graphicsAndComputeIndex = static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), graphicsAndComputeQueueFamilyProperty));

    // determine a queueFamilyIndex that supports present
    // first check if the graphicsIndex is good enough
    auto presentIndex = physicalDevice.getSurfaceSupportKHR(graphicsAndComputeIndex, surface)
                                       ? graphicsAndComputeIndex
                                       : ~0;
    if (presentIndex == queueFamilyProperties.size()) {
        // the graphicsIndex doesn't support present -> look for another family index that supports both
        // graphics and present
        for (size_t i = 0; i < queueFamilyProperties.size(); i++) {
            if (((queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) && 
                 (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eCompute)) &&
                 physicalDevice.getSurfaceSupportKHR(static_cast<uint32_t>(i), surface)) {
                graphicsAndComputeIndex = static_cast<uint32_t>(i);
                presentIndex = graphicsAndComputeIndex;
                break;
            }
        }
        if (presentIndex == queueFamilyProperties.size()) {
            // there's nothing like a single family index that supports both graphics and present -> look for another
            // family index that supports present
            for (size_t i = 0; i < queueFamilyProperties.size(); i++) {
                if (physicalDevice.getSurfaceSupportKHR(static_cast<uint32_t>(i), surface)) {
                    presentIndex = static_cast<uint32_t>(i);
                    break;
                }
            }
        }
    }
    if ((graphicsAndComputeIndex == queueFamilyProperties.size()) || (presentIndex == queueFamilyProperties.size())) {
        throw std::runtime_error("Could not find a queue for graphics or present -> terminating");
    }

    // query for Vulkan 1.3 features
    vk::PhysicalDeviceFeatures2 features2;
    features2.features.samplerAnisotropy = true;
    
    vk::PhysicalDeviceVulkan13Features vulkan13Features;
    vulkan13Features.synchronization2 = true;
    vulkan13Features.dynamicRendering = true;
    
    vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT extendedDynamicStateFeatures;
    extendedDynamicStateFeatures.extendedDynamicState = true;
    
    vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR timelineSemaphoreFeatures;
    timelineSemaphoreFeatures.timelineSemaphore = true;
    
    // Chain the features
    features2.pNext = &vulkan13Features;
    vulkan13Features.pNext = &extendedDynamicStateFeatures;
    extendedDynamicStateFeatures.pNext = &timelineSemaphoreFeatures;

    // create a Device
        float queuePriority = 0.0f;
        vk::DeviceQueueCreateInfo deviceQueueCreateInfo;
        deviceQueueCreateInfo.queueFamilyIndex = graphicsAndComputeIndex;
        deviceQueueCreateInfo.queueCount = 1;
        deviceQueueCreateInfo.pQueuePriorities = &queuePriority;
    
        vk::DeviceCreateInfo deviceCreateInfo;
        deviceCreateInfo.pNext = &features2;
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
        deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtension.size());
        deviceCreateInfo.ppEnabledExtensionNames = requiredDeviceExtension.data();

        device = vk::raii::Device(physicalDevice, deviceCreateInfo);
        graphicsQueue = vk::raii::Queue(device, graphicsAndComputeIndex, 0);
        computeQueue = vk::raii::Queue(device, graphicsAndComputeIndex, 0);
        presentQueue = vk::raii::Queue(device, presentIndex, 0);
    }

    void createSwapChain() {
    auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
    swapChainImageFormat = chooseSwapSurfaceFormat(physicalDevice.getSurfaceFormatsKHR(surface));
    swapChainExtent = chooseSwapExtent(surfaceCapabilities);
    auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
    minImageCount = (surfaceCapabilities.maxImageCount > 0 && minImageCount > surfaceCapabilities.maxImageCount) ? surfaceCapabilities.maxImageCount : minImageCount;
    
    vk::SwapchainCreateInfoKHR swapChainCreateInfo;
    swapChainCreateInfo.flags = vk::SwapchainCreateFlagsKHR();
    swapChainCreateInfo.surface = surface;
    swapChainCreateInfo.minImageCount = minImageCount;
    swapChainCreateInfo.imageFormat = swapChainImageFormat.format;
    swapChainCreateInfo.imageColorSpace = swapChainImageFormat.colorSpace;
    swapChainCreateInfo.imageExtent = swapChainExtent;
    swapChainCreateInfo.imageArrayLayers = 1;
    swapChainCreateInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
    swapChainCreateInfo.imageSharingMode = vk::SharingMode::eExclusive;
    swapChainCreateInfo.preTransform = surfaceCapabilities.currentTransform;
    swapChainCreateInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    swapChainCreateInfo.presentMode = chooseSwapPresentMode(physicalDevice.getSurfacePresentModesKHR(surface));
    swapChainCreateInfo.clipped = true;
    swapChainCreateInfo.oldSwapchain = nullptr;

    swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
    swapChainImages = swapChain.getImages();
}

void createImageViews() {
    vk::ImageViewCreateInfo imageViewCreateInfo;
    imageViewCreateInfo.viewType = vk::ImageViewType::e2D;
    imageViewCreateInfo.format = swapChainImageFormat.format;
    imageViewCreateInfo.components = {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity};
    imageViewCreateInfo.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    
    for (auto image : swapChainImages) {
        imageViewCreateInfo.image = image;
        swapChainImageViews.emplace_back(device, imageViewCreateInfo);
    }
}

void createComputeDescriptorSetLayout() {
    std::array layoutBindings{
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
        vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
        vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr)
    };

    vk::DescriptorSetLayoutCreateInfo layoutInfo;
    layoutInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
    layoutInfo.pBindings = layoutBindings.data();
    computeDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
}

void createGraphicsPipeline() {
    vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo;
    vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
    vertShaderStageInfo.module = shaderModule;
    vertShaderStageInfo.pName = "vertMain";
    
    vk::PipelineShaderStageCreateInfo fragShaderStageInfo;
    fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
    fragShaderStageInfo.module = shaderModule;
    fragShaderStageInfo.pName = "fragMain";
    
    vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    auto bindingDescription = Particle::getBindingDescription();
    auto attributeDescriptions = Particle::getAttributeDescriptions();

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
    
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly;
    inputAssembly.topology = vk::PrimitiveTopology::ePointList;
    inputAssembly.primitiveRestartEnable = vk::False;
    
    vk::PipelineViewportStateCreateInfo viewportState;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    
    vk::PipelineRasterizationStateCreateInfo rasterizer;
    rasterizer.depthClampEnable = vk::False;
    rasterizer.rasterizerDiscardEnable = vk::False;
    rasterizer.polygonMode = vk::PolygonMode::eFill;
    rasterizer.cullMode = vk::CullModeFlagBits::eBack;
    rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
    rasterizer.depthBiasEnable = vk::False;
    rasterizer.lineWidth = 1.0f;
    
    vk::PipelineMultisampleStateCreateInfo multisampling;
    multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
    multisampling.sampleShadingEnable = vk::False;

    vk::PipelineColorBlendAttachmentState colorBlendAttachment;
    colorBlendAttachment.blendEnable = vk::True;
    colorBlendAttachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
    colorBlendAttachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    colorBlendAttachment.colorBlendOp = vk::BlendOp::eAdd;
    colorBlendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    colorBlendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eZero;
    colorBlendAttachment.alphaBlendOp = vk::BlendOp::eAdd;
    colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;

    vk::PipelineColorBlendStateCreateInfo colorBlending;
    colorBlending.logicOpEnable = vk::False;
    colorBlending.logicOp = vk::LogicOp::eCopy;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    std::vector dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor
    };
    vk::PipelineDynamicStateCreateInfo dynamicState;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo;
    pipelineRenderingCreateInfo.colorAttachmentCount = 1;
    pipelineRenderingCreateInfo.pColorAttachmentFormats = &swapChainImageFormat.format;
    
    vk::GraphicsPipelineCreateInfo pipelineInfo;
    pipelineInfo.pNext = &pipelineRenderingCreateInfo;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = *pipelineLayout;
    pipelineInfo.subpass = 0;

    graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
}

    void createComputePipeline() {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

        vk::PipelineShaderStageCreateInfo computeShaderStageInfo;
        computeShaderStageInfo.stage = vk::ShaderStageFlagBits::eCompute;
        computeShaderStageInfo.module = shaderModule;
        computeShaderStageInfo.pName = "compMain";
    
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &*computeDescriptorSetLayout;
        computePipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);
    
        vk::ComputePipelineCreateInfo pipelineInfo;
        pipelineInfo.stage = computeShaderStageInfo;
        pipelineInfo.layout = *computePipelineLayout;
        computePipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
    }

    void createCommandPool() {
        vk::CommandPoolCreateInfo poolInfo{};
        poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
        poolInfo.queueFamilyIndex = graphicsAndComputeIndex;
        commandPool = vk::raii::CommandPool(device, poolInfo);
    }

    void createShaderStorageBuffers() {
        // Initialize particles
        std::default_random_engine rndEngine(static_cast<unsigned>(time(nullptr)));
        std::uniform_real_distribution rndDist(0.0f, 1.0f);

        // Initial particle positions on a circle
        std::vector<Particle> particles(PARTICLE_COUNT);
        for (auto& particle : particles) {
            float r = 0.25f * sqrtf(rndDist(rndEngine));
            float theta = rndDist(rndEngine) * 2.0f * 3.14159265358979323846f;
            float x = r * cosf(theta) * HEIGHT / WIDTH;
            float y = r * sinf(theta);
            particle.position = glm::vec2(x, y);
            particle.velocity = normalize(glm::vec2(x,y)) * 0.00025f;
            particle.color = glm::vec4(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine), 1.0f);
        }

        vk::DeviceSize bufferSize = sizeof(Particle) * PARTICLE_COUNT;

        // Create a staging buffer used to upload data to the gpu
        vk::raii::Buffer stagingBuffer({});
        vk::raii::DeviceMemory stagingBufferMemory({});
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(dataStaging, particles.data(), (size_t)bufferSize);
        stagingBufferMemory.unmapMemory();

        shaderStorageBuffers.clear();
        shaderStorageBuffersMemory.clear();

        // Copy initial particle data to all storage buffers
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vk::raii::Buffer shaderStorageBufferTemp({});
            vk::raii::DeviceMemory shaderStorageBufferTempMemory({});
            createBuffer(bufferSize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal, shaderStorageBufferTemp, shaderStorageBufferTempMemory);
            copyBuffer(stagingBuffer, shaderStorageBufferTemp, bufferSize);
            shaderStorageBuffers.emplace_back(std::move(shaderStorageBufferTemp));
            shaderStorageBuffersMemory.emplace_back(std::move(shaderStorageBufferTempMemory));
        }
    }

    void createUniformBuffers() {
        uniformBuffers.clear();
        uniformBuffersMemory.clear();
        uniformBuffersMapped.clear();

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
            vk::raii::Buffer buffer({});
            vk::raii::DeviceMemory bufferMem({});
            createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
            uniformBuffers.emplace_back(std::move(buffer));
            uniformBuffersMemory.emplace_back(std::move(bufferMem));
            uniformBuffersMapped.emplace_back( uniformBuffersMemory[i].mapMemory(0, bufferSize));
        }
    }

    void createDescriptorPool() {
        std::array poolSize {
            vk::DescriptorPoolSize( vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT),
            vk::DescriptorPoolSize(  vk::DescriptorType::eStorageBuffer, MAX_FRAMES_IN_FLIGHT * 2)
        };
        vk::DescriptorPoolCreateInfo poolInfo{};
        poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
        poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;
        poolInfo.poolSizeCount = poolSize.size();
        poolInfo.pPoolSizes = poolSize.data();
        descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
    }

    void createComputeDescriptorSets() {
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, computeDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = *descriptorPool;
    allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    allocInfo.pSetLayouts = layouts.data();
    computeDescriptorSets.clear();
    computeDescriptorSets = device.allocateDescriptorSets(allocInfo);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::DescriptorBufferInfo bufferInfo(uniformBuffers[i], 0, sizeof(UniformBufferObject));

        vk::DescriptorBufferInfo storageBufferInfoLastFrame(shaderStorageBuffers[(i - 1) % MAX_FRAMES_IN_FLIGHT], 0, sizeof(Particle) * PARTICLE_COUNT);
        vk::DescriptorBufferInfo storageBufferInfoCurrentFrame(shaderStorageBuffers[i], 0, sizeof(Particle) * PARTICLE_COUNT);
        
        // Используем стандартный синтаксис конструктора вместо designated initializers
        std::array<vk::WriteDescriptorSet, 3> descriptorWrites = {{
            vk::WriteDescriptorSet(
                *computeDescriptorSets[i],          // dstSet
                0,                                  // dstBinding
                0,                                  // dstArrayElement
                1,                                  // descriptorCount
                vk::DescriptorType::eUniformBuffer, // descriptorType
                nullptr,                            // pImageInfo
                &bufferInfo,                        // pBufferInfo
                nullptr                             // pTexelBufferView
            ),
            vk::WriteDescriptorSet(
                *computeDescriptorSets[i],          // dstSet
                1,                                  // dstBinding
                0,                                  // dstArrayElement
                1,                                  // descriptorCount
                vk::DescriptorType::eStorageBuffer, // descriptorType
                nullptr,                            // pImageInfo
                &storageBufferInfoLastFrame,        // pBufferInfo
                nullptr                             // pTexelBufferView
            ),
            vk::WriteDescriptorSet(
                *computeDescriptorSets[i],          // dstSet
                2,                                  // dstBinding
                0,                                  // dstArrayElement
                1,                                  // descriptorCount
                vk::DescriptorType::eStorageBuffer, // descriptorType
                nullptr,                            // pImageInfo
                &storageBufferInfoCurrentFrame,     // pBufferInfo
                nullptr                             // pTexelBufferView
            )
        }};
        
            device.updateDescriptorSets(descriptorWrites, {});
        }
    }


    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory) const {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = vk::SharingMode::eExclusive;
        buffer = vk::raii::Buffer(device, bufferInfo);
        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
        bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
        buffer.bindMemory(bufferMemory, 0);
    }

    [[nodiscard]] vk::raii::CommandBuffer beginSingleTimeCommands() const {
        vk::CommandBufferAllocateInfo allocInfo{};
        allocInfo.commandPool = *commandPool;
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandBufferCount = 1;
        vk::raii::CommandBuffer commandBuffer = std::move(vk::raii::CommandBuffers(device, allocInfo).front());

        vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        commandBuffer.begin(beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(const vk::raii::CommandBuffer& commandBuffer) const {
        commandBuffer.end();

        vk::SubmitInfo submitInfo{};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &*commandBuffer;
        graphicsQueue.submit(submitInfo, nullptr);
        graphicsQueue.waitIdle();
    }

    void copyBuffer(const vk::raii::Buffer & srcBuffer, const vk::raii::Buffer & dstBuffer, vk::DeviceSize size) const {
        vk::raii::CommandBuffer commandCopyBuffer = beginSingleTimeCommands();
        commandCopyBuffer.copyBuffer(srcBuffer, dstBuffer, vk::BufferCopy(0, 0, size));
        endSingleTimeCommands(commandCopyBuffer);
    }

    [[nodiscard]] uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createCommandBuffers() {
        commandBuffers.clear();
        vk::CommandBufferAllocateInfo allocInfo{};
        allocInfo.commandPool = *commandPool;
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
        commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
    }

    void createComputeCommandBuffers() {
        computeCommandBuffers.clear();
        vk::CommandBufferAllocateInfo allocInfo{};
        allocInfo.commandPool = *commandPool;
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
        computeCommandBuffers = vk::raii::CommandBuffers(device, allocInfo);
    }

    void recordCommandBuffer(uint32_t imageIndex) {
    commandBuffers[currentFrame].reset();
    commandBuffers[currentFrame].begin({});
    
    // Before starting rendering, transition the swapchain image to COLOR_ATTACHMENT_OPTIMAL
    transition_image_layout(
        imageIndex,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal,
        {}, // srcAccessMask (no need to wait for previous operations)
        vk::AccessFlagBits2::eColorAttachmentWrite, // dstAccessMask
        vk::PipelineStageFlagBits2::eTopOfPipe, // srcStage
        vk::PipelineStageFlagBits2::eColorAttachmentOutput // dstStage
    );
    
    vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
    
    // Правильное создание vk::RenderingAttachmentInfo
    vk::RenderingAttachmentInfo attachmentInfo{};
    attachmentInfo.imageView = *swapChainImageViews[imageIndex]; // Разыменование raii объекта
    attachmentInfo.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
    attachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
    attachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
    attachmentInfo.clearValue = clearColor;
    
    // Создаем vk::Rect2D для renderArea
    vk::Rect2D renderArea(vk::Offset2D(0, 0), swapChainExtent);
    
    // Правильное создание vk::RenderingInfo
    vk::RenderingInfo renderingInfo{};
    renderingInfo.renderArea = renderArea;
    renderingInfo.layerCount = 1;
    renderingInfo.viewMask = 0;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &attachmentInfo;
    
    commandBuffers[currentFrame].beginRendering(renderingInfo);
    commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
    commandBuffers[currentFrame].setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffers[currentFrame].setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    commandBuffers[currentFrame].bindVertexBuffers(0, { *shaderStorageBuffers[currentFrame] }, {0}); // Разыменование raii объекта
    commandBuffers[currentFrame].draw(PARTICLE_COUNT, 1, 0, 0);
    commandBuffers[currentFrame].endRendering();
    
    // After rendering, transition the swapchain image to PRESENT_SRC
    transition_image_layout(
        imageIndex,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits2::eColorAttachmentWrite, // srcAccessMask
        {}, // dstAccessMask
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, // srcStage
        vk::PipelineStageFlagBits2::eBottomOfPipe // dstStage
    );
    
    commandBuffers[currentFrame].end();
}

        void transition_image_layout(
            uint32_t imageIndex,
            vk::ImageLayout old_layout,
    vk::ImageLayout new_layout,
    vk::AccessFlags2 src_access_mask,
    vk::AccessFlags2 dst_access_mask,
    vk::PipelineStageFlags2 src_stage_mask,
    vk::PipelineStageFlags2 dst_stage_mask
) {
    // Создаем subresourceRange
    vk::ImageSubresourceRange subresourceRange(
        vk::ImageAspectFlagBits::eColor,  // aspectMask
        0,                                // baseMipLevel
        1,                                // levelCount
        0,                                // baseArrayLayer
        1                                 // layerCount
    );
    
    // Используем обычный конструктор
    vk::ImageMemoryBarrier2 barrier(
        src_stage_mask,                   // srcStageMask
        src_access_mask,                  // srcAccessMask
        dst_stage_mask,                   // dstStageMask
        dst_access_mask,                  // dstAccessMask
        old_layout,                       // oldLayout
        new_layout,                       // newLayout
        VK_QUEUE_FAMILY_IGNORED,          // srcQueueFamilyIndex
        VK_QUEUE_FAMILY_IGNORED,          // dstQueueFamilyIndex
        swapChainImages[imageIndex],      // image
        subresourceRange                  // subresourceRange
    );
    
    vk::DependencyInfo dependency_info(
        {},                               // dependencyFlags
        0,                                // memoryBarrierCount
        nullptr,                          // pMemoryBarriers
        0,                                // bufferMemoryBarrierCount
        nullptr,                          // pBufferMemoryBarriers
        1,                                // imageMemoryBarrierCount
        &barrier                          // pImageMemoryBarriers
        );
    
        commandBuffers[currentFrame].pipelineBarrier2(dependency_info);
    }   

    void recordComputeCommandBuffer() {
        computeCommandBuffers[currentFrame].reset();
        computeCommandBuffers[currentFrame].begin({});
        computeCommandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline);
        computeCommandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, computePipelineLayout, 0, {computeDescriptorSets[currentFrame]}, {});
        computeCommandBuffers[currentFrame].dispatch( PARTICLE_COUNT / 256, 1, 1 );
        computeCommandBuffers[currentFrame].end();
    }

    void createSyncObjects() {
        inFlightFences.clear();
    
        vk::SemaphoreTypeCreateInfo semaphoreType{};
        semaphoreType.sType = vk::StructureType::eSemaphoreTypeCreateInfo;
    semaphoreType.pNext = nullptr;
    semaphoreType.semaphoreType = vk::SemaphoreType::eTimeline;
    semaphoreType.initialValue = 0;
    
    vk::SemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = vk::StructureType::eSemaphoreCreateInfo;
    semaphoreInfo.pNext = &semaphoreType;
    semaphoreInfo.flags = {};
    
    semaphore = vk::raii::Semaphore(device, semaphoreInfo);
    timelineValue = 0;
    
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::FenceCreateInfo fenceInfo{};
        fenceInfo.sType = vk::StructureType::eFenceCreateInfo;
        fenceInfo.pNext = nullptr;
        fenceInfo.flags = {};
        inFlightFences.emplace_back(device, fenceInfo);
    }
}

    void updateUniformBuffer(uint32_t currentImage) {
        UniformBufferObject ubo{};
        ubo.deltaTime = static_cast<float>(lastFrameTime) * 2.0f;

        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    void drawFrame() {
    auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, nullptr, *inFlightFences[currentFrame]);
    while ( vk::Result::eTimeout == device.waitForFences( *inFlightFences[currentFrame], vk::True, UINT64_MAX ));
    device.resetFences(*inFlightFences[currentFrame]);
    
    // Update timeline value for this frame
    uint64_t computeWaitValue = timelineValue;
    uint64_t computeSignalValue = ++timelineValue;
    uint64_t graphicsWaitValue = computeSignalValue;
    uint64_t graphicsSignalValue = ++timelineValue;
    
    updateUniformBuffer(currentFrame);
    
    {
        recordComputeCommandBuffer();
        
        // Submit compute work
        vk::TimelineSemaphoreSubmitInfo computeTimelineInfo{};
        computeTimelineInfo.sType = vk::StructureType::eTimelineSemaphoreSubmitInfo;
        computeTimelineInfo.pNext = nullptr;
        computeTimelineInfo.waitSemaphoreValueCount = 1;
        computeTimelineInfo.pWaitSemaphoreValues = &computeWaitValue;
        computeTimelineInfo.signalSemaphoreValueCount = 1;
        computeTimelineInfo.pSignalSemaphoreValues = &computeSignalValue;
        
        vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eComputeShader};
        
        vk::SubmitInfo computeSubmitInfo{};
        computeSubmitInfo.sType = vk::StructureType::eSubmitInfo;
        computeSubmitInfo.pNext = &computeTimelineInfo;
        computeSubmitInfo.waitSemaphoreCount = 1;
        computeSubmitInfo.pWaitSemaphores = &*semaphore;
        computeSubmitInfo.pWaitDstStageMask = waitStages;
        computeSubmitInfo.commandBufferCount = 1;
        computeSubmitInfo.pCommandBuffers = &*computeCommandBuffers[currentFrame];
        computeSubmitInfo.signalSemaphoreCount = 1;
        computeSubmitInfo.pSignalSemaphores = &*semaphore;
        
        computeQueue.submit(computeSubmitInfo, nullptr);
    }
    
    {
        // Record graphics command buffer
        recordCommandBuffer(imageIndex);
        
        // Submit graphics work (waits for compute to finish)
        vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eVertexInput;
        
        vk::TimelineSemaphoreSubmitInfo graphicsTimelineInfo{};
        graphicsTimelineInfo.sType = vk::StructureType::eTimelineSemaphoreSubmitInfo;
        graphicsTimelineInfo.pNext = nullptr;
        graphicsTimelineInfo.waitSemaphoreValueCount = 1;
        graphicsTimelineInfo.pWaitSemaphoreValues = &graphicsWaitValue;
        graphicsTimelineInfo.signalSemaphoreValueCount = 1;
        graphicsTimelineInfo.pSignalSemaphoreValues = &graphicsSignalValue;
        
        vk::SubmitInfo graphicsSubmitInfo{};
        graphicsSubmitInfo.sType = vk::StructureType::eSubmitInfo;
        graphicsSubmitInfo.pNext = &graphicsTimelineInfo;
        graphicsSubmitInfo.waitSemaphoreCount = 1;
        graphicsSubmitInfo.pWaitSemaphores = &*semaphore;
        graphicsSubmitInfo.pWaitDstStageMask = &waitStage;
        graphicsSubmitInfo.commandBufferCount = 1;
        graphicsSubmitInfo.pCommandBuffers = &*commandBuffers[currentFrame];
        graphicsSubmitInfo.signalSemaphoreCount = 1;
        graphicsSubmitInfo.pSignalSemaphores = &*semaphore;
        
        graphicsQueue.submit(graphicsSubmitInfo, nullptr);
        
        // Present the image (wait for graphics to finish)
        vk::SemaphoreWaitInfo waitInfo{};
        waitInfo.sType = vk::StructureType::eSemaphoreWaitInfo;
        waitInfo.pNext = nullptr;
        waitInfo.flags = {};
        waitInfo.semaphoreCount = 1;
        waitInfo.pSemaphores = &*semaphore;
        waitInfo.pValues = &graphicsSignalValue;
        
        // Wait for graphics to complete before presenting
        while ( vk::Result::eTimeout == device.waitSemaphores(waitInfo, UINT64_MAX) );
        
        vk::PresentInfoKHR presentInfo{};
        presentInfo.sType = vk::StructureType::ePresentInfoKHR;
        presentInfo.pNext = nullptr;
        presentInfo.waitSemaphoreCount = 0; // No binary semaphores needed
        presentInfo.pWaitSemaphores = nullptr;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &*swapChain;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr;
        
        result = presentQueue.presentKHR(presentInfo);
        if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else if (result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to present swap chain image!");
        }
    }
    
    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const {
        vk::ShaderModuleCreateInfo createInfo{};
        createInfo.sType = vk::StructureType::eShaderModuleCreateInfo;
        createInfo.pNext = nullptr;
        createInfo.flags = {};
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
    
        vk::raii::ShaderModule shaderModule{ device, createInfo };
        return shaderModule;
    }

    static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                return availablePresentMode;
            }
        }
        return vk::PresentModeKHR::eFifo;
    }

    [[nodiscard]] vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) const {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            return {
                std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
                std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
            };
            }

    [[nodiscard]] std::vector<const char*> getRequiredExtensions() const {
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        if (enableValidationLayers) {
            extensions.push_back(vk::EXTDebugUtilsExtensionName );
        }

        return extensions;
    }

    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type, const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void*) {
        if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError || severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
            std::cerr << "validation layer: type " << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;
        }

        return vk::False;
    }

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }
        std::vector<char> buffer(file.tellg());
        file.seekg(0, std::ios::beg);
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        file.close();

        return buffer;
    }
};

int main() {
    try {
        ComputeShaderApplication app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
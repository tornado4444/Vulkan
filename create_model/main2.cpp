// main.cpp
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
#define VK_USE_PLATFORM_XCB_KHR 
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

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader/tiny_obj_loader.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr uint64_t FenceTimeOut = 100000000;
const std::string MODEL_PATH = "models/viking_room.obj";
const std::string TEXTURE_PATH = "textures/viking_room.png";

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
    constexpr bool enableValidationLayers = false;
#else
    constexpr bool enableValidationLayers = true;
#endif

static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT severity, 
    vk::DebugUtilsMessageTypeFlagsEXT type, 
    const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, 
    void*) {
    
    std::cerr << "validation layer: type " << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;
    return vk::False;
}

std::vector<const char*> deviceExtensions = {
    vk::KHRSwapchainExtensionName,
    vk::KHRSpirv14ExtensionName,
    vk::KHRSynchronization2ExtensionName,
    vk::KHRCreateRenderpass2ExtensionName
};

static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + filename);
    }

    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

std::vector dynamicStates = {
    vk::DynamicState::eViewport,
    vk::DynamicState::eScissor
};

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    bool operator==(const Vertex& other) const {
        return pos == other.pos && color == other.color && texCoord == other.texCoord;
    }


    static vk::VertexInputBindingDescription getBindingDescription() {
        return { 0, sizeof(Vertex), vk::VertexInputRate::eVertex };
    }

    static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
        return {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)),
            vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord))
        };
    }
};

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.pos) ^
                   (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
                   (hash<glm::vec2>()(vertex.texCoord) << 1);
        }
    };
}

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

class Vulkan {
public:
    Vulkan() = default;

public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window;
    vk::raii::Context context;
    std::unique_ptr<vk::raii::Device> device;
    std::unique_ptr<vk::raii::Instance> instance;
    std::unique_ptr<vk::raii::DebugUtilsMessengerEXT> debugMessenger;
    std::unique_ptr<vk::raii::PhysicalDevice> physicalDevice;
    std::unique_ptr<vk::raii::SurfaceKHR> surface;
    vk::PhysicalDeviceFeatures deviceFeatures;
    
    uint32_t graphicsIndex;
    uint32_t presentIndex;
    std::unique_ptr<vk::raii::CommandPool> commandPool;
    std::unique_ptr<vk::raii::Queue> graphicsQueue;
    std::unique_ptr<vk::raii::Queue> presentQueue;

    std::unique_ptr<vk::raii::SwapchainKHR> swapChain;
    std::vector<vk::raii::Image> swapChainImages;
    vk::Format swapChainImageFormat;
    vk::Extent2D swapChainExtent;
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
    std::vector<vk::raii::ImageView> swapChainImageViews;
    std::unique_ptr<vk::raii::PipelineLayout> pipelineLayout;
    std::unique_ptr<vk::raii::Pipeline> graphicsPipeline;

    std::unique_ptr<vk::raii::RenderPass> renderPass;
    std::vector<vk::raii::Framebuffer> swapChainFramebuffers;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::unique_ptr<vk::raii::Buffer> vertexBuffer = nullptr;
    std::unique_ptr<vk::raii::DeviceMemory> vertexBufferMemory = nullptr;
    std::unique_ptr<vk::raii::Buffer> indexBuffer = nullptr;
    std::unique_ptr<vk::raii::DeviceMemory> indexBufferMemory = nullptr;

    std::vector<std::unique_ptr<vk::raii::Buffer>> uniformBuffers;
    std::vector<std::unique_ptr<vk::raii::DeviceMemory>> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    std::unique_ptr<vk::raii::DescriptorPool> descriptorPool;

    std::vector<vk::raii::CommandBuffer> commandBuffers;
    std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
    std::vector<vk::raii::Fence> inFlightFences;

    std::unique_ptr<vk::raii::DescriptorSetLayout> descriptorSetLayout;
    std::vector<vk::raii::DescriptorSet> descriptorSets;

    uint32_t mipLevels;
    std::unique_ptr<vk::raii::Image> textureImage;
    std::unique_ptr<vk::raii::DeviceMemory> textureImageMemory;
    std::unique_ptr<vk::raii::ImageView> textureImageView;
    std::unique_ptr<vk::raii::Sampler> textureSampler;
    
    std::unique_ptr<vk::raii::Image> depthImage;
    std::unique_ptr<vk::raii::DeviceMemory> depthImageMemory;
    std::unique_ptr<vk::raii::ImageView> depthImageView;
    vk::Format depthFormat;
    std::unique_ptr<vk::raii::Buffer> stagingBuffer;

    vk::SampleCountFlagBits msaaSamples = vk::SampleCountFlagBits::e1;

    std::unique_ptr<vk::raii::Image> colorImage = nullptr;
    std::unique_ptr<vk::raii::DeviceMemory> colorImageMemory = nullptr;
    std::unique_ptr<vk::raii::ImageView> colorImageView = nullptr;

    const int MAX_FRAMES_IN_FLIGHT = 2;
    uint32_t currentFrame = 0;
    bool framebufferResized = false;
    
    void initWindow() {     
        glfwInit();
        
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height){
        auto app = reinterpret_cast<Vulkan*>(glfwGetWindowUserPointer(window));
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
        createCommandPool();
    
        depthFormat = findDepthFormat();  
        msaaSamples = getMaxUsableSampleCount();  
        
        createColorResources();  
        createDepthResources();  
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipelines(); 
        createFramebuffers();             
        createTextureImage();       
        createTextureImageView();
        createTextureSampler();
        loadModel();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObject();
    }

    void createInstance() {
        if(enableValidationLayers && !checkValidationLayerSupport()){
            throw std::runtime_error("validation layers requested, but not available!");
        }

        try {
            vk::ApplicationInfo appInfo{};
            appInfo.pApplicationName = "Vulkan";
            appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
            appInfo.pEngineName = "No Engine";
            appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
            appInfo.apiVersion = VK_API_VERSION_1_3;

            auto extensions = getRequiredExtensions();

            std::cout << "Available extensions:\n";
            auto availableExtensions = context.enumerateInstanceExtensionProperties();
            for (const auto& extension : availableExtensions) {
                std::cout << '\t' << extension.extensionName << '\n';
            }

            vk::InstanceCreateInfo createInfo{};
            createInfo.pApplicationInfo = &appInfo;
            createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
            createInfo.ppEnabledExtensionNames = extensions.data();

            if (enableValidationLayers) {
                createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
                createInfo.ppEnabledLayerNames = validationLayers.data();
            } else {
                createInfo.enabledLayerCount = 0;
            }

            instance = std::make_unique<vk::raii::Instance>(context, createInfo);
            std::cout << "Vulkan instance created successfully!\n";
            
        } catch (const vk::SystemError& err) {
            std::cerr << "Vulkan system error: " << err.what() << std::endl;
            throw std::runtime_error("Failed to create Vulkan instance");
        } catch (const std::exception& err) {
            std::cerr << "General error: " << err.what() << std::endl;
            throw std::runtime_error("Failed to create Vulkan instance");
        }
    }

    QueueFamilyIndices findQueueFamilies(vk::raii::PhysicalDevice& device) {
        QueueFamilyIndices indices;

        std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
                indices.graphicsFamily = i;
            }

            if (device.getSurfaceSupportKHR(i, **surface)) {
                indices.presentFamily = i;
            }

            if (indices.isComplete()) {
                break;
            }

            i++;
        }

        return indices;
    }

    bool checkValidationLayerSupport() {
        auto availableLayers = context.enumerateInstanceLayerProperties();
        
        for (const char* layerName : validationLayers) {
            bool layerFound = false;
            
            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }
            
            if (!layerFound) {
                return false;
            }
        }
        
        return true;
    }

    std::vector<const char*> getRequiredExtensions() { 
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        
        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    void setupDebugMessenger() {
        if(!enableValidationLayers) return;
        
        vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | 
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | 
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
            
        vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags( 
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | 
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | 
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);

        vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT(
            {}, severityFlags, messageTypeFlags, &debugCallback);
            
        debugMessenger = std::make_unique<vk::raii::DebugUtilsMessengerEXT>(
            *instance, debugUtilsMessengerCreateInfoEXT);
    }

    bool checkDeviceExtensionSupport(vk::raii::PhysicalDevice&device) {
        std::vector<vk::ExtensionProperties> availableExtensions = device.enumerateDeviceExtensionProperties();

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    SwapChainSupportDetails querySwapChainSupport(vk::raii::PhysicalDevice& device) {
        SwapChainSupportDetails details;

        details.capabilities = device.getSurfaceCapabilitiesKHR(**surface);
        details.formats = device.getSurfaceFormatsKHR(**surface);
        details.presentModes = device.getSurfacePresentModesKHR(**surface);

        return details;
    }


    void createSurface() {
        VkSurfaceKHR _surface;
        if (glfwCreateWindowSurface(**instance, window, nullptr, &_surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
        surface = std::make_unique<vk::raii::SurfaceKHR>(*instance, _surface);
        }

        bool isDeviceSuitable(vk::raii::PhysicalDevice& device) {
            auto indices = findQueueFamilies(device);
    
            bool extensionsSupported = checkDeviceExtensionSupport(device);
    
            bool swapChainAdequate = false;
                if (extensionsSupported) {
                auto swapChainSupport = querySwapChainSupport(device);
                swapChainAdequate = !swapChainSupport.formats.empty() && 
                           !swapChainSupport.presentModes.empty();
        }
    
            auto supportedFeatures = device.getFeatures();
    
            return indices.isComplete() && 
             extensionsSupported && 
            swapChainAdequate && 
            supportedFeatures.samplerAnisotropy;
    }


    void pickPhysicalDevice() {
        auto devices = vk::raii::PhysicalDevices(*instance);
        if (devices.empty()) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        for (const auto& device : devices) {
            vk::raii::PhysicalDevice deviceCopy = device;
        
            if (isDeviceSuitable(deviceCopy)) {
                physicalDevice = std::make_unique<vk::raii::PhysicalDevice>(std::move(deviceCopy));
                msaaSamples = getMaxUsableSampleCount();
                return;
            }
        }

        std::multimap<int, vk::raii::PhysicalDevice> candidates;

        for (const auto& device : devices) {
            auto deviceProperties = device.getProperties();
            auto deviceFeatures = device.getFeatures();
            uint32_t score = 0;

            if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
                score += 1000;
            }

            score += deviceProperties.limits.maxImageDimension2D;

            if (!deviceFeatures.geometryShader) {
                continue;
            }
        
            candidates.insert(std::make_pair(score, device));
        }

        if (!candidates.empty() && candidates.rbegin()->first > 0) {
            vk::raii::PhysicalDevice bestDevice = candidates.rbegin()->second;
            physicalDevice = std::make_unique<vk::raii::PhysicalDevice>(std::move(bestDevice));
        } else {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    void createLogicalDevice() {
        std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice->getQueueFamilyProperties();
        auto graphicsQueueFamilyProperty = std::find_if(queueFamilyProperties.begin(),
                                                        queueFamilyProperties.end(),
                                                        [](auto const & qfp) {
                                                            return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0);
                                                        });
        
        QueueFamilyIndices indices = findQueueFamilies(*physicalDevice);
        graphicsIndex = static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), graphicsQueueFamilyProperty));
        presentIndex = physicalDevice->getSurfaceSupportKHR(graphicsIndex, **surface)
                                           ? graphicsIndex
                                           : static_cast<uint32_t>(queueFamilyProperties.size());
        if (presentIndex == queueFamilyProperties.size()) {
            for (size_t i = 0; i < queueFamilyProperties.size(); i++) {
                if ((queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
                     physicalDevice->getSurfaceSupportKHR(static_cast<uint32_t>(i), **surface)) {
                    graphicsIndex = static_cast<uint32_t>(i);
                    presentIndex = graphicsIndex;
                    break;
                }
            }
            if (presentIndex == queueFamilyProperties.size()) {
                for (size_t i = 0; i < queueFamilyProperties.size(); i++) {
                    if (physicalDevice->getSurfaceSupportKHR(static_cast<uint32_t>(i), **surface)) {
                        presentIndex = static_cast<uint32_t>(i);
                        break;
                    }
                }
            }
        }
        if ((graphicsIndex == queueFamilyProperties.size()) || (presentIndex == queueFamilyProperties.size())) {
            throw std::runtime_error("Could not find a queue for graphics or present -> terminating");
        }

        auto features = physicalDevice->getFeatures2();
        vk::PhysicalDeviceVulkan13Features vulkan13Features;
        vk::PhysicalDeviceSynchronization2Features sync2Features;
        

        vulkan13Features.dynamicRendering = vk::True;
        sync2Features.synchronization2 = vk::True; 
        sync2Features.pNext = nullptr;
        vulkan13Features.pNext = &sync2Features;
        features.pNext = &vulkan13Features;
        
        float queuePriority = 0.0f;
        vk::DeviceQueueCreateInfo deviceQueueCreateInfo{};
        deviceQueueCreateInfo.queueFamilyIndex = graphicsIndex;
        deviceQueueCreateInfo.queueCount = 1;
        deviceQueueCreateInfo.pQueuePriorities = &queuePriority;
        
        vk::DeviceCreateInfo deviceCreateInfo{};
        deviceCreateInfo.pNext = &features;
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
        deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

        device = std::make_unique<vk::raii::Device>(*physicalDevice, deviceCreateInfo);
        graphicsQueue = std::make_unique<vk::raii::Queue>(*device, graphicsIndex, 0);
        presentQueue = std::make_unique<vk::raii::Queue>(*device, presentIndex, 0);
    }

    uint32_t findQueueFamilies() {
        std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice->getQueueFamilyProperties();

        auto graphicsQueueFamilyProperty =
        std::find_if( queueFamilyProperties.begin(),
                       queueFamilyProperties.end(),
                       []( vk::QueueFamilyProperties const & qfp ) { return qfp.queueFlags & vk::QueueFlagBits::eGraphics; } );

        return static_cast<uint32_t>( std::distance( queueFamilyProperties.begin(), graphicsQueueFamilyProperty ) );
    }

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
        for(const auto& availableFormat : availableFormats){
            if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                return availablePresentMode;
            }
        }
        return vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        } else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            vk::Extent2D actualExtent = {
              static_cast<uint32_t>(width),
             static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width, 
                capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, 
                capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const {
        vk::ShaderModuleCreateInfo createInfo{};
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
    
        return vk::raii::ShaderModule(*device, createInfo);
    }

    void createSwapChain() {
        auto surfaceCapabilities = physicalDevice->getSurfaceCapabilitiesKHR(**surface);
        auto availableFormats = physicalDevice->getSurfaceFormatsKHR(**surface);
        auto availablePresentModes = physicalDevice->getSurfacePresentModesKHR(**surface);

        auto surfaceFormat = chooseSwapSurfaceFormat(availableFormats);
        auto presentMode = chooseSwapPresentMode(availablePresentModes);
        auto extent = chooseSwapExtent(surfaceCapabilities);

        uint32_t imageCount = surfaceCapabilities.minImageCount + 1;
        if (surfaceCapabilities.maxImageCount > 0 && imageCount > surfaceCapabilities.maxImageCount) {
            imageCount = surfaceCapabilities.maxImageCount;
        }

        vk::SwapchainCreateInfoKHR createInfo{};
        createInfo.surface = **surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

        uint32_t queueFamilyIndices[] = {graphicsIndex, presentIndex};
        
        if (graphicsIndex != presentIndex) {
            createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode = vk::SharingMode::eExclusive;
            createInfo.queueFamilyIndexCount = 0;
            createInfo.pQueueFamilyIndices = nullptr;
        }

        createInfo.preTransform = surfaceCapabilities.currentTransform;
        createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
        createInfo.presentMode = presentMode;
        createInfo.clipped = vk::True;
        createInfo.oldSwapchain = nullptr;

        swapChain = std::make_unique<vk::raii::SwapchainKHR>(*device, createInfo);
        
        auto rawImages = swapChain->getImages();
        swapChainImages.clear();
        swapChainImages.reserve(rawImages.size());
        for (const auto& img : rawImages) {
            swapChainImages.emplace_back(*device, img);
        }
        
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;

        std::cout << "Swap chain created successfully with " << swapChainImages.size() << " images!\n";
    }

    void createImageViews() {
        swapChainImageViews.clear();
        
        swapChainImageViews.reserve(swapChainImages.size());

        vk::ImageViewCreateInfo imageViewCreateInfo{};
        imageViewCreateInfo.viewType = vk::ImageViewType::e2D;
        imageViewCreateInfo.format = swapChainImageFormat;
        
        imageViewCreateInfo.components.r = vk::ComponentSwizzle::eIdentity;
        imageViewCreateInfo.components.g = vk::ComponentSwizzle::eIdentity;
        imageViewCreateInfo.components.b = vk::ComponentSwizzle::eIdentity;
        imageViewCreateInfo.components.a = vk::ComponentSwizzle::eIdentity;
        
        imageViewCreateInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
        imageViewCreateInfo.subresourceRange.levelCount = 1;
        imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
        imageViewCreateInfo.subresourceRange.layerCount = 1;

        for (const auto& image : swapChainImages) {
            imageViewCreateInfo.image = image;
            swapChainImageViews.emplace_back(*device, imageViewCreateInfo);
        }
        
        std::cout << "Created " << swapChainImageViews.size() << " image views successfully!\n";
    }

    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(*physicalDevice);
        vk::CommandPoolCreateInfo poolInfo{};
        poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
        poolInfo.queueFamilyIndex = graphicsIndex;

        commandPool = std::make_unique<vk::raii::CommandPool>(*device, poolInfo);
    }

    vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features) {
        for (const auto format : candidates) {
            vk::FormatProperties props = physicalDevice->getFormatProperties(format);

            if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
                return format;
            }
            if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }

        throw std::runtime_error("failed to find supported format!");
    }

    vk::Format findDepthFormat() {  
        return findSupportedFormat(
            {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
            vk::ImageTiling::eOptimal,
            vk::FormatFeatureFlagBits::eDepthStencilAttachment
        );
    }

    [[nodiscard]]vk::raii::ImageView createImageView(vk::raii::Image& image, 
                                   vk::Format format, 
                                   vk::ImageAspectFlags aspectFlags, uint32_t mipLevels) {
        vk::ImageViewCreateInfo viewInfo{};
        viewInfo.image = *image;
        viewInfo.viewType = vk::ImageViewType::e2D;
        viewInfo.format = format;
    
        viewInfo.subresourceRange.aspectMask = aspectFlags;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = mipLevels;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
    
        return vk::raii::ImageView(*device, viewInfo);
    }

    void createDepthResources() {
        vk::Format depthFormat = findDepthFormat();
    
        createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, depthFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage, depthImageMemory);

        transitionImageLayout(depthImage, depthFormat, 
                     vk::ImageLayout::eUndefined,
                     vk::ImageLayout::eDepthStencilAttachmentOptimal, 1);
    
        depthImageView = std::make_unique<vk::raii::ImageView>(
        createImageView(*depthImage, 
                       depthFormat, 
                       vk::ImageAspectFlagBits::eDepth, 1)
        );
    }

    void updateImageViewCalls() {
        for (uint32_t i = 0; i < swapChainImages.size(); i++) {
        swapChainImageViews[i] = createImageView(swapChainImages[i], 
                                               swapChainImageFormat, 
                                               vk::ImageAspectFlagBits::eColor, 1);
        }
    
        textureImageView = std::make_unique<vk::raii::ImageView>(
        createImageView(*textureImage, 
                                      vk::Format::eR8G8B8A8Srgb, 
                                      vk::ImageAspectFlagBits::eColor, 1)
        );
    }

    void createCommandBuffers() {
        commandBuffers.clear();
        vk::CommandBufferAllocateInfo allocInfo{};
        allocInfo.commandPool = **commandPool;
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;

        auto tempCommandBuffers = vk::raii::CommandBuffers(*device, allocInfo);

        for (auto& cmdBuffer : tempCommandBuffers) {
            commandBuffers.emplace_back(std::move(cmdBuffer));
        }
    }

    std::unique_ptr<vk::raii::CommandBuffer> beginSingleTimeCommands() {
        vk::CommandBufferAllocateInfo allocInfo(*commandPool, vk::CommandBufferLevel::ePrimary, 1);
        auto commandBuffers = device->allocateCommandBuffers(allocInfo);
        auto commandBuffer = std::make_unique<vk::raii::CommandBuffer>(std::move(commandBuffers.front()));
    
        vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        commandBuffer->begin(beginInfo);
    
        return commandBuffer;
    }

    void endSingleTimeCommands(std::unique_ptr<vk::raii::CommandBuffer> commandBuffer) {
        commandBuffer->end();
    
        vk::SubmitInfo submitInfo({}, {}, {**commandBuffer});
        graphicsQueue->submit(submitInfo, nullptr);
        graphicsQueue->waitIdle();
    }

    void generateMipMaps(vk::raii::Image& image, vk::Format imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) {
        vk::FormatProperties formatProperties = physicalDevice->getFormatProperties(imageFormat);
        if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
            throw std::runtime_error("texture image format does not support linear blitting!");
        }

        auto commandBuffer = beginSingleTimeCommands();

        vk::ImageMemoryBarrier barrier{};
        barrier.image = *image;
        barrier.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
        barrier.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange.levelCount = mipLevels;

        int32_t mipWidth = texWidth;
        int32_t mipHeight = texHeight;

        for (uint32_t i = 1; i < mipLevels; i++) {
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
            barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

            commandBuffer->pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {},
                {}, {}, barrier);

            vk::ImageBlit blit{};
            blit.srcOffsets[0] = vk::Offset3D{0, 0, 0};
            blit.srcOffsets[1] = vk::Offset3D{mipWidth, mipHeight, 1};
            blit.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
            blit.srcSubresource.mipLevel = i - 1;
            blit.srcSubresource.baseArrayLayer = 0;
            blit.srcSubresource.layerCount = 1;
            blit.dstOffsets[0] = vk::Offset3D{0, 0, 0};
            blit.dstOffsets[1] = vk::Offset3D{mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1};
            blit.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
            blit.dstSubresource.mipLevel = i;
            blit.dstSubresource.baseArrayLayer = 0;
            blit.dstSubresource.layerCount = 1;

            commandBuffer->blitImage(
                *image, vk::ImageLayout::eTransferSrcOptimal,
                *image, vk::ImageLayout::eTransferDstOptimal,
                blit, vk::Filter::eLinear);

            barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
            barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            commandBuffer->pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {},
                {}, {}, barrier);

            if (mipWidth > 1) mipWidth /= 2;
            if (mipHeight > 1) mipHeight /= 2;
        }

        barrier.subresourceRange.baseMipLevel = mipLevels - 1;
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        commandBuffer->pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {},
            {}, {}, barrier);

        endSingleTimeCommands(std::move(commandBuffer));    
    }

    void createTextureImage() {
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        vk::DeviceSize imageSize = texWidth * texHeight * 4;
        mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }

        std::unique_ptr<vk::raii::Buffer> stagingBuffer;
        std::unique_ptr<vk::raii::DeviceMemory> stagingBufferMemory;

        createBuffer(imageSize, 
                vk::BufferUsageFlagBits::eTransferSrc, 
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, 
                stagingBuffer, 
                stagingBufferMemory);

        void* data = stagingBufferMemory->mapMemory(0, imageSize);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
        stagingBufferMemory->unmapMemory();

        stbi_image_free(pixels);

        createImage(texWidth, 
            texHeight, 
            mipLevels, 
            vk::SampleCountFlagBits::e1, 
            vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal, 
            vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, 
            vk::MemoryPropertyFlagBits::eDeviceLocal, 
            textureImage, 
            textureImageMemory);

        transitionImageLayout(textureImage, 
                vk::Format::eR8G8B8A8Srgb,
                vk::ImageLayout::eUndefined, 
                vk::ImageLayout::eTransferDstOptimal, 
                mipLevels);

        copyBufferToImage(*stagingBuffer, textureImage, 
                    static_cast<uint32_t>(texWidth), 
                    static_cast<uint32_t>(texHeight));

        generateMipMaps(*textureImage, vk::Format::eR8G8B8A8Srgb, texWidth, texHeight, mipLevels);
    }

    void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, vk::SampleCountFlagBits numSamples, 
                 vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, 
                 vk::MemoryPropertyFlags properties, std::unique_ptr<vk::raii::Image>& image,
                 std::unique_ptr<vk::raii::DeviceMemory>& imageMemory) const {
    
        vk::ImageCreateInfo imageInfo{};
        imageInfo.imageType = vk::ImageType::e2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = mipLevels;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = vk::ImageLayout::eUndefined;
        imageInfo.usage = usage;
        imageInfo.samples = numSamples; 
        imageInfo.sharingMode = vk::SharingMode::eExclusive;

        image = std::make_unique<vk::raii::Image>(*device, imageInfo);

        vk::MemoryRequirements memRequirements = image->getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        imageMemory = std::make_unique<vk::raii::DeviceMemory>(*device, allocInfo);
        image->bindMemory(**imageMemory, 0);
    }


    void createColorResources() {
        vk::Format colorFormat = swapChainImageFormat;
    
        createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, 
                colorFormat, 
                vk::ImageTiling::eOptimal, 
                vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,  
                vk::MemoryPropertyFlagBits::eDeviceLocal, 
                colorImage, colorImageMemory);
                
        colorImageView = std::make_unique<vk::raii::ImageView>(
            createImageView(*colorImage, colorFormat, vk::ImageAspectFlagBits::eColor, 1)
    );
    }

    void createTextureImageView() {
        vk::ImageViewCreateInfo viewInfo{};
        viewInfo.image = **textureImage;
        viewInfo.viewType = vk::ImageViewType::e2D;
        viewInfo.format = vk::Format::eR8G8B8A8Srgb;
        viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = mipLevels;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        int texWidth, texHeight;

        textureImageView = std::make_unique<vk::raii::ImageView>(*device, viewInfo);
    }

    void createTextureSampler() {
        vk::PhysicalDeviceProperties properties = physicalDevice->getProperties();

        vk::SamplerCreateInfo samplerInfo{};
    
        samplerInfo.magFilter = vk::Filter::eLinear;
    
        samplerInfo.minFilter = vk::Filter::eLinear;
    
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    
        samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
    
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.anisotropyEnable = vk::True;
    
        samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    
        samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
        samplerInfo.unnormalizedCoordinates = vk::False;
    
        samplerInfo.compareEnable = vk::False;
        samplerInfo.compareOp = vk::CompareOp::eAlways;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = static_cast<float>(mipLevels);
        textureSampler = std::make_unique<vk::raii::Sampler>(*device, samplerInfo);
    }

    vk::SampleCountFlagBits getMaxUsableSampleCount() {
        vk::PhysicalDeviceProperties physicalDeviceProperties = physicalDevice->getProperties();

        vk::SampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
        if (counts & vk::SampleCountFlagBits::e64) { return vk::SampleCountFlagBits::e64; }
        if (counts & vk::SampleCountFlagBits::e32) { return vk::SampleCountFlagBits::e32; }
        if (counts & vk::SampleCountFlagBits::e16) { return vk::SampleCountFlagBits::e16; }
        if (counts & vk::SampleCountFlagBits::e8) { return vk::SampleCountFlagBits::e8; }
        if (counts & vk::SampleCountFlagBits::e4) { return vk::SampleCountFlagBits::e4; }
        if (counts & vk::SampleCountFlagBits::e2) { return vk::SampleCountFlagBits::e2; }

        return vk::SampleCountFlagBits::e1;
    }

    bool hasStencilComponent(vk::Format format) {
        return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
    }

    void transitionImageLayout(std::unique_ptr<vk::raii::Image>& image, vk::Format format,
                          vk::ImageLayout oldLayout, vk::ImageLayout newLayout, uint32_t mipLevels) {
        vk::CommandBufferAllocateInfo allocInfo{};
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandPool = **commandPool;
        allocInfo.commandBufferCount = 1;
        vk::raii::CommandBuffer commandBuffer = std::move(device->allocateCommandBuffers(allocInfo).front());
    
        vk::CommandBufferBeginInfo beginInfo{};
        beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
        commandBuffer.begin(beginInfo);
    
        vk::ImageMemoryBarrier barrier{};
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = vk::QueueFamilyIgnored;  
        barrier.dstQueueFamilyIndex = vk::QueueFamilyIgnored;  
        barrier.image = **image;
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
    
        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;
    
        if (newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
            barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
            if (hasStencilComponent(format)) {
                barrier.subresourceRange.aspectMask |= vk::ImageAspectFlagBits::eStencil;  
            }
        } else {
            barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor; 
        }
    
        if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;  
            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;        
            destinationStage = vk::PipelineStageFlagBits::eTransfer;    
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
            sourceStage = vk::PipelineStageFlagBits::eTransfer;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        } else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }
    
        commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, {}, barrier);
        commandBuffer.end();
    
        vk::SubmitInfo submitInfo{};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &*commandBuffer;
        graphicsQueue->submit(submitInfo, nullptr);
        graphicsQueue->waitIdle();
    }

    void copyBufferToImage(vk::raii::Buffer& buffer, std::unique_ptr<vk::raii::Image>& image, 
                      uint32_t width, uint32_t height) {
        vk::CommandBufferAllocateInfo allocInfo{};
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandPool = **commandPool;
        allocInfo.commandBufferCount = 1;

        vk::raii::CommandBuffer commandBuffer = std::move(device->allocateCommandBuffers(allocInfo).front());

        vk::CommandBufferBeginInfo beginInfo{};
        beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
        commandBuffer.begin(beginInfo);

        vk::BufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = vk::Offset3D{0, 0, 0};
        region.imageExtent = vk::Extent3D{width, height, 1};

        commandBuffer.copyBufferToImage(*buffer, **image, vk::ImageLayout::eTransferDstOptimal, region);

        commandBuffer.end();

        vk::SubmitInfo submitInfo{};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &*commandBuffer;


        graphicsQueue->submit(submitInfo, nullptr);
        graphicsQueue->waitIdle();
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice->getMemoryProperties();
    
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && 
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
    
        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, 
                  std::unique_ptr<vk::raii::Buffer>& buffer, std::unique_ptr<vk::raii::DeviceMemory>& bufferMemory) {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = vk::SharingMode::eExclusive;
    
        buffer = std::make_unique<vk::raii::Buffer>(*device, bufferInfo);
    
        vk::MemoryRequirements memRequirements = buffer->getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
    
        bufferMemory = std::make_unique<vk::raii::DeviceMemory>(*device, allocInfo);
        buffer->bindMemory(**bufferMemory, 0);
    }

    void copyBuffer(vk::raii::Buffer & srcBuffer, vk::raii::Buffer & dstBuffer, vk::DeviceSize size){
        vk::CommandBufferAllocateInfo allocInfo{};
        allocInfo.commandPool = *commandPool;
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandBufferCount = 1;
        vk::raii::CommandBuffer commandCopyBuffer = std::move(device->allocateCommandBuffers(allocInfo).front());
        vk::CommandBufferBeginInfo beginInfo{};
        beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
        commandCopyBuffer.begin(beginInfo);

        commandCopyBuffer.copyBuffer(srcBuffer, dstBuffer, vk::BufferCopy(0, 0, size));
        commandCopyBuffer.end();

        vk::SubmitInfo submitInfo{};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &*commandCopyBuffer;

        graphicsQueue->submit(submitInfo, nullptr);
        graphicsQueue->waitIdle();
    }

    void loadModel() {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;
        std::unordered_map<Vertex, uint32_t> uniqueVertices{};

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str())) {
            throw std::runtime_error(warn + err);
        }

        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                Vertex vertex{};

                vertex.pos = {
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2]
                };


                vertex.texCoord = {
                    attrib.texcoords[2 * index.texcoord_index + 0],
                    1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                };

                vertex.color = {1.0f, 1.0f, 1.0f};

                vertices.push_back(vertex);
                indices.push_back(indices.size());
            }
        }
    }

    void createVertexBuffer() {
        vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        vk::BufferCreateInfo stagingBufferInfo{};
        stagingBufferInfo.size = bufferSize;
        stagingBufferInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;
        stagingBufferInfo.sharingMode = vk::SharingMode::eExclusive;
    
        auto stagingBuffer = std::make_unique<vk::raii::Buffer>(*device, stagingBufferInfo);
    
        vk::MemoryRequirements stagingMemRequirements = stagingBuffer->getMemoryRequirements();
        vk::MemoryAllocateInfo stagingAllocInfo{};
        stagingAllocInfo.allocationSize = stagingMemRequirements.size;
        stagingAllocInfo.memoryTypeIndex = findMemoryType(
        stagingMemRequirements.memoryTypeBits,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        );
    
        auto stagingBufferMemory = std::make_unique<vk::raii::DeviceMemory>(*device, stagingAllocInfo);
        stagingBuffer->bindMemory(**stagingBufferMemory, 0);

        void* data = stagingBufferMemory->mapMemory(0, bufferSize);
        memcpy(data, vertices.data(), static_cast<size_t>(bufferSize));
        stagingBufferMemory->unmapMemory();

        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = bufferSize;
        bufferInfo.usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst;
        bufferInfo.sharingMode = vk::SharingMode::eExclusive;
    
        vertexBuffer = std::make_unique<vk::raii::Buffer>(*device, bufferInfo);
    
        vk::MemoryRequirements memRequirements = vertexBuffer->getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(
        memRequirements.memoryTypeBits,
        vk::MemoryPropertyFlagBits::eDeviceLocal
        );
    
        vertexBufferMemory = std::make_unique<vk::raii::DeviceMemory>(*device, allocInfo);
        vertexBuffer->bindMemory(**vertexBufferMemory, 0);

        copyBuffer(*stagingBuffer, *vertexBuffer, bufferSize);
    }

    void createIndexBuffer(){
        vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        std::unique_ptr<vk::raii::Buffer> stagingBuffer;
        std::unique_ptr<vk::raii::DeviceMemory> stagingBufferMemory;
    
        createBuffer(bufferSize, 
                vk::BufferUsageFlagBits::eTransferSrc, 
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, 
                stagingBuffer, 
                stagingBufferMemory);

        void* data = stagingBufferMemory->mapMemory(0, bufferSize);
        memcpy(data, indices.data(), static_cast<size_t>(bufferSize));
        stagingBufferMemory->unmapMemory();

        createBuffer(bufferSize, 
                vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, 
                vk::MemoryPropertyFlagBits::eDeviceLocal, 
                indexBuffer, 
                indexBufferMemory);

        copyBuffer(*stagingBuffer, *indexBuffer, bufferSize);
    }

    void createUniformBuffers(){
        uniformBuffers.clear();
        uniformBuffersMemory.clear();
        uniformBuffersMapped.clear();
    
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
        
            std::unique_ptr<vk::raii::Buffer> buffer = std::make_unique<vk::raii::Buffer>(nullptr);
            std::unique_ptr<vk::raii::DeviceMemory> bufferMem = std::make_unique<vk::raii::DeviceMemory>(nullptr);
        
            createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, 
                    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, 
                    buffer, bufferMem);
        
            uniformBuffers.emplace_back(std::move(buffer));
            uniformBuffersMemory.emplace_back(std::move(bufferMem));
            uniformBuffersMapped.emplace_back(uniformBuffersMemory[i]->mapMemory(0, bufferSize));
        }
    }
    
    void createDescriptorPool() {
        std::array<vk::DescriptorPoolSize, 2> poolSizes = {
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT),
        vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, MAX_FRAMES_IN_FLIGHT)
        };
    
        vk::DescriptorPoolCreateInfo poolInfo{};
        poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
        poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;
        poolInfo.poolSizeCount = poolSizes.size();
        poolInfo.pPoolSizes = poolSizes.data();

        descriptorPool = std::make_unique<vk::raii::DescriptorPool>(*device, poolInfo);
    } 

    void createDescriptorSets() {
        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);
    
        vk::DescriptorSetAllocateInfo allocInfo{};
        allocInfo.descriptorPool = *descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
        allocInfo.pSetLayouts = layouts.data();
    
        descriptorSets.clear();
        descriptorSets = device->allocateDescriptorSets(allocInfo);
    
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vk::DescriptorBufferInfo bufferInfo(*uniformBuffers[i], 0, sizeof(UniformBufferObject));
        
            vk::DescriptorImageInfo imageInfo(*textureSampler, *textureImageView, vk::ImageLayout::eShaderReadOnlyOptimal);
        
            std::array<vk::WriteDescriptorSet, 2> descriptorWrites{
            vk::WriteDescriptorSet(descriptorSets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &bufferInfo),
            vk::WriteDescriptorSet(descriptorSets[i], 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageInfo)
            };
        
            device->updateDescriptorSets(descriptorWrites, {});
        }
    }

    void createSyncObject() {
        presentCompleteSemaphores.clear();
        renderFinishedSemaphores.clear();
        inFlightFences.clear();
    
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            presentCompleteSemaphores.emplace_back(*device, vk::SemaphoreCreateInfo());
            renderFinishedSemaphores.emplace_back(*device, vk::SemaphoreCreateInfo());
        
            vk::FenceCreateInfo fenceInfo{};
            fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;
            inFlightFences.emplace_back(*device, fenceInfo);
        }
    }

    void createGraphicsPipelines() {
        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();
    
        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
    
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shader/shader.spv"));

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
        vertShaderStageInfo.module = *shaderModule;  
        vertShaderStageInfo.pName = "vertMain";

        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
        fragShaderStageInfo.module = *shaderModule; 
        fragShaderStageInfo.pName = "fragMain";

        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;

        vk::PipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        vk::PipelineViewportStateCreateInfo viewportState({}, 1, {}, 1);

        vk::PipelineRasterizationStateCreateInfo rasterizer({}, vk::False, vk::False, vk::PolygonMode::eFill,
        vk::CullModeFlagBits::eBack, vk::FrontFace::eCounterClockwise, vk::False, 0.0f, 0.0f, 1.0f, 1.0f);
        vk::PipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sampleShadingEnable = vk::False;
        multisampling.rasterizationSamples = msaaSamples;

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | 
                                            vk::ColorComponentFlagBits::eG | 
                                            vk::ColorComponentFlagBits::eB | 
                                            vk::ColorComponentFlagBits::eA;
        colorBlendAttachment.blendEnable = vk::False;

        vk::PipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.depthTestEnable = vk::True;
        depthStencil.depthWriteEnable = vk::True;
        depthStencil.depthCompareOp = vk::CompareOp::eLess;
        depthStencil.depthBoundsTestEnable = vk::False;
        depthStencil.stencilTestEnable = vk::False;

        vk::PipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.logicOpEnable = vk::False;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &(**descriptorSetLayout);
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayout = std::make_unique<vk::raii::PipelineLayout>(*device, pipelineLayoutInfo);

        vk::GraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = **pipelineLayout;
        pipelineInfo.renderPass = **renderPass; 
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = nullptr;
        pipelineInfo.basePipelineIndex = -1;

        graphicsPipeline = std::make_unique<vk::raii::Pipeline>(*device, nullptr, pipelineInfo);
    }

    void recordCommandBuffer(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex){
        vk::CommandBufferBeginInfo beginInfo{};
        beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
        commandBuffer.begin(beginInfo);

        transition_image_layout(
            commandBuffer,
            imageIndex,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal,
            {},
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eTopOfPipe,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput
        );

        vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);

        vk::RenderingAttachmentInfo attachmentInfo{};
        attachmentInfo.imageView = *swapChainImageViews[imageIndex];
        attachmentInfo.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        attachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
        attachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
        attachmentInfo.clearValue = clearColor;

        vk::ClearValue depthClearValue = vk::ClearDepthStencilValue(1.0f, 0);
        vk::RenderingAttachmentInfo depthAttachmentInfo{};
        depthAttachmentInfo.imageView = *depthImageView;
        depthAttachmentInfo.imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
        depthAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
        depthAttachmentInfo.storeOp = vk::AttachmentStoreOp::eDontCare;
        depthAttachmentInfo.clearValue = depthClearValue;

        vk::RenderingInfo renderingInfo{};
        renderingInfo.renderArea = vk::Rect2D{ {0, 0}, swapChainExtent };
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &attachmentInfo;
        renderingInfo.pDepthAttachment = &depthAttachmentInfo;

        commandBuffer.beginRendering(renderingInfo);
    
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, **graphicsPipeline);
        
        vk::Viewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapChainExtent.width);
        viewport.height = static_cast<float>(swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        commandBuffer.setViewport(0, viewport);

        vk::Rect2D scissor{};
        scissor.offset = vk::Offset2D{0, 0};
        scissor.extent = swapChainExtent;
        commandBuffer.setScissor(0, scissor);
    
        commandBuffer.bindVertexBuffers(0, **vertexBuffer, {0});
        commandBuffer.bindIndexBuffer(**indexBuffer, 0, vk::IndexType::eUint32);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, **pipelineLayout, 0, *descriptorSets[currentFrame], nullptr);
        commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
    
        commandBuffer.endRendering();

        transition_image_layout(
            commandBuffer,
            imageIndex,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::ePresentSrcKHR,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            {},
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eBottomOfPipe
        );

        commandBuffer.end();
    }

    void transition_image_layout(
        vk::raii::CommandBuffer& commandBuffer,
        uint32_t imageIndex,
        vk::ImageLayout oldLayout,
        vk::ImageLayout newLayout,
        vk::AccessFlagBits2 srcAccessMask,
        vk::AccessFlagBits2 dstAccessMask,
        vk::PipelineStageFlagBits2 srcStageMask,
        vk::PipelineStageFlagBits2 dstStageMask) {
    
        vk::ImageMemoryBarrier2 barrier{};
        barrier.srcStageMask = srcStageMask;
        barrier.srcAccessMask = srcAccessMask;
        barrier.dstStageMask = dstStageMask;
        barrier.dstAccessMask = dstAccessMask;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = swapChainImages[imageIndex];
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        vk::DependencyInfo dependencyInfo{};
        dependencyInfo.dependencyFlags = {};
        dependencyInfo.imageMemoryBarrierCount = 1;
        dependencyInfo.pImageMemoryBarriers = &barrier;

        commandBuffer.pipelineBarrier2(dependencyInfo);
    }

    void createRenderPass() {
         vk::AttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = msaaSamples;  
        colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
        colorAttachment.finalLayout = vk::ImageLayout::eColorAttachmentOptimal;  

        vk::AttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

        vk::AttachmentDescription depthAttachment{};
        depthAttachment.format = findDepthFormat();
        depthAttachment.samples = msaaSamples;  
        depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        depthAttachment.storeOp = vk::AttachmentStoreOp::eDontCare;
        depthAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        depthAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        depthAttachment.initialLayout = vk::ImageLayout::eUndefined;
        depthAttachment.finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

        vk::AttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

        vk::AttachmentDescription colorAttachmentResolve{};
        colorAttachmentResolve.format = swapChainImageFormat;
        colorAttachmentResolve.samples = vk::SampleCountFlagBits::e1;
        colorAttachmentResolve.loadOp = vk::AttachmentLoadOp::eDontCare;
        colorAttachmentResolve.storeOp = vk::AttachmentStoreOp::eStore;
        colorAttachmentResolve.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        colorAttachmentResolve.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        colorAttachmentResolve.initialLayout = vk::ImageLayout::eUndefined;
        colorAttachmentResolve.finalLayout = vk::ImageLayout::ePresentSrcKHR;

        vk::AttachmentReference colorAttachmentResolveRef{};
        colorAttachmentResolveRef.attachment = 2;  
        colorAttachmentResolveRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

        vk::SubpassDescription subpass{};
        subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;
        subpass.pResolveAttachments = &colorAttachmentResolveRef;  

        vk::SubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eLateFragmentTests;
        dependency.dstStageMask = vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eColorAttachmentOutput;
        dependency.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite;  // Обновлено!
        dependency.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite | vk::AccessFlagBits::eColorAttachmentWrite;

        std::array<vk::AttachmentDescription, 3> attachments = {colorAttachment, depthAttachment, colorAttachmentResolve};

        vk::RenderPassCreateInfo renderPassInfo{};
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        renderPass = std::make_unique<vk::raii::RenderPass>(*device, renderPassInfo);
    }

    void createDescriptorSetLayout() {
        std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr),
        vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr)
        };

        vk::DescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.bindingCount = bindings.size();
        layoutInfo.pBindings = bindings.data();
    
        descriptorSetLayout = std::make_unique<vk::raii::DescriptorSetLayout>(*device, layoutInfo);
    }

    void createFramebuffers() {
        swapChainFramebuffers.clear();
        swapChainFramebuffers.reserve(swapChainImageViews.size()); 

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            std::array<vk::ImageView, 3> attachments = {
                **colorImageView,       
                **depthImageView,       
                *swapChainImageViews[i] 
            };

            vk::FramebufferCreateInfo framebufferInfo{};
            framebufferInfo.renderPass = **renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            swapChainFramebuffers.emplace_back(*device, framebufferInfo);
        }
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        device->waitIdle();
    }

    void drawFrame() {
         while (vk::Result::eTimeout == device->waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX));

        auto [result, imageIndex] = swapChain->acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[currentFrame], nullptr);

        if (result == vk::Result::eErrorOutOfDateKHR) {
            recreateSwapChain();
            return;
        }

        if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
        }

        updateUniformBuffer(currentFrame);

        device->resetFences(*inFlightFences[currentFrame]);

        commandBuffers[currentFrame].reset();
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);

        vk::SubmitInfo submitInfo{};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &(*presentCompleteSemaphores[currentFrame]);
        submitInfo.pWaitDstStageMask = &waitDestinationStageMask;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &(*commandBuffers[currentFrame]);
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &(*renderFinishedSemaphores[currentFrame]);

        graphicsQueue->submit(submitInfo, *inFlightFences[currentFrame]);

        vk::PresentInfoKHR presentInfo{};
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &(*renderFinishedSemaphores[currentFrame]);
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &(**swapChain);
        presentInfo.pImageIndices = &imageIndex;

        auto result_t = presentQueue->presentKHR(presentInfo);

        if (result_t == vk::Result::eErrorOutOfDateKHR || result_t == vk::Result::eSuboptimalKHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else if (result_t != vk::Result::eSuccess) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo{};
        ubo.model = rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;
    
        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    void cleanupSwapChain() {
        swapChainImageViews.clear();
        swapChain = nullptr;
    }

    void recreateSwapChain() {
        int width = 0, height = 0; 
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        device->waitIdle();
        depthImageView.reset();
        depthImage.reset();
        depthImageMemory.reset();
    
        createSwapChain();
        createImageViews();
        createColorResources();
        createDepthResources();  
        createFramebuffers();    
    }

    void cleanup() {
        device->waitIdle();

        descriptorSets.clear();
        descriptorPool.reset();

        textureImageView.reset();
        textureSampler.reset();
        textureImage.reset();
        textureImageMemory.reset();

        depthImageView.reset();
        depthImage.reset();
        depthImageMemory.reset();

        indexBuffer.reset();
        indexBufferMemory.reset();

        vertexBuffer.reset();
        vertexBufferMemory.reset();

        uniformBuffers.clear();
        uniformBuffersMemory.clear();

        cleanupSwapChain();

        glfwDestroyWindow(window);
        glfwTerminate();
    }
};

int main() {
    Vulkan app;
    
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}

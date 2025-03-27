#![allow(
    dead_code,
    // unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

use anyhow::{anyhow, Ok, Result};
use log::*;
use thiserror::Error;
use vulkanalia::bytecode::Bytecode;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::window as vk_window;
use vulkanalia::Version;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

use vulkanalia::vk::ExtDebugUtilsExtension;
use vulkanalia::vk::KhrSurfaceExtension;
use vulkanalia::vk::KhrSwapchainExtension;

use std::collections::HashSet;
use std::ffi::CStr;
use std::os::raw::c_void;

/// macOS에서 Vulkan을 사용할 때 필요한 버전  
const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);

/// validation layer를 활성화 할지 결정  
/// debug 빌드에서만 활성화하도록 설정함
const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

/// standard validation layer를 사용함  
/// validation layer는 Vulkan function call을 후킹해서 추가적인 연산을 적용함
const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

/// swapchain 확장의 이름을 포함한 리스트
const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];

/// 동시에 실행될 frame의 수
const MAX_FRAMES_IN_FLIGHT: usize = 2;

/// Our Vulkan app.  
/// Vulkan 프로그램동안 setup, rendering, destruction로직을 구현하는 구조체  
#[derive(Clone, Debug)]
struct App {
    /// vulkan entry point를 저장하기 위한 필드
    entry: Entry,
    /// vulkan instance를 저장하기 위한 필드
    instance: Instance,
    data: AppData,
    device: Device,
    /// frame track을 유지하기 위한 필드
    frame: usize,
}

impl App {
    /// Creates our Vulkan app.
    unsafe fn create(window: &Window) -> Result<Self> {
        // Vulkan command를 Vulkan shared library에서 로드하기 위해 사용됨
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let mut data = AppData::default();
        let instance = create_instance(window, &entry, &mut data)?;

        data.surface = vk_window::create_surface(&instance, &window, &window)?;

        pick_physical_device(&instance, &mut data)?;

        let device = create_logical_device(&entry, &instance, &mut data)?;
        create_swapchain(window, &instance, &device, &mut data)?;
        create_swapchain_image_views(&device, &mut data)?;
        create_render_pass(&instance, &device, &mut data)?;
        create_pipeline(&device, &mut data)?;
        create_framebuffers(&device, &mut data)?;
        create_command_pool(&instance, &device, &mut data)?;
        create_command_buffers(&device, &mut data)?;
        create_sync_objects(&device, &mut data)?;

        Ok(Self {
            entry,
            instance,
            data,
            device,
            frame: 0,
        })
    }

    /// Renders a frame for our Vulkan app.  
    ///   
    /// **render가 할 일:**  
    /// 1. swapchain으로부터 이미지를 얻음  
    /// 2. framebuffer에서 이미지를 attachment로 이용하여 command buffer를 실행
    /// 3. presentation을 위해 이미지를 swapchain으로 반환함
    ///
    /// 각각의 이벤트는 비동기적으로 실행됨 -> 세마포어 필요
    unsafe fn render(&mut self, window: &Window) -> Result<()> {
        // frame이 끝날 때 까지 대기
        self.device
            .wait_for_fences(&[self.data.in_flight_fences[self.frame]], true, u64::MAX)?;

        // swapchain으로부터 이미지를 얻어옴
        let image_index = self
            .device
            .acquire_next_image_khr(
                self.data.swapchain,
                // timeout. u64::MAX는 timeout을 비활성화
                u64::MAX,
                // presentation engine이 끝날 때 시그널될 세마포어
                // 시그널 된 때 부터 이미지를 그릴 수 있음
                self.data.image_available_semaphores[self.frame],
                vk::Fence::null(),
            )?
            .0 as usize;

        if !self.data.images_in_flight[image_index as usize].is_null() {
            self.device.wait_for_fences(
                &[self.data.images_in_flight[image_index as usize]],
                true,
                u64::MAX,
            )?;
        }

        self.data.images_in_flight[image_index as usize] = self.data.in_flight_fences[self.frame];

        // wait_semaphore와 wait_stages는 pipeline의 어느 시점에서 대기하고 있을 지 설정함
        let wait_semaphores = &[self.data.image_available_semaphores[self.frame]];
        // 색을 그리는것을 대기하기 위해 COLOR_ATTACHMENT_OUTPUT을 지정
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        // 얻은 swapchain image를 바인딩하는 command buffer를 제출해야함
        let command_buffers = &[self.data.command_buffers[image_index as usize]];
        // command buffer가 끝나면 시그널될 세마포어를 지정함
        let signal_semaphores = &[self.data.render_finished_semaphores[self.frame]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.device
            .reset_fences(&[self.data.in_flight_fences[self.frame]])?;

        // graphics queue에 command buffer를 제출함
        self.device.queue_submit(
            self.data.graphics_queue,
            &[submit_info],
            self.data.in_flight_fences[self.frame],
        )?;

        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        self.device
            .queue_present_khr(self.data.present_queue, &present_info)?;

        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    /// Destroys our Vulkan app.
    /// vk::DebugUtilsMessengerEXT오브젝트는 앱이 종료되기 전에 cleanup되어야 한다.
    unsafe fn destroy(&mut self) {
        // 모든 command들이 끝나고 synchronization이 필요하지 않으므로 semaphore를 파괴
        self.data
            .render_finished_semaphores
            .iter()
            .for_each(|s| self.device.destroy_semaphore(*s, None));
        self.data
            .image_available_semaphores
            .iter()
            .for_each(|s| self.device.destroy_semaphore(*s, None));
        // fence를 파괴
        self.data
            .in_flight_fences
            .iter()
            .for_each(|f| self.device.destroy_fence(*f, None));

        // command pool을 파괴
        self.device
            .destroy_command_pool(self.data.command_pool, None);
        // framebuffers를 파괴
        // image view와 render pass전에 파괴함
        self.data
            .framebuffers
            .iter()
            .for_each(|f| self.device.destroy_framebuffer(*f, None));
        // graphics pipeline을 파괴
        self.device.destroy_pipeline(self.data.pipeline, None);
        // pipeline layout을 파괴
        self.device
            .destroy_pipeline_layout(self.data.pipeline_layout, None);
        // render pass를 파괴
        self.device.destroy_render_pass(self.data.render_pass, None);

        if VALIDATION_ENABLED {
            // 프로그램이 종료되기 전에 디버그 메세지 핸들러를 파괴
            self.instance
                .destroy_debug_utils_messenger_ext(self.data.messenger, None);
        }

        // swapchain image view를 파괴
        self.data
            .swapchain_image_views
            .iter()
            .for_each(|v| self.device.destroy_image_view(*v, None));

        // device전에 청소되어야 함
        self.device.destroy_swapchain_khr(self.data.swapchain, None);
        self.device.destroy_device(None);
        // device가 파괴된 후에 instance를 파괴해야 함
        // 프로그램이 종료되면 instance가 파괴되기 전에 surface를 파괴해야 함
        self.instance.destroy_surface_khr(self.data.surface, None);
        // 프로그램이 종료되면 인스턴스를 파괴해야 함
        self.instance.destroy_instance(None);
    }
}

/// The Vulkan handles and associated properties used by our Vulkan app.
/// Vulkan 리소스 컨테이너 역할
#[derive(Clone, Debug, Default)]
struct AppData {
    /// surface
    surface: vk::SurfaceKHR,
    /// 디버그 메세지를 처리하기 위한 messenger 핸들러
    messenger: vk::DebugUtilsMessengerEXT,
    /// physical device 핸들
    physical_device: vk::PhysicalDevice,
    /// logical device와 함께 생성된 graphics queue를 컨트롤하기 위한 핸들
    graphics_queue: vk::Queue,
    /// present queue를 컨트롤하기 위한 핸들
    present_queue: vk::Queue,
    /// swapchain image를 위한 format
    swapchain_format: vk::Format,
    /// swapchain image를 위한 extent  
    /// swapchin의 해상도같은거
    swapchain_extent: vk::Extent2D,
    /// swapchain을 저장할 필드
    swapchain: vk::SwapchainKHR,
    /// swapchain의 이미지를 저장할 필드
    swapchain_images: Vec<vk::Image>,
    /// image view를 저장하기 위한 필드
    swapchain_image_views: Vec<vk::ImageView>,
    /// render pass를 저장하기 위한 필드  
    /// render pass는 프로그램 렌더링동안 계속 사용되므로 저장해둬야함
    render_pass: vk::RenderPass,
    /// shader의 uniform value를 저장하기 위한 필드  
    /// uniform 값들을 drawing time에 조작하기 위해 필요함
    pipeline_layout: vk::PipelineLayout,
    /// pipe line을 저장하기 위한 필드
    pipeline: vk::Pipeline,
    /// framebuffer들을 저장하기 위한 필드  
    /// swapchain이 반환하는 이미지에 따라 달라지므로, 모든 이미지에 대한 framebuffer들을 만들어줌
    framebuffers: Vec<vk::Framebuffer>,
    /// command pool을 저장하기 위한 필드  
    /// command buffer는 여기에 저장되고 관리되며 여기에서 할당됨
    command_pool: vk::CommandPool,
    /// command buffer들을 저장하기 위한 필드  
    /// swapchain의 모든 이미지에 대해 command buffer를 다시 기록해야함
    command_buffers: Vec<vk::CommandBuffer>,
    /// 이미지가 얻어졌고 rendering 준비가 됨을 알리기 위한 세마포어
    image_available_semaphores: Vec<vk::Semaphore>,
    /// rendering이 완료되었고 presentation가 일어났음을 알리기 위한 세마포어
    render_finished_semaphores: Vec<vk::Semaphore>,
    /// frame을 위한 fence
    in_flight_fences: Vec<vk::Fence>,
    /// swapchain image가 사용중인지 추적하기위한 필드
    images_in_flight: Vec<vk::Fence>,
}

#[derive(Debug, Error)]
#[error("Missing {0}.")]
pub struct SuitabilityError(pub &'static str);

#[derive(Copy, Clone, Debug)]
/// queue family의 인덱스를 저장하기 위한 구조체
struct QueueFamilyIndices {
    graphics: u32,
    // graphics queue family와 겹치지 않을 수 있으므로 present queue family를 따로 저장
    present: u32,
}

impl QueueFamilyIndices {
    unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        // 장치의 queue family 속성을 가져옴
        let properties = instance.get_physical_device_queue_family_properties(physical_device);

        let graphics = properties
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        let mut present = None;
        for (index, properties) in properties.iter().enumerate() {
            if instance.get_physical_device_surface_support_khr(
                physical_device,
                index as u32,
                data.surface,
            )? {
                present = Some(index as u32);
                break;
            }
        }
        if let (Some(graphics), Some(present)) = (graphics, present) {
            Ok(Self { graphics, present })
        } else {
            Err(anyhow!(SuitabilityError(
                "Missing required queue families."
            )))
        }
    }
}

/// swapchain이 window surface와 호환되는지 확인하기 위해 사용할 프로퍼티들을 담는 구조체
#[derive(Clone, Debug)]
struct SwapchainSupport {
    /// 최대/최소 swapchain 이미지 수, 이미지 크기 등의 정보를 담는 필드
    capabilities: vk::SurfaceCapabilitiesKHR,
    /// pixed format, color space 등의 정보를 담는 필드
    formats: Vec<vk::SurfaceFormatKHR>,
    /// available present modes를 담는 필드
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        Ok(Self {
            capabilities: instance
                .get_physical_device_surface_capabilities_khr(physical_device, data.surface)?,
            formats: instance
                .get_physical_device_surface_formats_khr(physical_device, data.surface)?,
            present_modes: instance
                .get_physical_device_surface_present_modes_khr(physical_device, data.surface)?,
        })
    }
}

/// Vulkan에서 발생하는 디버그 메세지를 처리하기 위한 콜백 함수  
/// Vulkan이 Rust함수를 호출하도록 허용하기 위해서 `extern "system"`을 사용함
extern "system" fn debug_callback(
    // 메세지의 심각도
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    // 메세지의 타입
    // 일반, 검증, 성능등의 타입이 있음
    type_: vk::DebugUtilsMessageTypeFlagsEXT,
    // 메세지의 데이터
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let data = unsafe { *data };
    let message = unsafe { CStr::from_ptr(data.message) }.to_string_lossy();

    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        error!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        warn!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        debug!("({:?}) {}", type_, message);
    } else {
        trace!("({:?}) {}", type_, message);
    }

    // 리턴값이 true라면, 프로그램이 abort됨. 지금은 항상 false를 반환시키면 됨
    vk::FALSE
}

/// physical device이 swapchain을 지원하는지 확인
/// 이걸 하더라도 window surface와 호환되는지 확인해야 함
unsafe fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    let extensions = instance
        .enumerate_device_extension_properties(physical_device, None)?
        .iter()
        .map(|e| e.extension_name)
        .collect::<HashSet<_>>();
    if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
        Ok(())
    } else {
        Err(anyhow!(SuitabilityError(
            "Missing required device extensions."
        )))
    }
}

/// physical device를 검사하고 적합한지 확인
unsafe fn check_physical_device(
    instance: &Instance,
    data: &AppData,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    // 장치의 속성을 가져옴
    // let properties = instance.get_physical_device_properties(physical_device);
    // 장치의 기능을 가져옴
    // let features = instance.get_physical_device_features(physical_device);

    QueueFamilyIndices::get(instance, data, physical_device)?;

    // swapchain이 지원되는지 확인
    check_physical_device_extensions(instance, physical_device)?;

    // swapchain이 지원되는지 확인한 후에 query해야하는것이 중요
    // swapchain이 window surface와 호환되는지 확인
    let support = SwapchainSupport::get(instance, data, physical_device)?;
    if support.formats.is_empty() || support.present_modes.is_empty() {
        return Err(anyhow!(SuitabilityError("Insufficient swapchain support.")));
    }

    Ok(())
}

/// 최적의 surface format(color depth) 찾기  
/// 가장 이상적으로 판단되는 surface format이 있으면 쓰고 없으면 첫번째 format을 사용
fn get_swapchain_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    formats
        .iter()
        .cloned()
        .find(|f| {
            // 가장 이상적인 surface format의 조건
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or_else(|| formats[0])
}

/// 최적의 Present mode 찾기  
/// FIFO는 항상 이용가능함이 보장됨  
/// MAILBOX가 가장 좋은데 안된다면 FIFO를 쓰도록 함
fn get_swapchain_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    present_modes
        .iter()
        .cloned()
        .find(|m| *m == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

/// 최적의 swap extent 찾기  
/// swapchain의 해상도임  
/// 가능한 해상도의 범위는 capabilities에 저장되어 있음
fn get_swapchain_extent(window: &Window, capabilities: vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        // width가 u32:MAX라면, 다른 해상도를 쓰는것을 허용하는 것임
        // 여기에서는 clamp를 사용하여 window의 최소/최대 사이의 값만 사용하도록 함
        vk::Extent2D::builder()
            .width(window.inner_size().width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ))
            .height(window.inner_size().height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ))
            .build()
    }
}

/// physical device를 찾아서 선택하고 AppData에 저장
unsafe fn pick_physical_device(instance: &Instance, data: &mut AppData) -> Result<()> {
    for physical_device in instance.enumerate_physical_devices()? {
        let properties = instance.get_physical_device_properties(physical_device);

        if let Err(error) = check_physical_device(instance, data, physical_device) {
            warn!(
                "Skipping physical device (`{}`): {}",
                properties.device_name, error
            );
        } else {
            info!("Selected physical device (`{}`).", properties.device_name);
            data.physical_device = physical_device;
            return Ok(());
        }
    }

    Err(anyhow!("Failed to find suitable physical device."))
}

/// instance 생성
unsafe fn create_instance(window: &Window, entry: &Entry, data: &mut AppData) -> Result<Instance> {
    // 애플리케이션 정보를 설정
    // 보통 optional이지만, 애플리케이션을 최적화하는데 유용한 정보를 드라이버에 제공할 수 있음
    // Vulkan은 UTF-8 문자열을 사용하므로 문자열 끝에 NULL 문자를 추가해야 함
    let application_info = vk::ApplicationInfo::builder()
        .application_name(b"Vulkan Tutorial\0")
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(b"No Engine\0")
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 0, 0));

    // 사용 가능한 레이어를 가져옴
    let available_layers = entry
        // 모든 레이어를 가져옴
        .enumerate_instance_layer_properties()?
        .iter()
        // 레이어의 이름을 HashSet에 모음
        .map(|l| l.layer_name)
        .collect::<HashSet<_>>();

    // validation layer가 요청되었지만 사용 가능한 레이어에 없다면 에러를 반환
    if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
        return Err(anyhow!("Validation layer requested but not supported."));
    }

    // validation layer의 활성 여부에 따라 레이어 목록을 설정
    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    // 필수 instance extension들을 가져옴
    let mut extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        // 이름들을 전부 const * const c_char로 변환
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    if VALIDATION_ENABLED {
        // validation layer가 활성화된 경우에만 디버그 유틸 확장 추가
        // 디버그 메세지를 핸들링하기 위해 필요함
        // 메세지를 debug_callback함수로 전달해서 핸들링 할 것임
        extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
    }

    // Required by Vulkan SDK on macOS since 1.3.216.
    let flags = if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        info!("Enabling extensions for macOS portability.");
        extensions.push(
            vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION
                .name
                .as_ptr(),
        );
        extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::empty()
    };

    // Vulkan 인스턴스 생성하기 위한 정보를 설정
    let mut info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        // 사용할 레이어 목록을 설정
        .enabled_layer_names(&layers)
        // 사용할 확장 목록을 설정
        .enabled_extension_names(&extensions)
        .flags(flags);

    // 디버그 정보를 설정
    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        // 알림을 받을 심각도를 설정
        // 사용할수 없을수도 있는 모든 flags를 사용하지만, 사용하지 않는 경우 문제가 없음
        // 그런 플래그를 사용하면 validation error를 발생시킴
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        // 알림을 받을 메세지 타입을 설정
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        // 디버그 콜백 설정
        .user_callback(Some(debug_callback));

    if VALIDATION_ENABLED {
        info = info.push_next(&mut debug_info);
    }

    let instance = entry.create_instance(&info, None)?;

    if VALIDATION_ENABLED {
        // debug info를 instance에 등록
        // 이것도 instance가 파괴되기 전에 해제해야 함
        data.messenger = instance.create_debug_utils_messenger_ext(&debug_info, None)?;
    }

    Ok(instance)
}

/// logical device를 생성
unsafe fn create_logical_device(
    entry: &Entry,
    instance: &Instance,
    data: &mut AppData,
) -> Result<Device> {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;

    // queue family를 생성하기 위해 여러개의 DeviceQeuueCreateInfo가 필요하므로
    // 세트를 생성해서 관리함
    let mut unique_indices = HashSet::new();
    unique_indices.insert(indices.graphics);
    unique_indices.insert(indices.present);

    let queue_priorities = &[1.0];
    let queue_infos = unique_indices
        .iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*i)
                .queue_priorities(queue_priorities)
        })
        .collect::<Vec<_>>();

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        vec![]
    };

    // swapchain 확장을 활성화함
    let mut extensions = DEVICE_EXTENSIONS
        .iter()
        .map(|n| n.as_ptr())
        .collect::<Vec<_>>();

    // Required by Vulkan SDK on macOS since 1.3.216.
    if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        extensions.push(vk::KHR_PORTABILITY_SUBSET_EXTENSION.name.as_ptr());
    }

    let features = vk::PhysicalDeviceFeatures::builder();

    // DeviceCreateInfo를 생성
    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .enabled_features(&features);

    let device = instance.create_device(data.physical_device, &info, None)?;

    // queue family가 같은경우 index를 한번만 넘겨줘도 됨
    data.graphics_queue = device.get_device_queue(indices.graphics, 0);
    data.present_queue = device.get_device_queue(indices.graphics, 0);

    Ok(device)
}

/// Swapchain 생성
unsafe fn create_swapchain(
    window: &Window,
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
    let support = SwapchainSupport::get(instance, data, data.physical_device)?;

    // swapchain을 생성하기 위해 필요한 세 가지 정보
    let surface_format = get_swapchain_surface_format(&support.formats);
    let present_mode = get_swapchain_present_mode(&support.present_modes);
    let extent = get_swapchain_extent(window, support.capabilities);

    // 이미지 개수를 설정
    // 이미지 개수는 min_image_count보다 1개 더 많아야 함. 드라이버 내부 연산이 완료가 되어야만 이미지를 얻을수 있는 문제를 피하기 위함.
    let mut image_count = support.capabilities.min_image_count + 1;

    // 이미지 수가 최대 이미지 수를 초과하지 않도록 함
    if support.capabilities.max_image_count != 0
        && image_count > support.capabilities.max_image_count
    {
        image_count = support.capabilities.max_image_count;
    }

    let mut queue_family_indices = vec![];
    let image_sharing_mode = if indices.graphics != indices.present {
        // queue family들이 다르다면, concurrent모드를 사용해서 소유권 없이 image를 사용할 수 있도록 함
        // 다를 때 소유권 핸들링이 어려워서 튜토리얼에서는 패스
        queue_family_indices.push(indices.graphics);
        queue_family_indices.push(indices.present);
        vk::SharingMode::CONCURRENT
    } else {
        // 웬만한 하드웨어는 queue family가 같고. 그런경우는 exclusive를 사용함.
        // 이 옵션이 최고의 퍼포먼스를 제공함
        vk::SharingMode::EXCLUSIVE
    };

    // swapchain을 생성하기 위한 정보를 채워야함
    let info = vk::SwapchainCreateInfoKHR::builder()
        .surface(data.surface)
        .min_image_count(image_count)
        // 위에서 생성한 세 가지 정보
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        // 이미지가 구성하는 레이어의 양.
        // stereoscopic 3D rendering이 아닌 경우는 항상 1임
        .image_array_layers(1)
        // swapchain의 이미지들이 어떤 작업에 사용될 지 지정
        // 지금은 color_attachment로 사용할 것
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(image_sharing_mode)
        .queue_family_indices(&queue_family_indices)
        // 90도 시계방향 회전 등의 변환이 지원되면 해주기
        // current_transform만 사용함.
        .pre_transform(support.capabilities.current_transform)
        // alpha채널이 window system에서 다른 window와 blending되어야 하는지 지정
        // 무시하기를 원하므로 OPAQUE 사용
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        // 다른 윈도우가 해당 픽셀 앞에 있는 경우 등의 obscure한 픽셀은 색상에 관해 신경쓰지 않음
        .clipped(true)
        // resize되는 등의 경우 swapchain이 invalid가 되거나 unoptimized되는 경우를 처리하기 위한 필드
        // 그러면 swapchain은 다시 생성되어야 하고 기존 swapchain에 대한 참조가 여기에 지정되어야 함
        // 기본값도 null이긴 함
        .old_swapchain(vk::SwapchainKHR::null());

    data.swapchain_format = surface_format.format;
    data.swapchain_extent = extent;
    data.swapchain = device.create_swapchain_khr(&info, None)?;

    // 이미지의 핸들을 가져옴
    data.swapchain_images = device.get_swapchain_images_khr(data.swapchain)?;

    Ok(())
}

/// swapchain image view 생성
unsafe fn create_swapchain_image_views(device: &Device, data: &mut AppData) -> Result<()> {
    // 모든 swapchain의 image에 대해 imageview를 생성
    data.swapchain_image_views = data
        .swapchain_images
        .iter()
        .map(|i| {
            // color component mapping
            // 특정 부분은 ONE이나 ZERO로 설정하면 해당 색 채널을 고정시킬 수 있음
            // default를 사용함
            let components = vk::ComponentMapping::builder()
                .r(vk::ComponentSwizzle::IDENTITY)
                .g(vk::ComponentSwizzle::IDENTITY)
                .b(vk::ComponentSwizzle::IDENTITY)
                .a(vk::ComponentSwizzle::IDENTITY);

            // image의 목적과 어느 부분이 접근될 지 설정
            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                // mipmap 없음
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                // multiple layer를 사용하지 않음
                // stereographic 3D가 아니므로 필요없음
                .layer_count(1);

            let info = vk::ImageViewCreateInfo::builder()
                .image(*i)
                // 이미지가 2D texture로 해석될 수 있도록 설정
                .view_type(vk::ImageViewType::_2D)
                .format(data.swapchain_format)
                .components(components)
                .subresource_range(subresource_range);

            device.create_image_view(&info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(())
}

/// render pass 생성  
/// rendering동안 사용될 framebuffer attachment를 vulkan에 알려주기 위함  
/// color/depth buffer의 수, 각 buffer를 위해 사용할 sample의 수, rendering동안 content가 어떻게 다루어져야할지를 설정함
unsafe fn create_render_pass(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // 여기에서는 swapchain의 이미지마다 single color buffer attachment를 가짐
    // 그 정보를 AttachmentDescription으로 표현할거임
    let color_attachment = vk::AttachmentDescription::builder()
        // format은 swapchain의 format과 일치해야함
        .format(data.swapchain_format)
        // multisample을 하지 않으므로 1로 설정
        .samples(vk::SampleCountFlags::_1)
        // 새 frame을 그릴 때 이전의 내용을 지움
        .load_op(vk::AttachmentLoadOp::CLEAR)
        // render된 content가 메모리에 저장되고 나중에 읽을 수 있도록 함
        // 화면에 렌더링된 삼각형을 볼거니까 STORE로 설정
        .store_op(vk::AttachmentStoreOp::STORE)
        // stencil buffer는 사용하지 않으므로 DONT_CARE로 설정
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        // render가 시작하기 전 갖게 될 layout
        // undefined로 설정하면 이미지의 보존이 보장되지 않음
        // 그러나 어차피 지울 이미지라서 문제없음
        .initial_layout(vk::ImageLayout::UNDEFINED)
        // rendering후에 swapchain을 사용하여 이미지가 presentation을 위해 준비되기를 원하므로 설정
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    // single render pass는 multiple subpass를 구성할 수 있음
    // 잇따라 적용되는 post-processing effect시퀀스임
    // 여기에서는 single subpass를 쓰도록 함
    // 모든 subpass는 이전에 만든 한 개 이상의 attachment를 참조함. 이것을 AttachmentReference로 표현함
    let color_attachment_ref = vk::AttachmentReference::builder()
        // color_attachment하나만 만들었으므로 index는 0
        .attachment(0)
        // subpass동안 attachment가 가지길 원하는 layout
        // subpass가 시작될 때 vulkan은 자동으로 attachment를 이 layout으로 변환함
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    // subpass에서 참조할 attachment를 설정
    let color_attachments = &[color_attachment_ref];
    let subpass = vk::SubpassDescription::builder()
        // graphics subpass라는 것을 명시함. 추후 compute subpass를 추가할 수도 있기 때문
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        // 설정한 참조정보를 전달
        // render pass에 color_attachment하나만 전달하므로 그걸 사용하게 됨
        .color_attachments(color_attachments);
    let dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

    // render pass생성을 위한 정보
    let attachments = &[color_attachment];
    let subpasses = &[subpass];
    let dependencies = &[dependency];
    let info = vk::RenderPassCreateInfo::builder()
        .attachments(attachments)
        .subpasses(subpasses)
        .dependencies(dependencies);

    data.render_pass = device.create_render_pass(&info, None)?;

    Ok(())
}

/// pipeline 생성
unsafe fn create_pipeline(device: &Device, data: &mut AppData) -> Result<()> {
    // 컴파일된 셰이더를 읽어옴
    let vert = include_bytes!("../shaders/vert.spv");
    let frag = include_bytes!("../shaders/frag.spv");

    // shader module은 단순히 wrapper이고 pipeline생성시에 machine code로 변환하고 링크됨
    // 즉 pipeline이 생성된 후에 shader module은 파괴해도 되므로, local scope에서 생성했고 이 함수 끝에서 파괴함
    let vert_shader_module = create_shader_module(device, &vert[..])?;
    let frag_shader_module = create_shader_module(device, &frag[..])?;

    // pipeline에 할당하기 위해서는 stage로 만들어서 붙여줘야함
    let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_shader_module)
        .name(b"main\0");

    let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_shader_module)
        .name(b"main\0");

    // 지금은 vertex shader에 정점정보를 하드코딩했기 때문에 로드될 vertex date가 없음
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder();

    // 어떤 종류의 geometry가 vertex로부터 그려질지를 설정함
    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
        // reuse없이 3개마다 삼각형
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    // 결과물이 렌더링될 framebuffer의 region을 서술함
    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(data.swapchain_extent.width as f32)
        .height(data.swapchain_extent.height as f32)
        // 특별한 일을 하지 않는 이상 표준인 0.0 - 1.0을 사용함
        .min_depth(0.0)
        .max_depth(1.0);

    // scissor는 transform보다는 filter처럼 작동함
    // 즉 이 범위 밖의 이미지는 폐기됨
    let scissor = vk::Rect2D::builder()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(data.swapchain_extent);

    // 어떤 그래픽카드들은 여러개의 viewport와 scissor를 쓰는 것을 지원함
    // 그러므로 배열로 전달해주어야 함
    // 그러나 여러개 쓰려면 GPU feature활성화 필요
    let viewports = &[viewport];
    let scissors = &[scissor];
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(viewports)
        .scissors(scissors);

    // rasterizer는 vertex shader로부터 만들어진 vertex로부터 생성된 geometry를 갖고 fragment shader에 의해 색이 입혀진 fragment로 바꿈
    // rasterizer는 depth testing, face culling 그리고 scissor test를 수행함
    // 그 후에 전체 polygon을 채우거나, 또는 edge만 출력하는 wireframe을 출력하도록 구성할 수도 있음
    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
        // 이부분이 true라면, near과 far plane을 벗어난 fragment는 폐기되지 않고 clamp됨
        // shadow map같은 특별한 케이스에서 유용함
        // 사용하려면 GPU feature활성화 필요
        .depth_clamp_enable(false)
        // 이부분이 true라면, geometry가 절대로 rasterizer stage로 넘어가지 않음
        // 즉 frambuffer으로의 출력을 비활성화함
        .rasterizer_discard_enable(false)
        // polygon을 어떻게 채울지 설정함
        // FILL은 채우기, LINE은 선, POINT는 점
        .polygon_mode(vk::PolygonMode::FILL)
        // 라인의 두께를 설정함
        // 1.0보다 두꺼운 라인을 그리고 싶다면, GPU feature활성화 필요
        .line_width(1.0)
        // face culling의 타입을 결정함
        // BACK으로 설정해서 front face만 그리도록 함
        .cull_mode(vk::CullModeFlags::BACK)
        // culling을 할 때 CLOCKWISE로 된걸 front face로 설정함
        .front_face(vk::FrontFace::CLOCKWISE)
        // shadow mapping을 위해 사용되지만, 잘 모름
        .depth_bias_enable(false);

    // anti-aliasing을 수행하기 위한 방법 중 하나
    // 같은 픽셀에 rasterize된 여러 polygon의 fragment shader결과를 조합하여 작동함
    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::_1);

    // fragment shader의 결과물을 framebuffer에 더하는 방법을 설정해야함
    // 이 변환은 color blending으로 알려져있음
    // color blending을 구성하기 위한 구조체중 하나인 attachment를 사용할것임
    // 이 구조체는 attached framebuffer마다 configuration을 가지고 있음
    let attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        // 이게 false라면 새 color가 그대로 넘겨짐. true라면 섞어서 새로운 색상을 계산해냄
        // result color는 바로 위에서 설정한 color_write_mask와 &연산하여 어떤 채널이 넘겨질 지 결정함
        .blend_enable(true)
        // 가장 일반적인 color blending을 위해 alpha blending을 구성할것임
        // 이를 위해 아래의 파라미터를 쓸 수 있음
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD);

    let attachments = &[attachment];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        // blending의 bitwise방식을 쓰려면 true로 설정해야함 -> 이러면 모든 attachment의 blend enable을 false로 설정하는것과 같아짐
        // false로 했으므로 fragment color가 그대로 framebuffer로 넘어감
        // 자세한건 노트 참조
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(attachments)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    let layout_info = vk::PipelineLayoutCreateInfo::builder();

    data.pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

    let stages = &[vert_stage, frag_stage];
    let info = vk::GraphicsPipelineCreateInfo::builder()
        // stage로 만들어진 shader module을 사용함
        .stages(stages)
        // fixed-funcition stage를 기술한 구조체들 전달
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .color_blend_state(&color_blend_state)
        // 참조가 아닌 handle을 전달
        .layout(data.pipeline_layout)
        // render pass와 sub pass의 index를 설정
        .render_pass(data.render_pass)
        .subpass(0);

    data.pipeline = device
        // 첫번째는 캐시설정임
        .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?
        .0[0];

    device.destroy_shader_module(vert_shader_module, None);
    device.destroy_shader_module(frag_shader_module, None);

    Ok(())
}

/// framebuffer 생성
unsafe fn create_framebuffers(device: &Device, data: &mut AppData) -> Result<()> {
    data.framebuffers = data
        .swapchain_image_views
        .iter()
        .map(|i| {
            let attachments = &[*i];
            let create_info = vk::FramebufferCreateInfo::builder()
                // 어떤 render pass와 호환될지 지정
                .render_pass(data.render_pass)
                .attachments(attachments)
                .width(data.swapchain_extent.width)
                .height(data.swapchain_extent.height)
                // swapchain의 이미지는 single image이므로 layer는 1
                .layers(1);

            device.create_framebuffer(&create_info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(())
}

/// command pool 생성
unsafe fn create_command_pool(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;

    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::empty()) // Optional.
        // drawing을 위한 command를 기록할것이므로 graphics queue를 사용함
        .queue_family_index(indices.graphics);

    data.command_pool = device.create_command_pool(&info, None)?;

    Ok(())
}

/// command buffer 생성
unsafe fn create_command_buffers(device: &Device, data: &mut AppData) -> Result<()> {
    let allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(data.command_pool)
        // 실행을 위해 queue로 전송될 수 있지만, 다른 command buffer에 의해 실행될 수는 없음
        .level(vk::CommandBufferLevel::PRIMARY)
        // 할당할 버퍼의 수
        .command_buffer_count(data.framebuffers.len() as u32);

    data.command_buffers = device.allocate_command_buffers(&allocate_info)?;

    for (i, command_buffer) in data.command_buffers.iter().enumerate() {
        let inheritance = vk::CommandBufferInheritanceInfo::builder();

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::empty()) // Optional.
            .inheritance_info(&inheritance); // Optional.

        // command buffer 기록 시작
        device.begin_command_buffer(*command_buffer, &info)?;

        // render pass를 시작하기 위한 정보
        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            // render area의 크기를 지정
            .extent(data.swapchain_extent);

        // clear color를 설정
        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let clear_values = &[color_clear_value];
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(data.render_pass)
            // framebuffer 설정
            .framebuffer(data.framebuffers[i])
            .render_area(render_area)
            .clear_values(clear_values);

        // render pass시작
        device.cmd_begin_render_pass(
            *command_buffer,
            &info,
            // primary command buffer에 그 자체로 임베드됨
            vk::SubpassContents::INLINE,
        );

        // graphics pipeline을 바인딩
        device.cmd_bind_pipeline(
            *command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            data.pipeline,
        );

        // 삼각형을 그리도록 알려줌
        device.cmd_draw(
            *command_buffer,
            // vertex의 갯수
            3,
            // instanced rendering을 위해 쓰이지만, 지금은 쓰지 않으므로 1
            1,
            // vertex buffer의 offset으로 사용됨
            // gl_VertexIndex의 가장 낮은 값을 정의함
            0,
            // instance의 offset으로 사용됨
            // gl_InstanceIndex의 가장 낮은 값을 정의함
            0,
        );

        // render pass를 끝냄
        device.cmd_end_render_pass(*command_buffer);
        // command buffer를 끝냄
        device.end_command_buffer(*command_buffer)?;
    }

    Ok(())
}

/// semaphore를 생성하는 함수  
unsafe fn create_sync_objects(device: &Device, data: &mut AppData) -> Result<()> {
    // draw command와 presentation의 queue operation을 동기화할것이므로 세마포어가 필요함
    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        data.image_available_semaphores
            .push(device.create_semaphore(&semaphore_info, None)?);
        data.render_finished_semaphores
            .push(device.create_semaphore(&semaphore_info, None)?);

        data.in_flight_fences
            .push(device.create_fence(&fence_info, None)?);
    }

    data.images_in_flight = data
        .swapchain_images
        .iter()
        .map(|_| vk::Fence::null())
        .collect();

    Ok(())
}

/// shader bytecode를 vk::ShaderModule로 래핑하는 helper function
unsafe fn create_shader_module(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule> {
    // bytecode는 u8 slice인데 ShaderModule은 u32 slice를 받으므로 변환이 필요함
    let bytecode = Bytecode::new(bytecode).unwrap();

    let info = vk::ShaderModuleCreateInfo::builder()
        .code_size(bytecode.code_size())
        .code(bytecode.code());

    Ok(device.create_shader_module(&info, None)?)
}

fn main() -> Result<()> {
    pretty_env_logger::init();

    // Window
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Vulkan Tutorial (Rust)")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)?;

    // App
    let mut app = unsafe { App::create(&window)? };
    event_loop.run(move |event, elwt| {
        match event {
            // Request a redraw when all events were processed.
            Event::AboutToWait => window.request_redraw(),
            Event::WindowEvent { event, .. } => match event {
                // Render a frame if our Vulkan app is not being destroyed.
                WindowEvent::RedrawRequested if !elwt.exiting() => {
                    // destroying flag를 체크해서 destroy후에 render를 호출하지 않도록 함
                    unsafe { app.render(&window) }.unwrap()
                }
                // Destroy our Vulkan app.
                WindowEvent::CloseRequested => {
                    elwt.exit();
                    unsafe {
                        app.device.device_wait_idle().unwrap();
                    }
                    unsafe {
                        app.destroy();
                    }
                }
                _ => {}
            },
            _ => {}
        }
    })?;

    Ok(())
}

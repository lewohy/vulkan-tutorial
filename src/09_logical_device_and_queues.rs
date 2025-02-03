#![allow(
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

use anyhow::{anyhow, Ok, Result};
use log::*;
use thiserror::Error;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::window as vk_window;
use vulkanalia::Version;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

use vulkanalia::vk::ExtDebugUtilsExtension;

use std::collections::HashSet;
use std::ffi::CStr;
use std::os::raw::c_void;

// macOS에서 Vulkan을 사용할 때 필요한 버전
const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);

// validation layer를 활성화 할지 결정
// debug 빌드에서만 활성화하도록 설정함
const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

// standard validation layer를 사용함
const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

// Vulkan에서 발생하는 디버그 메세지를 처리하기 위한 콜백 함수
// Vulkan이 Rust함수를 호출하도록 허용하기 위해서 `extern "system"`을 사용함
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

    vk::FALSE
}

// physical device를 검사하고 적합한지 확인
unsafe fn check_physical_device(
    instance: &Instance,
    data: &AppData,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    // 장치의 속성을 가져옴
    // let properties = instance.get_physical_device_properties(physical_device);
    // 장치의 기능을 가져옴
    // let features = instance.get_physical_device_features(physical_device);

    //
    QueueFamilyIndices::get(instance, data, physical_device)?;

    Ok(())
}

// physical device를 찾아서 선택하고 AppData에 저장
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

    // 필요한 인스턴스 확장을 가져옴
    let mut extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    if VALIDATION_ENABLED {
        // 디버그 유틸 확장 추가
        // 디버그 메세지를 핸들링하기 위해 필요함
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

// logical device를 생성
unsafe fn create_logical_device(
    entry: &Entry,
    instance: &Instance,
    data: &mut AppData,
) -> Result<Device> {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;

    let queue_priorities = &[1.0];
    let queue_info = vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(indices.graphics)
        .queue_priorities(queue_priorities);

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        vec![]
    };

    let mut extensions = vec![];

    // Required by Vulkan SDK on macOS since 1.3.216.
    if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        extensions.push(vk::KHR_PORTABILITY_SUBSET_EXTENSION.name.as_ptr());
    }

    let features = vk::PhysicalDeviceFeatures::builder();

    let queue_infos = &[queue_info];
    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(queue_infos)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .enabled_features(&features);

    let device = instance.create_device(data.physical_device, &info, None)?;

    data.graphics_queue = device.get_device_queue(indices.graphics, 0);

    Ok(device)
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
                    unsafe { app.render(&window) }.unwrap()
                }
                // Destroy our Vulkan app.
                WindowEvent::CloseRequested => {
                    elwt.exit();
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

/// Our Vulkan app.
#[derive(Clone, Debug)]
struct App {
    entry: Entry,
    instance: Instance,
    data: AppData,
    device: Device,
}

impl App {
    /// Creates our Vulkan app.
    unsafe fn create(window: &Window) -> Result<Self> {
        // Vulkan command를 Vulkan shared library에서 로드하기 위해 사용됨
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let mut data = AppData::default();
        let instance = create_instance(window, &entry, &mut data)?;

        pick_physical_device(&instance, &mut data)?;

        let device = create_logical_device(&entry, &instance, &mut data)?;

        Ok(Self {
            entry,
            instance,
            data,
            device,
        })
    }

    /// Renders a frame for our Vulkan app.
    unsafe fn render(&mut self, window: &Window) -> Result<()> {
        Ok(())
    }

    /// Destroys our Vulkan app.
    /// vk::DebugUtilsMessengerEXT오브젝트는 앱이 종료되기 전에 cleanup되어야 한다.
    unsafe fn destroy(&mut self) {
        if VALIDATION_ENABLED {
            // 프로그램이 종료되기 전에 디버그 메세지 핸들러를 파괴
            self.instance
                .destroy_debug_utils_messenger_ext(self.data.messenger, None);
        }

        // 프로그램이 종료되면 인스턴스를 파괴해야 함
        self.instance.destroy_instance(None);
        self.device.destroy_device(None);
    }
}

#[derive(Debug, Error)]
#[error("Missing {0}.")]
pub struct SuitabilityError(pub &'static str);

/// The Vulkan handles and associated properties used by our Vulkan app.
#[derive(Clone, Debug, Default)]
struct AppData {
    // 디버그 메세지를 처리하기 위한 메세지 핸들러
    messenger: vk::DebugUtilsMessengerEXT,
    // physical device 핸들
    physical_device: vk::PhysicalDevice,
    // logical device와 함께 생성된 queue를 컨트롤하기 위한 핸들
    graphics_queue: vk::Queue,
}

#[derive(Copy, Clone, Debug)]
struct QueueFamilyIndices {
    graphics: u32,
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

        if let Some(graphics) = graphics {
            Ok(Self { graphics })
        } else {
            Err(anyhow!(SuitabilityError(
                "Missing required queue families."
            )))
        }
    }
}

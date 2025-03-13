use anyhow::Context;
use cgmath::Vector4;
use hist::*;
use std::sync::Arc;
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{CommandBufferExecFuture, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo};
use vulkano::image::{Image, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::library::VulkanLibrary;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::Pipeline;
use vulkano::render_pass::RenderPass;
use vulkano::shader::ShaderModule;
use vulkano::swapchain::{
    self, PresentFuture, Surface, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo,
    SwapchainPresentInfo,
};
use vulkano::sync::future::{FenceSignalFuture, JoinFuture};
use vulkano::sync::{self, GpuFuture};
use vulkano::{Validated, VulkanError};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::raw_window_handle::HasDisplayHandle;
use winit::window::Window;

struct VkContext {
    device: Arc<Device>,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    images: Vec<Arc<Image>>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    vertex_buffer: Subbuffer<[MyVertex]>,
    uniform_buffer_allocator: SubbufferAllocator,
    descriptor_set: Arc<DescriptorSet>,
    descriptor_set_layout_index: u32,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>>,
}

fn rgb_cycle(counter: f32) -> cgmath::Vector4<f32> {
    let phase = counter * 6.0; // Scale to match the 6 phases of the color wheel
    let phase_int = phase.floor() as i32; // Determine the current phase

    match phase_int {
        0 => Vector4::new(1.0, phase.fract(), 0.0, 1.0), // Red to Yellow
        1 => Vector4::new(1.0 - phase.fract(), 1.0, 0.0, 1.0), // Yellow to Green
        2 => Vector4::new(0.0, 1.0, phase.fract(), 1.0), // Green to Cyan
        3 => Vector4::new(0.0, 1.0 - phase.fract(), 1.0, 1.0), // Cyan to Blue
        4 => Vector4::new(phase.fract(), 0.0, 1.0, 1.0), // Blue to Magenta
        5 => Vector4::new(1.0, 0.0, 1.0 - phase.fract(), 1.0), // Magenta to Red
        _ => unreachable!(),                             // Should never reach here
    }
}

impl VkContext {
    fn new(window: Arc<Window>, event_loop: &impl HasDisplayHandle) -> anyhow::Result<VkContext> {
        let library = VulkanLibrary::new()?;
        let required_extensions = Surface::required_extensions(&event_loop)?;
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )?;

        let surface = Surface::from_window(instance.clone(), window.clone())?;

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) =
            select_physical_device(&instance, &surface, &device_extensions)?;

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                ..Default::default()
            },
        )
        .context("failed to create device")?;

        let queue = queues.next().context("no queue found")?;

        let caps = physical_device
            .surface_capabilities(&surface, Default::default())
            .context("failed to get surface capabilities")?;

        let dimensions = window.inner_size();
        let composite_alpha = caps
            .supported_composite_alpha
            .into_iter()
            .next()
            .context("no supported composite alpha found")?;
        let image_format = physical_device.surface_formats(&surface, Default::default())?[0].0;

        let (swapchain, images) = Swapchain::new(
            device.clone(),
            surface,
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count + 1,
                image_format,
                image_extent: dimensions.into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha,
                ..Default::default()
            },
        )?;

        let render_pass = get_render_pass(device.clone(), &swapchain)?;
        let framebuffers = get_framebuffers(&images, render_pass.clone())?;

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let vertex1 = MyVertex {
            position: [-0.5, -0.5],
        };
        let vertex2 = MyVertex {
            position: [0.0, 0.5],
        };
        let vertex3 = MyVertex {
            position: [0.5, -0.25],
        };
        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![vertex1, vertex2, vertex3],
        )?;

        let uniform_buffer_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let vs = load_shader(device.clone(), "./shader/test.vert.spv")
            .context("failed to create vertex shader")?;
        let fs = load_shader(device.clone(), "./shader/test.frag.spv")
            .context("failed to create fragment shader")?;

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window.inner_size().into(),
            depth_range: 0.0..=1.0,
        };

        let pipeline = get_pipeline(
            device.clone(),
            vs.clone(),
            fs.clone(),
            render_pass.clone(),
            viewport.clone(),
        )?;

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let uniform_buffer = uniform_buffer_allocator.allocate_sized()?;

        let ub = UB {
            color: cgmath::Vector4 {
                x: 0.0,
                y: 1.0,
                z: 1.0,
                w: 1.0,
            },
        };

        *uniform_buffer.write().unwrap() = ub;

        let descriptor_set_layout_index = 0;
        let descriptor_set_layout = pipeline
            .layout()
            .set_layouts()
            .get(descriptor_set_layout_index)
            .unwrap();

        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            descriptor_set_layout.clone(),
            [WriteDescriptorSet::buffer(0, uniform_buffer.clone())],
            [],
        )?;

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let command_buffers = get_command_buffers(
            &command_buffer_allocator,
            &queue,
            &pipeline,
            &framebuffers,
            &descriptor_set,
            descriptor_set_layout_index as u32,
            &vertex_buffer,
        )?;

        Ok(VkContext {
            device,
            queue,
            swapchain,
            images,
            render_pass,
            viewport,
            vs,
            fs,
            vertex_buffer,
            uniform_buffer_allocator,
            descriptor_set,
            descriptor_set_layout_index: descriptor_set_layout_index as u32,
            command_buffer_allocator,
            command_buffers,
        })
    }
}

#[derive(Default)]
struct App {
    window: Option<Arc<Window>>,
    context: Option<VkContext>,

    windows_resized: bool,
    recreate_swapchain: bool,

    counter: f32,

    fences: Vec<
        Option<
            Arc<
                FenceSignalFuture<
                    PresentFuture<
                        CommandBufferExecFuture<
                            JoinFuture<Box<dyn GpuFuture>, SwapchainAcquireFuture>,
                        >,
                    >,
                >,
            >,
        >,
    >,
    previous_fence_i: u32,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let attributes = Window::default_attributes();

        if let Ok(window) = event_loop.create_window(attributes) {
            let first_window_handle = self.window.is_none();
            let _window_handle = Arc::new(window);

            if first_window_handle {
                let window = Arc::new(
                    event_loop
                        .create_window(Window::default_attributes().with_visible(true))
                        .unwrap(),
                );
                self.window = Some(window.clone());

                let vk_context = VkContext::new(window.clone(), &event_loop).unwrap();

                let frames_in_flight = vk_context.images.len();
                self.fences = vec![None; frames_in_flight];

                self.context = Some(vk_context);
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                self.window
                    .as_ref()
                    .expect("resize without a window")
                    .request_redraw();

                self.windows_resized = true;
            }
            WindowEvent::RedrawRequested => {
                let window = self
                    .window
                    .as_ref()
                    .expect("redraw request without a window");

                window.pre_present_notify();

                let VkContext {
                    device,
                    queue,
                    ref mut swapchain,
                    render_pass,
                    ref mut viewport,
                    vs,
                    fs,
                    vertex_buffer,
                    uniform_buffer_allocator,
                    ref mut descriptor_set,
                    descriptor_set_layout_index,
                    command_buffer_allocator,
                    ref mut command_buffers,
                    ..
                } = self.context.as_mut().expect("no context found");

                self.counter += 0.01;

                println!("{}", self.counter);

                let uniform_buffer = uniform_buffer_allocator.allocate_sized().unwrap();

                let ub = UB {
                    color: rgb_cycle(self.counter),
                };

                *uniform_buffer.write().unwrap() = ub;

                // i have no fucking idea
                unsafe {
                    descriptor_set
                        .update_by_ref([WriteDescriptorSet::buffer(0, uniform_buffer.clone())], [])
                        .unwrap();
                }

                if self.windows_resized || self.recreate_swapchain {
                    self.recreate_swapchain = false;
                    let new_dimensions = window.inner_size();

                    let (new_swapchain, new_images) = swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent: new_dimensions.into(),
                            ..swapchain.create_info()
                        })
                        .unwrap();

                    *swapchain = new_swapchain;
                    let new_framebuffers =
                        get_framebuffers(&new_images, render_pass.clone()).unwrap();

                    if self.windows_resized {
                        self.windows_resized = false;

                        viewport.extent = new_dimensions.into();
                        let new_pipeline = get_pipeline(
                            device.clone(),
                            vs.clone(),
                            fs.clone(),
                            render_pass.clone(),
                            viewport.clone(),
                        )
                        .unwrap();

                        *command_buffers = get_command_buffers(
                            &command_buffer_allocator,
                            &queue,
                            &new_pipeline,
                            &new_framebuffers,
                            &descriptor_set,
                            *descriptor_set_layout_index,
                            &vertex_buffer,
                        )
                        .unwrap();
                    }
                }

                let (image_i, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(swapchain.clone(), None)
                        .map_err(Validated::unwrap)
                    {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) => {
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };

                if suboptimal {
                    self.recreate_swapchain = true;
                }

                if let Some(image_fence) = &self.fences[image_i as usize] {
                    image_fence.wait(None).unwrap();
                }

                let previous_future = match self.fences[self.previous_fence_i as usize].clone() {
                    None => {
                        let mut now = sync::now(device.clone());
                        now.cleanup_finished();

                        now.boxed()
                    }
                    Some(fence) => fence.boxed(),
                };

                let future = previous_future
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffers[image_i as usize].clone())
                    .unwrap()
                    .then_swapchain_present(
                        queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_i),
                    )
                    .then_signal_fence_and_flush();

                self.fences[image_i as usize] = match future.map_err(Validated::unwrap) {
                    Ok(value) => Some(Arc::new(value)),
                    Err(VulkanError::OutOfDate) => {
                        self.recreate_swapchain = true;
                        None
                    }
                    Err(e) => {
                        println!("failed to flush future: {e}");
                        None
                    }
                };

                self.previous_fence_i = image_i;
            }
            _ => (),
        }
    }
}

fn main() -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;

    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).map_err(Into::into)
}

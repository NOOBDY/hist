use anyhow::Context;
use cgmath::Vector4;
use clap::Parser;
use hist::*;
use std::path::PathBuf;
use std::sync::Arc;
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::DescriptorBindingFlags;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
};
use vulkano::image::ImageUsage;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::library::VulkanLibrary;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, ColorBlendAttachmentState, ColorBlendState,
};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{Framebuffer, RenderPass, Subpass};
use vulkano::swapchain::{
    acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::{self, GpuFuture};
use vulkano::{Validated, VulkanError};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Window;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, default_value = "./shader/test.vert.spv")]
    vertex: PathBuf,

    #[arg(short, default_value = "./shader/test.frag.spv")]
    fragment: PathBuf,
}

struct VkContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    pipeline: Arc<GraphicsPipeline>,
    viewport: Viewport,
    descriptor_set: Arc<DescriptorSet>,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

struct App {
    counter: i32,

    vs_filepath: PathBuf,
    fs_filepath: PathBuf,

    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    uniform_buffer_allocator: SubbufferAllocator,
    vertex_buffer: Subbuffer<[MyVertex]>,
    index_buffer: Subbuffer<[u32]>,
    rcx: Option<VkContext>,
}

impl App {
    fn new(
        event_loop: &EventLoop<()>,
        vs_filepath: PathBuf,
        fs_filepath: PathBuf,
    ) -> anyhow::Result<App> {
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

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) =
            select_physical_device(&instance, event_loop, &device_extensions)?;

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                enabled_features: DeviceFeatures {
                    descriptor_indexing: true,
                    shader_sampled_image_array_non_uniform_indexing: true,
                    runtime_descriptor_array: true,
                    descriptor_binding_variable_descriptor_count: true,
                    ..DeviceFeatures::empty()
                },

                ..Default::default()
            },
        )
        .context("failed to create device")?;

        let queue = queues.next().context("no queue found")?;

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let uniform_buffer_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let vertex1 = MyVertex {
            position: [1.0, 1.0],
        };
        let vertex2 = MyVertex {
            position: [-1.0, 1.0],
        };
        let vertex3 = MyVertex {
            position: [-1.0, -1.0],
        };
        let vertex4 = MyVertex {
            position: [1.0, -1.0],
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
            vec![vertex1, vertex2, vertex3, vertex4],
        )?;

        let indices = vec![0, 2, 1, 0, 3, 2];

        let index_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            indices,
        )?;

        // let uploads = AutoCommandBufferBuilder::primary(
        //     command_buffer_allocator.clone(),
        //     queue.queue_family_index(),
        //     CommandBufferUsage::OneTimeSubmit,
        // )?;

        // let data = UB {
        //     color: Vector4::new(1.0, 0.0, 0.0, 1.0),
        // };

        // let buf = Buffer::from_data(
        //     memory_allocator.clone(),
        //     BufferCreateInfo {
        //         usage: BufferUsage::UNIFORM_BUFFER,
        //         ..Default::default()
        //     },
        //     AllocationCreateInfo {
        //         memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
        //             | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
        //         ..Default::default()
        //     },
        //     data,
        // )?;

        // let _ = uploads.build()?.execute(queue.clone())?;

        Ok(App {
            counter: 0,
            vs_filepath,
            fs_filepath,
            instance,
            device,
            queue,
            descriptor_set_allocator,
            command_buffer_allocator,
            uniform_buffer_allocator,
            vertex_buffer,
            index_buffer,
            rcx: None,
        })
    }

    fn resumed_internal(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
    ) -> anyhow::Result<()> {
        let window = Arc::new(event_loop.create_window(Window::default_attributes())?);
        let surface = Surface::from_window(self.instance.clone(), window.clone())?;
        let window_size = window.inner_size();

        let (swapchain, images) = {
            let surface_capabilites = self
                .device
                .physical_device()
                .surface_capabilities(&surface, Default::default())?;

            let (image_format, _) = self
                .device
                .physical_device()
                .surface_formats(&surface, Default::default())?[0];

            Swapchain::new(
                self.device.clone(),
                surface,
                SwapchainCreateInfo {
                    min_image_count: surface_capabilites.min_image_count.max(2),
                    image_format,
                    image_extent: window_size.into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha: surface_capabilites
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .context("")?,
                    ..Default::default()
                },
            )?
        };

        let render_pass = vulkano::single_pass_renderpass! {
            self.device.clone(),
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            }
        }?;

        let framebuffers = get_framebuffers(&images, &render_pass)?;

        let pipeline = {
            let vs = load_shader(self.device.clone(), &self.vs_filepath)?
                .entry_point("main")
                .context("entry point not found")?;
            let fs = load_shader(self.device.clone(), &self.fs_filepath)?
                .entry_point("main")
                .context("entry point not found")?;

            let vertex_input_state = MyVertex::per_vertex().definition(&vs)?;
            let stages = [
                PipelineShaderStageCreateInfo::new(vs.clone()),
                PipelineShaderStageCreateInfo::new(fs.clone()),
            ];

            let layout = {
                let mut layout_create_info =
                    PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);

                println!("{:?}", layout_create_info.set_layouts[0].bindings);

                println!("{:?}", fs.info().descriptor_binding_requirements);

                let binding = layout_create_info.set_layouts[0]
                    .bindings
                    .get_mut(&0)
                    .context("binding not found")?;
                binding.binding_flags |= DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT;
                binding.descriptor_count = 1;

                PipelineLayout::new(
                    self.device.clone(),
                    layout_create_info.into_pipeline_layout_create_info(self.device.clone())?,
                )?
            };

            let subpass = Subpass::from(render_pass.clone(), 0).context("subpass not found")?;

            GraphicsPipeline::new(
                self.device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState {
                            blend: Some(AttachmentBlend::alpha()),
                            ..Default::default()
                        },
                    )),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )?
        };

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
        };

        let buf = self.uniform_buffer_allocator.allocate_sized()?;

        let ub = UB {
            color: Vector4 {
                x: 1.0,
                y: 1.0,
                z: 1.0,
                w: 0.0,
            },
        };

        *buf.write().unwrap() = ub;

        let layout = &pipeline.layout().set_layouts()[0];
        let descriptor_set = DescriptorSet::new_variable(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            1,
            [WriteDescriptorSet::buffer(0, buf)],
            [],
        )?;

        let previous_frame_end = Some(sync::now(self.device.clone()).boxed());

        self.rcx = Some(VkContext {
            window,
            swapchain,
            render_pass,
            framebuffers,
            pipeline,
            viewport,
            descriptor_set,
            recreate_swapchain: false,
            previous_frame_end,
        });

        Ok(())
    }

    fn window_event_internal(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) -> anyhow::Result<()> {
        let ctx = self.rcx.as_mut().context("no vk context found")?;

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                ctx.recreate_swapchain = true;
            }
            WindowEvent::RedrawRequested => {
                let window_size = ctx.window.inner_size();

                if window_size.width == 0 || window_size.height == 0 {
                    return Ok(());
                }

                ctx.previous_frame_end
                    .as_mut()
                    .context("previous_frame_end is None")?
                    .cleanup_finished();

                if ctx.recreate_swapchain {
                    let (new_swapchain, new_images) =
                        ctx.swapchain.recreate(SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            ..ctx.swapchain.create_info()
                        })?;

                    ctx.swapchain = new_swapchain;
                    ctx.framebuffers = get_framebuffers(&new_images, &ctx.render_pass)?;
                    ctx.viewport.extent = window_size.into();
                    ctx.recreate_swapchain = false;
                }

                let (image_index, suboptimal, acquire_future) = match acquire_next_image(
                    ctx.swapchain.clone(),
                    None,
                )
                .map_err(Validated::unwrap)
                {
                    Ok(r) => r,
                    Err(VulkanError::OutOfDate) => {
                        ctx.recreate_swapchain = true;
                        return Ok(());
                    }
                    Err(e) => {
                        return Err(anyhow::anyhow!("failed to acquire next image: {e}"));
                    }
                };

                if suboptimal {
                    ctx.recreate_swapchain = true;
                }

                self.counter += 1;

                let buf = self.uniform_buffer_allocator.allocate_sized()?;

                let ub = UB {
                    color: rgb_cycle(self.counter),
                };

                *buf.write().unwrap() = ub;

                unsafe {
                    ctx.descriptor_set
                        .update_by_ref([WriteDescriptorSet::buffer(0, buf)], [])?;
                }

                let mut builder = AutoCommandBufferBuilder::primary(
                    self.command_buffer_allocator.clone(),
                    self.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )?;

                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
                            ..RenderPassBeginInfo::framebuffer(
                                ctx.framebuffers[image_index as usize].clone(),
                            )
                        },
                        Default::default(),
                    )?
                    .set_viewport(0, [ctx.viewport.clone()].into_iter().collect())?
                    .bind_pipeline_graphics(ctx.pipeline.clone())?
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        ctx.pipeline.layout().clone(),
                        0,
                        ctx.descriptor_set.clone(),
                    )?
                    .bind_vertex_buffers(0, self.vertex_buffer.clone())?
                    .bind_index_buffer(self.index_buffer.clone())?;

                unsafe {
                    builder.draw_indexed(self.index_buffer.len() as u32, 1, 0, 0, 0)?;
                }

                builder.end_render_pass(Default::default())?;

                let command_buffer = builder.build()?;
                let future = ctx
                    .previous_frame_end
                    .take()
                    .context("previous_frame_end is None")?
                    .join(acquire_future)
                    .then_execute(self.queue.clone(), command_buffer)?
                    .then_swapchain_present(
                        self.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            ctx.swapchain.clone(),
                            image_index,
                        ),
                    )
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => ctx.previous_frame_end = Some(future.boxed()),
                    Err(VulkanError::OutOfDate) => {
                        ctx.recreate_swapchain = true;
                        ctx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("failed to flush future: {e}");
                        ctx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                    }
                }

                ctx.window.request_redraw();
            }
            _ => {}
        }

        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.resumed_internal(event_loop).unwrap();
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        self.window_event_internal(event_loop, window_id, event)
            .unwrap();
    }
}

fn main() -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;

    let args = Args::parse();

    println!("{:?}", args.vertex);
    println!("{:?}", args.fragment);

    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(&event_loop, args.vertex, args.fragment)?;

    event_loop.run_app(&mut app).map_err(Into::into)
}

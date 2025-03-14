use anyhow::Context;
use std::path::Path;
use std::sync::Arc;
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
    SubpassBeginInfo, SubpassContents, SubpassEndInfo,
};
use vulkano::descriptor_set::DescriptorSet;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceExtensions, Queue, QueueFlags};
use vulkano::image::view::ImageView;
use vulkano::image::Image;
use vulkano::instance::Instance;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};
use vulkano::swapchain::{Surface, Swapchain};

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct MyVertex {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct UB {
    pub color: cgmath::Vector4<f32>,
}

unsafe impl bytemuck::Pod for UB {}
unsafe impl bytemuck::Zeroable for UB {}

pub fn select_physical_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface>,
    device_extensions: &DeviceExtensions,
) -> anyhow::Result<(Arc<PhysicalDevice>, u32)> {
    instance
        .enumerate_physical_devices()
        .context("could not enumerate devices")?
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|q| (p, q as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 4,
        })
        .context("no devices available")
}

pub fn get_render_pass(
    device: Arc<Device>,
    swapchain: &Arc<Swapchain>,
) -> anyhow::Result<Arc<RenderPass>> {
    vulkano::single_pass_renderpass!(
        device,
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
            depth_stencil: {}
        }
    )
    .context("failed to get render pass")
}

pub fn get_framebuffers(
    images: &[Arc<Image>],
    render_pass: Arc<RenderPass>,
) -> anyhow::Result<Vec<Arc<Framebuffer>>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone())?;
            let framebuffer = Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )?;
            Ok(framebuffer)
        })
        .collect()
}

pub fn get_pipeline(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> anyhow::Result<Arc<GraphicsPipeline>> {
    let vs = vs.entry_point("main").context("entry point not found")?;
    let fs = fs.entry_point("main").context("entry point not found")?;

    let vertex_input_state = MyVertex::per_vertex().definition(&vs)?;

    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];

    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())?,
    )?;

    let subpass = Subpass::from(render_pass.clone(), 0).context("failed to create subpass")?;

    GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState {
                viewports: [viewport].into_iter().collect(),
                ..Default::default()
            }),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .context("failed to create graphics pipeline")
}

pub fn get_command_buffers(
    command_buffer_allocator: &Arc<StandardCommandBufferAllocator>,
    queue: &Arc<Queue>,
    pipeline: &Arc<GraphicsPipeline>,
    framebuffers: &Vec<Arc<Framebuffer>>,
    descriptor_set: &Arc<DescriptorSet>,
    descriptor_set_index: u32,
    vertex_buffer: &Subbuffer<[MyVertex]>,
    index_buffer: &Subbuffer<[u32]>,
) -> anyhow::Result<Vec<Arc<PrimaryAutoCommandBuffer>>> {
    framebuffers
        .iter()
        .map(|framebuffer| {
            let mut builder = AutoCommandBufferBuilder::primary(
                command_buffer_allocator.clone(),
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )?;

            unsafe {
                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0.1, 0.1, 0.1, 1.0].into())],
                            ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                        },
                        SubpassBeginInfo {
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )?
                    .bind_pipeline_graphics(pipeline.clone())?
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipeline.layout().clone(),
                        descriptor_set_index,
                        descriptor_set.clone(),
                    )?
                    .bind_vertex_buffers(0, vertex_buffer.clone())?
                    .bind_index_buffer(index_buffer.clone())?
                    .draw_indexed(index_buffer.len() as u32, 1, 0, 0, 0)?
                    .end_render_pass(SubpassEndInfo::default())?;
            };

            Ok(builder.build()?)
        })
        .collect()
}

pub fn load_shader<P>(device: Arc<Device>, filepath: P) -> anyhow::Result<Arc<ShaderModule>>
where
    P: AsRef<Path>,
{
    let bytes = std::fs::read(filepath)?;
    let words = vulkano::shader::spirv::bytes_to_words(&bytes)?;
    let shader = unsafe { ShaderModule::new(device, ShaderModuleCreateInfo::new(&words))? };
    Ok(shader)
}

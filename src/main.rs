mod texture;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use futures::executor::block_on;
use naga::{front::wgsl, valid::Validator};
use notify::{Config, RecommendedWatcher, Watcher};
use std::{
    borrow::Cow,
    fs::{read_to_string, OpenOptions},
    io::Write,
    path::PathBuf,
    sync::mpsc::channel,
    time::Instant,
};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Adapter, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BufferBindingType, BufferUsages, ColorTargetState, CommandEncoderDescriptor,
    CompositeAlphaMode, Device, DeviceDescriptor, Features, Limits, LoadOp, Operations,
    PrimitiveState, Queue, RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline,
    RequestAdapterOptions, ShaderSource, ShaderStages, Surface, SurfaceConfiguration,
    TextureFormat,
};
use winit::event::Event::UserEvent;
use winit::{
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy},
    window::{Window, WindowBuilder, WindowLevel},
};

#[derive(Debug)]
enum UserEvents {
    Reload,
    WGPUError,
}

#[derive(Parser)]
struct Opts {
    wgsl_file: PathBuf,

    #[clap(short, long)]
    create: bool,

    #[clap(short, long)]
    always_on_top: bool,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod)]
struct Uniforms {
    pub mouse: [f32; 2],
    pub time: f32,
    pub pad: f32,
    pub window_size: [f32; 2],
}

impl Default for Uniforms {
    fn default() -> Uniforms {
        Uniforms {
            time: 0.,
            mouse: [0.0, 0.0],
            pad: 0.,
            window_size: [0., 0.],
        }
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone, ValueEnum)]
enum WindowLevelArg {
    Normal,
    AlwaysOnTop,
    AlwaysOnBottom,
}

impl From<WindowLevelArg> for WindowLevel {
    fn from(value: WindowLevelArg) -> Self {
        match value {
            WindowLevelArg::AlwaysOnBottom => WindowLevel::AlwaysOnBottom,
            WindowLevelArg::Normal => WindowLevel::Normal,
            WindowLevelArg::AlwaysOnTop => WindowLevel::AlwaysOnTop,
        }
    }
}

#[derive(Parser)]
struct Opts {
    fs_file: PathBuf,

    #[clap(short, long)]
    create: bool,

    #[arg(short, long, value_enum, default_value_t = WindowLevelArg::AlwaysOnTop)]
    window_level: WindowLevelArg,

    vs_file: Option<PathBuf>,
}

struct Instance {
    column_offset: f32,
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model: self.column_offset,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: f32,
}

impl InstanceRaw {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32,
            }],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod)]
struct VertexUniforms {
    pub tex_size: [f32; 2],
    pub resolution: [f32; 2],
}

impl Default for VertexUniforms {
    fn default() -> VertexUniforms {
        VertexUniforms {
            tex_size: [64., 64.],
            resolution: [512., 512.],
        }
    }
}

impl VertexUniforms {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod)]
struct FragmentUniforms {
    pub index: u32,
}

impl Default for FragmentUniforms {
    fn default() -> FragmentUniforms {
        FragmentUniforms { index: 0 }
    }
}

impl FragmentUniforms {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

fn recreate_pipeline(playground: &Playground) -> Result<Renderer> {
    println!("Reloading render pipeline.");

    let surface = unsafe { playground.instance.create_surface(&playground.window)? };
    let (adapter, device, queue) =
        block_on(Playground::get_async_stuff(&playground.instance, &surface));

    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps
        .formats
        .iter()
        .copied()
        .filter(|f| f.is_srgb())
        .next()
        .unwrap_or(surface_caps.formats[0]);

    let surface_config = SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: playground.size.width,
        height: playground.size.height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Auto,
        view_formats: vec![],
    };

    surface.configure(&device, &surface_config);
    playground.create_pipeline(
        adapter,
        device,
        queue,
        surface,
        surface_config,
        surface_format,
    )
}

struct Playground {
    pub frag_shader_path: PathBuf,
    pub vert_shader_path: PathBuf,
    pub renderer: Option<Renderer>,
    pub window: Window,
    pub size: PhysicalSize<u32>,
    pub vertex_uniforms: VertexUniforms,
    pub fragment_uniforms: FragmentUniforms,
    pub instance: wgpu::Instance,
    pub error_state: bool,
}

struct Renderer {
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub surface: Surface,
    pub surface_config: SurfaceConfiguration,
    pub surface_format: TextureFormat,
    pub render_pipeline: RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub vertex_uniforms_buffer: wgpu::Buffer,
    pub fragment_uniforms_buffer: wgpu::Buffer,
    pub diffuse_bind_group: wgpu::BindGroup,
    pub uniforms_buffer_bind_group: wgpu::BindGroup,
    //clip_rect: (u32, u32, u32, u32),
}

impl Playground {
    fn reload(&mut self) {
        let pipeline = recreate_pipeline(self);

        match pipeline {
            Ok(renderer) => {
                let _ = self.renderer.replace(renderer);
            }
            Err(e) => {
                let _ = self.renderer.take();
                println!("{}", e);
            }
        }

        self.window.request_redraw();
    }

    fn listen(watch_path_0: PathBuf, watch_path_1: PathBuf, proxy: EventLoopProxy<UserEvents>) {
        let (tx, rx) = channel();

        let mut watcher: RecommendedWatcher =
            RecommendedWatcher::new(tx, Config::default()).unwrap();
        watcher
            .watch(&watch_path_0, notify::RecursiveMode::NonRecursive)
            .unwrap();
        watcher
            .watch(&watch_path_1, notify::RecursiveMode::NonRecursive)
            .unwrap();

        loop {
            match rx.recv() {
                Ok(RawEvent {
                    path: Some(_),
                    op: Ok(_),
                    ..
                }) => {
                    proxy.send_event(UserEvents::Reload).unwrap();
                }
                Err(e) => println!("watch error: {:?}", e),
            }
        }
    }

    async fn get_async_stuff(
        instance: &wgpu::Instance,
        surface: &Surface,
    ) -> (Adapter, Device, Queue) {
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    features: Features::empty(),
                    limits: Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        (adapter, device, queue)
    }

    fn create_pipeline(
        &self,
        adapter: Adapter,
        device: Device,
        queue: Queue,
        surface: Surface,
        surface_config: SurfaceConfiguration,
        surface_format: TextureFormat,
    ) -> anyhow::Result<Renderer> {
        let frag_wgsl = match read_to_string(self.frag_shader_path.clone()) {
            Ok(res) => res,
            Err(e) => {
                return Err(anyhow::anyhow!("Unable to load frag shader: {:?}", e));
            }
        };
        let vert_wgsl = match read_to_string(self.vert_shader_path.clone()) {
            Ok(res) => res,
            Err(e) => {
                return Err(anyhow::anyhow!("Unable to load vert shader: {:?}", e));
            }
        };

        let fg_module = match wgsl::parse_str(&frag_wgsl) {
            Ok(res) => res,
            Err(e) => {
                return Err(anyhow::anyhow!("Syntax Error in Shader: {:?}", e));
            }
        };

        let vs_module = match wgsl::parse_str(&vert_wgsl) {
            Ok(res) => res,
            Err(e) => {
                return Err(anyhow::anyhow!("Syntax Error in Shader: {:?}", e));
            }
        };

        let mut validator = Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );

        let fg_validation_res = validator.validate(&fg_module);
        let vs_validation_result = validator.validate(&vs_module);

        if fg_validation_res.is_err() || vs_validation_result.is_err() {
            return Err(anyhow::anyhow!("Shaders failed compilation."));
        }

        // Fragment Shader
        let fragement_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fragment shader"),
            source: ShaderSource::Wgsl(Cow::Owned(frag_wgsl)),
        });

        let vertex_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vertex shader"),
            source: wgpu::ShaderSource::Wgsl(vert_wgsl.into()),
        });

        // Create vertex buffer; array-of-array of position and texture coordinates
        let vertex_data: [[f32; 2]; 4] = [
            // One full-screen triangle
            [-1.0, 1.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [1.0, -1.0],
        ];

        let vertex_data_slice = bytemuck::cast_slice(&vertex_data);
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pixels_scaling_renderer_vertex_buffer"),
            contents: vertex_data_slice,
            usage: wgpu::BufferUsages::VERTEX,
        });

        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: (vertex_data_slice.len() / vertex_data.len()) as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            }],
        };

        let vertex_uniforms = VertexUniforms::default();
        let fragment_uniforms = VertexUniforms::default();

        let vertex_uniforms_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("vertex_uniforms_buffer"),
            contents: vertex_uniforms.as_bytes(),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let fragment_uniforms_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("fragment_uniforms_buffer"),
            contents: fragment_uniforms.as_bytes(),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let uniforms_buffer_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("uniforms_buffer_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<VertexUniforms>() as _,
                        ),
                    },
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<FragmentUniforms>() as _,
                        ),
                    },
                },
            ],
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("uniforms_and_texture"),
            bind_group_layouts: &[&uniforms_buffer_layout, &texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        // TODO: Load the file instead. Also allow reloading?
        let diffuse_bytes = include_bytes!("../default_noteskin.png");
        let diffuse_texture =
            texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "happy-tree.png", false)
                .unwrap();

        let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
            label: Some("diffuse_bind_group"),
        });

        let uniforms_buffer_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("uniforms_binding_group"),
            layout: &uniforms_buffer_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: vertex_uniforms_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: fragment_uniforms_buffer.as_entire_binding(),
                },
            ],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader_module,
                entry_point: "vs_main",
                buffers: &[vertex_buffer_layout, InstanceRaw::desc()],
            },
            primitive: PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            fragment: Some(wgpu::FragmentState {
                module: &fragement_shader_module,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    format: surface_config.format,
                    write_mask: wgpu::ColorWrites::default(),
                })],
            }),
        });

        Ok(Renderer {
            render_pipeline,
            vertex_buffer,
            vertex_uniforms_buffer,
            fragment_uniforms_buffer,
            uniforms_buffer_bind_group,
            diffuse_bind_group,
            adapter,
            device,
            queue,
            surface,
            surface_config,
            surface_format,
        })
    }

    pub fn resize(&mut self, new_size: &PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            if let Some(renderer) = &mut self.renderer {
                renderer.surface_config.width = new_size.width;
                renderer.surface_config.height = new_size.height;

                renderer
                    .surface
                    .configure(&renderer.device, &renderer.surface_config);
            }

            self.window.request_redraw();
        }
    }

    pub fn run(opts: &Opts) {
        let event_loop = EventLoopBuilder::<UserEvents>::with_user_event().build();

        let proxy = event_loop.create_proxy();
        {
            let watch_fs_path = opts.fs_file.clone();
            let watch_vs_path = opts
                .vs_file
                .clone()
                .unwrap_or(PathBuf::from("./src/vertex.wgsl"));
            std::thread::spawn(move || Self::listen(watch_fs_path, watch_vs_path, proxy));
        }

        let window = WindowBuilder::new()
            .with_inner_size(PhysicalSize::new(512, 512))
            .with_title("WGSL Playground")
            .build(&event_loop)
            .unwrap();
        let size = window.inner_size();

        window.set_window_level(opts.window_level.into());

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });

        let mut playground = Playground {
            frag_shader_path: opts.fs_file.clone(),
            vert_shader_path: opts
                .vs_file
                .clone()
                .unwrap_or(PathBuf::from("./src/vertex.wgsl")),
            renderer: None,
            window,
            size,
            vertex_uniforms: VertexUniforms::default(),
            fragment_uniforms: FragmentUniforms::default(),
            instance,
            error_state: false,
        };

        playground.reload();

        let instant = Instant::now();

        let arrow_representations = Vec::<f32>::from([128., 48., -48., -128.]);

        let wgpu_error_proxy = event_loop.create_proxy();
        event_loop.run(move |event, _, control_flow| {
            match event {
                winit::event::Event::WindowEvent { ref event, .. } => match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(new_size) => playground.resize(new_size),
                    WindowEvent::CursorMoved { .. } => {}
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        playground.resize(new_inner_size)
                    }
                    _ => {}
                },
                winit::event::Event::RedrawRequested(_) => {
                    if let Some(renderer) = &playground.renderer {
                        let instances = arrow_representations
                            .iter()
                            .map(|arrow| Instance {
                                column_offset: *arrow,
                            })
                            .collect::<Vec<_>>();

                        let instance_data =
                            instances.iter().map(Instance::to_raw).collect::<Vec<_>>();

                        let instance_buffer =
                            renderer
                                .device
                                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some("Instance Buffer"),
                                    contents: bytemuck::cast_slice(&instance_data),
                                    usage: wgpu::BufferUsages::VERTEX,
                                });

                        let output_frame = renderer.surface.get_current_texture();
                        let output = output_frame.unwrap();
                        let view = output
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());

                        playground.fragment_uniforms.index =
                            instant.elapsed().as_secs_f32() as u32 % 10;
                        renderer.queue.write_buffer(
                            &renderer.vertex_uniforms_buffer,
                            0,
                            playground.vertex_uniforms.as_bytes(),
                        );
                        renderer.queue.write_buffer(
                            &renderer.fragment_uniforms_buffer,
                            0,
                            playground.fragment_uniforms.as_bytes(),
                        );

                        let mut encoder = renderer
                            .device
                            .create_command_encoder(&CommandEncoderDescriptor { label: None });

                        {
                            let mut render_pass =
                                encoder.begin_render_pass(&RenderPassDescriptor {
                                    label: None,
                                    color_attachments: &[Some(RenderPassColorAttachment {
                                        view: &view,
                                        resolve_target: None,
                                        ops: Operations {
                                            load: LoadOp::Clear(wgpu::Color::BLACK),
                                            store: true,
                                        },
                                    })],
                                    depth_stencil_attachment: None,
                                });
                            render_pass.set_pipeline(&renderer.render_pipeline);
                            render_pass.set_bind_group(
                                0,
                                &renderer.uniforms_buffer_bind_group,
                                &[],
                            );
                            render_pass.set_bind_group(1, &renderer.diffuse_bind_group, &[]);
                            render_pass.set_vertex_buffer(0, renderer.vertex_buffer.slice(..));
                            render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
                            render_pass.draw(0..4, 0..4);
                        }

                        renderer.queue.submit(std::iter::once(encoder.finish()));
                        output.present();
                    }
                }
                winit::event::Event::UserEvent(user_events) => {
                    match user_events {
                        UserEvents::Reload => {
                            playground.error_state = false;
                            playground.reload();

                            if let Some(renderer) = &playground.renderer {
                                let proxy = wgpu_error_proxy.to_owned();
                                renderer.device.on_uncaptured_error(Box::new(move |error| {
                                    // Sending the event will stop the redraw
                                    proxy.send_event(UserEvents::WGPUError).unwrap();

                                    if let wgpu::Error::Validation {
                                        source: _,
                                        description,
                                    } = error
                                    {
                                        if let Some(_) =
                                            description.find("note: label = `Fragment shader`")
                                        {
                                            println!("{}", description);
                                        }
                                    } else {
                                        println!("{}", error);
                                    }
                                }))
                            }
                        }
                        UserEvents::WGPUError => {
                            playground.error_state = true;
                        }
                    }
                }
                winit::event::Event::MainEventsCleared => {
                    if !playground.error_state {
                        playground.window.request_redraw();
                    }
                }
                _ => {}
            }
        });
    }
}

fn main() {
    wgpu_subscriber::initialize_default_subscriber(None);
    let opts = Opts::parse();

    if opts.create {
        let mut file = if let Ok(file) = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(opts.fs_file.clone())
        {
            file
        } else {
            println!(
                "Couldn't create file {:?}, make sure it doesn't already exist.",
                &opts.fs_file
            );
            return;
        };
        file.write_all(include_bytes!("frag.default.wgsl")).unwrap();
    }

    Playground::run(&opts);
}

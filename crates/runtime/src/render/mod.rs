//! Headless GPU render of a scene into a texture: background, skinned
//! characters, outline, per-frame readback into PNG with alpha.
//!
//! One renderer serves both the viewport (later) and offscreen export.
//! The Bevy World is a derived representation of the core document: the scene
//! world is rebuilt every frame from state computed by core.

pub mod mesh_gen;
pub mod metrics;
pub mod outline;

use std::path::Path;
use std::sync::mpsc;

use bevy::{
    asset::RenderAssetUsages,
    camera::RenderTarget,
    core_pipeline::tonemapping::Tonemapping,
    image::Image,
    mesh::{
        skinning::{SkinnedMesh, SkinnedMeshInverseBindposes},
        Indices,
    },
    pbr::MeshMaterial3d,
    prelude::*,
    render::{
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::{Extent3d, PrimitiveTopology, TextureDimension, TextureFormat, TextureUsages},
        settings::{Backends, RenderCreation, WgpuSettings},
    },
};
use phantom_core::{evaluate, evaluate_object, project::RigRef, Frame, Scene};

use crate::import::{load_model, ModelData};

#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error(transparent)]
    InvalidAction(#[from] phantom_core::CoreError),
    #[error("GPU unavailable: {0}")]
    GpuUnavailable(String),
    #[error("import: {0}")]
    Import(#[from] crate::import::ImportError),
    #[error("image: {0}")]
    Image(#[from] image::ImageError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("frame readback did not complete")]
    ReadbackTimeout,
}

#[derive(Clone, Debug)]
pub struct RenderOptions {
    pub width: u32,
    pub height: u32,
    /// "On twos" mode for the Phantom's pose.
    pub on_twos: bool,
}

/// Rendered frame: RGBA8 (sRGB), rows tightly packed.
pub struct RenderedFrame {
    pub frame: Frame,
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    /// Number of objects with at least one mesh part that passed Bevy
    /// visibility for the active camera of this frame.
    pub objects_in_frame: usize,
}

impl RenderedFrame {
    pub fn save_png(&self, path: &Path) -> Result<(), RenderError> {
        let img: image::RgbaImage =
            image::ImageBuffer::from_raw(self.width, self.height, self.rgba.clone())
                .expect("buffer size checked");
        img.save(path)?;
        Ok(())
    }
}

/// Marker for scene entities recreated on every frame.
#[derive(Component)]
struct SceneEntity;

/// Cache of assets that do not depend on the frame.
#[derive(Resource, Default)]
pub struct SceneCache {
    builtin: std::collections::HashMap<phantom_core::project::BuiltinActor, UploadedModel>,
}

/// Persistent live scene: structure is rebuilt only when the document
/// revision or the active scene changes; a frame only changes transforms.
#[derive(Default)]
pub struct LiveSync {
    revision: u64,
    scene: Option<phantom_core::SceneId>,
    forge: Option<ForgeResult>,
}

impl LiveSync {
    /// Syncs the world structure with the document if the document changed.
    pub fn ensure(
        &mut self,
        world: &mut World,
        project: &phantom_core::Project,
        scene_id: phantom_core::SceneId,
        project_dir: &Path,
        target: &Handle<Image>,
    ) -> Result<(), RenderError> {
        if self.scene == Some(scene_id) && self.revision == project.revision && self.forge.is_some() {
            return Ok(());
        }
        let scene = project
            .scenes
            .get(&scene_id)
            .ok_or_else(|| RenderError::GpuUnavailable(format!("scene {scene_id} not found")))?;
        self.forge = Some(forge_scene(world, project, scene, project_dir, target)?);
        self.scene = Some(scene_id);
        self.revision = project.revision;
        Ok(())
    }

    /// Applies a frame to the already-synced scene.
    pub fn apply(
        &self,
        world: &mut World,
        scene: &Scene,
        frame: Frame,
        on_twos: bool,
        control: Option<(phantom_core::PhantomId, glam::Vec3, f32, phantom_core::rig::Pose)>,
        onion_skin: bool,
        shot: Option<phantom_core::ShotId>,
    ) {
        if let Some(forge) = &self.forge {
            apply_scene_frame(world, scene, forge, frame, on_twos, control, onion_skin, shot);
        }
    }
}

pub struct Renderer {
    app: App,
    target: Handle<Image>,
    width: u32,
    height: u32,

}

impl Renderer {
    /// Creates a headless renderer. Without a suitable GPU it returns a
    /// diagnosable error; project editing is not blocked.
    pub fn new(width: u32, height: u32) -> Result<Self, RenderError> {
        let mut app = App::new();
        app.add_plugins(
            DefaultPlugins
                .set(bevy::render::RenderPlugin {
                    render_creation: RenderCreation::Automatic(WgpuSettings {
                        backends: Some(Backends::VULKAN),
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
        );
        app.add_plugins(outline::OutlinePlugin);
        app.init_resource::<SceneCache>();
        app.init_resource::<DrawCache>();
        app.init_resource::<VideoCache>();
        app.init_resource::<ClipCache>();
        app.finish();
        app.cleanup();

        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        let size = Extent3d { width, height, depth_or_array_layers: 1 };
        let mut image = Image::new_fill(
            size,
            TextureDimension::D2,
            &[0, 0, 0, 0],
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        );
        image.texture_descriptor.usage =
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC | TextureUsages::TEXTURE_BINDING;
        let target = images.add(image);
        drop(images);

        // Warm-up: the first iteration initializes the GPU device.
        app.update();
        let world = app.world_mut();
        if world.get_resource::<bevy::render::renderer::RenderDevice>().is_none() {
            return Err(RenderError::GpuUnavailable(
                "no Vulkan device found; check the GPU driver".into(),
            ));
        }

        Ok(Self { app, target, width, height })
    }

    /// Renders one scene frame. Frame time is derived from the frame number,
    /// not from wall clock or GPU load.
    pub fn render_frame(
        &mut self,
        project: &phantom_core::Project,
        scene: &Scene,
        project_dir: &Path,
        frame: Frame,
        opts: &RenderOptions,
        shot: Option<phantom_core::ShotId>,
    ) -> Result<RenderedFrame, RenderError> {
        let forge = self.rebuild_scene(project, scene, project_dir, frame, opts, shot)?;

        // A new world builds mesh/material pipelines asynchronously. Issuing the
        // readback in that same update captured the transparent clear before the
        // first ready frame — especially for an imported static OBJ.
        // A few warm-up updates do not change the document's frame time.
        for _ in 0..12 {
            self.app.update();
        }
        let objects_in_frame = visible_object_count(self.app.world(), &forge);

        let (tx, rx) = mpsc::channel::<Vec<u8>>();
        let mut entity = self.app.world_mut().spawn((SceneEntity, Readback::texture(self.target.clone())));
        let readback_entity = entity.id();
        entity.observe(move |ev: On<ReadbackComplete>| {
            let _ = tx.send(ev.data.clone());
        });

        // Readback is active every frame until the component is removed. We issue
        // EXACTLY one GPU copy, then keep the entity with its observer until the
        // event arrives: otherwise the wait loop queued up to 60 map_async calls
        // and emitted a WARN "sending into a closed channel" on drop.
        self.app.update();
        self.app.world_mut().entity_mut(readback_entity).remove::<Readback>();
        let mut data = None;
        for _ in 0..60 {
            if let Ok(d) = rx.try_recv() {
                data = Some(d);
                break;
            }
            self.app.update();
        }
        let _ = self.app.world_mut().despawn(readback_entity);
        let data = data.ok_or(RenderError::ReadbackTimeout)?;

        let tight = (self.width * self.height * 4) as usize;
        let mut rgba = if data.len() == tight {
            data
        } else {
            // GPU row alignment (256 bytes): strip the padding.
            let stride = data.len() / self.height as usize;
            let row = self.width as usize * 4;
            let mut out = Vec::with_capacity(tight);
            for y in 0..self.height as usize {
                out.extend_from_slice(&data[y * stride..y * stride + row]);
            }
            out
        };
        // 2D layers over the 3D frame (in output coordinates).
        if scene.layers_2d.iter().any(|l| !l.exposures.is_empty()) {
            let mut img = image::RgbaImage::from_raw(self.width, self.height, rgba)
                .expect("buffer size checked");
            for layer in &scene.layers_2d {
                if let Some(canvas) = crate::draw_raster::rasterize_for_frame(layer, frame) {
                    crate::draw_raster::composite_over(&mut img, &canvas);
                }
            }
            rgba = img.into_raw();
        }
        Ok(RenderedFrame { frame, width: self.width, height: self.height, rgba, objects_in_frame })
    }

    /// Rebuilds the Bevy world from the document at the given frame.
    fn rebuild_scene(
        &mut self,
        project: &phantom_core::Project,
        scene: &Scene,
        project_dir: &Path,
        frame: Frame,
        opts: &RenderOptions,
        shot: Option<phantom_core::ShotId>,
    ) -> Result<ForgeResult, RenderError> {
        let world = self.app.world_mut();
        let forge = forge_scene(world, project, scene, project_dir, &self.target)?;
        apply_scene_frame(world, scene, &forge, frame, opts.on_twos, None, false, shot);
        Ok(forge)
    }
}

/// Entities of a built scene: the anchor for per-frame updates without
/// rebuilding the structure.
#[derive(Clone, Default)]
pub struct ForgeResult {
    pub camera: Option<Entity>,
    pub phantoms: std::collections::HashMap<phantom_core::PhantomId, PhantomEntities>,
    /// 2D-layer quads (main, onion prev, onion next), in layer order.
    pub layers_2d: Vec<Layer2DEntities>,
    /// Background: screen texture and zone lights.
    pub background: Option<BackgroundEntities>,
    /// Root entities of prop objects, keyed by their id.
    pub objects: std::collections::HashMap<phantom_core::ObjectId, Entity>,
    /// Entities of document light sources, keyed by their id (B1).
    pub lights: std::collections::HashMap<phantom_core::id::LightId, Entity>,
    /// Entities of effects, keyed by their id (B3).
    pub vfx: std::collections::HashMap<phantom_core::id::VfxId, VfxEntities>,
}

/// Counts props whose mesh part is visible in the current view. The root itself
/// has no Mesh3d, so child entities are walked and each object counts once.
fn visible_object_count(world: &World, forge: &ForgeResult) -> usize {
    forge
        .objects
        .values()
        .filter(|root| {
            let mut pending = vec![**root];
            while let Some(entity) = pending.pop() {
                if world.get::<Mesh3d>(entity).is_some()
                    && world.get::<ViewVisibility>(entity).is_some_and(|visibility| visibility.get())
                {
                    return true;
                }
                if let Some(children) = world.get::<Children>(entity) {
                    pending.extend(children.iter());
                }
            }
            false
        })
        .count()
}

/// Entities of one effect (B3).
#[derive(Clone, Default)]
pub struct VfxEntities {
    /// Particle quads (order = particle index); empty for water.
    pub quads: Vec<Entity>,
    /// Water plane and its material (UV scroll); None for particles.
    pub water: Option<(Entity, Handle<StandardMaterial>)>,
}

#[derive(Clone)]
pub struct BackgroundEntities {
    pub texture: Handle<Image>,
    pub zone_lights: Vec<Entity>,
    pub source_asset: phantom_core::AssetId,
    pub is_video: bool,
}

/// Video-background decoders per asset (shared by viewport and export).
#[derive(Resource, Default)]
pub struct VideoCache {
    decoders: std::collections::HashMap<phantom_core::AssetId, crate::video::VideoDecoder>,
}

#[derive(Clone)]
pub struct Layer2DEntities {
    pub main: Entity,
    pub prev: Entity,
    pub next: Entity,
}

/// Cache of exposure textures/materials: the raster is recomputed only
/// for a new exposure or when its strokes change.
#[derive(Resource, Default)]
pub struct DrawCache {
    /// (layer, exposure, fully opaque?) → (stroke version, material).
    exposures: std::collections::HashMap<(phantom_core::LayerId, Frame, bool), (u64, Handle<StandardMaterial>)>,
    quad_mesh: Option<Handle<Mesh>>,
}

#[derive(Clone)]
pub struct PhantomEntities {
    pub root: Entity,
    pub joints: Vec<Entity>,
    /// Bone rest locals (translation, rotation, scale) — the rig's pose.
    pub rest: Vec<(glam::Vec3, glam::Quat, glam::Vec3)>,
    /// Bone → clip-track maps keyed by (asset, stack, retargeting profile):
    /// the legacy clip (stack None) and each `clip_track` usage.
    pub clip_maps: std::collections::HashMap<ClipKey, Vec<Option<usize>>>,
    /// Bone names (in `joints` order) — for attachments (B2).
    pub bone_names: Vec<String>,
    /// Bone parents (topological order).
    pub bone_parents: Vec<Option<usize>>,
}

/// Clip key: (asset, stack, retargeting profile); stack `None` — the longest
/// one (legacy), profile `None` — direct bone-name matching.
pub type ClipKey = (phantom_core::AssetId, Option<String>, Option<phantom_core::RetargetProfile>);

/// Motion clips per asset.
#[derive(Resource, Default)]
pub struct ClipCache {
    pub clips: std::collections::HashMap<ClipKey, std::sync::Arc<crate::import::ClipData>>,
}

/// Builds the scene structure in the world: lighting, background, camera,
/// Phantoms in rest pose. Old scene entities are removed. The frame pose is
/// applied by [`apply_scene_frame`].
pub fn forge_scene(
    world: &mut World,
    project: &phantom_core::Project,
    scene: &Scene,
    project_dir: &Path,
    target: &Handle<Image>,
) -> Result<ForgeResult, RenderError> {
    // Despawn is recursive: remove only top-level scene entities
    // (otherwise despawning children again spams warnings).
    let all: std::collections::HashSet<Entity> = world
        .query_filtered::<Entity, With<SceneEntity>>()
        .iter(world)
        .collect();
    let roots: Vec<Entity> = world
        .query_filtered::<(Entity, Option<&ChildOf>), With<SceneEntity>>()
        .iter(world)
        .filter(|(_, parent)| parent.map(|p| !all.contains(&p.parent())).unwrap_or(true))
        .map(|(e, _)| e)
        .collect();
    for e in roots {
        let _ = world.despawn(e);
    }
    let mut result = ForgeResult::default();

    // Lighting: the scene's three-point stage light (key/fill/rim) with shadows;
    // zone lights and background ambient are a separate contribution below.
    world.insert_resource(AmbientLight { color: Color::WHITE, brightness: 350.0, ..default() });
    spawn_lighting(world, &scene.lighting);
    let lights = spawn_scene_lights(world, scene);
    if let Some(floor) = &scene.floor {
        spawn_floor(world, floor);
    }

    // Background: arc screen with an image or video.
    if let Some(bg) = &scene.background {
        let source_asset = bg.image.asset();
        let is_video = matches!(bg.image, phantom_core::project::BackgroundSource::Video { .. });
        if let Some(path) = project.assets.get(&source_asset).map(|a| project_dir.join(&a.path)) {
            let img = if is_video {
                let mut cache = world.resource_mut::<VideoCache>();
                let dec = cache
                    .decoders
                    .entry(source_asset)
                    .or_insert_with(|| crate::video::VideoDecoder::open(&path).expect("video decoder"));
                dec.frame_at(0).cloned()
            } else {
                Some(image::open(&path)?.to_rgba8())
            };
            if let Some(img) = img {
                let img_zones = img.clone();
                let (w, h) = img.dimensions();
                let image = Image::new(
                    Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    TextureDimension::D2,
                    img.into_raw(),
                    TextureFormat::Rgba8UnormSrgb,
                    RenderAssetUsages::default(),
                );
                let tex = world.resource_mut::<Assets<Image>>().add(image);
                let (iw, ih) = img_zones.dimensions();
                let mesh_data = mesh_gen::arc_screen_mesh(bg, 48, Some(iw as f32 / ih.max(1) as f32));
                let mesh = world.resource_mut::<Assets<Mesh>>().add(mesh_from_data(&mesh_data));
                let material = world.resource_mut::<Assets<StandardMaterial>>().add(StandardMaterial {
                    base_color_texture: Some(tex.clone()),
                    unlit: true,
                    cull_mode: None,
                    ..default()
                });
                world.spawn((SceneEntity, Mesh3d(mesh), MeshMaterial3d(material)));

                // 3×2 zone lights along the screen plane; initial colors come
                // from the first frame (constant for a static image).
                let zones = crate::video::zone_averages(&img_zones);
                let mut avg = [0.0f32; 3];
                for z in zones {
                    for c in 0..3 {
                        avg[c] += z[c] / 6.0;
                    }
                }
                let zone_intensity = zone_light_intensity(avg);
                let mut zone_lights = Vec::new();
                for row in 0..2 {
                    for col in 0..3 {
                        let z = zones[row * 3 + col];
                        let x = (col as f32 - 1.0) * bg.radius * (bg.arc_radians / 2.0).sin() * 0.7;
                        let y = bg.center.y + (row as f32 - 0.5) * bg.height * 0.5;
                        let e = world
                            .spawn((
                                SceneEntity,
                                PointLight {
                                    color: Color::linear_rgba(z[0], z[1], z[2], 1.0),
                                    intensity: zone_intensity,
                                    range: bg.radius * 2.0,
                                    shadows_enabled: false,
                                    ..default()
                                },
                                Transform::from_translation(glam::Vec3::new(
                                    bg.center.x + x,
                                    y,
                                    bg.center.z + 1.0,
                                )),
                            ))
                            .id();
                        zone_lights.push(e);
                    }
                }
                let (ambient_color, ambient_brightness) = ambient_from_background(avg);
                world.insert_resource(AmbientLight {
                    color: Color::linear_rgba(ambient_color[0], ambient_color[1], ambient_color[2], 1.0),
                    brightness: ambient_brightness,
                    ..default()
                });
                result.background = Some(BackgroundEntities {
                    texture: tex,
                    zone_lights,
                    source_asset,
                    is_video,
                });
            }
        }
    }

    // Camera (apply_scene_frame sets its transform). The FOV formula is shared
    // with the pure diagnostics: phantom_core::vertical_fov.
    let fov = phantom_core::vertical_fov(scene.camera.focal_mm);
    let camera = world
        .spawn((
            SceneEntity,
            Camera3d::default(),
            Camera {
                target: RenderTarget::Image(target.clone().into()),
                // B5: the sky/clear color fills everything not covered
                // (and makes the frame opaque); without it — the previous alpha.
                clear_color: bevy::camera::ClearColorConfig::Custom(match scene.sky {
                    Some(c) => Color::linear_rgba(c[0], c[1], c[2], 1.0),
                    None => Color::NONE,
                }),
                ..default()
            },
            Projection::Perspective(PerspectiveProjection { fov, ..default() }),
            // TonyMcMapface: predictable light rolloff without clipping;
            // one shared forge path — identical in viewport and export.
            // TonyMcMapface needs Bevy's tonemapping_luts feature, which is
            // intentionally not shipped in this lean offline build. Reinhard
            // is LUT-free and keeps the render camera pipeline valid.
            Tonemapping::Reinhard,
            Msaa::Sample4,
            Transform::default(),
        ))
        .id();
    result.camera = Some(camera);
    result.lights = lights;
    // B3: fog and color grading on the shared-path camera.
    if let Some(fog) = &scene.fog {
        world.entity_mut(camera).insert(bevy::pbr::DistanceFog {
            color: Color::linear_rgb(fog.color[0], fog.color[1], fog.color[2]),
            falloff: bevy::pbr::FogFalloff::Exponential { density: fog.density },
            ..default()
        });
    }
    if let Some(g) = &scene.grade {
        let global = bevy::render::view::ColorGradingGlobal {
            exposure: g.exposure,
            temperature: g.temperature,
            tint: g.tint,
            hue: g.hue,
            post_saturation: g.post_saturation,
            ..default()
        };
        let section = bevy::render::view::ColorGradingSection {
            saturation: g.saturation,
            contrast: g.contrast,
            gamma: g.gamma,
            gain: g.gain,
            lift: g.lift,
        };
        world.entity_mut(camera).insert(bevy::render::view::ColorGrading { global, shadows: section, midtones: section, highlights: section });
    }
    // B3: scene effects.
    for vfx in &scene.vfx {
        let entities = spawn_vfx(world, vfx, project, project_dir);
        result.vfx.insert(vfx.id, entities);
    }

    // Ambient as its own scene setting (B1): fixed/off override the derivation
    // from the background; from_background keeps the previous path (set by the
    // background code above and updated per frame for a video background).
    match &scene.ambient {
        phantom_core::AmbientMode::FromBackground => {}
        phantom_core::AmbientMode::Fixed { color, brightness } => {
            world.insert_resource(AmbientLight {
                color: Color::linear_rgb(color[0], color[1], color[2]),
                brightness: *brightness,
                ..default()
            });
        }
        phantom_core::AmbientMode::Off => {
            world.insert_resource(AmbientLight { brightness: 0.0, ..default() });
        }
    }

    // 2D-layer quads: children of the camera, covering the whole frame.
    {
        let (vw, vh) = world
            .resource::<Assets<Image>>()
            .get(target)
            .map(|i| (i.width() as f32, i.height() as f32))
            .unwrap_or((1920.0, 1080.0));
        let quad = quad_mesh(world, fov, vw / vh);
        for _layer in &scene.layers_2d {
            let mut spawn = |z: f32| {
                let mat = world.resource_mut::<Assets<StandardMaterial>>().add(StandardMaterial {
                    unlit: true,
                    alpha_mode: AlphaMode::Blend,
                    cull_mode: None,
                    ..default()
                });
                let e = world
                    .spawn((
                        SceneEntity,
                        Mesh3d(quad.clone()),
                        MeshMaterial3d(mat),
                        Transform::from_translation(glam::Vec3::new(0.0, 0.0, z)),
                        Visibility::Hidden,
                    ))
                    .id();
                world.entity_mut(camera).add_child(e);
                e
            };
            result.layers_2d.push(Layer2DEntities {
                main: spawn(-1.0),
                prev: spawn(-1.01),
                next: spawn(-1.02),
            });
        }
    }

    // Phantoms in rest pose.
    for phantom in scene.phantoms.values() {
        for key in phantom.motion.iter().chain(phantom.takes.iter().flat_map(|t| &t.motion)) {
            key.validate()?;
            phantom.rig.validate_action(key.action)?;
        }
        match &phantom.rig {
            RigRef::Builtin | RigRef::BuiltinActor { .. } => {
                let actor = phantom.rig.builtin_actor().expect("builtin");
                let uploaded = builtin_assets(world, actor);
                let model = mesh_gen::builtin_actor_model(actor);
                let entities = spawn_skinned(world, &model, &uploaded, glam::Vec3::ZERO, None, phantom.outline_px, &phantom.material_patches);
                result.phantoms.insert(phantom.id, entities);
            }
            RigRef::Imported { asset } => {
                let Some(path) = project.assets.get(asset).map(|a| project_dir.join(&a.path)) else {
                    continue;
                };
                let model = load_model(&path, true)?;
                for d in &model.diagnostics {
                    bevy::log::warn!("import {}: {d}", path.display());
                }
                let project_root = project_dir.canonicalize().unwrap_or_else(|_| project_dir.to_path_buf());
                let tex_root = path
                    .parent()
                    .map(|dir| TextureRoot { model_dir: dir.to_path_buf(), project_root });
                let uploaded = upload_model(world, &model, tex_root.as_ref());
                let mut entities = spawn_skinned(world, &model, &uploaded, glam::Vec3::ZERO, None, phantom.outline_px, &phantom.material_patches);
                // Motion clips: matched by bone name. A non-empty track replaces
                // the legacy `clip`; an empty one keeps legacy behavior.
                if let Some(skin) = &model.skin {
                    let bone_names: Vec<String> = skin.bones.iter().map(|b| b.name.clone()).collect();
                    let mut wanted: Vec<ClipKey> = Vec::new();
                    if phantom.clip_track.is_empty() {
                        if let Some(clip_asset) = phantom.clip {
                            wanted.push((clip_asset, None, None));
                        }
                    } else {
                        for u in &phantom.clip_track {
                            wanted.push((u.asset, u.stack.clone(), u.retarget));
                        }
                    }
                    for (clip_asset, stack, retarget) in wanted {
                        let Some(clip_path) = project.assets.get(&clip_asset).map(|a| project_dir.join(&a.path)) else {
                            continue;
                        };
                        let key = (clip_asset, stack.clone(), retarget);
                        let clip = {
                            let mut cache = world.resource_mut::<ClipCache>();
                            if !cache.clips.contains_key(&key) {
                                // A clip read error propagates as RenderError::Import
                                // (GUI status / failed render job), not a panic.
                                let loaded = match retarget {
                                    Some(profile) => {
                                        let (clip, diagnostics) =
                                            crate::import::load_clip_retarget(&clip_path, profile, &bone_names, stack.as_deref())?;
                                        for d in &diagnostics {
                                            bevy::log::warn!("retargeting {}: {d}", clip_path.display());
                                        }
                                        clip
                                    }
                                    None => crate::import::load_clip(&clip_path, &bone_names, stack.as_deref())?,
                                };
                                cache.clips.insert(key.clone(), std::sync::Arc::new(loaded));
                            }
                            cache.clips.get(&key).expect("just inserted").clone()
                        };
                        let map: Vec<Option<usize>> = skin
                            .bones
                            .iter()
                            .map(|b| clip.bones.iter().position(|n| n == &b.name))
                            .collect();
                        entities.clip_maps.insert(key, map);
                    }
                }
                result.phantoms.insert(phantom.id, entities);
            }
        }
    }

    // Props: static (non-skinned) objects; a skinned model renders in bind
    // pose without a clip.
    for object in scene.objects.values() {
        // Catalog entry (K1): the mesh is built procedurally, no asset file needed.
        if let Some(prop) = &object.builtin {
            let Some(model) = crate::props::prop_model(prop) else {
                bevy::log::warn!("object {}: unknown catalog entry {prop}", object.name);
                continue;
            };
            let uploaded = upload_model(world, &model, None);
            let root = spawn_static(world, &uploaded, &object.material_patches);
            result.objects.insert(object.id, root);
            continue;
        }
        let Some(path) = project.assets.get(&object.asset).map(|a| project_dir.join(&a.path)) else {
            continue;
        };
        let model = load_model(&path, false)?;
        for d in &model.diagnostics {
            bevy::log::warn!("import {}: {d}", path.display());
        }
        let project_root = project_dir.canonicalize().unwrap_or_else(|_| project_dir.to_path_buf());
        let tex_root = path
            .parent()
            .map(|dir| TextureRoot { model_dir: dir.to_path_buf(), project_root });
        let uploaded = upload_model(world, &model, tex_root.as_ref());
        let root = spawn_static(world, &uploaded, &object.material_patches);
        result.objects.insert(object.id, root);
    }
    Ok(result)
}

/// Spawns a static object: a root with a transform and child parts with
/// materials (skin attributes on meshes without SkinnedMesh are ignored — bind pose).
fn spawn_static(world: &mut World, uploaded: &UploadedModel, patches: &[phantom_core::MaterialPatch]) -> Entity {
    let root = world.spawn((SceneEntity, Transform::default(), GlobalTransform::default())).id();
    for (i, (mesh, material)) in uploaded.parts.iter().enumerate() {
        // B4: material patches — the asset is cloned only when the target matches.
        let material = uploaded_material(world, material, patches, uploaded.part_names.get(i).and_then(|n| n.as_deref()));
        let part = world
            .spawn((SceneEntity, Mesh3d(mesh.clone()), MeshMaterial3d(material)))
            .id();
        world.entity_mut(root).add_child(part);
    }
    root
}

/// Three-point stage lighting: directional key/fill/rim with shadows.
/// Directions are fixed by the setup (key — the previous slice direction);
/// color, intensity (lux) and per-source enable are configurable.
/// Background ambient (corrective A1): color is the half-desaturated zone
/// average (50 % toward its own brightness, so the background does not tint the
/// actors); brightness is derived from the background's mean linear luma rather
/// than a constant: dark night (~0.02) → ~190 lux, daytime background (~0.4) →
/// ~950 lux, bright (~0.8) → ~1750 lux.
pub fn ambient_from_background(avg: [f32; 3]) -> ([f32; 3], f32) {
    let mean = 0.2126 * avg[0] + 0.7152 * avg[1] + 0.0722 * avg[2];
    let color = [
        0.5 * avg[0] + 0.5 * mean,
        0.5 * avg[1] + 0.5 * mean,
        0.5 * avg[2] + 0.5 * mean,
    ];
    (color, 150.0 + 2000.0 * mean)
}

/// Zone PointLight intensity for the background (lumens). A1: the previous
/// 150 lm gave ~1–9 lux on an actor — hundreds of times weaker than the key;
/// the formula is tied to background brightness and yields hundreds of lux of
/// fill: 250_000·(0.25+mean) lm ≈ (0.25+mean)·1200 lux at ~4 m distance.
pub fn zone_light_intensity(avg: [f32; 3]) -> f32 {
    let mean = 0.2126 * avg[0] + 0.7152 * avg[1] + 0.0722 * avg[2];
    250_000.0 * (0.25 + mean)
}

fn spawn_lighting(world: &mut World, lighting: &phantom_core::SceneLighting) {
    // (source, yaw, pitch) — YXZ, like the camera.
    let rig = [
        (&lighting.key, 0.6f32, -0.9f32),   // key: front-right, from above
        (&lighting.fill, -1.2f32, -0.5f32), // fill: softer, from the left
        (&lighting.rim, 3.1f32, -0.7f32),   // rim: behind, from above
    ];
    for (light, yaw, pitch) in rig {
        if !light.enabled {
            continue;
        }
        world.spawn((
            SceneEntity,
            DirectionalLight {
                color: Color::linear_rgb(light.color[0], light.color[1], light.color[2]),
                illuminance: light.intensity,
                // A2: shadows are opt-in per source; a skinned mesh plus a shadow
                // pass currently fails in wgpu pipeline validation.
                shadows_enabled: light.shadows,
                ..default()
            },
            Transform::from_rotation(Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0)),
        ));
    }
}

/// Document light sources (B1): point/spot/directional with keys;
/// apply_scene_frame applies per-frame state via `SceneLightSource::at`.
fn spawn_scene_lights(world: &mut World, scene: &Scene) -> std::collections::HashMap<phantom_core::id::LightId, Entity> {
    let mut out = std::collections::HashMap::new();
    for light in &scene.lights {
        let (pos, yaw, pitch, _, _) = light.at(scene.range.start);
        let transform = Transform {
            translation: pos,
            rotation: Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0),
            ..default()
        };
        let entity = match &light.kind {
            phantom_core::SceneLightKind::Point => world
                .spawn((
                    SceneEntity,
                    PointLight {
                        color: Color::linear_rgb(light.color[0], light.color[1], light.color[2]),
                        intensity: light.intensity,
                        range: 100.0,
                        shadows_enabled: light.shadows,
                        ..default()
                    },
                    transform,
                ))
                .id(),
            phantom_core::SceneLightKind::Spot { cone } => world
                .spawn((
                    SceneEntity,
                    SpotLight {
                        color: Color::linear_rgb(light.color[0], light.color[1], light.color[2]),
                        intensity: light.intensity,
                        range: 100.0,
                        inner_angle: cone * 0.8,
                        outer_angle: *cone,
                        shadows_enabled: light.shadows,
                        ..default()
                    },
                    transform,
                ))
                .id(),
            phantom_core::SceneLightKind::Directional => world
                .spawn((
                    SceneEntity,
                    DirectionalLight {
                        color: Color::linear_rgb(light.color[0], light.color[1], light.color[2]),
                        illuminance: light.intensity,
                        shadows_enabled: light.shadows,
                        ..default()
                    },
                    transform,
                ))
                .id(),
        };
        out.insert(light.id, entity);
    }
    out
}

/// Max particles per effect (memory guard).
const MAX_PARTICLES: usize = 1024;

/// Particle position and fade at `time` (seconds since scene start).
/// Deterministic: a hash of the index, no RNG state — a frame renders the same
/// regardless of frame order.
pub fn particle_state(vfx: &phantom_core::SceneVfx, index: u32, time: f32) -> (glam::Vec3, f32) {
    // splitmix32-like hash → [0,1).
    let h = |salt: u32| {
        let mut z = index.wrapping_mul(0x9E37_79B9).wrapping_add(salt.wrapping_mul(0x85EB_CA6B)).wrapping_add(0x27D4_EB2F);
        z = (z ^ (z >> 15)).wrapping_mul(0x2C1B_3C6D);
        z = (z ^ (z >> 12)).wrapping_mul(0x297A_2D39);
        (z ^ (z >> 15)) as f32 / u32::MAX as f32
    };
    let spawn_t = index as f32 / vfx.emit_rate.max(1e-6);
    let age = (time - spawn_t).rem_euclid(vfx.lifetime);
    let t = age / vfx.lifetime;
    let dir_y = if vfx.gravity <= 0.0 { -1.0 } else { 1.0 };
    let pos = vfx.position
        + glam::Vec3::new(
            (h(1) - 0.5) * vfx.extent.x + (h(4) - 0.5) * vfx.speed * 0.4 * age,
            (h(2) - 0.5) * vfx.extent.y + vfx.speed * dir_y * age + 0.5 * vfx.gravity * age * age,
            (h(3) - 0.5) * vfx.extent.z + (h(5) - 0.5) * vfx.speed * 0.4 * age,
        );
    // Soft life edges: sin(πt).
    (pos, (std::f32::consts::PI * t).sin())
}

/// Number of live particle quads for a configuration (capped at MAX_PARTICLES).
pub fn particle_count(vfx: &phantom_core::SceneVfx) -> usize {
    ((vfx.emit_rate * vfx.lifetime).ceil() as usize).clamp(1, MAX_PARTICLES)
}

/// Spawns an effect (B3): particles are a pool of unlit quads; water is a plane
/// with a normal map and UV scroll.
fn spawn_vfx(world: &mut World, vfx: &phantom_core::SceneVfx, project: &phantom_core::Project, project_dir: &Path) -> VfxEntities {
    match vfx.kind {
        phantom_core::VfxKind::Particles => {
            let n = particle_count(vfx);
            let mesh = world.resource_mut::<Assets<Mesh>>().add(Plane3d::default().mesh().size(1.0, 1.0));
            let material = world.resource_mut::<Assets<StandardMaterial>>().add(StandardMaterial {
                base_color: Color::linear_rgba(vfx.color[0], vfx.color[1], vfx.color[2], 1.0),
                unlit: true,
                alpha_mode: AlphaMode::Blend,
                cull_mode: None,
                ..default()
            });
            let quads = (0..n)
                .map(|_| {
                    world
                        .spawn((
                            SceneEntity,
                            Mesh3d(mesh.clone()),
                            MeshMaterial3d(material.clone()),
                            Transform::default(),
                            Visibility::Hidden,
                        ))
                        .id()
                })
                .collect();
            VfxEntities { quads, water: None }
        }
        phantom_core::VfxKind::Water => {
            let mesh = world
                .resource_mut::<Assets<Mesh>>()
                .add(Plane3d::default().mesh().size(vfx.extent.x.max(0.1), vfx.extent.z.max(0.1)));
            let normal_map_texture = vfx.normal_map.and_then(|id| {
                let path = project_dir.join(&project.assets.get(&id)?.path);
                let img = image::open(&path).ok()?.to_rgba8();
                let (w, h) = img.dimensions();
                let image = Image::new(
                    Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    TextureDimension::D2,
                    img.into_raw(),
                    TextureFormat::Rgba8Unorm,
                    RenderAssetUsages::default(),
                );
                Some(world.resource_mut::<Assets<Image>>().add(image))
            });
            let material = world.resource_mut::<Assets<StandardMaterial>>().add(StandardMaterial {
                base_color: Color::linear_rgba(vfx.color[0], vfx.color[1], vfx.color[2], 1.0),
                perceptual_roughness: 0.08,
                metallic: 0.0,
                normal_map_texture,
                ..default()
            });
            let plane = world
                .spawn((SceneEntity, Mesh3d(mesh), MeshMaterial3d(material.clone()), Transform::from_translation(vfx.position)))
                .id();
            VfxEntities { quads: vec![], water: Some((plane, material)) }
        }
    }
}

/// Matte floor at y=0: an opaque plane that receives shadows.
fn spawn_floor(world: &mut World, floor: &phantom_core::SceneFloor) {
    let mesh = world
        .resource_mut::<Assets<Mesh>>()
        .add(Plane3d::default().mesh().size(200.0, 200.0));
    let material = world.resource_mut::<Assets<StandardMaterial>>().add(StandardMaterial {
        base_color: Color::linear_rgb(floor.color[0], floor.color[1], floor.color[2]),
        perceptual_roughness: 0.95,
        metallic: 0.0,
        ..default()
    });
    world.spawn((SceneEntity, Mesh3d(mesh), MeshMaterial3d(material)));
}

/// Applies frame state to the built scene: camera and Phantom poses.
/// `control` is live control: (phantom, position, yaw in radians, pose) replaces
/// evaluate for the selected Phantom.
pub fn apply_scene_frame(
    world: &mut World,
    scene: &Scene,
    forge: &ForgeResult,
    frame: Frame,
    on_twos: bool,
    control: Option<(phantom_core::PhantomId, glam::Vec3, f32, phantom_core::rig::Pose)>,
    onion_skin: bool,
    shot: Option<phantom_core::ShotId>,
) {
    apply_background_frame(world, forge, frame, &scene.ambient);
    apply_draw_layers(world, scene, forge, frame, onion_skin);
    // Document lights: pose/intensity/enable at the frame (B1).
    for light in &scene.lights {
        let Some(entity) = forge.lights.get(&light.id) else { continue };
        let (pos, yaw, pitch, intensity, enabled) = light.at(frame);
        if let Some(mut tr) = world.get_mut::<Transform>(*entity) {
            tr.translation = pos;
            tr.rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
        }
        if let Some(mut vis) = world.get_mut::<Visibility>(*entity) {
            *vis = if enabled { Visibility::Inherited } else { Visibility::Hidden };
        }
        match &light.kind {
            phantom_core::SceneLightKind::Point => {
                if let Some(mut l) = world.get_mut::<PointLight>(*entity) {
                    l.intensity = intensity;
                }
            }
            phantom_core::SceneLightKind::Spot { .. } => {
                if let Some(mut l) = world.get_mut::<SpotLight>(*entity) {
                    l.intensity = intensity;
                }
            }
            phantom_core::SceneLightKind::Directional => {
                if let Some(mut l) = world.get_mut::<DirectionalLight>(*entity) {
                    l.illuminance = intensity;
                }
            }
        }
    }
    for object in scene.objects.values() {
        let Some(root) = forge.objects.get(&object.id) else { continue };
        let t = evaluate_object(object, &scene.range, frame, on_twos);
        if let Some(mut tr) = world.get_mut::<Transform>(*root) {
            tr.translation = t.position;
            tr.rotation = Quat::from_euler(EulerRot::YXZ, t.yaw, t.pitch, t.roll);
            tr.scale = t.scale;
        }
    }
    if let Some(cam) = forge.camera {
        let cam_ref = scene.camera_for_shot(shot);
        let pose = sample_camera(cam_ref, frame);
        if let Some(mut tr) = world.get_mut::<Transform>(cam) {
            tr.translation = pose.translation;
            // C1.1: horizon roll — the third YXZ component.
            tr.rotation = Quat::from_euler(EulerRot::YXZ, pose.yaw, pose.pitch, pose.roll);
        }
        // Focus comes from the selected camera (a shot may differ from the legacy
        // camera), including focal-length keys (C1.2): the projection is updated per
        // frame, no forge rebuild is needed.
        let fov = phantom_core::vertical_fov(pose.focal_mm);
        if let Some(mut proj) = world.get_mut::<Projection>(cam) {
            *proj = Projection::Perspective(PerspectiveProjection { fov, ..default() });
        }
    }
    // Per-Phantom bone poses for this frame — the basis of attachments (B2).
    let mut bone_frames: std::collections::HashMap<phantom_core::PhantomId, (glam::Vec3, f32, Vec<(glam::Vec3, glam::Quat)>)> =
        Default::default();
    for phantom in scene.phantoms.values() {
        let Some(entities) = forge.phantoms.get(&phantom.id) else { continue };
        let (root, yaw, bone_poses): (glam::Vec3, f32, Vec<(glam::Vec3, glam::Quat, glam::Vec3)>) = match &control {
            Some((id, pos, yaw, pose))
                if *id == phantom.id
                    && phantom.rig.builtin_actor() == Some(phantom_core::project::BuiltinActor::Humanoid)
                    && pose.bones.len() == entities.joints.len() =>
            {
                (*pos, *yaw, pose.bones.iter().map(|b| (b.translation, b.rotation, glam::Vec3::ONE)).collect())
            }
            _ => {
                let state = evaluate(phantom, &scene.range, frame, on_twos);
                match &phantom.rig {
                    RigRef::Builtin | RigRef::BuiltinActor { .. } => {
                        (state.root, state.yaw, state.pose.bones.iter().map(|b| (b.translation, b.rotation, glam::Vec3::ONE)).collect())
                    }
                    RigRef::Imported { .. } => {
                        let cache = world.resource::<ClipCache>();
                        if phantom.clip_track.is_empty() {
                            // Legacy: one clip (the longest stack), looped over the whole scene.
                            let clip_pose = phantom.clip.and_then(|asset| {
                                let map = entities.clip_maps.get(&(asset, None, None))?;
                                let clip = cache.clips.get(&(asset, None, None))?;
                                let f = (frame.max(0) as u32 % clip.frames.max(1)) as usize;
                                Some(sample_clip_pose(&entities.rest, map, clip, f as f32))
                            });
                            (state.root, state.yaw, clip_pose.unwrap_or_else(|| entities.rest.clone()))
                        } else {
                            // Clip track: the usage active at this frame; outside it —
                            // bind pose (manual controller keys work as before).
                            let active = phantom_core::active_clip_use(&phantom.clip_track, frame);
                            let mut root = state.root;
                            let pose = active.and_then(|u| {
                                let key = (u.asset, u.stack.clone(), u.retarget);
                                let map = entities.clip_maps.get(&key)?;
                                let clip = cache.clips.get(&key)?;
                                // Clip frame: (frame - start) * speed, clamped to the clip's
                                // last frame (no looping — a single playthrough).
                                let f = phantom_core::clip_frame_at(u, frame).clamp(0.0, (clip.frames.max(1) - 1) as f32);
                                let mut pose = sample_clip_pose(&entities.rest, map, clip, f);
                                // Blending: windows inside the usage range; the base pose is
                                // rest at the track edges, and the neighbouring clip's pose at
                                // butt-adjacent joints.
                                let (w_in, w_out) = phantom_core::blend_weights(u, frame);
                                if w_in < 1.0 {
                                    let base = phantom_core::clips::contiguous_prev(&phantom.clip_track, u)
                                        .and_then(|p| {
                                            let k = (p.asset, p.stack.clone(), p.retarget);
                                            let c = cache.clips.get(&k)?;
                                            let m = entities.clip_maps.get(&k)?;
                                            let pf = phantom_core::clip_frame_at(p, p.range.end - 1).clamp(0.0, (c.frames.max(1) - 1) as f32);
                                            Some(sample_clip_pose(&entities.rest, m, c, pf))
                                        })
                                        .unwrap_or_else(|| entities.rest.clone());
                                    pose = blend_clip_pose(&base, &pose, w_in);
                                }
                                if w_out < 1.0 {
                                    let base = phantom_core::clips::contiguous_next(&phantom.clip_track, u)
                                        .and_then(|n| {
                                            let k = (n.asset, n.stack.clone(), n.retarget);
                                            let c = cache.clips.get(&k)?;
                                            let m = entities.clip_maps.get(&k)?;
                                            Some(sample_clip_pose(&entities.rest, m, c, 0.0))
                                        })
                                        .unwrap_or_else(|| entities.rest.clone());
                                    pose = blend_clip_pose(&pose, &base, 1.0 - w_out);
                                }
                                // Root motion: Clip — the clip's root-bone delta from the frame
                                // the usage starts on; the controller is frozen at that start
                                // frame (the two trajectories are not summed). The root bone's
                                // translation in the pose is frozen at clip frame 0, otherwise
                                // the displacement would count twice: both in the bone's local
                                // pose and in the root entity's offset. Controller — the clip
                                // pose is applied as sampled (the root track is not stripped)
                                // and the controller owns the offset.
                                if u.root_motion == phantom_core::RootMotion::Clip {
                                    freeze_clip_root_translation(&mut pose, map, clip);
                                    if let Some(delta) = root_motion_delta(map, clip, f) {
                                        let base = evaluate(phantom, &scene.range, u.range.start, on_twos).root;
                                        root = base + delta;
                                    }
                                }
                                Some(pose)
                            });
                            (root, state.yaw, pose.unwrap_or_else(|| entities.rest.clone()))
                        }
                    }
                }
            }
        };
        if let Some(mut tr) = world.get_mut::<Transform>(entities.root) {
            tr.translation = root;
            tr.rotation = Quat::from_rotation_y(yaw);
            // B2: the actor's scale is applied to the root.
            tr.scale = glam::Vec3::splat(phantom.scale);
        }
        bone_frames.insert(phantom.id, (root, yaw, bone_poses.iter().map(|(t, r, _)| (*t, *r)).collect()));
        for (joint, (t, r, sc)) in entities.joints.iter().zip(bone_poses.iter()) {
            if let Some(mut tr) = world.get_mut::<Transform>(*joint) {
                tr.translation = *t;
                tr.rotation = *r;
                tr.scale = *sc;
            }
        }
    }
    // Attachments (B2): an object/light follows a Phantom bone at this frame.
    apply_attachments(world, scene, forge, &bone_frames);
    // Effects (B3): particles/water at this frame.
    apply_vfx(world, scene, forge, frame, &bone_frames);
}

/// Effects at this frame: particle positions/fade (Y-billboard toward the
/// camera), water UV scroll, enabled/density keys, emitter attached to a bone.
fn apply_vfx(
    world: &mut World,
    scene: &Scene,
    forge: &ForgeResult,
    frame: Frame,
    bone_frames: &std::collections::HashMap<phantom_core::PhantomId, (glam::Vec3, f32, Vec<(glam::Vec3, glam::Quat)>)>,
) {
    let time = (frame - scene.range.start) as f32 / phantom_core::FPS as f32;
    let cam_pos = forge.camera.and_then(|c| world.get::<GlobalTransform>(c)).map(|t| t.translation());
    for vfx in &scene.vfx {
        let Some(entities) = forge.vfx.get(&vfx.id) else { continue };
        let enabled = vfx.enabled_at(frame);
        let density = vfx.density_at(frame);
        match vfx.kind {
            phantom_core::VfxKind::Particles => {
                let center = match &vfx.attach {
                    Some(a) => attached_bone_world(scene, forge, bone_frames, a).map(|(p, _, _)| p).unwrap_or(vfx.position),
                    None => vfx.position,
                };
                let alive = ((entities.quads.len() as f32) * density.clamp(0.0, 1.0)) as usize;
                for (i, quad) in entities.quads.iter().enumerate() {
                    let Some(mut vis) = world.get_mut::<Visibility>(*quad) else { continue };
                    if !enabled || i >= alive {
                        *vis = Visibility::Hidden;
                        continue;
                    }
                    *vis = Visibility::Inherited;
                    let mut local = vfx.clone();
                    local.position = center;
                    let (pos, fade) = particle_state(&local, i as u32, time);
                    if let Some(mut tr) = world.get_mut::<Transform>(*quad) {
                        tr.translation = pos;
                        tr.scale = glam::Vec3::splat(vfx.size * fade.max(0.05));
                        // Y-billboard: the quad is turned toward the camera around Y.
                        if let Some(cp) = cam_pos {
                            let d = cp - pos;
                            tr.rotation = Quat::from_rotation_y(d.x.atan2(d.z)) * Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
                        }
                    }
                }
            }
            phantom_core::VfxKind::Water => {
                let Some((plane, material)) = &entities.water else { continue };
                if let Some(mut vis) = world.get_mut::<Visibility>(*plane) {
                    *vis = if enabled { Visibility::Inherited } else { Visibility::Hidden };
                }
                if let Some(mut tr) = world.get_mut::<Transform>(*plane) {
                    tr.translation = match &vfx.attach {
                        Some(a) => attached_bone_world(scene, forge, bone_frames, a).map(|(p, _, _)| p).unwrap_or(vfx.position),
                        None => vfx.position,
                    };
                }
                // Normal-map scroll: uv_transform is a translation over time.
                if let Some(m) = world.resource_mut::<Assets<StandardMaterial>>().get_mut(material) {
                    m.uv_transform = bevy::math::Affine2::from_translation(glam::Vec2::new(vfx.scroll_uv[0] * time, vfx.scroll_uv[1] * time));
                }
            }
        }
    }
}

/// Bone transform in the world: Phantom root (position + yaw + scale) ⋅ the
/// accumulated bone hierarchy. Names/parents: builtin — `rig::actor_bones`,
/// import — the model's SkinData. None if the Phantom or the bone is not found.
fn attached_bone_world(
    scene: &Scene,
    forge: &ForgeResult,
    bone_frames: &std::collections::HashMap<phantom_core::PhantomId, (glam::Vec3, f32, Vec<(glam::Vec3, glam::Quat)>)>,
    attach: &phantom_core::project::Attachment,
) -> Option<(glam::Vec3, glam::Quat, f32)> {
    let phantom = scene.phantoms.get(&attach.phantom)?;
    let entities = forge.phantoms.get(&attach.phantom)?;
    let (root, yaw, locals) = bone_frames.get(&attach.phantom)?;
    let idx = entities.bone_names.iter().position(|n| n.eq_ignore_ascii_case(&attach.bone))?;
    let world_bones = phantom_core::rig::accumulate_bones(&entities.bone_parents, locals);
    let (bt, br) = world_bones.get(idx).copied()?;
    let root_q = Quat::from_rotation_y(*yaw);
    let s = phantom.scale;
    let pos = *root + root_q * (bt * s);
    Some((pos, root_q * br, s))
}

/// Attachments of objects and lights to bones (corrective B2).
fn apply_attachments(
    world: &mut World,
    scene: &Scene,
    forge: &ForgeResult,
    bone_frames: &std::collections::HashMap<phantom_core::PhantomId, (glam::Vec3, f32, Vec<(glam::Vec3, glam::Quat)>)>,
) {
    for object in scene.objects.values() {
        let Some(attach) = &object.attach else { continue };
        let Some(root) = forge.objects.get(&object.id) else { continue };
        let Some((pos, rot, scale)) = attached_bone_world(scene, forge, bone_frames, attach) else { continue };
        let offset_q = Quat::from_euler(EulerRot::YXZ, attach.offset_yaw, attach.offset_pitch, attach.offset_roll);
        if let Some(mut tr) = world.get_mut::<Transform>(*root) {
            tr.translation = pos + rot * (attach.offset_pos * scale);
            tr.rotation = rot * offset_q;
            // The actor's scale is inherited; the key's own scale goes on top.
            let own = phantom_core::evaluate_object(object, &scene.range, 0, false).scale;
            tr.scale = own * scale;
        }
    }
    for light in &scene.lights {
        let Some(attach) = &light.attach else { continue };
        let Some(entity) = forge.lights.get(&light.id) else { continue };
        let Some((pos, rot, _)) = attached_bone_world(scene, forge, bone_frames, attach) else { continue };
        let offset_q = Quat::from_euler(EulerRot::YXZ, attach.offset_yaw, attach.offset_pitch, attach.offset_roll);
        if let Some(mut tr) = world.get_mut::<Transform>(*entity) {
            tr.translation = pos + rot * attach.offset_pos;
            tr.rotation = rot * offset_q;
        }
    }
}

/// Bone pose from a clip at fractional frame `f` (linear interpolation between
/// neighboring samples: lerp translations, nlerp rotations). Bones without a
/// track stay in rest pose. `f` must already be clamped to [0, frames - 1].
pub fn sample_clip_pose(
    rest: &[(glam::Vec3, glam::Quat, glam::Vec3)],
    map: &[Option<usize>],
    clip: &crate::import::ClipData,
    f: f32,
) -> Vec<(glam::Vec3, glam::Quat, glam::Vec3)> {
    let f = f.clamp(0.0, (clip.frames.max(1) - 1) as f32);
    let i0 = f.floor() as usize;
    let i1 = (i0 + 1).min((clip.frames.max(1) - 1) as usize);
    let t = f - i0 as f32;
    rest.iter()
        .enumerate()
        .map(|(bi, rest)| {
            map.get(bi)
                .and_then(|tr| {
                    tr.map(|ti| {
                        let a = clip.samples[ti][i0];
                        let b = clip.samples[ti][i1];
                        (a.0.lerp(b.0, t), a.1.lerp(b.1, t).normalize(), rest.2)
                    })
                })
                .unwrap_or(*rest)
        })
        .collect()
}

/// Blend of two bone poses: lerp translations, nlerp rotations; scale comes from `to`.
pub fn blend_clip_pose(
    from: &[(glam::Vec3, glam::Quat, glam::Vec3)],
    to: &[(glam::Vec3, glam::Quat, glam::Vec3)],
    alpha: f32,
) -> Vec<(glam::Vec3, glam::Quat, glam::Vec3)> {
    from.iter()
        .zip(to.iter())
        .map(|(a, b)| (a.0.lerp(b.0, alpha), a.1.lerp(b.1, alpha).normalize(), b.2))
        .collect()
}

/// Clip root-motion delta: the root bone's displacement (first skin bone) from
/// clip frame 0 to frame `f`. `None` if the root bone has no track.
pub fn root_motion_delta(map: &[Option<usize>], clip: &crate::import::ClipData, f: f32) -> Option<glam::Vec3> {
    let ti = map.first().and_then(|t| *t)?;
    let f = f.clamp(0.0, (clip.frames.max(1) - 1) as f32);
    let i0 = f.floor() as usize;
    let i1 = (i0 + 1).min((clip.frames.max(1) - 1) as usize);
    let t = f - i0 as f32;
    let at = clip.samples[ti][i0].0.lerp(clip.samples[ti][i1].0, t);
    Some(at - clip.samples[ti][0].0)
}

/// Freezes the root bone's translation (first skin bone) in the pose at clip
/// frame 0. In [`phantom_core::RootMotion::Clip`] mode the root displacement
/// comes from [`root_motion_delta`] on the root entity; without the freeze the
/// same displacement would count twice — both in the bone's local pose (the
/// clip's absolute root track) and in the root offset.
pub fn freeze_clip_root_translation(
    pose: &mut [(glam::Vec3, glam::Quat, glam::Vec3)],
    map: &[Option<usize>],
    clip: &crate::import::ClipData,
) {
    let Some(ti) = map.first().and_then(|t| *t) else { return };
    if let Some(root) = pose.first_mut() {
        root.0 = clip.samples[ti][0].0;
    }
}

fn builtin_assets(world: &mut World, actor: phantom_core::project::BuiltinActor) -> UploadedModel {
    let cached = world.resource::<SceneCache>().builtin.get(&actor).cloned();
    if let Some(c) = cached {
        return c;
    }
    let model = mesh_gen::builtin_actor_model(actor);
    let uploaded = upload_model(world, &model, None);
    world.resource_mut::<SceneCache>().builtin.insert(actor, uploaded.clone());
    uploaded
}

/// Spawns a character: bones, skinned meshes of the parts, and the outline.
fn spawn_skinned(
    world: &mut World,
    model: &ModelData,
    uploaded: &UploadedModel,
    root: glam::Vec3,
    pose: Option<&phantom_core::rig::Pose>,
    outline_px: f32,
    patches: &[phantom_core::MaterialPatch],
) -> PhantomEntities {
    {
        let skin = model.skin.as_ref().expect("skinned model");

        let root_entity = world
            .spawn((SceneEntity, Transform::from_translation(root), GlobalTransform::default()))
            .id();

        // Bones: a parent always precedes its child (topological order by depth).
        let depth = |mut i: usize| {
            let mut d = 0;
            while let Some(p) = skin.bones[i].parent {
                d += 1;
                i = p;
            }
            d
        };
        let mut order: Vec<usize> = (0..skin.bones.len()).collect();
        order.sort_by_key(|&i| depth(i));
        let mut entities: Vec<Option<Entity>> = vec![None; skin.bones.len()];
        for &i in &order {
            let bone = &skin.bones[i];
            let (t, r, sc) = match pose {
                Some(p) => (p.bones[i].translation, p.bones[i].rotation, glam::Vec3::from_array(bone.rest_scale)),
                None => (
                    glam::Vec3::from_array(bone.rest_translation),
                    glam::Quat::from_array(bone.rest_rotation),
                    glam::Vec3::from_array(bone.rest_scale),
                ),
            };
            let parent = bone.parent.and_then(|p| entities[p]).unwrap_or(root_entity);
            let e = world
                .spawn((SceneEntity, Transform { translation: t, rotation: r, scale: sc, ..default() }, GlobalTransform::default()))
                .id();
            world.entity_mut(parent).add_child(e);
            entities[i] = Some(e);
        }
        let joints: Vec<Entity> = entities.into_iter().map(|e| e.expect("all bones spawned")).collect();

        let skinned = SkinnedMesh { joints: joints.clone(), inverse_bindposes: uploaded.ibp.clone() };
        // One entity per part (material group); the skeleton is shared.
        for (i, (mesh, material)) in uploaded.parts.iter().enumerate() {
            // B4: actor material patches (wet wolf, glowing lantern).
            let material = uploaded_material(world, material, patches, uploaded.part_names.get(i).and_then(|n| n.as_deref()));
            let body = world
                .spawn((
                    SceneEntity,
                    Mesh3d(mesh.clone()),
                    MeshMaterial3d(material),
                    skinned.clone(),
                ))
                .id();
            world.entity_mut(root_entity).add_child(body);
        }

        // Outline: one combined mesh, inverted unlit pass.
        let outline_mat = world.resource_mut::<Assets<outline::OutlineMaterial>>().add(
            outline::OutlineMaterial { color: LinearRgba::BLACK, thickness_px: outline_px },
        );
        let hull = world
            .spawn((
                SceneEntity,
                Mesh3d(uploaded.combined.clone()),
                bevy::pbr::MeshMaterial3d(outline_mat),
                skinned,
            ))
            .id();
        world.entity_mut(root_entity).add_child(hull);

        let rest = skin
            .bones
            .iter()
            .map(|b| {
                (
                    glam::Vec3::from_array(b.rest_translation),
                    glam::Quat::from_array(b.rest_rotation),
                    glam::Vec3::from_array(b.rest_scale),
                )
            })
            .collect();
        PhantomEntities {
            root: root_entity,
            joints,
            rest,
            clip_maps: Default::default(),
            bone_names: skin.bones.iter().map(|b| b.name.clone()).collect(),
            bone_parents: skin.bones.iter().map(|b| b.parent).collect(),
        }
    }
}

/// Video background and zone lighting at this frame. A static image
/// updates its texture only once (a cheap check).
fn apply_background_frame(world: &mut World, forge: &ForgeResult, frame: Frame, ambient: &phantom_core::AmbientMode) {
    // Ambient from the background per frame — only in from_background mode (B1);
    // fixed/off are set at build time and are not overwritten.
    let Some(bg) = &forge.background else { return };
    if !bg.is_video {
        return; // Static image: texture and zones were set at build time.
    }
    if !matches!(ambient, phantom_core::AmbientMode::FromBackground) {
        return;
    }
    let img = {
        let mut cache = world.resource_mut::<VideoCache>();
        match cache.decoders.get_mut(&bg.source_asset) {
            Some(dec) => dec.frame_at(frame.max(0) as u64).cloned(),
            None => None,
        }
    };
    let Some(img) = img else { return };
    // Replace the screen texture's data.
    let (w, h) = img.dimensions();
    let image = Image::new(
        Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        TextureDimension::D2,
        img.clone().into_raw(),
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    let _ = world.resource_mut::<Assets<Image>>().insert(bg.texture.id(), image);
    // 3×2 zones → light colors and strength; ambient is the zone average.
    let zones = crate::video::zone_averages(&img);
    let mut avg = [0.0f32; 3];
    for (i, e) in bg.zone_lights.iter().enumerate() {
        let z = zones[i];
        for c in 0..3 {
            avg[c] += z[c] / 6.0;
        }
        if let Some(mut light) = world.get_mut::<PointLight>(*e) {
            light.color = Color::linear_rgba(z[0], z[1], z[2], 1.0);
            light.intensity = zone_light_intensity(avg);
        }
    }
    let (ambient_color, ambient_brightness) = ambient_from_background(avg);
    world.insert_resource(AmbientLight {
        color: Color::linear_rgba(ambient_color[0], ambient_color[1], ambient_color[2], 1.0),
        brightness: ambient_brightness,
        ..default()
    });
}

/// Textures and quad visibility of the 2D layers at this frame.
fn apply_draw_layers(
    world: &mut World,
    scene: &Scene,
    forge: &ForgeResult,
    frame: Frame,
    onion_skin: bool,
) {
    for (i, layer) in scene.layers_2d.iter().enumerate() {
        let Some(entities) = forge.layers_2d.get(i) else { break };
        let current = layer.exposure_for_frame(frame).map(|(f, _)| *f);
        set_exposure_quad(world, layer, entities.main, current, 1.0);
        let (prev, next) = if onion_skin { layer.onion(frame) } else { (None, None) };
        set_exposure_quad(world, layer, entities.prev, prev, 0.3);
        set_exposure_quad(world, layer, entities.next, next, 0.3);
    }
}

fn set_exposure_quad(
    world: &mut World,
    layer: &phantom_core::Layer2D,
    entity: Entity,
    exposure: Option<Frame>,
    opacity: f32,
) {
    let Some(exp_frame) = exposure else {
        if let Some(mut v) = world.get_mut::<Visibility>(entity) {
            *v = Visibility::Hidden;
        }
        return;
    };
    let Some(exp) = layer.exposures.get(&exp_frame) else { return };
    let key = (layer.id, exp_frame, opacity >= 1.0);
    let version = exp.strokes.len() as u64;
    let material = {
        let cache = world.resource::<DrawCache>();
        cache.exposures.get(&key).filter(|(v, _)| *v == version).map(|(_, m)| m.clone())
    };
    let material = match material {
        Some(m) => m,
        None => {
            let raster = crate::draw_raster::rasterize(layer.canvas, exp);
            let (w, h) = raster.dimensions();
            let image = Image::new(
                Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                TextureDimension::D2,
                raster.into_raw(),
                TextureFormat::Rgba8UnormSrgb,
                RenderAssetUsages::default(),
            );
            let tex = world.resource_mut::<Assets<Image>>().add(image);
            let mat = world.resource_mut::<Assets<StandardMaterial>>().add(StandardMaterial {
                base_color_texture: Some(tex),
                base_color: Color::WHITE.with_alpha(opacity),
                unlit: true,
                alpha_mode: AlphaMode::Blend,
                cull_mode: None,
                ..default()
            });
            world.resource_mut::<DrawCache>().exposures.insert(key, (version, mat.clone()));
            mat
        }
    };
    if let Some(mut m) = world.get_mut::<MeshMaterial3d<StandardMaterial>>(entity) {
        *m = MeshMaterial3d(material);
    }
    if let Some(mut v) = world.get_mut::<Visibility>(entity) {
        *v = Visibility::Inherited;
    }
}

/// Quad in the camera plane, covering the frame at distance 1.
fn quad_mesh(world: &mut World, fov: f32, aspect: f32) -> Handle<Mesh> {
    if let Some(m) = world.resource::<DrawCache>().quad_mesh.clone() {
        return m;
    }
    let h = (fov / 2.0).tan(); // half height at z = -1
    let w = h * aspect;
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vec![
            [-w, -h, 0.0],
            [w, -h, 0.0],
            [w, h, 0.0],
            [-w, h, 0.0],
        ],
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 0.0, 1.0]; 4]);
    // v=0 is the top of the image.
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_UV_0,
        vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
    );
    mesh.insert_indices(Indices::U32(vec![0, 1, 2, 0, 2, 3]));
    let handle = world.resource_mut::<Assets<Mesh>>().add(mesh);
    world.resource_mut::<DrawCache>().quad_mesh = Some(handle.clone());
    handle
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use phantom_core::*;

    /// Live path for 2D layers: forge + apply + a frame through the same
    /// readback as the offline render. Requires a GPU.
    #[test]
    #[ignore = "requires a GPU"]
    fn live_layer_quads_render() {
        let mut project = Project::new("t");
        let mut history = History::default();
        let scene = history
            .apply(&mut project, phantom_core::Command::AddScene { name: "s".into(), range: FrameRange::new(0, 48).unwrap() })
            .unwrap()
            .scene
            .unwrap();
        history
            .apply(
                &mut project,
                phantom_core::Command::BrushStroke {
                    scene,
                    layer: None,
                    frame: 0,
                    stroke: Stroke {
                        points: vec![[0.0, 20.0], [1920.0, 20.0]],
                        color: [1.0, 0.0, 0.0, 1.0],
                        width: 30.0,
                        erase: false,
                    },
                },
            )
            .unwrap();

        let mut renderer = Renderer::new(320, 180).expect("GPU");
        // Like the GUI: a windowed 2D camera for egui (no window — no target).
        renderer.app.world_mut().spawn(Camera2d);
        let target = renderer.target.clone();
        let scene_doc = project.scenes[&scene].clone();
        let world = renderer.app.world_mut();
        let forge = forge_scene(world, &project, &scene_doc, Path::new("/tmp"), &target).unwrap();
        apply_scene_frame(world, &scene_doc, &forge, 0, false, None, true, None);
        for _ in 0..30 {
            renderer.app.update();
        }

        let (tx, rx) = std::sync::mpsc::channel::<Vec<u8>>();
        renderer
            .app
            .world_mut()
            .spawn(Readback::texture(target))
            .observe(move |ev: On<ReadbackComplete>| {
                let _ = tx.send(ev.data.clone());
            });
        let mut data = None;
        for _ in 0..60 {
            renderer.app.update();
            if let Ok(d) = rx.try_recv() {
                data = Some(d);
                break;
            }
        }
        let data = data.expect("readback");
        let o = ((3 * 320 + 160) * 4) as usize; // y=3: stroke y=20/1080*180≈3
        let (r, a) = (data[o], data[o + 3]);
        assert!(r > 200 && a == 255, "stroke through the quad: r={r} a={a}");
    }
}

/// Camera position at a frame: linear interpolation of keys.
/// The shared implementation is `Camera::sample` in core; it takes a camera
/// (the scene's legacy one or a shot's) — choosing it is up to the caller.
fn sample_camera(camera: &phantom_core::Camera, frame: Frame) -> phantom_core::project::CameraPose {
    camera.sample(frame)
}

/// GPU assets of a model: the combined mesh (outline, legacy), part meshes
/// with materials, and the shared inverse-bindposes.
#[derive(Clone)]
pub struct UploadedModel {
    pub combined: Handle<Mesh>,
    pub parts: Vec<(Handle<Mesh>, Handle<StandardMaterial>)>,
    pub ibp: Handle<SkinnedMeshInverseBindposes>,
    /// Material names of the batches (in `parts` order); None for a legacy single one.
    pub part_names: Vec<Option<String>>,
}

/// Patches a batch's material from assets; with no matching patch — the same handle.
fn uploaded_material(
    world: &mut World,
    base: &Handle<StandardMaterial>,
    patches: &[phantom_core::MaterialPatch],
    part_name: Option<&str>,
) -> Handle<StandardMaterial> {
    let source = match world.resource::<Assets<StandardMaterial>>().get(base) {
        Some(m) => m.clone(),
        None => return base.clone(),
    };
    match apply_material_patches(&source, patches, part_name) {
        Some(m) => world.resource_mut::<Assets<StandardMaterial>>().add(m),
        None => base.clone(),
    }
}

/// Applies material patches to a batch (B4); None — no patches for this batch.
pub fn apply_material_patches(
    base: &StandardMaterial,
    patches: &[phantom_core::MaterialPatch],
    part_name: Option<&str>,
) -> Option<StandardMaterial> {
    let matching: Vec<&phantom_core::MaterialPatch> = patches
        .iter()
        .filter(|p| p.target.as_deref().is_none_or(|t| Some(t) == part_name))
        .collect();
    if matching.is_empty() {
        return None;
    }
    let mut m = base.clone();
    for p in matching {
        if let Some(c) = p.base_color {
            m.base_color = Color::srgba(c[0], c[1], c[2], c[3]);
        }
        if let Some(v) = p.metallic {
            m.metallic = v;
        }
        if let Some(v) = p.roughness {
            m.perceptual_roughness = v;
        }
        if let Some(e) = p.emissive {
            m.emissive = bevy::color::LinearRgba::rgb(e[0], e[1], e[2]);
        }
        if let Some(o) = p.opacity {
            let mut c = m.base_color.to_srgba();
            c.alpha = o;
            m.base_color = Color::Srgba(c);
            if o < 1.0 {
                m.alpha_mode = AlphaMode::Blend;
                m.cull_mode = None;
            }
        }
        if let Some(u) = p.unlit {
            m.unlit = u;
        }
    }
    Some(m)
}

/// Root for resolving a model's texture paths: everything stays inside the project.
struct TextureRoot {
    model_dir: std::path::PathBuf,
    project_root: std::path::PathBuf,
}

impl TextureRoot {
    /// Resolves a texture path from the model file. Rejects paths escaping the
    /// project root; also tries the bare file name in the model's directory (the
    /// import bundle puts embedded textures next to the model).
    fn resolve(&self, rel: &str) -> Option<std::path::PathBuf> {
        let rel_path = Path::new(rel);
        let mut candidates = vec![self.model_dir.join(rel_path)];
        if let Some(name) = rel_path.file_name() {
            candidates.push(self.model_dir.join(name));
        }
        for cand in candidates {
            if let Ok(canon) = cand.canonicalize() {
                if canon.is_file() && canon.starts_with(&self.project_root) {
                    return Some(canon);
                }
            }
        }
        None
    }
}

/// Loads ModelData into GPU assets: part meshes with PBR materials.
fn upload_model(world: &mut World, model: &ModelData, tex: Option<&TextureRoot>) -> UploadedModel {
    let combined = world.resource_mut::<Assets<Mesh>>().add(mesh_from_data_skinned(model));
    let ibp = world.resource_mut::<Assets<SkinnedMeshInverseBindposes>>().add(SkinnedMeshInverseBindposes::from(
        model
            .skin
            .as_ref()
            .map(|s| s.bones.iter().map(|b| Mat4::from_cols_array_2d(&b.inverse_bind)).collect::<Vec<_>>())
            .unwrap_or_default(),
    ));
    let (parts, part_names) = if model.parts.is_empty() {
        // Legacy/material-less model: one default material.
        let mat = world.resource_mut::<Assets<StandardMaterial>>().add(StandardMaterial {
            base_color: Color::srgb(0.85, 0.65, 0.45),
            perceptual_roughness: 0.9,
            ..default()
        });
        (vec![(combined.clone(), mat)], vec![None])
    } else {
        let parts: Vec<_> = model
            .parts
            .iter()
            .map(|part| {
                let rec = &model.materials[part.material.min(model.materials.len().saturating_sub(1))];
                let (material, tangents) = material_from_record(world, rec, tex);
                let mesh = world.resource_mut::<Assets<Mesh>>().add(part_mesh(model, part, tangents));
                (mesh, material)
            })
            .collect();
        let names = model
            .parts
            .iter()
            .map(|part| Some(model.materials[part.material.min(model.materials.len().saturating_sub(1))].name.clone()))
            .collect();
        (parts, names)
    };
    UploadedModel { combined, parts, ibp, part_names }
}

/// Mesh of one part: shared vertex/skin arrays, its own index range;
/// tangents are generated for normal-mapped parts (after the attributes).
pub fn part_mesh(model: &ModelData, part: &crate::import::PartData, tangents: bool) -> Mesh {
    let mut mesh = mesh_from_data_skinned(model);
    let start = part.start as usize;
    let end = (part.start + part.count).min(model.mesh.indices.len() as u32) as usize;
    mesh.insert_indices(Indices::U32(model.mesh.indices[start..end].to_vec()));
    if tangents {
        if let Err(e) = mesh.generate_tangents() {
            bevy::log::warn!("failed to generate tangents for normal map: {e}");
        }
    }
    mesh
}

/// Builds a StandardMaterial from an import record: base color (sRGB) + texture,
/// metallic/roughness (factors + packed B/G texture, linear), normal map
/// (linear). Also returns the "tangents needed" flag.
fn material_from_record(
    world: &mut World,
    rec: &crate::import::MaterialRecord,
    tex: Option<&TextureRoot>,
) -> (Handle<StandardMaterial>, bool) {
    // Decoding is pure CPU work before touching Bevy resources.
    let decode = |t: &crate::import::TextureRef| -> Option<image::RgbaImage> {
        let path = tex.and_then(|root| root.resolve(&t.path)).or_else(|| {
            bevy::log::warn!("texture not available inside the project: {}", t.path);
            None
        })?;
        match image::open(&path) {
            Ok(i) => Some(i.to_rgba8()),
            Err(e) => {
                bevy::log::warn!("texture {}: {e}", path.display());
                None
            }
        }
    };
    let to_image = |img: image::RgbaImage, srgb: bool| {
        let (w, h) = img.dimensions();
        Image::new(
            Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            TextureDimension::D2,
            img.into_raw(),
            if srgb { TextureFormat::Rgba8UnormSrgb } else { TextureFormat::Rgba8Unorm },
            RenderAssetUsages::default(),
        )
    };
    let base_tex = rec.base_color_texture.as_ref().and_then(|t| decode(t)).map(|i| to_image(i, true));
    let normal_tex = rec.normal_map_texture.as_ref().and_then(|t| decode(t)).map(|i| to_image(i, false));
    let mr_tex = match (rec.roughness_texture.as_ref(), rec.metalness_texture.as_ref()) {
        (None, None) => None,
        // A shared file already uses the glTF layout (G=roughness, B=metallic).
        (Some(r), Some(m)) if r.path == m.path => decode(r).map(|i| to_image(i, false)),
        (r, m) => {
            let rough = r.and_then(|t| decode(t));
            let metal = m.and_then(|t| decode(t));
            crate::import::pack_metallic_roughness(rough.as_ref(), metal.as_ref(), rec.invert_roughness)
                .map(|i| to_image(i, false))
        }
    };
    let needs_tangents = normal_tex.is_some();
    let mut images = world.resource_mut::<Assets<Image>>();
    let base_tex = base_tex.map(|i| images.add(i));
    let normal_tex = normal_tex.map(|i| images.add(i));
    let mr_tex = mr_tex.map(|i| images.add(i));
    drop(images);
    let [r, g, b, a] = rec.base_color;
    let mat = world.resource_mut::<Assets<StandardMaterial>>().add(StandardMaterial {
        base_color: Color::srgba(r, g, b, a),
        base_color_texture: base_tex,
        metallic: rec.metallic.clamp(0.0, 1.0),
        perceptual_roughness: rec.roughness.clamp(0.0, 1.0),
        metallic_roughness_texture: mr_tex,
        normal_map_texture: normal_tex,
        ..default()
    });
    (mat, needs_tangents)
}

fn mesh_from_data(data: &crate::import::MeshData) -> Mesh {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, data.positions.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, data.normals.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, data.uvs.clone());
    mesh.insert_indices(Indices::U32(data.indices.clone()));
    mesh
}

fn mesh_from_data_skinned(model: &ModelData) -> Mesh {
    let mut mesh = mesh_from_data(&model.mesh);
    if let Some(skin) = &model.skin {
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_JOINT_INDEX,
            bevy::mesh::VertexAttributeValues::Uint16x4(skin.joint_indices.clone()),
        );
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_JOINT_WEIGHT,
            bevy::mesh::VertexAttributeValues::Float32x4(skin.joint_weights.clone()),
        );
    }
    mesh
}

#[cfg(test)]
mod ambient_tests {
    use super::{ambient_from_background, zone_light_intensity};

    #[test]
    fn ambient_brightness_tracks_background_luma_not_constant() {
        let (_, night) = ambient_from_background([0.02, 0.02, 0.02]);
        let (_, day) = ambient_from_background([0.4, 0.4, 0.4]);
        let (_, bright) = ambient_from_background([0.8, 0.8, 0.8]);
        assert!(night > 150.0 && night < 250.0, "night: {night}");
        assert!((day - 950.0).abs() < 1.0, "day: {day}");
        assert!(bright > day && day > night, "monotonic in background brightness");
    }

    #[test]
    fn ambient_color_is_half_desaturated_toward_own_luma() {
        // Bluish background (the A1 case): the ambient color shifts toward neutral.
        let (color, _) = ambient_from_background([0.342, 0.417, 0.635]);
        let mean = 0.2126 * 0.342 + 0.7152 * 0.417 + 0.0722 * 0.635;
        for (c, a) in color.iter().zip([0.342, 0.417, 0.635]) {
            assert!((c - 0.5 * (a + mean)).abs() < 1e-6);
        }
        assert!(color[2] - color[0] < 0.635 - 0.342, "blue bias is reduced");
    }

    #[test]
    fn zone_lights_are_hundreds_of_lux_not_single_lux() {
        let day = zone_light_intensity([0.4, 0.4, 0.4]);
        let night = zone_light_intensity([0.02, 0.02, 0.02]);
        // At ~4 m from the actor: lux ≈ lumens / (4π·16). Day — around 800 lux,
        // night — around 330 lux (the previous 150 lm gave ~1–9 lux).
        assert!((day / (4.0 * std::f32::consts::PI * 16.0) - 810.0).abs() < 20.0, "day: {day}");
        assert!(night / (4.0 * std::f32::consts::PI * 16.0) > 300.0, "night: {night}");
        assert!(day > night);
    }
}

#[cfg(test)]
mod material_patch_tests {
    use super::apply_material_patches;
    use bevy::prelude::*;
    use phantom_core::MaterialPatch;

    fn base() -> StandardMaterial {
        StandardMaterial { base_color: Color::srgb(0.5, 0.5, 0.5), perceptual_roughness: 0.9, ..default() }
    }

    #[test]
    fn patches_apply_only_to_matching_target() {
        let patches = vec![MaterialPatch { target: Some("metal".into()), roughness: Some(0.1), ..Default::default() }];
        assert!(apply_material_patches(&base(), &patches, Some("fur")).is_none());
        let m = apply_material_patches(&base(), &patches, Some("metal")).unwrap();
        assert!((m.perceptual_roughness - 0.1).abs() < 1e-6);
        // A patch without target applies to all batches.
        let all = vec![MaterialPatch { opacity: Some(0.4), ..Default::default() }];
        let m = apply_material_patches(&base(), &all, None).unwrap();
        assert!(matches!(m.alpha_mode, AlphaMode::Blend), "opacity < 1 → blend");
    }

    #[test]
    fn later_patches_win_and_emissive_unlit_set() {
        let patches = vec![
            MaterialPatch { roughness: Some(0.8), ..Default::default() },
            MaterialPatch { roughness: Some(0.2), emissive: Some([2.0, 0.0, 0.0]), unlit: Some(true), ..Default::default() },
        ];
        let m = apply_material_patches(&base(), &patches, None).unwrap();
        assert!((m.perceptual_roughness - 0.2).abs() < 1e-6);
        assert!(m.unlit);
        assert!(m.emissive.red > 1.5, "emissive may exceed 1 (glow)");
    }
}

#[cfg(test)]
mod vfx_tests {
    use super::{particle_count, particle_state, MAX_PARTICLES};

    #[test]
    fn particles_are_deterministic_and_capped() {
        let vfx = phantom_core::vfx_preset("snow").unwrap();
        assert_eq!(particle_count(&vfx), 1024.min((200.0f32 * 6.0).ceil() as usize));
        let dense = phantom_core::SceneVfx { emit_rate: 100_000.0, ..phantom_core::vfx_preset("rain").unwrap() };
        assert_eq!(particle_count(&dense), MAX_PARTICLES);
        // Determinism: the same frame gives the same position; different particles differ.
        let a1 = particle_state(&vfx, 3, 1.5);
        let a2 = particle_state(&vfx, 3, 1.5);
        assert_eq!(a1, a2);
        let b = particle_state(&vfx, 4, 1.5);
        assert_ne!(a1.0, b.0);
        // Snow falls: y decreases over time (after spawn).
        let early = particle_state(&vfx, 0, 0.2).0.y;
        let late = particle_state(&vfx, 0, 1.0).0.y;
        assert!(late < early, "snow falls: {early} -> {late}");
        // Fade at the life edges is soft.
        assert!(particle_state(&vfx, 0, 0.01).1 < 0.1);
    }
}

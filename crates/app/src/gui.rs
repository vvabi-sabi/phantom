//! Storyboard editor: viewport, timeline, outliner.
//! Every document change goes through phantom_core::History — the same command
//! dispatcher used by the CLI and (later) MCP.

use std::path::PathBuf;
use std::process::{Child, Stdio};

use anyhow::Result;
use bevy::{
    asset::RenderAssetUsages,
    image::Image,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
};
use bevy_egui::{egui, EguiContext, EguiPlugin, EguiPrimaryContextPass, EguiUserTextures, PrimaryEguiContext};
use phantom_core::*;
use phantom_core::project::BackgroundSource;
use phantom_core::Command as DocCommand;
use phantom_runtime::{
    render::{LiveSync, SceneCache},
    render::outline::OutlinePlugin,
    storage,
};

const VIEWPORT_W: u32 = 1280;
const VIEWPORT_H: u32 = 720;
const DEJAVU: &str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";

pub fn run_editor(project_dir: PathBuf) -> Result<()> {
    let project = storage::load_project(&project_dir)?;
    let state = EditorState::new(project, project_dir);

    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(bevy::render::RenderPlugin {
                render_creation: bevy::render::settings::RenderCreation::Automatic(
                    bevy::render::settings::WgpuSettings {
                        backends: Some(bevy::render::settings::Backends::VULKAN),
                        ..Default::default()
                    },
                ),
                ..Default::default()
            })
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: format!("Phantom-animate — {}", state.project().name),
                    resolution: (1440u32, 860u32).into(),
                    ..default()
                }),
                ..default()
            }),
    );
    app.add_plugins(EguiPlugin::default());
    app.add_plugins(OutlinePlugin);
    app.init_resource::<SceneCache>();
    app.init_resource::<phantom_runtime::render::DrawCache>();
    app.init_resource::<phantom_runtime::render::VideoCache>();

    // The viewport texture is created in Startup (needs Assets<Image> and egui textures).
    app.add_systems(Startup, startup_viewport);
    app.add_systems(Update, scene_update);
    app.add_systems(EguiPrimaryContextPass, ui_system);

    // Bridge for MCP: first the shared atomic project lock, then the socket.
    // If bind fails, the local guard is dropped immediately and removes only
    // its own token; the editor never starts without an exclusive lock.
    let project_dir = state.project_dir.clone();
    let sock = crate::bridge::socket_path(&project_dir)?;
    let acquired = crate::project_lock::LockGuard::acquire(&project_dir, Some(sock.clone()), false)?;
    if let Some(owner) = &acquired.reclaimed {
        eprintln!("bridge: displaced stale lock {}", owner.summary());
    }
    let rx = match crate::bridge::host(&sock) {
        Ok(rx) => rx,
        Err(e) => {
            drop(acquired);
            return Err(e.into());
        }
    };
    app.insert_non_send_resource(BridgeGuard { lock: acquired.guard, socket: sock });
    app.insert_non_send_resource(BridgeQueue(Some(rx)));

    // EditorState is not a Bevy Resource (it holds a std::process::Child etc.),
    // so it lives in a NonSend.
    app.insert_non_send_resource(state);
    app.insert_non_send_resource(FontsLoaded::default());

    app.add_systems(Update, bridge_drain);
    app.run();
    Ok(())
}

struct BridgeGuard {
    lock: crate::project_lock::LockGuard,
    socket: PathBuf,
}

impl Drop for BridgeGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.socket);
        // LockGuard subsequently removes only the record with its own token.
    }
}

struct BridgeQueue(Option<std::sync::mpsc::Receiver<crate::bridge::BridgeRequest>>);

/// Executes MCP bridge requests on the editor's main thread.
fn bridge_drain(world: &mut World) {
    let requests: Vec<crate::bridge::BridgeRequest> = world
        .get_non_send_resource::<BridgeQueue>()
        .and_then(|q| q.0.as_ref().map(|rx| rx.try_iter().collect()))
        .unwrap_or_default();
    if requests.is_empty() {
        return;
    }
    let bridge_dir = world
        .get_non_send_resource::<BridgeGuard>()
        .map(|g| g.lock.dir().to_path_buf());
    let bridge_owner = world
        .get_non_send_resource::<BridgeGuard>()
        .and_then(|g| g.lock.owner().ok());
    let Some(mut state) = world.remove_non_send_resource::<EditorState>() else { return };
    for req in requests {
        let result: Result<serde_json::Value, crate::session::SessionError> =
            {
                match req.op {
                    crate::session::Op::ProjectOpen { dir, .. }
                        if bridge_dir.as_ref().is_some_and(|current| {
                            dir.canonicalize().unwrap_or_else(|_| dir.clone()) == *current
                        }) =>
                    {
                        Ok(serde_json::json!({ "ok": true, "already_open": true, "warnings": [] }))
                    }
                    crate::session::Op::LockRelease { dir, .. }
                        if bridge_dir.as_ref().is_some_and(|current| {
                            dir.canonicalize().unwrap_or_else(|_| dir.clone()) == *current
                        }) =>
                    {
                        Ok(serde_json::json!({
                            "ok": true,
                            "released": false,
                            "already_owned": true,
                            "observed_owner": bridge_owner.clone(),
                        }))
                    }
                    crate::session::Op::ProjectNew { .. } => {
                        Err(crate::session::SessionError::BridgeUnavailable {
                            operation: "project_new",
                            reason: "editor_already_owns_project",
                        })
                    }
                    crate::session::Op::ProjectOpen { dir, .. } => {
                        Err(crate::session::SessionError::ProjectSwitchRejected {
                            current: bridge_dir.clone().unwrap_or_else(|| state.project_dir.clone()),
                            target: dir,
                        })
                    }
                    op => state.session.execute(op),
                }
            };
        let response = match result {
            Ok(value) => crate::bridge::BridgeResponse::success(req.id, value),
            Err(error) => crate::bridge::BridgeResponse::failure(req.id, error.error_data()),
        };
        let _ = req.respond.send(serde_json::to_value(response).expect("bridge response"));
    }
    world.insert_non_send_resource(state);
}

/// Viewport texture and its egui id.
struct Viewport {
    image: Handle<Image>,
    tex: egui::TextureId,
}

fn startup_viewport(world: &mut World) {
    // Window camera for egui: without it the primary egui context attaches
    // to the scene camera and draws the UI into the viewport texture.
    world.spawn(Camera2d);
    let size = Extent3d { width: VIEWPORT_W, height: VIEWPORT_H, depth_or_array_layers: 1 };
    let mut image = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage =
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC;
    let handle = world.resource_mut::<Assets<Image>>().add(image);
    let tex = world
        .resource_mut::<EguiUserTextures>()
        .add_image(bevy_egui::EguiTextureHandle::Strong(handle.clone()));
    world.insert_non_send_resource(Viewport { image: handle, tex });
}

// ---------------------------------------------------------------------------
// Editor state
// ---------------------------------------------------------------------------

struct ExportJob {
    child: Child,
    rx: std::sync::mpsc::Receiver<String>,
    last_line: String,
    done: Option<bool>,
    /// Temporary file removed on cancel (MP4 .part).
    cleanup: Option<PathBuf>,
}

struct EditorState {
    session: crate::session::Session,
    project_dir: PathBuf,

    scene: Option<SceneId>,
    selected: Option<PhantomId>,
    frame: Frame,
    playing: bool,
    play_accum: f64,

    live: LiveSync,
    status: String,
    export: Option<ExportJob>,

    // Audio.
    playback: Option<phantom_runtime::audio::Playback>,
    pcm: PcmCache,
    audio_note: Option<String>,
    last_peak: f32,

    // Drawing.
    draw_mode: bool,
    draw_erase: bool,
    draw_color: [f32; 4],
    draw_width: f32,
    onion_skin: bool,
    stroke_in_progress: Option<Vec<[f32; 2]>>,

    // Panel fields.
    cam_pos: [f32; 3],
    cam_yaw: f32,
    cam_pitch: f32,
    cam_focal: f32,
    // Stage lighting and floor of the active scene (editable copies).
    lighting: SceneLighting,
    floor_on: bool,
    floor_color: [f32; 3],
    // Scene stage (layout mode): the box actors and props are confined to.
    stage: Stage,
    import_path: String,
    bg_asset: u64,
    // Flat backdrop geometry (editable copies of the active scene's screen).
    bg_width: f32,
    bg_height: f32,
    bg_center: [f32; 3],
    export_out: String,
    film_out: String,
    trim_edit: Option<(ClipId, Frame, Frame)>,
    // Audio clip editor: target track, source asset and start frame.
    audio_track: u64,
    audio_asset: u64,
    audio_start: Frame,
    bg_is_video: bool,
    // Builtin actor chosen when creating a Phantom.
    new_actor: phantom_core::project::BuiltinActor,
    // Asset id of the imported FBX rig used when creating a Phantom.
    new_phantom_asset: u64,
    // Prop: selected object and the fields of its transform key.
    selected_object: Option<phantom_core::ObjectId>,
    new_object_asset: u64,
    obj_pos: [f32; 3],
    obj_yaw: f32,
    obj_scale: [f32; 3],
}

impl EditorState {
    fn new(project: Project, project_dir: PathBuf) -> Self {
        let scene = project.scenes.keys().next().copied();
        let mut session = crate::session::Session::new();
        session.project = Some(project);
        session.dir = Some(project_dir.clone());
        let mut s = Self {
            session,
            project_dir,
            scene: None,
            selected: None,
            frame: 0,
            playing: false,
            play_accum: 0.0,
            live: LiveSync::default(),
            status: String::new(),
            export: None,
            playback: None,
            pcm: PcmCache::default(),
            audio_note: None,
            last_peak: 0.0,
            draw_mode: false,
            draw_erase: false,
            draw_color: [0.1, 0.1, 0.1, 1.0],
            draw_width: 6.0,
            onion_skin: true,
            stroke_in_progress: None,
            cam_pos: [0.0, 1.4, 4.0],
            cam_yaw: 0.0,
            cam_pitch: 0.0,
            cam_focal: 35.0,
            lighting: SceneLighting::default(),
            floor_on: false,
            floor_color: SceneFloor::default().color,
            stage: Stage::default(),
            import_path: String::new(),
            bg_asset: 0,
            bg_width: 24.0,
            bg_height: 8.0,
            bg_center: [0.0, 4.0, -5.0],
            export_out: "render_out".into(),
            film_out: "film.mp4".into(),
            trim_edit: None,
            audio_track: 0,
            audio_asset: 0,
            audio_start: 0,
            bg_is_video: false,
            new_actor: phantom_core::project::BuiltinActor::Humanoid,
            new_phantom_asset: 0,
            selected_object: None,
            new_object_asset: 0,
            obj_pos: [0.0, 0.0, 0.0],
            obj_yaw: 0.0,
            obj_scale: [1.0, 1.0, 1.0],
        };
        s.set_scene(scene);
        s
    }

    fn project(&self) -> &Project {
        self.session.project.as_ref().expect("project open")
    }

    fn scene(&self) -> Option<&Scene> {
        self.scene.and_then(|id| self.project().scenes.get(&id))
    }

    fn set_scene(&mut self, id: Option<SceneId>) {
        self.scene = id;
        self.selected = None;
        self.selected_object = None;
        self.playing = false;
        let snapshot = self.scene().map(|scene| {
            let active = scene.camera_for(Some(scene.active_camera));
            (
                scene.range,
                active.keys.first().copied(),
                active.focal_mm,
                scene.phantoms.keys().next().copied(),
                scene.phantoms.values().next().and_then(|p| p.motion.first().map(|m| (m.position, m.yaw))),
                scene.background.as_ref().map(|b| (b.image.asset().0, b.width, b.height, b.center.to_array())),
                scene.lighting,
                scene.floor,
                scene.stage,
            )
        });
        if let Some((range, cam_key, focal, first_phantom, first_pos, bg, lighting, floor, stage)) = snapshot {
            let (bg_asset, bg_width, bg_height, bg_center) = match bg {
                Some((asset, width, height, center)) => (asset, width, height, center),
                None => (0, self.bg_width, self.bg_height, self.bg_center),
            };
            self.lighting = lighting;
            self.floor_on = floor.is_some();
            self.floor_color = floor.map(|f| f.color).unwrap_or_else(|| SceneFloor::default().color);
            self.stage = stage;
            self.frame = range.start;
            let k = cam_key.unwrap_or(TransformKey::at(0, glam::Vec3::new(0.0, 1.4, 4.0), 0.0, 0.0));
            self.cam_pos = k.translation.to_array();
            self.cam_yaw = k.yaw;
            self.cam_pitch = k.pitch;
            self.cam_focal = focal;
            self.selected = first_phantom;
            let _ = first_pos;
            self.bg_asset = bg_asset;
            self.bg_width = bg_width;
            self.bg_height = bg_height;
            self.bg_center = bg_center;
        }
        // Structure changed — LiveSync rebuilds by scene id.
    }

    /// Switches the scene's active camera: the document records it, the panel
    /// fields take the camera's first key, and the frame is clamped to its range.
    fn set_active_camera(&mut self, camera: CameraId) {
        let Some(scene_id) = self.scene else { return };
        // AddCamera already activates the new camera, so only an explicit
        // switch issues a command (and an undo step).
        if self.scene().map(|s| s.active_camera) != Some(camera) {
            self.apply(DocCommand::ActivateCamera { scene: scene_id, camera });
        }
        let Some(scene) = self.scene() else { return };
        let cam = scene.camera_for(Some(camera));
        let key = cam.keys.first().map(|k| (k.translation.to_array(), k.yaw, k.pitch));
        let focal = cam.focal_mm;
        let range = scene.range_for(Some(camera));
        if let Some((pos, yaw, pitch)) = key {
            self.cam_pos = pos;
            self.cam_yaw = yaw;
            self.cam_pitch = pitch;
        }
        self.cam_focal = focal;
        self.frame = range.clamp(self.frame);
    }

    fn apply_full(&mut self, cmd: DocCommand) -> Result<CommandOutput, CoreError> {
        let session = &mut self.session;
        session.history.apply(session.project.as_mut().expect("project open"), cmd)
    }

    fn apply(&mut self, cmd: DocCommand) {
        let session = &mut self.session;
        let project = session.project.as_mut().expect("project open");
        match session.history.apply(project, cmd) {
            Ok(_) => self.status.clear(),
            Err(e) => self.status = e.to_string(),
        }
    }

    fn save(&mut self) {
        match storage::save_project(&self.project_dir, self.project()) {
            Ok(()) => self.status = "saved".into(),
            Err(e) => self.status = format!("save error: {e}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Per-frame scene update (timeline advance and world sync)
// ---------------------------------------------------------------------------

/// Debug viewport dump: PHANTOM_DEBUG_SHOT=/path.png — after 90 frames
/// save the viewport to PNG and exit the process (GUI check without a screenshot tool).
fn debug_shot(world: &mut World) -> bool {
    let Ok(path) = std::env::var("PHANTOM_DEBUG_SHOT") else { return false };
    let mut frames = world.remove_non_send_resource::<DebugShot>().unwrap_or_default();
    frames.0 += 1;
    if frames.0 < 90 || frames.1 {
        world.insert_non_send_resource(frames);
        return false;
    }
    frames.1 = true;
    world.insert_non_send_resource(frames);
    let target = world.non_send_resource::<Viewport>().image.clone();
    if let Some(state) = world.get_non_send_resource::<EditorState>() {
        eprintln!("debug shot: status={:?} scene={:?} frame={}", state.status, state.scene, state.frame);
    }
    world.spawn(bevy::render::gpu_readback::Readback::texture(target)).observe(
        move |ev: On<bevy::render::gpu_readback::ReadbackComplete>| {
            let data = &ev.data;
            let tight = (VIEWPORT_W * VIEWPORT_H * 4) as usize;
            let rgba = if data.len() == tight {
                data.clone()
            } else {
                let stride = data.len() / VIEWPORT_H as usize;
                let row = VIEWPORT_W as usize * 4;
                let mut out = Vec::with_capacity(tight);
                for y in 0..VIEWPORT_H as usize {
                    out.extend_from_slice(&data[y * stride..y * stride + row]);
                }
                out
            };
            let img: image::RgbaImage =
                image::ImageBuffer::from_raw(VIEWPORT_W, VIEWPORT_H, rgba).unwrap();
            img.save(&path).expect("debug shot saved");
            eprintln!("debug shot -> {path}");
            std::process::exit(0);
        },
    );
    true
}

#[derive(Default)]
struct DebugShot(u32, bool);

fn scene_update(world: &mut World) {
    if debug_shot(world) {
        return;
    }
    let dt = world.resource::<Time>().delta_secs_f64().min(0.1);
    let Some(mut state) = world.remove_non_send_resource::<EditorState>() else { return };
    {
            let Some(scene_id) = state.scene else {
                world.insert_non_send_resource(state);
                return;
            };
            let range = match state.scene() {
                Some(s) => s.range,
                None => return,
            };

            // Timeline advance.
            let advancing = state.playing;
            if advancing && state.playback.is_none() && state.audio_note.is_none() {
                // Start preview; without a device — silent fallback with a note.
                match phantom_runtime::audio::Playback::start() {
                    Ok(Some(p)) => state.playback = Some(p),
                    Ok(None) => state.audio_note = Some("no audio device — silent preview".into()),
                    Err(e) => state.audio_note = Some(format!("audio preview unavailable: {e}")),
                }
                let audio_assets: Vec<_> = state
                    .project()
                    .assets
                    .values()
                    .filter(|a| a.kind == AssetKind::Audio)
                    .map(|a| (a.id, state.project_dir.join(&a.path)))
                    .collect();
                if let Err(e) = phantom_runtime::audio::ensure_pcm(audio_assets, &mut state.pcm) {
                    state.status = format!("audio decode: {e}");
                }
            }
            if !advancing {
                state.playback = None;
                state.audio_note = None;
            }
            if advancing {
                state.play_accum += dt * FPS as f64;
                let passed = state.play_accum.floor() as Frame;
                state.play_accum -= passed as f64;
                if passed > 0 {
                    let from = state.frame;
                    state.frame = (state.frame + passed).min(range.end - 1);
                    // Feed audio for every frame passed.
                    {
                        let scene_tracks = state.scene().map(|s| s.audio_tracks.clone());
                        if let (Some(pb), Some(tracks)) = (state.playback.as_ref(), scene_tracks) {
                            let mut peak = 0.0f32;
                            for f in from + 1..=state.frame {
                                let mut block = [[0.0f32; 2]; SAMPLES_PER_FRAME as usize];
                                peak = peak.max(mix_frame(&tracks, f, &state.pcm, &mut block));
                                pb.push(&block);
                            }
                            state.last_peak = peak;
                        }
                    }
                    if state.frame >= range.end - 1 {
                        state.playing = false;
                    }
                }
            }

            // World synchronization.
            let (project_dir, revision_scene) = (state.project_dir.clone(), scene_id);
            let target = world.non_send_resource::<Viewport>().image.clone();
            {
                let project_ref = state.session.project.as_ref().expect("project open");
                if let Err(e) = state.live.ensure(world, project_ref, revision_scene, &project_dir, &target) {
                    state.status = e.to_string();
                    world.insert_non_send_resource(state);
                    return;
                }
            }
            {
                if let Some(project_ref) = state.session.project.as_ref() {
                    if let Some(scene) = project_ref.scenes.get(&revision_scene) {
                        state.live.apply(world, project_ref, scene, state.frame, state.onion_skin, Some(scene.active_camera));
                    }
                }
            }
    }
    world.insert_non_send_resource(state);
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

fn ui_system(world: &mut World) {
    let ctx = {
        let mut q = world.query_filtered::<&mut EguiContext, With<PrimaryEguiContext>>();
        q.single_mut(world).expect("primary egui context").get_mut().clone()
    };
    let viewport_tex = world.non_send_resource::<Viewport>().tex;

    // DejaVu font — once (project data may still hold Cyrillic text).
    {
        let mut fonts_added = world.remove_non_send_resource::<FontsLoaded>().unwrap_or_default();
        if !fonts_added.0 {
            if let Ok(data) = std::fs::read(DEJAVU) {
                let mut fonts = egui::FontDefinitions::default();
                fonts.font_data.insert("dejavu".into(), std::sync::Arc::new(egui::FontData::from_owned(data)));
                fonts
                    .families
                    .entry(egui::FontFamily::Proportional)
                    .or_default()
                    .insert(0, "dejavu".into());
                ctx.set_fonts(fonts);
            }
            fonts_added.0 = true;
        }
        world.insert_non_send_resource(fonts_added);
    }

    if let Some(mut state) = world.remove_non_send_resource::<EditorState>() {
        top_panel(&ctx, &mut state);
        right_panel(&ctx, &mut state);
        bottom_panel(&ctx, &mut state);
        viewport_panel(&ctx, &mut state, viewport_tex);
        world.insert_non_send_resource(state);
    }
}

#[derive(Default)]
struct FontsLoaded(bool);

fn top_panel(ctx: &egui::Context, state: &mut EditorState) {
    // Hotkeys.
    let (save, undo, redo) = ctx.input(|i| {
        (
            i.modifiers.command && i.key_pressed(egui::Key::S),
            i.modifiers.command && !i.modifiers.shift && i.key_pressed(egui::Key::Z),
            i.modifiers.command && (i.key_pressed(egui::Key::Y) || (i.modifiers.shift && i.key_pressed(egui::Key::Z))),
        )
    });
    if save {
        state.save();
    }
    if undo {
        { let s = &mut state.session; s.history.undo(s.project.as_mut().unwrap()); }
    }
    if redo {
        { let s = &mut state.session; s.history.redo(s.project.as_mut().unwrap()); }
    }

    egui::TopBottomPanel::top("top").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.label(format!("Phantom-animate — {}", state.project().name));
            ui.separator();
            if ui.button("Save (Ctrl+S)").clicked() {
                state.save();
            }
            if ui.button("Undo").clicked() {
                { let s = &mut state.session; s.history.undo(s.project.as_mut().unwrap()); }
            }
            if ui.button("Redo").clicked() {
                { let s = &mut state.session; s.history.redo(s.project.as_mut().unwrap()); }
            }
            ui.separator();
            ui.label("Export to:");
            ui.add(egui::TextEdit::singleline(&mut state.export_out).desired_width(140.0));
            let exporting = state.export.is_some();
            if ui.add_enabled(!exporting, egui::Button::new("Render PNG")).clicked() {
                start_export(state);
            }
            poll_export(state);
            if let Some(job) = &state.export {
                match job.done {
                    Some(true) => {
                        ui.label("export finished");
                        if ui.small_button("✕").clicked() {
                            state.export = None;
                        }
                    }
                    Some(false) => {
                        ui.label("export: error");
                        if ui.small_button("✕").clicked() {
                            state.export = None;
                        }
                    }
                    None => {
                        ui.label(format!("rendering… {}", job.last_line));
                        if ui.small_button("cancel").clicked() {
                            if let Some(mut job) = state.export.take() {
                                let _ = job.child.kill();
                                let _ = job.child.wait();
                                if let Some(tmp) = job.cleanup {
                                    let _ = std::fs::remove_file(tmp);
                                }
                            }
                        }
                    }
                };
            }
        });
    });
}

fn spawn_export(state: &mut EditorState, args: Vec<std::ffi::OsString>) {
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(e) => {
            state.status = format!("binary path not found: {e}");
            return;
        }
    };
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    match std::process::Command::new(exe)
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(mut child) => {
            let stdout = child.stdout.take().unwrap();
            std::thread::spawn(move || {
                use std::io::{BufRead, BufReader};
                for line in BufReader::new(stdout).lines() {
                    if let Ok(line) = line {
                        let _ = tx.send(line);
                    }
                }
            });
            state.export = Some(ExportJob { child, rx, last_line: String::new(), done: None, cleanup: None });
        }
        Err(e) => state.status = format!("cannot start export: {e}"),
    }
}

fn start_export(state: &mut EditorState) {
    let Some(scene_id) = state.scene else { return };
    state.save();
    let out = state.project_dir.join(&state.export_out);
    let mut args = vec![
        "render".into(),
        state.project_dir.clone().into(),
        "--scene".into(),
        scene_id.to_string().into(),
        "--out".into(),
        out.into(),
    ];
    if let Some(scene) = state.scene() {
        // The render child uses the scene's active camera, as the viewport does.
        args.push("--camera".into());
        args.push(scene.active_camera.to_string().into());
    }
    spawn_export(state, args);
}

fn start_film_export(state: &mut EditorState) {
    if state.project().film.is_empty() {
        state.status = "film is empty — add clips".into();
        return;
    }
    state.save();
    let out = state.project_dir.join(&state.film_out);
    let cleanup = out.with_extension("part.mp4");
    spawn_export(
        state,
        vec![
            "render-film".into(),
            state.project_dir.clone().into(),
            "--out".into(),
            out.into(),
            // Button + explicit file name = permission to overwrite.
            "--force".into(),
        ],
    );
    if let Some(job) = &mut state.export {
        job.cleanup = Some(cleanup);
    }
}

fn poll_export(state: &mut EditorState) {
    let Some(job) = &mut state.export else { return };
    while let Ok(line) = job.rx.try_recv() {
        job.last_line = line;
    }
    match job.child.try_wait() {
        Ok(Some(status)) => job.done = Some(status.success()),
        Ok(None) => {}
        Err(_) => job.done = Some(false),
    }
}

fn right_panel(ctx: &egui::Context, state: &mut EditorState) {
    egui::SidePanel::right("right").default_width(260.0).show(ctx, |ui| {
        ui.heading("Scene");
        let mut chosen = state.scene;
        egui::ComboBox::from_id_salt("scene_combo")
            .selected_text(
                state.scene().map(|s| s.name.clone()).unwrap_or_else(|| "—".into()),
            )
            .show_ui(ui, |ui| {
                for (id, scene) in &state.project().scenes {
                    ui.selectable_value(&mut chosen, Some(*id), &scene.name);
                }
            });
        if chosen != state.scene {
            state.set_scene(chosen);
        }
        ui.horizontal(|ui| {
            if ui.button("New scene").clicked() {
                let n = state.project().scenes.len() + 1;
                match FrameRange::new(0, 96) {
                    Ok(range) => {
                        let out = state.apply_full(
                            DocCommand::AddScene { name: format!("Scene {n}"), range },
                        );
                        match out {
                            Ok(o) => state.set_scene(o.scene),
                            Err(e) => state.status = e.to_string(),
                        }
                    }
                    Err(e) => state.status = e.to_string(),
                }
            }
        });

        ui.separator();
        ui.heading("Phantoms");
        let mut select = state.selected;
        if let Some(scene) = state.scene() {
            for (id, p) in &scene.phantoms {
                ui.selectable_value(&mut select, Some(*id), &p.name);
            }
        }
        if select != state.selected {
            state.selected = select;
        }
        ui.horizontal(|ui| {
            ui.label("Actor:");
            egui::ComboBox::from_id_salt("new_actor")
                .selected_text(crate::builtin::actor_name(state.new_actor))
                .show_ui(ui, |ui| {
                    for (actor, name) in crate::builtin::ACTORS {
                        ui.selectable_value(&mut state.new_actor, *actor, *name);
                    }
                });
            if ui.button("Add Phantom").clicked() {
                if let Some(scene_id) = state.scene {
                    let n = state.scene().map(|s| s.phantoms.len() + 1).unwrap_or(1);
                    let out = state.apply_full(
                        DocCommand::AddPhantom {
                            scene: scene_id,
                            name: format!("Phantom {n}"),
                            rig: RigRef::BuiltinActor { actor: state.new_actor },
                        },
                    );
                    match out {
                        Ok(o) => state.selected = o.phantom,
                        Err(e) => state.status = e.to_string(),
                    }
                }
            }
        });
        // Phantom from an imported FBX rig (RigRef::Imported).
        ui.horizontal(|ui| {
            ui.label("FBX rig (asset id)");
            ui.add(egui::DragValue::new(&mut state.new_phantom_asset));
            if ui.button("Add from FBX").clicked() {
                if let Some(scene_id) = state.scene {
                    let n = state.scene().map(|s| s.phantoms.len() + 1).unwrap_or(1);
                    let out = state.apply_full(DocCommand::AddPhantom {
                        scene: scene_id,
                        name: format!("Phantom {n}"),
                        rig: RigRef::Imported { asset: AssetId(state.new_phantom_asset) },
                    });
                    match out {
                        Ok(o) => state.selected = o.phantom,
                        Err(e) => state.status = e.to_string(),
                    }
                }
            }
        });
        // Hint: imported FBX models usable as a rig.
        {
            let rigs: Vec<String> = state
                .project()
                .assets
                .values()
                .filter(|a| a.kind == AssetKind::ModelFbx)
                .map(|a| format!("{} ({})", a.id, a.path.display()))
                .collect();
            if !rigs.is_empty() {
                ui.label(format!("FBX rigs: {}", rigs.join(", ")));
            }
        }

        ui.separator();
        ui.heading("Objects");
        {
            let mut select_obj = state.selected_object;
            if let Some(scene) = state.scene() {
                for (id, o) in &scene.objects {
                    ui.selectable_value(&mut select_obj, Some(*id), &o.name);
                }
            }
            if select_obj != state.selected_object {
                state.selected_object = select_obj;
                // Key fields come from the computed transform at the current frame.
                if let (Some(scene), Some(oid)) = (state.scene(), select_obj) {
                    let t = phantom_core::evaluate_object(&scene.objects[&oid], state.frame);
                    state.obj_pos = t.position.to_array();
                    state.obj_yaw = t.yaw;
                    state.obj_scale = t.scale.to_array();
                }
            }
            // Selected object: transform keys at the current frame.
            if let Some(oid) = state.selected_object {
                let has_key_here = state
                    .scene()
                    .and_then(|s| s.objects.get(&oid))
                    .map(|o| o.keys.iter().any(|k| k.frame == state.frame))
                    .unwrap_or(false);
                ui.horizontal(|ui| {
                    ui.label("Position");
                    for v in &mut state.obj_pos {
                        ui.add(egui::DragValue::new(v).speed(0.05));
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Yaw");
                    ui.add(egui::DragValue::new(&mut state.obj_yaw).speed(0.01));
                    ui.label("Scale");
                    for v in &mut state.obj_scale {
                        ui.add(egui::DragValue::new(v).speed(0.05).range(0.01..=100.0));
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("Key object at current frame").clicked() {
                        if let Some(sid) = state.scene {
                            state.apply(DocCommand::SetObjectKey {
                                scene: sid,
                                object: oid,
                                key: phantom_core::ObjectKey {
                                    frame: state.frame,
                                    position: glam::Vec3::from_array(state.obj_pos),
                                    yaw: state.obj_yaw,
                                    pitch: 0.0,
                                    roll: 0.0,
                                    scale: glam::Vec3::from_array(state.obj_scale),
                                },
                            });
                        }
                    }
                    if has_key_here && ui.small_button("Remove key").clicked() {
                        if let Some(sid) = state.scene {
                            state.apply(DocCommand::RemoveObjectKey { scene: sid, object: oid, frame: state.frame });
                        }
                    }
                    if ui.small_button("Remove object").clicked() {
                        if let Some(sid) = state.scene {
                            state.apply(DocCommand::RemoveObject { scene: sid, object: oid });
                            state.selected_object = None;
                        }
                    }
                });
            }
            ui.horizontal(|ui| {
                ui.label("Model (asset id)");
                ui.add(egui::DragValue::new(&mut state.new_object_asset));
                if ui.button("Add object").clicked() {
                    if let Some(sid) = state.scene {
                        let n = state.scene().map(|s| s.objects.len() + 1).unwrap_or(1);
                        let out = state.apply_full(DocCommand::AddObject {
                            scene: sid,
                            name: format!("Object {n}"),
                            asset: AssetId(state.new_object_asset),
                        });
                        match out {
                            Ok(o) => state.selected_object = o.object,
                            Err(e) => state.status = e.to_string(),
                        }
                    }
                }
            });
            // Hint: imported models usable as props.
            let models: Vec<String> = state
                .project()
                .assets
                .values()
                .filter(|a| matches!(a.kind, AssetKind::ModelObj | AssetKind::ModelFbx))
                .map(|a| format!("{} ({:?})", a.id, a.kind))
                .collect();
            if !models.is_empty() {
                ui.label(format!("models: {}", models.join(", ")));
            }
        }

        ui.separator();
        ui.heading("Cameras");
        if let Some(scene_id) = state.scene {
            let cameras: Vec<(CameraId, String)> = state
                .scene()
                .map(|s| s.cameras.values().map(|c| (c.id, c.name.clone())).collect())
                .unwrap_or_default();
            let active = state.scene().map(|s| s.active_camera).unwrap_or(SCENE_CAMERA);
            let mut select_camera: Option<CameraId> = None;
            let mut remove_camera = None;
            let selected_name = cameras
                .iter()
                .find(|(id, _)| *id == active)
                .map(|(_, n)| n.clone())
                .unwrap_or_else(|| "—".into());
            egui::ComboBox::from_label("Active camera").selected_text(selected_name).show_ui(ui, |ui| {
                for (id, name) in &cameras {
                    let label = if *id == SCENE_CAMERA { format!("{name} (scene)") } else { name.clone() };
                    if ui.selectable_label(active == *id, label).clicked() {
                        select_camera = Some(*id);
                    }
                }
            });
            if let Some(id) = select_camera {
                state.set_active_camera(id);
            }
            // Camera 0 covers the whole scene and cannot be narrowed.
            if active != SCENE_CAMERA {
                let cur = state.scene().and_then(|s| s.cameras.get(&active)).map(|c| (c.range.start, c.range.end));
                if let Some((mut start, mut end)) = cur {
                    ui.horizontal(|ui| {
                        ui.label("Camera range");
                        let rs = ui.add(egui::DragValue::new(&mut start));
                        let re = ui.add(egui::DragValue::new(&mut end));
                        if rs.changed() || re.changed() {
                            match FrameRange::new(start, end) {
                                Ok(range) => state.apply(DocCommand::SetCameraRange { scene: scene_id, camera: active, range }),
                                Err(e) => state.status = e.to_string(),
                            }
                        }
                    });
                }
            }
            // Target picker: none | track | follow plus the tracked subject.
            {
                let current = state.scene().and_then(|s| s.cameras.get(&active)).and_then(|c| c.target);
                let subjects: Vec<(CameraSubject, String)> = state
                    .scene()
                    .map(|s| {
                        let mut v: Vec<(CameraSubject, String)> = s
                            .phantoms
                            .values()
                            .map(|p| (CameraSubject::Phantom(p.id), format!("actor {} ({})", p.id.0, p.name)))
                            .collect();
                        v.extend(
                            s.objects
                                .values()
                                .map(|o| (CameraSubject::Object(o.id), format!("object {} ({})", o.id.0, o.name))),
                        );
                        v
                    })
                    .unwrap_or_default();
                let mut mode: Option<CameraTargetMode> = current.map(|t| t.mode);
                let mut subject = current.map(|t| t.subject);
                let mode_label = match mode {
                    None => "none",
                    Some(CameraTargetMode::Track) => "track",
                    Some(CameraTargetMode::Follow) => "follow",
                };
                ui.horizontal(|ui| {
                    ui.label("Target");
                    egui::ComboBox::from_id_salt("camera_target_mode").selected_text(mode_label).show_ui(ui, |ui| {
                        ui.selectable_value(&mut mode, None, "none");
                        ui.selectable_value(&mut mode, Some(CameraTargetMode::Track), "track");
                        ui.selectable_value(&mut mode, Some(CameraTargetMode::Follow), "follow");
                    });
                    if mode.is_some() {
                        let subject_label = subject
                            .and_then(|s| subjects.iter().find(|(cs, _)| *cs == s).map(|(_, n)| n.clone()))
                            .unwrap_or_else(|| "—".into());
                        egui::ComboBox::from_id_salt("camera_target_subject").selected_text(subject_label).show_ui(ui, |ui| {
                            for (cs, name) in &subjects {
                                ui.selectable_value(&mut subject, Some(*cs), name);
                            }
                        });
                    }
                });
                let changed = mode != current.map(|t| t.mode) || subject != current.map(|t| t.subject);
                if changed {
                    let target = match (mode, subject) {
                        (Some(mode), Some(subject)) => Some(CameraTarget { mode, subject }),
                        _ => None,
                    };
                    state.apply(DocCommand::SetCameraTarget { scene: scene_id, camera: active, target });
                }
            }
            ui.horizontal(|ui| {
                if ui.button("Add camera").clicked() {
                    if let Some(scene) = state.scene() {
                        let n = scene.cameras.len() + 1;
                        // None — the new camera covers the whole scene and becomes active.
                        match state.apply_full(DocCommand::AddCamera { scene: scene_id, name: format!("Camera {n}"), range: None }) {
                            Ok(o) => {
                                if let Some(id) = o.camera {
                                    state.set_active_camera(id);
                                }
                                state.status.clear();
                            }
                            Err(e) => state.status = e.to_string(),
                        }
                    }
                }
                if active != SCENE_CAMERA && ui.button("Remove camera").clicked() {
                    remove_camera = Some(active);
                }
            });
            if let Some(id) = remove_camera {
                state.apply(DocCommand::RemoveCamera { scene: scene_id, camera: id });
                // The document falls back to camera 0 — reload the panel fields.
                if state.status.is_empty() {
                    state.set_active_camera(SCENE_CAMERA);
                }
            }
        }

        ui.separator();
        ui.heading("Camera");
        ui.horizontal(|ui| {
            ui.label("Position");
            for v in &mut state.cam_pos {
                ui.add(egui::DragValue::new(v).speed(0.05));
            }
        });
        ui.horizontal(|ui| {
            ui.label("Yaw/Pitch");
            ui.add(egui::DragValue::new(&mut state.cam_yaw).speed(0.01));
            ui.add(egui::DragValue::new(&mut state.cam_pitch).speed(0.01));
        });
        ui.horizontal(|ui| {
            ui.label("Focal length, mm");
            ui.add(egui::DragValue::new(&mut state.cam_focal).speed(0.5).range(8.0..=200.0));
        });
        if ui.button("Key camera at current frame").clicked() {
            if let Some(scene_id) = state.scene {
                let active = state.scene().map(|s| s.active_camera).unwrap_or(SCENE_CAMERA);
                let key = TransformKey::at(
                    state.frame,
                    glam::Vec3::from_array(state.cam_pos),
                    state.cam_yaw,
                    state.cam_pitch,
                );
                state.apply(DocCommand::SetCameraKey { scene: scene_id, camera: active, key });
                state.apply(DocCommand::SetCameraLens { scene: scene_id, camera: active, focal_mm: state.cam_focal });
            }
        }

        ui.separator();
        ui.heading("Stage");
        {
            // The stage is the boundary actors and props are confined to; a
            // size change clamps everything already staged (one undo step).
            let mut stage = state.stage;
            let mut changed = false;
            ui.horizontal(|ui| {
                ui.label("Preset");
                for preset in StagePreset::ALL {
                    let selected = stage.preset == Some(preset);
                    if ui.selectable_label(selected, preset.name()).clicked() && !selected {
                        stage = Stage::from_preset(preset);
                        changed = true;
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label("Size W×H×D");
                changed |= ui.add(egui::DragValue::new(&mut stage.size.width).speed(0.5).range(0.1..=1000.0)).changed();
                changed |= ui.add(egui::DragValue::new(&mut stage.size.height).speed(0.5).range(0.1..=1000.0)).changed();
                changed |= ui.add(egui::DragValue::new(&mut stage.size.depth).speed(0.5).range(0.1..=1000.0)).changed();
                if changed {
                    stage.preset = None;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Center x,z");
                changed |= ui.add(egui::DragValue::new(&mut stage.center[0]).speed(0.5)).changed();
                changed |= ui.add(egui::DragValue::new(&mut stage.center[1]).speed(0.5)).changed();
            });
            if changed {
                if let Some(sid) = state.scene {
                    state.apply(DocCommand::SetStage { scene: sid, stage });
                    if state.status.is_empty() {
                        state.stage = stage;
                    }
                }
            }
        }

        ui.separator();
        ui.heading("Lighting and floor");
        {
            // egui color fields are sRGB; the document stores linear RGB,
            // direct value mapping is acceptable for the editor.
            let mut lighting = state.lighting;
            let mut changed = false;
            for (label, light) in [("Key", &mut lighting.key), ("Fill", &mut lighting.fill), ("Rim", &mut lighting.rim)] {
                ui.horizontal(|ui| {
                    ui.label(label);
                    changed |= ui.checkbox(&mut light.enabled, "").changed();
                    changed |= ui.color_edit_button_rgb(&mut light.color).changed();
                    changed |= ui.add(egui::DragValue::new(&mut light.intensity).speed(50.0).range(0.0..=100000.0)).changed();
                });
            }
            if changed {
                if let Some(sid) = state.scene {
                    state.apply(DocCommand::SetLighting { scene: sid, lighting });
                    if state.status.is_empty() {
                        state.lighting = lighting;
                    }
                }
            }
            let mut floor_on = state.floor_on;
            let mut floor_color = state.floor_color;
            let mut floor_changed = false;
            ui.horizontal(|ui| {
                floor_changed |= ui.checkbox(&mut floor_on, "Floor (matte, y=0)").changed();
                floor_changed |= ui.color_edit_button_rgb(&mut floor_color).changed();
            });
            if floor_changed {
                if let Some(sid) = state.scene {
                    let floor = if floor_on { Some(SceneFloor { color: floor_color }) } else { None };
                    state.apply(DocCommand::SetFloor { scene: sid, floor });
                    if state.status.is_empty() {
                        state.floor_on = floor_on;
                        state.floor_color = floor_color;
                    }
                }
            }
        }

        ui.separator();
        ui.heading("Audio");
        let mut track_cmds: Vec<DocCommand> = Vec::new();
        let scene_id_opt = state.scene;
        // Tracks of the current scene and the film.
        let mut track_rows: Vec<(TrackId, String, f32, bool, Vec<String>)> = Vec::new();
        if let Some(scene) = state.scene() {
            for t in &scene.audio_tracks {
                track_rows.push((t.id, format!("scene: {}", t.name), t.volume, t.muted, t.clips.iter().map(|c| format!("clip {} @{} ({} samples)", c.id, c.start, c.len_samples)).collect()));
            }
        }
        for t in &state.project().film_audio {
            track_rows.push((t.id, format!("film: {}", t.name), t.volume, t.muted, t.clips.iter().map(|c| format!("clip {} @{} ({} samples)", c.id, c.start, c.len_samples)).collect()));
        }
        for (tid, label, mut volume, muted, clips) in track_rows {
            ui.horizontal(|ui| {
                ui.label(label);
                if ui.add(egui::DragValue::new(&mut volume).speed(0.05).range(0.0..=4.0)).changed() {
                    track_cmds.push(DocCommand::SetTrackVolume { track: tid, volume });
                }
                let mut m = muted;
                if ui.checkbox(&mut m, "mute").changed() {
                    track_cmds.push(DocCommand::SetTrackMuted { track: tid, muted: m });
                }
            });
            for c in clips {
                ui.label(format!("   {c}"));
            }
        }
        for cmd in track_cmds {
            state.apply(cmd);
        }
        ui.horizontal(|ui| {
            if ui.button("+ scene track").clicked() {
                if let Some(sid) = scene_id_opt {
                    let out = state.apply_full(DocCommand::AddAudioTrack { scene: Some(sid), name: "Track".into() });
                    if let Err(e) = out {
                        state.status = e.to_string();
                    }
                }
            }
            if ui.button("+ film track").clicked() {
                let out = state.apply_full(DocCommand::AddAudioTrack { scene: None, name: "Film".into() });
                if let Err(e) = out {
                    state.status = e.to_string();
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("clip track:");
            ui.add(egui::DragValue::new(&mut state.audio_track));
            ui.label("asset");
            ui.add(egui::DragValue::new(&mut state.audio_asset));
            ui.label("frame");
            ui.add(egui::DragValue::new(&mut state.audio_start));
            if ui.button("Add").clicked() {
                // Length is the whole asset: only the decoder knows the exact
                // length, so take a covering value — the mixer trims to reality.
                state.apply(DocCommand::AddAudioClip {
                    track: TrackId(state.audio_track),
                    asset: AssetId(state.audio_asset),
                    start: state.audio_start,
                    offset_samples: 0,
                    len_samples: 96 * 48000,
                    fade_in_samples: 0,
                    fade_out_samples: 0,
                });
            }
        });
        if let Some(note) = &state.audio_note {
            ui.label(note);
        }
        if state.last_peak > 1.0 {
            ui.colored_label(egui::Color32::RED, format!("CLIPPING: peak {:.2}", state.last_peak));
        }

        ui.separator();
        ui.heading("Film");
        let mut film_action = None;
        let mut trim_toggle = None;
        let film_len_frames = film_len(&state.project().film);
        for (i, clip) in state.project().film.iter().enumerate() {
            let scene_name = state
                .project()
                .scenes
                .get(&clip.scene)
                .map(|s| s.name.clone())
                .unwrap_or_else(|| "?".into());
            ui.horizontal(|ui| {
                let camera_name = clip.camera.and_then(|cid| {
                    state.project().scenes.get(&clip.scene).and_then(|s| s.cameras.get(&cid)).map(|c| c.name.clone())
                });
                let camera_suffix = camera_name.map(|n| format!(" camera \"{n}\"")).unwrap_or_default();
                ui.label(format!("{}: {}{} [{}, {})", i, scene_name, camera_suffix, clip.range.start, clip.range.end));
                if ui.small_button("↑").clicked() && i > 0 {
                    film_action = Some(("move", clip.id, i - 1));
                }
                if ui.small_button("↓").clicked() {
                    film_action = Some(("move", clip.id, i + 1));
                }
                if ui.small_button("✕").clicked() {
                    film_action = Some(("remove", clip.id, 0));
                }
                if ui.small_button("✂").clicked() {
                    trim_toggle = Some((clip.id, clip.range.start, clip.range.end));
                }
            });
        }
        if let Some(t) = trim_toggle {
            state.trim_edit = Some(t);
        }
        // Trim row for the active clip.
        let mut trim_apply = None;
        if let Some((clip_id, s, e)) = &mut state.trim_edit {
            ui.horizontal(|ui| {
                ui.label("trim:");
                ui.add(egui::DragValue::new(s).speed(1));
                ui.add(egui::DragValue::new(e).speed(1));
                if ui.button("OK").clicked() {
                    trim_apply = Some((*clip_id, *s, *e));
                }
            });
        }
        if let Some((clip, s, e)) = trim_apply {
            match FrameRange::new(s, e) {
                Ok(range) => state.apply(DocCommand::SetFilmClipRange { clip, range }),
                Err(err) => state.status = err.to_string(),
            }
        }
        if let Some((action, clip, index)) = film_action {
            match action {
                "move" => state.apply(DocCommand::MoveFilmClip { clip, index }),
                "remove" => state.apply(DocCommand::RemoveFilmClip { clip }),
                _ => {}
            }
        }
        if ui.button("Add scene to film").clicked() {
            if let Some(scene) = state.scene() {
                let camera = scene.active_camera;
                let (id, range) = (scene.id, scene.range_for(Some(camera)));
                state.apply(DocCommand::AddFilmClip { scene: id, range, camera: Some(camera) });
            }
        }
        ui.label(format!("film length: {} frames", film_len_frames));
        ui.horizontal(|ui| {
            ui.label("MP4:");
            ui.add(egui::TextEdit::singleline(&mut state.film_out).desired_width(120.0));
            if ui.button("Export film").clicked() {
                start_film_export(state);
            }
        });

        ui.separator();
        ui.heading("Background and assets");
        ui.horizontal(|ui| {
            ui.label("Image (asset id)");
            ui.add(egui::DragValue::new(&mut state.bg_asset));
            ui.checkbox(&mut state.bg_is_video, "video");
            ui.label("W×H");
            ui.add(egui::DragValue::new(&mut state.bg_width).speed(0.5).range(0.1..=1000.0));
            ui.add(egui::DragValue::new(&mut state.bg_height).speed(0.5).range(0.1..=1000.0));
            if ui.button("Apply").clicked() {
                if let Some(scene_id) = state.scene {
                    state.apply(DocCommand::SetBackground {
                        scene: scene_id,
                        background: Background {
                            image: if state.bg_is_video {
                                BackgroundSource::Video { video: AssetId(state.bg_asset) }
                            } else {
                                BackgroundSource::Image(AssetId(state.bg_asset))
                            },
                            width: state.bg_width,
                            height: state.bg_height,
                            center: glam::Vec3::from(state.bg_center),
                            fit: Default::default(),
                        },
                    });
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("Import");
            ui.add(egui::TextEdit::singleline(&mut state.import_path).desired_width(120.0));
            if ui.button("File…").clicked() {
                import_asset(state);
            }
        });
    });
}

/// Import a file from the `import_path` field into the project: obj/fbx go
/// through bundle import with MTL and textures, anything else is a single file.
fn import_asset(state: &mut EditorState) {
    let path = PathBuf::from(state.import_path.trim());
    if !path.exists() {
        state.status = "file does not exist".into();
        return;
    }
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    let is_model = matches!(ext.as_deref(), Some("obj") | Some("fbx"));
    // Models — bundle import with MTL and textures inside the project.
    let mut import_diags: Vec<String> = Vec::new();
    let imported: Result<(PathBuf, Option<phantom_core::ModelBounds>), String> = if is_model {
        match phantom_runtime::import::import_model_bundle(&state.project_dir, &path) {
            Ok(b) => {
                import_diags = b.diagnostics;
                Ok((b.model, b.bounds))
            }
            Err(e) => Err(e.to_string()),
        }
    } else {
        storage::import_file(&state.project_dir, &path).map(|p| (p, None)).map_err(|e| e.to_string())
    };
    match imported {
        Ok((rel, bounds)) => {
            let kind = match ext.as_deref() {
                Some("obj") => AssetKind::ModelObj,
                Some("fbx") => AssetKind::ModelFbx,
                _ => AssetKind::Image,
            };
            match state.apply_full(DocCommand::RegisterAsset { kind, path: rel, bounds }) {
                Ok(o) => {
                    state.bg_asset = o.asset.map(|a| a.0).unwrap_or(0);
                    state.status = format!("imported asset {}", state.bg_asset);
                    if !import_diags.is_empty() {
                        state.status += &format!("; diagnostics: {}", import_diags.join("; "));
                    }
                }
                Err(e) => state.status = e.to_string(),
            }
        }
        Err(e) => state.status = e,
    }
}

fn bottom_panel(ctx: &egui::Context, state: &mut EditorState) {
    egui::TopBottomPanel::bottom("timeline").min_height(110.0).show(ctx, |ui| {
        let Some(scene_id) = state.scene else {
            ui.label("no scene — create one via \"New scene\"");
            return;
        };
        let range = state.scene().map(|s| s.range).unwrap();
        ui.horizontal(|ui| {
            if ui.button("⏮").clicked() {
                state.frame = range.start;
                state.playing = false;
            }
            let play_label = if state.playing { "⏸ Pause" } else { "▶ Play" };
            if ui.button(play_label).clicked() {
                state.playing = !state.playing;
                state.play_accum = 0.0;
            }
            if ui.button("✚ Key").clicked() {
                if let (Some(scene_id), Some(pid)) = (state.scene, state.selected) {
                    let scene = state.scene().unwrap();
                    let pose = evaluate(&scene.phantoms[&pid], state.frame);
                    let key = MotionKey { frame: state.frame, position: pose.root, yaw: pose.yaw, pose: pose.pose };
                    state.apply(DocCommand::SetMotionKey { scene: scene_id, phantom: pid, key });
                }
            }
            ui.separator();
            // Drawing tools.
            if ui.selectable_label(state.draw_mode, "✏ Draw").clicked() {
                state.draw_mode = !state.draw_mode;
                state.stroke_in_progress = None;
            }
            if state.draw_mode {
                ui.selectable_label(!state.draw_erase, "brush").clicked().then(|| state.draw_erase = false);
                ui.selectable_label(state.draw_erase, "eraser").clicked().then(|| state.draw_erase = true);
                let c = egui::Rgba::from_rgba_unmultiplied(
                    state.draw_color[0], state.draw_color[1], state.draw_color[2], state.draw_color[3],
                );
                if ui.color_edit_button_rgba_unmultiplied(&mut c.to_array()).changed() {
                    state.draw_color = c.to_array();
                }
                ui.add(egui::Slider::new(&mut state.draw_width, 1.0..=40.0).text("px"));
                ui.checkbox(&mut state.onion_skin, "onion skin");
                if ui.button("Remove exposure").clicked() {
                    if let Some(layer) = state.scene().and_then(|s| s.layers_2d.first()) {
                        let layer_id = layer.id;
                        if let Some((f, _)) = layer.exposure_for_frame(state.frame) {
                            let f = *f;
                            state.apply(DocCommand::RemoveExposure { scene: scene_id, layer: layer_id, frame: f });
                        }
                    }
                }
            }
            ui.separator();
            ui.label(format!("frame {} / [{}, {})", state.frame, range.start, range.end));
        });

        // Timeline ruler with the selected Phantom's keys.
        let height = 46.0;
        let (rect, response) =
            ui.allocate_exact_size(egui::vec2(ui.available_width(), height), egui::Sense::click_and_drag());
        if response.dragged() || response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                let t = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                state.frame = range.start + (t * range.len() as f32) as Frame;
                state.frame = state.frame.min(range.end - 1);
            }
        }
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 2.0, egui::Color32::from_gray(30));
        // Ticks every 12 frames.
        let span = range.len().max(1) as f32;
        let step = if span > 240.0 { 24 } else { 12 };
        let mut f = range.start;
        while f < range.end {
            let x = rect.left() + rect.width() * (f - range.start) as f32 / span;
            painter.line_segment(
                [egui::pos2(x, rect.bottom() - 8.0), egui::pos2(x, rect.bottom())],
                egui::Stroke::new(1.0_f32, egui::Color32::GRAY),
            );
            painter.text(
                egui::pos2(x + 2.0, rect.top() + 2.0),
                egui::Align2::LEFT_TOP,
                f.to_string(),
                egui::FontId::proportional(9.0),
                egui::Color32::GRAY,
            );
            f += step;
        }
        // Edit the timeline state, not the humanoid live controller: birds retain height.
        if let (Some(scene_id), Some(pid)) = (state.scene, state.selected) {
            if let Some(scene) = state.scene() {
                if let Some(phantom) = scene.phantoms.get(&pid) {
                    let pose = evaluate(phantom, state.frame);
                    let mut key = MotionKey { frame: state.frame, position: pose.root, yaw: pose.yaw, pose: pose.pose };
                    let mut changed = false;
                    ui.horizontal(|ui| {
                        ui.label("Key: XYZ (m), yaw (rad)");
                        for value in [&mut key.position.x, &mut key.position.y, &mut key.position.z, &mut key.yaw] {
                            changed |= ui.add(egui::DragValue::new(value).speed(0.01)).changed();
                        }
                        // Poses are filtered by the actor profile; an imported rig
                        // has none and renders in bind pose.
                        if let Some(actor) = phantom.rig.builtin_actor() {
                            egui::ComboBox::from_id_salt("key_pose")
                                .selected_text(crate::builtin::pose_name(key.pose))
                                .show_ui(ui, |ui| {
                                    for &pose in actor.poses() {
                                        changed |= ui
                                            .selectable_value(&mut key.pose, pose, crate::builtin::pose_name(pose))
                                            .changed();
                                    }
                                });
                        } else {
                            ui.label("(bind pose)");
                        }
                    });
                    if changed {
                        state.apply(DocCommand::SetMotionKey { scene: scene_id, phantom: pid, key });
                    }
                }
            }
        }

        // Keys of the selected Phantom — diamonds.
        if let (Some(scene), Some(pid)) = (state.scene(), state.selected) {
            if let Some(p) = scene.phantoms.get(&pid) {
                for key in &p.motion {
                    let x = rect.left() + rect.width() * (key.frame - range.start) as f32 / span;
                    let y = rect.center().y + 10.0;
                    painter.add(egui::Shape::convex_polygon(
                        vec![
                            egui::pos2(x, y - 5.0),
                            egui::pos2(x + 4.0, y),
                            egui::pos2(x, y + 5.0),
                            egui::pos2(x - 4.0, y),
                        ],
                        egui::Color32::from_rgb(240, 200, 80),
                        egui::Stroke::NONE,
                    ));
                }
            }
        }
        // 2D layer exposures — blue ticks at the top.
        if let Some(layer) = state.scene().and_then(|s| s.layers_2d.first()) {
            for f in layer.exposures.keys() {
                let x = rect.left() + rect.width() * (*f - range.start) as f32 / span;
                painter.line_segment(
                    [egui::pos2(x, rect.top()), egui::pos2(x, rect.top() + 10.0)],
                    egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(90, 160, 250)),
                );
            }
        }
        // Playhead.
        let x = rect.left() + rect.width() * (state.frame - range.start) as f32 / span;
        painter.line_segment(
            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
            egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(240, 80, 80)),
        );
        let _ = scene_id;
    });
}

fn viewport_panel(ctx: &egui::Context, state: &mut EditorState, tex: egui::TextureId) {
    egui::CentralPanel::default().show(ctx, |ui| {
        let available = ui.available_size();
        let scale = (available.x / VIEWPORT_W as f32).min(available.y / VIEWPORT_H as f32);
        let size = egui::vec2(VIEWPORT_W as f32 * scale, VIEWPORT_H as f32 * scale);
        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
        ui.painter().image(
            tex,
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );

        // Drawing: drag collects a stroke, release issues the command.
        if state.draw_mode {
            let canvas = state
                .scene()
                .and_then(|s| s.layers_2d.first().map(|l| l.canvas))
                .unwrap_or(DEFAULT_CANVAS);
            let to_canvas = |p: egui::Pos2| {
                [
                    (p.x - rect.left()) / rect.width() * canvas.0 as f32,
                    (p.y - rect.top()) / rect.height() * canvas.1 as f32,
                ]
            };
            let to_screen = |pt: [f32; 2]| {
                egui::pos2(
                    rect.left() + pt[0] / canvas.0 as f32 * rect.width(),
                    rect.top() + pt[1] / canvas.1 as f32 * rect.height(),
                )
            };
            if response.drag_started() {
                state.stroke_in_progress = response.interact_pointer_pos().map(|p| vec![to_canvas(p)]);
            }
            if response.dragged() {
                if let (Some(pts), Some(pos)) = (&mut state.stroke_in_progress, response.interact_pointer_pos()) {
                    let pt = to_canvas(pos);
                    if let Some(last) = pts.last() {
                        let d = ((pt[0] - last[0]).powi(2) + (pt[1] - last[1]).powi(2)).sqrt();
                        if d > 1.5 {
                            pts.push(pt);
                        }
                    }
                }
            }
            // Stroke preview on top of the viewport.
            if let Some(pts) = &state.stroke_in_progress {
                if pts.len() > 1 {
                    let screen: Vec<egui::Pos2> = pts.iter().map(|p| to_screen(*p)).collect();
                    let zoom = rect.width() / canvas.0 as f32;
                    let color = if state.draw_erase {
                        egui::Color32::from_rgba_unmultiplied(255, 255, 255, 140)
                    } else {
                        egui::Color32::from_rgba_unmultiplied(
                            (state.draw_color[0] * 255.0) as u8,
                            (state.draw_color[1] * 255.0) as u8,
                            (state.draw_color[2] * 255.0) as u8,
                            (state.draw_color[3] * 255.0) as u8,
                        )
                    };
                    ui.painter().add(egui::Shape::line(
                        screen,
                        egui::Stroke::new((state.draw_width * zoom).max(1.0), color),
                    ));
                }
            }
            if response.drag_stopped() {
                if let Some(pts) = state.stroke_in_progress.take() {
                    if pts.len() >= 2 {
                        if let Some(scene_id) = state.scene {
                            state.apply(DocCommand::BrushStroke {
                                scene: scene_id,
                                layer: None,
                                frame: state.frame,
                                stroke: phantom_core::Stroke {
                                    points: pts,
                                    color: state.draw_color,
                                    width: state.draw_width,
                                    erase: state.draw_erase,
                                },
                            });
                        }
                    }
                }
            }
        }

        // Hint and status.
        let hint = format!("frame {} — the timeline places actors and the camera", state.frame);
        let color = egui::Color32::from_gray(200);
        ui.painter().text(
            rect.left_bottom() + egui::vec2(8.0, -8.0),
            egui::Align2::LEFT_BOTTOM,
            hint,
            egui::FontId::proportional(13.0),
            color,
        );
        if !state.status.is_empty() {
            ui.painter().text(
                rect.left_top() + egui::vec2(8.0, 8.0),
                egui::Align2::LEFT_TOP,
                state.status.clone(),
                egui::FontId::proportional(13.0),
                egui::Color32::from_rgb(255, 200, 80),
            );
        }
    });
}

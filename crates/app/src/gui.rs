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
    app.init_resource::<phantom_runtime::render::ClipCache>();

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
    app.insert_non_send_resource(FrameInput::default());
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
            if state.recording.is_some()
                && !matches!(req.op, crate::session::Op::StateGet | crate::session::Op::JobStatus { .. })
            {
                Err(crate::session::SessionError::BridgeBusy)
            } else {
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

struct Recording {
    phantom: PhantomId,
    /// First frame not yet recorded.
    next_frame: Frame,
    keys: Vec<MotionKey>,
}

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
    on_twos: bool,
    play_accum: f64,

    control: ControlState,
    control_active: bool,
    recording: Option<Recording>,

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
    /// Active shot: its camera is used in the viewport and in scene export;
    /// None — the scene's legacy camera.
    active_shot: Option<ShotId>,
    cam_pos: [f32; 3],
    cam_yaw: f32,
    cam_pitch: f32,
    cam_focal: f32,
    // Stage lighting and floor of the active scene (editable copies).
    lighting: SceneLighting,
    floor_on: bool,
    floor_color: [f32; 3],
    import_path: String,
    bg_asset: u64,
    export_out: String,
    film_out: String,
    trim_edit: Option<(ClipId, Frame, Frame)>,
    clip_track: u64,
    clip_asset: u64,
    clip_start: Frame,
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
    // Clip track of the selected Phantom: fields for a new use.
    cu_asset: u64,
    cu_stack: String,
    cu_start: Frame,
    cu_end: Frame,
    cu_speed: f32,
    cu_blend_in: f32,
    cu_blend_out: f32,
    cu_root_clip: bool,
    // Retargeting profile for the new use: 0 = none, 1 = humanoid, 2 = quadruped.
    cu_retarget: usize,
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
            on_twos: false,
            play_accum: 0.0,
            control: ControlState::default(),
            control_active: false,
            recording: None,
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
            active_shot: None,
            cam_pos: [0.0, 1.4, 4.0],
            cam_yaw: 0.0,
            cam_pitch: 0.0,
            cam_focal: 35.0,
            lighting: SceneLighting::default(),
            floor_on: false,
            floor_color: SceneFloor::default().color,
            import_path: String::new(),
            bg_asset: 0,
            export_out: "render_out".into(),
            film_out: "film.mp4".into(),
            trim_edit: None,
            clip_track: 0,
            clip_asset: 0,
            clip_start: 0,
            bg_is_video: false,
            new_actor: phantom_core::project::BuiltinActor::Humanoid,
            new_phantom_asset: 0,
            selected_object: None,
            new_object_asset: 0,
            obj_pos: [0.0, 0.0, 0.0],
            obj_yaw: 0.0,
            obj_scale: [1.0, 1.0, 1.0],
            cu_asset: 0,
            cu_stack: String::new(),
            cu_start: 0,
            cu_end: 48,
            cu_speed: 1.0,
            cu_blend_in: 0.0,
            cu_blend_out: 0.0,
            cu_root_clip: false,
            cu_retarget: 0,
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
        self.active_shot = None;
        self.recording = None;
        self.playing = false;
        let snapshot = self.scene().map(|scene| {
            (
                scene.range,
                scene.camera.keys.first().copied(),
                scene.camera.focal_mm,
                scene.phantoms.keys().next().copied(),
                scene.phantoms.values().next().and_then(|p| p.motion.first().map(|m| (m.position, m.yaw))),
                scene.background.as_ref().map(|b| b.image.asset().0),
                scene.lighting,
                scene.floor,
            )
        });
        if let Some((range, cam_key, focal, first_phantom, first_pos, bg, lighting, floor)) = snapshot {
            self.lighting = lighting;
            self.floor_on = floor.is_some();
            self.floor_color = floor.map(|f| f.color).unwrap_or_else(|| SceneFloor::default().color);
            self.frame = range.start;
            let k = cam_key.unwrap_or(TransformKey::at(0, glam::Vec3::new(0.0, 1.4, 4.0), 0.0, 0.0));
            self.cam_pos = k.translation.to_array();
            self.cam_yaw = k.yaw;
            self.cam_pitch = k.pitch;
            self.cam_focal = focal;
            self.selected = first_phantom;
            if let Some(start) = first_pos {
                self.control.position = start.0;
                self.control.yaw = start.1;
            }
            self.bg_asset = bg.unwrap_or(0);
        }
        // Structure changed — LiveSync rebuilds by scene id.
    }

    /// Viewport's active shot: its camera is loaded into the panel fields;
    /// None — the scene's legacy camera. The frame is clamped to the shot range.
    fn set_active_shot(&mut self, shot: Option<ShotId>) {
        self.active_shot = shot;
        let Some(scene) = self.scene() else { return };
        let cam = scene.camera_for_shot(shot);
        let key = cam.keys.first().map(|k| (k.translation.to_array(), k.yaw, k.pitch));
        let focal = cam.focal_mm;
        let range = scene.range_for_shot(shot);
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
// Per-frame scene update (input, recording, world sync)
// ---------------------------------------------------------------------------

/// Keyboard input gathered by the UI system during the previous frame.
#[derive(Default)]
struct FrameInput {
    move_x: f32,
    move_z: f32,
    jump: bool,
    viewport_focused: bool,
}

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
    let input = world.remove_non_send_resource::<FrameInput>().unwrap_or_default();
    let Some(mut state) = world.remove_non_send_resource::<EditorState>() else { return };
    {
            let Some(scene_id) = state.scene else {
                world.insert_non_send_resource(input);
                world.insert_non_send_resource(state);
                return;
            };
            let range = match state.scene() {
                Some(s) => s.range,
                None => return,
            };

            // Timeline advance.
            let advancing = state.playing || state.recording.is_some();
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
                        if state.recording.is_some() {
                            stop_recording(&mut state);
                        }
                    }
                }
            }

            // Live control of the selected Phantom: always while recording,
            // otherwise when the viewport is focused (off in draw mode).
            // The live controller emits humanoid Walk/Jump only; other rigs
            // remain timeline-driven, without incompatible joints or actions.
            let live_humanoid = state.selected
                .and_then(|id| state.scene()?.phantoms.get(&id))
                .and_then(|p| p.rig.builtin_actor()) == Some(phantom_core::project::BuiltinActor::Humanoid);
            let controlling = live_humanoid
                && !state.draw_mode
                && (state.recording.is_some() || (input.viewport_focused && !state.playing));
            state.control_active = controlling;
            if controlling {
                let mut acc = dt;
                let step = 1.0 / CONTROL_HZ as f64;
                while acc > 0.0 {
                    let h = acc.min(step) as f32;
                    step_control(
                        &mut state.control,
                        &ControlInput { move_x: input.move_x, move_z: input.move_z, jump: input.jump },
                        h,
                    );
                    acc -= h as f64;
                }
                // Recording: capture every frame passed.
                if let Some(rec) = &mut state.recording {
                    while rec.next_frame <= state.frame {
                        rec.keys.push(MotionKey {
                            yaw: state.control.yaw, frame: rec.next_frame,
                            position: state.control.position,
                            action: state.control.action,
                        });
                        rec.next_frame += 1;
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
                    world.insert_non_send_resource(input);
                    world.insert_non_send_resource(state);
                    return;
                }
            }
            let control_override = if controlling {
                Some((state.selected.unwrap(), state.control.position, state.control.yaw, control_pose(&state.control)))
            } else {
                None
            };
            {
                if let Some(scene) = state.session.project.as_ref().and_then(|p| p.scenes.get(&revision_scene)) {
                    state.live.apply(world, scene, state.frame, state.on_twos, control_override, state.onion_skin, state.active_shot);
                }
            }
    }
    world.insert_non_send_resource(input);
    world.insert_non_send_resource(state);
}

fn start_recording(state: &mut EditorState) {
    let Some(phantom) = state.selected else {
        state.status = "select a Phantom to record".into();
        return;
    };
    if state.scene().and_then(|s| s.phantoms.get(&phantom)).and_then(|p| p.rig.builtin_actor())
        != Some(phantom_core::project::BuiltinActor::Humanoid)
    {
        state.status = "Live recording supports only the builtin humanoid; use keys for other rigs".into();
        return;
    }
    if let Some(scene) = state.scene() {
        let pose = evaluate(&scene.phantoms[&phantom], &scene.range, state.frame, state.on_twos);
        state.control = ControlState { position: pose.root, yaw: pose.yaw, ..Default::default() };
    }
    state.recording = Some(Recording { phantom, next_frame: state.frame, keys: Vec::new() });
    state.play_accum = 0.0;
}

fn stop_recording(state: &mut EditorState) {
    let Some(rec) = state.recording.take() else { return };
    if rec.keys.is_empty() {
        return;
    }
    let Some(scene_id) = state.scene else { return };
    // Re-recording — a new take; the previous one is not destroyed. One
    // command — one undo step; the take becomes active.
    let start = rec.keys.first().unwrap().frame;
    let end = rec.keys.last().unwrap().frame + 1;
    let n = state
        .scene()
        .and_then(|s| s.phantoms.get(&rec.phantom))
        .map(|p| p.takes.len() + 1)
        .unwrap_or(1);
    let Some(range) = FrameRange::new(start, end).ok() else { return };
    state.apply(DocCommand::AddTake {
        scene: scene_id,
        phantom: rec.phantom,
        name: format!("Take {n}"),
        range,
        motion: rec.keys,
    });
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

    let mut input = FrameInput::default();
    if let Some(mut state) = world.remove_non_send_resource::<EditorState>() {
        top_panel(&ctx, &mut state);
        right_panel(&ctx, &mut state);
        bottom_panel(&ctx, &mut state);
        viewport_panel(&ctx, &mut state, viewport_tex, &mut input);
        world.insert_non_send_resource(state);
    }
    world.insert_non_send_resource(input);
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
    if let Some(shot) = state.active_shot {
        args.push("--shot".into());
        args.push(shot.to_string().into());
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
            if let (Some(scene), Some(pid)) = (state.scene(), select) {
                let start =
                    scene.phantoms[&pid].motion.first().map(|m| (m.position, m.yaw)).unwrap_or_default();
                state.control = ControlState { position: start.0, yaw: start.1, ..Default::default() };
            }
        }
        // Takes of the selected Phantom.
        if let Some(pid) = state.selected {
            let takes: Vec<(phantom_core::project::TakeId, String, bool)> = state
                .scene()
                .and_then(|s| s.phantoms.get(&pid))
                .map(|p| {
                    p.takes
                        .iter()
                        .map(|t| (t.id, format!("{} [{}, {})", t.name, t.range.start, t.range.end), p.active_take == Some(t.id)))
                        .collect()
                })
                .unwrap_or_default();
            if !takes.is_empty() {
                ui.label("Takes:");
                for (tid, label, active) in takes {
                    ui.horizontal(|ui| {
                        if ui.radio(active, "").clicked() && !active {
                            if let Some(sid) = state.scene {
                                state.apply(DocCommand::SetActiveTake { scene: sid, phantom: pid, take: Some(tid) });
                            }
                        }
                        ui.label(label);
                        if ui.small_button("✕").clicked() {
                            if let Some(sid) = state.scene {
                                state.apply(DocCommand::RemoveTake { scene: sid, phantom: pid, take: tid });
                            }
                        }
                    });
                }
            }
        }
        // Clip track of the selected Phantom (imported rig only).
        if let Some(pid) = state.selected {
            let imported = state
                .scene()
                .and_then(|s| s.phantoms.get(&pid))
                .is_some_and(|p| matches!(p.rig, RigRef::Imported { .. }));
            if imported {
                let uses: Vec<(phantom_core::ClipUseId, String)> = state
                    .scene()
                    .and_then(|s| s.phantoms.get(&pid))
                    .map(|p| {
                        p.clip_track
                            .iter()
                            .map(|u| {
                                (
                                    u.id,
                                    format!(
                                        "clip {} {} [{}, {}) ×{:.2}{}",
                                        u.asset,
                                        u.stack.as_deref().unwrap_or("(longest)"),
                                        u.range.start,
                                        u.range.end,
                                        u.speed,
                                        u.retarget.map(|r| format!(" ↺{}", r.name())).unwrap_or_default(),
                                    ),
                                )
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let legacy = state.scene().and_then(|s| s.phantoms.get(&pid)).and_then(|p| p.clip);
                ui.collapsing("Clip track", |ui| {
                    if let Some(a) = legacy {
                        ui.label(format!("legacy clip {} (looped)", a));
                    }
                    let mut remove = None;
                    for (id, label) in uses {
                        ui.horizontal(|ui| {
                            ui.label(label);
                            if ui.small_button("✕").clicked() {
                                remove = Some(id);
                            }
                        });
                    }
                    if let (Some(sid), Some(id)) = (state.scene, remove) {
                        state.apply(DocCommand::RemoveClipUse { scene: sid, phantom: pid, clip_use: id });
                    }
                    let motion_assets: Vec<(u64, String)> = state
                        .project()
                        .assets
                        .values()
                        .filter(|a| a.kind == phantom_core::AssetKind::MotionFbx)
                        .map(|a| (a.id.0, format!("{}: {}", a.id, a.path.display())))
                        .collect();
                    if !motion_assets.is_empty() {
                        if !motion_assets.iter().any(|(id, _)| *id == state.cu_asset) {
                            state.cu_asset = motion_assets[0].0;
                        }
                        ui.horizontal(|ui| {
                            egui::ComboBox::from_id_salt("cu_asset")
                                .selected_text(
                                    motion_assets.iter().find(|(id, _)| *id == state.cu_asset).map(|(_, l)| l.clone()).unwrap_or_default(),
                                )
                                .show_ui(ui, |ui| {
                                    for (id, label) in &motion_assets {
                                        ui.selectable_value(&mut state.cu_asset, *id, label);
                                    }
                                });
                            ui.label("stack:");
                            ui.add(egui::TextEdit::singleline(&mut state.cu_stack).desired_width(90.0).hint_text("(longest)"));
                        });
                        ui.horizontal(|ui| {
                            ui.label("frames");
                            ui.add(egui::DragValue::new(&mut state.cu_start));
                            ui.add(egui::DragValue::new(&mut state.cu_end));
                            ui.label("×");
                            ui.add(egui::DragValue::new(&mut state.cu_speed).speed(0.05));
                        });
                        ui.horizontal(|ui| {
                            ui.label("in/out, s");
                            ui.add(egui::DragValue::new(&mut state.cu_blend_in).speed(0.05));
                            ui.add(egui::DragValue::new(&mut state.cu_blend_out).speed(0.05));
                            ui.checkbox(&mut state.cu_root_clip, "clip root motion");
                            ui.label("retargeting:");
                            egui::ComboBox::from_id_salt("cu_retarget")
                                .selected_text(["none", "humanoid", "quadruped"][state.cu_retarget])
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut state.cu_retarget, 0, "none");
                                    ui.selectable_value(&mut state.cu_retarget, 1, "humanoid");
                                    ui.selectable_value(&mut state.cu_retarget, 2, "quadruped");
                                });
                        });
                        if ui.button("Add clip").clicked() {
                            if let (Some(sid), Ok(range)) = (state.scene, FrameRange::new(state.cu_start, state.cu_end)) {
                                let stack = if state.cu_stack.trim().is_empty() { None } else { Some(state.cu_stack.trim().to_string()) };
                                let clip_use = phantom_core::ClipUse {
                                    id: phantom_core::ClipUseId(0), // assigned by the command
                                    asset: AssetId(state.cu_asset),
                                    stack,
                                    range,
                                    speed: state.cu_speed,
                                    blend_in: state.cu_blend_in,
                                    blend_out: state.cu_blend_out,
                                    root_motion: if state.cu_root_clip {
                                        phantom_core::RootMotion::Clip
                                    } else {
                                        phantom_core::RootMotion::Controller
                                    },
                                    retarget: match state.cu_retarget {
                                        1 => Some(phantom_core::RetargetProfile::Humanoid),
                                        2 => Some(phantom_core::RetargetProfile::Quadruped),
                                        _ => None,
                                    },
                                };
                                state.apply(DocCommand::AddClipUse { scene: sid, phantom: pid, clip_use });
                            } else {
                                state.status = "invalid frame range".into();
                            }
                        }
                    } else {
                        ui.label("no MotionFbx assets");
                    }
                });
            }
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
                    let t = phantom_core::evaluate_object(&scene.objects[&oid], &scene.range, state.frame, false);
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
        ui.heading("Shots");
        if let Some(scene_id) = state.scene {
            let shots: Vec<(ShotId, String)> = state
                .scene()
                .map(|s| s.shots.values().map(|sh| (sh.id, sh.name.clone())).collect())
                .unwrap_or_default();
            let mut select_shot: Option<Option<ShotId>> = None;
            let mut remove_shot = None;
            let selected_name = state
                .active_shot
                .and_then(|id| shots.iter().find(|(sid, _)| *sid == id).map(|(_, n)| n.clone()))
                .unwrap_or_else(|| "— (scene camera)".into());
            egui::ComboBox::from_label("Active shot").selected_text(selected_name).show_ui(ui, |ui| {
                if ui.selectable_label(state.active_shot.is_none(), "— (scene camera)").clicked() {
                    select_shot = Some(None);
                }
                for (id, name) in &shots {
                    if ui.selectable_label(state.active_shot == Some(*id), name).clicked() {
                        select_shot = Some(Some(*id));
                    }
                }
            });
            if let Some(sel) = select_shot {
                state.set_active_shot(sel);
            }
            // Active shot range: edited through SetShotRange.
            if let Some(active) = state.active_shot {
                let cur = state.scene().and_then(|s| s.shots.get(&active)).map(|sh| (sh.range.start, sh.range.end));
                if let Some((mut start, mut end)) = cur {
                    ui.horizontal(|ui| {
                        ui.label("Shot range");
                        let rs = ui.add(egui::DragValue::new(&mut start));
                        let re = ui.add(egui::DragValue::new(&mut end));
                        if rs.changed() || re.changed() {
                            match FrameRange::new(start, end) {
                                Ok(range) => state.apply(DocCommand::SetShotRange { scene: scene_id, shot: active, range }),
                                Err(e) => state.status = e.to_string(),
                            }
                        }
                    });
                }
            }
            ui.horizontal(|ui| {
                if ui.button("Add shot").clicked() {
                    if let Some(scene) = state.scene() {
                        let (range, n) = (scene.range, scene.shots.len() + 1);
                        match state.apply_full(DocCommand::AddShot { scene: scene_id, name: format!("Shot {n}"), range }) {
                            Ok(o) => {
                                state.set_active_shot(o.shot);
                                state.status.clear();
                            }
                            Err(e) => state.status = e.to_string(),
                        }
                    }
                }
                if let Some(active) = state.active_shot {
                    if ui.button("Remove shot").clicked() {
                        remove_shot = Some(active);
                    }
                }
            });
            if let Some(id) = remove_shot {
                state.apply(DocCommand::RemoveShot { scene: scene_id, shot: id });
                if state.status.is_empty() {
                    state.active_shot = None;
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
                let key = TransformKey::at(
                    state.frame,
                    glam::Vec3::from_array(state.cam_pos),
                    state.cam_yaw,
                    state.cam_pitch,
                );
                match state.active_shot {
                    Some(shot) => {
                        state.apply(DocCommand::SetShotCameraKey { scene: scene_id, shot, key });
                        state.apply(DocCommand::SetShotFocal { scene: scene_id, shot, focal_mm: state.cam_focal });
                    }
                    None => {
                        let mut keys = state.scene().map(|s| s.camera.keys.clone()).unwrap_or_default();
                        match keys.binary_search_by_key(&key.frame, |k| k.frame) {
                            Ok(i) => keys[i] = key,
                            Err(i) => keys.insert(i, key),
                        }
                        state.apply(DocCommand::SetCameraKeys { scene: scene_id, keys });
                        state.apply(DocCommand::SetCameraFocal { scene: scene_id, focal_mm: state.cam_focal });
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
            ui.add(egui::DragValue::new(&mut state.clip_track));
            ui.label("asset");
            ui.add(egui::DragValue::new(&mut state.clip_asset));
            ui.label("frame");
            ui.add(egui::DragValue::new(&mut state.clip_start));
            if ui.button("Add").clicked() {
                // Length is the whole asset: only the decoder knows the exact
                // length, so take a covering value — the mixer trims to reality.
                state.apply(DocCommand::AddAudioClip {
                    track: TrackId(state.clip_track),
                    asset: AssetId(state.clip_asset),
                    start: state.clip_start,
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
                let shot_name = clip.shot.and_then(|sid| {
                    state.project().scenes.get(&clip.scene).and_then(|s| s.shots.get(&sid)).map(|sh| sh.name.clone())
                });
                let shot_suffix = shot_name.map(|n| format!(" shot \"{n}\"")).unwrap_or_default();
                ui.label(format!("{}: {}{} [{}, {})", i, scene_name, shot_suffix, clip.range.start, clip.range.end));
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
                let (id, range) = (scene.id, scene.range_for_shot(state.active_shot));
                state.apply(DocCommand::AddFilmClip { scene: id, range, shot: state.active_shot });
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
                            arc_radians: 2.0,
                            radius: 6.0,
                            height: 4.0,
                            center: glam::Vec3::new(0.0, 2.0, -5.0),
                            bulge: Default::default(),
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
                import_asset(state, None);
            }
            if ui.button("Import motion (FBX)…").clicked() {
                import_asset(state, Some(AssetKind::MotionFbx));
            }
        });
    });
}

/// Import a file from the `import_path` field into the project. `kind_override`
/// sets the registration AssetKind (motion-fbx for motion clips); the import
/// route is unchanged: obj/fbx — bundle import with MTL and textures, anything else as a single file.
fn import_asset(state: &mut EditorState, kind_override: Option<AssetKind>) {
    let path = PathBuf::from(state.import_path.trim());
    if !path.exists() {
        state.status = "file does not exist".into();
        return;
    }
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    if kind_override == Some(AssetKind::MotionFbx) && ext.as_deref() != Some("fbx") {
        state.status = "motion import: expected a .fbx file".into();
        return;
    }
    let is_model = matches!(ext.as_deref(), Some("obj") | Some("fbx"));
    // Models — bundle import with MTL and textures inside the project.
    let mut import_diags: Vec<String> = Vec::new();
    let imported: Result<PathBuf, String> = if is_model {
        match phantom_runtime::import::import_model_bundle(&state.project_dir, &path) {
            Ok(b) => {
                import_diags = b.diagnostics;
                Ok(b.model)
            }
            Err(e) => Err(e.to_string()),
        }
    } else {
        storage::import_file(&state.project_dir, &path).map_err(|e| e.to_string())
    };
    match imported {
        Ok(rel) => {
            let kind = kind_override.unwrap_or(match ext.as_deref() {
                Some("obj") => AssetKind::ModelObj,
                Some("fbx") => AssetKind::ModelFbx,
                _ => AssetKind::Image,
            });
            match state.apply_full(DocCommand::RegisterAsset { kind, path: rel }) {
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
                if state.recording.is_none() {
                    state.playing = !state.playing;
                    state.play_accum = 0.0;
                }
            }
            let rec = state.recording.is_some();
            let rec_label = if rec { "■ Stop" } else { "● Record" };
            if ui
                .add_enabled(state.selected.is_some(), egui::Button::new(rec_label).selected(rec))
                .clicked()
            {
                if rec {
                    stop_recording(state);
                    state.playing = false;
                } else {
                    start_recording(state);
                }
            }
            if ui.button("✚ Key").clicked() {
                if let (Some(scene_id), Some(pid)) = (state.scene, state.selected) {
                    let key = if state.control_active {
                        MotionKey { frame: state.frame, position: state.control.position, yaw: state.control.yaw, action: state.control.action }
                    } else {
                        let scene = state.scene().unwrap();
                        let pose = evaluate(&scene.phantoms[&pid], &scene.range, state.frame, false);
                        MotionKey { frame: state.frame, position: pose.root, yaw: pose.yaw, action: pose.action }
                    };
                    state.apply(DocCommand::SetMotionKey { scene: scene_id, phantom: pid, key });
                }
            }
            ui.checkbox(&mut state.on_twos, "on twos");
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
                    let pose = evaluate(phantom, &scene.range, state.frame, false);
                    let mut key = MotionKey { frame: state.frame, position: pose.root, yaw: pose.yaw, action: pose.action };
                    let mut changed = false;
                    ui.horizontal(|ui| {
                        ui.label("Key: XYZ (m), yaw (rad)");
                        for value in [&mut key.position.x, &mut key.position.y, &mut key.position.z, &mut key.yaw] {
                            changed |= ui.add(egui::DragValue::new(value).speed(0.01)).changed();
                        }
                        // Actions are filtered by the actor profile; imported rigs are clip-driven.
                        if let Some(actor) = phantom.rig.builtin_actor() {
                            egui::ComboBox::from_id_salt("key_action")
                                .selected_text(crate::builtin::action_name(key.action))
                                .show_ui(ui, |ui| {
                                    for &action in actor.actions() {
                                        changed |= ui
                                            .selectable_value(&mut key.action, action, crate::builtin::action_name(action))
                                            .changed();
                                    }
                                });
                        } else {
                            ui.label("(clip)");
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

fn viewport_panel(ctx: &egui::Context, state: &mut EditorState, tex: egui::TextureId, input: &mut FrameInput) {
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

        input.viewport_focused = response.hovered() || response.has_focus();
        let wants_kb = ctx.wants_keyboard_input();
        if input.viewport_focused && !wants_kb {
            let (mx, mz, jump) = ctx.input(|i| {
                let x = (i.key_down(egui::Key::D) || i.key_down(egui::Key::ArrowRight)) as i8
                    - (i.key_down(egui::Key::A) || i.key_down(egui::Key::ArrowLeft)) as i8;
                // The camera looks down -Z: W moves forward (-Z), S toward the camera (+Z).
                let z = (i.key_down(egui::Key::S) || i.key_down(egui::Key::ArrowDown)) as i8
                    - (i.key_down(egui::Key::W) || i.key_down(egui::Key::ArrowUp)) as i8;
                (x as f32, z as f32, i.key_down(egui::Key::Space))
            });
            input.move_x = mx;
            input.move_z = mz;
            input.jump = jump;
        }

        // Hint and status.
        let hint = if state.recording.is_some() {
            format!("● RECORDING frame {}  (WASD — move, Space — jump, ■ — stop)", state.frame)
        } else if state.control_active {
            "WASD — move, Space — jump ● control active".to_string()
        } else {
            "hover the viewport to control the Phantom (WASD, Space)".to_string()
        };
        let color = if state.recording.is_some() {
            egui::Color32::from_rgb(255, 90, 90)
        } else {
            egui::Color32::from_gray(200)
        };
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

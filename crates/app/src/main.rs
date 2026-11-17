//! phantom-animate: CLI over the core/runtime commands.
//! GUI and MCP later call the same command dispatcher.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use glam::Vec3;
use phantom_core::*;
use phantom_core::project::BackgroundSource;
use phantom_runtime::{render::{RenderOptions, Renderer}, storage};

mod analysis;
mod gui;
mod bridge;
mod builtin;
mod mcp;
mod project_lock;
mod session;

#[derive(Parser)]
#[command(name = "phantom-animate", about = "Phantom-animate — staging stylized animation")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Create a project in a directory.
    New { dir: PathBuf, name: String },
    /// Open a project in the editor window (storyboarding).
    Edit { project: PathBuf },
    /// MCP server for AI agents (stdio, no window).
    Mcp {
        /// Bridge to an open window: connect to the socket from the project's lock file.
        #[arg(long)]
        bridge: Option<PathBuf>,
    },
    /// Add a scene.
    AddScene {
        project: PathBuf,
        name: String,
        /// Frame range "start-end" (half-open).
        #[arg(long, default_value = "0-96")]
        frames: String,
    },
    /// Import an asset (obj/fbx/png/jpg).
    /// Import asset(s): obj/fbx (+MTL/textures as a bundle),
    /// png/jpg/tiff, audio, video. Several files in one call (B7).
    Import {
        project: PathBuf,
        files: Vec<PathBuf>,
        /// Explicit asset kind: model-obj|model-fbx|image|audio|video
        /// (by default taken from the file extension; with bundle import it applies
        /// to every file).
        #[arg(long)]
        kind: Option<String>,
    },
    /// Version, build features and limits (D8) — machine-readable analog of
    /// MCP studio_capabilities for the CLI user.
    Capabilities,
    /// Set the background on the flat screen (a rectangle facing the camera
    /// side through `center`). The default 24×8 m stands at z = −5 and covers
    /// the frame for the default camera; MCP background_fit computes the width
    /// and height that close the frame from the cameras actually used.
    SetBackground {
        project: PathBuf,
        #[arg(long)]
        scene: u64,
        #[arg(long)]
        asset: u64,
        /// Background is a video (image by default).
        #[arg(long)]
        video: bool,
        /// Screen width in meters.
        #[arg(long, default_value_t = 24.0)]
        width: f32,
        /// Screen height in meters.
        #[arg(long, default_value_t = 8.0)]
        height: f32,
        /// Screen center "x,y,z" in meters; the screen faces +Z.
        #[arg(long, default_value = "0,4,-5")]
        center: String,
        /// Panorama UV mode: cover (cropped, default), contain (whole
        /// picture, clamp bands), stretch (stretched, pre-A3 behavior).
        #[arg(long, default_value = "cover")]
        fit: String,
    },
    /// Add a Phantom (builtin rig, builtin actor, or imported FBX character).
    AddPhantom {
        project: PathBuf,
        #[arg(long)]
        scene: u64,
        name: String,
        /// "builtin" (legacy humanoid), "humanoid"/"wolf"/"horse"/"bird" or a ModelFbx asset ID.
        #[arg(long, default_value = "builtin")]
        rig: String,
    },
    /// Set a Phantom controller key.
    SetKey {
        project: PathBuf,
        #[arg(long)]
        scene: u64,
        #[arg(long)]
        phantom: u64,
        #[arg(long)]
        frame: i64,
        /// "x,y,z" in meters.
        #[arg(long, default_value = "0,0,0")]
        pos: String,
        /// Root yaw about +Y in radians (zero faces +Z).
        #[arg(long, default_value_t = 0.0)]
        yaw: f32,
        /// Held pose (stand|sit|lie); a species that cannot hold it is an error.
        #[arg(long, default_value = "stand")]
        pose: String,
    },
    /// Simple prop catalog: names, metric dimensions, origin,
    /// material groups (K1). Without --query prints the whole section.
    Catalog {
        /// Section: primitive|furniture|interior.
        #[arg(long)]
        kind: Option<String>,
        /// Substring of the name or description.
        #[arg(long)]
        query: Option<String>,
    },
    /// Place a catalog item (K1): the mesh is built procedurally, no file needed.
    /// A furnished room (a room_dining-like entry) arrives as one package: the
    /// shell plus every piece of its furniture as its own object, one undo step.
    PropAdd {
        project: PathBuf,
        #[arg(long)]
        scene: u64,
        name: String,
        /// Catalog entry name: table, chair, room_kitchen, cylinder, ...
        prop: String,
        /// Position "x,y,z" in meters; without it the prop sits at the origin.
        #[arg(long)]
        pos: Option<String>,
        #[arg(long, default_value_t = 0.0)]
        yaw: f32,
        /// Scale "sx,sy,sz".
        #[arg(long, default_value = "1,1,1")]
        scale: String,
    },
    /// Fill a formation with people in one atomic step (one undo): line, arc,
    /// wedge, scattered or queue. Every person is a real Phantom with one pose
    /// key, so the crowd stays actable — you can move someone afterwards.
    Crowd {
        project: PathBuf,
        #[arg(long)]
        scene: u64,
        /// Formation: line | arc | wedge | scattered | queue.
        formation: String,
        /// How many people (up to 128).
        #[arg(long, default_value_t = 12)]
        count: usize,
        /// Who stands there: mixed (default) | adults | a role name
        /// (adult|child|teen|elder).
        #[arg(long)]
        roles: Option<String>,
        /// Pose held by everyone: stand (default) | sit | lie.
        #[arg(long)]
        pose: Option<String>,
        /// Meters between neighbors; 0.8 by default.
        #[arg(long)]
        spacing: Option<f32>,
        /// Rows for a line/scattered crowd; by default close to square.
        #[arg(long)]
        rows: Option<usize>,
        /// How far a person may look off the formation's facing, degrees.
        #[arg(long)]
        scatter_deg: Option<f32>,
        /// Determinism seed; a random one is drawn and printed when omitted.
        #[arg(long)]
        seed: Option<u64>,
        /// Name prefix; without it — the scene name.
        #[arg(long)]
        prefix: Option<String>,
        /// Where the formation's origin stands "x,y,z".
        #[arg(long)]
        pos: Option<String>,
        /// Which way the formation faces, degrees.
        #[arg(long, default_value_t = 0.0)]
        yaw: f32,
        /// Frame the pose key is set at.
        #[arg(long)]
        frame: Option<u32>,
    },
    /// Fill an area or a line from a preset in one atomic step (one undo):
    /// forest, grove, undergrowth, hedge, street_row, pole_row. Inside
    /// --action the fill places movable props; the ring beyond it becomes one
    /// band object whose mesh replays the same seed.
    Scatter {
        project: PathBuf,
        #[arg(long)]
        scene: u64,
        /// Fill preset: forest | grove | undergrowth | hedge | street_row | pole_row.
        preset: String,
        /// World rectangle "x0,z0,x1,z1" (or "x0,z0:x1,z1") in meters.
        #[arg(long)]
        area: Option<String>,
        /// Object id of a ground plate (street_yard/street_park/street_crossing).
        #[arg(long)]
        plate: Option<u64>,
        /// Keep this margin (m) clear of the plate edge.
        #[arg(long)]
        inset: Option<f32>,
        /// The zone the scene is shot in (editable props): "x0,z0,x1,z1".
        #[arg(long)]
        action: Option<String>,
        /// Props per square meter; the preset's own default by default.
        #[arg(long)]
        density: Option<f32>,
        /// Line presets: segment endpoints "x,z".
        #[arg(long)]
        from: Option<String>,
        #[arg(long)]
        to: Option<String>,
        #[arg(long)]
        step: Option<f32>,
        /// Determinism seed; a random one is drawn and printed when omitted.
        #[arg(long)]
        seed: Option<u64>,
        /// Name prefix; without it — the scene name.
        #[arg(long)]
        prefix: Option<String>,
    },
    /// Place a grid of catalog props in one atomic step (one undo): rows along
    /// +Z, cols along +X. The step defaults to the prop's own size times --gap.
    PropGrid {
        project: PathBuf,
        #[arg(long)]
        scene: u64,
        /// Catalog entry name: chair, stool, box, room_kitchen, ...
        prop: String,
        #[arg(long, default_value_t = 1)]
        rows: u32,
        #[arg(long, default_value_t = 1)]
        cols: u32,
        /// Position of the first object "x,y,z" in meters.
        #[arg(long, default_value = "0,0,0")]
        origin: String,
        /// Explicit spacing "dx,dz" in meters; without it — the prop's size times --gap.
        #[arg(long)]
        step: Option<String>,
        #[arg(long, default_value_t = 1.2)]
        gap: f32,
        /// Name prefix; without it — the prop name.
        #[arg(long)]
        prefix: Option<String>,
        #[arg(long, default_value_t = 0.0)]
        yaw: f32,
        /// Scale "sx,sy,sz".
        #[arg(long, default_value = "1,1,1")]
        scale: String,
    },
    /// Review sheet: several views of the staging, or a strip of frames, tiled
    /// into one PNG. Requires a GPU.
    Look {
        project: PathBuf,
        #[arg(long)]
        scene: u64,
        /// angles (default) | frames.
        #[arg(long, default_value = "angles")]
        mode: String,
        /// Where to write the sheet PNG.
        #[arg(long, default_value = "look.png")]
        out: PathBuf,
        /// Actor/object names to frame (repeatable), or phantom:<id>/object:<id>.
        #[arg(long = "target")]
        targets: Vec<String>,
        /// angles: view names (repeatable): front|back|left|right|top|three_quarter.
        #[arg(long = "view")]
        views: Vec<String>,
        /// Metres from the target to the eye; without it — auto-framed.
        #[arg(long)]
        distance: Option<f32>,
        /// angles: the frame to review.
        #[arg(long)]
        frame: Option<i64>,
        /// frames: camera whose lens and pose to use (id); the active camera by default.
        #[arg(long)]
        camera: Option<u64>,
        /// frames: explicit comma-separated frames.
        #[arg(long)]
        frames: Option<String>,
        /// frames: every Nth frame of the range.
        #[arg(long)]
        every: Option<u32>,
        /// frames: how many evenly spaced frames (default 5, max 12).
        #[arg(long)]
        count: Option<u32>,
        #[arg(long, default_value_t = 480)]
        width: u32,
        #[arg(long, default_value_t = 270)]
        height: u32,
    },
    /// Add a static object (prop) from a model asset (ModelObj/ModelFbx).
    AddObject {
        project: PathBuf,
        #[arg(long)]
        scene: u64,
        name: String,
        #[arg(long)]
        asset: u64,
    },
    /// Set an object (prop) transform key.
    SetObjectKey {
        project: PathBuf,
        #[arg(long)]
        scene: u64,
        #[arg(long)]
        object: u64,
        #[arg(long)]
        frame: i64,
        /// "x,y,z" in meters.
        #[arg(long, default_value = "0,0,0")]
        pos: String,
        /// Yaw about +Y in radians.
        #[arg(long, default_value_t = 0.0)]
        yaw: f32,
        #[arg(long, default_value_t = 0.0)]
        pitch: f32,
        #[arg(long, default_value_t = 0.0)]
        roll: f32,
        /// Scale "sx,sy,sz".
        #[arg(long, default_value = "1,1,1")]
        scale: String,
    },
    /// Configure a camera key at frame 0 (the scene's active camera by default).
    SetCamera {
        project: PathBuf,
        #[arg(long)]
        scene: u64,
        /// Camera id to key; the scene's active camera by default.
        #[arg(long)]
        camera: Option<u64>,
        #[arg(long, default_value = "0,1.4,4")]
        pos: String,
        #[arg(long, default_value_t = 0.0)]
        yaw: f32,
        #[arg(long, default_value_t = 0.0)]
        pitch: f32,
        #[arg(long, default_value_t = 35.0)]
        focal: f32,
        /// Horizon roll in degrees (C1).
        #[arg(long, default_value_t = 0.0)]
        roll: f32,
        /// Aim point "x,y,z": yaw/pitch are derived from it and --yaw/--pitch
        /// are ignored (C1.3).
        #[arg(long)]
        look_at: Option<String>,
        /// Smoothing toward the next key: linear|ease_in|ease_out|ease_in_out.
        #[arg(long, default_value = "linear")]
        ease: String,
    },
    /// Add a named camera (it becomes the scene's active camera). Without
    /// --frames the camera covers the whole scene. Alias: add-camera.
    #[command(visible_alias = "add-camera")]
    CameraAdd {
        project: PathBuf,
        #[arg(long)]
        scene: u64,
        /// Camera name; "Camera N" by default.
        #[arg(long)]
        name: Option<String>,
        /// Frame range "start-end" (half-open); the whole scene by default.
        #[arg(long)]
        frames: Option<String>,
    },
    /// Remove a camera; film clips onto it go away in the same step. Camera 0
    /// cannot be removed.
    CameraRemove {
        project: PathBuf,
        #[arg(long)]
        scene: u64,
        /// Camera id to remove.
        #[arg(long)]
        camera: u64,
    },
    /// Pick the scene's active camera (the viewport's and a render's default).
    CameraSelect {
        project: PathBuf,
        #[arg(long)]
        scene: u64,
        /// Camera id to activate.
        #[arg(long)]
        camera: u64,
    },
    /// Attach a camera to a subject: track (stand and turn to it) or follow
    /// (travel and turn with it). Without --subject the target is cleared.
    CameraTarget {
        project: PathBuf,
        #[arg(long)]
        scene: u64,
        /// Camera id; the scene's active camera by default.
        #[arg(long)]
        camera: Option<u64>,
        /// track (stand and turn) | follow (travel with the subject).
        #[arg(long)]
        mode: Option<String>,
        /// Subject "phantom:<id>" or "object:<id>".
        #[arg(long)]
        subject: Option<String>,
    },
    /// List the scene's cameras; the active one is marked with "*".
    CameraList {
        project: PathBuf,
        #[arg(long)]
        scene: u64,
    },
    /// Configure the scene's stage lighting (key/fill/rim).
    SetLighting {
        project: PathBuf,
        #[arg(long)]
        scene: u64,
        /// "r,g,b,lux[,on|off]" — only the given light sources change.
        #[arg(long)]
        key: Option<String>,
        #[arg(long)]
        fill: Option<String>,
        #[arg(long)]
        rim: Option<String>,
    },
    /// Enable/remove the scene's matte floor (y=0, receives shadows).
    SetFloor {
        project: PathBuf,
        #[arg(long)]
        scene: u64,
        /// Floor color "r,g,b" (linear RGB); neutral by default.
        #[arg(long)]
        color: Option<String>,
        /// Remove the floor.
        #[arg(long)]
        off: bool,
    },
    /// Set the scene stage (layout mode): the box actors and props are confined
    /// to. Anything already staged that no longer fits is clamped in the same
    /// step and reported.
    SetStage {
        project: PathBuf,
        #[arg(long)]
        scene: u64,
        /// Stage preset: small (10x10x4 m) | medium (30x20x8) | large (80x60x20).
        #[arg(long)]
        preset: Option<String>,
        /// Explicit size "width x depth x height" in meters, e.g. 12x8x4
        /// (with a preset it overrides those axes).
        #[arg(long)]
        size: Option<String>,
        /// Center of the box on the floor plane "x,z"; the current center by default.
        #[arg(long)]
        center: Option<String>,
    },
    /// Print the scene stage.
    Stage {
        project: PathBuf,
        #[arg(long)]
        scene: u64,
    },
    /// Render the film to MP4 (H.264 + AAC 48 kHz stereo audio when the
    /// project has audio tracks; without tracks — silent).
    RenderFilm {
        project: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        frames: Option<String>,
        #[arg(long, default_value_t = 1280)]
        width: u32,
        #[arg(long, default_value_t = 720)]
        height: u32,
        /// Allow overwriting an existing file.
        #[arg(long)]
        force: bool,
        /// Peak-normalize the mix to --target-peak-db (off by default).
        #[arg(long)]
        normalize: bool,
        /// Target peak dBFS with --normalize.
        #[arg(long, default_value_t = -1.0)]
        target_peak_db: f32,
        /// Contact sheet (K0.4): instead of MP4 — a PNG of every --every-th film
        /// frame into the --out directory (camera, as in export).
        #[arg(long)]
        preview: bool,
        /// Frame step for --preview (6 by default).
        #[arg(long, default_value_t = 6)]
        every: u32,
    },
    /// Mix the film audio to WAV without rendering video (mix preview, B6).
    RenderAudio {
        project: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        normalize: bool,
        #[arg(long, default_value_t = -1.0)]
        target_peak_db: f32,
    },
    /// Change an audio clip's fragment/fades (values in 48 kHz samples or
    /// seconds via the *-secs flags; any subset of the fields may be set).
    ClipUpdate {
        project: PathBuf,
        #[arg(long)]
        track: u64,
        #[arg(long)]
        clip: u64,
        #[arg(long)]
        offset: Option<u64>,
        #[arg(long)]
        len: Option<u64>,
        #[arg(long)]
        fade_in: Option<u64>,
        #[arg(long)]
        fade_out: Option<u64>,
        #[arg(long)]
        offset_secs: Option<f64>,
        #[arg(long)]
        len_secs: Option<f64>,
        #[arg(long)]
        fade_in_secs: Option<f64>,
        #[arg(long)]
        fade_out_secs: Option<f64>,
    },
    /// Add a clip to the film (at the end): a scene range or a camera clip
    /// (--camera; without --frames the camera's range is used).
    FilmAdd {
        project: PathBuf,
        #[arg(long)]
        scene: u64,
        /// Frame range "start-end" (half-open); the camera's range by default.
        #[arg(long)]
        frames: Option<String>,
        /// Camera whose range and lens the clip uses.
        #[arg(long)]
        camera: Option<u64>,
    },
    /// Add an audio track to a scene (or to the film without --scene).
    TrackAdd { project: PathBuf, name: String, #[arg(long)] scene: Option<u64> },
    /// Add an audio clip to a track (48 kHz samples; offset/len/fade are also
    /// accepted in seconds via the *-secs flags).
    ClipAdd {
        project: PathBuf,
        #[arg(long)]
        track: u64,
        #[arg(long)]
        asset: u64,
        #[arg(long, default_value_t = 0)]
        start: i64,
        #[arg(long, default_value_t = 0)]
        offset: u64,
        #[arg(long)]
        len: Option<u64>,
        #[arg(long)]
        offset_secs: Option<f64>,
        #[arg(long)]
        len_secs: Option<f64>,
        /// Fade-in in 48 kHz samples (or --fade-in-secs).
        #[arg(long)]
        fade_in: Option<u64>,
        /// Fade-out in 48 kHz samples (or --fade-out-secs).
        #[arg(long)]
        fade_out: Option<u64>,
        #[arg(long)]
        fade_in_secs: Option<f64>,
        #[arg(long)]
        fade_out_secs: Option<f64>,
    },
    /// Render a PNG sequence.
    Render {
        project: PathBuf,
        #[arg(long)]
        scene: u64,
        #[arg(long)]
        out: PathBuf,
        /// Camera to render with (id); the scene's active camera by default.
        #[arg(long)]
        camera: Option<u64>,
        /// Frame range "start-end" (upper bound EXCLUSIVE:
        /// 0-1 — frame 0 only); by default the whole scene (or camera).
        #[arg(long)]
        frames: Option<String>,
        #[arg(long, default_value_t = 1280)]
        width: u32,
        #[arg(long, default_value_t = 720)]
        height: u32,
        /// Cheap contact sheet (B7): every --every-th frame at 480x270,
        /// when the size is not given explicitly.
        #[arg(long)]
        preview: bool,
        /// Frame step for --preview (6 by default).
        #[arg(long, default_value_t = 6)]
        every: u32,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::New { dir, name } => {
            storage::create_project(&dir, &name)?;
            println!("project \"{name}\" created in {}", dir.display());
        }
        Cmd::Edit { project } => {
            gui::run_editor(project)?;
        }
        Cmd::Mcp { bridge } => {
            let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
            rt.block_on(mcp::serve(bridge))?;
        }
        Cmd::AddScene { project, name, frames } => {
            let range = parse_range(&frames)?;
            with_project(&project, |p, h| {
                h.apply(p, Command::AddScene { name, range })
                    .map(|o| println!("scene {}", o.scene.unwrap()))
                    .map_err(anyhow::Error::from)
            })?;
        }
        Cmd::Capabilities => {
            println!("phantom-animate {}", env!("CARGO_PKG_VERSION"));
            println!("fps: 24 (fixed), audio: 48 kHz stereo");
            println!("import: obj, fbx, png, jpg, tiff, mp3/wav/ogg/flac/m4a, mp4/webm/mov/gif; gltf/glb, webp/exr are not supported yet (see docs/architecture.md)");
            println!("output: png sequence, mp4 (h.264 + aac 48 kHz)");
            println!("bevy: render+pbr+core_pipeline without bloom/vignette/dof (not in 0.17.3), LUT-free Reinhard tonemapping, msaa 4x");
            println!("limits: frame <= 4096x4096, particles per effect <= 1024, outputs — inside the project directory");
        }
        Cmd::Import { project, files, kind } => {
            let forced_kind = match kind.as_deref() {
                Some("model-obj") => Some(AssetKind::ModelObj),
                Some("model-fbx") => Some(AssetKind::ModelFbx),
                Some("image") => Some(AssetKind::Image),
                Some("audio") => Some(AssetKind::Audio),
                Some("video") => Some(AssetKind::Video),
                Some(k) => bail!("unsupported asset kind: {k}"),
                None => None,
            };
            if files.is_empty() {
                bail!("import: no file given");
            }
            let mut failed = 0usize;
            for file in &files {
                // kind is either explicit or derived from EACH file's extension.
                let kind = match forced_kind {
                    Some(k) => k,
                    None => match file.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref() {
                        Some("obj") => AssetKind::ModelObj,
                        Some("fbx") => AssetKind::ModelFbx,
                        Some("png" | "jpg" | "jpeg" | "tiff" | "tif") => AssetKind::Image,
                        Some("mp3" | "wav" | "ogg" | "flac" | "m4a") => AssetKind::Audio,
                        Some("mp4" | "gif" | "webm" | "mov") => AssetKind::Video,
                        Some(e) => {
                            eprintln!("import {}: unsupported extension: {e}", file.display());
                            failed += 1;
                            continue;
                        }
                        None => {
                            eprintln!("import {}: no file extension", file.display());
                            failed += 1;
                            continue;
                        }
                    },
                };
                let result = (|| -> Result<()> {
                    // Models — bundle import with MTL and textures inside the project;
                    let (rel, bounds) = if matches!(kind, AssetKind::ModelObj | AssetKind::ModelFbx) {
                        let bundle = phantom_runtime::import::import_model_bundle(&project, file)?;
                        for d in &bundle.diagnostics {
                            eprintln!("import: {d}");
                        }
                        (bundle.model, bundle.bounds)
                    } else {
                        (storage::import_file(&project, file)?, None)
                    };
                    with_project(&project, |p, h| {
                        h.apply(p, Command::RegisterAsset { kind, path: rel.clone(), bounds })
                            .map(|o| println!("asset {} {}", o.asset.unwrap(), rel.display()))
                            .map_err(anyhow::Error::from)
                    })?;
                    Ok(())
                })();
                if let Err(e) = result {
                    eprintln!("import {}: {e:#}", file.display());
                    failed += 1;
                }
            }
            if failed > 0 {
                bail!("import: {failed}/{} files not imported", files.len());
            }
        }
        Cmd::SetBackground { project, scene, asset, video, width, height, center, fit } => {
            let fit = match fit.as_str() {
                "cover" => phantom_core::project::BackgroundFit::Cover,
                "contain" => phantom_core::project::BackgroundFit::Contain,
                "stretch" => phantom_core::project::BackgroundFit::Stretch,
                other => anyhow::bail!("fit: cover|contain|stretch, got {other}"),
            };
            with_project(&project, |p, h| {
                h.apply(
                    p,
                    Command::SetBackground {
                        scene: SceneId(scene),
                        background: Background {
                            image: if video { BackgroundSource::Video { video: AssetId(asset) } } else { BackgroundSource::Image(AssetId(asset)) },
                            width,
                            height,
                            center: parse_vec3(&center)?,
                            fit,
                        },
                    },
                )?;
                Ok(())
            })?;
        }
        Cmd::AddPhantom { project, scene, name, rig } => {
            let rig = builtin::parse_rig(&rig).map_err(anyhow::Error::msg)?;
            with_project(&project, |p, h| {
                h.apply(p, Command::AddPhantom { scene: SceneId(scene), name, rig })
                    .map(|o| println!("phantom {}", o.phantom.unwrap()))
                    .map_err(anyhow::Error::from)
            })?;
        }
        Cmd::SetKey { project, scene, phantom, frame, pos, pose, yaw } => {
            let pose = builtin::parse_pose(&pose).map_err(anyhow::Error::msg)?;
            with_project(&project, |p, h| {
                h.apply(
                    p,
                    Command::SetMotionKey {
                        scene: SceneId(scene),
                        phantom: PhantomId(phantom),
                        key: MotionKey { yaw, frame, position: parse_vec3(&pos)?, pose },
                    },
                )?;
                Ok(())
            })?;
        }
        Cmd::Catalog { kind, query } => {
            let kind = match kind.as_deref() {
                None => None,
                Some("primitive") => Some(phantom_core::catalog::CatalogKind::Primitive),
                Some("furniture") => Some(phantom_core::catalog::CatalogKind::Furniture),
                Some("interior") => Some(phantom_core::catalog::CatalogKind::Interior),
                Some("animal") => Some(phantom_core::catalog::CatalogKind::Animal),
                Some("street") => Some(phantom_core::catalog::CatalogKind::Street),
                Some("exterior") => Some(phantom_core::catalog::CatalogKind::Exterior),
                other => bail!("kind: primitive|furniture|interior, got {other:?}"),
            };
            let entries = phantom_core::catalog::catalog_query(kind, query.as_deref());
            println!("{:<16} {:<10} {:<22} origin      materials", "name", "section", "dimensions x,y,z m");
            for e in &entries {
                let size = format!("{:.2}x{:.2}x{:.2}", e.size[0], e.size[1], e.size[2]);
                println!("{:<16} {:<10} {:<22} {:<11} {}", e.name, e.kind.as_str(), size, format!("{:?}", e.origin).to_lowercase(), e.materials.join(","));
            }
            println!("total: {}", entries.len());
            println!("human reference lines (m): {}", phantom_core::catalog::HUMAN_ROLES.iter().map(|r| format!("{}: height {:.2}, eye {:.2}", r.role, r.height_m, r.eye_m)).collect::<Vec<_>>().join("; "));
        }
        Cmd::PropAdd { project, scene, name, prop, pos, yaw, scale } => {
            let key = pos
                .as_deref()
                .map(|p| -> Result<ObjectKey> {
                    Ok(ObjectKey { frame: 0, position: parse_vec3(p)?, yaw: yaw.to_radians(), pitch: 0.0, roll: 0.0, scale: parse_vec3(&scale)? })
                })
                .transpose()?;
            with_project(&project, |p, h| {
                h.apply(p, Command::AddBuiltinObject { scene: SceneId(scene), name, prop, key })
                    .map(|o| println!("object {}", o.object.unwrap()))
                    .map_err(anyhow::Error::from)
            })?;
        }
        Cmd::PropGrid { project, scene, prop, rows, cols, origin, step, gap, prefix, yaw, scale } => {
            let origin = parse_vec3(&origin)?;
            let step = match step {
                Some(s) => {
                    let v = parse_vec3(&s)?;
                    Some([v.x, v.z])
                }
                None => None,
            };
            let scale = parse_vec3(&scale)?;
            with_project(&project, |p, h| {
                let entry = phantom_core::catalog::catalog_find(&prop)
                    .ok_or_else(|| anyhow::anyhow!("unknown catalog entry: {prop} (see catalog)"))?;
                if !entry.furniture.is_empty() {
                    return Err(anyhow::anyhow!(
                        "{prop} comes furnished — place it once with prop-add (it arrives as one package); a grid is for single props"
                    ));
                }
                let [dx, dz] = step.unwrap_or([entry.size[0] * scale.x * gap, entry.size[2] * scale.z * gap]);
                let base = prefix.clone().unwrap_or_else(|| prop.clone());
                let mut commands = Vec::new();
                for row in 0..rows {
                    for col in 0..cols {
                        let key = ObjectKey {
                            frame: 0,
                            position: glam::Vec3::new(
                                origin.x + col as f32 * dx,
                                origin.y,
                                origin.z + row as f32 * dz,
                            ),
                            yaw: yaw.to_radians(),
                            pitch: 0.0,
                            roll: 0.0,
                            scale,
                        };
                        commands.push(Command::AddBuiltinObject {
                            scene: SceneId(scene),
                            name: format!("{} {}", base.trim(), commands.len() + 1),
                            prop: prop.clone(),
                            key: Some(key),
                        });
                    }
                }
                let out = h.apply(p, Command::Batch(commands))?;
                println!("grid {}x{} done", rows, cols);
                for c in &out.clamped {
                    println!("clamped object {} frame {}: {:?} -> {:?}", c.id, c.frame, c.from, c.to);
                }
                Ok(())
            })?;
        }
        Cmd::Look { project, scene, mode, out, targets, views, distance, frame, camera, frames, every, count, width, height } => {
            let frames = match frames {
                Some(list) => Some(
                    list.split(',')
                        .map(|v| v.trim().parse::<i64>().context("--frames is a comma-separated list of frames"))
                        .collect::<Result<Vec<_>>>()?,
                ),
                None => None,
            };
            let views: Vec<crate::session::ViewSpec> = if views.is_empty() {
                Vec::new()
            } else {
                views.into_iter().map(crate::session::ViewSpec::Name).collect()
            };
            let mut session = crate::session::Session::new();
            session.project = Some(storage::load_project(&project)?);
            session.dir = Some(project.clone());
            let value = session.execute(crate::session::Op::Look {
                scene,
                mode: Some(mode),
                target: if targets.is_empty() { None } else { Some(targets) },
                views: Some(views),
                distance,
                frame,
                camera,
                frames,
                every,
                count,
                view: None,
                width: Some(width),
                height: Some(height),
            })?;
            use base64::Engine as _;
            let encoded = value["png_base64"].as_str().context("look returned no PNG")?;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .context("look returned a broken PNG")?;
            std::fs::write(&out, bytes).with_context(|| format!("writing {}", out.display()))?;
            println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                "sheet": value["sheet"], "tiles": value["tiles"], "findings": value["findings"],
            }))?);
            println!("sheet -> {}", out.display());
        }
        Cmd::Crowd {
            project, scene, formation, count, roles, pose, spacing, rows, scatter_deg, seed, prefix, pos, yaw, frame,
        } => {
            let mut session = crate::session::Session::new();
            session.project = Some(storage::load_project(&project)?);
            session.dir = Some(project.clone());
            let value = session.execute(crate::session::Op::Crowd {
                scene,
                formation,
                count,
                roles,
                pose,
                spacing,
                rows,
                scatter_deg,
                seed,
                prefix,
                pos: match pos.as_deref() {
                    Some(s) => {
                        let v = parse_vec3(s)?;
                        Some([v.x, v.y, v.z])
                    }
                    None => None,
                },
                yaw: yaw.to_radians(),
                frame: frame.map(|f| f as phantom_core::Frame),
            })?;
            storage::save_project(&project, session.project.as_ref().expect("project was loaded"))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "crowd": value["crowd"], "count": value["count"], "seed": value["seed"],
                    "clamped": value["clamped"], "shifted": value["shifted"],
                }))?
            );
        }
        Cmd::Scatter { project, scene, preset, area, plate, inset, action, density, from, to, step, seed, prefix } => {
            // The same Session Op the MCP tool calls: one implementation of
            // "what a fill is", so CLI and agent can never disagree.
            // Corners may be separated from each other by ":" (reads as
            // "from:to", which is how a rectangle is said out loud); within a
            // corner it is a comma. Both spellings mean the same box.
            let rect = |s: &Option<String>| -> Result<Option<[f32; 4]>> {
                match s {
                    None => Ok(None),
                    Some(v) => {
                        let flat = v.replace(':', ",");
                        let n: Vec<f32> = flat
                            .split(',')
                            .filter(|x| !x.trim().is_empty())
                            .map(|x| x.trim().parse::<f32>().with_context(|| format!("parsing \"{v}\"")))
                            .collect::<Result<Vec<_>, _>>()?;
                        if n.len() != 4 {
                            return Err(anyhow::anyhow!("expected \"x0,z0,x1,z1\" (or \"x0,z0:x1,z1\"), got \"{v}\""));
                        }
                        Ok(Some([n[0], n[1], n[2], n[3]]))
                    }
                }
            };
            // Exactly two numbers: padding with a dummy 0 and reading x/z would
            // slide the caller's z into the y slot and silently place the row on
            // the origin instead of where it was asked for.
            let point = |s: &Option<String>| -> Result<Option<[f32; 2]>> {
                match s {
                    None => Ok(None),
                    Some(v) => {
                        let n: Vec<f32> = v
                            .split(',')
                            .map(|x| x.trim().parse::<f32>().with_context(|| format!("parsing \"{v}\"")))
                            .collect::<Result<Vec<_>, _>>()?;
                        if n.len() != 2 {
                            return Err(anyhow::anyhow!("expected \"x,z\", got \"{v}\""));
                        }
                        Ok(Some([n[0], n[1]]))
                    }
                }
            };
            let mut session = crate::session::Session::new();
            session.project = Some(storage::load_project(&project)?);
            session.dir = Some(project.clone());
            let value = session.execute(crate::session::Op::Scatter {
                scene,
                preset,
                area: rect(&area)?,
                plate,
                inset,
                action: rect(&action)?,
                density,
                from: point(&from)?,
                to: point(&to)?,
                step,
                seed,
                prefix,
            })?;
            // A fill mutates the document, so it has to reach the disk like
            // every other mutation command (unlike `look`, which only reads).
            storage::save_project(&project, session.project.as_ref().expect("project was loaded"))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "objects": value["objects"], "count": value["objects"].as_array().map(|v| v.len()).unwrap_or(0),
                    "band": value["band"], "seed": value["seed"], "clamped": value["clamped"],
                }))?
            );
        }
        Cmd::AddObject { project, scene, name, asset } => {
            with_project(&project, |p, h| {
                h.apply(p, Command::AddObject { scene: SceneId(scene), name, asset: AssetId(asset) })
                    .map(|o| println!("object {}", o.object.unwrap()))
                    .map_err(anyhow::Error::from)
            })?;
        }
        Cmd::SetObjectKey { project, scene, object, frame, pos, yaw, pitch, roll, scale } => {
            with_project(&project, |p, h| {
                h.apply(
                    p,
                    Command::SetObjectKey {
                        scene: SceneId(scene),
                        object: ObjectId(object),
                        key: ObjectKey { frame, position: parse_vec3(&pos)?, yaw, pitch, roll, scale: parse_vec3(&scale)? },
                    },
                )?;
                Ok(())
            })?;
        }
        Cmd::SetCamera { project, scene, camera, pos, yaw, pitch, focal, roll, look_at, ease } => {
            let ease = parse_ease(&ease)?;
            let look_at = look_at.as_deref().map(parse_vec3).transpose()?;
            with_project(&project, |p, h| {
                let camera = resolve_active_camera(p, SceneId(scene), camera)?;
                let key = TransformKey {
                    frame: 0,
                    translation: parse_vec3(&pos)?,
                    yaw: yaw.to_radians(),
                    pitch: pitch.to_radians(),
                    roll: roll.to_radians(),
                    look_at,
                    focal_mm: Some(focal),
                    ease,
                };
                h.apply(p, Command::SetCameraKey { scene: SceneId(scene), camera, key })?;
                h.apply(p, Command::SetCameraLens { scene: SceneId(scene), camera, focal_mm: focal })?;
                Ok(())
            })?;
        }
        Cmd::CameraAdd { project, scene, name, frames } => {
            let range = frames.as_deref().map(parse_range).transpose()?;
            with_project(&project, |p, h| {
                let name = name.unwrap_or_else(|| {
                    let n = p.scenes.get(&SceneId(scene)).map(|s| s.cameras.len() + 1).unwrap_or(1);
                    format!("Camera {n}")
                });
                h.apply(p, Command::AddCamera { scene: SceneId(scene), name, range })
                    .map(|o| println!("camera {}", o.camera.unwrap()))
                    .map_err(anyhow::Error::from)
            })?;
        }
        Cmd::CameraRemove { project, scene, camera } => {
            with_project(&project, |p, h| {
                h.apply(p, Command::RemoveCamera { scene: SceneId(scene), camera: CameraId(camera) })?;
                Ok(())
            })?;
        }
        Cmd::CameraSelect { project, scene, camera } => {
            with_project(&project, |p, h| {
                h.apply(p, Command::ActivateCamera { scene: SceneId(scene), camera: CameraId(camera) })?;
                Ok(())
            })?;
        }
        Cmd::CameraTarget { project, scene, camera, mode, subject } => {
            let target = match subject.as_deref() {
                None => None,
                Some(raw) => {
                    let mode = match mode.as_deref() {
                        Some("track") => CameraTargetMode::Track,
                        Some("follow") => CameraTargetMode::Follow,
                        Some(other) => bail!("camera-target mode: track|follow, got {other}"),
                        None => bail!("camera-target: --mode is required with --subject"),
                    };
                    Some(CameraTarget { mode, subject: parse_camera_subject(raw)? })
                }
            };
            with_project(&project, |p, h| {
                let camera = resolve_active_camera(p, SceneId(scene), camera)?;
                h.apply(p, Command::SetCameraTarget { scene: SceneId(scene), camera, target })?;
                Ok(())
            })?;
        }
        Cmd::CameraList { project, scene } => {
            let project_doc = storage::load_project(&project)?;
            let s = project_doc.scenes.get(&SceneId(scene)).ok_or(CoreError::SceneNotFound(SceneId(scene)))?;
            for camera in s.cameras.values() {
                let marker = if camera.id == s.active_camera { "*" } else { " " };
                println!("{marker} {} {} [{}, {})", camera.id, camera.name, camera.range.start, camera.range.end);
            }
        }
        Cmd::SetLighting { project, scene, key, fill, rim } => {
            with_project(&project, |p, h| {
                let mut lighting = p
                    .scenes
                    .get(&SceneId(scene))
                    .map(|s| s.lighting)
                    .ok_or(CoreError::SceneNotFound(SceneId(scene)))?;
                for (spec, slot) in [(key, &mut lighting.key), (fill, &mut lighting.fill), (rim, &mut lighting.rim)] {
                    if let Some(spec) = spec {
                        *slot = parse_light(&spec)?;
                    }
                }
                h.apply(p, Command::SetLighting { scene: SceneId(scene), lighting })?;
                Ok(())
            })?;
        }
        Cmd::SetFloor { project, scene, color, off } => {
            let color = match color {
                Some(c) => Some(parse_color(&c)?),
                None => None,
            };
            with_project(&project, |p, h| {
                let floor = if off {
                    None
                } else {
                    Some(SceneFloor { color: color.unwrap_or(SceneFloor::default().color) })
                };
                h.apply(p, Command::SetFloor { scene: SceneId(scene), floor })?;
                Ok(())
            })?;
        }
        Cmd::SetStage { project, scene, preset, size, center } => {
            let (width, height, depth) = match size {
                Some(s) => {
                    let v: Vec<f32> = s
                        .split(['x', 'X', ','])
                        .map(|p| p.trim().parse::<f32>().context("size is \"width x depth x height\", e.g. 12x8x4"))
                        .collect::<Result<Vec<_>>>()?;
                    if v.len() != 3 {
                        anyhow::bail!("size is \"width x depth x height\", e.g. 12x8x4");
                    }
                    (Some(v[0]), Some(v[2]), Some(v[1]))
                }
                None => (None, None, None),
            };
            let center = match center {
                Some(c) => {
                    let v: Vec<f32> = c
                        .split(',')
                        .map(|p| p.trim().parse::<f32>().context("center is \"x,z\""))
                        .collect::<Result<Vec<_>>>()?;
                    if v.len() != 2 {
                        anyhow::bail!("center is \"x,z\"");
                    }
                    Some([v[0], v[1]])
                }
                None => None,
            };
            with_project(&project, |p, h| {
                let current = p.scenes.get(&SceneId(scene)).ok_or(CoreError::SceneNotFound(SceneId(scene)))?.stage;
                let named = match preset.as_deref() {
                    Some(name) => Some(StagePreset::parse(name).context("preset: small|medium|large")?),
                    None => None,
                };
                let size = match (named, width, height, depth) {
                    (Some(preset), w, h, d) => {
                        let base = preset.size();
                        StageSize::new(w.unwrap_or(base.width), h.unwrap_or(base.height), d.unwrap_or(base.depth))
                    }
                    (None, Some(w), Some(h), Some(d)) => StageSize::new(w, h, d),
                    (None, ..) => anyhow::bail!("set-stage needs --preset or --size"),
                };
                let preset = match (named, width, height, depth) {
                    (Some(p), None, None, None) => Some(p),
                    _ => None,
                };
                let stage = Stage { preset, size, center: center.unwrap_or(current.center) };
                let out = h.apply(p, Command::SetStage { scene: SceneId(scene), stage })?;
                for c in &out.clamped {
                    println!(
                        "clamped {} {} frame {}: {:?} -> {:?}",
                        match c.kind {
                            ClampKind::Phantom => "phantom",
                            ClampKind::Object => "object",
                        },
                        c.id,
                        c.frame,
                        c.from,
                        c.to
                    );
                }
                println!(
                    "stage: preset {} size {}x{}x{} center {},{} ({} clamped)",
                    preset.map(|p| p.name()).unwrap_or("custom"),
                    size.width,
                    size.height,
                    size.depth,
                    stage.center[0],
                    stage.center[1],
                    out.clamped.len()
                );
                Ok(())
            })?;
        }
        Cmd::Stage { project, scene } => {
            let project_doc = storage::load_project(&project)?;
            let s = project_doc.scenes.get(&SceneId(scene)).ok_or(CoreError::SceneNotFound(SceneId(scene)))?;
            println!(
                "stage: preset {} size {}x{}x{} center {},{}",
                s.stage.preset.map(|p| p.name()).unwrap_or("custom"),
                s.stage.size.width,
                s.stage.size.height,
                s.stage.size.depth,
                s.stage.center[0],
                s.stage.center[1]
            );
        }
        Cmd::RenderFilm { project, out, frames, width, height, force, normalize, target_peak_db, preview, every } => {
            let project_doc = storage::load_project(&project)?;
            let frames = match frames {
                Some(f) => Some(parse_range(&f).map(|r| (r.start, r.end))?),
                None => None,
            };
            if preview {
                // Cheap contact sheet: 480x270 when the size is not given explicitly.
                let (width, height) = if width == 1280 && height == 720 { (480, 270) } else { (width, height) };
                let started = std::time::Instant::now();
                let cancel = std::sync::atomic::AtomicBool::new(false);
                let count = phantom_runtime::export::render_film_preview(
                    &project_doc, &project, &out, width, height, every, frames,
                    |done, total| {
                        let elapsed = started.elapsed().as_secs_f64();
                        let eta = if done > 0 { elapsed / done as f64 * (total - done) as f64 } else { 0.0 };
                        println!("{done}/{total} (elapsed {elapsed:.1} s, ~{eta:.1} s left)");
                    },
                    &cancel,
                )?;
                println!("film preview: {count} frames -> {}", out.display());
            } else {
                let opts = phantom_runtime::export::FilmExportOptions { width, height, out, frames, overwrite: force, normalize, target_peak_dbfs: target_peak_db };
            let cancel = std::sync::atomic::AtomicBool::new(false);
            let started = std::time::Instant::now();
            let path = phantom_runtime::export::export_film(&project_doc, &project, &opts, move |done, total| {
                // D5: ETA from the measured pace, not just "47/48".
                let elapsed = started.elapsed().as_secs_f64();
                let eta = if done > 0 { elapsed / done as f64 * (total - done) as f64 } else { 0.0 };
                println!("{done}/{total} (elapsed {elapsed:.1} s, ~{eta:.1} s left)");
            }, &cancel)?;
            println!("film -> {}", path.display());
            }
        }
        Cmd::FilmAdd { project, scene, frames, camera } => {
            let range = match frames {
                Some(f) => parse_range(&f)?,
                None => {
                    let camera_id = camera.ok_or_else(|| anyhow::anyhow!("film-add: --camera is required without --frames"))?;
                    let doc = storage::load_project(&project)?;
                    doc.scenes
                        .get(&SceneId(scene))
                        .and_then(|s| s.cameras.get(&CameraId(camera_id)))
                        .map(|c| c.range)
                        .ok_or(CoreError::CameraNotFound(CameraId(camera_id)))?
                }
            };
            with_project(&project, |p, h| {
                h.apply(p, Command::AddFilmClip { scene: SceneId(scene), range, camera: camera.map(CameraId) })
                    .map(|o| println!("clip {}", o.clip.unwrap()))
                    .map_err(anyhow::Error::from)
            })?;
        }
        Cmd::TrackAdd { project, name, scene } => {
            with_project(&project, |p, h| {
                h.apply(p, Command::AddAudioTrack { scene: scene.map(SceneId), name })
                    .map(|o| println!("track {}", o.track.unwrap()))
                    .map_err(anyhow::Error::from)
            })?;
        }
        Cmd::RenderAudio { project, out, normalize, target_peak_db } => {
            let project_doc = storage::load_project(&project)?;
            let mut pcm = phantom_core::PcmCache::new();
            let audio_assets: Vec<_> = project_doc
                .assets
                .values()
                .filter(|a| a.kind == phantom_core::AssetKind::Audio)
                .map(|a| (a.id, project.join(&a.path)))
                .collect();
            phantom_runtime::audio::ensure_pcm(audio_assets, &mut pcm)?;
            let (mut film_pcm, peak) = phantom_runtime::audio::mix_film(&project_doc, &pcm);
            if normalize && peak > 0.0 {
                let scale = 10f32.powf(target_peak_db / 20.0) / peak;
                for s in &mut film_pcm.samples {
                    s[0] *= scale;
                    s[1] *= scale;
                }
            }
            phantom_runtime::audio::write_wav(&film_pcm, &out)?;
            println!("film audio -> {} (peak {:.1} dBFS{})", out.display(), 20.0 * peak.max(1e-9).log10(), if normalize { ", normalized" } else { "" });
        }
        Cmd::ClipUpdate { project, track, clip, offset, len, fade_in, fade_out, offset_secs, len_secs, fade_in_secs, fade_out_secs } => {
            let pick = |samples: Option<u64>, secs: Option<f64>, name: &str| -> anyhow::Result<Option<u64>> {
                match (samples, secs) {
                    (Some(_), Some(_)) => anyhow::bail!("clip-update: {name} and {name}-secs are mutually exclusive"),
                    (Some(v), None) => Ok(Some(v)),
                    (None, Some(v)) => {
                        if !v.is_finite() || v < 0.0 {
                            anyhow::bail!("clip-update: {name}-secs must be finite and >= 0");
                        }
                        Ok(Some((v * 48_000.0).round() as u64))
                    }
                    (None, None) => Ok(None),
                }
            };
            with_project(&project, |p, h| {
                h.apply(
                    p,
                    Command::UpdateAudioClip {
                        track: TrackId(track),
                        clip: phantom_core::audio::AudioClipId(clip),
                        offset_samples: pick(offset, offset_secs, "--offset")?,
                        len_samples: pick(len, len_secs, "--len")?,
                        fade_in_samples: pick(fade_in, fade_in_secs, "--fade-in")?,
                        fade_out_samples: pick(fade_out, fade_out_secs, "--fade-out")?,
                    },
                )?;
                Ok(())
            })?;
        }
        Cmd::ClipAdd { project, track, asset, start, offset, len, offset_secs, len_secs, fade_in, fade_out, fade_in_secs, fade_out_secs } => {
            let pick = |samples: Option<u64>, secs: Option<f64>, name: &str| -> anyhow::Result<Option<u64>> {
                match (samples, secs) {
                    (Some(_), Some(_)) => anyhow::bail!("clip-add: {name} and {name}-secs are mutually exclusive"),
                    (Some(v), None) => Ok(Some(v)),
                    (None, Some(v)) => {
                        if !v.is_finite() || v < 0.0 {
                            anyhow::bail!("clip-add: {name}-secs must be finite and >= 0");
                        }
                        Ok(Some((v * 48_000.0).round() as u64))
                    }
                    (None, None) => Ok(None),
                }
            };
            let offset_samples = pick(Some(offset), offset_secs, "--offset")?.unwrap_or(0);
            let len_samples = pick(len, len_secs, "--len")?.ok_or_else(|| anyhow::anyhow!("clip-add: --len or --len-secs is required"))?;
            with_project(&project, |p, h| {
                h.apply(
                    p,
                    Command::AddAudioClip {
                        track: TrackId(track),
                        asset: AssetId(asset),
                        start,
                        offset_samples,
                        len_samples,
                        fade_in_samples: pick(fade_in, fade_in_secs, "--fade-in")?.unwrap_or(0),
                        fade_out_samples: pick(fade_out, fade_out_secs, "--fade-out")?.unwrap_or(0),
                    },
                )
                .map(|o| println!("clip {}", o.audio_clip.unwrap()))
                .map_err(anyhow::Error::from)
            })?;
        }
        Cmd::Render { project, scene, out, camera, frames, width, height, preview, every } => {
            let project_doc = storage::load_project(&project)?;
            let scene_doc = project_doc
                .scenes
                .get(&SceneId(scene))
                .with_context(|| format!("scene {scene} not found"))?;
            let camera_id = match camera {
                Some(id) => {
                    if !scene_doc.cameras.contains_key(&CameraId(id)) {
                        return Err(CoreError::CameraNotFound(CameraId(id))).with_context(|| format!("camera {id}"));
                    }
                    CameraId(id)
                }
                // No --camera: render through the scene's active camera, as the
                // viewport does.
                None => scene_doc.active_camera,
            };
            let range = match frames {
                Some(f) => parse_range(&f)?,
                None => scene_doc.range_for(Some(camera_id)),
            };
            std::fs::create_dir_all(&out)?;
            // Preview (B7): 480x270 and every N-th frame; explicit --width/--height
            // override the preview size.
            let (width, height) = if preview && width == 1280 && height == 720 { (480, 270) } else { (width, height) };
            let step = if preview { every.max(1) as i64 } else { 1 };
            let frames: Vec<Frame> = (range.start..range.end).step_by(step as usize).collect();
            let opts = RenderOptions { width, height };
            let mut renderer = Renderer::new(width, height)
                .context("cannot create the GPU renderer")?;
            let started = std::time::Instant::now();
            for (done, frame) in frames.iter().enumerate() {
                let rendered =
                    renderer.render_frame(&project_doc, scene_doc, &project, *frame, &opts, Some(camera_id))?;
                let path = out.join(format!("frame_{frame:06}.png"));
                rendered.save_png(&path)?;
                // D5: progress with ETA, not just "47/48".
                let done = done + 1;
                let elapsed = started.elapsed().as_secs_f64();
                let eta = if done > 0 { elapsed / done as f64 * (frames.len() - done) as f64 } else { 0.0 };
                println!("{done}/{} (elapsed {elapsed:.1} s, ~{eta:.1} s left) -> {}", frames.len(), path.display());
            }
        }
    }
    Ok(())
}

fn with_project(
    dir: &Path,
    f: impl FnOnce(&mut Project, &mut History) -> Result<()>,
) -> Result<()> {
    let mut project = storage::load_project(dir)?;
    let mut history = History::default();
    f(&mut project, &mut history)?;
    storage::save_project(dir, &project)?;
    Ok(())
}

fn parse_range(s: &str) -> Result<FrameRange> {    let (a, b) = s.split_once('-').context("range of the form start-end")?;
    Ok(FrameRange::new(a.trim().parse()?, b.trim().parse()?)?)
}

/// The camera a command refers to: an explicit id, or the scene's active camera.
fn resolve_active_camera(p: &Project, scene: SceneId, camera: Option<u64>) -> Result<CameraId> {
    match camera {
        Some(id) => Ok(CameraId(id)),
        None => Ok(p.scenes.get(&scene).ok_or(CoreError::SceneNotFound(scene))?.active_camera),
    }
}

/// A camera subject on the command line: "phantom:<id>" or "object:<id>".
fn parse_camera_subject(s: &str) -> Result<CameraSubject> {
    let (kind, id) = s
        .split_once(':')
        .context("subject is \"phantom:<id>\" or \"object:<id>\"")?;
    let id: u64 = id.trim().parse().context("subject id is a number")?;
    match kind.trim().to_ascii_lowercase().as_str() {
        "phantom" => Ok(CameraSubject::Phantom(PhantomId(id))),
        "object" => Ok(CameraSubject::Object(ObjectId(id))),
        other => bail!("subject kind: phantom|object, got {other}"),
    }
}

/// "r,g,b" — linear RGB color.
fn parse_color(s: &str) -> Result<[f32; 3]> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 3 {
        bail!("color of the form r,g,b");
    }
    Ok([parts[0].trim().parse()?, parts[1].trim().parse()?, parts[2].trim().parse()?])
}

/// "r,g,b,lux[,on|off][,shadow|noshadow]" — one stage light source.
/// Shadows are opt-in (corrective A2: the skinned shadow pipeline crashes).
fn parse_light(s: &str) -> Result<SceneLight> {
    let parts: Vec<&str> = s.split(',').collect();
    if !(4..=6).contains(&parts.len()) {
        bail!("light source of the form r,g,b,lux[,on|off][,shadow|noshadow]");
    }
    let enabled = match parts.get(4).map(|p| p.trim()) {
        None | Some("on") => true,
        Some("off") => false,
        Some(p) => bail!("expected on|off, got {p}"),
    };
    let shadows = match parts.get(5).map(|p| p.trim()) {
        None | Some("noshadow") => false,
        Some("shadow") => true,
        Some(p) => bail!("expected shadow|noshadow, got {p}"),
    };
    Ok(SceneLight {
        color: [parts[0].trim().parse()?, parts[1].trim().parse()?, parts[2].trim().parse()?],
        intensity: parts[3].trim().parse()?,
        enabled,
        shadows,
    })
}

/// Smoothing of a camera key segment (C1.5).
fn parse_ease(s: &str) -> Result<phantom_core::project::KeyEase> {
    use phantom_core::project::KeyEase;
    match s {
        "linear" => Ok(KeyEase::Linear),
        "ease_in" => Ok(KeyEase::EaseIn),
        "ease_out" => Ok(KeyEase::EaseOut),
        "ease_in_out" => Ok(KeyEase::EaseInOut),
        other => bail!("ease: linear|ease_in|ease_out|ease_in_out, got {other}"),
    }
}

fn parse_vec3(s: &str) -> Result<Vec3> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 3 {
        bail!("vector of the form x,y,z");
    }
    Ok(Vec3::new(parts[0].trim().parse()?, parts[1].trim().parse()?, parts[2].trim().parse()?))
}

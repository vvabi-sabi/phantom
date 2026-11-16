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
        /// Explicit asset kind: model-obj|model-fbx|motion-fbx|image|audio|video
        /// (by default taken from the file extension; with bundle import it applies
        /// to every file).
        #[arg(long)]
        kind: Option<String>,
    },
    /// Version, build features and limits (D8) — machine-readable analog of
    /// MCP studio_capabilities for the CLI user.
    Capabilities,
    /// Set the background on the arc screen.
    /// The default geometry (arc 2.0, radius 6, height 4) covers the frame only
    /// for focal < ~28 mm; MCP background_fit computes the needed arc, and the
    /// rule is: horizontal coverage ≈ 2·asin(sin(arc/2)·R/dist) rad must exceed
    /// the camera's hfov (hfov = 2·atan(18/focal·(9/16)) at 16:9).
    SetBackground {
        project: PathBuf,
        #[arg(long)]
        scene: u64,
        #[arg(long)]
        asset: u64,
        /// Background is a video (image by default).
        #[arg(long)]
        video: bool,
        #[arg(long, default_value_t = 2.0)]
        arc: f32,
        #[arg(long, default_value_t = 6.0)]
        radius: f32,
        #[arg(long, default_value_t = 4.0)]
        height: f32,
        #[arg(long, default_value = "0,2,-5")]
        center: String,
        /// Arc orientation: away_from_camera (screen wraps around the scene, default)
        /// or toward_camera (flat backdrop, pre-A3 behavior).
        #[arg(long, default_value = "away_from_camera")]
        bulge: String,
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
        /// Action from the actor catalog (idle|walk|jump|sit|lie|sneak|trot|gallop|fly|perch).
        #[arg(long, default_value = "idle")]
        action: String,
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
    /// Configure the camera (one static key).
    SetCamera {
        project: PathBuf,
        #[arg(long)]
        scene: u64,
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
        /// frame into the --out directory (shot camera, as in export).
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
    /// Add a clip to the film (at the end): a scene range or a shot (--shot;
    /// without --frames the shot range is used).
    FilmAdd { project: PathBuf, #[arg(long)] scene: u64, #[arg(long)] frames: Option<String>, #[arg(long)] shot: Option<u64> },
    /// Assign a motion clip to an imported rig. Without --frames —
    /// legacy (longest stack, looped over the whole scene); with --frames
    /// "start,end" — add a use to the clip track.
    SetClip {
        project: PathBuf,
        #[arg(long)]
        scene: u64,
        #[arg(long)]
        phantom: u64,
        #[arg(long)]
        asset: u64,
        #[arg(long)]
        frames: Option<String>,
        #[arg(long)]
        stack: Option<String>,
        #[arg(long, default_value_t = 1.0)]
        speed: f32,
        #[arg(long, default_value_t = 0.0)]
        blend_in: f32,
        #[arg(long, default_value_t = 0.0)]
        blend_out: f32,
        /// Clip's root motion instead of the controller.
        #[arg(long, default_value_t = false)]
        root_motion_clip: bool,
        /// Retargeting profile (humanoid|quadruped); without it — direct
        /// bone-name matching. Only together with --frames.
        #[arg(long)]
        retarget: Option<String>,
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
        /// Scene shot: its camera and range (if --frames is not given).
        #[arg(long)]
        shot: Option<u64>,
        /// Frame range "start-end" (upper bound EXCLUSIVE:
        /// 0-1 — frame 0 only); by default the whole scene (or shot).
        #[arg(long)]
        frames: Option<String>,
        #[arg(long, default_value_t = 1280)]
        width: u32,
        #[arg(long, default_value_t = 720)]
        height: u32,
        /// "On twos" mode for the Phantom pose.
        #[arg(long)]
        twos: bool,
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
                Some("motion-fbx") => Some(AssetKind::MotionFbx),
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
                    // kind only affects registration: motion-fbx is the same fbx bundle.
                    let rel = if matches!(kind, AssetKind::ModelObj | AssetKind::ModelFbx | AssetKind::MotionFbx) {
                        let bundle = phantom_runtime::import::import_model_bundle(&project, file)?;
                        for d in &bundle.diagnostics {
                            eprintln!("import: {d}");
                        }
                        bundle.model
                    } else {
                        storage::import_file(&project, file)?
                    };
                    with_project(&project, |p, h| {
                        h.apply(p, Command::RegisterAsset { kind, path: rel.clone() })
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
        Cmd::SetBackground { project, scene, asset, video, arc, radius, height, center, bulge, fit } => {
            let bulge = match bulge.as_str() {
                "away_from_camera" => phantom_core::project::Bulge::AwayFromCamera,
                "toward_camera" => phantom_core::project::Bulge::TowardCamera,
                other => anyhow::bail!("bulge: away_from_camera|toward_camera, got {other}"),
            };
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
                            arc_radians: arc,
                            radius,
                            height,
                            center: parse_vec3(&center)?,
                            bulge,
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
        Cmd::SetKey { project, scene, phantom, frame, pos, action, yaw } => {
            let action = builtin::parse_action(&action).map_err(anyhow::Error::msg)?;
            with_project(&project, |p, h| {
                h.apply(
                    p,
                    Command::SetMotionKey {
                        scene: SceneId(scene),
                        phantom: PhantomId(phantom),
                        key: MotionKey { yaw, frame, position: parse_vec3(&pos)?, action },
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
        Cmd::SetCamera { project, scene, pos, yaw, pitch, focal, roll, look_at, ease } => {
            let ease = parse_ease(&ease)?;
            let look_at = look_at.as_deref().map(parse_vec3).transpose()?;
            with_project(&project, |p, h| {
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
                h.apply(
                    p,
                    Command::SetCameraKeys {
                        scene: SceneId(scene),
                        keys: vec![key],
                    },
                )?;
                h.apply(p, Command::SetCameraFocal { scene: SceneId(scene), focal_mm: focal })?;
                Ok(())
            })?;
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
        Cmd::FilmAdd { project, scene, frames, shot } => {
            let range = match frames {
                Some(f) => parse_range(&f)?,
                None => {
                    let shot_id = shot.ok_or_else(|| anyhow::anyhow!("film-add: --shot is required without --frames"))?;
                    let doc = storage::load_project(&project)?;
                    doc.scenes
                        .get(&SceneId(scene))
                        .and_then(|s| s.shots.get(&ShotId(shot_id)))
                        .map(|sh| sh.range)
                        .ok_or(CoreError::ShotNotFound(ShotId(shot_id)))?
                }
            };
            with_project(&project, |p, h| {
                h.apply(p, Command::AddFilmClip { scene: SceneId(scene), range, shot: shot.map(ShotId) })
                    .map(|o| println!("clip {}", o.clip.unwrap()))
                    .map_err(anyhow::Error::from)
            })?;
        }
        Cmd::SetClip { project, scene, phantom, asset, frames, stack, speed, blend_in, blend_out, root_motion_clip, retarget } => {
            with_project(&project, |p, h| {
                match frames {
                    Some(frames) => {
                        let range = parse_range(&frames)?;
                        let retarget = match retarget.as_deref() {
                            None => None,
                            Some("humanoid") => Some(phantom_core::RetargetProfile::Humanoid),
                            Some("quadruped") => Some(phantom_core::RetargetProfile::Quadruped),
                            Some(other) => bail!("unknown retargeting profile: {other} (humanoid|quadruped)"),
                        };
                        let clip_use = phantom_core::ClipUse {
                            id: phantom_core::ClipUseId(0), // assigned by the command
                            asset: AssetId(asset),
                            stack,
                            range,
                            speed,
                            blend_in,
                            blend_out,
                            root_motion: if root_motion_clip { phantom_core::RootMotion::Clip } else { phantom_core::RootMotion::Controller },
                            retarget,
                        };
                        h.apply(p, Command::AddClipUse { scene: SceneId(scene), phantom: PhantomId(phantom), clip_use })
                            .map(|o| println!("clip_use {}", o.clip_use.unwrap()))?;
                    }
                    None => {
                        if retarget.is_some() {
                            bail!("--retarget is supported only with --frames (clip track)");
                        }
                        h.apply(p, Command::SetPhantomClip { scene: SceneId(scene), phantom: PhantomId(phantom), clip: Some(AssetId(asset)) })?;
                    }
                }
                Ok(())
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
        Cmd::Render { project, scene, out, shot, frames, width, height, twos, preview, every } => {
            let project_doc = storage::load_project(&project)?;
            let scene_doc = project_doc
                .scenes
                .get(&SceneId(scene))
                .with_context(|| format!("scene {scene} not found"))?;
            let shot_id = shot.map(ShotId);
            if let Some(id) = shot_id {
                if !scene_doc.shots.contains_key(&id) {
                    return Err(CoreError::ShotNotFound(id)).with_context(|| format!("shot {id}"));
                }
            }
            let range = match frames {
                Some(f) => parse_range(&f)?,
                None => scene_doc.range_for_shot(shot_id),
            };
            std::fs::create_dir_all(&out)?;
            // Preview (B7): 480x270 and every N-th frame; explicit --width/--height
            // override the preview size.
            let (width, height) = if preview && width == 1280 && height == 720 { (480, 270) } else { (width, height) };
            let step = if preview { every.max(1) as i64 } else { 1 };
            let frames: Vec<Frame> = (range.start..range.end).step_by(step as usize).collect();
            let opts = RenderOptions { width, height, on_twos: twos };
            let mut renderer = Renderer::new(width, height)
                .context("cannot create the GPU renderer")?;
            let started = std::time::Instant::now();
            for (done, frame) in frames.iter().enumerate() {
                let rendered =
                    renderer.render_frame(&project_doc, scene_doc, &project, *frame, &opts, shot_id)?;
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

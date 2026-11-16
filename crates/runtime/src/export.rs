//! Film export to MP4 via an external FFmpeg: frames streamed into stdin,
//! no shell; codec checked before starting; temp file + rename;
//! cancelling kills the process and removes the temp file.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};

use phantom_core::{film_len, film_map_clip, Frame, PcmCache, Project};

use crate::render::{RenderOptions, Renderer};

#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    #[error("ffmpeg not found: {0}")]
    FfmpegMissing(std::io::Error),
    #[error("ffmpeg has no libx264 codec — MP4 export is unavailable, a PNG sequence remains an option")]
    CodecMissing,
    #[error("render: {0}")]
    Render(#[from] crate::render::RenderError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("ffmpeg exited with an error: {0}")]
    FfmpegFailed(String),
    #[error("audio: {0}")]
    Audio(#[from] crate::audio::AudioError),
    #[error("export cancelled")]
    Cancelled,
    #[error("file {0} already exists (overwriting without explicit permission is not allowed; use --force)")]
    OutputExists(PathBuf),
    #[error("film is empty")]
    EmptyFilm,
}

pub struct FilmExportOptions {
    pub width: u32,
    pub height: u32,
    pub out: PathBuf,
    /// Film frame range [start, end); None — the whole film.
    pub frames: Option<(Frame, Frame)>,
    /// Allow overwriting an existing file.
    pub overwrite: bool,
    /// Peak-normalize the mix to `target_peak_dbfs` (off by default —
    /// corrective A6: "the quietest track pulls the mix up" must not
    /// happen silently).
    pub normalize: bool,
    /// Target peak dBFS when normalizing (typically −1.0).
    pub target_peak_dbfs: f32,
}

/// Film contact sheet (corrective_2 K0.4): every `every`-th film frame is
/// rendered to a PNG inside `out_dir`. Same clip→scene mapping and same shot
/// camera as the export, so the preview shows the film itself, not the whole
/// scene. One renderer for the entire run.
pub fn render_film_preview(
    project: &Project,
    project_dir: &Path,
    out_dir: &Path,
    width: u32,
    height: u32,
    every: u32,
    frames: Option<(Frame, Frame)>,
    mut progress: impl FnMut(Frame, Frame),
    cancel: &AtomicBool,
) -> Result<usize, ExportError> {
    if project.film.is_empty() {
        return Err(ExportError::EmptyFilm);
    }
    let step = every.max(1) as usize;
    let total = film_len(&project.film);
    let (start, end) = frames.unwrap_or((0, total));
    let end = end.min(total);
    if end <= start {
        return Err(ExportError::EmptyFilm);
    }
    std::fs::create_dir_all(out_dir)?;
    let opts = RenderOptions { width, height, on_twos: false };
    let mut renderer = Renderer::new(width, height)?;
    let picked: Vec<Frame> = (start..end).step_by(step).collect();
    for (done, film_frame) in picked.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err(ExportError::Cancelled);
        }
        let Some((clip, scene_frame)) = film_map_clip(&project.film, *film_frame) else { continue };
        let scene = project
            .scenes
            .get(&clip.scene)
            .ok_or(crate::render::RenderError::GpuUnavailable(format!("scene {} not found", clip.scene)))?;
        let frame = renderer.render_frame(project, scene, project_dir, scene_frame, &opts, clip.shot)?;
        frame.save_png(&out_dir.join(format!("film_{film_frame:06}.png")))?;
        progress(done as Frame + 1, picked.len() as Frame);
    }
    Ok(picked.len())
}

/// Check that ffmpeg is present and has the libx264 codec.
pub fn check_encoder() -> Result<(), ExportError> {
    let out = Command::new("ffmpeg")
        .arg("-hide_banner")
        .arg("-encoders")
        .output()
        .map_err(ExportError::FfmpegMissing)?;
    let text = String::from_utf8_lossy(&out.stdout);
    if !text.contains("libx264") {
        return Err(ExportError::CodecMissing);
    }
    Ok(())
}

/// Export the film to MP4. `progress(done, total)` is called on every frame;
/// `cancel` stops the export atomically.
pub fn export_film(
    project: &Project,
    project_dir: &Path,
    opts: &FilmExportOptions,
    mut progress: impl FnMut(Frame, Frame),
    cancel: &AtomicBool,
) -> Result<PathBuf, ExportError> {
    if project.film.is_empty() {
        return Err(ExportError::EmptyFilm);
    }
    if opts.out.exists() && !opts.overwrite {
        return Err(ExportError::OutputExists(opts.out.clone()));
    }
    check_encoder()?;

    let total = film_len(&project.film);
    let (start, end) = opts.frames.unwrap_or((0, total));
    let end = end.min(total);

    let tmp = opts.out.with_extension("part.mp4");

    // Audio: scenes joined at the cuts + film tracks -> a temp WAV.
    let has_audio = !project.film_audio.is_empty()
        || project.scenes.values().any(|s| !s.audio_tracks.is_empty());
    // A5: audio is a fact about the export, not a surprise: the message goes to
    // stderr (stdout stays a clean progress protocol; MCP shows the stderr tail).
    if has_audio {
        eprintln!(
            "audio: mixing scene and film tracks -> AAC 48 kHz stereo{}",
            if opts.normalize { " (peak normalization)" } else { "" }
        );
    } else {
        eprintln!("audio: no audio tracks — the film will be silent (see MCP audio_report)");
    }
    let wav = opts.out.with_extension("part.wav");
    if has_audio {
        let mut pcm = PcmCache::new();
        let audio_assets: Vec<_> = project
            .assets
            .values()
            .filter(|a| a.kind == phantom_core::AssetKind::Audio)
            .map(|a| (a.id, project_dir.join(&a.path)))
            .collect();
        crate::audio::ensure_pcm(audio_assets, &mut pcm)?;
        let (mut film_pcm, peak) = crate::audio::mix_film(project, &pcm);
        if opts.normalize && peak > 0.0 {
            // Peak normalization: a single linear gain over the whole mix
            // (no dynamic-range compression, no "pulling up" silence in
            // multichannel — the mix as a whole is scaled).
            let target = 10f32.powf(opts.target_peak_dbfs / 20.0);
            let scale = target / peak;
            for s in &mut film_pcm.samples {
                s[0] *= scale;
                s[1] *= scale;
            }
        }
        crate::audio::write_wav(&film_pcm, &wav)?;
    }

    let mut args: Vec<String> = vec![
        "-hide_banner".into(),
        "-loglevel".into(), "error".into(),
        "-f".into(), "rawvideo".into(),
        "-pix_fmt".into(), "rgba".into(),
        "-s".into(), format!("{}x{}", opts.width, opts.height),
        "-r".into(), "24".into(),
        "-i".into(), "pipe:0".into(),
    ];
    if has_audio {
        args.push("-i".into());
        args.push(wav.to_string_lossy().into());
        args.push("-c:a".into());
        args.push("aac".into());
        args.push("-shortest".into());
    } else {
        args.push("-an".into());
    }
    args.extend([
        "-c:v".into(), "libx264".into(),
        "-pix_fmt".into(), "yuv420p".into(),
        "-crf".into(), "18".into(),
        "-movflags".into(), "+faststart".into(),
        "-f".into(), "mp4".into(),
    ]);
    let mut child: Child = Command::new("ffmpeg")
        .args(&args)
        .arg(&tmp)
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(ExportError::FfmpegMissing)?;
    let mut stdin = child.stdin.take().expect("stdin piped");

    let mut renderer = Renderer::new(opts.width, opts.height)?;
    let render_opts = RenderOptions { width: opts.width, height: opts.height, on_twos: false };
    let mut result: Result<(), ExportError> = Ok(());
    for film_frame in start..end {
        if cancel.load(Ordering::Relaxed) {
            result = Err(ExportError::Cancelled);
            break;
        }
        let Some((clip, scene_frame)) = film_map_clip(&project.film, film_frame) else { continue };
        let scene = project
            .scenes
            .get(&clip.scene)
            .ok_or(crate::render::RenderError::GpuUnavailable(format!("scene {} not found", clip.scene)))?;
        // A clip-shot renders with its shot camera; audio maps through the scene
        // range in mix_film — there is no double mix.
        let frame = renderer.render_frame(project, scene, project_dir, scene_frame, &render_opts, clip.shot)?;
        if let Err(e) = stdin.write_all(&frame.rgba) {
            result = Err(ExportError::Io(e));
            break;
        }
        progress(film_frame - start + 1, end - start);
    }
    drop(stdin);

    if let Err(e) = result {
        let _ = child.kill();
        let out = child.wait_with_output();
        let _ = std::fs::remove_file(&tmp);
        let _ = std::fs::remove_file(&wav);
        if let Ok(out) = out {
            let tail = String::from_utf8_lossy(&out.stderr);
            if !tail.trim().is_empty() {
                return Err(ExportError::FfmpegFailed(format!("{e}; ffmpeg: {}", tail.chars().take(400).collect::<String>())));
            }
        }
        return Err(e);
    }

    let out = child.wait_with_output()?;
    let _ = std::fs::remove_file(&wav);
    if !out.status.success() {
        let _ = std::fs::remove_file(&tmp);
        let tail = String::from_utf8_lossy(&out.stderr);
        return Err(ExportError::FfmpegFailed(tail.chars().take(500).collect()));
    }
    std::fs::rename(&tmp, &opts.out)?;
    Ok(opts.out.clone())
}

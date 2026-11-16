//! Audio decoding through FFmpeg, film audio assembly, cpal preview.

use std::path::Path;
use std::process::{Command, Stdio};

use phantom_core::*;

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("ffmpeg: {0}")]
    Ffmpeg(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Whether ffmpeg is available (a cheap check by running `ffmpeg -version`).
pub fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-version"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Decode an audio file to f32 stereo 48 kHz through ffmpeg.
pub fn decode_audio(path: &Path) -> Result<PcmBuf, AudioError> {
    let out = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel", "error",
            "-i",
            &path.to_string_lossy(),
            "-f", "f32le",
            "-ac", "2",
            "-ar", "48000",
            "pipe:1",
        ])
        .stderr(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?
        .wait_with_output()?;
    if !out.status.success() {
        let tail = String::from_utf8_lossy(&out.stderr);
        return Err(AudioError::Ffmpeg(tail.chars().take(300).collect()));
    }
    let samples = out
        .stdout
        .chunks_exact(8)
        .map(|c| [f32::from_le_bytes([c[0], c[1], c[2], c[3]]), f32::from_le_bytes([c[4], c[5], c[6], c[7]])])
        .collect();
    Ok(PcmBuf { samples })
}

/// Decode audio assets into the cache (skipping the ones already present).
/// `assets` — (id, absolute path) of the project's audio assets.
pub fn ensure_pcm(
    assets: impl IntoIterator<Item = (AssetId, std::path::PathBuf)>,
    cache: &mut PcmCache,
) -> Result<(), AudioError> {
    for (id, path) in assets {
        if cache.contains_key(&id) {
            continue;
        }
        let buf = decode_audio(&path)?;
        cache.insert(id, buf);
    }
    Ok(())
}

/// Mix a single scene frame.
pub fn mix_scene_frame(scene: &Scene, frame: Frame, pcm: &PcmCache) -> ([[f32; 2]; SAMPLES_PER_FRAME as usize], f32) {
    let mut out = [[0.0f32; 2]; SAMPLES_PER_FRAME as usize];
    let peak = mix_frame(&scene.audio_tracks, frame, pcm, &mut out);
    (out, peak)
}

/// Assemble the film's full audio: scenes joined at the cuts (film_map) plus
/// film tracks. Returns the PCM and the maximum peak.
pub fn mix_film(project: &Project, pcm: &PcmCache) -> (PcmBuf, f32) {
    let total = film_len(&project.film);
    let mut result = Vec::with_capacity(total as usize * SAMPLES_PER_FRAME as usize);
    let mut peak = 0.0f32;
    let mut block = [[0.0f32; 2]; SAMPLES_PER_FRAME as usize];
    for film_frame in 0..total {
        block.fill([0.0, 0.0]);
        if let Some((scene_id, scene_frame)) = film_map(&project.film, film_frame) {
            if let Some(scene) = project.scenes.get(&scene_id) {
                mix_into(&scene.audio_tracks, scene_frame, pcm, &mut block);
            }
        }
        mix_into(&project.film_audio, film_frame, pcm, &mut block);
        peak = peak.max(peak_of(&block));
        result.extend_from_slice(&block);
    }
    (PcmBuf { samples: result }, peak)
}

/// Write the PCM to a temp WAV (s16 stereo 48 kHz), clamping to full scale.
pub fn write_wav(pcm: &PcmBuf, path: &Path) -> Result<(), AudioError> {
    let data_len = (pcm.samples.len() * 4) as u32;
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    use std::io::Write;
    w.write_all(b"RIFF")?;
    w.write_all(&(36 + data_len).to_le_bytes())?;
    w.write_all(b"WAVEfmt ")?;
    w.write_all(&16u32.to_le_bytes())?;
    w.write_all(&1u16.to_le_bytes())?; // PCM
    w.write_all(&2u16.to_le_bytes())?; // stereo
    w.write_all(&SAMPLE_RATE.to_le_bytes())?;
    w.write_all(&(SAMPLE_RATE * 4).to_le_bytes())?;
    w.write_all(&4u16.to_le_bytes())?;
    w.write_all(&16u16.to_le_bytes())?;
    w.write_all(b"data")?;
    w.write_all(&data_len.to_le_bytes())?;
    for s in &pcm.samples {
        w.write_all(&clamp_i16(s[0]).to_le_bytes())?;
        w.write_all(&clamp_i16(s[1]).to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Preview through cpal: per-frame delivery of mixed blocks.
// ---------------------------------------------------------------------------

/// Live playback. Without a device — silent fallback (None + reason).
pub struct Playback {
    _stream: cpal::Stream,
    producer: std::sync::mpsc::SyncSender<Vec<[f32; 2]>>,
}

impl Playback {
    /// Start a stream on the default device.
    pub fn start() -> Result<Option<Self>, AudioError> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
        let host = cpal::default_host();
        let Some(device) = host.default_output_device() else {
            return Ok(None);
        };
        let config = cpal::StreamConfig { channels: 2, sample_rate: SAMPLE_RATE, buffer_size: cpal::BufferSize::Default };
        let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<[f32; 2]>>(8);
        let stream = device
            .build_output_stream(
                &config,
                move |out: &mut [f32], _| {
                    let mut filled = 0;
                    while let Ok(block) = rx.try_recv() {
                        for (i, s) in block.iter().enumerate() {
                            if filled + 1 < out.len() {
                                out[filled] = s[0];
                                out[filled + 1] = s[1];
                                filled += 2;
                                let _ = i;
                            }
                        }
                    }
                    for x in &mut out[filled..] {
                        *x = 0.0;
                    }
                },
                |e| eprintln!("audio preview: {e}"),
                None,
            )
            .map_err(|e| AudioError::Ffmpeg(format!("output stream: {e}")))?;
        stream.play().map_err(|e| AudioError::Ffmpeg(format!("stream play: {e}")))?;
        Ok(Some(Self { _stream: stream, producer: tx }))
    }

    /// Feed one frame of audio (non-blocking; on overflow the frame is dropped).
    pub fn push(&self, block: &[[f32; 2]; SAMPLES_PER_FRAME as usize]) {
        let _ = self.producer.try_send(block.to_vec());
    }
}

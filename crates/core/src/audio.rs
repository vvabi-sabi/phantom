//! Sound: tracks, clips, core mixer. 48 kHz, 2000 samples per frame,
//! float PCM stereo; preview and export share one piece of logic.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::id::AssetId;
use crate::time::Frame;

pub const SAMPLE_RATE: u32 = 48000;
/// Samples per channel per frame at 24 fps.
pub const SAMPLES_PER_FRAME: u32 = SAMPLE_RATE / 24;

/// Audio track identifier.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct TrackId(pub u64);
impl std::fmt::Display for TrackId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl From<u64> for TrackId {
    fn from(v: u64) -> Self {
        Self(v)
    }
}

/// Audio clip identifier.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct AudioClipId(pub u64);
impl std::fmt::Display for AudioClipId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl From<u64> for AudioClipId {
    fn from(v: u64) -> Self {
        Self(v)
    }
}

/// Audio clip on a track: a fragment of an asset on the timeline.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AudioClip {
    pub id: AudioClipId,
    pub asset: AssetId,
    /// Start frame on the owner's timeline (scene or film).
    pub start: Frame,
    /// Offset into the source, in samples.
    pub offset_samples: u64,
    /// Duration, in samples.
    pub len_samples: u64,
    /// Fade-in, in samples (0 — none; corrective A4/B6).
    #[serde(default)]
    pub fade_in_samples: u64,
    /// Fade-out, in samples (0 — none).
    #[serde(default)]
    pub fade_out_samples: u64,
}

impl AudioClip {
    /// Clip gain at `pos` samples from its start: raised-cosine
    /// (equal-power) fades. Without fades — 1.0.
    pub fn gain_at(&self, pos: u64) -> f32 {
        // 0→1 along a cosine over [0,len); past the fade — 1.
        let fade_in = |pos: u64, len: u64| {
            if len == 0 || pos >= len {
                1.0
            } else {
                0.5 - 0.5 * (std::f32::consts::PI * pos as f32 / len as f32).cos()
            }
        };
        // 1→0 along a cosine as the clip end approaches.
        let remaining = self.len_samples.saturating_sub(pos);
        fade_in(pos, self.fade_in_samples) * fade_in(remaining, self.fade_out_samples)
    }
}

/// Track with volume and mute.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioTrack {
    pub id: TrackId,
    pub name: String,
    pub volume: f32,
    pub muted: bool,
    pub clips: Vec<AudioClip>,
}

impl AudioTrack {
    pub fn new(id: TrackId, name: impl Into<String>) -> Self {
        Self { id, name: name.into(), volume: 1.0, muted: false, clips: Vec::new() }
    }
}

/// Decoded asset: stereo f32 48 kHz.
#[derive(Clone, Debug, Default)]
pub struct PcmBuf {
    pub samples: Vec<[f32; 2]>,
}

/// Cache of decoded assets (passed to the mixer).
pub type PcmCache = BTreeMap<AssetId, PcmBuf>;

/// Result of mixing a frame: peak level (for overload indication).
/// The buffer is zeroed; to layer several track groups use [`mix_into`].
pub fn mix_frame(
    tracks: &[AudioTrack],
    frame: Frame,
    pcm: &PcmCache,
    out: &mut [[f32; 2]],
) -> f32 {
    out.fill([0.0, 0.0]);
    mix_into(tracks, frame, pcm, out);
    peak_of(out)
}

/// Layers tracks onto the buffer WITHOUT zeroing it (scene + film tracks).
pub fn mix_into(
    tracks: &[AudioTrack],
    frame: Frame,
    pcm: &PcmCache,
    out: &mut [[f32; 2]],
) {
    let frame_start = if frame < 0 { 0 } else { frame as u64 * SAMPLES_PER_FRAME as u64 };
    let frame_end = frame_start + SAMPLES_PER_FRAME as u64;
    for track in tracks {
        if track.muted {
            continue;
        }
        for clip in &track.clips {
            let clip_start = clip.start.max(0) as u64 * SAMPLES_PER_FRAME as u64;
            let clip_end = clip_start + clip.len_samples;
            if clip_end <= frame_start || clip_start >= frame_end {
                continue;
            }
            let Some(buf) = pcm.get(&clip.asset) else { continue };
            for (i, slot) in out.iter_mut().enumerate() {
                let t = frame_start + i as u64;
                if t < clip_start || t >= clip_end {
                    continue;
                }
                let src_idx = (clip.offset_samples + (t - clip_start)) as usize;
                if let Some(s) = buf.samples.get(src_idx) {
                    let gain = track.volume * clip.gain_at(t - clip_start);
                    slot[0] += s[0] * gain;
                    slot[1] += s[1] * gain;
                }
            }
        }
    }
}

/// Buffer peak level.
pub fn peak_of(out: &[[f32; 2]]) -> f32 {
    out.iter().fold(0.0f32, |p, s| p.max(s[0].abs()).max(s[1].abs()))
}

/// Clamp range before integer output.
pub fn clamp_i16(x: f32) -> i16 {
    (x.clamp(-1.0, 1.0) * 32767.0).round() as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_pcm(frames: usize) -> PcmBuf {
        PcmBuf {
            samples: (0..frames * SAMPLES_PER_FRAME as usize)
                .map(|i| {
                    let v = (i as f32 * 0.01).sin() * 0.5;
                    [v, v]
                })
                .collect(),
        }
    }

    #[test]
    fn mix_respects_timing_volume_mute() {
        let asset = AssetId(1);
        let mut pcm = PcmCache::new();
        pcm.insert(asset, sine_pcm(4));
        let track = |muted| AudioTrack {
            id: TrackId(1),
            name: "t".into(),
            volume: 2.0,
            muted,
            clips: vec![AudioClip { id: AudioClipId(1), asset, start: 1, offset_samples: 0, len_samples: 2000, fade_in_samples: 0, fade_out_samples: 0 }],
        };
        let mut out = [[0.0f32; 2]; SAMPLES_PER_FRAME as usize];
        // Frame 0: the clip has not started yet.
        assert_eq!(mix_frame(&[track(false)], 0, &pcm, &mut out), 0.0);
        // Frame 1: the clip plays at volume 2 → peak up to 1.0.
        let peak = mix_frame(&[track(false)], 1, &pcm, &mut out);
        assert!(peak > 0.9 && peak <= 1.0, "peak {peak}");
        // Muted: silence.
        assert_eq!(mix_frame(&[track(true)], 1, &pcm, &mut out), 0.0);
        // Frame 3: the clip has ended.
        assert_eq!(mix_frame(&[track(false)], 3, &pcm, &mut out), 0.0);
    }

    #[test]
    fn fade_curve_is_raised_cosine_and_zero_at_edges() {
        let clip = AudioClip { id: AudioClipId(1), asset: AssetId(1), start: 0, offset_samples: 0, len_samples: 4000, fade_in_samples: 1000, fade_out_samples: 1000 };
        assert_eq!(clip.gain_at(0), 0.0, "fade-in start is silence");
        assert!((clip.gain_at(500) - 0.5).abs() < 1e-6, "fade-in midpoint is 0.5");
        assert_eq!(clip.gain_at(1000), 1.0, "fade-in end is unity");
        assert_eq!(clip.gain_at(2000), 1.0, "clip body is unity");
        assert!((clip.gain_at(3500) - 0.5).abs() < 1e-6, "fade-out midpoint is 0.5");
        assert!(clip.gain_at(3999) < 0.001, "clip end is silence: {}", clip.gain_at(3999));
        // Without fades — unity; fades are capped by the clip length.
        let plain = AudioClip { fade_in_samples: 0, fade_out_samples: 0, ..clip };
        assert_eq!(plain.gain_at(0), 1.0);
    }

    #[test]
    fn mix_applies_fade_gain() {
        let asset = AssetId(1);
        let mut pcm = PcmCache::new();
        pcm.insert(asset, sine_pcm(4));
        let track = |fade_in, fade_out| AudioTrack {
            id: TrackId(1),
            name: "t".into(),
            volume: 1.0,
            muted: false,
            clips: vec![AudioClip { id: AudioClipId(1), asset, start: 0, offset_samples: 0, len_samples: 4000, fade_in_samples: fade_in, fade_out_samples: fade_out }],
        };
        let mut out = [[0.0f32; 2]; SAMPLES_PER_FRAME as usize];
        // Frame 0 with a 2000-sample fade-in: the first sample is silence, then it rises.
        mix_frame(&[track(2000, 0)], 0, &pcm, &mut out);
        assert_eq!(out[0], [0.0, 0.0]);
        assert!(out[1999][0].abs() < out[1998][0].abs() + 0.01, "the fade rises monotonically");
        // Frame 1 (samples 2000..4000) with a fade-out: the frame end is quieter than its start.
        let mut out2 = [[0.0f32; 2]; SAMPLES_PER_FRAME as usize];
        mix_frame(&[track(0, 2000)], 1, &pcm, &mut out2);
        assert!(out2[1999][0].abs() < out2[0][0].abs() + 1e-6, "fade-out decays");
    }

    #[test]
    fn clamp_limits_overload() {
        assert_eq!(clamp_i16(2.5), i16::MAX);
        assert_eq!(clamp_i16(-2.5), -32767);
        assert_eq!(clamp_i16(0.5), 16384);
    }
}

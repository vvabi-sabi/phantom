//! Phantom motion clip sequence (`clip_track`): pure functions for
//! selecting the active clip at a frame and for validation.
//!
//! Frames are integers, 24 fps, ranges are half-open. Clips in the track do
//! not overlap (overlaps are rejected by command validation). Sampling inside
//! a clip: `(frame - range.start) * speed` clip frames from its start;
//! at the end — clamp to the clip's last frame (no looping — one performance
//! per range).
//!
//! Core/runtime split: core owns the schedule (which clip and which clip frame
//! we are on) but has no access to the clip's data. Root motion in
//! [`RootMotion::Clip`] mode (the clip's root-bone delta from the use start
//! frame) is applied in runtime, where `ClipData` lives; in
//! [`RootMotion::Controller`] mode the clip pose is applied as sampled (the
//! root track is not stripped), and the controller owns the root entity's
//! offset. The two trajectories are never summed: in Clip mode the controller
//! root is frozen at the use start frame, the root-bone translation in the
//! pose is at clip frame 0, and only the clip's delta on the root entity
//! produces movement.

use serde::{Deserialize, Serialize};

use crate::id::{AssetId, ClipUseId};
use crate::time::{Frame, FrameRange};

/// Who owns the horizontal root offset for the duration of the clip.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum RootMotion {
    /// The Phantom controller (keys): the controller owns the root entity's
    /// offset; the clip pose is applied as sampled.
    #[default]
    Controller,
    /// The clip's root motion: the root-bone delta from the use start frame
    /// goes to the root entity; the root-bone translation in the pose is
    /// frozen at clip frame 0 (so movement is not counted twice).
    Clip,
}

/// One clip use in a Phantom track.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClipUse {
    pub id: ClipUseId,
    /// The clip asset (MotionFbx).
    pub asset: AssetId,
    /// Animation stack name; `None` — the longest one (legacy behavior).
    #[serde(default)]
    pub stack: Option<String>,
    /// Frame range inside the scene (half-open).
    pub range: FrameRange,
    /// Playback speed (> 0, finite; 1.0 — original).
    #[serde(default = "default_speed")]
    pub speed: f32,
    /// Smooth in, seconds (>= 0): blending from the base pose.
    #[serde(default)]
    pub blend_in: f32,
    /// Smooth out, seconds (>= 0): blending to the base pose.
    #[serde(default)]
    pub blend_out: f32,
    /// Root motion owner for the duration of the clip.
    #[serde(default)]
    pub root_motion: RootMotion,
    /// Retargeting profile; `None` — direct bone-name matching
    /// (legacy behavior). Unmapped roles are diagnosed at load time;
    /// zero coverage is a `NoMatchingBones` error, as before.
    #[serde(default)]
    pub retarget: Option<crate::retarget::RetargetProfile>,
}

fn default_speed() -> f32 {
    1.0
}

/// Validates one use against the scene and the rest of the track.
pub fn validate_clip_use(u: &ClipUse, track: &[ClipUse], scene: &FrameRange) -> Result<(), crate::CoreError> {
    if u.range.start < scene.start || u.range.end > scene.end {
        return Err(crate::CoreError::ClipOutOfScene);
    }
    if !u.speed.is_finite() || u.speed <= 0.0 {
        return Err(crate::CoreError::InvalidClipUseParam("speed"));
    }
    if !u.blend_in.is_finite() || u.blend_in < 0.0 {
        return Err(crate::CoreError::InvalidClipUseParam("blend_in"));
    }
    if !u.blend_out.is_finite() || u.blend_out < 0.0 {
        return Err(crate::CoreError::InvalidClipUseParam("blend_out"));
    }
    for other in track {
        if other.id == u.id {
            continue;
        }
        if u.range.start < other.range.end && other.range.start < u.range.end {
            return Err(crate::CoreError::ClipUseOverlap(other.id));
        }
    }
    Ok(())
}

/// The active use at a frame (the track does not overlap — at most one).
pub fn active_clip_use(track: &[ClipUse], frame: Frame) -> Option<&ClipUse> {
    track.iter().find(|u| u.range.contains(frame))
}

/// The neighbouring use that ends flush before `u` starts (for blending
/// at the seam): its range ends exactly at `u.range.start`.
pub fn contiguous_prev<'a>(track: &'a [ClipUse], u: &ClipUse) -> Option<&'a ClipUse> {
    track.iter().find(|o| o.id != u.id && o.range.end == u.range.start)
}

/// The neighbouring use that starts flush after `u` ends.
pub fn contiguous_next<'a>(track: &'a [ClipUse], u: &ClipUse) -> Option<&'a ClipUse> {
    track.iter().find(|o| o.id != u.id && o.range.start == u.range.end)
}

/// Clip frame (fractional, from the clip start) for scene frame `frame`
/// inside use `u`. Speed scales local time. The upper bound (the clip's
/// frame count) is only known in runtime — the clamp happens there.
pub fn clip_frame_at(u: &ClipUse, frame: Frame) -> f32 {
    (frame - u.range.start) as f32 * u.speed
}

/// Blend weights at a frame: `(w_in, w_out)`, both in [0, 1]; 1 — full clip.
/// Blend windows live inside the use's range.
pub fn blend_weights(u: &ClipUse, frame: Frame) -> (f32, f32) {
    let w_in = if u.blend_in > 0.0 {
        ((frame - u.range.start) as f32 / (u.blend_in * crate::time::FPS as f32)).min(1.0)
    } else {
        1.0
    };
    let w_out = if u.blend_out > 0.0 {
        ((u.range.end - 1 - frame) as f32 / (u.blend_out * crate::time::FPS as f32)).min(1.0)
    } else {
        1.0
    };
    (w_in.clamp(0.0, 1.0), w_out.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(id: u64, start: Frame, end: Frame) -> ClipUse {
        ClipUse {
            id: ClipUseId(id),
            asset: AssetId(1),
            stack: None,
            range: FrameRange::new(start, end).unwrap(),
            speed: 1.0,
            blend_in: 0.0,
            blend_out: 0.0,
            root_motion: RootMotion::Controller,
            retarget: None,
        }
    }

    #[test]
    fn selects_active_use() {
        let track = vec![u(1, 0, 24), u(2, 48, 96)];
        assert_eq!(active_clip_use(&track, 0).unwrap().id.0, 1);
        assert_eq!(active_clip_use(&track, 23).unwrap().id.0, 1);
        assert!(active_clip_use(&track, 24).is_none(), "between clips — manual keys");
        assert_eq!(active_clip_use(&track, 48).unwrap().id.0, 2);
        assert!(active_clip_use(&track, 96).is_none(), "half-open range");
    }

    #[test]
    fn speed_maps_clip_frame() {
        let mut x = u(1, 10, 50);
        x.speed = 2.0;
        assert_eq!(clip_frame_at(&x, 10), 0.0);
        assert_eq!(clip_frame_at(&x, 20), 20.0);
    }

    #[test]
    fn overlap_rejected() {
        let track = vec![u(1, 0, 24)];
        assert!(validate_clip_use(&u(2, 24, 48), &track, &FrameRange::new(0, 96).unwrap()).is_ok(), "a flush seam is not an overlap");
        assert!(matches!(
            validate_clip_use(&u(2, 12, 48), &track, &FrameRange::new(0, 96).unwrap()),
            Err(crate::CoreError::ClipUseOverlap(_))
        ));
    }

    #[test]
    fn params_validated() {
        let scene = FrameRange::new(0, 96).unwrap();
        let mut x = u(2, 0, 24);
        x.speed = 0.0;
        assert!(matches!(validate_clip_use(&x, &[], &scene), Err(crate::CoreError::InvalidClipUseParam("speed"))));
        x.speed = f32::NAN;
        assert!(matches!(validate_clip_use(&x, &[], &scene), Err(crate::CoreError::InvalidClipUseParam("speed"))));
        x.speed = 1.0;
        x.blend_in = -0.5;
        assert!(matches!(validate_clip_use(&x, &[], &scene), Err(crate::CoreError::InvalidClipUseParam("blend_in"))));
        x.blend_in = 0.0;
        x.range = FrameRange::new(90, 100).unwrap();
        assert!(matches!(validate_clip_use(&x, &[], &scene), Err(crate::CoreError::ClipOutOfScene)));
    }

    #[test]
    fn blend_windows() {
        let scene_fps = crate::time::FPS as i64;
        let mut x = u(1, 0, 10 * scene_fps);
        x.blend_in = 1.0;
        x.blend_out = 1.0;
        assert_eq!(blend_weights(&x, 0), (0.0, 1.0));
        assert_eq!(blend_weights(&x, scene_fps), (1.0, 1.0));
        let (wi, wo) = blend_weights(&x, scene_fps / 2);
        assert!((wi - 0.5).abs() < 1e-6);
        assert_eq!(wo, 1.0);
        let (_, wo_end) = blend_weights(&x, 10 * scene_fps - 1);
        assert_eq!(wo_end, 0.0, "the last frame is fully the base pose with blend_out");
    }

    #[test]
    fn contiguous_neighbors() {
        let track = vec![u(1, 0, 24), u(2, 24, 48), u(3, 72, 96)];
        assert_eq!(contiguous_prev(&track, &track[1]).unwrap().id.0, 1);
        assert!(contiguous_next(&track, &track[1]).is_none(), "a gap is not a seam");
    }
}

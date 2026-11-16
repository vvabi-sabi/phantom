//! Phantom state at a frame: a pure function of the frame,
//! independent of wall clock and call order (seek == sequential
//! playback).

use glam::Vec3;

use crate::project::{Action, Phantom};
use crate::rig::{self, Pose};
use crate::time::{frame_to_secs, Frame, FrameRange};

/// Computed Phantom state at a frame.
#[derive(Clone, Debug)]
pub struct PhantomPose {
    /// Controller root position in the scene (y >= 0, the floor clamps downward).
    pub root: Vec3,
    /// Root rotation about +Y, radians.
    pub yaw: f32,
    /// Current action.
    pub action: Action,
    /// Skeleton pose in controller space.
    pub pose: Pose,
}

/// Computes the Phantom state at `frame`.
///
/// `on_twos` — "on twos" mode: pose and position quantize to even frames
/// from the scene start; the camera and background are not affected.
pub fn evaluate(phantom: &Phantom, scene: &FrameRange, frame: Frame, on_twos: bool) -> PhantomPose {
    let frame = if on_twos {
        crate::time::on_twos(frame, scene.start)
    } else {
        frame
    };

    // The active take overrides the manual track within its range.
    let keys = phantom
        .active_take
        .and_then(|id| phantom.takes.iter().find(|t| t.id == id))
        .filter(|t| t.range.contains(frame))
        .map(|t| &t.motion)
        .unwrap_or(&phantom.motion);

    let (root, yaw, mut action, t_action) = sample_motion(keys, frame);
    let pose = if let Some(actor) = phantom.rig.builtin_actor() {
        // Empty/pre-roll tracks use the profile's rest action. Invalid externally
        // authored documents remain safe; command and renderer validation diagnose them.
        if keys.first().is_none_or(|k| frame < k.frame) || !actor.actions().contains(&action) {
            action = actor.default_action();
        }
        rig::sample_actor_action(actor, action, t_action).expect("supported action")
    } else {
        rig::sample_action(action, t_action)
    };
    PhantomPose { root, yaw, action, pose }
}

/// Interpolates controller keys. Returns (position, action,
/// local action time in seconds).
fn sample_motion(keys: &[crate::project::MotionKey], frame: Frame) -> (Vec3, f32, Action, f32) {
    if keys.is_empty() {
        return (Vec3::ZERO, 0.0, Action::Idle, 0.0);
    }

    // Position: linear interpolation between neighbouring keys; before the
    // first key and after the last one — we stand where that key stands.
    let idx = match keys.binary_search_by_key(&frame, |k| k.frame) {
        Ok(i) => i,
        Err(0) => {
            return (floor(keys[0].position), keys[0].yaw, Action::Idle, 0.0);
        }
        Err(i) => i - 1,
    };
    let from = keys[idx];
    let (pos, yaw) = match keys.get(idx + 1) {
        Some(to) => {
            let span = (to.frame - from.frame) as f32;
            let alpha = (frame - from.frame) as f32 / span;
            (floor(from.position.lerp(to.position, alpha)), shortest_angle_lerp(from.yaw, to.yaw, alpha))
        }
        None => (floor(from.position), from.yaw),
    };

    // Action: idle before the first key; after that a key's action starts
    // at the middle of the transition from the previous key (smooth switching).
    // Local action time runs from the last such boundary.
    let mut seg_start = keys[0].frame;
    let mut action = Action::Idle;
    if keys[0].frame <= frame {
        action = keys[0].action;
    }
    for w in keys.windows(2) {
        let mid = w[0].frame + (w[1].frame - w[0].frame) / 2;
        if mid <= frame {
            seg_start = mid;
            action = w[1].action;
        } else {
            break;
        }
    }
    let t_action = frame_to_secs((frame - seg_start).max(0)) as f32;
    (pos, yaw, action, t_action)
}

/// The floor clamps movement downward.
fn floor(p: Vec3) -> Vec3 {
    Vec3::new(p.x, p.y.max(0.0), p.z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::{MotionKey, RigRef};

    fn phantom(keys: Vec<MotionKey>) -> Phantom {
        Phantom {
            id: crate::id::PhantomId(1),
            name: "p".into(),
            rig: RigRef::Builtin,
            motion: keys,
            clip: None,
            clip_track: Vec::new(),
            takes: Vec::new(),
            active_take: None,
            outline_px: 2.0,
            scale: 1.0,
            material_patches: Vec::new(),
        }
    }

    #[test]
    fn seek_matches_playback() {
        let p = phantom(vec![
            MotionKey { yaw: 0.0, frame: 0, position: Vec3::ZERO, action: Action::Idle },
            MotionKey { yaw: 0.0, frame: 48, position: Vec3::new(2.0, 0.0, 0.0), action: Action::Walk },
        ]);
        let scene = FrameRange::new(0, 96).unwrap();
        let mut last = None;
        for f in 0..96 {
            let e = evaluate(&p, &scene, f, false);
            if let Some(prev) = last {
                // Continuity: the root does not teleport between adjacent frames.
                let prev: Vec3 = prev;
                assert!((e.root - prev).length() < 0.2, "jump at frame {f}");
            }
            last = Some(e.root);
        }
        // A direct seek gives the same values.
        for f in [0, 1, 24, 47, 48, 95] {
            let a = evaluate(&p, &scene, f, false);
            let b = evaluate(&p, &scene, f, false);
            assert_eq!(a.root, b.root);
        }
        // End of movement.
        let end = evaluate(&p, &scene, 48, false);
        assert!((end.root - Vec3::new(2.0, 0.0, 0.0)).length() < 1e-4);
        assert_eq!(end.action, Action::Walk);
    }

    #[test]
    fn twos_quantize_pose_but_not_below_floor() {
        let p = phantom(vec![
            MotionKey { yaw: 0.0, frame: 0, position: Vec3::new(0.0, -5.0, 0.0), action: Action::Idle },
            MotionKey { yaw: 0.0, frame: 24, position: Vec3::new(1.0, 0.0, 0.0), action: Action::Walk },
        ]);
        let scene = FrameRange::new(0, 48).unwrap();
        let a = evaluate(&p, &scene, 10, true);
        let b = evaluate(&p, &scene, 11, true);
        assert_eq!(a.root, b.root, "on twos adjacent odd/even frames match");
        assert_eq!(a.root.y, 0.0, "the floor clamps movement downward");
        let c = evaluate(&p, &scene, 12, true);
        assert_ne!(a.root, c.root);
    }
}

/// Shortest arc; an exact half-turn deterministically takes the negative arc.
pub fn shortest_angle_lerp(from: f32, to: f32, alpha: f32) -> f32 {
    let pi = std::f64::consts::PI;
    let delta = (to as f64 - from as f64 + pi).rem_euclid(2.0 * pi) - pi;
    (from as f64 + delta * alpha as f64) as f32
}

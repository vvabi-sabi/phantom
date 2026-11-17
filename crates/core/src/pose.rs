//! Phantom state at a frame: a pure function of the frame,
//! independent of wall clock and call order (seek == sequential
//! playback).

use glam::Vec3;

use crate::project::{ActorPose, Phantom};
use crate::rig::{self, Pose};
use crate::time::Frame;

/// Computed Phantom state at a frame.
#[derive(Clone, Debug)]
pub struct PhantomPose {
    /// Controller root position in the scene (y >= 0, the floor clamps downward).
    pub root: Vec3,
    /// Root rotation about +Y, radians.
    pub yaw: f32,
    /// The pose the actor holds at this frame.
    pub pose: ActorPose,
    /// Skeleton pose in controller space. For an imported rig this is the
    /// humanoid rest — an imported model has no named poses and the renderer
    /// keeps its bind pose, so nothing here reaches the frame.
    pub skeleton: Pose,
}

/// Computes the Phantom state at `frame`.
///
/// The pose switches at its key — a held pose is placed, not played, so there
/// is no half-way blend to hide. Position and yaw interpolate between keys and
/// hold outside them.
pub fn evaluate(phantom: &Phantom, frame: Frame) -> PhantomPose {
    let (root, yaw, pose) = sample_motion(&phantom.motion, frame);
    let skeleton = match phantom.rig.builtin_actor() {
        Some(actor) => rig::sample_actor_pose(actor, pose)
            // A hand-written document may name a pose the actor does not have;
            // the commands refuse it, and an external file falls back to the
            // standing pose instead of failing the frame.
            .unwrap_or_else(|_| rig::sample_actor_pose(actor, actor.default_pose()).expect("the default pose is always valid")),
        None => rig::Pose::rest(),
    };
    PhantomPose { root, yaw, pose, skeleton }
}

/// Interpolates controller keys: position and yaw between neighbours, the pose
/// of the key in force.
fn sample_motion(keys: &[crate::project::MotionKey], frame: Frame) -> (Vec3, f32, ActorPose) {
    if keys.is_empty() {
        return (Vec3::ZERO, 0.0, ActorPose::Stand);
    }
    let idx = match keys.binary_search_by_key(&frame, |k| k.frame) {
        Ok(i) => i,
        // Before the first key the value is held, not extrapolated backwards.
        Err(0) => return (floor(keys[0].position), keys[0].yaw, keys[0].pose),
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
    (pos, yaw, from.pose)
}

/// The floor clamps movement downward.
fn floor(p: Vec3) -> Vec3 {
    Vec3::new(p.x, p.y.max(0.0), p.z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::{ActorPose, MotionKey, RigRef};

    fn phantom(keys: Vec<MotionKey>) -> Phantom {
        Phantom {
            id: crate::id::PhantomId(1),
            name: "p".into(),
            rig: RigRef::Builtin,
            motion: keys,
            outline_px: 2.0,
            scale: 1.0,
            material_patches: Vec::new(),
        }
    }

    fn key(frame: Frame, position: Vec3, pose: ActorPose) -> MotionKey {
        MotionKey { yaw: 0.0, frame, position, pose }
    }

    #[test]
    fn seek_matches_playback() {
        let p = phantom(vec![
            key(0, Vec3::ZERO, ActorPose::Stand),
            key(48, Vec3::new(2.0, 0.0, 0.0), ActorPose::Sit),
        ]);
        let mut last = None;
        for f in 0..96 {
            let e = evaluate(&p, f);
            if let Some(prev) = last {
                // Continuity: the root does not teleport between adjacent frames.
                let prev: Vec3 = prev;
                assert!((e.root - prev).length() < 0.2, "jump at frame {f}");
            }
            last = Some(e.root);
        }
        // A direct seek gives the same values.
        for f in [0, 1, 24, 47, 48, 95] {
            assert_eq!(evaluate(&p, f).root, evaluate(&p, f).root);
        }
        // End of movement, and the pose in force is the key's own.
        let end = evaluate(&p, 48);
        assert!((end.root - Vec3::new(2.0, 0.0, 0.0)).length() < 1e-4);
        assert_eq!(end.pose, ActorPose::Sit);
        assert_eq!(end.skeleton.bones[0].translation.y, 0.95 - 0.50, "the sitting pose reaches the skeleton");
        // The pose switches at its key, not half-way.
        assert_eq!(evaluate(&p, 47).pose, ActorPose::Stand);
        assert_eq!(evaluate(&p, 48).pose, ActorPose::Sit);
    }

    #[test]
    fn the_floor_clamps_movement_and_an_empty_track_stands() {
        let p = phantom(vec![
            key(0, Vec3::new(0.0, -5.0, 0.0), ActorPose::Lie),
            key(24, Vec3::new(1.0, 0.0, 0.0), ActorPose::Stand),
        ]);
        // Below the floor at the first key: the root is held at ground level.
        for f in [0, 6, 12] {
            assert_eq!(evaluate(&p, f).root.y, 0.0, "frame {f}");
            assert_eq!(evaluate(&p, f).pose, ActorPose::Lie);
        }
        // An empty track: the actor stands at the origin in the default pose.
        let empty = phantom(Vec::new());
        let e = evaluate(&empty, 7);
        assert_eq!(e.root, Vec3::ZERO);
        assert_eq!(e.pose, ActorPose::Stand);
    }

    /// An imported rig has no named poses: it renders in bind pose, so nothing
    /// in the skeleton matters — but evaluation must not fail.
    #[test]
    fn an_imported_rig_evaluates_to_the_rest_skeleton() {
        let mut p = phantom(vec![key(0, Vec3::new(1.0, 0.0, 2.0), ActorPose::Stand)]);
        p.rig = RigRef::Imported { asset: crate::id::AssetId(3) };
        let e = evaluate(&p, 0);
        assert_eq!(e.root, Vec3::new(1.0, 0.0, 2.0));
        assert_eq!(e.skeleton.bones.len(), crate::rig::BONES.len());
    }
}

/// Shortest arc; an exact half-turn deterministically takes the negative arc.
pub fn shortest_angle_lerp(from: f32, to: f32, alpha: f32) -> f32 {
    let pi = std::f64::consts::PI;
    let delta = (to as f64 - from as f64 + pi).rem_euclid(2.0 * pi) - pi;
    (from as f64 + delta * alpha as f64) as f32
}

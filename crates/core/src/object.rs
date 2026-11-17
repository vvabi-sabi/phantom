//! Scene object transform at a frame: a pure function of the frame,
//! shared by render, GUI and tests (pose-style, like Phantoms).

use glam::Vec3;

use crate::pose::shortest_angle_lerp;
use crate::project::{ObjectKey, SceneObject};
use crate::time::Frame;

/// Computed object transform at a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObjectTransform {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub scale: Vec3,
}

impl Default for ObjectTransform {
    fn default() -> Self {
        Self { position: Vec3::ZERO, yaw: 0.0, pitch: 0.0, roll: 0.0, scale: Vec3::ONE }
    }
}

/// Computes the object transform at `frame`.
///
/// Position and scale interpolate linearly, angles along the shortest arc;
/// the value is held before the first and after the last key. An empty
/// track — the original transform (zero position, unit scale).
pub fn evaluate_object(object: &SceneObject, frame: Frame) -> ObjectTransform {
    sample_keys(&object.keys, frame)
}

/// Samples transform keys: linear interpolation, hold outside the range.
pub fn sample_keys(keys: &[ObjectKey], frame: Frame) -> ObjectTransform {
    let to_t = |k: &ObjectKey| ObjectTransform { position: k.position, yaw: k.yaw, pitch: k.pitch, roll: k.roll, scale: k.scale };
    let Some(first) = keys.first() else {
        return ObjectTransform::default();
    };
    let idx = match keys.binary_search_by_key(&frame, |k| k.frame) {
        Ok(i) => return to_t(&keys[i]),
        Err(0) => return to_t(first),
        Err(i) => i - 1,
    };
    let from = &keys[idx];
    let Some(to) = keys.get(idx + 1) else {
        return to_t(from);
    };
    let alpha = (frame - from.frame) as f32 / (to.frame - from.frame) as f32;
    ObjectTransform {
        position: from.position.lerp(to.position, alpha),
        yaw: shortest_angle_lerp(from.yaw, to.yaw, alpha),
        pitch: shortest_angle_lerp(from.pitch, to.pitch, alpha),
        roll: shortest_angle_lerp(from.roll, to.roll, alpha),
        scale: from.scale.lerp(to.scale, alpha),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{AssetId, ObjectId};

    fn key(frame: Frame, x: f32, yaw: f32) -> ObjectKey {
        ObjectKey { frame, position: Vec3::new(x, 0.0, 0.0), yaw, pitch: 0.0, roll: 0.0, scale: Vec3::ONE }
    }

    fn object(keys: Vec<ObjectKey>) -> SceneObject {
        SceneObject { id: ObjectId(1), name: "o".into(), asset: AssetId(1), builtin: None, keys , attach: None, material_patches: Vec::new(), scatter: None }
    }

    #[test]
    fn lerp_between_keys() {
        let keys = vec![key(0, 0.0, 0.0), key(10, 2.0, 1.0)];
        let t = sample_keys(&keys, 5);
        assert!((t.position.x - 1.0).abs() < 1e-6);
        assert!((t.yaw - 0.5).abs() < 1e-6);
    }

    #[test]
    fn hold_outside_range_and_empty_is_identity() {
        let keys = vec![key(10, 1.0, 0.3), key(20, 3.0, 0.5)];
        assert_eq!(sample_keys(&keys, 0).position.x, 1.0, "hold before the first key");
        assert_eq!(sample_keys(&keys, 99).position.x, 3.0, "hold after the last key");
        let t = sample_keys(&[], 7);
        assert_eq!(t, ObjectTransform::default());
    }

    #[test]
    fn rotation_takes_shortest_arc() {
        // 0 -> 3π/2 goes through the negative arc (-π/2), midpoint = -π/4.
        let keys = vec![key(0, 0.0, 0.0), key(10, 0.0, std::f32::consts::FRAC_PI_2 * 3.0)];
        let t = sample_keys(&keys, 5);
        assert!((t.yaw + std::f32::consts::FRAC_PI_4).abs() < 1e-5, "yaw={}", t.yaw);
    }

    #[test]
    fn interpolation_is_free_between_keys() {
        let obj = object(vec![key(0, 0.0, 0.0), key(24, 2.4, 0.0)]);
        assert_ne!(evaluate_object(&obj, 10), evaluate_object(&obj, 11));
        assert!(evaluate_object(&obj, 12).position.x > evaluate_object(&obj, 11).position.x);
    }
}

//! Built-in humanoid rig and procedural movements (idle/walk/jump).
//!
//! The skeleton profile is documented and stable: the bone names in [`BONES`]
//! are used when checking imported clips for compatibility.
//! Right-handed coordinates, Y up, units are meters.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

/// Bone description: name, parent index (None for the root), rest translation
/// relative to the parent.
pub struct BoneDef {
    pub name: &'static str,
    pub parent: Option<usize>,
    pub rest_translation: Vec3,
}

pub const HIPS: usize = 0;

/// Built-in rig profile. The order is topological (parent before child).
pub static BONES: &[BoneDef] = &[
    BoneDef { name: "hips", parent: None, rest_translation: Vec3::new(0.0, 0.95, 0.0) },
    BoneDef { name: "spine", parent: Some(0), rest_translation: Vec3::new(0.0, 0.12, 0.0) },
    BoneDef { name: "head", parent: Some(1), rest_translation: Vec3::new(0.0, 0.45, 0.0) },
    BoneDef { name: "upper_arm_l", parent: Some(1), rest_translation: Vec3::new(0.18, 0.38, 0.0) },
    BoneDef { name: "lower_arm_l", parent: Some(3), rest_translation: Vec3::new(0.25, 0.0, 0.0) },
    BoneDef { name: "upper_arm_r", parent: Some(1), rest_translation: Vec3::new(-0.18, 0.38, 0.0) },
    BoneDef { name: "lower_arm_r", parent: Some(5), rest_translation: Vec3::new(-0.25, 0.0, 0.0) },
    BoneDef { name: "upper_leg_l", parent: Some(0), rest_translation: Vec3::new(0.09, -0.05, 0.0) },
    BoneDef { name: "lower_leg_l", parent: Some(7), rest_translation: Vec3::new(0.0, -0.4, 0.0) },
    BoneDef { name: "upper_leg_r", parent: Some(0), rest_translation: Vec3::new(-0.09, -0.05, 0.0) },
    BoneDef { name: "lower_leg_r", parent: Some(9), rest_translation: Vec3::new(0.0, -0.4, 0.0) },
];

/// Local pose of a single bone.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct BonePose {
    pub translation: Vec3,
    pub rotation: Quat,
}

/// Parent⋅child composition for (translation, rotation).
pub fn combine_tr(parent: (Vec3, Quat), local: (Vec3, Quat)) -> (Vec3, Quat) {
    (parent.0 + parent.1 * local.0, parent.1 * local.1)
}

/// World (controller-space) bone transforms: hierarchy accumulation. The order
/// is topological (parent before child), as in [`BONES`] and in import
/// SkinData. The basis for attaching props/lights to bones (corrective B2).
pub fn accumulate_bones(parents: &[Option<usize>], locals: &[(Vec3, Quat)]) -> Vec<(Vec3, Quat)> {
    let mut out = Vec::with_capacity(locals.len());
    for (i, local) in locals.iter().enumerate() {
        let world = match parents.get(i).copied().flatten() {
            Some(p) if p < out.len() => combine_tr(out[p], *local),
            _ => *local,
        };
        out.push(world);
    }
    out
}

/// Bone index by name in a hierarchy (None if the bone is missing).
pub fn bone_index(names: &[&str], bone: &str) -> Option<usize> {
    names.iter().position(|n| n.eq_ignore_ascii_case(bone))
}

/// Full rig pose in Phantom controller space.
#[derive(Clone, Debug)]
pub struct Pose {
    /// Local bone transforms in [`BONES`] order.
    pub bones: Vec<BonePose>,
}

impl Pose {
    /// Rest pose.
    pub fn rest() -> Self {
        Self {
            bones: BONES
                .iter()
                .map(|b| BonePose {
                    translation: b.rest_translation,
                    rotation: Quat::IDENTITY,
                })
                .collect(),
        }
    }
}

/// Walk cycle duration in seconds (two steps).
pub const WALK_CYCLE_SECS: f32 = 1.0;
/// Jump duration in seconds.
pub const JUMP_SECS: f32 = 0.8;

fn rot_x(angle: f32) -> Quat {
    Quat::from_rotation_x(angle)
}

/// Arms hang down along the body (the rest pose is a T-pose).
const ARMS_DOWN: f32 = 1.25;

/// Procedural action pose at local time `t` (seconds since the action started).
///
/// Clip root motion is kept apart from the controller: the controller sets the
/// horizontal offset, the clip only adds a local hips offset (the walk bounce,
/// the jump arc).
pub fn sample_action(action: crate::project::Action, t: f32) -> Pose {
    use crate::project::Action;
    let mut pose = Pose::rest();
    match action {
        Action::Idle => {
            let s = (t * std::f32::consts::TAU / 3.0).sin();
            // Slight breathing: the spine sways; arms hang down with a small bob.
            pose.bones[1].rotation = rot_x(0.02 * s);
            pose.bones[3].rotation = Quat::from_rotation_z(-ARMS_DOWN - 0.04 * s);
            pose.bones[5].rotation = Quat::from_rotation_z(ARMS_DOWN + 0.04 * s);
        }
        Action::Walk => {
            let phase = t * std::f32::consts::TAU / WALK_CYCLE_SECS;
            let swing = phase.sin();
            // Legs in counter-phase.
            pose.bones[7].rotation = rot_x(0.55 * swing);
            pose.bones[9].rotation = rot_x(-0.55 * swing);
            // Knee bend during the leg swing.
            pose.bones[8].rotation = rot_x(-0.5 * (phase - 1.2).sin().max(0.0));
            pose.bones[10].rotation = rot_x(-0.5 * (phase + std::f32::consts::PI - 1.2).sin().max(0.0));
            // Arms hang down and swing counter-phase to the legs.
            pose.bones[3].rotation = rot_x(-0.4 * swing) * Quat::from_rotation_z(-ARMS_DOWN);
            pose.bones[5].rotation = rot_x(0.4 * swing) * Quat::from_rotation_z(ARMS_DOWN);
            // Root bounce: a local hips offset, not controller movement.
            pose.bones[HIPS].translation.y += 0.03 * (2.0 * phase).cos();
        }
        Action::Sit | Action::Lie | Action::Sneak | Action::Trot | Action::Gallop | Action::Fly | Action::Perch => {
            return sample_action(Action::Idle, t)
        }
        Action::Jump => {
            let p = (t / JUMP_SECS).clamp(0.0, 1.0);
            // Crouch → extend → airborne → land.
            let crouch = if p < 0.25 {
                (p / 0.25) * 0.5
            } else if p < 0.45 {
                0.5 * (1.0 - (p - 0.25) / 0.2)
            } else {
                0.0
            };
            let airborne = ((p - 0.3) / 0.5).clamp(0.0, 1.0);
            let arc = (airborne * std::f32::consts::PI).sin();
            pose.bones[HIPS].translation.y += 0.45 * arc - 0.25 * crouch;
            let tuck = rot_x(0.7 * arc + 0.6 * crouch);
            pose.bones[7].rotation = tuck;
            pose.bones[9].rotation = tuck;
            pose.bones[8].rotation = rot_x(-1.0 * arc - 0.8 * crouch);
            pose.bones[10].rotation = rot_x(-1.0 * arc - 0.8 * crouch);
            pose.bones[3].rotation = Quat::from_rotation_z(1.2 * arc);
            pose.bones[5].rotation = Quat::from_rotation_z(-1.2 * arc);
        }
    }
    pose
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bones_are_topologically_sorted() {
        for (i, b) in BONES.iter().enumerate() {
            if let Some(p) = b.parent {
                assert!(p < i, "parent {} must come before index {}", b.name, i);
            }
        }
    }

    #[test]
    fn walk_cycle_loops_smoothly() {
        let a = sample_action(crate::project::Action::Walk, 0.0);
        let b = sample_action(crate::project::Action::Walk, WALK_CYCLE_SECS);
        for (x, y) in a.bones.iter().zip(b.bones.iter()) {
            assert!((x.translation - y.translation).length() < 1e-5);
            assert!((x.rotation - y.rotation).length() < 1e-5);
        }
    }
}

/// Original bundled skeletons. Quadrupeds share topology, not proportions.
/// All rest rotations are identity, parents precede children.
pub fn actor_bones(actor: crate::project::BuiltinActor) -> Vec<BoneDef> {
    use crate::project::BuiltinActor::*;
    if actor == Humanoid {
        return BONES.iter().map(|b| BoneDef { name: b.name, parent: b.parent, rest_translation: b.rest_translation }).collect();
    }
    let mut bones = Vec::new();
    let mut add = |name, parent, x, y, z| bones.push(BoneDef { name, parent, rest_translation: Vec3::new(x, y, z) });
    if actor == Bird {
        add("body", None, 0., 0.32, 0.);
        add("head", Some(0), 0., 0.16, 0.23);
        add("wing_l", Some(0), 0.12, 0.04, 0.);
        add("wing_tip_l", Some(2), 0.32, 0., 0.);
        add("wing_r", Some(0), -0.12, 0.04, 0.);
        add("wing_tip_r", Some(4), -0.32, 0., 0.);
        add("tail", Some(0), 0., 0., -0.25);
        add("foot_l", Some(0), 0.07, -0.22, 0.);
        add("foot_r", Some(0), -0.07, -0.22, 0.);
    } else {
        let horse = actor == Horse;
        let (height, length, width, leg) = if horse { (1.25, 0.65, 0.23, 0.55) } else { (0.65, 0.43, 0.16, 0.28) };
        add("body", None, 0., height, 0.);
        add("neck", Some(0), 0., 0.12, length);
        add("head", Some(1), 0., if horse { 0.48 } else { 0.10 }, 0.16);
        add("tail", Some(0), 0., 0.02, -length);
        for (name, lower, x, z) in [
            ("front_l", "front_lower_l", width, length * 0.8),
            ("front_r", "front_lower_r", -width, length * 0.8),
            ("hind_l", "hind_lower_l", width, -length * 0.8),
            ("hind_r", "hind_lower_r", -width, -length * 0.8),
        ] {
            let index = match name {
                "front_l" => 4,
                "front_r" => 6,
                "hind_l" => 8,
                _ => 10,
            };
            add(name, Some(0), x, -0.1, z);
            add(lower, Some(index), 0., -leg, 0.);
        }
    }
    bones
}

/// Profile-checked, stateless animation. No horizontal root motion or yaw.
pub fn sample_actor_action(
    actor: crate::project::BuiltinActor,
    action: crate::project::Action,
    t: f32,
) -> Result<Pose, crate::error::CoreError> {
    use crate::project::{Action::*, BuiltinActor};
    crate::project::RigRef::BuiltinActor { actor }.validate_action(action)?;
    let t = if t.is_finite() { t } else { 0.0 };
    if actor == BuiltinActor::Humanoid {
        return Ok(sample_action(action, t));
    }
    let mut pose =
        Pose { bones: actor_bones(actor).iter().map(|b| BonePose { translation: b.rest_translation, rotation: Quat::IDENTITY }).collect() };
    let phase = t as f64 * std::f64::consts::TAU;
    let wave = |rate: f64, offset: f64| (phase * rate + offset).sin() as f32;
    match action {
        Idle => {
            pose.bones[1].rotation = rot_x(0.025 * wave(0.3, 0.));
            pose.bones[3].rotation = Quat::from_rotation_y(0.12 * wave(0.5, 0.));
        }
        Walk | Sneak | Trot | Gallop => {
            let (rate, amplitude) = match action {
                Sneak => (0.55, 0.25),
                Trot => (1.6, 0.65),
                Gallop => (2.1, 0.9),
                _ => (1., 0.5),
            };
            for (i, offset) in [0., std::f64::consts::PI, std::f64::consts::PI, 0.].into_iter().enumerate() {
                let offset = if action == Gallop { [0., 0.4, 2.0, 2.4][i] } else { offset };
                let swing = wave(rate, offset);
                pose.bones[4 + i * 2].rotation = rot_x(amplitude * swing);
                pose.bones[5 + i * 2].rotation = rot_x(-0.65 * swing.max(0.));
            }
            pose.bones[0].translation.y += if action == Sneak { -0.12 } else { 0.035 * wave(rate * 2., 0.) };
            pose.bones[1].rotation = rot_x(if action == Sneak { 0.18 } else { 0.04 * wave(rate, 0.) });
        }
        Sit => {
            pose.bones[0].translation.y -= 0.22;
            pose.bones[0].rotation = rot_x(-0.35);
            for i in [8, 10] {
                pose.bones[i].rotation = rot_x(-1.15);
                pose.bones[i + 1].rotation = rot_x(1.7);
            }
            pose.bones[1].rotation = rot_x(0.35);
        }
        Lie => {
            pose.bones[0].translation.y = 0.23;
            for i in [4, 6, 8, 10] {
                pose.bones[i].rotation = rot_x(1.35);
                pose.bones[i + 1].rotation = rot_x(-1.7);
            }
            pose.bones[1].rotation = rot_x(0.25);
        }
        Fly => {
            for (i, sign) in [(2, 1.), (4, -1.)] {
                pose.bones[i].rotation = Quat::from_rotation_z(sign * 0.8 * wave(2.5, 0.));
                pose.bones[i + 1].rotation = Quat::from_rotation_z(sign * 0.3 * wave(2.5, -0.7));
            }
            for i in [7, 8] {
                pose.bones[i].rotation = rot_x(-1.1);
            }
            pose.bones[6].rotation = rot_x(0.08 * wave(2.5, 0.));
        }
        Perch => {
            for (i, sign) in [(2, 1.), (4, -1.)] {
                pose.bones[i].rotation = Quat::from_rotation_y(sign * 1.25) * Quat::from_rotation_z(-sign * 0.65);
                pose.bones[i + 1].rotation = Quat::from_rotation_y(sign * 0.8);
            }
            pose.bones[1].rotation = rot_x(0.04 * wave(0.3, 0.));
        }
        Jump => unreachable!("validated humanoid action handled above"),
    }
    Ok(pose)
}

#[cfg(test)]
mod attach_tests {
    use super::*;

    #[test]
    fn accumulate_bones_composes_hierarchy() {
        let bones = actor_bones(crate::project::BuiltinActor::Humanoid);
        let parents: Vec<Option<usize>> = bones.iter().map(|b| b.parent).collect();
        // Rest pose: with no rotations, accumulation is a sum of translations.
        let locals: Vec<(Vec3, Quat)> = bones.iter().map(|b| (b.rest_translation, Quat::IDENTITY)).collect();
        let world = accumulate_bones(&parents, &locals);
        let hand = bone_index(&bones.iter().map(|b| b.name).collect::<Vec<_>>(), "lower_arm_r").unwrap();
        let (t, _) = world[hand];
        assert!((t - Vec3::new(-0.18 - 0.25, 0.95 + 0.12 + 0.38, 0.0)).length() < 1e-5, "{t}");
        // A parent rotation turns its child: hips 90° about Y.
        let mut locals2 = locals.clone();
        locals2[0].1 = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let world2 = accumulate_bones(&parents, &locals2);
        let (t2, r2) = world2[hand];
        // The arm is now in a different plane: x→-z.
        assert!((t2.z - (0.18 + 0.25)).abs() < 1e-4, "{t2}");
        assert!((r2.to_euler(glam::EulerRot::YXZ).0 - std::f32::consts::FRAC_PI_2).abs() < 1e-4);
    }

    #[test]
    fn bone_index_is_case_insensitive() {
        assert_eq!(bone_index(&["hips", "spine"], "SPINE"), Some(1));
        assert_eq!(bone_index(&["hips"], "tail"), None);
    }
}

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

fn rot_x(angle: f32) -> Quat {
    Quat::from_rotation_x(angle)
}

fn rot_z(angle: f32) -> Quat {
    Quat::from_rotation_z(angle)
}

/// Arms hang down along the body (the rest pose is a T-pose).
const ARMS_DOWN: f32 = 1.25;

/// How the humanoid figure's three body regions scale for a human role
/// (child/teen/adult/elder). The landmarks are in mesh space: the adult row is
/// the identity — waist 0.95, torso top 1.52, top of head 1.74 — so the
/// geometry drawn for `Humanoid` does not change.
#[derive(Clone, Copy, Debug)]
pub struct HumanProportions {
    /// Waist (hips) height of the drawn figure, m.
    pub waist: f32,
    /// Height where the torso region ends and the head begins, m.
    pub torso_top: f32,
    /// Top of the head, m.
    pub top: f32,
    /// Vertical (and horizontal) scale of the legs region, waist/0.95.
    pub leg: f32,
    /// Scale of the torso region, (torso_top−waist)/0.57.
    pub torso: f32,
    /// Scale of the head region, (top−torso_top)/0.22.
    pub head: f32,
}

/// The proportions of a human actor, from its [`crate::catalog::HumanRole`]
/// profile. The profile's own landmarks set the scale of each body region, so
/// a child is not just a small adult: its legs are much shorter than its
/// torso. The table's `eye`/`shoulder`/`waist`/`knee` are landmarks for the
/// scaling; `height` is what the figure's top matches.
pub fn human_proportions(actor: crate::project::BuiltinActor) -> HumanProportions {
    let role = actor.human_role().expect("human_proportions: a human actor");
    let adult = crate::catalog::human_role("adult").expect("the adult role");
    // Adult table landmarks: waist 0.99, shoulder 1.43, height 1.74; adult
    // geometry landmarks: waist 0.95, shoulder 1.45, top 1.74.
    let torso_ratio = (role.shoulder_m - role.waist_m) / (adult.shoulder_m - adult.waist_m);
    let head_ratio = (role.height_m - role.shoulder_m) / (adult.height_m - adult.shoulder_m);
    let waist = 0.95 * role.waist_m / adult.waist_m;
    let shoulder = waist + 0.50 * torso_ratio;
    let top = shoulder + 0.29 * head_ratio;
    let torso_top = shoulder + 0.07 * head_ratio;
    HumanProportions {
        waist,
        torso_top,
        top,
        leg: waist / 0.95,
        torso: (torso_top - waist) / 0.57,
        head: (top - torso_top) / 0.22,
    }
}

/// Humanoid: standing, sitting on a ~0.45 m seat, lying on the back. The poses
/// are held — an actor is never animated, it is placed.
fn humanoid_pose(actor: crate::project::BuiltinActor, pose: crate::project::ActorPose) -> Pose {
    use crate::project::ActorPose;
    let p = human_proportions(actor);
    let mut out = Pose {
        bones: actor_bones(actor)
            .iter()
            .map(|b| BonePose { translation: b.rest_translation, rotation: Quat::IDENTITY })
            .collect(),
    };
    out.bones[3].rotation = rot_z(-ARMS_DOWN);
    out.bones[5].rotation = rot_z(ARMS_DOWN);
    match pose {
        ActorPose::Stand => {}
        ActorPose::Sit => {
            // Hips down to the seat, thighs forward, shins hanging down. The
            // drop follows the role's legs: a child sits lower.
            out.bones[HIPS].translation.y -= 0.50 * p.leg;
            for thigh in [7, 9] {
                out.bones[thigh].rotation = rot_x(-1.45);
            }
            for shin in [8, 10] {
                out.bones[shin].rotation = rot_x(1.45);
            }
            out.bones[1].rotation = rot_x(-0.08);
        }
        ActorPose::Lie => {
            // On the back: the hips carry the whole body, which then lies along
            // the ground; the body's half thickness sets the hips' height.
            out.bones[HIPS].translation.y = 0.20 * p.torso;
            out.bones[HIPS].rotation = rot_x(-std::f32::consts::FRAC_PI_2);
        }
    }
    out
}

/// Quadrupeds and the bird. The rest pose is standing.
fn quadruped_pose(actor: crate::project::BuiltinActor, pose: crate::project::ActorPose) -> Pose {
    use crate::project::{ActorPose, BuiltinActor};
    let mut out = Pose {
        bones: actor_bones(actor)
            .iter()
            .map(|b| BonePose { translation: b.rest_translation, rotation: Quat::IDENTITY })
            .collect(),
    };
    match (actor, pose) {
        (_, ActorPose::Stand) => {}
        (BuiltinActor::Bird, ActorPose::Sit) => {
            // Perched: wings folded, body settled on the feet.
            out.bones[0].translation.y -= 0.06;
            for (i, sign) in [(2, 1.0f32), (4, -1.0f32)] {
                out.bones[i].rotation = Quat::from_rotation_y(sign * 1.25) * rot_z(-sign * 0.65);
                out.bones[i + 1].rotation = Quat::from_rotation_y(sign * 0.8);
            }
        }
        (_, ActorPose::Sit) => {
            // Hind legs fold under the body (a sitting dog).
            out.bones[0].translation.y -= 0.22;
            out.bones[0].rotation = rot_x(-0.35);
            for i in [8, 10] {
                out.bones[i].rotation = rot_x(-1.15);
                out.bones[i + 1].rotation = rot_x(1.7);
            }
            out.bones[1].rotation = rot_x(0.35);
        }
        (_, ActorPose::Lie) => {
            // Body down to a third of its standing height, all four legs folded.
            let standing = actor_bones(actor)[0].rest_translation.y;
            out.bones[0].translation.y = standing * 0.35;
            for i in [4, 6, 8, 10] {
                out.bones[i].rotation = rot_x(1.35);
                out.bones[i + 1].rotation = rot_x(-1.7);
            }
            out.bones[1].rotation = rot_x(0.25);
        }
    }
    out
}

/// Held pose of a builtin actor. Stateless and time-free: the same bones on
/// every frame, which is what "the actor does not animate" means.
pub fn sample_actor_pose(actor: crate::project::BuiltinActor, pose: crate::project::ActorPose) -> Result<Pose, crate::error::CoreError> {
    crate::project::RigRef::BuiltinActor { actor }.validate_pose(pose)?;
    Ok(match actor.human_role() {
        Some(_) => humanoid_pose(actor, pose),
        None => quadruped_pose(actor, pose),
    })
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

    /// A pose is held: sampling it twice gives the same bones, and each actor
    /// accepts only the poses it declares.
    #[test]
    fn poses_are_static_and_per_profile() {
        use crate::project::{ActorPose, BuiltinActor};
        for &actor in BuiltinActor::ALL {
            for pose in actor.poses() {
                let a = sample_actor_pose(actor, *pose).unwrap();
                let b = sample_actor_pose(actor, *pose).unwrap();
                for (x, y) in a.bones.iter().zip(b.bones.iter()) {
                    assert_eq!(x.translation, y.translation);
                    assert_eq!(x.rotation, y.rotation);
                }
                assert_eq!(a.bones.len(), actor_bones(actor).len());
            }
            assert!(sample_actor_pose(actor, ActorPose::Stand).is_ok(), "every actor can stand");
        }
        // A horse cannot sit, a bird cannot lie — the pose is refused, not faked.
        assert!(sample_actor_pose(BuiltinActor::Horse, ActorPose::Sit).is_err());
        assert!(sample_actor_pose(BuiltinActor::Bird, ActorPose::Lie).is_err());
        // Lying really is lower than standing, and sitting lowers the hips.
        let humanoid = |p| sample_actor_pose(BuiltinActor::Humanoid, p).unwrap();
        let stand = humanoid(ActorPose::Stand);
        let sit = humanoid(ActorPose::Sit);
        let lie = humanoid(ActorPose::Lie);
        assert!(sit.bones[HIPS].translation.y < stand.bones[HIPS].translation.y);
        assert!(lie.bones[HIPS].translation.y < sit.bones[HIPS].translation.y);
        assert_ne!(lie.bones[HIPS].rotation, Quat::IDENTITY, "lying turns the body");
    }
}

/// Original bundled skeletons. Quadrupeds share topology, not proportions.
/// All rest rotations are identity, parents precede children.
pub fn actor_bones(actor: crate::project::BuiltinActor) -> Vec<BoneDef> {
    use crate::project::BuiltinActor::*;
    if actor.human_role().is_some() {
        let p = human_proportions(actor);
        let ts = p.torso;
        // Same topology as the adult rig; the role scales each region's
        // offsets (the adult row is the identity).
        return BONES
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let r = b.rest_translation;
                let scaled = match i {
                    HIPS => Vec3::new(r.x, p.waist, r.z),
                    1 | 2 => r * ts,                    // spine, head offsets
                    3..=6 => Vec3::new(r.x * ts, r.y * ts, r.z * ts), // arms
                    _ => Vec3::new(r.x * ts, r.y * p.leg, r.z * ts),  // legs
                };
                BoneDef { name: b.name, parent: b.parent, rest_translation: scaled }
            })
            .collect();
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

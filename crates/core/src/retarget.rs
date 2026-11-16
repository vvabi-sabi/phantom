//! Clip retargeting profiles: an explicit bone-role map instead of a
//! universal automatic transfer of arbitrary anatomies.
//!
//! Each profile ([`RetargetProfile`]) declares a canonical minimal set of roles
//! and, per role, a list of accepted bone names (Mixamo `mixamorig:*` and the
//! same names without the prefix, Phantom's built-in rigs, common
//! KayKit/Blender variants). Matching is on normalized names (case, the prefix
//! up to `:` and the `. - ` separators are ignored), exact equality, no
//! "substring" substitution: incompatibility is diagnosed, not silently
//! distorted.
//!
//! core owns only the role map (it does not need the clip's data); loading the
//! source tracks and transferring samples onto the target rig's bones lives in
//! runtime (`load_clip_retarget`). Transfer limitation: rotations and
//! translations are transferred directly (assuming matching bone orientation);
//! there is no rest-pose or scale correction — rest-pose differences are
//! fixed by tuning/extending the profile's patterns, not by hidden math.

use serde::{Deserialize, Serialize};

/// Retargeting profile: the anatomy of the source clip and of the target rig.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetargetProfile {
    /// Biped: minimal set hips/spine/chest/neck/head/shoulders/arms/legs.
    Humanoid,
    /// Quadruped (built-in wolf/horse): body/neck/head/tail/legs.
    Quadruped,
}

impl RetargetProfile {
    /// Name for diagnostics and CLI/GUI.
    pub fn name(self) -> &'static str {
        match self {
            Self::Humanoid => "humanoid",
            Self::Quadruped => "quadruped",
        }
    }
}

/// Canonical role and accepted normalized bone names (source and target).
pub struct RoleDef {
    pub role: &'static str,
    pub patterns: &'static [&'static str],
}

/// Profile roles in matching order (each source name goes into
/// at most one role — the first role that matches).
pub fn roles(profile: RetargetProfile) -> &'static [RoleDef] {
    match profile {
        RetargetProfile::Humanoid => HUMANOID_ROLES,
        RetargetProfile::Quadruped => QUADRUPED_ROLES,
    }
}

/// Minimal humanoid set: Mixamo (`mixamorig:` is stripped by
/// normalization), the built-in rig (`upper_arm_l` etc.), KayKit
/// (Mixamo-like names without a prefix) and Blender suffixes `_l`/`_r`.
static HUMANOID_ROLES: &[RoleDef] = &[
    RoleDef { role: "hips", patterns: &["hips", "pelvis", "hip"] },
    RoleDef { role: "spine", patterns: &["spine", "spine0"] },
    RoleDef { role: "chest", patterns: &["chest", "spine1", "spine2", "torso", "upper_chest", "upperchest"] },
    RoleDef { role: "neck", patterns: &["neck"] },
    RoleDef { role: "head", patterns: &["head"] },
    RoleDef { role: "shoulder_l", patterns: &["leftshoulder", "shoulder_l", "l_shoulder", "left_shoulder", "clavicle_l"] },
    RoleDef { role: "shoulder_r", patterns: &["rightshoulder", "shoulder_r", "r_shoulder", "right_shoulder", "clavicle_r"] },
    RoleDef { role: "upperarm_l", patterns: &["leftarm", "upper_arm_l", "upperarm_l", "arm_l", "left_arm", "l_upperarm"] },
    RoleDef { role: "upperarm_r", patterns: &["rightarm", "upper_arm_r", "upperarm_r", "arm_r", "right_arm", "r_upperarm"] },
    RoleDef { role: "forearm_l", patterns: &["leftforearm", "lower_arm_l", "forearm_l", "lowerarm_l", "left_forearm", "l_forearm"] },
    RoleDef { role: "forearm_r", patterns: &["rightforearm", "lower_arm_r", "forearm_r", "lowerarm_r", "right_forearm", "r_forearm"] },
    RoleDef { role: "hand_l", patterns: &["lefthand", "hand_l", "left_hand", "l_hand"] },
    RoleDef { role: "hand_r", patterns: &["righthand", "hand_r", "right_hand", "r_hand"] },
    RoleDef { role: "upleg_l", patterns: &["leftupleg", "upper_leg_l", "upleg_l", "thigh_l", "left_up_leg", "l_thigh"] },
    RoleDef { role: "upleg_r", patterns: &["rightupleg", "upper_leg_r", "upleg_r", "thigh_r", "right_up_leg", "r_thigh"] },
    RoleDef { role: "leg_l", patterns: &["leftleg", "lower_leg_l", "leg_l", "shin_l", "calf_l", "left_leg", "l_shin"] },
    RoleDef { role: "leg_r", patterns: &["rightleg", "lower_leg_r", "leg_r", "shin_r", "calf_r", "right_leg", "r_shin"] },
    RoleDef { role: "foot_l", patterns: &["leftfoot", "foot_l", "left_foot", "l_foot"] },
    RoleDef { role: "foot_r", patterns: &["rightfoot", "foot_r", "right_foot", "r_foot"] },
];

/// Quadruped set for the built-in wolf/horse skeleton
/// (`body/neck/head/tail/front*/hind*`) and typical DCC name variants.
static QUADRUPED_ROLES: &[RoleDef] = &[
    RoleDef { role: "body", patterns: &["body", "hips", "pelvis", "spine", "root"] },
    RoleDef { role: "neck", patterns: &["neck"] },
    RoleDef { role: "head", patterns: &["head"] },
    RoleDef { role: "tail", patterns: &["tail", "tail0", "tail_0", "tail_base"] },
    RoleDef { role: "front_l", patterns: &["front_l", "frontleg_l", "front_leg_l", "foreleg_l", "front_upper_l", "l_front_leg"] },
    RoleDef { role: "front_r", patterns: &["front_r", "frontleg_r", "front_leg_r", "foreleg_r", "front_upper_r", "r_front_leg"] },
    RoleDef {
        role: "front_lower_l",
        patterns: &["front_lower_l", "frontlowerleg_l", "frontleg_lower_l", "front_lower_leg_l", "foreleg_lower_l", "front_l_lower"],
    },
    RoleDef {
        role: "front_lower_r",
        patterns: &["front_lower_r", "frontlowerleg_r", "frontleg_lower_r", "front_lower_leg_r", "foreleg_lower_r", "front_r_lower"],
    },
    RoleDef { role: "hind_l", patterns: &["hind_l", "hindleg_l", "hind_leg_l", "back_leg_l", "rear_leg_l", "hind_upper_l"] },
    RoleDef { role: "hind_r", patterns: &["hind_r", "hindleg_r", "hind_leg_r", "back_leg_r", "rear_leg_r", "hind_upper_r"] },
    RoleDef {
        role: "hind_lower_l",
        patterns: &["hind_lower_l", "hindlowerleg_l", "hindleg_lower_l", "hind_lower_leg_l", "rear_lower_leg_l", "hind_l_lower"],
    },
    RoleDef {
        role: "hind_lower_r",
        patterns: &["hind_lower_r", "hindlowerleg_r", "hindleg_lower_r", "hind_lower_leg_r", "rear_lower_leg_r", "hind_r_lower"],
    },
];

/// Bone-name normalization: lowercase, the prefix up to the last `:`
/// (`mixamorig:Hips` → `hips`), separators `.`, `-`, space → `_`.
pub fn normalize_bone_name(name: &str) -> String {
    let after_prefix = name.rsplit(':').next().unwrap_or(name);
    after_prefix
        .chars()
        .map(|c| match c {
            '.' | '-' | ' ' => '_',
            c => c.to_ascii_lowercase(),
        })
        .collect()
}

/// One matched retargeting link.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MappedBone {
    pub role: &'static str,
    /// Index into the source name list (the clip's file).
    pub source: usize,
    /// Index into the target rig's name list.
    pub target: usize,
}

/// Profile mapping result: matched roles plus incompatibility diagnostics on
/// both sides (it never stays silent).
#[derive(Clone, Debug, Default)]
pub struct RetargetMap {
    pub mapped: Vec<MappedBone>,
    /// (role, name in the source): the track exists, but the target rig has no such role.
    pub unmapped_target: Vec<(&'static str, String)>,
    /// (role, name in the target rig): the rig bone exists, but the clip has no such role.
    pub unmapped_source: Vec<(&'static str, String)>,
}

impl RetargetMap {
    /// Number of matched roles (== the number of tracks in the retargeted clip).
    pub fn matched(&self) -> usize {
        self.mapped.len()
    }

    /// Text diagnostics for unmapped roles (non-fatal).
    pub fn diagnostics(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (role, source) in &self.unmapped_target {
            out.push(format!("retargeting: role {role} (clip bone \"{source}\") not found in the target rig — track skipped"));
        }
        for (role, target) in &self.unmapped_source {
            out.push(format!("retargeting: role {role} (rig bone \"{target}\") not found in the clip — the bone stays in its rest pose"));
        }
        out
    }
}

/// Builds the role → (source, target) map. For each role the first matching
/// name is taken on each side; a source name already claimed by another role
/// is not reused. A role without a pair on either side goes to diagnostics and
/// the clip plays the matched subset; an empty map is a `NoMatchingBones`
/// error at load time (as with direct name matching).
pub fn build_map(profile: RetargetProfile, source_names: &[String], target_names: &[String]) -> RetargetMap {
    let norm_source: Vec<String> = source_names.iter().map(|n| normalize_bone_name(n)).collect();
    let norm_target: Vec<String> = target_names.iter().map(|n| normalize_bone_name(n)).collect();
    let mut claimed_source = vec![false; source_names.len()];
    let mut map = RetargetMap::default();
    for def in roles(profile) {
        let source = (0..norm_source.len()).find(|&i| !claimed_source[i] && def.patterns.contains(&norm_source[i].as_str()));
        let target = norm_target.iter().position(|n| def.patterns.contains(&n.as_str()));
        match (source, target) {
            (Some(s), Some(t)) => {
                claimed_source[s] = true;
                map.mapped.push(MappedBone { role: def.role, source: s, target: t });
            }
            (Some(s), None) => {
                claimed_source[s] = true;
                map.unmapped_target.push((def.role, source_names[s].clone()));
            }
            (None, Some(t)) => map.unmapped_source.push((def.role, target_names[t].clone())),
            (None, None) => {}
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_strips_prefix_and_separators() {
        assert_eq!(normalize_bone_name("mixamorig:LeftUpLeg"), "leftupleg");
        assert_eq!(normalize_bone_name("mixamorig:Hips"), "hips");
        assert_eq!(normalize_bone_name("UpperArm.L"), "upperarm_l");
        assert_eq!(normalize_bone_name("front-lower-l"), "front_lower_l");
        assert_eq!(normalize_bone_name("hips"), "hips");
    }

    #[test]
    fn source_name_claimed_once() {
        // "head" exists in humanoid and is potentially ambiguous via patterns;
        // check that one source bone does not go into two roles.
        let source = vec!["head".to_string()];
        let target = vec!["head".to_string(), "neck".to_string()];
        let map = build_map(RetargetProfile::Humanoid, &source, &target);
        assert_eq!(map.matched(), 1);
        assert_eq!(map.mapped[0].role, "head");
    }
}

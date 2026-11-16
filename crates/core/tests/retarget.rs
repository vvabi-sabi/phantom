//! Retargeting profiles: role matching by bone names from Mixamo, builtin
//! rigs and KayKit-like ones; incompatibility diagnostics; zero coverage.
//! No file loading and no GPU.

use phantom_core::project::BuiltinActor;
use phantom_core::retarget::{build_map, RetargetProfile};
use phantom_core::rig::{actor_bones, BONES};

/// Bone names of a Mixamo file (as in "Hip Hop Dancing.fbx"): `mixamorig:`
/// prefix and PascalCase without separators.
fn mixamo_humanoid() -> Vec<String> {
    [
        "mixamorig:Hips",
        "mixamorig:Spine",
        "mixamorig:Spine1",
        "mixamorig:Spine2",
        "mixamorig:Neck",
        "mixamorig:Head",
        "mixamorig:LeftShoulder",
        "mixamorig:LeftArm",
        "mixamorig:LeftForeArm",
        "mixamorig:LeftHand",
        "mixamorig:LeftUpLeg",
        "mixamorig:LeftLeg",
        "mixamorig:LeftFoot",
        "mixamorig:RightShoulder",
        "mixamorig:RightArm",
        "mixamorig:RightForeArm",
        "mixamorig:RightHand",
        "mixamorig:RightUpLeg",
        "mixamorig:RightLeg",
        "mixamorig:RightFoot",
        // Fingers and other bones outside the minimal set must not interfere.
        "mixamorig:LeftToeBase",
        "mixamorig:LeftHandThumb1",
        "mixamorig:RightHandIndex3",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

fn names(defs: &[phantom_core::rig::BoneDef]) -> Vec<String> {
    defs.iter().map(|b| b.name.to_string()).collect()
}

fn mapped_roles(map: &phantom_core::retarget::RetargetMap) -> Vec<&'static str> {
    map.mapped.iter().map(|m| m.role).collect()
}

#[test]
fn mixamo_clip_maps_to_builtin_humanoid() {
    let source = mixamo_humanoid();
    let target = names(BONES);
    let map = build_map(RetargetProfile::Humanoid, &source, &target);
    let roles = mapped_roles(&map);
    // Minimal set of the builtin rig: hips, spine, head, arms and legs x2.
    for role in [
        "hips", "spine", "head", "upperarm_l", "upperarm_r", "forearm_l", "forearm_r", "upleg_l", "upleg_r", "leg_l", "leg_r",
    ] {
        assert!(roles.contains(&role), "role {role} mapped: {roles:?}");
    }
    // Concrete pairs: Mixamo -> builtin names.
    let pair = |role: &str| {
        let m = map.mapped.iter().find(|m| m.role == role).unwrap();
        (source[m.source].as_str(), target[m.target].as_str())
    };
    assert_eq!(pair("hips"), ("mixamorig:Hips", "hips"));
    assert_eq!(pair("upperarm_l"), ("mixamorig:LeftArm", "upper_arm_l"));
    assert_eq!(pair("forearm_r"), ("mixamorig:RightForeArm", "lower_arm_r"));
    assert_eq!(pair("upleg_l"), ("mixamorig:LeftUpLeg", "upper_leg_l"));
    assert_eq!(pair("leg_r"), ("mixamorig:RightLeg", "lower_leg_r"));
    // The builtin rig has no chest/neck/shoulders/hands roles — that is
    // diagnostics; the clip plays the matched subset.
    let missing: Vec<&str> = map.unmapped_target.iter().map(|(r, _)| *r).collect();
    for role in ["chest", "neck", "shoulder_l", "shoulder_r", "hand_l", "hand_r", "foot_l", "foot_r"] {
        assert!(missing.contains(&role), "role {role} reported: {missing:?}");
    }
    assert!(map.unmapped_source.is_empty(), "every builtin rig bone is covered");
    assert!(!map.diagnostics().is_empty(), "incompatibility is reported as text");
}

#[test]
fn mixamo_clip_maps_to_kaykit_like_names() {
    // KayKit-like rig: mixamo names without the prefix (as in the verified
    // CC0 KayKit rig) — full coverage of the minimal set.
    let source = mixamo_humanoid();
    let target: Vec<String> = [
        "Hips", "Spine", "Spine1", "Neck", "Head", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftUpLeg", "LeftLeg",
        "LeftFoot", "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightUpLeg", "RightLeg", "RightFoot",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let map = build_map(RetargetProfile::Humanoid, &source, &target);
    assert_eq!(map.matched(), 19, "every role mapped: {:?}", map.mapped);
    assert!(map.unmapped_target.is_empty() && map.unmapped_source.is_empty());
    assert!(map.diagnostics().is_empty(), "full coverage — no diagnostics");
    // Rotations transfer role-to-role: LeftForeArm -> LeftForeArm.
    let m = map.mapped.iter().find(|m| m.role == "forearm_l").unwrap();
    assert_eq!(source[m.source], "mixamorig:LeftForeArm");
    assert_eq!(target[m.target], "LeftForeArm");
}

#[test]
fn quadruped_clip_maps_to_builtin_wolf_and_horse() {
    // Source uses DCC name variants (Blender suffixes/underscores).
    let source: Vec<String> = [
        "Body", "Neck", "Head", "Tail", "FrontLeg_L", "FrontLowerLeg_L", "FrontLeg_R", "FrontLowerLeg_R", "HindLeg_L",
        "HindLowerLeg_L", "HindLeg_R", "HindLowerLeg_R",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    for actor in [BuiltinActor::Wolf, BuiltinActor::Horse] {
        let target = names(&actor_bones(actor));
        let map = build_map(RetargetProfile::Quadruped, &source, &target);
        assert_eq!(map.matched(), 12, "full quadruped set for {actor:?}: {:?}", map.diagnostics());
        assert!(map.diagnostics().is_empty(), "{actor:?}: {:?}", map.diagnostics());
        let m = map.mapped.iter().find(|m| m.role == "front_lower_l").unwrap();
        assert_eq!(target[m.target], "front_lower_l");
        let m = map.mapped.iter().find(|m| m.role == "body").unwrap();
        assert_eq!(target[m.target], "body");
    }
    // Exact hits of builtin rig names match as well.
    let target = names(&actor_bones(BuiltinActor::Wolf));
    let map = build_map(RetargetProfile::Quadruped, &target, &target);
    assert_eq!(map.matched(), 12);
}

#[test]
fn zero_coverage_is_empty_map_not_silent_subset() {
    let source = vec!["cube".to_string(), "lamp".to_string()];
    let target = names(BONES);
    let map = build_map(RetargetProfile::Humanoid, &source, &target);
    assert_eq!(map.matched(), 0, "the loader turns this into NoMatchingBones");
    // Every target bone shows up in the unmapped_source diagnostics.
    assert_eq!(map.unmapped_source.len(), BONES.len());

    // Foreign anatomy (humanoid clip -> quadruped profile): hips/neck/head
    // match, legs do not — diagnostics list what stayed unmapped.
    let map = build_map(RetargetProfile::Quadruped, &mixamo_humanoid(), &names(&actor_bones(BuiltinActor::Wolf)));
    let missing: Vec<&str> = map.unmapped_source.iter().map(|(r, _)| *r).collect();
    assert!(missing.contains(&"front_l"), "front legs not found in humanoid clip: {missing:?}");
    assert!(missing.contains(&"hind_lower_r"), "{missing:?}");
}

#[test]
fn retarget_profile_serde_snake_case() {
    let h: RetargetProfile = serde_json::from_str("\"humanoid\"").unwrap();
    assert_eq!(h, RetargetProfile::Humanoid);
    let q: RetargetProfile = serde_json::from_str("\"quadruped\"").unwrap();
    assert_eq!(q, RetargetProfile::Quadruped);
    assert!(serde_json::from_str::<RetargetProfile>("\"centaur\"").is_err());
    assert_eq!(h.name(), "humanoid");
    assert_eq!(q.name(), "quadruped");
}

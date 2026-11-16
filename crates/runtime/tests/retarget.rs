//! Clip loading with profile-based retargeting: a Mixamo file → the builtin
//! humanoid rig and → KayKit-like names; diagnostics for unmatched roles;
//! zero coverage is NoMatchingBones; the legacy profile-less path is
//! unchanged. No GPU/render.

use std::path::Path;

use phantom_core::RetargetProfile;
use phantom_runtime::import::{load_clip, load_clip_retarget, ImportError};

fn fbx() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Hip Hop Dancing.fbx")
}

fn builtin_humanoid_bones() -> Vec<String> {
    phantom_core::rig::BONES.iter().map(|b| b.name.to_string()).collect()
}

#[test]
fn humanoid_profile_maps_mixamo_clip_to_builtin_rig() {
    let path = fbx();
    let target = builtin_humanoid_bones();
    let (clip, diagnostics) = load_clip_retarget(&path, RetargetProfile::Humanoid, &target, None).expect("retarget Mixamo → builtin");
    // Builtin rig: hips/spine/head + arms and legs ×2 = 11 bones, all covered.
    assert_eq!(clip.matched, 11, "{diagnostics:?}");
    // Clip tracks are named after the TARGET rig's bones (the bone→track map is shared).
    assert!(clip.bones.iter().all(|b| target.contains(b)), "{:?}", clip.bones);
    assert!(clip.bones.contains(&"hips".to_string()));
    assert!(clip.bones.contains(&"lower_arm_r".to_string()), "RightForeArm → lower_arm_r");
    assert_eq!(clip.samples.len(), clip.bones.len());
    assert!(clip.frames > 0);
    // The target rig has no chest/neck/shoulders/hands/feet roles — diagnostics.
    assert!(diagnostics.iter().any(|d| d.contains("chest")), "{diagnostics:?}");
    assert!(diagnostics.iter().any(|d| d.contains("neck")), "{diagnostics:?}");
    assert!(diagnostics.iter().any(|d| d.contains("hand_l")), "{diagnostics:?}");
}

#[test]
fn humanoid_profile_maps_mixamo_clip_to_kaykit_like_rig() {
    let path = fbx();
    // KayKit-like rig: mixamo names without the prefix — full coverage.
    let target: Vec<String> = [
        "Hips", "Spine", "Spine1", "Neck", "Head", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftUpLeg", "LeftLeg",
        "LeftFoot", "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightUpLeg", "RightLeg", "RightFoot",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let (clip, diagnostics) = load_clip_retarget(&path, RetargetProfile::Humanoid, &target, None).expect("full coverage");
    assert_eq!(clip.matched, 19, "{diagnostics:?}");
    assert!(diagnostics.is_empty(), "full coverage means no diagnostics: {diagnostics:?}");
    // Track names are the target's (the order follows the profile's role order).
    let mut sorted = clip.bones.clone();
    sorted.sort();
    let mut target_sorted = target.clone();
    target_sorted.sort();
    assert_eq!(sorted, target_sorted);
    // The samples really come from the source: not every rotation is identity.
    let track = &clip.samples[target.iter().position(|b| b == "LeftArm").unwrap()];
    assert!(track.iter().any(|(_, q)| (q.length() - 1.0).abs() < 1e-3 && *q != glam::Quat::IDENTITY));
}

#[test]
fn zero_coverage_is_no_matching_bones() {
    let path = fbx();
    let target = vec!["no_such_bone".to_string(), "another".to_string()];
    let err = load_clip_retarget(&path, RetargetProfile::Humanoid, &target, None).unwrap_err();
    assert!(matches!(err, ImportError::NoMatchingBones), "{err:?}");
    // A foreign profile on a humanoid file: quadruped matches only hips/neck/head,
    // the rest is diagnostics (not an error, and not silently mangled animation).
    let wolf: Vec<String> = phantom_core::rig::actor_bones(phantom_core::project::BuiltinActor::Wolf)
        .iter()
        .map(|b| b.name.to_string())
        .collect();
    let (clip, diagnostics) = load_clip_retarget(&path, RetargetProfile::Quadruped, &wolf, None).expect("hips/neck/head matched");
    assert_eq!(clip.matched, 3, "{diagnostics:?}");
    assert!(diagnostics.iter().any(|d| d.contains("front_l")), "{diagnostics:?}");
}

#[test]
fn legacy_direct_match_path_unchanged() {
    // Without a profile: direct full-name match, as before.
    let path = fbx();
    let mixamo = vec!["mixamorig:Hips".to_string(), "mixamorig:LeftHand".to_string()];
    let clip = load_clip(&path, &mixamo, None).expect("direct match");
    assert_eq!(clip.matched, 2);
    assert_eq!(clip.bones, mixamo);
    // Builtin names without a profile do not match mixamorig:* — error.
    let err = load_clip(&path, &builtin_humanoid_bones(), None).unwrap_err();
    assert!(matches!(err, ImportError::NoMatchingBones), "{err:?}");
}

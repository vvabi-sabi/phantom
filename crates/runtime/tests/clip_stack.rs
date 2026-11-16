//! Selecting a stack by name, sampling, pose blending and root motion —
//! no GPU/render.

use std::path::Path;

use glam::{Quat, Vec3};
use phantom_runtime::import::{list_stacks, load_clip, ClipData, ImportError};
use phantom_runtime::render::{blend_clip_pose, root_motion_delta, sample_clip_pose};

fn fbx() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Hip Hop Dancing.fbx")
}

/// FBX node names as rig bone candidates (the test needs no real rig:
/// name matching works with any list of names).
fn fbx_node_names(path: &Path) -> Vec<String> {
    let scene = ufbx::load_file(&path.to_string_lossy(), Default::default()).expect("the fbx loads");
    scene.nodes.iter().map(|n| n.element.name.to_string()).collect()
}

#[test]
fn stack_listing_and_selection_by_name() {
    let path = fbx();
    let stacks = list_stacks(&path).expect("stack list");
    assert!(!stacks.is_empty(), "the file has animation stacks");
    assert!(stacks.iter().all(|s| s.duration_secs > 0.0));

    let bones = fbx_node_names(&path);
    // The longest stack is the legacy None choice.
    let longest = stacks.iter().max_by(|a, b| a.duration_secs.partial_cmp(&b.duration_secs).unwrap()).unwrap();
    let by_default = load_clip(&path, &bones, None).expect("clip None");
    let by_name = load_clip(&path, &bones, Some(&longest.name)).expect("clip by name");
    assert_eq!(by_default.frames, by_name.frames, "longest stack name == the None choice");
    assert_eq!(by_default.samples, by_name.samples);

    // A nonexistent stack is a diagnosable error.
    let err = load_clip(&path, &bones, Some("no such stack")).unwrap_err();
    assert!(matches!(err, ImportError::NoSuchStack(_)));
}

fn toy_clip(frames: u32) -> ClipData {
    // One "root" bone: moves along +X, rotates around Y.
    let samples = vec![
        (0..frames)
            .map(|i| {
                let t = i as f32;
                (Vec3::new(t, 1.0, 0.0), Quat::from_rotation_y(t * 0.1))
            })
            .collect::<Vec<_>>(),
    ];
    ClipData { fps: 24.0, frames, bones: vec!["root".into()], samples, matched: 1 }
}

#[test]
fn pose_sampling_interpolates_and_clamps() {
    let clip = toy_clip(4);
    let rest = vec![(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE)];
    let map = vec![Some(0usize)];
    let p = sample_clip_pose(&rest, &map, &clip, 1.5);
    assert!((p[0].0 - Vec3::new(1.5, 1.0, 0.0)).length() < 1e-6);
    // Clamp past the last frame (the clip track does not loop).
    let end = sample_clip_pose(&rest, &map, &clip, 99.0);
    assert!((end[0].0 - Vec3::new(3.0, 1.0, 0.0)).length() < 1e-6);
    // A bone with no track stays at rest.
    let unmapped = sample_clip_pose(&rest, &[None], &clip, 1.0);
    assert_eq!(unmapped[0], rest[0]);
}

#[test]
fn blend_pose_lerps_between_segments() {
    let rest = vec![(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE)];
    let clip = toy_clip(4);
    let map = vec![Some(0usize)];
    let clip_pose = sample_clip_pose(&rest, &map, &clip, 2.0);
    // alpha = 0 → base (rest), 1 → clip.
    let at_zero = blend_clip_pose(&rest, &clip_pose, 0.0);
    assert!((at_zero[0].0 - rest[0].0).length() < 1e-6);
    let at_half = blend_clip_pose(&rest, &clip_pose, 0.5);
    assert!((at_half[0].0 - Vec3::new(1.0, 0.5, 0.0)).length() < 1e-6);
    assert!((at_half[0].1.length() - 1.0).abs() < 1e-5, "nlerp keeps the quaternion unit-length");
    let at_one = blend_clip_pose(&rest, &clip_pose, 1.0);
    assert!((at_one[0].0 - clip_pose[0].0).length() < 1e-6);
}

#[test]
fn root_motion_delta_from_use_start() {
    let clip = toy_clip(4);
    let map = vec![Some(0usize)];
    // Delta from clip frame 0 (the use start maps to 0).
    let d = root_motion_delta(&map, &clip, 2.0).unwrap();
    assert!((d - Vec3::new(2.0, 0.0, 0.0)).length() < 1e-6);
    let d0 = root_motion_delta(&map, &clip, 0.0).unwrap();
    assert!(d0.length() < 1e-6, "delta is zero at the use start — the controller's starting position");
    // Without a root bone track there is no delta.
    assert!(root_motion_delta(&[None], &clip, 2.0).is_none());
}

#[test]
fn clip_root_motion_composes_without_double_translation() {
    // Regression: in RootMotion::Clip mode the root entity gets
    // base + root_motion_delta, while the root bone's local translation in the
    // pose carries the same absolute root track from the clip — without
    // freezing, the travel was counted twice.
    use phantom_runtime::render::freeze_clip_root_translation;
    let clip = toy_clip(4); // the root moves +1 along X per frame
    let rest = vec![(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE)];
    let map = vec![Some(0usize)];
    let base = Vec3::new(5.0, 0.0, -2.0); // controller position at the use-start frame
    let f = 3.0;
    let authored_travel = clip.samples[0][3].0 - clip.samples[0][0].0;

    let mut pose = sample_clip_pose(&rest, &map, &clip, f);
    freeze_clip_root_translation(&mut pose, &map, &clip);
    let root = base + root_motion_delta(&map, &clip, f).unwrap();
    let world = root + pose[0].0;
    let authored_world = base + clip.samples[0][3].0;
    assert!(
        (world - authored_world).length() < 1e-6,
        "world = base + authored travel, not doubled: {world:?} vs {authored_world:?}"
    );
    assert!((world.x - base.x - authored_travel.x).abs() < 1e-6);
    // Without freezing, the path would be doubled.
    let unfrozen = sample_clip_pose(&rest, &map, &clip, f);
    let doubled = root + unfrozen[0].0;
    assert!((doubled.x - authored_world.x).abs() > 0.5, "control: no freeze means doubled travel");
}

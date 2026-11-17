//! Bundled actor profiles: skeletons, the poses each one can hold, and the
//! document round trip. Actors are not animated — a pose is held.

use glam::Vec3;
use phantom_core::{project::*, rig::*, *};

const ACTORS: &[BuiltinActor] = BuiltinActor::ALL;

#[test]
fn legacy_and_new_rigs_roundtrip() {
    assert_eq!(serde_json::to_string(&RigRef::Builtin).unwrap(), "\"Builtin\"");
    assert_eq!(serde_json::from_str::<RigRef>("\"Builtin\"").unwrap().builtin_actor(), Some(BuiltinActor::Humanoid));
    for &actor in ACTORS {
        let rig = RigRef::BuiltinActor { actor };
        assert_eq!(serde_json::from_str::<RigRef>(&serde_json::to_string(&rig).unwrap()).unwrap().builtin_actor(), Some(actor));
        for pose in actor.poses() {
            assert_eq!(*pose, serde_json::from_str::<ActorPose>(&serde_json::to_string(pose).unwrap()).unwrap());
        }
    }
}

#[test]
fn skeletons_are_finite_deterministic_and_poses_distinct() {
    for &actor in ACTORS {
        let bones = actor_bones(actor);
        for (i, b) in bones.iter().enumerate() {
            assert!(b.parent.is_none_or(|p| p < i));
            assert!(b.rest_translation.is_finite());
            assert!(!bones[..i].iter().any(|other| other.name == b.name));
        }
        let mut signatures = Vec::new();
        for &pose in actor.poses() {
            let sampled = sample_actor_pose(actor, pose).unwrap();
            let again = sample_actor_pose(actor, pose).unwrap();
            assert_eq!(sampled.bones.len(), bones.len());
            let mut signature = Vec::new();
            for (a, b) in sampled.bones.iter().zip(again.bones) {
                assert!(a.translation.is_finite() && a.rotation.is_finite());
                assert!((a.rotation.length() - 1.).abs() < 1e-5);
                assert_eq!(a.translation, b.translation, "a pose is held, not sampled over time");
                assert_eq!(a.rotation, b.rotation);
                signature.extend(a.translation.to_array());
                signature.extend(a.rotation.to_array());
            }
            assert!(!signatures.contains(&signature), "duplicate pose for {actor:?}: {pose:?}");
            signatures.push(signature);
        }
    }
}

#[test]
fn a_pose_the_species_does_not_have_is_refused() {
    // A horse cannot sit and a bird cannot lie — a clear error, not a silent
    // approximation.
    assert!(sample_actor_pose(BuiltinActor::Horse, ActorPose::Sit).is_err());
    assert!(sample_actor_pose(BuiltinActor::Bird, ActorPose::Lie).is_err());
    assert!(RigRef::BuiltinActor { actor: BuiltinActor::Horse }.validate_pose(ActorPose::Sit).is_err());
    assert!(RigRef::BuiltinActor { actor: BuiltinActor::Horse }.validate_pose(ActorPose::Stand).is_ok());
    // An imported rig has no named poses: any pose is accepted and ignored.
    assert!(RigRef::Imported { asset: AssetId(1) }.validate_pose(ActorPose::Lie).is_ok());
}

#[test]
fn timeline_holds_the_pose_of_the_key_in_force() {
    for &actor in ACTORS {
        let mut p = Phantom {
            id: PhantomId(1),
            name: "actor".into(),
            rig: RigRef::BuiltinActor { actor },
            motion: vec![],
            outline_px: 2.,
            scale: 1.0,
            material_patches: Vec::new(),
        };
        // An empty track stands at the origin in the default pose.
        let empty = evaluate(&p, 0);
        assert_eq!(empty.pose, actor.default_pose());
        assert_eq!(empty.root, Vec3::ZERO);
        for (i, &pose) in actor.poses().iter().enumerate() {
            p.motion = vec![MotionKey { yaw: 0.0, frame: 0, position: Vec3::new(1., 2., 3.), pose }];
            for f in [0, 1, 48, 95] {
                let e = evaluate(&p, f);
                assert_eq!(e.pose, pose);
                assert_eq!(e.root, Vec3::new(1., 2., 3.), "no motion, only a held pose");
                assert_eq!(e.skeleton.bones.len(), actor_bones(actor).len());
            }
            // A later key switches the pose at its own frame.
            let other = actor.poses()[(i + 1) % actor.poses().len()];
            p.motion.push(MotionKey { yaw: 0.0, frame: 48, position: Vec3::new(4., 2., 3.), pose: other });
            assert_eq!(evaluate(&p, 47).pose, pose);
            assert_eq!(evaluate(&p, 48).pose, other);
        }
    }
}

#[test]
fn unsupported_poses_are_atomic_and_projects_roundtrip() {
    let mut project = Project::new("actors");
    let mut history = History::default();
    let range = FrameRange::new(0, 48).unwrap();
    let scene = history.apply(&mut project, Command::AddScene { name: "s".into(), range }).unwrap().scene.unwrap();
    for &actor in ACTORS {
        let phantom = history
            .apply(&mut project, Command::AddPhantom { scene, name: format!("{actor:?}"), rig: RigRef::BuiltinActor { actor } })
            .unwrap()
            .phantom
            .unwrap();
        // A pose the species cannot hold is refused by every command that sets one.
        let unsupported = ActorPose::Lie;
        if !actor.poses().contains(&unsupported) {
            let bad = MotionKey { yaw: 0.0, frame: 0, position: Vec3::ZERO, pose: unsupported };
            for command in [
                Command::SetMotionKey { scene, phantom, key: bad },
                Command::ReplaceMotion { scene, phantom, motion: vec![bad] },
            ] {
                let before = serde_json::to_value(&project).unwrap();
                assert!(history.apply(&mut project, command).is_err(), "{actor:?} accepted {unsupported:?}");
                assert_eq!(before, serde_json::to_value(&project).unwrap());
            }
        }
        for (i, &pose) in actor.poses().iter().enumerate() {
            history
                .apply(
                    &mut project,
                    Command::SetMotionKey { scene, phantom, key: MotionKey { yaw: 0.0, frame: i as i64, position: Vec3::ZERO, pose } },
                )
                .unwrap();
        }
    }
    let json = serde_json::to_value(&project).unwrap();
    let restored: Project = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(json, serde_json::to_value(restored).unwrap());
}

/// An old manifest carries `action` on a key and takes/clips on the Phantom:
/// the key falls back to the default pose and the animation data is dropped.
#[test]
fn a_pre_layout_manifest_still_loads() {
    let json = r#"{
        "id": 1, "name": "old", "rig": "Builtin",
        "motion": [{"frame": 0, "position": [1, 2, 3], "action": "Walk", "yaw": 0.5}],
        "clip": 7, "clip_track": [], "takes": [], "active_take": null,
        "outline_px": 2.0
    }"#;
    let phantom: Phantom = serde_json::from_str(json).unwrap();
    assert_eq!(phantom.motion.len(), 1);
    assert_eq!(phantom.motion[0].pose, ActorPose::Stand, "the dropped action falls back to the default pose");
    assert_eq!(phantom.motion[0].yaw, 0.5);
    let round = serde_json::to_value(&phantom).unwrap();
    assert!(round.get("clip").is_none(), "the animation fields are gone from the document");
    assert!(round["motion"][0].get("action").is_none());
}

/// Human roles (child/teen/elder): one rig, proportions from HUMAN_ROLES. The
/// adult row is the identity, and a child is genuinely shorter-legged, not a
/// scaled adult.
#[test]
fn human_roles_scale_the_figure_by_their_profile() {
    use phantom_core::rig::human_proportions;
    for actor in [BuiltinActor::Humanoid, BuiltinActor::Child, BuiltinActor::Teen, BuiltinActor::Elder] {
        let p = human_proportions(actor);
        let role = actor.human_role().unwrap();
        // The figure's top matches the profile's published height.
        assert!((p.top - role.height_m).abs() < 0.01, "{actor:?}: top {} vs profile {}", p.top, role.height_m);
        // Regions stack: waist < torso_top < top, every scale positive.
        assert!(0.0 < p.waist && p.waist < p.torso_top && p.torso_top < p.top, "{actor:?}: {p:?}");
    }
    // The adult is the identity: its geometry keeps the exact reference numbers.
    let adult = human_proportions(BuiltinActor::Humanoid);
    assert!((adult.waist - 0.95).abs() < 1e-6 && (adult.torso_top - 1.52).abs() < 1e-6 && (adult.top - 1.74).abs() < 1e-6);
    assert!((adult.leg - 1.0).abs() < 1e-6 && (adult.torso - 1.0).abs() < 1e-6 && (adult.head - 1.0).abs() < 1e-6);
    // A child is not a scaled adult: its legs are short relative to its torso
    // (a uniform scale would keep the adult leg/torso ratio).
    let child = human_proportions(BuiltinActor::Child);
    let ratio = |p: &phantom_core::rig::HumanProportions| p.leg / p.torso;
    assert!(ratio(&child) < ratio(&adult) - 0.02, "child leg/torso {} vs adult {}", ratio(&child), ratio(&adult));
}

/// Roles share the adult rig and poses: same bone names, same pose set, and
/// every pose is lower for a child in absolute terms.
#[test]
fn human_roles_share_the_rig_and_poses_but_scale_them() {
    use phantom_core::rig::{actor_bones, sample_actor_pose};
    let adult_bones = actor_bones(BuiltinActor::Humanoid);
    for actor in [BuiltinActor::Child, BuiltinActor::Teen, BuiltinActor::Elder] {
        let bones = actor_bones(actor);
        assert_eq!(bones.len(), adult_bones.len(), "{actor:?}: the same topology");
        for (a, b) in adult_bones.iter().zip(bones.iter()) {
            assert_eq!(a.name, b.name, "{actor:?}: attachments find bones by name");
            assert_eq!(a.parent, b.parent);
        }
        assert_eq!(actor.poses(), BuiltinActor::Humanoid.poses(), "{actor:?}: the same pose set");
        for pose in actor.poses() {
            sample_actor_pose(actor, *pose).unwrap();
        }
        let role = actor.human_role().unwrap();
        let footprint = actor.footprint(phantom_core::ActorPose::Stand);
        assert!((footprint.half[1] * 2.0 - role.height_m).abs() < 0.01, "{actor:?}: the stage clamp uses the profile height");
    }
    // A child stands lower than an adult, in the rig and in the clamp.
    let child_hips = actor_bones(BuiltinActor::Child)[0].rest_translation.y;
    assert!(child_hips < 0.70, "child hips at {child_hips}");
    let child_fp = BuiltinActor::Child.footprint(phantom_core::ActorPose::Stand);
    let adult_fp = BuiltinActor::Humanoid.footprint(phantom_core::ActorPose::Stand);
    assert!(child_fp.half[1] < adult_fp.half[1]);
    // Sitting scales too: a child sits lower than an adult.
    let child_sit = BuiltinActor::Child.footprint(phantom_core::ActorPose::Sit);
    let adult_sit = BuiltinActor::Humanoid.footprint(phantom_core::ActorPose::Sit);
    assert!(child_sit.half[1] < adult_sit.half[1], "child sit {} vs adult {}", child_sit.half[1], adult_sit.half[1]);
}

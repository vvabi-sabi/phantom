use glam::Vec3;
use phantom_core::{project::*, rig::*, *};

const ACTORS: [BuiltinActor; 4] = [BuiltinActor::Humanoid, BuiltinActor::Wolf, BuiltinActor::Horse, BuiltinActor::Bird];

#[test]
fn legacy_and_new_rigs_roundtrip() {
    assert_eq!(serde_json::to_string(&RigRef::Builtin).unwrap(), "\"Builtin\"");
    assert_eq!(serde_json::from_str::<RigRef>("\"Builtin\"").unwrap().builtin_actor(), Some(BuiltinActor::Humanoid));
    for actor in ACTORS {
        let rig = RigRef::BuiltinActor { actor };
        assert_eq!(serde_json::from_str::<RigRef>(&serde_json::to_string(&rig).unwrap()).unwrap().builtin_actor(), Some(actor));
        for action in actor.actions() {
            assert_eq!(*action, serde_json::from_str::<Action>(&serde_json::to_string(action).unwrap()).unwrap());
        }
    }
}

#[test]
fn skeletons_poses_are_finite_deterministic_and_actions_distinct() {
    for actor in ACTORS {
        let bones = actor_bones(actor);
        for (i, b) in bones.iter().enumerate() {
            assert!(b.parent.is_none_or(|p| p < i));
            assert!(b.rest_translation.is_finite());
            assert!(!bones[..i].iter().any(|other| other.name == b.name));
        }
        let mut signatures = Vec::new();
        for &action in actor.actions() {
            let mut signature = Vec::new();
            for t in [0., 0.13, 0.37, 1., 10000.] {
                let pose = sample_actor_action(actor, action, t).unwrap();
                let again = sample_actor_action(actor, action, t).unwrap();
                assert_eq!(pose.bones.len(), bones.len());
                for (a, b) in pose.bones.iter().zip(again.bones) {
                    assert!(a.translation.is_finite() && a.rotation.is_finite());
                    assert!((a.rotation.length() - 1.).abs() < 1e-5);
                    assert_eq!(a.translation, b.translation);
                    assert_eq!(a.rotation, b.rotation);
                    signature.extend(a.translation.to_array());
                    signature.extend(a.rotation.to_array());
                }
            }
            assert!(!signatures.contains(&signature), "duplicate action for {actor:?}: {action:?}");
            signatures.push(signature);
        }
        assert!(sample_actor_action(actor, if actor == BuiltinActor::Humanoid { Action::Fly } else { Action::Jump }, 0.).is_err());
    }
}

#[test]
fn timeline_uses_profile_and_seeks_on_twos() {
    for actor in ACTORS {
        let mut p = Phantom {
            id: PhantomId(1),
            name: "actor".into(),
            rig: RigRef::BuiltinActor { actor },
            motion: vec![],
            clip: None,
            clip_track: vec![],
            takes: vec![],
            active_take: None,
            outline_px: 2.,
            scale: 1.0,
            material_patches: Vec::new(),
        };
        let range = FrameRange::new(0, 96).unwrap();
        assert_eq!(evaluate(&p, &range, 0, false).action, actor.default_action());
        for &action in actor.actions() {
            p.motion = vec![MotionKey { yaw: 0.0, frame: 0, position: Vec3::new(1., 2., 3.), action }];
            let poses: Vec<_> = (0..96).map(|f| evaluate(&p, &range, f, true)).collect();
            for f in (0..96).rev() {
                let seek = evaluate(&p, &range, f, true);
                assert_eq!(seek.action, action);
                assert_eq!(seek.root, Vec3::new(1., 2., 3.));
                for (a, b) in seek.pose.bones.iter().zip(&poses[(f / 2 * 2) as usize].pose.bones) {
                    assert_eq!(a.translation, b.translation);
                    assert_eq!(a.rotation, b.rotation);
                }
            }
        }
    }
}

#[test]
fn unsupported_actions_are_atomic_and_projects_roundtrip() {
    let mut project = Project::new("actors");
    let mut history = History::default();
    let range = FrameRange::new(0, 48).unwrap();
    let scene = history.apply(&mut project, Command::AddScene { name: "s".into(), range }).unwrap().scene.unwrap();
    for actor in ACTORS {
        let phantom = history
            .apply(&mut project, Command::AddPhantom { scene, name: format!("{actor:?}"), rig: RigRef::BuiltinActor { actor } })
            .unwrap()
            .phantom
            .unwrap();
        let bad =
            MotionKey { yaw: 0.0, frame: 0, position: Vec3::ZERO, action: if actor == BuiltinActor::Humanoid { Action::Fly } else { Action::Jump } };
        for command in [
            Command::SetMotionKey { scene, phantom, key: bad },
            Command::ReplaceMotion { scene, phantom, motion: vec![bad] },
            Command::AddTake { scene, phantom, name: "bad".into(), range, motion: vec![bad] },
        ] {
            let before = serde_json::to_value(&project).unwrap();
            assert!(history.apply(&mut project, command).is_err());
            assert_eq!(before, serde_json::to_value(&project).unwrap());
        }
        for (i, &action) in actor.actions().iter().enumerate() {
            history
                .apply(
                    &mut project,
                    Command::SetMotionKey { scene, phantom, key: MotionKey { yaw: 0.0, frame: i as i64, position: Vec3::ZERO, action } },
                )
                .unwrap();
        }
        let motion = project.scenes[&scene].phantoms[&phantom].motion.clone();
        history.apply(&mut project, Command::AddTake { scene, phantom, name: "take".into(), range, motion }).unwrap();
    }
    let json = serde_json::to_value(&project).unwrap();
    let restored: Project = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(json, serde_json::to_value(restored).unwrap());
}

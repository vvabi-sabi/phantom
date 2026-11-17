use glam::Vec3;
use phantom_core::*;

fn setup() -> (Project, History, SceneId, PhantomId) {
    let mut p = Project::new("yaw");
    let mut h = History::default();
    let scene = h.apply(&mut p, Command::AddScene { name: "scene".into(), range: FrameRange::new(3, 50).unwrap() }).unwrap().scene.unwrap();
    let id = h.apply(&mut p, Command::AddPhantom { scene, name: "bird".into(), rig: RigRef::BuiltinActor { actor: BuiltinActor::Bird } }).unwrap().phantom.unwrap();
    (p, h, scene, id)
}

#[test]
fn legacy_serde_and_finite_deserialization() {
    // A pre-layout key carries `action`; it is dropped and the pose defaults.
    let key: MotionKey = serde_json::from_str(r#"{"frame":0,"position":[0,2,0],"action":"Fly"}"#).unwrap();
    assert_eq!(key.yaw, 0.0);
    assert_eq!(key.pose, ActorPose::Stand);
    let key = MotionKey { yaw: 1.25, ..key };
    assert_eq!(key, serde_json::from_str(&serde_json::to_string(&key).unwrap()).unwrap());
    for json in [r#"{"frame":0,"position":[0,2,0],"action":"Fly","yaw":1e100}"#, r#"{"frame":0,"position":[1e100,2,0],"action":"Fly"}"#] {
        assert!(serde_json::from_str::<MotionKey>(json).is_err());
    }
}

#[test]
fn wrap_seek_and_bird_height() {
    let (mut p, mut h, scene, id) = setup();
    for (frame, height, degrees) in [(3, 2., 170_f32), (7, 6., -170_f32), (11, -2., -150_f32)] {
        h.apply(&mut p, Command::SetMotionKey { scene, phantom: id, key: MotionKey { frame, position: Vec3::new(0., height, 0.), yaw: degrees.to_radians(), pose: ActorPose::Stand } }).unwrap();
    }
    let s = &p.scenes[&scene]; let bird = &s.phantoms[&id];
    let mid = evaluate(bird, 5);
    assert!((mid.yaw.abs() - std::f32::consts::PI).abs() < 1e-5);
    assert_eq!(mid.root.y, 4.);
    // A bird keyed below the floor ends at ground level. The stage clamp runs
    // through the rotation matrix, so the bound carries f32 round-off (~1e-8)
    // instead of landing on an exact zero.
    assert!(evaluate(bird, 11).root.y.abs() < 1e-6);
    // A seek gives exactly the value sequential evaluation gives.
    let playback: Vec<_> = (3..50).map(|f| evaluate(bird, f)).collect();
    for f in (3..50).rev() {
        let seek = evaluate(bird, f);
        assert_eq!(seek.yaw, playback[(f-3) as usize].yaw);
        assert_eq!(seek.root, playback[(f-3) as usize].root);
    }
    assert_eq!(evaluate(bird, 0).yaw, bird.motion[0].yaw);
}

#[test]
fn invalid_keys_and_motion_are_atomic() {
    let (mut p, mut h, scene, id) = setup();
    for (position, yaw) in [(Vec3::ZERO, f32::NAN), (Vec3::ZERO, f32::INFINITY), (Vec3::NAN, 0.)] {
        let key = MotionKey { frame: 3, position, yaw, pose: ActorPose::Stand };
        for command in [Command::SetMotionKey { scene, phantom: id, key }, Command::ReplaceMotion { scene, phantom: id, motion: vec![key] }] {
            let before = serde_json::to_string(&p).unwrap();
            assert!(h.apply(&mut p, command).is_err());
            assert_eq!(before, serde_json::to_string(&p).unwrap());
        }
    }
    // Failed commands did not add undo entries.
    assert!(h.undo(&mut p));
    assert!(p.scenes[&scene].phantoms.is_empty());
}

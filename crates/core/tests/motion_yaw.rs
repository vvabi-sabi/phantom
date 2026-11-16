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
    let key: MotionKey = serde_json::from_str(r#"{"frame":0,"position":[0,2,0],"action":"Fly"}"#).unwrap();
    assert_eq!(key.yaw, 0.0);
    let key = MotionKey { yaw: 1.25, ..key };
    assert_eq!(key, serde_json::from_str(&serde_json::to_string(&key).unwrap()).unwrap());
    for json in [r#"{"frame":0,"position":[0,2,0],"action":"Fly","yaw":1e100}"#, r#"{"frame":0,"position":[1e100,2,0],"action":"Fly"}"#] {
        assert!(serde_json::from_str::<MotionKey>(json).is_err());
    }
}

#[test]
fn wrap_seek_scene_relative_twos_and_bird_height() {
    let (mut p, mut h, scene, id) = setup();
    for (frame, height, degrees) in [(3, 2., 170_f32), (7, 6., -170_f32), (11, -2., -150_f32)] {
        h.apply(&mut p, Command::SetMotionKey { scene, phantom: id, key: MotionKey { frame, position: Vec3::new(0., height, 0.), yaw: degrees.to_radians(), action: Action::Fly } }).unwrap();
    }
    let s = &p.scenes[&scene]; let bird = &s.phantoms[&id];
    let mid = evaluate(bird, &s.range, 5, false);
    assert!((mid.yaw.abs() - std::f32::consts::PI).abs() < 1e-5);
    assert_eq!(mid.root.y, 4.);
    assert_eq!(evaluate(bird, &s.range, 11, false).root.y, 0.);
    let playback: Vec<_> = (3..50).map(|f| evaluate(bird, &s.range, f, true)).collect();
    for f in (3..50).rev() {
        let seek = evaluate(bird, &s.range, f, true);
        assert_eq!(seek.yaw, playback[(f-3) as usize].yaw);
        assert_eq!(seek.root, playback[((f-3)/2*2) as usize].root);
        assert_eq!(seek.yaw, playback[((f-3)/2*2) as usize].yaw);
    }
    assert_eq!(evaluate(bird, &s.range, 0, false).yaw, bird.motion[0].yaw);
}

#[test]
fn invalid_keys_motion_and_takes_are_atomic() {
    let (mut p, mut h, scene, id) = setup();
    for (position, yaw) in [(Vec3::ZERO, f32::NAN), (Vec3::ZERO, f32::INFINITY), (Vec3::NAN, 0.)] {
        let key = MotionKey { frame: 3, position, yaw, action: Action::Fly };
        for command in [Command::SetMotionKey { scene, phantom: id, key }, Command::ReplaceMotion { scene, phantom: id, motion: vec![key] }, Command::AddTake { scene, phantom: id, name: "bad".into(), range: FrameRange::new(3, 5).unwrap(), motion: vec![key] }] {
            let before = serde_json::to_string(&p).unwrap();
            assert!(h.apply(&mut p, command).is_err());
            assert_eq!(before, serde_json::to_string(&p).unwrap());
        }
    }
    // Failed commands did not add undo entries.
    assert!(h.undo(&mut p));
    assert!(p.scenes[&scene].phantoms.is_empty());
}

#[test]
fn live_facing_preserves_stationary_yaw() {
    use phantom_core::control::*;
    let mut state = ControlState { yaw: 0.7, ..Default::default() };
    step_control(&mut state, &ControlInput::default(), 1./120.);
    assert_eq!(state.yaw, 0.7);
    step_control(&mut state, &ControlInput { move_x: 1., ..Default::default() }, 1./120.);
    assert_eq!(state.yaw, std::f32::consts::FRAC_PI_2);
    step_control(&mut state, &ControlInput::default(), 1./120.);
    assert_eq!(state.yaw, std::f32::consts::FRAC_PI_2);
}

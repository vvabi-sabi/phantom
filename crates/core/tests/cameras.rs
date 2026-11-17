//! Scene cameras: one camera entity per camera, the legacy migration, the
//! subject attachment and the rule that a camera must look at the scene.

use glam::Vec3;
use phantom_core::*;

fn setup(range: FrameRange) -> (Project, History, SceneId) {
    let mut p = Project::new("cameras");
    let mut h = History::default();
    let scene = h.apply(&mut p, Command::AddScene { name: "scene".into(), range }).unwrap().scene.unwrap();
    (p, h, scene)
}

fn key(frame: i64, pos: Vec3, yaw: f32) -> TransformKey {
    TransformKey::at(frame, pos, yaw, 0.0)
}

#[test]
fn a_new_scene_has_its_own_camera_and_it_is_active() {
    let (p, _, scene) = setup(FrameRange::new(0, 48).unwrap());
    let s = &p.scenes[&scene];
    assert_eq!(s.cameras.len(), 1);
    assert!(s.cameras.contains_key(&SCENE_CAMERA));
    assert_eq!(s.active_camera, SCENE_CAMERA);
    let own = &s.cameras[&SCENE_CAMERA];
    assert_eq!(own.range, s.range, "the scene's own camera covers the whole scene");
    assert!(own.target.is_none());
    // Round trip keeps the shape.
    let back: Project = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
    assert_eq!(back.scenes[&scene].cameras.len(), 1);
    assert_eq!(back.scenes[&scene].active_camera, SCENE_CAMERA);
}

/// A pre-camera manifest carries one `camera` and a `shots` map: the camera
/// becomes camera 0 and every shot becomes a camera with its own id.
#[test]
fn a_legacy_manifest_migrates_into_cameras() {
    let json = r#"{
        "id": 1, "name": "old", "range": {"start": 0, "end": 48},
        "background": null,
        "camera": {"keys": [{"frame": 0, "translation": [1, 2, 3], "yaw": 0.25, "pitch": 0.0}], "focal_mm": 50.0},
        "phantoms": {}, "shots": {
            "7": {"id": 7, "name": "close", "range": {"start": 0, "end": 24},
                  "camera": {"keys": [{"frame": 0, "translation": [4, 2, 0], "yaw": 1.5, "pitch": 0.0}], "focal_mm": 85.0}}
        }
    }"#;
    let mut scene: Scene = serde_json::from_str(json).unwrap();
    scene.normalize_legacy();
    assert_eq!(scene.cameras.len(), 2);
    let own = &scene.cameras[&SCENE_CAMERA];
    assert!((own.camera.focal_mm - 50.0).abs() < 1e-5, "the legacy camera lands in camera 0");
    assert_eq!(own.range, scene.range, "and covers the whole scene");
    let quoted = &scene.cameras[&CameraId(7)];
    assert_eq!(quoted.name, "close");
    assert_eq!(quoted.range, FrameRange::new(0, 24).unwrap());
    assert!((quoted.camera.focal_mm - 85.0).abs() < 1e-5);
    assert_eq!(scene.active_camera, SCENE_CAMERA);
    // Writing the scene never emits the legacy scene-level field again.
    let written: serde_json::Value = serde_json::to_value(&scene).unwrap();
    assert!(written.get("camera").is_none(), "the legacy field is not written: {written}");
    assert!(written["cameras"].is_object());
}

#[test]
fn cameras_are_added_activated_ranged_and_removed_with_their_clips() {
    let (mut p, mut h, scene) = setup(FrameRange::new(0, 48).unwrap());
    let first = h.apply(&mut p, Command::AddCamera { scene, name: "wide".into(), range: None }).unwrap().camera.unwrap();
    assert_ne!(first, SCENE_CAMERA);
    assert_eq!(p.scenes[&scene].active_camera, first, "a new camera becomes active");
    assert_eq!(p.scenes[&scene].cameras[&first].range, p.scenes[&scene].range);

    let second = h.apply(&mut p, Command::AddCamera { scene, name: "close".into(), range: Some(FrameRange::new(0, 24).unwrap()) }).unwrap().camera.unwrap();
    h.apply(&mut p, Command::ActivateCamera { scene, camera: first }).unwrap();
    assert_eq!(p.scenes[&scene].active_camera, first);

    // A film clip pinned to the narrow camera repeats its range.
    let clip = h
        .apply(&mut p, Command::AddFilmClip { scene, range: FrameRange::new(0, 24).unwrap(), camera: Some(second) })
        .unwrap()
        .clip
        .unwrap();
    assert!(h.apply(&mut p, Command::AddFilmClip { scene, range: FrameRange::new(0, 48).unwrap(), camera: Some(second) }).is_err());

    // Widening the camera retimes its clip.
    h.apply(&mut p, Command::SetCameraRange { scene, camera: second, range: FrameRange::new(0, 48).unwrap() }).unwrap();
    assert_eq!(p.film.iter().find(|c| c.id == clip).unwrap().range, FrameRange::new(0, 48).unwrap());

    // A range outside the scene is refused; camera 0 has no range of its own.
    assert!(h.apply(&mut p, Command::SetCameraRange { scene, camera: second, range: FrameRange::new(0, 96).unwrap() }).is_err());
    assert!(h.apply(&mut p, Command::SetCameraRange { scene, camera: SCENE_CAMERA, range: FrameRange::new(0, 24).unwrap() }).is_err());

    // Removing a camera takes its clips with it; the active camera falls back.
    let before_film = p.film.len();
    assert_eq!(before_film, 1);
    h.apply(&mut p, Command::RemoveCamera { scene, camera: second }).unwrap();
    assert!(p.film.is_empty(), "the clip went away with its camera");
    assert_eq!(p.scenes[&scene].cameras.len(), 2);
    assert!(h.apply(&mut p, Command::RemoveCamera { scene, camera: SCENE_CAMERA }).is_err(), "camera 0 stays");
    // Undo restores the camera and its clip in one step.
    assert!(h.undo(&mut p));
    assert_eq!(p.film.len(), 1);
    assert!(p.scenes[&scene].cameras.contains_key(&second));
}

#[test]
fn keys_lens_and_focus_are_per_camera() {
    let (mut p, mut h, scene) = setup(FrameRange::new(0, 48).unwrap());
    let cam = h.apply(&mut p, Command::AddCamera { scene, name: "a".into(), range: None }).unwrap().camera.unwrap();
    h.apply(&mut p, Command::SetCameraKey { scene, camera: cam, key: key(0, Vec3::new(0.0, 3.0, 14.0), 0.0) }).unwrap();
    h.apply(&mut p, Command::SetCameraKey { scene, camera: cam, key: key(24, Vec3::new(0.0, 3.0, 6.0), 0.0) }).unwrap();
    h.apply(&mut p, Command::SetCameraLens { scene, camera: cam, focal_mm: 85.0 }).unwrap();
    h.apply(&mut p, Command::SetCameraFocalKey { scene, camera: cam, frame: 24, focal_mm: 35.0 }).unwrap();
    let entry = &p.scenes[&scene].cameras[&cam];
    assert_eq!(entry.camera.keys.len(), 2);
    assert!((entry.camera.focal_mm - 85.0).abs() < 1e-5);
    assert_eq!(entry.camera.sample(0).focal_mm, 85.0, "a key without its own focal follows the base lens");
    assert_eq!(entry.camera.sample(24).focal_mm, 35.0, "a key with its own focal wins");
    // The last key cannot go.
    h.apply(&mut p, Command::RemoveCameraKey { scene, camera: cam, frame: 0 }).unwrap();
    assert!(h.apply(&mut p, Command::RemoveCameraKey { scene, camera: cam, frame: 24 }).is_err());
    assert!(h.apply(&mut p, Command::SetCameraLens { scene, camera: cam, focal_mm: 0.0 }).is_err());
    assert!(h.apply(&mut p, Command::SetCameraKey { scene, camera: CameraId(999), key: key(0, Vec3::ZERO, 0.0) }).is_err());
    // A focal key never moves the camera: at a frame without a key the pose is
    // sampled from the track, so a zoom can be added anywhere inside the range.
    h.apply(&mut p, Command::SetCameraFocalKey { scene, camera: cam, frame: 7, focal_mm: 35.0 }).unwrap();
    let added = p.scenes[&scene].cameras[&cam].camera.keys.iter().find(|k| k.frame == 7).expect("the zoom key");
    assert_eq!(added.focal_mm, Some(35.0));
    assert_eq!(added.translation.z, 6.0, "the pose is sampled from the remaining key, not invented");
}

/// The rule: a camera must look at the scene. Letting it drift off the stage is
/// refused, and the error carries the number so the agent can act on it.
#[test]
fn a_key_that_looks_away_is_refused_with_numbers() {
    let (mut p, mut h, scene) = setup(FrameRange::new(0, 48).unwrap());
    let cam = h.apply(&mut p, Command::AddCamera { scene, name: "a".into(), range: None }).unwrap().camera.unwrap();
    // Inside the stage: always accepted, whatever it points at.
    h.apply(&mut p, Command::SetCameraKey { scene, camera: cam, key: key(0, Vec3::new(0.0, 2.0, 0.0), std::f32::consts::PI) }).unwrap();
    // Outside the stage, looking at it: accepted (most of the volume is in frame).
    h.apply(&mut p, Command::SetCameraKey { scene, camera: cam, key: key(1, Vec3::new(0.0, 3.0, 25.0), 0.0) }).unwrap();
    // Outside the stage, turned around: refused, and nothing changed.
    let before = serde_json::to_string(&p).unwrap();
    let err = h.apply(&mut p, Command::SetCameraKey { scene, camera: cam, key: key(2, Vec3::new(0.0, 3.0, 25.0), std::f32::consts::PI) }).unwrap_err();
    assert!(matches!(err, CoreError::CameraFacesAway { .. }), "{err:?}");
    let message = err.to_string();
    assert!(message.contains("looks away"), "{message}");
    assert!(message.contains('%'), "{message}");
    assert_eq!(before, serde_json::to_string(&p).unwrap());
    // The failing key was not inserted.
    assert_eq!(p.scenes[&scene].cameras[&cam].camera.keys.len(), 2);
}

#[test]
fn a_camera_can_be_attached_to_a_subject_and_the_attachment_is_validated() {
    let (mut p, mut h, scene) = setup(FrameRange::new(0, 48).unwrap());
    let phantom = h
        .apply(&mut p, Command::AddPhantom { scene, name: "hero".into(), rig: RigRef::BuiltinActor { actor: BuiltinActor::Humanoid } })
        .unwrap()
        .phantom
        .unwrap();
    h.apply(&mut p, Command::SetMotionKey { scene, phantom, key: MotionKey { frame: 0, position: Vec3::new(2.0, 0.0, 0.0), pose: ActorPose::Stand, yaw: 0.0 } }).unwrap();
    let cam = h.apply(&mut p, Command::AddCamera { scene, name: "chase".into(), range: None }).unwrap().camera.unwrap();
    // A third-person rig inside the stage: always inside, so the rule passes.
    h.apply(&mut p, Command::SetCameraKey { scene, camera: cam, key: key(0, Vec3::new(0.0, 1.5, -4.0), 0.0) }).unwrap();
    h.apply(
        &mut p,
        Command::SetCameraTarget { scene, camera: cam, target: Some(CameraTarget::new(CameraTargetMode::Follow, CameraSubject::Phantom(phantom))) },
    )
    .unwrap();
    let entry = &p.scenes[&scene].cameras[&cam];
    let target = entry.target.unwrap();
    assert_eq!(target.mode, CameraTargetMode::Follow);
    assert_eq!(target.subject, CameraSubject::Phantom(phantom));

    // The pose follows the subject: 4 m behind it (local -Z), 1.5 m up.
    let pose = phantom_core::camera_pose(&p, &p.scenes[&scene], Some(cam), 0);
    assert!((pose.translation - Vec3::new(2.0, 1.5, -4.0)).length() < 1e-4, "{:?}", pose.translation);

    // A subject that does not exist is refused; a dangling one (the phantom is
    // removed afterwards) degrades to the camera's own keys.
    assert!(h
        .apply(
            &mut p,
            Command::SetCameraTarget { scene, camera: cam, target: Some(CameraTarget::new(CameraTargetMode::Track, CameraSubject::Object(ObjectId(99)))) }
        )
        .is_err());
    h.apply(&mut p, Command::RemovePhantom { scene, phantom }).unwrap();
    let orphan = phantom_core::camera_pose(&p, &p.scenes[&scene], Some(cam), 0);
    assert_eq!(orphan.translation, Vec3::new(0.0, 1.5, -4.0), "the camera falls back to its own keys");

    // Detaching restores the free camera.
    h.apply(&mut p, Command::SetCameraTarget { scene, camera: cam, target: None }).unwrap();
    assert!(p.scenes[&scene].cameras[&cam].target.is_none());
}

#[test]
fn tracking_keeps_the_camera_in_place_and_turns_it() {
    let (mut p, mut h, scene) = setup(FrameRange::new(0, 48).unwrap());
    let phantom = h
        .apply(&mut p, Command::AddPhantom { scene, name: "hero".into(), rig: RigRef::BuiltinActor { actor: BuiltinActor::Humanoid } })
        .unwrap()
        .phantom
        .unwrap();
    h.apply(&mut p, Command::SetMotionKey { scene, phantom, key: MotionKey { frame: 0, position: Vec3::new(6.0, 0.0, 0.0), pose: ActorPose::Stand, yaw: 0.0 } }).unwrap();
    let cam = h.apply(&mut p, Command::AddCamera { scene, name: "fixed".into(), range: None }).unwrap().camera.unwrap();
    // In front of the medium stage, high enough to keep the volume in frame.
    h.apply(&mut p, Command::SetCameraKey { scene, camera: cam, key: key(0, Vec3::new(0.0, 3.0, 25.0), 0.0) }).unwrap();
    h.apply(
        &mut p,
        Command::SetCameraTarget { scene, camera: cam, target: Some(CameraTarget::new(CameraTargetMode::Track, CameraSubject::Phantom(phantom))) },
    )
    .unwrap();
    let pose = phantom_core::camera_pose(&p, &p.scenes[&scene], Some(cam), 0);
    assert_eq!(pose.translation, Vec3::new(0.0, 3.0, 25.0));
    assert!(pose.yaw < 0.0, "turned toward the subject at +X: {}", pose.yaw);
}

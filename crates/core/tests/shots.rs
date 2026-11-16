//! Named shots: commands (validation, atomicity, undo), camera selection,
//! legacy serde for scenes and clips without shots, film clips bound to shots.

use glam::Vec3;
use phantom_core::*;

fn setup() -> (Project, History, SceneId) {
    let mut project = Project::new("t");
    let mut h = History::default();
    let scene = h
        .apply(&mut project, Command::AddScene { name: "s".into(), range: FrameRange::new(0, 96).unwrap() })
        .unwrap()
        .scene
        .unwrap();
    (project, h, scene)
}

fn add_shot(h: &mut History, p: &mut Project, scene: SceneId, start: Frame, end: Frame) -> ShotId {
    h.apply(p, Command::AddShot { scene, name: "a".into(), range: FrameRange::new(start, end).unwrap() })
        .unwrap()
        .shot
        .unwrap()
}

#[test]
fn add_shot_validates_range_and_name() {
    let (mut p, mut h, scene) = setup();
    assert!(matches!(
        h.apply(&mut p, Command::AddShot { scene, name: "x".into(), range: FrameRange::new(48, 100).unwrap() }),
        Err(CoreError::ShotOutOfScene)
    ));
    assert!(matches!(
        h.apply(&mut p, Command::AddShot { scene, name: "x".into(), range: FrameRange::new(-1, 10).unwrap() }),
        Err(CoreError::ShotOutOfScene)
    ));
    assert!(matches!(
        h.apply(&mut p, Command::AddShot { scene, name: " ".into(), range: FrameRange::new(0, 10).unwrap() }),
        Err(CoreError::EmptyName)
    ));
    // No failed command changed the document.
    assert!(p.scenes[&scene].shots.is_empty());
}

#[test]
fn shot_commands_roundtrip_and_undo() {
    let (mut p, mut h, scene) = setup();
    let shot = add_shot(&mut h, &mut p, scene, 0, 48);
    h.apply(&mut p, Command::SetShotCameraKey {
        scene,
        shot,
        key: TransformKey::at(10, Vec3::new(1.0, 2.0, 3.0), 0.5, -0.1),
    })
    .unwrap();
    h.apply(&mut p, Command::SetShotFocal { scene, shot, focal_mm: 50.0 }).unwrap();
    h.apply(&mut p, Command::SetShotRange { scene, shot, range: FrameRange::new(8, 40).unwrap() }).unwrap();
    {
        let sh = &p.scenes[&scene].shots[&shot];
        assert_eq!(sh.range, FrameRange::new(8, 40).unwrap());
        assert_eq!(sh.camera.keys.len(), 2);
        assert_eq!(sh.camera.focal_mm, 50.0);
    }
    h.undo(&mut p); // SetShotRange
    h.undo(&mut p); // SetShotFocal
    h.undo(&mut p); // SetShotCameraKey
    {
        let sh = &p.scenes[&scene].shots[&shot];
        assert_eq!(sh.range, FrameRange::new(0, 48).unwrap());
        assert_eq!(sh.camera.keys.len(), 1);
        assert_eq!(sh.camera.focal_mm, 35.0);
    }
    h.redo(&mut p);
    assert_eq!(p.scenes[&scene].shots[&shot].camera.keys.len(), 2);
    // Removing a shot and undoing it.
    h.apply(&mut p, Command::RemoveShot { scene, shot }).unwrap();
    assert!(p.scenes[&scene].shots.is_empty());
    h.undo(&mut p);
    assert!(p.scenes[&scene].shots.contains_key(&shot));
}

#[test]
fn shot_camera_key_remove_keeps_last_key() {
    let (mut p, mut h, scene) = setup();
    let shot = add_shot(&mut h, &mut p, scene, 0, 48);
    // A shot has one key by default — nothing to remove.
    assert!(h.apply(&mut p, Command::RemoveShotCameraKey { scene, shot, frame: 0 }).is_err());
    h.apply(&mut p, Command::SetShotCameraKey {
        scene,
        shot,
        key: TransformKey::at(5, Vec3::ZERO, 0.0, 0.0),
    })
    .unwrap();
    h.apply(&mut p, Command::RemoveShotCameraKey { scene, shot, frame: 5 }).unwrap();
    assert_eq!(p.scenes[&scene].shots[&shot].camera.keys.len(), 1);
    // A nonexistent shot is reported in diagnostics.
    assert!(matches!(
        h.apply(&mut p, Command::SetShotFocal { scene, shot: ShotId(999), focal_mm: 50.0 }),
        Err(CoreError::ShotNotFound(_))
    ));
}

#[test]
fn film_clip_on_shot_requires_matching_range() {
    let (mut p, mut h, scene) = setup();
    let shot = add_shot(&mut h, &mut p, scene, 24, 48);
    // A range that differs from the shot range is rejected.
    assert!(matches!(
        h.apply(&mut p, Command::AddFilmClip { scene, range: FrameRange::new(0, 48).unwrap(), shot: Some(shot) }),
        Err(CoreError::ShotRangeMismatch)
    ));
    let clip = h
        .apply(&mut p, Command::AddFilmClip { scene, range: FrameRange::new(24, 48).unwrap(), shot: Some(shot) })
        .unwrap()
        .clip
        .unwrap();
    assert_eq!(p.film[0].shot, Some(shot));
    // A legacy clip without a shot keeps the previous scene-range validation.
    h.apply(&mut p, Command::AddFilmClip { scene, range: FrameRange::new(0, 96).unwrap(), shot: None }).unwrap();
    assert!(matches!(
        h.apply(&mut p, Command::AddFilmClip { scene, range: FrameRange::new(90, 100).unwrap(), shot: None }),
        Err(CoreError::ClipOutOfScene)
    ));
    // Trimming a shot clip must match the shot range.
    assert!(matches!(
        h.apply(&mut p, Command::SetFilmClipRange { clip, range: FrameRange::new(24, 40).unwrap() }),
        Err(CoreError::ShotRangeMismatch)
    ));
}

#[test]
fn shot_range_edit_retims_film_clips_and_remove_drops_them() {
    let (mut p, mut h, scene) = setup();
    let shot = add_shot(&mut h, &mut p, scene, 24, 48);
    h.apply(&mut p, Command::AddFilmClip { scene, range: FrameRange::new(24, 48).unwrap(), shot: Some(shot) }).unwrap();
    let legacy = h
        .apply(&mut p, Command::AddFilmClip { scene, range: FrameRange::new(0, 10).unwrap(), shot: None })
        .unwrap()
        .clip
        .unwrap();
    h.apply(&mut p, Command::SetShotRange { scene, shot, range: FrameRange::new(12, 36).unwrap() }).unwrap();
    assert_eq!(p.film[0].range, FrameRange::new(12, 36).unwrap());
    assert_eq!(p.film[1].range, FrameRange::new(0, 10).unwrap());
    // One undo rolls back both the shot and the clip retiming.
    h.undo(&mut p);
    assert_eq!(p.scenes[&scene].shots[&shot].range, FrameRange::new(24, 48).unwrap());
    assert_eq!(p.film[0].range, FrameRange::new(24, 48).unwrap());
    // Removing a shot drops only its clips; the legacy clip stays.
    h.apply(&mut p, Command::RemoveShot { scene, shot }).unwrap();
    assert_eq!(p.film.len(), 1);
    assert_eq!(p.film[0].id, legacy);
    h.undo(&mut p);
    assert_eq!(p.film.len(), 2);
    assert!(p.scenes[&scene].shots.contains_key(&shot));
}

#[test]
fn film_map_routes_shot_clip_to_scene_frames() {
    let (mut p, mut h, scene) = setup();
    let shot = add_shot(&mut h, &mut p, scene, 24, 48);
    h.apply(&mut p, Command::AddFilmClip { scene, range: FrameRange::new(0, 10).unwrap(), shot: None }).unwrap();
    h.apply(&mut p, Command::AddFilmClip { scene, range: FrameRange::new(24, 48).unwrap(), shot: Some(shot) }).unwrap();
    assert_eq!(film_len(&p.film), 10 + 24);
    // A shot clip maps into the scene range [24, 48).
    assert_eq!(film_map(&p.film, 10), Some((scene, 24)));
    assert_eq!(film_map(&p.film, 33), Some((scene, 47)));
    let (clip, frame) = film_map_clip(&p.film, 10).unwrap();
    assert_eq!(clip.shot, Some(shot));
    assert_eq!(frame, 24);
    // A legacy clip does not gain a shot.
    let (clip, _) = film_map_clip(&p.film, 0).unwrap();
    assert_eq!(clip.shot, None);
}

#[test]
fn camera_sample_interpolates_keys() {
    let cam = Camera {
        keys: vec![
            TransformKey::at(0, Vec3::new(0.0, 0.0, 0.0), 0.0, 0.0),
            TransformKey::at(10, Vec3::new(10.0, 0.0, 0.0), 1.0, 0.0),
        ],
        focal_mm: 35.0,
    };
    let mid = cam.sample(5);
    assert!((mid.translation.x - 5.0).abs() < 1e-5);
    assert!((mid.yaw - 0.5).abs() < 1e-5);
    // Before the first and after the last key — clamped.
    assert_eq!(cam.sample(-3).translation, Vec3::ZERO);
    assert_eq!(cam.sample(99).translation, Vec3::new(10.0, 0.0, 0.0));
    // Exact hit on a key.
    assert_eq!(cam.sample(10).translation, Vec3::new(10.0, 0.0, 0.0));
}

#[test]
fn camera_for_shot_falls_back_to_legacy() {
    let (mut p, mut h, scene) = setup();
    let shot = add_shot(&mut h, &mut p, scene, 0, 48);
    let s = &p.scenes[&scene];
    // Legacy: None — the scene camera; an unknown id — also the scene camera.
    assert!(std::ptr::eq(s.camera_for_shot(None), &s.camera));
    assert!(std::ptr::eq(s.camera_for_shot(Some(ShotId(999))), &s.camera));
    assert!(std::ptr::eq(s.camera_for_shot(Some(shot)), &s.shots[&shot].camera));
    assert_eq!(s.range_for_shot(None), s.range);
    assert_eq!(s.range_for_shot(Some(shot)), s.shots[&shot].range);
}

#[test]
fn legacy_manifests_without_shots_deserialize() {
    // Scene and film clip in the old format (no shots/shot fields).
    let scene_json = r#"{
        "id": 1, "name": "s", "range": {"start": 0, "end": 96},
        "background": null, "camera": {"keys": [{"frame": 0, "translation": [0.0, 1.4, 4.0], "yaw": 0.0, "pitch": 0.0}], "focal_mm": 35.0},
        "phantoms": {}
    }"#;
    let scene: Scene = serde_json::from_str(scene_json).unwrap();
    assert!(scene.shots.is_empty());
    assert!(scene.objects.is_empty());
    let clip_json = r#"{"id": 1, "scene": 1, "range": {"start": 0, "end": 48}}"#;
    let clip: FilmClip = serde_json::from_str(clip_json).unwrap();
    assert_eq!(clip.shot, None);
    // Full project round-trip with a shot.
    let (mut p, mut h, scene) = setup();
    add_shot(&mut h, &mut p, scene, 0, 48);
    let text = serde_json::to_string(&p).unwrap();
    let back: Project = serde_json::from_str(&text).unwrap();
    assert_eq!(back.scenes[&scene].shots.len(), 1);
}

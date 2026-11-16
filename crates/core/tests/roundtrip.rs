use glam::Vec3;
use phantom_core::*;

fn sample_project() -> (Project, History, SceneId, PhantomId) {
    let mut project = Project::new("test");
    let mut history = History::default();

    let asset = history
        .apply(
            &mut project,
            Command::RegisterAsset {
                kind: AssetKind::Image,
                path: "assets/bg.png".into(),
            },
        )
        .unwrap()
        .asset
        .unwrap();
    let scene = history
        .apply(
            &mut project,
            Command::AddScene { name: "s1".into(), range: FrameRange::new(0, 96).unwrap() },
        )
        .unwrap()
        .scene
        .unwrap();
    history
        .apply(
            &mut project,
            Command::SetBackground {
                scene,
                background: Background {
                    image: BackgroundSource::Image(asset),
                    arc_radians: 2.0,
                    radius: 6.0,
                    height: 4.0,
                    center: Vec3::new(0.0, 2.0, -5.0),
                    bulge: Default::default(),
                    fit: Default::default(),
                },
            },
        )
        .unwrap();
    let phantom = history
        .apply(&mut project, Command::AddPhantom { scene, name: "boy".into(), rig: RigRef::Builtin })
        .unwrap()
        .phantom
        .unwrap();
    history
        .apply(
            &mut project,
            Command::SetMotionKey {
                scene,
                phantom,
                key: MotionKey { yaw: 0.0, frame: 0, position: Vec3::ZERO, action: Action::Idle },
            },
        )
        .unwrap();
    history
        .apply(
            &mut project,
            Command::SetMotionKey {
                scene,
                phantom,
                key: MotionKey { yaw: 0.0, frame: 48, position: Vec3::new(2.0, 0.0, 0.0), action: Action::Walk },
            },
        )
        .unwrap();
    (project, history, scene, phantom)
}

#[test]
fn project_json_round_trip() {
    let (project, _h, scene, phantom) = sample_project();
    let json = serde_json::to_string_pretty(&project).unwrap();
    let back: Project = serde_json::from_str(&json).unwrap();
    assert_eq!(back.schema_version, SCHEMA_VERSION);
    assert_eq!(back.scenes.len(), 1);
    assert_eq!(back.scenes[&scene].phantoms[&phantom].motion.len(), 2);
}

#[test]
fn undo_restores_document() {
    let (mut project, mut history, scene, phantom) = sample_project();
    let before = serde_json::to_string(&project).unwrap();

    history
        .apply(
            &mut project,
            Command::SetMotionKey {
                scene,
                phantom,
                key: MotionKey { yaw: 0.0, frame: 24, position: Vec3::ONE, action: Action::Jump },
            },
        )
        .unwrap();
    assert_eq!(project.scenes[&scene].phantoms[&phantom].motion.len(), 3);
    assert!(history.undo(&mut project));
    assert_eq!(serde_json::to_string(&project).unwrap(), before);
    assert!(history.redo(&mut project));
    assert_eq!(project.scenes[&scene].phantoms[&phantom].motion.len(), 3);
}

#[test]
fn invalid_commands_do_not_mutate() {
    let (mut project, mut history, scene, phantom) = sample_project();
    let before = serde_json::to_string(&project).unwrap();

    // Non-finite value in a key.
    assert!(history
        .apply(
            &mut project,
            Command::SetMotionKey {
                scene,
                phantom,
                key: MotionKey { yaw: 0.0, frame: 5, position: Vec3::NAN, action: Action::Idle },
            },
        )
        .is_err());
    // Nonexistent phantom.
    assert!(history
        .apply(
            &mut project,
            Command::SetMotionKey {
                scene,
                phantom: PhantomId(999),
                key: MotionKey { yaw: 0.0, frame: 5, position: Vec3::ZERO, action: Action::Idle },
            },
        )
        .is_err());
    // Empty name.
    assert!(history
        .apply(
            &mut project,
            Command::AddScene { name: "  ".into(), range: FrameRange::new(0, 10).unwrap() },
        )
        .is_err());
    // Background with a nonexistent asset.
    assert!(history
        .apply(
            &mut project,
            Command::SetBackground {
                scene,
                background: Background {
                    image: BackgroundSource::Image(AssetId(999)),
                    arc_radians: 1.0,
                    radius: 1.0,
                    height: 1.0,
                    center: Vec3::ZERO, bulge: Default::default(), fit: Default::default() },
            },
        )
        .is_err());

    assert_eq!(serde_json::to_string(&project).unwrap(), before);
}

#[test]
fn replace_motion_is_atomic_and_single_undo() {
    let (mut project, mut history, scene, phantom) = sample_project();
    let before = project.scenes[&scene].phantoms[&phantom].motion.clone();
    // Re-"recording": new keys inside [12, 24), old keys outside it.
    let mut motion: Vec<MotionKey> = before
        .iter()
        .filter(|k| k.frame < 12 || k.frame >= 24)
        .copied()
        .collect();
    for f in 12..24 {
        motion.push(MotionKey { yaw: 0.0, frame: f, position: Vec3::new(f as f32 * 0.1, 0.0, 0.0), action: Action::Walk });
    }
    history
        .apply(&mut project, Command::ReplaceMotion { scene, phantom, motion })
        .unwrap();
    let m = &project.scenes[&scene].phantoms[&phantom].motion;
    assert_eq!(m.len(), 2 + 12);
    assert!(m.windows(2).all(|w| w[0].frame < w[1].frame), "sorted");
    // One undo step restores the whole track.
    assert!(history.undo(&mut project));
    assert_eq!(project.scenes[&scene].phantoms[&phantom].motion, before);
}

#[test]
fn brush_stroke_creates_layer_and_exposure() {
    let (mut project, mut history, scene, _p) = sample_project();
    let stroke = Stroke {
        points: vec![[10.0, 10.0], [50.0, 30.0]],
        color: [1.0, 0.0, 0.0, 1.0],
        width: 6.0,
        erase: false,
    };
    let out = history
        .apply(&mut project, Command::BrushStroke { scene, layer: None, frame: 24, stroke })
        .unwrap();
    let layer_id = out.layer.unwrap();
    let layer = &project.scenes[&scene].layers_2d[0];
    assert_eq!(layer.id, layer_id);
    assert_eq!(layer.exposures.len(), 1);
    // The exposure at 24 holds until the next one; at 10 it does not exist yet.
    assert!(layer.exposure_for_frame(24).is_some());
    assert!(layer.exposure_for_frame(47).is_some());
    assert!(layer.exposure_for_frame(10).is_none());
    assert_eq!(layer.onion(30), (Some(24), None));

    // A one-point stroke (single character/point) is rejected without mutation.
    let bad = Stroke { points: vec![[0.0, 0.0]], color: [0.0; 4], width: 1.0, erase: false };
    assert!(history
        .apply(&mut project, Command::BrushStroke { scene, layer: None, frame: 0, stroke: bad })
        .is_err());

    // Undo removes both the exposure and the created layer.
    assert!(history.undo(&mut project));
    assert!(project.scenes[&scene].layers_2d.is_empty());
    assert!(history.redo(&mut project));
    // Removing the exposure.
    history
        .apply(&mut project, Command::RemoveExposure { scene, layer: layer_id, frame: 24 })
        .unwrap();
    assert!(project.scenes[&scene].layers_2d[0].exposures.is_empty());
}

#[test]
fn undo_redo_keeps_ids_stable() {
    let (mut project, mut history, scene, phantom) = sample_project();
    let json_before = serde_json::to_string(&project).unwrap();
    // Removing and re-adding entities does not change the document.
    assert!(history.undo(&mut project)); // undo SetMotionKey
    assert!(history.redo(&mut project));
    assert!(history.undo(&mut project));
    assert!(history.undo(&mut project)); // undo AddPhantom
    assert!(history.redo(&mut project));
    assert!(history.redo(&mut project));
    assert_eq!(serde_json::to_string(&project).unwrap(), json_before);
    // New commands reference the same ids.
    history
        .apply(
            &mut project,
            Command::SetMotionKey {
                scene,
                phantom,
                key: MotionKey { yaw: 0.0, frame: 96, position: Vec3::ZERO, action: Action::Idle },
            },
        )
        .unwrap();
}

#[test]
fn takes_override_manual_in_range_only() {
    let (mut project, mut history, scene, phantom) = sample_project();
    // Take: [10, 20) moves left (manual keys move right).
    let motion: Vec<MotionKey> = (10..20)
        .map(|f| MotionKey { yaw: 0.0, frame: f, position: Vec3::new(-(f as f32), 0.0, 0.0), action: Action::Walk })
        .collect();
    let take = history
        .apply(&mut project, Command::AddTake {
            scene,
            phantom,
            name: "Take 1".into(),
            range: FrameRange::new(10, 20).unwrap(),
            motion,
        })
        .unwrap()
        .take
        .unwrap();
    {
        let scene_doc = &project.scenes[&scene];
        let p = &scene_doc.phantoms[&phantom];
        // Inside the take — its track; outside — the manual one.
        assert!(evaluate(p, &scene_doc.range, 15, false).root.x < 0.0, "inside the take");
        assert!(evaluate(p, &scene_doc.range, 48, false).root.x > 0.0, "outside — manual track");
    }
    // Deactivating restores the manual performance.
    history.apply(&mut project, Command::SetActiveTake { scene, phantom, take: None }).unwrap();
    let range = project.scenes[&scene].range;
    let p = &project.scenes[&scene].phantoms[&phantom];
    assert!(evaluate(p, &range, 15, false).root.x >= 0.0);
    // A take is one undo step; re-recording does not destroy the previous take.
    history.apply(&mut project, Command::SetActiveTake { scene, phantom, take: Some(take) }).unwrap();
    let take2 = history
        .apply(&mut project, Command::AddTake {
            scene,
            phantom,
            name: "Take 2".into(),
            range: FrameRange::new(10, 20).unwrap(),
            motion: vec![MotionKey { yaw: 0.0, frame: 10, position: Vec3::ZERO, action: Action::Idle }],
        })
        .unwrap()
        .take
        .unwrap();
    assert_eq!(project.scenes[&scene].phantoms[&phantom].takes.len(), 2);
    assert_eq!(project.scenes[&scene].phantoms[&phantom].active_take, Some(take2));
    // Removing the active take clears the selection.
    history.apply(&mut project, Command::RemoveTake { scene, phantom, take: take2 }).unwrap();
    assert_eq!(project.scenes[&scene].phantoms[&phantom].takes.len(), 1);
    assert_eq!(project.scenes[&scene].phantoms[&phantom].active_take, None);
    assert!(history.undo(&mut project));
    assert_eq!(project.scenes[&scene].phantoms[&phantom].takes.len(), 2);
}

#[test]
fn evaluate_on_sample_project() {
    let (project, _h, scene_id, phantom_id) = sample_project();
    let scene = &project.scenes[&scene_id];
    let phantom = &scene.phantoms[&phantom_id];
    let p0 = evaluate(phantom, &scene.range, 0, false);
    assert_eq!(p0.action, Action::Idle);
    let p48 = evaluate(phantom, &scene.range, 48, false);
    assert_eq!(p48.action, Action::Walk);
    assert!((p48.root.x - 2.0).abs() < 1e-4);
    // Walk bones deviate from rest.
    let rest = phantom_core::rig::Pose::rest();
    let dev: f32 = p48
        .pose
        .bones
        .iter()
        .zip(rest.bones.iter())
        .map(|(a, b)| (a.rotation - b.rotation).length())
        .sum();
    assert!(dev > 0.01, "walk must differ from the rest pose");
}

#[test]
fn legacy_scene_without_lighting_floor_loads_with_defaults() {
    // Pre-P0 manifest: no lighting/floor fields — the scene gets default
    // three-point lighting and does NOT gain a floor.
    let json = r#"{
        "id": 1,
        "name": "s",
        "range": {"start": 0, "end": 24},
        "background": null,
        "camera": {"keys": [{"frame": 0, "translation": [0.0, 1.4, 4.0], "yaw": 0.0, "pitch": 0.0}], "focal_mm": 35.0},
        "phantoms": {}
    }"#;
    let scene: Scene = serde_json::from_str(json).unwrap();
    assert_eq!(scene.lighting, SceneLighting::default());
    assert_eq!(scene.lighting.key.intensity, 8000.0);
    assert!(scene.floor.is_none());
}

#[test]
fn lighting_and_floor_round_trip() {
    let (mut project, mut history, scene, _p) = sample_project();
    let lighting = SceneLighting {
        key: SceneLight { color: [1.0, 0.8, 0.6], intensity: 12000.0, enabled: true, shadows: false },
        fill: SceneLight { color: [0.5, 0.6, 1.0], intensity: 0.0, enabled: false, shadows: false },
        rim: SceneLight { color: [1.0, 1.0, 1.0], intensity: 5000.0, enabled: true, shadows: false },
    };
    history.apply(&mut project, Command::SetLighting { scene, lighting }).unwrap();
    let floor = SceneFloor { color: [0.3, 0.3, 0.35] };
    history.apply(&mut project, Command::SetFloor { scene, floor: Some(floor) }).unwrap();

    let json = serde_json::to_string_pretty(&project).unwrap();
    let back: Project = serde_json::from_str(&json).unwrap();
    assert_eq!(back.scenes[&scene].lighting, lighting);
    assert_eq!(back.scenes[&scene].floor, Some(floor));
}

#[test]
fn lighting_floor_commands_validate_and_undo() {
    let (mut project, mut history, scene, _p) = sample_project();
    let before = serde_json::to_string(&project).unwrap();

    // Negative/infinite intensity and color are rejected without mutation.
    let mut bad = SceneLighting::default();
    bad.key.intensity = -1.0;
    assert!(history.apply(&mut project, Command::SetLighting { scene, lighting: bad }).is_err());
    bad.key.intensity = f32::NAN;
    assert!(history.apply(&mut project, Command::SetLighting { scene, lighting: bad }).is_err());
    let mut bad = SceneLighting::default();
    bad.rim.color = [1.0, -0.5, 0.0];
    assert!(history.apply(&mut project, Command::SetLighting { scene, lighting: bad }).is_err());
    assert!(history
        .apply(&mut project, Command::SetFloor { scene, floor: Some(SceneFloor { color: [f32::INFINITY, 0.0, 0.0] }) })
        .is_err());
    assert_eq!(serde_json::to_string(&project).unwrap(), before);

    // Valid changes are atomic and each one undoes in a single step.
    let lighting = SceneLighting {
        key: SceneLight { color: [1.0, 0.9, 0.8], intensity: 9000.0, enabled: true, shadows: false },
        ..SceneLighting::default()
    };
    history.apply(&mut project, Command::SetLighting { scene, lighting }).unwrap();
    history.apply(&mut project, Command::SetFloor { scene, floor: Some(SceneFloor { color: [0.2; 3] }) }).unwrap();
    assert_eq!(project.scenes[&scene].lighting, lighting);
    assert_eq!(project.scenes[&scene].floor, Some(SceneFloor { color: [0.2; 3] }));
    assert!(history.undo(&mut project));
    assert_eq!(project.scenes[&scene].floor, None);
    assert!(history.undo(&mut project));
    assert_eq!(project.scenes[&scene].lighting, SceneLighting::default());
    assert!(history.redo(&mut project));
    assert!(history.redo(&mut project));
    assert_eq!(project.scenes[&scene].floor, Some(SceneFloor { color: [0.2; 3] }));
    // Removing the floor is an undoable command too.
    history.apply(&mut project, Command::SetFloor { scene, floor: None }).unwrap();
    assert_eq!(project.scenes[&scene].floor, None);
    assert!(history.undo(&mut project));
    assert_eq!(project.scenes[&scene].floor, Some(SceneFloor { color: [0.2; 3] }));
}

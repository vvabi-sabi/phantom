//! Props: commands over scene objects, undo, manifest compatibility.

use glam::Vec3;
use phantom_core::*;

fn setup() -> (Project, History, SceneId, AssetId) {
    let mut project = Project::new("t");
    let mut history = History::default();
    let scene = history
        .apply(&mut project, Command::AddScene { name: "s".into(), range: FrameRange::new(0, 48).unwrap() })
        .unwrap()
        .scene
        .unwrap();
    let asset = history
        .apply(&mut project, Command::RegisterAsset { kind: AssetKind::ModelObj, path: "assets/rock/rock.obj".into() })
        .unwrap()
        .asset
        .unwrap();
    (project, history, scene, asset)
}

fn key(frame: Frame) -> ObjectKey {
    ObjectKey { frame, position: Vec3::ZERO, yaw: 0.0, pitch: 0.0, roll: 0.0, scale: Vec3::ONE }
}

#[test]
fn add_object_validates_asset_kind_and_scene() {
    let (mut project, mut history, scene, asset) = setup();
    let out = history.apply(&mut project, Command::AddObject { scene, name: "Rock".into(), asset }).unwrap();
    let id = out.object.unwrap();
    assert_eq!(project.scenes[&scene].objects[&id].asset, asset);

    // A non-model asset is rejected.
    let img = history
        .apply(&mut project, Command::RegisterAsset { kind: AssetKind::Image, path: "assets/bg.png".into() })
        .unwrap()
        .asset
        .unwrap();
    assert!(matches!(
        history.apply(&mut project, Command::AddObject { scene, name: "x".into(), asset: img }),
        Err(CoreError::WrongAssetKind(_))
    ));
    // Nonexistent asset and scene.
    assert!(matches!(
        history.apply(&mut project, Command::AddObject { scene, name: "x".into(), asset: AssetId(999) }),
        Err(CoreError::AssetNotFound(_))
    ));
    assert!(matches!(
        history.apply(&mut project, Command::AddObject { scene: SceneId(999), name: "x".into(), asset }),
        Err(CoreError::SceneNotFound(_))
    ));
    // Errors are atomic: there is still exactly one object.
    assert_eq!(project.scenes[&scene].objects.len(), 1);
}

#[test]
fn object_keys_set_replace_remove_and_undo() {
    let (mut project, mut history, scene, asset) = setup();
    let object = history
        .apply(&mut project, Command::AddObject { scene, name: "o".into(), asset })
        .unwrap()
        .object
        .unwrap();
    let revision_before = project.revision;

    // Keys are inserted sorted; a repeat on a frame replaces.
    history.apply(&mut project, Command::SetObjectKey { scene, object, key: key(10) }).unwrap();
    history.apply(&mut project, Command::SetObjectKey { scene, object, key: key(2) }).unwrap();
    {
        let o = &project.scenes[&scene].objects[&object];
        assert_eq!(o.keys.iter().map(|k| k.frame).collect::<Vec<_>>(), vec![2, 10]);
    }
    history
        .apply(&mut project, Command::SetObjectKey {
            scene,
            object,
            key: ObjectKey { position: Vec3::X, ..key(10) },
        })
        .unwrap();
    assert_eq!(project.scenes[&scene].objects[&object].keys.len(), 2);
    assert_eq!(project.scenes[&scene].objects[&object].keys[1].position, Vec3::X);

    // Non-numeric values are rejected before the document changes.
    let bad = ObjectKey { position: Vec3::splat(f32::NAN), ..key(3) };
    assert!(matches!(
        history.apply(&mut project, Command::SetObjectKey { scene, object, key: bad }),
        Err(CoreError::NonFinite(_))
    ));
    let zero_scale = ObjectKey { scale: Vec3::ZERO, ..key(3) };
    assert!(matches!(
        history.apply(&mut project, Command::ReplaceObjectMotion { scene, object, keys: vec![zero_scale] }),
        Err(CoreError::NonFinite(_))
    ));
    assert_eq!(project.scenes[&scene].objects[&object].keys.len(), 2);

    // ReplaceObjectMotion — one command, one undo; sorts and deduplicates.
    history
        .apply(&mut project, Command::ReplaceObjectMotion { scene, object, keys: vec![key(5), key(1), key(5)] })
        .unwrap();
    assert_eq!(project.scenes[&scene].objects[&object].keys.iter().map(|k| k.frame).collect::<Vec<_>>(), vec![1, 5]);
    assert!(history.undo(&mut project));
    assert_eq!(project.scenes[&scene].objects[&object].keys.len(), 2);

    // RemoveObjectKey and RemoveObject.
    history.apply(&mut project, Command::RemoveObjectKey { scene, object, frame: 2 }).unwrap();
    assert_eq!(project.scenes[&scene].objects[&object].keys.len(), 1);
    history.apply(&mut project, Command::RemoveObject { scene, object }).unwrap();
    assert!(project.scenes[&scene].objects.is_empty());
    assert!(matches!(
        history.apply(&mut project, Command::RemoveObject { scene, object }),
        Err(CoreError::ObjectNotFound(_))
    ));

    // Full rollback to an empty document (including setup); the revision grows.
    while history.undo(&mut project) {}
    assert!(project.revision > revision_before);
    assert!(project.scenes.is_empty() && project.assets.is_empty());
}

#[test]
fn legacy_scene_without_objects_loads_empty() {
    // Scene manifest from before props existed: the objects field is absent.
    let json = r#"{
        "id": 1, "name": "s", "range": {"start": 0, "end": 24},
        "background": null,
        "camera": {"keys": [{"frame": 0, "translation": [0.0, 1.4, 4.0], "yaw": 0.0, "pitch": 0.0}], "focal_mm": 35.0},
        "phantoms": {}, "layers_2d": [], "audio_tracks": [],
        "lighting": {"key": {"color": [1.0, 1.0, 1.0], "intensity": 8000.0, "enabled": true},
                     "fill": {"color": [0.9, 0.95, 1.0], "intensity": 3000.0, "enabled": true},
                     "rim": {"color": [1.0, 0.95, 0.9], "intensity": 6000.0, "enabled": true}},
        "floor": null
    }"#;
    let scene: Scene = serde_json::from_str(json).unwrap();
    assert!(scene.objects.is_empty());
    // And it serializes back.
    let back: Scene = serde_json::from_str(&serde_json::to_string(&scene).unwrap()).unwrap();
    assert!(back.objects.is_empty());
}

#[test]
fn object_serde_round_trip_with_optional_fields() {
    // An old key without pitch/roll/scale gets default values.
    let json = r#"{"frame": 4, "position": [1.0, 0.0, 0.0], "yaw": 0.5}"#;
    let legacy: ObjectKey = serde_json::from_str(json).unwrap();
    assert_eq!(legacy.pitch, 0.0);
    assert_eq!(legacy.roll, 0.0);
    assert_eq!(legacy.scale, Vec3::ONE);

    let mut keys = vec![key(0), key(24)];
    keys[1].position = Vec3::new(2.0, 0.0, 0.0);
    let obj = SceneObject { id: ObjectId(3), name: "o".into(), asset: AssetId(1), builtin: None, keys , attach: None, material_patches: Vec::new() };
    let back: SceneObject = serde_json::from_str(&serde_json::to_string(&obj).unwrap()).unwrap();
    assert_eq!(back.keys.len(), 2);
    assert_eq!(back.keys[1].position, Vec3::new(2.0, 0.0, 0.0));
}

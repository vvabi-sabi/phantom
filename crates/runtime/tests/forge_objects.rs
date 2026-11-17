//! Forge for props without GPU: scene objects spawn as static entities with
//! the model's parts/materials; apply_scene_frame sets the frame's transform
//! (lerp/hold) on the object root.

use bevy::prelude::*;
use glam::Vec3;
use phantom_core::*;
use phantom_runtime::render::{apply_scene_frame, forge_scene};

fn test_world() -> World {
    let mut world = World::new();
    world.insert_resource(Assets::<Image>::default());
    world.insert_resource(Assets::<Mesh>::default());
    world.insert_resource(Assets::<StandardMaterial>::default());
    world.insert_resource(Assets::<bevy::mesh::skinning::SkinnedMeshInverseBindposes>::default());
    world.insert_resource(phantom_runtime::render::DrawCache::default());
    world
}

/// Project with an OBJ prop in a temporary directory.
fn project_with_object(dir: &std::path::Path) -> (Project, SceneId, ObjectId) {
    std::fs::create_dir_all(dir.join("assets/quad")).unwrap();
    std::fs::write(
        dir.join("assets/quad/quad.obj"),
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1/1 2/2/1 3/3/1\n",
    )
    .unwrap();
    let mut project = Project::new("t");
    let mut history = History::default();
    let scene = history
        .apply(&mut project, phantom_core::Command::AddScene { name: "s".into(), range: FrameRange::new(0, 48).unwrap() })
        .unwrap()
        .scene
        .unwrap();
    let asset = history
        .apply(&mut project, phantom_core::Command::RegisterAsset { kind: AssetKind::ModelObj, path: "assets/quad/quad.obj".into(), bounds: None })
        .unwrap()
        .asset
        .unwrap();
    let object = history
        .apply(&mut project, phantom_core::Command::AddObject { scene, name: "quad".into(), asset })
        .unwrap()
        .object
        .unwrap();
    let key = |frame: Frame, x: f32| ObjectKey { frame, position: Vec3::new(x, 0.0, 0.0), yaw: 0.0, pitch: 0.0, roll: 0.0, scale: Vec3::ONE };
    history.apply(&mut project, phantom_core::Command::SetObjectKey { scene, object, key: key(0, 0.0) }).unwrap();
    history.apply(&mut project, phantom_core::Command::SetObjectKey { scene, object, key: key(24, 2.4) }).unwrap();
    (project, scene, object)
}

#[test]
fn forge_spawns_static_object_and_apply_moves_root() {
    let dir = std::env::temp_dir().join(format!("phantom-forge-objects-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let (project, scene_id, object_id) = project_with_object(&dir);

    let mut world = test_world();
    let target = Handle::<Image>::default();
    let scene = project.scenes[&scene_id].clone();
    let forge = forge_scene(&mut world, &project, &scene, &dir, &target).unwrap();

    // Object root + a child carrying the part's mesh; no skinning (bind pose).
    let root = forge.objects[&object_id];
    let children = world.get::<Children>(root).expect("the root has parts");
    assert!(!children.is_empty());
    let mesh = world.get::<Mesh3d>(children[0]).expect("the part has a Mesh3d");
    assert!(world.resource::<Assets<Mesh>>().get(&mesh.0).and_then(|mesh| mesh.indices()).is_some_and(|indices| indices.len() >= 3), "OBJ triangles must not be lost on import");
    assert!(world.get::<MeshMaterial3d<StandardMaterial>>(children[0]).is_some());
    // Transform components are required by Mesh3d and must exist on the part.
    assert!(world.get::<Transform>(children[0]).is_some());
    assert!(world.get::<GlobalTransform>(children[0]).is_some());
    assert!(world.get::<bevy::mesh::skinning::SkinnedMesh>(children[0]).is_none());

    // Frame 12 is midway between the keys: the root is halfway.
    apply_scene_frame(&mut world, &Project::new("t"), &scene, &forge, 12, false, None);
    let t = world.get::<Transform>(root).unwrap();
    assert!((t.translation.x - 1.2).abs() < 1e-4, "x={}", t.translation.x);
    // Hold after the last key.
    apply_scene_frame(&mut world, &Project::new("t"), &scene, &forge, 47, false, None);
    let t = world.get::<Transform>(root).unwrap();
    assert!((t.translation.x - 2.4).abs() < 1e-4, "x={}", t.translation.x);

    let _ = std::fs::remove_dir_all(&dir);
}

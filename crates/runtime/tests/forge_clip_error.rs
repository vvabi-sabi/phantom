//! Forge without GPU: an unreadable motion clip is a diagnosable
//! `RenderError::Import`, not a panic (the GUI shows the status, the render job
//! fails).

use bevy::prelude::*;
use phantom_core::*;
use phantom_runtime::render::{forge_scene, RenderError};

fn test_world() -> World {
    let mut world = World::new();
    world.insert_resource(Assets::<Image>::default());
    world.insert_resource(Assets::<Mesh>::default());
    world.insert_resource(Assets::<StandardMaterial>::default());
    world.insert_resource(Assets::<bevy::mesh::skinning::SkinnedMeshInverseBindposes>::default());
    world.insert_resource(Assets::<phantom_runtime::render::outline::OutlineMaterial>::default());
    world.insert_resource(phantom_runtime::render::DrawCache::default());
    world.insert_resource(phantom_runtime::render::ClipCache::default());
    world
}

#[test]
fn unreadable_clip_is_error_not_panic() {
    let tmp = std::env::temp_dir().join(format!("phantom-forge-clip-err-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    // Unreadable "clip": the file exists, but it is not FBX.
    let bad_clip = tmp.join("broken.fbx");
    std::fs::write(&bad_clip, b"definitely not an fbx").unwrap();
    // A valid skinned model — the real FBX from the repository root.
    let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Hip Hop Dancing.fbx");

    let mut project = Project::new("t");
    let mut history = History::default();
    let scene = history
        .apply(&mut project, phantom_core::Command::AddScene { name: "s".into(), range: FrameRange::new(0, 48).unwrap() })
        .unwrap()
        .scene
        .unwrap();
    let model_asset = history
        .apply(&mut project, phantom_core::Command::RegisterAsset { kind: AssetKind::ModelFbx, path: model.clone() })
        .unwrap()
        .asset
        .unwrap();
    let clip_asset = history
        .apply(&mut project, phantom_core::Command::RegisterAsset { kind: AssetKind::MotionFbx, path: bad_clip.clone() })
        .unwrap()
        .asset
        .unwrap();
    let phantom = history
        .apply(&mut project, phantom_core::Command::AddPhantom { scene, name: "p".into(), rig: RigRef::Imported { asset: model_asset } })
        .unwrap()
        .phantom
        .unwrap();
    history
        .apply(
            &mut project,
            phantom_core::Command::AddClipUse {
                scene,
                phantom,
                clip_use: ClipUse {
                    id: ClipUseId(0),
                    asset: clip_asset,
                    stack: None,
                    range: FrameRange::new(0, 48).unwrap(),
                    speed: 1.0,
                    blend_in: 0.0,
                    blend_out: 0.0,
                    root_motion: RootMotion::Clip,
                    retarget: None,
                },
            },
        )
        .unwrap();

    let mut world = test_world();
    let target = Handle::<Image>::default();
    let scene_ref = &project.scenes[&scene];
    match forge_scene(&mut world, &project, scene_ref, std::path::Path::new("."), &target) {
        Err(err) => assert!(matches!(err, RenderError::Import(_)), "expected a clip import error, got: {err:?}"),
        Ok(_) => panic!("an unreadable clip must produce an error, not a successful forge"),
    }

    let _ = std::fs::remove_dir_all(&tmp);
}

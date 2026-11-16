//! Bone attachments (B2) without GPU: an object/light source follows a bone of
//! a builtin actor; the actor's scale lands on the root entity.

use bevy::prelude::*;
use glam::{Quat, Vec3};
use phantom_core::{Action, Attachment, BuiltinActor, Camera as _, FrameRange, MotionKey, ObjectId, Phantom, PhantomId, Project, RigRef, Scene, SceneFloor as _, SceneLighting, SceneObject, AssetId, SceneId, evaluate};
use phantom_core::Camera as CoreCamera;
use phantom_runtime::render::{apply_scene_frame, forge_scene};

fn test_world() -> World {
    let mut world = World::new();
    world.insert_resource(Assets::<Image>::default());
    world.insert_resource(Assets::<Mesh>::default());
    world.insert_resource(Assets::<StandardMaterial>::default());
    world.insert_resource(Assets::<phantom_runtime::render::outline::OutlineMaterial>::default());
    world.insert_resource(phantom_runtime::render::DrawCache::default());
    world.insert_resource(phantom_runtime::render::SceneCache::default());
    world.insert_resource(phantom_runtime::render::ClipCache::default());
    world.insert_resource(phantom_runtime::render::VideoCache::default());
    world.insert_resource(Assets::<bevy::mesh::skinning::SkinnedMeshInverseBindposes>::default());
    world
}

fn scene_with_phantom() -> Scene {
    let mut scene = Scene {
        id: SceneId(1),
        name: "s".into(),
        range: FrameRange::new(0, 48).unwrap(),
        background: None,
        camera: CoreCamera::default(),
        phantoms: Default::default(),
        layers_2d: vec![],
        audio_tracks: vec![],
        lighting: SceneLighting::default(),
        floor: None,
        objects: Default::default(),
        shots: Default::default(),
        lights: vec![],
        ambient: Default::default(),
        sky: None,
        grade: None,
        fog: None,
        vfx: vec![],
    };
    scene.lighting.key.enabled = false;
    scene.lighting.fill.enabled = false;
    scene.lighting.rim.enabled = false;
    let mut p = Phantom {
        id: PhantomId(2),
        name: "hero".into(),
        rig: RigRef::BuiltinActor { actor: BuiltinActor::Humanoid },
        motion: vec![MotionKey { frame: 0, position: Vec3::new(1.0, 0.0, -2.0), action: Action::Idle, yaw: 0.0 }],
        clip: None,
        clip_track: vec![],
        takes: vec![],
        active_take: None,
        outline_px: 2.0,
        scale: 1.0,
        material_patches: vec![],
    };
    // Frame 24: the actor moved ahead — the attachment must follow.
    p.motion.push(MotionKey { frame: 24, position: Vec3::new(3.0, 0.0, -2.0), action: Action::Idle, yaw: 0.0 });
    scene.phantoms.insert(p.id, p);
    scene
}

#[test]
fn attached_object_follows_bone_across_frames() {
    let mut world = test_world();
    let mut scene = scene_with_phantom();
    scene.objects.insert(
        ObjectId(5),
        SceneObject {
            id: ObjectId(5),
            name: "torch".into(),
            asset: AssetId(9), // the model is not loaded: the object's mesh is not needed for a transform test
            builtin: None,
            keys: vec![],
            attach: Some(Attachment {
                phantom: PhantomId(2),
                bone: "lower_arm_r".into(),
                offset_pos: Vec3::new(0.0, -0.1, 0.0),
                offset_yaw: 0.0,
                offset_pitch: 0.0,
                offset_roll: 0.0,
            }),
            material_patches: vec![],
        },
    );
    let project = Project::new("t");
    let target = Handle::<Image>::default();
    let forge = forge_scene(&mut world, &project, &scene, std::path::Path::new("."), &target).unwrap();
    // An object with no spawned entity is skipped — add a dummy one: the real
    // spawn path requires a model, so the math is checked through apply by
    // swapping the forge map for a dummy entity.
    let mut forge = forge;
    let probe = world.spawn((SceneEntity2, Transform::default(), GlobalTransform::default())).id();
    forge.objects.insert(ObjectId(5), probe);

    let expect = |frame: i64| {
        let phantom = &scene.phantoms[&PhantomId(2)];
        let state = evaluate(phantom, &scene.range, frame, false);
        let locals: Vec<(Vec3, Quat)> = state.pose.bones.iter().map(|b| (b.translation, b.rotation)).collect();
        let defs = phantom_core::rig::actor_bones(BuiltinActor::Humanoid);
        let parents: Vec<Option<usize>> = defs.iter().map(|b| b.parent).collect();
        let world_bones = phantom_core::rig::accumulate_bones(&parents, &locals);
        let idx = defs.iter().position(|b| b.name == "lower_arm_r").unwrap();
        let (bt, br) = world_bones[idx];
        let root_q = Quat::from_rotation_y(state.yaw);
        (state.root + root_q * bt) + (root_q * br) * Vec3::new(0.0, -0.1, 0.0)
    };

    apply_scene_frame(&mut world, &scene, &forge, 0, false, None, false, None);
    let t0 = world.get::<Transform>(probe).unwrap().translation;
    assert!((t0 - expect(0)).length() < 1e-4, "frame 0: {t0} != {}", expect(0));
    apply_scene_frame(&mut world, &scene, &forge, 24, false, None, false, None);
    let t24 = world.get::<Transform>(probe).unwrap().translation;
    assert!((t24 - expect(24)).length() < 1e-4, "the attachment follows the actor");
    // The actor moved +2 along x; idle bone sway adds centimeters on top.
    assert!((t24.x - t0.x - 2.0).abs() < 0.05, "the object followed the actor: dx={}", t24.x - t0.x);
}

#[derive(bevy::prelude::Component)]
struct SceneEntity2;

#[test]
fn phantom_scale_lands_on_root() {
    let mut world = test_world();
    let mut scene = scene_with_phantom();
    scene.phantoms.get_mut(&PhantomId(2)).unwrap().scale = 2.0;
    let project = Project::new("t");
    let target = Handle::<Image>::default();
    let forge = forge_scene(&mut world, &project, &scene, std::path::Path::new("."), &target).unwrap();
    apply_scene_frame(&mut world, &scene, &forge, 0, false, None, false, None);
    let root = forge.phantoms[&PhantomId(2)].root;
    assert_eq!(world.get::<Transform>(root).unwrap().scale, Vec3::splat(2.0));
}

//! Scene stage and the clamp to it (layout mode): every placement command
//! keeps the entity's whole bounding box inside the stage box.

use glam::Vec3;
use phantom_core::*;

fn setup(range: FrameRange) -> (Project, History, SceneId) {
    let mut p = Project::new("stage");
    let mut h = History::default();
    let scene = h.apply(&mut p, Command::AddScene { name: "scene".into(), range }).unwrap().scene.unwrap();
    (p, h, scene)
}

fn key(frame: i64, position: Vec3, pose: ActorPose) -> MotionKey {
    MotionKey { frame, position, pose, yaw: 0.0 }
}

#[test]
fn a_new_scene_gets_the_medium_stage_and_it_survives_a_roundtrip() {
    let (p, _, scene) = setup(FrameRange::new(0, 48).unwrap());
    let stage = p.scenes[&scene].stage;
    assert_eq!(stage.preset, Some(StagePreset::Medium));
    assert_eq!((stage.size.width, stage.size.height, stage.size.depth), (30.0, 8.0, 20.0));
    assert_eq!(stage.center, [0.0, 0.0]);
    // An old manifest without the field reads as the medium stage.
    let json = serde_json::to_string(&p).unwrap();
    let back: Project = serde_json::from_str(&json).unwrap();
    assert_eq!(back.scenes[&scene].stage, stage);
    let legacy: Scene = serde_json::from_str(
        &serde_json::to_string(&p.scenes[&scene]).unwrap().replace("\"stage\":", "\"unused_stage\":"),
    )
    .unwrap();
    assert_eq!(legacy.stage, Stage::default());
}

#[test]
fn setting_a_key_clamps_the_whole_body_and_reports_it() {
    let (mut p, mut h, scene) = setup(FrameRange::new(0, 48).unwrap());
    h.apply(&mut p, Command::SetStage { scene, stage: Stage::from_preset(StagePreset::Small) }).unwrap();
    let phantom = h
        .apply(&mut p, Command::AddPhantom { scene, name: "actor".into(), rig: RigRef::BuiltinActor { actor: BuiltinActor::Humanoid } })
        .unwrap()
        .phantom
        .unwrap();
    // A humanoid stands 1.74 m tall and takes ~0.6 × 0.5 m of floor.
    let out = h
        .apply(&mut p, Command::SetMotionKey { scene, phantom, key: key(0, Vec3::new(9.0, 0.0, -9.0), ActorPose::Stand) })
        .unwrap();
    assert_eq!(out.clamped.len(), 1, "the clamp is reported");
    let clamp = out.clamped[0];
    assert_eq!(clamp.kind, ClampKind::Phantom);
    assert_eq!(clamp.id, phantom.0);
    assert_eq!(clamp.frame, 0);
    assert_eq!(clamp.from, [9.0, 0.0, -9.0]);
    assert_eq!(clamp.to, [4.7, 0.0, -4.75]);
    let stored = p.scenes[&scene].phantoms[&phantom].motion[0];
    assert_eq!(stored.position, Vec3::new(4.7, 0.0, -4.75));
    // A key already inside the stage is untouched and silent.
    let out = h
        .apply(&mut p, Command::SetMotionKey { scene, phantom, key: key(10, Vec3::new(1.0, 0.0, 1.0), ActorPose::Stand) })
        .unwrap();
    assert!(out.clamped.is_empty());
    assert_eq!(p.scenes[&scene].phantoms[&phantom].motion[1].position, Vec3::new(1.0, 0.0, 1.0));
}

#[test]
fn replace_motion_clamps_every_key_and_a_prop_uses_its_catalog_size() {
    let (mut p, mut h, scene) = setup(FrameRange::new(0, 48).unwrap());
    h.apply(&mut p, Command::SetStage { scene, stage: Stage::from_preset(StagePreset::Small) }).unwrap();
    let phantom = h
        .apply(&mut p, Command::AddPhantom { scene, name: "a".into(), rig: RigRef::BuiltinActor { actor: BuiltinActor::Wolf } })
        .unwrap()
        .phantom
        .unwrap();
    let out = h
        .apply(
            &mut p,
            Command::ReplaceMotion {
                scene,
                phantom,
                motion: vec![key(0, Vec3::new(0.0, 0.0, 0.0), ActorPose::Stand), key(24, Vec3::new(0.0, 0.0, 50.0), ActorPose::Stand)],
            },
        )
        .unwrap();
    assert_eq!(out.clamped.len(), 1);
    assert_eq!(out.clamped[0].frame, 24);
    // A wolf is 1.1 m long: its box stops 0.55 m short of the wall.
    assert_eq!(out.clamped[0].to, [0.0, 0.0, 4.45]);
    // A catalog prop is clamped by its published dimensions, origin at the base center.
    let table = h
        .apply(&mut p, Command::AddBuiltinObject { scene, name: "table".into(), prop: "table".into(), key: Some(ObjectKey { frame: 0, position: Vec3::new(99.0, 0.0, 0.0), yaw: 0.0, pitch: 0.0, roll: 0.0, scale: Vec3::ONE }) })
        .unwrap();
    assert_eq!(table.clamped.len(), 1);
    assert_eq!(table.clamped[0].to, [4.3, 0.0, 0.0]);
    let object = table.object.unwrap();
    assert_eq!(p.scenes[&scene].objects[&object].keys[0].position, Vec3::new(4.3, 0.0, 0.0));
}

#[test]
fn a_stage_change_reclamps_everything_in_one_undo_step() {
    let (mut p, mut h, scene) = setup(FrameRange::new(0, 48).unwrap());
    let phantom = h
        .apply(&mut p, Command::AddPhantom { scene, name: "a".into(), rig: RigRef::BuiltinActor { actor: BuiltinActor::Humanoid } })
        .unwrap()
        .phantom
        .unwrap();
    h.apply(&mut p, Command::SetMotionKey { scene, phantom, key: key(0, Vec3::new(14.0, 0.0, 9.0), ActorPose::Stand) }).unwrap();
    let object = h
        .apply(&mut p, Command::AddBuiltinObject { scene, name: "chair".into(), prop: "chair".into(), key: Some(ObjectKey { frame: 12, position: Vec3::new(-14.0, 0.0, 0.0), yaw: 0.0, pitch: 0.0, roll: 0.0, scale: Vec3::ONE }) })
        .unwrap()
        .object
        .unwrap();
    let depth = h.undo_depth();
    let out = h.apply(&mut p, Command::SetStage { scene, stage: Stage::from_preset(StagePreset::Small) }).unwrap();
    assert_eq!(h.undo_depth(), depth + 1, "a stage change is one undo step");
    assert_eq!(out.clamped.len(), 2, "both the actor and the prop were moved");
    assert_eq!(p.scenes[&scene].phantoms[&phantom].motion[0].position, Vec3::new(4.7, 0.0, 4.75));
    assert_eq!(p.scenes[&scene].objects[&object].keys[0].position, Vec3::new(-4.775, 0.0, 0.0));
    assert_eq!(p.scenes[&scene].stage.preset, Some(StagePreset::Small));
    // Undo restores the positions and the old stage in one step.
    assert!(h.undo(&mut p));
    assert_eq!(p.scenes[&scene].phantoms[&phantom].motion[0].position, Vec3::new(14.0, 0.0, 9.0));
    assert_eq!(p.scenes[&scene].objects[&object].keys[0].position, Vec3::new(-14.0, 0.0, 0.0));
    assert_eq!(p.scenes[&scene].stage.preset, Some(StagePreset::Medium));
    assert!(h.redo(&mut p));
    assert_eq!(p.scenes[&scene].phantoms[&phantom].motion[0].position, Vec3::new(4.7, 0.0, 4.75));
}

#[test]
fn growing_an_actor_reclamps_it_and_a_bad_stage_is_rejected() {
    let (mut p, mut h, scene) = setup(FrameRange::new(0, 48).unwrap());
    h.apply(&mut p, Command::SetStage { scene, stage: Stage::from_preset(StagePreset::Small) }).unwrap();
    let phantom = h
        .apply(&mut p, Command::AddPhantom { scene, name: "a".into(), rig: RigRef::BuiltinActor { actor: BuiltinActor::Humanoid } })
        .unwrap()
        .phantom
        .unwrap();
    h.apply(&mut p, Command::SetMotionKey { scene, phantom, key: key(0, Vec3::new(4.7, 0.0, 0.0), ActorPose::Stand) }).unwrap();
    let out = h.apply(&mut p, Command::SetPhantomScale { scene, phantom, scale: 2.0 }).unwrap();
    assert_eq!(out.clamped.len(), 1, "twice the actor no longer fits where it stood");
    assert_eq!(out.clamped[0].to, [4.4, 0.0, 0.0]);
    assert_eq!(p.scenes[&scene].phantoms[&phantom].scale, 2.0);
    assert_eq!(p.scenes[&scene].phantoms[&phantom].motion[0].position, Vec3::new(4.4, 0.0, 0.0));
    // An unusable stage never reaches the document.
    let before = serde_json::to_string(&p).unwrap();
    for stage in [
        Stage { preset: None, size: StageSize::new(0.0, 4.0, 4.0), center: [0.0, 0.0] },
        Stage { preset: None, size: StageSize::new(10.0, -1.0, 10.0), center: [0.0, 0.0] },
        Stage { preset: None, size: StageSize::new(f32::NAN, 4.0, 4.0), center: [0.0, 0.0] },
        Stage { preset: None, size: StageSize::new(10.0, 4.0, 10.0), center: [f32::INFINITY, 0.0] },
    ] {
        assert!(h.apply(&mut p, Command::SetStage { scene, stage }).is_err());
        assert_eq!(before, serde_json::to_string(&p).unwrap());
    }
}

#[test]
fn an_imported_model_is_clamped_by_its_recorded_bounds() {
    let (mut p, mut h, scene) = setup(FrameRange::new(0, 48).unwrap());
    h.apply(&mut p, Command::SetStage { scene, stage: Stage::from_preset(StagePreset::Small) }).unwrap();
    let asset = h
        .apply(
            &mut p,
            Command::RegisterAsset {
                kind: AssetKind::ModelObj,
                path: "assets/rock.obj".into(),
                bounds: Some(ModelBounds { min: [-0.5, 0.0, -0.5], max: [0.5, 1.0, 0.5] }),
            },
        )
        .unwrap()
        .asset
        .unwrap();
    let object = h.apply(&mut p, Command::AddObject { scene, name: "rock".into(), asset }).unwrap().object.unwrap();
    let out = h
        .apply(&mut p, Command::SetObjectKey { scene, object, key: ObjectKey { frame: 0, position: Vec3::new(99.0, 0.0, 99.0), yaw: 0.0, pitch: 0.0, roll: 0.0, scale: Vec3::ONE } })
        .unwrap();
    assert_eq!(out.clamped[0].to, [4.5, 0.0, 4.5]);
    // A model without recorded bounds is clamped by its origin point.
    let bare = h
        .apply(&mut p, Command::RegisterAsset { kind: AssetKind::ModelObj, path: "assets/bare.obj".into(), bounds: None })
        .unwrap()
        .asset
        .unwrap();
    let object = h.apply(&mut p, Command::AddObject { scene, name: "bare".into(), asset: bare }).unwrap().object.unwrap();
    let out = h
        .apply(&mut p, Command::SetObjectKey { scene, object, key: ObjectKey { frame: 0, position: Vec3::new(99.0, 0.0, 0.0), yaw: 0.0, pitch: 0.0, roll: 0.0, scale: Vec3::ONE } })
        .unwrap();
    assert_eq!(out.clamped[0].to, [5.0, 0.0, 0.0]);
}

#[test]
fn the_clamp_is_per_axis_and_respects_the_stage_center() {
    let (mut p, mut h, scene) = setup(FrameRange::new(0, 48).unwrap());
    h.apply(
        &mut p,
        Command::SetStage { scene, stage: Stage { preset: None, size: StageSize::new(10.0, 4.0, 10.0), center: [5.0, -5.0] } },
    )
    .unwrap();
    let phantom = h
        .apply(&mut p, Command::AddPhantom { scene, name: "a".into(), rig: RigRef::BuiltinActor { actor: BuiltinActor::Humanoid } })
        .unwrap()
        .phantom
        .unwrap();
    let out = h
        .apply(&mut p, Command::SetMotionKey { scene, phantom, key: key(0, Vec3::new(-99.0, 99.0, 99.0), ActorPose::Stand) })
        .unwrap();
    // x: center 5, width 10 → [0, 10]; z: center −5, depth 10 → [−10, 0];
    // y: the 1.74 m actor under a 4 m ceiling. 1.74 is not exact in f32.
    let to = out.clamped[0].to;
    for (got, want) in to.iter().zip([0.3_f32, 2.26, -0.25]) {
        assert!((got - want).abs() < 1e-4, "{to:?} against {want}");
    }
}

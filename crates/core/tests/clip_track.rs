//! Phantom clip track: commands, validation, undo, legacy serde.

use phantom_core::*;

fn project_with_imported_phantom() -> (Project, History, SceneId, PhantomId, AssetId) {
    let mut project = Project::new("t");
    let mut history = History::default();
    let model = history
        .apply(&mut project, Command::RegisterAsset { kind: AssetKind::ModelFbx, path: "assets/actor.fbx".into() })
        .unwrap()
        .asset
        .unwrap();
    let motion = history
        .apply(&mut project, Command::RegisterAsset { kind: AssetKind::MotionFbx, path: "assets/move.fbx".into() })
        .unwrap()
        .asset
        .unwrap();
    let scene = history
        .apply(&mut project, Command::AddScene { name: "s".into(), range: FrameRange::new(0, 96).unwrap() })
        .unwrap()
        .scene
        .unwrap();
    let phantom = history
        .apply(&mut project, Command::AddPhantom { scene, name: "p".into(), rig: RigRef::Imported { asset: model } })
        .unwrap()
        .phantom
        .unwrap();
    (project, history, scene, phantom, motion)
}

fn clip_use(id: u64, asset: AssetId, start: Frame, end: Frame) -> ClipUse {
    ClipUse {
        id: ClipUseId(id),
        asset,
        stack: Some("Run".into()),
        range: FrameRange::new(start, end).unwrap(),
        speed: 1.5,
        blend_in: 0.25,
        blend_out: 0.5,
        root_motion: RootMotion::Clip,
        retarget: None,
    }
}

#[test]
fn add_and_remove_clip_use_undoable() {
    let (mut p, mut h, scene, phantom, motion) = project_with_imported_phantom();
    let u = clip_use(10, motion, 0, 48);
    let out = h.apply(&mut p, Command::AddClipUse { scene, phantom, clip_use: u.clone() }).unwrap();
    let id = out.clip_use.expect("the command assigns an id");
    let stored = &p.scenes[&scene].phantoms[&phantom].clip_track;
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].id, id, "the id is allocated by the command");
    assert_eq!(stored[0].range, u.range);

    // An overlap is rejected; the document does not change.
    let rev = p.revision;
    let err = h.apply(&mut p, Command::AddClipUse { scene, phantom, clip_use: clip_use(11, motion, 24, 72) }).unwrap_err();
    assert!(matches!(err, CoreError::ClipUseOverlap(_)));
    assert_eq!(p.revision, rev, "a rejected command does not touch the document");

    // Adjacent — allowed.
    h.apply(&mut p, Command::AddClipUse { scene, phantom, clip_use: clip_use(11, motion, 48, 96) }).unwrap();

    h.apply(&mut p, Command::RemoveClipUse { scene, phantom, clip_use: id }).unwrap();
    assert_eq!(p.scenes[&scene].phantoms[&phantom].clip_track.len(), 1);

    // Undo restores both clip uses, one step each.
    assert!(h.undo(&mut p));
    assert_eq!(p.scenes[&scene].phantoms[&phantom].clip_track.len(), 2);
    assert!(h.undo(&mut p));
    assert!(h.undo(&mut p));
    assert!(p.scenes[&scene].phantoms[&phantom].clip_track.is_empty());
}

#[test]
fn set_clip_track_validates_whole_track_atomically() {
    let (mut p, mut h, scene, phantom, motion) = project_with_imported_phantom();
    // Wrong asset kind is rejected.
    let img = h
        .apply(&mut p, Command::RegisterAsset { kind: AssetKind::Image, path: "assets/bg.png".into() })
        .unwrap()
        .asset
        .unwrap();
    let err = h
        .apply(&mut p, Command::SetClipTrack { scene, phantom, track: vec![clip_use(1, img, 0, 24)] })
        .unwrap_err();
    assert!(matches!(err, CoreError::WrongAssetKind(_)));

    // An overlapping pair inside one command is rejected as a whole.
    let track = vec![clip_use(1, motion, 0, 48), clip_use(2, motion, 24, 72)];
    let err = h.apply(&mut p, Command::SetClipTrack { scene, phantom, track }).unwrap_err();
    assert!(matches!(err, CoreError::ClipUseOverlap(_)));
    assert!(p.scenes[&scene].phantoms[&phantom].clip_track.is_empty(), "atomic: nothing written");

    // A valid track is sorted by start.
    let track = vec![clip_use(2, motion, 48, 96), clip_use(1, motion, 0, 48)];
    h.apply(&mut p, Command::SetClipTrack { scene, phantom, track }).unwrap();
    let ids: Vec<u64> = p.scenes[&scene].phantoms[&phantom].clip_track.iter().map(|u| u.id.0).collect();
    assert_eq!(ids, vec![1, 2]);
    assert!(h.undo(&mut p));
    assert!(p.scenes[&scene].phantoms[&phantom].clip_track.is_empty());
}

#[test]
fn legacy_phantom_json_loads_with_empty_track() {
    // Old manifest: the clip field is present, clip_track is absent.
    let json = r#"{
        "id": 7,
        "name": "boy",
        "rig": { "Imported": { "asset": 3 } },
        "motion": [],
        "clip": 5,
        "takes": [],
        "active_take": null,
        "outline_px": 2.0
    }"#;
    let p: Phantom = serde_json::from_str(json).unwrap();
    assert_eq!(p.clip, Some(AssetId(5)), "the legacy clip is preserved");
    assert!(p.clip_track.is_empty(), "serde default: empty track — legacy behavior");

    // Writing back does not lose clip.
    let out = serde_json::to_string(&p).unwrap();
    let p2: Phantom = serde_json::from_str(&out).unwrap();
    assert_eq!(p2.clip, Some(AssetId(5)));
}

#[test]
fn clip_use_serde_defaults() {
    // Minimal JSON: speed/root_motion/blends/stack use defaults.
    let json = r#"{
        "id": 1, "asset": 2, "stack": null,
        "range": { "start": 0, "end": 24 }
    }"#;
    let u: ClipUse = serde_json::from_str(json).unwrap();
    assert_eq!(u.speed, 1.0);
    assert_eq!(u.blend_in, 0.0);
    assert_eq!(u.blend_out, 0.0);
    assert_eq!(u.root_motion, RootMotion::Controller, "by default the controller owns the offset");
    assert_eq!(u.stack, None, "None — the longest stack (legacy)");
    assert_eq!(u.retarget, None, "by default — direct name match");

    let json = r#"{
        "id": 1, "asset": 2, "stack": null,
        "range": { "start": 0, "end": 24 },
        "retarget": "humanoid"
    }"#;
    let u: ClipUse = serde_json::from_str(json).unwrap();
    assert_eq!(u.retarget, Some(RetargetProfile::Humanoid));
    let out = serde_json::to_string(&u).unwrap();
    assert!(out.contains("\"retarget\":\"humanoid\""), "snake_case: {out}");
}

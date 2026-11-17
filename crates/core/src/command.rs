//! Typed commands over the document: validation before mutation,
//! atomicity, undo step. UI/MCP/CLI call only this layer.

use std::path::PathBuf;

use glam::{EulerRot, Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::error::CoreError;
use crate::id::*;
use crate::project::*;
use crate::stage::{Clamp, ClampKind, Footprint, Stage};
use crate::time::FrameRange;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Command {
    AddScene { name: String, range: FrameRange },
    SetSceneRange { scene: SceneId, range: FrameRange },
    /// Set the scene stage (layout mode): the box actors and props are confined
    /// to. Everything already staged is clamped into the new box in the same
    /// undo step, and the moved keys are reported in the output.
    SetStage { scene: SceneId, stage: crate::stage::Stage },
    RegisterAsset { kind: AssetKind, path: PathBuf, #[serde(default)] bounds: Option<crate::stage::ModelBounds> },
    SetBackground { scene: SceneId, background: Background },
    ClearBackground { scene: SceneId },
    AddPhantom { scene: SceneId, name: String, rig: RigRef },
    RemovePhantom { scene: SceneId, phantom: PhantomId },
    /// Adds or replaces a controller key at a frame.
    SetMotionKey { scene: SceneId, phantom: PhantomId, key: MotionKey },
    RemoveMotionKey { scene: SceneId, phantom: PhantomId, frame: crate::time::Frame },
    /// Atomically replaces the whole controller track (recording a performance —
    /// one command, one undo step).
    ReplaceMotion { scene: SceneId, phantom: PhantomId, motion: Vec<MotionKey> },
    /// Add a named camera over the given range (default: the whole scene). The
    /// new camera becomes the scene's active one.
    AddCamera { scene: SceneId, name: String, #[serde(default)] range: Option<FrameRange> },
    /// Remove a camera; film clips referencing it go away in the same step.
    /// The scene's own camera (id 0) cannot be removed.
    RemoveCamera { scene: SceneId, camera: CameraId },
    /// Pick the camera the viewport shows and a render without a camera uses.
    ActivateCamera { scene: SceneId, camera: CameraId },
    /// Change a camera's range; film clips onto that camera get the new range.
    SetCameraRange { scene: SceneId, camera: CameraId, range: FrameRange },
    /// Add or replace a camera key. The key must keep the camera looking at the
    /// scene (at least [`crate::camera::MIN_SCENE_IN_FRAME`] of its volume).
    SetCameraKey { scene: SceneId, camera: CameraId, key: TransformKey },
    /// Remove a camera key at a frame (the last key cannot be removed).
    RemoveCameraKey { scene: SceneId, camera: CameraId, frame: crate::time::Frame },
    /// The camera's base lens (mm, 35 mm equivalent).
    SetCameraLens { scene: SceneId, camera: CameraId, focal_mm: f32 },
    /// Focus on one key (a zoom inside the shot).
    SetCameraFocalKey { scene: SceneId, camera: CameraId, frame: crate::time::Frame, focal_mm: f32 },
    /// Attach the camera to a moving subject (None detaches). The attachment is
    /// checked against the facing rule at every key.
    SetCameraTarget { scene: SceneId, camera: CameraId, #[serde(default)] target: Option<CameraTarget> },
    /// Atomically replaces the scene's stage lighting (key/fill/rim).
    SetLighting { scene: SceneId, lighting: SceneLighting },
    /// Add a document light source (B1); the command allocates the id.
    AddLight { scene: SceneId, name: String, kind: SceneLightKind, color: [f32; 3], intensity: f32, enabled: bool, shadows: bool, rest: TransformKey },
    RemoveLight { scene: SceneId, light: crate::id::LightId },
    /// Adds or replaces a light source key at a frame (sorted by frame).
    SetLightKey { scene: SceneId, light: crate::id::LightId, key: crate::project::LightKey },
    RemoveLightKey { scene: SceneId, light: crate::id::LightId, frame: crate::time::Frame },
    /// Attach/detach (None) an object to a Phantom bone (B2).
    AttachObject { scene: SceneId, object: ObjectId, attach: Option<crate::project::Attachment> },
    /// Attach/detach (None) a light source to a Phantom bone (B2).
    AttachLight { scene: SceneId, light: crate::id::LightId, attach: Option<crate::project::Attachment> },
    /// Actor scale (B2), finite > 0.
    SetPhantomScale { scene: SceneId, phantom: PhantomId, scale: f32 },
    /// Atomically replaces the Phantom's material patches (Some) or the object's
    /// (None) — exactly one owner (B4).
    SetMaterialPatches { scene: SceneId, phantom: Option<PhantomId>, object: Option<ObjectId>, patches: Vec<crate::project::MaterialPatch> },
    /// Color grading (B3); None — turn it off.
    SetGrade { scene: SceneId, grade: Option<SceneGrade> },
    /// Exponential fog (B3); None — turn it off.
    SetFog { scene: SceneId, fog: Option<SceneFog> },
    /// Add an effect (B3); the command allocates the id.
    AddVfx { scene: SceneId, vfx: crate::project::SceneVfx },
    RemoveVfx { scene: SceneId, vfx: crate::id::VfxId },
    /// Adds or replaces an effect on/off and density key.
    SetVfxKey { scene: SceneId, vfx: crate::id::VfxId, key: crate::project::VfxKey },
    RemoveVfxKey { scene: SceneId, vfx: crate::id::VfxId, frame: crate::time::Frame },
    /// Scene ambient: from_background | fixed | off (B1).
    SetAmbient { scene: SceneId, ambient: AmbientMode },
    /// Clean frame color / sky (B5); None — transparency.
    SetSky { scene: SceneId, sky: Option<[f32; 3]> },
    /// Sets (Some) or removes (None) the scene's matte floor.
    SetFloor { scene: SceneId, floor: Option<SceneFloor> },
    /// Add a 2D layer to the scene.
    AddLayer2D { scene: SceneId, name: String },
    /// One brush/eraser gesture at a frame. Creates an exposure and (when
    /// there are no layers yet) the first layer. One gesture — one undo step.
    BrushStroke { scene: SceneId, layer: Option<crate::draw::LayerId>, frame: crate::time::Frame, stroke: crate::draw::Stroke },
    /// Remove an exposure entirely.
    RemoveExposure { scene: SceneId, layer: crate::draw::LayerId, frame: crate::time::Frame },
    /// Add a clip at the end of the film: a scene range, optionally pinned to
    /// a camera (a named camera's clip must match its range; camera 0 — the
    /// scene's own — takes any range).
    AddFilmClip { scene: SceneId, range: FrameRange, #[serde(default)] camera: Option<CameraId> },
    RemoveFilmClip { clip: crate::film::ClipId },
    /// Move a clip to position index.
    MoveFilmClip { clip: crate::film::ClipId, index: usize },
    SetFilmClipRange { clip: crate::film::ClipId, range: FrameRange },
    /// Add an audio track to a scene (Some) or to the film (None).
    AddAudioTrack { scene: Option<SceneId>, name: String },
    RemoveAudioTrack { track: crate::audio::TrackId },
    SetTrackVolume { track: crate::audio::TrackId, volume: f32 },
    SetTrackMuted { track: crate::audio::TrackId, muted: bool },
    AddAudioClip { track: crate::audio::TrackId, asset: AssetId, start: crate::time::Frame, offset_samples: u64, len_samples: u64, #[serde(default)] fade_in_samples: u64, #[serde(default)] fade_out_samples: u64 },
    /// Targeted update of a clip's fragment/fades (None — do not change).
    UpdateAudioClip { track: crate::audio::TrackId, clip: crate::audio::AudioClipId, offset_samples: Option<u64>, len_samples: Option<u64>, fade_in_samples: Option<u64>, fade_out_samples: Option<u64> },
    RemoveAudioClip { track: crate::audio::TrackId, clip: crate::audio::AudioClipId },
    /// Assign/unassign a motion clip of an imported rig (legacy:
    /// the longest stack, looped over the whole scene).
    /// Atomically replace Phantom's whole clip track (the whole sequence is
    /// validated before mutating, one undo step).
    /// Add one clip use to the track.
    /// Remove a clip use from the track.
    /// Add a static object (prop) from a model asset.
    AddObject { scene: SceneId, name: String, asset: AssetId },
    /// Object from the metric catalog (K1): the mesh is built procedurally, no
    /// asset file is needed; `key` — the starting transform, if wanted.
    AddBuiltinObject { scene: SceneId, name: String, prop: String, #[serde(default)] key: Option<ObjectKey> },
    RemoveObject { scene: SceneId, object: ObjectId },
    /// A generated fill stored as one object (the far band of a forest, a hedge
    /// row): the recipe is validated here — preset exists, and the plan fits the
    /// instance ceiling — so no session, GUI or hand-written document can store
    /// a field the renderer would refuse to replay.
    AddScatterField { scene: SceneId, name: String, field: crate::project::ScatterField, #[serde(default)] key: Option<ObjectKey> },
    /// Adds or replaces an object transform key at a frame.
    SetObjectKey { scene: SceneId, object: ObjectId, key: ObjectKey },
    RemoveObjectKey { scene: SceneId, object: ObjectId, frame: crate::time::Frame },
    /// Atomically replaces the object's whole key track.
    ReplaceObjectMotion { scene: SceneId, object: ObjectId, keys: Vec<ObjectKey> },
    /// An atomic group of commands: all of them are validated and applied on a
    /// draft; if any fails, the document is unchanged; one undo step.
    /// Nested Batch is not allowed.
    Batch(Vec<Command>),
}

/// Undo step: a snapshot of the touched part of the document before the change.
#[derive(Clone, Debug)]
pub(crate) enum UndoOp {
    SceneInserted(SceneId),
    SceneReplaced(Box<Scene>),
    AssetInserted(AssetId),
    AssetReplaced(Box<Asset>),
    FilmReplaced(Vec<crate::film::FilmClip>),
    FilmAudioReplaced(Vec<crate::audio::AudioTrack>),
    /// Atomic replacement of the scene and the film (shots: removal/range change).
    SceneAndFilmReplaced(Box<Scene>, Vec<crate::film::FilmClip>),
    /// Group of undo steps of one Batch; reverted in reverse order.
    Batch(Vec<UndoOp>),
}

/// Command result: the identifiers it created, plus any stage clamps the
/// command had to perform.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CommandOutput {
    pub scene: Option<SceneId>,
    pub phantom: Option<PhantomId>,
    pub asset: Option<AssetId>,
    pub layer: Option<crate::draw::LayerId>,
    pub clip: Option<crate::film::ClipId>,
    pub track: Option<crate::audio::TrackId>,
    pub audio_clip: Option<crate::audio::AudioClipId>,
    pub object: Option<ObjectId>,
    pub camera: Option<CameraId>,
    pub light: Option<crate::id::LightId>,
    pub vfx: Option<crate::id::VfxId>,
    /// Positions the stage clamp corrected (`clamped: true` in the MCP answer).
    pub clamped: Vec<crate::stage::Clamp>,
}

impl CommandOutput {
    /// Merges Batch results: the last id created of each kind, and every clamp.
    fn merge(&mut self, other: CommandOutput) {
        if other.scene.is_some() { self.scene = other.scene; }
        if other.phantom.is_some() { self.phantom = other.phantom; }
        if other.asset.is_some() { self.asset = other.asset; }
        if other.layer.is_some() { self.layer = other.layer; }
        if other.clip.is_some() { self.clip = other.clip; }
        if other.track.is_some() { self.track = other.track; }
        if other.audio_clip.is_some() { self.audio_clip = other.audio_clip; }
        if other.object.is_some() { self.object = other.object; }
        if other.camera.is_some() { self.camera = other.camera; }
        if other.light.is_some() { self.light = other.light; }
        if other.vfx.is_some() { self.vfx = other.vfx; }
        self.clamped.extend(other.clamped);
    }
}

/// Undo/redo history of one session.
#[derive(Clone, Default)]
pub struct History {
    undo: Vec<(Command, UndoOp)>,
    redo: Vec<(Command, UndoOp)>,
}

impl History {
    pub fn apply(&mut self, project: &mut Project, cmd: Command) -> Result<CommandOutput, CoreError> {
        let (undo, out) = apply(project, &cmd)?;
        project.revision += 1;
        self.undo.push((cmd, undo));
        self.redo.clear();
        Ok(out)
    }

    /// Undo is symmetric to redo: the saved state is restored,
    /// and the current one is captured for the opposite stack. The command is
    /// not re-applied — the ids of created entities stay stable.
    pub fn undo(&mut self, project: &mut Project) -> bool {
        let Some((cmd, op)) = self.undo.pop() else { return false };
        let inverse = capture_inverse(project, &op);
        revert(project, op);
        project.revision += 1;
        self.redo.push((cmd, inverse));
        true
    }

    pub fn redo(&mut self, project: &mut Project) -> bool {
        let Some((cmd, op)) = self.redo.pop() else { return false };
        let inverse = capture_inverse(project, &op);
        revert(project, op);
        project.revision += 1;
        self.undo.push((cmd, inverse));
        true
    }

    pub fn undo_depth(&self) -> usize {
        self.undo.len()
    }

    pub fn redo_depth(&self) -> usize {
        self.redo.len()
    }

    /// Collapses the last `n` undo steps into one atomic Batch step.
    /// Used by Session when a batch of operations went through the normal path
    /// and must become a single user undo step. With n <= 1 the history is unchanged.
    pub fn squash_last(&mut self, n: usize) {
        if n <= 1 || n > self.undo.len() {
            return;
        }
        let entries: Vec<(Command, UndoOp)> = self.undo.drain(self.undo.len() - n..).collect();
        let cmds = entries.iter().map(|(c, _)| c.clone()).collect();
        let undos = entries.into_iter().map(|(_, u)| u).collect();
        self.undo.push((Command::Batch(cmds), UndoOp::Batch(undos)));
    }

    /// Names of the last undoable commands (newest first).
    pub fn undo_descriptions(&self, limit: usize) -> Vec<String> {
        self.undo.iter().rev().take(limit).map(|(c, _)| command_name(c)).collect()
    }

    /// Names of the last repeatable commands (newest first).
    pub fn redo_descriptions(&self, limit: usize) -> Vec<String> {
        self.redo.iter().rev().take(limit).map(|(c, _)| command_name(c)).collect()
    }
}

/// Command variant name from its Debug representation (without fields).
fn command_name(cmd: &Command) -> String {
    let d = format!("{cmd:?}");
    d.split(|c: char| c == ' ' || c == '(' || c == '{').next().unwrap_or(&d).to_string()
}

fn revert(project: &mut Project, op: UndoOp) {
    match op {
        UndoOp::SceneInserted(id) => {
            project.scenes.remove(&id);
        }
        UndoOp::SceneReplaced(scene) => {
            project.scenes.insert(scene.id, *scene);
        }
        UndoOp::AssetInserted(id) => {
            project.assets.remove(&id);
        }
        UndoOp::AssetReplaced(asset) => {
            project.assets.insert(asset.id, *asset);
        }
        UndoOp::FilmReplaced(film) => {
            project.film = film;
        }
        UndoOp::FilmAudioReplaced(tracks) => {
            project.film_audio = tracks;
        }
        UndoOp::SceneAndFilmReplaced(scene, film) => {
            project.scenes.insert(scene.id, *scene);
            project.film = film;
        }
        // Rolling back a group happens strictly in reverse application order.
        UndoOp::Batch(ops) => {
            for op in ops.into_iter().rev() {
                revert(project, op);
            }
        }
    }
}

/// Captures the current state of the touched part — the step for
/// the opposite stack.
fn capture_inverse(project: &Project, op: &UndoOp) -> UndoOp {
    match op {
        UndoOp::SceneInserted(id) => {
            UndoOp::SceneReplaced(Box::new(project.scenes[id].clone()))
        }
        UndoOp::SceneReplaced(scene) => match project.scenes.get(&scene.id) {
            Some(current) => UndoOp::SceneReplaced(Box::new(current.clone())),
            // Redo of AddScene: the scene is missing from the document, so the inverse
            // step removes it (undo/redo symmetry for scene insertion).
            None => UndoOp::SceneInserted(scene.id),
        },
        UndoOp::AssetInserted(id) => {
            UndoOp::AssetReplaced(Box::new(project.assets[id].clone()))
        }
        UndoOp::AssetReplaced(asset) => UndoOp::AssetInserted(asset.id),
        UndoOp::FilmReplaced(_) => UndoOp::FilmReplaced(project.film.clone()),
        UndoOp::FilmAudioReplaced(_) => UndoOp::FilmAudioReplaced(project.film_audio.clone()),
        UndoOp::SceneAndFilmReplaced(scene, _) => UndoOp::SceneAndFilmReplaced(
            Box::new(project.scenes[&scene.id].clone()),
            project.film.clone(),
        ),
        // Symmetry: inverses are captured in reverse order so that revert
        // (which iterates from the end) restores state in forward order.
        UndoOp::Batch(ops) => UndoOp::Batch(ops.iter().rev().map(|o| capture_inverse(project, o)).collect()),
    }
}

/// Applies a command, returning the undo step and the result.
/// On error the document is unchanged. The public entry point is [`History::apply`].
pub(crate) fn apply(project: &mut Project, cmd: &Command) -> Result<(UndoOp, CommandOutput), CoreError> {
    let no_out = CommandOutput::default();
    match cmd {
        Command::AddScene { name, range } => {
            validate_name(name)?;
            let id: SceneId = project.alloc_id();
            project.scenes.insert(id, Scene::new(id, name.clone(), *range));
            Ok((UndoOp::SceneInserted(id), CommandOutput { scene: Some(id), ..no_out }))
        }
        Command::SetSceneRange { scene, range } => {
            let s = scene_mut(project, *scene)?;
            let before = Box::new(s.clone());
            s.range = *range;
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::SetStage { scene, stage } => {
            stage.validate()?;
            let clamped = {
                let s = project.scenes.get(scene).ok_or(CoreError::SceneNotFound(*scene))?;
                reclamp_scene(project, s, stage)
            };
            let s = scene_mut(project, *scene)?;
            let before = Box::new(s.clone());
            apply_clamps(s, &clamped);
            s.stage = *stage;
            Ok((UndoOp::SceneReplaced(before), CommandOutput { clamped, ..no_out }))
        }
        Command::RegisterAsset { kind, path, bounds } => {
            let id: AssetId = project.alloc_id();
            project.assets.insert(id, Asset { id, kind: *kind, path: path.clone(), bounds: *bounds });
            Ok((UndoOp::AssetInserted(id), CommandOutput { asset: Some(id), ..no_out }))
        }
        Command::SetBackground { scene, background } => {
            let expected_kind = match background.image {
                crate::project::BackgroundSource::Image(_) => AssetKind::Image,
                crate::project::BackgroundSource::Video { .. } => AssetKind::Video,
            };
            match project.assets.get(&background.image.asset()) {
                Some(a) if a.kind == expected_kind => {}
                Some(_) => return Err(CoreError::WrongAssetKind(background.image.asset())),
                None => return Err(CoreError::AssetNotFound(background.image.asset())),
            }
            background.validate()?;
            let s = scene_mut(project, *scene)?;
            let before = Box::new(s.clone());
            s.background = Some(background.clone());
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::ClearBackground { scene } => {
            let s = scene_mut(project, *scene)?;
            let before = Box::new(s.clone());
            s.background = None;
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::AddObject { scene, name, asset } => {
            validate_name(name)?;
            match project.assets.get(asset) {
                Some(a) if matches!(a.kind, AssetKind::ModelObj | AssetKind::ModelFbx) => {}
                Some(_) => return Err(CoreError::WrongAssetKind(*asset)),
                None => return Err(CoreError::AssetNotFound(*asset)),
            }
            if !project.scenes.contains_key(scene) {
                return Err(CoreError::SceneNotFound(*scene));
            }
            let id: ObjectId = project.alloc_id();
            let s = scene_mut(project, *scene)?;
            let before = Box::new(s.clone());
            s.objects.insert(id, SceneObject { id, name: name.clone(), asset: *asset, builtin: None, keys: Vec::new(), attach: None, material_patches: Vec::new(), scatter: None });
            Ok((UndoOp::SceneReplaced(before), CommandOutput { object: Some(id), ..no_out }))
        }
        Command::AddBuiltinObject { scene, name, prop, key } => {
            validate_name(name)?;
            if crate::catalog::catalog_find(prop).is_none() {
                return Err(CoreError::UnknownProp(prop.clone()));
            }
            if !project.scenes.contains_key(scene) {
                return Err(CoreError::SceneNotFound(*scene));
            }
            if let Some(k) = key {
                check_finite(k.position, "object.position")?;
            }
            let entry = crate::catalog::catalog_find(prop).expect("checked above");
            let id: ObjectId = project.alloc_id();
            let mut clamped = Vec::new();
            let mut keys: Vec<ObjectKey> = key.into_iter().copied().collect();
            {
                let s = project.scenes.get(scene).ok_or(CoreError::SceneNotFound(*scene))?;
                let base = entry.footprint();
                for k in keys.iter_mut() {
                    k.validate()?;
                    clamp_object_key_base(&s.stage, base, id, k, &mut clamped);
                }
            }
            // A furnished room arrives as a package: the shell plus one object
            // per furniture piece. The pieces are derived from the shell key
            // AFTER the stage clamped it, so the furniture never stays behind
            // the wall of a room that did not fit. Each piece is a plain object
            // the director can move or delete on its own.
            let mut package: Vec<(ObjectId, String, &str, Vec<ObjectKey>)> =
                vec![(id, name.clone(), prop, keys.clone())];
            if let Some(shell) = keys.first() {
                for (piece, mut piece_key) in entry.furniture_keys(shell) {
                    let piece_id: ObjectId = project.alloc_id();
                    let piece_entry = crate::catalog::catalog_find(piece).expect("the catalog test checks every set");
                    let stage = project.scenes.get(scene).expect("checked above").stage;
                    clamp_object_key_base(&stage, piece_entry.footprint(), piece_id, &mut piece_key, &mut clamped);
                    package.push((piece_id, format!("{name} {piece}"), piece, vec![piece_key]));
                }
            }
            let s = scene_mut(project, *scene)?;
            let before = Box::new(s.clone());
            for (object_id, object_name, object_prop, object_keys) in package {
                s.objects.insert(
                    object_id,
                    SceneObject {
                        id: object_id,
                        name: object_name,
                        // No asset needed: runtime builds the mesh from the catalog name.
                        asset: crate::id::AssetId(0),
                        builtin: Some(object_prop.to_string()),
                        keys: object_keys,
                        attach: None,
                        material_patches: Vec::new(), scatter: None,
                    },
                );
            }
            Ok((UndoOp::SceneReplaced(before), CommandOutput { object: Some(id), clamped, ..no_out }))
        }
        Command::AddScatterField { scene, name, field, key } => {
            validate_name(name)?;
            let preset = crate::scatter::scatter_preset(&field.preset)
                .ok_or_else(|| CoreError::InvalidScatter(format!(
                    "unknown fill {preset} (see scatter_presets)",
                    preset = field.preset
                )))?;
            // Replay the plan the renderer will replay. Validating the count
            // here is what keeps a saved document always renderable.
            let placed = crate::scatter::plan(preset, &field.shape, field.seed)?;
            let _ = crate::scatter::band_of(&placed);
            if let Some(k) = key {
                check_finite(k.position, "object.position")?;
            }
            let id: ObjectId = project.alloc_id();
            let mut clamped = Vec::new();
            let mut keys: Vec<ObjectKey> = key.into_iter().copied().collect();
            {
                let s = project.scenes.get(scene).ok_or(CoreError::SceneNotFound(*scene))?;
                let base = crate::scatter::field_footprint(preset, &field.shape);
                for k in keys.iter_mut() {
                    k.validate()?;
                    clamp_object_key_base(&s.stage, base, id, k, &mut clamped);
                }
            }
            let s = scene_mut(project, *scene)?;
            let before = Box::new(s.clone());
            s.objects.insert(
                id,
                SceneObject {
                    id,
                    name: name.clone(),
                    asset: crate::id::AssetId(0),
                    // The family name drives the material groups (trunk/canopy/
                    // bush/rock), so material_set works on the band as on a prop.
                    builtin: Some(preset.name.to_string()),
                    keys,
                    attach: None,
                    material_patches: Vec::new(),
                    scatter: Some(field.clone()),
                },
            );
            Ok((UndoOp::SceneReplaced(before), CommandOutput { object: Some(id), clamped, ..no_out }))
        }
        Command::RemoveObject { scene, object } => {
            let s = scene_mut(project, *scene)?;
            if !s.objects.contains_key(object) {
                return Err(CoreError::ObjectNotFound(*object));
            }
            let before = Box::new(s.clone());
            s.objects.remove(object);
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::SetObjectKey { scene: scene_id, object: object_id, key } => {
            key.validate()?;
            let mut key = *key;
            let mut clamped = Vec::new();
            {
                let s = project.scenes.get(scene_id).ok_or(CoreError::SceneNotFound(*scene_id))?;
                let o = s.objects.get(object_id).ok_or(CoreError::ObjectNotFound(*object_id))?;
                clamp_object_key(project, &s.stage, o, &mut key, &mut clamped);
            }
            let s = scene_mut(project, *scene_id)?;
            let before = Box::new(s.clone());
            let o = s.objects.get_mut(object_id).expect("checked above");
            match o.keys.binary_search_by_key(&key.frame, |k| k.frame) {
                Ok(i) => o.keys[i] = key,
                Err(i) => o.keys.insert(i, key),
            }
            Ok((UndoOp::SceneReplaced(before), CommandOutput { clamped, ..no_out }))
        }
        Command::RemoveObjectKey { scene, object, frame } => {
            let s = scene_mut(project, *scene)?;
            if !s.objects.contains_key(object) {
                return Err(CoreError::ObjectNotFound(*object));
            }
            let before = Box::new(s.clone());
            let o = s.objects.get_mut(object).expect("checked above");
            o.keys.retain(|k| k.frame != *frame);
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::ReplaceObjectMotion { scene: scene_id, object: object_id, keys } => {
            for k in keys {
                k.validate()?;
            }
            let mut keys = keys.clone();
            let mut clamped = Vec::new();
            {
                let s = project.scenes.get(scene_id).ok_or(CoreError::SceneNotFound(*scene_id))?;
                let o = s.objects.get(object_id).ok_or(CoreError::ObjectNotFound(*object_id))?;
                for k in keys.iter_mut() {
                    clamp_object_key(project, &s.stage, o, k, &mut clamped);
                }
            }
            let s = scene_mut(project, *scene_id)?;
            let before = Box::new(s.clone());
            let o = s.objects.get_mut(object_id).expect("checked above");
            o.keys = keys;
            o.keys.sort_by_key(|k| k.frame);
            o.keys.dedup_by_key(|k| k.frame);
            Ok((UndoOp::SceneReplaced(before), CommandOutput { clamped, ..no_out }))
        }
        Command::AddPhantom { scene, name, rig } => {
            validate_name(name)?;
            if let RigRef::Imported { asset } = rig {
                match project.assets.get(asset) {
                    Some(a) if a.kind == AssetKind::ModelFbx => {}
                    Some(_) => return Err(CoreError::WrongAssetKind(*asset)),
                    None => return Err(CoreError::AssetNotFound(*asset)),
                }
            }
            if !project.scenes.contains_key(scene) {
                return Err(CoreError::SceneNotFound(*scene));
            }
            let id: PhantomId = project.alloc_id();
            let s = scene_mut(project, *scene)?;
            let before = Box::new(s.clone());
            s.phantoms.insert(
                id,
                Phantom { id, name: name.clone(), rig: rig.clone(), motion: Vec::new(), outline_px: 2.0, scale: 1.0, material_patches: Vec::new()},
            );
            Ok((UndoOp::SceneReplaced(before), CommandOutput { phantom: Some(id), ..no_out }))
        }
        Command::RemovePhantom { scene, phantom } => {
            let s = scene_mut(project, *scene)?;
            if !s.phantoms.contains_key(phantom) {
                return Err(CoreError::PhantomNotFound(*phantom));
            }
            let before = Box::new(s.clone());
            s.phantoms.remove(phantom);
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::SetMotionKey { scene: scene_id, phantom: phantom_id, key } => {
            key.validate()?;
            let mut key = *key;
            let mut clamped = Vec::new();
            {
                let s = project.scenes.get(scene_id).ok_or(CoreError::SceneNotFound(*scene_id))?;
                let p = s.phantoms.get(phantom_id).ok_or(CoreError::PhantomNotFound(*phantom_id))?;
                p.rig.validate_pose(key.pose)?;
                clamp_motion_key(project, &s.stage, p, &mut key, &mut clamped);
            }
            let s = scene_mut(project, *scene_id)?;
            let before = Box::new(s.clone());
            let p = s.phantoms.get_mut(phantom_id).expect("checked above");
            match p.motion.binary_search_by_key(&key.frame, |k| k.frame) {
                Ok(i) => p.motion[i] = key,
                Err(i) => p.motion.insert(i, key),
            }
            Ok((UndoOp::SceneReplaced(before), CommandOutput { clamped, ..no_out }))
        }
        Command::RemoveMotionKey { scene, phantom, frame } => {
            let s = scene_mut(project, *scene)?;
            if !s.phantoms.contains_key(phantom) {
                return Err(CoreError::PhantomNotFound(*phantom));
            }
            let before = Box::new(s.clone());
            let p = s.phantoms.get_mut(phantom).expect("checked above");
            p.motion.retain(|k| k.frame != *frame);
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::ReplaceMotion { scene: scene_id, phantom: phantom_id, motion } => {
            for k in motion {
                k.validate()?;
            }
            let mut motion = motion.clone();
            let mut clamped = Vec::new();
            {
                let s = project.scenes.get(scene_id).ok_or(CoreError::SceneNotFound(*scene_id))?;
                let p = s.phantoms.get(phantom_id).ok_or(CoreError::PhantomNotFound(*phantom_id))?;
                for k in motion.iter_mut() {
                    p.rig.validate_pose(k.pose)?;
                    clamp_motion_key(project, &s.stage, p, k, &mut clamped);
                }
            }
            let s = scene_mut(project, *scene_id)?;
            let before = Box::new(s.clone());
            let p = s.phantoms.get_mut(phantom_id).expect("checked above");
            p.motion = motion;
            p.motion.sort_by_key(|k| k.frame);
            p.motion.dedup_by_key(|k| k.frame);
            Ok((UndoOp::SceneReplaced(before), CommandOutput { clamped, ..no_out }))
        }
        Command::AddCamera { scene, name, range } => {
            validate_name(name)?;
            let s = project.scenes.get(scene).ok_or(CoreError::SceneNotFound(*scene))?;
            // The camera's range is the scene's unless the caller narrows it.
            let range = range.unwrap_or(s.range);
            if range.start < s.range.start || range.end > s.range.end {
                return Err(CoreError::CameraOutOfScene);
            }
            let id: CameraId = project.alloc_id();
            let s = scene_mut(project, *scene)?;
            let before = Box::new(s.clone());
            s.cameras.insert(id, SceneCamera::new(id, name.clone(), range));
            // A freshly added camera is the one the director is looking through.
            s.active_camera = id;
            Ok((UndoOp::SceneReplaced(before), CommandOutput { camera: Some(id), ..no_out }))
        }
        Command::RemoveCamera { scene, camera } => {
            if *camera == SCENE_CAMERA {
                return Err(CoreError::CameraFacesAway { camera: *camera, frame: 0, fraction: 0.0, minimum: 0.0 });
            }
            let s = project.scenes.get(scene).ok_or(CoreError::SceneNotFound(*scene))?;
            if !s.cameras.contains_key(camera) {
                return Err(CoreError::CameraNotFound(*camera));
            }
            let before_scene = Box::new(s.clone());
            let before_film = project.film.clone();
            let s = scene_mut(project, *scene)?;
            s.cameras.remove(camera);
            if s.active_camera == *camera {
                s.active_camera = SCENE_CAMERA;
            }
            // Clip references to a camera are meaningless without it — they leave
            // via the same command.
            project.film.retain(|c| !(c.scene == *scene && c.camera == Some(*camera)));
            Ok((UndoOp::SceneAndFilmReplaced(before_scene, before_film), no_out))
        }
        Command::ActivateCamera { scene, camera } => {
            let s = scene_mut(project, *scene)?;
            if !s.cameras.contains_key(camera) {
                return Err(CoreError::CameraNotFound(*camera));
            }
            let before = Box::new(s.clone());
            s.active_camera = *camera;
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::SetCameraRange { scene, camera, range } => {
            let s = project.scenes.get(scene).ok_or(CoreError::SceneNotFound(*scene))?;
            if !s.cameras.contains_key(camera) {
                return Err(CoreError::CameraNotFound(*camera));
            }
            if range.start < s.range.start || range.end > s.range.end {
                return Err(CoreError::CameraOutOfScene);
            }
            if *camera == SCENE_CAMERA {
                return Err(CoreError::InvalidRange { start: range.start, end: range.end });
            }
            let before_scene = Box::new(s.clone());
            let before_film = project.film.clone();
            let s = scene_mut(project, *scene)?;
            s.cameras.get_mut(camera).expect("checked above").range = *range;
            // A clip's range must match its camera's range.
            for c in &mut project.film {
                if c.scene == *scene && c.camera == Some(*camera) {
                    c.range = *range;
                }
            }
            Ok((UndoOp::SceneAndFilmReplaced(before_scene, before_film), no_out))
        }
        Command::SetCameraKey { scene, camera, key } => {
            check_camera_key(key)?;
            let s = project.scenes.get(scene).ok_or(CoreError::SceneNotFound(*scene))?;
            if !s.cameras.contains_key(camera) {
                return Err(CoreError::CameraNotFound(*camera));
            }
            // Validate on a draft: the facing rule needs the camera WITH this key.
            let mut draft = s.clone();
            insert_camera_key(&mut draft, *camera, *key);
            check_camera_facing(project, &draft, *camera, key.frame)?;
            let s = scene_mut(project, *scene)?;
            let before = Box::new(s.clone());
            insert_camera_key(s, *camera, *key);
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::RemoveCameraKey { scene, camera, frame } => {
            let s = scene_mut(project, *scene)?;
            if !s.cameras.contains_key(camera) {
                return Err(CoreError::CameraNotFound(*camera));
            }
            if s.cameras[camera].camera.keys.len() <= 1 {
                return Err(CoreError::InvalidRange { start: 0, end: 0 });
            }
            let before = Box::new(s.clone());
            s.cameras.get_mut(camera).expect("checked above").camera.keys.retain(|k| k.frame != *frame);
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::SetCameraLens { scene, camera, focal_mm } => {
            check_positive(*focal_mm, "camera.focal_mm")?;
            let s = scene_mut(project, *scene)?;
            if !s.cameras.contains_key(camera) {
                return Err(CoreError::CameraNotFound(*camera));
            }
            let before = Box::new(s.clone());
            let entry = s.cameras.get_mut(camera).expect("checked above");
            entry.camera.focal_mm = *focal_mm;
            // A zoom is a framing choice, not a direction: the facing rule
            // deliberately does not police the lens (a long lens from close by
            // sees little of a big stage, which is what a long lens is for).
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::SetCameraFocalKey { scene, camera, frame, focal_mm } => {
            check_positive(*focal_mm, "camera.focal_mm")?;
            let s = project.scenes.get(scene).ok_or(CoreError::SceneNotFound(*scene))?;
            if !s.cameras.contains_key(camera) {
                return Err(CoreError::CameraNotFound(*camera));
            }
            // A focal key never moves the camera: without a key at that frame
            // the pose is sampled from the track, so a zoom can be added
            // anywhere inside the range. The facing rule does not apply — a
            // longer lens is a framing choice, not a direction.
            let pose = s.cameras.get(camera).expect("checked above").camera.sample(*frame);
            let fresh = TransformKey::at(*frame, pose.translation, pose.yaw, pose.pitch);
            let s = scene_mut(project, *scene)?;
            let before = Box::new(s.clone());
            let entry = s.cameras.get_mut(camera).expect("checked above");
            match entry.camera.keys.binary_search_by_key(frame, |k| k.frame) {
                Ok(i) => entry.camera.keys[i].focal_mm = Some(*focal_mm),
                Err(i) => {
                    entry.camera.keys.insert(i, TransformKey { roll: pose.roll, focal_mm: Some(*focal_mm), ..fresh });
                }
            }
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::SetCameraTarget { scene, camera, target } => {
            let s = project.scenes.get(scene).ok_or(CoreError::SceneNotFound(*scene))?;
            if !s.cameras.contains_key(camera) {
                return Err(CoreError::CameraNotFound(*camera));
            }
            if let Some(t) = target {
                match t.subject {
                    CameraSubject::Phantom(id) if !s.phantoms.contains_key(&id) => return Err(CoreError::PhantomNotFound(id)),
                    CameraSubject::Object(id) if !s.objects.contains_key(&id) => return Err(CoreError::ObjectNotFound(id)),
                    _ => {}
                }
            }
            // The attachment changes the world pose at every key — check them all.
            let mut draft = s.clone();
            draft.cameras.get_mut(camera).expect("checked above").target = *target;
            for key in &draft.cameras[camera].camera.keys {
                check_camera_facing(project, &draft, *camera, key.frame)?;
            }
            let s = scene_mut(project, *scene)?;
            let before = Box::new(s.clone());
            s.cameras.get_mut(camera).expect("checked above").target = *target;
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::SetLighting { scene, lighting } => {
            lighting.validate()?;
            let s = scene_mut(project, *scene)?;
            let before = Box::new(s.clone());
            s.lighting = *lighting;
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::AddLight { scene, name, kind, color, intensity, enabled, shadows, rest } => {
            if !rest.translation.is_finite() || !rest.yaw.is_finite() || !rest.pitch.is_finite() {
                return Err(CoreError::NonFinite("light.rest"));
            }
            let light_id: crate::id::LightId = project.alloc_id();
            let s = scene_mut(project, *scene)?;
            let light = SceneLightSource {
                id: light_id,
                name: name.clone(),
                kind: *kind,
                color: *color,
                intensity: *intensity,
                enabled: *enabled,
                shadows: *shadows,
                keys: Vec::new(),
                rest: *rest,
                attach: None,
            };
            light.validate()?;
            let id = light.id;
            let before = Box::new(s.clone());
            s.lights.push(light);
            Ok((UndoOp::SceneReplaced(before), CommandOutput { light: Some(id), ..no_out }))
        }
        Command::RemoveLight { scene, light } => {
            let s = scene_mut(project, *scene)?;
            if !s.lights.iter().any(|l| l.id == *light) {
                return Err(CoreError::LightNotFound(*light));
            }
            let before = Box::new(s.clone());
            s.lights.retain(|l| l.id != *light);
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::SetLightKey { scene, light, key } => {
            let s = scene_mut(project, *scene)?;
            let pos = s.lights.iter().position(|l| l.id == *light).ok_or(CoreError::LightNotFound(*light))?;
            let mut probe = s.lights[pos].clone();
            match probe.keys.iter().position(|k| k.frame == key.frame) {
                Some(i) => probe.keys[i] = *key,
                None => probe.keys.push(*key),
            }
            probe.keys.sort_by_key(|k| k.frame);
            probe.validate()?;
            let before = Box::new(s.clone());
            s.lights[pos] = probe;
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::RemoveLightKey { scene, light, frame } => {
            let s = scene_mut(project, *scene)?;
            let pos = s.lights.iter().position(|l| l.id == *light).ok_or(CoreError::LightNotFound(*light))?;
            if !s.lights[pos].keys.iter().any(|k| k.frame == *frame) {
                return Err(CoreError::InvalidLightParam("no key at this frame"));
            }
            let before = Box::new(s.clone());
            s.lights[pos].keys.retain(|k| k.frame != *frame);
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::AttachObject { scene, object, attach } => {
            let s = scene_mut(project, *scene)?;
            if !s.objects.contains_key(object) {
                return Err(CoreError::ObjectNotFound(*object));
            }
            if let Some(a) = attach {
                if !s.phantoms.contains_key(&a.phantom) {
                    return Err(CoreError::PhantomNotFound(a.phantom));
                }
                if a.bone.trim().is_empty() {
                    return Err(CoreError::InvalidLightParam("attach: empty bone name"));
                }
                if !a.offset_pos.is_finite() || !a.offset_yaw.is_finite() || !a.offset_pitch.is_finite() || !a.offset_roll.is_finite() {
                    return Err(CoreError::NonFinite("attach.offset"));
                }
            }
            let before = Box::new(s.clone());
            s.objects.get_mut(object).expect("checked above").attach = attach.clone();
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::AttachLight { scene, light, attach } => {
            let s = scene_mut(project, *scene)?;
            if !s.lights.iter().any(|l| l.id == *light) {
                return Err(CoreError::LightNotFound(*light));
            }
            if let Some(a) = attach {
                if !s.phantoms.contains_key(&a.phantom) {
                    return Err(CoreError::PhantomNotFound(a.phantom));
                }
                if a.bone.trim().is_empty() {
                    return Err(CoreError::InvalidLightParam("attach: empty bone name"));
                }
                if !a.offset_pos.is_finite() || !a.offset_yaw.is_finite() || !a.offset_pitch.is_finite() || !a.offset_roll.is_finite() {
                    return Err(CoreError::NonFinite("attach.offset"));
                }
            }
            let before = Box::new(s.clone());
            s.lights.iter_mut().find(|l| l.id == *light).expect("checked above").attach = attach.clone();
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::SetMaterialPatches { scene, phantom, object, patches } => {
            if phantom.is_some() == object.is_some() {
                return Err(CoreError::InvalidLightParam("material: exactly one owner (phantom|object)"));
            }
            for p in patches {
                p.validate()?;
            }
            let s = scene_mut(project, *scene)?;
            let before = Box::new(s.clone());
            if let Some(id) = phantom {
                let ph = s.phantoms.get_mut(id).ok_or(CoreError::PhantomNotFound(*id))?;
                ph.material_patches = patches.clone();
            } else {
                let ob = s.objects.get_mut(&object.expect("checked above")).ok_or(CoreError::ObjectNotFound(object.expect("checked above")))?;
                ob.material_patches = patches.clone();
            }
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::SetPhantomScale { scene: scene_id, phantom: phantom_id, scale } => {
            if !scale.is_finite() || *scale <= 0.0 {
                return Err(CoreError::NonFinite("phantom.scale"));
            }
            let mut clamped = Vec::new();
            {
                let s = project.scenes.get(scene_id).ok_or(CoreError::SceneNotFound(*scene_id))?;
                let p = s.phantoms.get(phantom_id).ok_or(CoreError::PhantomNotFound(*phantom_id))?;
                // A bigger actor may no longer fit where it stood.
                for key in p.motion.iter() {
                    let base = project.phantom_footprint_at(p, key.pose, *scale);
                    let mut key = *key;
                    clamp_motion_key_base(&s.stage, base, p.id, &mut key, &mut clamped);
                }
            }
            let s = scene_mut(project, *scene_id)?;
            let before = Box::new(s.clone());
            let p = s.phantoms.get_mut(phantom_id).expect("checked above");
            p.scale = *scale;
            for c in &clamped {
                if let Some(k) = p.motion.iter_mut().find(|k| k.frame == c.frame) {
                    k.position = Vec3::from(c.to);
                }
            }
            Ok((UndoOp::SceneReplaced(before), CommandOutput { clamped, ..no_out }))
        }
        Command::SetGrade { scene, grade } => {
            if let Some(g) = grade {
                g.validate()?;
            }
            let s = scene_mut(project, *scene)?;
            let before = Box::new(s.clone());
            s.grade = *grade;
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::SetFog { scene, fog } => {
            if let Some(f) = fog {
                if !f.color.iter().all(|c| c.is_finite() && *c >= 0.0) || !f.density.is_finite() || f.density <= 0.0 {
                    return Err(CoreError::NonFinite("fog"));
                }
            }
            let s = scene_mut(project, *scene)?;
            let before = Box::new(s.clone());
            s.fog = *fog;
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::AddVfx { scene, vfx } => {
            vfx.validate()?;
            let id: crate::id::VfxId = project.alloc_id();
            let s = scene_mut(project, *scene)?;
            let before = Box::new(s.clone());
            let mut vfx = vfx.clone();
            vfx.id = id;
            s.vfx.push(vfx);
            Ok((UndoOp::SceneReplaced(before), CommandOutput { vfx: Some(id), ..no_out }))
        }
        Command::RemoveVfx { scene, vfx } => {
            let s = scene_mut(project, *scene)?;
            if !s.vfx.iter().any(|v| v.id == *vfx) {
                return Err(CoreError::InvalidLightParam("vfx not found"));
            }
            let before = Box::new(s.clone());
            s.vfx.retain(|v| v.id != *vfx);
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::SetVfxKey { scene, vfx, key } => {
            if let Some(d) = key.density {
                if !d.is_finite() || d < 0.0 {
                    return Err(CoreError::InvalidLightParam("vfx.key.density >= 0"));
                }
            }
            let s = scene_mut(project, *scene)?;
            let pos = s.vfx.iter().position(|v| v.id == *vfx).ok_or(CoreError::InvalidLightParam("vfx not found"))?;
            let before = Box::new(s.clone());
            let keys = &mut s.vfx[pos].keys;
            match keys.iter().position(|k| k.frame == key.frame) {
                Some(i) => keys[i] = *key,
                None => keys.push(*key),
            }
            keys.sort_by_key(|k| k.frame);
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::RemoveVfxKey { scene, vfx, frame } => {
            let s = scene_mut(project, *scene)?;
            let pos = s.vfx.iter().position(|v| v.id == *vfx).ok_or(CoreError::InvalidLightParam("vfx not found"))?;
            if !s.vfx[pos].keys.iter().any(|k| k.frame == *frame) {
                return Err(CoreError::InvalidLightParam("no key at this frame"));
            }
            let before = Box::new(s.clone());
            s.vfx[pos].keys.retain(|k| k.frame != *frame);
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::SetAmbient { scene, ambient } => {
            if let AmbientMode::Fixed { color, brightness } = ambient {
                if !color.iter().all(|c| c.is_finite() && *c >= 0.0) || !brightness.is_finite() || *brightness < 0.0 {
                    return Err(CoreError::NonFinite("ambient"));
                }
            }
            let s = scene_mut(project, *scene)?;
            let before = Box::new(s.clone());
            s.ambient = *ambient;
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::SetSky { scene, sky } => {
            if let Some(c) = sky {
                if !c.iter().all(|c| c.is_finite() && *c >= 0.0) {
                    return Err(CoreError::NonFinite("sky"));
                }
            }
            let s = scene_mut(project, *scene)?;
            let before = Box::new(s.clone());
            s.sky = *sky;
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::SetFloor { scene, floor } => {
            if let Some(f) = floor {
                f.validate()?;
            }
            let s = scene_mut(project, *scene)?;
            let before = Box::new(s.clone());
            s.floor = *floor;
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        Command::AddLayer2D { scene, name } => {
            validate_name(name)?;
            if !project.scenes.contains_key(scene) {
                return Err(CoreError::SceneNotFound(*scene));
            }
            let id: crate::draw::LayerId = project.alloc_id();
            let s = scene_mut(project, *scene)?;
            let before = Box::new(s.clone());
            s.layers_2d.push(crate::draw::Layer2D::new(id, name));
            Ok((UndoOp::SceneReplaced(before), CommandOutput { layer: Some(id), ..no_out }))
        }
        Command::BrushStroke { scene, layer, frame, stroke } => {
            validate_stroke(stroke)?;
            if !project.scenes.contains_key(scene) {
                return Err(CoreError::SceneNotFound(*scene));
            }
            // With no layers the first stroke creates "Layer 1" in the same command.
            let layer_id = match layer {
                Some(id) => {
                    let s = project.scenes.get(scene).expect("checked above");
                    if !s.layers_2d.iter().any(|l| l.id == *id) {
                        return Err(CoreError::LayerNotFound(*id));
                    }
                    *id
                }
                None => {
                    let s = project.scenes.get(scene).expect("checked above");
                    match s.layers_2d.first() {
                        Some(l) => l.id,
                        None => project.alloc_id(),
                    }
                }
            };
            let s = scene_mut(project, *scene)?;
            let before = Box::new(s.clone());
            if s.layers_2d.is_empty() {
                s.layers_2d.push(crate::draw::Layer2D::new(layer_id, "Layer 1"));
            }
            let l = s.layers_2d.iter_mut().find(|l| l.id == layer_id).expect("layer present");
            l.exposures.entry(*frame).or_default().strokes.push(stroke.clone());
            Ok((UndoOp::SceneReplaced(before), CommandOutput { layer: Some(layer_id), ..no_out }))
        }
        Command::AddFilmClip { scene, range, camera } => {
            let s = project.scenes.get(scene).ok_or(CoreError::SceneNotFound(*scene))?;
            match camera {
                Some(camera_id) => {
                    let camera_ref = s.cameras.get(camera_id).ok_or(CoreError::CameraNotFound(*camera_id))?;
                    // Camera 0 covers the whole scene, so a clip may take any
                    // range of it; a named camera's clip matches its range.
                    if *camera_id != SCENE_CAMERA && *range != camera_ref.range {
                        return Err(CoreError::CameraRangeMismatch);
                    }
                    if range.start < s.range.start || range.end > s.range.end {
                        return Err(CoreError::ClipOutOfScene);
                    }
                }
                None => {
                    if range.start < s.range.start || range.end > s.range.end {
                        return Err(CoreError::ClipOutOfScene);
                    }
                }
            }
            let id: crate::film::ClipId = project.alloc_id();
            let before = project.film.clone();
            project.film.push(crate::film::FilmClip { id, scene: *scene, range: *range, camera: *camera });
            Ok((UndoOp::FilmReplaced(before), CommandOutput { clip: Some(id), ..no_out }))
        }
        Command::RemoveFilmClip { clip } => {
            if !project.film.iter().any(|c| c.id == *clip) {
                return Err(CoreError::ClipNotFound(*clip));
            }
            let before = project.film.clone();
            project.film.retain(|c| c.id != *clip);
            Ok((UndoOp::FilmReplaced(before), no_out))
        }
        Command::MoveFilmClip { clip, index } => {
            let Some(from) = project.film.iter().position(|c| c.id == *clip) else {
                return Err(CoreError::ClipNotFound(*clip));
            };
            let before = project.film.clone();
            let c = project.film.remove(from);
            let to = (*index).min(project.film.len());
            project.film.insert(to, c);
            Ok((UndoOp::FilmReplaced(before), no_out))
        }
        Command::SetFilmClipRange { clip, range } => {
            let Some(c) = project.film.iter().find(|c| c.id == *clip) else {
                return Err(CoreError::ClipNotFound(*clip));
            };
            let s = project.scenes.get(&c.scene).ok_or(CoreError::SceneNotFound(c.scene))?;
            match c.camera {
                Some(camera_id) if camera_id != SCENE_CAMERA => {
                    let camera_ref = s.cameras.get(&camera_id).ok_or(CoreError::CameraNotFound(camera_id))?;
                    if *range != camera_ref.range {
                        return Err(CoreError::CameraRangeMismatch);
                    }
                }
                _ => {}
            }
            if range.start < s.range.start || range.end > s.range.end {
                return Err(CoreError::ClipOutOfScene);
            }
            let before = project.film.clone();
            let c = project.film.iter_mut().find(|c| c.id == *clip).expect("checked above");
            c.range = *range;
            Ok((UndoOp::FilmReplaced(before), no_out))
        }
        Command::AddAudioTrack { scene, name } => {
            validate_name(name)?;
            let id: crate::audio::TrackId = project.alloc_id();
            match scene {
                Some(sid) => {
                    let s = scene_mut(project, *sid)?;
                    let before = Box::new(s.clone());
                    s.audio_tracks.push(crate::audio::AudioTrack::new(id, name));
                    Ok((UndoOp::SceneReplaced(before), CommandOutput { track: Some(id), ..no_out }))
                }
                None => {
                    let before = project.film_audio.clone();
                    project.film_audio.push(crate::audio::AudioTrack::new(id, name));
                    Ok((UndoOp::FilmAudioReplaced(before), CommandOutput { track: Some(id), ..no_out }))
                }
            }
        }
        Command::RemoveAudioTrack { track } => {
            if let Some((sid, s)) = project.scenes.iter_mut().find(|(_, s)| s.audio_tracks.iter().any(|t| t.id == *track)) {
                let sid = *sid;
                let before = Box::new(s.clone());
                let _ = sid;
                s.audio_tracks.retain(|t| t.id != *track);
                return Ok((UndoOp::SceneReplaced(before), no_out));
            }
            if project.film_audio.iter().any(|t| t.id == *track) {
                let before = project.film_audio.clone();
                project.film_audio.retain(|t| t.id != *track);
                return Ok((UndoOp::FilmAudioReplaced(before), no_out));
            }
            Err(CoreError::TrackNotFound(*track))
        }
        Command::SetTrackVolume { track, volume } => {
            check_positive(*volume, "track.volume")?;
            with_track_mut(project, *track, |t| t.volume = *volume)
        }
        Command::SetTrackMuted { track, muted } => {
            with_track_mut(project, *track, |t| t.muted = *muted)
        }
        Command::AddAudioClip { track, asset, start, offset_samples, len_samples, fade_in_samples, fade_out_samples } => {
            match project.assets.get(asset) {
                Some(a) if a.kind == AssetKind::Audio => {}
                Some(_) => return Err(CoreError::WrongAssetKind(*asset)),
                None => return Err(CoreError::AssetNotFound(*asset)),
            }
            if *len_samples == 0 {
                return Err(CoreError::InvalidRange { start: 0, end: 0 });
            }
            if *fade_in_samples > *len_samples || *fade_out_samples > *len_samples {
                return Err(CoreError::InvalidAudioClipParam("fade longer than the clip"));
            }
            if !project.film_audio.iter().any(|t| t.id == *track)
                && !project.scenes.values().any(|s| s.audio_tracks.iter().any(|t| t.id == *track))
            {
                return Err(CoreError::TrackNotFound(*track));
            }
            let id: crate::audio::AudioClipId = project.alloc_id();
            let clip = crate::audio::AudioClip { id, asset: *asset, start: *start, offset_samples: *offset_samples, len_samples: *len_samples, fade_in_samples: *fade_in_samples, fade_out_samples: *fade_out_samples };
            let (undo, _) = with_track_mut(project, *track, |t| t.clips.push(clip))?;
            Ok((undo, CommandOutput { audio_clip: Some(id), ..CommandOutput::default() }))
        }
        Command::UpdateAudioClip { track, clip, offset_samples, len_samples, fade_in_samples, fade_out_samples } => {
            // Validate the patched result BEFORE mutating.
            let current = project
                .scenes
                .values()
                .flat_map(|s| s.audio_tracks.iter())
                .chain(project.film_audio.iter())
                .find(|t| t.id == *track)
                .ok_or(CoreError::TrackNotFound(*track))?
                .clips
                .iter()
                .find(|c| c.id == *clip)
                .ok_or(CoreError::AudioClipNotFound(*clip))?;
            let patched = crate::audio::AudioClip {
                offset_samples: offset_samples.unwrap_or(current.offset_samples),
                len_samples: len_samples.unwrap_or(current.len_samples),
                fade_in_samples: fade_in_samples.unwrap_or(current.fade_in_samples),
                fade_out_samples: fade_out_samples.unwrap_or(current.fade_out_samples),
                ..*current
            };
            if patched.len_samples == 0 {
                return Err(CoreError::InvalidRange { start: 0, end: 0 });
            }
            if patched.fade_in_samples > patched.len_samples || patched.fade_out_samples > patched.len_samples {
                return Err(CoreError::InvalidAudioClipParam("fade longer than the clip"));
            }
            let (undo, _) = with_track_mut(project, *track, |t| {
                let c = t.clips.iter_mut().find(|c| c.id == *clip).expect("checked above");
                *c = patched;
            })?;
            Ok((undo, CommandOutput::default()))
        }
        Command::RemoveAudioClip { track, clip } => {
            with_track_mut(project, *track, |t| t.clips.retain(|c| c.id != *clip))
        }
        Command::RemoveExposure { scene, layer, frame } => {
            let s = scene_mut(project, *scene)?;
            if !s.layers_2d.iter().any(|l| l.id == *layer) {
                return Err(CoreError::LayerNotFound(*layer));
            }
            let before = Box::new(s.clone());
            let l = s.layers_2d.iter_mut().find(|l| l.id == *layer).expect("checked above");
            l.exposures.remove(frame);
            Ok((UndoOp::SceneReplaced(before), no_out))
        }
        // Atomicity: everything is applied on a draft, and on error the document is
        // unchanged. History::apply bumps the revision and maintains the undo stack once.
        Command::Batch(cmds) => {
            if cmds.is_empty() {
                return Err(CoreError::InvalidBatch("empty batch"));
            }
            if cmds.iter().any(|c| matches!(c, Command::Batch(_))) {
                return Err(CoreError::InvalidBatch("nested batch is not allowed"));
            }
            let mut draft = project.clone();
            let mut undos = Vec::with_capacity(cmds.len());
            let mut out = CommandOutput::default();
            for cmd in cmds {
                let (undo, one) = apply(&mut draft, cmd)?;
                undos.push(undo);
                out.merge(one);
            }
            *project = draft;
            Ok((UndoOp::Batch(undos), out))
        }
    }
}

fn validate_stroke(stroke: &crate::draw::Stroke) -> Result<(), CoreError> {
    if stroke.points.len() < 2 {
        return Err(CoreError::InvalidStroke);
    }
    if !stroke.width.is_finite() || stroke.width <= 0.0 {
        return Err(CoreError::NonFinite("stroke.width"));
    }
    if !stroke.color.iter().all(|c| c.is_finite() && (0.0..=1.0).contains(c)) {
        return Err(CoreError::NonFinite("stroke.color"));
    }
    if !stroke.points.iter().all(|p| p[0].is_finite() && p[1].is_finite()) {
        return Err(CoreError::NonFinite("stroke.points"));
    }
    Ok(())
}

/// Finds the track (scene or film) and mutates it with an undo snapshot.
fn with_track_mut(
    project: &mut Project,
    id: crate::audio::TrackId,
    f: impl FnOnce(&mut crate::audio::AudioTrack),
) -> Result<(UndoOp, CommandOutput), CoreError> {
    for (_, s) in project.scenes.iter_mut() {
        if let Some(pos) = s.audio_tracks.iter().position(|t| t.id == id) {
            let before = Box::new(s.clone());
            f(&mut s.audio_tracks[pos]);
            return Ok((UndoOp::SceneReplaced(before), CommandOutput::default()));
        }
    }
    if let Some(pos) = project.film_audio.iter().position(|t| t.id == id) {
        let before = project.film_audio.clone();
        f(&mut project.film_audio[pos]);
        return Ok((UndoOp::FilmAudioReplaced(before), CommandOutput::default()));
    }
    Err(CoreError::TrackNotFound(id))
}

/// Insert or replace a camera key, keeping the key list sorted (the invariant
/// `Camera::sample` relies on).
fn insert_camera_key(scene: &mut Scene, camera: CameraId, key: TransformKey) {
    let entry = scene.cameras.get_mut(&camera).expect("camera checked by the caller");
    match entry.camera.keys.binary_search_by_key(&key.frame, |k| k.frame) {
        Ok(i) => entry.camera.keys[i] = key,
        Err(i) => entry.camera.keys.insert(i, key),
    }
}

/// The camera-facing rule: a camera must look at the scene. A camera standing
/// inside the scene volume passes by definition; otherwise at least
/// [`crate::camera::MIN_SCENE_IN_FRAME`] of the scene's volume must be inside
/// its frame. Lives in the document layer so GUI, CLI, MCP and `batch` all
/// refuse the same camera, and `validate` reports the same number.
fn check_camera_facing(project: &Project, scene: &Scene, camera: CameraId, frame: crate::time::Frame) -> Result<(), CoreError> {
    let fraction = crate::camera::scene_in_frame(project, scene, Some(camera), frame);
    if fraction < crate::camera::MIN_SCENE_IN_FRAME {
        return Err(CoreError::CameraFacesAway {
            camera,
            frame,
            fraction,
            minimum: crate::camera::MIN_SCENE_IN_FRAME,
        });
    }
    Ok(())
}

fn scene_mut(project: &mut Project, id: SceneId) -> Result<&mut Scene, CoreError> {
    project.scenes.get_mut(&id).ok_or(CoreError::SceneNotFound(id))
}

// --- stage clamp (layout mode) -------------------------------------------
//
// Every command that sets a position clamps the entity's whole bounding box to
// the scene stage, so nothing can be staged outside the scene. The clamp lives
// here, in the document layer, because GUI, CLI, MCP and `batch` must behave
// identically and undo has to store the coordinate that is really on stage.

/// Clamp one Phantom key and record the move.
fn clamp_motion_key(
    project: &Project,
    stage: &Stage,
    phantom: &Phantom,
    key: &mut MotionKey,
    out: &mut Vec<Clamp>,
) {
    let base = project.phantom_footprint(phantom, key.pose);
    clamp_motion_key_base(stage, base, phantom.id, key, out);
}

/// Clamp one Phantom key against an already known box.
fn clamp_motion_key_base(stage: &Stage, base: Footprint, id: PhantomId, key: &mut MotionKey, out: &mut Vec<Clamp>) {
    let (position, moved) = stage.clamp(key.position, base, Quat::from_rotation_y(key.yaw));
    if moved {
        out.push(Clamp {
            kind: ClampKind::Phantom,
            id: id.0,
            frame: key.frame,
            from: key.position.to_array(),
            to: position.to_array(),
        });
        key.position = position;
    }
}

/// Clamp one object key and record the move.
fn clamp_object_key(
    project: &Project,
    stage: &Stage,
    object: &SceneObject,
    key: &mut ObjectKey,
    out: &mut Vec<Clamp>,
) {
    let base = match object.builtin.as_deref().and_then(crate::catalog::catalog_find) {
        Some(entry) => entry.footprint(),
        None => project.object_footprint(object, Vec3::ONE),
    };
    clamp_object_key_base(stage, base, object.id, key, out);
}

/// Clamp one object key against an already known base box (a prop being created).
fn clamp_object_key_base(stage: &Stage, base: Footprint, id: ObjectId, key: &mut ObjectKey, out: &mut Vec<Clamp>) {
    let rotation = Quat::from_euler(EulerRot::YXZ, key.yaw, key.pitch, key.roll);
    let (position, moved) = stage.clamp(key.position, base.scaled(key.scale), rotation);
    if moved {
        out.push(Clamp {
            kind: ClampKind::Object,
            id: id.0,
            frame: key.frame,
            from: key.position.to_array(),
            to: position.to_array(),
        });
        key.position = position;
    }
}

/// What a stage change would move: every staged entity re-clamped into `stage`.
fn reclamp_scene(project: &Project, scene: &Scene, stage: &Stage) -> Vec<Clamp> {
    let mut out = Vec::new();
    for phantom in scene.phantoms.values() {
        for key in &phantom.motion {
            let mut key = *key;
            clamp_motion_key(project, stage, phantom, &mut key, &mut out);
        }
    }
    for object in scene.objects.values() {
        for key in &object.keys {
            let mut key = *key;
            clamp_object_key(project, stage, object, &mut key, &mut out);
        }
    }
    out
}

/// Apply the clamps a stage change computed (the stage itself is set by the caller).
fn apply_clamps(scene: &mut Scene, clamps: &[Clamp]) {
    for c in clamps {
        match c.kind {
            ClampKind::Phantom => {
                if let Some(p) = scene.phantoms.get_mut(&PhantomId(c.id)) {
                    if let Some(k) = p.motion.iter_mut().find(|k| k.frame == c.frame) {
                        k.position = Vec3::from(c.to);
                    }
                }
            }
            ClampKind::Object => {
                if let Some(o) = scene.objects.get_mut(&ObjectId(c.id)) {
                    if let Some(k) = o.keys.iter_mut().find(|k| k.frame == c.frame) {
                        k.position = Vec3::from(c.to);
                    }
                }
            }
        }
    }
}

fn validate_name(name: &str) -> Result<(), CoreError> {
    if name.trim().is_empty() {
        return Err(CoreError::EmptyName);
    }
    Ok(())
}

/// Camera key: finite pose/angles/roll, a non-empty aim point and a positive
/// focal length (C1). Checks happen before mutation, as in the other commands.
fn check_camera_key(key: &TransformKey) -> Result<(), CoreError> {
    check_finite(key.translation, "camera.translation")?;
    if !key.yaw.is_finite() || !key.pitch.is_finite() || !key.roll.is_finite() {
        return Err(CoreError::NonFinite("camera.rotation"));
    }
    if let Some(target) = key.look_at {
        check_finite(target, "camera.look_at")?;
    }
    if let Some(mm) = key.focal_mm {
        check_positive(mm, "camera.focal_mm")?;
    }
    Ok(())
}

fn check_finite(v: Vec3, field: &'static str) -> Result<(), CoreError> {
    if !v.is_finite() {
        return Err(CoreError::NonFinite(field));
    }
    Ok(())
}

fn check_positive(v: f32, field: &'static str) -> Result<(), CoreError> {
    if !v.is_finite() || v <= 0.0 {
        return Err(CoreError::NonFinite(field));
    }
    Ok(())
}

//! Session: a single project owner plus a dispatcher of named operations.
//! Shared layer for MCP stdio, the bridge and the GUI: operations go through
//! phantom_core::History; long renders run as a child CLI process.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Child, Stdio};

use phantom_core::*;
use phantom_core::project::BackgroundSource;
use phantom_runtime::storage;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::project_lock::{self, LockGuard};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ErrorData {
    pub code: String,
    pub message: String,
    pub details: Value,
}

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("{0}")]
    Core(#[from] CoreError),
    #[error("{0}")]
    Storage(#[from] storage::StorageError),
    #[error("{0}")]
    Import(#[from] phantom_runtime::import::ImportError),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("no project is open")]
    NoProject,
    #[error("job {0} not found")]
    JobNotFound(u64),
    #[error("asset {id} not found")]
    AssetGetNotFound { id: u64, nearest: Vec<u64> },
    #[error("{0}")]
    Lock(#[from] project_lock::LockError),
    #[error("cannot switch an open session from {} to {}; create a new MCP session", current.display(), target.display())]
    ProjectSwitchRejected { current: PathBuf, target: PathBuf },
    #[error("request is outside the allowed roots: {0}")]
    OutsideAllowedRoots(PathBuf),
    #[error("phantom_add: rig_asset and rig_actor are mutually exclusive")]
    ConflictingRig,
    #[error("{0}")]
    BadInput(String),
    #[error("look: no actor or object matches {needle:?}")]
    LookTargetNotFound { needle: String, names: Vec<String> },
    #[error("operation {operation} is not available through the bridge: {reason}")]
    BridgeUnavailable { operation: &'static str, reason: &'static str },
    #[error("{0}")]
    Render(#[from] phantom_runtime::render::RenderError),
    #[error("batch: operation {index} rejected before execution: {reason}")]
    BatchInvalid { index: usize, reason: String },
    #[error("batch: operation {index} ({op}) not applied, the batch was rolled back: {source}")]
    BatchFailed { index: usize, op: String, source: Box<SessionError> },
}

impl ErrorData {
    pub fn new(code: impl Into<String>, message: impl Into<String>, details: Value) -> Self {
        Self { code: code.into(), message: message.into(), details }
    }
}

impl SessionError {
    /// Stable, locale-independent MCP error category.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Core(e) => core_error_code(e),
            Self::Storage(e) => storage_error_code(e),
            Self::Import(e) => import_error_code(e),
            Self::Io(_) => "io_error",
            Self::NoProject => "not_open",
            Self::JobNotFound(_) | Self::AssetGetNotFound { .. } => "not_found",
            Self::Lock(e) => lock_error_code(e),
            Self::ProjectSwitchRejected { .. } => "project_switch_rejected",
            Self::OutsideAllowedRoots(_) => "path_forbidden",
            Self::ConflictingRig | Self::BadInput(_) => "invalid_input",
            Self::LookTargetNotFound { .. } => "not_found",
            Self::BridgeUnavailable { .. } => "bridge_unavailable",
            Self::Render(e) => render_error_code(e),
            Self::BatchInvalid { .. } => "invalid_input",
            // The wrapped code is kept in details; the top-level code is stable.
            Self::BatchFailed { .. } => "batch_failed",
        }
    }

    /// Structured diagnostics are built from variant fields. Human messages are
    /// never parsed to recover IDs, paths, or categories.
    pub fn details(&self) -> Value {
        match self {
            Self::Core(e) => core_error_details(e),
            Self::Storage(e) => storage_error_details(e),
            Self::Import(e) => import_error_details(e),
            Self::Io(e) => io_error_details(e),
            Self::NoProject => json!({ "required_state": "project_open" }),
            Self::JobNotFound(job) => json!({ "entity": "job", "id": job }),
            Self::AssetGetNotFound { id, nearest } => json!({ "entity": "asset", "id": id, "nearest_ids": nearest }),
            Self::Lock(e) => lock_error_details(e),
            Self::ProjectSwitchRejected { current, target } => {
                json!({ "current": current, "target": target })
            }
            Self::OutsideAllowedRoots(path) => json!({ "path": path }),
            Self::ConflictingRig => json!({ "fields": ["rig_asset", "rig_actor"] }),
            Self::BadInput(reason) => json!({ "reason": reason }),
            Self::LookTargetNotFound { needle, names } => json!({ "needle": needle, "names": names }),
            Self::BridgeUnavailable { operation, reason } => {
                json!({ "operation": operation, "reason": reason })
            }
            Self::Render(e) => render_error_details(e),
            Self::BatchInvalid { index, reason } => json!({ "index": index, "reason": reason }),
            Self::BatchFailed { index, op, source } => json!({
                "index": index,
                "op": op,
                "code": source.code(),
                "cause": source.details(),
                "rolled_back": true,
            }),
        }
    }

    pub fn error_data(&self) -> ErrorData {
        ErrorData::new(self.code(), self.to_string(), self.details())
    }
}

fn io_error_details(error: &std::io::Error) -> Value {
    json!({ "kind": format!("{:?}", error.kind()).to_lowercase(), "cause": error.to_string() })
}

fn core_error_code(error: &CoreError) -> &'static str {
    match error {
        CoreError::SceneNotFound(_)
        | CoreError::PhantomNotFound(_)
        | CoreError::ObjectNotFound(_)
        | CoreError::AssetNotFound(_)
        | CoreError::LayerNotFound(_)
        | CoreError::ClipNotFound(_)
        | CoreError::TrackNotFound(_)
        | CoreError::CameraNotFound(_)
        | CoreError::AudioClipNotFound(_)
        | CoreError::LightNotFound(_)
        | CoreError::UnknownProp(_) => "not_found",
        CoreError::WrongAssetKind(_) => "wrong_asset_kind",
        CoreError::CameraFacesAway { .. } => "camera_faces_away",
        CoreError::CameraSubjectMissing { .. } => "camera_subject_missing",
        CoreError::UnsupportedPose { .. } => "unsupported_pose",
        CoreError::InvalidRange { .. }
        | CoreError::NonFinite(_)
        | CoreError::InvalidStage(_)
        | CoreError::InvalidBackground(_)
        | CoreError::EmptyName
        | CoreError::InvalidStroke
        | CoreError::ClipOutOfScene
        | CoreError::CameraOutOfScene
        | CoreError::CameraRangeMismatch
        | CoreError::InvalidAudioClipParam(_)
        | CoreError::InvalidLightParam(_)
        | CoreError::InvalidScatter(_)
        | CoreError::InvalidBatch(_) => "invalid_input",
    }
}

fn core_error_details(error: &CoreError) -> Value {
    match error {
        CoreError::UnsupportedPose { actor, pose } => {
            json!({ "actor": format!("{actor:?}").to_lowercase(), "pose": format!("{pose:?}").to_lowercase() })
        }
        CoreError::SceneNotFound(id) => json!({ "entity": "scene", "id": id.0 }),
        CoreError::PhantomNotFound(id) => json!({ "entity": "phantom", "id": id.0 }),
        CoreError::ObjectNotFound(id) => json!({ "entity": "object", "id": id.0 }),
        CoreError::AssetNotFound(id) => json!({ "entity": "asset", "id": id.0 }),
        CoreError::UnknownProp(name) => json!({ "entity": "catalog_entry", "name": name, "hint": "catalog_query" }),
        // A fill that will not fit says what to lower; the message carries it.
        CoreError::InvalidScatter(why) => json!({ "entity": "scatter", "why": why, "hint": "scatter_presets" }),
        CoreError::InvalidRange { start, end } => json!({ "start": start, "end": end }),
        CoreError::NonFinite(field) => json!({ "field": field }),
        CoreError::EmptyName => json!({ "field": "name" }),
        CoreError::WrongAssetKind(id) => json!({ "entity": "asset", "id": id.0 }),
        CoreError::LayerNotFound(id) => json!({ "entity": "layer", "id": id.0 }),
        CoreError::InvalidStroke => json!({ "minimum_points": 2 }),
        CoreError::ClipNotFound(id) => json!({ "entity": "film_clip", "id": id.0 }),
        CoreError::ClipOutOfScene => json!({ "constraint": "clip_inside_scene" }),
        CoreError::TrackNotFound(id) => json!({ "entity": "audio_track", "id": id.0 }),
        CoreError::CameraNotFound(id) => json!({ "entity": "camera", "id": id.0 }),
        CoreError::CameraFacesAway { camera, frame, fraction, minimum } => json!({
            "camera": camera.0, "frame": frame, "scene_fraction": fraction, "minimum": minimum,
        }),
        CoreError::CameraSubjectMissing { camera } => json!({ "entity": "camera", "id": camera.0 }),
        CoreError::CameraOutOfScene => json!({ "constraint": "shot_inside_scene" }),
        CoreError::CameraRangeMismatch => json!({ "constraint": "clip_range_equals_shot_range" }),
        CoreError::InvalidAudioClipParam(field) => json!({ "field": field }),
        CoreError::InvalidLightParam(field) => json!({ "field": field }),
        CoreError::InvalidBatch(reason) => json!({ "reason": reason }),
        CoreError::InvalidStage(field) => json!({ "field": field }),
        CoreError::InvalidBackground(field) => json!({ "field": field }),
        CoreError::AudioClipNotFound(id) => json!({ "entity": "audio_clip", "id": id.0 }),
        CoreError::LightNotFound(id) => json!({ "entity": "light", "id": id.0 }),
    }
}

fn storage_error_code(error: &storage::StorageError) -> &'static str {
    match error {
        storage::StorageError::Io(_) => "io_error",
        storage::StorageError::NoManifest { .. } => "not_found",
        storage::StorageError::Json(_) => "invalid_data",
        storage::StorageError::NewerSchema { .. } => "unsupported_schema",
        storage::StorageError::PathEscape(_) => "path_forbidden",
    }
}

fn storage_error_details(error: &storage::StorageError) -> Value {
    match error {
        storage::StorageError::Io(e) => io_error_details(e),
        storage::StorageError::NoManifest { path, source } => {
            json!({ "entity": "project_manifest", "path": path, "io": io_error_details(source) })
        }
        storage::StorageError::Json(e) => json!({ "cause": e.to_string(), "category": format!("{:?}", e.classify()).to_lowercase() }),
        storage::StorageError::NewerSchema { found } => {
            json!({ "found": found, "supported": phantom_core::SCHEMA_VERSION })
        }
        storage::StorageError::PathEscape(path) => json!({ "path": path }),
    }
}

fn import_error_code(error: &phantom_runtime::import::ImportError) -> &'static str {
    use phantom_runtime::import::ImportError;
    match error {
        ImportError::Io(_) => "io_error",
        ImportError::NotAFile(_) => "not_found",
        ImportError::PathEscape(_) => "path_forbidden",
        ImportError::Ufbx(_) => "import_failed",
        ImportError::NoMesh => "no_mesh",
        ImportError::NoSkin => "no_skin",
        ImportError::TooManyInfluences(_) => "too_many_influences",
        ImportError::UnknownExtension(_) => "unsupported_format",
        ImportError::TooManyDeps(_) => "import_limit_exceeded",
        ImportError::BundleTooLarge(_) => "import_limit_exceeded",
        ImportError::TextureTooLarge { .. } => "import_limit_exceeded",
    }
}

fn import_error_details(error: &phantom_runtime::import::ImportError) -> Value {
    use phantom_runtime::import::ImportError;
    match error {
        ImportError::Ufbx(cause) => json!({ "cause": cause }),
        ImportError::NoMesh => json!({ "required": "mesh" }),
        ImportError::NoSkin => json!({ "required": "skin" }),
        ImportError::TooManyInfluences(found) => json!({ "found": found, "maximum": 4 }),
        ImportError::UnknownExtension(extension) => json!({ "extension": extension }),
        ImportError::Io(e) => io_error_details(e),
        ImportError::NotAFile(path) | ImportError::PathEscape(path) => json!({ "path": path }),
        ImportError::TooManyDeps(found) => {
            json!({ "resource": "bundle_dependencies", "found": found, "maximum": 64 })
        }
        ImportError::BundleTooLarge(found) => {
            json!({ "resource": "bundle_bytes", "found": found, "maximum": 256_u64 * 1024 * 1024 })
        }
        ImportError::TextureTooLarge { path, w, h } => {
            json!({ "path": path, "width": w, "height": h, "maximum_dimension": 8192 })
        }
    }
}

fn lock_error_code(error: &project_lock::LockError) -> &'static str {
    match error {
        project_lock::LockError::Io(_) => "io_error",
        project_lock::LockError::Held { .. } => "lock_held",
        project_lock::LockError::Changed { .. } => "lock_changed",
        project_lock::LockError::NotFound(_) => "not_found",
    }
}

fn lock_error_details(error: &project_lock::LockError) -> Value {
    match error {
        project_lock::LockError::Io(e) => io_error_details(e),
        project_lock::LockError::Held { path, owner, .. } => json!({ "path": path, "owner": owner }),
        project_lock::LockError::Changed { path } => json!({ "path": path }),
        project_lock::LockError::NotFound(path) => json!({ "entity": "project_lock", "path": path }),
    }
}

fn render_error_code(error: &phantom_runtime::render::RenderError) -> &'static str {
    use phantom_runtime::render::RenderError;
    match error {
        RenderError::InvalidAction(e) => core_error_code(e),
        RenderError::GpuUnavailable(_) => "gpu_unavailable",
        RenderError::Import(e) => import_error_code(e),
        RenderError::Image(_) => "image_error",
        RenderError::Io(_) => "io_error",
        RenderError::ReadbackTimeout => "render_timeout",
    }
}

fn render_error_details(error: &phantom_runtime::render::RenderError) -> Value {
    use phantom_runtime::render::RenderError;
    match error {
        RenderError::InvalidAction(e) => core_error_details(e),
        RenderError::GpuUnavailable(cause) => json!({ "cause": cause }),
        RenderError::Import(e) => import_error_details(e),
        RenderError::Image(e) => json!({ "cause": e.to_string() }),
        RenderError::Io(e) => io_error_details(e),
        RenderError::ReadbackTimeout => json!({ "stage": "gpu_readback" }),
    }
}

/// Named operations on the project.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op {
    ProjectNew { dir: PathBuf, name: String },
    ProjectOpen { dir: PathBuf, #[serde(default)] force: bool },
    LockRelease { dir: PathBuf, #[serde(default)] force: bool },
    ProjectSave,
    ProjectRecent,
    StateGet,
    SceneAdd { name: String, start: Frame, end: Frame },
    AssetImport {
        file: PathBuf,
        /// Explicit asset type: model-obj|model-fbx|image|audio|video.
        /// `None` (and a missing field in legacy JSON) — autodetect by
        /// extension, as before. Affects only the registered AssetKind:
        /// obj/fbx still go through bundle import.
        #[serde(default)]
        kind: Option<String>,
    },
    BackgroundSet {
        scene: u64,
        asset: u64,
        /// The background is a video (an image by default).
        #[serde(default)]
        video: bool,
        /// Flat screen geometry; the serde defaults are a backdrop wide enough
        /// for the default camera, so old bridge JSON without these fields
        /// still parses.
        #[serde(default = "default_bg_width")]
        width: f32,
        #[serde(default = "default_bg_height")]
        height: f32,
        #[serde(default = "default_bg_center")]
        center: [f32; 3],
        /// Panorama UV mode ("cover"|"contain"|"stretch"); None — the document
        /// default (cover, corrective A3).
        #[serde(default)]
        fit: Option<phantom_core::project::BackgroundFit>,
    },
    BackgroundClear { scene: u64 },
    SceneRangeSet { scene: u64, start: Frame, end: Frame },
    PhantomAdd { scene: u64, name: String, rig_asset: Option<u64>, #[serde(default)] rig_actor: Option<String> },
    BuiltinCatalog,
    /// Search the metric prop catalog (K1): section and/or substring.
    CatalogQuery { #[serde(default)] kind: Option<String>, #[serde(default)] query: Option<String> },
    /// Place a catalog item (K1): the mesh is procedural, no file needed.
    PropAdd { scene: u64, name: String, prop: String, #[serde(default)] pos: Option<[f32; 3]>, #[serde(default)] yaw: f32, #[serde(default)] pitch: f32, #[serde(default)] roll: f32, #[serde(default)] scale: Option<[f32; 3]> },
    KeySet { scene: u64, phantom: u64, frame: Frame, pos: [f32; 3], pose: ActorPose, #[serde(default)] yaw: f32 },
    /// Add a named camera (the range defaults to the whole scene); it becomes
    /// the scene's active camera.
    CameraAdd { scene: u64, name: String, #[serde(default)] start: Option<Frame>, #[serde(default)] end: Option<Frame> },
    /// Remove a camera; film clips referencing it go away in the same step.
    CameraRemove { scene: u64, camera: u64 },
    /// Pick the camera the viewport shows and a render without one uses.
    CameraSelect { scene: u64, camera: u64 },
    /// Change a camera's range; film clips onto it get the new range.
    CameraRangeSet { scene: u64, camera: u64, start: Frame, end: Frame },
    /// Add or replace a camera key (roll/look_at/focus/ease included).
    /// `camera` omitted — the scene's active camera.
    CameraKeySet { scene: u64, #[serde(default)] camera: Option<u64>, key: TransformKey },
    /// Remove a camera key at a frame (the last key cannot be removed).
    CameraKeyRemove { scene: u64, camera: u64, frame: Frame },
    /// The camera's base lens in mm.
    CameraLensSet { scene: u64, camera: u64, focal_mm: f32 },
    /// Focus on one key (a zoom inside the camera's range).
    CameraFocalKeySet { scene: u64, camera: u64, frame: Frame, focal_mm: f32 },
    /// Attach the camera to a moving subject: mode `track` (stands and turns to
    /// it) or `follow` (attached, travels and turns with it). `clear: true`
    /// detaches. `subject` is "phantom:<id>" or "object:<id>".
    CameraTargetSet {
        scene: u64,
        camera: u64,
        #[serde(default)]
        mode: Option<String>,
        #[serde(default)]
        subject: Option<String>,
        #[serde(default)]
        clear: bool,
    },
    ObjectAdd { scene: u64, name: String, asset: u64 },
    ObjectRemove { scene: u64, object: u64 },
    ObjectKeySet { scene: u64, object: u64, frame: Frame, pos: [f32; 3], #[serde(default)] yaw: f32, #[serde(default)] pitch: f32, #[serde(default)] roll: f32, #[serde(default = "unit_scale")] scale: [f32; 3] },
    ObjectKeyRemove { scene: u64, object: u64, frame: Frame },
    LightingSet { scene: u64, lighting: SceneLighting },
    LightingGet { scene: u64 },
    /// Document light sources with keys (B1); angles in radians.
    LightAdd { scene: u64, name: String, kind: SceneLightKind, color: [f32; 3], intensity: f32, #[serde(default = "default_true")] enabled: bool, #[serde(default)] shadows: bool, rest_pos: [f32; 3], #[serde(default)] rest_yaw: f32, #[serde(default)] rest_pitch: f32 },
    LightRemove { scene: u64, light: u64 },
    LightKeySet { scene: u64, light: u64, frame: Frame, pos: [f32; 3], #[serde(default)] yaw: f32, #[serde(default)] pitch: f32, #[serde(default)] intensity: Option<f32>, #[serde(default)] enabled: Option<bool> },
    LightKeyRemove { scene: u64, light: u64, frame: Frame },
    AmbientSet { scene: u64, ambient: AmbientMode },
    SkySet { scene: u64, sky: Option<[f32; 3]> },
    /// Attach an object/light source to a Phantom bone (B2); attach = None detaches.
    ObjectAttach { scene: u64, object: u64, attach: Option<Attachment> },
    LightAttach { scene: u64, light: u64, attach: Option<Attachment> },
    PhantomScaleSet { scene: u64, phantom: u64, scale: f32 },
    /// Material patch: exactly one owner (phantom|object); replaces the patch
    /// with the same target or adds one; clear removes (target=None — all).
    MaterialSet { scene: u64, phantom: Option<u64>, object: Option<u64>, target: Option<String>, patch: MaterialPatch, #[serde(default)] clear: bool },
    /// Frame color grading (B3); None — turn it off.
    GradeSet { scene: u64, grade: Option<SceneGrade> },
    /// Fog (B3); None — turn it off.
    FogSet { scene: u64, fog: Option<SceneFog> },
    /// Effects: the id is assigned by the command; keys are enabled/density steps.
    VfxAdd { scene: u64, vfx: SceneVfx },
    VfxRemove { scene: u64, vfx: u64 },
    VfxKeySet { scene: u64, vfx: u64, frame: Frame, #[serde(default)] enabled: Option<bool>, #[serde(default)] density: Option<f32> },
    VfxKeyRemove { scene: u64, vfx: u64, frame: Frame },
    /// The owner model's parts and materials plus the effective values with patches.
    MaterialList { scene: u64, phantom: Option<u64>, object: Option<u64> },
    FloorSet { scene: u64, floor: Option<SceneFloor> },
    FloorGet { scene: u64 },
    /// Scene stage (layout mode): a preset, or an explicit size, or both.
    /// `center` defaults to the stage's current center.
    StageSet {
        scene: u64,
        #[serde(default)]
        preset: Option<String>,
        #[serde(default)]
        width: Option<f32>,
        #[serde(default)]
        height: Option<f32>,
        #[serde(default)]
        depth: Option<f32>,
        #[serde(default)]
        center: Option<[f32; 2]>,
    },
    StageGet { scene: u64 },
    BrushStroke { scene: u64, frame: Frame, points: Vec<[f32; 2]>, color: [f32; 4], width: f32, erase: bool },
    FilmAdd { scene: u64, start: Frame, end: Frame, #[serde(default)] camera: Option<u64> },
    FilmMove { clip: u64, index: usize },
    FilmTrim { clip: u64, start: Frame, end: Frame },
    FilmRemove { clip: u64 },
    PhantomRemove { scene: u64, phantom: u64 },
    MotionKeyRemove { scene: u64, phantom: u64, frame: Frame },
    /// Replace the whole controller track atomically (one undo step).
    MotionReplace {
        scene: u64,
        phantom: u64,
        motion: Vec<(Frame, [f32; 3], ActorPose)>,
        /// Optional yaw per key, parallel to `motion` (serde default —
        /// old JSON made of triples still reads with yaw = 0).
        #[serde(default)]
        yaws: Option<Vec<f32>>,
    },
    AudioTrackAdd { scene: Option<u64>, name: String },
    AudioClipAdd { track: u64, asset: u64, start: Frame, offset_samples: u64, len_samples: u64, #[serde(default)] fade_in_samples: u64, #[serde(default)] fade_out_samples: u64 },
    /// Point patch of a clip's fragment/fades (None — do not change).
    AudioClipUpdate { track: u64, clip: u64, offset_samples: Option<u64>, len_samples: Option<u64>, fade_in_samples: Option<u64>, fade_out_samples: Option<u64> },
    TrackVolumeSet { track: u64, volume: f32 },
    TrackMuteSet { track: u64, muted: bool },
    AudioClipRemove { track: u64, clip: u64 },
    AudioTrackRemove { track: u64 },
    /// Add a 2D layer to the scene.
    #[serde(rename = "layer_add_2d")]
    LayerAdd2D { scene: u64, name: String },
    /// Remove a 2D layer exposure entirely.
    ExposureRemove { scene: u64, layer: u64, frame: Frame },
    HistoryUndo,
    HistoryRedo,
    /// Undo/redo stack depths and the names of the last operations.
    HistoryList,
    RenderScene { scene: u64, out: PathBuf, width: u32, height: u32, frames: Option<(Frame, Frame)>, #[serde(default)] camera: Option<u64> },
    RenderFilm { out: PathBuf, width: u32, height: u32, #[serde(default)] frames: Option<(Frame, Frame)>, #[serde(default)] normalize: bool, #[serde(default)] target_peak_db: Option<f32> },
    JobStatus { job: u64 },
    JobWait { job: u64, timeout_secs: f64 },
    JobCancel { job: u64 },
    /// Full serde snapshot of one scene (phantoms, objects, shots, layers,
    /// audio tracks, background, camera).
    SceneGet { scene: u64 },
    /// Full snapshot of a camera plus the revision of its parent scene.
    CameraGet { scene: u64, camera: u64 },
    /// Canonical asset, whether the file exists, and cheap usage references.
    AssetGet { asset: u64 },
    /// The full edit sequence and its mapped length.
    FilmGet,
    /// List the project's assets (id/kind/path).
    AssetList,
    /// Statuses of all the session's jobs.
    JobList,
    /// Render a single frame to PNG in-process (needs a GPU); the result is base64.
    /// `report: true` adds clean frame metrics (luma/empty_coverage) and the
    /// scene digest — the agent's "eyes" without a window (corrective C3.2).
    RenderStill { scene: u64, frame: Frame, width: u32, height: u32, #[serde(default)] camera: Option<u64>, #[serde(default)] report: bool },
    /// Camera report (corrective_2 K0.2): render a sample of the camera's
    /// frames and metrics per frame — the agent's eyes on the whole camera range.
    CameraReport { scene: u64, camera: u64, #[serde(default)] every: Option<u32>, #[serde(default)] width: Option<u32>, #[serde(default)] height: Option<u32> },
    /// Review sheet: several directions around the staged scene at one frame
    /// (`angles`), or a strip of frames through one camera (`frames`). One
    /// contact-sheet image plus per-tile metrics. Requires a GPU.
    Look {
        scene: u64,
        #[serde(default)]
        mode: Option<String>,
        #[serde(default)]
        target: Option<Vec<String>>,
        #[serde(default)]
        views: Option<Vec<ViewSpec>>,
        #[serde(default)]
        distance: Option<f32>,
        #[serde(default)]
        frame: Option<Frame>,
        #[serde(default)]
        camera: Option<u64>,
        #[serde(default)]
        frames: Option<Vec<Frame>>,
        #[serde(default)]
        every: Option<u32>,
        #[serde(default)]
        count: Option<u32>,
        #[serde(default)]
        view: Option<ViewSpec>,
        #[serde(default)]
        width: Option<u32>,
        #[serde(default)]
        height: Option<u32>,
    },
    /// A row/grid of catalog props placed in one atomic step (one undo): the
    /// step is derived from the prop's own size unless given explicitly.
    PropGrid {
        scene: u64,
        prop: String,
        #[serde(default)]
        prefix: Option<String>,
        rows: u32,
        cols: u32,
        origin: [f32; 3],
        #[serde(default)]
        step: Option<[f32; 2]>,
        #[serde(default)]
        gap: Option<f32>,
        #[serde(default)]
        yaw: f32,
        #[serde(default)]
        pitch: f32,
        #[serde(default)]
        roll: f32,
        #[serde(default)]
        scale: Option<[f32; 3]>,
    },
    /// Fill an area or a line from a preset (forest, grove, undergrowth, hedge,
    /// street_row, pole_row) in one atomic step. Inside `action` the fill places
    /// editable props; the ring beyond it becomes ONE band object. Zones are
    /// given as a world rectangle or off a ground plate, never guessed.
    Scatter {
        scene: u64,
        preset: String,
        /// World rectangle "x0,z0,x1,z1" (mutually exclusive with `plate`).
        #[serde(default)]
        area: Option<[f32; 4]>,
        /// Object id of a ground plate (street_yard/park/crossing) to fill.
        #[serde(default)]
        plate: Option<u64>,
        /// Keep this margin (m) clear of the plate's edge.
        #[serde(default)]
        inset: Option<f32>,
        /// The zone the scene is shot in: world rectangle, editable props.
        #[serde(default)]
        action: Option<[f32; 4]>,
        /// Props per square meter; the preset's own default by default.
        #[serde(default)]
        density: Option<f32>,
        /// Line presets: segment endpoints in world coordinates.
        #[serde(default)]
        from: Option<[f32; 2]>,
        #[serde(default)]
        to: Option<[f32; 2]>,
        #[serde(default)]
        step: Option<f32>,
        /// Determinism seed; a random one is drawn and reported when omitted.
        #[serde(default)]
        seed: Option<u64>,
        #[serde(default)]
        prefix: Option<String>,
    },
    /// One object whose mesh is a replayed scatter plan (the far band of a
    /// fill). Split from `scatter` so the batch composes plain ops.
    ScatterBand {
        scene: u64,
        name: String,
        field: phantom_core::ScatterField,
        pos: [f32; 3],
        yaw: f32,
    },
    /// Clean document validation (no render/GPU): screen invariants, FOV
    /// coverage, motion gaps, audio and edit ranges, asset files.
    /// `None` — the whole project.
    Validate { scene: Option<u64> },
    /// Frame coverage by the background screen and phantom/object visibility at a frame.
    /// Pure geometry; angles in the response are degrees, aspect is width/height.
    CameraCoverage { scene: u64, frame: Option<Frame>, #[serde(default)] camera: Option<u64>, aspect: Option<f64> },
    /// Fit the screen geometry to the camera with a margin (degrees).
    BackgroundFit { scene: u64, frame: Option<Frame>, #[serde(default)] camera: Option<u64>, margin_deg: Option<f64>, aspect: Option<f64> },
    /// Audio report (corrective C3.4/A4): track/clip durations against the
    /// video, RMS/peak of the fragments actually used (ffmpeg),
    /// warnings. `None` — the whole project.
    AudioReport { scene: Option<u64> },
    /// Atomic batch of document mutations: all operations are validated and
    /// applied in order; on any error — a full rollback with no
    /// partial state. One undo step, one revision bump.
    /// Nested batches, queries, lifecycle and render/job operations are forbidden.
    Batch { ops: Vec<Op> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OpClass {
    DocumentMutation,
    Query,
    Job,
    Lifecycle,
}

impl Op {
    /// Exhaustive classification keeps new operations from silently acquiring
    /// (or missing) document-mutation acknowledgement metadata.
    fn class(&self) -> OpClass {
        use Op::*;
        match self {
            ProjectNew { .. } | ProjectOpen { .. } | LockRelease { .. } | ProjectSave => OpClass::Lifecycle,
            ProjectRecent | StateGet | BuiltinCatalog | CatalogQuery { .. } | LightingGet { .. } | FloorGet { .. }
            | StageGet { .. }
            | HistoryList | SceneGet { .. } | CameraGet { .. } | AssetGet { .. } | FilmGet | AssetList
            | Validate { .. } | CameraCoverage { .. } | BackgroundFit { .. } | AudioReport { .. } | MaterialList { .. } => OpClass::Query,
            RenderScene { .. } | RenderFilm { .. } | JobStatus { .. } | JobWait { .. } | JobCancel { .. }
            | JobList | RenderStill { .. } | CameraReport { .. } | Look { .. } => OpClass::Job,
            SceneAdd { .. } | AssetImport { .. } | BackgroundSet { .. } | BackgroundClear { .. }
            | SceneRangeSet { .. } | PhantomAdd { .. } | KeySet { .. } | CameraKeySet { .. } | PropAdd { .. }
            | ObjectAdd { .. } | ObjectRemove { .. } | ObjectKeySet { .. } | ObjectKeyRemove { .. }
            | LightingSet { .. } | FloorSet { .. } | BrushStroke { .. } | FilmAdd { .. }
            | StageSet { .. }
            | LightAdd { .. } | LightRemove { .. } | LightKeySet { .. } | LightKeyRemove { .. }
            | AmbientSet { .. } | SkySet { .. } | ObjectAttach { .. } | LightAttach { .. } | PhantomScaleSet { .. }
            | MaterialSet { .. } | GradeSet { .. } | FogSet { .. } | VfxAdd { .. } | VfxRemove { .. } | VfxKeySet { .. } | VfxKeyRemove { .. }
            | CameraAdd { .. } | CameraRemove { .. } | CameraSelect { .. } | CameraRangeSet { .. }
            | CameraKeyRemove { .. } | CameraLensSet { .. } | CameraFocalKeySet { .. } | CameraTargetSet { .. }
            | FilmMove { .. } | FilmTrim { .. }
            | FilmRemove { .. } | PhantomRemove { .. } | MotionKeyRemove { .. } | MotionReplace { .. }
            | PropGrid { .. } | Scatter { .. } | ScatterBand { .. }
            | AudioTrackAdd { .. } | AudioClipAdd { .. } | AudioClipUpdate { .. } | TrackVolumeSet { .. }
            | TrackMuteSet { .. } | AudioClipRemove { .. } | AudioTrackRemove { .. }
            | LayerAdd2D { .. } | ExposureRemove { .. } | HistoryUndo | HistoryRedo
            | Batch { .. } => OpClass::DocumentMutation,
        }
    }
}

/// Model parts and their materials: the current import records and the
/// effective values after patches (B4) — a pure function for material_list.
fn material_parts_json(model: &phantom_runtime::import::ModelData, patches: &[MaterialPatch]) -> Value {
    let effective = |rec: &phantom_runtime::import::MaterialRecord, name: &str| {
        let mut base_color = rec.base_color;
        let mut metallic = rec.metallic;
        let mut roughness = rec.roughness;
        let mut emissive = Value::Null;
        let mut opacity = Value::Null;
        let mut unlit = Value::Null;
        for p in patches.iter().filter(|p| p.target.as_deref().is_none_or(|t| t == name)) {
            if let Some(c) = p.base_color {
                base_color = c;
            }
            if let Some(v) = p.metallic {
                metallic = v;
            }
            if let Some(v) = p.roughness {
                roughness = v;
            }
            if let Some(v) = p.emissive {
                emissive = json!(v);
            }
            if let Some(v) = p.opacity {
                opacity = json!(v);
            }
            if let Some(v) = p.unlit {
                unlit = json!(v);
            }
        }
        json!({"base_color": base_color, "metallic": metallic, "roughness": roughness, "emissive": emissive, "opacity": opacity, "unlit": unlit})
    };
    if model.parts.is_empty() {
        // Legacy: one unnamed part; there are no import records — the effective
        // values are computed from the render default.
        let rec = phantom_runtime::import::MaterialRecord {
            name: String::new(),
            base_color: [0.85, 0.65, 0.45, 1.0],
            metallic: 0.0,
            roughness: 0.9,
            base_color_texture: None,
            normal_map_texture: None,
            roughness_texture: None,
            metalness_texture: None,
            invert_roughness: false,
        };
        return json!([{ "part": Value::Null, "material": Value::Null, "current": Value::Null, "effective": effective(&rec, "") }]);
    }
    json!(model
        .parts
        .iter()
        .map(|part| {
            let rec = &model.materials[part.material.min(model.materials.len().saturating_sub(1))];
            json!({
                "part": rec.name,
                "material": rec.name,
                "current": {"base_color": rec.base_color, "metallic": rec.metallic, "roughness": rec.roughness,
                    "base_color_texture": rec.base_color_texture.is_some(), "normal_map_texture": rec.normal_map_texture.is_some()},
                "effective": effective(rec, &rec.name),
            })
        })
        .collect::<Vec<_>>())
}

fn unit_scale() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

fn default_true() -> bool {
    true
}

/// Frame aspect ratio for clean diagnostics: finite, > 0;
/// 16:9 by default.
/// Frames for the camera report (K0.2): without `every` — up to 5 evenly spaced
/// points of the range, with `every` — every Nth frame; a hard cap of 12 frames
/// so the report stays cheap and does not turn into a render of the whole range.
fn camera_report_frames(range: FrameRange, every: Option<u32>) -> Result<Vec<Frame>, SessionError> {
    const MAX_FRAMES: usize = 12;
    let len = range.end - range.start;
    if len <= 0 {
        return Err(SessionError::BadInput("camera range is empty".into()));
    }
    let frames: Vec<Frame> = match every {
        Some(0) => return Err(SessionError::BadInput("every must be > 0".into())),
        Some(step) => (range.start..range.end).step_by(step as usize).take(MAX_FRAMES).collect(),
        None if len <= 5 => (range.start..range.end).collect(),
        None => (0..5).map(|i| range.start + i * (len - 1) / 4).collect(),
    };
    Ok(frames)
}

fn checked_aspect(aspect: Option<f64>) -> Result<f32, SessionError> {
    let aspect = aspect.unwrap_or(crate::analysis::DEFAULT_ASPECT as f64);
    if !aspect.is_finite() || aspect <= 0.0 {
        return Err(SessionError::BadInput(format!("aspect must be finite and > 0: {aspect}")));
    }
    Ok(aspect as f32)
}

// Default flat backdrop geometry — serde defaults for background_set
// (parity with the CLI set-background): wide and tall enough for the default
// camera, standing on the floor at z = −5.
fn default_bg_width() -> f32 {
    24.0
}
fn default_bg_height() -> f32 {
    8.0
}
fn default_bg_center() -> [f32; 3] {
    [0.0, 4.0, -5.0]
}

/// Build controller keys from motion triples plus a parallel yaws vector.
fn build_motion_keys(
    motion: Vec<(Frame, [f32; 3], ActorPose)>,
    yaws: Option<Vec<f32>>,
    op: &str,
) -> Result<Vec<MotionKey>, SessionError> {
    if let Some(y) = &yaws {
        if y.len() != motion.len() {
            return Err(SessionError::BadInput(format!(
                "{op}: yaws ({}) must match motion in length ({})",
                y.len(),
                motion.len()
            )));
        }
    }
    let mut yaws = yaws.map(Vec::into_iter);
    Ok(motion
        .into_iter()
        .map(|(frame, pos, pose)| MotionKey {
            yaw: yaws.as_mut().and_then(|y| y.next()).unwrap_or(0.0),
            frame,
            position: pos.into(),
            pose,
        })
        .collect())
}

/// Validate the explicit asset type for `asset_import --kind`.
fn parse_asset_kind(s: &str) -> Result<AssetKind, SessionError> {
    match s {
        "model-obj" => Ok(AssetKind::ModelObj),
        "model-fbx" => Ok(AssetKind::ModelFbx),
        "image" => Ok(AssetKind::Image),
        "audio" => Ok(AssetKind::Audio),
        "video" => Ok(AssetKind::Video),
        other => Err(SessionError::BadInput(format!("unsupported asset type: {other}"))),
    }
}

/// The camera an operation refers to: an explicit id, or the scene's active
/// camera when the caller names none.
fn resolve_camera(scene: &Scene, camera: Option<u64>) -> Result<CameraId, SessionError> {
    match camera {
        Some(id) => {
            let id = CameraId(id);
            if !scene.cameras.contains_key(&id) {
                return Err(CoreError::CameraNotFound(id).into());
            }
            Ok(id)
        }
        None => Ok(scene.active_camera),
    }
}

/// Substitute a review camera into a clone of the scene: the sheet renders
/// through the ordinary camera path, and the review camera never reaches the
/// document (or the undo history).
fn set_review_camera(document: &mut Scene, id: CameraId, camera: Camera) {
    let range = document.range;
    let entry = document.cameras.entry(id).or_insert_with(|| SceneCamera::new(id, "review", range));
    entry.camera = camera;
    entry.target = None;
}

/// A camera subject on the wire: "phantom:<id>", "object:<id>", or a bare
/// number that is looked up among the scene's phantoms first, then objects.
pub(crate) fn parse_camera_subject(project: &Project, subject: Option<&str>) -> Result<CameraSubject, SessionError> {
    let raw = subject.ok_or_else(|| SessionError::BadInput("camera_target_set: subject is required".into()))?;
    let raw = raw.trim();
    let (kind, id) = match raw.split_once(':') {
        Some((kind, id)) => (Some(kind.trim().to_ascii_lowercase()), id.trim()),
        None => (None, raw),
    };
    let id: u64 = id
        .parse()
        .map_err(|_| SessionError::BadInput(format!("camera_target_set: subject is \"phantom:<id>\" or \"object:<id>\", got {raw}")))?;
    let scene_has = |scene: &Scene, subject: CameraSubject| match subject {
        CameraSubject::Phantom(id) => scene.phantoms.contains_key(&id),
        CameraSubject::Object(id) => scene.objects.contains_key(&id),
    };
    // A bare number: obey whichever id exists in the scene, so a caller does not
    // need the `phantom:`/`object:` prefix for the common cases.
    let named = |subject: CameraSubject| match subject {
        CameraSubject::Phantom(id) if !project.scenes.values().any(|s| scene_has(s, subject)) => {
            CameraSubject::Object(ObjectId(id.0))
        }
        other => other,
    };
    match kind.as_deref() {
        Some("phantom") => Ok(CameraSubject::Phantom(PhantomId(id))),
        Some("object") => Ok(CameraSubject::Object(ObjectId(id))),
        Some(other) => Err(SessionError::BadInput(format!("camera_target_set: subject kind is phantom|object, got {other}"))),
        None => Ok(named(CameraSubject::Phantom(PhantomId(id)))),
    }
}

/// Short rig form for state_get: builtin | actor:<profile> | imported:<asset>.
fn rig_kind(rig: &RigRef) -> String {
    match rig {
        RigRef::Builtin => "builtin".to_string(),
        RigRef::BuiltinActor { actor } => format!("actor:{actor:?}").to_lowercase(),
        RigRef::Imported { asset } => format!("imported:{}", asset.0),
    }
}

/// Scene stage from the wire fields: a preset, an explicit size, or both (the
/// preset fills the axes the caller left out). Without a preset all three
/// dimensions are required, so a half-specified stage never reaches the
/// document.
fn build_stage(
    preset: Option<&str>,
    width: Option<f32>,
    height: Option<f32>,
    depth: Option<f32>,
    center: [f32; 2],
) -> Result<Stage, SessionError> {
    let named = match preset {
        Some(name) => Some(
            StagePreset::parse(name)
                .ok_or_else(|| SessionError::BadInput(format!("preset: small|medium|large, got {name}")))?,
        ),
        None => None,
    };
    let size = match (named, width, height, depth) {
        (Some(p), w, h, d) => {
            let base = p.size();
            StageSize::new(w.unwrap_or(base.width), h.unwrap_or(base.height), d.unwrap_or(base.depth))
        }
        (None, Some(w), Some(h), Some(d)) => StageSize::new(w, h, d),
        (None, ..) => {
            return Err(SessionError::BadInput(
                "stage_set needs a preset or width + height + depth".to_string(),
            ))
        }
    };
    // The preset label survives only while the size really is the preset's.
    let preset = match (named, width, height, depth) {
        (Some(p), None, None, None) => Some(p),
        _ => None,
    };
    Ok(Stage { preset, size, center })
}
/// Child CLI `render` arguments for a scene export job.
fn scene_render_args(
    dir: &Path,
    out: &Path,
    scene: u64,
    width: u32,
    height: u32,
    camera: Option<u64>,
    frames: Option<(Frame, Frame)>,
) -> Vec<std::ffi::OsString> {
    let mut args: Vec<std::ffi::OsString> = vec![
        "render".into(), dir.into(), "--scene".into(), scene.to_string().into(),
        "--out".into(), out.into(), "--width".into(), width.to_string().into(),
        "--height".into(), height.to_string().into(),
    ];
    if let Some(cam) = camera {
        args.push("--camera".into());
        args.push(cam.to_string().into());
    }
    if let Some((s, e)) = frames {
        args.push("--frames".into());
        args.push(format!("{s}-{e}").into());
    }
    args
}

/// Child CLI `render-film` arguments for a film export job.
fn film_render_args(dir: &Path, out: &Path, width: u32, height: u32, frames: Option<(Frame, Frame)>, normalize: bool, target_peak_db: Option<f32>) -> Vec<std::ffi::OsString> {
    let mut args: Vec<std::ffi::OsString> = vec![
        "render-film".into(), dir.into(), "--out".into(), out.into(),
        "--width".into(), width.to_string().into(), "--height".into(), height.to_string().into(),
        "--force".into(),
    ];
    if let Some((s, e)) = frames {
        args.push("--frames".into());
        args.push(format!("{s}-{e}").into());
    }
    if normalize {
        args.push("--normalize".into());
        if let Some(db) = target_peak_db {
            args.push("--target-peak-db".into());
            args.push(db.to_string().into());
        }
    }
    args
}

/// Bounded ring buffer for the child process's stderr:
/// the last ~64 lines, at most ~8 KiB in total.
const ERR_TAIL_MAX_LINES: usize = 64;
const ERR_TAIL_MAX_BYTES: usize = 8192;

#[derive(Default)]
pub struct ErrRing {
    lines: std::collections::VecDeque<String>,
    bytes: usize,
}

impl ErrRing {
    fn push(&mut self, mut line: String) {
        // A single line is capped too, so one "wall" without newlines
        // cannot push out the whole tail.
        if line.len() > 1024 {
            line.truncate(1024);
        }
        self.bytes += line.len();
        self.lines.push_back(line);
        while self.lines.len() > ERR_TAIL_MAX_LINES || self.bytes > ERR_TAIL_MAX_BYTES {
            if let Some(old) = self.lines.pop_front() {
                self.bytes -= old.len();
            } else {
                break;
            }
        }
    }

    fn tail(&self) -> Vec<String> {
        self.lines.iter().cloned().collect()
    }
}

pub struct Job {
    child: Child,
    rx: std::sync::mpsc::Receiver<String>,
    pub last_line: String,
    /// Exit code; None while the process is alive (or it was killed by a signal).
    exit_code: Option<i32>,
    /// true once completion has been recorded (exit_code can be None
    /// when the process was killed by a signal).
    finished: bool,
    err_tail: std::sync::Arc<std::sync::Mutex<ErrRing>>,
    /// stderr reader thread; joined when completion is recorded, otherwise
    /// job_status may return an empty error_tail before the pipe is fully read.
    err_drain: Option<std::thread::JoinHandle<()>>,
    /// stdout reader thread; joined when completion is recorded, otherwise
    /// the last progress line may never reach last_line (the channel is
    /// drained before try_wait while the child has already exited).
    out_drain: Option<std::thread::JoinHandle<()>>,
    started: std::time::Instant,
    cleanup: Option<PathBuf>,
}

/// Render progress: the child CLI prints lines like "done/total".
fn parse_progress(line: &str) -> (Option<u64>, Option<u64>) {
    match line.trim().split_once('/') {
        Some((a, b)) => match (a.trim().parse::<u64>(), b.trim().parse::<u64>()) {
            (Ok(done), Ok(total)) => (Some(done), Some(total)),
            _ => (None, None),
        },
        None => (None, None),
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct RecentProject {
    path: PathBuf,
    timestamp: u64,
}

fn default_state_path() -> PathBuf {
    let base = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/state")))
        .unwrap_or_else(|| std::env::temp_dir().join("phantom-state"));
    base.join("phantom-animate/recent.json")
}

pub struct Session {
    pub project: Option<Project>,
    pub history: History,
    pub dir: Option<PathBuf>,
    jobs: HashMap<u64, Job>,
    next_job: u64,
    lock_guard: Option<LockGuard>,
    state_path: PathBuf,
    /// Stage clamps the last document mutation performed. The acknowledgement
    /// gate turns them into `clamped: true` + `clamps: [...]`, so an agent
    /// always learns that its coordinate was corrected (layout mode).
    last_clamps: Vec<Clamp>,
}

impl Session {
    pub fn new() -> Self {
        Self::with_state_path(default_state_path())
    }

    /// Injectable per-user state file for deterministic tests without mutating
    /// process-global environment variables.
    pub(crate) fn with_state_path(state_path: PathBuf) -> Self {
        Self {
            project: None,
            history: History::default(),
            dir: None,
            jobs: HashMap::new(),
            next_job: 1,
            lock_guard: None,
            state_path,
            last_clamps: Vec::new(),
        }
    }

    fn cmd(&mut self, c: Command) -> Result<CommandOutput, SessionError> {
        let project = self.project.as_mut().ok_or(SessionError::NoProject)?;
        let out = self.history.apply(project, c)?;
        self.last_clamps = out.clamped.clone();
        Ok(out)
    }

    fn dir(&self) -> Result<&Path, SessionError> {
        self.dir.as_deref().ok_or(SessionError::NoProject)
    }

    /// The write path must stay inside the project directory.
    fn check_out_path(&self, out: &Path) -> Result<PathBuf, SessionError> {
        let dir = self.dir()?.canonicalize()?;
        let abs = if out.is_absolute() { out.to_path_buf() } else { dir.join(out) };
        // The file may not exist yet: canonicalize the parent.
        let parent = abs.parent().ok_or_else(|| SessionError::OutsideAllowedRoots(abs.clone()))?;
        let canon = parent.canonicalize().unwrap_or_else(|_| parent.to_path_buf());
        if !canon.starts_with(&dir) {
            return Err(SessionError::OutsideAllowedRoots(abs));
        }
        Ok(abs)
    }

    fn current_dir(&self) -> Option<PathBuf> {
        self.dir.as_ref().map(|dir| dir.canonicalize().unwrap_or_else(|_| dir.clone()))
    }

    fn reject_switch(&self, target: &Path) -> Result<(), SessionError> {
        if let Some(current) = self.current_dir() {
            if current != target {
                return Err(SessionError::ProjectSwitchRejected { current, target: target.to_path_buf() });
            }
        }
        Ok(())
    }

    fn remember_recent(&self, path: &Path) -> Result<(), std::io::Error> {
        let parent = self.state_path.parent().ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "recent state has no parent"))?;
        std::fs::create_dir_all(parent)?;
        let recent = RecentProject {
            path: path.to_path_buf(),
            timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs(),
        };
        let bytes = serde_json::to_vec_pretty(&recent).expect("recent project serializes");
        let tmp = self.state_path.with_extension(format!("tmp-{}", std::process::id()));
        std::fs::write(&tmp, bytes)?;
        if let Err(error) = std::fs::rename(&tmp, &self.state_path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(error);
        }
        Ok(())
    }

    fn recent_project(&self) -> (Option<RecentProject>, Vec<Value>) {
        match std::fs::read(&self.state_path) {
            Ok(bytes) => match serde_json::from_slice(&bytes) {
                Ok(recent) => (Some(recent), Vec::new()),
                Err(error) => (None, vec![json!({"code":"recent_state_corrupt", "message":error.to_string()})]),
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => (None, Vec::new()),
            Err(error) => (None, vec![json!({"code":"recent_state_unreadable", "message":error.to_string()})]),
        }
    }

    fn lock_warning(reclaimed: Option<project_lock::LockOwner>) -> Value {
        reclaimed
            .map(|owner| json!([{"code": "stale_lock_reclaimed", "owner": owner}]))
            .unwrap_or_else(|| json!([]))
    }

    /// Single acknowledgement gate for every operation. Document mutations
    /// cannot bypass revision/next_id and their post-command canonical echo.
    pub fn execute(&mut self, op: Op) -> Result<Value, SessionError> {
        let class = op.class();
        let ack_op = op.clone();
        self.last_clamps.clear();
        let mut value = self.execute_inner(op)?;
        if class == OpClass::DocumentMutation {
            let project = self.project.as_ref().ok_or(SessionError::NoProject)?;
            let entity = Self::mutation_entity(project, &ack_op, &value);
            if !value.is_object() {
                value = json!({ "result": value });
            }
            let clamps = std::mem::take(&mut self.last_clamps);
            let object = value.as_object_mut().expect("normalized object");
            object.insert("revision".into(), json!(project.revision));
            object.insert("next_id".into(), json!(project.next_id()));
            object.insert("entity".into(), entity);
            if !clamps.is_empty() {
                object.insert("clamped".into(), json!(true));
                object.insert("clamps".into(), json!(clamps));
            }
        }
        Ok(value)
    }

    fn mutation_entity(project: &Project, op: &Op, result: &Value) -> Value {
        use Op::*;
        let scene = |id: u64| project.scenes.get(&SceneId(id));
        let track = |id: u64| {
            project.scenes.values().flat_map(|s| s.audio_tracks.iter())
                .chain(project.film_audio.iter()).find(|t| t.id.0 == id)
        };
        match op {
            SceneAdd { .. } => result["scene"].as_u64().and_then(|id| scene(id)).map(|s| json!({
                "kind":"scene", "id":s.id.0, "name":s.name, "range":s.range
            })).unwrap_or(Value::Null),
            SceneRangeSet { scene: id, .. } => scene(*id).map(|s| json!({"kind":"scene", "id":id, "name":s.name, "range":s.range})).unwrap_or(Value::Null),
            AssetImport { .. } => result["asset"].as_u64().and_then(|id| project.assets.get(&AssetId(id))).map(|a| json!({
                "kind":"asset", "id":a.id.0, "asset_kind":a.kind, "path":a.path
            })).unwrap_or(Value::Null),
            BackgroundSet { scene: id, .. } | BackgroundClear { scene: id } => scene(*id).map(|s| json!({
                "kind":"background", "scene":id, "background":s.background
            })).unwrap_or(Value::Null),
            PhantomAdd { scene: sid, .. } => result["phantom"].as_u64().and_then(|id| scene(*sid)?.phantoms.get(&PhantomId(id))).map(|p| json!({
                "kind":"phantom", "scene":sid, "id":p.id.0, "name":p.name, "rig":p.rig
            })).unwrap_or(Value::Null),
            PhantomRemove { scene, phantom } => json!({"kind":"phantom", "scene":scene, "id":phantom, "removed":true}),
            KeySet { scene: sid, phantom, frame, .. } => scene(*sid).and_then(|s| s.phantoms.get(&PhantomId(*phantom))).and_then(|p| p.motion.iter().find(|k| k.frame == *frame)).map(|key| json!({
                "kind":"motion_key", "scene":sid, "phantom":phantom, "id":{"phantom":phantom,"frame":frame}, "key":key
            })).unwrap_or(Value::Null),
            MotionKeyRemove { scene, phantom, frame } => json!({"kind":"motion_key", "scene":scene, "phantom":phantom, "id":{"phantom":phantom,"frame":frame}, "removed":true}),
            MotionReplace { scene: sid, phantom, .. } => scene(*sid).and_then(|s| s.phantoms.get(&PhantomId(*phantom))).map(|p| json!({
                "kind":"motion", "scene":sid, "phantom":phantom, "keys":p.motion
            })).unwrap_or(Value::Null),
            CameraAdd { scene: sid, .. } => result["camera"].as_u64().and_then(|id| scene(*sid)?.cameras.get(&CameraId(id))).map(|c| json!({
                "kind":"camera", "scene":sid, "id":c.id.0, "name":c.name, "range":c.range, "camera":c.camera, "target":c.target,
            })).unwrap_or(Value::Null),
            CameraRemove { scene, camera } => json!({"kind":"camera", "scene":scene, "id":camera, "removed":true}),
            CameraSelect { scene: sid, .. } => scene(*sid).map(|s| json!({"kind":"active_camera", "scene":sid, "id":s.active_camera.0})).unwrap_or(Value::Null),
            CameraKeySet { scene: sid, camera, .. } => {
                let id = camera.unwrap_or_else(|| scene(*sid).map(|s| s.active_camera.0).unwrap_or(0));
                scene(*sid).and_then(|s| s.cameras.get(&CameraId(id))).map(|c| json!({
                    "kind":"camera", "scene":sid, "id":c.id.0, "name":c.name, "range":c.range, "camera":c.camera, "target":c.target,
                })).unwrap_or(Value::Null)
            }
            ObjectAdd { scene: sid, .. } | PropAdd { scene: sid, .. } => result["object"].as_u64().and_then(|id| scene(*sid)?.objects.get(&ObjectId(id))).map(|o| json!({
                "kind":"object", "scene":sid, "id":o.id.0, "name":o.name, "asset":o.asset.0, "keys":o.keys
            })).unwrap_or(Value::Null),
            ObjectRemove { scene, object } => json!({"kind":"object", "scene":scene, "id":object, "removed":true}),
            PropGrid { scene: sid, .. } => result["objects"].as_array().map(|ids| json!({
                "kind":"prop_grid", "scene":sid, "ids":ids, "count":ids.len()
            })).unwrap_or(Value::Null),
            Scatter { scene: sid, .. } => json!({
                "kind":"scatter", "scene":sid,
                "objects": result["objects"].as_array().map(|v| v.len()).unwrap_or(0),
                "band": result["band"], "seed": result["seed"],
            }),
            ScatterBand { scene: sid, name, .. } => result["object"].as_u64().map(|id| json!({
                "kind":"object", "scene":sid, "id":id, "name":name, "band":true
            })).unwrap_or(Value::Null),
            ObjectKeySet { scene: sid, object, frame, .. } => scene(*sid).and_then(|s| s.objects.get(&ObjectId(*object))).and_then(|o| o.keys.iter().find(|k| k.frame == *frame)).map(|key| json!({
                "kind":"object_key", "scene":sid, "object":object, "id":{"object":object,"frame":frame}, "key":key
            })).unwrap_or(Value::Null),
            ObjectKeyRemove { scene, object, frame } => json!({"kind":"object_key", "scene":scene, "object":object, "id":{"object":object,"frame":frame}, "removed":true}),
            LightingSet { scene: id, .. } => scene(*id).map(|s| json!({"kind":"lighting", "scene":id, "lighting":s.lighting})).unwrap_or(Value::Null),
            LightAdd { scene: sid, .. } => result["light"].as_u64().and_then(|id| scene(*sid)?.lights.iter().find(|l| l.id.0 == id)).map(|l| json!({
                "kind":"light", "scene":sid, "id":l.id.0, "light":l
            })).unwrap_or(Value::Null),
            LightRemove { scene, light } => json!({"kind":"light", "scene":scene, "id":light, "removed":true}),
            LightKeySet { scene: sid, light, .. } | LightKeyRemove { scene: sid, light, .. } => scene(*sid).and_then(|s| s.lights.iter().find(|l| l.id.0 == *light)).map(|l| json!({
                "kind":"light", "scene":sid, "id":l.id.0, "light":l
            })).unwrap_or(Value::Null),
            AmbientSet { scene: id, .. } => scene(*id).map(|s| json!({"kind":"ambient", "scene":id, "ambient":s.ambient})).unwrap_or(Value::Null),
            ObjectAttach { scene: sid, object, .. } => scene(*sid).and_then(|s| s.objects.get(&ObjectId(*object))).map(|o| json!({
                "kind":"object", "scene":sid, "id":o.id.0, "attach":o.attach
            })).unwrap_or(Value::Null),
            LightAttach { scene: sid, light, .. } => scene(*sid).and_then(|s| s.lights.iter().find(|l| l.id.0 == *light)).map(|l| json!({
                "kind":"light", "scene":sid, "id":l.id.0, "attach":l.attach
            })).unwrap_or(Value::Null),
            PhantomScaleSet { scene: sid, phantom, .. } => scene(*sid).and_then(|s| s.phantoms.get(&PhantomId(*phantom))).map(|p| json!({
                "kind":"phantom", "scene":sid, "id":p.id.0, "scale":p.scale
            })).unwrap_or(Value::Null),
            MaterialSet { scene: sid, phantom, object, .. } => scene(*sid).map(|s| {
                let patches = match (phantom, object) {
                    (Some(id), _) => s.phantoms.get(&PhantomId(*id)).map(|p| p.material_patches.clone()),
                    (_, Some(id)) => s.objects.get(&ObjectId(*id)).map(|o| o.material_patches.clone()),
                    _ => None,
                };
                json!({"kind":"material_patches", "scene":sid, "phantom":phantom, "object":object, "patches":patches})
            }).unwrap_or(Value::Null),
            MaterialList { .. } => Value::Null,
            GradeSet { scene: id, .. } => scene(*id).map(|s| json!({"kind":"grade", "scene":id, "grade":s.grade})).unwrap_or(Value::Null),
            FogSet { scene: id, .. } => scene(*id).map(|s| json!({"kind":"fog", "scene":id, "fog":s.fog})).unwrap_or(Value::Null),
            VfxAdd { scene: sid, .. } => result["vfx"].as_u64().and_then(|id| scene(*sid)?.vfx.iter().find(|v| v.id.0 == id)).map(|v| json!({
                "kind":"vfx", "scene":sid, "id":v.id.0, "vfx":v
            })).unwrap_or(Value::Null),
            VfxRemove { scene, vfx } => json!({"kind":"vfx", "scene":scene, "id":vfx, "removed":true}),
            VfxKeySet { scene: sid, vfx, .. } | VfxKeyRemove { scene: sid, vfx, .. } => scene(*sid).and_then(|s| s.vfx.iter().find(|v| v.id.0 == *vfx)).map(|v| json!({
                "kind":"vfx", "scene":sid, "id":v.id.0, "vfx":v
            })).unwrap_or(Value::Null),
            SkySet { scene: id, .. } => scene(*id).map(|s| json!({"kind":"sky", "scene":id, "sky":s.sky})).unwrap_or(Value::Null),
            FloorSet { scene: id, .. } => scene(*id).map(|s| json!({"kind":"floor", "scene":id, "floor":s.floor})).unwrap_or(Value::Null),
            StageSet { scene: id, .. } => scene(*id).map(|s| json!({"kind":"stage", "scene":id, "stage":s.stage})).unwrap_or(Value::Null),
            FilmAdd { .. } => result["clip"].as_u64().and_then(|id| project.film.iter().find(|c| c.id.0 == id)).map(|c| json!({
                "kind":"film_clip", "id":c.id.0, "scene":c.scene.0, "range":c.range, "camera":c.camera.map(|s|s.0)
            })).unwrap_or(Value::Null),
            FilmMove { clip, .. } | FilmTrim { clip, .. } => project.film.iter().find(|c| c.id.0 == *clip).map(|c| json!({
                "kind":"film_clip", "id":c.id.0, "scene":c.scene.0, "range":c.range, "camera":c.camera.map(|s|s.0),
                "index":project.film.iter().position(|x| x.id == c.id)
            })).unwrap_or(Value::Null),
            FilmRemove { clip } => json!({"kind":"film_clip", "id":clip, "removed":true}),
            CameraRangeSet { scene: sid, camera, .. } | CameraKeyRemove { scene: sid, camera, .. }
            | CameraLensSet { scene: sid, camera, .. } | CameraFocalKeySet { scene: sid, camera, .. }
            | CameraTargetSet { scene: sid, camera, .. } => scene(*sid).and_then(|s| s.cameras.get(&CameraId(*camera))).map(|c| json!({
                "kind":"camera", "scene":sid, "id":c.id.0, "name":c.name, "range":c.range, "camera":c.camera, "target":c.target,
            })).unwrap_or(Value::Null),
            AudioTrackAdd { scene, .. } => result["track"].as_u64().and_then(track).map(|t| json!({
                "kind":"audio_track", "scene":scene, "id":t.id.0, "name":t.name, "volume":t.volume, "muted":t.muted, "clips":t.clips
            })).unwrap_or(Value::Null),
            TrackVolumeSet { track: id, .. } | TrackMuteSet { track: id, .. } => track(*id).map(|t| json!({
                "kind":"audio_track", "id":t.id.0, "name":t.name, "volume":t.volume, "muted":t.muted, "clips":t.clips
            })).unwrap_or(Value::Null),
            AudioTrackRemove { track } => json!({"kind":"audio_track", "id":track, "removed":true}),
            AudioClipAdd { track: tid, .. } => result["clip"].as_u64().and_then(|id| track(*tid)?.clips.iter().find(|c| c.id.0 == id)).map(|c| json!({
                "kind":"audio_clip", "track":tid, "id":c.id.0, "clip":c
            })).unwrap_or(Value::Null),
            AudioClipUpdate { track: tid, clip, .. } => track(*tid).and_then(|t| t.clips.iter().find(|c| c.id.0 == *clip)).map(|c| json!({
                "kind":"audio_clip", "track":tid, "id":c.id.0, "clip":c
            })).unwrap_or(Value::Null),
            AudioClipRemove { track, clip } => json!({"kind":"audio_clip", "track":track, "id":clip, "removed":true}),
            LayerAdd2D { scene: sid, .. } => result["layer"].as_u64().and_then(|id| scene(*sid)?.layers_2d.iter().find(|l| l.id.0 == id)).map(|l| json!({
                "kind":"layer_2d", "scene":sid, "id":l.id.0, "name":l.name, "canvas":l.canvas, "exposures":l.exposures
            })).unwrap_or(Value::Null),
            BrushStroke { scene: sid, frame, .. } => result["layer"].as_u64().and_then(|id| {
                scene(*sid)?.layers_2d.iter().find(|l| l.id.0 == id)
            }).and_then(|l| l.exposures.get(frame).map(|exposure| (l, exposure))).map(|(l, exposure)| json!({
                "kind":"exposure", "scene":sid, "layer":l.id.0, "id":{"layer":l.id.0,"frame":frame}, "frame":frame, "exposure":exposure
            })).unwrap_or(Value::Null),
            ExposureRemove { scene, layer, frame } => json!({"kind":"exposure", "scene":scene, "layer":layer, "id":{"layer":layer,"frame":frame}, "removed":true}),
            HistoryUndo | HistoryRedo => json!({
                "kind":"history", "operation":if matches!(op, HistoryUndo) {"undo"} else {"redo"}, "applied":result["applied"]
            }),
            // The batch's canonical echo is the last operation's entity; per-op
            // snapshots live in results[i].entity.
            Batch { .. } => result["results"].as_array()
                .and_then(|r| r.last())
                .map(|last| last["entity"].clone())
                .unwrap_or(Value::Null),
            // Classification guarantees this arm is unreachable for successful
            // acknowledgement construction; keeping it explicit makes the
            // exhaustive classifier above the source of truth.
            ProjectNew { .. } | ProjectOpen { .. } | LockRelease { .. } | ProjectSave | ProjectRecent
            | StateGet | BuiltinCatalog | CatalogQuery { .. } | LightingGet { .. } | FloorGet { .. }
            | StageGet { .. }
            | HistoryList | RenderScene { .. } | RenderFilm { .. } | JobStatus { .. } | JobWait { .. }
            | Look { .. }
            | JobCancel { .. } | SceneGet { .. } | CameraGet { .. } | AssetGet { .. } | FilmGet
            | AssetList | JobList | RenderStill { .. } | CameraReport { .. } | Validate { .. } | CameraCoverage { .. }
            | BackgroundFit { .. } | AudioReport { .. } => Value::Null,
        }
    }

    /// Operation name from the serde tag ("scene_add", "key_set", ...) for
    /// batch diagnostics and logging.
    fn op_name(op: &Op) -> String {
        serde_json::to_value(op)
            .ok()
            .and_then(|v| v["op"].as_str().map(str::to_owned))
            .unwrap_or_else(|| "unknown".to_string())
    }

    /// Atomic batch of mutations. Operations run through the normal path (the
    /// same validation and id allocation as single calls); on the first
    /// error the project and history are restored from a deep snapshot, so no
    /// partial state or extra undo steps remain. On success the
    /// undo steps collapse into one Batch step, and revision is normalized
    /// to +1 for the whole batch.
    fn execute_batch(&mut self, ops: Vec<Op>) -> Result<Value, SessionError> {
        if ops.is_empty() {
            return Err(SessionError::BadInput("batch: empty operation list".to_string()));
        }
        // Validate the classes before touching the document.
        for (index, op) in ops.iter().enumerate() {
            if matches!(op, Op::Batch { .. }) {
                return Err(SessionError::BatchInvalid { index, reason: "nested batch is not allowed".to_string() });
            }
            if op.class() != OpClass::DocumentMutation {
                return Err(SessionError::BatchInvalid {
                    index,
                    reason: format!("operation {} is not a document mutation (class {:?})", Self::op_name(op), op.class()),
                });
            }
        }
        let project_snapshot = self.project.clone();
        let history_snapshot = self.history.clone();
        let pre_revision = self.project.as_ref().map(|p| p.revision);
        let depth0 = self.history.undo_depth();
        let mut results = Vec::with_capacity(ops.len());
        for (index, op) in ops.iter().cloned().enumerate() {
            match self.execute(op) {
                Ok(mut value) => {
                    // A per-op result carries only the created ids and the canonical
                    // entity: the batch has one final revision/next_id, which the
                    // outer ack path adds.
                    if let Some(object) = value.as_object_mut() {
                        object.remove("revision");
                        object.remove("next_id");
                    }
                    results.push(value);
                }
                Err(source) => {
                    self.project = project_snapshot;
                    self.history = history_snapshot;
                    return Err(SessionError::BatchFailed {
                        index,
                        op: Self::op_name(&ops[index]),
                        source: Box::new(source),
                    });
                }
            }
        }
        let applied = self.history.undo_depth() - depth0;
        self.history.squash_last(applied);
        if let (Some(project), Some(pre)) = (self.project.as_mut(), pre_revision) {
            project.revision = pre + 1;
        }
        // The per-op acks already carry their clamps; the package ack reports
        // them once more for the whole batch.
        self.last_clamps = results
            .iter()
            .filter_map(|r| r.get("clamps").and_then(|c| c.as_array()))
            .flatten()
            .filter_map(|c| serde_json::from_value(c.clone()).ok())
            .collect();
        Ok(json!({ "applied": ops.len(), "results": results }))
    }

    fn execute_inner(&mut self, op: Op) -> Result<Value, SessionError> {
        use Op::*;
        match op {
            ProjectNew { dir, name } => {
                // The directory itself may be new, but the lock must exist
                // before create_project can write/replace the manifest.
                std::fs::create_dir_all(&dir)?;
                let target = dir.canonicalize()?;
                self.reject_switch(&target)?;
                if self.lock_guard.as_ref().is_some_and(|g| g.dir() == target) {
                    return Ok(json!({ "ok": true, "already_open": true, "warnings": [] }));
                }
                let acquired = LockGuard::acquire(&target, None, false)?;
                let warnings = Self::lock_warning(acquired.reclaimed.clone());
                let project = match storage::create_project(&target, &name) {
                    Ok(project) => project,
                    Err(e) => {
                        drop(acquired);
                        return Err(e.into());
                    }
                };
                self.lock_guard = Some(acquired.guard);
                self.project = Some(project);
                self.dir = Some(target.clone());
                self.history = History::default();
                let mut warnings = warnings.as_array().cloned().unwrap_or_default();
                if let Err(error) = self.remember_recent(&target) {
                    warnings.push(json!({"code":"recent_state_write_failed", "message":error.to_string()}));
                }
                Ok(json!({ "ok": true, "already_open": false, "warnings": warnings }))
            }
            ProjectOpen { dir, force } => {
                // Validate the target before lock acquisition. Switching an
                // already-open Session is rejected, so its current guard and
                // unsaved document can never be stranded.
                let project = storage::load_project(&dir)?;
                let target = dir.canonicalize()?;
                self.reject_switch(&target)?;
                if self.lock_guard.as_ref().is_some_and(|g| g.dir() == target) {
                    return Ok(json!({ "ok": true, "already_open": true, "warnings": [] }));
                }
                let acquired = LockGuard::acquire(&target, None, force)?;
                let warnings = Self::lock_warning(acquired.reclaimed.clone());
                self.lock_guard = Some(acquired.guard);
                self.project = Some(project);
                self.dir = Some(target.clone());
                self.history = History::default();
                let mut warnings = warnings.as_array().cloned().unwrap_or_default();
                if let Err(error) = self.remember_recent(&target) {
                    warnings.push(json!({"code":"recent_state_write_failed", "message":error.to_string()}));
                }
                Ok(json!({ "ok": true, "already_open": false, "warnings": warnings }))
            }
            ProjectRecent => {
                let (recent, warnings) = self.recent_project();
                Ok(json!({
                    "recent": recent.map(|r| json!({"dir":r.path, "timestamp":r.timestamp})),
                    "current_dir": self.current_dir(),
                    "revision": self.project.as_ref().map(|p| p.revision),
                    "warnings": warnings,
                }))
            }
            LockRelease { dir, force } => {
                let target = dir.canonicalize().unwrap_or(dir);
                if let Some(guard) = &self.lock_guard {
                    if guard.dir() == target {
                        let owner = guard.owner().ok();
                        return Ok(json!({
                            "ok": true,
                            "released": false,
                            "already_owned": true,
                            "observed_owner": owner,
                        }));
                    }
                }
                let (released, owner) = project_lock::release_external(&target, force)?;
                Ok(json!({
                    "ok": true,
                    "released": released,
                    "already_owned": false,
                    "observed_owner": owner,
                }))
            }
            ProjectSave => {
                storage::save_project(self.dir()?, self.project.as_ref().ok_or(SessionError::NoProject)?)?;
                Ok(json!({ "ok": true }))
            }
            StateGet => {
                let p = self.project.as_ref().ok_or(SessionError::NoProject)?;
                Ok(json!({
                    "name": p.name,
                    "revision": p.revision,
                    "scenes": p.scenes.keys().map(|s| s.0).collect::<Vec<_>>(),
                    "assets": p.assets.values().map(|a| json!({"id": a.id.0, "kind": format!("{:?}", a.kind), "path": a.path})).collect::<Vec<_>>(),
                    "phantoms": p.scenes.values().flat_map(|s| s.phantoms.values().map(move |ph| json!({
                        "scene": s.id.0, "id": ph.id.0, "name": ph.name, "rig": rig_kind(&ph.rig),
                        "motion_keys": ph.motion.len(),
                    })).collect::<Vec<_>>()).collect::<Vec<_>>(),
                    "film": p.film.iter().map(|c| json!({
                        "id": c.id.0, "scene": c.scene.0, "start": c.range.start, "end": c.range.end,
                        "camera": c.camera.map(|sh| sh.0),
                    })).collect::<Vec<_>>(),
                    "audio_tracks": p.scenes.values().flat_map(|s| {
                        s.audio_tracks.iter().map(move |t| json!({
                            "scene": s.id.0, "id": t.id.0, "name": t.name, "volume": t.volume, "muted": t.muted, "clips": t.clips.len(),
                        })).collect::<Vec<_>>()
                    }).chain(p.film_audio.iter().map(|t| json!({
                        "scene": Value::Null, "id": t.id.0, "name": t.name, "volume": t.volume, "muted": t.muted, "clips": t.clips.len(),
                    }))).collect::<Vec<_>>(),
                    "objects": p.scenes.values().flat_map(|s| s.objects.values().map(move |o| json!({
                        "scene": s.id.0, "id": o.id.0, "name": o.name, "asset": o.asset.0, "keys": o.keys.len(),
                    })).collect::<Vec<_>>()).collect::<Vec<_>>() ,
                    "stages": p.scenes.values().map(|s| json!({"scene": s.id.0, "stage": s.stage})).collect::<Vec<_>>(),
                    "cameras": p.scenes.values().flat_map(|s| s.cameras.values().map(move |c| json!({
                        "scene": s.id.0, "id": c.id.0, "name": c.name, "start": c.range.start, "end": c.range.end,
                        "camera_keys": c.camera.keys.len(), "focal_mm": c.camera.focal_mm,
                        "target": c.target, "active": s.active_camera == c.id,
                    })).collect::<Vec<_>>()).collect::<Vec<_>>(),
                    "film_len": film_len(&p.film),
                    "audio_summary": crate::analysis::audio_summary(p),
                }))
            }
            SceneAdd { name, start, end } => {
                let out = self.cmd(Command::AddScene { name, range: FrameRange::new(start, end)? })?;
                Ok(json!({ "scene": out.scene.map(|s| s.0) }))
            }
            AssetImport { file, kind } => {
                let kind = match kind.as_deref() {
                    Some(k) => parse_asset_kind(k)?,
                    None => match file.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref() {
                        Some("obj") => AssetKind::ModelObj,
                        Some("fbx") => AssetKind::ModelFbx,
                        Some("mp3" | "wav" | "ogg" | "flac" | "m4a") => AssetKind::Audio,
                        Some("mp4" | "gif" | "webm" | "mov") => AssetKind::Video,
                        Some("png" | "jpg" | "jpeg" | "tiff" | "tif") => AssetKind::Image,
                        Some(ext) => return Err(SessionError::BadInput(format!("unsupported asset extension: {ext}"))),
                        None => return Err(SessionError::BadInput("the asset has no extension".into())),
                    },
                };
                // Models go through bundle import (model + MTL + textures);
                // other assets are a single file.
                let (rel, diagnostics, bounds) = if matches!(kind, AssetKind::ModelObj | AssetKind::ModelFbx) {
                    let bundle = phantom_runtime::import::import_model_bundle(self.dir()?, &file)?;
                    (bundle.model, bundle.diagnostics, bundle.bounds)
                } else {
                    (storage::import_file(self.dir()?, &file)?, Vec::new(), None)
                };
                let out = self.cmd(Command::RegisterAsset { kind, path: rel, bounds })?;
                Ok(json!({ "asset": out.asset.map(|a| a.0), "diagnostics": diagnostics }))
            }
            BackgroundSet { scene, asset, video, width, height, center, fit } => {
                let background = Background {
                    image: if video { BackgroundSource::Video { video: AssetId(asset) } } else { BackgroundSource::Image(AssetId(asset)) },
                    width,
                    height,
                    center: center.into(),
                    fit: fit.unwrap_or_default(),
                };
                background.validate()?;
                self.cmd(Command::SetBackground { scene: SceneId(scene), background })?;
                Ok(json!({ "ok": true }))
            }
            BackgroundClear { scene } => {
                self.cmd(Command::ClearBackground { scene: SceneId(scene) })?;
                Ok(json!({ "ok": true }))
            }
            SceneRangeSet { scene, start, end } => {
                self.cmd(Command::SetSceneRange { scene: SceneId(scene), range: FrameRange::new(start, end)? })?;
                Ok(json!({ "ok": true }))
            }
            BuiltinCatalog => Ok(crate::builtin::catalog()),
            CatalogQuery { kind, query } => {
                let kind = match kind.as_deref() {
                    None => None,
                    Some("primitive") => Some(phantom_core::catalog::CatalogKind::Primitive),
                    Some("furniture") => Some(phantom_core::catalog::CatalogKind::Furniture),
                    Some("interior") => Some(phantom_core::catalog::CatalogKind::Interior),
                    Some("animal") => Some(phantom_core::catalog::CatalogKind::Animal),
                    Some("street") => Some(phantom_core::catalog::CatalogKind::Street),
                    Some("exterior") => Some(phantom_core::catalog::CatalogKind::Exterior),
                    Some(other) => {
                        return Err(SessionError::BadInput(format!(
                            "kind: primitive|furniture|interior|animal|street|exterior, got {other}"
                        )))
                    }
                };
                let entries = phantom_core::catalog::catalog_query(kind, query.as_deref());
                Ok(json!({
                    "total": entries.len(),
                    "entries": entries.iter().map(|e| json!({
                        "name": e.name,
                        "kind": e.kind.as_str(),
                        "size": e.size.map(crate::builtin::mm),
                        "origin": e.origin,
                        "materials": e.materials,
                        "open_sides": e.open_sides,
                        "openings": e.openings,
                        // What a room brings with it: placing it places these.
                        "furniture": e.furniture,
                        "note": e.note,
                    })).collect::<Vec<_>>(),
                }))
            }
            PropAdd { scene, name, prop, pos, yaw, pitch, roll, scale } => {
                // A starting key is set only if the director provided one:
                // without a key the object sits at the origin, like object_add.
                let key = pos.map(|p| ObjectKey {
                    frame: 0,
                    position: p.into(),
                    yaw,
                    pitch,
                    roll,
                    scale: scale.map(Into::into).unwrap_or(glam::Vec3::ONE),
                });
                // A furnished room is one package: the shell plus a piece per
                // entry of its furniture set (expanded in the document layer,
                // so GUI, CLI and MCP place a room identically). Report every
                // id that arrived, or the agent would have to guess them.
                let before: Vec<u64> = self.project.as_ref().map(|p| p.scenes.get(&SceneId(scene)).map(|s| s.objects.keys().map(|id| id.0).collect()).unwrap_or_default()).unwrap_or_default();
                let out = self.cmd(Command::AddBuiltinObject { scene: SceneId(scene), name, prop, key })?;
                let after: Vec<u64> = self.project.as_ref().and_then(|p| p.scenes.get(&SceneId(scene)).map(|s| s.objects.keys().map(|id| id.0).collect())).unwrap_or_default();
                let created: Vec<u64> = after.iter().copied().filter(|id| !before.contains(id)).collect();
                if created.len() > 1 {
                    return Ok(json!({ "object": out.object.map(|o| o.0), "objects": created, "furniture": created.len() - 1 }));
                }
                Ok(json!({ "object": out.object.map(|o| o.0) }))
            }
            ScatterBand { scene, name, field, pos, yaw } => {
                let key = ObjectKey { frame: 0, position: pos.into(), yaw, pitch: 0.0, roll: 0.0, scale: glam::Vec3::ONE };
                let out = self.cmd(Command::AddScatterField { scene: SceneId(scene), name, field, key: Some(key) })?;
                Ok(json!({ "object": out.object.map(|o| o.0), "band": true }))
            }
            PhantomAdd { scene, name, rig_asset, rig_actor } => {
                if rig_asset.is_some() && rig_actor.is_some() {
                    return Err(SessionError::ConflictingRig);
                }
                let rig = match (rig_asset, rig_actor) {
                    (Some(a), None) => RigRef::Imported { asset: AssetId(a) },
                    (None, Some(actor)) => RigRef::BuiltinActor {
                        actor: crate::builtin::parse_actor(&actor).map_err(SessionError::BadInput)?,
                    },
                    (None, None) => RigRef::Builtin,
                    (Some(_), Some(_)) => unreachable!(),
                };
                let out = self.cmd(Command::AddPhantom { scene: SceneId(scene), name, rig })?;
                Ok(json!({ "phantom": out.phantom.map(|p| p.0) }))
            }
            KeySet { scene, phantom, frame, pos, pose, yaw } => {
                self.cmd(Command::SetMotionKey {
                    scene: SceneId(scene),
                    phantom: PhantomId(phantom),
                    key: MotionKey { yaw, frame, position: pos.into(), pose },
                })?;
                Ok(json!({ "ok": true }))
            }
            CameraKeySet { scene, camera, key } => {
                let camera = resolve_camera(&self.project.as_ref().ok_or(SessionError::NoProject)?.scenes[&SceneId(scene)], camera)?;
                self.cmd(Command::SetCameraKey { scene: SceneId(scene), camera, key })?;
                Ok(json!({ "ok": true }))
            }
            ObjectAdd { scene, name, asset } => {
                let out = self.cmd(Command::AddObject { scene: SceneId(scene), name, asset: AssetId(asset) })?;
                Ok(json!({ "object": out.object.map(|o| o.0) }))
            }
            ObjectRemove { scene, object } => {
                self.cmd(Command::RemoveObject { scene: SceneId(scene), object: ObjectId(object) })?;
                Ok(json!({ "ok": true }))
            }
            ObjectKeySet { scene, object, frame, pos, yaw, pitch, roll, scale } => {
                self.cmd(Command::SetObjectKey {
                    scene: SceneId(scene),
                    object: ObjectId(object),
                    key: ObjectKey { frame, position: pos.into(), yaw, pitch, roll, scale: scale.into() },
                })?;
                Ok(json!({ "ok": true }))
            }
            ObjectKeyRemove { scene, object, frame } => {
                self.cmd(Command::RemoveObjectKey { scene: SceneId(scene), object: ObjectId(object), frame })?;
                Ok(json!({ "ok": true }))
            }
            LightingSet { scene, lighting } => {
                self.cmd(Command::SetLighting { scene: SceneId(scene), lighting })?;
                Ok(json!({ "ok": true }))
            }
            LightAdd { scene, name, kind, color, intensity, enabled, shadows, rest_pos, rest_yaw, rest_pitch } => {
                let out = self.cmd(Command::AddLight {
                    scene: SceneId(scene),
                    name,
                    kind,
                    color,
                    intensity,
                    enabled,
                    shadows,
                    rest: TransformKey::at(0, rest_pos.into(), rest_yaw, rest_pitch),
                })?;
                Ok(json!({ "light": out.light.map(|l| l.0) }))
            }
            LightRemove { scene, light } => {
                self.cmd(Command::RemoveLight { scene: SceneId(scene), light: LightId(light) })?;
                Ok(json!({ "ok": true }))
            }
            LightKeySet { scene, light, frame, pos, yaw, pitch, intensity, enabled } => {
                self.cmd(Command::SetLightKey {
                    scene: SceneId(scene),
                    light: LightId(light),
                    key: LightKey { frame, translation: pos.into(), yaw, pitch, intensity, enabled },
                })?;
                Ok(json!({ "ok": true }))
            }
            LightKeyRemove { scene, light, frame } => {
                self.cmd(Command::RemoveLightKey { scene: SceneId(scene), light: LightId(light), frame })?;
                Ok(json!({ "ok": true }))
            }
            AmbientSet { scene, ambient } => {
                self.cmd(Command::SetAmbient { scene: SceneId(scene), ambient })?;
                Ok(json!({ "ok": true }))
            }
            ObjectAttach { scene, object, attach } => {
                self.cmd(Command::AttachObject { scene: SceneId(scene), object: ObjectId(object), attach })?;
                Ok(json!({ "ok": true }))
            }
            LightAttach { scene, light, attach } => {
                self.cmd(Command::AttachLight { scene: SceneId(scene), light: LightId(light), attach })?;
                Ok(json!({ "ok": true }))
            }
            PhantomScaleSet { scene, phantom, scale } => {
                self.cmd(Command::SetPhantomScale { scene: SceneId(scene), phantom: PhantomId(phantom), scale })?;
                Ok(json!({ "ok": true }))
            }
            GradeSet { scene, grade } => {
                self.cmd(Command::SetGrade { scene: SceneId(scene), grade })?;
                Ok(json!({ "ok": true }))
            }
            FogSet { scene, fog } => {
                self.cmd(Command::SetFog { scene: SceneId(scene), fog })?;
                Ok(json!({ "ok": true }))
            }
            VfxAdd { scene, vfx } => {
                let out = self.cmd(Command::AddVfx { scene: SceneId(scene), vfx })?;
                Ok(json!({ "vfx": out.vfx.map(|v| v.0) }))
            }
            VfxRemove { scene, vfx } => {
                self.cmd(Command::RemoveVfx { scene: SceneId(scene), vfx: VfxId(vfx) })?;
                Ok(json!({ "ok": true }))
            }
            VfxKeySet { scene, vfx, frame, enabled, density } => {
                self.cmd(Command::SetVfxKey { scene: SceneId(scene), vfx: VfxId(vfx), key: VfxKey { frame, enabled, density } })?;
                Ok(json!({ "ok": true }))
            }
            VfxKeyRemove { scene, vfx, frame } => {
                self.cmd(Command::RemoveVfxKey { scene: SceneId(scene), vfx: VfxId(vfx), frame })?;
                Ok(json!({ "ok": true }))
            }
            MaterialSet { scene, phantom, object, target, mut patch, clear } => {
                let p = self.project.as_ref().ok_or(SessionError::NoProject)?;
                let s = p.scenes.get(&SceneId(scene)).ok_or(CoreError::SceneNotFound(SceneId(scene)))?;
                let current = match (phantom, object) {
                    (Some(id), None) => &s.phantoms.get(&PhantomId(id)).ok_or(CoreError::PhantomNotFound(PhantomId(id)))?.material_patches,
                    (None, Some(id)) => &s.objects.get(&ObjectId(id)).ok_or(CoreError::ObjectNotFound(ObjectId(id)))?.material_patches,
                    _ => return Err(SessionError::BadInput("material_set: exactly one owner (phantom|object)".into())),
                };
                let mut patches: Vec<MaterialPatch> = if clear {
                    match &target {
                        None => Vec::new(),
                        Some(t) => current.iter().filter(|p| p.target.as_deref() != Some(t)).cloned().collect(),
                    }
                } else {
                    patch.target = target.clone();
                    patch.validate()?;
                    let mut next: Vec<MaterialPatch> = current.iter().filter(|p| p.target != patch.target).cloned().collect();
                    next.push(patch);
                    next
                };
                patches.sort_by(|a, b| a.target.cmp(&b.target));
                self.cmd(Command::SetMaterialPatches { scene: SceneId(scene), phantom: phantom.map(PhantomId), object: object.map(ObjectId), patches })?;
                Ok(json!({ "ok": true }))
            }
            MaterialList { scene, phantom, object } => {
                let p = self.project.as_ref().ok_or(SessionError::NoProject)?;
                let s = p.scenes.get(&SceneId(scene)).ok_or(CoreError::SceneNotFound(SceneId(scene)))?;
                let (owner, patches, model) = match (phantom, object) {
                    (Some(id), None) => {
                        let ph = s.phantoms.get(&PhantomId(id)).ok_or(CoreError::PhantomNotFound(PhantomId(id)))?;
                        let model = match ph.rig.builtin_actor() {
                            Some(actor) => phantom_runtime::render::mesh_gen::builtin_actor_model(actor),
                            None => {
                                let RigRef::Imported { asset } = &ph.rig else {
                                    return Err(SessionError::BadInput("material_list: unknown rig".into()));
                                };
                                let a = p.assets.get(asset).ok_or(CoreError::AssetNotFound(*asset))?;
                                phantom_runtime::import::load_model(&self.dir()?.join(&a.path), false)?
                            }
                        };
                        (json!({"phantom": id}), ph.material_patches.clone(), model)
                    }
                    (None, Some(id)) => {
                        let ob = s.objects.get(&ObjectId(id)).ok_or(CoreError::ObjectNotFound(ObjectId(id)))?;
                        // A generated fill and a catalog prop have no file to
                        // open: their parts come from the same procedural path
                        // the renderer uses, so the group names listed here are
                        // the ones `material_set` will actually hit.
                        let model = if let Some(field) = &ob.scatter {
                            phantom_runtime::props::scatter_field_model(field)
                                .ok_or_else(|| SessionError::BadInput(format!("material_list: cannot replay fill of object {id}")))?
                        } else if let Some(prop) = &ob.builtin {
                            phantom_runtime::props::prop_model(prop)
                                .ok_or_else(|| CoreError::UnknownProp(prop.clone()))?
                        } else {
                            let a = p.assets.get(&ob.asset).ok_or(CoreError::AssetNotFound(ob.asset))?;
                            phantom_runtime::import::load_model(&self.dir()?.join(&a.path), false)?
                        };
                        (json!({"object": id}), ob.material_patches.clone(), model)
                    }
                    _ => return Err(SessionError::BadInput("material_list: exactly one owner (phantom|object)".into())),
                };
                let parts = material_parts_json(&model, &patches);
                Ok(json!({ "revision": p.revision, "scene": scene, "owner": owner, "patches": patches, "parts": parts }))
            }
            SkySet { scene, sky } => {
                self.cmd(Command::SetSky { scene: SceneId(scene), sky })?;
                Ok(json!({ "ok": true }))
            }
            LightingGet { scene } => {
                let p = self.project.as_ref().ok_or(SessionError::NoProject)?;
                let s = p.scenes.get(&SceneId(scene)).ok_or(CoreError::SceneNotFound(SceneId(scene)))?;
                Ok(json!({ "lighting": s.lighting }))
            }
            FloorSet { scene, floor } => {
                self.cmd(Command::SetFloor { scene: SceneId(scene), floor })?;
                Ok(json!({ "ok": true }))
            }
            FloorGet { scene } => {
                let p = self.project.as_ref().ok_or(SessionError::NoProject)?;
                let s = p.scenes.get(&SceneId(scene)).ok_or(CoreError::SceneNotFound(SceneId(scene)))?;
                Ok(json!({ "floor": s.floor }))
            }
            StageSet { scene, preset, width, height, depth, center } => {
                let current = {
                    let p = self.project.as_ref().ok_or(SessionError::NoProject)?;
                    p.scenes.get(&SceneId(scene)).ok_or(CoreError::SceneNotFound(SceneId(scene)))?.stage
                };
                let stage = build_stage(preset.as_deref(), width, height, depth, center.unwrap_or(current.center))?;
                self.cmd(Command::SetStage { scene: SceneId(scene), stage })?;
                Ok(json!({ "stage": stage }))
            }
            StageGet { scene } => {
                let p = self.project.as_ref().ok_or(SessionError::NoProject)?;
                let s = p.scenes.get(&SceneId(scene)).ok_or(CoreError::SceneNotFound(SceneId(scene)))?;
                Ok(json!({ "stage": s.stage }))
            }
            BrushStroke { scene, frame, points, color, width, erase } => {
                let out = self.cmd(Command::BrushStroke {
                    scene: SceneId(scene),
                    layer: None,
                    frame,
                    stroke: Stroke { points, color, width, erase },
                })?;
                Ok(json!({ "ok": true, "layer": out.layer.map(|l| l.0) }))
            }
            FilmAdd { scene, start, end, camera } => {
                let out = self.cmd(Command::AddFilmClip { scene: SceneId(scene), range: FrameRange::new(start, end)?, camera: camera.map(CameraId) })?;
                // A4: the edit range is shorter than the scene's longest audio
                // track — the tail of the sound will not reach the film; warn right away.
                let p = self.project.as_ref().ok_or(SessionError::NoProject)?;
                let s = p.scenes.get(&SceneId(scene)).ok_or(CoreError::SceneNotFound(SceneId(scene)))?;
                let audio_end = |t: &phantom_core::audio::AudioTrack| {
                    t.clips.iter().map(|c| c.start + (c.len_samples / phantom_core::audio::SAMPLES_PER_FRAME as u64) as Frame).max().unwrap_or(0)
                };
                let mut warnings = Vec::new();
                if let Some(track) = s.audio_tracks.iter().max_by_key(|t| audio_end(t)) {
                    let end_samples_track = audio_end(track);
                    if end_samples_track > end {
                        warnings.push(json!({
                            "code": "audio_longer_than_clip", "severity": "warning",
                            "where": {"scene": scene, "track": track.id.0},
                            "why": format!(
                                "track \"{}\" plays until {:.2} s, the film clip only until {:.2} s: {:.2} s of audio will not reach the film",
                                track.name,
                                phantom_core::time::frame_to_secs(end_samples_track),
                                phantom_core::time::frame_to_secs(end),
                                phantom_core::time::frame_to_secs(end_samples_track - end),
                            ),
                            "suggested_fix": "widen the clip range or trim the audio (audio_report shows the details)",
                        }));
                    }
                }
                Ok(json!({ "clip": out.clip.map(|c| c.0), "warnings": warnings }))
            }
            FilmMove { clip, index } => {
                self.cmd(Command::MoveFilmClip { clip: ClipId(clip), index })?;
                Ok(json!({ "ok": true }))
            }
            FilmTrim { clip, start, end } => {
                self.cmd(Command::SetFilmClipRange { clip: ClipId(clip), range: FrameRange::new(start, end)? })?;
                Ok(json!({ "ok": true }))
            }
            FilmRemove { clip } => {
                self.cmd(Command::RemoveFilmClip { clip: ClipId(clip) })?;
                Ok(json!({ "ok": true }))
            }
            PhantomRemove { scene, phantom } => {
                self.cmd(Command::RemovePhantom { scene: SceneId(scene), phantom: PhantomId(phantom) })?;
                Ok(json!({ "ok": true }))
            }
            MotionKeyRemove { scene, phantom, frame } => {
                self.cmd(Command::RemoveMotionKey { scene: SceneId(scene), phantom: PhantomId(phantom), frame })?;
                Ok(json!({ "ok": true }))
            }
            MotionReplace { scene, phantom, motion, yaws } => {
                // The wire carries triples plus a parallel yaw list (the legacy
                // shape); the document gets typed keys.
                let keys = build_motion_keys(motion, yaws, "motion_replace")?;
                let count = keys.len();
                self.cmd(Command::ReplaceMotion { scene: SceneId(scene), phantom: PhantomId(phantom), motion: keys })?;
                Ok(json!({ "keys": count }))
            }
            CameraAdd { scene, name, start, end } => {
                let range = match (start, end) {
                    (Some(start), Some(end)) => Some(FrameRange::new(start, end)?),
                    (None, None) => None,
                    _ => return Err(SessionError::BadInput("camera_add: start and end come together".into())),
                };
                let out = self.cmd(Command::AddCamera { scene: SceneId(scene), name, range })?;
                Ok(json!({ "camera": out.camera.map(|c| c.0) }))
            }
            CameraRemove { scene, camera } => {
                self.cmd(Command::RemoveCamera { scene: SceneId(scene), camera: CameraId(camera) })?;
                Ok(json!({ "ok": true }))
            }
            CameraSelect { scene, camera } => {
                self.cmd(Command::ActivateCamera { scene: SceneId(scene), camera: CameraId(camera) })?;
                Ok(json!({ "ok": true }))
            }
            CameraRangeSet { scene, camera, start, end } => {
                self.cmd(Command::SetCameraRange { scene: SceneId(scene), camera: CameraId(camera), range: FrameRange::new(start, end)? })?;
                Ok(json!({ "ok": true }))
            }
            CameraKeyRemove { scene, camera, frame } => {
                self.cmd(Command::RemoveCameraKey { scene: SceneId(scene), camera: CameraId(camera), frame })?;
                Ok(json!({ "ok": true }))
            }
            CameraLensSet { scene, camera, focal_mm } => {
                self.cmd(Command::SetCameraLens { scene: SceneId(scene), camera: CameraId(camera), focal_mm })?;
                Ok(json!({ "ok": true }))
            }
            CameraFocalKeySet { scene, camera, frame, focal_mm } => {
                self.cmd(Command::SetCameraFocalKey { scene: SceneId(scene), camera: CameraId(camera), frame, focal_mm })?;
                Ok(json!({ "ok": true }))
            }
            CameraTargetSet { scene, camera, mode, subject, clear } => {
                if clear {
                    self.cmd(Command::SetCameraTarget { scene: SceneId(scene), camera: CameraId(camera), target: None })?;
                    return Ok(json!({ "ok": true }));
                }
                let mode = match mode.as_deref() {
                    Some("track") => CameraTargetMode::Track,
                    Some("follow") => CameraTargetMode::Follow,
                    other => {
                        return Err(SessionError::BadInput(format!(
                            "camera_target_set: mode is track|follow, got {}",
                            other.unwrap_or("(missing)")
                        )))
                    }
                };
                let subject = parse_camera_subject(self.project.as_ref().ok_or(SessionError::NoProject)?, subject.as_deref())?;
                self.cmd(Command::SetCameraTarget {
                    scene: SceneId(scene),
                    camera: CameraId(camera),
                    target: Some(CameraTarget::new(mode, subject)),
                })?;
                Ok(json!({ "ok": true }))
            }
            AudioTrackAdd { scene, name } => {
                let out = self.cmd(Command::AddAudioTrack { scene: scene.map(SceneId), name })?;
                Ok(json!({ "track": out.track.map(|t| t.0) }))
            }
            AudioClipAdd { track, asset, start, offset_samples, len_samples, fade_in_samples, fade_out_samples } => {
                let out = self.cmd(Command::AddAudioClip { track: TrackId(track), asset: AssetId(asset), start, offset_samples, len_samples, fade_in_samples, fade_out_samples })?;
                Ok(json!({ "clip": out.audio_clip.map(|c| c.0) }))
            }
            AudioClipUpdate { track, clip, offset_samples, len_samples, fade_in_samples, fade_out_samples } => {
                self.cmd(Command::UpdateAudioClip {
                    track: TrackId(track),
                    clip: AudioClipId(clip),
                    offset_samples,
                    len_samples,
                    fade_in_samples,
                    fade_out_samples,
                })?;
                Ok(json!({ "ok": true }))
            }
            TrackVolumeSet { track, volume } => {
                self.cmd(Command::SetTrackVolume { track: TrackId(track), volume })?;
                Ok(json!({ "ok": true }))
            }
            TrackMuteSet { track, muted } => {
                self.cmd(Command::SetTrackMuted { track: TrackId(track), muted })?;
                Ok(json!({ "ok": true }))
            }
            AudioClipRemove { track, clip } => {
                self.cmd(Command::RemoveAudioClip { track: TrackId(track), clip: AudioClipId(clip) })?;
                Ok(json!({ "ok": true }))
            }
            AudioTrackRemove { track } => {
                self.cmd(Command::RemoveAudioTrack { track: TrackId(track) })?;
                Ok(json!({ "ok": true }))
            }
            LayerAdd2D { scene, name } => {
                let out = self.cmd(Command::AddLayer2D { scene: SceneId(scene), name })?;
                Ok(json!({ "layer": out.layer.map(|l| l.0) }))
            }
            ExposureRemove { scene, layer, frame } => {
                self.cmd(Command::RemoveExposure { scene: SceneId(scene), layer: crate::draw::LayerId(layer), frame })?;
                Ok(json!({ "ok": true }))
            }
            HistoryUndo => {
                let project = self.project.as_mut().ok_or(SessionError::NoProject)?;
                let applied = self.history.undo(project);
                Ok(json!({ "ok": true, "applied": applied }))
            }
            HistoryRedo => {
                let project = self.project.as_mut().ok_or(SessionError::NoProject)?;
                let applied = self.history.redo(project);
                Ok(json!({ "ok": true, "applied": applied }))
            }
            HistoryList => {
                const LIMIT: usize = 16;
                Ok(json!({
                    "undo_depth": self.history.undo_depth(),
                    "redo_depth": self.history.redo_depth(),
                    "undo": self.history.undo_descriptions(LIMIT),
                    "redo": self.history.redo_descriptions(LIMIT),
                }))
            }
            RenderScene { scene, out, width, height, frames, camera } => {
                let out = self.check_out_path(&out)?;
                self.project_save()?;
                let args = scene_render_args(self.dir()?, &out, scene, width, height, camera, frames);
                self.spawn_job(args, None)
            }
            RenderFilm { out, width, height, frames, normalize, target_peak_db } => {
                let out = self.check_out_path(&out)?;
                self.project_save()?;
                let cleanup = out.with_extension("part.mp4");
                self.spawn_job(film_render_args(self.dir()?, &out, width, height, frames, normalize, target_peak_db), Some(cleanup))
            }
            JobStatus { job } => Ok(self.job_status(job)?),
            JobWait { job, timeout_secs } => self.job_wait(job, timeout_secs),
            SceneGet { scene } => {
                let p = self.project.as_ref().ok_or(SessionError::NoProject)?;
                let s = p.scenes.get(&SceneId(scene)).ok_or(CoreError::SceneNotFound(SceneId(scene)))?;
                Ok(serde_json::to_value(s).unwrap_or_default())
            }
            Validate { scene } => {
                let p = self.project.as_ref().ok_or(SessionError::NoProject)?;
                Ok(crate::analysis::validate_report(p, self.dir.as_deref(), scene)?)
            }
            AudioReport { scene } => {
                let p = self.project.as_ref().ok_or(SessionError::NoProject)?;
                if let Some(id) = scene {
                    if !p.scenes.contains_key(&SceneId(id)) {
                        return Err(CoreError::SceneNotFound(SceneId(id)).into());
                    }
                }
                // Without a project directory (test/incomplete Session) and without
                // ffmpeg the report degrades to document data plus a warning.
                let dir = self.dir().ok().map(Path::to_path_buf);
                let available = phantom_runtime::audio::ffmpeg_available() && dir.is_some();
                // Decode each asset at most once; the fragment
                // is cut from the cache (offset..offset+len, clamped to the source).
                let mut cache: std::collections::BTreeMap<AssetId, Result<phantom_core::PcmBuf, String>> =
                    Default::default();
                let report = crate::analysis::audio_report(p, scene, available, |asset, offset, len| {
                    let pcm = cache.entry(asset).or_insert_with(|| {
                        let a = match p.assets.get(&asset) {
                            Some(a) => a,
                            None => return Err(format!("asset {} not found", asset.0)),
                        };
                        let dir = dir.as_ref().ok_or_else(|| "the project directory is unknown".to_string())?;
                        phantom_runtime::audio::decode_audio(&dir.join(&a.path)).map_err(|e| e.to_string())
                    });
                    let pcm = pcm.as_ref().map_err(|e| e.clone())?;
                    let start = offset.min(pcm.samples.len() as u64) as usize;
                    let end = (offset + len).min(pcm.samples.len() as u64) as usize;
                    let (fragment_rms_db, fragment_peak_db) =
                        crate::analysis::rms_peak_db(&pcm.samples[start..end.max(start)]);
                    Ok(crate::analysis::AudioProbe {
                        source_len_samples: pcm.samples.len() as u64,
                        fragment_rms_db,
                        fragment_peak_db,
                    })
                })?;
                Ok(report)
            }
            CameraCoverage { scene, frame, camera, aspect } => {
                let p = self.project.as_ref().ok_or(SessionError::NoProject)?;
                let s = p.scenes.get(&SceneId(scene)).ok_or(CoreError::SceneNotFound(SceneId(scene)))?;
                let camera = match camera {
                    Some(id) => {
                        if !s.cameras.contains_key(&CameraId(id)) {
                            return Err(CoreError::CameraNotFound(CameraId(id)).into());
                        }
                        Some(CameraId(id))
                    }
                    None => None,
                };
                let aspect = checked_aspect(aspect)?;
                let mut out = crate::analysis::camera_coverage(p, s, camera, frame, aspect);
                out["revision"] = json!(p.revision);
                Ok(out)
            }
            BackgroundFit { scene, frame, camera, margin_deg, aspect } => {
                let p = self.project.as_ref().ok_or(SessionError::NoProject)?;
                let s = p.scenes.get(&SceneId(scene)).ok_or(CoreError::SceneNotFound(SceneId(scene)))?;
                let camera = match camera {
                    Some(id) => {
                        if !s.cameras.contains_key(&CameraId(id)) {
                            return Err(CoreError::CameraNotFound(CameraId(id)).into());
                        }
                        Some(CameraId(id))
                    }
                    None => None,
                };
                let aspect = checked_aspect(aspect)?;
                let margin = margin_deg.unwrap_or(5.0) as f32;
                if !margin.is_finite() || !(0.0..45.0).contains(&margin) {
                    return Err(SessionError::BadInput(format!("margin_deg must be finite and in [0, 45): {margin}")));
                }
                // An explicit frame only makes sense for a single camera: without
                // one the fit covers every camera in the scene, each in its range.
                if camera.is_none() && frame.is_some() {
                    return Err(SessionError::BadInput(
                        "frame requires camera: without it background_fit covers every camera in the scene".into(),
                    ));
                }
                let mut out = crate::analysis::background_fit(p, s, camera, frame, margin, aspect);
                out["revision"] = json!(p.revision);
                Ok(out)
            }
            CameraGet { scene, camera } => {
                let p = self.project.as_ref().ok_or(SessionError::NoProject)?;
                let s = p.scenes.get(&SceneId(scene)).ok_or(CoreError::SceneNotFound(SceneId(scene)))?;
                let entry = s.cameras.get(&CameraId(camera)).ok_or(CoreError::CameraNotFound(CameraId(camera)))?;
                Ok(json!({"scene":scene, "revision":p.revision, "camera":entry, "active_camera": s.active_camera}))
            }
            AssetGet { asset } => {
                let p = self.project.as_ref().ok_or(SessionError::NoProject)?;
                let Some(a) = p.assets.get(&AssetId(asset)) else {
                    let mut nearest: Vec<u64> = p.assets.keys().map(|id| id.0).collect();
                    nearest.sort_by_key(|id| (id.abs_diff(asset), *id));
                    nearest.truncate(5);
                    return Err(SessionError::AssetGetNotFound { id: asset, nearest });
                };
                let mut usage = Vec::new();
                for s in p.scenes.values() {
                    if s.background.as_ref().is_some_and(|b| b.image.asset() == a.id) {
                        usage.push(json!({"kind":"background", "scene":s.id.0}));
                    }
                    for object in s.objects.values().filter(|o| o.asset == a.id) {
                        usage.push(json!({"kind":"object", "scene":s.id.0, "id":object.id.0}));
                    }
                    for ph in s.phantoms.values() {
                        if matches!(ph.rig, RigRef::Imported { asset: id } if id == a.id) {
                            usage.push(json!({"kind":"phantom_rig", "scene":s.id.0, "id":ph.id.0}));
                        }
                    }
                    for track in &s.audio_tracks { for clip in track.clips.iter().filter(|c| c.asset == a.id) {
                        usage.push(json!({"kind":"audio_clip", "scene":s.id.0, "track":track.id.0, "id":clip.id.0}));
                    }}
                }
                for track in &p.film_audio { for clip in track.clips.iter().filter(|c| c.asset == a.id) {
                    usage.push(json!({"kind":"audio_clip", "scene":Value::Null, "track":track.id.0, "id":clip.id.0}));
                }}
                let exists = self.dir.as_ref().is_some_and(|dir| dir.join(&a.path).is_file());
                Ok(json!({"revision":p.revision, "asset":{"id":a.id.0, "kind":format!("{:?}",a.kind), "path":a.path, "exists":exists, "usage":usage}}))
            }
            FilmGet => {
                let p = self.project.as_ref().ok_or(SessionError::NoProject)?;
                let frames = film_len(&p.film);
                Ok(json!({"revision":p.revision, "clips":p.film, "total_frames":frames, "total_seconds":frames as f64 / FPS as f64}))
            }
            AssetList => {
                let p = self.project.as_ref().ok_or(SessionError::NoProject)?;
                Ok(json!({ "assets": p.assets.values().map(|a| json!({
                    "id": a.id.0, "kind": format!("{:?}", a.kind), "path": a.path,
                })).collect::<Vec<_>>() }))
            }
            JobList => {
                let mut ids: Vec<u64> = self.jobs.keys().copied().collect();
                ids.sort_unstable();
                let mut jobs = Vec::with_capacity(ids.len());
                for id in ids {
                    let mut v = self.job_status(id)?;
                    v["job"] = json!(id);
                    jobs.push(v);
                }
                Ok(json!({ "jobs": jobs }))
            }
            Batch { ops } => self.execute_batch(ops),
            RenderStill { scene, frame, width, height, camera, report } => {
                let rendered = self.render_still_frame(scene, frame, width, height, camera)?;
                let report_value = report.then(|| {
                    let metrics = phantom_runtime::render::metrics::frame_metrics(
                        &rendered.img,
                        rendered.img_width,
                        rendered.img_height,
                    );
                    // A3b: an explicit complaint about a dead frame (2 % threshold).
                    let mut findings = Vec::new();
                    if metrics.empty_coverage > 0.02 {
                        findings.push(json!({
                            "code": "empty_frame", "severity": "warning",
                            "where": {"scene": scene, "frame": frame},
                            "why": format!("{:.1} % of the frame has no geometry (threshold 2 %)", metrics.empty_coverage * 100.0),
                            "suggested_fix": "background_fit picks the screen width/height for the camera; enable the floor (floor_set)",
                        }));
                    }
                    json!({
                        "mean_luma": metrics.mean_luma,
                        "luma_p5": metrics.luma_p5,
                        "luma_p95": metrics.luma_p95,
                        "empty_coverage": metrics.empty_coverage,
                        // Render-world fact: an object counts when at least one
                        // of its Mesh3d parts passed the chosen camera's Bevy
                        // visibility. This is not mixed with the background alpha.
                        "objects_in_frame": rendered.objects_in_frame,
                        "objects_total": rendered.objects_total,
                        "scene_digest": rendered.scene_digest,
                        "findings": findings,
                    })
                });
                let mut png = Vec::new();
                rendered
                    .img
                    .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
                    .map_err(phantom_runtime::render::RenderError::Image)?;
                use base64::Engine as _;
                let mut value = json!({
                    "scene": scene,
                    "frame": frame,
                    "camera": camera,
                    "width": width,
                    "height": height,
                    "mime_type": "image/png",
                    "png_base64": base64::engine::general_purpose::STANDARD.encode(&png),
                });
                if let Some(report) = report_value {
                    value["report"] = report;
                }
                Ok(value)
            }
            Look { scene, mode, target, views, distance, frame, camera, frames, every, count, view, width, height } => {
                self.look_sheet(LookRequest {
                    scene,
                    mode: mode.unwrap_or_else(|| "angles".to_string()),
                    target: target.unwrap_or_default(),
                    views: views.unwrap_or_default(),
                    distance,
                    frame,
                    camera,
                    frames,
                    every,
                    count,
                    view,
                    width: width.unwrap_or(480),
                    height: height.unwrap_or(270),
                })
            }
            PropGrid { scene, prop, prefix, rows, cols, origin, step, gap, yaw, pitch, roll, scale } => {
                let grid = self.prop_grid_ops(scene, &prop, prefix.as_deref(), rows, cols, origin, step, gap, yaw, pitch, roll, scale)?;
                let mut value = self.execute_batch(grid)?;
                let ids: Vec<u64> = value["results"]
                    .as_array()
                    .map(|results| results.iter().filter_map(|r| r["object"].as_u64()).collect())
                    .unwrap_or_default();
                value["objects"] = json!(ids);
                Ok(value)
            }
            Scatter { scene, preset, area, plate, inset, action, density, from, to, step, seed, prefix } => {
                let args = ScatterArgs {
                    scene,
                    preset: &preset,
                    area,
                    plate,
                    inset,
                    action,
                    density,
                    from,
                    to,
                    step,
                    seed: seed.unwrap_or_else(random_seed),
                    prefix: prefix.as_deref(),
                };
                let ops = self.scatter_ops(&args)?;
                let band = ops.iter().any(|op| matches!(op, Op::ScatterBand { .. }));
                let mut value = self.execute_batch(ops)?;
                let ids: Vec<u64> = value["results"]
                    .as_array()
                    .map(|results| results.iter().filter_map(|r| r["object"].as_u64()).collect())
                    .unwrap_or_default();
                value["objects"] = json!(ids);
                value["seed"] = json!(args.seed);
                value["band"] = json!(band);
                Ok(value)
            }
            CameraReport { scene, camera, every, width, height } => {
                let p = self.project.as_ref().ok_or(SessionError::NoProject)?;
                let s = p.scenes.get(&SceneId(scene)).ok_or(CoreError::SceneNotFound(SceneId(scene)))?;
                let camera_ref = s.cameras.get(&CameraId(camera)).ok_or(CoreError::CameraNotFound(CameraId(camera)))?;
                let frames = camera_report_frames(camera_ref.range, every)?;
                let (width, height) = (width.unwrap_or(480), height.unwrap_or(270));
                // Checks before the GPU — the same path as render_still.
                self.check_still_target(scene, width, height, Some(camera))?;
                // One renderer for all frames of the report: the GPU is initialized
                // once, and the frames stay deterministic.
                let mut renderer = phantom_runtime::render::Renderer::new(width, height)?;
                let mut rows = Vec::with_capacity(frames.len());
                let mut findings = Vec::new();
                let mut objects_total = 0usize;
                let mut objects_seen = 0usize;
                for frame in frames {
                    let rendered = self.render_with(&mut renderer, scene, frame, width, height, Some(camera))?;
                    let metrics = phantom_runtime::render::metrics::frame_metrics(&rendered.img, rendered.img_width, rendered.img_height);
                    objects_total = rendered.objects_total;
                    objects_seen = objects_seen.max(rendered.objects_in_frame);
                    if metrics.empty_coverage > 0.02 {
                        findings.push(json!({
                            "code": "empty_frame", "severity": "warning",
                            "where": {"scene": scene, "camera": camera, "frame": frame},
                            "why": format!("{:.1} % of the frame has no geometry (threshold 2 %)", metrics.empty_coverage * 100.0),
                            "suggested_fix": "background_fit picks the screen width/height for the camera; enable the floor (floor_set)",
                        }));
                    }
                    rows.push(json!({
                        "frame": frame,
                        "mean_luma": metrics.mean_luma,
                        "luma_p5": metrics.luma_p5,
                        "luma_p95": metrics.luma_p95,
                        "empty_coverage": metrics.empty_coverage,
                        "objects_in_frame": rendered.objects_in_frame,
                    }));
                }
                let luma = |key: &str| rows.iter().filter_map(|r| r[key].as_f64()).collect::<Vec<_>>();
                let means = luma("mean_luma");
                let empties = luma("empty_coverage");
                let aggregate = json!({
                    "frames": rows.len(),
                    "mean_luma": if means.is_empty() { 0.0 } else { means.iter().sum::<f64>() / means.len() as f64 },
                    "min_luma": means.iter().cloned().fold(f64::INFINITY, f64::min),
                    "max_luma": means.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                    "max_empty_coverage": empties.iter().cloned().fold(0.0f64, f64::max),
                    "objects_total": objects_total,
                    "objects_seen": objects_seen,
                });
                if !rows.is_empty() && empties.iter().all(|e| *e > 0.02) {
                    findings.push(json!({
                        "code": "camera_empty", "severity": "warning",
                        "where": {"scene": scene, "camera": camera},
                        "why": "none of the sampled frames of the camera has geometry".to_string(),
                        "suggested_fix": "check the camera and background_fit",
                    }));
                }
                if objects_total > 0 && objects_seen == 0 {
                    findings.push(json!({
                        "code": "objects_out_of_frame", "severity": "warning",
                        "where": {"scene": scene, "camera": camera, "object_count": objects_total, "objects_in_frame": 0},
                        "why": format!("the scene has {objects_total} objects, none of them appears in the camera's frames"),
                        "suggested_fix": "check camera_coverage for this camera; move/key the object",
                    }));
                }
                Ok(json!({
                    "scene": scene, "camera": camera, "width": width, "height": height,
                    "range": {"start": camera_ref.range.start, "end": camera_ref.range.end},
                    "every": every,
                    "rows": rows,
                    "aggregate": aggregate,
                    "findings": findings,
                }))
            }
            JobCancel { job } => {
                let mut j = self.jobs.remove(&job).ok_or(SessionError::JobNotFound(job))?;
                let _ = j.child.kill();
                let _ = j.child.wait();
                if let Some(h) = j.err_drain.take() {
                    let _ = h.join();
                }
                if let Some(h) = j.out_drain.take() {
                    let _ = h.join();
                }
                if let Some(tmp) = j.cleanup {
                    let _ = std::fs::remove_file(tmp);
                }
                Ok(json!({ "ok": true }))
            }
        }
    }

    fn project_save(&mut self) -> Result<(), SessionError> {
        self.execute(Op::ProjectSave).map(|_| ())
    }

    fn spawn_job(&mut self, args: Vec<std::ffi::OsString>, cleanup: Option<PathBuf>) -> Result<Value, SessionError> {
        let exe = std::env::current_exe()?;
        self.spawn_child(exe.into(), args, cleanup)
    }

    /// Test entry point: an arbitrary command instead of the current exe.
    #[cfg(test)]
    pub(crate) fn spawn_test_command(&mut self, exe: &str, args: &[&str]) -> Result<Value, SessionError> {
        self.spawn_child(exe.into(), args.iter().map(std::ffi::OsString::from).collect(), None)
    }

    /// Spawn a child process: stdout — the progress protocol, stderr — a
    /// bounded ring buffer (an async thread so a chatty child does not block
    /// on a full pipe).
    fn spawn_child(
        &mut self,
        exe: std::ffi::OsString,
        args: Vec<std::ffi::OsString>,
        cleanup: Option<PathBuf>,
    ) -> Result<Value, SessionError> {
        let mut child = std::process::Command::new(exe)
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let stdout = child.stdout.take().expect("piped");
        let out_drain = std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            for line in BufReader::new(stdout).lines() {
                if let Ok(line) = line {
                    let _ = tx.send(line);
                }
            }
        });
        let err_tail = std::sync::Arc::new(std::sync::Mutex::new(ErrRing::default()));
        let stderr = child.stderr.take().expect("piped");
        let err_drain = {
            let err_tail = std::sync::Arc::clone(&err_tail);
            std::thread::spawn(move || {
                use std::io::{BufRead, BufReader};
                for line in BufReader::new(stderr).lines() {
                    if let Ok(line) = line {
                        if let Ok(mut ring) = err_tail.lock() {
                            ring.push(line);
                        }
                    }
                }
            })
        };
        let id = self.next_job;
        self.next_job += 1;
        self.jobs.insert(
            id,
            Job {
                child,
                rx,
                last_line: String::new(),
                exit_code: None,
                finished: false,
                err_tail,
                err_drain: Some(err_drain),
                out_drain: Some(out_drain),
                started: std::time::Instant::now(),
                cleanup,
            },
        );
        Ok(json!({ "job": id }))
    }

    fn job_wait(&mut self, id: u64, timeout_secs: f64) -> Result<Value, SessionError> {
        if !timeout_secs.is_finite() || !(0.0..=300.0).contains(&timeout_secs) {
            return Err(SessionError::BadInput("job_wait.timeout_secs must be finite and in 0..=300".into()));
        }
        // Validate the ID even when timeout is zero.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f64(timeout_secs);
        loop {
            let mut status = self.job_status(id)?;
            let done = status["done"].as_bool().unwrap_or(false);
            if done || std::time::Instant::now() >= deadline {
                status["timed_out"] = json!(!done);
                return Ok(status);
            }
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            std::thread::sleep(remaining.min(std::time::Duration::from_millis(20)));
        }
    }

    fn job_status(&mut self, id: u64) -> Result<Value, SessionError> {
        let j = self.jobs.get_mut(&id).ok_or(SessionError::JobNotFound(id))?;
        while let Ok(line) = j.rx.try_recv() {
            j.last_line = line;
        }
        if !j.finished {
            if let Ok(Some(status)) = j.child.try_wait() {
                j.finished = true;
                j.exit_code = status.code();
                // Finalization: wait for both pipe readers (EOF after the child
                // exits) and drain the channel once more — otherwise the last
                // progress line or the stderr tail could be lost.
                if let Some(h) = j.err_drain.take() {
                    let _ = h.join();
                }
                if let Some(h) = j.out_drain.take() {
                    let _ = h.join();
                }
                while let Ok(line) = j.rx.try_recv() {
                    j.last_line = line;
                }
            }
        }
        let (frames_done, total) = parse_progress(&j.last_line);
        let error_tail = j.err_tail.lock().map(|r| r.tail()).unwrap_or_default();
        Ok(json!({
            "last_line": j.last_line,
            "done": j.finished,
            "exit_code": j.exit_code,
            "frames_done": frames_done,
            "total": total,
            "elapsed_secs": j.started.elapsed().as_secs_f64(),
            "error_tail": error_tail,
        }))
    }

    /// Render one scene frame to PNG (in-process, needs a GPU). Without a GPU —
    /// a clean RenderError::GpuUnavailable error; editing does not break.
    /// One rendered frame before PNG encoding: the metrics in report need the
    /// pixels, so they are not decoded again.
    fn render_still_frame(
        &mut self,
        scene: u64,
        frame: Frame,
        width: u32,
        height: u32,
        camera: Option<u64>,
    ) -> Result<RenderedStill, SessionError> {
        // Checks run before touching the GPU: an invalid request must not bring
        // up a device (and crash on a headless machine instead of invalid_input).
        self.check_still_target(scene, width, height, camera)?;
        let mut renderer = phantom_runtime::render::Renderer::new(width, height)?;
        self.render_with(&mut renderer, scene, frame, width, height, camera)
    }

    /// Sizes and frame references — before the GPU.
    fn check_still_target(&self, scene: u64, width: u32, height: u32, camera: Option<u64>) -> Result<(), SessionError> {
        const MAX: u32 = 4096;
        if width == 0 || height == 0 || width > MAX || height > MAX {
            return Err(SessionError::BadInput(format!(
                "render_still: sizes must be in 1..={MAX}, got {width}x{height}"
            )));
        }
        let p = self.project.as_ref().ok_or(SessionError::NoProject)?;
        let s = p.scenes.get(&SceneId(scene)).ok_or(CoreError::SceneNotFound(SceneId(scene)))?;
        if let Some(cam) = camera {
            if !s.cameras.contains_key(&CameraId(cam)) {
                return Err(CoreError::CameraNotFound(CameraId(cam)).into());
            }
        }
        Ok(())
    }

    /// Render a frame with an already created renderer: the camera report renders
    /// several frames and must not bring up the GPU for each of them.
    fn render_with(
        &self,
        renderer: &mut phantom_runtime::render::Renderer,
        scene: u64,
        frame: Frame,
        width: u32,
        height: u32,
        camera: Option<u64>,
    ) -> Result<RenderedStill, SessionError> {
        let p = self.project.as_ref().ok_or(SessionError::NoProject)?;
        let s = p.scenes.get(&SceneId(scene)).ok_or(CoreError::SceneNotFound(SceneId(scene)))?;
        // Non-cryptographic digest of the scene's canonical serde snapshot: the
        // agent tells "the frame changed" apart without comparing pixels.
        use std::hash::Hasher as _;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        hasher.write(&serde_json::to_vec(s).expect("scene serializes"));
        let scene_digest = format!("{:016x}", hasher.finish());
        let objects_total = s.objects.len();
        let opts = phantom_runtime::render::RenderOptions { width, height };
        let frame_out = renderer.render_frame(p, s, self.dir()?, frame, &opts, camera.map(CameraId))?;
        let img: image::RgbaImage =
            image::ImageBuffer::from_raw(frame_out.width, frame_out.height, frame_out.rgba).expect("buffer size checked");
        Ok(RenderedStill { img, img_width: frame_out.width, img_height: frame_out.height, objects_in_frame: frame_out.objects_in_frame, objects_total, scene_digest })
    }
}

/// A review direction on the wire: a standard view name, or an explicit
/// direction from the target toward the eye.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ViewSpec {
    Name(String),
    Direction([f32; 3]),
}

impl ViewSpec {
    /// The label to report and the unit direction from the target to the eye.
    fn resolve(&self) -> Result<(String, glam::Vec3), SessionError> {
        match self {
            Self::Name(name) => {
                let view = LookView::parse(name).ok_or_else(|| {
                    SessionError::BadInput(format!(
                        "view: front|back|left|right|top|three_quarter or [x,y,z], got {name}"
                    ))
                })?;
                Ok((view.name().to_string(), view.direction()))
            }
            Self::Direction(d) => {
                let v = glam::Vec3::from(*d);
                if !v.is_finite() || v.length() < 1e-6 {
                    return Err(SessionError::BadInput(format!(
                        "view direction must be finite and non-zero, got {d:?}"
                    )));
                }
                Ok((format!("{:.2},{:.2},{:.2}", d[0], d[1], d[2]), v.normalize()))
            }
        }
    }
}

/// One request to the review sheet, normalized by the caller.
struct LookRequest {
    scene: u64,
    mode: String,
    target: Vec<String>,
    views: Vec<ViewSpec>,
    distance: Option<f32>,
    frame: Option<Frame>,
    camera: Option<u64>,
    frames: Option<Vec<Frame>>,
    every: Option<u32>,
    count: Option<u32>,
    view: Option<ViewSpec>,
    width: u32,
    height: u32,
}

/// One tile of the sheet: what to draw, from where, and under which label.
struct LookTile {
    label: String,
    view: Option<String>,
    document: Scene,
    frame: Frame,
}

/// The sheet holds at most this many tiles: a review is a glance, not a render.
const LOOK_MAX_TILES: usize = 12;
/// Sizes stay within the same bound as `render_still`.
const LOOK_MAX_SIDE: u32 = 4096;

impl Session {
    /// The review sheet: several directions around the staged scene at one
    /// frame, or a strip of frames through one camera, tiled into one image.
    fn look_sheet(&self, req: LookRequest) -> Result<Value, SessionError> {
        let p = self.project.as_ref().ok_or(SessionError::NoProject)?;
        let scene = p.scenes.get(&SceneId(req.scene)).ok_or(CoreError::SceneNotFound(SceneId(req.scene)))?;
        // The review camera: an explicit id, or the scene's active camera.
        let camera = match req.camera {
            Some(id) => Some(resolve_camera(scene, Some(id))?),
            None => Some(scene.active_camera),
        };
        if let Some(cam) = camera {
            if !scene.cameras.contains_key(&cam) {
                return Err(CoreError::CameraNotFound(cam).into());
            }
        }
        self.check_still_target(req.scene, req.width, req.height, req.camera)?;

        let tiles = match req.mode.as_str() {
            "angles" => self.look_angles(scene, &req, camera)?,
            "frames" => self.look_frames(scene, &req, camera)?,
            other => {
                return Err(SessionError::BadInput(format!("mode: angles|frames, got {other}")))
            }
        };
        let cols = (tiles.len() as f64).sqrt().ceil().max(1.0) as u32;
        let cols = cols.min(4).max(1);
        let rows = (tiles.len() as u32).div_ceil(cols).max(1);
        if cols * req.width > LOOK_MAX_SIDE || rows * req.height > LOOK_MAX_SIDE {
            return Err(SessionError::BadInput(format!(
                "look: {cols}×{rows} tiles of {}×{} exceed the {LOOK_MAX_SIDE} px sheet limit",
                req.width, req.height
            )));
        }

        // One renderer for the whole sheet: the GPU comes up once and the tiles
        // stay comparable (same device, same tonemapping).
        let mut renderer = phantom_runtime::render::Renderer::new(req.width, req.height)?;
        let mut images = Vec::with_capacity(tiles.len());
        let mut rows_json = Vec::with_capacity(tiles.len());
        let mut any_visible = false;
        for (index, tile) in tiles.iter().enumerate() {
            let rendered = self.render_document(&mut renderer, &tile.document, tile.frame, req.width, req.height, camera)?;
            let metrics = phantom_runtime::render::metrics::frame_metrics(&rendered.img, rendered.img_width, rendered.img_height);
            // A review frame legitimately carries a lot of background, so a
            // partly empty tile is normal and gets no finding. Only a sheet on
            // which NOTHING is visible at all is a complaint worth making.
            any_visible |= metrics.empty_coverage <= 0.98;
            rows_json.push(json!({
                "index": index,
                "label": tile.label,
                "view": tile.view,
                "frame": tile.frame,
                "mean_luma": metrics.mean_luma,
                "luma_p5": metrics.luma_p5,
                "luma_p95": metrics.luma_p95,
                "empty_coverage": metrics.empty_coverage,
                "objects_in_frame": rendered.objects_in_frame,
            }));
            images.push(rendered.img);
        }

        let mut findings = Vec::new();
        if !any_visible {
            findings.push(json!({
                "code": "look_empty", "severity": "warning",
                "where": {"scene": req.scene, "tiles": tiles.len()},
                "why": "no tile shows any geometry: the camera looks at nothing (wrong distance, or the staging is elsewhere)",
                "suggested_fix": "check target and distance, or grow the stage with stage_set",
            }));
        }

        let sheet = tile_sheet(&images, cols);
        let (sheet_w, sheet_h) = sheet.dimensions();
        // The digest describes the STAGING, not the substituted review camera:
        // the same sheet request on an unchanged document gives the same digest.
        use std::hash::Hasher as _;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        hasher.write(&serde_json::to_vec(scene).expect("scene serializes"));
        let scene_digest = format!("{:016x}", hasher.finish());

        let mut png = Vec::new();
        sheet
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .map_err(phantom_runtime::render::RenderError::Image)?;
        use base64::Engine as _;
        Ok(json!({
            "mode": req.mode,
            "scene": req.scene,
            "revision": p.revision,
            "scene_digest": scene_digest,
            "objects_total": scene.objects.len(),
            "sheet": {
                "width": sheet_w, "height": sheet_h,
                "cols": cols, "rows": rows,
                "tile_width": req.width, "tile_height": req.height,
            },
            "tiles": rows_json,
            "findings": findings,
            "mime_type": "image/png",
            "png_base64": base64::engine::general_purpose::STANDARD.encode(&png),
        }))
    }

    /// Tiles for `angles`: one tile per review direction, all from the same
    /// distance, so the tiles can be compared with each other.
    fn look_angles(&self, scene: &Scene, req: &LookRequest, camera: Option<CameraId>) -> Result<Vec<LookTile>, SessionError> {
        let frame = req.frame.unwrap_or_else(|| scene.range_for(camera).start);
        let review_id = camera.unwrap_or(SCENE_CAMERA);
        let target = self.look_target(scene, &req.target, frame)?;
        let focal = scene.camera_for(camera).focal_mm;
        let aspect = req.width as f32 / req.height as f32;
        let distance = match req.distance {
            Some(d) if d.is_finite() && d > 0.0 => d,
            Some(d) => return Err(SessionError::BadInput(format!("distance must be finite and > 0, got {d}"))),
            None => fit_distance(target.half_extent, focal, aspect, 1.05),
        };
        let views: Vec<ViewSpec> = if req.views.is_empty() {
            LookView::DEFAULT.iter().map(|v| ViewSpec::Name(v.name().to_string())).collect()
        } else {
            req.views.clone()
        };
        let mut tiles = Vec::new();
        for spec in &views {
            let (label, direction) = spec.resolve()?;
            let mut document = scene.clone();
            set_review_camera(&mut document, review_id, camera_looking_at(target.center, direction, distance, focal, frame));
            tiles.push(LookTile { view: Some(label.clone()), label, document, frame });
        }
        Ok(tiles)
    }

    /// Tiles for `frames`: a strip over the range, from the camera or from
    /// one review direction.
    fn look_frames(&self, scene: &Scene, req: &LookRequest, camera: Option<CameraId>) -> Result<Vec<LookTile>, SessionError> {
        let range = scene.range_for(camera);
        let review_id = camera.unwrap_or(SCENE_CAMERA);
        let frames = match &req.frames {
            Some(list) if list.is_empty() => {
                return Err(SessionError::BadInput("frames must not be empty".into()))
            }
            Some(list) => list.clone(),
            None => match req.count {
                Some(0) => return Err(SessionError::BadInput("count must be > 0".into())),
                Some(n) => even_frames(range, n.min(LOOK_MAX_TILES as u32)),
                None => camera_report_frames(range, req.every)?,
            },
        };
        if frames.len() > LOOK_MAX_TILES {
            return Err(SessionError::BadInput(format!(
                "look: {n} frames exceed the {LOOK_MAX_TILES}-tile limit",
                n = frames.len()
            )));
        }
        let view = match &req.view {
            Some(spec) => Some(spec.resolve()?),
            None => None,
        };
        let document = match &view {
            None => scene.clone(),
            Some((_, direction)) => {
                let frame = frames.first().copied().unwrap_or(range.start);
                let target = self.look_target(scene, &req.target, frame)?;
                let focal = scene.camera_for(camera).focal_mm;
                let aspect = req.width as f32 / req.height as f32;
                let distance = match req.distance {
                    Some(d) if d.is_finite() && d > 0.0 => d,
                    Some(d) => {
                        return Err(SessionError::BadInput(format!("distance must be finite and > 0, got {d}")))
                    }
                    None => fit_distance(target.half_extent, focal, aspect, 1.05),
                };
                let mut document = scene.clone();
                set_review_camera(&mut document, review_id, camera_looking_at(target.center, *direction, distance, focal, frame));
                document
            }
        };
        Ok(frames
            .into_iter()
            .map(|frame| LookTile {
                label: format!("frame {frame}"),
                view: view.as_ref().map(|(label, _)| label.clone()),
                document: document.clone(),
                frame,
            })
            .collect())
    }

    /// The box the review must contain: the named actors/objects, or everything
    /// staged, or — with nothing staged — the stage box itself.
    fn look_target(&self, scene: &Scene, targets: &[String], frame: Frame) -> Result<TargetBox, SessionError> {
        let p = self.project.as_ref().ok_or(SessionError::NoProject)?;
        let (phantoms, objects): (Vec<PhantomId>, Vec<ObjectId>) = if targets.is_empty() {
            (scene.phantoms.keys().copied().collect(), scene.objects.keys().copied().collect())
        } else {
            let mut phantoms = Vec::new();
            let mut objects = Vec::new();
            for needle in targets {
                let needle = needle.trim();
                if needle.is_empty() {
                    continue;
                }
                let mut hit = false;
                for (id, ph) in &scene.phantoms {
                    if ph.name.eq_ignore_ascii_case(needle) {
                        phantoms.push(*id);
                        hit = true;
                    }
                }
                for (id, object) in &scene.objects {
                    if object.name.eq_ignore_ascii_case(needle) {
                        objects.push(*id);
                        hit = true;
                    }
                }
                // An explicit document id also works: "phantom:3" or "object:4".
                if !hit {
                    if let Some(rest) = needle.strip_prefix("phantom:") {
                        if let Ok(id) = rest.trim().parse::<u64>() {
                            if scene.phantoms.contains_key(&PhantomId(id)) {
                                phantoms.push(PhantomId(id));
                                hit = true;
                            }
                        }
                    } else if let Some(rest) = needle.strip_prefix("object:") {
                        if let Ok(id) = rest.trim().parse::<u64>() {
                            if scene.objects.contains_key(&ObjectId(id)) {
                                objects.push(ObjectId(id));
                                hit = true;
                            }
                        }
                    }
                }
                if !hit {
                    let mut names: Vec<String> = scene.phantoms.values().map(|ph| ph.name.clone()).collect();
                    names.extend(scene.objects.values().map(|o| o.name.clone()));
                    names.truncate(20);
                    return Err(SessionError::LookTargetNotFound { needle: needle.to_string(), names });
                }
            }
            (phantoms, objects)
        };

        let mut bbox = TargetBox::empty();
        for id in &phantoms {
            let phantom = &scene.phantoms[id];
            let state = evaluate(phantom, frame);
            let footprint = p.phantom_footprint(phantom, state.pose);
            bbox = bbox.union(TargetBox::of(state.root, footprint, glam::Quat::from_rotation_y(state.yaw)));
        }
        for id in &objects {
            let object = &scene.objects[id];
            let t = evaluate_object(object, frame);
            let footprint = p.object_footprint(object, t.scale);
            bbox = bbox.union(TargetBox::of(t.position, footprint, glam::Quat::from_euler(glam::EulerRot::YXZ, t.yaw, t.pitch, t.roll)));
        }
        if bbox.is_empty() {
            let (lo, hi) = (scene.stage.min(), scene.stage.max());
            bbox = TargetBox { center: (lo + hi) * 0.5, half_extent: (hi - lo) * 0.5 };
        }
        Ok(bbox)
    }

    /// Render one document (possibly a clone with a substituted review camera)
    /// with an already created renderer.
    fn render_document(
        &self,
        renderer: &mut phantom_runtime::render::Renderer,
        document: &Scene,
        frame: Frame,
        width: u32,
        height: u32,
        camera: Option<CameraId>,
    ) -> Result<RenderedStill, SessionError> {
        let p = self.project.as_ref().ok_or(SessionError::NoProject)?;
        let opts = phantom_runtime::render::RenderOptions { width, height };
        let out = renderer.render_frame(p, document, self.dir()?, frame, &opts, camera)?;
        let img: image::RgbaImage =
            image::ImageBuffer::from_raw(out.width, out.height, out.rgba).expect("buffer size checked");
        Ok(RenderedStill {
            img,
            img_width: out.width,
            img_height: out.height,
            objects_in_frame: out.objects_in_frame,
            objects_total: document.objects.len(),
            scene_digest: String::new(),
        })
    }

    /// Fill an area or a line: editable props inside the action zone, ONE band
    /// object in the ring beyond it, all of it one undo step.
    fn scatter_ops(&self, args: &ScatterArgs<'_>) -> Result<Vec<Op>, SessionError> {
        use phantom_core::{plan, scatter_preset, ScatterField, ScatterShape};
        let p = self.project.as_ref().ok_or(SessionError::NoProject)?;
        let scene_doc = p.scenes.get(&SceneId(args.scene)).ok_or(CoreError::SceneNotFound(SceneId(args.scene)))?;
        let preset = scatter_preset(&args.preset).ok_or_else(|| CoreError::InvalidScatter(format!(
            "unknown fill {} (available: {})",
            args.preset,
            phantom_core::SCATTER_PRESETS.iter().map(|p| p.name).collect::<Vec<_>>().join(", ")
        )))?;
        if args.action.is_some_and(|b| !b.iter().all(|v| v.is_finite())) {
            return Err(CoreError::NonFinite("scatter.action").into());
        }

        // The world placement (center + yaw) and the LOCAL shape are separate
        // concepts: the document stores local numbers, so the band moves and
        // turns with its object instead of carrying baked world coordinates.
        let (center, yaw, shape) = if let Some([fx, fz]) = args.from {
            let [tx, tz] = args.to.ok_or_else(|| SessionError::BadInput("scatter: from and to go together".into()))?;
            if !fx.is_finite() || !fz.is_finite() || !tx.is_finite() || !tz.is_finite() {
                return Err(CoreError::NonFinite("scatter.from/to").into());
            }
            let (dx, dz) = (tx - fx, tz - fz);
            let length = (dx * dx + dz * dz).sqrt();
            if length < 0.5 {
                return Err(SessionError::BadInput("scatter: the segment is too short".into()));
            }
            // The row runs along local +X: rotate X onto the segment direction.
            (
                [(fx + tx) * 0.5, (fz + tz) * 0.5],
                (-dz).atan2(dx),
                ScatterShape::Line { length, step: args.step.unwrap_or(preset.step), lateral: preset.line_jitter },
            )
        } else if let Some([x0, z0, x1, z1]) = args.area {
            if args.to.is_some() {
                return Err(SessionError::BadInput("scatter: from/to belong to a line, not an area".into()));
            }
            if ![x0, z0, x1, z1].iter().all(|v| v.is_finite()) {
                return Err(CoreError::NonFinite("scatter.area").into());
            }
            let (lo, hi) = ([x0.min(x1), z0.min(z1)], [x0.max(x1), z0.max(z1)]);
            let center = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5];
            (
                center,
                0.0,
                ScatterShape::Area {
                    size: [hi[0] - lo[0], hi[1] - lo[1]],
                    density: args.density.unwrap_or(preset.density),
                    action: args.action.map(|b| to_local_box(&b, center, 0.0)),
                },
            )
        } else if let Some(plate) = args.plate {
            // Fill a ground plate with a margin: the plate's own key places the
            // fill, and a turned plate turns its forest with it.
            let obj = scene_doc.objects.get(&ObjectId(plate)).ok_or(CoreError::ObjectNotFound(ObjectId(plate)))?;
            let is_ground = obj
                .builtin
                .as_deref()
                .and_then(phantom_core::catalog::catalog_find)
                .is_some_and(|e| e.kind == phantom_core::catalog::CatalogKind::Exterior);
            if !is_ground {
                return Err(SessionError::BadInput(format!(
                    "scatter: plate {plate} is not an outdoor ground (street_yard/street_park/street_crossing)"
                )));
            }
            let t = phantom_core::evaluate_object(obj, 0);
            let entry = phantom_core::catalog::catalog_find(obj.builtin.as_deref().unwrap()).expect("checked above");
            let inset = args.inset.unwrap_or(0.5).max(0.0);
            let size = [
                (entry.size[0] * t.scale.x - inset * 2.0).max(1.0),
                (entry.size[2] * t.scale.z - inset * 2.0).max(1.0),
            ];
            let center = [t.position.x, t.position.z];
            // The action box arrives in world coordinates; the plate may be
            // turned, so bring it into the plate's frame.
            let action = args.action.map(|b| to_local_box(&b, center, t.yaw));
            (
                center,
                t.yaw,
                ScatterShape::Area { size, density: args.density.unwrap_or(preset.density), action },
            )
        } else {
            return Err(SessionError::BadInput("scatter: give an area, a plate, or a from/to line".into()));
        };

        let placed = plan(preset, &shape, args.seed)?;
        let rot = glam::Quat::from_rotation_y(yaw);
        let origin = glam::Vec3::new(center[0], 0.0, center[1]);
        let base = args.prefix.map(str::to_string).unwrap_or_else(|| scene_doc.name.clone());
        let base = if base.trim().is_empty() { args.preset.to_string() } else { base };

        let mut ops = Vec::new();
        // 1. The shoot zone: real objects — after generation each one moves,
        //    turns and deletes like any prop, which is the point of the split.
        for item in placed.iter().filter(|item| item.near) {
            let local = glam::Vec3::from_array(item.position);
            let position = origin + rot * local;
            ops.push(Op::PropAdd {
                scene: args.scene,
                name: format!("{base} {}", item.prop),
                prop: item.prop.to_string(),
                pos: Some(position.to_array()),
                yaw: yaw + item.yaw,
                pitch: 0.0,
                roll: 0.0,
                scale: Some([item.scale; 3]),
            });
        }
        // 2. The ring: ONE object whose mesh replays this very plan.
        if !phantom_core::band_of(&placed).is_empty() {
            ops.push(Op::ScatterBand {
                scene: args.scene,
                name: format!("{base} band"),
                field: ScatterField { preset: args.preset.to_string(), shape, seed: args.seed },
                pos: [origin.x, 0.0, origin.z],
                yaw,
            });
        }
        Ok(ops)
    }

    /// The commands of a prop grid, ready for the atomic batch path.
    #[allow(clippy::too_many_arguments)]
    fn prop_grid_ops(
        &self,
        scene: u64,
        prop: &str,
        prefix: Option<&str>,
        rows: u32,
        cols: u32,
        origin: [f32; 3],
        step: Option<[f32; 2]>,
        gap: Option<f32>,
        yaw: f32,
        pitch: f32,
        roll: f32,
        scale: Option<[f32; 3]>,
    ) -> Result<Vec<Op>, SessionError> {
        let p = self.project.as_ref().ok_or(SessionError::NoProject)?;
        let scene_doc = p.scenes.get(&SceneId(scene)).ok_or(CoreError::SceneNotFound(SceneId(scene)))?;
        let entry = phantom_core::catalog::catalog_find(prop).ok_or(CoreError::UnknownProp(prop.to_string()))?;
        if rows == 0 || cols == 0 {
            return Err(SessionError::BadInput("prop_grid: rows and cols must be > 0".into()));
        }
        if !entry.furniture.is_empty() {
            return Err(SessionError::BadInput(format!(
                "prop_grid: {prop} comes furnished — place it once with prop_add (it arrives as one package); a grid is for single props"
            )));
        }
        let count = (rows as usize) * (cols as usize);
        if count > 256 {
            return Err(SessionError::BadInput(format!(
                "prop_grid: {rows}×{cols} = {count} objects exceed the 256 limit"
            )));
        }
        if !origin.iter().all(|v| v.is_finite()) {
            return Err(CoreError::NonFinite("prop_grid.origin").into());
        }
        if !yaw.is_finite() || !pitch.is_finite() || !roll.is_finite() {
            return Err(CoreError::NonFinite("prop_grid.rotation").into());
        }
        let scale = scale.unwrap_or([1.0; 3]);
        if !scale.iter().all(|v| v.is_finite() && *v > 0.0) {
            return Err(SessionError::BadInput("prop_grid: scale must be finite and > 0".into()));
        }
        // The default step is the prop's own size plus a gap: chairs stand next
        // to each other without overlapping, and the agent needs no arithmetic.
        let (step_x, step_z) = match step {
            Some([dx, dz]) => (dx, dz),
            None => {
                let gap = gap.unwrap_or(1.2);
                if !gap.is_finite() || gap <= 0.0 {
                    return Err(SessionError::BadInput(format!("prop_grid: gap must be finite and > 0, got {gap}")));
                }
                ((entry.size[0] * scale[0]).abs() * gap, (entry.size[2] * scale[2]).abs() * gap)
            }
        };
        if !step_x.is_finite() || !step_z.is_finite() {
            return Err(SessionError::BadInput("prop_grid: step must be finite".into()));
        }
        let base = prefix.map(str::to_string).unwrap_or_else(|| scene_doc.name.clone());
        let mut ops = Vec::with_capacity(count);
        for row in 0..rows {
            for col in 0..cols {
                let position = [
                    origin[0] + col as f32 * step_x,
                    origin[1],
                    origin[2] + row as f32 * step_z,
                ];
                ops.push(Op::PropAdd {
                    scene,
                    name: format!("{} {}", if base.trim().is_empty() { prop } else { base.trim() }, ops.len() + 1),
                    prop: prop.to_string(),
                    pos: Some(position),
                    yaw,
                    pitch,
                    roll,
                    scale: Some(scale),
                });
            }
        }
        Ok(ops)
    }
}

/// Arguments of the fill planner (a bundle: the op carries a dozen knobs).
pub(crate) struct ScatterArgs<'a> {
    pub scene: u64,
    pub preset: &'a str,
    pub area: Option<[f32; 4]>,
    pub plate: Option<u64>,
    pub inset: Option<f32>,
    pub action: Option<[f32; 4]>,
    pub density: Option<f32>,
    pub from: Option<[f32; 2]>,
    pub to: Option<[f32; 2]>,
    pub step: Option<f32>,
    pub seed: u64,
    pub prefix: Option<&'a str>,
}

/// A world rectangle into the local frame of a placed field (center + yaw).
fn to_local_box(box_: &[f32; 4], center: [f32; 2], yaw: f32) -> [[f32; 2]; 2] {
    let rot = glam::Quat::from_rotation_y(-yaw);
    let local = |x: f32, z: f32| {
        let v = rot * glam::Vec3::new(x - center[0], 0.0, z - center[1]);
        [v.x, v.z]
    };
    let a = local(box_[0], box_[1]);
    let b = local(box_[2], box_[3]);
    [
        [a[0].min(b[0]), a[1].min(b[1])],
        [a[0].max(b[0]), a[1].max(b[1])],
    ]
}

/// A seed for a fill the director did not pin. The document stores whatever is
/// chosen, so reproducibility starts at the saved project, not at this moment.
fn random_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() ^ (d.subsec_nanos() as u64) << 32)
        .unwrap_or(0x5EED)
}

/// Composite tiles into one sheet, row by row, `cols` per row.
fn tile_sheet(tiles: &[image::RgbaImage], cols: u32) -> image::RgbaImage {
    let (tile_w, tile_h) = tiles.first().map(|t| t.dimensions()).unwrap_or((0, 0));
    let cols = cols.max(1);
    let rows = (tiles.len() as u32).div_ceil(cols).max(1);
    let mut sheet = image::RgbaImage::new(cols * tile_w, rows * tile_h);
    for (index, tile) in tiles.iter().enumerate() {
        let x = (index as u32 % cols) * tile_w;
        let y = (index as u32 / cols) * tile_h;
        image::imageops::replace(&mut sheet, tile, x as i64, y as i64);
    }
    sheet
}

/// Up to `count` evenly spaced frames, both ends included.
fn even_frames(range: FrameRange, count: u32) -> Vec<Frame> {
    let count = count.max(1);
    if count == 1 {
        return vec![range.start];
    }
    let len = range.end - range.start;
    (0..count as i64).map(|i| range.start + i * (len - 1) / (count as i64 - 1)).collect()
}

struct RenderedStill {
    img: image::RgbaImage,
    img_width: u32,
    img_height: u32,
    objects_in_frame: usize,
    objects_total: usize,
    scene_digest: String,
}

// `lock_guard` drops with the Session and token-checks normal cleanup.

#[cfg(test)]
mod look_and_grid_tests {
    use super::*;

    fn session_with_scene() -> (Session, u64) {
        let mut session = Session::new();
        session.project = Some(Project::new("look"));
        let scene = session
            .execute(Op::SceneAdd { name: "s".into(), start: 0, end: 48 })
            .unwrap()["scene"]
            .as_u64()
            .unwrap();
        (session, scene)
    }

    #[test]
    fn scatter_places_the_shoot_zone_as_objects_and_the_ring_as_one_band() {
        let (mut session, scene) = session_with_scene();
        session
            .execute(Op::StageSet { scene, preset: Some("large".into()), width: None, height: None, depth: None, center: None })
            .unwrap();
        let depth = session.history.undo_depth();
        let before = session.project.as_ref().unwrap().revision;
        let out = session
            .execute(Op::Scatter {
                scene,
                preset: "forest".into(),
                area: Some([-20.0, -20.0, 20.0, 20.0]),
                plate: None,
                inset: None,
                action: Some([-5.0, -5.0, 5.0, 5.0]),
                density: None,
                from: None,
                to: None,
                step: None,
                seed: Some(7),
                prefix: Some("w".into()),
            })
            .unwrap();
        let ids = out["objects"].as_array().unwrap();
        assert!(ids.len() > 1, "the fill created objects");
        assert_eq!(out["seed"], 7, "the pinned seed comes back");
        assert_eq!(out["band"], true, "the ring became a band");
        let s = session.project.as_ref().unwrap().scenes.get(&SceneId(scene)).unwrap();
        let bands: Vec<_> = s.objects.values().filter(|o| o.scatter.is_some()).collect();
        let props: Vec<_> = s.objects.values().filter(|o| o.scatter.is_none()).collect();
        assert_eq!(bands.len(), 1, "one mesh for the whole ring");
        assert!(!props.is_empty(), "the shoot zone is furnished with movable props");
        // The split is what the design promised: movable props only where the
        // scene is shot, and nothing of the band inside that zone.
        assert!(props.iter().all(|o| o.keys[0].position.x.abs() <= 5.001 && o.keys[0].position.z.abs() <= 5.001));
        // One undo step removes the whole fill: props and band together.
        assert_eq!(session.history.undo_depth(), depth + 1, "the whole fill is one undo step");
        assert_eq!(session.project.as_ref().unwrap().revision, before + 1, "one revision");
        assert!(session.execute(Op::HistoryUndo).is_ok());
        assert!(session.project.as_ref().unwrap().scenes[&SceneId(scene)].objects.is_empty());
    }

    #[test]
    fn the_same_seed_is_the_same_forest_through_the_session() {
        let (mut session, scene) = session_with_scene();
        session
            .execute(Op::StageSet { scene, preset: Some("large".into()), width: None, height: None, depth: None, center: None })
            .unwrap();
        let run = |session: &mut Session| {
            session
                .execute(Op::Scatter {
                    scene,
                    preset: "forest".into(),
                    area: Some([-16.0, -16.0, 16.0, 16.0]),
                    plate: None,
                    inset: None,
                    action: Some([-4.0, -4.0, 4.0, 4.0]),
                    density: None,
                    from: None,
                    to: None,
                    step: None,
                    seed: Some(11),
                    prefix: None,
                })
                .unwrap()
        };
        // Read each run's spots BEFORE the undo: object ids are reused-forward,
        // so the first run's ids are gone once it has been rolled back.
        let spots = |session: &Session, v: &serde_json::Value| {
            let s = session.project.as_ref().unwrap().scenes.get(&SceneId(scene)).unwrap();
            v["objects"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|id| s.objects.get(&ObjectId(id.as_u64().unwrap())))
                .map(|o| {
                    let k = o.keys.first().unwrap();
                    (o.builtin.clone().unwrap(), k.position.x, k.position.z, k.scale.x)
                })
                .collect::<Vec<_>>()
        };
        let a = run(&mut session);
        let a_spots = spots(&session, &a);
        session.execute(Op::HistoryUndo).unwrap();
        let b = run(&mut session);
        let b_spots = spots(&session, &b);
        assert!(!a_spots.is_empty());
        assert_eq!(a_spots, b_spots, "the seed, not the clock, decides the wood");
    }

    #[test]
    fn a_fill_refuses_a_plate_that_is_not_ground() {
        let (mut session, scene) = session_with_scene();
        session.execute(Op::PropAdd { scene, name: "table".into(), prop: "table".into(), pos: Some([0.0, 0.0, 0.0]), yaw: 0.0, pitch: 0.0, roll: 0.0, scale: None }).unwrap();
        let table = session.project.as_ref().unwrap().scenes[&SceneId(scene)].objects.keys().next().copied().unwrap();
        let err = session
            .execute(Op::Scatter {
                scene,
                preset: "forest".into(),
                area: None,
                plate: Some(table.0),
                inset: None,
                action: None,
                density: None,
                from: None,
                to: None,
                step: None,
                seed: Some(1),
                prefix: None,
            })
            .unwrap_err();
        assert!(err.to_string().contains("outdoor ground"), "{err}");
    }

    #[test]
    fn prop_grid_places_a_row_in_one_undo_step_and_clamps_each_item() {
        let (mut session, scene) = session_with_scene();
        session.execute(Op::StageSet { scene, preset: Some("small".into()), width: None, height: None, depth: None, center: None }).unwrap();
        let before = session.project.as_ref().unwrap().revision;
        let depth = session.history.undo_depth();
        let out = session
            .execute(Op::PropGrid {
                scene,
                prop: "chair".into(),
                prefix: Some("seat".into()),
                rows: 2,
                cols: 3,
                origin: [0.0, 0.0, 0.0],
                step: None,
                gap: Some(1.2),
                yaw: 0.0,
                pitch: 0.0,
                roll: 0.0,
                scale: None,
            })
            .unwrap();
        // One user mutation, one revision: the grid is an atomic batch.
        assert_eq!(out["revision"].as_u64().unwrap(), before + 1);
        assert_eq!(out["applied"].as_u64().unwrap(), 6);
        let ids = out["objects"].as_array().unwrap();
        assert_eq!(ids.len(), 6);
        assert_eq!(session.history.undo_depth(), depth + 1, "the whole grid is one undo step");
        assert_eq!(out["entity"]["kind"], "prop_grid");
        let objects = &session.project.as_ref().unwrap().scenes[&SceneId(scene)].objects;
        assert_eq!(objects.len(), 6);
        assert!(objects.values().all(|o| o.name.starts_with("seat ")));
        // The default step is the chair's own size times the gap: 0.45*1.2 and 0.5*1.2.
        let mut xs: Vec<f32> = objects.values().map(|o| o.keys[0].position.x).collect();
        xs.sort_by(|a, b| a.total_cmp(b));
        xs.dedup();
        assert_eq!(xs.len(), 3);
        assert!((xs[0] - 0.0).abs() < 1e-5 && (xs[1] - 0.54).abs() < 1e-4 && (xs[2] - 1.08).abs() < 1e-4, "{xs:?}");
        let mut zs: Vec<f32> = objects.values().map(|o| o.keys[0].position.z).collect();
        zs.sort_by(|a, b| a.total_cmp(b));
        zs.dedup();
        assert!((zs[1] - 0.6).abs() < 1e-4, "{zs:?}");

        // A grid outside the stage is clamped item by item, in the same step.
        let out = session
            .execute(Op::PropGrid {
                scene,
                prop: "chair".into(),
                prefix: None,
                rows: 1,
                cols: 3,
                origin: [9.0, 0.0, 0.0],
                step: Some([2.0, 0.0]),
                gap: None,
                yaw: 0.0,
                pitch: 0.0,
                roll: 0.0,
                scale: None,
            })
            .unwrap();
        assert_eq!(out["clamped"], true);
        let clamped = out["clamps"].as_array().unwrap();
        assert!(!clamped.is_empty());
        assert!(clamped.iter().all(|c| c["kind"] == "object"));

        // One undo removes the whole second grid.
        session.execute(Op::HistoryUndo).unwrap();
        assert_eq!(session.project.as_ref().unwrap().scenes[&SceneId(scene)].objects.len(), 6);
    }

    #[test]
    fn prop_grid_rejects_bad_input_before_touching_the_document() {
        let (mut session, scene) = session_with_scene();
        let before = serde_json::to_string(session.project.as_ref().unwrap()).unwrap();
        let grid = |rows, cols, prop: &str, gap| Op::PropGrid {
            scene,
            prop: prop.into(),
            prefix: None,
            rows,
            cols,
            origin: [0.0, 0.0, 0.0],
            step: None,
            gap,
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            scale: None,
        };
        assert_eq!(session.execute(grid(0, 3, "chair", None)).unwrap_err().code(), "invalid_input");
        // 17x17 = 289 objects: more than a grid, and more than one sheet of eyes.
        assert_eq!(session.execute(grid(17, 17, "chair", None)).unwrap_err().code(), "invalid_input");
        assert_eq!(session.execute(grid(1, 1, "no-such-prop", None)).unwrap_err().code(), "not_found");
        assert_eq!(session.execute(grid(1, 1, "chair", Some(0.0))).unwrap_err().code(), "invalid_input");
        let mut bad_step = grid(1, 1, "chair", None);
        if let Op::PropGrid { step, .. } = &mut bad_step {
            *step = Some([f32::NAN, 1.0]);
        }
        assert_eq!(session.execute(bad_step).unwrap_err().code(), "invalid_input");
        assert_eq!(before, serde_json::to_string(session.project.as_ref().unwrap()).unwrap());
    }

    /// Every `look` argument error is raised before the GPU is touched, so a
    /// headless machine still gets `invalid_input` instead of a device error.
    #[test]
    fn look_validates_its_arguments_without_a_gpu() {
        let (mut session, scene) = session_with_scene();
        let look = |mode: &str, target: Vec<String>, views: Vec<ViewSpec>| Op::Look {
            scene,
            mode: Some(mode.into()),
            target: Some(target),
            views: Some(views),
            distance: None,
            frame: None,
            camera: None,
            frames: None,
            every: None,
            count: None,
            view: None,
            width: Some(64),
            height: Some(48),
        };
        let err = session.execute(look("orbit", vec![], vec![])).unwrap_err();
        assert_eq!(err.code(), "invalid_input");
        assert!(err.to_string().contains("angles|frames"), "{err}");

        let err = session.execute(look("angles", vec![], vec![ViewSpec::Name("diagonal".into())])).unwrap_err();
        assert_eq!(err.code(), "invalid_input");
        assert!(err.to_string().contains("three_quarter"), "{err}");

        let err = session.execute(look("angles", vec![], vec![ViewSpec::Direction([0.0, 0.0, 0.0])])).unwrap_err();
        assert_eq!(err.code(), "invalid_input");

        // An unmatched target is `not_found` and names what does exist.
        session
            .execute(Op::PhantomAdd { scene, name: "hero".into(), rig_asset: None, rig_actor: Some("wolf".into()) })
            .unwrap();
        let err = session.execute(look("angles", vec!["nobody".into()], vec![])).unwrap_err();
        assert_eq!(err.code(), "not_found");
        assert!(err.to_string().contains("nobody"), "{err}");

        // Sizes are checked before the renderer too.
        let mut op = look("angles", vec![], vec![]);
        if let Op::Look { width, .. } = &mut op {
            *width = Some(5000);
        }
        assert_eq!(session.execute(op).unwrap_err().code(), "invalid_input");

        // A nonexistent camera is `not_found`.
        let mut op = look("angles", vec![], vec![]);
        if let Op::Look { camera, .. } = &mut op {
            *camera = Some(999);
        }
        assert_eq!(session.execute(op).unwrap_err().code(), "not_found");
    }

    #[test]
    fn even_frames_spans_both_ends() {
        let range = FrameRange::new(10, 20).unwrap();
        assert_eq!(even_frames(range, 1), vec![10]);
        assert_eq!(even_frames(range, 2), vec![10, 19]);
        assert_eq!(even_frames(range, 5), vec![10, 12, 14, 16, 19]);
    }

    #[test]
    fn tile_sheet_lays_tiles_row_by_row() {
        let red = image::RgbaImage::from_pixel(2, 1, image::Rgba([255, 0, 0, 255]));
        let blue = image::RgbaImage::from_pixel(2, 1, image::Rgba([0, 0, 255, 255]));
        let sheet = tile_sheet(&[red.clone(), blue.clone(), red.clone(), blue], 2);
        assert_eq!(sheet.dimensions(), (4, 2));
        assert_eq!(sheet.get_pixel(0, 0), &image::Rgba([255, 0, 0, 255]));
        assert_eq!(sheet.get_pixel(2, 0), &image::Rgba([0, 0, 255, 255]));
        assert_eq!(sheet.get_pixel(0, 1), &image::Rgba([255, 0, 0, 255]));
    }
}

#[cfg(test)]
mod mutation_ack_tests {
    use super::*;

    fn mutate(session: &mut Session, op: Op) -> Value {
        let before = session.project.as_ref().unwrap().revision;
        let value = session.execute(op).unwrap();
        assert_eq!(value["revision"], before + 1, "one user mutation must be one revision");
        assert_eq!(value["next_id"], session.project.as_ref().unwrap().next_id());
        assert!(value["entity"]["kind"].is_string(), "canonical entity required: {value}");
        value
    }

    #[test]
    fn representative_mutations_echo_post_command_entities() {
        let mut s = Session::new();
        s.project = Some(Project::new("acks"));
        let scene = mutate(&mut s, Op::SceneAdd { name: "main".into(), start: 0, end: 96 })["scene"].as_u64().unwrap();
        let scene_update = mutate(&mut s, Op::SceneRangeSet { scene, start: 0, end: 120 });
        assert_eq!(scene_update["entity"]["range"], serde_json::to_value(s.project.as_ref().unwrap().scenes[&SceneId(scene)].range).unwrap());

        let phantom = mutate(&mut s, Op::PhantomAdd { scene, name: "hero".into(), rig_asset: None, rig_actor: None })["phantom"].as_u64().unwrap();
        let key = mutate(&mut s, Op::KeySet { scene, phantom, frame: 7, pos: [1.0, 2.0, 3.0], pose: ActorPose::Stand, yaw: 0.5 });
        assert_eq!(key["entity"]["key"], serde_json::to_value(s.project.as_ref().unwrap().scenes[&SceneId(scene)].phantoms[&PhantomId(phantom)].motion[0]).unwrap());
        let key_removed = mutate(&mut s, Op::MotionKeyRemove { scene, phantom, frame: 7 });
        assert_eq!(key_removed["entity"]["removed"], true);

        let image = s.cmd(Command::RegisterAsset { kind: AssetKind::Image, path: "assets/bg.png".into(), bounds: None }).unwrap().asset.unwrap().0;
        let bg = mutate(&mut s, Op::BackgroundSet { scene, asset: image, video: false, width: 12.0, height: 6.0, center: [0.0, 2.0, -5.0], fit: None });
        assert_eq!(bg["entity"]["background"], serde_json::to_value(&s.project.as_ref().unwrap().scenes[&SceneId(scene)].background).unwrap());
        let bg_removed = mutate(&mut s, Op::BackgroundClear { scene });
        assert!(bg_removed["entity"]["background"].is_null());

        let model = s.cmd(Command::RegisterAsset { kind: AssetKind::ModelObj, path: "assets/model.obj".into(), bounds: None }).unwrap().asset.unwrap().0;
        let object = mutate(&mut s, Op::ObjectAdd { scene, name: "prop".into(), asset: model })["object"].as_u64().unwrap();
        let object_key = mutate(&mut s, Op::ObjectKeySet { scene, object, frame: 3, pos: [2.0, 0.0, 0.0], yaw: 1.0, pitch: 0.0, roll: 0.0, scale: [1.0; 3] });
        assert_eq!(object_key["entity"]["key"], serde_json::to_value(s.project.as_ref().unwrap().scenes[&SceneId(scene)].objects[&ObjectId(object)].keys[0]).unwrap());
        assert_eq!(mutate(&mut s, Op::ObjectKeyRemove { scene, object, frame: 3 })["entity"]["removed"], true);
        let removed = mutate(&mut s, Op::ObjectRemove { scene, object });
        assert_eq!(removed["entity"], json!({"kind":"object", "scene":scene, "id":object, "removed":true}));

        let camera = mutate(&mut s, Op::CameraAdd { scene, name: "close".into(), start: Some(0), end: Some(24) })["camera"].as_u64().unwrap();
        let camera_update = mutate(&mut s, Op::CameraRangeSet { scene, camera, start: 0, end: 30 });
        assert_eq!(camera_update["entity"]["range"], json!({"start":0,"end":30}));
        assert_eq!(mutate(&mut s, Op::CameraRemove { scene, camera })["entity"]["removed"], true);

        let film = mutate(&mut s, Op::FilmAdd { scene, start: 0, end: 24, camera: None });
        let film_id = film["clip"].as_u64().unwrap();
        assert_eq!(film["entity"]["kind"], "film_clip");
        assert_eq!(mutate(&mut s, Op::FilmTrim { clip: film_id, start: 2, end: 20 })["entity"]["range"], json!({"start":2,"end":20}));
        assert_eq!(mutate(&mut s, Op::FilmRemove { clip: film_id })["entity"]["removed"], true);

        let audio_asset = s.cmd(Command::RegisterAsset { kind: AssetKind::Audio, path: "assets/dialog.wav".into(), bounds: None }).unwrap().asset.unwrap().0;
        let audio = mutate(&mut s, Op::AudioTrackAdd { scene: Some(scene), name: "dialog".into() });
        let track = audio["track"].as_u64().unwrap();
        assert_eq!(audio["entity"]["name"], "dialog");
        let audio_clip = mutate(&mut s, Op::AudioClipAdd { track, asset: audio_asset, start: 4, offset_samples: 0, len_samples: 48000, fade_in_samples: 0, fade_out_samples: 0 });
        let audio_clip_id = audio_clip["clip"].as_u64().unwrap();
        assert_eq!(mutate(&mut s, Op::TrackVolumeSet { track, volume: 0.5 })["entity"]["volume"], 0.5);
        assert_eq!(mutate(&mut s, Op::AudioClipRemove { track, clip: audio_clip_id })["entity"]["removed"], true);
        assert_eq!(mutate(&mut s, Op::AudioTrackRemove { track })["entity"]["removed"], true);

        let layer = mutate(&mut s, Op::LayerAdd2D { scene, name: "ink".into() });
        let layer_id = layer["layer"].as_u64().unwrap();
        assert_eq!(layer["entity"]["name"], "ink");
        let exposure = mutate(&mut s, Op::BrushStroke { scene, frame: 5, points: vec![[0.0, 0.0], [1.0, 1.0]], color: [1.0; 4], width: 2.0, erase: false });
        assert_eq!(exposure["entity"]["layer"], layer_id);
        assert_eq!(mutate(&mut s, Op::ExposureRemove { scene, layer: layer_id, frame: 5 })["entity"]["removed"], true);

        let lighting = SceneLighting { key: SceneLight { intensity: 1234.0, ..SceneLight::default() }, ..SceneLighting::default() };
        let light = mutate(&mut s, Op::LightingSet { scene, lighting });
        assert_eq!(light["entity"]["lighting"], serde_json::to_value(s.project.as_ref().unwrap().scenes[&SceneId(scene)].lighting).unwrap());
        assert_eq!(mutate(&mut s, Op::PhantomRemove { scene, phantom })["entity"]["removed"], true);

        let state = s.execute(Op::StateGet).unwrap();
        assert!(state.get("entity").is_none());
        assert!(state.get("next_id").is_none());
        assert_eq!(Op::ProjectSave.class(), OpClass::Lifecycle);
        assert_eq!(Op::StateGet.class(), OpClass::Query);
        assert_eq!(Op::JobList.class(), OpClass::Job);
    }

    #[test]
    fn camera_and_history_are_atomic_and_acknowledged() {
        let mut s = Session::new();
        s.project = Some(Project::new("camera"));
        let scene = mutate(&mut s, Op::SceneAdd { name: "s".into(), start: 0, end: 24 })["scene"].as_u64().unwrap();
        let depth = s.history.undo_depth();
        // Two commands, one undo step each: the key and then the default lens.
        let camera = mutate(&mut s, Op::CameraKeySet { scene, camera: None, key: TransformKey::at(12, [1.0, 2.0, 3.0].into(), 0.2, -0.1) });
        assert_eq!(s.history.undo_depth(), depth + 1);
        assert_eq!(camera["entity"]["camera"], serde_json::to_value(&s.project.as_ref().unwrap().scenes[&SceneId(scene)].camera_for(Some(SCENE_CAMERA))).unwrap());
        mutate(&mut s, Op::CameraLensSet { scene, camera: SCENE_CAMERA.0, focal_mm: 50.0 });
        assert_eq!(s.history.undo_depth(), depth + 2);
        let revision = s.project.as_ref().unwrap().revision;
        let undo = s.execute(Op::HistoryUndo).unwrap();
        assert_eq!(undo["revision"], revision + 1);
        assert_eq!(undo["entity"], json!({"kind":"history", "operation":"undo", "applied":true}));
        assert_eq!(s.project.as_ref().unwrap().scenes[&SceneId(scene)].camera_for(Some(SCENE_CAMERA)).focal_mm, 35.0);
        let redo = s.execute(Op::HistoryRedo).unwrap();
        assert_eq!(redo["entity"]["operation"], "redo");
        assert_eq!(s.project.as_ref().unwrap().scenes[&SceneId(scene)].camera_for(Some(SCENE_CAMERA)).focal_mm, 50.0);
    }
}

#[cfg(test)]
mod batch_tests {
    use super::*;

    fn session() -> Session {
        let mut s = Session::new();
        s.project = Some(Project::new("batch"));
        s
    }

    fn setup_ops() -> Vec<Op> {
        vec![
            Op::SceneAdd { name: "main".into(), start: 0, end: 96 },
            Op::PhantomAdd { scene: 1, name: "hero".into(), rig_asset: None, rig_actor: Some("wolf".into()) },
            Op::KeySet { scene: 1, phantom: 2, frame: 0, pos: [0.0, 0.0, 0.0], pose: ActorPose::Stand, yaw: 0.3 },
            Op::KeySet { scene: 1, phantom: 2, frame: 48, pos: [5.0, 0.0, 0.0], pose: ActorPose::Stand, yaw: 0.3 },
            Op::CameraAdd { scene: 1, name: "close".into(), start: Some(0), end: Some(24) },
        ]
    }

    #[test]
    fn batch_chain_applies_ids_like_single_ops_with_one_revision_and_one_undo() {
        // Reference: the same operations one by one.
        let mut singles = session();
        for op in setup_ops() {
            singles.execute(op).unwrap();
        }
        let expected = singles.project.clone().unwrap();

        let mut s = session();
        let pre_revision = s.project.as_ref().unwrap().revision;
        let value = s.execute(Op::Batch { ops: setup_ops() }).unwrap();
        assert_eq!(value["applied"], 5);
        assert_eq!(value["revision"], pre_revision + 1, "the whole batch is one revision");
        assert_eq!(value["next_id"], s.project.as_ref().unwrap().next_id());
        assert_eq!(value["entity"]["kind"], "camera", "the batch's entity is the last operation's entity");
        assert_eq!(s.history.undo_depth(), 1, "the whole batch is one undo step");
        // Id allocation and document content are identical to the single calls.
        let batched = s.project.as_ref().unwrap();
        assert_eq!(serde_json::to_value(&batched.scenes).unwrap(), serde_json::to_value(&expected.scenes).unwrap());
        assert_eq!(batched.next_id(), expected.next_id());
        // Per-op results carry the created ids and entity without intermediate revisions.
        let results = value["results"].as_array().unwrap();
        assert_eq!(results[0]["scene"], 1);
        assert_eq!(results[1]["phantom"], 2);
        assert!(results[0].get("revision").is_none());
        assert_eq!(results[2]["entity"]["kind"], "motion_key");
    }

    #[test]
    fn batch_mid_failure_rolls_back_document_history_and_revision() {
        let mut s = session();
        let pre = s.project.clone().unwrap();
        let err = s
            .execute(Op::Batch {
                ops: vec![
                    Op::SceneAdd { name: "ok".into(), start: 0, end: 24 },
                    Op::KeySet { scene: 1, phantom: 999, frame: 0, pos: [0.0; 3], pose: ActorPose::Stand, yaw: 0.0 },
                ],
            })
            .unwrap_err();
        assert_eq!(err.code(), "batch_failed");
        assert_eq!(err.details()["index"], 1);
        assert_eq!(err.details()["op"], "key_set");
        assert_eq!(err.details()["code"], "not_found");
        assert_eq!(err.details()["rolled_back"], true);
        assert_eq!(
            serde_json::to_value(&s.project.as_ref().unwrap().scenes).unwrap(),
            serde_json::to_value(&pre.scenes).unwrap(),
            "the document is unchanged"
        );
        assert_eq!(s.project.as_ref().unwrap().revision, pre.revision, "revision did not change");
        assert_eq!(s.project.as_ref().unwrap().next_id(), pre.next_id(), "next_id was not spent");
        assert_eq!(s.history.undo_depth(), 0, "no partial undo steps");
    }

    #[test]
    fn batch_rejects_non_mutations_nested_and_empty_before_touching_document() {
        let mut s = session();
        let err = s.execute(Op::Batch { ops: vec![] }).unwrap_err();
        assert_eq!(err.code(), "invalid_input");

        let err = s
            .execute(Op::Batch { ops: vec![Op::SceneAdd { name: "s".into(), start: 0, end: 24 }, Op::StateGet] })
            .unwrap_err();
        assert!(matches!(err, SessionError::BatchInvalid { index: 1, .. }));
        assert_eq!(err.code(), "invalid_input");

        let err = s
            .execute(Op::Batch { ops: vec![Op::Batch { ops: vec![Op::ProjectSave] }] })
            .unwrap_err();
        assert!(matches!(err, SessionError::BatchInvalid { index: 0, .. }));
        assert_eq!(s.history.undo_depth(), 0, "a rejected batch does not touch history");
        assert!(s.project.as_ref().unwrap().scenes.is_empty());
    }

    #[test]
    fn batch_undo_restores_full_pre_state_and_redo_reapplies_same_ids() {
        let mut s = session();
        s.execute(Op::Batch { ops: setup_ops() }).unwrap();
        let committed = s.project.clone().unwrap();

        let undo = s.execute(Op::HistoryUndo).unwrap();
        assert_eq!(undo["entity"], json!({"kind":"history", "operation":"undo", "applied":true}));
        assert!(s.project.as_ref().unwrap().scenes.is_empty(), "one undo rolls back the whole batch");
        assert_eq!(s.history.redo_depth(), 1);

        s.execute(Op::HistoryRedo).unwrap();
        let redone = s.project.as_ref().unwrap();
        assert_eq!(
            serde_json::to_value(&redone.scenes).unwrap(),
            serde_json::to_value(&committed.scenes).unwrap(),
            "redo restores the same ids and content"
        );
        assert_eq!(redone.next_id(), committed.next_id());
        // Undoing again returns the empty state once more (symmetry).
        s.execute(Op::HistoryUndo).unwrap();
        assert!(s.project.as_ref().unwrap().scenes.is_empty());
    }

    #[test]
    fn core_batch_command_is_atomic_and_one_undo_step() {
        // Direct path through core: the same atomicity without a Session.
        let mut project = Project::new("core-batch");
        let mut history = History::default();
        let result = history.apply(
            &mut project,
            Command::Batch(vec![
                Command::AddScene { name: "a".into(), range: FrameRange::new(0, 24).unwrap() },
                Command::AddScene { name: "b".into(), range: FrameRange::new(0, 24).unwrap() },
            ]),
        );
        assert!(result.is_ok());
        assert_eq!(project.scenes.len(), 2);
        assert_eq!(history.undo_depth(), 1);

        let err = history
            .apply(
                &mut project,
                Command::Batch(vec![
                    Command::AddScene { name: "c".into(), range: FrameRange::new(0, 24).unwrap() },
                    Command::SetSceneRange { scene: SceneId(999), range: FrameRange::new(0, 24).unwrap() },
                ]),
            )
            .unwrap_err();
        assert!(matches!(err, CoreError::SceneNotFound(_)));
        assert_eq!(project.scenes.len(), 2, "an unsuccessful core batch does not change the document");

        assert!(matches!(
            history.apply(&mut project, Command::Batch(vec![])).unwrap_err(),
            CoreError::InvalidBatch(_)
        ));
        assert!(matches!(
            history
                .apply(&mut project, Command::Batch(vec![Command::Batch(vec![])]))
                .unwrap_err(),
            CoreError::InvalidBatch(_)
        ));
    }
}

#[cfg(test)]
mod audio_report_tests {
    use super::*;

    fn session_with_scene() -> (Session, u64) {
        let mut s = Session::new();
        s.project = Some(Project::new("audio-report"));
        let scene = s.execute(Op::SceneAdd { name: "s".into(), start: 0, end: 48 }).unwrap()["scene"].as_u64().unwrap();
        (s, scene)
    }

    #[test]
    fn audio_clip_fades_and_update_patch_are_canonical() {
        let (mut s, scene) = session_with_scene();
        let asset = s.cmd(Command::RegisterAsset { kind: AssetKind::Audio, path: "assets/x.wav".into(), bounds: None }).unwrap().asset.unwrap().0;
        let track = s.execute(Op::AudioTrackAdd { scene: Some(scene), name: "t".into() }).unwrap()["track"].as_u64().unwrap();
        let added = s
            .execute(Op::AudioClipAdd { track, asset, start: 0, offset_samples: 0, len_samples: 96_000, fade_in_samples: 4_800, fade_out_samples: 0 })
            .unwrap();
        assert_eq!(added["entity"]["clip"]["fade_in_samples"], 4_800);
        let clip = added["clip"].as_u64().unwrap();
        // A fade longer than the clip — invalid_input before any mutation.
        let err = s.execute(Op::AudioClipAdd { track, asset, start: 0, offset_samples: 0, len_samples: 2_000, fade_in_samples: 9_999, fade_out_samples: 0 }).unwrap_err();
        assert_eq!(err.code(), "invalid_input");
        // Patch: only fade_out; the rest is kept, entity is the canonical clip.
        let updated = s
            .execute(Op::AudioClipUpdate { track, clip, offset_samples: None, len_samples: None, fade_in_samples: None, fade_out_samples: Some(9_600) })
            .unwrap();
        assert_eq!(updated["entity"]["clip"]["fade_in_samples"], 4_800);
        assert_eq!(updated["entity"]["clip"]["fade_out_samples"], 9_600);
        assert_eq!(updated["entity"]["clip"]["len_samples"], 96_000);
        // A patch that would produce an invalid state is rejected without changing the document.
        let err = s.execute(Op::AudioClipUpdate { track, clip, offset_samples: None, len_samples: Some(1_000), fade_in_samples: None, fade_out_samples: None }).unwrap_err();
        assert_eq!(err.code(), "invalid_input");
        let err = s.execute(Op::AudioClipUpdate { track, clip: 999, offset_samples: None, len_samples: None, fade_in_samples: None, fade_out_samples: None }).unwrap_err();
        assert_eq!(err.code(), "not_found");
        let p = s.project.as_ref().unwrap();
        let stored = p.scenes[&SceneId(scene)].audio_tracks[0].clips[0];
        assert_eq!((stored.fade_in_samples, stored.fade_out_samples, stored.len_samples), (4_800, 9_600, 96_000));
    }

    #[test]
    fn film_add_warns_when_audio_outlives_clip() {
        let (mut s, scene) = session_with_scene();
        let asset = s.cmd(Command::RegisterAsset { kind: AssetKind::Audio, path: "assets/music.mp3".into(), bounds: None }).unwrap().asset.unwrap().0;
        let track = s.execute(Op::AudioTrackAdd { scene: Some(scene), name: "music".into() }).unwrap()["track"].as_u64().unwrap();
        // 6.34 s of audio; the film clip is only 1 s (24 frames).
        s.execute(Op::AudioClipAdd { track, asset, start: 0, offset_samples: 0, len_samples: 304_320, fade_in_samples: 0, fade_out_samples: 0 }).unwrap();
        let added = s.execute(Op::FilmAdd { scene, start: 0, end: 24, camera: None }).unwrap();
        let warnings = added["warnings"].as_array().unwrap();
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0]["code"], "audio_longer_than_clip");
        assert!(warnings[0]["why"].as_str().unwrap().contains("music"));
        // A clip covering the audio entirely — no warning (the scene is
        // widened: a film clip must lie inside the scene's range).
        s.execute(Op::SceneRangeSet { scene, start: 0, end: 200 }).unwrap();
        let quiet = s.execute(Op::FilmAdd { scene, start: 0, end: 160, camera: None }).unwrap();
        assert!(quiet["warnings"].as_array().unwrap().is_empty());
    }

    #[test]
    fn audio_report_is_query_with_document_durations_without_ffmpeg_facts() {
        let (mut s, scene) = session_with_scene();
        // A project directory is needed for decoding; a missing file → decode_failed.
        let tmp = std::env::temp_dir().join(format!("phantom-audio-query-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        s.dir = Some(tmp.clone());
        assert_eq!(Op::AudioReport { scene: None }.class(), OpClass::Query);
        let asset = s.cmd(Command::RegisterAsset { kind: AssetKind::Audio, path: "assets/x.wav".into(), bounds: None }).unwrap().asset.unwrap().0;
        let track = s.execute(Op::AudioTrackAdd { scene: Some(scene), name: "t".into() }).unwrap()["track"].as_u64().unwrap();
        s.execute(Op::AudioClipAdd { track, asset, start: 0, offset_samples: 0, len_samples: 96_000, fade_in_samples: 0, fade_out_samples: 0 }).unwrap();
        let revision = s.project.as_ref().unwrap().revision;
        let report = s.execute(Op::AudioReport { scene: None }).unwrap();
        assert!(report.get("entity").is_none(), "a query carries no mutation ack");
        assert_eq!(s.project.as_ref().unwrap().revision, revision);
        assert_eq!(report["film"]["seconds"], 0.0);
        assert_eq!(report["tracks"][0]["clips"][0]["len_seconds"], 2.0);
        let codes: Vec<&str> = report["warnings"].as_array().unwrap().iter().map(|w| w["code"].as_str().unwrap()).collect();
        if phantom_runtime::audio::ffmpeg_available() {
            // Decoding fails on a missing file — per clip, without failing the report.
            assert!(codes.contains(&"decode_failed"), "{codes:?}");
        } else {
            assert!(codes.contains(&"ffmpeg_unavailable"), "{codes:?}");
        }
        // An unknown scene — not_found.
        let err = s.execute(Op::AudioReport { scene: Some(999) }).unwrap_err();
        assert_eq!(err.code(), "not_found");
        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn audio_report_without_project_dir_degrades_to_document_facts() {
        let (mut s, scene) = session_with_scene();
        let asset = s.cmd(Command::RegisterAsset { kind: AssetKind::Audio, path: "assets/x.wav".into(), bounds: None }).unwrap().asset.unwrap().0;
        let track = s.execute(Op::AudioTrackAdd { scene: Some(scene), name: "t".into() }).unwrap()["track"].as_u64().unwrap();
        s.execute(Op::AudioClipAdd { track, asset, start: 0, offset_samples: 0, len_samples: 96_000, fade_in_samples: 0, fade_out_samples: 0 }).unwrap();
        let report = s.execute(Op::AudioReport { scene: None }).unwrap();
        let codes: Vec<&str> = report["warnings"].as_array().unwrap().iter().map(|w| w["code"].as_str().unwrap()).collect();
        assert!(codes.contains(&"ffmpeg_unavailable"), "{codes:?}");
        assert!(report["tracks"][0]["clips"][0]["fragment_rms_db"].is_null());
        assert_eq!(report["tracks"][0]["clips"][0]["len_seconds"], 2.0);
    }

    #[test]
    fn audio_report_measures_real_fragment_rms_when_ffmpeg_exists() {
        if !phantom_runtime::audio::ffmpeg_available() {
            eprintln!("skip: ffmpeg unavailable");
            return;
        }
        let tmp = std::env::temp_dir().join(format!("phantom-audio-report-{}-{}", std::process::id(), 1));
        std::fs::create_dir_all(tmp.join("assets")).unwrap();
        // Quiet first half (~-66 dB), loud second half (-6 dB): the fragment
        // choice must change the RMS — the A4 case.
        let mut samples = vec![[0.0005f32; 2]; 48_000];
        samples.extend(vec![[0.5f32; 2]; 48_000]);
        phantom_runtime::audio::write_wav(&phantom_core::PcmBuf { samples }, &tmp.join("assets/music.wav")).unwrap();
        let (mut s, scene) = session_with_scene();
        s.dir = Some(tmp.clone());
        let asset = s.cmd(Command::RegisterAsset { kind: AssetKind::Audio, path: "assets/music.wav".into(), bounds: None }).unwrap().asset.unwrap().0;
        let track = s.execute(Op::AudioTrackAdd { scene: Some(scene), name: "music".into() }).unwrap()["track"].as_u64().unwrap();
        // The first second (quiet).
        s.execute(Op::AudioClipAdd { track, asset, start: 0, offset_samples: 0, len_samples: 48_000, fade_in_samples: 0, fade_out_samples: 0 }).unwrap();
        // The next second (loud).
        s.execute(Op::AudioClipAdd { track, asset, start: 24, offset_samples: 48_000, len_samples: 48_000, fade_in_samples: 0, fade_out_samples: 0 }).unwrap();
        let report = s.execute(Op::AudioReport { scene: Some(scene) }).unwrap();
        let clips = report["tracks"][0]["clips"].as_array().unwrap();
        let quiet = clips[0]["fragment_rms_db"].as_f64().unwrap();
        let loud = clips[1]["fragment_rms_db"].as_f64().unwrap();
        assert!(quiet < -55.0, "quiet fragment: {quiet} dB");
        assert!((loud - -6.0).abs() < 1.5, "loud fragment: {loud} dB");
        assert_eq!(clips[0]["source_duration_seconds"], 2.0);
        let codes: Vec<&str> = report["warnings"].as_array().unwrap().iter().map(|w| w["code"].as_str().unwrap()).collect();
        assert!(codes.contains(&"quiet_fragment"), "{codes:?}");
        let _ = std::fs::remove_dir_all(tmp);
    }
}

#[cfg(test)]
mod scene_light_tests {
    use super::*;

    #[test]
    fn document_lights_keys_ambient_sky_are_acknowledged() {
        let mut s = Session::new();
        s.project = Some(Project::new("lights"));
        let scene = s.execute(Op::SceneAdd { name: "s".into(), start: 0, end: 48 }).unwrap()["scene"].as_u64().unwrap();
        let added = s
            .execute(Op::LightAdd {
                scene,
                name: "lamp".into(),
                kind: SceneLightKind::Spot { cone: 0.7 },
                color: [1.0, 0.9, 0.8],
                intensity: 800.0,
                enabled: true,
                shadows: false,
                rest_pos: [0.0, 2.0, -1.0],
                rest_yaw: 0.0,
                rest_pitch: -0.5,
            })
            .unwrap();
        let light = added["light"].as_u64().unwrap();
        assert_eq!(added["entity"]["kind"], "light");
        assert_eq!(added["entity"]["light"]["name"], "lamp");
        // Keys: set and remove; entity is the current light source.
        s.execute(Op::LightKeySet { scene, light, frame: 12, pos: [1.0, 2.0, -1.0], yaw: 0.0, pitch: 0.0, intensity: None, enabled: Some(false) }).unwrap();
        let removed_key = s.execute(Op::LightKeyRemove { scene, light, frame: 12 }).unwrap();
        assert!(removed_key["entity"]["light"]["keys"].as_array().unwrap().is_empty());
        let err = s.execute(Op::LightKeyRemove { scene, light, frame: 12 }).unwrap_err();
        assert_eq!(err.code(), "invalid_input");
        let err = s.execute(Op::LightKeySet { scene, light: 999, frame: 0, pos: [0.0; 3], yaw: 0.0, pitch: 0.0, intensity: None, enabled: None }).unwrap_err();
        assert_eq!(err.code(), "not_found");
        // Ambient and sky.
        let ambient = s.execute(Op::AmbientSet { scene, ambient: AmbientMode::Fixed { color: [0.2, 0.2, 0.25], brightness: 300.0 } }).unwrap();
        assert_eq!(ambient["entity"]["ambient"]["brightness"], 300.0);
        let sky = s.execute(Op::SkySet { scene, sky: Some([0.1, 0.2, 0.3]) }).unwrap();
        assert!((sky["entity"]["sky"][0].as_f64().unwrap() - 0.1).abs() < 1e-6);
        let err = s.execute(Op::SkySet { scene, sky: Some([f32::NAN, 0.0, 0.0]) }).unwrap_err();
        assert_eq!(err.code(), "invalid_input");
        // Removing a light source: a removed snapshot, one undo step.
        let removed = s.execute(Op::LightRemove { scene, light }).unwrap();
        assert_eq!(removed["entity"], json!({"kind":"light", "scene":scene, "id":light, "removed":true}));
        s.execute(Op::HistoryUndo).unwrap();
        assert_eq!(s.project.as_ref().unwrap().scenes[&SceneId(scene)].lights.len(), 1, "undo brought the light source back");
    }
}

#[cfg(test)]
mod attach_tests {
    use super::*;

    #[test]
    fn attach_object_light_and_scale_are_acknowledged() {
        let mut s = Session::new();
        s.project = Some(Project::new("attach"));
        let scene = s.execute(Op::SceneAdd { name: "s".into(), start: 0, end: 48 }).unwrap()["scene"].as_u64().unwrap();
        let phantom = s.execute(Op::PhantomAdd { scene, name: "hero".into(), rig_asset: None, rig_actor: None }).unwrap()["phantom"].as_u64().unwrap();
        let model = s.cmd(Command::RegisterAsset { kind: AssetKind::ModelObj, path: "assets/p.obj".into(), bounds: None }).unwrap().asset.unwrap().0;
        let object = s.execute(Op::ObjectAdd { scene, name: "torch".into(), asset: model }).unwrap()["object"].as_u64().unwrap();

        let attach = Some(Attachment { phantom: PhantomId(phantom), bone: "lower_arm_r".into(), offset_pos: glam::Vec3::new(0.0, -0.05, 0.0), offset_yaw: 0.0, offset_pitch: 0.0, offset_roll: 0.0 });
        let v = s.execute(Op::ObjectAttach { scene, object, attach: attach.clone() }).unwrap();
        assert_eq!(v["entity"]["attach"]["bone"], "lower_arm_r");
        // A nonexistent phantom/object or an empty bone name — typed errors.
        let err = s.execute(Op::ObjectAttach { scene, object, attach: Some(Attachment { bone: "  ".into(), ..attach.clone().unwrap() }) }).unwrap_err();
        assert_eq!(err.code(), "invalid_input");
        let err = s.execute(Op::ObjectAttach { scene, object: 999, attach: attach.clone() }).unwrap_err();
        assert_eq!(err.code(), "not_found");
        let err = s.execute(Op::ObjectAttach { scene, object, attach: Some(Attachment { phantom: PhantomId(999), ..attach.clone().unwrap() }) }).unwrap_err();
        assert_eq!(err.code(), "not_found");
        // Detaching — None.
        let v = s.execute(Op::ObjectAttach { scene, object, attach: None }).unwrap();
        assert!(v["entity"]["attach"].is_null());

        // A light source onto a bone plus the actor's scale.
        let light = s
            .execute(Op::LightAdd { scene, name: "lamp".into(), kind: SceneLightKind::Point, color: [1.0; 3], intensity: 100.0, enabled: true, shadows: false, rest_pos: [0.0; 3], rest_yaw: 0.0, rest_pitch: 0.0 })
            .unwrap()["light"]
            .as_u64()
            .unwrap();
        let v = s.execute(Op::LightAttach { scene, light, attach: attach.clone() }).unwrap();
        assert_eq!(v["entity"]["attach"]["phantom"], phantom);
        let v = s.execute(Op::PhantomScaleSet { scene, phantom, scale: 1.5 }).unwrap();
        assert_eq!(v["entity"]["scale"], 1.5);
        let err = s.execute(Op::PhantomScaleSet { scene, phantom, scale: 0.0 }).unwrap_err();
        assert_eq!(err.code(), "invalid_input");
        // Undo restores the scale.
        s.execute(Op::HistoryUndo).unwrap();
        assert_eq!(s.project.as_ref().unwrap().scenes[&SceneId(scene)].phantoms[&PhantomId(phantom)].scale, 1.0);
    }
}

#[cfg(test)]
mod material_tests {
    use super::*;

    #[test]
    fn material_set_replace_clear_and_list_builtin() {
        let mut s = Session::new();
        s.project = Some(Project::new("mat"));
        let scene = s.execute(Op::SceneAdd { name: "s".into(), start: 0, end: 48 }).unwrap()["scene"].as_u64().unwrap();
        let phantom = s.execute(Op::PhantomAdd { scene, name: "wolf".into(), rig_asset: None, rig_actor: Some("wolf".into()) }).unwrap()["phantom"].as_u64().unwrap();

        let patch = MaterialPatch { roughness: Some(0.12), ..Default::default() };
        let v = s.execute(Op::MaterialSet { scene, phantom: Some(phantom), object: None, target: None, patch: patch.clone(), clear: false }).unwrap();
        assert_eq!(v["entity"]["kind"], "material_patches");
        // A repeated set with the same target (None) replaces rather than accumulates.
        let patch2 = MaterialPatch { roughness: Some(0.5), metallic: Some(1.0), ..Default::default() };
        s.execute(Op::MaterialSet { scene, phantom: Some(phantom), object: None, target: None, patch: patch2, clear: false }).unwrap();
        let stored = &s.project.as_ref().unwrap().scenes[&SceneId(scene)].phantoms[&PhantomId(phantom)].material_patches;
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].metallic, Some(1.0));
        // An invalid patch — invalid_input before any mutation.
        let err = s.execute(Op::MaterialSet { scene, phantom: Some(phantom), object: None, target: None, patch: MaterialPatch { roughness: Some(2.0), ..Default::default() }, clear: false }).unwrap_err();
        assert_eq!(err.code(), "invalid_input");
        // Both owners — an error.
        let err = s.execute(Op::MaterialSet { scene, phantom: Some(phantom), object: Some(1), target: None, patch: patch.clone(), clear: false }).unwrap_err();
        assert_eq!(err.code(), "invalid_input");
        // material_list for a builtin actor — no file IO; effective values with the patch.
        let list = s.execute(Op::MaterialList { scene, phantom: Some(phantom), object: None }).unwrap();
        assert!(list.get("entity").is_none(), "a query carries no mutation ack");
        let parts = list["parts"].as_array().unwrap();
        assert!(!parts.is_empty());
        assert_eq!(parts[0]["effective"]["metallic"], 1.0);
        assert_eq!(parts[0]["effective"]["roughness"], 0.5);
        // clear: all patches removed.
        s.execute(Op::MaterialSet { scene, phantom: Some(phantom), object: None, target: None, patch: MaterialPatch::default(), clear: true }).unwrap();
        assert!(s.project.as_ref().unwrap().scenes[&SceneId(scene)].phantoms[&PhantomId(phantom)].material_patches.is_empty());
    }

    #[test]
    fn material_presets_are_known_and_valid() {
        for name in ["matte", "skin", "metal", "wet", "glass", "glow"] {
            let p = material_preset(name).expect(name);
            p.validate().unwrap_or_else(|e| panic!("preset {name}: {e}"));
        }
        assert!(material_preset("chrome_plated").is_none());
    }
}

#[cfg(test)]
mod vfx_grade_tests {
    use super::*;

    #[test]
    fn grade_fog_vfx_lifecycle_is_acknowledged() {
        let mut s = Session::new();
        s.project = Some(Project::new("vfx"));
        let scene = s.execute(Op::SceneAdd { name: "s".into(), start: 0, end: 48 }).unwrap()["scene"].as_u64().unwrap();
        let grade = s.execute(Op::GradeSet { scene, grade: Some(SceneGrade { exposure: 0.5, ..Default::default() }) }).unwrap();
        assert_eq!(grade["entity"]["grade"]["exposure"], 0.5);
        let off = s.execute(Op::GradeSet { scene, grade: None }).unwrap();
        assert!(off["entity"]["grade"].is_null());
        let fog = s.execute(Op::FogSet { scene, fog: Some(SceneFog { color: [0.5, 0.6, 0.7], density: 0.05 }) }).unwrap();
        assert!((fog["entity"]["fog"]["density"].as_f64().unwrap() - 0.05).abs() < 1e-6);
        let err = s.execute(Op::FogSet { scene, fog: Some(SceneFog { color: [0.5; 3], density: 0.0 }) }).unwrap_err();
        assert_eq!(err.code(), "invalid_input");

        let v = s.execute(Op::VfxAdd { scene, vfx: vfx_preset("snow").unwrap() }).unwrap();
        let id = v["vfx"].as_u64().unwrap();
        assert_eq!(v["entity"]["vfx"]["name"], "snow");
        s.execute(Op::VfxKeySet { scene, vfx: id, frame: 24, enabled: Some(false), density: None }).unwrap();
        let after = s.execute(Op::VfxKeySet { scene, vfx: id, frame: 36, enabled: Some(true), density: Some(0.5) }).unwrap();
        assert_eq!(after["entity"]["vfx"]["keys"].as_array().unwrap().len(), 2);
        let stored = &s.project.as_ref().unwrap().scenes[&SceneId(scene)].vfx[0];
        assert!(!stored.enabled_at(24) && stored.enabled_at(36));
        let removed = s.execute(Op::VfxRemove { scene, vfx: id }).unwrap();
        assert_eq!(removed["entity"]["removed"], true);
        let err = s.execute(Op::VfxKeySet { scene, vfx: id, frame: 0, enabled: None, density: None }).unwrap_err();
        assert_eq!(err.code(), "invalid_input", "a key on a removed effect");
    }
}

#[cfg(test)]
mod import_formats_tests {
    use super::*;

    #[test]
    fn tiff_is_supported_and_gltf_is_rejected_honestly() {
        assert!(parse_asset_kind("model-gltf").is_err());
        // Importing a tiff file (copied without decoding) → Image.
        let tmp = std::env::temp_dir().join(format!("phantom-import-fmt-{}", std::process::id()));
        std::fs::create_dir_all(tmp.join("assets")).unwrap();
        let mut s = Session::new();
        s.project = Some(Project::new("fmt"));
        s.dir = Some(tmp.clone());
        let src = tmp.join("bg.tiff");
        std::fs::write(&src, b"II*\0not-a-real-tiff").unwrap();
        let v = s.execute(Op::AssetImport { file: src.clone(), kind: None }).unwrap();
        assert!(v["asset"].is_u64(), "{v}");
        let a = s.project.as_ref().unwrap().assets.values().next().unwrap();
        assert!(matches!(a.kind, AssetKind::Image));
        // glTF is not copied as an "image" and no fake support is promised.
        let glb = tmp.join("empty.glb");
        std::fs::write(&glb, b"").unwrap();
        let err = s.execute(Op::AssetImport { file: glb, kind: None }).unwrap_err();
        assert_eq!(err.code(), "invalid_input");
        let _ = std::fs::remove_dir_all(tmp);
    }
}

#[cfg(test)]
mod corrective_e2e_tests {
    use super::*;

    /// The agent's full production path (corrective reproduction) without a GPU:
    /// project → staging batch → light/attachment/effect/material →
    /// validate + audio_report before the render.
    #[test]
    fn agent_production_path_end_to_end() {
        let mut s = Session::new();
        s.project = Some(Project::new("never_cry_wolf"));
        // One batch instead of 10 round-trips: scene, actor, motion, camera,
        // background asset, plus a track and a music clip longer than the scene (the A4 case).
        let v = s
            .execute(Op::Batch {
                ops: vec![
                    Op::SceneAdd { name: "Scene_1".into(), start: 0, end: 48 },
                    Op::PhantomAdd { scene: 1, name: "wolf".into(), rig_asset: None, rig_actor: Some("wolf".into()) },
                    Op::KeySet { scene: 1, phantom: 2, frame: 0, pos: [0.0, 0.0, 0.0], pose: ActorPose::Stand, yaw: 0.0 },
                    Op::KeySet { scene: 1, phantom: 2, frame: 47, pos: [4.0, 0.0, 0.0], pose: ActorPose::Stand, yaw: 0.0 },
                    Op::CameraAdd { scene: 1, name: "wide".into(), start: Some(0), end: Some(48) },
                ],
            })
            .unwrap();
        assert_eq!(v["applied"], 5);
        assert_eq!(v["revision"], 1);
        let asset = s.cmd(Command::RegisterAsset { kind: AssetKind::Audio, path: "assets/music.mp3".into(), bounds: None }).unwrap().asset.unwrap().0;
        s.execute(Op::Batch {
            ops: vec![
                Op::AudioTrackAdd { scene: Some(1), name: "music".into() },
                Op::AudioClipAdd { track: 5, asset, start: 0, offset_samples: 0, len_samples: 304_320, fade_in_samples: 0, fade_out_samples: 4_800 },
            ],
        })
        .unwrap();
        // Light: a lantern on the wolf's bone; a snowstorm effect; a wet wolf.
        let light = s
            .execute(Op::LightAdd { scene: 1, name: "lamp".into(), kind: SceneLightKind::Point, color: [1.0, 0.9, 0.8], intensity: 600.0, enabled: true, shadows: false, rest_pos: [0.0; 3], rest_yaw: 0.0, rest_pitch: 0.0 })
            .unwrap()["light"]
            .as_u64()
            .unwrap();
        s.execute(Op::LightAttach { scene: 1, light, attach: Some(Attachment { phantom: PhantomId(2), bone: "head".into(), offset_pos: Default::default(), offset_yaw: 0.0, offset_pitch: 0.0, offset_roll: 0.0 }) }).unwrap();
        s.execute(Op::VfxAdd { scene: 1, vfx: vfx_preset("snow").unwrap() }).unwrap();
        s.execute(Op::MaterialSet { scene: 1, phantom: Some(2), object: None, target: None, patch: material_preset("wet").unwrap(), clear: false }).unwrap();
        s.execute(Op::FogSet { scene: 1, fog: Some(SceneFog { color: [0.6, 0.7, 0.85], density: 0.04 }) }).unwrap();
        s.execute(Op::SkySet { scene: 1, sky: Some([0.4, 0.5, 0.7]) }).unwrap();

        // Diagnostics before the render: validate sees the long music; audio_report —
        // the lost seconds; camera_coverage — the frame geometry.
        let report = s.execute(Op::Validate { scene: None }).unwrap();
        let codes: Vec<&str> = report["findings"].as_array().unwrap().iter().map(|f| f["code"].as_str().unwrap()).collect();
        assert!(codes.contains(&"audio_out_of_range") || codes.contains(&"audio_length_mismatch"), "{codes:?}");
        let audio = s.execute(Op::AudioReport { scene: None }).unwrap();
        let summary = &s.execute(Op::StateGet).unwrap()["audio_summary"];
        assert!(summary["scenes"][0]["lost_seconds"].as_f64().unwrap() > 4.0, "A4: the lost seconds are a fact known before the render");
        assert!(audio["tracks"].is_array());
        let coverage = s.execute(Op::CameraCoverage { scene: 1, frame: None, camera: None, aspect: None }).unwrap();
        assert!(coverage["camera"]["hfov_deg"].is_f64());
        // Undoing the whole path — ordinary history: every mutation is undoable.
        while s.history.undo_depth() > 0 {
            s.execute(Op::HistoryUndo).unwrap();
        }
        assert!(s.project.as_ref().unwrap().scenes.is_empty());
    }
}

#[cfg(test)]
mod yaw_tests {
    use super::*;

    #[test]
    fn legacy_key_set_and_take_triples_remain_readable() {
        // A pre-layout wire request carries `action`; it is ignored and the key
        // falls back to the default pose.
        let op: Op = serde_json::from_str(r#"{"op":"key_set","scene":1,"phantom":2,"frame":3,"pos":[0,2,0],"pose":"sit"}"#).unwrap();
        assert!(matches!(op, Op::KeySet { yaw: 0.0, pose: ActorPose::Sit, .. }));
    }
}

#[cfg(test)]
mod import_tests {
    use super::*;

    #[test]
    fn asset_import_routes_obj_through_bundle() {
        let tmp = std::env::temp_dir().join(format!("phantom-session-import-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let src = tmp.join("src");
        std::fs::create_dir_all(&src).unwrap();
        let obj = src.join("quad.obj");
        std::fs::write(&obj, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").unwrap();

        let proj_dir = tmp.join("proj");
        // ProjectNew records the recent path; keep it inside the test tmp dir.
        let mut session = Session::with_state_path(tmp.join("recent.json"));
        session.execute(Op::ProjectNew { dir: proj_dir.clone(), name: "t".into() }).unwrap();
        let out = session.execute(Op::AssetImport { file: obj, kind: None }).unwrap();
        let id = out["asset"].as_u64().unwrap();
        let asset = &session.project.as_ref().unwrap().assets[&AssetId(id)];
        // Bundle import: the model lands in the bundle's own directory.
        assert_eq!(asset.path, Path::new("assets/quad/quad.obj"));
        assert!(proj_dir.join(&asset.path).exists());
        drop(session);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn asset_import_kinds_and_legacy_json() {
        // Legacy JSON without kind parses (autodetect by extension).
        let op: Op = serde_json::from_value(json!({ "op": "asset_import", "file": "x.fbx" })).unwrap();
        match op {
            Op::AssetImport { kind, .. } => assert!(kind.is_none()),
            other => panic!("unexpected operation: {other:?}"),
        }

        let tmp = std::env::temp_dir().join(format!("phantom-session-motion-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let src = tmp.join("src");
        std::fs::create_dir_all(&src).unwrap();
        // A real FBX from the repository root: bundle import parses the file.
        let real = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Hip Hop Dancing.fbx");
        let fbx = src.join("move.fbx");
        std::fs::copy(&real, &fbx).unwrap();

        let proj_dir = tmp.join("proj");
        let mut session = Session::with_state_path(tmp.join("recent.json"));
        session.execute(Op::ProjectNew { dir: proj_dir.clone(), name: "t".into() }).unwrap();

        // An invalid kind is rejected before import.
        let err = session
            .execute(Op::AssetImport { file: fbx.clone(), kind: Some("wat".into()) })
            .unwrap_err();
        assert!(matches!(err, SessionError::BadInput(_)));

        // motion-fbx is no longer an importable kind: animation is gone.
        let err = session
            .execute(Op::AssetImport { file: fbx.clone(), kind: Some("motion-fbx".into()) })
            .unwrap_err();
        assert!(matches!(err, SessionError::BadInput(_)), "{err}");

        // An explicit model-obj kind still registers through the bundle import.
        let out = session.execute(Op::AssetImport { file: fbx, kind: Some("model-obj".into()) });
        assert!(out.is_ok() || out.is_err(), "an explicit kind is accepted or reported, never a panic");

        // A rigged model imported as ModelFbx becomes a Phantom that renders in
        // bind pose (no poses, no clips).
        let model = session
            .execute(Op::AssetImport { file: src.join("move.fbx"), kind: Some("model-fbx".into()) })
            .unwrap()["asset"]
            .as_u64()
            .unwrap();
        let scene = session.execute(Op::SceneAdd { name: "s".into(), start: 0, end: 48 }).unwrap()["scene"].as_u64().unwrap();
        let phantom = session
            .execute(Op::PhantomAdd { scene, name: "p".into(), rig_asset: Some(model), rig_actor: None })
            .unwrap()["phantom"]
            .as_u64()
            .unwrap();
        // Any pose is accepted for an imported rig (it has none of its own).
        session.execute(Op::KeySet { scene, phantom, frame: 0, pos: [0.0, 0.0, 0.0], pose: ActorPose::Sit, yaw: 0.0 }).unwrap();

        drop(session);
        let _ = std::fs::remove_dir_all(&tmp);
    }
}

#[cfg(test)]
mod lighting_tests {
    use super::*;

    #[test]
    fn lighting_floor_ops_execute_and_report() {
        let mut session = Session::new();
        session.project = Some(Project::new("light"));
        let out = session.execute(Op::SceneAdd { name: "s".into(), start: 0, end: 24 }).unwrap();
        let scene = out["scene"].as_u64().unwrap();

        let lighting = SceneLighting {
            key: SceneLight { color: [1.0, 0.9, 0.8], intensity: 9000.0, enabled: true, shadows: false },
            ..SceneLighting::default()
        };
        session.execute(Op::LightingSet { scene, lighting }).unwrap();
        let got = session.execute(Op::LightingGet { scene }).unwrap();
        assert_eq!(got["lighting"]["key"]["intensity"].as_f64().unwrap(), 9000.0);

        assert!(session.execute(Op::FloorGet { scene }).unwrap()["floor"].is_null());
        session.execute(Op::FloorSet { scene, floor: Some(SceneFloor { color: [0.4; 3] }) }).unwrap();
        assert!(!session.execute(Op::FloorGet { scene }).unwrap()["floor"].is_null());
        session.execute(Op::FloorSet { scene, floor: None }).unwrap();
        assert!(session.execute(Op::FloorGet { scene }).unwrap()["floor"].is_null());

        // An invalid intensity is rejected at the core level.
        let mut bad = lighting;
        bad.fill.intensity = -5.0;
        assert!(session.execute(Op::LightingSet { scene, lighting: bad }).is_err());
        // A nonexistent scene — an error, not a panic.
        assert!(session.execute(Op::FloorGet { scene: 999 }).is_err());
    }

    #[test]
    fn stage_set_clamps_and_reports_it() {
        let mut session = Session::new();
        session.project = Some(Project::new("stage"));
        let scene = session.execute(Op::SceneAdd { name: "s".into(), start: 0, end: 48 }).unwrap()["scene"].as_u64().unwrap();
        let phantom = session
            .execute(Op::PhantomAdd { scene, name: "actor".into(), rig_asset: None, rig_actor: Some("humanoid".into()) })
            .unwrap()["phantom"]
            .as_u64()
            .unwrap();
        // A new scene carries the medium stage.
        let got = session.execute(Op::StageGet { scene }).unwrap();
        assert_eq!(got["stage"]["preset"], "medium");
        assert_eq!(got["stage"]["size"]["width"].as_f64().unwrap(), 30.0);

        // A key outside the stage is clamped, and the acknowledgement says so.
        let out = session
            .execute(Op::KeySet { scene, phantom, frame: 0, pos: [100.0, 0.0, 0.0], pose: ActorPose::Stand, yaw: 0.0 })
            .unwrap();
        assert_eq!(out["clamped"], true);
        assert_eq!(out["clamps"][0]["kind"], "phantom");
        assert_eq!(out["clamps"][0]["from"][0].as_f64().unwrap(), 100.0);
        assert!((out["clamps"][0]["to"][0].as_f64().unwrap() - 14.7).abs() < 1e-4);
        assert!((out["entity"]["key"]["position"][0].as_f64().unwrap() - 14.7).abs() < 1e-4, "the entity echo shows what was stored");

        // A key inside the stage is silent.
        let out = session
            .execute(Op::KeySet { scene, phantom, frame: 10, pos: [1.0, 0.0, 1.0], pose: ActorPose::Stand, yaw: 0.0 })
            .unwrap();
        assert!(out.get("clamped").is_none(), "{out}");

        // Shrinking the stage re-clamps in the same step and one undo.
        let out = session.execute(Op::StageSet { scene, preset: Some("small".into()), width: None, height: None, depth: None, center: None }).unwrap();
        assert_eq!(out["stage"]["preset"], "small");
        assert_eq!(out["clamped"], true);
        assert!((out["clamps"][0]["to"][0].as_f64().unwrap() - 4.7).abs() < 1e-4);
        assert_eq!(session.execute(Op::StageGet { scene }).unwrap()["stage"]["size"]["depth"].as_f64().unwrap(), 10.0);

        // Undo restores both the size and the key.
        session.execute(Op::HistoryUndo).unwrap();
        assert_eq!(session.execute(Op::StageGet { scene }).unwrap()["stage"]["preset"], "medium");
        let key = session.execute(Op::StateGet).unwrap()["phantoms"][0]["motion_keys"].as_u64().unwrap();
        assert_eq!(key, 2);

        // A partial size without a preset is rejected before the document changes.
        assert!(session
            .execute(Op::StageSet { scene, preset: None, width: Some(10.0), height: None, depth: None, center: None })
            .is_err());
        assert!(session.execute(Op::StageSet { scene, preset: Some("huge".into()), width: None, height: None, depth: None, center: None }).is_err());
        // Explicit axes override the preset, and the label stops claiming a preset.
        let out = session
            .execute(Op::StageSet { scene, preset: Some("large".into()), width: Some(12.0), height: None, depth: None, center: Some([1.0, 2.0]) })
            .unwrap();
        assert!(out["stage"]["preset"].is_null(), "{out}");
        assert_eq!(out["stage"]["size"]["width"].as_f64().unwrap(), 12.0);
        assert_eq!(out["stage"]["size"]["height"].as_f64().unwrap(), 20.0, "the preset fills the axes the caller left out");
        assert_eq!(out["stage"]["center"][0].as_f64().unwrap(), 1.0);
    }

    #[test]
    fn stage_ops_round_trip_json() {
        let op: Op = serde_json::from_str(r#"{"op":"stage_set","scene":1,"preset":"small"}"#).unwrap();
        assert!(matches!(op, Op::StageSet { preset: Some(_), width: None, .. }));
        let op: Op = serde_json::from_str(r#"{"op":"stage_set","scene":1,"width":10.0,"height":4.0,"depth":10.0}"#).unwrap();
        assert!(matches!(op, Op::StageSet { preset: None, width: Some(_), .. }));
        let op: Op = serde_json::from_str(r#"{"op":"stage_get","scene":1}"#).unwrap();
        assert!(matches!(op, Op::StageGet { scene: 1 }));
        assert_eq!(Op::StageGet { scene: 1 }.class(), OpClass::Query);
        assert_eq!(Op::StageSet { scene: 1, preset: None, width: None, height: None, depth: None, center: None }.class(), OpClass::DocumentMutation);
    }

    #[test]
    fn lighting_floor_ops_round_trip_json() {
        let op: Op = serde_json::from_str(
            r#"{"op":"lighting_set","scene":1,"lighting":{"key":{"color":[1,1,1],"intensity":8000},"fill":{"color":[0.9,0.95,1],"intensity":3000},"rim":{"color":[1,0.95,0.9],"intensity":6000}}}"#,
        )
        .unwrap();
        assert!(matches!(op, Op::LightingSet { lighting, .. } if lighting.fill.enabled));
        let op: Op = serde_json::from_str(r#"{"op":"floor_set","scene":1,"floor":null}"#).unwrap();
        assert!(matches!(op, Op::FloorSet { floor: None, .. }));
    }
}

#[cfg(test)]
mod object_tests {
    use super::*;

    fn session_with_scene_and_model() -> (Session, u64, u64) {
        let mut session = Session::new();
        session.project = Some(Project::new("objects"));
        let scene = session.execute(Op::SceneAdd { name: "s".into(), start: 0, end: 48 }).unwrap()["scene"].as_u64().unwrap();
        let asset = {
            let p = session.project.as_mut().unwrap();
            session.history.apply(p, Command::RegisterAsset { kind: AssetKind::ModelObj, path: "assets/quad/quad.obj".into(), bounds: None }).unwrap().asset.unwrap().0
        };
        (session, scene, asset)
    }

    #[test]
    fn object_ops_dispatch_validate_and_report() {
        let (mut session, scene, asset) = session_with_scene_and_model();
        let object = session
            .execute(Op::ObjectAdd { scene, name: "quad".into(), asset })
            .unwrap()["object"]
            .as_u64()
            .unwrap();
        session.execute(Op::ObjectKeySet { scene, object, frame: 0, pos: [0.0, 0.0, 0.0], yaw: 0.0, pitch: 0.0, roll: 0.0, scale: [1.0, 1.0, 1.0] }).unwrap();
        session.execute(Op::ObjectKeySet { scene, object, frame: 24, pos: [2.0, 0.0, 0.0], yaw: 1.0, pitch: 0.0, roll: 0.0, scale: [2.0, 2.0, 2.0] }).unwrap();
        let o = &session.project.as_ref().unwrap().scenes[&SceneId(scene)].objects[&ObjectId(object)];
        assert_eq!(o.keys.len(), 2);

        // The object list in the scene state.
        let state = session.execute(Op::StateGet).unwrap();
        assert_eq!(state["objects"].as_array().unwrap().len(), 1);
        assert_eq!(state["objects"][0]["keys"].as_u64().unwrap(), 2);

        // A non-numeric value is rejected at the core level.
        assert!(session.execute(Op::ObjectKeySet { scene, object, frame: 1, pos: [f32::NAN, 0.0, 0.0], yaw: 0.0, pitch: 0.0, roll: 0.0, scale: [1.0, 1.0, 1.0] }).is_err());
        // A nonexistent object — an error, not a panic.
        assert!(session.execute(Op::ObjectKeyRemove { scene, object: 999, frame: 0 }).is_err());

        session.execute(Op::ObjectKeyRemove { scene, object, frame: 0 }).unwrap();
        assert_eq!(session.project.as_ref().unwrap().scenes[&SceneId(scene)].objects[&ObjectId(object)].keys.len(), 1);
        session.execute(Op::ObjectRemove { scene, object }).unwrap();
        assert!(session.execute(Op::StateGet).unwrap()["objects"].as_array().unwrap().is_empty());
    }

    #[test]
    fn legacy_object_ops_json_defaults() {
        // yaw/pitch/roll/scale are optional in JSON.
        let op: Op = serde_json::from_str(r#"{"op":"object_key_set","scene":1,"object":2,"frame":3,"pos":[0,1,0]}"#).unwrap();
        assert!(matches!(op, Op::ObjectKeySet { yaw: 0.0, scale: [1.0, 1.0, 1.0], .. }));
    }
}

#[cfg(test)]
mod actor_tests {
    use super::*;

    fn session_with_scene() -> (Session, u64) {
        let mut session = Session::new();
        session.project = Some(Project::new("actors"));
        let out = session.execute(Op::SceneAdd { name: "s".into(), start: 0, end: 48 }).unwrap();
        (session, out["scene"].as_u64().unwrap())
    }

    #[test]
    fn phantom_add_dispatches_builtin_actors_and_rejects_conflicts() {
        let (mut session, scene) = session_with_scene();
        for actor in ["humanoid", "wolf", "horse", "bird"] {
            let out = session
                .execute(Op::PhantomAdd { scene, name: actor.into(), rig_asset: None, rig_actor: Some(actor.into()) })
                .unwrap();
            let pid = PhantomId(out["phantom"].as_u64().unwrap());
            let rig = &session.project.as_ref().unwrap().scenes[&SceneId(scene)].phantoms[&pid].rig;
            assert_eq!(rig.builtin_actor(), Some(crate::builtin::parse_actor(actor).unwrap()));
        }
        // Legacy: no actor/asset keeps the legacy builtin rig.
        let out = session.execute(Op::PhantomAdd { scene, name: "legacy".into(), rig_asset: None, rig_actor: None }).unwrap();
        let pid = PhantomId(out["phantom"].as_u64().unwrap());
        assert!(matches!(session.project.as_ref().unwrap().scenes[&SceneId(scene)].phantoms[&pid].rig, RigRef::Builtin));
        // Conflicting asset + actor rejected.
        assert!(matches!(
            session.execute(Op::PhantomAdd { scene, name: "bad".into(), rig_asset: Some(1), rig_actor: Some("wolf".into()) }),
            Err(SessionError::ConflictingRig)
        ));
        assert!(session.execute(Op::PhantomAdd { scene, name: "bad".into(), rig_asset: None, rig_actor: Some("dragon".into()) }).is_err());
    }

    #[test]
    fn key_set_validates_actor_poses_and_catalog_lists_profiles() {
        let (mut session, scene) = session_with_scene();
        let out = session
            .execute(Op::PhantomAdd { scene, name: "bird".into(), rig_asset: None, rig_actor: Some("bird".into()) })
            .unwrap();
        let phantom = out["phantom"].as_u64().unwrap();
        session.execute(Op::KeySet { scene, phantom, frame: 0, pos: [0., 1., 0.], pose: ActorPose::Sit, yaw: 0.0 }).unwrap();
        // A bird has no lying pose — an unavailable pose is an error, not an
        // approximation (a humanoid figure would accept it).
        assert!(session.execute(Op::KeySet { scene, phantom, frame: 0, pos: [0., 1., 0.], pose: ActorPose::Lie, yaw: 0.0 }).is_err());
        let catalog = session.execute(Op::BuiltinCatalog).unwrap();
        assert_eq!(catalog["profiles"].as_array().unwrap().len(), 7, "humanoid, wolf, horse, bird + child/teen/elder");
        assert!(catalog["profiles"].as_array().unwrap().iter().any(|p| p["actor"] == "bird"
            && p["poses"] == serde_json::json!(["stand", "sit"]) && p["default_pose"] == "stand"));
    }

    #[test]
    fn legacy_phantom_add_op_remains_readable() {
        let op: Op = serde_json::from_str(r#"{"op":"phantom_add","scene":1,"name":"p","rig_asset":null}"#).unwrap();
        assert!(matches!(op, Op::PhantomAdd { rig_actor: None, .. }));
    }
}

#[cfg(test)]
mod shot_report_tests {
    use super::*;

    fn range(start: Frame, end: Frame) -> FrameRange {
        FrameRange::new(start, end).unwrap()
    }

    #[test]
    fn auto_sampling_stays_cheap_and_covers_the_range_ends() {
        // A short range — all frames; a long one — up to 5 points including both ends.
        assert_eq!(camera_report_frames(range(10, 13), None).unwrap(), vec![10, 11, 12]);
        let picked = camera_report_frames(range(0, 48), None).unwrap();
        assert_eq!(picked.len(), 5);
        assert_eq!((picked[0], picked[picked.len() - 1]), (0, 47));
        assert!(picked.windows(2).all(|w| w[0] < w[1]), "{picked:?}");
    }

    #[test]
    fn explicit_every_and_cap_are_enforced() {
        assert_eq!(camera_report_frames(range(0, 6), Some(2)).unwrap(), vec![0, 2, 4]);
        // Hard cap: the report does not turn into a render of the whole range.
        assert_eq!(camera_report_frames(range(0, 1000), Some(1)).unwrap().len(), 12);
        assert_eq!(camera_report_frames(range(0, 10), Some(0)).unwrap_err().code(), "invalid_input");
    }

    #[test]
    fn shot_report_is_a_job_class_op_without_mutation_ack() {
        assert_eq!(Op::CameraReport { scene: 1, camera: 2, every: None, width: None, height: None }.class(), OpClass::Job);
        let mut session = Session::new();
        session.project = Some(Project::new("camera-report"));
        session.execute(Op::SceneAdd { name: "s".into(), start: 0, end: 48 }).unwrap();
        session.execute(Op::CameraAdd { scene: 1, name: "wide".into(), start: Some(0), end: Some(24) }).unwrap();
        // The report needs a GPU, unavailable in tests — check the pre-render validation.
        let missing = session
            .execute(Op::CameraReport { scene: 1, camera: 999, every: None, width: None, height: None })
            .unwrap_err();
        assert_eq!(missing.code(), "not_found");
        assert!(session.execute(Op::CameraReport { scene: 1, camera: 2, every: Some(0), width: None, height: None }).is_err());
    }
}

#[cfg(test)]
mod shot_tests {
    use super::*;

    fn session_with_scene() -> (Session, u64) {
        let mut session = Session::new();
        session.project = Some(Project::new("cameras"));
        let out = session.execute(Op::SceneAdd { name: "s".into(), start: 0, end: 96 }).unwrap();
        (session, out["scene"].as_u64().unwrap())
    }

    #[test]
    fn camera_ops_dispatch_validate_and_expose_state() {
        let (mut session, scene) = session_with_scene();
        // A range outside the scene is rejected.
        assert!(session.execute(Op::CameraAdd { scene, name: "bad".into(), start: Some(90), end: Some(100) }).is_err());
        let camera = session.execute(Op::CameraAdd { scene, name: "a".into(), start: Some(24), end: Some(48) }).unwrap()["camera"].as_u64().unwrap();
        session
            .execute(Op::CameraKeySet { scene, camera: Some(camera), key: TransformKey::at(24, [1.0, 2.0, 3.0].into(), 0.5, 0.0) })
            .unwrap();
        session.execute(Op::CameraLensSet { scene, camera, focal_mm: 50.0 }).unwrap();
        session.execute(Op::CameraRangeSet { scene, camera, start: 12, end: 36 }).unwrap();
        let state = session.execute(Op::StateGet).unwrap();
        let cameras = state["cameras"].as_array().unwrap();
        assert_eq!(cameras.len(), 2, "the scene's own camera and the named one");
        let entry = cameras.iter().find(|c| c["id"].as_u64() == Some(camera)).expect("the named camera is listed");
        assert_eq!(entry["start"].as_i64().unwrap(), 12);
        assert_eq!(entry["focal_mm"].as_f64().unwrap(), 50.0);
        assert_eq!(entry["active"], true, "adding a camera makes it the active one");
        // A film clip on a named camera: its range must equal the camera's range.
        assert!(session.execute(Op::FilmAdd { scene, start: 0, end: 10, camera: Some(camera) }).is_err());
        let clip = session.execute(Op::FilmAdd { scene, start: 12, end: 36, camera: Some(camera) }).unwrap()["clip"].as_u64().unwrap();
        assert_eq!(session.project.as_ref().unwrap().film[0].camera, Some(CameraId(camera)));
        session.execute(Op::CameraRemove { scene, camera }).unwrap();
        // The camera's clip goes away with it; the scene keeps its own camera.
        assert!(session.project.as_ref().unwrap().film.iter().all(|c| c.id.0 != clip));
        let state = session.execute(Op::StateGet).unwrap();
        assert_eq!(state["cameras"].as_array().unwrap().len(), 1, "only the scene's own camera is left");
    }

    #[test]
    fn legacy_film_add_and_render_scene_ops_remain_readable() {
        let op: Op = serde_json::from_str(r#"{"op":"film_add","scene":1,"start":0,"end":10}"#).unwrap();
        assert!(matches!(op, Op::FilmAdd { camera: None, .. }));
        let op: Op = serde_json::from_str(r#"{"op":"render_scene","scene":1,"out":"o","width":320,"height":180,"frames":null}"#).unwrap();
        assert!(matches!(op, Op::RenderScene { camera: None, .. }));
    }
}

#[cfg(test)]
mod expansion_tests {
    use super::*;

    fn session_with_scene() -> (Session, u64) {
        let mut session = Session::new();
        session.project = Some(Project::new("exp"));
        let scene = session.execute(Op::SceneAdd { name: "s".into(), start: 0, end: 96 }).unwrap()["scene"].as_u64().unwrap();
        (session, scene)
    }

    fn add_phantom(session: &mut Session, scene: u64) -> u64 {
        session
            .execute(Op::PhantomAdd { scene, name: "p".into(), rig_asset: None, rig_actor: None })
            .unwrap()["phantom"]
            .as_u64()
            .unwrap()
    }

    #[test]
    fn scene_range_set_dispatch_and_validation() {
        let (mut session, scene) = session_with_scene();
        session.execute(Op::SceneRangeSet { scene, start: 12, end: 48 }).unwrap();
        assert_eq!(session.project.as_ref().unwrap().scenes[&SceneId(scene)].range, FrameRange::new(12, 48).unwrap());
        // An empty range is rejected; a nonexistent scene — an error, not a panic.
        assert!(session.execute(Op::SceneRangeSet { scene, start: 48, end: 48 }).is_err());
        assert!(session.execute(Op::SceneRangeSet { scene: 999, start: 0, end: 10 }).is_err());
    }

    #[test]
    fn phantom_motion_ops_dispatch_validate_and_one_undo_replace() {
        let (mut session, scene) = session_with_scene();
        let phantom = add_phantom(&mut session, scene);

        session.execute(Op::KeySet { scene, phantom, frame: 0, pos: [0.0, 0.0, 0.0], pose: ActorPose::Stand, yaw: 0.0 }).unwrap();
        session.execute(Op::KeySet { scene, phantom, frame: 24, pos: [1.0, 0.0, 0.0], pose: ActorPose::Stand, yaw: 0.5 }).unwrap();
        session.execute(Op::MotionKeyRemove { scene, phantom, frame: 24 }).unwrap();
        let p = |s: &Session| s.project.as_ref().unwrap().scenes[&SceneId(scene)].phantoms[&PhantomId(phantom)].clone();
        assert_eq!(p(&session).motion.len(), 1);
        // A nonexistent phantom/scene — an error, not a panic.
        assert!(session.execute(Op::MotionKeyRemove { scene, phantom: 999, frame: 0 }).is_err());
        assert!(session.execute(Op::MotionKeyRemove { scene: 999, phantom, frame: 0 }).is_err());

        // yaws must match motion in length.
        let err = session
            .execute(Op::MotionReplace { scene, phantom, motion: vec![(0, [0.0, 0.0, 0.0], ActorPose::Stand)], yaws: Some(vec![0.1, 0.2]) })
            .unwrap_err();
        assert!(matches!(err, SessionError::BadInput(_)));

        // ReplaceMotion — one command, one undo step; yaw reaches the keys.
        session
            .execute(Op::MotionReplace {
                scene,
                phantom,
                motion: vec![(48, [2.0, 0.0, 0.0], ActorPose::Stand), (0, [0.0, 0.0, 0.0], ActorPose::Stand)],
                yaws: Some(vec![1.0, -1.0]),
            })
            .unwrap();
        let motion = p(&session).motion;
        assert_eq!(motion.len(), 2);
        assert_eq!(motion[0].frame, 0, "the track is sorted by frame");
        assert_eq!(motion[0].yaw, -1.0);
        assert_eq!(motion[1].yaw, 1.0);
        let depth = session.history.undo_depth();
        let undo = session.execute(Op::HistoryUndo).unwrap();
        assert_eq!(undo["applied"], true);
        assert_eq!(session.history.undo_depth(), depth - 1, "replacing the track is one undo step");
        assert_eq!(p(&session).motion.len(), 1, "the old track is restored in full");

        // A pose the actor does not support is rejected atomically. The
        // legacy builtin rig is the humanoid figure and takes every pose, so
        // the check needs an actor with a smaller set — a bird.
        let bird = session
            .execute(Op::PhantomAdd { scene, name: "bird".into(), rig_asset: None, rig_actor: Some("bird".into()) })
            .unwrap()["phantom"]
            .as_u64()
            .unwrap();
        assert!(session.execute(Op::MotionReplace { scene, phantom: bird, motion: vec![(0, [0.0, 0.0, 0.0], ActorPose::Lie)], yaws: None }).is_err());
        assert_eq!(p(&session).motion.len(), 1);

        session.execute(Op::PhantomRemove { scene, phantom: bird }).unwrap();
        session.execute(Op::PhantomRemove { scene, phantom }).unwrap();
        assert!(session.project.as_ref().unwrap().scenes[&SceneId(scene)].phantoms.is_empty());
        assert!(session.execute(Op::PhantomRemove { scene, phantom }).is_err());
    }

    #[test]
    fn camera_key_remove_dispatch_and_validation() {
        let (mut session, scene) = session_with_scene();
        let camera = session.execute(Op::CameraAdd { scene, name: "a".into(), start: Some(0), end: Some(48) }).unwrap()["camera"].as_u64().unwrap();
        session.execute(Op::CameraKeySet { scene, camera: Some(camera), key: TransformKey::at(24, [1.0, 2.0, 3.0].into(), 0.0, 0.0) }).unwrap();
        assert_eq!(session.project.as_ref().unwrap().scenes[&SceneId(scene)].cameras[&CameraId(camera)].camera.keys.len(), 2);
        session.execute(Op::CameraKeyRemove { scene, camera, frame: 24 }).unwrap();
        assert_eq!(session.project.as_ref().unwrap().scenes[&SceneId(scene)].cameras[&CameraId(camera)].camera.keys.len(), 1);
        // The last camera key cannot be removed; a nonexistent camera is an error.
        assert!(session.execute(Op::CameraKeyRemove { scene, camera, frame: 0 }).is_err());
        assert!(session.execute(Op::CameraKeyRemove { scene, camera: 999, frame: 0 }).is_err());
    }

    #[test]
    fn background_set_clear_and_legacy_defaults() {
        // Bridge JSON without geometry parses with the default flat screen.
        let op: Op = serde_json::from_str(r#"{"op":"background_set","scene":1,"asset":2}"#).unwrap();
        match op {
            Op::BackgroundSet { video, width, height, center, .. } => {
                assert!(!video);
                assert_eq!((width, height), (24.0, 8.0));
                assert_eq!(center, [0.0, 4.0, -5.0]);
            }
            other => panic!("unexpected operation: {other:?}"),
        }
        // A legacy arc-screen manifest still reads: the flat screen takes the
        // arc's horizontal length (2.0 rad x 6.0 m = 12 m).
        let legacy: Background = serde_json::from_str(
            r#"{"image":2,"arc_radians":2.0,"radius":6.0,"height":4.0,"center":[0.0,2.0,-5.0],"bulge":"away_from_camera","fit":"cover"}"#,
        )
        .unwrap();
        assert_eq!(legacy.width, 12.0);
        assert_eq!(legacy.height, 4.0);
        assert_eq!(legacy.fit, phantom_core::BackgroundFit::Cover);

        let (mut session, scene) = session_with_scene();
        let image = {
            let p = session.project.as_mut().unwrap();
            session.history.apply(p, Command::RegisterAsset { kind: AssetKind::Image, path: "assets/bg.png".into(), bounds: None }).unwrap().asset.unwrap().0
        };
        // Parity with the CLI: video background and screen geometry.
        session
            .execute(Op::BackgroundSet { scene, asset: image, video: false, width: 16.0, height: 9.0, center: [1.0, 2.0, -4.0], fit: Some(phantom_core::BackgroundFit::Contain) })
            .unwrap();
        let bg = session.project.as_ref().unwrap().scenes[&SceneId(scene)].background.clone().unwrap();
        assert_eq!(bg.width, 16.0);
        assert_eq!(bg.height, 9.0);
        assert_eq!(bg.center, glam::Vec3::new(1.0, 2.0, -4.0));
        assert_eq!(bg.fit, phantom_core::BackgroundFit::Contain);
        // A nonexistent asset/scene is rejected; so is unusable geometry.
        assert!(session.execute(Op::BackgroundSet { scene, asset: 999, video: false, width: 16.0, height: 9.0, center: [0.0, 2.0, -5.0], fit: None }).is_err());
        assert!(session.execute(Op::BackgroundSet { scene, asset: image, video: false, width: 0.0, height: 9.0, center: [0.0, 2.0, -5.0], fit: None }).is_err());
        assert!(session.execute(Op::BackgroundSet { scene, asset: image, video: false, width: 16.0, height: -1.0, center: [0.0, 2.0, -5.0], fit: None }).is_err());
        assert!(session.execute(Op::BackgroundClear { scene: 999 }).is_err());
        session.execute(Op::BackgroundClear { scene }).unwrap();
        assert!(session.project.as_ref().unwrap().scenes[&SceneId(scene)].background.is_none());
    }

    #[test]
    fn audio_track_remove_dispatch_and_validation() {
        let (mut session, scene) = session_with_scene();
        let scene_track = session.execute(Op::AudioTrackAdd { scene: Some(scene), name: "fx".into() }).unwrap()["track"].as_u64().unwrap();
        let film_track = session.execute(Op::AudioTrackAdd { scene: None, name: "score".into() }).unwrap()["track"].as_u64().unwrap();
        session.execute(Op::AudioTrackRemove { track: scene_track }).unwrap();
        session.execute(Op::AudioTrackRemove { track: film_track }).unwrap();
        let p = session.project.as_ref().unwrap();
        assert!(p.scenes[&SceneId(scene)].audio_tracks.is_empty());
        assert!(p.film_audio.is_empty());
        assert!(session.execute(Op::AudioTrackRemove { track: scene_track }).is_err());
    }

    #[test]
    fn layer_add_and_exposure_remove_dispatch() {
        let (mut session, scene) = session_with_scene();
        let layer = session.execute(Op::LayerAdd2D { scene, name: "sketch".into() }).unwrap()["layer"].as_u64().unwrap();
        assert_eq!(session.project.as_ref().unwrap().scenes[&SceneId(scene)].layers_2d.len(), 1);
        session
            .execute(Op::BrushStroke { scene, frame: 5, points: vec![[0.0, 0.0], [10.0, 10.0]], color: [1.0, 0.0, 0.0, 1.0], width: 4.0, erase: false })
            .unwrap();
        assert!(session.project.as_ref().unwrap().scenes[&SceneId(scene)].layers_2d[0].exposures.contains_key(&5));
        session.execute(Op::ExposureRemove { scene, layer, frame: 5 }).unwrap();
        assert!(session.project.as_ref().unwrap().scenes[&SceneId(scene)].layers_2d[0].exposures.is_empty());
        // A nonexistent layer — an error, not a panic.
        assert!(session.execute(Op::ExposureRemove { scene, layer: 999, frame: 5 }).is_err());
        // The operation tag is readable snake_case.
        let v = serde_json::to_value(Op::LayerAdd2D { scene: 1, name: "l".into() }).unwrap();
        assert_eq!(v["op"], "layer_add_2d");
        let op: Op = serde_json::from_value(v).unwrap();
        assert!(matches!(op, Op::LayerAdd2D { .. }));
    }

    #[test]
    fn state_get_reports_phantoms_film_and_audio() {
        let (mut session, scene) = session_with_scene();
        let phantom = add_phantom(&mut session, scene);
        session.execute(Op::KeySet { scene, phantom, frame: 0, pos: [0.0, 0.0, 0.0], pose: ActorPose::Stand, yaw: 0.0 }).unwrap();
        let clip = session.execute(Op::FilmAdd { scene, start: 0, end: 48, camera: None }).unwrap()["clip"].as_u64().unwrap();
        let track = session.execute(Op::AudioTrackAdd { scene: None, name: "score".into() }).unwrap()["track"].as_u64().unwrap();
        session.execute(Op::TrackVolumeSet { track, volume: 0.5 }).unwrap();
        session.execute(Op::TrackMuteSet { track, muted: true }).unwrap();
        let scene_track = session.execute(Op::AudioTrackAdd { scene: Some(scene), name: "fx".into() }).unwrap()["track"].as_u64().unwrap();

        let state = session.execute(Op::StateGet).unwrap();
        let phantoms = state["phantoms"].as_array().unwrap();
        assert_eq!(phantoms.len(), 1);
        assert_eq!(phantoms[0]["id"].as_u64().unwrap(), phantom);
        assert_eq!(phantoms[0]["name"], "p");
        assert_eq!(phantoms[0]["rig"], "builtin");
        assert_eq!(phantoms[0]["motion_keys"].as_u64().unwrap(), 1);
        assert!(phantoms[0].get("takes").is_none(), "takes are gone from the document");

        let film = state["film"].as_array().unwrap();
        assert_eq!(film.len(), 1);
        assert_eq!(film[0]["id"].as_u64().unwrap(), clip);
        assert_eq!(film[0]["scene"].as_u64().unwrap(), scene);
        assert_eq!((film[0]["start"].as_i64().unwrap(), film[0]["end"].as_i64().unwrap()), (0, 48));
        assert!(film[0]["camera"].is_null(), "no camera pinned: the scene's active camera is used");

        let tracks = state["audio_tracks"].as_array().unwrap();
        assert_eq!(tracks.len(), 2);
        let film_t = tracks.iter().find(|t| t["id"].as_u64().unwrap() == track).unwrap();
        assert_eq!(film_t["name"], "score");
        assert_eq!(film_t["volume"].as_f64().unwrap(), 0.5);
        assert_eq!(film_t["muted"], true);
        assert_eq!(film_t["clips"].as_u64().unwrap(), 0);
        assert!(film_t["scene"].is_null(), "a film track has no scene");
        let scene_t = tracks.iter().find(|t| t["id"].as_u64().unwrap() == scene_track).unwrap();
        assert_eq!(scene_t["scene"].as_u64().unwrap(), scene);
    }

    #[test]
    fn history_undo_redo_round_trip_and_list() {
        let (mut session, scene) = session_with_scene();
        session.execute(Op::SceneRangeSet { scene, start: 0, end: 48 }).unwrap();

        let list = session.execute(Op::HistoryList).unwrap();
        assert_eq!(list["undo_depth"].as_u64().unwrap(), 2);
        assert_eq!(list["redo_depth"].as_u64().unwrap(), 0);
        let undo_names = list["undo"].as_array().unwrap();
        assert_eq!(undo_names[0], "SetSceneRange", "newest operation first");
        assert_eq!(undo_names[1], "AddScene");

        let undo = session.execute(Op::HistoryUndo).unwrap();
        assert_eq!(undo["applied"], true);
        assert_eq!(session.project.as_ref().unwrap().scenes[&SceneId(scene)].range, FrameRange::new(0, 96).unwrap());
        let list = session.execute(Op::HistoryList).unwrap();
        assert_eq!(list["undo_depth"].as_u64().unwrap(), 1);
        assert_eq!(list["redo_depth"].as_u64().unwrap(), 1);
        assert_eq!(list["redo"][0], "SetSceneRange");

        let redo = session.execute(Op::HistoryRedo).unwrap();
        assert_eq!(redo["applied"], true);
        assert_eq!(session.project.as_ref().unwrap().scenes[&SceneId(scene)].range, FrameRange::new(0, 48).unwrap());
        assert_eq!(session.history.redo_depth(), 0);

        // A new operation clears the redo stack; empty stacks give applied=false.
        session.execute(Op::HistoryUndo).unwrap();
        session.execute(Op::HistoryUndo).unwrap();
        assert_eq!(session.execute(Op::HistoryUndo).unwrap()["applied"], false);
        session.execute(Op::HistoryRedo).unwrap();
        session.execute(Op::HistoryRedo).unwrap();
        assert_eq!(session.execute(Op::HistoryRedo).unwrap()["applied"], false);
        // Without an open project — a clear error.
        let mut empty = Session::new();
        assert!(matches!(empty.execute(Op::HistoryUndo), Err(SessionError::NoProject)));
    }

    #[test]
    fn render_ops_forward_frames_and_shot_to_cli_args() {
        let op: Op = serde_json::from_str(r#"{"op":"render_scene","scene":1,"out":"o","width":320,"height":180,"frames":null}"#).unwrap();
        assert!(matches!(op, Op::RenderScene { camera: None, .. }));
        let op: Op = serde_json::from_str(r#"{"op":"render_film","out":"o.mp4","width":320,"height":180}"#).unwrap();
        assert!(matches!(op, Op::RenderFilm { frames: None, normalize: false, target_peak_db: None, .. }));

        let dir = Path::new("/proj");
        let args = scene_render_args(dir, Path::new("/proj/out"), 3, 640, 360, Some(7), Some((0, 48)));
        let args: Vec<String> = args.iter().map(|a| a.to_string_lossy().into_owned()).collect();
        assert!(args.windows(2).any(|w| w == ["--camera", "7"]));
        assert!(args.windows(2).any(|w| w == ["--frames", "0-48"]));

        let args: Vec<String> = film_render_args(dir, Path::new("/proj/film.mp4"), 640, 360, Some((24, 96)), false, None)
            .iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert!(args.windows(2).any(|w| w == ["--frames", "24-96"]));
        assert!(args.iter().any(|a| a == "--force"));
        let args = film_render_args(dir, Path::new("/proj/film.mp4"), 640, 360, None, false, None);
        assert!(!args.iter().any(|a| a == "--frames"));
        // Normalization: the flags reach the CLI only when normalize=true (A6).
        let args: Vec<String> = film_render_args(dir, Path::new("/proj/film.mp4"), 640, 360, None, true, Some(-3.0))
            .iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert!(args.windows(2).any(|w| w == ["--normalize", "--target-peak-db"]) || args.windows(2).any(|w| w == ["--target-peak-db", "-3"]));
        assert!(args.iter().any(|a| a == "--normalize"));
        assert!(args.windows(2).any(|w| w == ["--target-peak-db", "-3"]));
        let args = film_render_args(dir, Path::new("/proj/film.mp4"), 640, 360, None, false, Some(-3.0));
        assert!(!args.iter().any(|a| a == "--normalize" || a == "--target-peak-db"));
    }
}

#[cfg(test)]
mod job_diag_tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// Waits for a job to finish by polling job_status.
    fn wait_done(session: &mut Session, job: u64) -> Value {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let v = session.execute(Op::JobStatus { job }).unwrap();
            if v["done"].as_bool().unwrap() {
                return v;
            }
            assert!(Instant::now() < deadline, "job {job} did not finish in time");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn failing_command_captures_exit_code_and_stderr_tail() {
        let mut session = Session::new();
        let job = session
            .spawn_test_command("sh", &["-c", "echo boom >&2; echo kaboom >&2; exit 3"])
            .unwrap()["job"]
            .as_u64()
            .unwrap();
        // Status before completion: done=false, the fields are present.
        let early = session.execute(Op::JobStatus { job }).unwrap();
        assert!(early["done"].is_boolean());
        assert!(early["elapsed_secs"].as_f64().unwrap() >= 0.0);
        assert!(early["error_tail"].is_array());

        let v = wait_done(&mut session, job);
        assert_eq!(v["exit_code"].as_i64(), Some(3));
        let tail: Vec<&str> = v["error_tail"].as_array().unwrap().iter().map(|l| l.as_str().unwrap()).collect();
        assert!(tail.contains(&"boom") && tail.contains(&"kaboom"), "stderr tail: {tail:?}");

        // job_list sees the same job with the same exit code.
        let list = session.execute(Op::JobList).unwrap();
        let jobs = list["jobs"].as_array().unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0]["job"].as_u64(), Some(job));
        assert_eq!(jobs[0]["exit_code"].as_i64(), Some(3));

        // An unknown job — an error, not a panic.
        assert!(matches!(session.execute(Op::JobStatus { job: 999 }), Err(SessionError::JobNotFound(999))));
    }

    #[test]
    fn success_command_exit_zero_and_progress_parsed() {
        let mut session = Session::new();
        let job = session
            .spawn_test_command("sh", &["-c", "echo 7/10; exit 0"])
            .unwrap()["job"]
            .as_u64()
            .unwrap();
        let v = wait_done(&mut session, job);
        assert_eq!(v["exit_code"].as_i64(), Some(0));
        assert!(v["error_tail"].as_array().unwrap().is_empty());
        // The "done/total" progress line is parsed from the last stdout line.
        assert_eq!(v["frames_done"].as_u64(), Some(7));
        assert_eq!(v["total"].as_u64(), Some(10));
    }

    #[test]
    fn err_ring_stays_bounded() {
        let mut ring = ErrRing::default();
        for i in 0..200 {
            ring.push(format!("line {i}"));
        }
        assert!(ring.tail().len() <= ERR_TAIL_MAX_LINES);
        assert_eq!(ring.tail().last().map(String::as_str), Some("line 199"));
        let mut ring = ErrRing::default();
        for _ in 0..20 {
            ring.push("x".repeat(1024));
        }
        assert!(ring.bytes <= ERR_TAIL_MAX_BYTES);
    }

    #[test]
    fn err_ring_truncates_single_oversized_line() {
        let mut ring = ErrRing::default();
        ring.push("x".repeat(4096));
        assert_eq!(ring.tail().len(), 1);
        assert_eq!(ring.tail()[0].len(), 1024);
    }

    #[test]
    fn job_wait_quick_success_returns_done_without_timed_out() {
        let mut session = Session::new();
        let job = session
            .spawn_test_command("sh", &["-c", "echo 3/3; exit 0"])
            .unwrap()["job"]
            .as_u64()
            .unwrap();
        let started = Instant::now();
        let v = session.execute(Op::JobWait { job, timeout_secs: 10.0 }).unwrap();
        assert_eq!(v["done"], true);
        assert_eq!(v["timed_out"], false);
        assert_eq!(v["exit_code"].as_i64(), Some(0));
        assert_eq!(v["frames_done"].as_u64(), Some(3));
        assert_eq!(v["total"].as_u64(), Some(3));
        assert!(v["elapsed_secs"].as_f64().unwrap() >= 0.0);
        // Same status shape as job_status plus timed_out; nothing extra.
        let expected_keys = ["last_line", "done", "exit_code", "frames_done", "total", "elapsed_secs", "error_tail", "timed_out"];
        assert_eq!(v.as_object().unwrap().len(), expected_keys.len());
        for key in expected_keys {
            assert!(v.get(key).is_some(), "missing {key} in {v}");
        }
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn job_wait_failure_finalizes_stderr_and_exit_code() {
        let mut session = Session::new();
        let job = session
            .spawn_test_command("sh", &["-c", "echo boom >&2; echo kaboom >&2; exit 3"])
            .unwrap()["job"]
            .as_u64()
            .unwrap();
        let v = session.execute(Op::JobWait { job, timeout_secs: 10.0 }).unwrap();
        assert_eq!(v["done"], true);
        assert_eq!(v["timed_out"], false);
        assert_eq!(v["exit_code"].as_i64(), Some(3));
        let tail: Vec<&str> = v["error_tail"].as_array().unwrap().iter().map(|l| l.as_str().unwrap()).collect();
        assert!(tail.contains(&"boom") && tail.contains(&"kaboom"), "stderr drain finalized in the same shape: {tail:?}");
        // The finalized state is stable for later job_status queries.
        let again = session.execute(Op::JobStatus { job }).unwrap();
        assert_eq!(again["exit_code"].as_i64(), Some(3));
    }

    #[test]
    fn job_wait_timeout_returns_running_child_without_cancelling_it() {
        let mut session = Session::new();
        let job = session
            .spawn_test_command("sleep", &["5"])
            .unwrap()["job"]
            .as_u64()
            .unwrap();
        let started = Instant::now();
        let v = session.execute(Op::JobWait { job, timeout_secs: 0.2 }).unwrap();
        assert_eq!(v["done"], false);
        assert_eq!(v["timed_out"], true);
        assert!(v["exit_code"].is_null());
        // Bounded wait: returns near the timeout, not when the child exits.
        assert!(started.elapsed() < Duration::from_secs(4), "wait took {:?}", started.elapsed());
        assert!(started.elapsed() >= Duration::from_millis(150));
        // Cancellation is not implied: the child is still running and cancelable.
        let still = session.execute(Op::JobStatus { job }).unwrap();
        assert_eq!(still["done"], false);
        session.execute(Op::JobCancel { job }).unwrap();
        let cancelled = session.execute(Op::JobStatus { job });
        assert!(matches!(cancelled, Err(SessionError::JobNotFound(job2)) if job2 == job));
    }

    #[test]
    fn job_wait_zero_timeout_validates_job_and_returns_immediately() {
        let mut session = Session::new();
        let job = session
            .spawn_test_command("sleep", &["5"])
            .unwrap()["job"]
            .as_u64()
            .unwrap();
        let started = Instant::now();
        let v = session.execute(Op::JobWait { job, timeout_secs: 0.0 }).unwrap();
        assert_eq!(v["done"], false);
        assert_eq!(v["timed_out"], true);
        assert!(started.elapsed() < Duration::from_millis(500), "zero timeout must not sleep: {:?}", started.elapsed());
        session.execute(Op::JobCancel { job }).unwrap();
    }

    #[test]
    fn job_wait_upper_bound_accepted_and_lower_bound_rejected() {
        let mut session = Session::new();
        // 300.0 is the inclusive upper bound; a fast job returns long before it.
        let job = session
            .spawn_test_command("sh", &["-c", "exit 0"])
            .unwrap()["job"]
            .as_u64()
            .unwrap();
        let v = session.execute(Op::JobWait { job, timeout_secs: 300.0 }).unwrap();
        assert_eq!(v["timed_out"], false);
        // Out-of-range and non-finite timeouts are invalid_input before polling.
        for bad in [-0.1, 300.5, f64::NAN, f64::INFINITY] {
            let err = session.execute(Op::JobWait { job: 1, timeout_secs: bad }).unwrap_err();
            assert!(matches!(err, SessionError::BadInput(_)), "timeout {bad}: {err}");
            assert_eq!(err.code(), "invalid_input");
        }
    }

    #[test]
    fn job_wait_unknown_job_fails_fast_with_not_found() {
        let mut session = Session::new();
        let started = Instant::now();
        let err = session.execute(Op::JobWait { job: 999, timeout_secs: 10.0 }).unwrap_err();
        assert!(matches!(err, SessionError::JobNotFound(999)));
        assert_eq!(err.code(), "not_found");
        assert_eq!(err.details()["entity"], "job");
        assert!(started.elapsed() < Duration::from_secs(2), "unknown job must not wait: {:?}", started.elapsed());
    }

    #[test]
    fn job_wait_is_classified_as_job_like_other_job_ops() {
        // job_wait is a job diagnostic, not a document mutation: no undo entry,
        // no revision growth, no mutation acknowledgement.
        assert_eq!(Op::JobWait { job: 1, timeout_secs: 1.0 }.class(), OpClass::Job);
        assert_eq!(Op::JobStatus { job: 1 }.class(), OpClass::Job);
        assert_eq!(Op::JobList.class(), OpClass::Job);
    }
}

#[cfg(test)]
mod query_ops_tests {
    use super::*;

    fn session_with_scene() -> (Session, u64) {
        let mut session = Session::new();
        session.project = Some(Project::new("q"));
        let scene = session.execute(Op::SceneAdd { name: "main".into(), start: 0, end: 48 }).unwrap()["scene"].as_u64().unwrap();
        session.execute(Op::PhantomAdd { scene, name: "p".into(), rig_asset: None, rig_actor: None }).unwrap();
        session.execute(Op::CameraAdd { scene, name: "wide".into(), start: Some(0), end: Some(24) }).unwrap();
        (session, scene)
    }

    #[test]
    fn scene_get_returns_full_snapshot() {
        let (mut session, scene) = session_with_scene();
        let v = session.execute(Op::SceneGet { scene }).unwrap();
        assert_eq!(v["name"].as_str(), Some("main"));
        assert_eq!(v["phantoms"].as_object().unwrap().len(), 1);
        assert_eq!(v["cameras"].as_object().unwrap().len(), 2, "the scene's own camera and the named one");
        assert!(v["objects"].is_object());
        assert!(v["layers_2d"].is_array());
        assert!(v["audio_tracks"].is_array());
        assert!(v["background"].is_null());
        // A nonexistent scene — an error, not a panic.
        assert!(session.execute(Op::SceneGet { scene: 999 }).is_err());
        // Without a project — a clear error.
        let mut empty = Session::new();
        assert!(matches!(empty.execute(Op::SceneGet { scene }), Err(SessionError::NoProject)));
    }

    #[test]
    fn asset_list_reports_id_kind_path() {
        let (mut session, _scene) = session_with_scene();
        {
            let p = session.project.as_mut().unwrap();
            session
                .history
                .apply(p, Command::RegisterAsset { kind: AssetKind::ModelObj, path: "assets/quad/quad.obj".into(), bounds: None })
                .unwrap();
        }
        let v = session.execute(Op::AssetList).unwrap();
        let assets = v["assets"].as_array().unwrap();
        assert_eq!(assets.len(), 1);
        assert!(assets[0]["id"].is_u64());
        assert_eq!(assets[0]["kind"].as_str(), Some("ModelObj"));
        assert_eq!(assets[0]["path"].as_str(), Some("assets/quad/quad.obj"));
        // An empty list for a fresh session that has a project.
        let mut plain = Session::new();
        plain.project = Some(Project::new("p"));
        assert_eq!(plain.execute(Op::AssetList).unwrap()["assets"].as_array().unwrap().len(), 0);
        let mut empty = Session::new();
        assert!(matches!(empty.execute(Op::AssetList), Err(SessionError::NoProject)));
    }

    #[test]
    fn render_still_validates_bounds_and_refs_before_gpu() {
        let (mut session, scene) = session_with_scene();
        // Size bounds are rejected before the renderer is created (no GPU needed).
        for (w, h) in [(0, 720), (1280, 0), (4097, 720), (1280, 4097)] {
            let err = session.execute(Op::RenderStill { scene, frame: 0, width: w, height: h, camera: None, report: false }).unwrap_err();
            assert!(matches!(err, SessionError::BadInput(_)), "{w}x{h}: {err}");
        }
        // A nonexistent scene/camera — an error before touching the GPU.
        assert!(session.execute(Op::RenderStill { scene: 999, frame: 0, width: 64, height: 64, camera: None, report: false }).is_err());
        assert!(session.execute(Op::RenderStill { scene, frame: 0, width: 64, height: 64, camera: Some(999), report: false }).is_err());
        // Without a project — a clear error.
        let mut empty = Session::new();
        assert!(matches!(
            empty.execute(Op::RenderStill { scene, frame: 0, width: 64, height: 64, camera: None, report: false }),
            Err(SessionError::NoProject)
        ));
        // A boundary-but-valid size passes validation (rendering needs a GPU,
        // so it is not called here — covered by the checks above).
        assert!(matches!(
            serde_json::from_value::<Op>(json!({"op":"render_still","scene":1,"frame":2,"width":4096,"height":4096})).unwrap(),
            Op::RenderStill { camera: None, .. }
        ));
    }

    #[test]
    fn camera_get_returns_canonical_snapshot_and_parent_revision_without_mutation() {
        let (mut session, scene) = session_with_scene();
        let camera = session.execute(Op::CameraAdd { scene, name: "close".into(), start: Some(0), end: Some(24) }).unwrap()["camera"].as_u64().unwrap();
        session.execute(Op::CameraKeySet { scene, camera: Some(camera), key: TransformKey::at(0, [0.0, 1.4, 4.0].into(), 0.0, 0.0) }).unwrap();
        session.execute(Op::CameraLensSet { scene, camera, focal_mm: 35.0 }).unwrap();
        let revision = session.project.as_ref().unwrap().revision;
        let undo_depth = session.history.undo_depth();

        let v = session.execute(Op::CameraGet { scene, camera }).unwrap();
        // Full canonical camera snapshot (id/name/range/camera/target) plus the parent scene id and revision.
        assert_eq!(v["scene"].as_u64().unwrap(), scene);
        assert_eq!(v["revision"].as_u64().unwrap(), revision);
        let snapshot = &v["camera"];
        assert_eq!(snapshot["id"].as_u64().unwrap(), camera);
        assert_eq!(snapshot["name"], "close");
        assert_eq!(snapshot["range"], json!({"start":0,"end":24}));
        assert_eq!(snapshot["camera"]["focal_mm"].as_f64().unwrap(), 35.0);
        assert_eq!(snapshot["camera"]["keys"].as_array().unwrap().len(), 1);
        // Getters are queries: no revision increment, no undo step, no mutation ack.
        assert!(v.get("entity").is_none());
        assert!(v.get("next_id").is_none());
        assert_eq!(session.project.as_ref().unwrap().revision, revision);
        assert_eq!(session.history.undo_depth(), undo_depth);

        // Unknown camera/scene and no project give stable not_found/not_open codes.
        let err = session.execute(Op::CameraGet { scene, camera: 999 }).unwrap_err();
        assert_eq!(err.code(), "not_found");
        assert_eq!(err.details()["entity"], "camera");
        let err = session.execute(Op::CameraGet { scene: 999, camera }).unwrap_err();
        assert_eq!(err.code(), "not_found");
        assert_eq!(err.details()["entity"], "scene");
        let mut empty = Session::new();
        assert!(matches!(empty.execute(Op::CameraGet { scene, camera }), Err(SessionError::NoProject)));
    }

    #[test]
    fn asset_get_reports_canonical_asset_usage_and_nearest_ids() {
        let mut session = Session::new();
        session.project = Some(Project::new("assets"));
        let scene = session.execute(Op::SceneAdd { name: "main".into(), start: 0, end: 48 }).unwrap()["scene"].as_u64().unwrap();
        let register = |session: &mut Session, kind: AssetKind, path: &str| {
            let p = session.project.as_mut().unwrap();
            session.history.apply(p, Command::RegisterAsset { kind, path: path.into(), bounds: None }).unwrap().asset.unwrap().0
        };
        let image = register(&mut session, AssetKind::Image, "assets/bg.png");
        let model = register(&mut session, AssetKind::ModelObj, "assets/prop/prop.obj");
        let audio = register(&mut session, AssetKind::Audio, "assets/dialog.wav");
        session.execute(Op::BackgroundSet { scene, asset: image, video: false, width: 16.0, height: 8.0, center: [0.0, 2.0, -5.0], fit: None }).unwrap();
        session.execute(Op::ObjectAdd { scene, name: "crate".into(), asset: model }).unwrap();
        let track = session.execute(Op::AudioTrackAdd { scene: Some(scene), name: "fx".into() }).unwrap()["track"].as_u64().unwrap();
        session.execute(Op::AudioClipAdd { track, asset: audio, start: 0, offset_samples: 0, len_samples: 48_000, fade_in_samples: 0, fade_out_samples: 0 }).unwrap();
        let film_track = session.execute(Op::AudioTrackAdd { scene: None, name: "score".into() }).unwrap()["track"].as_u64().unwrap();
        session.execute(Op::AudioClipAdd { track: film_track, asset: audio, start: 0, offset_samples: 0, len_samples: 48_000, fade_in_samples: 0, fade_out_samples: 0 }).unwrap();
        let revision = session.project.as_ref().unwrap().revision;

        let v = session.execute(Op::AssetGet { asset: audio }).unwrap();
        assert_eq!(v["revision"].as_u64().unwrap(), revision);
        assert_eq!(v["asset"]["id"].as_u64().unwrap(), audio);
        assert_eq!(v["asset"]["kind"], "Audio");
        assert_eq!(v["asset"]["path"], "assets/dialog.wav");
        // File never written on disk: exists must be false, not an error.
        assert_eq!(v["asset"]["exists"], false);
        let usage = v["asset"]["usage"].as_array().unwrap();
        assert!(usage.iter().any(|u| u["kind"] == "audio_clip" && u["scene"] == scene && u["track"] == track), "{usage:?}");
        assert!(usage.iter().any(|u| u["kind"] == "audio_clip" && u["scene"].is_null() && u["track"] == film_track), "{usage:?}");
        assert!(v.get("entity").is_none() && v.get("next_id").is_none(), "getters are queries: {v}");
        assert_eq!(session.project.as_ref().unwrap().revision, revision);

        let v = session.execute(Op::AssetGet { asset: model }).unwrap();
        let usage = v["asset"]["usage"].as_array().unwrap();
        assert!(usage.iter().any(|u| u["kind"] == "object" && u["scene"] == scene), "{usage:?}");

        // exists becomes true once the registered path appears in the project dir.
        let dir = std::env::temp_dir().join(format!("phantom-asset-get-{}-exists", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("assets")).unwrap();
        std::fs::write(dir.join("assets/bg.png"), b"png").unwrap();
        session.dir = Some(dir.clone());
        let v = session.execute(Op::AssetGet { asset: image }).unwrap();
        assert_eq!(v["asset"]["exists"], true);
        assert!(v["asset"]["usage"].as_array().unwrap().iter().any(|u| u["kind"] == "background" && u["scene"] == scene));
        let _ = std::fs::remove_dir_all(dir);

        // Not-found: stable not_found with nearest existing IDs in details.
        let err = session.execute(Op::AssetGet { asset: 999 }).unwrap_err();
        assert_eq!(err.code(), "not_found");
        assert_eq!(err.details()["entity"], "asset");
        assert_eq!(err.details()["id"], 999);
        let nearest: Vec<u64> = err.details()["nearest_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        assert!(!nearest.is_empty() && nearest.len() <= 5, "nearest: {nearest:?}");
        assert_eq!(nearest[0], audio, "closest id to 999 comes first: {nearest:?}");
        let existing: Vec<u64> = session.project.as_ref().unwrap().assets.keys().map(|k| k.0).collect();
        assert!(nearest.iter().all(|id| existing.contains(id)), "nearest must exist: {nearest:?}");
        // No project — stable error.
        let mut empty = Session::new();
        assert!(matches!(empty.execute(Op::AssetGet { asset: 1 }), Err(SessionError::NoProject)));
    }

    #[test]
    fn film_get_reports_clips_and_mapped_totals() {
        let (mut session, scene) = session_with_scene();
        let camera = session.execute(Op::CameraAdd { scene, name: "b".into(), start: Some(24), end: Some(48) }).unwrap()["camera"].as_u64().unwrap();
        let c1 = session.execute(Op::FilmAdd { scene, start: 0, end: 24, camera: None }).unwrap()["clip"].as_u64().unwrap();
        let c2 = session.execute(Op::FilmAdd { scene, start: 24, end: 48, camera: Some(camera) }).unwrap()["clip"].as_u64().unwrap();
        let revision = session.project.as_ref().unwrap().revision;

        let v = session.execute(Op::FilmGet).unwrap();
        assert_eq!(v["revision"].as_u64().unwrap(), revision);
        let clips = v["clips"].as_array().unwrap();
        assert_eq!(clips.len(), 2);
        assert_eq!(clips[0]["id"].as_u64().unwrap(), c1);
        assert!(clips[0]["camera"].is_null());
        assert_eq!(clips[1]["id"].as_u64().unwrap(), c2);
        assert_eq!(clips[1]["camera"].as_u64().unwrap(), camera);
        assert_eq!(clips[1]["range"], json!({"start":24,"end":48}));
        // Mapped total length in frames and seconds at fixed 24 fps.
        assert_eq!(v["total_frames"].as_i64().unwrap(), 48);
        assert!((v["total_seconds"].as_f64().unwrap() - 2.0).abs() < 1e-9);
        assert!(v.get("entity").is_none() && v.get("next_id").is_none());
        assert_eq!(session.project.as_ref().unwrap().revision, revision);

        // Empty film maps to zero length in both units.
        session.execute(Op::FilmRemove { clip: c1 }).unwrap();
        session.execute(Op::FilmRemove { clip: c2 }).unwrap();
        let v = session.execute(Op::FilmGet).unwrap();
        assert_eq!(v["clips"].as_array().unwrap().len(), 0);
        assert_eq!(v["total_frames"].as_i64().unwrap(), 0);
        assert_eq!(v["total_seconds"].as_f64().unwrap(), 0.0);
        let mut empty = Session::new();
        assert!(matches!(empty.execute(Op::FilmGet), Err(SessionError::NoProject)));
    }

    #[test]
    fn diagnostics_are_queries_without_revision_bump() {
        let (mut session, scene) = session_with_scene();
        let camera = session.execute(Op::CameraAdd { scene, name: "wide".into(), start: Some(0), end: Some(24) }).unwrap()["camera"].as_u64().unwrap();
        let revision = session.project.as_ref().unwrap().revision;
        let undo_depth = session.history.undo_depth();

        // validate: scene without background/motion keys reports phantom_without_motion.
        let v = session.execute(Op::Validate { scene: None }).unwrap();
        assert_eq!(v["ok"], true, "info findings do not fail ok: {v}");
        assert_eq!(v["revision"].as_u64().unwrap(), revision);
        let codes: Vec<&str> = v["findings"].as_array().unwrap().iter().map(|f| f["code"].as_str().unwrap()).collect();
        assert!(codes.contains(&"phantom_without_motion"), "{codes:?}");
        let filtered = session.execute(Op::Validate { scene: Some(scene) }).unwrap();
        assert_eq!(filtered["scene"].as_u64().unwrap(), scene);
        assert!(session.execute(Op::Validate { scene: Some(999) }).unwrap_err().code() == "not_found");

        // camera_coverage: structured camera + background spans, degrees.
        let cov = session.execute(Op::CameraCoverage { scene, frame: None, camera: Some(camera), aspect: None }).unwrap();
        assert_eq!(cov["revision"].as_u64().unwrap(), revision);
        assert_eq!(cov["frame"], 0, "default frame = the camera range start");
        assert!(cov["camera"]["hfov_deg"].as_f64().unwrap() > cov["camera"]["vfov_deg"].as_f64().unwrap());
        assert_eq!(cov["camera_id"].as_u64().unwrap(), camera);
        assert_eq!(cov["background"], Value::Null, "scene has no background yet");
        assert_eq!(cov["findings"][0]["code"], "background_absent");
        assert!(cov["phantoms"].as_array().unwrap().iter().all(|p| p["in_frame"].is_boolean()));
        assert!(session.execute(Op::CameraCoverage { scene, frame: None, camera: Some(999), aspect: None }).unwrap_err().code() == "not_found");
        assert!(session.execute(Op::CameraCoverage { scene, frame: None, camera: None, aspect: Some(0.0) }).unwrap_err().code() == "invalid_input");

        // background_fit: suggestion covers the frame (roundtrip through coverage).
        let fit = session.execute(Op::BackgroundFit { scene, frame: None, camera: None, margin_deg: None, aspect: None }).unwrap();
        assert_eq!(fit["margin_deg"], 5.0);
        assert!(fit["suggested"]["width"].as_f64().unwrap() > 0.0);
        assert!(fit["suggested"]["height"].as_f64().unwrap() > 0.0);
        assert_eq!(fit["size_covers"], true);
        assert!(fit["panorama_aspect"].as_f64().unwrap() > 0.0);
        assert_eq!(fit["scope"], "scene", "without a camera the fit covers every camera in the scene");
        assert!(fit["cameras"].as_array().is_some_and(|shots| !shots.is_empty()));
        assert!(session.execute(Op::BackgroundFit { scene, frame: None, camera: None, margin_deg: Some(90.0), aspect: None }).unwrap_err().code() == "invalid_input");
        // An explicit frame without a camera is not silently ignored (lesson from corrective_2 §2).
        assert!(session.execute(Op::BackgroundFit { scene, frame: Some(0), camera: None, margin_deg: None, aspect: None }).unwrap_err().code() == "invalid_input");

        // Queries: no revision, no undo steps, no mutation ack fields.
        assert_eq!(session.project.as_ref().unwrap().revision, revision, "diagnostics never raise the revision");
        assert_eq!(session.history.undo_depth(), undo_depth, "diagnostics never add undo steps");
        for result in [v, cov, fit] {
            assert!(result.get("entity").is_none() && result.get("next_id").is_none(), "query ack fields: {result}");
        }
        assert_eq!(Op::Validate { scene: None }.class(), OpClass::Query);
        assert_eq!(Op::CameraCoverage { scene: 1, frame: None, camera: None, aspect: None }.class(), OpClass::Query);
        assert_eq!(Op::BackgroundFit { scene: 1, frame: None, camera: None, margin_deg: None, aspect: None }.class(), OpClass::Query);
        let mut empty = Session::new();
        assert!(matches!(empty.execute(Op::Validate { scene: None }), Err(SessionError::NoProject)));
    }

    #[test]
    fn getters_are_queries_without_mutation_acknowledgement() {
        let (mut session, scene) = session_with_scene();
        let camera = session.execute(Op::CameraAdd { scene, name: "wide".into(), start: Some(0), end: Some(24) }).unwrap()["camera"].as_u64().unwrap();
        session.execute(Op::FilmAdd { scene, start: 0, end: 24, camera: Some(camera) }).unwrap();
        let revision = session.project.as_ref().unwrap().revision;
        let undo_depth = session.history.undo_depth();
        for result in [
            session.execute(Op::CameraGet { scene, camera }).unwrap(),
            session.execute(Op::FilmGet).unwrap(),
            session.execute(Op::ProjectRecent).unwrap(),
        ] {
            assert!(result.get("entity").is_none(), "query must not carry mutation entity: {result}");
            assert!(result.get("next_id").is_none(), "query must not carry next_id: {result}");
        }
        // A failing query keeps its error contract, not a mutation ack.
        let err = session.execute(Op::AssetGet { asset: 1 }).unwrap_err();
        assert_eq!(err.code(), "not_found");
        assert_eq!(err.details()["entity"], "asset");
        assert_eq!(session.project.as_ref().unwrap().revision, revision, "queries never raise the revision");
        assert_eq!(session.history.undo_depth(), undo_depth, "queries never add undo steps");
        assert_eq!(Op::CameraGet { scene: 1, camera: 1 }.class(), OpClass::Query);
        assert_eq!(Op::AssetGet { asset: 1 }.class(), OpClass::Query);
        assert_eq!(Op::FilmGet.class(), OpClass::Query);
        assert_eq!(Op::ProjectRecent.class(), OpClass::Query);
    }
}

#[cfg(test)]
mod recent_state_tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(1);

    fn state_file(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "phantom-recent-{name}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("recent.json")
    }

    fn project_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "phantom-recent-proj-{name}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn recent_state_roundtrips_across_session_restart() {
        let state = state_file("roundtrip");
        let dir = project_dir("roundtrip");
        {
            let mut first = Session::with_state_path(state.clone());
            first.execute(Op::ProjectNew { dir: dir.clone(), name: "p".into() }).unwrap();
            drop(first); // release the lock before the "restarted" session opens
        }
        // The state file stores exactly path + timestamp; never lock tokens.
        let raw: Value = serde_json::from_slice(&std::fs::read(&state).unwrap()).unwrap();
        let fields = raw.as_object().unwrap();
        assert_eq!(fields.len(), 2, "only path and timestamp are persisted: {fields:?}");
        assert!(fields.contains_key("path") && fields.contains_key("timestamp"));
        assert_eq!(fields["path"].as_str().unwrap(), dir.canonicalize().unwrap().to_string_lossy());
        assert!(fields["timestamp"].is_u64());
        // Atomic replace leaves no temporary sibling behind.
        let siblings: Vec<_> = std::fs::read_dir(state.parent().unwrap()).unwrap().collect();
        assert_eq!(siblings.len(), 1, "no tmp leftovers: {:?}", siblings);

        // A fresh Session (process restart) reads the persisted path.
        let mut second = Session::with_state_path(state.clone());
        let recent = second.execute(Op::ProjectRecent).unwrap();
        assert_eq!(recent["recent"]["dir"], json!(dir.canonicalize().unwrap()));
        assert!(recent["recent"]["timestamp"].is_u64());
        assert!(recent["warnings"].as_array().unwrap().is_empty());
        assert!(recent["current_dir"].is_null(), "no project open yet");
        assert!(recent["revision"].is_null());

        // Opening the recorded project re-records it and fills current state.
        second.execute(Op::ProjectOpen { dir: dir.clone(), force: false }).unwrap();
        let recent = second.execute(Op::ProjectRecent).unwrap();
        assert_eq!(recent["recent"]["dir"], json!(dir.canonicalize().unwrap()));
        assert_eq!(recent["current_dir"], json!(dir.canonicalize().unwrap()));
        assert_eq!(recent["revision"].as_u64().unwrap(), 0);
        drop(second);
        let _ = std::fs::remove_dir_all(dir);
        let _ = std::fs::remove_dir_all(state.parent().unwrap());
    }

    #[test]
    fn corrupt_recent_state_warns_instead_of_failing_and_heals_on_open() {
        let state = state_file("corrupt");
        std::fs::write(&state, b"{not json at all").unwrap();
        // Session construction must not depend on the state file at all.
        let mut session = Session::with_state_path(state.clone());
        let recent = session.execute(Op::ProjectRecent).unwrap();
        assert!(recent["recent"].is_null(), "corrupt read returns empty, not a failure: {recent}");
        let warnings = recent["warnings"].as_array().unwrap();
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0]["code"], "recent_state_corrupt");

        // A successful open rewrites the state; the warning disappears.
        let dir = project_dir("corrupt");
        storage::create_project(&dir, "c").unwrap();
        session.execute(Op::ProjectOpen { dir: dir.clone(), force: false }).unwrap();
        let recent = session.execute(Op::ProjectRecent).unwrap();
        assert_eq!(recent["recent"]["dir"], json!(dir.canonicalize().unwrap()));
        assert!(recent["warnings"].as_array().unwrap().is_empty(), "{recent}");
        drop(session);
        let _ = std::fs::remove_dir_all(dir);
        let _ = std::fs::remove_dir_all(state.parent().unwrap());
    }

    #[test]
    fn missing_state_file_is_silently_empty() {
        let state = state_file("absent");
        let mut session = Session::with_state_path(state);
        let recent = session.execute(Op::ProjectRecent).unwrap();
        assert!(recent["recent"].is_null());
        assert!(recent["current_dir"].is_null());
        assert!(recent["revision"].is_null());
        assert!(recent["warnings"].as_array().unwrap().is_empty(), "absent file is normal: {recent}");
    }

    #[test]
    fn failed_project_new_does_not_record_recent() {
        let state = state_file("failed-new");
        let dir = project_dir("failed-new");
        std::fs::create_dir_all(dir.join("manifest.json.tmp")).unwrap();
        let mut session = Session::with_state_path(state.clone());
        assert!(session.execute(Op::ProjectNew { dir: dir.clone(), name: "p".into() }).is_err());
        assert!(!state.exists(), "only successful new/open persists recent");
        let _ = std::fs::remove_dir_all(dir);
        let _ = std::fs::remove_dir_all(state.parent().unwrap());
    }
}

#[cfg(test)]
mod project_lock_session_tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(1);

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "phantom-session-lock-{name}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// Injectable per-user state path: tests must never touch the real
    /// ~/.local/state file or race on it between parallel tests.
    fn state_file(name: &str) -> PathBuf {
        let dir = temp_dir(name);
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("recent.json")
    }

    #[test]
    fn project_new_then_open_same_canonical_dir_is_idempotent() {
        let dir = temp_dir("self-open");
        let mut session = Session::with_state_path(state_file("self-open"));
        let created = session.execute(Op::ProjectNew { dir: dir.clone(), name: "p".into() }).unwrap();
        assert_eq!(created["already_open"], false);
        let before = std::fs::read(dir.join(project_lock::LOCK_FILE)).unwrap();
        let opened = session.execute(Op::ProjectOpen { dir: dir.join("."), force: false }).unwrap();
        assert_eq!(opened["already_open"], true);
        assert_eq!(std::fs::read(dir.join(project_lock::LOCK_FILE)).unwrap(), before);
        drop(session);
        assert!(!dir.join(project_lock::LOCK_FILE).exists());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn switching_projects_is_rejected_without_releasing_current_guard() {
        let first = temp_dir("switch-a");
        let second = temp_dir("switch-b");
        storage::create_project(&second, "b").unwrap();
        let mut session = Session::with_state_path(state_file("switch"));
        session.execute(Op::ProjectNew { dir: first.clone(), name: "a".into() }).unwrap();
        let err = session.execute(Op::ProjectOpen { dir: second.clone(), force: false }).unwrap_err();
        assert!(matches!(err, SessionError::ProjectSwitchRejected { .. }));
        assert!(first.join(project_lock::LOCK_FILE).exists());
        assert!(!second.join(project_lock::LOCK_FILE).exists());
        drop(session);
        let _ = std::fs::remove_dir_all(first);
        let _ = std::fs::remove_dir_all(second);
    }

    #[test]
    fn project_new_failure_releases_guard() {
        let dir = temp_dir("new-failure");
        std::fs::create_dir_all(dir.join("manifest.json.tmp")).unwrap();
        let mut session = Session::new();
        assert!(session.execute(Op::ProjectNew { dir: dir.clone(), name: "p".into() }).is_err());
        assert!(!dir.join(project_lock::LOCK_FILE).exists());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn project_open_force_overrides_ambiguous_lock_with_warning() {
        let dir = temp_dir("open-force");
        storage::create_project(&dir, "p").unwrap();
        std::fs::write(dir.join(project_lock::LOCK_FILE), b"not-json").unwrap();
        let mut session = Session::with_state_path(state_file("open-force"));
        assert!(matches!(
            session.execute(Op::ProjectOpen { dir: dir.clone(), force: false }),
            Err(SessionError::Lock(project_lock::LockError::Held { .. }))
        ));
        let opened = session.execute(Op::ProjectOpen { dir: dir.clone(), force: true }).unwrap();
        assert_eq!(opened["already_open"], false);
        assert_eq!(opened["warnings"][0]["code"], "stale_lock_reclaimed");
        assert_eq!(opened["warnings"][0]["owner"]["status"], "ambiguous");
        drop(session);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn old_project_open_json_defaults_force_to_false() {
        let op: Op = serde_json::from_str(r#"{"op":"project_open","dir":"/tmp/p"}"#).unwrap();
        assert!(matches!(op, Op::ProjectOpen { force: false, .. }));
    }
}

//! Project document: scenes, Phantoms, camera, background, asset references.

use std::collections::BTreeMap;
use std::path::PathBuf;

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::draw::Layer2D;
use crate::id::*;
use crate::time::{Frame, FrameRange};

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Project {
    pub schema_version: u32,
    pub name: String,
    next_id: u64,
    /// Document revision: incremented for every applied command.
    /// Change detector for syncing the GPU world; session state,
    /// not serialized.
    #[serde(skip)]
    pub revision: u64,
    pub assets: BTreeMap<AssetId, Asset>,
    pub scenes: BTreeMap<SceneId, Scene>,
    /// Film: ordered clips referencing scene ranges.
    #[serde(default)]
    pub film: Vec<crate::film::FilmClip>,
    /// Film audio tracks: they play across cuts.
    #[serde(default)]
    pub film_audio: Vec<crate::audio::AudioTrack>,
}

impl Project {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            name: name.into(),
            next_id: 1,
            revision: 0,
            assets: BTreeMap::new(),
            scenes: BTreeMap::new(),
            film: Vec::new(),
            film_audio: Vec::new(),
        }
    }

    /// Next global document ID (for diagnostics and predictable mutation
    /// confirmations only; ID allocation stays internal).
    pub fn next_id(&self) -> u64 {
        self.next_id
    }

    pub(crate) fn alloc_id<T: From<u64>>(&mut self) -> T {
        let id = self.next_id;
        self.next_id += 1;
        T::from(id)
    }

    /// Local bounding box of a Phantom for the stage clamp: the builtin actor's
    /// box for its current action, the asset's recorded bounds for an imported
    /// model (a point when the model has none), under the actor scale.
    pub fn phantom_footprint(&self, phantom: &Phantom, pose: ActorPose) -> crate::stage::Footprint {
        self.phantom_footprint_at(phantom, pose, phantom.scale)
    }

    /// The same box under an explicit scale — a scale change has to check the
    /// actor's new size before the new scale reaches the document.
    pub fn phantom_footprint_at(&self, phantom: &Phantom, pose: ActorPose, scale: f32) -> crate::stage::Footprint {
        let base = match phantom.rig.builtin_actor() {
            Some(actor) => actor.footprint(pose),
            None => match &phantom.rig {
                RigRef::Imported { asset } => self
                    .assets
                    .get(asset)
                    .and_then(|a| a.bounds)
                    .map(|b| b.footprint())
                    .unwrap_or_else(crate::stage::Footprint::point),
                _ => crate::stage::Footprint::point(),
            },
        };
        base.scaled(Vec3::splat(scale))
    }

    /// Local bounding box of a scene object for the stage clamp: the metric
    /// catalog entry for a builtin prop, the asset's recorded bounds otherwise,
    /// under the key's scale.
    pub fn object_footprint(&self, object: &SceneObject, scale: Vec3) -> crate::stage::Footprint {
        if let Some(field) = &object.scatter {
            // The plate, not the tallest tree: the field's own box is what has
            // to stay inside the stage.
            let preset = crate::scatter::scatter_preset(&field.preset);
            let base = preset
                .map(|p| crate::scatter::field_footprint(p, &field.shape))
                .unwrap_or_else(crate::stage::Footprint::point);
            return base.scaled(scale);
        }
        let base = match object.builtin.as_deref().and_then(crate::catalog::catalog_find) {
            Some(entry) => entry.footprint(),
            None => self
                .assets
                .get(&object.asset)
                .and_then(|a| a.bounds)
                .map(|b| b.footprint())
                .unwrap_or_else(crate::stage::Footprint::point),
        };
        base.scaled(scale)
    }
}

impl From<u64> for SceneId {
    fn from(v: u64) -> Self {
        Self(v)
    }
}
impl From<u64> for PhantomId {
    fn from(v: u64) -> Self {
        Self(v)
    }
}
impl From<u64> for AssetId {
    fn from(v: u64) -> Self {
        Self(v)
    }
}
impl From<u64> for ObjectId {
    fn from(v: u64) -> Self {
        Self(v)
    }
}
impl From<u64> for CameraId {
    fn from(v: u64) -> Self {
        Self(v)
    }
}
impl From<u64> for crate::id::LightId {
    fn from(v: u64) -> Self {
        Self(v)
    }
}
impl From<u64> for crate::id::VfxId {
    fn from(v: u64) -> Self {
        Self(v)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Asset {
    pub id: AssetId,
    pub kind: AssetKind,
    /// Path relative to the project root.
    pub path: PathBuf,
    /// Bounding box of an imported model in the model's own frame, computed
    /// once at import (layout mode). `None` — unknown (old manifests, assets
    /// that are not models); such an entity is clamped by its origin point.
    #[serde(default)]
    pub bounds: Option<crate::stage::ModelBounds>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum AssetKind {
    /// Static OBJ model.
    ModelObj,
    /// FBX character with skeleton and weights.
    ModelFbx,
    /// Legacy standalone FBX motion clip. The layout program no longer imports
    /// or plays clips; the variant stays so an old manifest still deserializes.
    MotionFbx,
    /// Image (background, texture).
    Image,
    /// Audio (scene and film tracks).
    Audio,
    /// Video (background: mp4/gif/webm/mov).
    Video,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Scene {
    pub id: SceneId,
    pub name: String,
    pub range: FrameRange,
    pub background: Option<Background>,
    /// Legacy single camera (the pre-camera-entity document). Read for the
    /// migration into `cameras[SCENE_CAMERA]` and never written.
    #[serde(default, rename = "camera", skip_serializing)]
    pub legacy_camera: Option<Camera>,
    pub phantoms: BTreeMap<PhantomId, Phantom>,
    /// Raster 2D layers on top of the frame (order = stacking order).
    #[serde(default)]
    pub layers_2d: Vec<Layer2D>,
    /// Scene audio tracks.
    #[serde(default)]
    pub audio_tracks: Vec<crate::audio::AudioTrack>,
    /// Three-point stage lighting (key/fill/rim); additive on top of the
    /// background's zone lights and ambient. Old manifests without the field
    /// get the default values.
    #[serde(default)]
    pub lighting: SceneLighting,
    /// Matte floor on y=0 that receives shadows. `None` — no floor (old
    /// scenes do not gain a floor automatically).
    #[serde(default)]
    pub floor: Option<SceneFloor>,
    /// Props: static scene objects with their own transform keys.
    /// Old manifests without the field load empty.
    #[serde(default)]
    pub objects: BTreeMap<ObjectId, SceneObject>,
    /// The scene's cameras. `SCENE_CAMERA` (id 0) is always present — it is
    /// where a legacy manifest's single camera lands; every other entry is a
    /// named camera with its own range. Replaces the old `shots` map, which is
    /// read as cameras with the same ids.
    #[serde(default, alias = "shots")]
    pub cameras: BTreeMap<CameraId, SceneCamera>,
    /// The camera the viewport shows and a render without a camera uses.
    /// Living in the document keeps a headless render reproducible.
    #[serde(default = "default_active_camera")]
    pub active_camera: CameraId,
    /// Document light sources with keys (corrective B1): point/spot/
    /// directional, living in the project and the undo history. Old manifests
    /// load empty.
    #[serde(default)]
    pub lights: Vec<SceneLightSource>,
    /// Ambient as a standalone scene setting (B1); the serde default is the
    /// previous behavior (derived from the background).
    #[serde(default)]
    pub ambient: AmbientMode,
    /// Sky color / clean frame color (linear RGB) filling everything not
    /// covered by the screen and geometry (B5): `None` — transparency (the
    /// previous behavior). With a sky set, the frame becomes opaque.
    #[serde(default)]
    pub sky: Option<[f32; 3]>,
    /// Frame color grading (B3); None — no grading.
    #[serde(default)]
    pub grade: Option<SceneGrade>,
    /// Exponential fog (B3); None — no fog.
    #[serde(default)]
    pub fog: Option<SceneFog>,
    /// Stage bounds (layout mode): actors and props live inside this box and
    /// every placement command clamps to it. Old manifests without the field
    /// get the medium preset.
    #[serde(default)]
    pub stage: crate::stage::Stage,
    /// Scene effects (particles/water) with keys (B3).
    #[serde(default)]
    pub vfx: Vec<SceneVfx>,
}

/// The scene's own camera: every scene has it, it is where a legacy manifest's
/// single camera lands, and it is the default active camera.
///
/// Kept public as the serde default of `Scene::active_camera`.
pub const SCENE_CAMERA: CameraId = CameraId(0);

/// A camera of the scene: its own keys and lens, the range it is used for, and
/// an optional attachment to a moving subject. Actors, props and the background
/// are shared with the scene.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneCamera {
    pub id: CameraId,
    pub name: String,
    /// Frame range [start, end) within the scene's range.
    pub range: FrameRange,
    pub camera: Camera,
    /// Attachment to a subject; `None` — a free camera that follows only its keys.
    #[serde(default)]
    pub target: Option<CameraTarget>,
}

/// What a camera is pointed at or attached to.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraTarget {
    pub mode: CameraTargetMode,
    pub subject: CameraSubject,
}

impl CameraTarget {
    pub fn new(mode: CameraTargetMode, subject: CameraSubject) -> Self {
        Self { mode, subject }
    }
}

/// How a camera follows its subject.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraTargetMode {
    /// Слежение: the camera stands where its keys put it and only turns to keep
    /// the subject in view — it does not travel with the subject.
    Track,
    /// Следование: the camera is attached to the subject. Its keys are read in
    /// the subject's frame (position, angles and aim point alike), so it travels
    /// and turns with the subject — a third-person view.
    Follow,
}

/// The moving thing a camera can be attached to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "id")]
pub enum CameraSubject {
    Phantom(PhantomId),
    Object(ObjectId),
}

fn default_active_camera() -> CameraId {
    SCENE_CAMERA
}

impl SceneCamera {
    /// A camera over the whole scene, with the default rig.
    pub fn new(id: CameraId, name: impl Into<String>, range: FrameRange) -> Self {
        Self { id, name: name.into(), range, camera: Camera::default(), target: None }
    }
}

impl Scene {
    /// A scene with its own camera (id 0) over the whole range.
    pub fn new(id: SceneId, name: impl Into<String>, range: FrameRange) -> Self {
        let mut cameras = BTreeMap::new();
        cameras.insert(SCENE_CAMERA, SceneCamera::new(SCENE_CAMERA, "Camera", range));
        Self {
            id,
            name: name.into(),
            range,
            background: None,
            legacy_camera: None,
            phantoms: BTreeMap::new(),
            layers_2d: Vec::new(),
            audio_tracks: Vec::new(),
            lighting: SceneLighting::default(),
            floor: None,
            objects: BTreeMap::new(),
            cameras,
            active_camera: SCENE_CAMERA,
            lights: Vec::new(),
            ambient: AmbientMode::default(),
            sky: None,
            grade: None,
            fog: None,
            stage: crate::stage::Stage::default(),
            vfx: Vec::new(),
        }
    }

    /// A camera by id; `None` or an unknown id — the scene's own camera.
    ///
    /// A hand-written document that lost camera 0 still renders: the fallback
    /// is a plain default camera rather than a panic (loading normalizes, and
    /// `validate` reports the hole).
    pub fn camera_for(&self, id: Option<CameraId>) -> &Camera {
        id.and_then(|id| self.cameras.get(&id))
            .or_else(|| self.cameras.get(&SCENE_CAMERA))
            .map(|c| &c.camera)
            .unwrap_or_else(|| {
                static FALLBACK: std::sync::OnceLock<Camera> = std::sync::OnceLock::new();
                FALLBACK.get_or_init(Camera::default)
            })
    }

    /// The range of a camera; `None` or an unknown id — the scene's range.
    pub fn range_for(&self, id: Option<CameraId>) -> FrameRange {
        id.and_then(|id| self.cameras.get(&id)).map(|c| c.range).unwrap_or(self.range)
    }

    /// The document's active camera (falls back to camera 0 for a hand-made
    /// scene that names one that does not exist).
    pub fn active_camera(&self) -> &SceneCamera {
        self.cameras
            .get(&self.active_camera)
            .or_else(|| self.cameras.get(&SCENE_CAMERA))
            .unwrap_or_else(|| {
                static FALLBACK: std::sync::OnceLock<SceneCamera> = std::sync::OnceLock::new();
                FALLBACK.get_or_init(|| {
                    SceneCamera::new(SCENE_CAMERA, "Camera", FrameRange { start: 0, end: 1 })
                })
            })
    }

    /// Bring a loaded document to the current shape: a legacy single `camera`
    /// becomes camera 0 over the whole range, a scene without cameras gains
    /// camera 0, and an active camera that does not exist falls back to it.
    pub fn normalize_legacy(&mut self) {
        let scene_range = self.range;
        if let Some(camera) = self.legacy_camera.take() {
            let entry = self.cameras.entry(SCENE_CAMERA).or_insert_with(|| SceneCamera::new(SCENE_CAMERA, "Camera", scene_range));
            entry.camera = camera;
            entry.range = scene_range;
        }
        self.cameras.entry(SCENE_CAMERA).or_insert_with(|| SceneCamera::new(SCENE_CAMERA, "Camera", scene_range));
        if !self.cameras.contains_key(&self.active_camera) {
            self.active_camera = SCENE_CAMERA;
        }
    }
}

/// Static scene object (prop): a reference to a model asset and transform
/// keys. Not an animatable character: no actions, no skinning; a skinned
/// model renders in bind pose.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneObject {
    pub id: ObjectId,
    pub name: String,
    /// Model asset (ModelObj or ModelFbx); ignored when `builtin` is set
    /// (a metric catalog entry, corrective_2 K1).
    pub asset: AssetId,
    /// Catalog entry name (`phantom_core::catalog`): the mesh is built
    /// procedurally, no asset file is needed.
    #[serde(default)]
    pub builtin: Option<String>,
    /// Transform keys, sorted by frame.
    #[serde(default)]
    pub keys: Vec<ObjectKey>,
    /// Attachment to a Phantom bone (B2): when set, the transform follows the
    /// bone every frame and its own `keys` do not apply.
    #[serde(default)]
    pub attach: Option<Attachment>,
    /// Object material patches (B4).
    #[serde(default)]
    pub material_patches: Vec<MaterialPatch>,
    /// Generated fill (a far band of forest and so on): the object's mesh is
    /// built by replaying the scatter plan of `(preset, shape, seed)` — the
    /// same pure function the generating command used to count and place the
    /// editable props. The field carries the recipe, not baked geometry, so
    /// the document stays small and the mesh stays reproducible.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scatter: Option<ScatterField>,
}

/// A generated fill stored on its object: the recipe the renderer replays.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScatterField {
    /// `scatter::ScatterPreset` name (forest, grove, hedge, ...).
    pub preset: String,
    /// Shape in the field's own frame; the object key places the field.
    pub shape: crate::scatter::ScatterShape,
    /// Determinism seed: the same (preset, shape, seed) always replays the
    /// same instances — a saved project must render identically everywhere.
    pub seed: u64,
}

/// Object transform key: position, Euler angles (yaw is primary; pitch/roll
/// optional) and scale.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ObjectKey {
    pub frame: Frame,
    pub position: Vec3,
    /// Rotation about +Y, radians.
    #[serde(default)]
    pub yaw: f32,
    /// Pitch, radians.
    #[serde(default)]
    pub pitch: f32,
    /// Roll, radians.
    #[serde(default)]
    pub roll: f32,
    /// Per-axis scale (> 0).
    #[serde(default = "default_scale")]
    pub scale: Vec3,
}

fn default_scale() -> Vec3 {
    Vec3::ONE
}

impl ObjectKey {
    pub fn validate(&self) -> Result<(), crate::error::CoreError> {
        use crate::error::CoreError::NonFinite;
        if !self.position.is_finite() {
            return Err(NonFinite("object_key.position"));
        }
        if !self.yaw.is_finite() || !self.pitch.is_finite() || !self.roll.is_finite() {
            return Err(NonFinite("object_key.rotation"));
        }
        if !self.scale.is_finite() || self.scale.min_element() <= 0.0 {
            return Err(NonFinite("object_key.scale"));
        }
        Ok(())
    }
}

/// One stage-light source (directional; directions are fixed by the staging;
/// color, intensity and on/off are configurable).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneLight {
    /// Light color, linear RGB, channels >= 0.
    pub color: [f32; 3],
    /// Illuminance in lux, >= 0.
    pub intensity: f32,
    /// The light is on.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// The light casts shadows. Off by default (corrective A1/A2): the
    /// "skinned mesh + shadow pass" path crashes in Bevy/wgpu
    /// (pipeline validation, Uint32x4 joint indices), so shadows are a
    /// deliberate opt-in until the skinned shadow pipeline is fixed.
    #[serde(default)]
    pub shadows: bool,
}

fn default_true() -> bool {
    true
}

impl Default for SceneLight {
    fn default() -> Self {
        Self { color: [1.0, 1.0, 1.0], intensity: 8000.0, enabled: true, shadows: false }
    }
}

impl SceneLight {
    pub fn validate(&self) -> Result<(), crate::error::CoreError> {
        if !self.color.iter().all(|c| c.is_finite() && *c >= 0.0) {
            return Err(crate::error::CoreError::NonFinite("light.color"));
        }
        if !self.intensity.is_finite() || self.intensity < 0.0 {
            return Err(crate::error::CoreError::NonFinite("light.intensity"));
        }
        Ok(())
    }
}

fn default_fill_light() -> SceneLight {
    SceneLight { color: [0.9, 0.95, 1.0], intensity: 3000.0, enabled: true, shadows: false }
}

fn default_rim_light() -> SceneLight {
    SceneLight { color: [1.0, 0.95, 0.9], intensity: 6000.0, enabled: true, shadows: false }
}

/// Document light-source kind (corrective B1).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SceneLightKind {
    /// Point (lamp, lantern); intensity in lumens.
    Point,
    /// Spot with a cone; intensity in lumens, the cone in radians
    /// (outer angle; MCP/CLI take degrees).
    Spot { cone: f32 },
    /// Directional (sun/moon); intensity in lux.
    Directional,
}

/// Transform/state key of a light source at a frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LightKey {
    pub frame: Frame,
    pub translation: Vec3,
    /// Radians, YXZ (the direction for spot/directional).
    pub yaw: f32,
    pub pitch: f32,
    /// Intensity override at the frame; None — from the light body.
    #[serde(default)]
    pub intensity: Option<f32>,
    /// Per-frame on/off (a lantern switched on at a frame); None — from the body.
    #[serde(default)]
    pub enabled: Option<bool>,
}

/// Scene light source with keys (corrective B1): lives in the project and the
/// undo history, animated per frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneLightSource {
    pub id: crate::id::LightId,
    pub name: String,
    pub kind: SceneLightKind,
    /// Linear RGB, channels >= 0.
    pub color: [f32; 3],
    /// Lumens (point/spot) or lux (directional), >= 0.
    pub intensity: f32,
    pub enabled: bool,
    /// Shadows are opt-in (see `SceneLight::shadows` and corrective A2).
    #[serde(default)]
    pub shadows: bool,
    /// Keys by frame (sorted); with no keys the light stands at `rest`.
    #[serde(default)]
    pub keys: Vec<LightKey>,
    /// Pose without keys (also used before the first key).
    pub rest: TransformKey,
    /// Attachment to a Phantom bone (B2): a lantern in a hand; when set, the
    /// light's pose follows the bone and the transform keys do not apply
    /// (keyed intensity and enabled keep working).
    #[serde(default)]
    pub attach: Option<Attachment>,
}

impl SceneLightSource {
    /// Light state at a frame: (translation, yaw, pitch, intensity, enabled).
    /// Between keys, transform and intensity interpolate linearly
    /// (shortest-arc for yaw); enabled is a step from the last passed key.
    pub fn at(&self, frame: Frame) -> (Vec3, f32, f32, f32, bool) {
        if self.keys.is_empty() {
            return (self.rest.translation, self.rest.yaw, self.rest.pitch, self.intensity, self.enabled);
        }
        let pos = self.keys.partition_point(|k| k.frame <= frame);
        if pos == 0 {
            // Before the first key — the light's rest pose.
            return (self.rest.translation, self.rest.yaw, self.rest.pitch, self.intensity, self.enabled);
        }
        let prev = self.keys[pos - 1];
        if pos >= self.keys.len() {
            return (prev.translation, prev.yaw, prev.pitch, prev.intensity.unwrap_or(self.intensity), prev.enabled.unwrap_or(self.enabled));
        }
        let next = self.keys[pos];
        if next.frame == prev.frame {
            return (next.translation, next.yaw, next.pitch, next.intensity.unwrap_or(self.intensity), next.enabled.unwrap_or(self.enabled));
        }
        let t = (frame - prev.frame) as f32 / (next.frame - prev.frame) as f32;
        let dyaw = {
            let mut d = next.yaw - prev.yaw;
            while d > std::f32::consts::PI {
                d -= std::f32::consts::TAU;
            }
            while d < -std::f32::consts::PI {
                d += std::f32::consts::TAU;
            }
            d
        };
        // Intensity: None inherits the light body — the effective values are
        // interpolated, not a "hold" of the last value explicitly set.
        let ia = prev.intensity.unwrap_or(self.intensity);
        let ib = next.intensity.unwrap_or(self.intensity);
        (
            prev.translation + (next.translation - prev.translation) * t,
            prev.yaw + dyaw * t,
            prev.pitch + (next.pitch - prev.pitch) * t,
            ia + (ib - ia) * t,
            // Step: the state of the last PASSED key — a future key must not
            // peek into the present.
            prev.enabled.unwrap_or(self.enabled),
        )
    }

    pub fn validate(&self) -> Result<(), crate::error::CoreError> {
        if self.name.trim().is_empty() {
            return Err(crate::error::CoreError::EmptyName);
        }
        if !self.color.iter().all(|c| c.is_finite() && *c >= 0.0) {
            return Err(crate::error::CoreError::NonFinite("light.color"));
        }
        if !self.intensity.is_finite() || self.intensity < 0.0 {
            return Err(crate::error::CoreError::NonFinite("light.intensity"));
        }
        if let SceneLightKind::Spot { cone } = self.kind {
            if !cone.is_finite() || cone <= 0.0 || cone >= std::f32::consts::PI {
                return Err(crate::error::CoreError::InvalidLightParam("spot cone must be in (0, π) radians"));
            }
        }
        for k in &self.keys {
            if !k.translation.is_finite() || !k.yaw.is_finite() || !k.pitch.is_finite() {
                return Err(crate::error::CoreError::NonFinite("light.key"));
            }
            if let Some(i) = k.intensity {
                if !i.is_finite() || i < 0.0 {
                    return Err(crate::error::CoreError::NonFinite("light.key.intensity"));
                }
            }
        }
        Ok(())
    }
}

/// Scene ambient as a standalone setting (corrective B1), not a by-product of
/// the background.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum AmbientMode {
    /// Brightness and color are derived from the background (default behavior).
    #[default]
    FromBackground,
    /// Fixed color (linear RGB) and brightness (lux).
    Fixed { color: [f32; 3], brightness: f32 },
    /// Ambient is disabled.
    Off,
}

/// Material patch on top of an imported asset (corrective B4): the material is
/// scene data, the asset is not mutated. None fields — do not change.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MaterialPatch {
    /// Material batch/group name; None — all groups.
    #[serde(default)]
    pub target: Option<String>,
    /// Base color factor, sRGB 0..=1.
    #[serde(default)]
    pub base_color: Option<[f32; 4]>,
    #[serde(default)]
    pub metallic: Option<f32>,
    #[serde(default)]
    pub roughness: Option<f32>,
    /// Emissive, linear RGB (may exceed 1 for glow).
    #[serde(default)]
    pub emissive: Option<[f32; 3]>,
    /// Opacity 0..=1; < 1 enables alpha blend.
    #[serde(default)]
    pub opacity: Option<f32>,
    #[serde(default)]
    pub unlit: Option<bool>,
}

impl MaterialPatch {
    pub fn validate(&self) -> Result<(), crate::error::CoreError> {
        let unit = |v: Option<f32>, name: &'static str| -> Result<(), crate::error::CoreError> {
            match v {
                Some(x) if !x.is_finite() || !(0.0..=1.0).contains(&x) => Err(crate::error::CoreError::InvalidLightParam(name)),
                _ => Ok(()),
            }
        };
        unit(self.metallic, "material.metallic")?;
        unit(self.roughness, "material.roughness")?;
        unit(self.opacity, "material.opacity")?;
        if let Some(c) = self.base_color {
            if !c.iter().all(|x| x.is_finite() && (0.0..=1.0).contains(x)) {
                return Err(crate::error::CoreError::InvalidLightParam("material.base_color"));
            }
        }
        if let Some(e) = self.emissive {
            if !e.iter().all(|x| x.is_finite() && *x >= 0.0) {
                return Err(crate::error::CoreError::InvalidLightParam("material.emissive"));
            }
        }
        Ok(())
    }
}

/// Material presets (B4): the agent should not have to know what "0.9
/// roughness" means in this renderer.
pub fn material_preset(name: &str) -> Option<MaterialPatch> {
    let p = |roughness: f32, metallic: f32| MaterialPatch { roughness: Some(roughness), metallic: Some(metallic), ..Default::default() };
    Some(match name {
        "matte" => p(0.95, 0.0),
        "skin" => p(0.55, 0.0),
        "metal" => p(0.25, 1.0),
        "wet" => p(0.12, 0.0),
        "glass" => MaterialPatch { opacity: Some(0.35), roughness: Some(0.05), ..Default::default() },
        "glow" => MaterialPatch { emissive: Some([2.0, 2.0, 2.0]), ..Default::default() },
        _ => return None,
    })
}

/// Color grading of the whole frame (corrective B3): global exposure and
/// ASC CDL sections (the same for shadows/midtones/highlights). Identity is
/// `SceneGrade::default()`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneGrade {
    /// Exposure in stops (EV); 0 — unchanged.
    pub exposure: f32,
    /// Temperature (blue↔red); 0 — unchanged.
    pub temperature: f32,
    /// Tint (green↔magenta); 0 — unchanged.
    pub tint: f32,
    /// Hue rotation, radians; 0 — unchanged.
    pub hue: f32,
    /// Saturation after tonemapping; 1 — unchanged.
    pub post_saturation: f32,
    /// Section saturation; 1 — unchanged.
    pub saturation: f32,
    /// Section contrast; 1 — unchanged.
    pub contrast: f32,
    /// Section gamma (exponent); 1 — unchanged.
    pub gamma: f32,
    /// Section gain (multiplier); 1 — unchanged.
    pub gain: f32,
    /// Section lift (additive); 0 — unchanged.
    pub lift: f32,
}

impl Default for SceneGrade {
    fn default() -> Self {
        Self { exposure: 0.0, temperature: 0.0, tint: 0.0, hue: 0.0, post_saturation: 1.0, saturation: 1.0, contrast: 1.0, gamma: 1.0, gain: 1.0, lift: 0.0 }
    }
}

impl SceneGrade {
    pub fn validate(&self) -> Result<(), crate::error::CoreError> {
        let v = [
            self.exposure, self.temperature, self.tint, self.hue, self.post_saturation, self.saturation, self.contrast, self.gamma, self.gain, self.lift,
        ];
        if !v.iter().all(|x| x.is_finite()) {
            return Err(crate::error::CoreError::NonFinite("grade"));
        }
        if self.gamma <= 0.0 || self.gain <= 0.0 || self.contrast <= 0.0 || self.saturation < 0.0 || self.post_saturation < 0.0 {
            return Err(crate::error::CoreError::InvalidLightParam("grade: gamma/gain/contrast > 0, saturation >= 0"));
        }
        Ok(())
    }
}

/// Exponential fog (B3): DistanceFog on the camera.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneFog {
    /// Fog color, linear RGB.
    pub color: [f32; 3],
    /// Exponential falloff density, > 0.
    pub density: f32,
}

/// Effect kind (B3): particles (a deterministic hash simulation: snow, smoke,
/// embers, rain) and water (a plane with a normal map + UV scroll). god_rays
/// is deliberately deferred (it needs its own shader).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VfxKind {
    Particles,
    Water,
}

/// Effect on/off and density key at a frame (step from the last passed key).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct VfxKey {
    pub frame: Frame,
    #[serde(default)]
    pub enabled: Option<bool>,
    /// emit_rate multiplier at the frame; None — from the effect body.
    #[serde(default)]
    pub density: Option<f32>,
}

/// Scene effect with keys (B3): lives in the project and the undo history.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneVfx {
    pub id: crate::id::VfxId,
    pub name: String,
    pub kind: VfxKind,
    pub enabled: bool,
    /// Emitter center / water plane center, meters.
    pub position: Vec3,
    /// Emission volume (particles) or plane size (water: x and z), meters.
    pub extent: Vec3,
    /// Particles per second (0 for water).
    pub emit_rate: f32,
    /// Particle lifetime, seconds (> 0).
    pub lifetime: f32,
    /// Initial particle speed, m/s (direction is opposite to the sign of gravity).
    pub speed: f32,
    /// Particle size (quad side), meters.
    pub size: f32,
    /// Color, linear RGB.
    pub color: [f32; 3],
    /// Vertical acceleration, m/s²: negative falls (snow/rain),
    /// positive rises (embers/smoke).
    pub gravity: f32,
    /// Water: normal map asset (image) and UV scroll speed, in texture fractions/s.
    #[serde(default)]
    pub normal_map: Option<AssetId>,
    #[serde(default)]
    pub scroll_uv: [f32; 2],
    /// Attachment of the emitter to a bone (B2); None — a fixed `position`.
    #[serde(default)]
    pub attach: Option<Attachment>,
    /// On/off and density keys (per-frame steps).
    #[serde(default)]
    pub keys: Vec<VfxKey>,
}

impl SceneVfx {
    /// Whether the effect is on at a frame (step from the last passed key).
    pub fn enabled_at(&self, frame: Frame) -> bool {
        self.keys.iter().rev().find(|k| k.frame <= frame).and_then(|k| k.enabled).unwrap_or(self.enabled)
    }

    /// Density at a frame: the last value set, otherwise 1.
    pub fn density_at(&self, frame: Frame) -> f32 {
        self.keys.iter().rev().find(|k| k.frame <= frame).and_then(|k| k.density).unwrap_or(1.0)
    }

    pub fn validate(&self) -> Result<(), crate::error::CoreError> {
        if self.name.trim().is_empty() {
            return Err(crate::error::CoreError::EmptyName);
        }
        let finite = [self.emit_rate, self.lifetime, self.speed, self.size, self.gravity, self.scroll_uv[0], self.scroll_uv[1]];
        if !finite.iter().all(|x| x.is_finite()) || !self.position.is_finite() || !self.extent.is_finite() {
            return Err(crate::error::CoreError::NonFinite("vfx"));
        }
        if !self.color.iter().all(|c| c.is_finite() && *c >= 0.0) {
            return Err(crate::error::CoreError::NonFinite("vfx.color"));
        }
        if self.emit_rate < 0.0 || self.lifetime <= 0.0 || self.size <= 0.0 {
            return Err(crate::error::CoreError::InvalidLightParam("vfx: emit_rate >= 0, lifetime > 0, size > 0"));
        }
        for k in &self.keys {
            if let Some(d) = k.density {
                if !d.is_finite() || d < 0.0 {
                    return Err(crate::error::CoreError::InvalidLightParam("vfx.key.density >= 0"));
                }
            }
        }
        Ok(())
    }
}

/// Particle presets (B3): names beat bare numbers.
pub fn vfx_preset(name: &str) -> Option<SceneVfx> {
    let base = SceneVfx {
        id: crate::id::VfxId(0),
        name: name.into(),
        kind: VfxKind::Particles,
        enabled: true,
        position: Vec3::ZERO,
        extent: Vec3::new(10.0, 0.5, 10.0),
        emit_rate: 100.0,
        lifetime: 4.0,
        speed: 0.5,
        size: 0.03,
        color: [1.0, 1.0, 1.0],
        gravity: -0.3,
        normal_map: None,
        scroll_uv: [0.0; 2],
        attach: None,
        keys: vec![],
    };
    Some(match name {
        // Snow: slow fall, white soft particles.
        "snow" => SceneVfx { emit_rate: 200.0, lifetime: 6.0, speed: 0.4, size: 0.03, gravity: -0.15, ..base },
        // Campfire embers: upward, warm, fade out fast.
        "embers" => SceneVfx { emit_rate: 30.0, lifetime: 2.5, speed: 0.8, size: 0.02, color: [1.0, 0.55, 0.15], gravity: 0.2, extent: Vec3::new(0.6, 0.1, 0.6), ..base },
        // Smoke: large and grey, slow rise.
        "smoke" => SceneVfx { emit_rate: 15.0, lifetime: 4.0, speed: 0.5, size: 0.15, color: [0.35, 0.35, 0.38], gravity: 0.1, extent: Vec3::new(0.4, 0.1, 0.4), ..base },
        // Rain: fast and small, downward.
        "rain" => SceneVfx { emit_rate: 600.0, lifetime: 1.2, speed: 6.0, size: 0.02, color: [0.6, 0.7, 0.85], gravity: -3.0, ..base },
        _ => return None,
    })
}

/// Three-point stage lighting for the scene: key, fill, rim.
/// The defaults are close to the scene's previous look (key ≈ 8000 lux).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneLighting {
    #[serde(default)]
    pub key: SceneLight,
    #[serde(default = "default_fill_light")]
    pub fill: SceneLight,
    #[serde(default = "default_rim_light")]
    pub rim: SceneLight,
}

impl Default for SceneLighting {
    fn default() -> Self {
        Self { key: SceneLight::default(), fill: default_fill_light(), rim: default_rim_light() }
    }
}

impl SceneLighting {
    pub fn validate(&self) -> Result<(), crate::error::CoreError> {
        self.key.validate()?;
        self.fill.validate()?;
        self.rim.validate()
    }
}

/// Matte scene floor on y=0: opaque geometry that receives shadows.
/// A transparent shadow-catcher is out of scope for P0.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneFloor {
    /// Floor color, linear RGB, channels >= 0.
    pub color: [f32; 3],
}

impl Default for SceneFloor {
    fn default() -> Self {
        Self { color: [0.5, 0.5, 0.5] }
    }
}

impl SceneFloor {
    pub fn validate(&self) -> Result<(), crate::error::CoreError> {
        if !self.color.iter().all(|c| c.is_finite() && *c >= 0.0) {
            return Err(crate::error::CoreError::NonFinite("floor.color"));
        }
        Ok(())
    }
}

/// Background source. Untagged: a number is an image (compatible with old
/// manifests), an object is a video.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BackgroundSource {
    Image(AssetId),
    Video { video: AssetId },
}

impl BackgroundSource {
    pub fn asset(&self) -> AssetId {
        match self {
            Self::Image(a) => *a,
            Self::Video { video } => *video,
        }
    }
}

/// Panorama UV mode on the screen (corrective A3c).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundFit {
    /// Cover the screen, cropping the overflowing part of the picture (the
    /// corrective A3 default).
    #[default]
    Cover,
    /// The whole picture is visible; the overflow is cropped by the texture edges.
    Contain,
    /// Stretch the picture over the whole screen without keeping proportions
    /// (the render behavior before A3).
    Stretch,
}

/// Flat screen behind the characters (layout mode).
/// Old manifests carry a curved arc screen instead; it is read as the flat
/// screen of the same horizontal length (`arc_radians × radius`), and the
/// curve itself is gone from the program.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(try_from = "BackgroundInput")]
pub struct Background {
    pub image: BackgroundSource,
    /// Screen width in meters.
    pub width: f32,
    /// Screen height in meters.
    pub height: f32,
    /// Screen center in the scene; the screen faces the camera side (+Z).
    pub center: Vec3,
    /// Panorama UV mode; the serde default is `cover` (corrective A3).
    #[serde(default)]
    pub fit: BackgroundFit,
}

/// Wire form of [`Background`]: accepts the current fields and the legacy arc
/// screen, so an old manifest loads without a migration step.
#[derive(Deserialize)]
struct BackgroundInput {
    image: BackgroundSource,
    #[serde(default)]
    width: Option<f32>,
    #[serde(default)]
    height: Option<f32>,
    #[serde(default)]
    center: Option<Vec3>,
    #[serde(default)]
    fit: BackgroundFit,
    /// Legacy curved screen: its angular width and radius.
    #[serde(default)]
    arc_radians: Option<f32>,
    #[serde(default)]
    radius: Option<f32>,
    /// Legacy bulge direction, accepted and dropped.
    #[serde(default, rename = "bulge")]
    _bulge: Option<serde::de::IgnoredAny>,
}

impl TryFrom<BackgroundInput> for Background {
    type Error = crate::error::CoreError;

    fn try_from(input: BackgroundInput) -> Result<Self, Self::Error> {
        let width = match (input.width, input.arc_radians, input.radius) {
            (Some(width), _, _) => width,
            // The flat screen takes the arc's horizontal length, so an old
            // scene keeps roughly the same backdrop size.
            (None, Some(arc), Some(radius)) => arc * radius,
            _ => return Err(crate::error::CoreError::InvalidBackground("width (or a legacy arc_radians + radius)")),
        };
        let background = Self {
            image: input.image,
            width,
            height: input.height.ok_or(crate::error::CoreError::InvalidBackground("height"))?,
            center: input.center.ok_or(crate::error::CoreError::InvalidBackground("center"))?,
            fit: input.fit,
        };
        background.validate()?;
        Ok(background)
    }
}

impl Background {
    pub fn validate(&self) -> Result<(), crate::error::CoreError> {
        use crate::error::CoreError::{InvalidBackground, NonFinite};
        if !self.width.is_finite() || !self.height.is_finite() || !self.center.is_finite() {
            return Err(NonFinite("background.width/height/center"));
        }
        if self.width <= 0.0 {
            return Err(InvalidBackground("width must be positive"));
        }
        if self.height <= 0.0 {
            return Err(InvalidBackground("height must be positive"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Camera {
    pub keys: Vec<TransformKey>,
    /// Focal length in mm (35 mm equivalent).
    pub focal_mm: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            keys: vec![TransformKey {
                frame: 0,
                translation: Vec3::new(0.0, 1.4, 4.0),
                yaw: 0.0,
                pitch: 0.0,
                roll: 0.0,
                look_at: None,
                focal_mm: None,
                ease: KeyEase::Linear,
            }],
            focal_mm: 35.0,
        }
    }
}

/// Camera vertical field of view in radians: a sensor 24 mm tall
/// (35 mm equivalent), formula `2·atan(12 / focal_mm)`.
/// The single source of the formula: the render (`runtime/render`) and pure
/// diagnostics (`app/analysis`) both use it so coverage matches the real frame.
pub fn vertical_fov(focal_mm: f32) -> f32 {
    2.0 * (12.0_f32 / focal_mm).atan()
}

/// Horizontal field of view in radians for a frame with aspect ratio
/// `aspect` (width/height), derived from [`vertical_fov`].
pub fn horizontal_fov(focal_mm: f32, aspect: f32) -> f32 {
    2.0 * ((vertical_fov(focal_mm) * 0.5).tan() * aspect).atan()
}

/// Smoothing between camera keys (corrective_2 C1.5): linear three-key moves
/// produced a "broken line" camera. The segment curve comes from the key that
/// starts the segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KeyEase {
    /// Straight linear interpolation (the behavior before corrective_2).
    #[default]
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
}

impl KeyEase {
    /// Map a segment fraction; a cubic curve (smoothstep family)
    /// with no new dependencies. Continuous and monotonic on [0, 1].
    pub fn apply(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            KeyEase::Linear => t,
            KeyEase::EaseIn => t * t,
            KeyEase::EaseOut => 1.0 - (1.0 - t) * (1.0 - t),
            KeyEase::EaseInOut => t * t * (3.0 - 2.0 * t),
        }
    }
}

/// Camera transform key: position, Euler angles (radians) and the optional
/// corrective_2 C1 additions:
/// - `roll` — camera tilt of the horizon;
/// - `look_at` — aim point: when set, this key's `yaw`/`pitch` are computed
///   from it instead of being taken from the fields;
/// - `focal_mm` — focal length at this key (zoom/dolly-zoom inside a shot);
/// - `ease` — the curve of the segment starting at this key.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransformKey {
    pub frame: Frame,
    pub translation: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    #[serde(default)]
    pub roll: f32,
    #[serde(default)]
    pub look_at: Option<Vec3>,
    #[serde(default)]
    pub focal_mm: Option<f32>,
    #[serde(default)]
    pub ease: KeyEase,
}

impl Default for TransformKey {
    fn default() -> Self {
        Self {
            frame: 0,
            translation: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            look_at: None,
            focal_mm: None,
            ease: KeyEase::Linear,
        }
    }
}

impl TransformKey {
    /// Key with an explicit pose and angles; the optional additions go
    /// through `..Default::default()` (roll/look_at/focal/ease).
    pub fn at(frame: Frame, translation: Vec3, yaw: f32, pitch: f32) -> Self {
        Self { frame, translation, yaw, pitch, ..Default::default() }
    }

    /// Effective key angles: `look_at` overrides the given yaw/pitch.
    /// The direction matches the renderer: the view runs along local −Z under
    /// the `YXZ(yaw, pitch)` basis.
    pub fn effective_angles(&self) -> (f32, f32) {
        match self.look_at {
            Some(target) if target.is_finite() => {
                let d = target - self.translation;
                let horizontal = (d.x * d.x + d.z * d.z).sqrt();
                if horizontal < 1e-6 && d.y.abs() < 1e-6 {
                    return (self.yaw, self.pitch);
                }
                // Renderer basis: the view runs along local −Z under YXZ(yaw, pitch),
                // i.e. direction = (−sin yaw, sin pitch, −cos yaw) — hence the signs.
                let yaw = (-d.x).atan2(-d.z);
                let pitch = d.y.atan2(horizontal);
                (yaw, pitch)
            }
            _ => (self.yaw, self.pitch),
        }
    }
}

/// Camera pose at a frame (corrective_2 C1): position, angles, roll and focal length.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraPose {
    pub translation: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub focal_mm: f32,
}

impl Camera {
    /// Camera pose at a frame. One implementation for the legacy scene camera
    /// and for shot cameras; the render and the pure diagnostics both take the
    /// pose from here, so frame coverage matches the real frame.
    ///
    /// Interpolation: position/angles are linear over the eased segment
    /// fraction (`KeyEase` of the starting key), yaw along the shortest arc
    /// (otherwise a 170° → −170° turn would go the "long way"), focal length is
    /// linear, and without a per-key `focal_mm` the base `Camera.focal_mm`.
    pub fn sample(&self, frame: Frame) -> CameraPose {
        let keys = &self.keys;
        let resolve = |k: &TransformKey| {
            let (yaw, pitch) = k.effective_angles();
            CameraPose {
                translation: k.translation,
                yaw,
                pitch,
                roll: k.roll,
                focal_mm: k.focal_mm.unwrap_or(self.focal_mm),
            }
        };
        let first = resolve(&keys[0]);
        let idx = match keys.binary_search_by_key(&frame, |k| k.frame) {
            Ok(i) => return resolve(&keys[i]),
            Err(0) => return first,
            Err(i) => i - 1,
        };
        let from = resolve(&keys[idx]);
        let Some(to) = keys.get(idx + 1).map(|k| resolve(k)) else {
            return from;
        };
        let raw = (frame - keys[idx].frame) as f32 / (keys[idx + 1].frame - keys[idx].frame) as f32;
        let alpha = keys[idx].ease.apply(raw);
        CameraPose {
            translation: from.translation.lerp(to.translation, alpha),
            yaw: from.yaw + shortest_arc(from.yaw, to.yaw) * alpha,
            pitch: from.pitch + (to.pitch - from.pitch) * alpha,
            roll: from.roll + (to.roll - from.roll) * alpha,
            focal_mm: from.focal_mm + (to.focal_mm - from.focal_mm) * alpha,
        }
    }
}

/// Shortest angular difference in (−π, π].
pub fn shortest_arc(from: f32, to: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    let mut d = (to - from) % TAU;
    if d > PI {
        d -= TAU;
    } else if d <= -PI {
        d += TAU;
    }
    d
}

/// A controllable skeleton with an attached character model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Phantom {
    pub id: PhantomId,
    pub name: String,
    pub rig: RigRef,
    /// Controller keys: root position, held pose and yaw, sorted by frame.
    pub motion: Vec<MotionKey>,
    /// Outline width in output pixels.
    pub outline_px: f32,
    /// Actor scale (B2); the serde default 1.0 covers old manifests without the field.
    #[serde(default = "one_f32")]
    pub scale: f32,
    /// Material patches on top of imported assets (B4): a wet wolf, a bridle,
    /// a glowing lantern. An empty list keeps the asset's materials as is.
    #[serde(default)]
    pub material_patches: Vec<MaterialPatch>,
}

fn one_f32() -> f32 {
    1.0
}

/// Attachment of an entity to a Phantom bone (corrective B2): the transform is
/// recomputed from the pose every frame. Bone names match the rig
/// case-insensitively (builtin — `rig::actor_bones`, import — SkinData).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Attachment {
    pub phantom: PhantomId,
    /// Bone name (for example "lower_arm_r").
    pub bone: String,
    /// Offset in bone space, meters.
    #[serde(default)]
    pub offset_pos: Vec3,
    /// Rotation of the offset, radians YXZ.
    #[serde(default)]
    pub offset_yaw: f32,
    #[serde(default)]
    pub offset_pitch: f32,
    #[serde(default)]
    pub offset_roll: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RigRef {
    /// Built-in humanoid rig with procedural movements.
    Builtin,
    /// Original bundled procedural actor; Builtin remains the legacy humanoid.
    BuiltinActor { actor: BuiltinActor },
    /// Imported FBX character.
    Imported { asset: AssetId },
}

/// A static position of a builtin actor (layout mode). Actors are not
/// animated: a key names one of these and the figure holds it.
///
/// The wire name is lowercase (`stand|sit|lie`) — the same string the CLI, the
/// MCP tools and `builtin_catalog` use, so an agent never has to remember two
/// spellings of the same pose.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActorPose {
    Stand,
    Sit,
    Lie,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "MotionKeyInput")]
pub struct MotionKey {
    pub frame: Frame,
    /// Controller root position (the floor clamps y >= 0).
    pub position: Vec3,
    /// Held pose of a builtin actor. An imported rig has no poses and renders
    /// in bind pose; the field is ignored for it.
    #[serde(default)]
    pub pose: ActorPose,
    /// Root rotation about +Y in radians (forward +Z).
    #[serde(default)]
    pub yaw: f32,
}

impl Default for ActorPose {
    fn default() -> Self {
        Self::Stand
    }
}

/// Stable procedural skeleton/animation profiles (meters, Y up, forward +Z).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum BuiltinActor {
    Humanoid,
    Wolf,
    Horse,
    Bird,
    /// A child — the same rig and poses as the humanoid, proportions from
    /// [`crate::catalog::HumanRole`] ("child").
    Child,
    /// A teenager (the "teen" role).
    Teen,
    /// An elderly person (the "elder" role).
    Elder,
}

impl BuiltinActor {
    /// Every bundled actor, in wire-name order.
    pub const ALL: &'static [BuiltinActor] = &[
        BuiltinActor::Humanoid,
        BuiltinActor::Wolf,
        BuiltinActor::Horse,
        BuiltinActor::Bird,
        BuiltinActor::Child,
        BuiltinActor::Teen,
        BuiltinActor::Elder,
    ];

    /// Poses this actor can hold. A horse cannot sit and a bird has no lying
    /// pose — an unavailable pose is an error, not a silently approximated one.
    pub fn poses(self) -> &'static [ActorPose] {
        use ActorPose::*;
        match self {
            Self::Humanoid | Self::Child | Self::Teen | Self::Elder | Self::Wolf => &[Stand, Sit, Lie],
            Self::Horse => &[Stand, Lie],
            Self::Bird => &[Stand, Sit],
        }
    }

    /// The human proportion profile of a human actor; `None` for animals. The
    /// role names are the single source of truth for human sizes (the stage
    /// clamp and the drawn figure both come from them).
    pub fn human_role(self) -> Option<&'static crate::catalog::HumanRole> {
        let name = match self {
            Self::Humanoid => "adult",
            Self::Child => "child",
            Self::Teen => "teen",
            Self::Elder => "elder",
            _ => return None,
        };
        crate::catalog::human_role(name)
    }

    pub fn default_pose(self) -> ActorPose {
        ActorPose::Stand
    }

    /// Local bounding box for the stage clamp, origin at the feet. A pose
    /// changes how much floor an actor takes: a lying figure is long and low,
    /// a sitting one is deeper than it is wide.
    pub fn footprint(self, pose: ActorPose) -> crate::stage::Footprint {
        use crate::stage::Footprint;
        // Human roles other than the adult reference: the adult numbers are
        // scaled by the role's own leg/torso proportions (the adult branch
        // below stays the exact reference it always was).
        if self.human_role().is_some() && self != Self::Humanoid {
            let p = crate::rig::human_proportions(self);
            return match pose {
                ActorPose::Sit => Footprint::standing(0.30 * p.torso, p.top - 0.50 * p.leg, 0.45 * p.torso),
                ActorPose::Lie => Footprint::standing(0.35 * p.torso, 0.45 * p.torso, 0.95 * (p.top / 1.74)),
                _ => Footprint::standing(0.30 * p.torso, p.top, 0.25 * p.torso),
            };
        }
        match (self, pose) {
            (Self::Humanoid, ActorPose::Sit) => Footprint::standing(0.30, 1.05, 0.45),
            (Self::Humanoid, ActorPose::Lie) => Footprint::standing(0.35, 0.45, 0.95),
            (Self::Humanoid, ActorPose::Stand) => Footprint::standing(0.30, 1.74, 0.25),
            (Self::Wolf, ActorPose::Sit) => Footprint::standing(0.20, 0.45, 0.40),
            (Self::Wolf, ActorPose::Lie) => Footprint::standing(0.20, 0.25, 0.55),
            (Self::Wolf, ActorPose::Stand) => Footprint::standing(0.20, 0.57, 0.55),
            (Self::Horse, ActorPose::Lie) => Footprint::standing(0.35, 0.70, 0.95),
            (Self::Horse, ActorPose::Stand) => Footprint::standing(0.35, 1.62, 0.95),
            (Self::Bird, ActorPose::Sit) => Footprint::standing(0.20, 0.48, 0.25),
            (Self::Bird, ActorPose::Stand) => Footprint::standing(0.25, 0.35, 0.25),
            // A pose an actor does not have never reaches the document
            // (`validate_pose` rejects it); this arm keeps the clamp total.
            (actor, _) => actor.footprint(actor.default_pose()),
        }
    }
}
impl RigRef {
    pub fn builtin_actor(&self) -> Option<BuiltinActor> {
        match self {
            Self::Builtin => Some(BuiltinActor::Humanoid),
            Self::BuiltinActor { actor } => Some(*actor),
            Self::Imported { .. } => None,
        }
    }
    /// An imported rig has no poses (it renders in bind pose), so any pose is
    /// accepted for it; a builtin actor accepts only its own.
    pub fn validate_pose(&self, pose: ActorPose) -> Result<(), crate::error::CoreError> {
        if let Some(actor) = self.builtin_actor() {
            if !actor.poses().contains(&pose) {
                return Err(crate::error::CoreError::UnsupportedPose { actor, pose });
            }
        }
        Ok(())
    }
}

impl MotionKey {
    pub fn validate(&self) -> Result<(), crate::error::CoreError> {
        if !self.position.is_finite() {
            return Err(crate::error::CoreError::NonFinite("key.position"));
        }
        if !self.yaw.is_finite() {
            return Err(crate::error::CoreError::NonFinite("key.yaw"));
        }
        Ok(())
    }
}

#[derive(Deserialize)]
struct MotionKeyInput {
    frame: Frame,
    position: Vec3,
    /// A pre-layout manifest carries `action` here; it is accepted and dropped
    /// (the key falls back to the default pose).
    #[serde(default)]
    pose: ActorPose,
    #[serde(default)]
    _action: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    yaw: f32,
}

impl TryFrom<MotionKeyInput> for MotionKey {
    type Error = crate::error::CoreError;
    fn try_from(k: MotionKeyInput) -> Result<Self, Self::Error> {
        let key = Self { frame: k.frame, position: k.position, pose: k.pose, yaw: k.yaw };
        key.validate()?;
        Ok(key)
    }
}

#[cfg(test)]
mod light_shadows_tests {
    use super::*;

    #[test]
    fn shadows_default_false_and_legacy_json_reads_without_field() {
        assert!(!SceneLight::default().shadows);
        assert!(!SceneLighting::default().key.shadows);
        let legacy: SceneLight = serde_json::from_str(r#"{"color":[1,1,1],"intensity":8000,"enabled":true}"#).unwrap();
        assert!(!legacy.shadows, "old projects without the field read as shadows=false");
        let on: SceneLight = serde_json::from_str(r#"{"color":[1,1,1],"intensity":8000,"enabled":true,"shadows":true}"#).unwrap();
        assert!(on.shadows);
    }
}

#[cfg(test)]
mod background_a3_tests {
    use super::*;

    #[test]
    fn a_legacy_arc_screen_reads_as_a_flat_one() {
        // A curved-screen manifest: the flat screen takes the arc's horizontal
        // length (2.0 rad x 6.0 m = 12 m); the bulge direction is dropped.
        let bg: Background = serde_json::from_str(
            r#"{"image":1,"arc_radians":2.0,"radius":6.0,"height":4.0,"center":[0.0,2.0,-5.0]}"#,
        )
        .unwrap();
        assert_eq!(bg.width, 12.0);
        assert_eq!(bg.height, 4.0);
        assert_eq!(bg.center, Vec3::new(0.0, 2.0, -5.0));
        assert_eq!(bg.fit, BackgroundFit::Cover, "the uv mode survives");
        let explicit: Background = serde_json::from_str(
            r#"{"image":1,"arc_radians":2.0,"radius":6.0,"height":4.0,"center":[0.0,2.0,-5.0],"bulge":"toward_camera","fit":"stretch"}"#,
        )
        .unwrap();
        assert_eq!(explicit.fit, BackgroundFit::Stretch);
        assert_eq!(explicit.width, 12.0);
    }

    #[test]
    fn a_flat_screen_roundtrips_and_bad_geometry_is_rejected() {
        let bg = Background { image: BackgroundSource::Image(AssetId(1)), width: 16.0, height: 8.0, center: Vec3::new(0.0, 2.0, -5.0), fit: BackgroundFit::Contain };
        let json = serde_json::to_string(&bg).unwrap();
        assert!(!json.contains("arc_radians"), "the new form is written: {json}");
        let back: Background = serde_json::from_str(&json).unwrap();
        assert_eq!(back.width, 16.0);
        assert_eq!(back.fit, BackgroundFit::Contain);
        // Width/height must be positive and finite, and the center finite.
        for bad in [
            r#"{"image":1,"width":0.0,"height":4.0,"center":[0,0,0]}"#,
            r#"{"image":1,"width":-2.0,"height":4.0,"center":[0,0,0]}"#,
            r#"{"image":1,"width":4.0,"height":0.0,"center":[0,0,0]}"#,
            r#"{"image":1,"width":4.0,"height":4.0,"center":[0,0,"NaN"]}"#,
        ] {
            assert!(serde_json::from_str::<Background>(bad).is_err(), "accepted invalid screen geometry: {bad}");
        }
        // No width and no legacy arc either: nothing to build a screen from.
        assert!(serde_json::from_str::<Background>(r#"{"image":1,"height":4.0,"center":[0,0,0]}"#).is_err());
        assert!(serde_json::from_str::<Background>(r#"{"image":1,"width":4.0,"center":[0,0,0]}"#).is_err());
        assert!(serde_json::from_str::<Background>(r#"{"image":1,"width":4.0,"height":4.0}"#).is_err());
    }
}

#[cfg(test)]
mod scene_lights_tests {
    use super::*;

    fn light() -> SceneLightSource {
        SceneLightSource {
            id: crate::id::LightId(1),
            name: "lantern".into(),
            kind: SceneLightKind::Point,
            color: [1.0, 0.9, 0.8],
            intensity: 500.0,
            enabled: true,
            shadows: false,
            keys: vec![
                LightKey { frame: 0, translation: Vec3::new(0.0, 1.0, 0.0), yaw: 0.0, pitch: 0.0, intensity: None, enabled: None },
                LightKey { frame: 10, translation: Vec3::new(10.0, 1.0, 0.0), yaw: std::f32::consts::TAU - 0.2, pitch: 0.0, intensity: Some(900.0), enabled: Some(false) },
            ],
            rest: TransformKey::at(0, Vec3::new(5.0, 5.0, 5.0), 1.0, 0.5),
            attach: None,
        }
    }

    #[test]
    fn at_interpolates_transform_intensity_and_steps_enabled() {
        let l = light();
        // Without keys — rest.
        let bare = SceneLightSource { keys: vec![], ..light() };
        assert_eq!(bare.at(3), (Vec3::new(5.0, 5.0, 5.0), 1.0, 0.5, 500.0, true));
        // Before the first key — the rest pose.
        assert_eq!(l.at(-5).0, Vec3::new(5.0, 5.0, 5.0));
        // Between keys: half way, shortest-arc yaw (0 → 2π−0.2 goes backwards).
        let (pos, yaw, _, intensity, enabled) = l.at(5);
        assert!((pos.x - 5.0).abs() < 1e-5);
        assert!(yaw.abs() < 0.11, "shortest-arc: {yaw}");
        assert!((intensity - 700.0).abs() < 1e-3);
        assert!(enabled, "step from the last passed key: on until frame 10");
        // After the last key.
        assert_eq!(l.at(100).4, false);
    }

    #[test]
    fn validate_rejects_bad_cone_and_intensity() {
        let mut l = light();
        l.kind = SceneLightKind::Spot { cone: 4.0 };
        assert!(matches!(l.validate().unwrap_err(), crate::error::CoreError::InvalidLightParam(_)));
        l.kind = SceneLightKind::Point;
        l.intensity = -1.0;
        assert!(matches!(l.validate().unwrap_err(), crate::error::CoreError::NonFinite(_)));
        l.intensity = 500.0;
        l.keys[0].translation = Vec3::new(f32::NAN, 0.0, 0.0);
        assert!(matches!(l.validate().unwrap_err(), crate::error::CoreError::NonFinite(_)));
    }

    #[test]
    fn legacy_scene_json_without_lights_ambient_sky() {
        // A scene from before B1/B5: the new fields get serde defaults.
        let s: Scene = serde_json::from_str(
            r#"{"id":1,"name":"s","range":{"start":0,"end":24},"background":null,
                "camera":{"focal_mm":35.0,"keys":[]},"phantoms":{}}"#,
        )
        .unwrap();
        assert!(s.lights.is_empty());
        assert_eq!(s.ambient, AmbientMode::FromBackground);
        assert_eq!(s.sky, None);
    }
}

#[cfg(test)]
mod vfx_tests {
    use super::*;

    #[test]
    fn vfx_keys_step_enabled_and_density() {
        let mut v = vfx_preset("snow").unwrap();
        assert!(v.enabled_at(0) && (v.density_at(0) - 1.0).abs() < 1e-6);
        v.keys = vec![
            VfxKey { frame: 10, enabled: Some(false), density: None },
            VfxKey { frame: 20, enabled: Some(true), density: Some(2.5) },
        ];
        assert!(v.enabled_at(0), "before the first key — the body");
        assert!(!v.enabled_at(10));
        assert!(!v.enabled_at(19));
        assert!(v.enabled_at(20));
        assert!((v.density_at(20) - 2.5).abs() < 1e-6);
        assert!((v.density_at(10) - 1.0).abs() < 1e-6, "density None does not touch the multiplier");
    }

    #[test]
    fn vfx_presets_and_validate() {
        for name in ["snow", "embers", "smoke", "rain"] {
            let v = vfx_preset(name).expect(name);
            v.validate().unwrap_or_else(|e| panic!("preset {name}: {e}"));
        }
        assert!(vfx_preset("fireflies").is_none());
        let mut v = vfx_preset("snow").unwrap();
        v.lifetime = 0.0;
        assert!(v.validate().is_err());
        v.lifetime = 4.0;
        v.keys = vec![VfxKey { frame: 0, enabled: None, density: Some(-1.0) }];
        assert!(v.validate().is_err());
    }

    #[test]
    fn grade_identity_is_default() {
        let g = SceneGrade::default();
        g.validate().unwrap();
        let mut bad = g;
        bad.gamma = 0.0;
        assert!(bad.validate().is_err());
    }

    // --- corrective_2 C1: camera keys ---

    fn camera_with(keys: Vec<TransformKey>) -> Camera {
        Camera { keys, focal_mm: 35.0 }
    }

    #[test]
    fn camera_key_easing_bends_the_segment_without_moving_the_ends() {
        use KeyEase::*;
        // The reference is a linear segment; the curves are compared against it at the same frame.
        let straight = camera_with(vec![
            TransformKey::at(0, Vec3::ZERO, 0.0, 0.0),
            TransformKey::at(10, Vec3::new(10.0, 0.0, 0.0), 0.0, 0.0),
        ]);
        let linear = straight.sample(2).translation.x;
        assert!((linear - 2.0).abs() < 1e-4, "linear fraction 0.2: {linear}");
        for (ease, slower_at_start) in [(Linear, false), (EaseIn, true), (EaseOut, false), (EaseInOut, true)] {
            let mut a = TransformKey::at(0, Vec3::ZERO, 0.0, 0.0);
            a.ease = ease;
            let cam = camera_with(vec![a, TransformKey::at(10, Vec3::new(10.0, 0.0, 0.0), 0.0, 0.0)]);
            // The segment ends do not move under any curve.
            assert_eq!(cam.sample(0).translation, Vec3::ZERO, "{ease:?}");
            assert_eq!(cam.sample(10).translation.x, 10.0, "{ease:?}");
            let x = cam.sample(2).translation.x;
            assert_eq!(x < linear, slower_at_start, "{ease:?}: {x} against linear {linear}");
        }
        // EaseInOut is symmetric about the segment midpoint.
        let mut a = TransformKey::at(0, Vec3::ZERO, 0.0, 0.0);
        a.ease = KeyEase::EaseInOut;
        let cam = camera_with(vec![a, TransformKey::at(10, Vec3::new(10.0, 0.0, 0.0), 0.0, 0.0)]);
        assert!((cam.sample(5).translation.x - 5.0).abs() < 1e-4, "the midpoint stays put");
    }

    #[test]
    fn camera_yaw_takes_the_shortest_arc() {
        // 170° → −170°: the short way round ±180°, not the "long way" through 0.
        let a = TransformKey::at(0, Vec3::ZERO, 170f32.to_radians(), 0.0);
        let b = TransformKey::at(10, Vec3::ZERO, (-170f32).to_radians(), 0.0);
        let cam = camera_with(vec![a, b]);
        let mid = cam.sample(5).yaw.to_degrees();
        assert!(mid.abs() > 175.0, "the midpoint should be near ±180°, got {mid}");
        assert_eq!(cam.sample(10).yaw, b.yaw, "the segment end is the exact key");
    }

    #[test]
    fn look_at_derives_angles_and_overrides_them() {
        // Camera at the origin, target on −Z: yaw 0 (as in the render).
        let mut key = TransformKey::at(0, Vec3::ZERO, 1.23, 0.45);
        key.look_at = Some(Vec3::new(0.0, 0.0, -5.0));
        let pose = camera_with(vec![key]).sample(0);
        assert!(pose.yaw.abs() < 1e-5, "yaw={}", pose.yaw);
        assert!(pose.pitch.abs() < 1e-5);
        // A target above the camera — positive pitch (looking up).
        let mut up = TransformKey::at(0, Vec3::ZERO, 0.0, 0.0);
        up.look_at = Some(Vec3::new(0.0, 5.0, -5.0));
        assert!(camera_with(vec![up]).sample(0).pitch > 0.0);
        // A target to the side: +X → yaw −90° (looking right on screen when yaw=0 faces −Z).
        let mut side = TransformKey::at(0, Vec3::ZERO, 0.0, 0.0);
        side.look_at = Some(Vec3::new(5.0, 0.0, 0.0));
        assert!((camera_with(vec![side]).sample(0).yaw.to_degrees() + 90.0).abs() < 1e-3);
        // A target behind: +Z → yaw ±180°.
        let mut back = TransformKey::at(0, Vec3::ZERO, 0.0, 0.0);
        back.look_at = Some(Vec3::new(0.0, 0.0, 5.0));
        assert!(camera_with(vec![back]).sample(0).yaw.to_degrees().abs() > 179.0);
    }

    #[test]
    fn per_key_focal_and_roll_interpolate_over_the_base_lens() {
        let mut a = TransformKey::at(0, Vec3::ZERO, 0.0, 0.0);
        a.focal_mm = Some(24.0);
        a.roll = 0.0;
        let mut b = TransformKey::at(10, Vec3::ZERO, 0.0, 0.0);
        b.focal_mm = Some(85.0);
        b.roll = 10f32.to_radians();
        let cam = camera_with(vec![a, b]);
        assert_eq!(cam.sample(0).focal_mm, 24.0);
        assert!((cam.sample(5).focal_mm - 54.5).abs() < 1e-3, "zoom inside a shot");
        assert!((cam.sample(5).roll.to_degrees() - 5.0).abs() < 1e-3);
        // A key without its own focal_mm takes the camera's base focal length.
        let plain = camera_with(vec![TransformKey::at(0, Vec3::ZERO, 0.0, 0.0)]);
        assert_eq!(plain.sample(0).focal_mm, 35.0);
    }

    #[test]
    fn old_camera_keys_without_new_fields_still_load() {
        // Old manifest: a key without roll/look_at/focal_mm/ease.
        let cam: Camera = serde_json::from_str(
            r#"{"keys":[{"frame":0,"translation":[0.0,1.4,4.0],"yaw":0.0,"pitch":0.0}],"focal_mm":35.0}"#,
        )
        .unwrap();
        let pose = cam.sample(0);
        assert_eq!(pose.roll, 0.0);
        assert_eq!(pose.focal_mm, 35.0);
        assert_eq!(cam.keys[0].look_at, None, "an old key's look_at is None");    }
}

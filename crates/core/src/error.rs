use crate::id::*;
use crate::time::Frame;

#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("pose {pose:?} is not supported by builtin {actor:?}")]
    UnsupportedPose { actor: crate::project::BuiltinActor, pose: crate::project::ActorPose },
    #[error("invalid stage: {0}")]
    InvalidStage(&'static str),
    #[error("invalid background: {0}")]
    InvalidBackground(&'static str),
    #[error("scene {0} not found")]
    SceneNotFound(SceneId),
    #[error("phantom {0} not found")]
    PhantomNotFound(PhantomId),
    #[error("object {0} not found")]
    ObjectNotFound(ObjectId),
    #[error("asset {0} not found")]
    AssetNotFound(AssetId),
    /// A name that is not in the metric catalog (K1).
    #[error("unknown catalog entry: {0} (see catalog_query)")]
    UnknownProp(String),
    // D7/B8: the upper bound is always exclusive — the hint is in the message itself.
    #[error("invalid frame range [{start}, {end}): the upper bound is exclusive (0-1 = frame 0 only)")]
    InvalidRange { start: Frame, end: Frame },
    #[error("non-numeric or infinite value in field {0}")]
    NonFinite(&'static str),
    #[error("empty name")]
    EmptyName,
    #[error("asset {0} has the wrong kind for this operation")]
    WrongAssetKind(AssetId),
    #[error("2D layer {0} not found")]
    LayerNotFound(crate::draw::LayerId),
    #[error("a stroke must contain at least 2 points")]
    InvalidStroke,
    #[error("clip {0} not found")]
    ClipNotFound(crate::film::ClipId),
    #[error("clip range is outside the scene")]
    ClipOutOfScene,
    #[error("audio track {0} not found")]
    TrackNotFound(crate::audio::TrackId),
    #[error("audio clip {0} not found")]
    AudioClipNotFound(crate::audio::AudioClipId),
    #[error("camera {0} not found")]
    CameraNotFound(CameraId),
    #[error("camera {camera} at frame {frame} looks away from the scene: only {fraction:.0} % of the scene volume is in frame (minimum {minimum:.0} %)")]
    CameraFacesAway { camera: CameraId, frame: Frame, fraction: f32, minimum: f32 },
    #[error("camera {camera} is attached to a subject that no longer exists")]
    CameraSubjectMissing { camera: CameraId },
    #[error("camera range is outside the scene")]
    CameraOutOfScene,
    #[error("clip range must match the camera's range")]
    CameraRangeMismatch,
    #[error("invalid batch: {0}")]
    InvalidBatch(&'static str),
    #[error("invalid scatter: {0}")]
    InvalidScatter(String),
    #[error("invalid audio clip parameter: {0}")]
    InvalidAudioClipParam(&'static str),
    #[error("light source {0} not found")]
    LightNotFound(crate::id::LightId),
    #[error("invalid light source parameter: {0}")]
    InvalidLightParam(&'static str),
}

//! phantom-core: project document, commands, undo history, time,
//! scene state evaluation. Independent of Bevy, egui, GPU, MCP.

pub mod audio;
pub mod catalog;
pub mod clips;
pub mod command;
pub mod control;
pub mod draw;
pub mod error;
pub mod film;
pub mod id;
pub mod object;
pub mod pose;
pub mod project;
pub mod retarget;
pub mod rig;
pub mod time;

pub use audio::{
    clamp_i16, mix_frame, mix_into, peak_of, AudioClip, AudioClipId, AudioTrack, PcmBuf, PcmCache, TrackId, SAMPLES_PER_FRAME, SAMPLE_RATE,
};
pub use clips::{active_clip_use, blend_weights, clip_frame_at, validate_clip_use, ClipUse, RootMotion};
pub use command::{Command, CommandOutput, History};
pub use control::{control_pose, step_control, ControlInput, ControlState, CONTROL_HZ, WALK_SPEED};
pub use draw::{Exposure, Layer2D, LayerId, Stroke, DEFAULT_CANVAS};
pub use error::CoreError;
pub use film::{film_len, film_map, film_map_clip, ClipId, FilmClip};
pub use id::{AssetId, ClipUseId, LightId, ObjectId, PhantomId, SceneId, ShotId, VfxId};
pub use object::{evaluate_object, ObjectTransform};
pub use pose::{evaluate, PhantomPose};
pub use retarget::RetargetProfile;
pub use project::{
    horizontal_fov, vertical_fov, Action, AmbientMode, Asset, AssetKind, Attachment, Background, BackgroundFit, BackgroundSource, Bulge, BuiltinActor,
    Camera, LightKey, MaterialPatch, MotionKey, ObjectKey, Phantom, Project, RigRef, Scene, SceneFloor, SceneFog, SceneGrade, SceneLight, SceneLightKind,
    SceneLightSource, SceneLighting, SceneObject, SceneVfx, Shot, TransformKey, VfxKey, VfxKind, SCHEMA_VERSION, material_preset, vfx_preset,
};
pub use time::{Frame, FrameRange, FPS};

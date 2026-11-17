//! phantom-core: project document, commands, undo history, time,
//! scene state evaluation. Independent of Bevy, egui, GPU, MCP.

pub mod audio;
pub mod camera;
pub mod catalog;
pub mod command;
pub mod crowd;
pub mod draw;
pub mod error;
pub mod film;
pub mod id;
pub mod object;
pub mod pose;
pub mod project;
pub mod rig;
pub mod scatter;
pub mod stage;
pub mod time;
pub mod view;

pub use audio::{
    clamp_i16, mix_frame, mix_into, peak_of, AudioClip, AudioClipId, AudioTrack, PcmBuf, PcmCache, TrackId, SAMPLES_PER_FRAME, SAMPLE_RATE,
};
pub use command::{Command, CommandOutput, History};
pub use crowd::{
    crowd_formation, formation_name, plan_crowd, CrowdFormation, CrowdMix, CrowdSpec, PlacedPerson, CROWD_FORMATIONS,
    CROWD_SPACING, MAX_CROWD_PHANTOMS,
};
pub use draw::{Exposure, Layer2D, LayerId, Stroke, DEFAULT_CANVAS};
pub use error::CoreError;
pub use film::{film_len, film_map, film_map_clip, ClipId, FilmClip};
pub use id::{AssetId, LightId, ObjectId, PhantomId, SceneId, CameraId, VfxId};
pub use object::{evaluate_object, ObjectTransform};
pub use pose::{evaluate, PhantomPose};
pub use camera::{
    camera_frustum, camera_pose, camera_target, scene_in_frame, scene_in_frame_with, subject_aim, subject_transform, MIN_SCENE_IN_FRAME,
};
pub use scatter::{
    band_of, band_top_y, plan, scatter_preset, PlacedProp, ScatterPreset, ScatterShape, SeedRng,
    ScatterSpecies, MAX_SCATTER_INSTANCES, MAX_SCATTER_OBJECTS, SCATTER_PRESETS,
};
pub use stage::{Clamp, ClampKind, Footprint, ModelBounds, Stage, StagePreset, StageSize};
pub use view::{camera_looking_at, fit_distance, Frustum, LookView, TargetBox, DEFAULT_ASPECT};
pub use project::{
    horizontal_fov, vertical_fov, ActorPose, AmbientMode, Asset, AssetKind, Attachment, Background, BackgroundFit, BackgroundSource, BuiltinActor,
    Camera, CameraSubject, CameraTarget, CameraTargetMode, LightKey, MaterialPatch, MotionKey, ObjectKey, Phantom, Project, RigRef, Scene,
    SceneCamera, ScatterField,
    SceneFloor, SceneFog, SceneGrade, SceneLight, SceneLightKind, SceneLightSource, SceneLighting, SceneObject, SceneVfx, TransformKey, VfxKey, VfxKind,
    SCHEMA_VERSION, SCENE_CAMERA, material_preset, vfx_preset,
};
pub use time::{Frame, FrameRange, FPS};

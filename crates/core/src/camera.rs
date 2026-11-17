//! Scene cameras: the world pose of a camera at a frame — including its
//! attachment to a moving subject — and the rule that a camera must look at the
//! scene.
//!
//! Everything here is a pure function of the document, so the numbers an agent
//! is shown are unit-testable without a GPU and mean the same thing in the
//! renderer, in the commands and in `validate`.

use glam::{Quat, Vec3};

use crate::object::evaluate_object;
use crate::pose::evaluate;
use crate::id::CameraId;
use crate::project::{CameraPose, CameraSubject, CameraTarget, CameraTargetMode, Project, Scene};
use crate::stage::Stage;
use crate::time::Frame;
use crate::view::{Frustum, DEFAULT_ASPECT};

/// A camera must keep at least this fraction of the scene's volume in frame.
/// A camera standing *inside* the scene volume always counts as 1.0, so an
/// interior shot is never punished for seeing only a slice of a big stage.
pub const MIN_SCENE_IN_FRAME: f32 = 0.20;

/// The stage volume is sampled on a `GRID × GRID × GRID` grid to estimate that
/// fraction: 125 points, deterministic and cheap.
pub const GRID: usize = 5;

/// The subject's world position and facing yaw at a frame. `None` when the
/// subject is gone — a deleted actor, or a hand-made document.
///
/// Only the yaw carries over: a subject tipping over (its pitch/roll) does not
/// roll the camera, which keeps the horizon where the director put it.
pub fn subject_transform(scene: &Scene, subject: CameraSubject, frame: Frame) -> Option<(Vec3, f32)> {
    match subject {
        CameraSubject::Phantom(id) => {
            let phantom = scene.phantoms.get(&id)?;
            let state = evaluate(phantom, frame);
            Some((state.root, state.yaw))
        }
        CameraSubject::Object(id) => {
            let object = scene.objects.get(&id)?;
            let t = evaluate_object(object, frame);
            Some((t.position, t.yaw))
        }
    }
}

/// The point a camera aims at when it tracks a subject: the middle of the
/// subject's own box, so a 1.74 m actor is framed around the chest and not
/// around the feet.
pub fn subject_aim(project: &Project, scene: &Scene, subject: CameraSubject, frame: Frame) -> Option<Vec3> {
    match subject {
        CameraSubject::Phantom(id) => {
            let phantom = scene.phantoms.get(&id)?;
            let state = evaluate(phantom, frame);
            let footprint = project.phantom_footprint(phantom, state.pose);
            Some(state.root + Quat::from_rotation_y(state.yaw) * footprint.offset_vec())
        }
        CameraSubject::Object(id) => {
            let object = scene.objects.get(&id)?;
            let t = evaluate_object(object, frame);
            let footprint = project.object_footprint(object, t.scale);
            Some(t.position + Quat::from_euler(glam::EulerRot::YXZ, t.yaw, t.pitch, t.roll) * footprint.offset_vec())
        }
    }
}

/// Angles that aim a camera at `target` from `eye`, in the renderer's basis
/// (`YXZ(yaw, pitch)`, looking down local −Z). A degenerate pair of points keeps
/// `fallback`.
pub fn aim_angles(eye: Vec3, target: Vec3, fallback: (f32, f32)) -> (f32, f32) {
    let d = target - eye;
    let horizontal = (d.x * d.x + d.z * d.z).sqrt();
    if horizontal < 1e-6 && d.y.abs() < 1e-6 {
        return fallback;
    }
    ((-d.x).atan2(-d.z), d.y.atan2(horizontal))
}

/// The camera's attachment (an unknown id has none).
pub fn camera_target(scene: &Scene, id: Option<CameraId>) -> Option<CameraTarget> {
    id.and_then(|id| scene.cameras.get(&id)).and_then(|c| c.target)
}

/// The world pose of a camera at a frame, with its subject attachment applied.
///
/// - A free camera is its own keys.
/// - **Tracking** (`Track`): the key position and angles stand, and the camera
///   only turns to keep the subject in view.
/// - **Following** (`Follow`): the keys are read in the subject's frame —
///   position, angles and aim point alike — so the camera travels and turns with
///   the subject, a third-person view. `Camera::sample` has already resolved a
///   key's `look_at` against that key's own (local) position, so the resolved
///   angles are local and simply compose with the subject's yaw.
///
/// A subject that is gone is ignored: the camera falls back to its own keys in
/// world space, which is deterministic and never a crash. `validate` reports the
/// dangling attachment (`camera_subject_missing`).
pub fn camera_pose(project: &Project, scene: &Scene, id: Option<CameraId>, frame: Frame) -> CameraPose {
    let pose = scene.camera_for(id).sample(frame);
    let Some(target) = camera_target(scene, id) else { return pose };
    match target.mode {
        CameraTargetMode::Track => {
            let Some(aim) = subject_aim(project, scene, target.subject, frame) else { return pose };
            let (yaw, pitch) = aim_angles(pose.translation, aim, (pose.yaw, pose.pitch));
            CameraPose { yaw, pitch, ..pose }
        }
        CameraTargetMode::Follow => {
            let Some((origin, yaw)) = subject_transform(scene, target.subject, frame) else { return pose };
            let rotation = Quat::from_rotation_y(yaw);
            CameraPose {
                translation: origin + rotation * pose.translation,
                yaw: yaw + pose.yaw,
                ..pose
            }
        }
    }
}

/// The camera's frame at a frame, with the attachment applied.
pub fn camera_frustum(project: &Project, scene: &Scene, id: Option<CameraId>, frame: Frame, aspect: f32) -> Frustum {
    Frustum::from_pose(camera_pose(project, scene, id, frame), aspect)
}

/// How much of the scene a camera takes in at a frame.
///
/// A camera standing **inside** the scene volume counts as 1.0: by decision, a
/// camera inside the room is looking at the scene, whatever it points at. That
/// also keeps the rule honest for big stages, where any interior shot sees only
/// a slice of the box while a distant camera sees most of it.
///
/// Otherwise the fraction of the stage's volume inside the frame, sampled on a
/// [`GRID`]³ grid.
pub fn scene_in_frame(project: &Project, scene: &Scene, id: Option<CameraId>, frame: Frame) -> f32 {
    scene_in_frame_with(project, scene, id, frame, DEFAULT_ASPECT)
}

/// [`scene_in_frame`] with an explicit frame aspect ratio.
pub fn scene_in_frame_with(project: &Project, scene: &Scene, id: Option<CameraId>, frame: Frame, aspect: f32) -> f32 {
    let pose = camera_pose(project, scene, id, frame);
    if scene.stage.contains(pose.translation) {
        return 1.0;
    }
    box_in_frame_fraction(&scene.stage, &Frustum::from_pose(pose, aspect))
}

/// The fraction of a stage box's volume inside a frame, sampled on a [`GRID`]³
/// grid of points (the grid avoids an exact volume intersection while staying
/// deterministic and cheap).
pub fn box_in_frame_fraction(stage: &Stage, frustum: &Frustum) -> f32 {
    let (lo, hi) = (stage.min(), stage.max());
    let span = hi - lo;
    let mut inside = 0usize;
    let mut total = 0usize;
    for ix in 0..GRID {
        for iy in 0..GRID {
            for iz in 0..GRID {
                let t = |i: usize| (i as f32 + 0.5) / GRID as f32;
                let point = lo + span * Vec3::new(t(ix), t(iy), t(iz));
                total += 1;
                if frustum.contains(point) {
                    inside += 1;
                }
            }
        }
    }
    if total == 0 {
        return 0.0;
    }
    inside as f32 / total as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{ObjectId, PhantomId};
    use crate::project::{Camera, ObjectKey, SceneCamera, SceneObject, TransformKey, SCENE_CAMERA};
    use crate::rig::Pose;
    use crate::time::FrameRange;

    fn scene_with_stage(preset: crate::stage::StagePreset) -> Scene {
        let mut scene = Scene::new(crate::id::SceneId(1), "s", FrameRange::new(0, 48).unwrap());
        scene.stage = crate::stage::Stage::from_preset(preset);
        scene
    }

    fn phantom_at(scene: &mut Scene, x: f32, z: f32, yaw: f32) {
        let key = crate::project::MotionKey { frame: 0, position: Vec3::new(x, 0.0, z), pose: crate::project::ActorPose::Stand, yaw };
        scene.phantoms.insert(
            PhantomId(2),
            crate::project::Phantom {
                id: PhantomId(2),
                name: "hero".into(),
                rig: crate::project::RigRef::BuiltinActor { actor: crate::project::BuiltinActor::Humanoid },
                motion: vec![key],
                outline_px: 2.0,
                scale: 1.0,
                material_patches: Vec::new(),
            },
        );
    }

    fn camera_at(scene: &mut Scene, id: CameraId, pos: Vec3, yaw: f32, target: Option<CameraTarget>) {
        let mut entry = SceneCamera::new(id, "cam", scene.range);
        entry.camera = Camera { keys: vec![TransformKey::at(0, pos, yaw, 0.0)], focal_mm: 35.0 };
        entry.target = target;
        scene.cameras.insert(id, entry);
    }

    #[test]
    fn a_camera_inside_the_scene_always_counts_as_looking_at_it() {
        let project = Project::new("t");
        let mut scene = scene_with_stage(crate::stage::StagePreset::Large);
        // Inside the volume, pointed at a wall: by decision still 100 %.
        camera_at(&mut scene, SCENE_CAMERA, Vec3::new(0.0, 2.0, 0.0), std::f32::consts::PI, None);
        assert_eq!(scene_in_frame(&project, &scene, Some(SCENE_CAMERA), 0), 1.0);
    }

    #[test]
    fn a_camera_outside_sees_a_fraction_and_a_camera_turned_away_sees_nothing() {
        let project = Project::new("t");
        let mut scene = scene_with_stage(crate::stage::StagePreset::Medium);
        // In front of the medium stage, looking at it: most of the volume is in frame.
        camera_at(&mut scene, SCENE_CAMERA, Vec3::new(0.0, 3.0, 15.0), 0.0, None);
        let looking = scene_in_frame(&project, &scene, Some(SCENE_CAMERA), 0);
        assert!(looking >= MIN_SCENE_IN_FRAME, "{looking}");
        // The same spot, turned around: nothing of the scene is in frame.
        camera_at(&mut scene, SCENE_CAMERA, Vec3::new(0.0, 3.0, 15.0), std::f32::consts::PI, None);
        let away = scene_in_frame(&project, &scene, Some(SCENE_CAMERA), 0);
        assert_eq!(away, 0.0);
        assert!(away < MIN_SCENE_IN_FRAME);
    }

    #[test]
    fn tracking_turns_the_camera_without_moving_it() {
        let project = Project::new("t");
        let mut scene = scene_with_stage(crate::stage::StagePreset::Medium);
        // The camera sits in front of the stage and aims at a subject behind it.
        camera_at(
            &mut scene,
            SCENE_CAMERA,
            Vec3::new(0.0, 2.0, 12.0),
            0.0,
            Some(CameraTarget::new(CameraTargetMode::Track, CameraSubject::Phantom(PhantomId(2)))),
        );
        phantom_at(&mut scene, 6.0, 0.0, 0.0);
        let pose = camera_pose(&project, &scene, Some(SCENE_CAMERA), 0);
        assert_eq!(pose.translation, Vec3::new(0.0, 2.0, 12.0), "tracking does not move the camera");
        // Looking at a subject 6 m to the right: the yaw turns toward it.
        assert!(pose.yaw < 0.0 && pose.yaw > -std::f32::consts::FRAC_PI_2, "yaw {}", pose.yaw);

        // The subject moves: the camera turns further, the position stays.
        scene.phantoms.get_mut(&PhantomId(2)).unwrap().motion[0].position = Vec3::new(-6.0, 0.0, 0.0);
        let moved = camera_pose(&project, &scene, Some(SCENE_CAMERA), 0);
        assert!(moved.yaw > 0.0, "yaw {}", moved.yaw);
        assert_eq!(moved.translation, pose.translation);
    }

    #[test]
    fn following_carries_the_camera_with_the_subject_and_turns_with_it() {
        let project = Project::new("t");
        let mut scene = scene_with_stage(crate::stage::StagePreset::Medium);
        // Third-person rig: 4 m behind (local -Z) and 1.5 m up, no look_at.
        camera_at(
            &mut scene,
            SCENE_CAMERA,
            Vec3::new(0.0, 1.5, -4.0),
            0.0,
            Some(CameraTarget::new(CameraTargetMode::Follow, CameraSubject::Phantom(PhantomId(2)))),
        );
        phantom_at(&mut scene, 5.0, 1.0, 0.0);
        let pose = camera_pose(&project, &scene, Some(SCENE_CAMERA), 0);
        assert!((pose.translation - Vec3::new(5.0, 1.5, -3.0)).length() < 1e-4, "{:?}", pose.translation);
        assert!((pose.yaw - 0.0).abs() < 1e-5);

        // The subject turns 90° about +Y: the camera orbits with it and turns too.
        scene.phantoms.get_mut(&PhantomId(2)).unwrap().motion[0].yaw = std::f32::consts::FRAC_PI_2;
        let turned = camera_pose(&project, &scene, Some(SCENE_CAMERA), 0);
        assert!((turned.yaw - std::f32::consts::FRAC_PI_2).abs() < 1e-4, "yaw {}", turned.yaw);
        // The local offset (0, 1.5, -4) rotates with the subject: behind it
        // (+Z at yaw 0) becomes 4 m to its left (-X) at yaw +90°.
        assert!((turned.translation - Vec3::new(1.0, 1.5, 1.0)).length() < 1e-4, "{:?}", turned.translation);

        // A local aim point keeps the subject centred whatever the subject does:
        // aiming at the subject's own box centre (the same point `Track` uses).
        let aim_local = crate::project::BuiltinActor::Humanoid
            .footprint(crate::project::ActorPose::Stand)
            .offset_vec();
        let mut aimed = scene.cameras[&SCENE_CAMERA].clone();
        aimed.camera.keys[0].look_at = Some(aim_local);
        scene.cameras.insert(SCENE_CAMERA, aimed);
        let pose = camera_pose(&project, &scene, Some(SCENE_CAMERA), 0);
        let aim = subject_aim(&project, &scene, CameraSubject::Phantom(PhantomId(2)), 0).unwrap();
        let (yaw, pitch) = aim_angles(pose.translation, aim, (0.0, 0.0));
        assert!((pose.yaw - yaw).abs() < 1e-4, "the aim point stays on the subject: {} vs {yaw}", pose.yaw);
        assert!((pose.pitch - pitch).abs() < 1e-4);
    }

    #[test]
    fn a_subject_that_is_gone_falls_back_to_the_cameras_own_keys() {
        let project = Project::new("t");
        let mut scene = scene_with_stage(crate::stage::StagePreset::Medium);
        camera_at(
            &mut scene,
            SCENE_CAMERA,
            Vec3::new(0.0, 1.5, -4.0),
            0.3,
            Some(CameraTarget::new(CameraTargetMode::Follow, CameraSubject::Phantom(PhantomId(9)))),
        );
        let pose = camera_pose(&project, &scene, Some(SCENE_CAMERA), 0);
        assert_eq!(pose.translation, Vec3::new(0.0, 1.5, -4.0), "the camera keeps its own keys");
        assert!((pose.yaw - 0.3).abs() < 1e-5);
    }

    #[test]
    fn the_grid_fraction_is_monotone_and_bounded() {
        let mut scene = scene_with_stage(crate::stage::StagePreset::Medium);
        let stage = scene.stage;
        let project = Project::new("t");
        // Backing off cannot lose the scene: a distant camera frames more of
        // the box than a close one, never less.
        camera_at(&mut scene, SCENE_CAMERA, Vec3::new(0.0, 4.0, 20.0), 0.0, None);
        let near = scene_in_frame(&project, &scene, Some(SCENE_CAMERA), 0);
        camera_at(&mut scene, SCENE_CAMERA, Vec3::new(0.0, 4.0, 60.0), 0.0, None);
        let far = scene_in_frame(&project, &scene, Some(SCENE_CAMERA), 0);
        assert!(far >= near, "{far} vs {near}");
        // A frame that contains the whole box is 1.0.
        let wide = Frustum::from_pose(CameraPose { translation: Vec3::new(0.0, 4.0, 200.0), yaw: 0.0, pitch: 0.0, roll: 0.0, focal_mm: 8.0 }, DEFAULT_ASPECT);
        let fraction = box_in_frame_fraction(&stage, &wide);
        assert!((0.0..=1.0).contains(&fraction));
        assert!(fraction > 0.9, "{fraction}");
    }

    #[test]
    fn an_object_can_be_a_subject_too() {
        let mut project = Project::new("t");
        let mut scene = scene_with_stage(crate::stage::StagePreset::Medium);
        let object = ObjectId(3);
        scene.objects.insert(
            object,
            SceneObject {
                id: object,
                name: "crate".into(),
                asset: crate::id::AssetId(0),
                builtin: Some("box".into()),
                keys: vec![ObjectKey { frame: 0, position: Vec3::new(2.0, 0.0, 0.0), yaw: 0.0, pitch: 0.0, roll: 0.0, scale: Vec3::ONE }],
                attach: None,
                material_patches: Vec::new(),
                scatter: None,
            },
        );
        let _ = &mut project;
        camera_at(
            &mut scene,
            SCENE_CAMERA,
            Vec3::new(0.0, 2.0, 10.0),
            0.0,
            Some(CameraTarget::new(CameraTargetMode::Track, CameraSubject::Object(object))),
        );
        let pose = camera_pose(&project, &scene, Some(SCENE_CAMERA), 0);
        assert!(pose.yaw < 0.0, "the camera turned toward the crate: {}", pose.yaw);
        // The aim point is the box's own centre (a unit box sits on the floor).
        let aim = subject_aim(&project, &scene, CameraSubject::Object(object), 0).unwrap();
        assert!((aim - Vec3::new(2.0, 0.5, 0.0)).length() < 1e-4, "{aim:?}");
        let _: Pose = Pose::rest();
    }
}

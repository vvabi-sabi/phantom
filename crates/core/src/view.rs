//! Pure camera framing for the `look` tool: the standard review directions and
//! the distance at which a staged target fits the frame.
//!
//! Everything here is a pure function of the document — no renderer, no GPU —
//! so the numbers an agent is shown can be unit-tested, and the same
//! conventions can be reused by the camera tools later.

use glam::Vec3;

use crate::project::{horizontal_fov, vertical_fov, Camera, CameraPose, KeyEase, TransformKey};
use crate::time::Frame;

/// Frame aspect used whenever a caller does not name one (16:9 — the MCP
/// default and the shape of the neural pipeline's frames).
pub const DEFAULT_ASPECT: f32 = 16.0 / 9.0;

/// A perspective frame: the camera's position, basis and field of view at one
/// frame. One implementation for the renderer's geometry and for every
/// diagnostic, so a coverage number always describes the frame that is drawn.
///
/// Angles and the aspect ratio are the only inputs; the basis is the renderer's
/// (`YXZ(yaw, pitch)`, looking down local −Z).
#[derive(Clone, Copy, Debug)]
pub struct Frustum {
    pub pos: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub focal_mm: f32,
    pub vfov: f32,
    pub hfov: f32,
    /// world → camera (the conjugate of YXZ(yaw, pitch)).
    inv_rot: glam::Quat,
}

/// Points closer than this to the camera plane are treated as behind it.
pub const MIN_DEPTH: f32 = 1e-3;

impl Frustum {
    /// The frame of a camera pose.
    pub fn from_pose(pose: CameraPose, aspect: f32) -> Self {
        // Focal length comes from the pose: a key may set its own, and then the
        // FOV must match the real frame.
        let focal_mm = pose.focal_mm;
        Self {
            pos: pose.translation,
            yaw: pose.yaw,
            pitch: pose.pitch,
            focal_mm,
            vfov: vertical_fov(focal_mm),
            hfov: horizontal_fov(focal_mm, aspect),
            inv_rot: glam::Quat::from_euler(glam::EulerRot::YXZ, pose.yaw, pose.pitch, 0.0).conjugate(),
        }
    }

    /// The frame of a camera at a frame.
    pub fn new(camera: &Camera, frame: crate::time::Frame, aspect: f32) -> Self {
        Self::from_pose(camera.sample(frame), aspect)
    }

    /// A world point in camera space.
    pub fn local(&self, p: Vec3) -> Vec3 {
        self.inv_rot * (p - self.pos)
    }

    /// Whether a point is in frame: in front of the camera and inside both half-FOVs.
    pub fn contains(&self, p: Vec3) -> bool {
        let l = self.local(p);
        if l.z > -MIN_DEPTH {
            return false;
        }
        let az = l.x.atan2(-l.z);
        let el = l.y.atan2(-l.z);
        az.abs() <= self.hfov * 0.5 && el.abs() <= self.vfov * 0.5
    }
}

/// A standard review direction: where the eye stands relative to the target.
///
/// "Front" is the side the staged characters face (forward +Z), so a `front`
/// view shows the actors as the scene's own front — the same convention as the
/// rest of the document. "Left" and "right" are named from that front view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LookView {
    Front,
    Back,
    Left,
    Right,
    Top,
    ThreeQuarter,
}

impl LookView {
    /// Every direction, in the order a review sheet reads them.
    pub const ALL: [LookView; 6] =
        [Self::Front, Self::Right, Self::Back, Self::Left, Self::Top, Self::ThreeQuarter];

    pub fn name(self) -> &'static str {
        match self {
            Self::Front => "front",
            Self::Back => "back",
            Self::Left => "left",
            Self::Right => "right",
            Self::Top => "top",
            Self::ThreeQuarter => "three_quarter",
        }
    }

    /// The directions a sheet uses when the caller names none.
    pub const DEFAULT: [LookView; 4] = [Self::Front, Self::Right, Self::Top, Self::ThreeQuarter];

    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|v| v.name() == name)
    }

    /// Unit direction from the target toward the eye — `eye = target + dir · d`.
    ///
    /// The eye sits slightly above the target except for `top`; `top` keeps a
    /// small horizontal offset so the view direction stays well defined (with a
    /// perfectly vertical view the yaw would be arbitrary).
    pub fn direction(self) -> Vec3 {
        match self {
            Self::Front => Vec3::new(0.0, 0.25, 1.0),
            Self::Back => Vec3::new(0.0, 0.25, -1.0),
            Self::Left => Vec3::new(-1.0, 0.25, 0.0),
            Self::Right => Vec3::new(1.0, 0.25, 0.0),
            Self::Top => Vec3::new(0.0, 1.0, 0.02),
            Self::ThreeQuarter => Vec3::new(1.0, 0.5, 1.0),
        }
        .normalize()
    }
}

/// Distance from the target center at which a target of these half-extents fits
/// the frame, with `margin` (>= 1) of slack, at this lens and aspect ratio.
///
/// The target is treated as its bounding sphere: the tighter of the two field
/// angles decides, because a sphere that fits vertically may still be cut off
/// horizontally. A sphere is orientation-independent, so every review direction
/// backs off the same distance and the tiles stay comparable.
pub fn fit_distance(half_extent: Vec3, focal_mm: f32, aspect: f32, margin: f32) -> f32 {
    let radius = half_extent.length().max(1e-3);
    let half_fov = (vertical_fov(focal_mm).min(horizontal_fov(focal_mm, aspect)) * 0.5).max(1e-3);
    let margin = if margin.is_finite() { margin.max(1.0) } else { 1.0 };
    (radius / half_fov.sin()) * margin
}

/// A camera aimed at `target` from `direction` at `distance`: one key with
/// `look_at`, so the angles are computed with the renderer's own basis and the
/// frame matches what the camera sees (no trigonometry for the caller).
pub fn camera_looking_at(target: Vec3, direction: Vec3, distance: f32, focal_mm: f32, frame: Frame) -> Camera {
    let direction = direction.normalize_or(Vec3::Z);
    let eye = target + direction * distance.max(0.01);
    Camera {
        keys: vec![TransformKey {
            frame,
            translation: eye,
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            look_at: Some(target),
            focal_mm: None,
            ease: KeyEase::Linear,
        }],
        focal_mm: if focal_mm.is_finite() && focal_mm > 0.0 { focal_mm } else { 35.0 },
    }
}

/// Axis-aligned box of a staged target: center and half-extents, meters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TargetBox {
    pub center: Vec3,
    pub half_extent: Vec3,
}

impl TargetBox {
    /// A box around one placed entity: its footprint under a rotation, at a
    /// position. `footprint` is the entity's local box (origin-relative).
    pub fn of(position: Vec3, footprint: crate::stage::Footprint, rotation: glam::Quat) -> Self {
        let half = footprint.world_half(rotation);
        let center = position + rotation * footprint.offset_vec();
        Self { center, half_extent: half }
    }

    /// The smallest box containing both.
    pub fn union(self, other: Self) -> Self {
        let lo = self.center - self.half_extent;
        let hi = self.center + self.half_extent;
        let olo = other.center - other.half_extent;
        let ohi = other.center + other.half_extent;
        let lo = lo.min(olo);
        let hi = hi.max(ohi);
        Self { center: (lo + hi) * 0.5, half_extent: (hi - lo) * 0.5 }
    }

    /// A box that contains nothing yet (the neutral element of `union`).
    pub fn empty() -> Self {
        Self { center: Vec3::ZERO, half_extent: Vec3::splat(-1.0) }
    }

    pub fn is_empty(self) -> bool {
        self.half_extent.min_element() < 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stage::Footprint;

    #[test]
    fn view_names_round_trip_and_directions_are_unit() {
        for v in LookView::ALL {
            assert_eq!(LookView::parse(v.name()), Some(v));
            assert!((v.direction().length() - 1.0).abs() < 1e-5, "{v:?}");
        }
        assert_eq!(LookView::parse("diagonal"), None);
        // Front looks at the scene from the side the characters face (+Z).
        assert!(LookView::Front.direction().z > 0.0);
        assert!(LookView::Back.direction().z < 0.0);
        assert!(LookView::Top.direction().y > 0.9);
    }

    #[test]
    fn fit_distance_grows_with_the_target_and_shrinks_with_the_lens() {
        let small = fit_distance(Vec3::splat(0.5), 35.0, 16.0 / 9.0, 1.0);
        let big = fit_distance(Vec3::splat(5.0), 35.0, 16.0 / 9.0, 1.0);
        assert!(big > small * 9.0);
        let wide = fit_distance(Vec3::splat(1.0), 16.0, 16.0 / 9.0, 1.0);
        let tele = fit_distance(Vec3::splat(1.0), 135.0, 16.0 / 9.0, 1.0);
        assert!(tele > wide * 5.0, "a longer lens backs off further: {tele} vs {wide}");
        let margin = fit_distance(Vec3::splat(1.0), 35.0, 16.0 / 9.0, 1.2);
        assert!(margin > fit_distance(Vec3::splat(1.0), 35.0, 16.0 / 9.0, 1.0));
        // A degenerate size never divides by zero.
        assert!(fit_distance(Vec3::ZERO, 35.0, 1.0, 1.0).is_finite());
    }

    /// Backing off by `fit_distance` really does put the whole target inside the
    /// frame: the target's sphere is inside the tighter cone.
    #[test]
    fn the_fitted_camera_contains_the_target() {
        let target = Vec3::new(1.0, 0.9, -2.0);
        let half = Vec3::new(1.4, 0.9, 0.6);
        let (focal, aspect) = (35.0_f32, 16.0 / 9.0);
        let d = fit_distance(half, focal, aspect, 1.05);
        for view in LookView::ALL {
            let cam = camera_looking_at(target, view.direction(), d, focal, 0);
            let pose = cam.sample(0);
            let to_target = target - pose.translation;
            // The angle between the view axis and the target center must leave
            // room for the target's angular radius.
            let forward = Vec3::new(-pose.yaw.sin() * pose.pitch.cos(), pose.pitch.sin(), -pose.yaw.cos() * pose.pitch.cos());
            let cos_angle = forward.dot(to_target.normalize());
            let target_angle = cos_angle.clamp(-1.0, 1.0).acos();
            let angular_radius = (half.length() / d).min(1.0).asin();
            let half_fov = vertical_fov(focal).min(horizontal_fov(focal, aspect)) * 0.5;
            assert!(target_angle + angular_radius <= half_fov + 1e-3, "{view:?}: target is cut off");
        }
    }

    #[test]
    fn target_box_unions_and_reports_emptiness() {
        let empty = TargetBox::empty();
        assert!(empty.is_empty());
        let a = TargetBox::of(Vec3::new(0.0, 0.0, 0.0), Footprint::standing(0.3, 1.8, 0.3), glam::Quat::IDENTITY);
        let b = TargetBox::of(Vec3::new(4.0, 0.0, 0.0), Footprint::standing(0.3, 1.8, 0.3), glam::Quat::IDENTITY);
        let both = empty.union(a).union(b);
        assert!(!both.is_empty());
        assert!((both.center.x - 2.0).abs() < 1e-4);
        assert!((both.half_extent.x - 2.3).abs() < 1e-4);
        // The box really contains both footprints.
        assert!(both.half_extent.y >= a.half_extent.y - 1e-5);
    }
}

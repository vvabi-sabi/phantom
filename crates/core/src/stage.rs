//! Scene stage (layout mode): the bounded area a scene is staged in.
//!
//! A stage is a box standing on the floor: `y` runs from 0 to the stage height,
//! `x`/`z` are centered on [`Stage::center`]. The box is invisible — it is a
//! boundary, not geometry. Every command that sets a position clamps the
//! entity's bounding box ([`Footprint`]) to it, so nothing can be staged
//! outside the scene.

use glam::{Mat3, Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::error::CoreError;
use crate::time::Frame;

/// Named stage preset. The sizes are a product decision: a small scene is a
/// room, a medium one a hall or a yard, a large one a street or a square.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StagePreset {
    Small,
    Medium,
    Large,
}

impl StagePreset {
    /// Every preset, in growing order.
    pub const ALL: [StagePreset; 3] = [Self::Small, Self::Medium, Self::Large];

    pub fn name(self) -> &'static str {
        match self {
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Large => "large",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.name() == name)
    }

    /// Stage size of the preset, meters: width × depth × height.
    pub fn size(self) -> StageSize {
        match self {
            Self::Small => StageSize::new(10.0, 4.0, 10.0),
            Self::Medium => StageSize::new(30.0, 8.0, 20.0),
            Self::Large => StageSize::new(80.0, 20.0, 60.0),
        }
    }
}

/// Stage dimensions in meters. Height is the vertical axis; width and depth are
/// horizontal, so a size reads the way a director says it out loud.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct StageSize {
    pub width: f32,
    pub height: f32,
    pub depth: f32,
}

impl StageSize {
    pub const fn new(width: f32, height: f32, depth: f32) -> Self {
        Self { width, height, depth }
    }
}

/// The scene stage: a floor-level box that actors and props are confined to.
/// Old manifests without the field get the medium preset.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Stage {
    /// The preset the size came from; `None` — the size was set by hand.
    #[serde(default)]
    pub preset: Option<StagePreset>,
    pub size: StageSize,
    /// Center of the box on the floor plane: `[x, z]`. The floor stays at `y = 0`.
    #[serde(default)]
    pub center: [f32; 2],
}

impl Default for Stage {
    fn default() -> Self {
        Self { preset: Some(StagePreset::Medium), size: StagePreset::Medium.size(), center: [0.0, 0.0] }
    }
}

impl Stage {
    /// Stage from a preset, centered on the origin.
    pub fn from_preset(preset: StagePreset) -> Self {
        Self { preset: Some(preset), size: preset.size(), center: [0.0, 0.0] }
    }

    /// Lower corner of the box (the floor is at `y = 0`).
    pub fn min(&self) -> Vec3 {
        Vec3::new(
            self.center[0] - self.size.width * 0.5,
            0.0,
            self.center[1] - self.size.depth * 0.5,
        )
    }

    /// Upper corner of the box.
    pub fn max(&self) -> Vec3 {
        Vec3::new(
            self.center[0] + self.size.width * 0.5,
            self.size.height,
            self.center[1] + self.size.depth * 0.5,
        )
    }

    /// Is the point inside the box (borders included)?
    pub fn contains(&self, point: Vec3) -> bool {
        let (lo, hi) = (self.min(), self.max());
        point.cmpge(lo).all() && point.cmple(hi).all()
    }

    /// Clamp an entity so its whole bounding box stays inside the stage.
    ///
    /// `position` is the entity origin (the Phantom root, the object origin),
    /// `footprint` its local box, `rotation` its orientation. Returns the
    /// clamped position and whether the position had to move. A box larger than
    /// the stage on some axis is centered on that axis instead of being pushed
    /// out — the box never leaves the stage in the other axes.
    pub fn clamp(&self, position: Vec3, footprint: Footprint, rotation: Quat) -> (Vec3, bool) {
        let half = footprint.world_half(rotation);
        let offset = rotation * Vec3::from(footprint.offset);
        let center = position + offset;
        let (lo, hi) = (self.min() + half, self.max() - half);
        let mut clamped = center;
        for axis in 0..3 {
            clamped[axis] = if lo[axis] > hi[axis] {
                // The box is bigger than the stage on this axis and cannot fit
                // at all. On the floor axis it stands on the ground and passes
                // through the ceiling — the floor is a hard surface. On a
                // horizontal axis it is centered, so the overflow is symmetric.
                if axis == 1 {
                    lo[axis]
                } else {
                    (lo[axis] + hi[axis]) * 0.5
                }
            } else {
                center[axis].clamp(lo[axis], hi[axis])
            };
        }
        let result = clamped - offset;
        // Rotating the footprint's own offset introduces float noise far below
        // any visible size; reporting it as a correction would tell the
        // director their coordinate was fixed when nothing moved.
        (result, (result - position).length() > 1e-4)
    }

    /// Is the entity's whole bounding box inside the stage?
    pub fn fits(&self, position: Vec3, footprint: Footprint, rotation: Quat) -> bool {
        let (_, moved) = self.clamp(position, footprint, rotation);
        !moved
    }

    /// Validate finiteness and positivity before the stage reaches the document.
    pub fn validate(&self) -> Result<(), CoreError> {
        let size = self.size;
        if !size.width.is_finite() || !size.height.is_finite() || !size.depth.is_finite() {
            return Err(CoreError::InvalidStage("size"));
        }
        if size.width <= 0.0 || size.height <= 0.0 || size.depth <= 0.0 {
            return Err(CoreError::InvalidStage("size must be positive"));
        }
        if !self.center[0].is_finite() || !self.center[1].is_finite() {
            return Err(CoreError::InvalidStage("center"));
        }
        Ok(())
    }
}

/// Bounding box of an entity in its own local frame, meters, relative to the
/// entity origin — the point a key or a placement command sets.
///
/// One type serves actors (origin at the feet), catalog props (origin at the
/// base center or the geometric center) and imported models (origin wherever
/// the file put it), so the clamp is the same arithmetic everywhere.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Footprint {
    /// Half-extents of the local box.
    pub half: [f32; 3],
    /// Box center relative to the entity origin.
    pub offset: [f32; 3],
}

impl Footprint {
    pub const fn new(half: [f32; 3], offset: [f32; 3]) -> Self {
        Self { half, offset }
    }

    /// A box standing on the origin: from `y = 0` up to `height`.
    pub const fn standing(half_x: f32, height: f32, half_z: f32) -> Self {
        Self {
            half: [half_x, height * 0.5, half_z],
            offset: [0.0, height * 0.5, 0.0],
        }
    }

    /// A box centered on the origin.
    pub const fn centered(half: [f32; 3]) -> Self {
        Self { half, offset: [0.0, 0.0, 0.0] }
    }

    /// A bare point: no extent at all (an entity whose size is unknown).
    pub const fn point() -> Self {
        Self::centered([0.0, 0.0, 0.0])
    }

    /// The same box under an entity scale.
    pub fn scaled(self, scale: Vec3) -> Self {
        Self {
            half: [
                self.half[0] * scale.x.abs(),
                self.half[1] * scale.y.abs(),
                self.half[2] * scale.z.abs(),
            ],
            offset: [self.offset[0] * scale.x, self.offset[1] * scale.y, self.offset[2] * scale.z],
        }
    }

    /// Half-extents of the world-axis-aligned box of this box under `rotation`:
    /// `|R| · half`. Exact for any orientation, so a turned chair is clamped by
    /// the space it really takes.
    pub fn world_half(self, rotation: Quat) -> Vec3 {
        let m = Mat3::from_quat(rotation);
        let h = Vec3::from(self.half);
        m.x_axis.abs() * h.x + m.y_axis.abs() * h.y + m.z_axis.abs() * h.z
    }

    /// Local box center as a vector.
    pub fn offset_vec(self) -> Vec3 {
        Vec3::from(self.offset)
    }
}

/// Which kind of entity a clamp moved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClampKind {
    Phantom,
    Object,
}

/// One clamp the stage performed: the entity, the key it belongs to, and the
/// coordinates before and after. Commands carry these so the agent learns that
/// its coordinate was corrected instead of silently getting something else.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Clamp {
    pub kind: ClampKind,
    /// Document id of the Phantom or the scene object.
    pub id: u64,
    /// Frame of the affected key (a stage change reports every moved key).
    pub frame: Frame,
    /// Requested position.
    pub from: [f32; 3],
    /// Position actually stored.
    pub to: [f32; 3],
}

/// Bounding box of an imported model, meters, relative to the model origin.
/// Computed once at import and stored on the asset; a model without recorded
/// bounds is clamped by its origin point instead.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelBounds {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl ModelBounds {
    pub fn footprint(&self) -> Footprint {
        Footprint {
            half: [
                (self.max[0] - self.min[0]).abs() * 0.5,
                (self.max[1] - self.min[1]).abs() * 0.5,
                (self.max[2] - self.min[2]).abs() * 0.5,
            ],
            offset: [
                (self.min[0] + self.max[0]) * 0.5,
                (self.min[1] + self.max[1]) * 0.5,
                (self.min[2] + self.max[2]) * 0.5,
            ],
        }
    }

    /// Half-extents of the box (a quick summary for reports).
    pub fn half(&self) -> [f32; 3] {
        self.footprint().half
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_grow_and_match_the_agreed_sizes() {
        let small = StagePreset::Small.size();
        let medium = StagePreset::Medium.size();
        let large = StagePreset::Large.size();
        assert_eq!((small.width, small.depth, small.height), (10.0, 10.0, 4.0));
        assert_eq!((medium.width, medium.depth, medium.height), (30.0, 20.0, 8.0));
        assert_eq!((large.width, large.depth, large.height), (80.0, 60.0, 20.0));
        assert!(small.width < medium.width && medium.width < large.width);
        assert!(small.height < medium.height && medium.height < large.height);
        for p in StagePreset::ALL {
            assert_eq!(StagePreset::parse(p.name()), Some(p));
        }
        assert_eq!(StagePreset::parse("huge"), None);
    }

    #[test]
    fn a_standing_actor_is_clamped_by_its_whole_body() {
        let stage = Stage::from_preset(StagePreset::Small); // x,z ∈ ±5, y ∈ [0,4]
        let human = Footprint::standing(0.3, 1.74, 0.25);
        // Far outside: the body lands just inside the wall, not half through it.
        let (p, moved) = stage.clamp(Vec3::new(99.0, 0.0, -99.0), human, Quat::IDENTITY);
        assert!(moved);
        assert!((p.x - 4.7).abs() < 1e-4, "{p:?}");
        assert!((p.z + 4.75).abs() < 1e-4, "{p:?}");
        assert_eq!(p.y, 0.0);
        // The whole body fits: the top of the head is below the ceiling.
        assert!(stage.fits(p, human, Quat::IDENTITY));
        // Already inside: nothing moves.
        let inside = Vec3::new(1.0, 0.0, 1.0);
        assert_eq!(stage.clamp(inside, human, Quat::IDENTITY), (inside, false));
    }

    #[test]
    fn height_is_clamped_too_and_a_huge_box_is_centered() {
        let stage = Stage::from_preset(StagePreset::Small);
        let tall = Footprint::standing(0.2, 1.0, 0.2);
        // Above the ceiling: the feet come down.
        let (p, moved) = stage.clamp(Vec3::new(0.0, 10.0, 0.0), tall, Quat::IDENTITY);
        assert!(moved);
        assert!((p.y - 3.0).abs() < 1e-4, "{p:?}");
        // A box bigger than the stage is centered horizontally and stands on
        // the floor, not pushed out.
        let giant = Footprint::standing(20.0, 40.0, 20.0);
        let (p, _) = stage.clamp(Vec3::new(3.0, 3.0, 3.0), giant, Quat::IDENTITY);
        assert_eq!(p, Vec3::ZERO);
        // Too tall for the stage: feet on the floor, head through the ceiling.
        let tall = Footprint::standing(0.3, 9.0, 0.3);
        let (p, _) = stage.clamp(Vec3::new(0.0, 5.0, 0.0), tall, Quat::IDENTITY);
        assert_eq!(p.y, 0.0, "an oversized actor still stands on the floor");
    }

    #[test]
    fn a_turned_box_is_clamped_by_the_space_it_really_takes() {
        let stage = Stage::from_preset(StagePreset::Small);
        // A long, thin object: 2 m long, 0.2 m wide.
        let plank = Footprint::centered([0.1, 0.05, 1.0]);
        // Along z it needs 1 m of clearance; turned 90° it needs it along x.
        let (along, _) = stage.clamp(Vec3::new(99.0, 0.05, 0.0), plank, Quat::IDENTITY);
        assert!((along.x - 4.9).abs() < 1e-4, "{along:?}");
        let turned = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let (across, _) = stage.clamp(Vec3::new(99.0, 0.05, 0.0), plank, turned);
        assert!((across.x - 4.0).abs() < 1e-4, "{across:?}");
        // The world half-extent really swaps axes.
        let h0 = plank.world_half(Quat::IDENTITY);
        let h1 = plank.world_half(turned);
        assert!((h0.x - 0.1).abs() < 1e-5 && (h0.z - 1.0).abs() < 1e-5);
        assert!((h1.x - 1.0).abs() < 1e-5 && (h1.z - 0.1).abs() < 1e-5);
    }

    #[test]
    fn a_centered_origin_is_handled_like_a_standing_one() {
        let stage = Stage::from_preset(StagePreset::Small);
        // A monitor with its origin at the geometric center, 0.6 × 0.4 × 0.06.
        let monitor = Footprint::centered([0.3, 0.2, 0.03]);
        let (p, _) = stage.clamp(Vec3::new(0.0, 0.05, 0.0), monitor, Quat::IDENTITY);
        assert!((p.y - 0.2).abs() < 1e-4, "the bottom edge stays on the floor: {p:?}");
        let (p, _) = stage.clamp(Vec3::new(0.0, 99.0, 0.0), monitor, Quat::IDENTITY);
        assert!((p.y - 3.8).abs() < 1e-4, "{p:?}");
    }

    #[test]
    fn a_scale_grows_the_box_and_the_stage_validates_itself() {
        let stage = Stage::from_preset(StagePreset::Small);
        let wolf = Footprint::standing(0.2, 0.57, 0.55);
        let doubled = wolf.scaled(Vec3::splat(2.0));
        assert!((doubled.half[1] - 0.57).abs() < 1e-5);
        let (p, _) = stage.clamp(Vec3::new(0.0, 0.0, 99.0), doubled, Quat::IDENTITY);
        assert!((p.z - 3.9).abs() < 1e-4, "{p:?}");
        assert!(stage.validate().is_ok());
        assert!(Stage { preset: None, size: StageSize::new(0.0, 4.0, 4.0), center: [0.0, 0.0] }.validate().is_err());
        assert!(Stage { preset: None, size: StageSize::new(f32::NAN, 4.0, 4.0), center: [0.0, 0.0] }.validate().is_err());
        assert!(Stage { preset: None, size: StageSize::new(1.0, 4.0, 4.0), center: [f32::INFINITY, 0.0] }.validate().is_err());
    }

    #[test]
    fn model_bounds_become_a_footprint() {
        let bounds = ModelBounds { min: [-0.5, 0.0, -0.2], max: [0.5, 1.0, 0.2] };
        let f = bounds.footprint();
        assert_eq!(f.half, [0.5, 0.5, 0.2]);
        assert_eq!(f.offset, [0.0, 0.5, 0.0]);
        assert_eq!(bounds.half(), [0.5, 0.5, 0.2]);
    }
}

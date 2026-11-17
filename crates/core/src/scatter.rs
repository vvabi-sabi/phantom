//! Generators that fill an area or a line with catalog props (forest,
//! undergrowth, hedge, street row, pole row).
//!
//! The plan is a **pure function** of (preset, shape, seed), and both the
//! command that places editable objects and the renderer that builds the far
//! band call it. That is the whole reason it lives here: what the director sees
//! in the command's answer and what appears in the frame must be the same
//! computation, or the contact sheet lies.
//!
//! Randomness is a local splitmix64, so the plan does not depend on a library's
//! internal hashing and the same seed grows the same forest on any machine.
//! Placement is a jittered grid: a cell of `1/√density` with one random point
//! inside. A plain grid reads as a plantation, and pure rejection sampling
//! makes the count depend on luck; a jittered grid reads as a wood and always
//! yields the same number of points for the same area.

use serde::{Deserialize, Serialize};

use crate::error::CoreError;

/// Editable props placed as separate scene objects. The same ceiling as
/// `prop_grid`: past it the document, its undo history and the per-frame
/// rebuild stop being pleasant to work with.
pub const MAX_SCATTER_OBJECTS: usize = 256;
/// Instances baked into one far-band mesh. They are geometry rather than
/// entities, so the ceiling is far higher — but the mesh is still built on the
/// CPU, and it is one mesh.
pub const MAX_SCATTER_INSTANCES: usize = 2000;

/// One kind the fill draws from, with its relative weight and the scale range
/// an instance is drawn at.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScatterSpecies {
    /// Catalog entry name.
    pub prop: &'static str,
    /// Relative chance of being picked.
    pub weight: f32,
    /// Scale range: a wood is not a row of identical trees.
    pub scale: [f32; 2],
}

const fn species(prop: &'static str, weight: f32, lo: f32, hi: f32) -> ScatterSpecies {
    ScatterSpecies { prop, weight, scale: [lo, hi] }
}

/// A fill recipe. `near` fills the zone the scene is shot in — those become
/// editable objects. `far` fills the band beyond it — that becomes one mesh, so
/// it favours tall species that read as a silhouette and drops the small stuff
/// that would only cost triangles. Line presets cycle `near` instead of
/// sampling it by weight: a street row is tree, lamp, bin, tree, lamp, bin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScatterPreset {
    pub name: &'static str,
    pub near: &'static [ScatterSpecies],
    pub far: &'static [ScatterSpecies],
    /// Default props per square meter.
    pub density: f32,
    /// Density multiplier of the far band: a wall of forest reads as solid long
    /// before it is dense enough to walk through.
    pub far_density: f32,
    /// Default spacing in meters for line presets.
    pub step: f32,
    /// Lateral offset in meters for line presets — a hedge is tight, a street
    /// row wanders a little off the curb.
    pub line_jitter: f32,
    /// Scale range for line presets. Manufactured items (lamps, poles) come at
    /// 1.0 exactly: a street with lamps of three heights is a bug, not variety.
    pub line_scale: [f32; 2],
    /// How far a line instance may turn off the row, in degrees.
    pub line_yaw_deg: f32,
    pub note: &'static str,
}

/// The fills available by name. Species and densities are chosen so each preset
/// reads as the thing it is named after at its own defaults.
pub const SCATTER_PRESETS: &[ScatterPreset] = &[
    ScatterPreset {
        name: "forest",
        // A mixed wood: two storeys of trees, bramble underneath, the odd stone.
        near: &[
            species("tree", 3.0, 0.8, 1.3),
            species("tree_pine", 3.0, 0.8, 1.3),
            species("bush", 3.0, 0.7, 1.4),
            species("rock", 1.0, 0.6, 1.4),
        ],
        far: &[
            species("tree", 3.0, 0.9, 1.3),
            species("tree_pine", 4.0, 0.9, 1.4),
            species("bush", 1.0, 0.8, 1.2),
        ],
        density: 0.2,
        far_density: 0.5,
        step: 2.5,
        line_jitter: 0.3,
        line_scale: [1.0, 1.0],
        line_yaw_deg: 30.0,
        note: "mixed wood: pines and broadleaves, undergrowth, stones",
    },
    ScatterPreset {
        name: "grove",
        // Open and walkable: the place you shoot IN, not the backdrop.
        near: &[
            species("tree", 3.0, 1.0, 1.4),
            species("tree_pine", 2.0, 1.0, 1.3),
            species("bush", 1.0, 0.8, 1.2),
            species("rock", 1.0, 0.7, 1.3),
        ],
        far: &[species("tree", 2.0, 1.0, 1.4), species("tree_pine", 3.0, 1.0, 1.4)],
        density: 0.06,
        far_density: 0.4,
        step: 6.0,
        line_jitter: 0.4,
        line_scale: [1.0, 1.0],
        line_yaw_deg: 40.0,
        note: "open grove: wide-spaced tall trees, almost no undergrowth",
    },
    ScatterPreset {
        name: "undergrowth",
        // Low, dense, impassable-looking: the background you shoot OVER.
        near: &[
            species("bush", 6.0, 0.6, 1.5),
            species("rock", 2.0, 0.6, 1.4),
            species("tree", 1.0, 0.35, 0.55),
        ],
        far: &[species("bush", 5.0, 0.7, 1.4), species("tree", 1.0, 0.4, 0.6)],
        density: 0.5,
        far_density: 0.6,
        step: 1.2,
        line_jitter: 0.25,
        line_scale: [0.85, 1.15],
        line_yaw_deg: 45.0,
        note: "thicket and clearing edge: bushes, saplings, stones",
    },
    ScatterPreset {
        name: "hedge",
        near: &[species("bush", 1.0, 0.9, 1.15)],
        far: &[],
        density: 1.0,
        far_density: 0.0,
        step: 1.0,
        line_jitter: 0.1,
        line_scale: [0.95, 1.1],
        line_yaw_deg: 12.0,
        note: "a shrub wall along a line: garden, park, plot boundary",
    },
    ScatterPreset {
        name: "street_row",
        // The curb in order: tree, lamp, litter bin, then repeat.
        near: &[
            species("tree", 1.0, 1.0, 1.0),
            species("streetlight", 1.0, 1.0, 1.0),
            species("bin", 1.0, 1.0, 1.0),
        ],
        far: &[],
        density: 1.0,
        far_density: 0.0,
        step: 8.0,
        line_jitter: 0.15,
        line_scale: [1.0, 1.0],
        line_yaw_deg: 0.0,
        note: "curb row: tree, streetlight and bin repeating along the road",
    },
    ScatterPreset {
        name: "pole_row",
        near: &[species("pole", 1.0, 1.0, 1.0)],
        far: &[],
        density: 1.0,
        far_density: 0.0,
        step: 5.0,
        line_jitter: 0.0,
        line_scale: [1.0, 1.0],
        line_yaw_deg: 0.0,
        note: "a line of utility poles along a fence line or road",
    },
];

/// Preset by name.
pub fn scatter_preset(name: &str) -> Option<&'static ScatterPreset> {
    SCATTER_PRESETS.iter().find(|p| p.name == name)
}

/// Where to fill, in the field's OWN frame: centered on the origin, running
/// along local +X. The command turns the world rectangle or segment the
/// director gave into local numbers, and the renderer is handed the same shape
/// — so the band turns and moves with its object instead of being a pile of
/// baked world coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ScatterShape {
    /// Rectangle of `size` meters, with the optional exclusion box (the zone
    /// the scene is shot in: the band never grows inside it).
    Area { size: [f32; 2], density: f32, action: Option<[[f32; 2]; 2]> },
    /// Segment of `length` meters along +X, centered on the origin, one
    /// instance every `step`, offset sideways by at most `lateral`.
    Line { length: f32, step: f32, lateral: f32 },
}

/// One placed thing. `near` marks the zone the scene is shot in: those become
/// scene objects, the rest belongs to the band mesh.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlacedProp {
    pub prop: &'static str,
    /// Local position, base on `y = 0`.
    pub position: [f32; 3],
    /// Radians.
    pub yaw: f32,
    /// Uniform scale.
    pub scale: f32,
    pub near: bool,
}

/// splitmix64: deterministic output that does not depend on a library version
/// or a platform hash, which is what a saved document has to be able to rely on.
#[derive(Clone, Copy, Debug)]
pub struct SeedRng(u64);

impl SeedRng {
    pub fn new(seed: u64) -> Self {
        SeedRng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        // 24 bits — exactly the mantissa of an f32.
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn between(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }

    /// Uniform integer in `0..below` (0 when the range is empty). splitmix64 is
    /// strong enough that the modulo bias is irrelevant at the sizes a document
    /// holds (hundreds), and the alternative — rejection sampling — would make
    /// the number of draws depend on the value, which a saved plan cannot have.
    pub fn below(&mut self, below: usize) -> usize {
        if below == 0 {
            0
        } else {
            (self.next_u64() % below as u64) as usize
        }
    }
}

/// Is the point inside the box given as two corners?
fn inside(box_: &[[f32; 2]; 2], x: f32, z: f32) -> bool {
    let (a, b) = (box_[0], box_[1]);
    x >= a[0].min(b[0])
        && x <= a[0].max(b[0])
        && z >= a[1].min(b[1])
        && z <= a[1].max(b[1])
}

fn pick(species: &[ScatterSpecies], rng: &mut SeedRng) -> ScatterSpecies {
    let total: f32 = species.iter().map(|s| s.weight.max(0.0)).sum();
    let mut roll = rng.next_f32() * total;
    for s in species {
        roll -= s.weight.max(0.0);
        if roll <= 0.0 {
            return *s;
        }
    }
    *species.last().expect("the preset test checks species lists are non-empty")
}

/// One point of a jittered grid, drawn in full before anything is culled: the
/// number of random draws must not depend on the exclusion box, or dragging the
/// clearing would reshuffle the wood standing behind it. The cell is clipped to
/// the rectangle, so no instance is ever planted outside the requested area.
fn cell_point(rng: &mut SeedRng, cell: f32, ix: usize, iz: usize, size: [f32; 2]) -> [f32; 2] {
    let mut axis = |i: usize, dim: f32| {
        let lo = -dim * 0.5 + i as f32 * cell;
        let hi = (lo + cell).min(dim * 0.5);
        lo + rng.next_f32() * (hi - lo).max(0.0)
    };
    [axis(ix, size[0]), axis(iz, size[1])]
}

fn push(
    out: &mut Vec<PlacedProp>,
    kind: ScatterSpecies,
    position: [f32; 2],
    yaw: f32,
    scale: f32,
    near: bool,
) {
    out.push(PlacedProp {
        prop: kind.prop,
        position: [position[0], 0.0, position[1]],
        yaw,
        scale: scale.clamp(kind.scale[0].min(kind.scale[1]), kind.scale[0].max(kind.scale[1])),
        near,
    });
}

fn validate(preset: &ScatterPreset, shape: &ScatterShape) -> Result<(), CoreError> {
    let bad = |why: &str| CoreError::InvalidScatter(why.to_string());
    if preset.near.is_empty() {
        return Err(bad("preset has nothing to place"));
    }
    match shape {
        ScatterShape::Area { size, density, action } => {
            if !size.iter().all(|v| v.is_finite() && *v > 0.0) {
                return Err(bad("area size must be finite and > 0"));
            }
            if !density.is_finite() || *density <= 0.0 {
                return Err(bad("density must be finite and > 0"));
            }
            if action.is_some_and(|b| !b.iter().flatten().all(|v| v.is_finite())) {
                return Err(bad("action box must be finite"));
            }
        }
        ScatterShape::Line { length, step, lateral } => {
            if !length.is_finite() || *length <= 0.0 {
                return Err(bad("line length must be finite and > 0"));
            }
            if !step.is_finite() || *step <= 0.0 {
                return Err(bad("line step must be finite and > 0"));
            }
            if !lateral.is_finite() || *lateral < 0.0 {
                return Err(bad("line lateral offset must be finite and >= 0"));
            }
        }
    }
    Ok(())
}

/// Fill the shape.
///
/// The near and far layers draw from **separate streams** of the same seed, so
/// the exclusion box only relabels and culls points: moving the clearing never
/// jumps the forest behind it.
pub fn plan(
    preset: &ScatterPreset,
    shape: &ScatterShape,
    seed: u64,
) -> Result<Vec<PlacedProp>, CoreError> {
    validate(preset, shape)?;
    let mut placed = Vec::new();
    match shape {
        ScatterShape::Area { size, density, action } => {
            // The action box is the ground the scene is shot ON: it gets
            // editable objects the director can move, turn and delete. Beyond
            // it, out to the edge of the area, stands the band — one mesh, a
            // silhouette you shoot AGAINST and do not edit tree by tree.
            // Without an action box the whole area is shootable ground, so
            // everything is an object and there is no band.
            let cell = 1.0 / density.sqrt();
            let nx = (size[0] / cell).ceil() as usize;
            let nz = (size[1] / cell).ceil() as usize;
            let mut rng = SeedRng::new(seed);
            for iz in 0..nz {
                for ix in 0..nx {
                    // Drawn first, always: see `cell_point`.
                    let kind = pick(preset.near, &mut rng);
                    let p = cell_point(&mut rng, cell, ix, iz, *size);
                    let yaw = rng.next_f32() * std::f32::consts::TAU;
                    let scale = rng.between(kind.scale[0], kind.scale[1]);
                    // Outside the shoot zone there is nothing to edit by hand.
                    if action.is_some_and(|b| !inside(&b, p[0], p[1])) {
                        continue;
                    }
                    push(&mut placed, kind, p, yaw, scale, true);
                }
            }
            if let (Some(box_), true) = (action, !preset.far.is_empty()) {
                let far_density = (density * preset.far_density).max(1e-4);
                let far_cell = 1.0 / far_density.sqrt();
                let fx = (size[0] / far_cell).ceil() as usize;
                let fz = (size[1] / far_cell).ceil() as usize;
                let mut rng = SeedRng::new(seed ^ 0xD1B5_4A32_D192_ED03);
                for iz in 0..fz {
                    for ix in 0..fx {
                        let kind = pick(preset.far, &mut rng);
                        let p = cell_point(&mut rng, far_cell, ix, iz, *size);
                        let yaw = rng.next_f32() * std::f32::consts::TAU;
                        let scale = rng.between(kind.scale[0], kind.scale[1]);
                        // The band is the RING: the clearing stays clear.
                        if inside(&box_, p[0], p[1]) {
                            continue;
                        }
                        push(&mut placed, kind, p, yaw, scale, false);
                    }
                }
            }
        }
        ScatterShape::Line { length, step, lateral } => {
            let count = (length / step).floor() as usize + 1;
            let half = (count as f32 - 1.0) * step * 0.5;
            let jitter = preset.line_yaw_deg.to_radians();
            let mut rng = SeedRng::new(seed);
            for i in 0..count {
                // A line cycles its composition instead of sampling it: the
                // curb order of a street is the point of the preset.
                let kind = preset.near[i % preset.near.len()];
                let p = [-half + i as f32 * step, rng.between(-*lateral, *lateral)];
                let yaw = rng.between(-jitter * 0.5, jitter * 0.5);
                let scale = rng.between(preset.line_scale[0], preset.line_scale[1]);
                push(&mut placed, kind, p, yaw, scale, true);
            }
        }
    }

    let near = placed.iter().filter(|p| p.near).count();
    let far = placed.len() - near;
    if near > MAX_SCATTER_OBJECTS {
        return Err(CoreError::InvalidScatter(format!(
            "{} would place {near} editable props, the limit is {MAX_SCATTER_OBJECTS}: lower the density, grow the step, or split the area into several calls",
            preset.name
        )));
    }
    if far > MAX_SCATTER_INSTANCES {
        return Err(CoreError::InvalidScatter(format!(
            "{} would bake {far} instances into one band, the limit is {MAX_SCATTER_INSTANCES}: lower the density or shrink the area",
            preset.name
        )));
    }
    Ok(placed)
}

/// The field's local box for the stage clamp: the full plate, tall enough for
/// the tallest instance the preset can draw at its largest scale.
pub fn field_footprint(preset: &ScatterPreset, shape: &ScatterShape) -> crate::stage::Footprint {
    let top = band_top_y(preset);
    match shape {
        ScatterShape::Area { size, .. } => crate::stage::Footprint::standing(size[0] * 0.5, top, size[1] * 0.5),
        ScatterShape::Line { length, lateral, .. } => {
            // Depth is the deepest species plus the lateral drift of the row.
            let source = if preset.near.is_empty() { preset.far } else { preset.near };
            let depth = source
                .iter()
                .map(|s| {
                    crate::catalog::catalog_find(s.prop)
                        .map(|e| e.size[2] * s.scale[1] * 0.5)
                        .unwrap_or(0.0)
                })
                .fold(0.0f32, f32::max);
            crate::stage::Footprint::standing(length * 0.5, top, depth + lateral)
        }
    }
}

/// Everything the band holds (the far layer). The renderer draws exactly this,
/// so what the command counted is what gets built.
pub fn band_of(placed: &[PlacedProp]) -> Vec<PlacedProp> {
    placed.iter().copied().filter(|p| !p.near).collect()
}

/// The tallest thing a preset can plant, at its largest scale — the height the
/// field's stage footprint is measured with.
pub fn band_top_y(preset: &ScatterPreset) -> f32 {
    let source = if preset.far.is_empty() { preset.near } else { preset.far };
    source
        .iter()
        .map(|s| {
            crate::catalog::catalog_find(s.prop)
                .map(|e| e.size[1] * s.scale[1])
                .unwrap_or(0.0)
        })
        .fold(0.0f32, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::catalog_find;

    fn area(size: [f32; 2], density: f32, action: Option<[[f32; 2]; 2]>) -> ScatterShape {
        ScatterShape::Area { size, density, action }
    }

    #[test]
    fn the_same_seed_grows_the_same_wood() {
        let preset = scatter_preset("forest").unwrap();
        let a = plan(preset, &area([10.0, 10.0], 0.2, None), 7).unwrap();
        let b = plan(preset, &area([10.0, 10.0], 0.2, None), 7).unwrap();
        assert_eq!(a, b, "a saved document must replay identically");
        assert!(!a.iter().any(|p| !p.near), "no action box — everything is shootable ground");
    }

    #[test]
    fn the_shoot_zone_gets_editable_props_and_the_ring_gets_the_band() {
        let preset = scatter_preset("forest").unwrap();
        let box_ = [[-3.0, -3.0], [3.0, 3.0]];
        let placed = plan(preset, &area([16.0, 16.0], 0.3, Some(box_)), 5).unwrap();
        // Everything you can move stands where you shoot; the band is outside.
        assert!(placed.iter().filter(|p| p.near).all(|p| p.position[0].abs() <= 3.0 && p.position[2].abs() <= 3.0));
        assert!(placed.iter().filter(|p| !p.near).all(|p| p.position[0].abs() > 3.0 || p.position[2].abs() > 3.0));
        assert!(placed.iter().any(|p| p.near), "the shoot zone is furnished");
        assert!(placed.iter().any(|p| !p.near), "a ring of forest stands around the clearing");
    }

    #[test]
    fn every_instance_lands_inside_the_area() {
        let preset = scatter_preset("grove").unwrap();
        let placed = plan(preset, &area([12.0, 8.0], 0.06, None), 3).unwrap();
        assert!(!placed.is_empty());
        for p in placed {
            assert!(p.position[0].abs() <= 6.0 + 1e-3 && p.position[2].abs() <= 4.0 + 1e-3, "{p:?}");
        }
    }

    #[test]
    fn a_line_cycles_its_composition_in_order() {
        let preset = scatter_preset("street_row").unwrap();
        let line = ScatterShape::Line { length: 40.0, step: 8.0, lateral: 0.2 };
        let placed = plan(preset, &line, 1).unwrap();
        let props: Vec<_> = placed.iter().map(|p| p.prop).collect();
        assert_eq!(props, vec!["tree", "streetlight", "bin", "tree", "streetlight", "bin"]);
        // A street is straight: the row runs along X with only the lateral drift.
        assert!(placed.windows(2).all(|w| w[1].position[0] > w[0].position[0]));
    }

    #[test]
    fn too_many_props_is_a_clear_error_not_a_thinner_forest() {
        let preset = scatter_preset("undergrowth").unwrap();
        // 20x20 at 0.5/m² wants ~200; push it past the 256 ceiling.
        let err = plan(preset, &area([40.0, 40.0], 0.5, None), 1).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("limit"), "{msg}");
        assert!(msg.contains("density"), "{msg} tells the director what to lower");
    }

    #[test]
    fn presets_are_well_formed_and_named() {
        let mut names: Vec<_> = SCATTER_PRESETS.iter().map(|p| p.name).collect();
        let n = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), n, "preset names are unique");
        for p in SCATTER_PRESETS {
            assert!(!p.near.is_empty() && !p.note.is_empty(), "{}", p.name);
            for s in p.near.iter().chain(p.far) {
                assert!(catalog_find(s.prop).is_some(), "{}: unknown prop {}", p.name, s.prop);
                assert!(s.scale[0] <= s.scale[1] && s.weight > 0.0, "{}: {}", p.name, s.prop);
            }
            assert!(p.density > 0.0 && p.step > 0.0, "{}", p.name);
        }
    }
}

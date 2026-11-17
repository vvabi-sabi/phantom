//! Crowds: a plan that turns a count and a formation into people.
//!
//! Same rule as [`crate::scatter`]: the plan is a **pure function** of
//! (formation, size, roles, seed), and both the command that creates the
//! phantoms and any later read of the crowd replay it. A director who re-runs
//! the command with the numbers the answer reported gets the same crowd back —
//! positions, roles and glances alike.
//!
//! A crowd is a batch of real phantoms, not one mesh. People are acted in, so
//! each of them has to stay individually movable and keyable; the ceiling is
//! correspondingly low (a skinned mesh costs more than a prop).

use serde::{Deserialize, Serialize};

use crate::catalog::HUMAN_ROLES;
use crate::error::CoreError;
use crate::scatter::SeedRng;

/// People in one crowd. A skinned humanoid is heavier than a prop, and every
/// one of them is an entity with its own key track; past this the document and
/// its undo history stop being comfortable. The error says what to lower.
pub const MAX_CROWD_PHANTOMS: usize = 128;

/// Default spacing between neighbors, in meters. Close enough that a crowd
/// reads as one mass from a camera, far enough that two people do not swap arms.
pub const CROWD_SPACING: f32 = 0.8;

/// How far a scattered person may drift off their grid point, as a fraction of
/// the cell. The closest two neighbors then cannot come nearer than 2/3 of the
/// spacing (0.53 m at the default 0.8). Half a cell reads wilder and put two
/// people 0.2 m apart in a smoke test — bodies through bodies, which is the one
/// thing a crowd generator may never do. [`crowd_footprint`] bounds the same
/// shape, so the two must not drift apart.
const JITTER_FRACTION: f32 = 1.0 / 6.0;

/// A person the planner put somewhere.
#[derive(Clone, Debug, PartialEq)]
pub struct PlacedPerson {
    /// Human role from [`crate::catalog::HUMAN_ROLES`] (`adult|child|teen|
    /// elder`). The core speaks roles; turning one into a rig name belongs to the
    /// surface that creates a phantom.
    pub actor: &'static str,
    /// Local position, feet on `y = 0`, in the crowd's own frame: +X across the
    /// formation, -Z the direction it faces.
    pub position: [f32; 3],
    /// Radians: the direction the person looks.
    pub yaw: f32,
}

/// How the crowd stands. Named after what the director says out loud.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CrowdFormation {
    /// Rank and file: soldiers, a queue of two, a class photo.
    Line,
    /// An arc opening toward -Z: a crowd that faces the camera head-on and
    /// still shows its width.
    Arc,
    /// A V opening toward -Z: the classic "crowd walking toward camera".
    Wedge,
    /// No structure — a market square, a waiting crowd.
    Scattered,
    /// One file: a queue at a counter, people through a doorway.
    Queue,
}

/// The formations by name.
pub fn crowd_formation(name: &str) -> Option<CrowdFormation> {
    use CrowdFormation::*;
    Some(match name {
        "line" => Line,
        "arc" => Arc,
        "wedge" => Wedge,
        "scattered" => Scattered,
        "queue" => Queue,
        _ => return None,
    })
}

/// The name a formation is addressed by on the wire and in errors.
pub fn formation_name(formation: CrowdFormation) -> &'static str {
    use CrowdFormation::*;
    match formation {
        Line => "line",
        Arc => "arc",
        Wedge => "wedge",
        Scattered => "scattered",
        Queue => "queue",
    }
}

/// Every formation name, for error messages and tool descriptions.
pub const CROWD_FORMATIONS: &[&str] = &["line", "arc", "wedge", "scattered", "queue"];

/// Who is allowed in the crowd. "mixed" is the interesting default: a crowd of
/// identical adults reads as a mannequin factory.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CrowdMix {
    /// Every human role in [`crate::catalog::HUMAN_ROLES`], weighted toward
    /// adults the way a street is.
    Mixed,
    /// Only grown-ups.
    Adults,
    /// A named role, exactly as spelled in the catalog.
    Role(String),
}

/// The plan's input: what to build, not where it stands. Position and turn of
/// the whole crowd belong to the objects it creates, so that the plan stays
/// local and the crowd can be moved afterwards.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CrowdSpec {
    pub formation: CrowdFormation,
    pub count: usize,
    pub mix: CrowdMix,
    /// Meters between neighbors.
    pub spacing: f32,
    /// Rows of an area formation; without it the rows follow from the count.
    pub rows: Option<usize>,
    /// Radians: how far a person may look off the formation's facing.
    pub scatter_deg: f32,
}

impl CrowdSpec {
    /// The spec a command carries, before validation. `scatter_deg` defaults to
    /// 60°: a crowd that all stare the same way is a firing squad.
    pub fn new(formation: CrowdFormation, count: usize, mix: CrowdMix) -> Self {
        Self { formation, count, mix, spacing: CROWD_SPACING, rows: None, scatter_deg: 60.0 }
    }
}

/// Roles the mix allows, with the weights that pick between them. Adult-heavy:
/// a street is mostly grown-ups, and the two child-sized bodies in the frame are
/// what makes it read as a real crowd.
fn roles_of(mix: &CrowdMix) -> Vec<(&'static str, f32)> {
    match mix {
        CrowdMix::Adults => vec![("adult", 1.0)],
        CrowdMix::Mixed => vec![("adult", 6.0), ("teen", 2.0), ("elder", 1.5), ("child", 1.0)],
        CrowdMix::Role(name) => {
            HUMAN_ROLES.iter().filter(|r| r.role == name.as_str()).map(|r| (r.role, 1.0)).collect()
        }
    }
}

fn pick_role(roles: &[(&'static str, f32)], rng: &mut SeedRng) -> &'static str {
    let total: f32 = roles.iter().map(|(_, w)| *w).sum();
    let mut roll = rng.next_f32() * total;
    for (role, weight) in roles {
        roll -= weight;
        if roll <= 0.0 {
            return role;
        }
    }
    roles.last().expect("the mix always keeps at least one role").0
}

/// Rows and files of an area formation: as close to square as the count allows,
/// because a crowd of 40 in one rank is a queue, not a crowd.
fn grid_of(count: usize, forced: Option<usize>) -> (usize, usize) {
    let rows = forced.unwrap_or_else(|| {
        let root = (count as f32).sqrt().floor() as usize;
        (1..=root).rev().find(|r| count % r == 0).unwrap_or(1)
    });
    (rows.max(1), count.div_ceil(rows.max(1)))
}

/// The spot of person `i` in a formation of `count`, in the crowd's local frame.
/// Four of the five shapes are pure geometry — a test can assert them without a
/// seed. `scattered` is the exception: its jitter comes from `rng`, which is why
/// [`crowd_footprint`] measures that shape's envelope from the grid instead.
fn slot(formation: CrowdFormation, spec: &CrowdSpec, i: usize, rng: &mut SeedRng) -> [f32; 2] {
    let d = spec.spacing;
    match formation {
        CrowdFormation::Queue => [0.0, -(i as f32) * d * 1.3],
        CrowdFormation::Line => {
            let (_, cols) = grid_of(spec.count, spec.rows);
            [(i % cols) as f32 * d - (cols as f32 - 1.0) * d * 0.5, -((i / cols) as f32) * d]
        }
        CrowdFormation::Wedge => {
            // Grow the V outward from the point: the tip stands alone, and every
            // later person joins one shoulder further out and one step deeper.
            // (Roughly i/2 would plant person 1 on the tip's spot.)
            let r = ((i + 1) as f32 * 0.5).floor();
            let side = if i % 2 == 0 { 1.0 } else { -1.0 };
            [side * r * d, -r * d]
        }
        CrowdFormation::Arc => {
            // An arc of people spread over up to 140°, opening away from the
            // camera: the ends turn toward the lens, which is why a crowd
            // photographed in an arc looks bigger than one in a line.
            let n = spec.count.max(1) as f32;
            let radius = (n * d).max(d) / 2.4;
            let sweep = 2.44; // ~140°
            let t = if spec.count <= 1 { 0.0 } else { i as f32 / (n - 1.0) - 0.5 };
            let angle = t * sweep;
            // The ends come toward the camera and the middle recedes — the arc
            // is concave on the lens and hugs it, which is why a crowd shot in
            // an arc reads as bigger than the same number in a line. Subtracting
            // cos(sweep/2) puts the end rank at the local origin.
            [angle.sin() * radius, radius * (angle.cos() - (sweep * 0.5).cos())]
        }
        CrowdFormation::Scattered => {
            // A jittered grid, same idea as the fill: cell `d`, point somewhere
            // inside it. Structured enough that nobody stands inside anybody,
            // irregular enough that it does not read as a formation.
            let (rows, cols) = grid_of(spec.count, spec.rows);
            let ix = i % cols;
            let iz = i / cols;
            [
                ix as f32 * d - (cols as f32 - 1.0) * d * 0.5 + rng.between(-JITTER_FRACTION, JITTER_FRACTION) * d,
                iz as f32 * d - (rows as f32 - 1.0) * d * 0.5 + rng.between(-JITTER_FRACTION, JITTER_FRACTION) * d,
            ]
        }
    }
}

/// Build the crowd: who stands where and who they look at.
///
/// Roles and glances are drawn in a fixed order (person 0 first), so adding one
/// person to the count never restyles the people already there — a director who
/// grows a crowd from 20 to 24 does not watch the first twenty change clothes.
pub fn plan_crowd(spec: &CrowdSpec, seed: u64) -> Result<Vec<PlacedPerson>, CoreError> {
    if spec.count == 0 {
        return Err(CoreError::InvalidCrowd("crowd: count must be at least 1".into()));
    }
    if spec.count > MAX_CROWD_PHANTOMS {
        return Err(CoreError::InvalidCrowd(format!(
            "crowd: {count} people is over the limit of {MAX_CROWD_PHANTOMS} — split it into several crowds (they stay one undo step per call), or lower the count",
            count = spec.count
        )));
    }
    if !spec.spacing.is_finite() || spec.spacing <= 0.0 {
        return Err(CoreError::InvalidCrowd(format!(
            "crowd: spacing must be finite and > 0, got {}",
            spec.spacing
        )));
    }
    if !spec.scatter_deg.is_finite() {
        return Err(CoreError::InvalidCrowd("crowd: scatter_deg must be finite".into()));
    }
    let roles = roles_of(&spec.mix);
    if roles.is_empty() {
        let known = HUMAN_ROLES.iter().map(|r| r.role).collect::<Vec<_>>().join(", ");
        let given = match &spec.mix {
            CrowdMix::Role(name) => name.clone(),
            _ => "?".into(),
        };
        return Err(CoreError::InvalidCrowd(format!(
            "crowd: unknown role \"{given}\" (known roles: {known})"
        )));
    }
    if let Some(rows) = spec.rows {
        if matches!(spec.formation, CrowdFormation::Arc | CrowdFormation::Wedge | CrowdFormation::Queue) {
            return Err(CoreError::InvalidCrowd(format!(
                "crowd: the {} formation has no rows (its shape follows from the count) — drop rows",
                formation_name(spec.formation)
            )));
        }
        if rows == 0 || rows > spec.count {
            return Err(CoreError::InvalidCrowd(format!(
                "crowd: rows must be between 1 and the count ({}), got {rows}",
                spec.count
            )));
        }
    }

    let jitter = spec.scatter_deg.to_radians() * 0.5;
    let mut rng = SeedRng::new(seed);
    let mut people = Vec::with_capacity(spec.count);
    for i in 0..spec.count {
        // Drawn in person order, always, before the spot is even computed:
        // nothing downstream may skip a draw, or one person's role would shift
        // the whole crowd's style. See the function's doc comment.
        let role = pick_role(&roles, &mut rng);
        let glance = rng.between(-jitter, jitter);
        let [x, z] = slot(spec.formation, spec, i, &mut rng);
        people.push(PlacedPerson {
            actor: role,
            position: [x, 0.0, z],
            yaw: glance,
        });
    }
    Ok(people)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(formation: &str, count: usize) -> CrowdSpec {
        CrowdSpec::new(crowd_formation(formation).unwrap(), count, CrowdMix::Mixed)
    }

    #[test]
    fn the_same_seed_gathers_the_same_crowd() {
        let a = plan_crowd(&spec("scattered", 12), 4).unwrap();
        let b = plan_crowd(&spec("scattered", 12), 4).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, plan_crowd(&spec("scattered", 12), 5).unwrap(), "the seed must matter");
    }

    #[test]
    fn a_mixed_crowd_holds_more_than_one_kind_of_person() {
        let people = plan_crowd(&spec("line", 40), 9).unwrap();
        let kinds: std::collections::BTreeSet<_> = people.iter().map(|p| p.actor).collect();
        assert!(kinds.len() > 1, "a crowd of one silhouette: {kinds:?}");
        assert!(kinds.iter().all(|k| crate::catalog::human_role(k).is_some() || *k == "adult"));
        // One named role is exactly that role.
        let only = CrowdSpec::new(CrowdFormation::Line, 8, CrowdMix::Adults);
        assert!(plan_crowd(&only, 1).unwrap().iter().all(|p| p.actor == "adult"));
    }

    #[test]
    fn a_formation_keeps_its_shape_and_everyone_stays_on_the_floor() {
        // Line: everyone in the ranks, no two people in one spot.
        let line = plan_crowd(&spec("line", 12), 2).unwrap();
        assert_eq!(line.len(), 12);
        assert!(line.iter().all(|p| p.position[1] == 0.0));
        // Compared at decimeter precision: the default spacing is 0.8 m, so
        // rounding to meters would report two people in one spot when they are
        // nearly a meter apart.
        let distinct = |people: &[PlacedPerson]| {
            people
                .iter()
                .map(|p| ((p.position[0] * 10.0).round() as i32, (p.position[2] * 10.0).round() as i32))
                .collect::<std::collections::BTreeSet<_>>()
        };
        assert_eq!(distinct(&line).len(), 12, "two people share a spot in a line");
        // Queue: one file, always deeper, never wider.
        let queue = plan_crowd(&spec("queue", 6), 2).unwrap();
        assert!(queue.iter().all(|p| p.position[0] == 0.0));
        assert!(queue.windows(2).all(|w| w[1].position[2] < w[0].position[2]));
        // Wedge: the tip is in front, and it widens as it goes back.
        let wedge = plan_crowd(&spec("wedge", 9), 2).unwrap();
        assert_eq!(wedge[0].position, [0.0, 0.0, 0.0]);
        let front = wedge[..3].iter().map(|p| p.position[0].abs()).fold(0.0f32, f32::max);
        let back = wedge[6..].iter().map(|p| p.position[0].abs()).fold(0.0f32, f32::max);
        assert!(back > front, "a wedge that narrows backward is not a wedge");
        // Arc: concave on the lens, so the ends stand nearest and the middle is
        // the deepest person of the rank.
        let arc = plan_crowd(&spec("arc", 11), 2).unwrap();
        let deepest = arc
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.position[2].total_cmp(&b.position[2]))
            .unwrap()
            .0;
        assert!(deepest > 2 && deepest < 8, "the arc's deep point should be its middle, got {deepest}");
        // ...and its neighbors do not overlap: the arc keeps the ~spacing stride.
        assert_eq!(distinct(&arc).len(), 11, "the arc planted two people in one spot");
    }

    #[test]
    fn nobody_stands_inside_anybody() {
        // The failure mode of a fill generator is not a wrong silhouette, it is
        // two bodies sharing one space — so the stride is an invariant, not a
        // look. Measured at the default spacing over a full 10x10 scatter.
        let people = plan_crowd(&spec("scattered", 100), 3).unwrap();
        let closest = |a: &PlacedPerson, b: &PlacedPerson| {
            ((a.position[0] - b.position[0]).powi(2) + (a.position[2] - b.position[2]).powi(2)).sqrt()
        };
        let mut worst = f32::MAX;
        for (i, a) in people.iter().enumerate() {
            for b in people[i + 1..].iter() {
                worst = worst.min(closest(a, b));
            }
        }
        assert!(worst >= CROWD_SPACING * 0.66, "closest pair {:.3} m apart", worst);
    }

    #[test]
    fn the_crowd_envelope_holds_what_the_plan_can_produce() {
        // The envelope decides whether a crowd is refused, so it may not be one
        // lucky sample: a tighter box would turn down a crowd that fits, and
        // which seed was asked for would decide the answer.
        let mut spec = spec("scattered", 60);
        spec.rows = Some(6);
        let box_ = crowd_footprint(&spec);
        for seed in 0..40u64 {
            for person in plan_crowd(&spec, seed).unwrap() {
                let at = [person.position[0], 0.0, person.position[2]];
                let lo = [
                    box_.offset[0] - box_.half[0],
                    box_.offset[1] - box_.half[1],
                    box_.offset[2] - box_.half[2],
                ];
                let hi = [
                    box_.offset[0] + box_.half[0],
                    box_.offset[1] + box_.half[1],
                    box_.offset[2] + box_.half[2],
                ];
                for axis in 0..3 {
                    assert!(
                        at[axis] >= lo[axis] - 1e-4 && at[axis] <= hi[axis] + 1e-4,
                        "seed {seed}: person outside the envelope on axis {axis}: {at:?} not in {lo:?}..{hi:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_too_big_crowd_is_a_clear_error_with_the_limit_in_it() {
        let err = plan_crowd(&spec("scattered", MAX_CROWD_PHANTOMS + 1), 1).unwrap_err().to_string();
        assert!(err.contains("128"), "{err}");
        assert!(err.contains("split"), "the error must say what to do: {err}");
    }

    #[test]
    fn nonsense_input_is_refused_before_anything_is_built() {
        assert!(plan_crowd(&spec("line", 0), 1).is_err(), "an empty crowd");
        assert!(plan_crowd(&CrowdSpec::new(CrowdFormation::Line, 4, CrowdMix::Role("goblin".into())), 1)
            .unwrap_err()
            .to_string()
            .contains("goblin"));
        let mut s = spec("arc", 6);
        s.rows = Some(2);
        assert!(plan_crowd(&s, 1).unwrap_err().to_string().contains("no rows"));
        let mut s = spec("line", 6);
        s.spacing = 0.0;
        assert!(plan_crowd(&s, 1).is_err());
    }

    #[test]
    fn formations_are_named_the_way_the_surface_spells_them() {
        for name in CROWD_FORMATIONS {
            let f = crowd_formation(name).unwrap_or_else(|| panic!("{name} missing from the lookup"));
            assert_eq!(formation_name(f), *name);
        }
        assert!(crowd_formation("phalanx").is_none());
    }
}

/// The crowd's local box: the whole formation, tall enough for the tallest role
/// the mix can draw.
///
/// This exists because a crowd must be clamped as ONE body. Clamping each
/// person by their own point — which is what a plain motion key does — pushes
/// everyone who sticks past the border to the border, and a queue in a narrow
/// room collapses into a pile at the wall: the formation stops being a
/// formation. So the batch shifts the entire crowd by one offset and every
/// person keeps their spacing.
pub fn crowd_footprint(spec: &CrowdSpec) -> crate::stage::Footprint {
    // The envelope is a real bounding box, not a radius: a queue is deep and
    // narrow, and calling it a square would turn down a queue that fits along
    // the wall. Half-extent plus the center, so the formation may sit anywhere
    // inside it — the origin is not its middle.
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
    if spec.formation == CrowdFormation::Scattered {
        // The jitter is a random draw, so the measured box above is one
        // particular crowd's, and an unlucky seed would report a tighter box
        // than the plan is allowed to produce — which is a refusal the caller
        // could not reproduce. Take the grid and add the cap instead: this box
        // holds every seed, and refuses nothing that would actually fit.
        let (rows, cols) = grid_of(spec.count, spec.rows);
        let (jx, jz) = (JITTER_FRACTION * spec.spacing, JITTER_FRACTION * spec.spacing);
        lo = [
            -(cols as f32 - 1.0) * spec.spacing * 0.5 - jx,
            -(rows as f32 - 1.0) * spec.spacing * 0.5 - jz,
        ];
        hi = [
            (cols as f32 - 1.0) * spec.spacing * 0.5 + jx,
            (rows as f32 - 1.0) * spec.spacing * 0.5 + jz,
        ];
    } else {
        // The other four shapes are deterministic, so their envelope is simply
        // the box their spots occupy.
        let mut rng = SeedRng::new(0);
        for i in 0..spec.count {
            let [x, z] = slot(spec.formation, spec, i, &mut rng);
            lo = [lo[0].min(x), lo[1].min(z)];
            hi = [hi[0].max(x), hi[1].max(z)];
        }
    }
    let height = roles_of(&spec.mix)
        .iter()
        .filter_map(|(role, _)| crate::catalog::human_role(role))
        .map(|r| r.height_m)
        .fold(0.0f32, f32::max)
        .max(1.74);
    // Half a body of slack each way: a person is not a point.
    let half = [
        (hi[0] - lo[0]) * 0.5 + 0.4,
        height * 0.5,
        (hi[1] - lo[1]) * 0.5 + 0.4,
    ];
    let offset = [(lo[0] + hi[0]) * 0.5, height * 0.5, (lo[1] + hi[1]) * 0.5];
    crate::stage::Footprint::new(half, offset)
}

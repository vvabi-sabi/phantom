//! Metric catalog of simple props (corrective_2 K1/K3).
//!
//! One table instead of verbal agreements: dimensions in meters, origin at base
//! center (except where noted), named material groups for `material_set`,
//! cut-away walls for interiors. Runtime (`props`) builds geometry from these
//! entries; here there are only the public sizes and composition, so an agent
//! can place items without opening the mesh.

use serde::{Deserialize, Serialize};

/// Catalog section.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogKind {
    /// Primitive: cube, cylinder, cone, sphere, prism, tube, ramp.
    Primitive,
    /// Furniture and items.
    Furniture,
    /// Interiors: a room box with cut-away walls.
    Interior,
}

impl CatalogKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Primitive => "primitive",
            Self::Furniture => "furniture",
            Self::Interior => "interior",
        }
    }
}

/// Where the origin sits relative to the dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogOrigin {
    /// Origin at base center: the item stands on `y = 0`, neither sunk into the
    /// floor nor floating.
    BaseCenter,
    /// Geometric center (for suspended and wall-mounted items).
    Center,
}

/// Catalog entry: everything needed to place an item without opening the mesh.
/// Built from constants, so it only serializes (outward — JSON responses).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct CatalogEntry {
    pub name: &'static str,
    pub kind: CatalogKind,
    /// Full dimensions [x, y, z] in meters.
    pub size: [f32; 3],
    pub origin: CatalogOrigin,
    /// Named material groups (match `target` in `material_set`).
    pub materials: &'static [&'static str],
    /// Interior walls that are not built (`north|south|east|west`) — the
    /// camera side must not cover the frame.
    pub open_sides: &'static [&'static str],
    /// Open passageways, if an interior has them (door/window).
    pub openings: &'static [&'static str],
    /// Short hint for the director.
    pub note: &'static str,
}

impl CatalogEntry {
    /// Height of the topmost point above the floor with origin `BaseCenter`.
    pub fn top_y(&self) -> f32 {
        match self.origin {
            CatalogOrigin::BaseCenter => self.size[1],
            CatalogOrigin::Center => self.size[1] * 0.5,
        }
    }
}

/// The catalog table — the single source of metric prop dimensions.
pub const CATALOG: &[CatalogEntry] = &[
    // --- primitives: unit sizes, scaled by the object key ---
    entry("box", CatalogKind::Primitive, [1.0, 1.0, 1.0], &["body"], "cube 1×1×1 m; scale via the object key"),
    entry("cylinder", CatalogKind::Primitive, [1.0, 1.0, 1.0], &["body"], "cylinder d=1 m, h=1 m, origin at base center"),
    entry("cone", CatalogKind::Primitive, [1.0, 1.0, 1.0], &["body"], "cone d=1 m, h=1 m"),
    entry("sphere", CatalogKind::Primitive, [1.0, 1.0, 1.0], &["body"], "ellipsoid d=1 m (scaling gives an ellipsoid)"),
    entry("prism", CatalogKind::Primitive, [1.0, 1.0, 1.0], &["body"], "triangular prism 1×1×1 m"),
    entry("tube", CatalogKind::Primitive, [1.0, 1.0, 1.0], &["body"], "tube d=1 m, h=1 m, wall 0.15 m"),
    entry("ramp", CatalogKind::Primitive, [1.0, 1.0, 1.0], &["body"], "ramp 1×1×1 m (inclined plane)"),
    // --- furniture and items: metric sizes, origin at base center ---
    entry("table", CatalogKind::Furniture, [1.4, 0.75, 0.8], &["top", "frame"], "dining table: tabletop at 0.75 m"),
    entry("chair", CatalogKind::Furniture, [0.45, 0.9, 0.5], &["seat", "back", "frame"], "chair: seat at 0.45 m, backrest up to 0.9 m"),
    entry("stool", CatalogKind::Furniture, [0.35, 0.45, 0.35], &["seat", "frame"], "stool: seat at 0.45 m"),
    entry("bench", CatalogKind::Furniture, [1.6, 0.45, 0.4], &["seat", "frame"], "bench 1.6 m"),
    entry("cabinet", CatalogKind::Furniture, [0.8, 0.8, 0.45], &["body", "front"], "cabinet: height 0.8 m"),
    entry("wardrobe", CatalogKind::Furniture, [1.2, 2.0, 0.6], &["body", "front"], "wardrobe: height 2.0 m"),
    entry("shelf", CatalogKind::Furniture, [0.9, 1.8, 0.35], &["body", "shelves"], "bookshelf: height 1.8 m"),
    entry("bed", CatalogKind::Furniture, [1.6, 0.55, 2.0], &["mattress", "frame"], "bed: sleeping surface 0.55 m, length 2.0 m"),
    entry("sofa", CatalogKind::Furniture, [1.9, 0.85, 0.9], &["seat", "back", "frame"], "sofa: seat at 0.45 m, backrest up to 0.85 m"),
    entry("monitor", CatalogKind::Furniture, [0.6, 0.4, 0.06], &["screen", "frame"], "monitor/TV: placed on a surface, origin at center"),
    entry("laptop", CatalogKind::Furniture, [0.34, 0.24, 0.24], &["screen", "keyboard"], "laptop: open, screen tilted up and back"),
    entry("book", CatalogKind::Furniture, [0.15, 0.23, 0.03], &["cover", "pages"], "book: height 0.23 m"),
    entry("glass", CatalogKind::Furniture, [0.08, 0.14, 0.08], &["body"], "drinking glass: height 0.14 m"),
    entry("plate", CatalogKind::Furniture, [0.24, 0.03, 0.24], &["body"], "plate: diameter 0.24 m"),
    entry("lamp", CatalogKind::Furniture, [0.35, 1.5, 0.35], &["shade", "frame"], "floor lamp: height 1.5 m; light is added separately (`light_add`)"),
    // --- interiors: a box with cut-away walls and published dimensions ---
    entry_room("room_kitchen", [4.0, 2.7, 5.0], &["north"], &["door_south"], "kitchen 4×5 m, ceiling 2.7 m; open wall is north (camera side)"),
    entry_room("room_bedroom", [4.0, 2.7, 4.0], &["north"], &["door_south"], "bedroom 4×4 m, ceiling 2.7 m"),
    entry_room("room_living", [5.0, 2.7, 6.0], &["north"], &["door_south"], "living room 5×6 m, ceiling 2.7 m"),
    entry_room("room_corridor", [1.6, 2.5, 6.0], &["north"], &["door_south", "door_north"], "corridor 1.6×6 m, ceiling 2.5 m"),
    entry_room("room_toilet", [1.6, 2.5, 2.2], &["north"], &["door_south"], "toilet 1.6×2.2 m, ceiling 2.5 m"),
    entry_room("room_bathroom", [2.4, 2.5, 2.6], &["north"], &["door_south"], "bathroom 2.4×2.6 m, ceiling 2.5 m"),
];

const fn entry(
    name: &'static str,
    kind: CatalogKind,
    size: [f32; 3],
    materials: &'static [&'static str],
    note: &'static str,
) -> CatalogEntry {
    CatalogEntry { name, kind, size, origin: CatalogOrigin::BaseCenter, materials, open_sides: &[], openings: &[], note }
}

const fn entry_room(
    name: &'static str,
    size: [f32; 3],
    open_sides: &'static [&'static str],
    openings: &'static [&'static str],
    note: &'static str,
) -> CatalogEntry {
    CatalogEntry {
        name,
        kind: CatalogKind::Interior,
        size,
        origin: CatalogOrigin::BaseCenter,
        materials: &["floor", "wall", "ceiling"],
        open_sides,
        openings,
        note,
    }
}

/// Entry by name.
pub fn catalog_find(name: &str) -> Option<&'static CatalogEntry> {
    CATALOG.iter().find(|e| e.name == name)
}

/// Catalog search: section and/or name substring (case-insensitive).
/// An empty query returns the whole section — the agent needs an overview, not
/// blind guessing.
pub fn catalog_query(kind: Option<CatalogKind>, query: Option<&str>) -> Vec<&'static CatalogEntry> {
    let needle = query.map(|q| q.to_lowercase());
    CATALOG
        .iter()
        .filter(|e| kind.is_none_or(|k| e.kind == k))
        .filter(|e| needle.as_deref().is_none_or(|n| n.is_empty() || e.name.contains(n) || e.note.to_lowercase().contains(n)))
        .collect()
}

/// Human body reference lines by role (K3) — for "not above the knee" checks
/// and for picking camera height. Heights are a product decision and do not change.
pub const HUMAN_ROLES: &[HumanRole] = &[
    HumanRole { role: "child", height_m: 1.15, eye_m: 1.05, shoulder_m: 0.95, waist_m: 0.65, knee_m: 0.30 },
    HumanRole { role: "teen", height_m: 1.55, eye_m: 1.44, shoulder_m: 1.28, waist_m: 0.88, knee_m: 0.42 },
    HumanRole { role: "adult", height_m: 1.74, eye_m: 1.62, shoulder_m: 1.43, waist_m: 0.99, knee_m: 0.47 },
    HumanRole { role: "elder", height_m: 1.66, eye_m: 1.53, shoulder_m: 1.36, waist_m: 0.95, knee_m: 0.45 },
];

/// Height and role reference lines in meters.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct HumanRole {
    pub role: &'static str,
    pub height_m: f32,
    pub eye_m: f32,
    pub shoulder_m: f32,
    pub waist_m: f32,
    pub knee_m: f32,
}

/// Role by name.
pub fn human_role(role: &str) -> Option<&'static HumanRole> {
    HUMAN_ROLES.iter().find(|r| r.role == role)
}

/// Withers height of built-in animals (K3): one table, not guesswork.
pub const ANIMAL_HEIGHTS: &[(&str, f32)] = &[
    ("humanoid", 1.74),
    ("wolf", 0.57),
    ("horse", 1.62),
    ("bird", 0.48),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_names_are_unique_and_metric() {
        let mut names: Vec<&str> = CATALOG.iter().map(|e| e.name).collect();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count, "catalog names are unique");
        for e in CATALOG {
            assert!(e.size.iter().all(|v| *v > 0.0 && v.is_finite()), "{}: {:?}", e.name, e.size);
            assert!(!e.materials.is_empty(), "{}: material groups are required", e.name);
            assert!(!e.note.is_empty(), "{}: a hint for the director", e.name);
        }
    }

    #[test]
    fn furniture_sits_on_the_floor_and_rooms_are_walkable() {
        // Origin at base center: the tabletop is at 0.75 m, not "waist-deep in the floor".
        assert!((catalog_find("table").unwrap().top_y() - 0.75).abs() < 1e-6);
        assert!((catalog_find("wardrobe").unwrap().top_y() - 2.0).abs() < 1e-6);
        let room = catalog_find("room_kitchen").unwrap();
        assert_eq!(room.kind, CatalogKind::Interior);
        assert_eq!(room.open_sides, &["north"], "the wall on the camera side is cut away");
        assert!(room.size[1] >= 2.4, "the ceiling is above adult height");
        assert_eq!(room.materials, &["floor", "wall", "ceiling"]);
    }

    #[test]
    fn query_filters_by_kind_and_substring() {
        assert!(catalog_query(Some(CatalogKind::Interior), None).len() >= 6);
        assert!(catalog_query(Some(CatalogKind::Furniture), None).len() >= 12);
        let found = catalog_query(None, Some("room"));
        assert!(found.iter().all(|e| e.kind == CatalogKind::Interior));
        let chairs = catalog_query(None, Some("chair"));
        assert_eq!(chairs.len(), 1);
        assert_eq!(chairs[0].name, "chair");
        assert!(catalog_query(None, Some("no-such")).is_empty());
        assert!(catalog_query(None, None).len() == CATALOG.len(), "an empty query gives an overview of everything");
    }

    #[test]
    fn human_reference_lines_grow_with_the_role() {
        let adult = human_role("adult").unwrap();
        let child = human_role("child").unwrap();
        assert!(adult.height_m > child.height_m);
        for r in HUMAN_ROLES {
            assert!(r.eye_m < r.height_m && r.knee_m < r.waist_m && r.waist_m < r.shoulder_m);
        }
        // "A dog is not above an adult's knee": the table confirms it.
        let knee = adult.knee_m;
        let wolf = ANIMAL_HEIGHTS.iter().find(|(n, _)| *n == "wolf").unwrap().1;
        assert!(wolf < knee + 0.15, "wolf {wolf} against the knee at {knee}");
        // A horse is above a human's shoulder — the edit went in the growing direction.
        let horse = ANIMAL_HEIGHTS.iter().find(|(n, _)| *n == "horse").unwrap().1;
        assert!(horse > adult.shoulder_m, "horse {horse} is above the shoulder at {}", adult.shoulder_m);
    }
}

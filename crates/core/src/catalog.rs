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
    /// Animal figures: static props (an animal moves by its transform keys),
    /// with pose variants where they read in frame (`cat_sitting`...).
    Animal,
    /// Street props: items that live outdoors (pole, streetlight, tree...).
    Street,
    /// Interiors: a room box with cut-away walls.
    Interior,
    /// Outdoor grounds: a ground plate with an orientation, no walls or
    /// ceiling (yard, park, crossing); props are placed separately.
    Exterior,
}

impl CatalogKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Primitive => "primitive",
            Self::Furniture => "furniture",
            Self::Animal => "animal",
            Self::Street => "street",
            Self::Interior => "interior",
            Self::Exterior => "exterior",
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
    /// Furniture that comes with a room. Placing the room places these as
    /// separate scene objects, so each piece can be moved or removed on its
    /// own; the offset is in the room's own frame at yaw 0.
    pub furniture: &'static [RoomFurniture],
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

    /// Local bounding box for the stage clamp, in the entry's own origin
    /// (`BaseCenter` stands on `y = 0`, `Center` is centered on it).
    pub fn footprint(&self) -> crate::stage::Footprint {
        let half = [self.size[0] * 0.5, self.size[1] * 0.5, self.size[2] * 0.5];
        match self.origin {
            CatalogOrigin::BaseCenter => crate::stage::Footprint::standing(half[0], self.size[1], half[2]),
            CatalogOrigin::Center => crate::stage::Footprint::centered(half),
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
    entry("door", CatalogKind::Furniture, [0.9, 2.05, 0.08], &["frame", "panel"], "door slab 0.9×2.05 m: frame around a recessed panel"),
    entry("window", CatalogKind::Furniture, [1.4, 1.4, 0.08], &["frame", "glass"], "window slab 1.4×1.4 m: frame, cross mullion and a glass pane"),
    entry("stairs", CatalogKind::Furniture, [1.0, 1.7, 2.8], &["steps"], "straight flight: 10 steps, rise 0.17 m, going 0.28 m, ascending toward +Z"),
    entry("carpet", CatalogKind::Furniture, [2.0, 0.02, 3.0], &["pile"], "carpet 2×3 m: a flat slab on the floor"),
    entry("picture", CatalogKind::Furniture, [0.6, 0.8, 0.03], &["frame", "canvas"], "wall picture 0.6×0.8 m, canvas on the +Z face"),
    entry("mirror", CatalogKind::Furniture, [0.6, 1.6, 0.03], &["frame", "glass"], "tall wall mirror 0.6×1.6 m, glass on the +Z face"),
    entry("bin", CatalogKind::Furniture, [0.3, 0.4, 0.3], &["body"], "waste bin d=0.3 m, h=0.4 m, hollow"),
    entry("plant", CatalogKind::Furniture, [0.4, 0.7, 0.4], &["pot", "leaves"], "potted plant h=0.7 m"),
    entry("cardboard", CatalogKind::Furniture, [0.4, 0.3, 0.3], &["box", "tape"], "cardboard box with a tape strip (the 1 m primitive keeps the name `box`)"),
    entry("crate", CatalogKind::Furniture, [0.5, 0.4, 0.5], &["body", "slats"], "wooden crate with slats on two opposite sides"),
    entry("barrel", CatalogKind::Furniture, [0.6, 0.9, 0.6], &["body", "hoops"], "barrel d=0.6 m, h=0.9 m with hoop bands"),
    // --- animals: static figures with pose variants where they read in frame ---
    entry("cat", CatalogKind::Animal, [0.18, 0.38, 0.57], &["body", "head", "legs", "tail"], "cat; moves by its transform keys"),
    entry("cat_sitting", CatalogKind::Animal, [0.18, 0.38, 0.57], &["body", "head", "legs", "tail"], "cat, sitting"),
    entry("cat_lying", CatalogKind::Animal, [0.18, 0.24, 0.73], &["body", "head", "legs", "tail"], "cat, lying"),
    entry("dog_big", CatalogKind::Animal, [0.31, 0.76, 1.14], &["body", "head", "legs", "tail"], "big dog (shepherd size)"),
    entry("dog_big_sitting", CatalogKind::Animal, [0.31, 0.76, 1.00], &["body", "head", "legs", "tail"], "big dog, sitting"),
    entry("dog_big_lying", CatalogKind::Animal, [0.31, 0.39, 1.31], &["body", "head", "legs", "tail"], "big dog, lying"),
    entry("dog_small", CatalogKind::Animal, [0.18, 0.42, 0.63], &["body", "head", "legs", "tail"], "small dog (terrier size)"),
    entry("dog_small_sitting", CatalogKind::Animal, [0.18, 0.42, 0.54], &["body", "head", "legs", "tail"], "small dog, sitting"),
    entry("dog_small_lying", CatalogKind::Animal, [0.18, 0.24, 0.70], &["body", "head", "legs", "tail"], "small dog, lying"),
    entry("cow", CatalogKind::Animal, [0.60, 1.75, 2.19], &["body", "head", "legs", "tail"], "cow"),
    entry("cow_lying", CatalogKind::Animal, [0.60, 0.79, 2.66], &["body", "head", "legs", "tail"], "cow, lying"),
    entry("pig", CatalogKind::Animal, [0.43, 0.95, 1.44], &["body", "head", "legs", "tail"], "pig"),
    entry("pig_lying", CatalogKind::Animal, [0.43, 0.48, 1.50], &["body", "head", "legs", "tail"], "pig, lying"),
    entry("sheep", CatalogKind::Animal, [0.37, 0.84, 1.12], &["body", "head", "legs", "tail"], "sheep"),
    entry("sheep_lying", CatalogKind::Animal, [0.37, 0.40, 1.12], &["body", "head", "legs", "tail"], "sheep, lying"),
    entry("goat", CatalogKind::Animal, [0.34, 0.80, 1.04], &["body", "head", "legs", "tail"], "goat (horned)"),
    entry("goat_lying", CatalogKind::Animal, [0.34, 0.40, 1.07], &["body", "head", "legs", "tail"], "goat, lying"),
    entry("chicken", CatalogKind::Animal, [0.21, 0.35, 0.46], &["body", "head", "legs"], "chicken, standing"),
    entry("fox", CatalogKind::Animal, [0.20, 0.44, 0.81], &["body", "head", "legs", "tail"], "fox"),
    entry("fox_lying", CatalogKind::Animal, [0.20, 0.25, 0.94], &["body", "head", "legs", "tail"], "fox, lying"),
    entry("hare", CatalogKind::Animal, [0.15, 0.38, 0.47], &["body", "head", "legs", "tail"], "hare"),
    entry("hare_lying", CatalogKind::Animal, [0.15, 0.26, 0.46], &["body", "head", "legs", "tail"], "hare, crouched"),
    entry("deer", CatalogKind::Animal, [0.41, 1.62, 1.55], &["body", "head", "legs", "tail"], "deer with antlers"),
    entry("deer_lying", CatalogKind::Animal, [0.41, 0.62, 1.57], &["body", "head", "legs", "tail"], "deer, lying"),
    entry("bear", CatalogKind::Animal, [0.66, 0.96, 1.72], &["body", "head", "legs", "tail"], "bear"),
    entry("bear_lying", CatalogKind::Animal, [0.66, 0.50, 1.72], &["body", "head", "legs", "tail"], "bear, lying"),
    entry("mouse", CatalogKind::Animal, [0.04, 0.08, 0.11], &["body", "head", "tail"], "mouse"),
    // --- street props: outdoor items, origin at base center ---
    entry("pole", CatalogKind::Street, [0.15, 5.0, 0.15], &["body"], "utility pole h=5 m"),
    entry("streetlight", CatalogKind::Street, [0.5, 4.5, 1.2], &["pole", "head", "lamp"], "street light h=4.5 m; the arm and lamp reach toward +Z (the lit side)"),
    entry("bench_outdoor", CatalogKind::Street, [1.6, 0.9, 0.6], &["seat", "back", "frame"], "street bench: seat at 0.45 m, backrest up to 0.9 m"),
    entry("car", CatalogKind::Street, [1.8, 1.5, 4.4], &["body", "cabin", "windows", "wheels"], "generic car, length along Z, cabin slightly toward −Z"),
    entry("tree", CatalogKind::Street, [2.0, 3.5, 2.0], &["trunk", "canopy"], "stylized tree h≈3.5 m: trunk plus canopy blobs"),
    entry("tree_pine", CatalogKind::Street, [1.8, 4.2, 1.8], &["trunk", "canopy"], "conifer h≈4.2 m: trunk with stacked cones, narrower than a broadleaf"),
    entry("bush", CatalogKind::Street, [1.2, 0.9, 1.2], &["canopy"], "bush: two canopy blobs, top at 0.9 m"),
    entry("rock", CatalogKind::Street, [1.0, 0.6, 0.8], &["body"], "low-poly rock, angular by design"),

    // --- interiors: a box with cut-away walls and published dimensions ---
    entry_room("room_kitchen", [4.0, 2.7, 5.0], &["north"], &["door_south"], "kitchen 4×5 m, ceiling 2.7 m; open wall is north (camera side)"),
    entry_room("room_bedroom", [4.0, 2.7, 4.0], &["north"], &["door_south"], "bedroom 4×4 m, ceiling 2.7 m"),
    entry_room("room_living", [5.0, 2.7, 6.0], &["north"], &["door_south"], "living room 5×6 m, ceiling 2.7 m"),
    entry_room("room_corridor", [1.6, 2.5, 6.0], &["north"], &["door_south", "door_north"], "corridor 1.6×6 m, ceiling 2.5 m"),
    entry_room("room_toilet", [1.6, 2.5, 2.2], &["north"], &["door_south"], "toilet 1.6×2.2 m, ceiling 2.5 m"),
    entry_room("room_bathroom", [2.4, 2.5, 2.6], &["north"], &["door_south"], "bathroom 2.4×2.6 m, ceiling 2.5 m"),
    entry_room_furnished("room_dining", [4.5, 2.7, 5.5], &["north"], &["door_south"], &[
        furn("table", 0.0, 0.0, 0.0),
        furn_yaw("chair", 0.0, 0.0, 0.85, 180.0),
        furn_yaw("chair", 0.0, 0.0, -0.85, 0.0),
        furn_yaw("chair", 1.1, 0.0, 0.0, 270.0),
        furn_yaw("chair", -1.1, 0.0, 0.0, 90.0),
        furn_yaw("cabinet", -1.7, 0.0, 2.3, 180.0),
        furn_yaw("shelf", 1.9, 0.0, 2.3, 180.0),
        furn_yaw("picture", 0.0, 1.4, 2.55, 180.0),
    ], "dining room 4.5×5.5 m: table with four seats around it, cabinet and shelf along the back wall"),
    entry_room_furnished("room_office", [3.5, 2.7, 4.0], &["north"], &["door_south"], &[
        furn_yaw("table", 0.0, 0.0, -1.3, 90.0),
        furn_yaw("chair", 0.0, 0.0, -0.55, 180.0),
        furn("monitor", 0.0, 0.75, -1.3),
        furn_yaw("shelf", -1.4, 0.0, 0.6, 90.0),
        furn("lamp", 1.3, 0.0, -1.4),
        furn("plant", 1.4, 0.0, 1.5),
        furn("window", 0.0, 0.8, 1.94),
    ], "office 3.5×4 m: desk with a monitor against the back wall, shelf, floor lamp"),
    entry_room_furnished("room_kids", [3.5, 2.7, 4.0], &["north"], &["door_south"], &[
        furn_yaw("bed", -0.8, 0.0, -0.8, 90.0),
        furn_yaw("wardrobe", 1.35, 0.0, -1.5, 180.0),
        furn_yaw("shelf", 1.4, 0.0, 0.4, 90.0),
        furn("cardboard", 0.4, 0.0, 1.2),
        furn("lamp", -1.4, 0.0, 1.5),
        furn("picture", 0.4, 1.4, 1.94),
    ], "kids room 3.5×4 m: bed, wardrobe, shelf and a box on the floor"),
    entry_room_furnished("room_hallway", [2.0, 2.5, 4.0], &["north"], &["door_south", "door_north"], &[
        furn_yaw("cabinet", -0.6, 0.0, -1.2, 90.0),
        furn_yaw("shelf", 0.7, 0.0, 0.6, 270.0),
        furn_yaw("mirror", 0.85, 0.9, -1.2, 270.0),
        furn("plant", -0.6, 0.0, 1.4),
    ], "hallway 2×4 m: cabinet, shelf, wall mirror and a plant"),
    entry_room_furnished("room_garage", [5.0, 2.7, 6.0], &["north"], &["door_south"], &[
        furn("car", 0.0, 0.0, -0.4),
        furn_yaw("shelf", -2.2, 0.0, 2.4, 180.0),
        furn_yaw("crate", 2.0, 0.0, 2.3, 0.0),
        furn("barrel", 2.0, 0.0, 1.4),
        furn("cardboard", -1.6, 0.0, 1.4),
        furn_yaw("door", 1.6, 0.0, 2.9, 180.0),
    ], "garage 5×6 m: a car in the middle, shelf and boxes along the walls"),
    entry_room_furnished("room_stairwell", [3.0, 5.0, 5.0], &["north"], &["door_south"], &[
        furn("stairs", -0.6, 0.0, -1.0),
        furn_yaw("door", 1.0, 0.0, 2.4, 180.0),
        furn_yaw("window", 1.4, 1.2, -2.4, 0.0),
    ], "stairwell 3×5 m, two storeys high: a flight of stairs and a door on the landing"),
    entry_room_furnished("room_basement", [4.0, 2.2, 5.0], &["north"], &["door_south"], &[
        furn_yaw("shelf", -1.7, 0.0, -1.5, 90.0),
        furn("crate", 1.4, 0.0, -1.8),
        furn("barrel", 1.4, 0.0, -0.8),
        furn("cardboard", -1.2, 0.0, 1.6),
        furn("bin", 1.5, 0.0, 1.8),
        furn("lamp", -1.6, 0.0, 2.0),
    ], "basement 4×5 m, ceiling 2.2 m: shelving, crates and a barrel"),
    entry_room_furnished("room_attic", [4.0, 2.2, 5.0], &["north"], &["door_south"], &[
        furn("crate", -1.3, 0.0, -1.4),
        furn("crate", -1.3, 0.4, -1.4),
        furn("barrel", 1.4, 0.0, -1.5),
        furn_yaw("wardrobe", 1.2, 0.0, 0.6, 180.0),
        furn("cardboard", 0.0, 0.0, 1.6),
        furn("bin", -1.5, 0.0, 1.8),
    ], "attic 4×5 m, ceiling 2.2 m: stacked crates, a wardrobe and boxes"),
    // --- outdoor grounds: a ground plate, no walls, no ceiling; the ground
    // top is at y=0 and props/cameras are placed on it (the slab hangs below).
    entry("street_yard", CatalogKind::Exterior, [20.0, 0.24, 20.0], &["ground"], "open yard 20×20 m; ground top at y=0 — props and cameras are placed on it"),
    entry("street_park", CatalogKind::Exterior, [30.0, 0.24, 30.0], &["ground", "path"], "park 30×30 m with a walking path along Z; ground top at y=0 — props and cameras are placed on it"),
    entry("street_crossing", CatalogKind::Exterior, [24.0, 0.24, 24.0], &["ground", "road", "marking"], "crossing 24×24 m: two 6 m roads with dashed center lines; ground top at y=0 — props and cameras are placed on it"),
];

const fn entry(
    name: &'static str,
    kind: CatalogKind,
    size: [f32; 3],
    materials: &'static [&'static str],
    note: &'static str,
) -> CatalogEntry {
    CatalogEntry { name, kind, size, origin: CatalogOrigin::BaseCenter, materials, open_sides: &[], openings: &[], furniture: &[], note }
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
        furniture: &[],
        note,
    }
}

/// One piece of a room's furniture set: a catalog prop plus where it stands in
/// the room's own frame (x right, z toward the far wall at yaw 0), its turn and
/// its height above the floor (a monitor sits on a table, not in it).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoomFurniture {
    pub prop: &'static str,
    pub offset: [f32; 3],
    /// Turn around the vertical axis, in degrees (the wire speaks degrees).
    pub yaw_deg: f32,
}

const fn furn(prop: &'static str, x: f32, y: f32, z: f32) -> RoomFurniture {
    RoomFurniture { prop, offset: [x, y, z], yaw_deg: 0.0 }
}

const fn furn_yaw(prop: &'static str, x: f32, y: f32, z: f32, yaw_deg: f32) -> RoomFurniture {
    RoomFurniture { prop, offset: [x, y, z], yaw_deg }
}

/// A room with its furniture set.
const fn entry_room_furnished(
    name: &'static str,
    size: [f32; 3],
    open_sides: &'static [&'static str],
    openings: &'static [&'static str],
    furniture: &'static [RoomFurniture],
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
        furniture,
        note,
    }
}

impl CatalogEntry {
    /// Place a furnished room: one key per piece of its furniture set, derived
    /// from the shell's key. Offsets are in the room's own frame, so they
    /// rotate, scale and translate with it, and each piece stays an ordinary
    /// object the director can move or delete on its own. Shared by the CLI,
    /// the MCP session and the GUI so all three place a room identically.
    pub fn furniture_keys(&self, shell: &crate::project::ObjectKey) -> Vec<(&'static str, crate::project::ObjectKey)> {
        if self.furniture.is_empty() {
            return Vec::new();
        }
        let rot = glam::Quat::from_rotation_y(shell.yaw);
        self.furniture
            .iter()
            .map(|piece| {
                let offset = glam::Vec3::from_array(piece.offset) * shell.scale;
                let position = shell.position + rot * offset;
                (
                    piece.prop,
                    crate::project::ObjectKey {
                        frame: shell.frame,
                        position,
                        yaw: shell.yaw + piece.yaw_deg.to_radians(),
                        pitch: 0.0,
                        roll: 0.0,
                        scale: shell.scale,
                    },
                )
            })
            .collect()
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
    ("child", 1.15),
    ("teen", 1.55),
    ("elder", 1.66),
    ("wolf", 0.57),
    ("horse", 1.62),
    ("bird", 0.48),
    ("cat", 0.38),
    ("dog_big", 0.76),
    ("dog_small", 0.42),
    ("cow", 1.75),
    ("pig", 0.95),
    ("sheep", 0.84),
    ("goat", 0.80),
    ("chicken", 0.35),
    ("fox", 0.44),
    ("hare", 0.38),
    ("deer", 1.62),
    ("bear", 0.96),
    ("mouse", 0.08),
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

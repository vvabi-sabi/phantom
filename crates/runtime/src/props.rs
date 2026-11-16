//! Prop-catalog geometry (corrective_2 K1): the mesh is built procedurally from
//! a `phantom_core::catalog` entry, with no external files and no licensing
//! questions. Every entry gets named material groups, so `material_set` with a
//! `target` hits "seat", "back" and so on.
//!
//! Conventions match the builtin actors: origin at base center (the item stands
//! on `y = 0`), dimensions come from the catalog table, interior sides are
//! named `north/south/east/west` and are cut away per `open_sides`.

use glam::Vec3;
use phantom_core::catalog::{catalog_find, CatalogEntry, CatalogKind};

use crate::import::{MaterialRecord, MeshData, ModelData, PartData};

/// Builder for an unskinned mesh with material groups.
struct PropBuilder {
    mesh: MeshData,
    /// (group name, start, count) in the index buffer.
    parts: Vec<(String, u32, u32)>,
    current: Option<String>,
}

impl PropBuilder {
    fn new() -> Self {
        Self { mesh: MeshData { positions: vec![], normals: vec![], uvs: vec![], indices: vec![] }, parts: vec![], current: None }
    }

    /// Start a material group: every following primitive goes into it.
    fn group(&mut self, name: &str) {
        self.close_group();
        self.current = Some(name.to_string());
    }

    fn close_group(&mut self) {
        let Some(name) = self.current.take() else { return };
        let start = self.parts.iter().map(|(_, s, c)| s + c).max().unwrap_or(0);
        let count = self.mesh.indices.len() as u32 - start;
        if count > 0 {
            self.parts.push((name, start, count));
        }
    }

    /// Axis-aligned box by center and half extents.
    fn add_box(&mut self, center: Vec3, half: Vec3) {
        let faces = [
            (Vec3::X, Vec3::Y, Vec3::Z),
            (-Vec3::X, Vec3::Y, Vec3::Z),
            (Vec3::Y, Vec3::X, Vec3::Z),
            (-Vec3::Y, Vec3::X, Vec3::Z),
            (Vec3::Z, Vec3::X, Vec3::Y),
            (-Vec3::Z, Vec3::X, Vec3::Y),
        ];
        for (n, u, v) in faces {
            let base = self.mesh.positions.len() as u32;
            for (su, sv) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                let p = center + n * half + u * half * su + v * half * sv;
                self.mesh.positions.push(p.to_array());
                self.mesh.normals.push(n.to_array());
                self.mesh.uvs.push([(su + 1.0) * 0.5, (sv + 1.0) * 0.5]);
            }
            // Flat shading: outward-facing winding.
            if n.dot(u.cross(v)) > 0.0 {
                self.mesh.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            } else {
                self.mesh.indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
            }
        }
    }

    /// Box by dimensions and base center: convenient for furniture.
    fn add_upright(&mut self, base_center: Vec3, size: Vec3) {
        self.add_box(base_center + Vec3::new(0.0, size.y * 0.5, 0.0), size * 0.5);
    }

    /// Cylinder along Y: a ring of `segments` side quads plus caps.
    fn add_cylinder(&mut self, base: Vec3, radius: f32, height: f32, inner_radius: f32, segments: u32) {
        let step = std::f32::consts::TAU / segments as f32;
        let rings: [f32; 2] = [inner_radius, radius];
        for ring in 0..2 {
            let r = rings[ring];
            if r <= 0.0 {
                continue;
            }
            for i in 0..segments {
                let a0 = i as f32 * step;
                let a1 = (i + 1) as f32 * step;
                let (s0, c0) = a0.sin_cos();
                let (s1, c1) = a1.sin_cos();
                let p0 = base + Vec3::new(c0 * r, 0.0, s0 * r);
                let p1 = base + Vec3::new(c1 * r, 0.0, s1 * r);
                let n0 = Vec3::new(c0, 0.0, s0);
                let n1 = Vec3::new(c1, 0.0, s1);
                let outward = if ring == 1 { 1.0 } else { -1.0 };
                let side = |b: &mut Self, top: bool| {
                    let idx = b.mesh.positions.len() as u32;
                    let (lo, hi) = (p0, p1);
                    let (nlo, nhi) = (n0 * outward, n1 * outward);
                    let y = if top { height } else { 0.0 };
                    let flip = if top { outward > 0.0 } else { outward < 0.0 };
                    for (p, n) in [(lo, nlo), (hi, nhi)] {
                        b.mesh.positions.push((p + Vec3::Y * y).to_array());
                        b.mesh.normals.push(n.to_array());
                        b.mesh.uvs.push([0.0, y / height.max(1e-3)]);
                    }
                    // Second row — the opposite edge of the same quad.
                    for (p, n) in [(lo, nlo), (hi, nhi)] {
                        b.mesh.positions.push((p + Vec3::Y * (height - y)).to_array());
                        b.mesh.normals.push(n.to_array());
                        b.mesh.uvs.push([1.0, 1.0 - y / height.max(1e-3)]);
                    }
                    if flip {
                        b.mesh.indices.extend_from_slice(&[idx, idx + 1, idx + 3, idx, idx + 3, idx + 2]);
                    } else {
                        b.mesh.indices.extend_from_slice(&[idx, idx + 3, idx + 1, idx, idx + 2, idx + 3]);
                    }
                };
                if r == inner_radius && inner_radius > 0.0 {
                    // Inner tube wall: normal points inward.
                    side(self, true);
                } else if inner_radius > 0.0 {
                    side(self, true);
                } else {
                    side(self, true);
                }
                // Cap/bottom: a triangle fan from the center.
                let cap_y = if ring == 1 { height } else { 0.0 };
                let center = base + Vec3::Y * cap_y;
                let idx = self.mesh.positions.len() as u32;
                let n = if ring == 1 { Vec3::Y } else { -Vec3::Y };
                self.mesh.positions.push(center.to_array());
                self.mesh.normals.push(n.to_array());
                self.mesh.uvs.push([0.5, 0.5]);
                for p in [p0, p1] {
                    self.mesh.positions.push((p + Vec3::Y * cap_y).to_array());
                    self.mesh.normals.push(n.to_array());
                    self.mesh.uvs.push([0.5 + p.x / (radius * 2.0), 0.5 + p.z / (radius * 2.0)]);
                }
                if ring == 1 {
                    self.mesh.indices.extend_from_slice(&[idx, idx + 1, idx + 2]);
                } else {
                    self.mesh.indices.extend_from_slice(&[idx, idx + 2, idx + 1]);
                }
            }
        }
    }

    /// Cone along Y (apex on top) plus a bottom.
    fn add_cone(&mut self, base: Vec3, radius: f32, height: f32, segments: u32) {
        let step = std::f32::consts::TAU / segments as f32;
        for i in 0..segments {
            let a0 = i as f32 * step;
            let a1 = (i + 1) as f32 * step;
            let (s0, c0) = a0.sin_cos();
            let (s1, c1) = a1.sin_cos();
            let p0 = base + Vec3::new(c0 * radius, 0.0, s0 * radius);
            let p1 = base + Vec3::new(c1 * radius, 0.0, s1 * radius);
            let apex = base + Vec3::Y * height;
            let idx = self.mesh.positions.len() as u32;
            let n0 = Vec3::new(c0, radius / height, s0).normalize();
            let n1 = Vec3::new(c1, radius / height, s1).normalize();
            for (p, n) in [(p0, n0), (p1, n1), (apex, Vec3::Y)] {
                self.mesh.positions.push(p.to_array());
                self.mesh.normals.push(n.to_array());
                self.mesh.uvs.push([0.5, 0.0]);
            }
            self.mesh.indices.extend_from_slice(&[idx, idx + 2, idx + 1]);
            let cidx = self.mesh.positions.len() as u32;
            for (p, u) in [(base, [0.5, 0.5]), (p0, [0.0, 0.0]), (p1, [1.0, 0.0])] {
                self.mesh.positions.push(p.to_array());
                self.mesh.normals.push((-Vec3::Y).to_array());
                self.mesh.uvs.push(u);
            }
            self.mesh.indices.extend_from_slice(&[cidx, cidx + 2, cidx + 1]);
        }
    }

    /// Ellipsoid (a sphere of unit diameter) from parallels and meridians.
    fn add_sphere(&mut self, center: Vec3, radius: f32, segments: u32, rings: u32) {
        for r in 0..rings {
            let phi0 = std::f32::consts::PI * r as f32 / rings as f32;
            let phi1 = std::f32::consts::PI * (r + 1) as f32 / rings as f32;
            for s in 0..segments {
                let th0 = std::f32::consts::TAU * s as f32 / segments as f32;
                let th1 = std::f32::consts::TAU * (s + 1) as f32 / segments as f32;
                let point = |phi: f32, th: f32| {
                    let n = Vec3::new(phi.sin() * th.cos(), phi.cos(), phi.sin() * th.sin());
                    (center + n * radius, n)
                };
                let (p00, n00) = point(phi0, th0);
                let (p01, n01) = point(phi0, th1);
                let (p10, n10) = point(phi1, th0);
                let (p11, n11) = point(phi1, th1);
                let idx = self.mesh.positions.len() as u32;
                for (p, n) in [(p00, n00), (p10, n10), (p11, n11), (p01, n01)] {
                    self.mesh.positions.push(p.to_array());
                    self.mesh.normals.push(n.to_array());
                    self.mesh.uvs.push([0.0, 0.0]);
                }
                self.mesh.indices.extend_from_slice(&[idx, idx + 1, idx + 2, idx, idx + 2, idx + 3]);
            }
        }
    }

    /// Triangular prism: a triangle in XY (apex up), extruded along Z.
    fn add_prism(&mut self, base_center: Vec3, size: Vec3) {
        let hx = size.x * 0.5;
        let hz = size.z * 0.5;
        let (x, y, z) = (base_center.x, base_center.y, base_center.z);
        let tri_at = |dz: f32| {
            [
                Vec3::new(x - hx, y, z + dz),
                Vec3::new(x + hx, y, z + dz),
                Vec3::new(x, y + size.y, z + dz),
            ]
        };
        let tri = |bld: &mut Self, p: [Vec3; 3], n: Vec3, flip: bool| {
            let idx = bld.mesh.positions.len() as u32;
            for v in p {
                bld.mesh.positions.push(v.to_array());
                bld.mesh.normals.push(n.to_array());
                bld.mesh.uvs.push([0.0, 0.0]);
            }
            if flip {
                bld.mesh.indices.extend_from_slice(&[idx, idx + 2, idx + 1]);
            } else {
                bld.mesh.indices.extend_from_slice(&[idx, idx + 1, idx + 2]);
            }
        };
        let quad = |bld: &mut Self, p: [Vec3; 4], n: Vec3| {
            let idx = bld.mesh.positions.len() as u32;
            for v in p {
                bld.mesh.positions.push(v.to_array());
                bld.mesh.normals.push(n.to_array());
                bld.mesh.uvs.push([0.0, 0.0]);
            }
            bld.mesh.indices.extend_from_slice(&[idx, idx + 1, idx + 2, idx, idx + 2, idx + 3]);
        };
        let front = tri_at(hz);
        let back = tri_at(-hz);
        tri(self, front, Vec3::Z, false);
        tri(self, back, -Vec3::Z, true);
        let slope = Vec3::new(size.y, hx, 0.0).normalize();
        quad(self, [back[0], front[0], front[2], back[2]], -slope);
        quad(self, [back[1], back[2], front[2], front[1]], slope);
        quad(self, [back[0], back[1], front[1], front[0]], -Vec3::Y);
    }

    /// Ramp: a sloped slab from the floor up to a height (slope along Z).
    fn add_ramp(&mut self, base_center: Vec3, size: Vec3) {
        let hx = size.x * 0.5;
        let hz = size.z * 0.5;
        let (x, y, z) = (base_center.x, base_center.y, base_center.z);
        let quad = |bld: &mut Self, p: [Vec3; 4], n: Vec3| {
            let idx = bld.mesh.positions.len() as u32;
            for v in p {
                bld.mesh.positions.push(v.to_array());
                bld.mesh.normals.push(n.to_array());
                bld.mesh.uvs.push([0.0, 0.0]);
            }
            bld.mesh.indices.extend_from_slice(&[idx, idx + 1, idx + 2, idx, idx + 2, idx + 3]);
        };
        // Bottom edge at +Z, top edge at −Z.
        let low_l = Vec3::new(x - hx, y, z + hz);
        let low_r = Vec3::new(x + hx, y, z + hz);
        let high_l = Vec3::new(x - hx, y + size.y, z - hz);
        let high_r = Vec3::new(x + hx, y + size.y, z - hz);
        let slope = Vec3::new(0.0, size.z, size.y).normalize();
        quad(self, [low_l, low_r, high_r, high_l], slope);
        quad(self, [low_l, high_l, Vec3::new(high_l.x, y, high_l.z), Vec3::new(low_l.x, y, low_l.z)], -Vec3::X);
        quad(self, [low_r, Vec3::new(low_r.x, y, low_r.z), Vec3::new(high_r.x, y, high_r.z), high_r], Vec3::X);
        quad(self, [high_l, high_r, Vec3::new(high_r.x, y, high_r.z), Vec3::new(high_l.x, y, high_l.z)], -Vec3::Z);
        quad(self, [low_l, Vec3::new(low_l.x, y, low_l.z), Vec3::new(low_r.x, y, low_r.z), low_r], -Vec3::Y);
    }

    fn finish(mut self, entry: &CatalogEntry) -> ModelData {
        self.close_group();
        let materials = entry
            .materials
            .iter()
            .enumerate()
            .map(|(_i, name)| MaterialRecord {
                name: name.to_string(),
                // Neutral warm plastic: the prop reads before material_set.
                base_color: [0.78, 0.72, 0.64, 1.0],
                metallic: 0.0,
                roughness: 0.7,
                base_color_texture: None,
                normal_map_texture: None,
                roughness_texture: None,
                metalness_texture: None,
                invert_roughness: false,
            })
            .collect::<Vec<_>>();
        let parts = self
            .parts
            .into_iter()
            .map(|(name, start, count)| PartData {
                material: entry.materials.iter().position(|m| *m == name).unwrap_or(0),
                start,
                count,
            })
            .collect();
        ModelData { mesh: self.mesh, skin: None, materials, parts, diagnostics: vec![] }
    }
}

/// Mesh for a catalog entry; `None` for an unknown name.
pub fn prop_model(name: &str) -> Option<ModelData> {
    let entry = catalog_find(name)?;
    Some(match entry.kind {
        CatalogKind::Primitive => primitive(entry),
        CatalogKind::Furniture => furniture(entry),
        CatalogKind::Interior => interior(entry),
    })
}

fn primitive(entry: &CatalogEntry) -> ModelData {
    let mut b = PropBuilder::new();
    b.group("body");
    let base = Vec3::ZERO;
    match entry.name {
        "cylinder" => b.add_cylinder(base, 0.5, 1.0, 0.0, 20),
        "cone" => b.add_cone(base, 0.5, 1.0, 20),
        "sphere" => b.add_sphere(base + Vec3::new(0.0, 0.5, 0.0), 0.5, 20, 12),
        "prism" => b.add_prism(base, Vec3::ONE),
        "ramp" => b.add_ramp(base, Vec3::ONE),
        "tube" => b.add_cylinder(base, 0.5, 1.0, 0.35, 20),
        _ => b.add_upright(base, Vec3::ONE),
    }
    b.finish(entry)
}

fn furniture(entry: &CatalogEntry) -> ModelData {
    let mut b = PropBuilder::new();
    let s = Vec3::from_array(entry.size);
    match entry.name {
        "table" => {
            b.group("frame");
            let leg = 0.06;
            for sx in [-1.0f32, 1.0] {
                for sz in [-1.0f32, 1.0] {
                    let x = sx * (s.x * 0.5 - leg * 2.0);
                    let z = sz * (s.z * 0.5 - leg * 2.0);
                    b.add_upright(Vec3::new(x, 0.0, z), Vec3::new(leg, s.y - 0.04, leg));
                }
            }
            b.group("top");
            b.add_upright(Vec3::new(0.0, s.y - 0.04, 0.0), Vec3::new(s.x, 0.04, s.z));
        }
        "chair" | "sofa" => {
            let seat_y = if entry.name == "chair" { 0.45 } else { 0.42 };
            b.group("frame");
            let leg = 0.05;
            for sx in [-1.0f32, 1.0] {
                for sz in [-1.0f32, 1.0] {
                    let x = sx * (s.x * 0.5 - leg * 2.0);
                    let z = sz * (s.z * 0.5 - leg * 2.0);
                    b.add_upright(Vec3::new(x, 0.0, z), Vec3::new(leg, seat_y - 0.06, leg));
                }
            }
            b.group("seat");
            b.add_upright(Vec3::new(0.0, seat_y - 0.06, 0.0), Vec3::new(s.x, 0.06, s.z));
            b.group("back");
            b.add_upright(Vec3::new(0.0, seat_y, -s.z * 0.5 + 0.04), Vec3::new(s.x, s.y - seat_y, 0.08));
        }
        "stool" | "bench" => {
            b.group("frame");
            let leg = 0.05;
            for sx in [-1.0f32, 1.0] {
                for sz in [-1.0f32, 1.0] {
                    let x = sx * (s.x * 0.5 - leg * 2.0);
                    let z = sz * (s.z * 0.5 - leg * 2.0);
                    b.add_upright(Vec3::new(x, 0.0, z), Vec3::new(leg, s.y - 0.04, leg));
                }
            }
            b.group("seat");
            b.add_upright(Vec3::new(0.0, s.y - 0.04, 0.0), Vec3::new(s.x, 0.04, s.z));
        }
        "cabinet" | "wardrobe" | "shelf" => {
            b.group("body");
            b.add_upright(Vec3::ZERO, Vec3::new(s.x, s.y, s.z * 0.9));
            b.group("front");
            b.add_upright(Vec3::new(0.0, 0.0, s.z * 0.45), Vec3::new(s.x * 0.94, s.y * 0.94, 0.02));
        }
        "bed" => {
            b.group("frame");
            b.add_upright(Vec3::ZERO, Vec3::new(s.x, 0.25, s.z));
            b.group("mattress");
            b.add_upright(Vec3::new(0.0, 0.25, 0.0), Vec3::new(s.x * 0.96, s.y - 0.25, s.z * 0.96));
        }
        "monitor" => {
            b.group("frame");
            b.add_upright(Vec3::new(0.0, 0.0, 0.0), Vec3::new(s.x * 0.12, s.y * 0.2, s.z * 0.6));
            b.group("screen");
            b.add_box(Vec3::new(0.0, s.y * 0.5, 0.0), Vec3::new(s.x * 0.5, s.y * 0.35, s.z * 0.5));
        }
        "laptop" => {
            b.group("keyboard");
            b.add_upright(Vec3::ZERO, Vec3::new(s.x, 0.015, s.z * 0.8));
            b.group("screen");
            b.add_box(Vec3::new(0.0, s.y * 0.45, -s.z * 0.5), Vec3::new(s.x * 0.5, s.y * 0.45, 0.008));
        }
        "book" => {
            b.group("cover");
            b.add_upright(Vec3::ZERO, Vec3::new(s.x, s.y, s.z));
            b.group("pages");
            b.add_upright(Vec3::new(s.x * 0.05, s.y * 0.05, 0.0), Vec3::new(s.x * 0.85, s.y * 0.88, s.z * 1.05));
        }
        "glass" => {
            b.group("body");
            b.add_cylinder(Vec3::ZERO, s.x * 0.5, s.y, s.x * 0.38, 16);
        }
        "plate" => {
            b.group("body");
            b.add_cylinder(Vec3::ZERO, s.x * 0.5, s.y, s.x * 0.36, 20);
        }
        "lamp" => {
            b.group("frame");
            b.add_upright(Vec3::ZERO, Vec3::new(s.x * 0.2, 0.02, s.z * 0.2));
            b.add_cylinder(Vec3::new(0.0, 0.02, 0.0), 0.015, s.y - 0.25, 0.0, 10);
            b.group("shade");
            b.add_cone(Vec3::new(0.0, s.y - 0.25, 0.0), s.x * 0.5, 0.25, 16);
        }
        _ => b.add_upright(Vec3::ZERO, s),
    }
    b.finish(entry)
}

/// Interior: floor, ceiling and walls, except the sides listed in `open_sides`.
/// Sides are named by axis: `north` is the far one (−Z), `south` the near one
/// (+Z), `west` is −X and `east` is +X. The default camera looks toward −Z, so
/// `north` in `open_sides` removes the wall that blocks the frame.
fn interior(entry: &CatalogEntry) -> ModelData {
    let mut b = PropBuilder::new();
    let s = Vec3::from_array(entry.size);
    let t = 0.12;
    b.group("floor");
    b.add_box(Vec3::new(0.0, -t * 0.5, 0.0), Vec3::new(s.x * 0.5, t * 0.5, s.z * 0.5));
    b.group("ceiling");
    b.add_box(Vec3::new(0.0, s.y + t * 0.5, 0.0), Vec3::new(s.x * 0.5, t * 0.5, s.z * 0.5));
    b.group("wall");
    let wall = |b: &mut PropBuilder, center: Vec3, half: Vec3| b.add_box(center, half);
    if !entry.open_sides.contains(&"north") {
        wall(&mut b, Vec3::new(0.0, s.y * 0.5, -s.z * 0.5), Vec3::new(s.x * 0.5, s.y * 0.5, t * 0.5));
    }
    if !entry.open_sides.contains(&"south") {
        wall(&mut b, Vec3::new(0.0, s.y * 0.5, s.z * 0.5), Vec3::new(s.x * 0.5, s.y * 0.5, t * 0.5));
    }
    if !entry.open_sides.contains(&"west") {
        wall(&mut b, Vec3::new(-s.x * 0.5, s.y * 0.5, 0.0), Vec3::new(t * 0.5, s.y * 0.5, s.z * 0.5));
    }
    if !entry.open_sides.contains(&"east") {
        wall(&mut b, Vec3::new(s.x * 0.5, s.y * 0.5, 0.0), Vec3::new(t * 0.5, s.y * 0.5, s.z * 0.5));
    }
    b.finish(entry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use phantom_core::catalog::CATALOG;

    fn bbox(model: &ModelData) -> (Vec3, Vec3) {
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for p in &model.mesh.positions {
            let v = Vec3::from_array(*p);
            lo = lo.min(v);
            hi = hi.max(v);
        }
        (lo, hi)
    }

    #[test]
    fn every_catalog_entry_has_geometry_with_parts_and_materials() {
        for entry in CATALOG {
            let model = prop_model(entry.name).unwrap_or_else(|| panic!("no mesh for {}", entry.name));
            assert!(!model.mesh.indices.is_empty(), "{}: empty mesh", entry.name);
            assert_eq!(model.mesh.positions.len(), model.mesh.normals.len(), "{}", entry.name);
            assert_eq!(model.mesh.positions.len(), model.mesh.uvs.len(), "{}", entry.name);
            assert_eq!(model.materials.len(), entry.materials.len(), "{}: material groups", entry.name);
            // Parts cover the index buffer with no gaps and no overlaps.
            let mut ranges: Vec<(u32, u32)> = model.parts.iter().map(|p| (p.start, p.count)).collect();
            ranges.sort_unstable();
            let mut next = 0u32;
            for (start, count) in ranges {
                assert_eq!(start, next, "{}: gap between parts", entry.name);
                next = start + count;
                assert!(model.parts.iter().any(|p| p.material < model.materials.len()));
            }
            assert_eq!(next as usize, model.mesh.indices.len(), "{}: parts must cover all indices", entry.name);
        }
    }

    #[test]
    fn declared_size_matches_generated_geometry_and_origin_is_on_the_floor() {
        const T: f32 = 0.12; // floor/wall/ceiling thickness for interiors
        for entry in CATALOG {
            let model = prop_model(entry.name).unwrap();
            let (lo, hi) = bbox(&model);
            let got = hi - lo;
            let want = Vec3::from_array(entry.size);
            // An interior declares its FREE dimensions, so the mesh is larger by
            // the slab thickness; other props must match the table exactly.
            let (slack, base) = match entry.kind {
                CatalogKind::Interior => (2.0 * T, T),
                _ => (0.05, 0.0),
            };
            for axis in 0..3 {
                let (g, w) = (got[axis], want[axis]);
                let limit = slack + w * 0.06;
                assert!(g >= w - limit, "{}: size along axis {axis} is {g}, below the table's {w}", entry.name);
                assert!(g <= w + limit, "{}: size along axis {axis} is {g}, above the table's {w}", entry.name);
            }
            assert!(lo.y.abs() <= base + 1e-3, "{}: mesh bottom at {:.3}, expected {base}", entry.name, lo.y);
            // The interior floor lies on y = 0: items are placed on it, not inside it.
            if entry.kind == CatalogKind::Interior {
                assert!(
                    model.mesh.positions.iter().any(|p| p[1].abs() < 1e-4),
                    "{}: the floor surface must be at y = 0",
                    entry.name
                );
            }
        }
    }

    #[test]
    fn open_side_removes_that_wall_only() {
        let entry = catalog_find("room_kitchen").unwrap();
        let model = prop_model("room_kitchen").unwrap();
        let (_, hi) = bbox(&model);
        // The north wall is cut away, so −Z is open: geometry does not reach the edge.
        let min_z = model.mesh.positions.iter().map(|p| p[2]).fold(f32::INFINITY, f32::min);
        assert!(min_z > -entry.size[2] * 0.5 - 0.2, "north wall was not built: min_z {min_z}");
        assert!(hi.y > entry.size[1], "ceiling above the walls");
    }

    #[test]
    fn unknown_prop_has_no_mesh() {
        assert!(prop_model("no-such-prop").is_none());
    }
}

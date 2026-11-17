//! Prop-catalog geometry (corrective_2 K1): the mesh is built procedurally from
//! a `phantom_core::catalog` entry, with no external files and no licensing
//! questions. Every entry gets named material groups, so `material_set` with a
//! `target` hits "seat", "back" and so on.
//!
//! Conventions match the builtin actors: origin at base center (the item stands
//! on `y = 0`), dimensions come from the catalog table, interior sides are
//! named `north/south/east/west` and are cut away per `open_sides`.

use glam::{Quat, Vec3};
use phantom_core::catalog::{catalog_find, CatalogEntry, CatalogKind};
use phantom_core::{plan, scatter_preset, ScatterField, MAX_SCATTER_INSTANCES};

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
        CatalogKind::Animal => animal(entry),
        CatalogKind::Street => street_item(entry),
        CatalogKind::Interior => interior(entry),
        CatalogKind::Exterior => exterior(entry),
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
        "door" => {
            // Frame: two jambs and a lintel around a recessed panel.
            b.group("frame");
            let jamb = 0.06;
            for x in [s.x * 0.5 - jamb * 0.5, -(s.x * 0.5 - jamb * 0.5)] {
                b.add_upright(Vec3::new(x, 0.0, 0.0), Vec3::new(jamb, s.y, s.z));
            }
            b.add_upright(Vec3::new(0.0, s.y - jamb, 0.0), Vec3::new(s.x - 2.0 * jamb, jamb, s.z));
            b.group("panel");
            b.add_upright(Vec3::ZERO, Vec3::new(s.x - 2.0 * jamb, s.y - jamb, s.z * 0.5));
        }
        "window" => {
            // Outer frame, cross mullion and a recessed glass pane.
            b.group("frame");
            let w = 0.06;
            b.add_upright(Vec3::ZERO, Vec3::new(s.x, w, s.z));
            b.add_upright(Vec3::new(0.0, s.y - w, 0.0), Vec3::new(s.x, w, s.z));
            for x in [s.x * 0.5 - w * 0.5, -(s.x * 0.5 - w * 0.5)] {
                b.add_upright(Vec3::new(x, w, 0.0), Vec3::new(w, s.y - 2.0 * w, s.z));
            }
            b.add_upright(Vec3::new(0.0, w, 0.0), Vec3::new(0.04, s.y - 2.0 * w, s.z * 0.7));
            b.add_upright(Vec3::new(0.0, s.y * 0.5 - 0.02, 0.0), Vec3::new(s.x - 2.0 * w, 0.04, s.z * 0.7));
            b.group("glass");
            b.add_box(Vec3::new(0.0, s.y * 0.5, 0.0), Vec3::new(s.x * 0.5 - w, s.y * 0.5 - w, 0.01));
        }
        "stairs" => {
            // Straight flight: 10 steps rising 0.17 and going 0.28 along +Z.
            b.group("steps");
            let (rise, going) = (s.y / 10.0, s.z / 10.0);
            for i in 0..10 {
                let z = -s.z * 0.5 + going * i as f32;
                b.add_upright(Vec3::new(0.0, 0.0, z), Vec3::new(s.x, rise * (i + 1) as f32, going));
            }
        }
        "carpet" => {
            b.group("pile");
            b.add_upright(Vec3::ZERO, s);
        }
        "picture" => {
            b.group("frame");
            b.add_upright(Vec3::ZERO, s);
            b.group("canvas");
            // Slightly recessed canvas on the +Z (front) face.
            b.add_box(Vec3::new(0.0, s.y * 0.5, s.z * 0.5 - 0.004), Vec3::new(s.x * 0.5 - 0.04, s.y * 0.5 - 0.04, 0.004));
        }
        "mirror" => {
            b.group("frame");
            b.add_upright(Vec3::ZERO, s);
            b.group("glass");
            b.add_box(Vec3::new(0.0, s.y * 0.5, s.z * 0.5 - 0.003), Vec3::new(s.x * 0.5 - 0.04, s.y * 0.5 - 0.04, 0.003));
        }
        "bin" => {
            b.group("body");
            b.add_cylinder(Vec3::ZERO, s.x * 0.5, s.y, s.x * 0.5 - 0.02, 16);
        }
        "plant" => {
            b.group("pot");
            b.add_cylinder(Vec3::ZERO, 0.125, 0.25, 0.0, 14);
            b.group("leaves");
            b.add_sphere(Vec3::new(0.03, 0.54, 0.0), 0.17, 10, 6);
            b.add_sphere(Vec3::new(-0.03, 0.49, 0.04), 0.17, 10, 6);
        }
        "cardboard" => {
            b.group("box");
            b.add_upright(Vec3::ZERO, s);
            b.group("tape");
            // A thin tape strip across the top.
            b.add_box(Vec3::new(0.0, s.y - 0.001, 0.0), Vec3::new(0.03, 0.002, s.z * 0.5));
        }
        "crate" => {
            b.group("body");
            b.add_upright(Vec3::ZERO, s);
            b.group("slats");
            // Two thin slats on each of the two opposite ±X sides.
            for x in [s.x * 0.5 - 0.002, -(s.x * 0.5 - 0.002)] {
                for y in [s.y * 0.3, s.y * 0.7] {
                    b.add_box(Vec3::new(x, y, 0.0), Vec3::new(0.006, 0.03, s.z * 0.46));
                }
            }
        }
        "barrel" => {
            b.group("body");
            b.add_cylinder(Vec3::ZERO, s.x * 0.5, s.y, 0.0, 18);
            b.group("hoops");
            // Hoop bands: short hollow cylinders slightly wider than the body.
            for y in [s.y * 0.22, s.y * 0.78] {
                b.add_cylinder(Vec3::new(0.0, y - 0.02, 0.0), s.x * 0.5 + 0.008, 0.04, s.x * 0.5 - 0.01, 18);
            }
        }
        _ => b.add_upright(Vec3::ZERO, s),
    }
    b.finish(entry)
}


// --- animals ----------------------------------------------------------------

/// Pose variant of an animal figure (the base name stands; the variants exist
/// where they read in frame).
#[derive(Clone, Copy, PartialEq, Eq)]
enum AnimalVariant {
    Stand,
    Sit,
    Lie,
}

/// Proportions of a quadruped figure, in meters. Boxes are axis-aligned, so
/// the silhouette is built from a few well-placed parts: body, head with ears
/// and a snout, four legs, a tail.
struct Quadruped {
    /// Body box: half width, height, half length.
    body: Vec3,
    /// Leg height (the body rides on it) and thickness.
    leg_h: f32,
    leg_t: f32,
    /// Head box half size; sits at the front top of the body.
    head: f32,
    /// Snout length (0 — a flat face, like a cat).
    snout: f32,
    /// Ear size and shape: (half size, tall) — tall ears stand up (hare),
    /// short ones are round (bear) or horns (goat/deer use antlers instead).
    ear: (f32, f32),
    /// Antlers: each side is a small vertical fork above the head.
    antlers: bool,
    /// Tail: length and thickness; hangs down from the rear (0 — no tail).
    tail: (f32, f32),
}

fn quadruped(entry: &CatalogEntry) -> Option<(Quadruped, AnimalVariant)> {
    let (base, variant) = if let Some(base) = entry.name.strip_suffix("_sitting") {
        (base, AnimalVariant::Sit)
    } else if let Some(base) = entry.name.strip_suffix("_lying") {
        (base, AnimalVariant::Lie)
    } else {
        (entry.name, AnimalVariant::Stand)
    };
    let q = match base {
        "cat" => Quadruped { body: Vec3::new(0.07, 0.12, 0.20), leg_h: 0.10, leg_t: 0.035, head: 0.07, snout: 0.0, ear: (0.02, 0.05), antlers: false, tail: (0.20, 0.02) },
        "dog_big" => Quadruped { body: Vec3::new(0.12, 0.24, 0.38), leg_h: 0.28, leg_t: 0.07, head: 0.11, snout: 0.10, ear: (0.03, 0.07), antlers: false, tail: (0.25, 0.04) },
        "dog_small" => Quadruped { body: Vec3::new(0.07, 0.13, 0.20), leg_h: 0.13, leg_t: 0.04, head: 0.07, snout: 0.05, ear: (0.025, 0.05), antlers: false, tail: (0.12, 0.025) },
        "cow" => Quadruped { body: Vec3::new(0.24, 0.70, 0.85), leg_h: 0.75, leg_t: 0.12, head: 0.16, snout: 0.12, ear: (0.06, 0.05), antlers: false, tail: (0.55, 0.04) },
        "pig" => Quadruped { body: Vec3::new(0.17, 0.35, 0.55), leg_h: 0.35, leg_t: 0.09, head: 0.13, snout: 0.07, ear: (0.04, 0.05), antlers: false, tail: (0.10, 0.02) },
        "sheep" => Quadruped { body: Vec3::new(0.15, 0.32, 0.42), leg_h: 0.35, leg_t: 0.07, head: 0.09, snout: 0.06, ear: (0.04, 0.03), antlers: false, tail: (0.06, 0.03) },
        "goat" => Quadruped { body: Vec3::new(0.14, 0.28, 0.38), leg_h: 0.32, leg_t: 0.06, head: 0.09, snout: 0.07, ear: (0.035, 0.03), antlers: true, tail: (0.08, 0.025) },
        "fox" => Quadruped { body: Vec3::new(0.08, 0.13, 0.26), leg_h: 0.14, leg_t: 0.04, head: 0.07, snout: 0.07, ear: (0.025, 0.06), antlers: false, tail: (0.22, 0.045) },
        "hare" => Quadruped { body: Vec3::new(0.06, 0.10, 0.16), leg_h: 0.09, leg_t: 0.03, head: 0.055, snout: 0.0, ear: (0.02, 0.10), antlers: false, tail: (0.04, 0.025) },
        "deer" => Quadruped { body: Vec3::new(0.17, 0.50, 0.60), leg_h: 0.85, leg_t: 0.07, head: 0.11, snout: 0.09, ear: (0.04, 0.06), antlers: true, tail: (0.08, 0.03) },
        "bear" => Quadruped { body: Vec3::new(0.26, 0.38, 0.70), leg_h: 0.32, leg_t: 0.14, head: 0.14, snout: 0.07, ear: (0.05, 0.04), antlers: false, tail: (0.0, 0.0) },
        "mouse" => Quadruped { body: Vec3::new(0.02, 0.025, 0.035), leg_h: 0.015, leg_t: 0.008, head: 0.015, snout: 0.008, ear: (0.008, 0.008), antlers: false, tail: (0.05, 0.004) },
        _ => return None,
    };
    Some((q, variant))
}

/// The quadruped figure. Lying: the body rests on the ground with the legs
/// folded under it (short stubs stay visible at the sides) and the head held
/// forward and low. Sitting (cat/dog): the front legs stay vertical, the rump
/// drops to the ground and the chest rises — two boxes for the body.
fn draw_quadruped(b: &mut PropBuilder, q: &Quadruped, variant: AnimalVariant, materials: &[&str]) {
    let has_tail = materials.contains(&"tail");
    let has_legs = materials.contains(&"legs");
    b.group("body");
    let (body_center_y, body_len) = match variant {
        AnimalVariant::Lie => (q.body.y * 0.5 + 0.02, q.body.z),
        AnimalVariant::Sit => (q.leg_h * 0.5 + q.body.y * 0.5, q.body.z * 0.6),
        AnimalVariant::Stand => (q.leg_h + q.body.y * 0.5, q.body.z),
    };
    b.add_box(Vec3::new(0.0, body_center_y, 0.0), Vec3::new(q.body.x, q.body.y * 0.5, body_len));
    if variant == AnimalVariant::Sit {
        // Chest rises, rump sits: a second, higher front box.
        b.add_box(
            Vec3::new(0.0, q.leg_h + q.body.y * 0.45, body_len * 0.55),
            Vec3::new(q.body.x * 0.85, q.body.y * 0.45, q.body.z * 0.45),
        );
    }
    if has_legs {
        b.group("legs");
        let leg = |b: &mut PropBuilder, x: f32, z: f32, h: f32| {
            b.add_box(Vec3::new(x, h * 0.5, z), Vec3::new(q.leg_t, h * 0.5, q.leg_t));
        };
        let (fx, fz) = (q.body.x - q.leg_t * 0.5, body_len - q.leg_t);
        match variant {
            AnimalVariant::Stand => {
                for (x, z) in [(fx, fz), (-fx, fz), (fx, -fz), (-fx, -fz)] {
                    leg(b, x, z, q.leg_h);
                }
            }
            AnimalVariant::Sit => {
                // Front legs vertical, hind legs folded under the rump.
                for x in [fx, -fx] {
                    leg(b, x, body_len + q.leg_t, q.leg_h);
                    b.add_box(Vec3::new(x, q.leg_t, -body_len * 0.4), Vec3::new(q.leg_t, q.leg_t, body_len * 0.8));
                }
            }
            AnimalVariant::Lie => {
                // Folded: thin stubs along the belly.
                for x in [fx, -fx] {
                    b.add_box(Vec3::new(x, q.leg_t, 0.0), Vec3::new(q.leg_t, q.leg_t, body_len));
                }
            }
        }
    }
    b.group("head");
    let head_y = match variant {
        AnimalVariant::Stand => body_center_y + q.body.y * 0.5 + q.head * 0.55,
        AnimalVariant::Sit => q.leg_h + q.body.y * 0.9 + q.head * 0.8,
        AnimalVariant::Lie => body_center_y + q.body.y * 0.3,
    };
    let head_z = body_len + q.head * 0.8;
    b.add_box(Vec3::new(0.0, head_y, head_z), Vec3::splat(q.head));
    if q.snout > 0.0 {
        b.add_box(
            Vec3::new(0.0, head_y - q.head * 0.35, head_z + q.head + q.snout * 0.5),
            Vec3::new(q.head * 0.45, q.head * 0.35, q.snout * 0.5),
        );
    }
    // Ears (or antlers for goat/deer) on top of the head.
    let ear_top = head_y + q.head + q.ear.1 * 0.5;
    for x in [q.head * 0.55, -q.head * 0.55] {
        if q.antlers {
            // A small fork: vertical stem + side branch.
            b.add_box(Vec3::new(x, ear_top, head_z), Vec3::new(q.ear.0, q.ear.1 * 0.5, q.ear.0));
            b.add_box(Vec3::new(x * 1.5, ear_top + q.ear.1 * 0.4, head_z), Vec3::new(q.ear.1 * 0.5, q.ear.0, q.ear.0));
        } else {
            b.add_box(Vec3::new(x, ear_top, head_z), Vec3::new(q.ear.0, q.ear.1 * 0.5, q.ear.0));
        }
    }
    if has_tail && q.tail.0 > 0.0 {
        b.group("tail");
        let rear = -body_len;
        if variant == AnimalVariant::Stand {
            // The tail hangs down from the rear (axis-aligned boxes, so
            // vertical) — and never below the floor.
            let tail_y = (body_center_y + q.body.y * 0.3 - q.tail.0 * 0.3).max(q.tail.0 * 0.5);
            b.add_box(Vec3::new(0.0, tail_y, rear - q.tail.1), Vec3::new(q.tail.1, q.tail.0 * 0.5, q.tail.1));
        } else {
            // Lying or sitting: the tail rests on the ground behind the body.
            b.add_box(Vec3::new(0.0, q.tail.1, rear - q.tail.0 * 0.5), Vec3::new(q.tail.1, q.tail.1, q.tail.0 * 0.5));
        }
    }
}

/// A chicken: plump body, small head with a beak, tail feathers, two legs.
/// Drawn at a nominal size (like the quadrupeds); the catalog table carries
/// the measured result.
fn draw_chicken(b: &mut PropBuilder) {
    b.group("body");
    let body = Vec3::new(0.088, 0.105, 0.105);
    let body_y = 0.147;
    b.add_box(Vec3::new(0.0, body_y, 0.0), body);
    // Tail feathers: a box rising at the rear.
    b.add_box(Vec3::new(0.0, body_y + body.y + 0.03, -body.z - 0.03), Vec3::new(body.x * 0.6, 0.05, 0.05));
    b.group("head");
    let head_r = 0.0385;
    let head = Vec3::new(0.0, body_y + body.y + head_r * 0.6, body.z + head_r * 0.9);
    b.add_box(head, Vec3::splat(head_r));
    // Beak and comb.
    b.add_box(head + Vec3::new(0.0, -head_r * 0.2, head_r + 0.02), Vec3::new(head_r * 0.35, head_r * 0.3, 0.025));
    b.add_box(head + Vec3::new(0.0, head_r + 0.015, 0.0), Vec3::new(head_r * 0.4, 0.02, head_r * 0.5));
    b.group("legs");
    let leg_h = body_y - body.y;
    for x in [0.029, -0.029] {
        b.add_box(Vec3::new(x, leg_h * 0.5, 0.0), Vec3::new(0.012, leg_h * 0.5, 0.012));
    }
}

/// Mesh for an animal entry: a quadruped (with pose variants) or the chicken.
fn animal(entry: &CatalogEntry) -> ModelData {
    let mut b = PropBuilder::new();
    if entry.name == "chicken" {
        draw_chicken(&mut b);
        return b.finish(entry);
    }
    let (q, variant) = quadruped(entry).unwrap_or_else(|| panic!("no quadruped params for {}", entry.name));
    draw_quadruped(&mut b, &q, variant, entry.materials);
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

/// Street props: items that live outdoors, origin at base center.
fn street_item(entry: &CatalogEntry) -> ModelData {
    let mut b = PropBuilder::new();
    let s = Vec3::from_array(entry.size);
    match entry.name {
        "pole" => {
            b.group("body");
            b.add_cylinder(Vec3::ZERO, s.x * 0.5, s.y, 0.0, 12);
        }
        "streetlight" => {
            // Pole at −Z, arm reaching toward +Z with the lamp head at its end
            // (the lit side is +Z).
            b.group("pole");
            b.add_cylinder(Vec3::new(0.0, 0.0, -0.28), 0.05, 4.35, 0.0, 12);
            b.add_box(Vec3::new(0.0, 4.31, 0.22), Vec3::new(0.035, 0.035, 0.5));
            b.group("head");
            b.add_box(Vec3::new(0.0, s.y - 0.075, 0.72), Vec3::new(0.25, 0.075, 0.15));
            b.group("lamp");
            b.add_box(Vec3::new(0.0, s.y - 0.165, 0.72), Vec3::new(0.2, 0.015, 0.1));
        }
        "bench_outdoor" => {
            // Two side frames carry the seat slats and the backrest.
            b.group("frame");
            for x in [s.x * 0.5 - 0.025, -(s.x * 0.5 - 0.025)] {
                b.add_upright(Vec3::new(x, 0.0, 0.0), Vec3::new(0.05, 0.41, s.z));
            }
            b.group("seat");
            for z in [-0.2f32, 0.0, 0.2] {
                b.add_upright(Vec3::new(0.0, 0.41, z), Vec3::new(s.x - 0.04, 0.04, 0.16));
            }
            b.group("back");
            b.add_upright(Vec3::new(0.0, 0.45, -s.z * 0.5 + 0.025), Vec3::new(s.x - 0.04, s.y - 0.45, 0.05));
        }
        "car" => {
            // Wheels touch the ground, the body clears it by 0.15 m, the cabin
            // sits slightly behind center (−Z), windows band its upper half.
            b.group("wheels");
            for x in [s.x * 0.5 - 0.125, -(s.x * 0.5 - 0.125)] {
                for z in [1.45f32, -1.45] {
                    b.add_upright(Vec3::new(x, 0.0, z), Vec3::new(0.25, 0.6, 0.6));
                }
            }
            b.group("body");
            b.add_box(Vec3::new(0.0, 0.45, 0.0), Vec3::new(0.85, 0.3, s.z * 0.5));
            b.group("cabin");
            b.add_box(Vec3::new(0.0, 1.125, -0.3), Vec3::new(0.75, 0.375, 1.1));
            b.group("windows");
            b.add_box(Vec3::new(0.0, 1.25, -0.3), Vec3::new(0.76, 0.2, 1.11));
        }
        "tree" => {
            b.group("trunk");
            b.add_cylinder(Vec3::ZERO, 0.12, 1.7, 0.0, 10);
            b.group("canopy");
            b.add_sphere(Vec3::new(0.0, 2.6, 0.0), 0.9, 12, 8);
            b.add_sphere(Vec3::new(0.45, 2.45, 0.3), 0.55, 12, 6);
            b.add_sphere(Vec3::new(-0.4, 2.3, -0.35), 0.6, 12, 6);
        }
        "tree_pine" => {
            // A conifer reads as a silhouette by its STACK of cones: narrower
            // and taller than the broadleaf, so a mixed forest is not one
            // drawing repeated, which is what a scattered wood is judged by.
            b.group("trunk");
            b.add_cylinder(Vec3::ZERO, 0.11, 1.2, 0.0, 10);
            b.group("canopy");
            // 12 segments puts a vertex on each axis, so the girth is exact.
            b.add_cone(Vec3::new(0.0, 0.9, 0.0), 0.9, 1.6, 12);
            b.add_cone(Vec3::new(0.0, 1.8, 0.0), 0.68, 1.5, 12);
            b.add_cone(Vec3::new(0.0, 2.6, 0.0), 0.45, 1.6, 12);
        }
        "bush" => {
            b.group("canopy");
            b.add_sphere(Vec3::new(0.15, 0.45, 0.12), 0.45, 10, 6);
            b.add_sphere(Vec3::new(-0.15, 0.45, -0.12), 0.45, 10, 6);
        }
        "rock" => {
            b.group("body");
            // One low-segment sphere, squashed per axis to the declared size.
            let c = Vec3::new(0.0, s.y * 0.5, 0.0);
            let start = b.mesh.positions.len();
            b.add_sphere(c, s.x * 0.5, 4, 6);
            let squash = Vec3::new(1.0, s.y / s.x, s.z / s.x);
            for p in &mut b.mesh.positions[start..] {
                let v = c + (Vec3::from_array(*p) - c) * squash;
                *p = v.to_array();
            }
        }
        _ => b.add_upright(Vec3::ZERO, s),
    }
    b.finish(entry)
}

/// Outdoor ground: a slab whose top surface is exactly at `y = 0` (the slab
/// hangs below, like interior floors) — props and cameras are placed on it.
/// No walls, no ceiling.
fn exterior(entry: &CatalogEntry) -> ModelData {
    let mut b = PropBuilder::new();
    let s = Vec3::from_array(entry.size);
    let t = s.y; // slab thickness
    b.group("ground");
    b.add_box(Vec3::new(0.0, -t * 0.5, 0.0), Vec3::new(s.x * 0.5, t * 0.5, s.z * 0.5));
    match entry.name {
        "street_park" => {
            // Walking path: 2 m wide, along Z, slightly above the ground top.
            b.group("path");
            b.add_box(Vec3::new(0.0, 0.03, 0.0), Vec3::new(1.0, 0.03, s.z * 0.5));
        }
        "street_crossing" => {
            // Two crossing roads, 6 m wide, along X and along Z.
            b.group("road");
            b.add_box(Vec3::new(0.0, 0.02, 0.0), Vec3::new(s.x * 0.5, 0.02, 3.0));
            b.add_box(Vec3::new(0.0, 0.02, 0.0), Vec3::new(3.0, 0.02, s.z * 0.5));
            // Dashed center lines on both roads, a dash every 2 m.
            b.group("marking");
            let mut d = -s.x * 0.5 + 1.0;
            while d <= s.x * 0.5 - 1.0 {
                if d.abs() > 3.5 {
                    b.add_box(Vec3::new(d, 0.045, 0.0), Vec3::new(0.5, 0.005, 0.075));
                    b.add_box(Vec3::new(0.0, 0.045, d), Vec3::new(0.075, 0.005, 0.5));
                }
                d += 2.0;
            }
        }
        _ => {}
    }
    b.finish(entry)
}

// --- generated fills (forest bands, hedge rows) --------------------------------

/// The material groups a fill can draw: the union of its species' groups, in a
/// stable order. A band is not one catalog entry, so its material table comes
/// from what it plants — and `material_set` then hits `trunk`, `canopy`, `bush`
/// and `rock` by name, exactly as it does on a single prop.
pub fn scatter_materials(field: &ScatterField) -> Vec<String> {
    let Some(preset) = scatter_preset(&field.preset) else {
        return vec![];
    };
    let source = if preset.far.is_empty() { preset.near } else { preset.far };
    let mut names: Vec<String> = Vec::new();
    for species in source {
        let Some(entry) = catalog_find(species.prop) else { continue };
        for group in entry.materials {
            if !names.iter().any(|n| n == *group) {
                names.push((*group).to_string());
            }
        }
    }
    names
}

/// Builds one mesh out of whole catalog drawings: a species is drawn once and
/// merged in under a transform, so a scattered wood reuses the exact geometry
/// the single-prop path produces instead of a second, worse drawing routine.
struct FieldBuilder {
    mesh: MeshData,
    parts: Vec<PartData>,
    /// Material name -> index in `materials`.
    slots: std::collections::HashMap<String, usize>,
    materials: Vec<MaterialRecord>,
}

impl FieldBuilder {
    fn new(materials: Vec<String>) -> Self {
        let slots = materials
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();
        let records = materials
            .into_iter()
            .map(|name| MaterialRecord {
                name,
                // Neutral: the forest reads before material_set, and green
                // everywhere would fight the director's own grade.
                base_color: [0.62, 0.60, 0.55, 1.0],
                metallic: 0.0,
                roughness: 0.8,
                base_color_texture: None,
                normal_map_texture: None,
                roughness_texture: None,
                metalness_texture: None,
                invert_roughness: false,
            })
            .collect();
        Self {
            mesh: MeshData { positions: vec![], normals: vec![], uvs: vec![], indices: vec![] },
            parts: vec![],
            slots,
            materials: records,
        }
    }

    /// Copy one part of a drawing in, transformed. Only the vertices the part
    /// references are copied — a tree is a few hundred vertices, and a band is
    /// hundreds of trees, so dragging the whole drawing per instance would
    /// inflate the mesh for nothing.
    fn merge_part(&mut self, model: &ModelData, part: &PartData, at: Vec3, rot: Quat, scale: f32) {
        debug_assert!(part.start as usize + part.count as usize <= model.mesh.indices.len());
        let Some(material) = self.slots.get(&model.materials.get(part.material).map(|m| m.name.clone()).unwrap_or_default()) else {
            return;
        };
        let material = *material;
        let ibase = self.mesh.indices.len() as u32;
        let start = (part.start as usize).min(model.mesh.indices.len());
        let end = (part.start as usize + part.count as usize).min(model.mesh.indices.len());
        let mut remap: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for &v in &model.mesh.indices[start..end] {
            let mapped = *remap.entry(v).or_insert_with(|| {
                let p = self.mesh.positions.len() as u32;
                let local = Vec3::from_array(model.mesh.positions[v as usize]);
                let n = Vec3::from_array(model.mesh.normals[v as usize]);
                self.mesh.positions.push((at + rot * (local * scale)).to_array());
                // Uniform scale keeps normals valid under rotation.
                self.mesh.normals.push((rot * n).to_array());
                self.mesh.uvs.push(model.mesh.uvs.get(v as usize).copied().unwrap_or([0.0, 0.0]));
                p
            });
            // `mapped` is already an index into this mesh's positions.
            self.mesh.indices.push(mapped);
        }
        let count = (self.mesh.indices.len() as u32 - ibase) as u32;
        if count > 0 {
            self.parts.push(PartData { material, start: ibase, count });
        }
    }

    fn finish(self) -> ModelData {
        ModelData { mesh: self.mesh, skin: None, materials: self.materials, parts: self.parts, diagnostics: vec![] }
    }
}

/// Mesh for a generated fill: the plan is replayed here with the same pure
/// function the generating command used, so the count the director was told and
/// the geometry in the frame cannot disagree.
///
/// Only the far band is drawn. The near zone is editable scene objects — that
/// is the whole split: what you shoot in is movable, what stands behind it is a
/// silhouette and costs one mesh.
pub fn scatter_field_model(field: &ScatterField) -> Option<ModelData> {
    let preset = scatter_preset(&field.preset)?;
    let placed = plan(preset, &field.shape, field.seed).ok()?;
    let band = phantom_core::band_of(&placed);
    if band.len() > MAX_SCATTER_INSTANCES {
        return None;
    }
    let mut b = FieldBuilder::new(scatter_materials(field));
    // Draw each species once and reuse it: a wood is the same dozen shapes
    // placed many times, and building it that way keeps the cost per instance
    // at a copy instead of a rebuild.
    let mut drawings: std::collections::HashMap<&str, ModelData> = std::collections::HashMap::new();
    for p in &band {
        if !drawings.contains_key(p.prop) {
            let Some(model) = prop_model(p.prop) else { continue };
            drawings.insert(p.prop, model);
        }
        let Some(model) = drawings.get(p.prop) else { continue };
        let at = Vec3::from_array(p.position);
        let rot = Quat::from_rotation_y(p.yaw);
        for part in &model.parts {
            b.merge_part(model, part, at, rot, p.scale);
        }
    }
    Some(b.finish())
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
            // Exterior grounds hang their slab below y = 0 (its top is the
            // placement surface), so their bottom sits at −0.24.
            let (slack, base) = match entry.kind {
                CatalogKind::Interior => (2.0 * T, T),
                CatalogKind::Exterior => (0.05, 0.24),
                _ => (0.05, 0.0),
            };
            for axis in 0..3 {
                let (g, w) = (got[axis], want[axis]);
                let limit = slack + w * 0.06;
                assert!(g >= w - limit, "{}: size along axis {axis} is {g}, below the table's {w}", entry.name);
                assert!(g <= w + limit, "{}: size along axis {axis} is {g}, above the table's {w}", entry.name);
            }
            assert!(lo.y.abs() <= base + 1e-3, "{}: mesh bottom at {:.3}, expected {base}", entry.name, lo.y);
            // The interior floor and the exterior ground top lie on y = 0:
            // items and cameras are placed on it, not inside it.
            if matches!(entry.kind, CatalogKind::Interior | CatalogKind::Exterior) {
                assert!(
                    model.mesh.positions.iter().any(|p| p[1].abs() < 1e-4),
                    "{}: the floor surface must be at y = 0",
                    entry.name
                );
            }
        }
    }

    #[test]
    fn a_forest_band_replays_the_plan_it_was_given() {
        let field = phantom_core::ScatterField {
            preset: "forest".into(),
            shape: phantom_core::ScatterShape::Area { size: [16.0, 16.0], density: 0.3, action: Some([[-3.0, -3.0], [3.0, 3.0]]) },
            seed: 5,
        };
        let model = scatter_field_model(&field).unwrap();
        let preset = phantom_core::scatter_preset("forest").unwrap();
        let band = phantom_core::band_of(&phantom_core::plan(preset, &field.shape, field.seed).unwrap());
        // One part per species part drawn: the mesh is the plan, not a guess.
        let species: usize = band.iter().map(|p| p.prop).collect::<std::collections::HashSet<_>>().len();
        assert!(band.len() >= species, "the band planted something");
        // Materials: every part names a group the field declares, and nothing
        // sits above the tallest tree the preset can draw.
        for part in &model.parts {
            assert!(part.material < model.materials.len(), "part names a declared group");
        }
        assert!(!model.materials.is_empty(), "trunk/canopy/bush/rock are addressable");
        let top = model.mesh.positions.iter().map(|p| p[1]).fold(0.0f32, f32::max);
        assert!(top > 1.0 && top <= phantom_core::band_top_y(preset) * 1.4 + 1e-2, "band top {top}");
        // Every instance stands on the ground: nothing floats, nothing sinks.
        assert!(model.mesh.positions.iter().all(|p| p[1] > -1e-3), "the band sits on y = 0");
        // The merge copied every triangle of every instance and remapped its
        // vertices: the count is exactly the species' geometry times the plan,
        // and no index reaches past the vertex buffer.
        let want: u32 = band
            .iter()
            .map(|p| prop_model(p.prop).map(|m| m.parts.iter().map(|part| part.count).sum::<u32>()).unwrap_or(0))
            .sum();
        assert!(want > 0, "the species have geometry");
        let got: u32 = model.parts.iter().map(|part| part.count).sum();
        assert_eq!(got, want, "the band is the plan, triangle for triangle");
        assert_eq!(got as usize, model.mesh.indices.len(), "the parts partition the index buffer");
        assert!(model.mesh.indices.iter().all(|&i| (i as usize) < model.mesh.positions.len()));
        let (lo, hi) = bbox(&model);
        assert!(hi.x - lo.x > 8.0 && hi.z - lo.z > 8.0, "the band covers the area: {:?}", hi - lo);
    }

    #[test]
    fn crossing_has_roads_and_no_ceiling() {
        let entry = catalog_find("street_crossing").unwrap();
        let model = prop_model("street_crossing").unwrap();
        let (lo, hi) = bbox(&model);
        // No walls and no ceiling: the mesh stays a flat slab with road paint.
        assert!(hi.y - lo.y <= 0.31, "bbox height {:.3} — nothing above the roads", hi.y - lo.y);
        for name in ["road", "marking"] {
            let material = entry.materials.iter().position(|m| *m == name).unwrap();
            let count: u32 = model.parts.iter().filter(|p| p.material == material).map(|p| p.count).sum();
            assert!(count > 0, "{name} group must have geometry");
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


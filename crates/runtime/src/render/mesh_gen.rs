//! Mesh generation: the built-in skinned character and the background arc screen.

use glam::{Mat4, Vec3};
use phantom_core::project::Background;
use phantom_core::project::BuiltinActor;
use phantom_core::rig::actor_bones;

use crate::import::{BoneData, MeshData, ModelData, SkinData};

struct MeshBuilder {
    mesh: MeshData,
    joint_indices: Vec<[u16; 4]>,
    joint_weights: Vec<[f32; 4]>,
}

impl MeshBuilder {
    fn new() -> Self {
        Self {
            mesh: MeshData { positions: vec![], normals: vec![], uvs: vec![], indices: vec![] },
            joint_indices: vec![],
            joint_weights: vec![],
        }
    }

    /// Axis-aligned box rigidly bound to a single bone.
    fn add_box(&mut self, center: Vec3, half: Vec3, bone: usize) {
        let faces = [
            // (normal, u axis, v axis)
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
                // Rounded normals: continuous across edges, so the outline has no
                // seams; the shading is "inflated" — a consistent stylization of the
                // built-in rig.
                self.mesh.normals.push(((p - center) / half).normalize().to_array());
                self.mesh.uvs.push([(su + 1.0) * 0.5, (sv + 1.0) * 0.5]);
                self.joint_indices.push([bone as u16, 0, 0, 0]);
                self.joint_weights.push([1.0, 0.0, 0.0, 0.0]);
            }
            // Winding: counter-clockwise seen from outside.
            if n.dot(u.cross(v)) > 0.0 {
                self.mesh.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            } else {
                self.mesh.indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
            }
        }
    }
}

/// Built-in character: boxes rigidly bound to the bones of the humanoid skeleton.
/// A stylized "robot"; the art model is replaced by an FBX import.
pub fn builtin_phantom_model() -> ModelData {
    let mut b = MeshBuilder::new();
    // Pelvis.
    b.add_box(Vec3::new(0.0, 1.01, 0.0), Vec3::new(0.14, 0.07, 0.09), 0);
    // Torso.
    b.add_box(Vec3::new(0.0, 1.30, 0.0), Vec3::new(0.17, 0.22, 0.10), 1);
    // Head.
    b.add_box(Vec3::new(0.0, 1.63, 0.0), Vec3::new(0.10, 0.11, 0.10), 2);
    // Arms: upper arm + forearm, along X.
    for s in [1.0f32, -1.0] {
        let upper = if s > 0.0 { 3 } else { 5 };
        let lower = if s > 0.0 { 4 } else { 6 };
        b.add_box(Vec3::new(s * 0.305, 1.45, 0.0), Vec3::new(0.125, 0.045, 0.045), upper);
        b.add_box(Vec3::new(s * 0.555, 1.45, 0.0), Vec3::new(0.125, 0.04, 0.04), lower);
        // Thigh + shin + foot.
        let uleg = if s > 0.0 { 7 } else { 9 };
        let lleg = if s > 0.0 { 8 } else { 10 };
        b.add_box(Vec3::new(s * 0.09, 0.70, 0.0), Vec3::new(0.055, 0.20, 0.055), uleg);
        b.add_box(Vec3::new(s * 0.09, 0.275, 0.0), Vec3::new(0.05, 0.225, 0.05), lleg);
        b.add_box(Vec3::new(s * 0.09, 0.03, 0.05), Vec3::new(0.05, 0.03, 0.10), lleg);
    }

    finish_model(b, BuiltinActor::Humanoid)
}

fn finish_model(b: MeshBuilder, actor: BuiltinActor) -> ModelData {
    let definitions = actor_bones(actor);
    let mut global: Vec<Vec3> = Vec::with_capacity(definitions.len());
    for bone in definitions.iter() {
        let g = match bone.parent {
            Some(p) => global[p] + bone.rest_translation,
            None => bone.rest_translation,
        };
        global.push(g);
    }
    let bones: Vec<BoneData> = definitions
        .iter()
        .enumerate()
        .map(|(i, bone)| BoneData {
            name: bone.name.to_string(),
            parent: bone.parent,
            inverse_bind: Mat4::from_translation(-global[i]).to_cols_array_2d(),
            rest_translation: bone.rest_translation.to_array(),
            rest_rotation: [0.0, 0.0, 0.0, 1.0],
            rest_scale: [1.0, 1.0, 1.0],
        })
        .collect();

    ModelData {
        mesh: b.mesh,
        skin: Some(SkinData { bones, joint_indices: b.joint_indices, joint_weights: b.joint_weights }),
        materials: vec![],
        parts: vec![],
        diagnostics: vec![],
    }
}

/// Original box-built low-poly actors, requiring no external assets.
pub fn builtin_actor_model(actor: BuiltinActor) -> ModelData {
    if actor == BuiltinActor::Humanoid {
        return builtin_phantom_model();
    }
    let definitions = actor_bones(actor);
    let mut global: Vec<Vec3> = Vec::new();
    for bone in &definitions {
        global.push(bone.rest_translation + bone.parent.map(|p| global[p]).unwrap_or(Vec3::ZERO));
    }
    let mut b = MeshBuilder::new();
    let mut part = |bone: usize, offset: Vec3, half: Vec3| b.add_box(global[bone] + offset, half, bone);
    if actor == BuiltinActor::Bird {
        part(0, Vec3::ZERO, Vec3::new(0.14, 0.13, 0.25));
        part(1, Vec3::ZERO, Vec3::splat(0.10));
        part(1, Vec3::new(0., -0.02, 0.14), Vec3::new(0.04, 0.035, 0.07));
        for (upper, lower, sign) in [(2, 3, 1.), (4, 5, -1.)] {
            part(upper, Vec3::new(sign * 0.16, 0., -0.03), Vec3::new(0.16, 0.035, 0.17));
            part(lower, Vec3::new(sign * 0.17, 0., -0.06), Vec3::new(0.17, 0.025, 0.12));
        }
        part(6, Vec3::new(0., 0., -0.12), Vec3::new(0.11, 0.035, 0.15));
        for i in [7, 8] {
            part(i, Vec3::new(0., -0.035, 0.), Vec3::new(0.02, 0.065, 0.02));
            part(i, Vec3::new(0., -0.085, 0.04), Vec3::new(0.035, 0.015, 0.07));
        }
    } else {
        let horse = actor == BuiltinActor::Horse;
        let (width, length, leg) = if horse { (0.28, 0.65, 0.55) } else { (0.20, 0.43, 0.28) };
        part(0, Vec3::ZERO, Vec3::new(width, if horse { 0.27 } else { 0.18 }, length));
        part(1, Vec3::new(0., if horse { 0.22 } else { 0.03 }, 0.), Vec3::new(width * 0.65, if horse { 0.30 } else { 0.15 }, 0.15));
        part(2, Vec3::ZERO, Vec3::new(width * 0.7, 0.16, 0.20));
        part(2, Vec3::new(0., -0.06, 0.22), Vec3::new(width * 0.5, 0.09, 0.15));
        for sign in [-1., 1.] {
            part(2, Vec3::new(sign * width * 0.5, 0.21, -0.06), Vec3::new(0.045, 0.10, 0.06));
        }
        part(3, Vec3::new(0., if horse { -0.25 } else { -0.06 }, -0.20), Vec3::new(0.07, if horse { 0.28 } else { 0.08 }, 0.25));
        for i in [4, 6, 8, 10] {
            part(i, Vec3::new(0., -leg * 0.5, 0.), Vec3::new(width * 0.32, leg * 0.5, 0.075));
            part(i + 1, Vec3::new(0., -leg * 0.5, 0.), Vec3::new(width * 0.23, leg * 0.5, 0.055));
            part(i + 1, Vec3::new(0., -leg, 0.035), Vec3::new(0.075, 0.05, 0.10));
        }
        if horse {
            part(1, Vec3::new(0., 0.22, -0.16), Vec3::new(0.045, 0.29, 0.06)); // mane
        }
    }
    finish_model(b, actor)
}

/// Arc screen: an arc with UV uniform along arc length.
/// Orientation — `bg.bulge` (corrective A3a): `toward_camera` gives
/// `z = (cos θ − 1)·R` (the middle is CLOSER to the viewer than `center` —
/// a screen-as-backdrop); `away_from_camera` (default) — `z = (1 − cos θ)·R`, edges
/// closer to the viewer, so the screen wraps around the scene. UV mode —
/// `bg.fit` (A3c): `stretch` — the previous stretching over the whole arc;
/// `cover`/`contain` preserve the panorama proportions from `image_aspect`
/// (texture width/height; None — stretch).
pub fn arc_screen_mesh(bg: &Background, segments: u32, image_aspect: Option<f32>) -> MeshData {
    use phantom_core::project::{BackgroundFit, Bulge};
    // UV span along each axis. Physical screen aspect = arc length / height.
    let (u_span, v_span) = match (bg.fit, image_aspect.filter(|a| a.is_finite() && *a > 0.0)) {
        (BackgroundFit::Stretch, _) | (_, None) => (1.0, 1.0),
        (BackgroundFit::Cover, Some(ai)) => {
            let screen = (bg.arc_radians * bg.radius) / bg.height;
            (if ai > screen { screen / ai } else { 1.0 }, if ai > screen { 1.0 } else { ai / screen })
        }
        (BackgroundFit::Contain, Some(ai)) => {
            // Whole image visible: the UV span goes OUTSIDE [0,1] along the wide
            // axis (screen edges sample the texture's clamped edges).
            let screen = (bg.arc_radians * bg.radius) / bg.height;
            (if ai > screen { 1.0 } else { screen / ai }, if ai > screen { ai / screen } else { 1.0 })
        }
    };
    let z_sign = match bg.bulge {
        Bulge::TowardCamera => -1.0,
        Bulge::AwayFromCamera => 1.0,
    };
    let mut mesh = MeshData { positions: vec![], normals: vec![], uvs: vec![], indices: vec![] };
    for i in 0..=segments {
        let t = i as f32 / segments as f32;
        let theta = (t - 0.5) * bg.arc_radians;
        // Arc middle (θ = 0) sits at `center`; edges are offset by (1 − cos θ)·R
        // with the sign of the bulge orientation.
        let x = theta.sin() * bg.radius;
        let z = z_sign * (1.0 - theta.cos()) * bg.radius;
        let u = 0.5 + (t - 0.5) * u_span;
        let base = mesh.positions.len() as u32;
        for (y, v) in [(0.0f32, 1.0f32), (bg.height, 0.0f32)] {
            mesh.positions
                .push((bg.center + Vec3::new(x, y - bg.height * 0.5, z)).to_array());
            // Normal points from the center of curvature toward the viewer (for
            // away_from_camera — toward the center of curvature; the material is
            // unlit and double-sided, so the normal is kept consistent with the orientation).
            mesh.normals.push(Vec3::new(-z_sign * theta.sin(), 0.0, -z_sign * theta.cos()).to_array());
            mesh.uvs.push([u, 0.5 + (v - 0.5) * v_span]);
        }
        if i > 0 {
            let a = base - 2;
            mesh.indices.extend_from_slice(&[a, base, a + 1, a + 1, base, base + 1]);
        }
    }
    mesh
}

#[cfg(test)]
mod actor_tests {
    use super::*;
    #[test]
    fn bundled_meshes_have_valid_weights_bind_matrices_and_geometry() {
        for actor in [BuiltinActor::Humanoid, BuiltinActor::Wolf, BuiltinActor::Horse, BuiltinActor::Bird] {
            let model = builtin_actor_model(actor);
            let mesh = &model.mesh;
            let skin = model.skin.as_ref().unwrap();
            assert!(!mesh.positions.is_empty());
            assert_eq!(mesh.positions.len(), mesh.normals.len());
            assert_eq!(mesh.positions.len(), mesh.uvs.len());
            assert_eq!(mesh.positions.len(), skin.joint_indices.len());
            assert_eq!(mesh.positions.len(), skin.joint_weights.len());
            assert!(mesh.indices.iter().all(|&i| (i as usize) < mesh.positions.len()));
            for p in &mesh.positions {
                assert!(Vec3::from_array(*p).is_finite());
            }
            for n in &mesh.normals {
                assert!((Vec3::from_array(*n).length() - 1.).abs() < 1e-5);
            }
            for uv in &mesh.uvs {
                assert!(uv.iter().all(|v| v.is_finite()));
            }
            let mut global = Vec::new();
            for (i, b) in skin.bones.iter().enumerate() {
                assert!(b.parent.is_none_or(|p| p < i));
                let g = Vec3::from_array(b.rest_translation) + b.parent.map(|p| global[p]).unwrap_or(Vec3::ZERO);
                global.push(g);
                let inverse = Mat4::from_cols_array_2d(&b.inverse_bind);
                assert!(inverse.is_finite());
                assert!((Mat4::from_translation(g) * inverse).abs_diff_eq(Mat4::IDENTITY, 1e-5));
            }
            for (joints, weights) in skin.joint_indices.iter().zip(&skin.joint_weights) {
                assert!(joints.iter().all(|&i| (i as usize) < skin.bones.len()));
                assert!(weights.iter().all(|&w| w.is_finite() && w >= 0.));
                assert!((weights.iter().sum::<f32>() - 1.).abs() < 1e-5);
            }
            for &action in actor.actions() {
                let pose = phantom_core::rig::sample_actor_action(actor, action, 0.37).unwrap();
                let mut transforms: Vec<Mat4> = Vec::new();
                for (i, b) in skin.bones.iter().enumerate() {
                    let local = Mat4::from_rotation_translation(pose.bones[i].rotation, pose.bones[i].translation);
                    transforms.push(b.parent.map(|p| transforms[p]).unwrap_or(Mat4::IDENTITY) * local);
                }
                for (i, p) in mesh.positions.iter().enumerate() {
                    let joint = skin.joint_indices[i][0] as usize;
                    let skinned = transforms[joint] * Mat4::from_cols_array_2d(&skin.bones[joint].inverse_bind);
                    assert!(skinned.transform_point3(Vec3::from_array(*p)).is_finite());
                }
            }
        }
    }
}

#[cfg(test)]
mod screen_tests {
    use super::*;
    use phantom_core::project::{BackgroundFit, BackgroundSource, Bulge};
    use glam::Vec3;
    use phantom_core::AssetId;

    fn bg(bulge: Bulge, fit: BackgroundFit) -> Background {
        Background {
            image: BackgroundSource::Image(AssetId(1)),
            arc_radians: 2.0,
            radius: 6.0,
            height: 4.0,
            center: Vec3::new(0.0, 2.0, -5.0),
            bulge,
            fit,
        }
    }

    fn edge_center_z(mesh: &MeshData) -> (f32, f32) {
        // Top edge: odd vertices; the arc middle is the middle segment.
        let top: Vec<[f32; 3]> = mesh.positions.iter().skip(1).step_by(2).copied().collect();
        let mid = top[top.len() / 2];
        let edge = top[0];
        (edge[2], mid[2])
    }

    /// Orientation pin (corrective A3a): must not change silently.
    #[test]
    fn bulge_orientation_is_pinned() {
        let toward = arc_screen_mesh(&bg(Bulge::TowardCamera, BackgroundFit::Stretch), 16, None);
        let (te, tm) = edge_center_z(&toward);
        assert!(te < tm, "toward_camera: edges FARTHER than the middle (z={te} < {tm}) — screen-as-backdrop");

        let away = arc_screen_mesh(&bg(Bulge::AwayFromCamera, BackgroundFit::Stretch), 16, None);
        let (ae, am) = edge_center_z(&away);
        assert!(ae > am, "away_from_camera: edges CLOSER than the middle (z={ae} > {am}) — screen wraps the scene");
        // The middle of both orientations is exactly at center.z.
        assert!((am - -5.0).abs() < 1e-5 && (tm - -5.0).abs() < 1e-5);
    }

    #[test]
    fn fit_modes_shape_uv_spans() {
        // Screen: arc length 12 m, height 4 m → aspect 3.0; panorama 16:9 ≈ 1.78.
        let image = Some(16.0 / 9.0);
        let stretch = arc_screen_mesh(&bg(Bulge::AwayFromCamera, BackgroundFit::Stretch), 8, image);
        let us: Vec<f32> = stretch.uvs.iter().map(|uv| uv[0]).collect();
        assert_eq!(us.iter().cloned().fold(0. / 0., f32::min), 0.0);
        assert_eq!(us.iter().cloned().fold(0. / 0., f32::max), 1.0);
        let vs: Vec<f32> = stretch.uvs.iter().map(|uv| uv[1]).collect();
        assert_eq!(vs.iter().cloned().fold(0. / 0., f32::min), 0.0);
        assert_eq!(vs.iter().cloned().fold(0. / 0., f32::max), 1.0);

        // cover: a 1.78 image on a 3.0 screen — top/bottom cropped (v span < 1),
        // the full range horizontally.
        let cover = arc_screen_mesh(&bg(Bulge::AwayFromCamera, BackgroundFit::Cover), 8, image);
        let cu: Vec<f32> = cover.uvs.iter().map(|uv| uv[0]).collect();
        assert_eq!(cu.iter().cloned().fold(0. / 0., f32::min), 0.0);
        let cv: Vec<f32> = cover.uvs.iter().map(|uv| uv[1]).collect();
        let vmin = cv.iter().cloned().fold(0. / 0., f32::min);
        let vmax = cv.iter().cloned().fold(0. / 0., f32::max);
        let span = (16.0 / 9.0f32) / 3.0;
        assert!((vmax - vmin - span).abs() < 1e-5, "cover v span {span}: {vmin}..{vmax}");
        assert!((vmin - (0.5 - span / 2.0)).abs() < 1e-5, "crop is symmetric");

        // contain: whole image visible — u goes outside [0,1] (side bars, the image
        // is narrower than the screen). A wide image — top/bottom bars (v).
        let contain = arc_screen_mesh(&bg(Bulge::AwayFromCamera, BackgroundFit::Contain), 8, image);
        let ku: Vec<f32> = contain.uvs.iter().map(|uv| uv[0]).collect();
        let kmin = ku.iter().cloned().fold(0. / 0., f32::min);
        let kmax = ku.iter().cloned().fold(0. / 0., f32::max);
        assert!(kmin < 0.0 && kmax > 1.0, "contain u goes outside the texture: {kmin}..{kmax}");
        let kv: Vec<f32> = contain.uvs.iter().map(|uv| uv[1]).collect();
        assert_eq!(kv.iter().cloned().fold(0. / 0., f32::min), 0.0);
        assert_eq!(kv.iter().cloned().fold(0. / 0., f32::max), 1.0);
        let contain_w = arc_screen_mesh(&bg(Bulge::AwayFromCamera, BackgroundFit::Contain), 8, Some(4.0));
        let wv: Vec<f32> = contain_w.uvs.iter().map(|uv| uv[1]).collect();
        assert!(wv.iter().cloned().fold(0. / 0., f32::min) < 0.0, "contain wide image: v < 0");

        // Wide 4:1 panorama on a 3.0 screen: cover crops the sides.
        let wide = Some(4.0);
        let cover_w = arc_screen_mesh(&bg(Bulge::AwayFromCamera, BackgroundFit::Cover), 8, wide);
        let wu: Vec<f32> = cover_w.uvs.iter().map(|uv| uv[0]).collect();
        let wspan = 3.0f32 / 4.0;
        let wmin = wu.iter().cloned().fold(0. / 0., f32::min);
        let wmax = wu.iter().cloned().fold(0. / 0., f32::max);
        assert!((wmax - wmin - wspan).abs() < 1e-5, "cover u span {wspan}: {wmin}..{wmax}");
    }
}

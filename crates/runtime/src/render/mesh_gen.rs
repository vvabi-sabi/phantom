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
/// Map one box of the adult humanoid into a human role's proportions. The
/// figure is three vertical regions — legs (below the waist), torso, head —
/// and each scales by its own ratio from the role's profile, so a child is not
/// just a small adult. The adult row is the identity: every box keeps its
/// exact numbers.
fn map_humanoid_box(center: Vec3, half: Vec3, p: &phantom_core::rig::HumanProportions) -> (Vec3, Vec3) {
    const WAIST: f32 = 0.95;
    const TORSO_TOP: f32 = 1.52;
    let (center, half) = if center.y <= WAIST {
        (Vec3::new(center.x * p.leg, center.y * p.leg, center.z * p.leg), half * p.leg)
    } else if center.y <= TORSO_TOP {
        let y = p.waist + (center.y - WAIST) * p.torso;
        (Vec3::new(center.x * p.torso, y, center.z * p.torso), half * p.torso)
    } else {
        let y = p.torso_top + (center.y - TORSO_TOP) * p.head;
        (Vec3::new(center.x * p.head, y, center.z * p.head), half * p.head)
    };
    (center, half)
}

/// A stylized "robot" person; the art model is replaced by an FBX import. The
/// same drawing serves every human role — the boxes are mapped through the
/// role's proportions (the adult keeps its exact numbers).
pub fn builtin_humanoid_model(actor: BuiltinActor) -> ModelData {
    let p = phantom_core::rig::human_proportions(actor);
    let mut b = MeshBuilder::new();
    let mut box_ = |center: Vec3, half: Vec3, bone: usize| {
        let (c, h) = map_humanoid_box(center, half, &p);
        b.add_box(c, h, bone);
    };
    // Pelvis.
    box_(Vec3::new(0.0, 1.01, 0.0), Vec3::new(0.14, 0.07, 0.09), 0);
    // Torso.
    box_(Vec3::new(0.0, 1.30, 0.0), Vec3::new(0.17, 0.22, 0.10), 1);
    // Head.
    box_(Vec3::new(0.0, 1.63, 0.0), Vec3::new(0.10, 0.11, 0.10), 2);
    // Arms: upper arm + forearm, along X.
    for s in [1.0f32, -1.0] {
        let upper = if s > 0.0 { 3 } else { 5 };
        let lower = if s > 0.0 { 4 } else { 6 };
        box_(Vec3::new(s * 0.305, 1.45, 0.0), Vec3::new(0.125, 0.045, 0.045), upper);
        box_(Vec3::new(s * 0.555, 1.45, 0.0), Vec3::new(0.125, 0.04, 0.04), lower);
        // Thigh + shin + foot.
        let uleg = if s > 0.0 { 7 } else { 9 };
        let lleg = if s > 0.0 { 8 } else { 10 };
        box_(Vec3::new(s * 0.09, 0.70, 0.0), Vec3::new(0.055, 0.20, 0.055), uleg);
        box_(Vec3::new(s * 0.09, 0.275, 0.0), Vec3::new(0.05, 0.225, 0.05), lleg);
        box_(Vec3::new(s * 0.09, 0.03, 0.05), Vec3::new(0.05, 0.03, 0.10), lleg);
    }

    finish_model(b, actor)
}

/// The adult humanoid — kept as the named entry point of the default actor.
pub fn builtin_phantom_model() -> ModelData {
    builtin_humanoid_model(BuiltinActor::Humanoid)
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
    if actor.human_role().is_some() {
        return builtin_humanoid_model(actor);
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

/// Flat backdrop screen: a rectangle `width × height` centered on `bg.center`,
/// facing the camera side (+Z). UV mode — `bg.fit`: `stretch` — the picture is
/// stretched over the whole screen; `cover`/`contain` preserve the panorama
/// proportions from `image_aspect` (texture width/height; None — stretch).
pub fn screen_mesh(bg: &Background, image_aspect: Option<f32>) -> MeshData {
    use phantom_core::project::BackgroundFit;
    // UV span along each axis. Physical screen aspect = width / height.
    let (u_span, v_span) = match (bg.fit, image_aspect.filter(|a| a.is_finite() && *a > 0.0)) {
        (BackgroundFit::Stretch, _) | (_, None) => (1.0, 1.0),
        (BackgroundFit::Cover, Some(ai)) => {
            let screen = bg.width / bg.height;
            (if ai > screen { screen / ai } else { 1.0 }, if ai > screen { 1.0 } else { ai / screen })
        }
        (BackgroundFit::Contain, Some(ai)) => {
            // Whole image visible: the UV span goes OUTSIDE [0,1] along the wide
            // axis (screen edges sample the texture's clamped edges).
            let screen = bg.width / bg.height;
            (if ai > screen { 1.0 } else { screen / ai }, if ai > screen { ai / screen } else { 1.0 })
        }
    };
    let half_w = bg.width * 0.5;
    let half_h = bg.height * 0.5;
    let mut mesh = MeshData { positions: vec![], normals: vec![], uvs: vec![], indices: vec![] };
    for (x, y, u, v) in [
        (-half_w, -half_h, 0.5 - 0.5 * u_span, 0.5 + 0.5 * v_span),
        (half_w, -half_h, 0.5 + 0.5 * u_span, 0.5 + 0.5 * v_span),
        (half_w, half_h, 0.5 + 0.5 * u_span, 0.5 - 0.5 * v_span),
        (-half_w, half_h, 0.5 - 0.5 * u_span, 0.5 - 0.5 * v_span),
    ] {
        mesh.positions.push((bg.center + Vec3::new(x, y, 0.0)).to_array());
        mesh.normals.push([0.0, 0.0, 1.0]);
        mesh.uvs.push([u, v]);
    }
    // Counter-clockwise seen from +Z (the camera side).
    mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
    mesh
}

#[cfg(test)]
mod actor_tests {
    use super::*;
    #[test]
    fn bundled_meshes_have_valid_weights_bind_matrices_and_geometry() {
        for &actor in BuiltinActor::ALL {
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
            for &held in actor.poses() {
                let pose = phantom_core::rig::sample_actor_pose(actor, held).unwrap();
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
    use glam::Vec3;
    use phantom_core::project::{BackgroundFit, BackgroundSource};
    use phantom_core::AssetId;

    fn bg(fit: BackgroundFit) -> Background {
        Background {
            image: BackgroundSource::Image(AssetId(1)),
            // 12 x 4 m at z = -5: a 3.0 screen aspect, the same numbers the arc
            // screen had (2.0 rad * 6.0 m of arc length).
            width: 12.0,
            height: 4.0,
            center: Vec3::new(0.0, 2.0, -5.0),
            fit,
        }
    }

    /// The screen is flat: four corners, one plane, the documented size.
    #[test]
    fn the_screen_is_a_flat_rectangle_of_the_documented_size() {
        let mesh = screen_mesh(&bg(BackgroundFit::Stretch), None);
        assert_eq!(mesh.positions.len(), 4);
        assert_eq!(mesh.indices, vec![0, 1, 2, 0, 2, 3]);
        // Every corner sits on the screen plane z = center.z — no curvature.
        assert!(mesh.positions.iter().all(|p| (p[2] + 5.0).abs() < 1e-6), "{:?}", mesh.positions);
        let xs: Vec<f32> = mesh.positions.iter().map(|p| p[0]).collect();
        let ys: Vec<f32> = mesh.positions.iter().map(|p| p[1]).collect();
        assert_eq!(xs.iter().cloned().fold(f32::INFINITY, f32::min), -6.0);
        assert_eq!(xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max), 6.0);
        assert_eq!(ys.iter().cloned().fold(f32::INFINITY, f32::min), 0.0);
        assert_eq!(ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max), 4.0);
        // All normals face the camera side (+Z).
        assert!(mesh.normals.iter().all(|n| n[2] > 0.0 && n[0] == 0.0 && n[1] == 0.0));
    }

    #[test]
    fn fit_modes_shape_uv_spans() {
        // Screen aspect 3.0; panorama 16:9 ≈ 1.78.
        let image = Some(16.0 / 9.0);
        let stretch = screen_mesh(&bg(BackgroundFit::Stretch), image);
        let us: Vec<f32> = stretch.uvs.iter().map(|uv| uv[0]).collect();
        assert_eq!(us.iter().cloned().fold(0. / 0., f32::min), 0.0);
        assert_eq!(us.iter().cloned().fold(0. / 0., f32::max), 1.0);
        let vs: Vec<f32> = stretch.uvs.iter().map(|uv| uv[1]).collect();
        assert_eq!(vs.iter().cloned().fold(0. / 0., f32::min), 0.0);
        assert_eq!(vs.iter().cloned().fold(0. / 0., f32::max), 1.0);

        // cover: a 1.78 image on a 3.0 screen — top/bottom cropped (v span < 1),
        // the full range horizontally.
        let cover = screen_mesh(&bg(BackgroundFit::Cover), image);
        let cu: Vec<f32> = cover.uvs.iter().map(|uv| uv[0]).collect();
        assert_eq!(cu.iter().cloned().fold(0. / 0., f32::min), 0.0);
        assert_eq!(cu.iter().cloned().fold(0. / 0., f32::max), 1.0);
        let cv: Vec<f32> = cover.uvs.iter().map(|uv| uv[1]).collect();
        let vmin = cv.iter().cloned().fold(0. / 0., f32::min);
        let vmax = cv.iter().cloned().fold(0. / 0., f32::max);
        let span = (16.0 / 9.0f32) / 3.0;
        assert!((vmax - vmin - span).abs() < 1e-5, "cover v span {span}: {vmin}..{vmax}");
        assert!((vmin - (0.5 - span / 2.0)).abs() < 1e-5, "crop is symmetric");

        // contain: whole image visible — u goes outside [0,1] (side bars, the image
        // is narrower than the screen). A wide image — top/bottom bars (v).
        let contain = screen_mesh(&bg(BackgroundFit::Contain), image);
        let ku: Vec<f32> = contain.uvs.iter().map(|uv| uv[0]).collect();
        let kmin = ku.iter().cloned().fold(0. / 0., f32::min);
        let kmax = ku.iter().cloned().fold(0. / 0., f32::max);
        assert!(kmin < 0.0 && kmax > 1.0, "contain u goes outside the texture: {kmin}..{kmax}");
        let kv: Vec<f32> = contain.uvs.iter().map(|uv| uv[1]).collect();
        assert_eq!(kv.iter().cloned().fold(0. / 0., f32::min), 0.0);
        assert_eq!(kv.iter().cloned().fold(0. / 0., f32::max), 1.0);
        let contain_w = screen_mesh(&bg(BackgroundFit::Contain), Some(4.0));
        let wv: Vec<f32> = contain_w.uvs.iter().map(|uv| uv[1]).collect();
        assert!(wv.iter().cloned().fold(0. / 0., f32::min) < 0.0, "contain wide image: v < 0");

        // Wide 4:1 panorama on a 3.0 screen: cover crops the sides.
        let cover_w = screen_mesh(&bg(BackgroundFit::Cover), Some(4.0));
        let wu: Vec<f32> = cover_w.uvs.iter().map(|uv| uv[0]).collect();
        let wspan = 3.0f32 / 4.0;
        let wmin = wu.iter().cloned().fold(0. / 0., f32::min);
        let wmax = wu.iter().cloned().fold(0. / 0., f32::max);
        assert!((wmax - wmin - wspan).abs() < 1e-5, "cover u span {wspan}: {wmin}..{wmax}");
    }
}

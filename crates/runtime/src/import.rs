//! ufbx adapter: OBJ/FBX → internal model format.
//! Normalizes axes (right-handed, Y up) and units (meters).
//! Invalid data is not silently turned into a motionless model: a "character"
//! without skinning, or vertices without weights, is an error.

use std::collections::{HashMap, HashSet};
use std::path::{Component, Path, PathBuf};

use glam::{Mat4, Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    #[error("ufbx: {0}")]
    Ufbx(String),
    #[error("the file has no meshes")]
    NoMesh,
    #[error("skinning expected (FBX character), but there are no deformers")]
    NoSkin,
    #[error("too many influences on a vertex: {0} > 4")]
    TooManyInfluences(usize),
    #[error("unsupported file extension: {0}")]
    UnknownExtension(String),
    #[error("the file has no animation stacks")]
    NoClip,
    #[error("no rig bone name was found in the clip — the skeletons are incompatible")]
    NoMatchingBones,
    #[error("animation stack not found: {0}")]
    NoSuchStack(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a model file: {0}")]
    NotAFile(PathBuf),
    #[error("dependency path escapes the allowed root: {0}")]
    PathEscape(PathBuf),
    #[error("too many files in the import bundle: {0} > {MAX_BUNDLE_DEPS}")]
    TooManyDeps(usize),
    #[error("import bundle is too large: {0} bytes > {MAX_BUNDLE_BYTES}")]
    BundleTooLarge(u64),
    #[error("texture {path} exceeds the {MAX_TEXTURE_DIM}px limit: {w}x{h}")]
    TextureTooLarge { path: PathBuf, w: u32, h: u32 },
}

/// Normalized model — derived data, not serialized into the manifest.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelData {
    pub mesh: MeshData,
    /// Skeleton and weights; None for a static OBJ.
    pub skin: Option<SkinData>,
    /// PBR material table. Empty for files without materials and for old
    /// data: the render uses one default material (legacy path).
    #[serde(default)]
    pub materials: Vec<MaterialRecord>,
    /// Index groups per material: a hole-free, overlap-free partition of
    /// `mesh.indices`, in the same order as the indices. Empty for legacy models.
    #[serde(default)]
    pub parts: Vec<PartData>,
    /// Non-fatal import diagnostics (unsupported features and the like).
    #[serde(default)]
    pub diagnostics: Vec<String>,
}

/// PBR material of a mesh part (Bevy StandardMaterial metallic/roughness model).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaterialRecord {
    pub name: String,
    /// Base color factor, sRGB components 0..1 (as the author set it in the DCC).
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    /// Base color; decoded as sRGB.
    pub base_color_texture: Option<TextureRef>,
    /// Normal map; decoded as linear.
    pub normal_map_texture: Option<TextureRef>,
    /// Roughness map; linear, packed into the G channel.
    pub roughness_texture: Option<TextureRef>,
    /// Metalness map; linear, packed into the B channel.
    pub metalness_texture: Option<TextureRef>,
    /// Spec/gloss file: the roughness map stores glossiness, so invert the
    /// factor and the G channel when packing.
    #[serde(default)]
    pub invert_roughness: bool,
}

/// Reference to a texture file: path relative to the model file's directory
/// (as recorded in the source file, separators normalized to '/').
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextureRef {
    pub path: String,
}

/// A mesh part with a single material: an index range in the shared buffer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartData {
    /// Index into `ModelData.materials`.
    #[serde(default)]
    pub material: usize,
    /// Range `[start, start + count)` in `MeshData.indices`.
    pub start: u32,
    pub count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
}

/// Animation clip: bone local transforms sampled at 24 fps.
#[derive(Clone, Debug)]
pub struct ClipData {
    pub fps: f32,
    pub frames: u32,
    /// Bone names in `samples` order.
    pub bones: Vec<String>,
    /// samples[bone][frame] = (translation, rotation), local.
    pub samples: Vec<Vec<(Vec3, Quat)>>,
    /// How many rig names were found in the clip (compatibility diagnostics).
    pub matched: usize,
}

/// Load an FBX animation stack as a clip. `stack` is the stack name;
/// `None` — the longest one (legacy behavior).
/// Compatibility is checked against the rig bone names after normalization;
/// a complete mismatch is an error, not a silently motionless model.
pub fn load_clip(path: &Path, rig_bones: &[String], stack_name: Option<&str>) -> Result<ClipData, ImportError> {
    let scene = load_anim_scene(path)?;
    let node_names: Vec<String> = scene.nodes.iter().map(|n| n.element.name.to_string()).collect();
    // Direct name match: a track is a rig bone found in the file.
    let pairs: Vec<(String, String)> = rig_bones.iter().filter(|b| node_names.contains(b)).map(|b| (b.clone(), b.clone())).collect();
    sample_pairs(&scene, stack_name, pairs)
}

/// Load a clip with profile-based retargeting: source bones are located in the
/// file by the profile's role patterns, and samples are carried over onto the
/// target rig's bone names. `ClipData.bones` tracks are TARGET-rig names (the
/// bone→track map in the render is built by the shared code).
///
/// Unmatched roles on either side are non-fatal diagnostics (the second result
/// element); the clip plays the matched subset. Zero coverage is
/// `NoMatchingBones`, as with direct name matching. Limitation: rotations and
/// translations transfer directly (assuming matching bone orientation); there
/// is no rest-pose or scale correction.
pub fn load_clip_retarget(
    path: &Path,
    profile: phantom_core::RetargetProfile,
    target_bones: &[String],
    stack_name: Option<&str>,
) -> Result<(ClipData, Vec<String>), ImportError> {
    let scene = load_anim_scene(path)?;
    let node_names: Vec<String> = scene.nodes.iter().map(|n| n.element.name.to_string()).collect();
    let map = phantom_core::retarget::build_map(profile, &node_names, target_bones);
    if map.matched() == 0 {
        return Err(ImportError::NoMatchingBones);
    }
    let pairs: Vec<(String, String)> = map
        .mapped
        .iter()
        .map(|m| (node_names[m.source].clone(), target_bones[m.target].clone()))
        .collect();
    let clip = sample_pairs(&scene, stack_name, pairs)?;
    Ok((clip, map.diagnostics()))
}

/// FBX scene for reading animation (no geometry, with axis/unit normalization).
fn load_anim_scene(path: &Path) -> Result<ufbx::SceneRoot, ImportError> {
    let opts = ufbx::LoadOpts {
        target_axes: ufbx::CoordinateAxes::right_handed_y_up(),
        target_unit_meters: 1.0,
        space_conversion: ufbx::SpaceConversion::AdjustTransforms,
        ignore_geometry: true,
        ..Default::default()
    };
    ufbx::load_file(&path.to_string_lossy(), opts).map_err(|e| ImportError::Ufbx(format!("{e:?}")))
}

/// Sample the (file node name, clip track name) pairs at 24 fps.
/// Stack order: the stack is picked before the coverage check (as in the legacy path).
fn sample_pairs(
    scene: &ufbx::Scene,
    stack_name: Option<&str>,
    pairs: Vec<(String, String)>,
) -> Result<ClipData, ImportError> {
    let stack = match stack_name {
        Some(name) => scene
            .anim_stacks
            .iter()
            .find(|s| s.element.name.as_ref() == name)
            .ok_or_else(|| ImportError::NoSuchStack(name.to_string())),
        None => scene
            .anim_stacks
            .iter()
            .max_by(|a, b| (a.time_end - a.time_begin).partial_cmp(&(b.time_end - b.time_begin)).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or(ImportError::NoClip),
    }?;
    let stack = stack;
    const FPS: f32 = 24.0;
    let duration = (stack.time_end - stack.time_begin) as f32;
    let frames = (duration * FPS).ceil().max(1.0) as u32;

    // Source node name → node; the track is stored under the output name.
    let mut matched = 0;
    let mut bones = Vec::new();
    let mut samples = Vec::new();
    for (source_name, track_name) in pairs {
        let node = scene.nodes.iter().find(|n| n.element.name.as_ref() == source_name.as_str());
        let Some(node) = node else { continue };
        matched += 1;
        bones.push(track_name);
        let mut track = Vec::with_capacity(frames as usize);
        for f in 0..frames {
            let t = stack.time_begin + f as f64 / FPS as f64;
            let tr = node.evaluate_transform(&stack.anim, t);
            track.push((v3(tr.translation), q4v(tr.rotation)));
        }
        samples.push(track);
    }
    if matched == 0 {
        return Err(ImportError::NoMatchingBones);
    }
    Ok(ClipData { fps: FPS, frames, bones, samples, matched })
}

/// Cheap listing of FBX animation stacks: name and duration in seconds.
/// For MCP `clip_list` / picking a stack by name; no sampling is done.
pub fn list_stacks(path: &Path) -> Result<Vec<StackInfo>, ImportError> {
    let opts = ufbx::LoadOpts { ignore_geometry: true, ..Default::default() };
    let scene = ufbx::load_file(&path.to_string_lossy(), opts)
        .map_err(|e| ImportError::Ufbx(format!("{e:?}")))?;
    Ok(scene
        .anim_stacks
        .iter()
        .map(|s| StackInfo { name: s.element.name.to_string(), duration_secs: (s.time_end - s.time_begin) as f32 })
        .collect())
}

/// Description of an FBX animation stack.
#[derive(Clone, Debug)]
pub struct StackInfo {
    pub name: String,
    pub duration_secs: f32,
}

fn q4v(q: ufbx::Quat) -> Quat {
    Quat::from_xyzw(q.x as f32, q.y as f32, q.z as f32, q.w as f32)
}

#[derive(Default)]
struct SkinBuild {
    bone_ids: Vec<u32>,
    bones: Vec<BoneData>,
    joint_indices: Vec<[u16; 4]>,
    joint_weights: Vec<[f32; 4]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoneData {
    pub name: String,
    pub parent: Option<usize>,
    /// Inverse bind matrix (geometry→bone in the bind pose).
    pub inverse_bind: [[f32; 4]; 4],
    /// Rest transform relative to the parent.
    pub rest_translation: [f32; 3],
    pub rest_rotation: [f32; 4],
    /// Rest scale (Mixamo has root = 0.01, cm→m).
    #[serde(default = "one3")]
    pub rest_scale: [f32; 3],
}

fn one3() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SkinData {
    pub bones: Vec<BoneData>,
    /// 4 bone indices per vertex.
    pub joint_indices: Vec<[u16; 4]>,
    /// 4 normalized weights per vertex.
    pub joint_weights: Vec<[f32; 4]>,
}

/// Load an OBJ or FBX. For FBX with `require_skin`, check that a skeleton exists.
/// glTF/glb is deliberately rejected: the current ufbx decoder cannot read it.
pub fn load_model(path: &Path, require_skin: bool) -> Result<ModelData, ImportError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "obj" | "fbx" => {}
        other => return Err(ImportError::UnknownExtension(other.to_string())),
    }

    let opts = ufbx::LoadOpts {
        target_axes: ufbx::CoordinateAxes::right_handed_y_up(),
        target_unit_meters: 1.0,
        space_conversion: ufbx::SpaceConversion::AdjustTransforms,
        geometry_transform_handling: ufbx::GeometryTransformHandling::HelperNodes,
        generate_missing_normals: true,
        clean_skin_weights: true,
        // Read MTL/external files: OBJ materials do not parse without them.
        // Missing ones are a warning, not a rejection; dependency paths are
        // authorized by the bundle import, not by ufbx.
        load_external_files: true,
        ignore_missing_external_files: true,
        ..Default::default()
    };
    let scene = ufbx::load_file(&path.to_string_lossy(), opts)
        .map_err(|e| ImportError::Ufbx(format!("{e:?}")))?;

    // Collect the meshes of all nodes into one; node transforms are baked into
    // the vertices (statics) or used as the bind pose (skinning).
    let mut mesh = MeshData { positions: vec![], normals: vec![], uvs: vec![], indices: vec![] };
    let mut skin: Option<SkinData> = None;
    // The skeleton is built globally across all skinned meshes of the scene.
    let mut skin_state: Option<SkinBuild> = None;
    // Materials: a global table (dedup by element_id) and index groups.
    let mut materials: Vec<MaterialRecord> = Vec::new();
    let mut mat_map: HashMap<u32, usize> = HashMap::new();
    let mut parts: Vec<PartData> = Vec::new();
    let mut diagnostics: Vec<String> = Vec::new();
    const NO_SLOT: u32 = u32::MAX;

    for node in scene.nodes.iter() {
        let Some(m) = node.mesh.as_ref() else { continue };
        let base = mesh.positions.len() as u32;
        // Vertices are baked through geometry_to_world: node meshes can live in
        // different spaces (the head in its local space) while inverse_bind is
        // unique per skeleton — so the single mesh space is the bind-pose world.
        let has_skin = !m.skin_deformers.is_empty();
        let to_world: Mat4 = mat_from(&node.geometry_to_world);

        // Vertex attributes: expand by indices (index → vertex).
        for i in 0..m.num_indices {
            let p = v3(m.vertex_position[i]);
            let p = to_world.transform_point3(p);
            mesh.positions.push(p.to_array());
            let n = if m.vertex_normal.exists { v3(m.vertex_normal[i]) } else { Vec3::Y };
            let n = to_world.transform_vector3(n).try_normalize().unwrap_or(Vec3::Y);
            mesh.normals.push(n.to_array());
            let uv = if m.vertex_uv.exists { v2(m.vertex_uv[i]) } else { Vec2::ZERO };
            mesh.uvs.push(uv.to_array());
        }

        // Face triangulation grouped by material slot: parts must be
        // contiguous index ranges. Group order follows the slot's first
        // appearance; within a group, the face order is preserved.
        let slots: &[ufbx::Ref<ufbx::Material>] =
            if !m.materials.is_empty() { &m.materials } else { &node.materials };
        let use_slots = !slots.is_empty() && !m.face_material.is_empty();
        let mut tri: Vec<u32> = Vec::new();
        let mut bucket_order: Vec<u32> = Vec::new();
        let mut buckets: HashMap<u32, Vec<u32>> = HashMap::new();
        for (fi, face) in m.faces.iter().enumerate() {
            let slot = if use_slots {
                match m.face_material.get(fi).copied() {
                    Some(s) if (s as usize) < slots.len() => s,
                    _ => NO_SLOT,
                }
            } else {
                NO_SLOT
            };
            tri.clear();
            // ufbx 0.4's helper returns an empty vector for an already
            // triangular face. OBJ/FBX exports commonly are triangles, so
            // preserve their three expanded vertex indices directly.
            match face.num_indices {
                3 => tri.extend(face.index_begin..face.index_begin + 3),
                n if n > 3 => {
                    ufbx::triangulate_face_vec(&mut tri, m, *face);
                }
                _ => {}
            };
            if !buckets.contains_key(&slot) {
                bucket_order.push(slot);
                buckets.insert(slot, Vec::new());
            }
            buckets.get_mut(&slot).expect("bucket").extend(tri.iter().map(|&i| base + i));
        }
        for slot in bucket_order {
            let idxs = buckets.remove(&slot).expect("bucket");
            let start = mesh.indices.len() as u32;
            let count = idxs.len() as u32;
            mesh.indices.extend(idxs);
            let material = if slot == NO_SLOT {
                None
            } else {
                let mat: &ufbx::Material = &slots[slot as usize];
                let eid = mat.element.element_id;
                Some(*mat_map.entry(eid).or_insert_with(|| {
                    materials.push(extract_material(mat, &mut diagnostics));
                    materials.len() - 1
                }))
            };
            parts.push(PartData { material: material.unwrap_or(usize::MAX), start, count });
        }

        if has_skin {
            let deform = &m.skin_deformers[0];
            // Bone node.element_id → bone index (global across the node's
            // meshes: a character has one skeleton spread over several meshes).
            let state = skin_state.get_or_insert_with(SkinBuild::default);
            for cluster in deform.clusters.iter() {
                let Some(bone_node) = cluster.bone_node.as_ref() else { continue };
                let eid = bone_node.element.element_id;
                if !state.bone_ids.contains(&eid) {
                    state.bone_ids.push(eid);
                    // bind_to_world: the bone's world bind pose; inverting it
                    // gives inverse_bindposes (bind world → bone).
                    state.bones.push(BoneData {
                        name: bone_node.element.name.to_string(),
                        parent: None, // filled in after all meshes are collected
                        inverse_bind: mat_to_array(&mat_from(&cluster.bind_to_world).inverse()),
                        rest_translation: v3(bone_node.local_transform.translation).to_array(),
                        rest_rotation: q4(bone_node.local_transform.rotation),
                        rest_scale: [1.0, 1.0, 1.0],
                    });
                }
            }
            // Weights: skin.vertices are indexed by the mesh's "vertex index"
            // (canonical), but we expanded by face-index — so a map is needed.
            let state = skin_state.as_mut().expect("skin build");
            let num_v = m.num_vertices;
            let mut ji = vec![[0u16; 4]; num_v];
            let mut jw = vec![[0.0f32; 4]; num_v];
            for vi in 0..num_v {
                let sv = deform.vertices[vi];
                let begin = sv.weight_begin as usize;
                let count = sv.num_weights as usize;
                if count > 4 {
                    return Err(ImportError::TooManyInfluences(count));
                }
                let mut total = 0.0f32;
                for w in 0..count {
                    let sw = &deform.weights[begin + w];
                    let Some(bi) = state
                        .bone_ids
                        .iter()
                        .position(|e| {
                            deform.clusters[sw.cluster_index as usize]
                                .bone_node
                                .as_ref()
                                .map(|b| b.element.element_id)
                                == Some(*e)
                        })
                        else { continue };
                    ji[vi][w] = bi as u16;
                    jw[vi][w] = sw.weight as f32;
                    total += sw.weight as f32;
                }
                if total > 0.0 {
                    for w in jw[vi].iter_mut() {
                        *w /= total;
                    }
                } else {
                    jw[vi][0] = 1.0;
                }
            }
            // Move the weights from canonical vertices onto the expanded ones.
            for i in 0..m.num_indices {
                let vi = m.vertex_indices[i] as usize;
                state.joint_indices.push(ji[vi]);
                state.joint_weights.push(jw[vi]);
            }
        }
    }

    if mesh.positions.is_empty() {
        return Err(ImportError::NoMesh);
    }
    if let Some(mut state) = skin_state {
        // Bone parents from the node hierarchy.
        for bi in 0..state.bones.len() {
            let eid = state.bone_ids[bi];
            let node = scene
                .nodes
                .iter()
                .find(|n| n.element.element_id == eid)
                .expect("bone node in scene");
            let mut p = node.parent.as_ref();
            while let Some(par) = p {
                if let Some(pi) = state.bone_ids.iter().position(|e| *e == par.element.element_id) {
                    state.bones[bi].parent = Some(pi);
                    break;
                }
                p = par.parent.as_ref();
            }
        }
        // Rest locals come from the bind matrices, not from the nodes' current
        // transforms: global = inverse(inverse_bind), local =
        // inverse(global_parent) * global. Otherwise an FBX with a "built-in"
        // root orientation (a typical -90° X) renders lying down.
        let globals: Vec<Mat4> = state
            .bones
            .iter()
            .map(|b| Mat4::from_cols_array_2d(&b.inverse_bind).inverse())
            .collect();
        for (bi, bone) in state.bones.iter_mut().enumerate() {
            let local = match bone.parent {
                Some(p) => globals[p].inverse() * globals[bi],
                None => globals[bi],
            };
            let (scale, rot, trans) = local.to_scale_rotation_translation();
            bone.rest_translation = trans.to_array();
            bone.rest_rotation = rot.to_array();
            bone.rest_scale = scale.to_array();
        }
        skin = Some(SkinData { bones: state.bones, joint_indices: state.joint_indices, joint_weights: state.joint_weights });
    }
    if require_skin && skin.is_none() {
        return Err(ImportError::NoSkin);
    }
    // A file without materials needs no parts — legacy path with one material.
    // A file with materials but meshes without a slot: add an explicit default
    // material so the partition stays complete.
    if materials.is_empty() {
        parts.clear();
    } else if parts.iter().any(|p| p.material == usize::MAX) {
        let default_idx = materials.len();
        materials.push(MaterialRecord {
            name: "default".into(),
            base_color: [0.8, 0.8, 0.8, 1.0],
            metallic: 0.0,
            roughness: 0.9,
            base_color_texture: None,
            normal_map_texture: None,
            roughness_texture: None,
            metalness_texture: None,
            invert_roughness: false,
        });
        for part in parts.iter_mut() {
            if part.material == usize::MAX {
                part.material = default_idx;
            }
        }
    }
    Ok(ModelData { mesh, skin, materials, parts, diagnostics })
}

/// Extract a PBR record from a ufbx material: pbr fields first, then the
/// fbx fallback. Unsupported features (layered/procedural textures, UV
/// transforms, heightmap bump) produce explicit diagnostics, not a silent
/// white mesh.
fn extract_material(mat: &ufbx::Material, diagnostics: &mut Vec<String>) -> MaterialRecord {
    let name = mat.element.name.to_string();
    let pbr = &mat.pbr;
    let fbx = &mat.fbx;
    let invert_roughness = mat.features.roughness_as_glossiness.enabled;

    let v4 = |m: &ufbx::MaterialMap| (m.value_vec4.x as f32, m.value_vec4.y as f32, m.value_vec4.z as f32, m.value_vec4.w as f32);
    let (r, g, b, a) = if pbr.base_color.has_value {
        v4(&pbr.base_color)
    } else if fbx.diffuse_color.has_value {
        let (r, g, b, _) = v4(&fbx.diffuse_color);
        (r, g, b, 1.0)
    } else {
        (0.8, 0.8, 0.8, 1.0)
    };
    let factor = if pbr.base_factor.has_value {
        pbr.base_factor.value_vec4.x as f32
    } else if fbx.diffuse_factor.has_value {
        fbx.diffuse_factor.value_vec4.x as f32
    } else {
        1.0
    };
    let c = |v: f32| (v * factor).clamp(0.0, 1.0);
    let base_color = [c(r), c(g), c(b), a.clamp(0.0, 1.0)];

    let mut roughness = 0.5;
    if pbr.roughness.has_value {
        let v = (pbr.roughness.value_vec4.x as f32).clamp(0.0, 1.0);
        roughness = if invert_roughness { 1.0 - v } else { v };
    }
    let metallic = if pbr.metalness.has_value { (pbr.metalness.value_vec4.x as f32).clamp(0.0, 1.0) } else { 0.0 };

    let mut tex = |map: &ufbx::MaterialMap, role: &str| -> Option<TextureRef> {
        let t = map.texture.as_ref()?;
        if t.type_ != ufbx::TextureType::File {
            diagnostics.push(format!(
                "material \"{name}\": {role}: layered/procedural textures are not supported, using the factor"
            ));
            return None;
        }
        if t.has_uv_transform {
            diagnostics.push(format!(
                "material \"{name}\": {role}: texture UV transform is not supported, the texture is applied without it"
            ));
        }
        let path = texture_path(t)?;
        Some(TextureRef { path })
    };
    let base_color_texture = tex(&pbr.base_color, "base color").or_else(|| tex(&fbx.diffuse_color, "base color"));
    let normal_map_texture = tex(&pbr.normal_map, "normal map").or_else(|| tex(&fbx.normal_map, "normal map"));
    let roughness_texture = tex(&pbr.roughness, "roughness");
    let metalness_texture = tex(&pbr.metalness, "metalness");
    if normal_map_texture.is_none() && fbx.bump.texture.is_some() {
        diagnostics.push(format!(
            "material \"{name}\": heightmap bump is not a normal map and is not supported — relief skipped"
        ));
    }

    MaterialRecord {
        name,
        base_color,
        metallic,
        roughness,
        base_color_texture,
        normal_map_texture,
        roughness_texture,
        metalness_texture,
        invert_roughness,
    }
}

/// Texture path as written in the model file: relative, otherwise as recorded.
fn texture_path(t: &ufbx::Texture) -> Option<String> {
    let p: &str = if !t.relative_filename.is_empty() {
        t.relative_filename.as_ref()
    } else if !t.filename.is_empty() {
        t.filename.as_ref()
    } else {
        t.absolute_filename.as_ref()
    };
    if p.is_empty() { None } else { Some(p.replace('\\', "/")) }
}

/// ufbx Matrix is 3×4, columns (m00,m10,m20) ... (m03,m13,m23);
/// the last column is the translation.
fn mat_from(m: &ufbx::Matrix) -> Mat4 {
    Mat4::from_cols_array(&[
        m.m00 as f32, m.m10 as f32, m.m20 as f32, 0.0,
        m.m01 as f32, m.m11 as f32, m.m21 as f32, 0.0,
        m.m02 as f32, m.m12 as f32, m.m22 as f32, 0.0,
        m.m03 as f32, m.m13 as f32, m.m23 as f32, 1.0,
    ])
}

fn mat_to_array(m: &Mat4) -> [[f32; 4]; 4] {
    m.to_cols_array_2d()
}

fn v3(v: ufbx::Vec3) -> Vec3 {
    Vec3::new(v.x as f32, v.y as f32, v.z as f32)
}

fn v2(v: ufbx::Vec2) -> Vec2 {
    Vec2::new(v.x as f32, v.y as f32)
}

fn q4(q: ufbx::Quat) -> [f32; 4] {
    [q.x as f32, q.y as f32, q.z as f32, q.w as f32]
}

// ---------------------------------------------------------------------------
// Bundle import: model + MTL + textures into the project's own directory.
// ---------------------------------------------------------------------------

/// Result of a bundle model import.
#[derive(Debug)]
pub struct BundleImport {
    /// Path of the main model file relative to the project root.
    pub model: PathBuf,
    /// Copied dependencies (paths relative to the project root).
    pub dependencies: Vec<PathBuf>,
    /// Non-fatal diagnostics (skipped/unsupported dependencies).
    pub diagnostics: Vec<String>,
}

/// Maximum number of dependency files in a bundle.
pub const MAX_BUNDLE_DEPS: usize = 64;
/// Total bundle byte limit (model + dependencies).
pub const MAX_BUNDLE_BYTES: u64 = 256 * 1024 * 1024;
/// Maximum texture size along any side.
pub const MAX_TEXTURE_DIM: u32 = 8192;

/// Import an OBJ/FBX together with its dependencies (MTL, textures) into the
/// project: `assets/<name>/…` keeping the relative layout, so re-parsing from
/// the bundle resolves the same paths. Each dependency path is canonicalized
/// and must stay inside the source directory; an escape (via `..` or a symlink)
/// rejects the import. Source files are not modified.
pub fn import_model_bundle(project_dir: &Path, source: &Path) -> Result<BundleImport, ImportError> {
    let ext = source
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "obj" | "fbx" => {}
        other => return Err(ImportError::UnknownExtension(other.to_string())),
    }
    let source_abs = source.canonicalize()?;
    if !source_abs.is_file() {
        return Err(ImportError::NotAFile(source.to_path_buf()));
    }
    let source_root = source_abs
        .parent()
        .ok_or_else(|| ImportError::PathEscape(source_abs.clone()))?
        .to_path_buf();
    let file_name = source_abs
        .file_name()
        .ok_or_else(|| ImportError::PathEscape(source_abs.clone()))?
        .to_os_string();
    let stem = source_abs.file_stem().and_then(|s| s.to_str()).unwrap_or("model").to_string();

    // Bundle directory without collisions: <stem>, <stem>-1, <stem>-2, …
    let mut bundle_rel = PathBuf::from(crate::storage::ASSETS_DIR).join(&stem);
    for i in 1u32.. {
        if !project_dir.join(&bundle_rel).exists() {
            break;
        }
        bundle_rel = PathBuf::from(crate::storage::ASSETS_DIR).join(format!("{stem}-{i}"));
    }
    let bundle_abs = project_dir.join(&bundle_rel);
    std::fs::create_dir_all(&bundle_abs)?;

    let result = import_bundle_inner(project_dir, &source_abs, &source_root, &file_name, &ext, &bundle_rel, &bundle_abs);
    if result.is_err() {
        // Do not leave a half-built bundle in the project.
        let _ = std::fs::remove_dir_all(&bundle_abs);
    }
    result
}

fn import_bundle_inner(
    _project_dir: &Path,
    source_abs: &Path,
    source_root: &Path,
    file_name: &std::ffi::OsStr,
    ext: &str,
    bundle_rel: &Path,
    bundle_abs: &Path,
) -> Result<BundleImport, ImportError> {
    let mut total: u64 = 0;
    let mut dependencies: Vec<PathBuf> = Vec::new();
    let mut diagnostics: Vec<String> = Vec::new();
    let mut dests: HashSet<PathBuf> = HashSet::new();

    // The main model file.
    copy_bounded(source_abs, &bundle_abs.join(file_name), &mut total)?;
    let model_rel = bundle_rel.join(file_name);

    let mut add_dep =
        |dest_rel: PathBuf, copy: &mut dyn FnMut(&Path) -> Result<(), ImportError>| -> Result<(), ImportError> {
        if dests.contains(&dest_rel) {
            return Ok(());
        }
        if dependencies.len() >= MAX_BUNDLE_DEPS {
            return Err(ImportError::TooManyDeps(dependencies.len() + 1));
        }
        let dest = bundle_abs.join(&dest_rel);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        copy(&dest)?;
        dests.insert(dest_rel.clone());
        dependencies.push(bundle_rel.join(&dest_rel));
        Ok(())
    };

    // OBJ: MTL files are listed by mtllib lines in the file text.
    if ext == "obj" {
        for mtl in scan_mtllib(source_abs) {
            let rel = safe_relative(&mtl).ok_or_else(|| ImportError::PathEscape(PathBuf::from(&mtl)))?;
            let canon = match source_root.join(&rel).canonicalize() {
                Ok(c) if c.is_file() => c,
                _ => {
                    diagnostics.push(format!("mtllib not found: {mtl}"));
                    continue;
                }
            };
            if !canon.starts_with(source_root) {
                return Err(ImportError::PathEscape(canon));
            }
            add_dep(rel, &mut |dest| copy_bounded(&canon, dest, &mut total))?;
        }
    }

    // Textures: enumerated through ufbx (a shared path for OBJ and FBX).
    // mtllib entries were authorized above, before this read.
    let opts = ufbx::LoadOpts { load_external_files: true, ignore_missing_external_files: true, ..Default::default() };
    let scene = ufbx::load_file(&source_abs.to_string_lossy(), opts)
        .map_err(|e| ImportError::Ufbx(format!("{e:?}")))?;
    for texture in scene.textures.iter() {
        if texture.type_ != ufbx::TextureType::File {
            diagnostics.push(format!(
                "texture \"{}\": layered/procedural is not supported, skipped",
                texture.element.name
            ));
            continue;
        }
        let Some(rel_str) = texture_path(texture) else { continue };
        let candidate = if !texture.absolute_filename.is_empty() {
            PathBuf::from(texture.absolute_filename.replace('\\', "/"))
        } else {
            source_root.join(rel_str.replace('\\', "/"))
        };
        match candidate.canonicalize() {
            Ok(canon) if canon.is_file() => {
                if !canon.starts_with(source_root) {
                    return Err(ImportError::PathEscape(canon));
                }
                let rel = safe_relative(&rel_str).ok_or_else(|| ImportError::PathEscape(PathBuf::from(&rel_str)))?;
                if !check_texture_dims(&canon, || image::image_dimensions(&canon).map_err(|_| ()), &mut diagnostics)? {
                    continue; // unknown format: the diagnostic is already recorded
                }
                add_dep(rel, &mut |dest| copy_bounded(&canon, dest, &mut total))?;
            }
            // NB: a Blob with size=0 has a null pointer — Deref is not allowed,
            // check the size directly.
            _ if texture.content.size > 0 => {
                // Texture embedded in the FBX: write the contents to the bundle root.
                let bytes: &[u8] = &texture.content;
                let parsed = image::ImageReader::new(std::io::Cursor::new(bytes))
                    .with_guessed_format()
                    .ok()
                    .and_then(|r| r.into_dimensions().ok());
                let Some((w, h)) = parsed else {
                    diagnostics.push(format!("embedded texture \"{rel_str}\": unknown format, skipped"));
                    continue;
                };
                if w > MAX_TEXTURE_DIM || h > MAX_TEXTURE_DIM {
                    return Err(ImportError::TextureTooLarge { path: PathBuf::from(&rel_str), w, h });
                }
                let name = Path::new(&rel_str)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("embedded.png")
                    .to_string();
                // add_dep writes the file synchronously, so a disk check is
                // enough for name uniqueness.
                let mut rel = PathBuf::from(&name);
                for i in 1u32.. {
                    if !bundle_abs.join(&rel).exists() {
                        break;
                    }
                    rel = PathBuf::from(format!("{i}-{name}"));
                }
                add_dep(rel, &mut |dest| {
                    total += bytes.len() as u64;
                    if total > MAX_BUNDLE_BYTES {
                        return Err(ImportError::BundleTooLarge(total));
                    }
                    std::fs::write(dest, bytes)?;
                    Ok(())
                })?;
            }
            _ => diagnostics.push(format!("texture not found and not embedded: {rel_str}")),
        }
    }

    Ok(BundleImport { model: model_rel, dependencies, diagnostics })
}

/// Check the texture dimensions; an unreadable format is a diagnostic and a
/// skip (the file is not copied: the renderer will not need it anyway).
fn check_texture_dims(
    path: &Path,
    dims: impl FnOnce() -> Result<(u32, u32), ()>,
    diagnostics: &mut Vec<String>,
) -> Result<bool, ImportError> {
    match dims() {
        Ok((w, h)) => {
            if w > MAX_TEXTURE_DIM || h > MAX_TEXTURE_DIM {
                return Err(ImportError::TextureTooLarge { path: path.to_path_buf(), w, h });
            }
            Ok(true)
        }
        Err(()) => {
            diagnostics.push(format!("texture of unknown format, skipped: {}", path.display()));
            Ok(false)
        }
    }
}

/// Copy a file while respecting the total bundle byte limit.
fn copy_bounded(src: &Path, dest: &Path, total: &mut u64) -> Result<(), ImportError> {
    let bytes = std::fs::copy(src, dest)?;
    *total += bytes;
    if *total > MAX_BUNDLE_BYTES {
        return Err(ImportError::BundleTooLarge(*total));
    }
    Ok(())
}

/// Relative path that does not escape the root: plain components only.
fn safe_relative(rel: &str) -> Option<PathBuf> {
    let p = PathBuf::from(rel);
    if p.is_absolute() || p.components().any(|c| !matches!(c, Component::Normal(_) | Component::CurDir)) {
        None
    } else {
        Some(p)
    }
}

/// OBJ `mtllib` lines: material file names, as written.
fn scan_mtllib(obj: &Path) -> Vec<String> {
    let Ok(bytes) = std::fs::read(obj) else { return Vec::new() };
    let text = String::from_utf8_lossy(&bytes);
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("mtllib") {
            out.extend(rest.split_whitespace().map(str::to_string));
        }
    }
    out
}

/// Pack metallic/roughness into one RGBA texture for Bevy StandardMaterial:
/// G = roughness, B = metallic, R/A = 255 (neutral: the factors multiply by 1).
/// The caller handles the case where both maps share a path (glTF layout).
pub fn pack_metallic_roughness(
    roughness: Option<&image::RgbaImage>,
    metalness: Option<&image::RgbaImage>,
    invert_roughness: bool,
) -> Option<image::RgbaImage> {
    let (w, h) = match (roughness, metalness) {
        (Some(r), _) => r.dimensions(),
        (None, Some(m)) => m.dimensions(),
        (None, None) => return None,
    };
    let metal_resized;
    let metalness = match metalness {
        Some(m) if m.dimensions() != (w, h) => {
            metal_resized = image::imageops::resize(m, w, h, image::imageops::FilterType::Triangle);
            Some(&metal_resized)
        }
        other => other,
    };
    let mut out = image::RgbaImage::from_pixel(w, h, image::Rgba([255, 255, 255, 255]));
    for (x, y, px) in out.enumerate_pixels_mut() {
        if let Some(r) = roughness {
            let v = r.get_pixel(x, y).0[0];
            px.0[1] = if invert_roughness { 255 - v } else { v };
        }
        if let Some(m) = metalness {
            px.0[2] = m.get_pixel(x, y).0[0];
        }
    }
    Some(out)
}

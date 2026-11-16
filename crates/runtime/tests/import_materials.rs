//! P0 materials: OBJ+MTL with two materials and generated textures
//! (no third-party assets). Coverage: index partitioning, texture roles,
//! tangents, bundle import with a round-trip after the source is deleted,
//! traversal/symlink dependency rejection, legacy ModelData.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use image::RgbaImage;
use phantom_runtime::import::{import_model_bundle, load_model, pack_metallic_roughness, ImportError, ModelData};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "phantom-mat-{}-{}-{tag}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn write_png(path: &Path, rgba: [u8; 4]) {
    RgbaImage::from_pixel(4, 4, image::Rgba(rgba)).save(path).unwrap();
}

const TWO_QUADS_OBJ: &str = "\
mtllib two.mtl
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 2 0 0
v 3 0 0
v 3 1 0
v 2 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 1
usemtl red
f 1/1/1 2/2/1 3/3/1 4/4/1
usemtl green
f 5/5/1 6/6/1 7/7/1 8/8/1
";

const TWO_MATS_MTL: &str = "\
newmtl red
Kd 1.0 0.0 0.0
map_Kd red.png
Pr 0.25
Pm 0.75
newmtl green
Kd 0.0 1.0 0.0
map_Kd green.png
norm normal.png
";

/// Source model: two quads, two materials, three textures.
fn write_source(dir: &Path) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("two.obj"), TWO_QUADS_OBJ).unwrap();
    std::fs::write(dir.join("two.mtl"), TWO_MATS_MTL).unwrap();
    write_png(&dir.join("red.png"), [255, 0, 0, 255]);
    write_png(&dir.join("green.png"), [0, 255, 0, 255]);
    write_png(&dir.join("normal.png"), [128, 128, 255, 255]);
    dir.join("two.obj")
}

#[test]
fn two_material_obj_partitions_indices_and_preserves_materials() {
    let tmp = TempDir::new("parts");
    let obj = write_source(tmp.path());
    let model = load_model(&obj, false).expect("OBJ with MTL imports");

    assert_eq!(model.materials.len(), 2, "red/green materials");
    assert_eq!(model.parts.len(), 2, "one part per material");
    assert!(model.skin.is_none());

    // Partition: the parts cover every index with no gaps or overlaps.
    let mut ranges: Vec<(u32, u32)> = model.parts.iter().map(|p| (p.start, p.start + p.count)).collect();
    ranges.sort();
    let mut cursor = 0;
    for (s, e) in &ranges {
        assert_eq!(*s, cursor, "gap or overlap in the partition");
        cursor = *e;
    }
    assert_eq!(cursor as usize, model.mesh.indices.len());

    // Materials: Kd colors, Pr/Pm factors, texture roles.
    let red = model.materials.iter().find(|m| m.name == "red").expect("red");
    assert!((red.base_color[0] - 1.0).abs() < 1e-4 && red.base_color[1].abs() < 1e-4, "{:?}", red.base_color);
    assert_eq!(red.base_color_texture.as_ref().unwrap().path, "red.png");
    assert!((red.roughness - 0.25).abs() < 1e-4, "Pr: {}", red.roughness);
    assert!((red.metallic - 0.75).abs() < 1e-4, "Pm: {}", red.metallic);
    let green = model.materials.iter().find(|m| m.name == "green").expect("green");
    assert!((green.base_color[1] - 1.0).abs() < 1e-4 && green.base_color[0].abs() < 1e-4, "{:?}", green.base_color);
    assert_eq!(green.normal_map_texture.as_ref().unwrap().path, "normal.png");
    // red has no normal map, green has no metallic/roughness maps.
    assert!(red.normal_map_texture.is_none());
    assert!(green.roughness_texture.is_none() && green.metalness_texture.is_none());

    // The red material's part refers only to the first quad (x in 0..=1).
    let red_idx = model.materials.iter().position(|m| m.name == "red").unwrap();
    let red_part = model.parts.iter().find(|p| p.material == red_idx).unwrap();
    for i in &model.mesh.indices[red_part.start as usize..(red_part.start + red_part.count) as usize] {
        let x = model.mesh.positions[*i as usize][0];
        assert!((0.0..=1.0).contains(&x), "red part, x={x}");
    }
}

#[test]
fn pack_metallic_roughness_places_channels_with_neutral_fill() {
    let rough = RgbaImage::from_pixel(2, 2, image::Rgba([10, 99, 99, 255]));
    let metal = RgbaImage::from_pixel(2, 2, image::Rgba([200, 0, 0, 255]));
    let packed = pack_metallic_roughness(Some(&rough), Some(&metal), false).unwrap();
    assert_eq!(packed.get_pixel(0, 0).0, [255, 10, 200, 255], "G=rough, B=metal");
    // Roughness only: B is neutral (255), the metallic factor works alone.
    let only_r = pack_metallic_roughness(Some(&rough), None, false).unwrap();
    assert_eq!(only_r.get_pixel(0, 0).0, [255, 10, 255, 255]);
    // Metalness only: G is neutral.
    let only_m = pack_metallic_roughness(None, Some(&metal), false).unwrap();
    assert_eq!(only_m.get_pixel(0, 0).0, [255, 255, 200, 255]);
    // Spec/gloss: the G channel is inverted.
    let inv = pack_metallic_roughness(Some(&rough), None, true).unwrap();
    assert_eq!(inv.get_pixel(0, 0).0, [255, 245, 255, 255]);
    assert!(pack_metallic_roughness(None, None, false).is_none());
}

#[test]
fn tangents_generated_for_normal_mapped_parts() {
    let tmp = TempDir::new("tangents");
    let obj = write_source(tmp.path());
    let model = load_model(&obj, false).unwrap();
    let part = model
        .parts
        .iter()
        .find(|p| model.materials[p.material].normal_map_texture.is_some())
        .expect("part with a normal map");
    let mesh = phantom_runtime::render::part_mesh(&model, part, true);
    assert!(
        mesh.attribute(bevy::mesh::Mesh::ATTRIBUTE_TANGENT).is_some(),
        "tangents for the normal-mapped part"
    );
}

#[test]
fn bundle_round_trip_after_source_deleted() {
    let tmp = TempDir::new("roundtrip");
    let src = tmp.path().join("src");
    let obj = write_source(&src);
    let proj = tmp.path().join("proj");
    std::fs::create_dir_all(&proj).unwrap();

    let bundle = import_model_bundle(&proj, &obj).expect("bundle import");
    assert_eq!(bundle.model, PathBuf::from("assets/two/two.obj"));
    for dep in ["assets/two/two.mtl", "assets/two/red.png", "assets/two/green.png", "assets/two/normal.png"] {
        assert!(bundle.dependencies.contains(&PathBuf::from(dep)), "missing {dep} in {:?}", bundle.dependencies);
        assert!(proj.join(dep).exists(), "{dep} was not copied");
    }

    // Re-import — the catalog stays collision-free.
    let second = import_model_bundle(&proj, &obj).expect("re-import");
    assert_eq!(second.model, PathBuf::from("assets/two-1/two.obj"));

    // Source deleted: the bundle is self-contained.
    std::fs::remove_dir_all(&src).unwrap();
    let model = load_model(&proj.join(&bundle.model), false).expect("parse from the bundle");
    assert_eq!(model.materials.len(), 2);
    assert_eq!(model.parts.len(), 2);
    let red = model.materials.iter().find(|m| m.name == "red").unwrap();
    assert_eq!(red.base_color_texture.as_ref().unwrap().path, "red.png");
    assert!(proj.join("assets/two/red.png").exists());
}

#[test]
fn traversal_and_symlink_dependencies_rejected() {
    // mtllib escaping the source directory.
    let tmp = TempDir::new("traverse-mtl");
    let src = tmp.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(tmp.path().join("outside.mtl"), "newmtl x\nKd 1 1 1\n").unwrap();
    let obj = src.join("evil.obj");
    std::fs::write(&obj, TWO_QUADS_OBJ.replace("mtllib two.mtl", "mtllib ../outside.mtl")).unwrap();
    let proj = tmp.path().join("proj");
    std::fs::create_dir_all(&proj).unwrap();
    let err = import_model_bundle(&proj, &obj).unwrap_err();
    assert!(matches!(err, ImportError::PathEscape(_)), "{err}");
    assert!(!proj.join("assets/evil").exists(), "the half-built bundle was removed");

    // The ../ texture exists, but outside the allowed root.
    let tmp = TempDir::new("traverse-tex");
    let src = tmp.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    write_png(&tmp.path().join("evil.png"), [1, 2, 3, 255]);
    let obj = write_source(&src);
    std::fs::write(src.join("two.mtl"), TWO_MATS_MTL.replace("map_Kd red.png", "map_Kd ../evil.png")).unwrap();
    let proj = tmp.path().join("proj");
    std::fs::create_dir_all(&proj).unwrap();
    let err = import_model_bundle(&proj, &obj).unwrap_err();
    assert!(matches!(err, ImportError::PathEscape(_)), "{err}");

    // A symlinked texture points outside: canonicalization resolves the target, so reject.
    let tmp = TempDir::new("symlink");
    let src = tmp.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    write_png(&tmp.path().join("real.png"), [9, 9, 9, 255]);
    std::os::unix::fs::symlink(tmp.path().join("real.png"), src.join("link.png")).unwrap();
    let obj = write_source(&src);
    std::fs::write(src.join("two.mtl"), TWO_MATS_MTL.replace("map_Kd red.png", "map_Kd link.png")).unwrap();
    let proj = tmp.path().join("proj");
    std::fs::create_dir_all(&proj).unwrap();
    let err = import_model_bundle(&proj, &obj).unwrap_err();
    assert!(matches!(err, ImportError::PathEscape(_)), "{err}");
}

#[test]
fn missing_texture_is_diagnostic_not_failure() {
    let tmp = TempDir::new("missing");
    let src = tmp.path().join("src");
    let obj = write_source(&src);
    std::fs::remove_file(src.join("red.png")).unwrap();
    let proj = tmp.path().join("proj");
    std::fs::create_dir_all(&proj).unwrap();
    let bundle = import_model_bundle(&proj, &obj).expect("a missing texture is not a failure");
    assert!(
        bundle.diagnostics.iter().any(|d| d.contains("red.png") && d.contains("not found")),
        "{:?}",
        bundle.diagnostics
    );
}

#[test]
fn legacy_model_data_without_materials_deserializes() {
    let json = r#"{"mesh":{"positions":[],"normals":[],"uvs":[],"indices":[]},"skin":null}"#;
    let model: ModelData = serde_json::from_str(json).unwrap();
    assert!(model.materials.is_empty());
    assert!(model.parts.is_empty());
    assert!(model.diagnostics.is_empty());
    // The new record reads back.
    let text = serde_json::to_string(&model).unwrap();
    let model2: ModelData = serde_json::from_str(&text).unwrap();
    assert!(model2.parts.is_empty());
}

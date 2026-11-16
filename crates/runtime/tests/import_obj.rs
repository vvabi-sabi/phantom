use std::path::Path;
use phantom_runtime::import::load_model;

#[test]
fn wolf_obj_imports_as_static_mesh() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../43-obj/obj/Wolf_obj.obj");
    let model = load_model(&path, false).expect("the OBJ must import");
    assert!(model.skin.is_none());
    assert!(!model.mesh.positions.is_empty());
    assert_eq!(model.mesh.positions.len(), model.mesh.normals.len());
    assert_eq!(model.mesh.indices.len() % 3, 0);
    // Meters, Y up: the wolf must not be kilometers big.
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in &model.mesh.positions {
        for i in 0..3 {
            lo[i] = lo[i].min(p[i]);
            hi[i] = hi[i].max(p[i]);
        }
    }
    let size = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    assert!(size.iter().all(|s| *s < 100.0), "size {size:?}");
    assert!(size.iter().all(|s| *s > 0.001), "size {size:?}");
}

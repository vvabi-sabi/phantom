fn main() {
    let path = std::env::args().nth(1).unwrap();
    let model = phantom_runtime::import::load_model(std::path::Path::new(&path), true).unwrap();
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in &model.mesh.positions {
        for i in 0..3 { lo[i] = lo[i].min(p[i]); hi[i] = hi[i].max(p[i]); }
    }
    println!("bbox: {:?} .. {:?}", lo, hi);
    let skin = model.skin.unwrap();
    println!("bones: {}", skin.bones.len());
    let g = glam::Mat4::from_cols_array_2d(&skin.bones[0].inverse_bind).inverse();
    let (_s, _r, t) = g.to_scale_rotation_translation();
    println!("bind root: t={:?} scale={:?}", t, _s);
}

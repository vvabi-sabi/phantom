//! E2E: project → frames. Requires a GPU; run with:
//! `cargo test -p phantom-runtime --test render_e2e -- --ignored`


use glam::Vec3;
use phantom_core::*;
use phantom_runtime::render::{RenderOptions, Renderer};
use phantom_runtime::storage;

fn build_project(dir: &std::path::Path) -> (Project, SceneId) {
    let mut project = storage::create_project(dir, "e2e").unwrap();
    let mut history = History::default();
    // Simplest 4x4 background.
    let img = image::RgbaImage::from_pixel(4, 4, image::Rgba([20, 60, 140, 255]));
    img.save(dir.join("assets/bg.png")).unwrap();
    let asset = history
        .apply(&mut project, Command::RegisterAsset { kind: AssetKind::Image, path: "assets/bg.png".into() })
        .unwrap()
        .asset
        .unwrap();
    let scene = history
        .apply(&mut project, Command::AddScene { name: "s".into(), range: FrameRange::new(0, 48).unwrap() })
        .unwrap()
        .scene
        .unwrap();
    history
        .apply(
            &mut project,
            Command::SetBackground {
                scene,
                background: Background { image: BackgroundSource::Image(asset), arc_radians: 2.0, radius: 6.0, height: 4.0, center: Vec3::new(0.0, 2.0, -5.0), bulge: Default::default(), fit: Default::default() },
            },
        )
        .unwrap();
    let phantom = history
        .apply(&mut project, Command::AddPhantom { scene, name: "p".into(), rig: RigRef::Builtin })
        .unwrap()
        .phantom
        .unwrap();
    for (frame, x, action) in [(0, 0.0, Action::Idle), (48, 2.0, Action::Walk)] {
        history
            .apply(
                &mut project,
                Command::SetMotionKey {
                    scene,
                    phantom,
                    key: MotionKey { yaw: 0.0, frame, position: Vec3::new(x, 0.0, 0.0), action },
                },
            )
            .unwrap();
    }
    // A stroke across the top of the canvas: exposed at frame 0, it holds for the whole scene.
    history
        .apply(
            &mut project,
            Command::BrushStroke {
                scene,
                layer: None,
                frame: 0,
                stroke: Stroke {
                    points: vec![[0.0, 20.0], [1920.0, 20.0]],
                    color: [1.0, 0.0, 0.0, 1.0],
                    width: 30.0,
                    erase: false,
                },
            },
        )
        .unwrap();
    (project, scene)
}

#[test]
#[ignore = "requires GPU"]
fn scene_to_frames() {
    let dir = std::env::temp_dir().join(format!("phantom-render-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("assets")).unwrap();
    let (project, scene_id) = build_project(&dir);
    storage::save_project(&dir, &project).unwrap();

    let opts = RenderOptions { width: 320, height: 180, on_twos: false };
    let mut renderer = Renderer::new(320, 180).expect("GPU renderer");
    let scene = &project.scenes[&scene_id];

    let f0 = renderer.render_frame(&project, scene, &dir, 0, &opts, None).unwrap();
    let f24 = renderer.render_frame(&project, scene, &dir, 24, &opts, None).unwrap();
    let _ = std::fs::remove_dir_all(&dir);

    let px = |f: &phantom_runtime::render::RenderedFrame, x: u32, y: u32| {
        let o = ((y * f.width + x) * 4) as usize;
        (f.rgba[o], f.rgba[o + 1], f.rgba[o + 2], f.rgba[o + 3])
    };
    // The bottom corners are transparent (the stroke only covers the top).
    assert_eq!(px(&f0, 1, f0.height - 2).3, 0, "the corner must be transparent");
    assert_eq!(px(&f0, f0.width - 2, f0.height - 2).3, 0);
    // The center is opaque (background or character).
    assert_eq!(px(&f0, 160, 90).3, 255);
    // Outline: the frame has black opaque pixels.
    let has_outline = (0..f0.width * f0.height).step_by(7).any(|i| {
        let o = (i * 4) as usize;
        f0.rgba[o] < 20 && f0.rgba[o + 1] < 20 && f0.rgba[o + 2] < 20 && f0.rgba[o + 3] == 255
    });
    assert!(has_outline, "expected a black outline");
    // The character moves: the centroid of opaque pixels shifts right.
    let cx = |f: &phantom_runtime::render::RenderedFrame| {
        let mut sx = 0u64;
        let mut n = 0u64;
        for y in 0..f.height {
            for x in 0..f.width {
                let o = ((y * f.width + x) * 4) as usize;
                // The character is warm (r > b), the background is blue.
                if f.rgba[o + 3] == 255 && f.rgba[o] as i32 > f.rgba[o + 2] as i32 + 30 {
                    sx += x as u64;
                    n += 1;
                }
            }
        }
        sx as f64 / n.max(1) as f64
    };
    assert!(cx(&f24) > cx(&f0) + 20.0, "walking shifts the character: {} vs {}", cx(&f0), cx(&f24));

    // 2D drawing on top of the frame: a red band at the top in both frames
    // (canvas 1920×1080 → output 320×180; canvas y=20 ≈ output y=3).
    for f in [&f0, &f24] {
        let (r, _g, _b, a) = px(f, 160, 3);
        assert!(a == 255 && r > 200, "stroke in frame: r={r} a={a}");
    }
}

#[test]
#[ignore = "requires GPU"]
fn imported_obj_changes_pixels_from_empty_baseline() {
    // D1 regression: import → static object hierarchy → transform → extraction
    // → readback. The plain triangle faces the default camera at z=4.
    let dir = std::env::temp_dir().join(format!("phantom-render-object-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("assets/triangle")).unwrap();
    std::fs::write(
        dir.join("assets/triangle/triangle.obj"),
        // ufbx normalizes OBJ centimeters to meters; author a 2×2 m triangle.
        "v -100 0 0\nv 100 0 0\nv 0 200 0\nvn 0 0 1\nf 1//1 2//1 3//1\n",
    )
    .unwrap();
    let mut project = storage::create_project(&dir, "object-e2e").unwrap();
    let mut history = History::default();
    let scene_id = history
        .apply(&mut project, Command::AddScene { name: "s".into(), range: FrameRange::new(0, 1).unwrap() })
        .unwrap()
        .scene
        .unwrap();
    let opts = RenderOptions { width: 320, height: 180, on_twos: false };
    // One renderer: several Bevy App/GPU devices in one process are not
    // supported; a repeated render_still is covered by the separate D3.4 test.

    project.scenes.get_mut(&scene_id).unwrap().sky = Some([0.0, 0.0, 0.0]);
    let asset = history
        .apply(&mut project, Command::RegisterAsset { kind: AssetKind::ModelObj, path: "assets/triangle/triangle.obj".into() })
        .unwrap()
        .asset
        .unwrap();
    let object = history
        .apply(&mut project, Command::AddObject { scene: scene_id, name: "triangle".into(), asset })
        .unwrap()
        .object
        .unwrap();
    history
        .apply(
            &mut project,
            Command::SetObjectKey {
                scene: scene_id,
                object,
                key: ObjectKey { frame: 0, position: Vec3::ZERO, yaw: 0.0, pitch: 0.0, roll: 0.0, scale: Vec3::ONE },
            },
        )
        .unwrap();
    // Exclude lighting/tonemapping: this test targets the OBJ path's geometry.
    project.scenes.get_mut(&scene_id).unwrap().objects.get_mut(&object).unwrap().material_patches.push(MaterialPatch {
        target: None, base_color: Some([1.0, 0.0, 1.0, 1.0]), metallic: None,
        roughness: None, emissive: None, opacity: None, unlit: Some(true),
    });
    let mut renderer = Renderer::new(opts.width, opts.height).expect("GPU renderer");
    let with_object = renderer.render_frame(&project, &project.scenes[&scene_id], &dir, 0, &opts, None).unwrap();
    let magenta = with_object.rgba.chunks_exact(4).filter(|pixel| pixel[0] > 120 && pixel[2] > 80 && pixel[1] < 80).count();
    assert!(magenta > 100, "the imported OBJ must cover a visible area of the frame, magenta={magenta}, visible_objects={}", with_object.objects_in_frame);
    assert_eq!(with_object.objects_in_frame, 1, "the render report must tell real visible props apart from a background-only frame");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "requires GPU"]
fn builtin_catalog_prop_renders_without_any_asset_file() {
    // K1: a catalog prop must draw without an import — otherwise the catalog
    // repeats the D1 defect class (the object exists, no pixels).
    let dir = std::env::temp_dir().join(format!("phantom-render-prop-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut project = storage::create_project(&dir, "prop-e2e").unwrap();
    let mut history = History::default();
    let scene_id = history
        .apply(&mut project, Command::AddScene { name: "s".into(), range: FrameRange::new(0, 1).unwrap() })
        .unwrap()
        .scene
        .unwrap();
    project.scenes.get_mut(&scene_id).unwrap().sky = Some([0.0, 0.0, 0.0]);
    // A 1.4×0.75×0.8 m table 4 m in front of the camera (default camera at [0,1.4,4]).
    let key = ObjectKey { frame: 0, position: Vec3::new(0.0, 0.0, 0.0), yaw: 0.0, pitch: 0.0, roll: 0.0, scale: Vec3::ONE };
    let object = history
        .apply(&mut project, Command::AddBuiltinObject { scene: scene_id, name: "table".into(), prop: "table".into(), key: Some(key) })
        .unwrap()
        .object
        .unwrap();
    project.scenes.get_mut(&scene_id).unwrap().objects.get_mut(&object).unwrap().material_patches.push(MaterialPatch {
        target: None, base_color: Some([1.0, 0.0, 1.0, 1.0]), metallic: None,
        roughness: None, emissive: None, opacity: None, unlit: Some(true),
    });
    let opts = RenderOptions { width: 320, height: 180, on_twos: false };
    let mut renderer = Renderer::new(opts.width, opts.height).expect("GPU renderer");
    let frame = renderer.render_frame(&project, &project.scenes[&scene_id], &dir, 0, &opts, None).unwrap();
    let magenta = frame.rgba.chunks_exact(4).filter(|p| p[0] > 120 && p[2] > 80 && p[1] < 80).count();
    assert!(magenta > 100, "the catalog prop must be visible: magenta={magenta}");
    assert_eq!(frame.objects_in_frame, 1);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "requires GPU"]
fn project_round_trip_through_disk() {
    let dir = std::env::temp_dir().join(format!("phantom-storage-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("assets")).unwrap();
    let (project, _s) = build_project(&dir);
    storage::save_project(&dir, &project).unwrap();
    let loaded = storage::load_project(&dir).unwrap();
    assert_eq!(
        serde_json::to_string(&project).unwrap(),
        serde_json::to_string(&loaded).unwrap()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

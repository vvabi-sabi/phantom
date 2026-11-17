//! Forge scenes without GPU: stage lighting with shadows, tonemapping and MSAA
//! on the shared-path camera (viewport + export), matte floor. World structure
//! only; the visual result is checked by the user.

use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::prelude::*;
use phantom_core::*;
use phantom_runtime::render::{apply_scene_frame, forge_scene};

fn test_world() -> World {
    let mut world = World::new();
    world.insert_resource(Assets::<Image>::default());
    world.insert_resource(Assets::<Mesh>::default());
    world.insert_resource(Assets::<StandardMaterial>::default());
    // The 2D-layer quad cache is read by forge even for a scene without layers.
    world.insert_resource(phantom_runtime::render::DrawCache::default());
    world
}

fn test_scene(lighting: SceneLighting, floor: Option<SceneFloor>) -> Scene {
    let mut scene = Scene::new(SceneId(1), "s", FrameRange::new(0, 24).unwrap());
    scene.lighting = lighting;
    scene.floor = floor;
    scene
}

fn forge(world: &mut World, scene: &Scene) {
    let project = Project::new("t");
    let target = Handle::<Image>::default();
    forge_scene(world, &project, scene, std::path::Path::new("."), &target).unwrap();
}

fn apply(world: &mut World, scene: &Scene, forge: &phantom_runtime::render::ForgeResult, frame: i64) {
    let project = Project::new("t");
    apply_scene_frame(world, &project, scene, forge, frame, false, None);
}

#[test]
fn forge_applies_shadows_tonemapping_and_msaa() {
    let mut world = test_world();
    let scene = test_scene(SceneLighting::default(), None);
    forge(&mut world, &scene);

    // Three stage directional light sources; shadows are off by default
    // (corrective A2: the skinned shadow pipeline crashes, shadows are opt-in).
    let lights: Vec<&DirectionalLight> = world.query::<&DirectionalLight>().iter(&world).collect();
    assert_eq!(lights.len(), 3);
    assert!(lights.iter().all(|l| !l.shadows_enabled));
    assert!(lights.iter().any(|l| (l.illuminance - 8000.0).abs() < 1e-3), "key ≈ 8000 lux");

    // Shared-path camera: tonemapping and MSAA 4x.
    let (tonemapping, msaa) = world.query::<(&Tonemapping, &Msaa)>().single(&world).unwrap();
    assert_eq!(*tonemapping, Tonemapping::Reinhard);
    assert_eq!(*msaa, Msaa::Sample4);

    // Without a floor there is no opaque floor geometry.
    assert_eq!(world.query::<&MeshMaterial3d<StandardMaterial>>().iter(&world).count(), 0);
}

#[test]
fn forge_respects_light_colors_intensity_and_enabled() {
    let mut world = test_world();
    let lighting = SceneLighting {
        key: SceneLight { color: [1.0, 0.5, 0.25], intensity: 12000.0, enabled: true, shadows: true },
        fill: SceneLight { color: [0.5, 0.6, 1.0], intensity: 1000.0, enabled: false, shadows: false },
        rim: SceneLight { color: [1.0, 1.0, 1.0], intensity: 5000.0, enabled: true, shadows: false },
    };
    let scene = test_scene(lighting, None);
    forge(&mut world, &scene);

    // A disabled fill is not spawned.
    let lights: Vec<&DirectionalLight> = world.query::<&DirectionalLight>().iter(&world).collect();
    assert_eq!(lights.len(), 2);
    let key = lights.iter().find(|l| (l.illuminance - 12000.0).abs() < 1e-3).unwrap();
    let c = key.color.to_linear();
    assert!((c.red - 1.0).abs() < 1e-4 && (c.green - 0.5).abs() < 1e-4 && (c.blue - 0.25).abs() < 1e-4);
    // The shadows flag passes through to the light source one-to-one.
    assert!(key.shadows_enabled);
    let rim = lights.iter().find(|l| (l.illuminance - 5000.0).abs() < 1e-3).unwrap();
    assert!(!rim.shadows_enabled);
}

#[test]
fn forge_spawns_matte_floor_when_enabled() {
    let mut world = test_world();
    let scene = test_scene(SceneLighting::default(), Some(SceneFloor { color: [0.3, 0.4, 0.5] }));
    forge(&mut world, &scene);

    let mut q = world.query::<&MeshMaterial3d<StandardMaterial>>();
    let mats: Vec<_> = q.iter(&world).collect();
    assert_eq!(mats.len(), 1, "exactly one material entity — the floor");
    let assets = world.resource::<Assets<StandardMaterial>>();
    let mat = assets.get(&mats[0].0).expect("floor material loaded");
    let c = mat.base_color.to_linear();
    assert!((c.red - 0.3).abs() < 1e-4 && (c.green - 0.4).abs() < 1e-4 && (c.blue - 0.5).abs() < 1e-4);
    assert!(mat.perceptual_roughness > 0.9, "matte floor");
    assert!(mat.alpha_mode == AlphaMode::Opaque, "the floor is opaque geometry");
}

#[test]
fn forge_spawns_document_lights_and_applies_keys_per_frame() {
    use phantom_core::{LightKey, SceneLightKind, SceneLightSource, TransformKey};
    let mut world = test_world();
    let mut scene = test_scene(SceneLighting::default(), None);
    // No stage key/fill/rim — a clean test of document light sources.
    scene.lighting.key.enabled = false;
    scene.lighting.fill.enabled = false;
    scene.lighting.rim.enabled = false;
    scene.lights = vec![
        SceneLightSource {
            id: phantom_core::LightId(11),
            name: "lamp".into(),
            kind: SceneLightKind::Point,
            color: [1.0, 0.8, 0.6],
            intensity: 700.0,
            enabled: true,
            shadows: false,
            keys: vec![LightKey { frame: 12, translation: Vec3::new(2.0, 3.0, -1.0), yaw: 0.5, pitch: -0.3, intensity: Some(1200.0), enabled: None }],
            rest: TransformKey::at(0, Vec3::ZERO, 0.0, 0.0),
            attach: None,
        },
        SceneLightSource {
            id: phantom_core::LightId(12),
            name: "door".into(),
            kind: SceneLightKind::Spot { cone: 0.7 },
            color: [1.0; 3],
            intensity: 2000.0,
            enabled: true,
            shadows: true,
            keys: vec![LightKey { frame: 12, translation: Vec3::ZERO, yaw: 0.0, pitch: 0.0, intensity: None, enabled: Some(false) }],
            rest: TransformKey::at(0, Vec3::ZERO, 0.0, 0.0),
            attach: None,
        },
    ];
    let project = Project::new("t");
    let target = Handle::<Image>::default();
    let forge = forge_scene(&mut world, &project, &scene, std::path::Path::new("."), &target).unwrap();
    assert_eq!(forge.lights.len(), 2);

    // Spawn: point with lumens, spot with a 0.7 rad cone (inner 0.8×) and shadows.
    let points: Vec<&PointLight> = world.query::<&PointLight>().iter(&world).collect();
    assert!(points.iter().any(|l| (l.intensity - 700.0).abs() < 1e-3 && !l.shadows_enabled), "point 700 lumens without shadows");
    let spots: Vec<&SpotLight> = world.query::<&SpotLight>().iter(&world).collect();
    let spot = spots.iter().find(|l| (l.intensity - 2000.0).abs() < 1e-3).expect("spot");
    assert!((spot.outer_angle - 0.7).abs() < 1e-4 && (spot.inner_angle - 0.56).abs() < 1e-4);
    assert!(spot.shadows_enabled);
    assert_eq!(world.query::<&DirectionalLight>().iter(&world).count(), 0, "stage lighting is disabled");

    // Before the key (frame 0): rest pose, enabled from the body.
    apply(&mut world, &scene, &forge, 0);
    let lamp = forge.lights[&phantom_core::LightId(11)];
    assert_eq!(world.get::<Transform>(lamp).unwrap().translation, Vec3::ZERO);
    // At the key (frame 12): key pose, intensity overridden, spot disabled.
    apply(&mut world, &scene, &forge, 12);
    let tr = world.get::<Transform>(lamp).unwrap();
    assert_eq!(tr.translation, Vec3::new(2.0, 3.0, -1.0));
    assert!((world.get::<PointLight>(lamp).unwrap().intensity - 1200.0).abs() < 1e-3);
    let door = forge.lights[&phantom_core::LightId(12)];
    assert_eq!(*world.get::<Visibility>(door).unwrap(), Visibility::Hidden);
}

#[test]
fn forge_applies_sky_clear_color_and_ambient_modes() {
    use phantom_core::AmbientMode;
    // Sky: a clear color instead of transparency.
    let mut world = test_world();
    let mut scene = test_scene(SceneLighting::default(), None);
    scene.sky = Some([0.2, 0.3, 0.5]);
    forge(&mut world, &scene);
    let cam = world.query::<&bevy::prelude::Camera>().single(&world).unwrap();
    match &cam.clear_color {
        bevy::camera::ClearColorConfig::Custom(c) => {
            let l = c.to_linear();
            assert!((l.red - 0.2).abs() < 1e-4 && (l.alpha - 1.0).abs() < 1e-4, "the sky is opaque");
        }
        other => panic!("expected a custom clear color: {other:?}"),
    }
    // Ambient fixed/off override the background.
    let mut world = test_world();
    let mut scene = test_scene(SceneLighting::default(), None);
    scene.ambient = AmbientMode::Fixed { color: [0.1, 0.2, 0.3], brightness: 777.0 };
    forge(&mut world, &scene);
    let a = world.resource::<AmbientLight>();
    assert!((a.brightness - 777.0).abs() < 1e-3);
    let c = a.color.to_linear();
    assert!((c.blue - 0.3).abs() < 1e-4);
    let mut world = test_world();
    let mut scene = test_scene(SceneLighting::default(), None);
    scene.ambient = AmbientMode::Off;
    forge(&mut world, &scene);
    assert_eq!(world.resource::<AmbientLight>().brightness, 0.0);
}

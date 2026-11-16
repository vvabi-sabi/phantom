//! Optional GPU smoke test: all original actors render without external resources.
use glam::Vec3;
use phantom_core::*;
use phantom_runtime::render::{RenderOptions, Renderer};

#[test]
#[ignore = "requires GPU"]
fn bundled_profiles_render_distinct_silhouettes() {
    let mut project = Project::new("bundled");
    let mut history = History::default();
    let scene =
        history.apply(&mut project, Command::AddScene { name: "s".into(), range: FrameRange::new(0, 48).unwrap() }).unwrap().scene.unwrap();
    let id =
        history.apply(&mut project, Command::AddPhantom { scene, name: "actor".into(), rig: RigRef::Builtin }).unwrap().phantom.unwrap();
    let mut renderer = Renderer::new(320, 180).expect("GPU renderer");
    let options = RenderOptions { width: 320, height: 180, on_twos: false };
    let mut pictures = Vec::new();
    for actor in [BuiltinActor::Humanoid, BuiltinActor::Wolf, BuiltinActor::Horse, BuiltinActor::Bird] {
        let phantom = project.scenes.get_mut(&scene).unwrap().phantoms.get_mut(&id).unwrap();
        phantom.rig = RigRef::BuiltinActor { actor };
        phantom.motion = vec![MotionKey { yaw: 0.0, frame: 0, position: Vec3::ZERO, action: actor.actions()[0] }];
        let image = renderer.render_frame(&project, &project.scenes[&scene], std::path::Path::new("/tmp"), 9, &options, None).unwrap();
        assert!(image.rgba.chunks_exact(4).filter(|p| p[3] > 0).count() > 20, "empty {actor:?}");
        assert!(image.rgba.chunks_exact(4).any(|p| p[3] == 0), "no transparent background");
        assert!(!pictures.contains(&image.rgba), "profile cache reused wrong model");
        pictures.push(image.rgba);
    }
}

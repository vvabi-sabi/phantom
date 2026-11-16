//! A shot clip inside a film: mapping into the scene range, scene audio is not
//! duplicated.

#[test]
fn shot_clip_maps_to_scene_range_and_audio_mixes_once() {
    use glam::Vec3;
    use phantom_core::*;
    let mut project = Project::new("t");
    let mut h = History::default();
    let asset = h.apply(&mut project, Command::RegisterAsset { kind: AssetKind::Audio, path: "a.wav".into() }).unwrap().asset.unwrap();
    let scene = h.apply(&mut project, Command::AddScene { name: "s".into(), range: FrameRange::new(0, 96).unwrap() }).unwrap().scene.unwrap();
    let track = h.apply(&mut project, Command::AddAudioTrack { scene: Some(scene), name: "m".into() }).unwrap().track.unwrap();
    h.apply(&mut project, Command::AddAudioClip { track, asset, start: 0, offset_samples: 0, len_samples: 192000, fade_in_samples: 0, fade_out_samples: 0 }).unwrap();
    let shot = h
        .apply(&mut project, Command::AddShot { scene, name: "a".into(), range: FrameRange::new(24, 48).unwrap() })
        .unwrap()
        .shot
        .unwrap();
    // Film: a legacy clip over the scene range + a shot clip of the same scene.
    h.apply(&mut project, Command::AddFilmClip { scene, range: FrameRange::new(0, 10).unwrap(), shot: None }).unwrap();
    h.apply(&mut project, Command::AddFilmClip { scene, range: FrameRange::new(24, 48).unwrap(), shot: Some(shot) }).unwrap();

    // Mapping: the shot clip goes to scene frames [24, 48); the shot is kept on the clip.
    assert_eq!(film_map(&project.film, 10), Some((scene, 24)));
    let (clip, frame) = film_map_clip(&project.film, 20).unwrap();
    assert_eq!(clip.shot, Some(shot));
    assert_eq!(frame, 34);

    // Audio: one source per film frame — no double mix
    // (a single clip peaks at 0.5, double addition would give 1.0).
    let mut pcm = PcmCache::new();
    pcm.insert(asset, PcmBuf { samples: vec![[0.5, 0.5]; 192000] });
    let (film, peak) = phantom_runtime::audio::mix_film(&project, &pcm);
    assert_eq!(film.samples.len(), (10 + 24) * 2000);
    assert!(peak <= 0.5 + 1e-6, "double mix: peak {peak}");
    assert!(film.samples.iter().all(|s| (s[0] - 0.5).abs() < 1e-6));

    // The shot's camera differs from the scene's legacy camera and is picked by id.
    let s = &project.scenes[&scene];
    let _ = Vec3::ZERO; // glam is used in the test manifest
    assert!(std::ptr::eq(s.camera_for_shot(Some(shot)), &s.shots[&shot].camera));
    assert!(std::ptr::eq(s.camera_for_shot(None), &s.camera));
}

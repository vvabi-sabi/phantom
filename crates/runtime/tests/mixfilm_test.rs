#[test]
fn mix_film_maps_scene_audio_through_cuts() {
    use phantom_core::*;
    let mut project = Project::new("t");
    let mut h = History::default();
    let asset = h.apply(&mut project, Command::RegisterAsset { kind: AssetKind::Audio, path: "a.wav".into(), bounds: None }).unwrap().asset.unwrap();
    let scene = h.apply(&mut project, Command::AddScene { name: "s".into(), range: FrameRange::new(0, 96).unwrap() }).unwrap().scene.unwrap();
    let track = h.apply(&mut project, Command::AddAudioTrack { scene: Some(scene), name: "m".into() }).unwrap().track.unwrap();
    h.apply(&mut project, Command::AddAudioClip { track, asset, start: 0, offset_samples: 0, len_samples: 192000, fade_in_samples: 0, fade_out_samples: 0 }).unwrap();
    h.apply(&mut project, Command::AddFilmClip { scene, range: FrameRange::new(0, 48).unwrap(), camera: None }).unwrap();

    let mut pcm = PcmCache::new();
    pcm.insert(asset, PcmBuf { samples: vec![[0.5, 0.5]; 192000] });
    let (film, peak) = phantom_runtime::audio::mix_film(&project, &pcm);
    assert_eq!(film.samples.len(), 48 * 2000);
    assert!(peak > 0.4, "peak {peak}");
    assert!(film.samples.iter().any(|s| s[0] > 0.4), "there are non-zero samples");
}

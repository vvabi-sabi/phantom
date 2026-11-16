//! Silent playback fallback check: no device gives None, no panic.

#[test]
fn playback_fallback_is_silent() {
    match phantom_runtime::audio::Playback::start() {
        Ok(Some(_)) => eprintln!("device present — stream started"),
        Ok(None) => eprintln!("no device — silent fallback"),
        Err(e) => panic!("the error must be None, not Err: {e}"),
    }
}

#[test]
fn decode_and_wav_roundtrip() {
    // Generate a tone, decode it back, write WAV, check it is not silent.
    let dir = std::env::temp_dir().join(format!("phantom-audio-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let tone = dir.join("tone.wav");
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1"])
        .arg(&tone)
        .status()
        .unwrap();
    assert!(status.success());
    let pcm = phantom_runtime::audio::decode_audio(&tone).unwrap();
    assert!(pcm.samples.len() > 47000, "samples: {}", pcm.samples.len());
    assert!(pcm.samples.iter().any(|s| s[0].abs() > 0.05), "the signal is non-zero");
    let out = dir.join("out.wav");
    phantom_runtime::audio::write_wav(&pcm, &out).unwrap();
    let pcm2 = phantom_runtime::audio::decode_audio(&out).unwrap();
    assert!(pcm2.samples.iter().any(|s| s[0].abs() > 0.04));
    let _ = std::fs::remove_dir_all(&dir);
}

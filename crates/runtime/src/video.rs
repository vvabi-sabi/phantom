//! Streaming decode for the video background: ffmpeg resamples by timestamp to
//! the project's 24 fps (`-vf fps=24`), frame k ↔ time k/24 s; a VFR source
//! does not break the mapping. Ring buffer; looping restarts the process.

use std::collections::VecDeque;
use std::io::Read;
use std::path::Path;
use std::process::{Child, Command, Stdio};

use image::RgbaImage;

#[derive(Debug, thiserror::Error)]
pub enum VideoError {
    #[error("ffmpeg: {0}")]
    Ffmpeg(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("video has no frames")]
    NoFrames,
}

/// Preserve a 1080p video background for a 1920px master. This is still a
/// bounded decoded frame (~8 MiB RGBA), not an unbounded source-resolution cache.
pub const MAX_WIDTH: u32 = 1920;
/// Ring buffer depth (frames at the project's 24 fps).
const BUFFER_AHEAD: u64 = 24;

struct DecoderProc {
    child: Child,
    // Mutex: mpsc::Receiver is not Sync, and VideoCache is a Bevy resource (Send+Sync).
    rx: std::sync::Mutex<std::sync::mpsc::Receiver<RgbaImage>>,
}

pub struct VideoDecoder {
    path: std::path::PathBuf,
    pub width: u32,
    pub height: u32,
    proc: Option<DecoderProc>,
    buffer: VecDeque<RgbaImage>,
    /// Index of the next frame that will be pulled out of the process.
    next_pull: u64,
    eof: bool,
}

impl VideoDecoder {
    /// Open the video and read its dimensions via ffprobe.
    pub fn open(path: &Path) -> Result<Self, VideoError> {
        let probe = Command::new("ffprobe")
            .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0"])
            .arg(path)
            .output()?;
        let text = String::from_utf8_lossy(&probe.stdout);
        let mut parts = text.trim().split(',');
        let w: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        let h: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        if w == 0 || h == 0 {
            return Err(VideoError::Ffmpeg(format!("ffprobe could not read {}", path.display())));
        }
        let (mut width, mut height) = (w, h);
        if width > MAX_WIDTH {
            height = height * MAX_WIDTH / width;
            width = MAX_WIDTH;
        }
        // Even dimensions for rawvideo.
        width &= !1;
        height &= !1;
        let mut dec = Self {
            path: path.to_path_buf(),
            width,
            height,
            proc: None,
            buffer: VecDeque::new(),
            next_pull: 0,
            eof: false,
        };
        dec.restart()?;
        // The first frame is mandatory already at open: wait for it up to 10 s.
        for _ in 0..2000 {
            let got = dec.pull();
            if got > 0 {
                dec.next_pull += got as u64;
                return Ok(dec);
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        Err(VideoError::NoFrames)
    }

    fn restart(&mut self) -> Result<(), VideoError> {
        if let Some(mut p) = self.proc.take() {
            let _ = p.child.kill();
            let _ = p.child.wait();
        }
        let mut child = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel", "error",
                "-i", &self.path.to_string_lossy(),
                "-vf", &format!("fps=24,scale={}:{}", self.width, self.height),
                "-f", "rawvideo",
                "-pix_fmt", "rgba",
                "pipe:1",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let mut stdout = child.stdout.take().expect("piped");
        let (tx, rx) = std::sync::mpsc::sync_channel::<RgbaImage>(4);
        let (w, h) = (self.width as usize, self.height as usize);
        std::thread::spawn(move || {
            let frame_bytes = w * h * 4;
            loop {
                let mut buf = vec![0u8; frame_bytes];
                match stdout.read_exact(&mut buf) {
                    Ok(()) => {
                        let img = match RgbaImage::from_raw(w as u32, h as u32, buf) {
                            Some(i) => i,
                            None => break,
                        };
                        if tx.send(img).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        self.proc = Some(DecoderProc { child, rx: std::sync::Mutex::new(rx) });
        Ok(())
    }

    /// Pull available frames into the buffer; returns how many were new.
    fn pull(&mut self) -> usize {
        let mut got = 0;
        if let Some(p) = &self.proc {
            let rx = p.rx.lock().expect("decoder rx");
            while let Ok(img) = rx.try_recv() {
                self.buffer.push_back(img);
                got += 1;
            }
        }
        got
    }

    /// Frame with index `index` (project 24 fps). Pulls decoding ahead; at the
    /// end of the video — restart (loop, indices keep growing).
    pub fn frame_at(&mut self, index: u64) -> Option<&RgbaImage> {
        let first_buffered = self.next_pull - self.buffer.len() as u64;
        if index < first_buffered {
            // Backwards scrubbing: rewind from the start of the video.
            self.buffer.clear();
            self.next_pull = 0;
            self.eof = false;
            self.restart().ok()?;
        }
        let mut idle = 0u32;
        loop {
            let first = self.next_pull - self.buffer.len() as u64;
            if index >= first && index < self.next_pull {
                return self.buffer.get((index - first) as usize);
            }
            let got = self.pull();
            self.next_pull += got as u64;
            // Buffer is bounded: drop frames older than the requested one.
            while self.buffer.len() as u64 > BUFFER_AHEAD && self.next_pull - self.buffer.len() as u64 <= index.saturating_sub(BUFFER_AHEAD) {
                self.buffer.pop_front();
            }
            if got > 0 {
                idle = 0;
                continue;
            }
            let alive = self
                .proc
                .as_mut()
                .map(|p| matches!(p.child.try_wait(), Ok(None)))
                .unwrap_or(false);
            if !alive {
                // EOF: loop — restart, indices keep counting up (looping video).
                self.buffer.clear();
                self.restart().ok()?;
            }
            idle += 1;
            if idle > 2000 {
                self.eof = true;
                return None;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }
}

/// Average colors of the 3×2 grid zones in linear color (for zone light
/// sources). Alpha-weighted mean (corrective A7): transparent PNG pixels do not
/// drag a zone to black; a fully transparent zone is a neutral 0.
pub fn zone_averages(img: &RgbaImage) -> [[f32; 3]; 6] {
    let (w, h) = img.dimensions();
    let mut zones = [[0.0f64; 3]; 6];
    let mut weights = [0.0f64; 6];
    for (x, y, px) in img.enumerate_pixels() {
        let col = (x * 3 / w).min(2) as usize;
        let row = (y * 2 / h).min(1) as usize;
        let zi = row * 3 + col;
        let a = px.0[3] as f64 / 255.0;
        for c in 0..3 {
            let v = px.0[c] as f64 / 255.0;
            zones[zi][c] += v * v * a; // sRGB → approximate linear, alpha as weight
        }
        weights[zi] += a;
    }
    let mut out = [[0.0f32; 3]; 6];
    for zi in 0..6 {
        for c in 0..3 {
            let lin = (zones[zi][c] / weights[zi].max(1e-9)).sqrt();
            out[zi][c] = if weights[zi] > 0.0 { lin as f32 } else { 0.0 };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zones_average_quadrants() {
        let mut img = RgbaImage::new(6, 4);
        for (x, y, px) in img.enumerate_pixels_mut() {
            *px = if x < 3 { image::Rgba([255, 0, 0, 255]) } else { image::Rgba([0, 0, 255, 255]) };
            let _ = y;
        }
        let z = zone_averages(&img);
        assert!(z[0][0] > 0.9 && z[0][2] < 0.1, "left zone is red: {:?}", z[0]);
        assert!(z[2][2] > 0.9, "right zone is blue: {:?}", z[2]);
    }

    #[test]
    fn zones_weight_by_alpha_not_black() {
        // A7: semitransparent white plus a fully transparent "black": the zone
        // must stay bright instead of falling to black.
        let mut img = RgbaImage::new(6, 4);
        for (x, _, px) in img.enumerate_pixels_mut() {
            *px = if x < 3 { image::Rgba([255, 255, 255, 128]) } else { image::Rgba([0, 0, 0, 0]) };
        }
        let z = zone_averages(&img);
        assert!(z[0][0] > 0.9, "semitransparent white stays white: {:?}", z[0]);
        assert_eq!(z[2], [0.0, 0.0, 0.0], "fully transparent zone is a neutral 0: {:?}", z[2]);
    }
}

//! Metrics of a rendered RGBA8 frame — the agent's "eyes" without a window
//! (corrective C3.2/A1/A3). No GPU involved: they run over an already-read
//! buffer, so they test deterministically on synthetic frames.

use serde::Serialize;

/// Frame metrics. Brightness is Rec.709 luma 0..=255, computed over opaque
/// pixels only (alpha > 0): transparency means "no geometry here".
#[derive(Clone, Copy, Debug, Serialize)]
pub struct FrameMetrics {
    /// Mean brightness of opaque pixels (0 if the frame is empty).
    pub mean_luma: f64,
    /// 5th percentile of opaque-pixel brightness.
    pub luma_p5: f64,
    /// 95th percentile of opaque-pixel brightness.
    pub luma_p95: f64,
    /// Share of the frame with no geometry (alpha == 0), 0.0..=1.0.
    pub empty_coverage: f64,
}

/// Computes metrics over an RGBA8 readback buffer (4 bytes per pixel).
pub fn frame_metrics(rgba: &[u8], width: u32, height: u32) -> FrameMetrics {
    let pixels = (width as usize) * (height as usize);
    debug_assert_eq!(rgba.len(), pixels * 4);
    if pixels == 0 {
        return FrameMetrics { mean_luma: 0.0, luma_p5: 0.0, luma_p95: 0.0, empty_coverage: 0.0 };
    }
    let mut histogram = [0u64; 256];
    let mut opaque = 0u64;
    for px in rgba.chunks_exact(4) {
        if px[3] == 0 {
            continue;
        }
        opaque += 1;
        let luma = 0.2126 * px[0] as f64 + 0.7152 * px[1] as f64 + 0.0722 * px[2] as f64;
        histogram[luma.round().clamp(0.0, 255.0) as usize] += 1;
    }
    if opaque == 0 {
        return FrameMetrics { mean_luma: 0.0, luma_p5: 0.0, luma_p95: 0.0, empty_coverage: 1.0 };
    }
    let mut sum = 0.0f64;
    let mut weighted = 0.0f64;
    for (bin, &count) in histogram.iter().enumerate() {
        sum += count as f64;
        weighted += bin as f64 * count as f64;
    }
    debug_assert_eq!(sum as u64, opaque);
    let percentile = |p: f64| {
        let target = (p * opaque as f64).ceil() as u64;
        let mut seen = 0u64;
        for (bin, &count) in histogram.iter().enumerate() {
            seen += count;
            if seen >= target.max(1) {
                return bin as f64;
            }
        }
        255.0
    };
    FrameMetrics {
        mean_luma: weighted / sum,
        luma_p5: percentile(0.05),
        luma_p95: percentile(0.95),
        empty_coverage: 1.0 - opaque as f64 / pixels as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buffer(pixels: &[[u8; 4]]) -> Vec<u8> {
        pixels.iter().flatten().copied().collect()
    }

    #[test]
    fn uniform_opaque_frame() {
        let rgba = buffer(&[[100, 100, 100, 255]; 16]);
        let m = frame_metrics(&rgba, 4, 4);
        assert!((m.mean_luma - 100.0).abs() < 1e-9);
        assert_eq!(m.luma_p5, 100.0);
        assert_eq!(m.luma_p95, 100.0);
        assert_eq!(m.empty_coverage, 0.0);
    }

    #[test]
    fn half_transparent_is_half_empty_and_ignored_in_luma() {
        let mut pixels = vec![[255, 255, 255, 255]; 8];
        pixels.extend([[0, 0, 0, 0]; 8]);
        let rgba = buffer(&pixels);
        let m = frame_metrics(&rgba, 4, 4);
        assert_eq!(m.empty_coverage, 0.5);
        assert!((m.mean_luma - 255.0).abs() < 1e-9, "transparent pixels do not count toward brightness");
    }

    #[test]
    fn fully_empty_frame() {
        let rgba = buffer(&[[0, 0, 0, 0]; 4]);
        let m = frame_metrics(&rgba, 2, 2);
        assert_eq!(m.empty_coverage, 1.0);
        assert_eq!(m.mean_luma, 0.0);
    }

    #[test]
    fn percentiles_track_dark_tail_and_bright_body() {
        // 19 pixels of brightness 200 and 1 pixel of brightness 10: p5 = 10, p95 = 200.
        let mut pixels = vec![[200, 200, 200, 255]; 19];
        pixels.push([10, 10, 10, 255]);
        let rgba = buffer(&pixels);
        let m = frame_metrics(&rgba, 5, 4);
        assert_eq!(m.luma_p5, 10.0);
        assert_eq!(m.luma_p95, 200.0);
        assert!((m.mean_luma - (19.0 * 200.0 + 10.0) / 20.0).abs() < 0.5);
    }

    #[test]
    fn rec709_luma_weights_green_over_red_and_blue() {
        let rgba = buffer(&[[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [0, 0, 0, 0]]);
        let m = frame_metrics(&rgba, 2, 2);
        let expected = (0.2126 * 255.0 + 0.7152 * 255.0 + 0.0722 * 255.0) / 3.0;
        assert!((m.mean_luma - expected).abs() < 0.5);
    }
}

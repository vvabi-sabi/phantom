//! Rasterization of 2D-layer strokes and compositing over a frame.
//! The document stores strokes; the raster is derived data owned by this module.

use image::{Rgba, RgbaImage};
use phantom_core::{Exposure, Layer2D, Stroke};

/// Rasterize an exposure into an RGBA canvas (transparent background).
pub fn rasterize(canvas: (u32, u32), exposure: &Exposure) -> RgbaImage {
    let mut img = RgbaImage::from_pixel(canvas.0, canvas.1, Rgba([0, 0, 0, 0]));
    for stroke in &exposure.strokes {
        stamp_stroke(&mut img, stroke);
    }
    img
}

/// Discs along the polyline at a width/4 step.
fn stamp_stroke(img: &mut RgbaImage, stroke: &Stroke) {
    let r = stroke.width / 2.0;
    let step = (stroke.width / 4.0).max(0.5);
    for w in stroke.points.windows(2) {
        let (p0, p1) = (w[0], w[1]);
        let (dx, dy) = (p1[0] - p0[0], p1[1] - p0[1]);
        let len = (dx * dx + dy * dy).sqrt();
        let steps = (len / step).ceil().max(1.0) as u32;
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            stamp_disc(img, p0[0] + dx * t, p0[1] + dy * t, r, stroke);
        }
    }
}

fn stamp_disc(img: &mut RgbaImage, cx: f32, cy: f32, r: f32, stroke: &Stroke) {
    let (w, h) = img.dimensions();
    let (x0, x1) = ((cx - r).floor().max(0.0) as u32, (cx + r).ceil().min(w as f32) as u32);
    let (y0, y1) = ((cy - r).floor().max(0.0) as u32, (cy + r).ceil().min(h as f32) as u32);
    for y in y0..y1 {
        for x in x0..x1 {
            let d = ((x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2)).sqrt();
            // Half-pixel soft edge.
            let coverage = (r + 0.5 - d).clamp(0.0, 1.0);
            if coverage <= 0.0 {
                continue;
            }
            let a = stroke.color[3] * coverage;
            let px = img.get_pixel_mut(x, y);
            if stroke.erase {
                // destination-out: subtract alpha and color proportionally.
                px.0[3] = (px.0[3] as f32 * (1.0 - a)) as u8;
            } else {
                blend_over(px, stroke.color, a);
            }
        }
    }
}

/// source-over with straight (non-premultiplied) alpha.
fn blend_over(dst: &mut Rgba<u8>, src: [f32; 4], src_a: f32) {
    let da = dst.0[3] as f32 / 255.0;
    let out_a = src_a + da * (1.0 - src_a);
    if out_a <= 0.0 {
        *dst = Rgba([0, 0, 0, 0]);
        return;
    }
    for c in 0..3 {
        let dc = dst.0[c] as f32 / 255.0;
        let out = (src[c] * src_a + dc * da * (1.0 - src_a)) / out_a;
        dst.0[c] = (out.clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    dst.0[3] = (out_a.clamp(0.0, 1.0) * 255.0).round() as u8;
}

/// Composite the canvas onto a frame (scaling the canvas to the frame size).
pub fn composite_over(frame: &mut RgbaImage, canvas_img: &RgbaImage) {
    let (w, h) = frame.dimensions();
    let scaled: RgbaImage = if canvas_img.dimensions() == (w, h) {
        canvas_img.clone()
    } else {
        image::imageops::resize(canvas_img, w, h, image::imageops::FilterType::Triangle)
    };
    for (dst, src) in frame.pixels_mut().zip(scaled.pixels()) {
        let a = src.0[3] as f32 / 255.0;
        if a > 0.0 {
            blend_over(dst, [src.0[0] as f32 / 255.0, src.0[1] as f32 / 255.0, src.0[2] as f32 / 255.0, a], a);
        }
    }
}

/// Exposure raster for a layer at a frame (if it has an active exposure).
pub fn rasterize_for_frame(layer: &Layer2D, frame: phantom_core::Frame) -> Option<RgbaImage> {
    layer.exposure_for_frame(frame).map(|(_, e)| rasterize(layer.canvas, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use phantom_core::Stroke;

    fn stroke(points: Vec<[f32; 2]>, erase: bool) -> Stroke {
        Stroke { points, color: [1.0, 0.0, 0.0, 1.0], width: 8.0, erase }
    }

    #[test]
    fn stroke_marks_pixels() {
        let mut e = Exposure::default();
        e.strokes.push(stroke(vec![[10.0, 10.0], [90.0, 10.0]], false));
        let img = rasterize((100, 100), &e);
        let mid = img.get_pixel(50, 10);
        assert_eq!(mid.0[3], 255);
        assert_eq!(mid.0[0], 255, "red stroke");
        assert_eq!(img.get_pixel(50, 60).0[3], 0, "transparent outside the stroke");
    }

    #[test]
    fn eraser_subtracts() {
        let mut e = Exposure::default();
        e.strokes.push(stroke(vec![[10.0, 10.0], [90.0, 10.0]], false));
        e.strokes.push(stroke(vec![[40.0, 10.0], [60.0, 10.0]], true));
        let img = rasterize((100, 100), &e);
        assert!(img.get_pixel(50, 10).0[3] < 20, "eraser cleared the middle");
        assert_eq!(img.get_pixel(20, 10).0[3], 255, "edges remain");
    }

    #[test]
    fn composite_draws_over_frame() {
        let mut e = Exposure::default();
        e.strokes.push(stroke(vec![[10.0, 10.0], [90.0, 10.0]], false));
        let canvas = rasterize((100, 100), &e);
        let mut frame = RgbaImage::from_pixel(200, 200, Rgba([0, 0, 0, 0]));
        composite_over(&mut frame, &canvas);
        assert_eq!(frame.get_pixel(100, 20).0[3], 255, "scaled stroke landed in the frame");
    }
}

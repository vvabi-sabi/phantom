//! Pure document diagnostics (corrective C3.1/C3.3): `validate`,
//! `camera_coverage`, `background_fit`. Only the core document and camera
//! geometry — no render, no GPU, no media decoding.
//!
//! Screen geometry is the one `mesh_gen::screen_mesh` builds: a flat rectangle
//! `width × height` centered on `center`, facing the camera side (+Z). Both
//! places must agree — a coverage number that describes a different plane than
//! the one drawn would be worse than no number.
//! Camera — `Camera::sample` plus a YXZ(yaw, pitch) rotation looking along
//! local −Z, as in the render (`Quat::from_euler(YXZ, yaw, pitch, 0)`).
//! Vertical FOV — `phantom_core::vertical_fov`, the formula shared with the
//! renderer; horizontal FOV derived from it via `phantom_core::horizontal_fov`.
//!
//! Documented approximations:
//! - coverage is estimated from 129 columns (and two edges in height):
//!   the screen's azimuth range is the min/max over samples, sampling error
//!   < width / 128 m of the plane;
//! - vertical coverage uses the worst column inside the frustum
//!   (conservative: the real frame may be covered slightly better);
//! - camera roll does not exist in the document and is not accounted for;
//!   pitch is part of the camera basis and is accounted for exactly;
//! - the phantom/object point test uses the root point (no model volume)
//!   and the position without "on twos" quantization.

use std::collections::BTreeSet;
use std::path::Path;

use glam::Vec3;
use serde_json::{json, Value};

#[cfg(test)]
use phantom_core::{SceneCamera, SCENE_CAMERA};

use phantom_core::audio::SAMPLES_PER_FRAME;
use phantom_core::time::frame_to_secs;
use phantom_core::{
    evaluate, evaluate_object, film_len, Background, Camera, Frame, FrameRange, Project, Scene, CameraId, CameraSubject,
};

pub use phantom_core::view::DEFAULT_ASPECT;
use phantom_core::view::{Frustum, MIN_DEPTH};

/// Screen columns sampled for the coverage estimate (both ends included).
const SCREEN_SAMPLES: usize = 128;

/// Why the screen geometry is invalid; `None` — the geometry is valid.
fn background_invalid_reason(bg: &Background) -> Option<&'static str> {
    match bg.validate() {
        Ok(()) => None,
        Err(phantom_core::CoreError::InvalidBackground(reason)) => Some(reason),
        Err(phantom_core::CoreError::NonFinite(_)) => Some("non-numeric values in width/height/center"),
        Err(_) => Some("invalid geometry"),
    }
}

/// Angular spans of the screen seen from the camera (radians, relative to the
/// optical axis).
struct ScreenSpan {
    min_az: f32,
    max_az: f32,
    /// Smallest elevation of the top edge among columns inside the frustum.
    top_el: f32,
    /// Largest elevation of the bottom edge among columns inside the frustum.
    bottom_el: f32,
    /// Some columns lie inside the horizontal frustum.
    any_in_frustum: bool,
    /// Some screen points are behind the camera (coverage is not confirmed).
    any_behind: bool,
}

/// The flat screen sampled along its width: two edges (top and bottom) per
/// column. One implementation for the renderer's geometry and the diagnostics,
/// so a coverage number always describes the plane that is really drawn.
fn screen_span(bg: &Background, fr: &Frustum) -> ScreenSpan {
    let mut span = ScreenSpan {
        min_az: f32::INFINITY,
        max_az: f32::NEG_INFINITY,
        top_el: f32::INFINITY,
        bottom_el: f32::NEG_INFINITY,
        any_in_frustum: false,
        any_behind: false,
    };
    for i in 0..=SCREEN_SAMPLES {
        let x = (i as f32 / SCREEN_SAMPLES as f32 - 0.5) * bg.width;
        for &y in &[bg.height * 0.5, -bg.height * 0.5] {
            let p = bg.center + Vec3::new(x, y, 0.0);
            let l = fr.local(p);
            if l.z > -MIN_DEPTH {
                // Points behind the camera do not project into the frame: they
                // do not break coverage, but they also take no part in the span.
                span.any_behind = true;
                continue;
            }
            let az = l.x.atan2(-l.z);
            let el = l.y.atan2(-l.z);
            span.min_az = span.min_az.min(az);
            span.max_az = span.max_az.max(az);
            if az.abs() <= fr.hfov * 0.5 {
                span.any_in_frustum = true;
                if y > 0.0 {
                    span.top_el = span.top_el.min(el);
                } else {
                    span.bottom_el = span.bottom_el.max(el);
                }
            }
        }
    }
    span
}

/// How much of the frame the screen covers, plus the per-side gaps (radians).
struct Coverage {
    covers_frame: bool,
    span: ScreenSpan,
    gap_left: f32,
    gap_right: f32,
    gap_top: f32,
    gap_bottom: f32,
}

fn coverage(bg: &Background, fr: &Frustum) -> Coverage {
    let span = screen_span(bg, fr);
    let half_h = fr.hfov * 0.5;
    let half_v = fr.vfov * 0.5;
    let covers_h = span.min_az <= -half_h && span.max_az >= half_h;
    let covers_v = span.any_in_frustum && span.top_el >= half_v && span.bottom_el <= -half_v;
    Coverage {
        covers_frame: covers_h && covers_v,
        gap_left: (span.min_az + half_h).max(0.0),
        gap_right: (half_h - span.max_az).max(0.0),
        gap_top: if span.any_in_frustum { (half_v - span.top_el).max(0.0) } else { half_v },
        gap_bottom: if span.any_in_frustum { (span.bottom_el + half_v).max(0.0) } else { half_v },
        span,
    }
}

/// Does the screen at this width close the horizontal frustum with `margin`
/// of slack per side?
fn width_covers(bg: &Background, width: f32, fr: &Frustum, margin: f32) -> bool {
    let mut probe = bg.clone();
    probe.width = width;
    let half_h = fr.hfov * 0.5 + margin;
    let span = screen_span(&probe, fr);
    span.min_az <= -half_h && span.max_az >= half_h
}

/// Does the screen at this height close the vertical frustum with `margin`?
/// The caller must already have a sufficient width: a screen no column of which
/// falls inside the frame cannot cover any part of it.
fn height_covers(bg: &Background, height: f32, fr: &Frustum, margin: f32) -> bool {
    let mut probe = bg.clone();
    probe.height = height;
    let half_v = fr.vfov * 0.5 + margin;
    let span = screen_span(&probe, fr);
    span.any_in_frustum && span.top_el >= half_v && span.bottom_el <= -half_v
}

/// Smallest screen width that closes the horizontal frustum with `margin` at
/// the screen's current plane. `None` when no width helps — the plane does not
/// face the camera or lies behind it, in which case the screen has to be moved.
///
/// Monotone in width (a wider plane only widens the azimuth span), so growth to
/// a covering bound plus bisection is exact.
fn needed_width(bg: &Background, fr: &Frustum, margin: f32) -> Option<f32> {
    let mut lo = 0.0f32;
    let mut hi = bg.width.max(1.0);
    let mut guard = 0;
    while !width_covers(bg, hi, fr, margin) {
        lo = hi;
        hi *= 1.5;
        guard += 1;
        if guard > 48 {
            return None;
        }
    }
    for _ in 0..48 {
        let mid = 0.5 * (lo + hi);
        if width_covers(bg, mid, fr, margin) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Some(hi)
}

/// Smallest screen width that closes EVERY sample's horizontal frustum.
fn needed_width_all(bg: &Background, samples: &[FitSample], margin: f32) -> Option<f32> {
    let mut need = 0.0f32;
    for s in samples {
        need = need.max(needed_width(bg, &s.fr, margin)?);
    }
    Some(need)
}

/// Smallest screen height that closes every sample's vertical frustum.
fn needed_height_all(bg: &Background, samples: &[FitSample], margin: f32) -> Option<f32> {
    let mut lo = 0.0f32;
    let mut hi = bg.height.max(1.0);
    let covers = |h: f32| samples.iter().all(|s| height_covers(bg, h, &s.fr, margin));
    let mut guard = 0;
    while !covers(hi) {
        lo = hi;
        hi *= 1.5;
        guard += 1;
        if guard > 48 {
            return None;
        }
    }
    for _ in 0..48 {
        let mid = 0.5 * (lo + hi);
        if covers(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Some(hi)
}

/// Suggested_fix line for a coverage gap: the minimal width at the current
/// plane, plus a focal-length alternative, or an admission that no width helps.
fn gap_fix_string(bg: &Background, fr: &Frustum, aspect: f32) -> String {
    match needed_width(bg, fr, 0.0) {
        Some(width) => match max_fitting_focal(bg, fr, aspect) {
            Some(f) => format!("width >= {width:.1} m at z = {:.1} m, or focal_mm <= {f:.1}", bg.center.z),
            None => format!("width >= {width:.1} m at z = {:.1} m", bg.center.z),
        },
        None => format!(
            "no width closes the frame at z = {:.1} m — the plane does not face this camera; move the screen back (background_set center)",
            bg.center.z
        ),
    }
}

/// Largest focal length (mm) at which the current screen still closes the
/// frame; `None` when a side already has a gap (focal narrows both FOVs, so
/// the limit is the minimum over the two axes).
fn max_fitting_focal(bg: &Background, fr: &Frustum, aspect: f32) -> Option<f32> {
    let span = screen_span(bg, fr);
    let h_limit = span.max_az.min(-span.min_az); // symmetric horizontal slack
    if h_limit <= 0.0 || !span.any_in_frustum {
        return None;
    }
    let v_limit = span.top_el.min(-span.bottom_el);
    if v_limit <= 0.0 {
        return None;
    }
    // vfov <= 2·v_limit and hfov <= 2·h_limit; focal = 12 / tan(vfov/2),
    // tan(hfov/2) = tan(vfov/2)·aspect.
    let from_v = 12.0 / v_limit.tan();
    let from_h = 12.0 * aspect / h_limit.tan();
    Some(from_v.min(from_h))
}

fn deg(rad: f32) -> f64 {
    (rad as f64).to_degrees()
}

fn v3(p: Vec3) -> Value {
    json!([p.x, p.y, p.z])
}

fn gap_json(cov: &Coverage) -> Value {
    json!({
        "left": deg(cov.gap_left), "right": deg(cov.gap_right),
        "top": deg(cov.gap_top), "bottom": deg(cov.gap_bottom),
    })
}

fn span_json(cov: &Coverage) -> Value {
    json!({
        "horizontal_deg": { "min": deg(cov.span.min_az), "max": deg(cov.span.max_az) },
        "vertical_deg": {
            "min": if cov.span.any_in_frustum { json!(deg(cov.span.bottom_el)) } else { Value::Null },
            "max": if cov.span.any_in_frustum { json!(deg(cov.span.top_el)) } else { Value::Null },
        },
    })
}

fn camera_json(scene_id: u64, camera: Option<CameraId>, frame: Frame, fr: &Frustum, aspect: f32) -> Value {
    json!({
        "scene": scene_id,
        "camera_id": camera.map(|s| s.0),
        "frame": frame,
        "camera": {
            "pos": v3(fr.pos),
            "yaw_deg": deg(fr.yaw),
            "pitch_deg": deg(fr.pitch),
            "focal_mm": fr.focal_mm,
            "hfov_deg": deg(fr.hfov),
            "vfov_deg": deg(fr.vfov),
            "aspect": aspect,
        },
    })
}

/// Frame coverage by the background screen and phantom/object point
/// visibility at a frame (corrective C3.3). Pure geometry, no render.
///
/// `frame` defaults to the start of the camera's range. The scene and camera
/// must exist (the calling Session checks that).
pub fn camera_coverage(project: &Project, scene: &Scene, camera: Option<CameraId>, frame: Option<Frame>, aspect: f32) -> Value {
    let range = scene.range_for(camera);
    let frame = frame.unwrap_or(range.start);
    let fr = phantom_core::camera_frustum(project, scene, camera, frame, aspect);
    let mut out = camera_json(scene.id.0, camera, frame, &fr, aspect);
    let object = out.as_object_mut().expect("object");

    let mut findings = Vec::new();
    match &scene.background {
        None => {
            object.insert("background".into(), Value::Null);
            findings.push(json!({
                "code": "background_absent", "severity": "info",
                "where": {"scene": scene.id.0, "frame": frame},
                "why": "the scene has no background screen; covers_frame is undefined",
                "suggested_fix": "background_set with geometry from background_fit",
            }));
        }
        Some(bg) => {
            if let Some(reason) = background_invalid_reason(bg) {
                object.insert("background".into(), json!({
                    "width": bg.width, "height": bg.height,
                    "center": v3(bg.center), "covers_frame": Value::Null,
                }));
                findings.push(json!({
                    "code": "bg_geometry_invalid", "severity": "error",
                    "where": {"scene": scene.id.0, "frame": frame},
                    "why": format!("screen geometry is invalid: {reason}"),
                    "suggested_fix": "fix background_set (width > 0, height > 0, finite center)",
                }));
            } else {
                let cov = coverage(bg, &fr);
                object.insert("background".into(), json!({
                    "width": bg.width, "height": bg.height,
                    "center": v3(bg.center),
                    "covers_frame": cov.covers_frame,
                    "span": span_json(&cov),
                    "gap_deg": gap_json(&cov),
                    "points_behind_camera": cov.span.any_behind,
                }));
                if !cov.covers_frame {
                    let needed_width = needed_width(bg, &fr, 0.0);
                    findings.push(json!({
                        "code": "screen_fov_gap", "severity": "warning",
                        "where": {"scene": scene.id.0, "frame": frame, "camera": camera.map(|s| s.0)},
                        "why": format!("the screen does not cover the frame: gaps in degrees {}", gap_json(&cov)),
                        "suggested_fix": gap_fix_string(bg, &fr, aspect),
                        "needed_width_m": needed_width.map(|w| (w as f64 * 100.0).ceil() / 100.0),
                    }));
                }
            }
        }
    }

    let phantoms: Vec<Value> = scene
        .phantoms
        .values()
        .map(|ph| {
            let state = evaluate(ph, frame);
            json!({"id": ph.id.0, "pos": v3(state.root), "in_frame": fr.contains(state.root)})
        })
        .collect();
    let objects: Vec<Value> = scene
        .objects
        .values()
        .map(|o| {
            let t = evaluate_object(o, frame);
            json!({"id": o.id.0, "pos": v3(t.position), "in_frame": fr.contains(t.position)})
        })
        .collect();
    object.insert("phantoms".into(), json!(phantoms));
    object.insert("objects".into(), json!(objects));
    object.insert("findings".into(), json!(findings));
    out
}

/// Fit the screen size to the camera (layout mode): the smallest honest
/// width/height that close the frame with `margin_deg` of slack per side, at
/// the screen's current plane.
///
/// The screen is a flat rectangle at `center`, facing +Z. `center` is preserved
/// (or the default [0,2,−5] when there is no background); a camera that never
/// faces the plane is not fixable by size alone, and then `size_covers` is
/// false with an empty suggestion — move the screen with `background_set`.
///
/// `width` — the minimum closing the horizontal frustum of every sample (growth
/// to a covering bound plus bisection; the span is monotone in width);
/// `height` — the same for the vertical frustum, at that width;
/// `panorama_aspect = width / height` — the ratio for the panorama generator.
///
/// Without `camera` the fit runs over the WHOLE scene (corrective_2 K0.1): each
/// camera's keys are sampled and the size must close every sample — the union of
/// the cameras' requirements, not one camera. Fitting a single camera that looks elsewhere would otherwise give a "fit for
/// the neighbouring frame" and leave a hole in its own.
/// An explicit `frame` without `camera` makes no sense (each camera has its own
/// ranges) and is rejected by the calling code.
///
/// Results are rounded up (0.01 m), so the suggestion really does close the
/// frame (roundtrip test).
pub fn background_fit(project: &Project, scene: &Scene, camera: Option<CameraId>, frame: Option<Frame>, margin_deg: f32, aspect: f32) -> Value {
    let margin = margin_deg.to_radians();
    let samples = fit_samples(project, scene, camera, frame, aspect);
    let scope = if camera.is_some() { "camera" } else { "scene" };

    let current = scene.background.as_ref();
    let center = current.map(|b| b.center).filter(|c| c.is_finite()).unwrap_or(Vec3::new(0.0, 2.0, -5.0));
    // The image source does not affect geometry — a placeholder id.
    let source = || phantom_core::BackgroundSource::Image(phantom_core::AssetId(0));
    let image = current.map(|b| b.image).unwrap_or_else(source);
    let fit = current.map(|b| b.fit).unwrap_or_default();
    let base = Background { image, width: current.map(|b| b.width).filter(|w| w.is_finite() && *w > 0.0).unwrap_or(8.0), height: 1.0, center, fit };

    // Width first (it decides whether any column is inside the frame at all),
    // then height at that width.
    let needed = needed_width_all(&base, &samples, margin);
    let width = needed.map(|w| (w as f64 * 100.0).ceil() / 100.0).unwrap_or(0.0);
    let mut with_width = base.clone();
    with_width.width = width as f32;
    let height = needed
        .and_then(|_| needed_height_all(&with_width, &samples, margin))
        .map(|h| (h as f64 * 100.0).ceil() / 100.0)
        .unwrap_or(0.0);
    let size_covers = needed.is_some() && height > 0.0;

    let panorama_aspect = if height > 0.0 { width / height } else { 0.0 };

    let suggested = Background { image, width: width as f32, height: height as f32, center, fit };

    let (current_json, fits, delta) = match current {
        Some(bg) => {
            let valid = background_invalid_reason(bg).is_none();
            let fits = if valid { Some(samples.iter().all(|s| coverage(bg, &s.fr).covers_frame)) } else { None };
            (
                json!({
                    "width": bg.width, "height": bg.height, "center": v3(bg.center),
                    "geometry_valid": valid,
                }),
                fits,
                json!({
                    "width": width - bg.width as f64,
                    "height": height - bg.height as f64,
                }),
            )
        }
        None => (Value::Null, None, Value::Null),
    };

    // Worst sample per camera: it is what sets the size; the rest are reference.
    let mut per_camera: Vec<Value> = Vec::new();
    for (id, worst) in worst_per_camera(&samples, &suggested, margin) {
        let cov = coverage(current.unwrap_or(&suggested), &worst.fr);
        per_camera.push(json!({
            "camera": id.map(|s| s.0),
            "frame": worst.frame,
            "covers_current": current.map(|_| cov.covers_frame),
            "gap_deg": gap_json(&cov),
            "needed_width_m": needed_width(&suggested, &worst.fr, margin).map(|w| (w as f64 * 100.0).ceil() / 100.0),
        }));
    }
    // Binding sample: the one needing the largest width.
    let binding = samples
        .iter()
        .max_by(|a, b| {
            let key = |s: &FitSample| needed_width(&suggested, &s.fr, margin).unwrap_or(f32::INFINITY);
            key(a).total_cmp(&key(b))
        })
        .expect("fit_samples is never empty");

    let mut out = camera_json(scene.id.0, binding.camera, binding.frame, &binding.fr, aspect);
    let object = out.as_object_mut().expect("object");
    object.insert("scope".into(), json!(scope));
    object.insert("camera_count".into(), json!(per_camera.len()));
    object.insert("samples".into(), json!(samples.len()));
    object.insert("margin_deg".into(), json!(margin_deg));
    object.insert("current".into(), current_json);
    object.insert("suggested".into(), json!({
        "width": width,
        "height": height,
        "center": v3(center),
    }));
    object.insert("panorama_aspect".into(), json!(panorama_aspect));
    object.insert("size_covers".into(), json!(size_covers));
    object.insert("what_fits".into(), json!("a flat screen at its current plane; no size helps when a camera never faces it — move the screen with background_set"));
    object.insert("fits".into(), fits.map(|f| json!(f)).unwrap_or(Value::Null));
    object.insert("delta".into(), delta);
    object.insert("cameras".into(), json!(per_camera));
    // A3b: the lower half of the frame without a floor is dead; suggest enabling one.
    object.insert(
        "floor".into(),
        json!({
            "present": scene.floor.is_some(),
            "suggested": scene.floor.is_none(),
            "why": "the lower part of the frame is not covered by the screen without a floor; set-floor/scene_floor_set removes the void",
        }),
    );
    out
}

/// Camera sample for the fit: the camera (None — the scene's own) and frame.
struct FitSample {
    camera: Option<CameraId>,
    frame: Frame,
    fr: Frustum,
}

/// Camera samples for the fit: an explicit `camera` — only it (the frame is given
/// explicitly or taken from the start of the range); otherwise the scene's
/// every camera at its keys and range ends.
fn fit_samples(project: &Project, scene: &Scene, camera: Option<CameraId>, frame: Option<Frame>, aspect: f32) -> Vec<FitSample> {
    let mut out = Vec::new();
    let mut sources: Vec<(Option<CameraId>, &Camera, FrameRange)> = match camera {
        Some(id) => vec![(Some(id), scene.camera_for(Some(id)), scene.range_for(Some(id)))],
        None => scene.cameras.values().map(|c| (Some(c.id), &c.camera, c.range)).collect(),
    };
    // An explicit frame only makes sense for a single camera; at scene scope
    // cameras are sampled at their own keys.
    let explicit = if camera.is_some() { frame } else { None };
    sources.retain(|(_, _, range)| range.end > range.start);
    for (id, camera, range) in sources {
        let frames = match explicit {
            Some(f) => vec![f],
            None => coverage_frames(camera, range),
        };
        for f in frames {
            out.push(FitSample { camera: id, frame: f, fr: phantom_core::camera_frustum(project, scene, id, f, aspect) });
        }
    }
    if out.is_empty() {
        let f = frame.unwrap_or(scene.range.start);
        out.push(FitSample { camera, frame: f, fr: phantom_core::camera_frustum(project, scene, camera, f, aspect) });
    }
    out
}

/// The worst sample of each camera (the one needing the largest arc under
/// geometry `bg`), in id order; `None` — the scene's legacy camera.
fn worst_per_camera<'a>(samples: &'a [FitSample], bg: &Background, margin: f32) -> Vec<(Option<CameraId>, &'a FitSample)> {
    let mut ids: Vec<Option<CameraId>> = samples.iter().map(|s| s.camera).collect();
    ids.sort_by_key(|id| id.map(|s| s.0));
    ids.dedup();
    ids.into_iter()
        .map(|id| {
            let worst = samples
                .iter()
                .filter(|s| s.camera == id)
                .max_by(|a, b| {
                    let key = |s: &FitSample| needed_width(bg, &s.fr, margin).unwrap_or(f32::INFINITY);
                    key(a).total_cmp(&key(b))
                })
                .expect("a camera has samples");
            (id, worst)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// validate (corrective C3.1)
// ---------------------------------------------------------------------------

fn finding(code: &str, severity: &str, scene: Option<u64>, frame: Option<Frame>, mut extra_where: Value, why: String, suggested_fix: String) -> Value {
    let object = extra_where.as_object_mut().expect("where object");
    if let Some(s) = scene {
        object.insert("scene".into(), json!(s));
    }
    if let Some(f) = frame {
        object.insert("frame".into(), json!(f));
    }
    json!({
        "code": code,
        "severity": severity,
        "where": extra_where,
        "why": why,
        "suggested_fix": suggested_fix,
    })
}

/// A camera subject as `details` JSON.
fn subject_json(subject: CameraSubject) -> Value {
    match subject {
        CameraSubject::Phantom(id) => json!({"kind": "phantom", "id": id.0}),
        CameraSubject::Object(id) => json!({"kind": "object", "id": id.0}),
    }
}

/// Coverage-check frames: the range ends and the camera keys inside the range.
fn coverage_frames(camera: &Camera, range: FrameRange) -> Vec<Frame> {
    let mut frames: BTreeSet<Frame> = [range.start, range.end - 1].into_iter().collect();
    frames.extend(camera.keys.iter().map(|k| k.frame).filter(|f| *f >= range.start && *f < range.end));
    frames.into_iter().collect()
}

/// End of an audio clip in its owner's timeline frames (rounded up).
fn audio_clip_end(clip: &phantom_core::AudioClip) -> Frame {
    let frames = (clip.len_samples as f64 / SAMPLES_PER_FRAME as f64).ceil() as Frame;
    clip.start + frames.max(0)
}

/// Pure project/scene validation (corrective C3.1). Returns findings in a
/// deterministic order (scene, code, frame). `dir` is only needed to check
/// that asset files exist on disk; `None` — that check is skipped.
pub fn validate(project: &Project, dir: Option<&Path>, scene_filter: Option<u64>) -> Result<Vec<Value>, phantom_core::CoreError> {
    let mut findings: Vec<(Option<u64>, &'static str, Option<Frame>, Value)> = Vec::new();
    let mut push = |scene: Option<u64>, code: &'static str, frame: Option<Frame>, f: Value| {
        findings.push((scene, code, frame, f));
    };

    let scenes: Vec<&Scene> = match scene_filter {
        Some(id) => vec![project
            .scenes
            .get(&phantom_core::SceneId(id))
            .ok_or(phantom_core::CoreError::SceneNotFound(phantom_core::SceneId(id)))?],
        None => project.scenes.values().collect(),
    };

    for scene in scenes {
        let sid = scene.id.0;

        // The camera must look at the scene. A command rejects such a key, so a
        // finding here means a document written by hand or by an older version;
        // severity is a warning because no command can produce it, and a camera
        // nobody keyed (a scene's untouched default camera) is not judged.
        for camera in scene.cameras.values() {
            if camera.camera.keys.is_empty() {
                continue;
            }
            for frame in coverage_frames(&camera.camera, camera.range) {
                let fraction = phantom_core::camera::scene_in_frame_with(project, scene, Some(camera.id), frame, DEFAULT_ASPECT);
                if fraction < phantom_core::camera::MIN_SCENE_IN_FRAME {
                    let percent = (fraction as f64 * 100.0 * 10.0).round() / 10.0;
                    let minimum = phantom_core::camera::MIN_SCENE_IN_FRAME * 100.0;
                    push(
                        Some(sid),
                        "camera_faces_away",
                        Some(frame),
                        finding(
                            "camera_faces_away",
                            "warning",
                            Some(sid),
                            Some(frame),
                            json!({"camera": camera.id.0, "name": camera.name, "scene_fraction": percent / 100.0, "minimum": phantom_core::camera::MIN_SCENE_IN_FRAME}),
                            format!("camera {} \"{}\" sees {percent}% of the scene at frame {frame} (minimum {minimum}%)", camera.id.0, camera.name),
                            "aim the camera at the scene: set look_at to a scene point, or move the key inside the stage".to_string(),
                        ),
                    );
                }
            }
        }

        // An attachment whose subject is gone (or has no keys): the camera
        // falls back to its own world-space keys, silently. Say so.
        for camera in scene.cameras.values() {
            let Some(target) = &camera.target else { continue };
            if phantom_core::camera::subject_transform(scene, target.subject, camera.range.start).is_none() {
                push(
                    Some(sid),
                    "camera_subject_missing",
                    Some(camera.range.start),
                    finding(
                        "camera_subject_missing",
                        "warning",
                        Some(sid),
                        Some(camera.range.start),
                        json!({"camera": camera.id.0, "name": camera.name, "subject": subject_json(target.subject)}),
                        format!("camera {} \"{}\" is attached to a subject that has no position (deleted, or without motion keys): it falls back to its own world-space keys", camera.id.0, camera.name),
                        "camera_target_set with clear, or give the subject a motion key".to_string(),
                    ),
                );
            }
        }

        // Background screen geometry.
        if let Some(bg) = &scene.background {
            if let Some(reason) = background_invalid_reason(bg) {
                push(
                    Some(sid),
                    "bg_geometry_invalid",
                    None,
                    finding(
                        "bg_geometry_invalid",
                        "error",
                        Some(sid),
                        None,
                        json!({}),
                        format!("screen geometry is invalid: {reason}"),
                        "background_set with width > 0 m, height > 0 m and a finite center".to_string(),
                    ),
                );
            } else {
                // FOV coverage: every camera of the scene over its own range.
                let checks: Vec<(Option<u64>, &Camera, FrameRange)> =
                    scene.cameras.values().map(|c| (Some(c.id.0), &c.camera, c.range)).collect();
                for (camera_id, camera, range) in checks {
                    for frame in coverage_frames(camera, range) {
                        let fr = Frustum::new(camera, frame, DEFAULT_ASPECT);
                        let cov = coverage(bg, &fr);
                        if cov.covers_frame {
                            continue;
                        }
                        let needed_width_m = needed_width(bg, &fr, 0.0).map(|w| (w as f64 * 100.0).ceil() / 100.0);
                        let suggested = gap_fix_string(bg, &fr, DEFAULT_ASPECT);
                        push(
                            Some(sid),
                            "screen_fov_gap",
                            Some(frame),
                            finding(
                                "screen_fov_gap",
                                "warning",
                                Some(sid),
                                Some(frame),
                                json!({"camera": camera_id, "needed_width_m": needed_width_m, "gap_deg": gap_json(&cov)}),
                                format!("the screen does not cover the frame at frame {frame}: gaps in degrees {}", gap_json(&cov)),
                                suggested,
                            ),
                        );
                    }
                }
            }
        }

        // Bevy 0.17/wgpu currently rejects the shadow prepass for skinned
        // meshes (joint-index interstage validation). Never discover that
        // after a long render: conservative warning for any Phantom, whose
        // rig may be skinned, with any enabled shadow source.
        let shadow_source = scene.lighting.key.shadows
            || scene.lighting.fill.shadows
            || scene.lighting.rim.shadows
            || scene.lights.iter().any(|light| light.shadows);
        if shadow_source && !scene.phantoms.is_empty() {
            push(
                Some(sid),
                "skinned_shadows_unsupported",
                None,
                finding(
                    "skinned_shadows_unsupported",
                    "warning",
                    Some(sid),
                    None,
                    json!({"phantoms": scene.phantoms.len()}),
                    "shadows are enabled while Phantom actors are present: Bevy 0.17/wgpu may reject the shadow prepass of a skinned mesh".to_string(),
                    "turn shadows off on the light sources before rendering; the skinned shadow pipeline fix is not implemented yet".to_string(),
                ),
            );
        }

        // Props: a pure staging check, not a claim about the importer/mesh
        // state. Even a visible background makes an alpha metric useless for
        // this question, so the roots are checked in the same frustum as
        // camera_coverage, at every camera point of the scene.
        if !scene.objects.is_empty() {
            let checks: Vec<(Option<u64>, &Camera, FrameRange)> =
                scene.cameras.values().map(|c| (Some(c.id.0), &c.camera, c.range)).collect();
            let any_in_frame = checks.into_iter().any(|(_, camera, range)| {
                coverage_frames(camera, range).into_iter().any(|frame| {
                    let fr = Frustum::new(camera, frame, DEFAULT_ASPECT);
                    scene.objects.values().any(|object| fr.contains(evaluate_object(object, frame).position))
                })
            });
            if !any_in_frame {
                let count = scene.objects.len();
                push(
                    Some(sid),
                    "objects_out_of_frame",
                    None,
                    finding(
                        "objects_out_of_frame",
                        "warning",
                        Some(sid),
                        None,
                        json!({"object_count": count, "objects_in_frame": 0, "estimate": "object_root_frustum"}),
                        format!("the scene has {count} objects, but no root lands inside any checked camera"),
                        "run camera_coverage for the intended camera; move/key the object or change the camera. This estimates object roots, it is not a render result".to_string(),
                    ),
                );
            }
        }

        // Stage bounds (layout mode): every staged entity must fit the box.
        // Commands clamp, so a finding here means a document that predates the
        // stage or was written by hand.
        for ph in scene.phantoms.values() {
            for key in &ph.motion {
                let footprint = project.phantom_footprint(ph, key.pose);
                let rotation = glam::Quat::from_rotation_y(key.yaw);
                if !scene.stage.fits(key.position, footprint, rotation) {
                    let (clamped, _) = scene.stage.clamp(key.position, footprint, rotation);
                    push(
                        Some(sid),
                        "out_of_stage",
                        Some(key.frame),
                        finding(
                            "out_of_stage",
                            "error",
                            Some(sid),
                            Some(key.frame),
                            json!({"kind": "phantom", "id": ph.id.0, "frame": key.frame, "position": key.position.to_array(), "clamped": clamped.to_array(), "stage": scene.stage}),
                            format!("phantom {} frame {} stands outside the stage", ph.id.0, key.frame),
                            "set the key again (key_set) or grow the stage (stage_set); a command clamps the position automatically".to_string(),
                        ),
                    );
                }
            }
        }
        for object in scene.objects.values() {
            for key in &object.keys {
                let footprint = project.object_footprint(object, key.scale);
                let rotation = glam::Quat::from_euler(glam::EulerRot::YXZ, key.yaw, key.pitch, key.roll);
                if !scene.stage.fits(key.position, footprint, rotation) {
                    let (clamped, _) = scene.stage.clamp(key.position, footprint, rotation);
                    push(
                        Some(sid),
                        "out_of_stage",
                        Some(key.frame),
                        finding(
                            "out_of_stage",
                            "error",
                            Some(sid),
                            Some(key.frame),
                            json!({"kind": "object", "id": object.id.0, "frame": key.frame, "position": key.position.to_array(), "clamped": clamped.to_array(), "stage": scene.stage}),
                            format!("object {} frame {} stands outside the stage", object.id.0, key.frame),
                            "set the key again (object_key_set) or grow the stage (stage_set); a command clamps the position automatically".to_string(),
                        ),
                    );
                }
            }
        }

        // Phantom motion: how position keys cover the scene range.
        for ph in scene.phantoms.values() {
            if ph.motion.is_empty() {
                push(
                    Some(sid),
                    "phantom_without_motion",
                    None,
                    finding(
                        "phantom_without_motion",
                        "info",
                        Some(sid),
                        None,
                        json!({"phantom": ph.id.0, "name": ph.name}),
                        format!("phantom {} has no motion keys", ph.id.0),
                        "key_set or motion_replace".to_string(),
                    ),
                );
                continue;
            }
            let start_cov: Option<Frame> = ph.motion.first().map(|k| k.frame);
            let end_cov: Option<Frame> = ph.motion.last().map(|k| k.frame + 1);
            if let (Some(first), Some(last)) = (start_cov, end_cov) {
                if first > scene.range.start || last < scene.range.end {
                    push(
                        Some(sid),
                        "motion_gap",
                        Some(first.max(scene.range.start)),
                        finding(
                            "motion_gap",
                            "warning",
                            Some(sid),
                            Some(first.max(scene.range.start)),
                            json!({"phantom": ph.id.0, "covered": {"start": first, "end": last}, "scene_range": scene.range}),
                            format!(
                                "phantom {} motion covers [{first}, {last}), scene [{}, {})",
                                ph.id.0, scene.range.start, scene.range.end
                            ),
                            "add keys across the whole scene range".to_string(),
                        ),
                    );
                }
            }
        }

        // Empty scene.
        if scene.phantoms.is_empty() && scene.objects.is_empty() && scene.layers_2d.is_empty() {
            push(
                Some(sid),
                "scene_empty",
                None,
                finding(
                    "scene_empty",
                    "info",
                    Some(sid),
                    None,
                    json!({}),
                    format!("scene {sid} contains no phantom, objects or 2D layers"),
                    "phantom_add/object_add/layer_add_2d".to_string(),
                ),
            );
        }

        // Scene audio tracks.
        for track in &scene.audio_tracks {
            let mut track_end: Option<Frame> = None;
            for clip in &track.clips {
                let end = audio_clip_end(clip);
                track_end = Some(track_end.map_or(end, |e: Frame| e.max(end)));
                if clip.start < scene.range.start || end > scene.range.end {
                    push(
                        Some(sid),
                        "audio_out_of_range",
                        Some(clip.start),
                        finding(
                            "audio_out_of_range",
                            "warning",
                            Some(sid),
                            Some(clip.start),
                            json!({"track": track.id.0, "clip": clip.id.0, "clip_end_frame": end,
                                   "start_secs": frame_to_secs(clip.start - scene.range.start),
                                   "len_secs": clip.len_samples as f64 / phantom_core::SAMPLE_RATE as f64}),
                            format!(
                                "audio clip {} on track {} is outside the scene range [{}, {})",
                                clip.id.0, track.id.0, scene.range.start, scene.range.end
                            ),
                            "audio_clip_add with start/len inside the scene range".to_string(),
                        ),
                    );
                }
            }
            if let Some(end) = track_end {
                if end != scene.range.end {
                    let longer = end > scene.range.end;
                    push(
                        Some(sid),
                        "audio_length_mismatch",
                        None,
                        finding(
                            "audio_length_mismatch",
                            "warning",
                            Some(sid),
                            None,
                            json!({"track": track.id.0, "track_end_frame": end, "scene_end_frame": scene.range.end,
                                   "track_secs": frame_to_secs(end - scene.range.start), "scene_secs": frame_to_secs(scene.range.len())}),
                            format!(
                                "track {} {} the scene: it ends at frame {end}, scene [{}, {})",
                                track.id.0,
                                if longer { "longer than" } else { "shorter than" },
                                scene.range.start, scene.range.end
                            ),
                            if longer { "trim the clip (len_secs) to the end of the scene" } else { "extend the track to the end of the scene" }.to_string(),
                        ),
                    );
                }
            }
        }
    }

    // Film edit clips.
    for clip in &project.film {
        let bad = match project.scenes.get(&clip.scene) {
            None => Some(format!("clip {} references a scene that does not exist: {}", clip.id.0, clip.scene.0)),
            Some(scene) => {
                if clip.range.start < scene.range.start || clip.range.end > scene.range.end {
                    Some(format!(
                        "clip {} [{}, {}) is outside the range of scene {} [{}, {})",
                        clip.id.0, clip.range.start, clip.range.end, clip.scene.0, scene.range.start, scene.range.end
                    ))
                } else if let Some(camera) = clip.camera {
                    match scene.cameras.get(&camera) {
                        None => Some(format!("clip {} references a camera that does not exist: camera {} of scene {}", clip.id.0, camera.0, clip.scene.0)),
                        Some(s) if s.range != clip.range && camera.0 != 0 => Some(format!(
                            "clip {} [{}, {}) does not match the range of camera {} [{}, {})",
                            clip.id.0, clip.range.start, clip.range.end, camera.0, s.range.start, s.range.end
                        )),
                        _ => None,
                    }
                } else {
                    None
                }
            }
        };
        if let Some(why) = bad {
            push(
                None,
                "film_clip_out_of_scene",
                Some(clip.range.start),
                finding(
                    "film_clip_out_of_scene",
                    "error",
                    None,
                    Some(clip.range.start),
                    json!({"clip": clip.id.0, "scene": clip.scene.0, "camera": clip.camera.map(|s| s.0), "range": clip.range}),
                    why,
                    "film_remove + film_add with a range inside the scene or the camera's own range".to_string(),
                ),
            );
        }
    }

    // Film assembly (corrective_2 K0.3, the internal part): whether every named
    // camera is assembled, whether scene frames go unshown, and whether ranges
    // overlap twice. Cross-checking against the storyboard is deliberately
    // not done here — that is the editor's/director's job on their side.
    {
        let camera_assembly = project.film.iter().any(|clip| clip.camera.is_some_and(|c| c.0 != 0));
        if camera_assembly {
            for scene in project.scenes.values() {
                for camera in scene.cameras.values().filter(|c| c.id.0 != 0) {
                    if !project.film.iter().any(|clip| clip.scene == scene.id && clip.camera == Some(camera.id)) {
                        push(
                            Some(scene.id.0),
                            "camera_not_in_film",
                            Some(camera.range.start),
                            finding(
                                "camera_not_in_film",
                                "warning",
                                Some(scene.id.0),
                                Some(camera.range.start),
                                json!({"camera": camera.id.0, "name": camera.name, "range": camera.range}),
                                format!("camera {} \"{}\" of scene {} is not assembled into the film", camera.id.0, camera.name, scene.id.0),
                                "film_add with camera, or delete the camera if it is not needed".to_string(),
                            ),
                        );
                    }
                }
            }
        }
        // Scene frames the film never shows (a hole inside the scene range)
        // and frames shown twice (an overlap).
        for scene in project.scenes.values() {
            let mut ranges: Vec<FrameRange> = project
                .film
                .iter()
                .filter(|clip| clip.scene == scene.id)
                .map(|clip| clip.range)
                .collect();
            if ranges.is_empty() {
                continue;
            }
            ranges.sort_by_key(|r| (r.start, r.end));
            let mut covered_end = scene.range.start;
            for r in &ranges {
                let from = r.start.max(scene.range.start);
                let to = r.end.min(scene.range.end);
                if from > covered_end {
                    push(
                        Some(scene.id.0),
                        "film_coverage_gap",
                        Some(covered_end),
                        finding(
                            "film_coverage_gap",
                            "warning",
                            Some(scene.id.0),
                            Some(covered_end),
                            json!({"scene_range": scene.range, "unshown": {"start": covered_end, "end": from}}),
                            format!("scene {}: frames [{covered_end}, {from}) are not shown in the film", scene.id.0),
                            "film_add over this range, or trim the scene with scene_range_set".to_string(),
                        ),
                    );
                }
                covered_end = covered_end.max(to);
            }
            if covered_end < scene.range.end {
                push(
                    Some(scene.id.0),
                    "film_coverage_gap",
                    Some(covered_end),
                    finding(
                        "film_coverage_gap",
                        "warning",
                        Some(scene.id.0),
                        Some(covered_end),
                        json!({"scene_range": scene.range, "unshown": {"start": covered_end, "end": scene.range.end}}),
                        format!("scene {}: frames [{covered_end}, {}) are not shown in the film", scene.id.0, scene.range.end),
                        "film_add over this range, or trim the scene with scene_range_set".to_string(),
                    ),
                );
            }
            for pair in ranges.windows(2) {
                let (a, b) = (pair[0], pair[1]);
                if b.start < a.end {
                    push(
                        Some(scene.id.0),
                        "film_overlap",
                        Some(b.start),
                        finding(
                            "film_overlap",
                            "warning",
                            Some(scene.id.0),
                            Some(b.start),
                            json!({"first": a, "second": b, "overlap": {"start": b.start, "end": a.end.min(b.end)}}),
                            format!("scene {}: frames [{}, {}) are shown twice in the film", scene.id.0, b.start, a.end.min(b.end)),
                            "acceptable if the repeat is intentional, otherwise film_remove the redundant clip".to_string(),
                        ),
                    );
                }
            }
        }
    }

    // Film audio tracks.
    let film_frames = film_len(&project.film);
    for track in &project.film_audio {
        let mut track_end: Option<Frame> = None;
        for clip in &track.clips {
            let end = audio_clip_end(clip);
            track_end = Some(track_end.map_or(end, |e: Frame| e.max(end)));
            if clip.start < 0 || end > film_frames {
                push(
                    None,
                    "audio_out_of_range",
                    Some(clip.start),
                    finding(
                        "audio_out_of_range",
                        "warning",
                        None,
                        Some(clip.start),
                        json!({"track": track.id.0, "clip": clip.id.0, "clip_end_frame": end, "film_frames": film_frames,
                               "len_secs": clip.len_samples as f64 / phantom_core::SAMPLE_RATE as f64}),
                        format!("audio clip {} on film track {} goes past the film length of {film_frames} frames", clip.id.0, track.id.0),
                        "audio_clip_add with start/len inside the film length".to_string(),
                    ),
                );
            }
        }
        if let Some(end) = track_end {
            if end > film_frames {
                push(
                    None,
                    "audio_length_mismatch",
                    None,
                    finding(
                        "audio_length_mismatch",
                        "warning",
                        None,
                        None,
                        json!({"track": track.id.0, "track_end_frame": end, "film_frames": film_frames,
                               "track_secs": frame_to_secs(end), "film_secs": frame_to_secs(film_frames)}),
                        format!("film track {} is longer than the film: it ends at frame {end}, film is {film_frames} frames", track.id.0),
                        "trim the clip to the film length".to_string(),
                    ),
                );
            }
        }
    }

    // Asset file existence.
    if let Some(dir) = dir {
        for asset in project.assets.values() {
            if !dir.join(&asset.path).exists() {
                push(
                    None,
                    "asset_missing",
                    None,
                    finding(
                        "asset_missing",
                        "error",
                        None,
                        None,
                        json!({"asset": asset.id.0, "path": asset.path, "kind": format!("{:?}", asset.kind)}),
                        format!("asset file {} ({}) not found in the project directory", asset.id.0, asset.path.display()),
                        "asset_import the file into the project again".to_string(),
                    ),
                );
            }
        }
    }

    // Deterministic order: scene (project-level findings first), code, frame.
    findings.sort_by(|a, b| (a.0, a.1, a.2).cmp(&(b.0, b.1, b.2)));
    Ok(findings.into_iter().map(|(_, _, _, f)| f).collect())
}

/// Summarised validate report for Session/MCP.
pub fn validate_report(project: &Project, dir: Option<&Path>, scene: Option<u64>) -> Result<Value, phantom_core::CoreError> {
    let findings = validate(project, dir, scene)?;
    let ok = !findings.iter().any(|f| f["severity"] == "error");
    Ok(json!({
        "ok": ok,
        "revision": project.revision,
        "scene": scene,
        "findings": findings,
    }))
}

/// Short audio-vs-video summary for `state_get` (A4 — "length must be a
/// checkable fact before rendering"). Document data only, no decoding:
/// source length and RMS live in `audio_report`.
pub fn audio_summary(project: &Project) -> Value {
    let track_end = |tracks: &[phantom_core::audio::AudioTrack]| -> Frame {
        tracks
            .iter()
            .flat_map(|t| t.clips.iter())
            .map(|c| c.start + (c.len_samples / SAMPLES_PER_FRAME as u64) as Frame)
            .max()
            .unwrap_or(0)
    };
    let entry = |video_end: Frame, audio_end: Frame| {
        json!({
            "video_seconds": frame_to_secs(video_end),
            "audio_seconds": frame_to_secs(audio_end),
            "lost_seconds": frame_to_secs((audio_end - video_end).max(0)),
        })
    };
    let scenes: Vec<Value> = project
        .scenes
        .values()
        .map(|s| {
            let mut e = entry(s.range.end, track_end(&s.audio_tracks));
            e["scene"] = json!(s.id.0);
            e
        })
        .collect();
    let film_frames = film_len(&project.film);
    let mut film = entry(film_frames, track_end(&project.film_audio));
    film["film_frames"] = json!(film_frames);
    json!({ "scenes": scenes, "film": film })
}

// ---------------------------------------------------------------- audio (A4)

/// Lower bound of the dB scale (instead of -inf for digital silence).
pub const DB_FLOOR: f64 = -120.0;
/// "Almost silence" threshold from corrective A4: a fragment quieter than this will not be heard.
pub const QUIET_FRAGMENT_DB: f64 = -60.0;

/// RMS and peak of a stereo f32 fragment in dBFS (20·log10, full scale 1.0).
/// Silence and an empty fragment both give [`DB_FLOOR`].
pub fn rms_peak_db(samples: &[[f32; 2]]) -> (f64, f64) {
    if samples.is_empty() {
        return (DB_FLOOR, DB_FLOOR);
    }
    let mut sum = 0.0f64;
    let mut peak = 0.0f64;
    for [l, r] in samples {
        sum += (*l as f64) * (*l as f64) + (*r as f64) * (*r as f64);
        peak = peak.max(l.abs() as f64).max(r.abs() as f64);
    }
    let rms = (sum / (samples.len() as f64 * 2.0)).sqrt();
    let to_db = |x: f64| if x > 0.0 { 20.0 * x.log10() } else { DB_FLOOR };
    (to_db(rms).max(DB_FLOOR), to_db(peak).max(DB_FLOOR))
}

/// Result of probing one audio clip (decoding happens on the caller's side,
/// this is only the struct). Durations are in 48 kHz samples.
pub struct AudioProbe {
    /// Full length of the source file in samples.
    pub source_len_samples: u64,
    /// RMS of the fragment actually used (offset..offset+len), dBFS.
    pub fragment_rms_db: f64,
    /// Peak of the fragment actually used, dBFS.
    pub fragment_peak_db: f64,
}

/// Project sound report (corrective C3.4/A4): track and clip durations
/// against video length, RMS/peak of the fragments actually used, and
/// machine-readable warnings. `probe` is called lazily once per unique
/// asset clip; an `Err` becomes a per-clip `decode_error` and a `decode_failed`
/// warning without dropping the whole report. `probe_available` says whether a
/// decoder exists at all (without ffmpeg the metrics are null plus an
/// `ffmpeg_unavailable` warning, while document durations stay).
pub fn audio_report(
    project: &Project,
    scene_filter: Option<u64>,
    probe_available: bool,
    mut probe: impl FnMut(phantom_core::AssetId, u64, u64) -> Result<AudioProbe, String>,
) -> Result<Value, phantom_core::CoreError> {
    use phantom_core::audio::SAMPLE_RATE;
    let mut warnings: Vec<Value> = Vec::new();
    // Key is (asset, offset, len): fragments of one file differ, while
    // identical clips share a single probe.
    let mut probed = std::collections::BTreeMap::<(phantom_core::AssetId, u64, u64), Option<Result<AudioProbe, String>>>::new();
    if !probe_available {
        warnings.push(json!({
            "code": "ffmpeg_unavailable", "severity": "warning", "where": {},
            "why": "ffmpeg not found: source RMS/durations are unavailable, only document data is shown",
            "suggested_fix": "install ffmpeg",
        }));
    }

    let film_frames = film_len(&project.film);
    let film_seconds = frame_to_secs(film_frames);

    let mut tracks = Vec::new();
    let mut collect = |tracks_src: &[phantom_core::audio::AudioTrack], kind: &str, scene_id: Option<u64>, video_end: Frame, warnings: &mut Vec<Value>| {
        for track in tracks_src {
            let mut clips = Vec::new();
            let mut span_end: Frame = 0;
            for clip in &track.clips {
                let clip_end = clip.start + (clip.len_samples / SAMPLES_PER_FRAME as u64) as Frame;
                span_end = span_end.max(clip_end);
                if clip_end > video_end {
                    warnings.push(json!({
                        "code": "audio_beyond_video", "severity": "warning",
                        "where": {"track": track.id.0, "clip": clip.id.0, "scene": scene_id},
                        "why": format!(
                            "the clip ends at {:.2} s while video ends at {:.2} s: {:.2} s of audio will not reach the render",
                            frame_to_secs(clip_end), frame_to_secs(video_end), frame_to_secs(clip_end - video_end),
                        ),
                        "suggested_fix": "extend the scene/film range or trim the clip (len_secs)",
                    }));
                }
                let asset_id = clip.asset;
                let entry = probed.entry((asset_id, clip.offset_samples, clip.len_samples)).or_insert_with(|| {
                    if probe_available { Some(probe(asset_id, clip.offset_samples, clip.len_samples)) } else { None }
                });
                let (source_seconds, rms_db, peak_db, decode_error) = match entry {
                    Some(Ok(p)) => {
                        let source_end = clip.offset_samples + clip.len_samples;
                        if source_end > p.source_len_samples {
                            warnings.push(json!({
                                "code": "audio_beyond_source", "severity": "warning",
                                "where": {"track": track.id.0, "clip": clip.id.0, "asset": asset_id.0},
                                "why": format!(
                                    "fragment {:.2}..{:.2} s is beyond the file length {:.2} s: the tail will be silence",
                                    clip.offset_samples as f64 / SAMPLE_RATE as f64,
                                    source_end as f64 / SAMPLE_RATE as f64,
                                    p.source_len_samples as f64 / SAMPLE_RATE as f64,
                                ),
                                "suggested_fix": "reduce offset_secs/len_secs",
                            }));
                        }
                        if p.fragment_rms_db < QUIET_FRAGMENT_DB && clip.len_samples > 0 {
                            warnings.push(json!({
                                "code": "quiet_fragment", "severity": "warning",
                                "where": {"track": track.id.0, "clip": clip.id.0, "asset": asset_id.0},
                                "why": format!(
                                    "the fragment in use is almost silence: RMS {:.1} dB (offset {:.2} s)",
                                    p.fragment_rms_db,
                                    clip.offset_samples as f64 / SAMPLE_RATE as f64,
                                ),
                                "suggested_fix": "move offset_secs to a loud part of the track (the A4 case: music starts with silence)",
                            }));
                        }
                        (
                            json!(p.source_len_samples as f64 / SAMPLE_RATE as f64),
                            json!(p.fragment_rms_db),
                            json!(p.fragment_peak_db),
                            Value::Null,
                        )
                    }
                    Some(Err(e)) => {
                        warnings.push(json!({
                            "code": "decode_failed", "severity": "warning",
                            "where": {"track": track.id.0, "clip": clip.id.0, "asset": asset_id.0},
                            "why": format!("asset cannot be decoded: {e}"),
                            "suggested_fix": "check the asset file (asset_get)",
                        }));
                        (Value::Null, Value::Null, Value::Null, json!(e))
                    }
                    None => (Value::Null, Value::Null, Value::Null, Value::Null),
                };
                clips.push(json!({
                    "id": clip.id.0,
                    "asset": asset_id.0,
                    "start_frame": clip.start,
                    "start_seconds": frame_to_secs(clip.start),
                    "offset_seconds": clip.offset_samples as f64 / SAMPLE_RATE as f64,
                    "len_seconds": clip.len_samples as f64 / SAMPLE_RATE as f64,
                    "source_duration_seconds": source_seconds,
                    "fragment_rms_db": rms_db,
                    "fragment_peak_db": peak_db,
                    "decode_error": decode_error,
                }));
            }
            let span_seconds = frame_to_secs(span_end);
            let video_seconds = frame_to_secs(video_end);
            if span_end > video_end {
                warnings.push(json!({
                    "code": "audio_longer_than_video", "severity": "warning",
                    "where": {"track": track.id.0, "scene": scene_id},
                    "why": format!("audio {span_seconds:.2} s vs video {video_seconds:.2} s: {:.2} s lost", span_seconds - video_seconds),
                    "suggested_fix": "extend the video range or shorten/shift the audio clips",
                }));
            }
            tracks.push(json!({
                "id": track.id.0,
                "name": track.name,
                "kind": kind,
                "scene": scene_id,
                "volume": track.volume,
                "muted": track.muted,
                "span_end_frame": span_end,
                "span_seconds": span_seconds,
                "clips": clips,
            }));
        }
    };

    for scene in project.scenes.values() {
        if scene_filter.is_some_and(|f| f != scene.id.0) {
            continue;
        }
        collect(&scene.audio_tracks, "scene", Some(scene.id.0), scene.range.end, &mut warnings);
    }
    collect(&project.film_audio, "film", None, film_frames, &mut warnings);

    Ok(json!({
        "revision": project.revision,
        "scene": scene_filter,
        "sample_rate_hz": SAMPLE_RATE,
        "film": {"frames": film_frames, "seconds": film_seconds},
        "tracks": tracks,
        "warnings": warnings,
    }))
}

#[cfg(test)]
mod audio_tests {
    use super::*;
    use phantom_core::audio::{AudioClip, AudioClipId, AudioTrack, TrackId};
    use phantom_core::{Asset, AssetId, AssetKind, SceneId};

    fn report_project() -> (Project, SceneId) {
        let mut p = Project::new("audio");
        let sid = SceneId(1);
        let mut s = Scene {
            id: sid,
            name: "s".into(),
            range: FrameRange::new(0, 48).unwrap(), // 2 s of video
            background: None,
            legacy_camera: Some(Camera::default()),
            cameras: Default::default(),
            active_camera: SCENE_CAMERA,
            phantoms: Default::default(),
            layers_2d: vec![],
            audio_tracks: vec![],
            lighting: Default::default(),
            floor: None,
            objects: Default::default(),
            lights: vec![],
            ambient: Default::default(),
            sky: None,
            grade: None,
            fog: None,
            vfx: vec![],
            stage: phantom_core::Stage::default(),
        };
        let aid = AssetId(2);
        p.assets.insert(aid, Asset { id: aid, kind: AssetKind::Audio, path: "assets/music.mp3".into(), bounds: None });
        // A 6.34 s clip in a 2-second scene — the A4 case.
        s.audio_tracks.push(AudioTrack {
            id: TrackId(3),
            name: "music".into(),
            volume: 1.0,
            muted: false,
            clips: vec![AudioClip { id: AudioClipId(4), asset: aid, start: 0, offset_samples: 0, len_samples: 304_320, fade_in_samples: 0, fade_out_samples: 0 }],
        });
        s.normalize_legacy();
        p.scenes.insert(sid, s);
        (p, sid)
    }

    #[test]
    fn rms_peak_db_levels() {
        assert_eq!(rms_peak_db(&[]), (DB_FLOOR, DB_FLOOR));
        assert_eq!(rms_peak_db(&[[0.0, 0.0]; 100]), (DB_FLOOR, DB_FLOOR));
        let (rms, peak) = rms_peak_db(&[[1.0, 1.0]; 100]);
        assert!((rms - 0.0).abs() < 1e-9);
        assert!((peak - 0.0).abs() < 1e-9);
        let (rms, peak) = rms_peak_db(&[[0.5, -0.5]; 100]);
        assert!((rms - (-6.0206)).abs() < 1e-3);
        assert!((peak - (-6.0206)).abs() < 1e-3);
        // One quiet channel is not lost: peak is the max across channels.
        let (_, peak) = rms_peak_db(&[[0.0, 0.25]; 100]);
        assert!((peak - (-12.0412)).abs() < 1e-3);
    }

    #[test]
    fn audio_report_flags_quiet_and_overlong_fragment() {
        let (p, sid) = report_project();
        // Probe: a 360 s source, the fragment in use is silence (-74 dB as in A4).
        let report = audio_report(
            &p,
            None,
            true,
            |asset, offset, len| {
                assert_eq!(asset.0, 2);
                assert_eq!((offset, len), (0, 304_320));
                Ok(AudioProbe { source_len_samples: 360 * 48_000, fragment_rms_db: -74.0, fragment_peak_db: -70.0 })
            },
        )
        .unwrap();
        let codes: Vec<&str> = report["warnings"].as_array().unwrap().iter().map(|w| w["code"].as_str().unwrap()).collect();
        assert!(codes.contains(&"quiet_fragment"), "{codes:?}");
        assert!(codes.contains(&"audio_beyond_video"), "{codes:?}");
        assert!(codes.contains(&"audio_longer_than_video"), "{codes:?}");
        let clip = &report["tracks"][0]["clips"][0];
        assert_eq!(clip["source_duration_seconds"], 360.0);
        assert_eq!(clip["fragment_rms_db"], -74.0);
        assert_eq!(clip["len_seconds"], 304_320.0 / 48_000.0);
        // Scene filter: a foreign scene is excluded.
        let filtered = audio_report(&p, Some(sid.0 + 999), true, |_, _, _| unreachable!()).unwrap();
        assert_eq!(filtered["tracks"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn audio_report_handles_beyond_source_decode_error_and_no_decoder() {
        let (p, _) = report_project();
        // The fragment goes past a short source.
        let report = audio_report(&p, None, true, |_, _, _| {
            Ok(AudioProbe { source_len_samples: 48_000, fragment_rms_db: -20.0, fragment_peak_db: -8.0 })
        })
        .unwrap();
        let codes: Vec<&str> = report["warnings"].as_array().unwrap().iter().map(|w| w["code"].as_str().unwrap()).collect();
        assert!(codes.contains(&"audio_beyond_source"), "{codes:?}");
        assert!(!codes.contains(&"quiet_fragment"), "-20 dB is not silence: {codes:?}");
        // A decode error is per-clip; the report does not fail.
        let report = audio_report(&p, None, true, |_, _, _| Err("bad file".into())).unwrap();
        let codes: Vec<&str> = report["warnings"].as_array().unwrap().iter().map(|w| w["code"].as_str().unwrap()).collect();
        assert!(codes.contains(&"decode_failed"), "{codes:?}");
        assert_eq!(report["tracks"][0]["clips"][0]["decode_error"], "bad file");
        assert!(report["tracks"][0]["clips"][0]["fragment_rms_db"].is_null());
        // Without ffmpeg: metrics are null, document durations are still there.
        let report = audio_report(&p, None, false, |_, _, _| unreachable!()).unwrap();
        let codes: Vec<&str> = report["warnings"].as_array().unwrap().iter().map(|w| w["code"].as_str().unwrap()).collect();
        assert!(codes.contains(&"ffmpeg_unavailable"), "{codes:?}");
        assert!(!codes.contains(&"audio_beyond_source"), "without a decoder the source length is unknown: {codes:?}");
        assert!(report["tracks"][0]["clips"][0]["source_duration_seconds"].is_null());
        assert_eq!(report["tracks"][0]["clips"][0]["len_seconds"], 304_320.0 / 48_000.0);
    }

    #[test]
    fn audio_summary_marks_lost_seconds_without_decoding() {
        let (p, _) = report_project();
        let summary = audio_summary(&p);
        let scene = &summary["scenes"][0];
        assert_eq!(scene["video_seconds"], 2.0);
        assert!((scene["audio_seconds"].as_f64().unwrap() - 304_320.0 / 48_000.0 / 24.0 * 24.0).abs() < 0.05);
        assert!(scene["lost_seconds"].as_f64().unwrap() > 4.0);
        assert_eq!(summary["film"]["audio_seconds"], 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phantom_core::audio::{AudioClip, AudioClipId, AudioTrack, TrackId};
    use phantom_core::film::{ClipId, FilmClip};
    use phantom_core::project::{ActorPose, BackgroundSource, MotionKey, ObjectKey, Phantom, RigRef, SceneObject, TransformKey};
    use phantom_core::{Asset, AssetId, AssetKind, ObjectId, PhantomId, SceneId};

    /// A deliberately small backdrop: the default camera at (0,1.4,4) sees a
    /// 9 m gap to its plane, so 8x4 m leaves a coverage gap to measure.
    fn bg() -> Background {
        Background {
            image: BackgroundSource::Image(AssetId(1)),
            width: 8.0,
            height: 4.0,
            center: Vec3::new(0.0, 2.0, -5.0),
            fit: Default::default(),
        }
    }

    fn scene(id: u64, start: Frame, end: Frame) -> Scene {
        let mut s = Scene {
            id: SceneId(id),
            name: format!("s{id}"),
            range: FrameRange::new(start, end).unwrap(),
            background: None,
            legacy_camera: Some(Camera::default()),
            cameras: Default::default(),
            active_camera: SCENE_CAMERA,
            phantoms: Default::default(),
            layers_2d: vec![],
            audio_tracks: vec![],
            lighting: Default::default(),
            floor: None,
            objects: Default::default(),
            lights: vec![],
            ambient: Default::default(),
            sky: None,
            grade: None,
            fog: None,
            vfx: vec![],
            stage: phantom_core::Stage::default(),
        };
        // Camera 0 exists in every scene (Scene::new inserts it); a literal
        // built here gets it from the legacy field, exactly like a loaded
        // manifest.
        s.normalize_legacy();
        s.background = Some(bg());
        s
    }

    /// `camera_coverage` / `background_fit` for a bare scene: they need a
    /// project only for the camera's attachment, and these scenes have none.
    fn cam_cov(scene: &Scene, camera: Option<CameraId>, frame: Option<Frame>) -> Value {
        camera_coverage(&Project::new("t"), scene, camera, frame, DEFAULT_ASPECT)
    }

    fn bg_fit(scene: &Scene, camera: Option<CameraId>, frame: Option<Frame>, margin: f32) -> Value {
        background_fit(&Project::new("t"), scene, camera, frame, margin, DEFAULT_ASPECT)
    }

    fn phantom(id: u64, keys: Vec<(Frame, [f32; 3])>) -> Phantom {
        Phantom {
            id: PhantomId(id),
            name: format!("p{id}"),
            rig: RigRef::Builtin,
            motion: keys
                .into_iter()
                .map(|(frame, pos)| MotionKey { frame, position: pos.into(), pose: ActorPose::Stand, yaw: 0.0 })
                .collect(),
            outline_px: 2.0,
            scale: 1.0,
            material_patches: Vec::new(),
        }
    }

    fn codes(findings: &[Value]) -> Vec<&str> {
        findings.iter().map(|f| f["code"].as_str().unwrap()).collect()
    }

    // --- validate: every finding class ---

    #[test]
    fn validate_clean_scene_has_no_error_findings() {
        let mut p = Project::new("t");
        let mut s = scene(1, 0, 48);
        // Covering geometry: the camera looks at the screen plane from 6 m and
        // a 20x8 m screen closes both axes.
        s.cameras.get_mut(&SCENE_CAMERA).unwrap().camera.keys = vec![TransformKey::at(0, Vec3::new(0.0, 2.0, -11.0), std::f32::consts::PI, 0.0)];
        s.background = Some(Background { width: 20.0, height: 8.0, ..bg() });
        s.phantoms.insert(PhantomId(2), phantom(2, vec![(0, [0.0, 0.0, 0.0]), (47, [1.0, 0.0, 0.0])]));
        p.scenes.insert(SceneId(1), s);
        let findings = validate(&p, None, None).unwrap();
        assert!(!findings.iter().any(|f| f["severity"] == "error"), "{findings:?}");
        assert!(!codes(&findings).contains(&"screen_fov_gap"), "{findings:?}");
        assert!(!codes(&findings).contains(&"motion_gap"), "{findings:?}");
    }

    #[test]
    fn validate_triggers_each_finding_class() {
        let mut p = Project::new("t");
        // 1: invalid screen geometry + empty scene.
        let mut s1 = scene(1, 0, 48);
        s1.background = Some(Background { width: 0.0, ..bg() });
        p.scenes.insert(SceneId(1), s1);
        // 2: FOV not covered (default geometry, camera at (0,1.4,4)),
        //    motion holes, a phantom without motion, audio out of range and length.
        let mut s2 = scene(2, 0, 48);
        s2.phantoms.insert(PhantomId(3), phantom(3, vec![(10, [0.0, 0.0, 0.0]), (20, [0.0, 0.0, 0.0])]));
        s2.phantoms.insert(PhantomId(4), phantom(4, vec![]));
        let mut track = AudioTrack::new(TrackId(5), "a");
        track.clips.push(AudioClip { id: AudioClipId(6), asset: AssetId(7), start: -12, offset_samples: 0, len_samples: 48_000, fade_in_samples: 0, fade_out_samples: 0 });
        s2.audio_tracks.push(track);
        let mut short = AudioTrack::new(TrackId(8), "short");
        short.clips.push(AudioClip { id: AudioClipId(9), asset: AssetId(7), start: 0, offset_samples: 0, len_samples: 2000, fade_in_samples: 0, fade_out_samples: 0 });
        s2.audio_tracks.push(short);
        p.scenes.insert(SceneId(2), s2);
        // Film: a clip outside the scene range; a film track longer than the film.
        p.film.push(FilmClip { id: ClipId(10), scene: SceneId(2), range: FrameRange::new(0, 96).unwrap(), camera: None });
        let mut film_track = AudioTrack::new(TrackId(11), "film");
        film_track.clips.push(AudioClip { id: AudioClipId(12), asset: AssetId(7), start: 0, offset_samples: 0, len_samples: 200 * 48_000 , fade_in_samples: 0, fade_out_samples: 0 });
        p.film_audio.push(film_track);
        // Missing asset file (dir exists, the file does not).
        p.assets.insert(AssetId(7), Asset { id: AssetId(7), kind: AssetKind::Audio, path: "missing.wav".into(), bounds: None });

        let dir = std::env::temp_dir().join(format!("phantom-analysis-validate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let findings = validate(&p, Some(&dir), None).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let codes = codes(&findings);
        for expected in [
            "bg_geometry_invalid",
            "screen_fov_gap",
            "motion_gap",
            "phantom_without_motion",
            "scene_empty",
            "audio_out_of_range",
            "audio_length_mismatch",
            "film_clip_out_of_scene",
            "asset_missing",
        ] {
            assert!(codes.contains(&expected), "{expected} missing from {codes:?}");
        }
        // ok=false: there is error severity (bg_geometry_invalid, film_clip_out_of_scene, asset_missing).
        let report = validate_report(&p, None, None).unwrap();
        assert_eq!(report["ok"], false);
    }

    #[test]
    fn validate_checks_film_assembly_cameras_gaps_and_overlaps() {
        let mut p = Project::new("t");
        let mut s = scene(1, 0, 96);
        let assembled = CameraId(2);
        let forgotten = CameraId(3);
        s.cameras.insert(assembled, SceneCamera { id: assembled, name: "kept".into(), range: FrameRange::new(0, 48).unwrap(), camera: Camera::default(), target: None });
        s.cameras.insert(forgotten, SceneCamera { id: forgotten, name: "lost".into(), range: FrameRange::new(48, 96).unwrap(), camera: Camera::default(), target: None });
        p.scenes.insert(SceneId(1), s);
        // The film assembles only the first camera and repeats part of its own frames.
        p.film.push(FilmClip { id: ClipId(4), scene: SceneId(1), range: FrameRange::new(0, 48).unwrap(), camera: Some(assembled) });
        p.film.push(FilmClip { id: ClipId(5), scene: SceneId(1), range: FrameRange::new(24, 48).unwrap(), camera: Some(assembled) });

        let findings = validate(&p, None, Some(1)).unwrap();
        let found = codes(&findings);
        assert!(found.contains(&"camera_not_in_film"), "{found:?}");
        assert!(found.contains(&"film_coverage_gap"), "frames 48..96 are not shown: {found:?}");
        assert!(found.contains(&"film_overlap"), "frames 24..48 are shown twice: {found:?}");
        let gap = findings.iter().find(|f| f["code"] == "film_coverage_gap").unwrap();
        assert_eq!(gap["where"]["unshown"]["start"], 48);
        assert_eq!(gap["where"]["unshown"]["end"], 96);
        let overlap = findings.iter().find(|f| f["code"] == "film_overlap").unwrap();
        assert_eq!(overlap["where"]["overlap"]["start"], 24);
        // An edit without camera references must not produce a false camera_not_in_film.
        let mut scene_level = p.clone();
        for clip in &mut scene_level.film { clip.camera = None; }
        let scene_level_findings = validate(&scene_level, None, Some(1)).unwrap();
        let scene_level_codes = codes(&scene_level_findings);
        assert!(!scene_level_codes.contains(&"camera_not_in_film"), "{scene_level_codes:?}");
    }

    #[test]
    fn validate_deterministic_order_and_scene_filter() {
        let mut p = Project::new("t");
        let mut s1 = scene(1, 0, 48); // screen_fov_gap on the default camera
        s1.phantoms.insert(PhantomId(2), phantom(2, vec![(5, [0.0; 3])])); // motion_gap
        p.scenes.insert(SceneId(1), s1);
        p.scenes.insert(SceneId(2), scene(2, 0, 24)); // screen_fov_gap
        let a = validate(&p, None, None).unwrap();
        let b = validate(&p, None, None).unwrap();
        assert_eq!(a, b, "deterministic output");
        let keys: Vec<(u64, &str, i64)> = a
            .iter()
            .map(|f| {
                (
                    f["where"]["scene"].as_u64().unwrap_or(0),
                    f["code"].as_str().unwrap(),
                    f["where"]["frame"].as_i64().unwrap_or(-1),
                )
            })
            .collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted, "order: scene, code, frame: {keys:?}");
        // Scene filter: only its findings; a nonexistent scene — not_found.
        let only2 = validate(&p, None, Some(2)).unwrap();
        assert!(only2.iter().all(|f| f["where"]["scene"].as_u64().unwrap_or(2) == 2));
        assert!(validate(&p, None, Some(99)).is_err());
    }

    #[test]
    fn validate_reports_a_motion_gap_but_not_a_covered_scene() {
        let mut p = Project::new("t");
        let mut s = scene(1, 0, 48);
        // Keys only in the middle: the scene ends are not covered.
        s.phantoms.insert(PhantomId(2), phantom(2, vec![(40, [0.0; 3])]));
        p.scenes.insert(SceneId(1), s);
        let findings = validate(&p, None, None).unwrap();
        let found = codes(&findings);
        assert!(found.contains(&"motion_gap"), "{found:?}");
        assert!(!found.contains(&"phantom_without_motion"), "{found:?}");

        // Keys at both ends: no gap, no complaint.
        let mut p = Project::new("t");
        let mut s = scene(1, 0, 48);
        s.phantoms.insert(PhantomId(2), phantom(2, vec![(0, [0.0; 3]), (47, [1.0; 3])]));
        p.scenes.insert(SceneId(1), s);
        let findings = validate(&p, None, None).unwrap();
        let found = codes(&findings);
        assert!(!found.contains(&"motion_gap"), "{found:?}");
    }

    // --- camera_coverage: known geometry ---

    /// The screen seen from the camera is a plane: its angular span is exactly
    /// `atan((width/2) / distance)` on each side. Pin that number.
    #[test]
    fn coverage_span_is_the_plane_seen_from_the_camera() {
        let mut s = scene(1, 0, 48);
        // The camera sits on the screen's axis 6 m in front of it.
        s.cameras.get_mut(&SCENE_CAMERA).unwrap().camera = Camera {
            keys: vec![TransformKey::at(0, Vec3::new(0.0, 2.0, -11.0), std::f32::consts::PI, 0.0)],
            focal_mm: 35.0,
        };
        s.background = Some(Background { width: 12.0, height: 6.0, ..bg() });
        let out = cam_cov(&s, None, Some(0));
        let bg = &out["background"];
        let span = &bg["span"]["horizontal_deg"];
        // atan(6/6) = 45° on each side.
        assert!((span["min"].as_f64().unwrap() + 45.0).abs() < 0.2, "{span}");
        assert!((span["max"].as_f64().unwrap() - 45.0).abs() < 0.2, "{span}");
        // hfov/2 (35 mm, 16:9) ≈ 31.3° < 45°, height 6 m: el = atan2(3,6) = 26.6° > vfov/2 ≈ 18.9°.
        assert_eq!(bg["covers_frame"], true, "{bg}");
        assert!(out["findings"].as_array().unwrap().is_empty());
    }

    #[test]
    fn coverage_small_screen_reports_gap() {
        // The default staging: camera (0,1.4,4), an 8x4 m screen at (0,2,−5).
        // Its edge sits at atan2(4, 9) ≈ 24.0°, hfov/2 ≈ 31.3° → a gap on both
        // sides, and the screen's 4 m height does not reach the frame either.
        let s = scene(1, 0, 48);
        let out = cam_cov(&s, None, Some(0));
        let bg = &out["background"];
        assert_eq!(bg["covers_frame"], false);
        let max = bg["span"]["horizontal_deg"]["max"].as_f64().unwrap();
        assert!((max - 24.0).abs() < 1.0, "max azimuth {max}");
        assert!(bg["gap_deg"]["left"].as_f64().unwrap() > 4.0);
        assert!(bg["gap_deg"]["right"].as_f64().unwrap() > 4.0);
        assert_eq!(out["findings"][0]["code"], "screen_fov_gap");
        // The fix names the width the frame needs, not an arc.
        let fix = out["findings"][0]["suggested_fix"].as_str().unwrap();
        assert!(fix.contains("width >="), "{fix}");
    }

    #[test]
    fn coverage_in_frame_points_and_absent_background() {
        let mut s = scene(1, 0, 48);
        s.background = None;
        s.phantoms.insert(PhantomId(2), phantom(2, vec![(0, [0.0, 0.0, -5.0]), (47, [0.0, 0.0, -5.0])]));
        s.objects.insert(
            ObjectId(3),
            SceneObject {
                id: ObjectId(3),
                name: "o".into(),
                asset: AssetId(1), builtin: None,
                keys: vec![ObjectKey { frame: 0, position: Vec3::new(100.0, 0.0, 0.0), yaw: 0.0, pitch: 0.0, roll: 0.0, scale: Vec3::ONE }],
                attach: None,
                material_patches: Vec::new(),
                scatter: None,
            },
        );
        let out = cam_cov(&s, None, Some(0));
        assert_eq!(out["background"], Value::Null);
        assert_eq!(out["findings"][0]["code"], "background_absent");
        assert_eq!(out["phantoms"][0]["in_frame"], true, "phantom on the axis in front of the camera");
        assert_eq!(out["objects"][0]["in_frame"], false, "object 100 m to the side is out of frame");
        // The default frame is the start of the range.
        assert_eq!(out["frame"], 0);
    }

    #[test]
    fn validate_warns_only_when_all_object_roots_miss_all_cameras() {
        let mut p = Project::new("t");
        let mut s = scene(1, 0, 48);
        s.objects.insert(
            ObjectId(3),
            SceneObject {
                id: ObjectId(3), name: "off-camera".into(), asset: AssetId(1), builtin: None,
                keys: vec![ObjectKey { frame: 0, position: Vec3::new(10.0, 0.0, -5.0), yaw: 0.0, pitch: 0.0, roll: 0.0, scale: Vec3::ONE }],
                attach: None, material_patches: Vec::new(),
                scatter: None,
            },
        );
        p.scenes.insert(SceneId(1), s);
        let findings = validate(&p, None, Some(1)).unwrap();
        let finding = findings.iter().find(|f| f["code"] == "objects_out_of_frame").expect("an out-of-frame staging must be warned about");
        assert_eq!(finding["severity"], "warning");
        assert_eq!(finding["where"]["objects_in_frame"], 0);
        assert_eq!(validate_report(&p, None, Some(1)).unwrap()["ok"], true, "a warning does not turn validate into an error");

        p.scenes.get_mut(&SceneId(1)).unwrap().objects.get_mut(&ObjectId(3)).unwrap().keys[0].position = Vec3::new(0.0, 0.0, -5.0);
        let findings = validate(&p, None, Some(1)).unwrap();
        let codes = codes(&findings);
        assert!(!codes.contains(&"objects_out_of_frame"), "{codes:?}");
    }

    #[test]
    fn validate_warns_before_skinned_shadow_prepass() {
        let mut p = Project::new("t");
        let mut s = scene(1, 0, 48);
        s.phantoms.insert(PhantomId(2), phantom(2, vec![(0, [0.0, 0.0, -5.0])]));
        s.lighting.key.shadows = true;
        p.scenes.insert(SceneId(1), s);
        let findings = validate(&p, None, Some(1)).unwrap();
        let codes = codes(&findings);
        assert!(codes.contains(&"skinned_shadows_unsupported"), "{codes:?}");
    }

    #[test]
    fn background_fit_scene_scope_covers_every_camera() {
        // corrective_2 K0.1: without camera the fit must cover ALL scene
        // cameras, not one. The second camera looks the opposite way.
        let mut s = scene(1, 0, 96);
        let forward = CameraId(2);
        let backward = CameraId(3);
        s.cameras.insert(
            forward,
            SceneCamera {
                id: forward, name: "fwd".into(), range: FrameRange::new(0, 48).unwrap(),
                camera: Camera { focal_mm: 35.0, keys: vec![TransformKey::at(0, Vec3::new(0.0, 2.0, -11.0), std::f32::consts::PI, 0.0)] },
                target: None,
            },
        );
        s.cameras.insert(
            backward,
            SceneCamera {
                id: backward, name: "back".into(), range: FrameRange::new(48, 96).unwrap(),
                camera: Camera { focal_mm: 35.0, keys: vec![TransformKey::at(48, Vec3::new(0.0, 2.0, 11.0), 0.0, 0.0)] },
                target: None,
            },
        );

        let scene_fit = bg_fit(&s, None, None, 5.0);
        assert_eq!(scene_fit["scope"], "scene");
        assert_eq!(scene_fit["camera_count"], 3, "the scene's own camera + two named ones");
        assert_eq!(scene_fit["size_covers"], true);
        // The suggested size covers each camera on its own.
        let width = scene_fit["suggested"]["width"].as_f64().unwrap() as f32;
        let height = scene_fit["suggested"]["height"].as_f64().unwrap() as f32;
        let mut fixed = s.clone();
        fixed.background = Some(Background { width, height, ..bg() });
        for (camera, frame) in [(0, 0), (forward.0, 0), (backward.0, 48)] {
            let cov = cam_cov(&fixed, Some(CameraId(camera)), Some(frame));
            assert_eq!(cov["background"]["covers_frame"], true, "camera {camera:?} is not covered: {}", cov["background"]);
        }
        // A fit for a single camera is never wider than a fit for the whole scene.
        let camera_fit = bg_fit(&s, Some(backward), None, 5.0);
        assert_eq!(camera_fit["scope"], "camera");
        assert!(camera_fit["suggested"]["width"].as_f64().unwrap() <= scene_fit["suggested"]["width"].as_f64().unwrap());
        // The backward camera is 16 m from the plane and sets the size: it is the
        // binding sample.
        assert_eq!(scene_fit["camera_id"].as_u64(), Some(backward.0));
    }

    // --- background_fit: roundtrip ---

    #[test]
    fn background_fit_roundtrip_covers_frame() {
        let s = scene(1, 0, 48); // default staging with a coverage gap
        let fit = bg_fit(&s, None, None, 5.0);
        assert_eq!(fit["fits"], false, "the current geometry does not cover the frame");
        assert_eq!(fit["size_covers"], true);
        let suggested = &fit["suggested"];
        let width = suggested["width"].as_f64().unwrap() as f32;
        let height = suggested["height"].as_f64().unwrap() as f32;
        assert!(width > 0.0 && height > 0.0);
        // The camera is 9 m from the plane: with a 5° margin the width must
        // exceed 2·9·tan(36.28°) ≈ 13.2 m.
        assert!(width >= 13.0, "width {width}");
        assert!(fit["panorama_aspect"].as_f64().unwrap() > 0.0);

        // Apply the suggestion: coverage is closed.
        let mut fixed = s.clone();
        fixed.background = Some(Background { width, height, ..bg() });
        let out = cam_cov(&fixed, None, Some(0));
        assert_eq!(out["background"]["covers_frame"], true, "{}", out["background"]);

        // Geometry that already covers: fits=true, and the fit does not ask for
        // more than the margin allows.
        let fit2 = bg_fit(&fixed, None, Some(0), 5.0);
        assert_eq!(fit2["fits"], true);
        assert!(fit2["delta"]["width"].as_f64().unwrap() <= 0.01, "{}", fit2["delta"]);
        assert!(fit2["delta"]["height"].as_f64().unwrap() <= 0.01, "{}", fit2["delta"]);
    }

    #[test]
    fn background_fit_without_background_uses_defaults() {
        let mut s = scene(1, 0, 48);
        s.background = None;
        let fit = bg_fit(&s, None, None, 5.0);
        assert_eq!(fit["current"], Value::Null);
        assert_eq!(fit["fits"], Value::Null);
        assert_eq!(fit["suggested"]["center"], json!([0.0, 2.0, -5.0]));
        assert!(fit["suggested"]["width"].as_f64().unwrap() > 0.0);
        assert!(fit["suggested"]["height"].as_f64().unwrap() > 0.0);
    }

    /// A camera that never faces the screen plane cannot be fixed by size: the
    /// fit says so instead of inventing a number.
    #[test]
    fn background_fit_reports_when_no_size_helps() {
        let mut s = scene(1, 0, 48);
        // The camera stands BEHIND the screen plane and looks away from it.
        s.cameras.get_mut(&SCENE_CAMERA).unwrap().camera = Camera {
            keys: vec![TransformKey::at(0, Vec3::new(0.0, 2.0, -11.0), 0.0, 0.0)],
            focal_mm: 35.0,
        };
        let fit = bg_fit(&s, None, Some(0), 5.0);
        assert_eq!(fit["size_covers"], false);
        assert_eq!(fit["suggested"]["width"].as_f64().unwrap(), 0.0);
        assert_eq!(fit["suggested"]["height"].as_f64().unwrap(), 0.0);
        assert!(fit["what_fits"].as_str().unwrap().contains("move the screen"));
    }
}

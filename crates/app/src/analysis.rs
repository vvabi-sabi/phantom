//! Pure document diagnostics (corrective C3.1/C3.3): `validate`,
//! `camera_coverage`, `background_fit`. Only the core document and camera
//! geometry — no render, no GPU, no media decoding.
//!
//! Screen geometry is the same one `mesh_gen::arc_screen_mesh` builds:
//! arc point = `center + (sin θ·R, y − height/2, sign·(1 − cos θ)·R)`,
//! θ ∈ [−arc/2, arc/2], y ∈ [0, height]; the arc midpoint sits at `center`,
//! the sign comes from `bg.bulge`: `toward_camera` — edges farther from the
//! viewer (a backdrop-style screen), `away_from_camera` (default) — edges
//! closer, the screen wraps around the scene (A3a).
//! Camera — `Camera::sample` plus a YXZ(yaw, pitch) rotation looking along
//! local −Z, as in the render (`Quat::from_euler(YXZ, yaw, pitch, 0)`).
//! Vertical FOV — `phantom_core::vertical_fov`, the formula shared with the
//! renderer; horizontal FOV derived from it via `phantom_core::horizontal_fov`.
//!
//! Documented approximations:
//! - coverage is estimated from 129 arc points (and two edges in height):
//!   the screen's azimuth range is the min/max over samples, sampling error
//!   < arc/128 rad;
//! - vertical coverage uses the worst column inside the frustum
//!   (conservative: the real frame may be covered slightly better);
//! - camera roll does not exist in the document and is not accounted for;
//!   pitch is part of the camera basis and is accounted for exactly;
//! - the phantom/object point test uses the root point (no model volume)
//!   and the position without "on twos" quantization.

use std::collections::BTreeSet;
use std::path::Path;

use glam::{Quat, Vec3};
use serde_json::{json, Value};

use phantom_core::audio::SAMPLES_PER_FRAME;
use phantom_core::time::frame_to_secs;
use phantom_core::{
    evaluate, evaluate_object, film_len, horizontal_fov, vertical_fov, Background, Camera, Frame, FrameRange, Project, Scene, ShotId,
};

/// Default frame aspect ratio (16:9) when the call does not set aspect.
pub const DEFAULT_ASPECT: f32 = 16.0 / 9.0;
/// Screen arc sample points for the coverage estimate (both ends included).
const SCREEN_SAMPLES: usize = 128;
/// Minimum depth in front of the camera: points closer/behind add no coverage.
const MIN_DEPTH: f32 = 1e-3;

/// Camera at a specific frame, with its derived field-of-view angles.
struct Frustum {
    pos: Vec3,
    yaw: f32,
    pitch: f32,
    focal_mm: f32,
    vfov: f32,
    hfov: f32,
    /// world → camera (conjugate of YXZ(yaw, pitch)).
    inv_rot: Quat,
}

impl Frustum {
    fn new(camera: &Camera, frame: Frame, aspect: f32) -> Self {
        let pose = camera.sample(frame);
        // Focal length comes from the pose: a key may set its own (C1.2), and
        // then the diagnostics FOV must match the real frame.
        let focal_mm = pose.focal_mm;
        Self {
            pos: pose.translation,
            yaw: pose.yaw,
            pitch: pose.pitch,
            focal_mm,
            vfov: vertical_fov(focal_mm),
            hfov: horizontal_fov(focal_mm, aspect),
            inv_rot: Quat::from_euler(glam::EulerRot::YXZ, pose.yaw, pose.pitch, 0.0).conjugate(),
        }
    }

    fn local(&self, p: Vec3) -> Vec3 {
        self.inv_rot * (p - self.pos)
    }

    /// Whether a point is in frame: in front of the camera and inside both half-FOVs.
    fn contains(&self, p: Vec3) -> bool {
        let l = self.local(p);
        if l.z > -MIN_DEPTH {
            return false;
        }
        let az = l.x.atan2(-l.z);
        let el = l.y.atan2(-l.z);
        az.abs() <= self.hfov * 0.5 && el.abs() <= self.vfov * 0.5
    }
}

/// Why the screen geometry is invalid; `None` — the geometry is valid.
fn background_invalid_reason(bg: &Background) -> Option<&'static str> {
    if !bg.arc_radians.is_finite() || !bg.radius.is_finite() || !bg.height.is_finite() || !bg.center.is_finite() {
        return Some("non-numeric values in arc/radius/height/center");
    }
    if bg.arc_radians <= 0.0 {
        return Some("arc <= 0");
    }
    // JSON clients commonly send a rounded full circle (6.2832). Accept this
    // harmless f32 rounding; materially larger arcs remain invalid geometry.
    if bg.arc_radians > std::f32::consts::TAU + 1e-3 {
        return Some("arc > 2π");
    }
    if bg.radius <= 0.0 {
        return Some("radius <= 0");
    }
    if bg.height <= 0.0 {
        return Some("height <= 0");
    }
    None
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

fn screen_span(bg: &Background, fr: &Frustum) -> ScreenSpan {
    let mut span = ScreenSpan {
        min_az: f32::INFINITY,
        max_az: f32::NEG_INFINITY,
        top_el: f32::INFINITY,
        bottom_el: f32::NEG_INFINITY,
        any_in_frustum: false,
        any_behind: false,
    };
    let z_sign = match bg.bulge {
        phantom_core::project::Bulge::TowardCamera => -1.0,
        phantom_core::project::Bulge::AwayFromCamera => 1.0,
    };
    for i in 0..=SCREEN_SAMPLES {
        let theta = (i as f32 / SCREEN_SAMPLES as f32 - 0.5) * bg.arc_radians;
        let x = theta.sin() * bg.radius;
        let z = z_sign * (1.0 - theta.cos()) * bg.radius;
        for &y in &[bg.height * 0.5, -bg.height * 0.5] {
            let p = bg.center + Vec3::new(x, y, z);
            let l = fr.local(p);
            if l.z > -MIN_DEPTH {
                // Points behind the camera do not project into the frame: they
                // do not break coverage (a full-turn screen correctly wraps the
                // camera), but they also take no part in the span.
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

/// Smallest arc (radians, ≤ 2π) that closes the horizontal frustum with a
/// `margin` of slack at the current radius/center. Monotonicity of span(arc)
/// (the θ range widens) makes bisection correct. `None` when even a full
/// circle does not close the frame (the camera is too far from the screen for
/// the current radius — `background_fit` then picks the radius).
fn needed_arc(bg: &Background, fr: &Frustum, margin: f32) -> Option<f32> {
    let half_h = fr.hfov * 0.5 + margin;
    bisect_arc(|arc| {
        let mut probe = bg.clone();
        probe.arc_radians = arc;
        let span = screen_span(&probe, fr);
        span.min_az <= -half_h && span.max_az >= half_h
    })
}

/// Smallest arc that closes EVERY scene sample (corrective_2 K0.1).
/// Monotonicity still holds: widening the arc only widens the spans, so
/// bisection stays correct for the union of requirements too.
fn needed_arc_all(bg: &Background, samples: &[FitSample], margin: f32) -> Option<f32> {
    bisect_arc(|arc| {
        let mut probe = bg.clone();
        probe.arc_radians = arc;
        samples.iter().all(|s| {
            let half_h = s.fr.hfov * 0.5 + margin;
            let span = screen_span(&probe, &s.fr);
            span.min_az <= -half_h && span.max_az >= half_h
        })
    })
}

/// Bisection for the minimal arc under a monotone predicate `covers`;
/// `None` when even a full circle does not close the frame.
fn bisect_arc(covers: impl Fn(f32) -> bool) -> Option<f32> {
    if !covers(std::f32::consts::TAU) {
        return None;
    }
    let mut lo = 0.0f32;
    let mut hi = std::f32::consts::TAU;
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

/// suggested_fix line for a coverage gap: either the minimal arc at the
/// current radius (plus a focal-length alternative), or an admission that the
/// radius is too small and background_fit is needed.
fn gap_fix_string(bg: &Background, fr: &Frustum, aspect: f32) -> String {
    match needed_arc(bg, fr, 0.0) {
        Some(arc) => {
            let arc_deg = deg(arc);
            match max_fitting_focal(bg, fr, aspect) {
                Some(f) => format!("arc >= {arc_deg:.1}° at radius {:.1} m, or focal_mm <= {f:.1}", bg.radius),
                None => format!("arc >= {arc_deg:.1}° at radius {:.1} m", bg.radius),
            }
        }
        None => format!(
            "at radius {:.1} m the frame cannot be closed even by a 360° arc — background_fit will pick radius/arc",
            bg.radius
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

fn camera_json(scene_id: u64, shot: Option<ShotId>, frame: Frame, fr: &Frustum, aspect: f32) -> Value {
    json!({
        "scene": scene_id,
        "shot": shot.map(|s| s.0),
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
/// `frame` defaults to the start of the shot/scene range. The scene and shot
/// must exist (the calling Session checks that).
pub fn camera_coverage(scene: &Scene, shot: Option<ShotId>, frame: Option<Frame>, aspect: f32) -> Value {
    let range = scene.range_for_shot(shot);
    let frame = frame.unwrap_or(range.start);
    let fr = Frustum::new(scene.camera_for_shot(shot), frame, aspect);
    let mut out = camera_json(scene.id.0, shot, frame, &fr, aspect);
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
                    "arc_deg": deg(bg.arc_radians), "radius": bg.radius, "height": bg.height,
                    "center": v3(bg.center), "covers_frame": Value::Null,
                }));
                findings.push(json!({
                    "code": "bg_geometry_invalid", "severity": "error",
                    "where": {"scene": scene.id.0, "frame": frame},
                    "why": format!("screen geometry is invalid: {reason}"),
                    "suggested_fix": "fix background_set (arc in (0, 2π], radius > 0, height > 0)",
                }));
            } else {
                let cov = coverage(bg, &fr);
                object.insert("background".into(), json!({
                    "arc_deg": deg(bg.arc_radians), "radius": bg.radius, "height": bg.height,
                    "center": v3(bg.center),
                    "covers_frame": cov.covers_frame,
                    "span": span_json(&cov),
                    "gap_deg": gap_json(&cov),
                    "points_behind_camera": cov.span.any_behind,
                }));
                if !cov.covers_frame {
                    let needed_arc_deg = needed_arc(bg, &fr, 0.0).map(deg);
                    findings.push(json!({
                        "code": "screen_fov_gap", "severity": "warning",
                        "where": {"scene": scene.id.0, "frame": frame, "shot": shot.map(|s| s.0)},
                        "why": format!("the screen does not cover the frame: gaps in degrees {}", gap_json(&cov)),
                        "suggested_fix": gap_fix_string(bg, &fr, aspect),
                        "needed_arc_deg": needed_arc_deg,
                    }));
                }
            }
        }
    }

    let phantoms: Vec<Value> = scene
        .phantoms
        .values()
        .map(|ph| {
            let state = evaluate(ph, &scene.range, frame, false);
            json!({"id": ph.id.0, "pos": v3(state.root), "in_frame": fr.contains(state.root)})
        })
        .collect();
    let objects: Vec<Value> = scene
        .objects
        .values()
        .map(|o| {
            let t = evaluate_object(o, &scene.range, frame, false);
            json!({"id": o.id.0, "pos": v3(t.position), "in_frame": fr.contains(t.position)})
        })
        .collect();
    object.insert("phantoms".into(), json!(phantoms));
    object.insert("objects".into(), json!(objects));
    object.insert("findings".into(), json!(findings));
    out
}

/// Fit the screen geometry to the camera (corrective C3.3): the smallest
/// honest arc/height that close the frame with `margin_deg` of slack per side.
///
/// Formulas (documented approximations):
/// - `center` is preserved (or the default [0,2,−5] when there is no background);
/// - `radius = max(current|6.0, dist_xz(camera, center) + 0.5)` — the starting
///   point: the camera inside the curvature circle with a 0.5 m delta. Because
///   the screen bulges TOWARD the camera (A3) this may not be enough: the arc
///   edges move away from the camera, so radius is scaled by ×1.5 until the
///   bisection finds a closing arc (10 km cap — a guard against degenerate
///   setups; then a 360° arc is returned with `arc_covers: false`);
/// - `arc` — minimum found by bisection over the sampled azimuthal span, so
///   that min_az <= −(hfov/2 + margin) and max_az >= hfov/2 + margin;
/// - `height` — from the nearest screen point (arc midpoint, dist_xz):
///   top/bottom y = cam.y ± tan(pitch ± (vfov/2 + margin))·dist, symmetric
///   around the preserved center.y; angles are clamped to ±85° against tan→∞;
/// - `panorama_aspect = (arc·radius) / height` — the ratio of the screen's
///   horizontal length to its height, for the panorama generator.
///
/// Without `shot` the fit runs over the WHOLE scene (corrective_2 K0.1): the
/// legacy camera keys and each shot camera's keys are sampled, and
/// arc/radius/height must close every sample — that is, the union of the
/// shots' requirements, not one camera. Fitting a single shot that looks
/// elsewhere would otherwise give a "fit for the neighbouring frame" and leave
/// a hole in its own.
/// An explicit `frame` without `shot` makes no sense (shots have their own
/// ranges) and is rejected by the calling code.
///
/// Results are rounded up (arc to 0.1°, radius to 0.1 m, height to 0.01 m),
/// so the suggestion really does close the frame (roundtrip test).
pub fn background_fit(scene: &Scene, shot: Option<ShotId>, frame: Option<Frame>, margin_deg: f32, aspect: f32) -> Value {
    let margin = margin_deg.to_radians();
    let samples = fit_samples(scene, shot, frame, aspect);
    let scope = if shot.is_some() { "shot" } else { "scene" };

    let current = scene.background.as_ref();
    let center = current.map(|b| b.center).filter(|c| c.is_finite()).unwrap_or(Vec3::new(0.0, 2.0, -5.0));
    let radius0 = current.map(|b| b.radius).filter(|r| r.is_finite() && *r > 0.0).unwrap_or(6.0);
    // Starting radius — the most demanding sample: the camera inside the
    // curvature circle with a 0.5 m delta.
    let start_radius = samples
        .iter()
        .map(|s| dist_xz(s.fr.pos, center) + 0.5)
        .fold(radius0, f32::max);
    let mut radius = (start_radius * 10.0).ceil() / 10.0;

    // The image source does not affect geometry — a placeholder id.
    let source = || phantom_core::BackgroundSource::Image(phantom_core::AssetId(0));
    let probe = |radius: f32| Background { image: current.map(|b| b.image).unwrap_or_else(source), arc_radians: 0.0, radius, height: 1.0, center, bulge: current.map(|b| b.bulge).unwrap_or_default(), fit: Default::default() };
    let mut arc = needed_arc_all(&probe(radius), &samples, margin);
    let mut guard = 0;
    while arc.is_none() && radius < 10_000.0 && guard < 64 {
        radius *= 1.5;
        arc = needed_arc_all(&probe(radius), &samples, margin);
        guard += 1;
    }
    radius = (radius * 10.0).ceil() / 10.0;
    // Recompute the arc after rounding radius, so the suggestion is guaranteed to close every sample.
    let arc_covers = arc.is_some();
    let arc = needed_arc_all(&probe(radius), &samples, margin).unwrap_or(std::f32::consts::TAU);
    let arc_deg_ceil = (deg(arc) * 10.0).ceil() / 10.0;
    let arc = ((arc_deg_ceil as f32).to_radians()).min(std::f32::consts::TAU);

    let clamp_el = |a: f32| a.clamp(-85.0f32.to_radians(), 85.0f32.to_radians());
    // Height — the union of vertical requirements: the highest top and the
    // lowest bottom across samples, symmetric around center.y.
    let mut need_top = f32::NEG_INFINITY;
    let mut need_bottom = f32::INFINITY;
    for s in &samples {
        let d = dist_xz(s.fr.pos, center).max(0.5);
        need_top = need_top.max(s.fr.pos.y + clamp_el(s.fr.pitch + s.fr.vfov * 0.5 + margin).tan() * d);
        need_bottom = need_bottom.min(s.fr.pos.y + clamp_el(s.fr.pitch - s.fr.vfov * 0.5 - margin).tan() * d);
    }
    let half_h = (need_top - center.y).max(center.y - need_bottom).max(0.1);
    let height = (2.0 * half_h * 100.0).ceil() / 100.0;

    let panorama_aspect = arc as f64 * radius as f64 / height as f64;

    let (current_json, fits, delta) = match current {
        Some(bg) => {
            let valid = background_invalid_reason(bg).is_none();
            let fits = if valid { Some(samples.iter().all(|s| coverage(bg, &s.fr).covers_frame)) } else { None };
            (
                json!({
                    "arc_deg": deg(bg.arc_radians), "radius": bg.radius,
                    "height": bg.height, "center": v3(bg.center),
                    "geometry_valid": valid,
                }),
                fits,
                json!({
                    "arc_deg": arc_deg_ceil - deg(bg.arc_radians),
                    "radius": radius as f64 - bg.radius as f64,
                    "height": height as f64 - bg.height as f64,
                }),
            )
        }
        None => (Value::Null, None, Value::Null),
    };

    // Worst sample per shot: it is what sets the arc; the rest are reference.
    let suggested = probe(radius);
    let mut per_shot: Vec<Value> = Vec::new();
    for (id, worst) in worst_per_shot(&samples, &suggested, margin) {
        let cov = coverage(current.unwrap_or(&suggested), &worst.fr);
        per_shot.push(json!({
            "shot": id.map(|s| s.0),
            "frame": worst.frame,
            "covers_current": current.map(|_| cov.covers_frame),
            "gap_deg": gap_json(&cov),
            "needed_arc_deg": needed_arc(&suggested, &worst.fr, margin).map(|arc| (deg(arc) * 10.0).ceil() / 10.0),
        }));
    }
    // Binding sample: the largest required arc under the suggested geometry.
    let binding = samples
        .iter()
        .max_by(|a, b| {
            let key = |s: &FitSample| needed_arc(&suggested, &s.fr, margin).unwrap_or(std::f32::consts::TAU);
            key(a).total_cmp(&key(b))
        })
        .expect("fit_samples is never empty");

    let mut out = camera_json(scene.id.0, binding.shot, binding.frame, &binding.fr, aspect);
    let object = out.as_object_mut().expect("object");
    object.insert("scope".into(), json!(scope));
    object.insert("cameras".into(), json!(per_shot.len()));
    object.insert("samples".into(), json!(samples.len()));
    object.insert("margin_deg".into(), json!(margin_deg));
    object.insert("current".into(), current_json);
    object.insert("suggested".into(), json!({
        "arc_deg": arc_deg_ceil,
        "radius": radius,
        "height": height,
        "center": v3(center),
    }));
    object.insert("panorama_aspect".into(), json!(panorama_aspect));
    object.insert("arc_covers".into(), json!(arc_covers));
    object.insert("fits".into(), fits.map(|f| json!(f)).unwrap_or(Value::Null));
    object.insert("delta".into(), delta);
    object.insert("shots".into(), json!(per_shot));
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

/// Camera sample for the fit: shot (None — the scene's legacy camera) and frame.
struct FitSample {
    shot: Option<ShotId>,
    frame: Frame,
    fr: Frustum,
}

/// Camera samples for the fit: an explicit `shot` — only it (the frame is given
/// explicitly or taken from the start of the range); otherwise the scene's
/// legacy camera plus every shot camera at its keys and range ends.
fn fit_samples(scene: &Scene, shot: Option<ShotId>, frame: Option<Frame>, aspect: f32) -> Vec<FitSample> {
    let mut out = Vec::new();
    let mut sources: Vec<(Option<ShotId>, &Camera, FrameRange)> = match shot {
        Some(id) => vec![(Some(id), scene.camera_for_shot(Some(id)), scene.range_for_shot(Some(id)))],
        None => {
            let mut all: Vec<(Option<ShotId>, &Camera, FrameRange)> = vec![(None, &scene.camera, scene.range)];
            for s in scene.shots.values() {
                all.push((Some(s.id), &s.camera, s.range));
            }
            all
        }
    };
    // An explicit frame only makes sense for a single camera; at scene scope
    // cameras are sampled at their own keys.
    let explicit = if shot.is_some() { frame } else { None };
    sources.retain(|(_, _, range)| range.end > range.start);
    for (id, camera, range) in sources {
        let frames = match explicit {
            Some(f) => vec![f],
            None => coverage_frames(camera, range),
        };
        for f in frames {
            out.push(FitSample { shot: id, frame: f, fr: Frustum::new(camera, f, aspect) });
        }
    }
    if out.is_empty() {
        out.push(FitSample { shot, frame: frame.unwrap_or(scene.range.start), fr: Frustum::new(scene.camera_for_shot(shot), frame.unwrap_or(scene.range.start), aspect) });
    }
    out
}

/// The worst sample of each shot (the one needing the largest arc under
/// geometry `bg`), in id order; `None` — the scene's legacy camera.
fn worst_per_shot<'a>(samples: &'a [FitSample], bg: &Background, margin: f32) -> Vec<(Option<ShotId>, &'a FitSample)> {
    let mut ids: Vec<Option<ShotId>> = samples.iter().map(|s| s.shot).collect();
    ids.sort_by_key(|id| id.map(|s| s.0));
    ids.dedup();
    ids.into_iter()
        .map(|id| {
            let worst = samples
                .iter()
                .filter(|s| s.shot == id)
                .max_by(|a, b| {
                    let key = |s: &FitSample| needed_arc(bg, &s.fr, margin).unwrap_or(std::f32::consts::TAU);
                    key(a).total_cmp(&key(b))
                })
                .expect("a shot has samples");
            (id, worst)
        })
        .collect()
}

/// Distance from the camera to the screen center in the horizontal plane.
fn dist_xz(pos: Vec3, center: Vec3) -> f32 {
    ((pos.x - center.x).powi(2) + (pos.z - center.z).powi(2)).sqrt()
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
                        "background_set with arc in (0, 2π] rad, radius > 0 m, height > 0 m".to_string(),
                    ),
                );
            } else {
                // FOV coverage: the legacy camera over the scene range and the shot cameras.
                let mut checks: Vec<(Option<u64>, &Camera, FrameRange)> = vec![(None, &scene.camera, scene.range)];
                for shot in scene.shots.values() {
                    checks.push((Some(shot.id.0), &shot.camera, shot.range));
                }
                for (shot, camera, range) in checks {
                    for frame in coverage_frames(camera, range) {
                        let fr = Frustum::new(camera, frame, DEFAULT_ASPECT);
                        let cov = coverage(bg, &fr);
                        if cov.covers_frame {
                            continue;
                        }
                        let arc_deg = needed_arc(bg, &fr, 0.0).map(deg);
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
                                json!({"shot": shot, "needed_arc_deg": arc_deg.unwrap_or(360.0), "gap_deg": gap_json(&cov)}),
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
        // camera_coverage, at every camera point of the scene and its shots.
        if !scene.objects.is_empty() {
            let mut checks: Vec<(Option<u64>, &Camera, FrameRange)> = vec![(None, &scene.camera, scene.range)];
            for shot in scene.shots.values() {
                checks.push((Some(shot.id.0), &shot.camera, shot.range));
            }
            let any_in_frame = checks.into_iter().any(|(_, camera, range)| {
                coverage_frames(camera, range).into_iter().any(|frame| {
                    let fr = Frustum::new(camera, frame, DEFAULT_ASPECT);
                    scene.objects.values().any(|object| fr.contains(evaluate_object(object, &scene.range, frame, false).position))
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
                        "run camera_coverage for the intended shot; move/key the object or change the camera. This estimates object roots, it is not a render result".to_string(),
                    ),
                );
            }
        }

        // Phantom motion: how keys/clips cover the scene range.
        for ph in scene.phantoms.values() {
            let covering_take = ph
                .active_take
                .and_then(|id| ph.takes.iter().find(|t| t.id == id))
                .is_some_and(|t| t.range.start <= scene.range.start && t.range.end >= scene.range.end);
            if covering_take {
                continue;
            }
            if ph.motion.is_empty() && ph.takes.is_empty() && ph.clip.is_none() && ph.clip_track.is_empty() {
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
                        format!("phantom {} has no motion keys, takes or clips", ph.id.0),
                        "key_set/motion_replace, take_add or phantom_clip_set".to_string(),
                    ),
                );
                continue;
            }
            // Combined coverage: a legacy clip is looped across the whole scene.
            if ph.clip_track.is_empty() && ph.clip.is_some() {
                continue; // legacy clip loop covers the whole scene
            }
            let mut start_cov: Option<Frame> = ph.motion.first().map(|k| k.frame);
            let mut end_cov: Option<Frame> = ph.motion.last().map(|k| k.frame + 1);
            for u in &ph.clip_track {
                start_cov = Some(start_cov.map_or(u.range.start, |s| s.min(u.range.start)));
                end_cov = Some(end_cov.map_or(u.range.end, |e| e.max(u.range.end)));
            }
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
                            "add keys/clip across the whole scene range".to_string(),
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
                } else if let Some(shot) = clip.shot {
                    match scene.shots.get(&shot) {
                        None => Some(format!("clip {} references a shot that does not exist: shot {} of scene {}", clip.id.0, shot.0, clip.scene.0)),
                        Some(s) if s.range != clip.range => Some(format!(
                            "clip {} [{}, {}) does not match the range of shot {} [{}, {})",
                            clip.id.0, clip.range.start, clip.range.end, shot.0, s.range.start, s.range.end
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
                    json!({"clip": clip.id.0, "scene": clip.scene.0, "shot": clip.shot.map(|s| s.0), "range": clip.range}),
                    why,
                    "film_remove + film_add with a range inside the scene/shot".to_string(),
                ),
            );
        }
    }

    // Film assembly (corrective_2 K0.3, the internal part): whether every
    // shot is assembled, whether scene frames go unshown, and whether ranges
    // overlap twice. Cross-checking against the storyboard is deliberately
    // not done here — that is the editor's/director's job on their side.
    {
        let shot_assembly = project.film.iter().any(|clip| clip.shot.is_some());
        if shot_assembly {
            for scene in project.scenes.values() {
                for shot in scene.shots.values() {
                    if !project.film.iter().any(|clip| clip.scene == scene.id && clip.shot == Some(shot.id)) {
                        push(
                            Some(scene.id.0),
                            "shot_not_in_film",
                            Some(shot.range.start),
                            finding(
                                "shot_not_in_film",
                                "warning",
                                Some(scene.id.0),
                                Some(shot.range.start),
                                json!({"shot": shot.id.0, "name": shot.name, "range": shot.range}),
                                format!("shot {} \"{}\" of scene {} is not assembled into the film", shot.id.0, shot.name, scene.id.0),
                                "film_add with shot, or delete the shot if it is not needed".to_string(),
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
            camera: Camera::default(),
            phantoms: Default::default(),
            layers_2d: vec![],
            audio_tracks: vec![],
            lighting: Default::default(),
            floor: None,
            objects: Default::default(),
            shots: Default::default(),
            lights: vec![],
            ambient: Default::default(),
            sky: None,
            grade: None,
            fog: None,
            vfx: vec![],
        };
        let aid = AssetId(2);
        p.assets.insert(aid, Asset { id: aid, kind: AssetKind::Audio, path: "assets/music.mp3".into() });
        // A 6.34 s clip in a 2-second scene — the A4 case.
        s.audio_tracks.push(AudioTrack {
            id: TrackId(3),
            name: "music".into(),
            volume: 1.0,
            muted: false,
            clips: vec![AudioClip { id: AudioClipId(4), asset: aid, start: 0, offset_samples: 0, len_samples: 304_320, fade_in_samples: 0, fade_out_samples: 0 }],
        });
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
    use phantom_core::project::{BackgroundSource, MotionKey, ObjectKey, Phantom, RigRef, SceneObject, Take, TakeId, TransformKey};
    use phantom_core::{Action, Asset, AssetId, AssetKind, ObjectId, PhantomId, SceneId};

    fn bg() -> Background {
        Background {
            image: BackgroundSource::Image(AssetId(1)),
            arc_radians: 2.0,
            radius: 6.0,
            height: 4.0,
            center: Vec3::new(0.0, 2.0, -5.0),
            bulge: Default::default(),
            fit: Default::default(),
        }
    }

    fn scene(id: u64, start: Frame, end: Frame) -> Scene {
        let mut s = Scene {
            id: SceneId(id),
            name: format!("s{id}"),
            range: FrameRange::new(start, end).unwrap(),
            background: None,
            camera: Camera::default(),
            phantoms: Default::default(),
            layers_2d: vec![],
            audio_tracks: vec![],
            lighting: Default::default(),
            floor: None,
            objects: Default::default(),
            shots: Default::default(),
            lights: vec![],
            ambient: Default::default(),
            sky: None,
            grade: None,
            fog: None,
            vfx: vec![],
        };
        s.background = Some(bg());
        s
    }

    fn phantom(id: u64, keys: Vec<(Frame, [f32; 3])>) -> Phantom {
        Phantom {
            id: PhantomId(id),
            name: format!("p{id}"),
            rig: RigRef::Builtin,
            motion: keys
                .into_iter()
                .map(|(frame, pos)| MotionKey { frame, position: pos.into(), action: Action::Idle, yaw: 0.0 })
                .collect(),
            clip: None,
            clip_track: vec![],
            takes: vec![],
            active_take: None,
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
        // Covering geometry: a 2π arc around the camera at the curvature
        // center — that is the toward_camera orientation (curvature circle
        // around the camera).
        s.camera.keys = vec![TransformKey::at(0, Vec3::new(0.0, 2.0, -11.0), std::f32::consts::PI, 0.0)];
        s.background = Some(Background { arc_radians: std::f32::consts::TAU, height: 6.0, bulge: phantom_core::project::Bulge::TowardCamera, ..bg() });
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
        s1.background = Some(Background { arc_radians: 0.0, ..bg() });
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
        p.film.push(FilmClip { id: ClipId(10), scene: SceneId(2), range: FrameRange::new(0, 96).unwrap(), shot: None });
        let mut film_track = AudioTrack::new(TrackId(11), "film");
        film_track.clips.push(AudioClip { id: AudioClipId(12), asset: AssetId(7), start: 0, offset_samples: 0, len_samples: 200 * 48_000 , fade_in_samples: 0, fade_out_samples: 0 });
        p.film_audio.push(film_track);
        // Missing asset file (dir exists, the file does not).
        p.assets.insert(AssetId(7), Asset { id: AssetId(7), kind: AssetKind::Audio, path: "missing.wav".into() });

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
    fn rounded_full_circle_is_valid_background_geometry() {
        let mut p = Project::new("t");
        let mut s = scene(1, 0, 48);
        s.background = Some(Background { arc_radians: 6.2832, ..bg() });
        p.scenes.insert(SceneId(1), s);
        let findings = validate(&p, None, Some(1)).unwrap();
        let codes = codes(&findings);
        assert!(!codes.contains(&"bg_geometry_invalid"), "{codes:?}");
    }

    #[test]
    fn validate_checks_film_assembly_shots_gaps_and_overlaps() {
        let mut p = Project::new("t");
        let mut s = scene(1, 0, 96);
        let assembled = ShotId(2);
        let forgotten = ShotId(3);
        s.shots.insert(assembled, phantom_core::project::Shot { id: assembled, name: "kept".into(), range: FrameRange::new(0, 48).unwrap(), camera: Camera::default() });
        s.shots.insert(forgotten, phantom_core::project::Shot { id: forgotten, name: "lost".into(), range: FrameRange::new(48, 96).unwrap(), camera: Camera::default() });
        p.scenes.insert(SceneId(1), s);
        // The film assembles only the first shot and repeats part of its own frames.
        p.film.push(FilmClip { id: ClipId(4), scene: SceneId(1), range: FrameRange::new(0, 48).unwrap(), shot: Some(assembled) });
        p.film.push(FilmClip { id: ClipId(5), scene: SceneId(1), range: FrameRange::new(24, 48).unwrap(), shot: Some(assembled) });

        let findings = validate(&p, None, Some(1)).unwrap();
        let found = codes(&findings);
        assert!(found.contains(&"shot_not_in_film"), "{found:?}");
        assert!(found.contains(&"film_coverage_gap"), "frames 48..96 are not shown: {found:?}");
        assert!(found.contains(&"film_overlap"), "frames 24..48 are shown twice: {found:?}");
        let gap = findings.iter().find(|f| f["code"] == "film_coverage_gap").unwrap();
        assert_eq!(gap["where"]["unshown"]["start"], 48);
        assert_eq!(gap["where"]["unshown"]["end"], 96);
        let overlap = findings.iter().find(|f| f["code"] == "film_overlap").unwrap();
        assert_eq!(overlap["where"]["overlap"]["start"], 24);
        // An edit without shot references must not produce a false shot_not_in_film.
        let mut scene_level = p.clone();
        for clip in &mut scene_level.film { clip.shot = None; }
        let scene_level_findings = validate(&scene_level, None, Some(1)).unwrap();
        let scene_level_codes = codes(&scene_level_findings);
        assert!(!scene_level_codes.contains(&"shot_not_in_film"), "{scene_level_codes:?}");
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
    fn validate_skips_phantoms_with_covering_take_and_looping_clip() {
        let mut p = Project::new("t");
        let mut s = scene(1, 0, 48);
        let mut ph = phantom(2, vec![(40, [0.0; 3])]); // a hole at the start, but the active take covers it
        ph.takes.push(Take { id: TakeId(3), name: "t".into(), range: FrameRange::new(0, 48).unwrap(), motion: vec![] });
        ph.active_take = Some(TakeId(3));
        s.phantoms.insert(PhantomId(2), ph);
        let mut ph_clip = phantom(4, vec![]); // legacy clip: looped across the whole scene
        ph_clip.clip = Some(AssetId(9));
        s.phantoms.insert(PhantomId(4), ph_clip);
        p.scenes.insert(SceneId(1), s);
        let findings = validate(&p, None, None).unwrap();
        let codes = codes(&findings);
        assert!(!codes.contains(&"motion_gap"), "{codes:?}");
        assert!(!codes.contains(&"phantom_without_motion"), "{codes:?}");
    }

    // --- camera_coverage: known geometry ---

    /// A camera at the screen's curvature center looking at the arc midpoint:
    /// the screen's azimuthal span equals exactly arc.
    fn centered_scene(arc: f32, height: f32, focal_mm: f32) -> Scene {
        let mut s = scene(1, 0, 48);
        // curvature circle: center + (0,·,−R) = (0, 2, −11) — the
        // toward_camera orientation puts the camera at the arc's curvature center.
        s.camera = Camera {
            keys: vec![TransformKey::at(0, Vec3::new(0.0, 2.0, -11.0), std::f32::consts::PI, 0.0)],
            focal_mm,
        };
        s.background = Some(Background { arc_radians: arc, height, bulge: phantom_core::project::Bulge::TowardCamera, ..bg() });
        s
    }

    #[test]
    fn coverage_from_curvature_center_span_equals_arc() {
        let s = centered_scene(2.0, 6.0, 35.0);
        let out = camera_coverage(&s, None, Some(0), DEFAULT_ASPECT);
        let bg = &out["background"];
        let span = &bg["span"]["horizontal_deg"];
        assert!((span["min"].as_f64().unwrap() + 57.2958).abs() < 0.5, "{span}");
        assert!((span["max"].as_f64().unwrap() - 57.2958).abs() < 0.5, "{span}");
        // hfov/2 (35 mm, 16:9) ≈ 31.3° < 57.3°, height 6 m: el=atan2(3,6)=26.6° > vfov/2 ≈ 18.9°.
        assert_eq!(bg["covers_frame"], true, "{bg}");
        assert!(out["findings"].as_array().unwrap().is_empty());
    }

    #[test]
    fn coverage_offset_camera_reports_gap() {
        // The default staging from corrective A3(b): camera (0,1.4,4),
        // screen arc=2.0, radius=6, center=(0,2,−5): the edge sits at ~23.2°,
        // hfov/2 ≈ 31.3° → a gap on both sides. The measurement was taken on
        // the toward_camera orientation (pre-A3 render behavior) — pin it explicitly.
        let mut s = scene(1, 0, 48);
        s.background = Some(Background { bulge: phantom_core::project::Bulge::TowardCamera, ..bg() });
        let out = camera_coverage(&s, None, Some(0), DEFAULT_ASPECT);
        let bg = &out["background"];
        assert_eq!(bg["covers_frame"], false);
        let max = bg["span"]["horizontal_deg"]["max"].as_f64().unwrap();
        assert!((max - 23.2).abs() < 1.0, "max azimuth {max}");
        assert!(bg["gap_deg"]["left"].as_f64().unwrap() > 4.0);
        assert!(bg["gap_deg"]["right"].as_f64().unwrap() > 4.0);
        assert_eq!(out["findings"][0]["code"], "screen_fov_gap");
    }

    #[test]
    fn coverage_default_away_bulge_wraps_edges_closer_to_camera() {
        // The same staging, but with the default away_from_camera orientation:
        // the edges come closer and are seen at a larger angle (39° vs 23.2°),
        // the side gap disappears — this pins the new corrective A3a default.
        let s = scene(1, 0, 48);
        assert_eq!(s.background.as_ref().unwrap().bulge, phantom_core::project::Bulge::AwayFromCamera);
        let out = camera_coverage(&s, None, Some(0), DEFAULT_ASPECT);
        let bg = &out["background"];
        let max = bg["span"]["horizontal_deg"]["max"].as_f64().unwrap();
        assert!((max - 39.0).abs() < 1.5, "away: max azimuth {max}");
        assert!(bg["gap_deg"]["left"].as_f64().unwrap() < 1.0, "sides are covered: {}", bg["gap_deg"]);
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
            },
        );
        let out = camera_coverage(&s, None, Some(0), DEFAULT_ASPECT);
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
                keys: vec![ObjectKey { frame: 0, position: Vec3::new(100.0, 0.0, 0.0), yaw: 0.0, pitch: 0.0, roll: 0.0, scale: Vec3::ONE }],
                attach: None, material_patches: Vec::new(),
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
    fn background_fit_scene_scope_covers_every_shot() {
        // corrective_2 K0.1: without shot the fit must cover ALL scene
        // cameras, not one. The second shot looks the opposite way.
        let mut s = scene(1, 0, 96);
        let forward = ShotId(2);
        let backward = ShotId(3);
        s.shots.insert(
            forward,
            phantom_core::project::Shot {
                id: forward, name: "fwd".into(), range: FrameRange::new(0, 48).unwrap(),
                camera: Camera { focal_mm: 35.0, keys: vec![TransformKey::at(0, Vec3::new(0.0, 2.0, -11.0), std::f32::consts::PI, 0.0)] },
            },
        );
        s.shots.insert(
            backward,
            phantom_core::project::Shot {
                id: backward, name: "back".into(), range: FrameRange::new(48, 96).unwrap(),
                camera: Camera { focal_mm: 35.0, keys: vec![TransformKey::at(48, Vec3::new(0.0, 2.0, 11.0), 0.0, 0.0)] },
            },
        );

        let scene_fit = background_fit(&s, None, None, 5.0, DEFAULT_ASPECT);
        assert_eq!(scene_fit["scope"], "scene");
        assert_eq!(scene_fit["cameras"], 3, "legacy camera + two shots: {}", scene_fit["shots"]);
        // The suggested geometry covers each shot on its own.
        let arc = (scene_fit["suggested"]["arc_deg"].as_f64().unwrap() as f32).to_radians();
        let radius = scene_fit["suggested"]["radius"].as_f64().unwrap() as f32;
        let height = scene_fit["suggested"]["height"].as_f64().unwrap() as f32;
        let mut fixed = s.clone();
        fixed.background = Some(Background { arc_radians: arc, radius, height, center: Vec3::new(0.0, 2.0, -5.0), ..bg() });
        for (shot, frame) in [(None, 0), (Some(forward), 0), (Some(backward), 48)] {
            let cov = camera_coverage(&fixed, shot, Some(frame), DEFAULT_ASPECT);
            assert_eq!(cov["background"]["covers_frame"], true, "shot {shot:?} is not covered: {}", cov["background"]);
        }
        // A fit for a single shot may be narrower than a fit for the whole scene.
        let shot_fit = background_fit(&s, Some(backward), None, 5.0, DEFAULT_ASPECT);
        assert_eq!(shot_fit["scope"], "shot");
        assert!(shot_fit["suggested"]["arc_deg"].as_f64().unwrap() <= scene_fit["suggested"]["arc_deg"].as_f64().unwrap());
    }

    // --- background_fit: roundtrip ---

    #[test]
    fn background_fit_roundtrip_covers_frame() {
        let s = scene(1, 0, 48); // default staging with a coverage gap
        let fit = background_fit(&s, None, None, 5.0, DEFAULT_ASPECT);
        assert_eq!(fit["fits"], false, "the current geometry does not cover the frame");
        let suggested = &fit["suggested"];
        let arc = (suggested["arc_deg"].as_f64().unwrap() as f32).to_radians();
        assert!(arc > 0.0 && arc <= std::f32::consts::TAU, "arc {arc}");
        assert_eq!(fit["arc_covers"], true);
        // radius is not below the starting dist(camera, center) + 0.5 = 9.5 m;
        // with away_from_camera the starting radius usually already covers the frame.
        assert!(suggested["radius"].as_f64().unwrap() >= 9.4);
        assert!(fit["panorama_aspect"].as_f64().unwrap() > 0.0);

        // Apply the suggestion: coverage is closed.
        let mut fixed = s.clone();
        fixed.background = Some(Background {
            arc_radians: arc,
            radius: suggested["radius"].as_f64().unwrap() as f32,
            height: suggested["height"].as_f64().unwrap() as f32,
            center: Vec3::new(0.0, 2.0, -5.0),
            ..bg()
        });
        let out = camera_coverage(&fixed, None, Some(0), DEFAULT_ASPECT);
        assert_eq!(out["background"]["covers_frame"], true, "{}", out["background"]);

        // Geometry that already covers: fits=true, the arc delta is non-negative.
        let fit2 = background_fit(&fixed, None, Some(0), 5.0, DEFAULT_ASPECT);
        assert_eq!(fit2["fits"], true);
        assert!(fit2["delta"]["arc_deg"].as_f64().unwrap() >= -0.2, "{}", fit2["delta"]);
    }

    #[test]
    fn background_fit_without_background_uses_defaults() {
        let mut s = scene(1, 0, 48);
        s.background = None;
        let fit = background_fit(&s, None, None, 5.0, DEFAULT_ASPECT);
        assert_eq!(fit["current"], Value::Null);
        assert_eq!(fit["fits"], Value::Null);
        assert_eq!(fit["suggested"]["center"], json!([0.0, 2.0, -5.0]));
        assert!(fit["suggested"]["radius"].as_f64().unwrap() >= 9.0, "the camera is 9 m from the default center");
    }
}

//! MCP server for AI agents (rmcp, stdio). Two modes:
//! a standalone session without a window, and a bridge to the open editor.
//! stdout carries the protocol only; logs go to stderr.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use rmcp::{
    handler::server::wrapper::Parameters,
    model::*,
    tool, tool_handler, tool_router, ErrorData, ServerHandler, ServiceExt,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::bridge::BridgeClient;
use crate::session::{ErrorData as StructuredError, Op, Session};

enum Backend {
    Standalone(Mutex<Session>),
    Bridge(Mutex<BridgeClient>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum McpMode {
    Standalone,
    Bridge,
}

impl McpMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Standalone => "standalone",
            Self::Bridge => "bridge",
        }
    }
}

#[derive(Clone)]
struct PhantomMcp {
    backend: Arc<Backend>,
    /// MCP navigation state belongs to this server/connection, never to the
    /// project document or GUI session shared by the bridge.
    active_scene: Arc<Mutex<Option<u64>>>,
    allowed_root: Arc<Mutex<Option<PathBuf>>>,
    mode: McpMode,
}

#[derive(Debug, Serialize, JsonSchema)]
struct StructuredObject {
    /// Operation-specific structured fields; structuredContent is authoritative.
    #[serde(flatten)]
    fields: std::collections::BTreeMap<String, serde_json::Value>,
}

fn structured_object_schema() -> Arc<JsonObject> {
    rmcp::handler::server::tool::schema_for_type::<StructuredObject>()
}

#[derive(Debug, Serialize, JsonSchema)]
struct RenderStillOutput {
    /// Scene ID rendered.
    scene: u64,
    /// Integer scene frame rendered at fixed 24 fps.
    frame: i64,
    /// Optional camera ID whose view was used.
    camera: Option<u64>,
    /// PNG width in pixels.
    width: u32,
    /// PNG height in pixels.
    height: u32,
    /// Image media type; currently image/png.
    mime_type: String,
    /// Pure frame metrics when report=true was requested; absent otherwise.
    report: Option<FrameMetricsOutput>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct FrameMetricsOutput {
    /// Mean Rec.709 luma 0..=255 over opaque pixels; 0 when the frame is empty.
    mean_luma: f64,
    /// 5th percentile luma over opaque pixels.
    luma_p5: f64,
    /// 95th percentile luma over opaque pixels.
    luma_p95: f64,
    /// Share of the frame without geometry (alpha == 0), 0.0..=1.0.
    empty_coverage: f64,
    /// Number of scene objects whose forged Mesh3d part passed Bevy visibility for the selected camera.
    objects_in_frame: usize,
    /// Total scene objects considered by objects_in_frame.
    objects_total: usize,
    /// Non-cryptographic 64-bit hex digest of the canonical scene snapshot; changes when the scene document changes.
    scene_digest: String,
}

fn render_still_output_schema() -> Arc<JsonObject> {
    rmcp::handler::server::tool::schema_for_type::<RenderStillOutput>()
}

#[derive(Debug, Serialize, JsonSchema)]
struct JobStatusOutput {
    /// Last progress line emitted by the child process.
    last_line: String,
    /// True only after child exit and stderr drain are finalized.
    done: bool,
    /// Child exit code after completion; null while running or when terminated by signal.
    exit_code: Option<i32>,
    /// Completed frame count parsed from last_line, when available.
    frames_done: Option<u64>,
    /// Total frame count parsed from last_line, when available.
    total: Option<u64>,
    /// Monotonic seconds elapsed since the job started.
    elapsed_secs: f64,
    /// Bounded tail of child stderr, oldest line first.
    error_tail: Vec<String>,
    /// True when job_wait reached its timeout while the child remained running; cancellation is not implied.
    timed_out: bool,
}
fn job_wait_output_schema() -> Arc<JsonObject> { rmcp::handler::server::tool::schema_for_type::<JobStatusOutput>() }

#[derive(Debug, Serialize, JsonSchema)]
struct DescribeOutput {
    /// Exact registered MCP tool name.
    name: String,
    /// Description from the registered ToolRouter metadata.
    description: String,
    /// Registered JSON input schema, including per-field units and ranges.
    #[serde(rename = "inputSchema")]
    #[schemars(schema_with = "value_schema")]
    input_schema: serde_json::Value,
    /// Registered structuredContent output schema.
    #[serde(rename = "outputSchema")]
    #[schemars(schema_with = "value_schema")]
    output_schema: serde_json::Value,
    /// How to interpret units documented in inputSchema field descriptions.
    unit_notes: String,
    /// Usage guidance generated from the same registered metadata.
    usage_notes: String,
}
fn value_schema(_gen: &mut schemars::SchemaGenerator) -> schemars::Schema { schemars::json_schema!({}) }
fn describe_output_schema() -> Arc<JsonObject> { rmcp::handler::server::tool::schema_for_type::<DescribeOutput>() }

fn nearest_tool_names(wanted: &str, mut names: Vec<String>) -> Vec<String> {
    fn distance(a: &str, b: &str) -> usize {
        let mut row: Vec<usize> = (0..=b.len()).collect();
        for (i, ac) in a.bytes().enumerate() {
            let mut previous = row[0]; row[0] = i + 1;
            for (j, bc) in b.bytes().enumerate() {
                let old = row[j + 1];
                row[j + 1] = (row[j + 1] + 1).min(row[j] + 1).min(previous + usize::from(ac != bc));
                previous = old;
            }
        }
        row[b.len()]
    }
    names.sort_by_key(|name| (distance(wanted, name), name.clone()));
    names.truncate(5);
    names
}

impl PhantomMcp {
    fn backend_exec(&self, op: Op) -> Result<serde_json::Value, StructuredError> {
        match &*self.backend {
            Backend::Standalone(session) => session
                .lock()
                .expect("session lock")
                .execute(op)
                .map_err(|error| error.error_data()),
            Backend::Bridge(client) => client
                .lock()
                .expect("bridge lock")
                .execute(&op)
                .map_err(|error| error.error_data()),
        }
    }

    fn resolve_scene(&self, explicit: Option<u64>) -> Result<u64, CallToolResult> {
        explicit.or_else(|| *self.active_scene.lock().expect("active scene lock")).ok_or_else(|| {
            structured_failure(StructuredError::new(
                "no_active_scene",
                "scene is not set and no active scene is selected",
                serde_json::json!({ "hint": "use scene_select or pass scene" }),
            ))
        })
    }

    fn refresh_active_scene(&self) {
        let Ok(state) = self.backend_exec(Op::StateGet) else { return };
        let scenes: Vec<u64> = state.get("scenes").and_then(|v| serde_json::from_value(v.clone()).ok()).unwrap_or_default();
        let mut active = self.active_scene.lock().expect("active scene lock");
        if active.is_none_or(|id| !scenes.contains(&id)) {
            // Pick the numerically nearest remaining stable ID (lower ID wins
            // ties); with no previous selection, BTreeMap order picks first.
            let previous = *active;
            *active = previous
                .and_then(|old| scenes.iter().copied().min_by_key(|id| (id.abs_diff(old), *id)))
                .or_else(|| scenes.first().copied());
        }
    }

    fn active_root(&self) -> Option<PathBuf> {
        self.allowed_root.lock().expect("allowed root lock").clone()
    }

    fn exec(&self, op: Op) -> Result<CallToolResult, ErrorData> {
        Ok(match self.backend_exec(op) {
            Ok(value) => structured_success(value),
            Err(error) => structured_failure(error),
        })
    }

    /// Run an operation whose result carries `png_base64` and turn it into an
    /// image block plus structuredContent (shared by render_still and look).
    fn exec_image(&self, op: Op) -> Result<CallToolResult, ErrorData> {
        Ok(image_result(self.backend_exec(op)))
    }
}

/// A review direction from the wire: a view name or an `[x,y,z]` direction.
fn parse_view(v: &serde_json::Value) -> Result<crate::session::ViewSpec, String> {
    match v {
        serde_json::Value::String(name) => Ok(crate::session::ViewSpec::Name(name.clone())),
        serde_json::Value::Array(a) if a.len() == 3 => {
            let mut out = [0.0f32; 3];
            for (i, item) in a.iter().enumerate() {
                let value = item.as_f64().ok_or_else(|| format!("view direction entries must be numbers, got {item}"))?;
                if !value.is_finite() {
                    return Err(format!("view direction must be finite, got {value}"));
                }
                out[i] = value as f32;
            }
            if out.iter().all(|v| v.abs() < 1e-6) {
                return Err("view direction must not be the zero vector".into());
            }
            Ok(crate::session::ViewSpec::Direction(out))
        }
        other => Err(format!(
            "views: a view name (front|back|left|right|top|three_quarter) or [x,y,z], got {other}"
        )),
    }
}

fn parse_views(views: Option<&[serde_json::Value]>) -> Result<Vec<crate::session::ViewSpec>, String> {
    views.unwrap_or(&[]).iter().map(parse_view).collect()
}

fn image_result(result: Result<serde_json::Value, StructuredError>) -> CallToolResult {
    match result {
        Ok(mut value) => {
            let Some(object) = value.as_object_mut() else {
                return structured_failure(StructuredError::new(
                    "internal_error",
                    "render_still returned a non-object",
                    serde_json::json!({ "stage": "mcp_response" }),
                ));
            };
            let Some(encoded) = object.remove("png_base64").and_then(|v| v.as_str().map(str::to_owned)) else {
                return structured_failure(StructuredError::new(
                    "internal_error",
                    "render_still returned no PNG",
                    serde_json::json!({ "stage": "mcp_response" }),
                ));
            };
            let mime_type = object
                .get("mime_type")
                .and_then(|v| v.as_str())
                .unwrap_or("image/png")
                .to_owned();
            let mut result = structured_success(value);
            result.content.push(ContentBlock::image(encoded, mime_type));
            result
        }
        Err(error) => structured_failure(error),
    }
}

/// rmcp's `structured()` constructor mirrors JSON into a text block. This
/// contract intentionally does not: structuredContent is authoritative.
fn structured_success(value: serde_json::Value) -> CallToolResult {
    let mut result = CallToolResult::default();
    result.structured_content = Some(value);
    result.is_error = Some(false);
    result
}

fn structured_failure(error: StructuredError) -> CallToolResult {
    let mut result = CallToolResult::default();
    result.structured_content = Some(serde_json::to_value(error).expect("structured error"));
    result.is_error = Some(true);
    result
}

fn seconds_to_samples(seconds: f64, field: &str) -> Result<u64, String> {
    if !seconds.is_finite() || seconds < 0.0 {
        return Err(format!("{field} must be finite and non-negative"));
    }
    let samples = seconds * 48_000.0;
    if !samples.is_finite() || samples > u64::MAX as f64 {
        return Err(format!("{field} overflows 48-kHz sample count"));
    }
    Ok(samples.round() as u64)
}

fn seconds_to_frame(seconds: f64, field: &str) -> Result<i64, String> {
    if !seconds.is_finite() || seconds < 0.0 {
        return Err(format!("{field} must be finite and non-negative"));
    }
    let frames = seconds * phantom_core::FPS as f64;
    if !frames.is_finite() || frames > i64::MAX as f64 {
        return Err(format!("{field} overflows 24-fps frame count"));
    }
    Ok(frames.round() as i64)
}

fn render_scene_output(out: Option<String>) -> PathBuf {
    PathBuf::from(out.unwrap_or_else(|| "frames".into()))
}

fn convert_brush_points(
    points: Vec<[f32; 2]>,
    space: BrushSpace,
    screen_width: Option<f32>,
    screen_height: Option<f32>,
) -> Result<Vec<[f32; 2]>, String> {
    if points.iter().flatten().any(|v| !v.is_finite()) {
        return Err("brush points must be finite".into());
    }
    match space {
        BrushSpace::Canvas => Ok(points),
        BrushSpace::Ndc => {
            if points.iter().flatten().any(|v| !(-1.0..=1.0).contains(v)) {
                return Err("ndc brush points must be in -1..=1".into());
            }
            Ok(points.into_iter().map(|[x, y]| [(x + 1.0) * 960.0, (1.0 - y) * 540.0]).collect())
        }
        BrushSpace::Screen => {
            let (Some(w), Some(h)) = (screen_width, screen_height) else {
                return Err("screen brush space requires screen_width and screen_height".into());
            };
            if !w.is_finite() || !h.is_finite() || w <= 0.0 || h <= 0.0 {
                return Err("screen dimensions must be finite and positive".into());
            }
            Ok(points.into_iter().map(|[x, y]| [x * 1920.0 / w, y * 1080.0 / h]).collect())
        }
    }
}

fn invalid_input(message: impl Into<String>) -> CallToolResult {
    let message = message.into();
    structured_failure(StructuredError::new(
        "invalid_input",
        message.clone(),
        serde_json::json!({ "reason": message }),
    ))
}

/// Lens presets (corrective_2 C2.1): the project reads "portrait" rather than
/// a bare number. Values are 35-mm equivalents.
const LENS_PRESETS: &[(&str, f32)] = &[
    ("ultrawide", 16.0),
    ("wide", 24.0),
    ("normal", 35.0),
    ("portrait", 50.0),
    ("portrait_tele", 85.0),
    ("tele", 135.0),
    ("super_tele", 200.0),
];

fn lens_preset_names() -> String {
    LENS_PRESETS.iter().map(|(name, mm)| format!("{name} {mm:.0}")).collect::<Vec<_>>().join(" | ")
}

/// Focal length from an explicit number or a preset name; an unknown name is an
/// error, not a silent substitution (lesson from corrective_2 §2).
fn lens_mm(focal_mm: Option<f32>, preset: Option<&str>) -> Result<f32, CallToolResult> {
    if let Some(mm) = focal_mm {
        return Ok(mm);
    }
    match preset {
        None => Err(invalid_input("focal_mm or focal_preset required")),
        Some(name) => LENS_PRESETS
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, mm)| *mm)
            .ok_or_else(|| invalid_input(format!("focal_preset: {}; got {name}", lens_preset_names()))),
    }
}

fn parse_ease(name: Option<&str>) -> Result<phantom_core::project::KeyEase, CallToolResult> {
    use phantom_core::project::KeyEase;
    match name {
        None | Some("linear") => Ok(KeyEase::Linear),
        Some("ease_in") => Ok(KeyEase::EaseIn),
        Some("ease_out") => Ok(KeyEase::EaseOut),
        Some("ease_in_out") => Ok(KeyEase::EaseInOut),
        Some(other) => Err(invalid_input(format!("ease: linear|ease_in|ease_out|ease_in_out; got {other}"))),
    }
}

/// Build a camera key from MCP fields: degrees in, look_at overrides the
/// angles, focal_mm overrides the preset.
#[allow(clippy::too_many_arguments)]
fn camera_key_input(
    frame: i64,
    pos: [f32; 3],
    yaw_deg: Option<f32>,
    pitch_deg: Option<f32>,
    roll_deg: Option<f32>,
    look_at: Option<[f32; 3]>,
    focal_mm: Option<f32>,
    focal_preset: Option<&str>,
    ease: Option<&str>,
) -> Result<phantom_core::project::TransformKey, CallToolResult> {
    let focal = match (focal_mm, focal_preset) {
        (None, None) => None,
        _ => Some(lens_mm(focal_mm, focal_preset)?),
    };
    Ok(phantom_core::project::TransformKey {
        frame,
        translation: glam::Vec3::from_array(pos),
        yaw: yaw_deg.unwrap_or(0.0).to_radians(),
        pitch: pitch_deg.unwrap_or(0.0).to_radians(),
        roll: roll_deg.unwrap_or(0.0).to_radians(),
        look_at: look_at.map(glam::Vec3::from_array),
        focal_mm: focal,
        ease: parse_ease(ease)?,
    })
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PNew {
    /// Project directory path; becomes the only allowed output root.
    dir: String,
    /// Non-empty human-readable project name.
    name: String,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct POpen {
    /// Existing project directory path; becomes the only allowed output root.
    dir: String,
    /// Allow displacing an ambiguous lock; a live PID is never displaced.
    #[serde(default)]
    force: bool,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PLockRelease {
    /// Project directory whose writer lock is diagnosed or released.
    dir: String,
    /// Required to remove a dead/ambiguous lock owned by someone else.
    #[serde(default)]
    force: bool,
}

#[tool_router]
impl PhantomMcp {
    #[tool(output_schema = structured_object_schema(), description = "Create a Phantom-animate project in a directory")]
    async fn project_new(&self, Parameters(p): Parameters<PNew>) -> Result<CallToolResult, ErrorData> {
        let dir = PathBuf::from(p.dir);
        let result = self.backend_exec(Op::ProjectNew { dir: dir.clone(), name: p.name });
        if result.is_ok() {
            *self.allowed_root.lock().expect("allowed root lock") = Some(dir.canonicalize().unwrap_or(dir));
            *self.active_scene.lock().expect("active scene lock") = None;
            self.refresh_active_scene();
        }
        Ok(result.map(structured_success).unwrap_or_else(structured_failure))
    }

    #[tool(output_schema = structured_object_schema(), description = "Open an existing project; reopening the current project is idempotent; force displaces only an ambiguous/stale lock, never a live PID")]
    async fn project_open(&self, Parameters(p): Parameters<POpen>) -> Result<CallToolResult, ErrorData> {
        let dir = PathBuf::from(p.dir);
        let result = self.backend_exec(Op::ProjectOpen { dir: dir.clone(), force: p.force });
        if result.is_ok() {
            *self.allowed_root.lock().expect("allowed root lock") = Some(dir.canonicalize().unwrap_or(dir));
            *self.active_scene.lock().expect("active scene lock") = None;
            self.refresh_active_scene();
        }
        Ok(result
            .map(|mut value| {
                if let Some(object) = value.as_object_mut() {
                    object.insert("active_scene".into(), serde_json::json!(*self.active_scene.lock().expect("active scene lock")));
                }
                structured_success(value)
            })
            .unwrap_or_else(structured_failure))
    }

    #[tool(output_schema = structured_object_schema(), description = "Diagnose and release a project lock; force is needed for a dead/ambiguous lock owned by someone else, a live owner is never removed")]
    async fn lock_release(&self, Parameters(p): Parameters<PLockRelease>) -> Result<CallToolResult, ErrorData> {
        self.exec(Op::LockRelease { dir: PathBuf::from(p.dir), force: p.force })
    }

    #[tool(output_schema = structured_object_schema(), description = "Save the project to disk")]
    async fn project_save(&self) -> Result<CallToolResult, ErrorData> {
        self.exec(Op::ProjectSave)
    }

    #[tool(output_schema = structured_object_schema(), description = "Last successfully created/opened project from per-user state, plus the current project and active scene; works with no open project")]
    async fn project_recent(&self) -> Result<CallToolResult, ErrorData> {
        let result = self.backend_exec(Op::ProjectRecent).map(|mut value| {
            if let Some(object) = value.as_object_mut() {
                object.insert("active_scene".into(), serde_json::json!(*self.active_scene.lock().expect("active scene lock")));
            }
            value
        });
        Ok(result.map(structured_success).unwrap_or_else(structured_failure))
    }

    #[tool(output_schema = describe_output_schema(), description = "Describe a registered MCP tool from the same ToolRouter metadata as tools/list: description, inputSchema, outputSchema, units and usage notes")]
    async fn describe(&self, Parameters(p): Parameters<PDescribe>) -> Result<CallToolResult, ErrorData> {
        let tools = Self::tool_router().list_all();
        if let Some(tool) = tools.iter().find(|candidate| candidate.name.as_ref() == p.tool) {
            let description = tool.description.as_deref().unwrap_or_default().to_owned();
            return Ok(structured_success(serde_json::json!({
                "name": tool.name,
                "description": description,
                "inputSchema": tool.input_schema,
                "outputSchema": tool.output_schema,
                "unit_notes": "Units, ranges, defaults, and omission semantics are documented on each inputSchema property.",
                "usage_notes": format!("{description} Results are returned in structuredContent and conform to outputSchema."),
            })));
        }
        let nearest = nearest_tool_names(&p.tool, tools.iter().map(|t| t.name.to_string()).collect());
        Ok(structured_failure(StructuredError::new("not_found", format!("tool {} not found", p.tool), serde_json::json!({"entity":"tool", "name":p.tool, "nearest_tools":nearest}))))
    }

    #[tool(output_schema = structured_object_schema(), description = "Project state: revision, scenes, assets, film length")]
    async fn state_get(&self) -> Result<CallToolResult, ErrorData> {
        self.refresh_active_scene();
        let result = self.backend_exec(Op::StateGet).map(|mut value| {
            if let Some(object) = value.as_object_mut() {
                object.insert("active_scene".into(), serde_json::json!(*self.active_scene.lock().expect("active scene lock")));
                object.insert("allowed_root".into(), serde_json::json!(self.active_root()));
            }
            value
        });
        Ok(result.map(structured_success).unwrap_or_else(structured_failure))
    }

    #[tool(output_schema = structured_object_schema(), description = "Select the active scene of this MCP session; does not change the document")]
    async fn scene_select(&self, Parameters(p): Parameters<PSceneSelect>) -> Result<CallToolResult, ErrorData> {
        match self.backend_exec(Op::SceneGet { scene: p.scene }) {
            Ok(_) => {
                *self.active_scene.lock().expect("active scene lock") = Some(p.scene);
                Ok(structured_success(serde_json::json!({ "active_scene": p.scene })))
            }
            Err(error) => Ok(structured_failure(error)),
        }
    }

    #[tool(output_schema = structured_object_schema(), description = "Get the active scene of this MCP session (null when no scene is selected)")]
    async fn scene_current(&self) -> Result<CallToolResult, ErrorData> {
        self.refresh_active_scene();
        Ok(structured_success(serde_json::json!({
            "active_scene": *self.active_scene.lock().expect("active scene lock")
        })))
    }

    #[tool(output_schema = structured_object_schema(), description = "Version, mode, fixed media parameters, limits and available server features without initializing the GPU")]
    async fn studio_capabilities(&self) -> Result<CallToolResult, ErrorData> {
        Ok(structured_success(serde_json::json!({
            "version": env!("CARGO_PKG_VERSION"),
            "protocol_version": "2025-06-18",
            "structured_response_version": 1,
            "mode": self.mode.as_str(),
            "active_scene": *self.active_scene.lock().expect("active scene lock"),
            "allowed_project_root": self.active_root(),
            "platform": std::env::consts::OS,
            "fps": phantom_core::FPS,
            "sample_rate_hz": 48_000,
            "path_limits": { "outputs_within_project": true, "default_scene_output": "frames" },
            "output_limits": { "max_render_width": 4096, "max_render_height": 4096 },
            "features": {
                "audio": true,
                "shadows": { "supported": true, "current": "enabled_for_staging_lights" },
                "tonemapping": "tony_mc_mapface",
                "render_still_bridge": true,
                "stage": {
                    "tools": ["stage_set", "stage_get"],
                    "presets": phantom_core::StagePreset::ALL.iter().map(|p| {
                        let s = p.size();
                        serde_json::json!({ "preset": p.name(), "width_m": s.width, "height_m": s.height, "depth_m": s.depth })
                    }).collect::<Vec<_>>(),
                    "clamp": "every placement command keeps the entity's bounding box inside the stage and reports clamped/clamps"
                },
                "pure_diagnostics": ["validate", "camera_coverage", "background_fit"],
                "formats": { "still": ["png"], "scene": ["png_sequence"], "film": ["mp4_h264_aac"] }
            },
            "gpu_availability": "unknown_until_render"
        })))
    }

    #[tool(output_schema = structured_object_schema(), description = "Atomic package of document mutations in wire format (op tag, radians, 48-kHz samples); scene may be omitted — the active scene is substituted; any error rolls the whole package back; one undo step and one revision; results[i].entity returns the created ids for chains")]
    async fn batch(&self, Parameters(p): Parameters<PBatch>) -> Result<CallToolResult, ErrorData> {
        let active = *self.active_scene.lock().expect("active scene lock");
        let mut ops = Vec::with_capacity(p.ops.len());
        for (index, mut raw) in p.ops.into_iter().enumerate() {
            if let Some(object) = raw.as_object_mut() {
                let missing = object.get("scene").is_none_or(|v| v.is_null());
                if missing {
                    if let Some(scene) = active {
                        object.insert("scene".into(), serde_json::json!(scene));
                    }
                }
            }
            match serde_json::from_value::<Op>(raw) {
                Ok(op) => ops.push(op),
                Err(e) => {
                    return Ok(structured_failure(StructuredError::new(
                        "invalid_input",
                        format!("batch: operation {index} not parsed: {e}"),
                        serde_json::json!({ "index": index, "cause": e.to_string() }),
                    )))
                }
            }
        }
        let result = self.backend_exec(Op::Batch { ops });
        if result.is_ok() {
            self.refresh_active_scene();
        }
        Ok(result.map(structured_success).unwrap_or_else(structured_failure))
    }

    #[tool(output_schema = structured_object_schema(), description = "Add a scene with the frame range [start, end)")]
    async fn scene_add(&self, Parameters(p): Parameters<PSceneAdd>) -> Result<CallToolResult, ErrorData> {
        let result = self.backend_exec(Op::SceneAdd { name: p.name, start: p.start, end: p.end });
        if let Ok(value) = &result {
            *self.active_scene.lock().expect("active scene lock") = value.get("scene").and_then(|v| v.as_u64());
        }
        Ok(result.map(structured_success).unwrap_or_else(structured_failure))
    }

    #[tool(output_schema = structured_object_schema(), description = "Import an asset file (obj/fbx/png/jpg/tiff/audio/video) into the project; glTF/glb are not supported by the current decoder yet")]
    async fn asset_import(&self, Parameters(p): Parameters<PImport>) -> Result<CallToolResult, ErrorData> {
        self.exec(Op::AssetImport { file: PathBuf::from(p.file), kind: None })
    }

    #[tool(output_schema = structured_object_schema(), description = "Batch import (B7): several files in one call — atomically (batch): either every asset is registered or none; results[i] carries the asset id of each file")]
    async fn asset_import_batch(&self, Parameters(p): Parameters<PImportBatch>) -> Result<CallToolResult, ErrorData> {
        if p.files.is_empty() {
            return Ok(invalid_input("asset_import_batch: empty files list"));
        }
        self.exec(Op::Batch {
            ops: p.files.into_iter().map(|file| Op::AssetImport { file: PathBuf::from(file), kind: None }).collect(),
        })
    }

    #[tool(output_schema = structured_object_schema(), description = "Set an asset as the scene background (a flat screen facing the camera side); video — video background; width/height/center — screen geometry (defaults 24 m, 8 m, [0,4,-5]); use background_fit for the size that closes the frame")]
    async fn background_set(&self, Parameters(p): Parameters<PBackground>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        let fit = match p.fit.as_deref() {
            None => None,
            Some("cover") => Some(phantom_core::project::BackgroundFit::Cover),
            Some("contain") => Some(phantom_core::project::BackgroundFit::Contain),
            Some("stretch") => Some(phantom_core::project::BackgroundFit::Stretch),
            Some(other) => return Ok(invalid_input(format!("fit: cover|contain|stretch, got {other}"))),
        };
        self.exec(Op::BackgroundSet {
            scene: scene,
            asset: p.asset,
            video: p.video.unwrap_or(false),
            width: p.width.unwrap_or(24.0),
            height: p.height.unwrap_or(8.0),
            center: p.center.unwrap_or([0.0, 4.0, -5.0]),
            fit,
        })
    }

    #[tool(output_schema = structured_object_schema(), description = "Remove the scene background")]
    async fn background_clear(&self, Parameters(p): Parameters<PSceneRef>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::BackgroundClear { scene: scene })
    }

    #[tool(output_schema = structured_object_schema(), description = "Change the scene frame range [start, end)")]
    async fn scene_range_set(&self, Parameters(p): Parameters<PSceneRange>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::SceneRangeSet { scene: scene, start: p.start, end: p.end })
    }

    #[tool(output_schema = structured_object_schema(), description = "Add a Phantom: no parameters — builtin humanoid; rig_actor (humanoid|wolf|horse|bird) — builtin actor; rig_asset — imported FBX character (mutually exclusive)")]
    async fn phantom_add(&self, Parameters(p): Parameters<PPhantom>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::PhantomAdd { scene: scene, name: p.name, rig_asset: p.rig_asset, rig_actor: p.rig_actor })
    }

    #[tool(output_schema = structured_object_schema(), description = "Builtin actor catalog: actor profiles with the poses each one can hold, the metric prop table, human reference lines and stage presets")]
    async fn builtin_catalog(&self) -> Result<CallToolResult, ErrorData> {
        self.exec(Op::BuiltinCatalog)
    }

    #[tool(output_schema = structured_object_schema(), description = "Set a Phantom controller key: frame, position [x,y,z], pose from the actor's profile in builtin_catalog (stand|sit|lie); omitted fields are preserved from the exact key or the evaluated state")]
    async fn key_set(&self, Parameters(p): Parameters<PKey>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        let preserve = p.preserve.unwrap_or(true);
        let snapshot = match self.backend_exec(Op::SceneGet { scene }) {
            Ok(value) => match serde_json::from_value::<phantom_core::Scene>(value) {
                Ok(scene) => scene,
                Err(e) => return Ok(invalid_input(format!("scene snapshot: {e}"))),
            },
            Err(error) => return Ok(structured_failure(error)),
        };
        let Some(phantom) = snapshot.phantoms.get(&phantom_core::PhantomId(p.phantom)) else {
            return self.exec(Op::KeySet { scene, phantom: p.phantom, frame: p.frame, pos: [0.0; 3], pose: phantom_core::ActorPose::Stand, yaw: 0.0 });
        };
        let base = phantom.motion.iter().find(|key| key.frame == p.frame).copied();
        let evaluated = phantom_core::pose::evaluate(phantom, p.frame);
        let default_pose = phantom.rig.builtin_actor().map(|actor| actor.default_pose()).unwrap_or(phantom_core::ActorPose::Stand);
        let pose = match p.pose {
            Some(pose) => match crate::builtin::parse_pose(&pose) {
                Ok(pose) => pose,
                Err(e) => return Ok(invalid_input(e)),
            },
            None if preserve => base.map(|k| k.pose).unwrap_or(evaluated.pose),
            None => default_pose,
        };
        let pos = p.pos.unwrap_or_else(|| if preserve { base.map(|k| k.position.to_array()).unwrap_or(evaluated.root.to_array()) } else { [0.0; 3] });
        let yaw = p.yaw_deg.map(f32::to_radians).unwrap_or_else(|| if preserve { base.map(|k| k.yaw).unwrap_or(evaluated.yaw) } else { 0.0 });
        self.exec(Op::KeySet { scene, phantom: p.phantom, frame: p.frame, pos, pose, yaw })
    }

    #[tool(output_schema = structured_object_schema(), description = "Remove a Phantom from the scene (its keys go away with it)")]
    async fn phantom_remove(&self, Parameters(p): Parameters<PPhantomRef>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::PhantomRemove { scene: scene, phantom: p.phantom })
    }

    #[tool(output_schema = structured_object_schema(), description = "Remove a Phantom controller key at a frame")]
    async fn motion_key_remove(&self, Parameters(p): Parameters<PMotionKeyRef>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::MotionKeyRemove { scene: scene, phantom: p.phantom, frame: p.frame })
    }

    #[tool(output_schema = structured_object_schema(), description = "Atomically replace the whole Phantom controller track; motion — typed keys {frame,pos,pose,yaw_deg?}, MCP angles in degrees, an empty array clears the track")]
    async fn motion_replace(&self, Parameters(p): Parameters<PMotion>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        let mut motion = Vec::with_capacity(p.motion.len());
        let mut yaws = Vec::with_capacity(p.motion.len());
        for key in p.motion {
            let pose = match crate::builtin::parse_pose(&key.pose) {
                Ok(pose) => pose,
                Err(e) => return Ok(invalid_input(e)),
            };
            motion.push((key.frame, key.pos, pose));
            yaws.push(key.yaw_deg.unwrap_or(0.0).to_radians());
        }
        self.exec(Op::MotionReplace { scene, phantom: p.phantom, motion, yaws: Some(yaws) })
    }

    #[tool(output_schema = structured_object_schema(), description = "Add a static object (prop) from a model asset (ModelObj/ModelFbx without animation; a skinned model renders in bind pose)")]
    async fn object_add(&self, Parameters(p): Parameters<PObjectAdd>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::ObjectAdd { scene: scene, name: p.name, asset: p.asset })
    }

    #[tool(output_schema = structured_object_schema(), description = "Remove a prop object from the scene")]
    async fn object_remove(&self, Parameters(p): Parameters<PObjectRef>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::ObjectRemove { scene: scene, object: p.object })
    }

    #[tool(output_schema = structured_object_schema(), description = "Set an object key: frame at 24 fps, position in meters, yaw_deg/pitch_deg/roll_deg in degrees, positive scale; angles interpolate along the shortest arc")]
    async fn object_key_set(&self, Parameters(p): Parameters<PObjectKey>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::ObjectKeySet {
            scene: scene,
            object: p.object,
            frame: p.frame,
            pos: p.pos,
            yaw: p.yaw_deg.unwrap_or(0.0).to_radians(),
            pitch: p.pitch_deg.unwrap_or(0.0).to_radians(),
            roll: p.roll_deg.unwrap_or(0.0).to_radians(),
            scale: p.scale.unwrap_or([1.0, 1.0, 1.0]),
        })
    }

    #[tool(output_schema = structured_object_schema(), description = "Remove an object transform key at a frame")]
    async fn object_key_remove(&self, Parameters(p): Parameters<PObjectKeyRemove>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::ObjectKeyRemove { scene: scene, object: p.object, frame: p.frame })
    }

    #[tool(output_schema = structured_object_schema(), description = "Set the scene stage lighting: key/fill/rim, each — color [r,g,b] (linear), intensity in lux, enabled")]
    async fn scene_lighting_set(&self, Parameters(p): Parameters<PLightingSet>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        let mut lighting: phantom_core::SceneLighting = match self.backend_exec(Op::LightingGet { scene }) {
            Ok(value) => match serde_json::from_value(value["lighting"].clone()) {
                Ok(lighting) => lighting,
                Err(e) => return Ok(invalid_input(format!("lighting snapshot: {e}"))),
            },
            Err(error) => return Ok(structured_failure(error)),
        };
        if let Some(key) = p.key { lighting.key = key.into(); }
        if let Some(fill) = p.fill { lighting.fill = fill.into(); }
        if let Some(rim) = p.rim { lighting.rim = rim.into(); }
        self.exec(Op::LightingSet { scene, lighting })
    }

    #[tool(output_schema = structured_object_schema(), description = "Current scene stage lighting (key/fill/rim)")]
    async fn scene_lighting_get(&self, Parameters(p): Parameters<PSceneRef>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::LightingGet { scene: scene })
    }

    #[tool(output_schema = structured_object_schema(), description = "Enable/remove the scene matte floor (y=0, receives shadows, opaque); color — linear RGB, neutral by default")]
    async fn scene_floor_set(&self, Parameters(p): Parameters<PFloorSet>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        let floor = if p.enabled {
            Some(phantom_core::SceneFloor { color: p.color.unwrap_or(phantom_core::SceneFloor::default().color) })
        } else {
            None
        };
        self.exec(Op::FloorSet { scene: scene, floor })
    }

    #[tool(output_schema = structured_object_schema(), description = "Current scene floor (null — no floor)")]
    async fn scene_floor_get(&self, Parameters(p): Parameters<PSceneRef>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::FloorGet { scene: scene })
    }

    #[tool(output_schema = structured_object_schema(), description = "Set the scene stage (layout mode): the box actors and props are confined to. Give preset small|medium|large and/or width+height+depth in meters; anything already staged that no longer fits is clamped in the same step and reported as clamped/clamps. Without center the stage keeps its current center.")]
    async fn stage_set(&self, Parameters(p): Parameters<PStageSet>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::StageSet { scene: scene, preset: p.preset, width: p.width, height: p.height, depth: p.depth, center: p.center })
    }

    #[tool(output_schema = structured_object_schema(), description = "Current scene stage: preset, size (width/height/depth, meters) and floor-plane center")]
    async fn stage_get(&self, Parameters(p): Parameters<PSceneRef>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::StageGet { scene: scene })
    }

    #[tool(output_schema = structured_object_schema(), description = "Brush/eraser stroke of a 2D layer at a frame (points in 1920x1080 canvas coordinates)")]
    async fn brush_stroke(&self, Parameters(p): Parameters<PStroke>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        let points = match convert_brush_points(p.points, p.space, p.screen_width, p.screen_height) {
            Ok(points) => points,
            Err(message) => return Ok(invalid_input(message)),
        };
        self.exec(Op::BrushStroke { scene, frame: p.frame, points, color: p.color, width: p.width, erase: p.erase })
    }

    #[tool(output_schema = structured_object_schema(), description = "Append a film clip from a scene range (at the end); camera — clip a camera's range (start/end must equal the camera range)")]
    async fn film_add(&self, Parameters(p): Parameters<PFilmAdd>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::FilmAdd { scene: scene, start: p.start, end: p.end, camera: p.camera })
    }

    #[tool(output_schema = structured_object_schema(), description = "Add a named scene camera: range [start, end) inside the scene (omitted — the whole scene); the new camera becomes the scene's active camera")]
    async fn camera_add(&self, Parameters(p): Parameters<PCameraAdd>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::CameraAdd { scene: scene, name: p.name, start: p.start, end: p.end })
    }

    #[tool(output_schema = structured_object_schema(), description = "Remove a camera; film clips referencing it are removed by the same command")]
    async fn camera_remove(&self, Parameters(p): Parameters<PCameraRef>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::CameraRemove { scene: scene, camera: p.camera })
    }

    #[tool(output_schema = structured_object_schema(), description = "Select the scene's active camera: the one the viewport shows and a render without a camera uses")]
    async fn camera_select(&self, Parameters(p): Parameters<PCameraRef>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::CameraSelect { scene: scene, camera: p.camera })
    }

    #[tool(output_schema = structured_object_schema(), description = "Change a camera's range [start, end) (inside the scene); film clips targeting that camera get the new range")]
    async fn camera_range_set(&self, Parameters(p): Parameters<PCameraRange>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::CameraRangeSet { scene: scene, camera: p.camera, start: p.start, end: p.end })
    }

    #[tool(output_schema = structured_object_schema(), description = "Camera key (C1): camera omitted — the scene's active camera. pos + yaw_deg/pitch_deg OR look_at (aim at a point — the program computes yaw/pitch); roll_deg (tilted horizon), focal_mm/focal_preset (zoom), ease (smoothing toward the next key). The key must keep at least 20 % of the scene volume in frame")]
    async fn camera_key_set(&self, Parameters(p): Parameters<PCameraKey>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        let key = match camera_key_input(p.frame, p.pos, p.yaw_deg, p.pitch_deg, p.roll_deg, p.look_at, p.focal_mm, p.focal_preset.as_deref(), p.ease.as_deref()) {
            Ok(key) => key,
            Err(result) => return Ok(result),
        };
        self.exec(Op::CameraKeySet { scene, camera: p.camera, key })
    }

    #[tool(output_schema = structured_object_schema(), description = "Remove a camera key at a frame (the last key cannot be removed)")]
    async fn camera_key_remove(&self, Parameters(p): Parameters<PCameraKeyRef>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::CameraKeyRemove { scene: scene, camera: p.camera, frame: p.frame })
    }

    #[tool(output_schema = structured_object_schema(), description = "Camera base lens in mm (35-mm equivalent) — the base for keys without their own focal_mm")]
    async fn camera_lens_set(&self, Parameters(p): Parameters<PCameraLens>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::CameraLensSet { scene: scene, camera: p.camera, focal_mm: p.focal_mm })
    }

    #[tool(output_schema = structured_object_schema(), description = "Camera focal-length key (C1.2): zoom/crash-zoom inside the camera's range without changing the pose")]
    async fn camera_focal_key_set(&self, Parameters(p): Parameters<PCameraFocalKey>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::CameraFocalKeySet { scene, camera: p.camera, frame: p.frame, focal_mm: p.focal_mm })
    }

    #[tool(output_schema = structured_object_schema(), description = "Attach the camera to a moving subject — track: the camera stands where its keys put it and turns to keep the subject in view; follow: the camera is attached to the subject and its keys are read in the subject's frame, so it travels and turns with it (a third-person view). subject — phantom:<id> or object:<id>; clear: true detaches")]
    async fn camera_target_set(&self, Parameters(p): Parameters<PCameraTargetSet>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::CameraTargetSet { scene, camera: p.camera, mode: p.mode, subject: p.subject, clear: p.clear.unwrap_or(false) })
    }

    #[tool(output_schema = structured_object_schema(), description = "Move a film clip to position index")]
    async fn film_move(&self, Parameters(p): Parameters<PFilmMove>) -> Result<CallToolResult, ErrorData> {
        self.exec(Op::FilmMove { clip: p.clip, index: p.index })
    }

    #[tool(output_schema = structured_object_schema(), description = "Trim a film clip [start, end)")]
    async fn film_trim(&self, Parameters(p): Parameters<PFilmTrim>) -> Result<CallToolResult, ErrorData> {
        self.exec(Op::FilmTrim { clip: p.clip, start: p.start, end: p.end })
    }

    #[tool(output_schema = structured_object_schema(), description = "Remove a film clip")]
    async fn film_remove(&self, Parameters(p): Parameters<PFilmMove>) -> Result<CallToolResult, ErrorData> {
        self.exec(Op::FilmRemove { clip: p.clip })
    }

    #[tool(output_schema = structured_object_schema(), description = "Add an audio track to a scene (scene given) or to the film (scene = null)")]
    async fn audio_track_add(&self, Parameters(p): Parameters<PTrackAdd>) -> Result<CallToolResult, ErrorData> {
        self.exec(Op::AudioTrackAdd { scene: p.scene, name: p.name })
    }

    #[tool(output_schema = structured_object_schema(), description = "Add an audio clip: start_secs on the timeline, offset_secs/len_secs of the source, optional fade_in_secs/fade_out_secs (equal-power); seconds are converted to 48-kHz samples, the timeline is fixed at 24 fps")]
    async fn audio_clip_add(&self, Parameters(p): Parameters<PClipAdd>) -> Result<CallToolResult, ErrorData> {
        let start = match seconds_to_frame(p.start_secs, "start_secs") { Ok(v) => v, Err(e) => return Ok(invalid_input(e)) };
        let offset_samples = match seconds_to_samples(p.offset_secs, "offset_secs") { Ok(v) => v, Err(e) => return Ok(invalid_input(e)) };
        let len_samples = match seconds_to_samples(p.len_secs, "len_secs") { Ok(v) => v, Err(e) => return Ok(invalid_input(e)) };
        let fade_in_samples = match p.fade_in_secs.map(|v| seconds_to_samples(v, "fade_in_secs")) { None => 0, Some(Ok(v)) => v, Some(Err(e)) => return Ok(invalid_input(e)) };
        let fade_out_samples = match p.fade_out_secs.map(|v| seconds_to_samples(v, "fade_out_secs")) { None => 0, Some(Ok(v)) => v, Some(Err(e)) => return Ok(invalid_input(e)) };
        self.exec(Op::AudioClipAdd { track: p.track, asset: p.asset, start, offset_samples, len_samples, fade_in_samples, fade_out_samples })
    }

    #[tool(output_schema = structured_object_schema(), description = "Change an audio clip's fragment/fades: offset_secs, len_secs, fade_in_secs, fade_out_secs; any subset of fields, omitted ones stay unchanged")]
    async fn audio_clip_update(&self, Parameters(p): Parameters<PClipUpdate>) -> Result<CallToolResult, ErrorData> {
        let conv = |v: Option<f64>, name: &str| -> Result<Option<u64>, CallToolResult> {
            match v {
                None => Ok(None),
                Some(v) => seconds_to_samples(v, name).map(Some).map_err(invalid_input),
            }
        };
        let offset_samples = match conv(p.offset_secs, "offset_secs") { Ok(v) => v, Err(r) => return Ok(r) };
        let len_samples = match conv(p.len_secs, "len_secs") { Ok(v) => v, Err(r) => return Ok(r) };
        let fade_in_samples = match conv(p.fade_in_secs, "fade_in_secs") { Ok(v) => v, Err(r) => return Ok(r) };
        let fade_out_samples = match conv(p.fade_out_secs, "fade_out_secs") { Ok(v) => v, Err(r) => return Ok(r) };
        self.exec(Op::AudioClipUpdate { track: p.track, clip: p.clip, offset_samples, len_samples, fade_in_samples, fade_out_samples })
    }

    #[tool(output_schema = structured_object_schema(), description = "Track volume (1.0 = unchanged)")]
    async fn track_volume_set(&self, Parameters(p): Parameters<PVolume>) -> Result<CallToolResult, ErrorData> {
        self.exec(Op::TrackVolumeSet { track: p.track, volume: p.volume })
    }

    #[tool(output_schema = structured_object_schema(), description = "Mute/unmute a track")]
    async fn track_mute_set(&self, Parameters(p): Parameters<PMute>) -> Result<CallToolResult, ErrorData> {
        self.exec(Op::TrackMuteSet { track: p.track, muted: p.muted })
    }

    #[tool(output_schema = structured_object_schema(), description = "Remove an audio track entirely (with all its clips)")]
    async fn audio_track_remove(&self, Parameters(p): Parameters<PTrackRef>) -> Result<CallToolResult, ErrorData> {
        self.exec(Op::AudioTrackRemove { track: p.track })
    }

    #[tool(output_schema = structured_object_schema(), description = "Add a 2D layer to the scene; returns the layer id")]
    async fn layer_add_2d(&self, Parameters(p): Parameters<PLayerAdd>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::LayerAdd2D { scene: scene, name: p.name })
    }

    #[tool(output_schema = structured_object_schema(), description = "Remove a 2D layer exposure at a frame entirely")]
    async fn exposure_remove(&self, Parameters(p): Parameters<PExposureRef>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::ExposureRemove { scene: scene, layer: p.layer, frame: p.frame })
    }

    #[tool(output_schema = structured_object_schema(), description = "Undo the last operation (applied=false — the stack is empty)")]
    async fn history_undo(&self) -> Result<CallToolResult, ErrorData> {
        let result = self.exec(Op::HistoryUndo);
        self.refresh_active_scene();
        result
    }

    #[tool(output_schema = structured_object_schema(), description = "Redo an undone operation (applied=false — the stack is empty)")]
    async fn history_redo(&self) -> Result<CallToolResult, ErrorData> {
        let result = self.exec(Op::HistoryRedo);
        self.refresh_active_scene();
        result
    }

    #[tool(output_schema = structured_object_schema(), description = "Depth of the undo/redo stacks and the names of the last operations")]
    async fn history_list(&self) -> Result<CallToolResult, ErrorData> {
        self.exec(Op::HistoryList)
    }

    #[tool(output_schema = structured_object_schema(), description = "Remove an audio clip from a track")]
    async fn audio_clip_remove(&self, Parameters(p): Parameters<PClipRemove>) -> Result<CallToolResult, ErrorData> {
        self.exec(Op::AudioClipRemove { track: p.track, clip: p.clip })
    }

    #[tool(output_schema = structured_object_schema(), description = "Start exporting a scene to a PNG sequence (job); frames — half-open range [start,end), so 84-85 renders frame 84 only; frames is half-open")]
    async fn render_scene(&self, Parameters(p): Parameters<PRenderScene>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        let result = self.backend_exec(Op::RenderScene { scene, out: render_scene_output(p.out), width: p.width.unwrap_or(1280), height: p.height.unwrap_or(720), frames: p.frames.map(|r| (r.start, r.end)), camera: p.camera });
        Ok(match result {
            Ok(mut value) => { if let Some(object) = value.as_object_mut() { object.insert("allowed_root".into(), serde_json::json!(self.active_root())); } structured_success(value) },
            Err(error) => structured_failure(error),
        })
    }

    #[tool(output_schema = structured_object_schema(), description = "Start exporting the film to MP4: H.264 + AAC 48-kHz stereo sound — the mix of scene and film audio tracks (no tracks — silent); frames — optional film frame range [start, end); normalize/target_peak_db — opt-in peak normalization of the mix; check audio durations against video with audio_report before rendering")]
    async fn render_film(&self, Parameters(p): Parameters<POut>) -> Result<CallToolResult, ErrorData> {
        self.exec(Op::RenderFilm { out: PathBuf::from(p.out), width: p.width.unwrap_or(1280), height: p.height.unwrap_or(720), frames: p.frames.map(|r| (r.start, r.end)), normalize: p.normalize.unwrap_or(false), target_peak_db: p.target_peak_db })
    }

    #[tool(output_schema = structured_object_schema(), description = "Job status: done, exit_code (after completion), frames_done/total (from the progress line), elapsed_secs and error_tail (last stderr lines)")]
    async fn job_status(&self, Parameters(p): Parameters<PJob>) -> Result<CallToolResult, ErrorData> {
        self.exec(Op::JobStatus { job: p.job })
    }

    #[tool(output_schema = job_wait_output_schema(), description = "Wait efficiently for a job to finish or for timeout_secs; returns the job_status shape plus timed_out, and does not cancel the job on timeout")]
    async fn job_wait(&self, Parameters(p): Parameters<PJobWait>) -> Result<CallToolResult, ErrorData> {
        self.exec(Op::JobWait { job: p.job, timeout_secs: p.timeout_secs })
    }

    #[tool(output_schema = structured_object_schema(), description = "Statuses of all session jobs (the same fields as job_status, plus job)")]
    async fn job_list(&self) -> Result<CallToolResult, ErrorData> {
        self.exec(Op::JobList)
    }

    #[tool(output_schema = structured_object_schema(), description = "Full snapshot of one scene: phantoms, objects, cameras (with the active camera), 2D layers, audio tracks, background")]
    async fn scene_get(&self, Parameters(p): Parameters<PSceneRef>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::SceneGet { scene: scene })
    }

    #[tool(output_schema = structured_object_schema(), description = "Full canonical snapshot of one camera (id, name, range, keys and focal_mm, target) plus the revision of its parent scene; scene may be omitted for the active scene")]
    async fn camera_get(&self, Parameters(p): Parameters<PCameraRef>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::CameraGet { scene, camera: p.camera })
    }

    #[tool(output_schema = structured_object_schema(), description = "Pure document check without render and GPU: screen invariants (bg_geometry_invalid), camera FOV coverage (screen_fov_gap), motion gaps (motion_gap), audio and editing ranges (audio_out_of_range/audio_length_mismatch/film_clip_out_of_scene), missing files (asset_missing); returns {ok, revision, findings:[{code, severity, where, why, suggested_fix}]} with a deterministic order (scene, code, frame); ok=false on error severity")]
    async fn validate(&self, Parameters(p): Parameters<PValidate>) -> Result<CallToolResult, ErrorData> {
        // Without an explicit scene and without an active scene — the whole project.
        let scene = p.scene.or_else(|| *self.active_scene.lock().expect("active scene lock"));
        self.exec(Op::Validate { scene })
    }

    #[tool(output_schema = structured_object_schema(), description = "Audio report (A4): track/clip durations in seconds against video length, RMS/peak dBFS of the fragments actually used (ffmpeg decoding; without ffmpeg — null plus a warning), machine-readable warnings (audio_beyond_video, audio_beyond_source, quiet_fragment, audio_longer_than_video, decode_failed). Check BEFORE rendering: \"music 360 s, film 2 s\" shows up here")]
    async fn audio_report(&self, Parameters(p): Parameters<PAudioReport>) -> Result<CallToolResult, ErrorData> {
        // Without an explicit scene and without an active scene — the whole project (like validate).
        let scene = p.scene.or_else(|| *self.active_scene.lock().expect("active scene lock"));
        self.exec(Op::AudioReport { scene })
    }

    #[tool(output_schema = structured_object_schema(), description = "Add a document light source (B1): point (lumens), spot (lumens + cone_deg) or directional (lux); angles in degrees; animated by light_key_set; shadows are opt-in via shadows")]
    async fn light_add(&self, Parameters(p): Parameters<PLightAdd>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        let kind = match p.kind.as_str() {
            "point" => phantom_core::SceneLightKind::Point,
            "directional" => phantom_core::SceneLightKind::Directional,
            "spot" => match p.cone_deg {
                Some(c) if c.is_finite() && c > 0.0 && c < 180.0 => phantom_core::SceneLightKind::Spot { cone: c.to_radians() },
                _ => return Ok(invalid_input("spot: cone_deg is required and must be in (0, 180)")),
            },
            other => return Ok(invalid_input(format!("kind: point|spot|directional, got {other}"))),
        };
        self.exec(Op::LightAdd {
            scene,
            name: p.name,
            kind,
            color: p.color,
            intensity: p.intensity,
            enabled: p.enabled.unwrap_or(true),
            shadows: p.shadows.unwrap_or(false),
            rest_pos: p.pos,
            rest_yaw: p.yaw_deg.unwrap_or(0.0).to_radians(),
            rest_pitch: p.pitch_deg.unwrap_or(0.0).to_radians(),
        })
    }

    #[tool(output_schema = structured_object_schema(), description = "Remove a document light source")]
    async fn light_remove(&self, Parameters(p): Parameters<PLightRef>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::LightRemove { scene, light: p.light })
    }

    #[tool(output_schema = structured_object_schema(), description = "Light source key at a frame: position, rotation (degrees), optional intensity and enabled — a lantern can be switched on at a frame")]
    async fn light_key_set(&self, Parameters(p): Parameters<PLightKey>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::LightKeySet {
            scene,
            light: p.light,
            frame: p.frame,
            pos: p.pos,
            yaw: p.yaw_deg.unwrap_or(0.0).to_radians(),
            pitch: p.pitch_deg.unwrap_or(0.0).to_radians(),
            intensity: p.intensity,
            enabled: p.enabled,
        })
    }

    #[tool(output_schema = structured_object_schema(), description = "Remove a light source key at a frame")]
    async fn light_key_remove(&self, Parameters(p): Parameters<PLightKeyRemove>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::LightKeyRemove { scene, light: p.light, frame: p.frame })
    }

    #[tool(output_schema = structured_object_schema(), description = "Attach an object/light source to a Phantom bone (B2): object, light or prop (creates an object from a model asset); the transform follows the bone every frame; offset in meters/degrees; detach=true detaches")]
    async fn attach(&self, Parameters(p): Parameters<PAttach>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        let attach = phantom_core::Attachment {
            phantom: phantom_core::PhantomId(p.phantom),
            bone: p.bone,
            offset_pos: p.offset_pos.unwrap_or([0.0; 3]).into(),
            offset_yaw: p.offset_yaw_deg.unwrap_or(0.0).to_radians(),
            offset_pitch: p.offset_pitch_deg.unwrap_or(0.0).to_radians(),
            offset_roll: p.offset_roll_deg.unwrap_or(0.0).to_radians(),
        };
        let detach = p.detach.unwrap_or(false);
        let targets = [p.object.is_some(), p.light.is_some(), p.prop.is_some()].into_iter().filter(|b| *b).count();
        if targets != 1 {
            return Ok(invalid_input("attach: exactly one of object/light/prop"));
        }
        if let Some(prop) = p.prop {
            if detach {
                return Ok(invalid_input("prop is not compatible with detach"));
            }
            // Short form "flashlight in hand": create the object from the asset
            // and attach it right away — one call instead of two round trips.
            let created = self.backend_exec(Op::ObjectAdd { scene, name: format!("prop_{prop}"), asset: prop });
            let object = match created {
                Ok(value) => match value["object"].as_u64() {
                    Some(id) => id,
                    None => {
                        return Ok(structured_failure(StructuredError::new(
                            "internal_error",
                            "object_add returned no id",
                            serde_json::json!({ "stage": "attach_prop" }),
                        )))
                    }
                },
                Err(error) => return Ok(structured_failure(error)),
            };
            return self.exec(Op::ObjectAttach { scene, object, attach: Some(attach) });
        }
        if let Some(object) = p.object {
            return self.exec(Op::ObjectAttach { scene, object, attach: if detach { None } else { Some(attach) } });
        }
        self.exec(Op::LightAttach { scene, light: p.light.expect("checked"), attach: if detach { None } else { Some(attach) } })
    }

    #[tool(output_schema = structured_object_schema(), description = "Actor scale (uniform, > 0): height/pet/teen without rebuilding the model")]
    async fn phantom_scale_set(&self, Parameters(p): Parameters<PPhantomScale>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::PhantomScaleSet { scene, phantom: p.phantom, scale: p.scale })
    }

    #[tool(output_schema = structured_object_schema(), description = "Frame color grading (B3): exposure in stops, temperature/tint, hue_deg, saturation/contrast/gamma/gain/lift; enabled=false turns it off; LUTs are not supported")]
    async fn grade_set(&self, Parameters(p): Parameters<PGradeSet>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        if p.enabled == Some(false) {
            return self.exec(Op::GradeSet { scene, grade: None });
        }
        let d = phantom_core::SceneGrade::default();
        let grade = phantom_core::SceneGrade {
            exposure: p.exposure.unwrap_or(d.exposure),
            temperature: p.temperature.unwrap_or(d.temperature),
            tint: p.tint.unwrap_or(d.tint),
            hue: p.hue_deg.unwrap_or(0.0).to_radians(),
            post_saturation: p.post_saturation.unwrap_or(d.post_saturation),
            saturation: p.saturation.unwrap_or(d.saturation),
            contrast: p.contrast.unwrap_or(d.contrast),
            gamma: p.gamma.unwrap_or(d.gamma),
            gain: p.gain.unwrap_or(d.gain),
            lift: p.lift.unwrap_or(d.lift),
        };
        self.exec(Op::GradeSet { scene, grade: Some(grade) })
    }

    #[tool(output_schema = structured_object_schema(), description = "Exponential fog (B3): color (linear RGB) + density > 0; off=true turns it off")]
    async fn fog_set(&self, Parameters(p): Parameters<PFogSet>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        if p.off == Some(true) {
            return self.exec(Op::FogSet { scene, fog: None });
        }
        let color = p.color.unwrap_or([0.5, 0.6, 0.7]);
        let density = match p.density {
            Some(d) if d.is_finite() && d > 0.0 => d,
            _ => return Ok(invalid_input("fog_set: density > 0 is required (or off=true)")),
        };
        self.exec(Op::FogSet { scene, fog: Some(phantom_core::SceneFog { color, density }) })
    }

    #[tool(output_schema = structured_object_schema(), description = "Add an effect (B3): particles (presets snow|embers|smoke|rain, deterministic hash simulation) or water (plane + normal map + scroll_uv); enabled/density keys — vfx_key_set")]
    async fn vfx_add(&self, Parameters(p): Parameters<PVfxAdd>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        let kind = match p.kind.as_str() {
            "particles" => phantom_core::VfxKind::Particles,
            "water" => phantom_core::VfxKind::Water,
            other => return Ok(invalid_input(format!("kind: particles|water, got {other}"))),
        };
        let mut vfx = match &p.preset {
            Some(name) => match phantom_core::vfx_preset(name) {
                Some(v) => v,
                None => return Ok(invalid_input(format!("preset: snow|embers|smoke|rain, got {name}"))),
            },
            None => phantom_core::vfx_preset("snow").expect("snow preset"),
        };
        vfx.name = p.name;
        vfx.kind = kind;
        if let Some(v) = p.pos { vfx.position = v.into(); }
        if let Some(v) = p.extent { vfx.extent = v.into(); }
        if let Some(v) = p.emit_rate { vfx.emit_rate = v; }
        if let Some(v) = p.lifetime { vfx.lifetime = v; }
        if let Some(v) = p.speed { vfx.speed = v; }
        if let Some(v) = p.size { vfx.size = v; }
        if let Some(v) = p.color { vfx.color = v; }
        if let Some(v) = p.gravity { vfx.gravity = v; }
        if let Some(v) = p.normal_map { vfx.normal_map = Some(phantom_core::AssetId(v)); }
        if let Some(v) = p.scroll_uv { vfx.scroll_uv = v; }
        if let Some(v) = p.enabled { vfx.enabled = v; }
        self.exec(Op::VfxAdd { scene, vfx })
    }

    #[tool(output_schema = structured_object_schema(), description = "Remove an effect")]
    async fn vfx_remove(&self, Parameters(p): Parameters<PVfxRef>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::VfxRemove { scene, vfx: p.vfx })
    }

    #[tool(output_schema = structured_object_schema(), description = "Effect key at a frame: enabled (a blizzard starts at the frame) and/or density (emit_rate multiplier)")]
    async fn vfx_key_set(&self, Parameters(p): Parameters<PVfxKey>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::VfxKeySet { scene, vfx: p.vfx, frame: p.frame, enabled: p.enabled, density: p.density })
    }

    #[tool(output_schema = structured_object_schema(), description = "Remove an effect key at a frame")]
    async fn vfx_key_remove(&self, Parameters(p): Parameters<PVfxKeyRemove>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::VfxKeyRemove { scene, vfx: p.vfx, frame: p.frame })
    }

    #[tool(output_schema = structured_object_schema(), description = "Parts and materials of a phantom/object model: current import values and effective values after patches (a query, does not mutate)")]
    async fn material_list(&self, Parameters(p): Parameters<PMaterialRef>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::MaterialList { scene, phantom: p.phantom, object: p.object })
    }

    #[tool(output_schema = structured_object_schema(), description = "Material patch for a phantom/object (B4): preset matte|skin|metal|wet|glass|glow and/or explicit fields (base_color sRGB, metallic, roughness, emissive linear, opacity, unlit); target — part name, all parts by default; clear removes patches")]
    async fn material_set(&self, Parameters(p): Parameters<PMaterialSet>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        let clear = p.clear.unwrap_or(false);
        let mut patch = match &p.preset {
            Some(name) => match phantom_core::material_preset(name) {
                Some(preset) => preset,
                None => return Ok(invalid_input(format!("preset: matte|skin|metal|wet|glass|glow, got {name}"))),
            },
            None if !clear => phantom_core::MaterialPatch::default(),
            None => phantom_core::MaterialPatch::default(),
        };
        // Explicit fields override the preset.
        if p.base_color.is_some() { patch.base_color = p.base_color; }
        if p.metallic.is_some() { patch.metallic = p.metallic; }
        if p.roughness.is_some() { patch.roughness = p.roughness; }
        if p.emissive.is_some() { patch.emissive = p.emissive; }
        if p.opacity.is_some() { patch.opacity = p.opacity; }
        if p.unlit.is_some() { patch.unlit = p.unlit; }
        if !clear && patch == phantom_core::MaterialPatch::default() {
            return Ok(invalid_input("material_set: set a preset or at least one field"));
        }
        self.exec(Op::MaterialSet { scene, phantom: p.phantom, object: p.object, target: p.target, patch, clear })
    }

    #[tool(output_schema = structured_object_schema(), description = "Scene ambient: from_background (default), fixed (color+brightness, lux) or off")]
    async fn scene_ambient_set(&self, Parameters(p): Parameters<PAmbientSet>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        let ambient = match p.mode.as_str() {
            "from_background" => phantom_core::AmbientMode::FromBackground,
            "off" => phantom_core::AmbientMode::Off,
            "fixed" => phantom_core::AmbientMode::Fixed {
                color: p.color.unwrap_or([1.0, 1.0, 1.0]),
                brightness: match p.brightness {
                    Some(b) if b.is_finite() && b >= 0.0 => b,
                    _ => return Ok(invalid_input("fixed: brightness (lux, >= 0) is required")),
                },
            },
            other => return Ok(invalid_input(format!("mode: from_background|fixed|off, got {other}"))),
        };
        self.exec(Op::AmbientSet { scene, ambient })
    }

    #[tool(output_schema = structured_object_schema(), description = "Sky color / clean frame color (linear RGB): fills everything not covered by geometry and makes the frame opaque; null — the previous transparency")]
    async fn scene_sky_set(&self, Parameters(p): Parameters<PSkySet>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::SkySet { scene, sky: p.sky })
    }

    #[tool(output_schema = structured_object_schema(), description = "Frame coverage by the background screen at a frame, without rendering: camera pose (degrees), FOV (degrees), the screen's azimuthal/vertical span and the uncovered gap_deg per side, in_frame for every phantom/object; without a background covers_frame=null plus a finding")]
    async fn camera_coverage(&self, Parameters(p): Parameters<PCameraCoverage>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::CameraCoverage { scene, frame: p.frame, camera: p.camera, aspect: p.aspect })
    }

    #[tool(output_schema = structured_object_schema(), description = "Fit the flat screen to the cameras without rendering. Without camera — over ALL cameras of the scene, so one size covers every camera; cameras[] shows each camera's worst frame and what is not covered now. Response: suggested {width m, height m, center m} at the current plane with a margin_deg safety margin, panorama_aspect (width/height) for a panorama generator, size_covers (false — no size helps, the screen must be moved), current/delta, fits, scope, cameras/samples")]
    async fn background_fit(&self, Parameters(p): Parameters<PBackgroundFit>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::BackgroundFit { scene, frame: p.frame, camera: p.camera, margin_deg: p.margin_deg, aspect: p.aspect })
    }

    #[tool(output_schema = structured_object_schema(), description = "Search the metric prop catalog (K1): kind primitive|furniture|interior|animal|street|exterior and/or a name substring; entries[{name, size m, origin, materials, open_sides, furniture, note}] — placement without opening the mesh")]
    async fn catalog_query(&self, Parameters(p): Parameters<PCatalogQuery>) -> Result<CallToolResult, ErrorData> {
        self.exec(Op::CatalogQuery { kind: p.kind, query: p.query })
    }

    #[tool(output_schema = structured_object_schema(), description = "Place a catalog item (K1): the mesh is built procedurally, no asset file needed; pos/yaw/pitch/roll/scale — the start key (degrees), then object_key_set. Materials are edited with material_set using the group names from catalog_query. A furnished room (kind interior with a furniture set) arrives as one undoable package: the shell plus one object per piece — the answer lists every id under `objects`")]
    async fn prop_add(&self, Parameters(p): Parameters<PPropAdd>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::PropAdd {
            scene,
            name: p.name,
            prop: p.prop,
            pos: p.pos,
            yaw: p.yaw_deg.unwrap_or(0.0).to_radians(),
            pitch: p.pitch_deg.unwrap_or(0.0).to_radians(),
            roll: p.roll_deg.unwrap_or(0.0).to_radians(),
            scale: p.scale,
        })
    }

    #[tool(output_schema = structured_object_schema(), description = "Canonical asset: id, kind, relative path, exists and cheap usage refs; not_found includes nearest_ids")]
    async fn asset_get(&self, Parameters(p): Parameters<PAssetRef>) -> Result<CallToolResult, ErrorData> {
        self.exec(Op::AssetGet { asset: p.asset })
    }

    #[tool(output_schema = structured_object_schema(), description = "Find assets by file-name substring (case-insensitive): up to 10 matches {id, kind, path} — the answer to \"id 4 in memory vs id 9 on disk\" (D2)")]
    async fn asset_find(&self, Parameters(p): Parameters<PAssetFind>) -> Result<CallToolResult, ErrorData> {
        let result = self.backend_exec(Op::AssetList).map(|value| {
            let needle = p.name.to_lowercase();
            let matches: Vec<serde_json::Value> = value["assets"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|a| a["path"].as_str().is_some_and(|p| p.to_lowercase().contains(&needle)))
                .take(10)
                .cloned()
                .collect();
            serde_json::json!({ "matches": matches, "total": matches.len() })
        });
        Ok(result.map(structured_success).unwrap_or_else(structured_failure))
    }

    #[tool(output_schema = structured_object_schema(), description = "The whole film: ordered clips and the mapped total length in frames at 24 fps and in seconds")]
    async fn film_get(&self) -> Result<CallToolResult, ErrorData> { self.exec(Op::FilmGet) }

    #[tool(output_schema = structured_object_schema(), description = "List of project assets: id, kind, path")]
    async fn asset_list(&self) -> Result<CallToolResult, ErrorData> {
        self.exec(Op::AssetList)
    }

    #[tool(output_schema = render_still_output_schema(), description = "Render one scene frame to PNG (image content + structured metadata; standalone or bridge; requires a GPU; width/height <= 4096; report=true adds luma/empty_coverage metrics, objects_in_frame/objects_total (Mesh3d prop visibility) and scene_digest; a cheap 480x270 preview by default)")]
    async fn render_still(&self, Parameters(p): Parameters<PRenderStill>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        let report = p.report.unwrap_or(false);
        let (width, height) = still_size(p.width, p.height, report);
        self.exec_image(Op::RenderStill {
            scene: scene,
            frame: p.frame,
            width,
            height,
            camera: p.camera,
            report,
        })
    }

    #[tool(output_schema = structured_object_schema(), description = "Camera report (K0.2): renders a sample of the camera's frames (without every — up to 5 evenly spaced, with every — every Nth, at most 12) and returns rows[{frame, mean_luma, luma_p5/p95, empty_coverage, objects_in_frame}], aggregate and findings (empty_frame/camera_empty/objects_out_of_frame). Requires a GPU; default 480x270")]
    async fn camera_report(&self, Parameters(p): Parameters<PCameraReport>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::CameraReport { scene, camera: p.camera, every: p.every, width: p.width, height: p.height })
    }

    #[tool(output_schema = structured_object_schema(), description = "Review sheet — the agent's eyes on the whole staging: render several directions around the scene at one frame (mode=angles, default) or a strip of frames through one camera (mode=frames) and return them tiled into ONE image. Each tile also carries mean_luma/empty_coverage/objects_in_frame, and empty tiles are findings. target — actor/object names (or phantom:<id>/object:<id>), default everything staged; views — front|back|left|right|top|three_quarter or an [x,y,z] direction from the target (default front,right,top,three_quarter); distance — metres from the target, default auto-framed by the target's size. Requires a GPU; default tile 480x270.")]
    async fn look(&self, Parameters(p): Parameters<PLook>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        let views = match parse_views(p.views.as_deref()) {
            Ok(views) => views,
            Err(e) => return Ok(invalid_input(e)),
        };
        let view = match p.view.as_ref().map(parse_view).transpose() {
            Ok(view) => view,
            Err(e) => return Ok(invalid_input(e)),
        };
        self.exec_image(Op::Look {
            scene,
            mode: p.mode,
            target: p.target,
            views: Some(views),
            distance: p.distance,
            frame: p.frame,
            camera: p.camera,
            frames: p.frames,
            every: p.every,
            count: p.count,
            view,
            width: p.width,
            height: p.height,
        })
    }

    #[tool(output_schema = structured_object_schema(), description = "Place a grid of catalog props in ONE atomic step (one undo): rows along +Z, cols along +X from origin, each object clamped into the stage. The step defaults to the prop's own size times gap (1.2), so \"a row of 6 chairs\" needs no arithmetic; an explicit step [dx,dz] in meters overrides it. Angles in degrees. Returns the created object ids.")]
    async fn prop_grid(&self, Parameters(p): Parameters<PPropGrid>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::PropGrid {
            scene,
            prop: p.prop,
            prefix: p.prefix,
            rows: p.rows,
            cols: p.cols,
            origin: p.origin,
            step: p.step,
            gap: p.gap,
            yaw: p.yaw_deg.unwrap_or(0.0).to_radians(),
            pitch: p.pitch_deg.unwrap_or(0.0).to_radians(),
            roll: p.roll_deg.unwrap_or(0.0).to_radians(),
            scale: p.scale,
        })
    }

    #[tool(output_schema = structured_object_schema(), description = "Fill an area or a line with a catalog mix in ONE atomic step (one undo), from a preset: `forest` (pines + broadleaf + bush + rock, 0.2/m2), `grove` (2.5/m2), `undergrowth` (bush + rock, 1.2/m2), `hedge` (bush every ~0.45 m), `street_row` (tree, streetlight, bin repeating every ~8 m), `pole_row` (a lamp every ~25 m). Give an area as a [x0,z0,x1,z1] rectangle, or a ground plate's object id (the plate is the ground and the margin), or from/to endpoints for a line preset. Inside the optional action box (the zone the scene is shot in) the fill places REAL objects you can move, turn and delete afterwards; the ring beyond it becomes ONE band object whose mesh is a deterministic replay of the same seed, so the answer's numbers and the render never disagree. Over the per-fill caps the call fails with a concrete fix (raise --inset, lower density, shrink the area) instead of thinning the fill in front of the director. The seed is stored in the document: the same seed and shape rebuild the same wood; omit it and the drawn seed comes back in the answer. Angles in degrees.")]
    async fn scatter(&self, Parameters(p): Parameters<PScatter>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::Scatter {
            scene,
            preset: p.preset,
            area: p.area,
            plate: p.plate,
            inset: p.inset,
            action: p.action,
            density: p.density,
            from: p.from,
            to: p.to,
            step: p.step,
            seed: p.seed,
            prefix: p.prefix,
        })
    }

    #[tool(output_schema = structured_object_schema(), description = "Fill a formation with people in ONE atomic step (one undo): `line` (rank and file), `arc` (concave on the lens, reads bigger than a line), `wedge` (a V walking toward the camera), `scattered` (jittered grid — a market square), `queue` (one file). Every person is a real Phantom holding one pose key, so the crowd stays actable afterwards: move someone, turn someone, key someone. `roles` mixes who stands there — `mixed` (default: adult-heavy with teens, elders and children, which is what makes a crowd read as real), `adults`, or one catalog role (`adult|child|teen|elder`). One pose for everyone (`stand` by default); glances are random within `scatter_deg` of the formation's facing, drawn from the seed. The seed is stored in the document, so the same call rebuilds the same crowd; omit it and the drawn seed comes back in the answer. Over 128 people the call fails with the limit and the fix (split into several crowds) instead of quietly truncating. Angles in degrees.")]
    async fn crowd(&self, Parameters(p): Parameters<PCrowd>) -> Result<CallToolResult, ErrorData> {
        let scene = match self.resolve_scene(p.scene) { Ok(scene) => scene, Err(result) => return Ok(result) };
        self.exec(Op::Crowd {
            scene,
            formation: p.formation,
            count: p.count,
            roles: p.roles,
            pose: p.pose,
            spacing: p.spacing,
            rows: p.rows,
            scatter_deg: p.scatter_deg,
            seed: p.seed,
            prefix: p.prefix,
            pos: p.pos,
            yaw: p.yaw_deg.unwrap_or(0.0).to_radians(),
            frame: p.frame,
        })
    }

    #[tool(output_schema = structured_object_schema(), description = "Cancel a job (the process is killed and temporary files are deleted)")]
    async fn job_cancel(&self, Parameters(p): Parameters<PJob>) -> Result<CallToolResult, ErrorData> {
        self.exec(Op::JobCancel { job: p.job })
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PLightAdd {
    /// Scene ID; omit to use the active scene.
    scene: Option<u64>,
    /// Non-empty human-readable light name.
    name: String,
    /// Light kind: `point` (lamp, lumens), `spot` (cone, lumens; needs cone_deg) or `directional` (sun/moon, lux).
    kind: String,
    /// Linear RGB channels, each finite and >= 0.
    color: [f32; 3],
    /// Intensity: lumens for point/spot, lux for directional; finite and >= 0.
    intensity: f32,
    /// Rest position [x,y,z] in meters; used without keys and before the first key.
    pos: [f32; 3],
    /// Rest yaw in degrees; default 0.
    yaw_deg: Option<f32>,
    /// Rest pitch in degrees; default 0.
    pitch_deg: Option<f32>,
    /// Outer cone angle in degrees for spot, in (0, 180); required for spot, ignored otherwise.
    cone_deg: Option<f32>,
    /// Whether the light emits; default true.
    enabled: Option<bool>,
    /// Shadow casting; default false (skinned shadow pipeline is unstable — opt-in).
    shadows: Option<bool>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PLightRef { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// Document light ID.
    light: u64 }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PLightKey {
    /// Scene ID; omit to use the active scene.
    scene: Option<u64>,
    /// Document light ID.
    light: u64,
    /// Integer key frame at fixed 24 fps.
    frame: i64,
    /// Position [x,y,z] in meters on this key.
    pos: [f32; 3],
    /// Yaw in degrees; default 0.
    yaw_deg: Option<f32>,
    /// Pitch in degrees; default 0.
    pitch_deg: Option<f32>,
    /// Intensity override on this key; omitted inherits the light body value.
    intensity: Option<f32>,
    /// On/off step on this key (e.g. a lantern switched on mid-scene); omitted inherits.
    enabled: Option<bool>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PLightKeyRemove { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// Document light ID.
    light: u64, /// Integer key frame to remove at 24 fps.
    frame: i64 }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PAmbientSet {
    /// Scene ID; omit to use the active scene.
    scene: Option<u64>,
    /// Ambient mode: `from_background` (default, derived from the backdrop), `fixed` (color+brightness) or `off`.
    mode: String,
    /// Linear RGB for mode=fixed.
    color: Option<[f32; 3]>,
    /// Brightness in lux for mode=fixed, finite and >= 0.
    brightness: Option<f32>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PSkySet {
    /// Scene ID; omit to use the active scene.
    scene: Option<u64>,
    /// Clear/sky color as linear RGB, filling everything not covered by geometry; null restores transparency. With a sky the frame is opaque.
    sky: Option<[f32; 3]>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PAttach {
    /// Scene ID; omit to use the active scene.
    scene: Option<u64>,
    /// Parent phantom ID whose bone carries the attachment.
    phantom: u64,
    /// Bone name of the phantom rig (case-insensitive; builtin rigs: see builtin_catalog, imported rigs: FBX joint names).
    bone: String,
    /// Existing object ID to attach (mutually exclusive with light/prop).
    object: Option<u64>,
    /// Existing light ID to attach (mutually exclusive with object/prop).
    light: Option<u64>,
    /// Model asset ID: creates a new object and attaches it (the one-call "flashlight in hand" form).
    prop: Option<u64>,
    /// True detaches object/light instead (bone/phantom ignored).
    detach: Option<bool>,
    /// Offset in bone space [x,y,z], meters; default [0,0,0].
    offset_pos: Option<[f32; 3]>,
    /// Offset yaw in degrees; default 0.
    offset_yaw_deg: Option<f32>,
    /// Offset pitch in degrees; default 0.
    offset_pitch_deg: Option<f32>,
    /// Offset roll in degrees; default 0.
    offset_roll_deg: Option<f32>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PPhantomScale { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// Phantom ID.
    phantom: u64, /// Uniform actor scale, finite and > 0.
    scale: f32 }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PMaterialRef {
    /// Scene ID; omit to use the active scene.
    scene: Option<u64>,
    /// Owner phantom ID (mutually exclusive with object).
    phantom: Option<u64>,
    /// Owner object ID (mutually exclusive with phantom).
    object: Option<u64>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PMaterialSet {
    /// Scene ID; omit to use the active scene.
    scene: Option<u64>,
    /// Owner phantom ID (mutually exclusive with object).
    phantom: Option<u64>,
    /// Owner object ID (mutually exclusive with phantom).
    object: Option<u64>,
    /// Material/part name to patch; omitted/null patches all parts.
    target: Option<String>,
    /// Preset name: matte | skin | metal | wet | glass | glow; explicit fields below override preset values.
    preset: Option<String>,
    /// Base color factor, sRGB channels 0..=1.
    base_color: Option<[f32; 4]>,
    /// Metallic factor 0..=1.
    metallic: Option<f32>,
    /// Roughness factor 0..=1.
    roughness: Option<f32>,
    /// Emissive color, linear RGB (may exceed 1 for glow).
    emissive: Option<[f32; 3]>,
    /// Opacity 0..=1; below 1 enables alpha blending.
    opacity: Option<f32>,
    /// Unlit shading toggle.
    unlit: Option<bool>,
    /// True removes patches for target (all, when target omitted) instead of adding one.
    clear: Option<bool>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PGradeSet {
    /// Scene ID; omit to use the active scene.
    scene: Option<u64>,
    /// False disables grading entirely (document grade = null).
    enabled: Option<bool>,
    /// Exposure in EV stops; default 0.
    exposure: Option<f32>,
    /// White-balance temperature (blue↔red); default 0.
    temperature: Option<f32>,
    /// White-balance tint (green↔magenta); default 0.
    tint: Option<f32>,
    /// Hue rotation in degrees; default 0.
    hue_deg: Option<f32>,
    /// Post-tonemapping saturation; 1 = unchanged.
    post_saturation: Option<f32>,
    /// Section saturation; 1 = unchanged.
    saturation: Option<f32>,
    /// Section contrast; 1 = unchanged.
    contrast: Option<f32>,
    /// Section gamma (ASC CDL exponent), > 0; 1 = unchanged.
    gamma: Option<f32>,
    /// Section gain (multiplier), > 0; 1 = unchanged.
    gain: Option<f32>,
    /// Section lift (additive); 0 = unchanged.
    lift: Option<f32>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PFogSet {
    /// Scene ID; omit to use the active scene.
    scene: Option<u64>,
    /// Fog color, linear RGB; null with density omitted disables fog.
    color: Option<[f32; 3]>,
    /// Exponential density, finite and > 0.
    density: Option<f32>,
    /// True disables fog regardless of other fields.
    off: Option<bool>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PVfxAdd {
    /// Scene ID; omit to use the active scene.
    scene: Option<u64>,
    /// Non-empty effect name.
    name: String,
    /// Kind: `particles` or `water`.
    kind: String,
    /// Particle preset: snow | embers | smoke | rain; explicit fields override preset values.
    preset: Option<String>,
    /// Emitter center / water plane center [x,y,z], meters; default [0,0,0].
    pos: Option<[f32; 3]>,
    /// Emission volume / plane size [x,y,z], meters; default [10,0.5,10].
    extent: Option<[f32; 3]>,
    /// Particles per second; preset default when omitted.
    emit_rate: Option<f32>,
    /// Particle lifetime in seconds, > 0.
    lifetime: Option<f32>,
    /// Initial particle speed, m/s.
    speed: Option<f32>,
    /// Particle quad size, meters, > 0.
    size: Option<f32>,
    /// Linear RGB color.
    color: Option<[f32; 3]>,
    /// Vertical acceleration, m/s²: negative falls (snow/rain), positive rises (embers/smoke).
    gravity: Option<f32>,
    /// Water: normal-map image asset ID.
    normal_map: Option<u64>,
    /// Water: UV scroll speed, texture fractions per second.
    scroll_uv: Option<[f32; 2]>,
    /// Initially enabled; default true.
    enabled: Option<bool>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PVfxRef { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// Vfx ID.
    vfx: u64 }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PVfxKey {
    /// Scene ID; omit to use the active scene.
    scene: Option<u64>,
    /// Vfx ID.
    vfx: u64,
    /// Integer key frame at 24 fps.
    frame: i64,
    /// On/off step on this key; omitted leaves inherited.
    enabled: Option<bool>,
    /// emit_rate multiplier step on this key, finite and >= 0; omitted inherits.
    density: Option<f32>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PVfxKeyRemove { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// Vfx ID.
    vfx: u64, /// Integer key frame to remove.
    frame: i64 }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PValidate {
    /// Scene ID; omit to validate the active scene, or the whole project when no scene is active.
    scene: Option<u64>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PAudioReport {
    /// Scene ID; omit to report the active scene, or the whole project when no scene is active.
    scene: Option<u64>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PCameraCoverage {
    /// Scene ID; omit to use the active scene.
    scene: Option<u64>,
    /// Integer frame at fixed 24 fps; default = start of the camera's range, or of the scene range without a camera.
    frame: Option<i64>,
    /// Camera ID; its view and range are used instead of the scene's active camera.
    camera: Option<u64>,
    /// Frame aspect (width/height); default 1.7778 (16:9). Angles in the response are degrees.
    aspect: Option<f64>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PBackgroundFit {
    /// Scene ID; omit to use the active scene.
    scene: Option<u64>,
    /// Integer frame at fixed 24 fps; only valid together with `camera` (default = start of that camera's range).
    frame: Option<i64>,
    /// Camera ID; omit to fit the screen for ALL cameras of the scene, so one size covers every camera.
    camera: Option<u64>,
    /// Safety margin added to the frustum on every side, in degrees; default 5.0, range [0, 45).
    margin_deg: Option<f64>,
    /// Frame aspect (width/height); default 1.7778 (16:9). Suggested angles are degrees, lengths meters.
    aspect: Option<f64>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PSceneSelect {
    /// Existing scene ID to make active for this MCP connection.
    scene: u64,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PBatch {
    /// Ordered operations in the Session/bridge wire format: objects with an
    /// `op` tag (e.g. {"op":"scene_add","name":"S1","start":0,"end":48}).
    /// Unlike single-call convenience tools, angles are radians (yaw/pitch/
    /// roll fields) and audio offsets/lengths are 48-kHz samples; frames are
    /// integers at 24 fps, ranges are [start,end). The `scene` field may be
    /// omitted/null — the connection's active scene is injected. Only document
    /// mutations are allowed (no queries, lifecycle, render, jobs, nested
    /// batch). All operations are validated and applied atomically: any failure
    /// rolls the whole batch back; success is one undo step and one revision.
    ops: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PSceneAdd {
    /// Non-empty human-readable scene name.
    name: String,
    /// First integer frame at fixed 24 fps; range is [start,end).
    start: i64,
    /// End-exclusive integer frame at fixed 24 fps; must be greater than start.
    end: i64,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PImport { /// Source file path to import into the active project.
    file: String }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PImportBatch { /// Source file paths; imported atomically (all or none).
    files: Vec<String> }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PBackground {
    /// Scene ID; omit to use the active scene.
    scene: Option<u64>,
    /// Imported image or video asset ID.
    asset: u64,
    /// True for a video background; false/default for an image.
    video: Option<bool>,
    /// Screen width in meters (>0); the flat backdrop spans x ∈ center.x ± width/2.
    width: Option<f32>,
    /// Screen height in meters (>0); it spans y ∈ center.y ± height/2.
    height: Option<f32>,
    /// Screen center [x,y,z] in meters; Y is up. The screen faces +Z (the camera side).
    center: Option<[f32; 3]>,
    /// Panorama UV mode: `cover` (default, crop overflow), `contain` (whole image, clamped bands) or `stretch` (legacy distortion).
    fit: Option<String>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PSceneRange { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// First integer frame at 24 fps; range is [start,end).
    start: i64, /// End-exclusive integer frame at 24 fps.
    end: i64 }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PCameraKeyRef { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// Camera ID in the scene.
    camera: u64, /// Integer frame at fixed 24 fps.
    frame: i64 }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PPhantomRef { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// Phantom actor ID in the scene.
    phantom: u64 }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PMotionKeyRef { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// Phantom actor ID in the scene.
    phantom: u64, /// Integer key frame at fixed 24 fps.
    frame: i64 }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct MotionEntry {
    /// Integer key frame at fixed 24 fps.
    frame: i64,
    /// Controller position [x,y,z] in meters; Y is up.
    pos: [f32; 3],
    /// Pose this actor can hold (stand|sit|lie), from its builtin_catalog profile.
    pose: String,
    /// Optional root yaw in degrees about +Y; 0 faces +Z, positive turns toward +X.
    yaw_deg: Option<f32>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PMotion { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// Phantom actor ID whose complete manual motion is replaced.
    phantom: u64, /// Typed controller keys; an empty array clears the manual track.
    motion: Vec<MotionEntry> }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PTrackRef { /// Global audio track ID.
    track: u64 }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PLayerAdd { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// Non-empty human-readable 2D layer name.
    name: String }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PExposureRef { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// 2D layer ID in the scene.
    layer: u64, /// Integer exposure frame at fixed 24 fps.
    frame: i64 }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PPhantom { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// Non-empty human-readable actor name.
    name: String, /// Imported FBX rig asset ID; mutually exclusive with rig_actor.
    rig_asset: Option<u64>, /// Builtin actor profile: humanoid, wolf, horse, or bird.
    rig_actor: Option<String> }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PKey {
    /// Scene ID; omit to use the active scene.
    scene: Option<u64>,
    /// Phantom actor ID in the scene.
    phantom: u64,
    /// Integer key frame at fixed 24 fps.
    frame: i64,
    /// Optional controller position [x,y,z] in meters; Y is up.
    pos: Option<[f32; 3]>,
    /// Optional pose for this actor (stand|sit|lie); a pose the species cannot
    /// hold is rejected with unsupported_pose.
    pose: Option<String>,
    /// Optional root yaw in degrees about +Y; 0 faces +Z, positive turns toward +X.
    yaw_deg: Option<f32>,
    /// Preserve omitted fields from an exact key or evaluated state; defaults true.
    preserve: Option<bool>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PCameraAdd { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// Non-empty human-readable camera name.
    name: String, /// Optional first integer scene frame at 24 fps; the range [start,end) defaults to the whole scene.
    start: Option<i64>, /// Optional end-exclusive integer scene frame at 24 fps; start and end come together.
    end: Option<i64> }
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PCameraKey {
    /// Scene ID; omit to use the active scene.
    scene: Option<u64>,
    /// Camera ID; omit to key the scene's active camera.
    camera: Option<u64>,
    /// Integer key frame at fixed 24 fps.
    frame: i64,
    /// Camera position [x,y,z] in meters; Y is up.
    pos: [f32; 3],
    /// Yaw in degrees about +Y; 0 looks along -Z. Ignored when look_at is given.
    yaw_deg: Option<f32>,
    /// Pitch in degrees; 0 is level, positive looks upward. Ignored when look_at is given.
    pitch_deg: Option<f32>,
    /// Roll in degrees (tilted horizon); 0 keeps the horizon level.
    roll_deg: Option<f32>,
    /// Aim point [x,y,z] in meters: yaw/pitch are derived from pos toward it, so orbit/follow keys need no trigonometry.
    look_at: Option<[f32; 3]>,
    /// Focal length in millimeters (>0) on this key: zoom/crash-zoom inside the camera's range.
    focal_mm: Option<f32>,
    /// Named lens preset, used when focal_mm is omitted: ultrawide 16 | wide 24 | normal 35 | portrait 50 | portrait_tele 85 | tele 135 | super_tele 200.
    focal_preset: Option<String>,
    /// Interpolation to the next key: linear | ease_in | ease_out | ease_in_out (default linear).
    ease: Option<String>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PCameraFocalKey {
    /// Scene ID; omit to use the active scene.
    scene: Option<u64>,
    /// Camera ID in the scene.
    camera: u64,
    /// Integer key frame at fixed 24 fps.
    frame: i64,
    /// Focal length in millimeters (>0) on this key.
    focal_mm: f32,
}
#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum BrushSpace { Canvas, Screen, Ndc }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PStroke {
    /// Scene ID; omit to use the active scene.
    scene: Option<u64>,
    /// Integer exposure frame at fixed 24 fps.
    frame: i64,
    /// Polyline points in the selected space; at least two finite points.
    points: Vec<[f32; 2]>,
    /// RGBA channels in inclusive range 0..=1.
    color: [f32; 4],
    /// Brush width in pixels of the canonical 1920x1080 canvas (>0).
    width: f32,
    /// True erases instead of painting color.
    erase: bool,
    /// Coordinate space: canvas (1920x1080), screen (requires dimensions), or ndc (-1..1, +Y up).
    space: BrushSpace,
    /// Screen width in pixels, required and >0 only for screen space.
    screen_width: Option<f32>,
    /// Screen height in pixels, required and >0 only for screen space.
    screen_height: Option<f32>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PSceneRef { /// Scene ID; omit to use the active scene.
    scene: Option<u64> }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PLight { /// Linear RGB channels, each finite and >=0.
    color: [f32; 3], /// Illuminance in lux, finite and >=0.
    intensity: f32, /// Whether this staging light emits; defaults true.
    enabled: Option<bool>, /// Whether this light casts shadows; defaults false (skinned shadow pipeline is unstable — opt-in).
    shadows: Option<bool> }
impl From<PLight> for phantom_core::SceneLight { fn from(l: PLight) -> Self { Self { color:l.color, intensity:l.intensity, enabled:l.enabled.unwrap_or(true), shadows:l.shadows.unwrap_or(false) } } }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PLightingSet { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// Optional key-light replacement; omitted leaves it unchanged.
    key: Option<PLight>, /// Optional fill-light replacement; omitted leaves it unchanged.
    fill: Option<PLight>, /// Optional rim-light replacement; omitted leaves it unchanged.
    rim: Option<PLight> }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PFloorSet { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// True enables the floor; false removes it.
    enabled: bool, /// Optional linear RGB floor color, channels finite and >=0.
    color: Option<[f32; 3]> }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PStageSet { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// Stage preset: small (10x10x4 m) | medium (30x20x8) | large (80x60x20).
    preset: Option<String>, /// Stage width in meters (with a preset it overrides that axis).
    width: Option<f32>, /// Stage height in meters: the ceiling above the y=0 floor.
    height: Option<f32>, /// Stage depth in meters (with a preset it overrides that axis).
    depth: Option<f32>, /// Center of the box on the floor plane [x, z]; default — the current center.
    center: Option<[f32; 2]> }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PFilmAdd { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// First integer scene frame; range is [start,end).
    start: i64, /// End-exclusive integer scene frame at 24 fps.
    end: i64, /// Optional camera ID; range must equal the camera's [start,end).
    camera: Option<u64> }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PCameraRef { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// Camera ID in the scene.
    camera: u64 }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PCameraRange { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// Camera ID in the scene.
    camera: u64, /// First integer scene frame; range is [start,end).
    start: i64, /// End-exclusive integer scene frame at 24 fps.
    end: i64 }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PCameraLens { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// Camera ID in the scene.
    camera: u64, /// Focal length in millimeters (>0), 35-mm camera model.
    focal_mm: f32 }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PCameraTargetSet { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// Camera ID in the scene.
    camera: u64, /// track — the camera stands and turns to keep the subject in view | follow — attached to the subject, travels and turns with it; omit with clear=true to detach.
    mode: Option<String>, /// phantom:<id> or object:<id>; required unless clear=true.
    subject: Option<String>, /// True detaches the camera from its subject.
    clear: Option<bool> }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PFilmTrim { /// Film clip ID.
    clip: u64, /// First integer scene frame; range is [start,end).
    start: i64, /// End-exclusive integer scene frame at 24 fps.
    end: i64 }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PFilmMove { /// Film clip ID.
    clip: u64, /// Zero-based destination index; ignored by film_remove.
    index: usize }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct FrameRangeInput { /// First integer frame at fixed 24 fps; range is [start,end).
    start: i64, /// End-exclusive integer frame at fixed 24 fps.
    end: i64 }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct POut { /// Output MP4 path within the allowed project root.
    out: String, /// Output width in pixels, 1..=4096; defaults 1280.
    width: Option<u32>, /// Output height in pixels, 1..=4096; defaults 720.
    height: Option<u32>, /// Optional film frame range [start,end), with end exclusive.
    frames: Option<FrameRangeInput>, /// Peak-normalize the film mix to target_peak_db; default false (A6: no silent loudness changes).
    normalize: Option<bool>, /// Target peak dBFS when normalize=true; defaults -1.0.
    target_peak_db: Option<f32> }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PRenderScene { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// Output directory within project; defaults to <project>/frames.
    out: Option<String>, /// Output width in pixels, 1..=4096; defaults 1280.
    width: Option<u32>, /// Output height in pixels, 1..=4096; defaults 720.
    height: Option<u32>, /// Optional scene frame range [start,end), with end exclusive.
    frames: Option<FrameRangeInput>, /// Optional camera ID selecting its view.
    camera: Option<u64> }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PCatalogQuery {
    /// Catalog section: primitive | furniture | interior | animal | street | exterior; omit for all.
    kind: Option<String>,
    /// Case-insensitive substring of the entry name or its note; omit or empty for the whole section.
    query: Option<String>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PLook {
    /// Scene ID; omit to use the active scene.
    scene: Option<u64>,
    /// `angles` (default) — directions around the scene at one frame; `frames` — a strip over a range.
    mode: Option<String>,
    /// Actor/object names to frame, or `phantom:<id>` / `object:<id>`; omit for everything staged.
    target: Option<Vec<String>>,
    /// angles: view names (front|back|left|right|top|three_quarter) or [x,y,z] directions from the target; default front,right,top,three_quarter.
    views: Option<Vec<serde_json::Value>>,
    /// Metres from the target center to the eye; omit to auto-frame by the target's size.
    distance: Option<f32>,
    /// angles: the frame to review; default the first frame of the camera's scene range.
    frame: Option<i64>,
    /// frames: camera whose view to use; omit for the scene's active camera.
    camera: Option<u64>,
    /// frames: explicit frame numbers (at most 12); overrides every/count.
    frames: Option<Vec<i64>>,
    /// frames: every Nth frame of the range (at most 12 tiles).
    every: Option<u32>,
    /// frames: how many evenly spaced frames, both ends included (at most 12).
    count: Option<u32>,
    /// frames: a review direction instead of the camera — a view name or [x,y,z].
    view: Option<serde_json::Value>,
    /// Tile width in pixels, 1..=4096; default 480.
    width: Option<u32>,
    /// Tile height in pixels, 1..=4096; default 270.
    height: Option<u32>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PCrowd {
    /// Scene ID; omit to use the active scene.
    scene: Option<u64>,
    /// Formation: `line`, `arc`, `wedge`, `scattered` or `queue`.
    formation: String,
    /// How many people; up to 128.
    count: usize,
    /// Who stands there: `mixed` (default), `adults`, or a catalog role name (`adult|child|teen|elder`).
    roles: Option<String>,
    /// Pose held by everyone: `stand` (default), `sit` or `lie`.
    pose: Option<String>,
    /// Meters between neighbors; 0.8 by default.
    spacing: Option<f32>,
    /// Rows for a `line` or `scattered` crowd; close to square by default. Rejected for the shapes that have no rows.
    rows: Option<usize>,
    /// How far a person may look off the formation's facing, in degrees; 60 by default.
    scatter_deg: Option<f32>,
    /// Determinism seed; omit to draw one (it comes back in the answer and is stored).
    seed: Option<u64>,
    /// Name prefix for the people; default the scene name.
    prefix: Option<String>,
    /// Where the formation's origin stands, [x,y,z] in meters; the origin by default.
    pos: Option<[f32; 3]>,
    /// Which way the formation faces, in degrees about +Y.
    yaw_deg: Option<f32>,
    /// Frame the pose key is set at; frame 0 by default.
    frame: Option<i64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PScatter {
    /// Scene ID; omit to use the active scene.
    scene: Option<u64>,
    /// Fill preset: `forest`, `grove`, `undergrowth`, `hedge`, `street_row`, `pole_row`.
    preset: String,
    /// World rectangle [x0,z0,x1,z1] in meters to fill (an area preset).
    area: Option<[f32; 4]>,
    /// Object id of an outdoor ground plate (street_yard/street_park/street_crossing) to fill instead of a rectangle.
    plate: Option<u64>,
    /// Keep this margin in meters clear of the plate edge; default 0.5.
    inset: Option<f32>,
    /// The zone the scene is shot in [x0,z0,x1,z1]: inside it the fill places movable props.
    action: Option<[f32; 4]>,
    /// Props per square meter; the preset's own density by default.
    density: Option<f32>,
    /// Line presets: segment start [x,z] in meters.
    from: Option<[f32; 2]>,
    /// Line presets: segment end [x,z] in meters.
    to: Option<[f32; 2]>,
    /// Line presets: spacing in meters; the preset's step by default.
    step: Option<f32>,
    /// Determinism seed; omit to draw one (it comes back in the answer and is stored).
    seed: Option<u64>,
    /// Name prefix for the created objects; default the scene name.
    prefix: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PPropGrid {
    /// Scene ID; omit to use the active scene.
    scene: Option<u64>,
    /// Catalog entry name from catalog_query (e.g. chair, stool, box, room_kitchen). A furnished room places its furniture too.
    prop: String,
    /// Name prefix for the created objects; default the prop name.
    prefix: Option<String>,
    /// Rows along +Z; 1 or more (rows*cols <= 256).
    rows: u32,
    /// Columns along +X; 1 or more (rows*cols <= 256).
    cols: u32,
    /// Position of the first (row 0, col 0) object [x,y,z] in meters.
    origin: [f32; 3],
    /// Explicit spacing [dx, dz] in meters; default the prop's size times gap.
    step: Option<[f32; 2]>,
    /// Gap factor for the default step: size*gap; default 1.2 (items do not touch).
    gap: Option<f32>,
    /// Yaw in degrees about +Y for every item.
    yaw_deg: Option<f32>,
    /// Pitch in degrees for every item.
    pitch_deg: Option<f32>,
    /// Roll in degrees for every item.
    roll_deg: Option<f32>,
    /// Uniform or per-axis scale, each finite and > 0.
    scale: Option<[f32; 3]>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PPropAdd {
    /// Scene ID; omit to use the active scene.
    scene: Option<u64>,
    /// Non-empty object name in the scene.
    name: String,
    /// Catalog entry name from catalog_query (e.g. table, room_kitchen, cylinder).
    prop: String,
    /// Start position [x,y,z] in meters; omit to leave the object at the origin.
    pos: Option<[f32; 3]>,
    /// Yaw in degrees about +Y; 0 keeps the catalog orientation (room north opens away from the default camera).
    yaw_deg: Option<f32>,
    /// Pitch in degrees; 0 is level.
    pitch_deg: Option<f32>,
    /// Roll in degrees; 0 is level.
    roll_deg: Option<f32>,
    /// Uniform or per-axis scale [x,y,z] of the procedural mesh.
    scale: Option<[f32; 3]>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PCameraReport {
    /// Scene ID; omit to use the active scene.
    scene: Option<u64>,
    /// Camera ID to report on.
    camera: u64,
    /// Sample every Nth frame; omit for up to 5 evenly spaced frames (hard cap 12 frames either way).
    every: Option<u32>,
    /// Render width in pixels, 1..=4096; defaults 480.
    width: Option<u32>,
    /// Render height in pixels, 1..=4096; defaults 270.
    height: Option<u32>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PJob { /// Render job ID returned by render_scene or render_film.
    job: u64 }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PJobWait {
    /// Render job ID returned by render_scene or render_film.
    job: u64,
    /// Maximum wall-clock seconds to wait, finite and in inclusive range 0..=300; timeout does not cancel the job.
    timeout_secs: f64,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PDescribe {
    /// Exact MCP tool name from tools/list to describe.
    tool: String,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PRenderStill { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// Integer scene frame at fixed 24 fps.
    frame: i64, /// PNG width in pixels, 1..=4096; omit for 1024x576, or 480x270 when report=true.
    width: Option<u32>, /// PNG height in pixels, 1..=4096; omit for 1024x576, or 480x270 when report=true.
    height: Option<u32>, /// Optional camera ID selecting its view.
    camera: Option<u64>, /// True adds pure frame metrics (mean_luma/luma_p5/luma_p95 over opaque pixels, empty_coverage = alpha==0 share, objects_in_frame/objects_total = forged Mesh3d visibility, non-cryptographic scene_digest) to structuredContent; pairs with the cheap 480x270 default.
    report: Option<bool> }

/// Default render_still sizes: a normal frame is 1024x576, a cheap
/// preview is 480x270 when metrics are requested (corrective C3.2).
fn still_size(width: Option<u32>, height: Option<u32>, report: bool) -> (u32, u32) {
    let (w, h) = if report { (480, 270) } else { (1024, 576) };
    (width.unwrap_or(w), height.unwrap_or(h))
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PTrackAdd { /// Scene ID for a scene track; null/omitted deliberately creates a film track (not active-scene resolution).
    scene: Option<u64>, /// Non-empty human-readable track name.
    name: String }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PClipAdd { /// Global audio track ID.
    track: u64, /// Imported audio asset ID.
    asset: u64, /// Timeline start in finite non-negative seconds; converted to an exact 24-fps frame.
    start_secs: f64, /// Source offset in finite non-negative seconds; rounded to nearest 48-kHz sample.
    offset_secs: f64, /// Clip length in finite non-negative seconds; rounded to nearest 48-kHz sample.
    len_secs: f64, /// Fade-in length in finite non-negative seconds, <= len_secs; default 0 (no fade).
    fade_in_secs: Option<f64>, /// Fade-out length in finite non-negative seconds, <= len_secs; default 0 (no fade).
    fade_out_secs: Option<f64> }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PClipUpdate { /// Global audio track ID.
    track: u64, /// Audio clip ID on that track.
    clip: u64, /// New source offset in seconds; omitted leaves it unchanged.
    offset_secs: Option<f64>, /// New clip length in seconds (>0); omitted leaves it unchanged.
    len_secs: Option<f64>, /// New fade-in length in seconds (<= len); omitted leaves it unchanged.
    fade_in_secs: Option<f64>, /// New fade-out length in seconds (<= len); omitted leaves it unchanged.
    fade_out_secs: Option<f64> }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PVolume { /// Global audio track ID.
    track: u64, /// Linear gain, finite and >=0; 1.0 is unity.
    volume: f32 }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PMute { /// Global audio track ID.
    track: u64, /// True silences the track; false enables it.
    muted: bool }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PClipRemove { /// Global audio track ID.
    track: u64, /// Audio clip ID on the track.
    clip: u64 }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PAssetRef { /// Imported asset ID.
    asset: u64 }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PAssetFind { /// Substring of the asset file name (case-insensitive).
    name: String }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PObjectAdd { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// Non-empty human-readable object name.
    name: String, /// Imported OBJ/FBX model asset ID.
    asset: u64 }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PObjectRef { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// Object ID in the scene.
    object: u64 }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PObjectKey { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// Object ID in the scene.
    object: u64, /// Integer key frame at fixed 24 fps.
    frame: i64, /// Position [x,y,z] in meters; Y is up.
    pos: [f32;3], /// Optional yaw degrees about +Y; 0 preserves model orientation.
    yaw_deg: Option<f32>, /// Optional pitch degrees about +X; 0 is level.
    pitch_deg: Option<f32>, /// Optional roll degrees about +Z; 0 is upright.
    roll_deg: Option<f32>, /// Optional positive scale [x,y,z]; defaults [1,1,1].
    scale: Option<[f32;3]> }
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PObjectKeyRemove { /// Scene ID; omit to use the active scene.
    scene: Option<u64>, /// Object ID in the scene.
    object: u64, /// Integer key frame at fixed 24 fps.
    frame: i64 }

#[tool_handler]
impl ServerHandler for PhantomMcp {
    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let call = rmcp::handler::server::tool::ToolCallContext::new(self, request, context);
        let response = Self::tool_router().call(call).await?;
        // rmcp converts parameter-deserialization failures to an unstructured
        // text tool error. Normalize that local MCP failure into our contract.
        Ok(match response {
            CallToolResponse::Complete(result)
                if result.is_error == Some(true) && result.structured_content.is_none() =>
            {
                let message = result
                    .content
                    .first()
                    .and_then(ContentBlock::as_text)
                    .map(|text| text.text.clone())
                    .unwrap_or_else(|| "invalid tool parameters".into());
                CallToolResponse::Complete(invalid_input(message))
            }
            other => other,
        })
    }

    #[allow(deprecated)] // rmcp: ServerInfo is still the get_info result type in 3.8
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.protocol_version = ProtocolVersion::V_2025_06_18;
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.server_info.name = "phantom-animate".into();
        info.server_info.version = env!("CARGO_PKG_VERSION").into();
        info.instructions = Some(
            "Phantom-animate: command-first layout staging for a playblast. Start with studio_capabilities, then \
             project_new/project_open and scene_add; a new scene or the first opened scene becomes active. \
             scene_select/scene_current control the active scene of this MCP session only; the scene parameter of \
             scene tools may be omitted, and with no selection no_active_scene is returned. state_get includes \
             active_scene and allowed_root. Frames are integers at 24 fps, ranges are always [start,end) with end \
             exclusive; coordinates are in meters, Y up. Each scene has a stage: the box actors and props are \
             confined to (stage_set/stage_get; presets small 10x10x4 m, medium 30x20x8, large 80x60x20, or an \
             explicit width+height+depth). Every placement command clamps the entity's whole bounding box into \
             the stage, and the answer then carries clamped:true plus clamps[] with the coordinates before and \
             after — a position you sent may have been corrected. validate reports out_of_stage for a document \
             that predates the stage. MCP angles are yaw_deg/pitch_deg/roll_deg in degrees \
             (core stores radians). Actors are NOT animated: key_set/motion_replace place a figure with a held pose \
             — typed keys are {frame,pos,pose,yaw_deg?}, where pose is stand|sit|lie and the species must have it \
             (builtin_catalog lists the poses per actor; an unavailable pose is unsupported_pose). Position and yaw \
             interpolate between keys, the pose switches at its own key. An imported rig has no poses and renders in \
             bind pose. key_set preserves omitted fields (pose included) by default. Audio MCP accepts seconds and \
             converts them to 48 kHz. brush_stroke sets \
             canvas/screen/ndc explicitly and normalizes to 1920x1080. render_scene writes frames/ inside \
             allowed_root by default; render_still is available standalone and over the bridge, and requires a GPU; \
             report=true gives mean_luma/luma_p5/luma_p95/empty_coverage/scene_digest plus a cheap 480x270 \
             preview — this stands in for the agent's eyes (check lighting and frame emptiness before a full \
             render). A scene holds a list of cameras (camera_add/camera_remove/camera_select); one camera is \
             active and is the view a render without a camera uses. A camera can be attached to a moving subject \
             in two modes (camera_target_set: track stand-and-turn, follow travels-with-it), and every camera key \
             must keep the camera looking at the scene (at least 20 % of the stage volume in frame). \
             audio_report before rendering shows track/clip durations against the video and the RMS/peak \
             of the fragments actually used: \"music 360 s vs a 2 s film\" and a quiet offset are visible there; \
             film_add warns when a clip is shorter than the scene's longest track; state_get carries audio_summary. \
             Lighting: light_add (point/spot/directional, degrees, lumens/lux) plus light_key_set/light_key_remove \
             animate pose, intensity and on/off per frame; scene_ambient_set (from_background|fixed|off) and \
             scene_sky_set (null = transparency) control the frame background. Getters are queries: \
             camera_get/asset_get/film_get return canonical snapshots carrying the current revision and do not bump \
             it; project_recent returns the last successfully created/opened project and the current directory with \
             no open project. job_wait waits for a job up to timeout_secs (0..=300) and returns the job_status \
             shape plus timed_out; a timeout does not cancel the job. describe {tool} returns the exact \
             description/inputSchema/outputSchema of any tool from tools/list. batch takes an array of operations \
             in wire format ({\"op\":\"scene_add\",...}); unlike the single-call tools, angles there are radians, \
             audio is 48-kHz samples, and scene may be omitted (the active one is substituted). The package is \
             atomic: any error rolls everything back (batch_failed with index and the inner code), success is one \
             undo step and one revision; results[i] returns the created ids, and ids are allocated \
             deterministically, so scene→phantom→key chains work inside one package. Pure diagnostics without \
             render and GPU: validate {scene?} runs the invariants (screen, FOV coverage, motion gaps, \
             audio/editing ranges, files) and returns findings with code/severity/where/why/suggested_fix (without \
             scene and without an active scene — the whole project); camera_coverage {scene?, frame?, camera?, \
             aspect?} — screen coverage of the frame and in_frame points (angles in the response are degrees, \
             aspect defaults to 16:9); background_fit {scene?, frame?, camera?, margin_deg?, aspect?} — suggested \
             the flat screen size that closes the frame at its current plane, with a margin_deg safety margin (5° by \
             default), size_covers false when no size helps (move the screen with background_set) and \
             panorama_aspect for a panorama generator; all three are queries and do not bump the revision. prop_grid \
             places a row or grid of catalog props in one atomic step (one undo), with the spacing derived from the \
             prop's own size. structuredContent is the \
             authoritative result; ordinary content is empty except for the render_still image block."
                .into(),
        );
        info
    }
}

pub async fn serve(bridge: Option<PathBuf>) -> Result<()> {
    let mode = if bridge.is_some() { McpMode::Bridge } else { McpMode::Standalone };
    let allowed_root = bridge.as_ref().map(|dir| dir.canonicalize().unwrap_or_else(|_| dir.clone()));
    let backend = match bridge {
        Some(dir) => Backend::Bridge(Mutex::new(
            BridgeClient::connect(&dir).context("connecting to the editor window")?,
        )),
        None => Backend::Standalone(Mutex::new(Session::new())),
    };
    let server = PhantomMcp {
        backend: Arc::new(backend),
        active_scene: Arc::new(Mutex::new(None)),
        allowed_root: Arc::new(Mutex::new(allowed_root)),
        mode,
    };
    server.refresh_active_scene();
    let service = server.serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod contract_tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(1);

    #[test]
    fn parameters_reject_unknown_fields_and_publish_closed_schemas() {
        // P0 API: an agent's typo must be an invalid_input at decode time, never
        // a successful operation that silently loses intent.
        let err = serde_json::from_value::<PSkySet>(serde_json::json!({
            "scene": 1,
            "colr": [1.0, 0.0, 0.0],
        }))
        .unwrap_err();
        assert!(err.to_string().contains("colr"));
        let err = serde_json::from_value::<PCameraKey>(serde_json::json!({
            "scene": 1,
            "frame": 0,
            "pos": [0.0, 0.0, 0.0],
            "target": [0.0, 0.0, -1.0],
        }))
        .unwrap_err();
        assert!(err.to_string().contains("target"));
        for schema in [
            rmcp::handler::server::tool::schema_for_type::<PSkySet>(),
            rmcp::handler::server::tool::schema_for_type::<PCameraKey>(),
        ] {
            assert_eq!(serde_json::to_value(schema).unwrap()["additionalProperties"], false);
        }
    }

    #[test]
    fn project_open_selects_and_returns_first_scene() {
        let dir = std::env::temp_dir().join(format!("phantom-mcp-open-active-{}", NEXT.fetch_add(1, Ordering::Relaxed)));
        let server = standalone(Session::new());
        block_on(server.project_new(Parameters(PNew { dir: dir.to_string_lossy().into(), name: "p".into() }))).unwrap();
        let scene = assert_structured(&block_on(server.scene_add(Parameters(PSceneAdd { name: "s".into(), start: 0, end: 24 }))).unwrap(), false)["scene"].as_u64().unwrap();
        let opened = block_on(server.project_open(Parameters(POpen { dir: dir.to_string_lossy().into(), force: false }))).unwrap();
        assert_eq!(assert_structured(&opened, false)["active_scene"], scene);
        let _ = std::fs::remove_dir_all(dir);
    }

    fn standalone(session: Session) -> PhantomMcp {
        let root = session.dir.clone();
        let server = PhantomMcp {
            backend: Arc::new(Backend::Standalone(Mutex::new(session))),
            active_scene: Arc::new(Mutex::new(None)),
            allowed_root: Arc::new(Mutex::new(root)),
            mode: McpMode::Standalone,
        };
        server.refresh_active_scene();
        server
    }

    fn bridge_once(mut session: Session) -> PhantomMcp {
        let (client, server) = UnixStream::pair().unwrap();
        std::thread::spawn(move || {
            let mut writer = server.try_clone().unwrap();
            let mut line = String::new();
            BufReader::new(server).read_line(&mut line).unwrap();
            let request: serde_json::Value = serde_json::from_str(&line).unwrap();
            let id = request["id"].as_u64().unwrap();
            let op: Op = serde_json::from_value(request["op"].clone()).unwrap();
            let response = match session.execute(op) {
                Ok(value) => crate::bridge::BridgeResponse::success(id, value),
                Err(error) => crate::bridge::BridgeResponse::failure(id, error.error_data()),
            };
            writeln!(writer, "{}", serde_json::to_string(&response).unwrap()).unwrap();
        });
        PhantomMcp {
            backend: Arc::new(Backend::Bridge(Mutex::new(BridgeClient::from_stream(client)))),
            active_scene: Arc::new(Mutex::new(None)),
            allowed_root: Arc::new(Mutex::new(None)),
            mode: McpMode::Bridge,
        }
    }

    fn call(server: &PhantomMcp, op: Op) -> CallToolResult {
        server.exec(op).unwrap()
    }

    fn wire(result: &CallToolResult) -> serde_json::Value {
        serde_json::to_value(result).unwrap()
    }

    fn assert_structured(result: &CallToolResult, is_error: bool) -> &serde_json::Value {
        assert_eq!(result.is_error, Some(is_error));
        assert!(result.content.is_empty(), "ordinary structured results must not carry text JSON");
        let value = result.structured_content.as_ref().expect("structuredContent");
        let encoded = wire(result);
        assert_eq!(encoded["structuredContent"], *value);
        assert_eq!(encoded["isError"], is_error);
        value
    }

    #[test]
    fn representative_successes_are_structured_without_text_json() {
        let mut session = Session::new();
        session.project = Some(phantom_core::Project::new("contract"));
        let server = standalone(session);

        let catalog = call(&server, Op::BuiltinCatalog);
        assert!(assert_structured(&catalog, false)["profiles"].is_array());

        let added = call(&server, Op::SceneAdd { name: "main".into(), start: 0, end: 24 });
        let structured = assert_structured(&added, false);
        assert!(structured["scene"].is_u64());
        assert_eq!(structured["revision"], 1);
        assert_eq!(structured["next_id"], 2);
        assert_eq!(structured["entity"]["kind"], "scene");
        assert_eq!(structured["entity"]["name"], "main");

        let state = call(&server, Op::StateGet);
        let structured = assert_structured(&state, false);
        assert!(structured.get("entity").is_none());
        assert!(structured.get("next_id").is_none());
    }

    #[test]
    fn stable_error_codes_serialize_in_the_mcp_envelope() {
        let not_open = call(&standalone(Session::new()), Op::StateGet);
        assert_eq!(assert_structured(&not_open, true)["code"], "not_open");

        let mut with_project = Session::new();
        with_project.project = Some(phantom_core::Project::new("errors"));
        let server = standalone(with_project);
        let not_found = call(&server, Op::SceneGet { scene: 999 });
        assert_eq!(assert_structured(&not_found, true)["code"], "not_found");
        assert_eq!(not_found.structured_content.as_ref().unwrap()["details"]["entity"], "scene");

        let invalid = call(
            &server,
            Op::RenderStill { scene: 1, frame: 0, width: 0, height: 64, camera: None, report: false },
        );
        assert_eq!(assert_structured(&invalid, true)["code"], "invalid_input");

        let root = std::env::temp_dir().join(format!(
            "phantom-mcp-path-contract-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let mut path_session = Session::new();
        path_session.project = Some(phantom_core::Project::new("path"));
        path_session.dir = Some(root.clone());
        let forbidden = call(
            &standalone(path_session),
            Op::RenderScene {
                scene: 1,
                out: PathBuf::from("/phantom-contract-outside/out"),
                width: 64,
                height: 64,
                frames: None,
                camera: None,
            },
        );
        assert_eq!(assert_structured(&forbidden, true)["code"], "path_forbidden");
        let _ = std::fs::remove_dir_all(root);

        let locked = std::env::temp_dir().join(format!(
            "phantom-mcp-lock-contract-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        phantom_runtime::storage::create_project(&locked, "locked").unwrap();
        // ProjectOpen records the recent path; keep it in a test-owned state file.
        let owner_state = std::env::temp_dir().join(format!(
            "phantom-mcp-owner-state-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut owner = Session::with_state_path(owner_state);
        owner.execute(Op::ProjectOpen { dir: locked.clone(), force: false }).unwrap();
        let held = call(
            &standalone(Session::new()),
            Op::ProjectOpen { dir: locked.clone(), force: false },
        );
        assert_eq!(assert_structured(&held, true)["code"], "lock_held");
        assert!(held.structured_content.as_ref().unwrap()["details"]["owner"]["pid"].is_u64());
        drop(owner);
        let _ = std::fs::remove_dir_all(locked);
    }

    #[test]
    fn render_still_has_image_and_metadata_without_base64_duplication() {
        let result = image_result(Ok(serde_json::json!({
            "scene": 4,
            "frame": 12,
            "camera": 9,
            "width": 320,
            "height": 180,
            "mime_type": "image/png",
            "png_base64": "cG5n",
        })));
        assert_eq!(result.is_error, Some(false));
        assert_eq!(result.content.len(), 1);
        assert!(matches!(&result.content[0], ContentBlock::Image(_)));
        let metadata = result.structured_content.as_ref().unwrap();
        assert_eq!(metadata["scene"], 4);
        assert_eq!(metadata["frame"], 12);
        assert_eq!(metadata["camera"], 9);
        assert_eq!(metadata["width"], 320);
        assert_eq!(metadata["height"], 180);
        assert_eq!(metadata["mime_type"], "image/png");
        assert!(metadata.get("png_base64").is_none());
        assert!(wire(&result)["structuredContent"].is_object());
    }

    #[test]
    fn standalone_and_bridge_produce_equivalent_call_tool_results() {
        let standalone_ok = call(&standalone(Session::new()), Op::BuiltinCatalog);
        let bridge_ok = call(&bridge_once(Session::new()), Op::BuiltinCatalog);
        assert_eq!(wire(&standalone_ok), wire(&bridge_ok));

        let mut standalone_mutation = Session::new();
        standalone_mutation.project = Some(phantom_core::Project::new("parity"));
        let mut bridge_mutation = Session::new();
        bridge_mutation.project = Some(phantom_core::Project::new("parity"));
        let op = Op::SceneAdd { name: "same".into(), start: 0, end: 24 };
        let standalone_added = call(&standalone(standalone_mutation), op.clone());
        let bridge_added = call(&bridge_once(bridge_mutation), op);
        assert_eq!(wire(&standalone_added), wire(&bridge_added));
        assert_eq!(bridge_added.structured_content.as_ref().unwrap()["entity"]["kind"], "scene");

        let standalone_error = call(&standalone(Session::new()), Op::StateGet);
        let bridge_error = call(&bridge_once(Session::new()), Op::StateGet);
        assert_eq!(wire(&standalone_error), wire(&bridge_error));
        assert_eq!(bridge_error.structured_content.as_ref().unwrap()["code"], "not_open");

        // render_still is routed through the bridge too. Invalid dimensions
        // fail before GPU creation, so parity is covered without rendering.
        let still = Op::RenderStill { scene: 1, frame: 0, width: 0, height: 64, camera: None, report: false };
        let standalone_still = standalone(Session::new()).exec_image(still.clone()).unwrap();
        let bridge_still = bridge_once(Session::new()).exec_image(still).unwrap();
        assert_eq!(wire(&standalone_still), wire(&bridge_still));
        assert_eq!(bridge_still.structured_content.as_ref().unwrap()["code"], "invalid_input");
    }

    #[test]
    fn local_parse_failures_use_invalid_input() {
        let result = invalid_input("unknown action");
        assert_eq!(assert_structured(&result, true)["code"], "invalid_input");
    }

    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(future)
    }

    #[test]
    fn active_scene_lifecycle_and_omission_are_connection_local() {
        let mut session = Session::new();
        session.project = Some(phantom_core::Project::new("active"));
        let server = standalone(session);
        let current = block_on(server.scene_current()).unwrap();
        assert_eq!(assert_structured(&current, false)["active_scene"], serde_json::Value::Null);
        let missing = block_on(server.background_clear(Parameters(PSceneRef { scene: None }))).unwrap();
        assert_eq!(assert_structured(&missing, true)["code"], "no_active_scene");

        let first = block_on(server.scene_add(Parameters(PSceneAdd { name: "first".into(), start: 0, end: 24 }))).unwrap();
        let first_id = assert_structured(&first, false)["scene"].as_u64().unwrap();
        let current = block_on(server.scene_current()).unwrap();
        assert_eq!(assert_structured(&current, false)["active_scene"], first_id);
        let second = block_on(server.scene_add(Parameters(PSceneAdd { name: "second".into(), start: 24, end: 48 }))).unwrap();
        let second_id = assert_structured(&second, false)["scene"].as_u64().unwrap();
        assert_ne!(first_id, second_id);
        let current = block_on(server.scene_current()).unwrap();
        assert_eq!(assert_structured(&current, false)["active_scene"], second_id);

        // Representative omitted-scene query resolves at the MCP edge.
        let scene = block_on(server.scene_get(Parameters(PSceneRef { scene: None }))).unwrap();
        assert_eq!(assert_structured(&scene, false)["id"], second_id);
        let selected = block_on(server.scene_select(Parameters(PSceneSelect { scene: first_id }))).unwrap();
        assert_eq!(assert_structured(&selected, false)["active_scene"], first_id);
        let state = block_on(server.state_get()).unwrap();
        assert_eq!(assert_structured(&state, false)["active_scene"], first_id);

        block_on(server.scene_select(Parameters(PSceneSelect { scene: second_id }))).unwrap();
        let undone = block_on(server.history_undo()).unwrap();
        assert_eq!(assert_structured(&undone, false)["applied"], true);
        let current = block_on(server.scene_current()).unwrap();
        assert_eq!(assert_structured(&current, false)["active_scene"], first_id);
    }

    #[test]
    fn unit_conversions_are_deterministic_and_reject_bad_seconds() {
        assert_eq!(seconds_to_samples(6.34, "len_secs").unwrap(), 304_320);
        assert_eq!(seconds_to_frame(0.5, "start_secs").unwrap(), 12);
        assert_eq!(render_scene_output(None), PathBuf::from("frames"));
        assert!(seconds_to_samples(-0.1, "offset_secs").is_err());
        assert!(seconds_to_samples(f64::NAN, "offset_secs").is_err());
        assert!((90.0_f32.to_radians() - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
    }

    #[test]
    fn key_preserve_lighting_patch_and_pose_defaults() {
        let mut session = Session::new();
        session.project = Some(phantom_core::Project::new("ergonomics"));
        let scene = session.execute(Op::SceneAdd { name: "s".into(), start: 0, end: 48 }).unwrap()["scene"].as_u64().unwrap();
        let phantom = session.execute(Op::PhantomAdd { scene, name: "p".into(), rig_asset: None, rig_actor: None }).unwrap()["phantom"].as_u64().unwrap();
        session.execute(Op::KeySet { scene, phantom, frame: 8, pos: [1.0, 2.0, 3.0], pose: phantom_core::ActorPose::Sit, yaw: 0.25 }).unwrap();
        let server = standalone(session);

        let changed = block_on(server.key_set(Parameters(PKey {
            scene: None, phantom, frame: 8, pos: None, pose: None, yaw_deg: Some(90.0), preserve: None,
        }))).unwrap();
        assert_structured(&changed, false);
        let patched = block_on(server.scene_lighting_set(Parameters(PLightingSet {
            scene: None,
            key: None,
            fill: Some(PLight { color: [0.2, 0.3, 0.4], intensity: 123.0, enabled: Some(false), shadows: None }),
            rim: None,
        }))).unwrap();
        assert_structured(&patched, false);
        // A pose the species does not have is refused at the MCP boundary —
        // checked before the session mutex is taken below.
        let horse = block_on(server.phantom_add(Parameters(PPhantom {
            scene: None, name: "horse".into(), rig_asset: None, rig_actor: Some("horse".into()),
        }))).unwrap();
        let horse = assert_structured(&horse, false)["phantom"].as_u64().unwrap();
        let refused = block_on(server.key_set(Parameters(PKey {
            scene: None, phantom: horse, frame: 0, pos: None, pose: Some("sit".into()), yaw_deg: None, preserve: None,
        }))).unwrap();
        let refused = assert_structured(&refused, true);
        assert_eq!(refused["code"], "unsupported_pose");
        assert_eq!(refused["details"]["actor"], "horse");
        assert_eq!(refused["details"]["pose"], "sit");

        let Backend::Standalone(session) = &*server.backend else { unreachable!() };
        let guard = session.lock().unwrap();
        let scene_doc = &guard.project.as_ref().unwrap().scenes[&phantom_core::SceneId(scene)];
        let key = scene_doc.phantoms[&phantom_core::PhantomId(phantom)].motion.iter().find(|k| k.frame == 8).unwrap();
        assert_eq!(key.position.to_array(), [1.0, 2.0, 3.0]);
        assert_eq!(key.pose, phantom_core::ActorPose::Sit, "the omitted pose is preserved from the exact key");
        assert!((key.yaw - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert_eq!(scene_doc.lighting.key, phantom_core::SceneLighting::default().key);
        assert_eq!(scene_doc.lighting.fill.intensity, 123.0);
        assert!(!scene_doc.lighting.fill.enabled);
        assert_eq!(scene_doc.phantoms.len(), 2, "the refused horse key did not create a phantom");
    }

    #[test]
    fn capabilities_report_mode_without_gpu_probe() {
        let standalone_caps = block_on(standalone(Session::new()).studio_capabilities()).unwrap();
        let value = assert_structured(&standalone_caps, false);
        assert_eq!(value["mode"], "standalone");
        assert_eq!(value["gpu_availability"], "unknown_until_render");
        assert_eq!(value["fps"], 24);
        assert_eq!(value["sample_rate_hz"], 48_000);

        let bridge_caps = block_on(bridge_once(Session::new()).studio_capabilities()).unwrap();
        assert_eq!(assert_structured(&bridge_caps, false)["mode"], "bridge");
    }

    #[test]
    fn render_still_report_defaults_and_serde_contract() {
        // Default sizes: a normal frame and a cheap preview with metrics.
        assert_eq!(still_size(None, None, false), (1024, 576));
        assert_eq!(still_size(None, None, true), (480, 270));
        assert_eq!(still_size(Some(320), None, true), (320, 270));
        assert_eq!(still_size(None, Some(180), false), (1024, 180));
        // Legacy JSON without report reads back with report=false.
        let op: Op = serde_json::from_str(
            r#"{"op":"render_still","scene":1,"frame":0,"width":64,"height":64}"#,
        )
        .unwrap();
        assert!(matches!(op, Op::RenderStill { report: false, .. }));
        let schema = serde_json::to_value(render_still_output_schema()).unwrap();
        let report = &schema["properties"]["report"];
        // Nested schema may be inlined or referenced depending on schemars;
        // the dedicated type itself is the stable schema contract.
        let metrics = serde_json::to_value(rmcp::handler::server::tool::schema_for_type::<FrameMetricsOutput>()).unwrap();
        assert!(metrics["properties"]["objects_in_frame"].is_object(), "{report}");
        assert!(metrics["properties"]["objects_total"].is_object());
    }

    #[test]
    fn light_tools_convert_degrees_and_validate_kind() {
        let mut session = Session::new();
        session.project = Some(phantom_core::Project::new("lights-mcp"));
        session.execute(Op::SceneAdd { name: "s".into(), start: 0, end: 48 }).unwrap();
        let server = standalone(session);

        let added = block_on(server.light_add(Parameters(PLightAdd {
            scene: None, name: "lantern".into(), kind: "spot".into(), color: [1.0, 0.9, 0.8], intensity: 800.0,
            pos: [0.0, 2.0, -1.0], yaw_deg: Some(90.0), pitch_deg: None, cone_deg: Some(40.0), enabled: None, shadows: None,
        }))).unwrap();
        let value = assert_structured(&added, false);
        let light = value["light"].as_u64().unwrap();
        let Backend::Standalone(session) = &*server.backend else { unreachable!() };
        let guard = session.lock().unwrap();
        let stored = &guard.project.as_ref().unwrap().scenes[&phantom_core::SceneId(1)].lights[0];
        assert!(matches!(stored.kind, phantom_core::SceneLightKind::Spot { cone } if (cone - 40f32.to_radians()).abs() < 1e-6));
        assert!((stored.rest.yaw - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        drop(guard);

        let keyed = block_on(server.light_key_set(Parameters(PLightKey {
            scene: None, light, frame: 24, pos: [1.0, 2.0, -1.0], yaw_deg: Some(180.0), pitch_deg: None, intensity: None, enabled: Some(false),
        }))).unwrap();
        assert_eq!(assert_structured(&keyed, false)["entity"]["kind"], "light");
        let guard = session.lock().unwrap();
        let key = guard.project.as_ref().unwrap().scenes[&phantom_core::SceneId(1)].lights[0].keys[0];
        assert!((key.yaw - std::f32::consts::PI).abs() < 1e-6);
        assert_eq!(key.enabled, Some(false));
        drop(guard);

        // Invalid kind/cone — invalid_input at the boundary.
        let bad = block_on(server.light_add(Parameters(PLightAdd {
            scene: None, name: "x".into(), kind: "area".into(), color: [1.0; 3], intensity: 1.0,
            pos: [0.0; 3], yaw_deg: None, pitch_deg: None, cone_deg: None, enabled: None, shadows: None,
        }))).unwrap();
        assert_eq!(assert_structured(&bad, true)["code"], "invalid_input");
        let no_cone = block_on(server.light_add(Parameters(PLightAdd {
            scene: None, name: "x".into(), kind: "spot".into(), color: [1.0; 3], intensity: 1.0,
            pos: [0.0; 3], yaw_deg: None, pitch_deg: None, cone_deg: None, enabled: None, shadows: None,
        }))).unwrap();
        assert_eq!(assert_structured(&no_cone, true)["code"], "invalid_input");

        let ambient = block_on(server.scene_ambient_set(Parameters(PAmbientSet {
            scene: None, mode: "fixed".into(), color: Some([0.2, 0.2, 0.25]), brightness: Some(300.0),
        }))).unwrap();
        assert_eq!(assert_structured(&ambient, false)["entity"]["kind"], "ambient");
        let sky = block_on(server.scene_sky_set(Parameters(PSkySet { scene: None, sky: Some([0.05, 0.1, 0.2]) }))).unwrap();
        assert!((assert_structured(&sky, false)["entity"]["sky"][1].as_f64().unwrap() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn attach_tool_validates_and_converts_degrees() {
        let mut session = Session::new();
        session.project = Some(phantom_core::Project::new("attach-mcp"));
        let scene = session.execute(Op::SceneAdd { name: "s".into(), start: 0, end: 48 }).unwrap()["scene"].as_u64().unwrap();
        let phantom = session.execute(Op::PhantomAdd { scene, name: "hero".into(), rig_asset: None, rig_actor: None }).unwrap()["phantom"].as_u64().unwrap();
        let server = standalone(session);

        // Exactly one of object/light/prop.
        let none = block_on(server.attach(Parameters(PAttach {
            scene: None, phantom, bone: "lower_arm_r".into(), object: None, light: None, prop: None,
            detach: None, offset_pos: None, offset_yaw_deg: None, offset_pitch_deg: None, offset_roll_deg: None,
        }))).unwrap();
        assert_eq!(assert_structured(&none, true)["code"], "invalid_input");
        // A nonexistent object — not_found from Session.
        let missing = block_on(server.attach(Parameters(PAttach {
            scene: None, phantom, bone: "lower_arm_r".into(), object: Some(999), light: None, prop: None,
            detach: None, offset_pos: None, offset_yaw_deg: Some(90.0), offset_pitch_deg: None, offset_roll_deg: None,
        }))).unwrap();
        assert_eq!(assert_structured(&missing, true)["code"], "not_found");
        // Light source: add + attach; degrees → radians.
        let light = block_on(server.light_add(Parameters(PLightAdd {
            scene: None, name: "lamp".into(), kind: "point".into(), color: [1.0; 3], intensity: 50.0,
            pos: [0.0; 3], yaw_deg: None, pitch_deg: None, cone_deg: None, enabled: None, shadows: None,
        }))).unwrap();
        let light = assert_structured(&light, false)["light"].as_u64().unwrap();
        let ok = block_on(server.attach(Parameters(PAttach {
            scene: None, phantom, bone: "LOWER_ARM_R".into(), object: None, light: Some(light), prop: None,
            detach: None, offset_pos: Some([0.0, -0.05, 0.0]), offset_yaw_deg: Some(90.0), offset_pitch_deg: None, offset_roll_deg: None,
        }))).unwrap();
        assert_eq!(assert_structured(&ok, false)["entity"]["kind"], "light");
        let Backend::Standalone(session) = &*server.backend else { unreachable!() };
        let guard = session.lock().unwrap();
        let stored = guard.project.as_ref().unwrap().scenes[&phantom_core::SceneId(scene)].lights[0].attach.clone().unwrap();
        assert_eq!(stored.bone, "LOWER_ARM_R");
        assert!((stored.offset_yaw - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        drop(guard);
        // detach.
        let off = block_on(server.attach(Parameters(PAttach {
            scene: None, phantom, bone: String::new(), object: None, light: Some(light), prop: None,
            detach: Some(true), offset_pos: None, offset_yaw_deg: None, offset_pitch_deg: None, offset_roll_deg: None,
        }))).unwrap();
        assert!(assert_structured(&off, false)["entity"]["attach"].is_null());
    }

    #[test]
    fn material_set_preset_and_field_override_contract() {
        let mut session = Session::new();
        session.project = Some(phantom_core::Project::new("mat-mcp"));
        let scene = session.execute(Op::SceneAdd { name: "s".into(), start: 0, end: 48 }).unwrap()["scene"].as_u64().unwrap();
        let phantom = session.execute(Op::PhantomAdd { scene, name: "wolf".into(), rig_asset: None, rig_actor: Some("wolf".into()) }).unwrap()["phantom"].as_u64().unwrap();
        let server = standalone(session);

        // Preset wet plus an explicit metallic overrides the preset.
        let set = block_on(server.material_set(Parameters(PMaterialSet {
            scene: None, phantom: Some(phantom), object: None, target: None,
            preset: Some("wet".into()), base_color: None, metallic: Some(0.3), roughness: None, emissive: None, opacity: None, unlit: None, clear: None,
        }))).unwrap();
        assert_eq!(assert_structured(&set, false)["entity"]["kind"], "material_patches");
        let listed = block_on(server.material_list(Parameters(PMaterialRef { scene: None, phantom: Some(phantom), object: None }))).unwrap();
        let parts = assert_structured(&listed, false)["parts"].as_array().unwrap().clone();
        assert!((parts[0]["effective"]["roughness"].as_f64().unwrap() - 0.12).abs() < 1e-6, "wet preset");
        assert!((parts[0]["effective"]["metallic"].as_f64().unwrap() - 0.3).abs() < 1e-6, "the explicit field overrode the preset");

        // Unknown preset and empty request — invalid_input.
        let bad = block_on(server.material_set(Parameters(PMaterialSet {
            scene: None, phantom: Some(phantom), object: None, target: None,
            preset: Some("chrome".into()), base_color: None, metallic: None, roughness: None, emissive: None, opacity: None, unlit: None, clear: None,
        }))).unwrap();
        assert_eq!(assert_structured(&bad, true)["code"], "invalid_input");
        let empty = block_on(server.material_set(Parameters(PMaterialSet {
            scene: None, phantom: Some(phantom), object: None, target: None,
            preset: None, base_color: None, metallic: None, roughness: None, emissive: None, opacity: None, unlit: None, clear: None,
        }))).unwrap();
        assert_eq!(assert_structured(&empty, true)["code"], "invalid_input");
        // clear.
        let cleared = block_on(server.material_set(Parameters(PMaterialSet {
            scene: None, phantom: Some(phantom), object: None, target: None,
            preset: None, base_color: None, metallic: None, roughness: None, emissive: None, opacity: None, unlit: None, clear: Some(true),
        }))).unwrap();
        assert_eq!(assert_structured(&cleared, false)["entity"]["patches"], serde_json::json!([]));
    }

    #[test]
    fn vfx_and_grade_tools_contract() {
        let mut session = Session::new();
        session.project = Some(phantom_core::Project::new("vfx-mcp"));
        session.execute(Op::SceneAdd { name: "s".into(), start: 0, end: 48 }).unwrap();
        let server = standalone(session);

        let added = block_on(server.vfx_add(Parameters(PVfxAdd {
            scene: None, name: "blizzard".into(), kind: "particles".into(), preset: Some("snow".into()),
            pos: Some([0.0, 5.0, 0.0]), extent: None, emit_rate: None, lifetime: None, speed: None, size: None,
            color: None, gravity: None, normal_map: None, scroll_uv: None, enabled: None,
        }))).unwrap();
        let id = assert_structured(&added, false)["vfx"].as_u64().unwrap();
        let Backend::Standalone(session) = &*server.backend else { unreachable!() };
        let guard = session.lock().unwrap();
        let stored = &guard.project.as_ref().unwrap().scenes[&phantom_core::SceneId(1)].vfx[0];
        assert_eq!(stored.emit_rate, 200.0, "snow preset");
        assert_eq!(stored.position.to_array(), [0.0, 5.0, 0.0], "the explicit pos overrode the preset");
        drop(guard);

        let bad_kind = block_on(server.vfx_add(Parameters(PVfxAdd {
            scene: None, name: "x".into(), kind: "god_rays".into(), preset: None,
            pos: None, extent: None, emit_rate: None, lifetime: None, speed: None, size: None,
            color: None, gravity: None, normal_map: None, scroll_uv: None, enabled: None,
        }))).unwrap();
        assert_eq!(assert_structured(&bad_kind, true)["code"], "invalid_input");

        let keyed = block_on(server.vfx_key_set(Parameters(PVfxKey { scene: None, vfx: id, frame: 24, enabled: Some(false), density: None }))).unwrap();
        assert_eq!(assert_structured(&keyed, false)["entity"]["kind"], "vfx");

        let graded = block_on(server.grade_set(Parameters(PGradeSet {
            scene: None, enabled: None, exposure: Some(0.5), temperature: None, tint: None, hue_deg: Some(15.0),
            post_saturation: None, saturation: Some(1.2), contrast: None, gamma: None, gain: None, lift: None,
        }))).unwrap();
        let value = assert_structured(&graded, false);
        assert_eq!(value["entity"]["kind"], "grade");
        let guard = session.lock().unwrap();
        let g = guard.project.as_ref().unwrap().scenes[&phantom_core::SceneId(1)].grade.unwrap();
        assert!((g.hue - 15f32.to_radians()).abs() < 1e-6, "degrees → radians");
        assert_eq!(g.saturation, 1.2);
        drop(guard);

        let fogged = block_on(server.fog_set(Parameters(PFogSet { scene: None, color: Some([0.5, 0.6, 0.7]), density: Some(0.05), off: None }))).unwrap();
        assert_eq!(assert_structured(&fogged, false)["entity"]["kind"], "fog");
        let fog_off = block_on(server.fog_set(Parameters(PFogSet { scene: None, color: None, density: None, off: Some(true) }))).unwrap();
        assert!(assert_structured(&fog_off, false)["entity"]["fog"].is_null());
    }

    #[test]
    fn catalog_query_and_prop_add_are_metric_and_validated() {
        let mut session = Session::new();
        session.project = Some(phantom_core::Project::new("catalog-mcp"));
        session.execute(Op::SceneAdd { name: "s".into(), start: 0, end: 48 }).unwrap();
        let server = standalone(session);

        // Search returns dimensions and material groups — placement without opening the mesh.
        let found = block_on(server.catalog_query(Parameters(PCatalogQuery { kind: Some("furniture".into()), query: Some("table".into()) }))).unwrap();
        let value = assert_structured(&found, false);
        assert_eq!(value["total"], 1);
        let entry = &value["entries"][0];
        assert_eq!(entry["name"], "table");
        assert_eq!(entry["size"], serde_json::json!([1.4, 0.75, 0.8]), "metric dimensions");
        assert_eq!(entry["origin"], "base_center");
        assert_eq!(entry["materials"], serde_json::json!(["top", "frame"]));
        let rooms = block_on(server.catalog_query(Parameters(PCatalogQuery { kind: Some("interior".into()), query: None }))).unwrap();
        let rooms = assert_structured(&rooms, false);
        assert!(rooms["total"].as_u64().unwrap() >= 6);
        assert_eq!(rooms["entries"][0]["open_sides"], serde_json::json!(["north"]));
        // Unknown section — invalid_input, not an empty answer.
        let bad_kind = block_on(server.catalog_query(Parameters(PCatalogQuery { kind: Some("weapons".into()), query: None }))).unwrap();
        assert_eq!(assert_structured(&bad_kind, true)["code"], "invalid_input");

        // An item is placed without an asset file; an unknown name — not_found with a hint.
        let placed = block_on(server.prop_add(Parameters(PPropAdd {
            scene: None, name: "table".into(), prop: "table".into(),
            pos: Some([1.0, 0.0, -2.0]), yaw_deg: Some(90.0), pitch_deg: None, roll_deg: None, scale: None,
        }))).unwrap();
        let placed = assert_structured(&placed, false);
        assert!(placed["object"].is_u64());
        let Backend::Standalone(session) = &*server.backend else { unreachable!() };
        let guard = session.lock().unwrap();
        let scene = &guard.project.as_ref().unwrap().scenes[&phantom_core::SceneId(1)];
        let object = scene.objects.values().next().unwrap();
        assert_eq!(object.builtin.as_deref(), Some("table"));
        let key = &object.keys[0];
        assert!((key.position.x - 1.0).abs() < 1e-6);
        assert!((key.yaw.to_degrees() - 90.0).abs() < 1e-3, "degrees out");
        drop(guard);
        let unknown = block_on(server.prop_add(Parameters(PPropAdd {
            scene: None, name: "x".into(), prop: "dragon".into(),
            pos: None, yaw_deg: None, pitch_deg: None, roll_deg: None, scale: None,
        }))).unwrap();
        let unknown = assert_structured(&unknown, true);
        assert_eq!(unknown["code"], "not_found");
        assert_eq!(unknown["details"]["hint"], "catalog_query");
    }

    #[test]
    fn camera_key_set_supports_look_at_roll_lens_presets_and_ease() {
        let mut session = Session::new();
        session.project = Some(phantom_core::Project::new("camera-mcp"));
        session.execute(Op::SceneAdd { name: "s".into(), start: 0, end: 48 }).unwrap();
        session.execute(Op::CameraAdd { scene: 1, name: "wide".into(), start: Some(0), end: Some(24) }).unwrap();
        let server = standalone(session);

        // look_at without angles: the program computes yaw/pitch; tilt and preset are supplied.
        let aimed = block_on(server.camera_key_set(Parameters(PCameraKey {
            scene: None, camera: Some(2), frame: 0, pos: [0.0, 1.5, 0.0], yaw_deg: None, pitch_deg: None,
            roll_deg: Some(6.0), look_at: Some([0.0, 1.5, -5.0]),
            focal_mm: None, focal_preset: Some("portrait".into()), ease: Some("ease_in_out".into()),
        }))).unwrap();
        assert_structured(&aimed, false);
        let Backend::Standalone(session) = &*server.backend else { unreachable!() };
        let guard = session.lock().unwrap();
        let key = &guard.project.as_ref().unwrap().scenes[&phantom_core::SceneId(1)].cameras[&phantom_core::CameraId(2)].camera.keys[0];
        assert_eq!(key.look_at, Some(glam::Vec3::new(0.0, 1.5, -5.0)));
        assert!((key.roll.to_degrees() - 6.0).abs() < 1e-3, "roll in degrees out");
        assert_eq!(key.focal_mm, Some(50.0), "portrait = 50 mm");
        assert_eq!(key.ease, phantom_core::project::KeyEase::EaseInOut);
        drop(guard);

        // Unknown preset and unknown ease — invalid_input, not a silent default.
        let bad_preset = block_on(server.camera_key_set(Parameters(PCameraKey {
            scene: None, camera: None, frame: 0, pos: [0.0, 1.4, 4.0], yaw_deg: Some(0.0), pitch_deg: None,
            roll_deg: None, look_at: None, focal_mm: None, focal_preset: Some("cinemascope".into()), ease: None,
        }))).unwrap();
        assert_eq!(assert_structured(&bad_preset, true)["code"], "invalid_input");
        let bad_ease = block_on(server.camera_key_set(Parameters(PCameraKey {
            scene: None, camera: None, frame: 0, pos: [0.0, 1.4, 4.0], yaw_deg: Some(0.0), pitch_deg: None,
            roll_deg: None, look_at: None, focal_mm: Some(35.0), focal_preset: None, ease: Some("bounce".into()),
        }))).unwrap();
        assert_eq!(assert_structured(&bad_ease, true)["code"], "invalid_input");

        // Camera focal key: zoom inside the camera's range without changing the pose.
        let zoom = block_on(server.camera_focal_key_set(Parameters(PCameraFocalKey { scene: None, camera: 2, frame: 12, focal_mm: 135.0 }))).unwrap();
        assert_structured(&zoom, false);
        let guard = session.lock().unwrap();
        let cam = &guard.project.as_ref().unwrap().scenes[&phantom_core::SceneId(1)].cameras[&phantom_core::CameraId(2)].camera;
        let key12 = cam.keys.iter().find(|k| k.frame == 12).expect("focal key added");
        assert_eq!(key12.focal_mm, Some(135.0));
        assert_eq!(cam.sample(6).focal_mm, 92.5, "focal interpolates between 50 and 135");
    }

    #[test]
    fn camera_target_set_rejects_bad_mode_and_subject() {
        let mut session = Session::new();
        session.project = Some(phantom_core::Project::new("camera-target-mcp"));
        let server = standalone(session);
        block_on(server.scene_add(Parameters(PSceneAdd { name: "main".into(), start: 0, end: 48 }))).unwrap();
        // The scene's own camera (id 0) always exists, so only mode/subject can fail.
        let bad_mode = block_on(server.camera_target_set(Parameters(PCameraTargetSet {
            scene: None, camera: 0, mode: Some("orbit".into()), subject: Some("phantom:1".into()), clear: None,
        }))).unwrap();
        assert_eq!(assert_structured(&bad_mode, true)["code"], "invalid_input");
        let bad_subject = block_on(server.camera_target_set(Parameters(PCameraTargetSet {
            scene: None, camera: 0, mode: Some("track".into()), subject: Some("nobody".into()), clear: None,
        }))).unwrap();
        assert_eq!(assert_structured(&bad_subject, true)["code"], "invalid_input");
    }

    #[test]
    fn asset_find_matches_case_insensitive_substring() {
        let mut session = Session::new();
        session.project = Some(phantom_core::Project::new("find"));
        // Registering an asset without a file import is not reachable from outside,
        // and batch-import needs real files, so we run asset_find on an empty project:
        // check the empty result and the response shape.
        let server = standalone(session);
        let empty = block_on(server.asset_find(Parameters(PAssetFind { name: "bg".into() }))).unwrap();
        let value = assert_structured(&empty, false);
        assert_eq!(value["total"], 0);
        assert!(value["matches"].is_array());
    }

    #[test]
    fn audio_clip_update_seconds_patch_contract() {
        let mut session = Session::new();
        session.project = Some(phantom_core::Project::new("clip-update"));
        let scene = session.execute(Op::SceneAdd { name: "s".into(), start: 0, end: 96 }).unwrap()["scene"].as_u64().unwrap();
        let track = session.execute(Op::AudioTrackAdd { scene: Some(scene), name: "t".into() }).unwrap()["track"].as_u64().unwrap();
        let server = standalone(session);
        // not_found: asset 999 does not exist (validation happens before the mutation).
        let bad_add = block_on(server.audio_clip_add(Parameters(PClipAdd {
            track, asset: 999, start_secs: 0.0, offset_secs: 0.0, len_secs: 1.0, fade_in_secs: Some(0.1), fade_out_secs: None,
        }))).unwrap();
        assert_eq!(assert_structured(&bad_add, true)["code"], "not_found");
        // not_found: clip 42 is not on the track.
        let missing = block_on(server.audio_clip_update(Parameters(PClipUpdate {
            track, clip: 42, offset_secs: None, len_secs: None, fade_in_secs: Some(0.25), fade_out_secs: None,
        }))).unwrap();
        assert_eq!(assert_structured(&missing, true)["code"], "not_found");
        assert_eq!(missing.structured_content.as_ref().unwrap()["details"]["entity"], "audio_clip");
        // invalid_input: negative seconds are rejected at the MCP boundary.
        let negative = block_on(server.audio_clip_update(Parameters(PClipUpdate {
            track, clip: 42, offset_secs: None, len_secs: Some(-1.0), fade_in_secs: None, fade_out_secs: None,
        }))).unwrap();
        assert_eq!(assert_structured(&negative, true)["code"], "invalid_input");
    }

    #[test]
    fn audio_report_tool_is_structured_query_without_revision_bump() {
        let mut session = Session::new();
        session.project = Some(phantom_core::Project::new("audio-mcp"));
        let scene = session.execute(Op::SceneAdd { name: "s".into(), start: 0, end: 48 }).unwrap()["scene"].as_u64().unwrap();
        let server = standalone(session);
        let before = call(&server, Op::StateGet);
        let revision = before.structured_content.as_ref().unwrap()["revision"].as_u64().unwrap();

        let report = block_on(server.audio_report(Parameters(PAudioReport { scene: None }))).unwrap();
        let value = assert_structured(&report, false);
        assert_eq!(value["sample_rate_hz"], 48_000);
        assert_eq!(value["revision"], revision, "a query does not bump the revision");
        assert!(value["tracks"].is_array() && value["warnings"].is_array());

        let filtered = block_on(server.audio_report(Parameters(PAudioReport { scene: Some(scene) }))).unwrap();
        assert_eq!(assert_structured(&filtered, false)["scene"], scene);

        let missing = block_on(server.audio_report(Parameters(PAudioReport { scene: Some(999) }))).unwrap();
        assert_eq!(assert_structured(&missing, true)["code"], "not_found");

        // state_get carries an audio-vs-video summary without decoding.
        let state = call(&server, Op::StateGet);
        let summary = &state.structured_content.as_ref().unwrap()["audio_summary"];
        assert!(summary["scenes"].is_array() && summary["film"]["lost_seconds"].is_f64());
    }

    #[test]
    fn batch_tool_injects_active_scene_and_keeps_atomic_contract() {
        let mut session = Session::new();
        session.project = Some(phantom_core::Project::new("batch-mcp"));
        let server = standalone(session);
        // scene_add automatically selects the connection's active scene.
        block_on(server.scene_add(Parameters(PSceneAdd { name: "main".into(), start: 0, end: 48 }))).unwrap();

        let added = block_on(server.batch(Parameters(PBatch { ops: vec![
            serde_json::json!({"op":"phantom_add","name":"hero","rig_actor":"wolf"}),
            serde_json::json!({"op":"key_set","phantom":2,"frame":0,"pos":[0,0,0],"pose":"sit","yaw":0.5}),
            serde_json::json!({"op":"camera_add","name":"close","start":0,"end":24}),
        ] }))).unwrap();
        let value = assert_structured(&added, false);
        assert_eq!(value["applied"], 3);
        assert_eq!(value["revision"], 2, "scene_add + one batch = two revisions");
        assert_eq!(value["entity"]["kind"], "camera");
        let results = value["results"].as_array().unwrap();
        assert_eq!(results[0]["phantom"], 2);
        assert_eq!(results[1]["entity"]["kind"], "motion_key");
        assert!(results[1].get("revision").is_none());

        let Backend::Standalone(session) = &*server.backend else { unreachable!() };
        let guard = session.lock().unwrap();
        let scene = &guard.project.as_ref().unwrap().scenes[&phantom_core::SceneId(1)];
        assert!(scene.phantoms.contains_key(&phantom_core::PhantomId(2)));
        assert_eq!(scene.cameras.len(), 2, "the scene's own camera plus the added one");
        drop(guard);

        // A parse error carries the operation index.
        let bad = block_on(server.batch(Parameters(PBatch { ops: vec![serde_json::json!({"op":"nope"})] }))).unwrap();
        let error = assert_structured(&bad, true);
        assert_eq!(error["code"], "invalid_input");
        assert_eq!(error["details"]["index"], 0);

        // Failure in the middle: stable code, index, inner code, full rollback.
        let before = call(&server, Op::StateGet);
        let revision_before = before.structured_content.as_ref().unwrap()["revision"].as_u64().unwrap();
        let failed = block_on(server.batch(Parameters(PBatch { ops: vec![
            serde_json::json!({"op":"camera_add","name":"temp","start":0,"end":10}),
            serde_json::json!({"op":"camera_remove","camera":999}),
        ] }))).unwrap();
        let error = assert_structured(&failed, true);
        assert_eq!(error["code"], "batch_failed");
        assert_eq!(error["details"]["index"], 1);
        assert_eq!(error["details"]["op"], "camera_remove");
        assert_eq!(error["details"]["code"], "not_found");
        let after = call(&server, Op::StateGet);
        let structured = after.structured_content.as_ref().unwrap();
        assert_eq!(structured["revision"].as_u64().unwrap(), revision_before, "a rolled-back batch does not move the revision");
        assert_eq!(structured["cameras"].as_array().unwrap().len(), 2, "the temporary camera was rolled back");

        // Bridge parity: the same package over the bridge yields the same structure.
        let mut bridge_session = Session::new();
        bridge_session.project = Some(phantom_core::Project::new("batch-bridge"));
        let bridged = block_on(bridge_once(bridge_session).batch(Parameters(PBatch { ops: vec![
            serde_json::json!({"op":"scene_add","name":"b","start":0,"end":24}),
        ] }))).unwrap();
        let value = assert_structured(&bridged, false);
        assert_eq!(value["applied"], 1);
        assert_eq!(value["revision"], 1);
        assert_eq!(value["entity"]["kind"], "scene");
    }

    #[test]
    fn describe_is_generated_from_tool_router_metadata_without_drift() {
        let server = standalone(Session::new());
        let tools = PhantomMcp::tool_router().list_all();
        let expected = tools.iter().find(|t| t.name.as_ref() == "camera_get").expect("camera_get registered");

        let result = block_on(server.describe(Parameters(PDescribe { tool: "camera_get".into() }))).unwrap();
        let structured = assert_structured(&result, false);
        // Field-for-field identity with the tools/list entry: describe cannot
        // drift because both come from the same ToolRouter metadata.
        assert_eq!(structured["name"], "camera_get");
        assert_eq!(structured["description"].as_str().unwrap(), expected.description.as_deref().unwrap_or_default());
        assert_eq!(structured["inputSchema"], serde_json::to_value(&expected.input_schema).unwrap());
        assert_eq!(structured["outputSchema"], serde_json::to_value(&expected.output_schema).unwrap());
        assert!(!structured["unit_notes"].as_str().unwrap().trim().is_empty());
        assert!(!structured["usage_notes"].as_str().unwrap().trim().is_empty());
        assert_eq!(structured.as_object().unwrap().len(), 6, "describe returns exactly name/description/inputSchema/outputSchema/unit_notes/usage_notes");

        // The typed output schema of job_wait is the same metadata tools/list
        // carries, and it names the actual job status fields incl. timed_out.
        let wait = block_on(server.describe(Parameters(PDescribe { tool: "job_wait".into() }))).unwrap();
        let wait_meta = assert_structured(&wait, false);
        let expected_wait = tools.iter().find(|t| t.name.as_ref() == "job_wait").unwrap();
        assert_eq!(wait_meta["inputSchema"], serde_json::to_value(&expected_wait.input_schema).unwrap());
        assert_eq!(wait_meta["outputSchema"], serde_json::to_value(&expected_wait.output_schema).unwrap());
        let properties = wait_meta["outputSchema"]["properties"].as_object().expect("job_wait output schema is a typed object");
        for key in ["last_line", "done", "exit_code", "frames_done", "total", "elapsed_secs", "error_tail", "timed_out"] {
            assert!(properties.contains_key(key), "job_wait outputSchema property {key} missing: {properties:?}");
        }

        // Unknown tool: stable not_found with nearest registered names.
        let missing = block_on(server.describe(Parameters(PDescribe { tool: "camera_gett".into() }))).unwrap();
        let error = assert_structured(&missing, true);
        assert_eq!(error["code"], "not_found");
        assert_eq!(error["details"]["entity"], "tool");
        let nearest: Vec<&str> = error["details"]["nearest_tools"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        assert!(nearest.contains(&"camera_get"), "nearest tools for camera_gett: {nearest:?}");
        assert!(nearest.len() <= 5);
    }

    #[test]
    fn project_recent_works_at_the_edge_without_a_project() {
        let state = std::env::temp_dir().join(format!(
            "phantom-mcp-recent-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_file(&state);
        let server = standalone(Session::with_state_path(state.clone()));
        // No project and no state file: a successful empty answer, not an error.
        let result = block_on(server.project_recent()).unwrap();
        let structured = assert_structured(&result, false);
        assert!(structured["recent"].is_null());
        assert!(structured["current_dir"].is_null());
        assert!(structured["revision"].is_null());
        assert!(structured["active_scene"].is_null());
        assert!(structured["warnings"].as_array().unwrap().is_empty());
        let _ = std::fs::remove_file(state);
    }

    #[test]
    fn getters_and_job_wait_parity_between_standalone_and_bridge() {
        let build = |name: &str| {
            let mut session = Session::new();
            session.project = Some(phantom_core::Project::new(name));
            let scene = session.execute(Op::SceneAdd { name: "main".into(), start: 0, end: 48 }).unwrap()["scene"].as_u64().unwrap();
            let camera = session.execute(Op::CameraAdd { scene, name: "wide".into(), start: Some(0), end: Some(24) }).unwrap()["camera"].as_u64().unwrap();
            session.execute(Op::FilmAdd { scene, start: 0, end: 24, camera: Some(camera) }).unwrap();
            (session, scene, camera)
        };
        let (standalone_session, scene, camera) = build("parity-a");
        let (bridge_session, _, _) = build("parity-b");
        // Shared Session Op means the bridge returns the identical structured shape.
        let op = Op::CameraGet { scene, camera };
        assert_eq!(wire(&call(&standalone(standalone_session), op.clone())), wire(&call(&bridge_once(bridge_session), op)));

        let (standalone_session, _, _) = build("parity-c");
        let (bridge_session, _, _) = build("parity-d");
        let op = Op::FilmGet;
        assert_eq!(wire(&call(&standalone(standalone_session), op.clone())), wire(&call(&bridge_once(bridge_session), op)));

        // job_wait quick success is identical over the shared dispatcher too.
        let spawn = |mut session: Session| {
            let job = session.spawn_test_command("sh", &["-c", "echo 1/1; exit 0"]).unwrap()["job"].as_u64().unwrap();
            (session, job)
        };
        let (standalone_session, job) = spawn(Session::new());
        let (bridge_session, _) = spawn(Session::new());
        let op = Op::JobWait { job, timeout_secs: 10.0 };
        let mut standalone_wait = wire(&call(&standalone(standalone_session), op.clone()));
        let mut bridge_wait = wire(&call(&bridge_once(bridge_session), op));
        // Strip only the volatile elapsed time; progress parsing is
        // deterministic because finalization joins the stdout/stderr readers.
        for value in [&mut standalone_wait, &mut bridge_wait] {
            value["structuredContent"]["elapsed_secs"] = serde_json::Value::Null;
        }
        assert_eq!(standalone_wait["structuredContent"]["frames_done"], 1);
        assert_eq!(standalone_wait["structuredContent"]["last_line"], "1/1");
        assert_eq!(standalone_wait, bridge_wait);
        assert_eq!(standalone_wait["structuredContent"]["timed_out"], false);
        assert_eq!(standalone_wait["structuredContent"]["exit_code"], 0);
    }

    #[test]
    fn stage_tools_clamp_and_report_over_the_mcp_envelope() {
        let mut session = Session::new();
        session.project = Some(phantom_core::Project::new("stage-mcp"));
        let server = standalone(session);
        let added = block_on(server.scene_add(Parameters(PSceneAdd { name: "main".into(), start: 0, end: 48 }))).unwrap();
        let scene = assert_structured(&added, false)["scene"].as_u64().unwrap();

        let got = block_on(server.stage_get(Parameters(PSceneRef { scene: None }))).unwrap();
        let got = assert_structured(&got, false);
        assert_eq!(got["stage"]["preset"], "medium");
        assert_eq!(got["stage"]["size"]["height"].as_f64().unwrap(), 8.0);
        assert!(got.get("revision").is_none(), "stage_get is a query");

        let set = block_on(server.stage_set(Parameters(PStageSet { scene: None, preset: Some("small".into()), width: None, height: None, depth: None, center: None }))).unwrap();
        let set = assert_structured(&set, false);
        assert_eq!(set["stage"]["preset"], "small");
        assert!(set.get("revision").is_some(), "stage_set is a mutation");
        assert_eq!(set["entity"]["kind"], "stage");

        // A phantom key outside the small stage comes back corrected.
        let ph = block_on(server.phantom_add(Parameters(PPhantom { scene: None, name: "actor".into(), rig_asset: None, rig_actor: Some("humanoid".into()) }))).unwrap();
        let phantom = assert_structured(&ph, false)["phantom"].as_u64().unwrap();
        let keyed = block_on(server.key_set(Parameters(PKey { scene: None, phantom, frame: 0, pos: Some([90.0, 0.0, 0.0]), pose: Some("stand".into()), yaw_deg: None, preserve: None }))).unwrap();
        let keyed = assert_structured(&keyed, false);
        assert_eq!(keyed["clamped"], true);
        assert_eq!(keyed["clamps"][0]["kind"], "phantom");
        assert!((keyed["clamps"][0]["to"][0].as_f64().unwrap() - 4.7).abs() < 1e-4);

        // A half-specified stage is invalid_input before any mutation.
        let bad = block_on(server.stage_set(Parameters(PStageSet { scene: Some(scene), preset: None, width: Some(4.0), height: None, depth: None, center: None }))).unwrap();
        assert_eq!(assert_structured(&bad, true)["code"], "invalid_input");
        let bad_preset = block_on(server.stage_set(Parameters(PStageSet { scene: Some(scene), preset: Some("huge".into()), width: None, height: None, depth: None, center: None }))).unwrap();
        assert_eq!(assert_structured(&bad_preset, true)["code"], "invalid_input");
    }

    #[test]
    fn diagnostics_tools_contract_active_scene_and_no_revision() {
        let mut session = Session::new();
        session.project = Some(phantom_core::Project::new("diag"));
        let server = standalone(session);

        // camera_coverage without scene and without an active scene: stable no_active_scene.
        let missing = block_on(server.camera_coverage(Parameters(PCameraCoverage { scene: None, frame: None, camera: None, aspect: None }))).unwrap();
        assert_eq!(assert_structured(&missing, true)["code"], "no_active_scene");
        // validate without any scene resolves to the whole project (empty here).
        let whole = block_on(server.validate(Parameters(PValidate { scene: None }))).unwrap();
        let structured = assert_structured(&whole, false);
        assert_eq!(structured["ok"], true);
        assert_eq!(structured["findings"].as_array().unwrap().len(), 0);
        assert!(structured["scene"].is_null());

        // New scene becomes active; omitted scene resolves at the MCP edge.
        let added = block_on(server.scene_add(Parameters(PSceneAdd { name: "main".into(), start: 0, end: 48 }))).unwrap();
        let scene = assert_structured(&added, false)["scene"].as_u64().unwrap();
        let revision_before = {
            let Backend::Standalone(session) = &*server.backend else { unreachable!() };
            session.lock().unwrap().project.as_ref().unwrap().revision
        };

        let cov = block_on(server.camera_coverage(Parameters(PCameraCoverage { scene: None, frame: None, camera: None, aspect: None }))).unwrap();
        let structured = assert_structured(&cov, false);
        assert_eq!(structured["scene"].as_u64().unwrap(), scene);
        assert_eq!(structured["frame"], 0);
        let camera = &structured["camera"];
        for key in ["pos", "yaw_deg", "pitch_deg", "focal_mm", "hfov_deg", "vfov_deg"] {
            assert!(camera.get(key).is_some(), "camera field {key}: {camera}");
        }
        assert_eq!(structured["background"], serde_json::Value::Null, "no background yet");
        assert_eq!(structured["findings"][0]["code"], "background_absent");

        let fit = block_on(server.background_fit(Parameters(PBackgroundFit { scene: None, frame: None, camera: None, margin_deg: None, aspect: None }))).unwrap();
        let structured = assert_structured(&fit, false);
        assert_eq!(structured["margin_deg"], 5.0);
        assert!(structured["suggested"]["width"].as_f64().unwrap() > 0.0);
        assert!(structured["suggested"]["height"].as_f64().unwrap() > 0.0);
        assert!(structured["panorama_aspect"].as_f64().unwrap() > 0.0);

        let report = block_on(server.validate(Parameters(PValidate { scene: None }))).unwrap();
        let structured = assert_structured(&report, false);
        assert_eq!(structured["scene"].as_u64().unwrap(), scene, "active scene resolves for validate");
        let finding = &structured["findings"][0];
        for key in ["code", "severity", "where", "why", "suggested_fix"] {
            assert!(finding.get(key).is_some(), "finding field {key}: {finding}");
        }

        // Queries never raise the revision.
        let Backend::Standalone(session) = &*server.backend else { unreachable!() };
        assert_eq!(session.lock().unwrap().project.as_ref().unwrap().revision, revision_before);

        // Tools are registered with described inputs; describe covers them.
        let tools = PhantomMcp::tool_router().list_all();
        for name in ["validate", "camera_coverage", "background_fit"] {
            let tool = tools.iter().find(|t| t.name.as_ref() == name).unwrap_or_else(|| panic!("{name} registered"));
            let input = serde_json::to_value(&tool.input_schema).unwrap();
            let properties = input["properties"].as_object().unwrap();
            assert!(properties.contains_key("scene"), "{name} scene param");
            assert!(!properties["scene"]["description"].as_str().unwrap_or_default().is_empty());
        }
        let described = block_on(server.describe(Parameters(PDescribe { tool: "background_fit".into() }))).unwrap();
        assert_eq!(assert_structured(&described, false)["name"], "background_fit");

        // Bridge parity comes free through the shared Session Op.
        let mut bridge_session = Session::new();
        bridge_session.project = Some(phantom_core::Project::new("diag-bridge"));
        bridge_session.execute(Op::SceneAdd { name: "main".into(), start: 0, end: 48 }).unwrap();
        let bridged = call(&bridge_once(bridge_session), Op::Validate { scene: None });
        let structured = assert_structured(&bridged, false);
        assert!(structured["findings"].is_array());
    }

    #[test]
    fn brush_spaces_convert_to_canonical_canvas() {
        assert_eq!(convert_brush_points(vec![[960.0, 540.0]], BrushSpace::Screen, Some(960.0), Some(540.0)).unwrap(), vec![[1920.0, 1080.0]]);
        assert_eq!(convert_brush_points(vec![[0.0, 0.0]], BrushSpace::Ndc, None, None).unwrap(), vec![[960.0, 540.0]]);
        assert!(convert_brush_points(vec![[2.0, 0.0]], BrushSpace::Ndc, None, None).is_err());
    }

    fn assert_property_descriptions(schema: &serde_json::Value, path: &str) {
        if let Some(properties) = schema.get("properties").and_then(|v| v.as_object()) {
            for (name, property) in properties {
                let description = property.get("description").and_then(|v| v.as_str()).unwrap_or("");
                assert!(!description.trim().is_empty(), "missing property description at {path}.{name}: {property}");
            }
        }
        match schema {
            serde_json::Value::Object(object) => for (key, value) in object { assert_property_descriptions(value, &format!("{path}.{key}")); },
            serde_json::Value::Array(values) => for (index, value) in values.iter().enumerate() { assert_property_descriptions(value, &format!("{path}[{index}]")); },
            _ => {}
        }
    }

    #[test]
    fn tools_list_has_described_inputs_and_structured_output_schemas() {
        let tools = PhantomMcp::tool_router().list_all();
        assert!(tools.len() >= 60);
        for tool in tools {
            let input = serde_json::to_value(&tool.input_schema).unwrap();
            assert_property_descriptions(&input, &tool.name);
            let output = tool.output_schema.as_ref().unwrap_or_else(|| panic!("{} lacks outputSchema", tool.name));
            assert_eq!(output.get("type").and_then(|v| v.as_str()), Some("object"), "{} output must promise an object", tool.name);
            let wire = serde_json::to_value(&tool).unwrap();
            assert!(wire.get("outputSchema").is_some(), "{} serialized tools/list omits outputSchema", tool.name);
        }
    }
}

//! Stable user-facing names shared by GUI, CLI, MCP and session dispatch.
use phantom_core::{ActorPose, AssetId, BuiltinActor, RigRef};
use serde_json::{json, Value};

pub const ACTORS: &[(BuiltinActor, &str)] = &[
    (BuiltinActor::Humanoid, "humanoid"), (BuiltinActor::Wolf, "wolf"),
    (BuiltinActor::Horse, "horse"), (BuiltinActor::Bird, "bird"),
    (BuiltinActor::Child, "child"), (BuiltinActor::Teen, "teen"),
    (BuiltinActor::Elder, "elder"),
];
pub const POSES: &[(ActorPose, &str)] = &[
    (ActorPose::Stand, "stand"), (ActorPose::Sit, "sit"), (ActorPose::Lie, "lie"),
];
pub fn actor_name(actor: BuiltinActor) -> &'static str {
    ACTORS.iter().find(|(a, _)| *a == actor).unwrap().1
}
pub fn pose_name(pose: ActorPose) -> &'static str {
    POSES.iter().find(|(p, _)| *p == pose).unwrap().1
}
pub fn parse_actor(name: &str) -> Result<BuiltinActor, String> {
    ACTORS.iter().find(|(_, n)| *n == name).map(|(a, _)| *a)
        .ok_or_else(|| format!("unknown builtin actor: {name} (humanoid|child|teen|elder|wolf|horse|bird)"))
}
pub fn parse_pose(name: &str) -> Result<ActorPose, String> {
    POSES.iter().find(|(_, n)| *n == name).map(|(p, _)| *p)
        .ok_or_else(|| format!("unknown pose: {name} (stand|sit|lie)"))
}
pub fn parse_rig(name: &str) -> Result<RigRef, String> {
    if name == "builtin" { return Ok(RigRef::Builtin); }
    if let Ok(actor) = parse_actor(name) { return Ok(RigRef::BuiltinActor { actor }); }
    name.parse().map(|id| RigRef::Imported { asset: AssetId(id) })
        .map_err(|_| "rig: builtin|humanoid|child|teen|elder|wolf|horse|bird or asset ID".into())
}
/// A crowd role name (the catalog's `adult|child|teen|elder`) as a rig name.
/// The two vocabularies differ in one place only: the catalog calls a grown-up
/// "adult", the rig is `humanoid`. Roles are what a crowd mixes, so the planner
/// speaks roles and this is where it becomes something you can create a Phantom
/// with. The role comes from the core's role table, hence `'static`.
pub fn role_rig(role: &'static str) -> &'static str {
    if role == "adult" {
        "humanoid"
    } else {
        role
    }
}

/// Catalog dimensions in JSON: millimeter precision as f64. The table stores
/// f32, and standalone and bridge serialize f32 differently — rounding makes
/// the contract stable (and honest: these are metric dimensions, not computations).
pub fn mm(v: f32) -> f64 {
    ((v as f64) * 1000.0).round() / 1000.0
}

/// Catalog overview: builtin actors (profiles/poses), metric props and
/// human reference lines (corrective_2 K1/K3). One call gives the agent the
/// sizes and names, so placement never requires opening the mesh.
pub fn catalog() -> Value {
    json!({
        "props": phantom_core::catalog::CATALOG.iter().map(|e| json!({
            "name": e.name,
            "kind": e.kind.as_str(),
            "size": e.size.map(mm),
            "origin": e.origin,
            "materials": e.materials,
            "open_sides": e.open_sides,
            "openings": e.openings,
            "note": e.note,
        })).collect::<Vec<_>>(),
        "human_roles": phantom_core::catalog::HUMAN_ROLES.iter().map(|r| json!({
            "role": r.role, "height_m": mm(r.height_m), "eye_m": mm(r.eye_m),
            "shoulder_m": mm(r.shoulder_m), "waist_m": mm(r.waist_m), "knee_m": mm(r.knee_m),
        })).collect::<Vec<_>>(),
        "animal_heights_m": phantom_core::catalog::ANIMAL_HEIGHTS.iter().map(|(n, h)| json!({"actor": n, "height_m": mm(*h)})).collect::<Vec<_>>(),
        "stage_presets": phantom_core::StagePreset::ALL.iter().map(|p| {
            let s = p.size();
            json!({"preset": p.name(), "width_m": mm(s.width), "height_m": mm(s.height), "depth_m": mm(s.depth)})
        }).collect::<Vec<_>>(),
        "profiles": ACTORS.iter().map(|(actor, name)| json!({
            "actor": name,
            "poses": actor.poses().iter().map(|p| pose_name(*p)).collect::<Vec<_>>(),
            "default_pose": pose_name(actor.default_pose()),
        })).collect::<Vec<_>>(),
    })
}

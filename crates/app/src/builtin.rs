//! Stable user-facing names shared by GUI, CLI, MCP and session dispatch.
use phantom_core::{Action, AssetId, BuiltinActor, RigRef};
use serde_json::{json, Value};

pub const ACTORS: &[(BuiltinActor, &str)] = &[
    (BuiltinActor::Humanoid, "humanoid"), (BuiltinActor::Wolf, "wolf"),
    (BuiltinActor::Horse, "horse"), (BuiltinActor::Bird, "bird"),
];
pub const ACTIONS: &[(Action, &str)] = &[
    (Action::Idle, "idle"), (Action::Walk, "walk"), (Action::Jump, "jump"),
    (Action::Sit, "sit"), (Action::Lie, "lie"), (Action::Sneak, "sneak"),
    (Action::Trot, "trot"), (Action::Gallop, "gallop"),
    (Action::Fly, "fly"), (Action::Perch, "perch"),
];
pub fn actor_name(actor: BuiltinActor) -> &'static str {
    ACTORS.iter().find(|(a, _)| *a == actor).unwrap().1
}
pub fn action_name(action: Action) -> &'static str {
    ACTIONS.iter().find(|(a, _)| *a == action).unwrap().1
}
pub fn parse_actor(name: &str) -> Result<BuiltinActor, String> {
    ACTORS.iter().find(|(_, n)| *n == name).map(|(a, _)| *a)
        .ok_or_else(|| format!("unknown builtin actor: {name} (humanoid|wolf|horse|bird)"))
}
pub fn parse_action(name: &str) -> Result<Action, String> {
    ACTIONS.iter().find(|(_, n)| *n == name).map(|(a, _)| *a)
        .ok_or_else(|| format!("unknown action: {name}; see builtin catalog"))
}
pub fn parse_rig(name: &str) -> Result<RigRef, String> {
    if name == "builtin" { return Ok(RigRef::Builtin); }
    if let Ok(actor) = parse_actor(name) { return Ok(RigRef::BuiltinActor { actor }); }
    name.parse().map(|id| RigRef::Imported { asset: AssetId(id) })
        .map_err(|_| "rig: builtin|humanoid|wolf|horse|bird or asset ID".into())
}
/// Catalog dimensions in JSON: millimeter precision as f64. The table stores
/// f32, and standalone and bridge serialize f32 differently — rounding makes
/// the contract stable (and honest: these are metric dimensions, not computations).
pub fn mm(v: f32) -> f64 {
    ((v as f64) * 1000.0).round() / 1000.0
}

/// Catalog overview: builtin actors (profiles/actions), metric props and
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
        "profiles": ACTORS.iter().map(|(actor, name)| json!({
            "actor": name,
            "actions": actor.actions().iter().map(|a| action_name(*a)).collect::<Vec<_>>(),
            "default_action": action_name(actor.default_action()),
        })).collect::<Vec<_>>(),
    })
}

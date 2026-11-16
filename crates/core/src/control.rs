//! Live Phantom control model: fixed 1/120 s step,
//! automatic idle/walk/jump transitions. Pure core functions — the same
//! for GUI, recording and (later) MCP.

use glam::Vec3;

use crate::project::Action;
use crate::rig::JUMP_SECS;

/// Live control integration rate.
pub const CONTROL_HZ: u32 = 120;
/// Controller walk speed, m/s.
pub const WALK_SPEED: f32 = 1.4;

/// Input for one step: desired direction (normalized internally) and jump.
#[derive(Clone, Copy, Debug, Default)]
pub struct ControlInput {
    pub move_x: f32,
    pub move_z: f32,
    pub jump: bool,
}

/// Phantom controller state.
#[derive(Clone, Copy, Debug)]
pub struct ControlState {
    /// Root position (y is always 0 — the floor; the jump arc lives in the clip).
    pub position: Vec3,
    /// Facing about +Y in radians, retained while stationary.
    pub yaw: f32,
    pub action: Action,
    /// Current action time in seconds (for the clip phase).
    pub action_t: f32,
}

impl Default for ControlState {
    fn default() -> Self {
        Self { yaw: 0.0, position: Vec3::ZERO, action: Action::Idle, action_t: 0.0 }
    }
}

/// One control step. `dt` is 1/CONTROL_HZ.
pub fn step_control(state: &mut ControlState, input: &ControlInput, dt: f32) {
    let wish = Vec3::new(input.move_x, 0.0, input.move_z);
    let wish = if wish.length_squared() > 1.0 { wish.normalize() } else { wish };
    let moving = wish.length_squared() > 1e-6;

    if moving { state.yaw = wish.x.atan2(wish.z); }
    state.position += wish * WALK_SPEED * dt;
    state.position.y = 0.0;

    // Jump: not interrupted until the arc finishes, then it auto-returns.
    if state.action == Action::Jump {
        state.action_t += dt;
        if state.action_t >= JUMP_SECS {
            state.action = if moving { Action::Walk } else { Action::Idle };
            state.action_t = 0.0;
        }
        return;
    }
    if input.jump {
        state.action = Action::Jump;
        state.action_t = 0.0;
        return;
    }
    let next = if moving { Action::Walk } else { Action::Idle };
    if next != state.action {
        state.action = next;
        state.action_t = 0.0;
    } else {
        state.action_t += dt;
    }
}

/// Live control pose at the current step (for the viewport and recording).
pub fn control_pose(state: &ControlState) -> crate::rig::Pose {
    crate::rig::sample_action(state.action, state.action_t)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / CONTROL_HZ as f32;

    #[test]
    fn walk_moves_and_stops() {
        let mut s = ControlState::default();
        let input = ControlInput { move_x: 1.0, move_z: 0.0, jump: false };
        for _ in 0..CONTROL_HZ {
            step_control(&mut s, &input, DT);
        }
        assert_eq!(s.action, Action::Walk);
        assert!((s.position.x - WALK_SPEED).abs() < 1e-3, "meters per second * speed");
        assert_eq!(s.position.y, 0.0);
        step_control(&mut s, &ControlInput::default(), DT);
        assert_eq!(s.action, Action::Idle);
    }

    #[test]
    fn jump_completes_and_returns() {
        let mut s = ControlState::default();
        step_control(&mut s, &ControlInput { jump: true, ..Default::default() }, DT);
        assert_eq!(s.action, Action::Jump);
        // A repeated jump input does not restart the arc.
        let t0 = s.action_t;
        step_control(&mut s, &ControlInput { jump: true, ..Default::default() }, DT);
        assert!(s.action_t > t0);
        for _ in 0..CONTROL_HZ {
            step_control(&mut s, &ControlInput::default(), DT);
        }
        assert_eq!(s.action, Action::Idle);
    }

    #[test]
    fn diagonal_is_normalized() {
        let mut s = ControlState::default();
        let input = ControlInput { move_x: 1.0, move_z: 1.0, jump: false };
        for _ in 0..CONTROL_HZ {
            step_control(&mut s, &input, DT);
        }
        let len = (s.position.x.powi(2) + s.position.z.powi(2)).sqrt();
        assert!((len - WALK_SPEED).abs() < 1e-3);
    }
}

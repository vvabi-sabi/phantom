//! Shared project lock used by standalone MCP sessions and the GUI bridge.
//!
//! The lock file is an ownership record, not merely a marker. Acquisition is
//! always an atomic `create_new`; stale records are removed only after a second
//! byte-for-byte observation, and normal release unlinks only the token owned by
//! the guard.

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

pub const LOCK_FILE: &str = ".phantom.lock";
const LOCK_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LockRecord {
    pub version: u32,
    pub pid: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub process_start: Option<String>,
    pub token: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub socket: Option<PathBuf>,
    pub heartbeat_unix_ms: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OwnerStatus {
    Live,
    Dead,
    Ambiguous,
}

impl fmt::Display for OwnerStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Live => f.write_str("live"),
            Self::Dead => f.write_str("dead"),
            Self::Ambiguous => f.write_str("status ambiguous"),
        }
    }
}

/// Safe diagnostic view of an observed lock. The ownership token itself is not
/// exposed, only whether a token was present.
#[derive(Clone, Debug, Serialize)]
pub struct LockOwner {
    pub pid: Option<u32>,
    pub status: OwnerStatus,
    pub process_start: Option<String>,
    pub version: Option<u32>,
    pub legacy: bool,
    pub token_present: bool,
    pub socket: Option<PathBuf>,
    pub socket_present: bool,
    pub heartbeat_unix_ms: Option<u64>,
    pub mtime_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl LockOwner {
    pub fn summary(&self) -> String {
        let pid = self.pid.map(|p| p.to_string()).unwrap_or_else(|| "unknown".into());
        format!("PID {pid} ({})", self.status)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LockError {
    #[error("io: {0}")]
    Io(#[from] io::Error),
    #[error("lock {path} is held by {owner_summary}")]
    Held {
        path: PathBuf,
        owner: LockOwner,
        owner_summary: String,
    },
    #[error("lock {path} changed during safe release; retry the operation")]
    Changed { path: PathBuf },
    #[error("lock {} not found", .0.display())]
    NotFound(PathBuf),
}

impl LockError {
    fn held(path: PathBuf, owner: LockOwner) -> Self {
        let owner_summary = owner.summary();
        Self::Held { path, owner, owner_summary }
    }
}

#[derive(Debug)]
struct Observation {
    owner: LockOwner,
    raw: Vec<u8>,
}

#[derive(Debug)]
pub struct LockAcquisition {
    pub guard: LockGuard,
    /// Present when a definitely dead lock, or an ambiguous lock under force,
    /// was conservatively displaced.
    pub reclaimed: Option<LockOwner>,
}

#[derive(Debug)]
pub struct LockGuard {
    dir: PathBuf,
    path: PathBuf,
    token: String,
}

impl LockGuard {
    pub fn acquire(dir: &Path, socket: Option<PathBuf>, force: bool) -> Result<LockAcquisition, LockError> {
        let dir = dir.canonicalize()?;
        let path = dir.join(LOCK_FILE);
        let mut reclaimed = None;

        // A changed observation means another contender made progress. Retrying
        // preserves create_new as the sole acquisition primitive.
        for _ in 0..16 {
            let record = LockRecord {
                version: LOCK_VERSION,
                pid: std::process::id(),
                process_start: process_start(std::process::id()),
                token: ownership_token(),
                socket: socket.clone(),
                heartbeat_unix_ms: unix_ms(SystemTime::now()),
            };
            match create_record(&path, &record) {
                Ok(()) => {
                    return Ok(LockAcquisition {
                        guard: LockGuard { dir, path, token: record.token },
                        reclaimed,
                    });
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
                Err(e) => return Err(e.into()),
            }

            let observed = match observe_path(&path) {
                Ok(v) => v,
                Err(LockError::NotFound(_)) => continue,
                Err(e) => return Err(e),
            };
            let reclaimable = observed.owner.status == OwnerStatus::Dead
                || (force && observed.owner.status == OwnerStatus::Ambiguous);
            if !reclaimable {
                return Err(LockError::held(path, observed.owner));
            }
            // Force never overrides a lock known to be live, including another
            // guard in this process. Dead legacy records are safe because /proc
            // has definitely established that their PID no longer exists.
            if remove_observed(&path, &observed.raw)? {
                reclaimed = Some(observed.owner);
            }
        }
        Err(LockError::Changed { path })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Release only if the on-disk record still contains this guard's token.
    /// A replacement lock is never removed.
    pub fn release(&mut self) -> Result<bool, LockError> {
        let observed = match observe_path(&self.path) {
            Ok(v) => v,
            Err(LockError::NotFound(_)) => return Ok(false),
            Err(e) => return Err(e),
        };
        let token = parsed_value(&observed.raw)
            .and_then(|v| v.get("token").and_then(|v| v.as_str()).map(str::to_owned));
        if token.as_deref() != Some(self.token.as_str()) {
            return Ok(false);
        }
        remove_observed(&self.path, &observed.raw)
    }

    pub fn owner(&self) -> Result<LockOwner, LockError> {
        Ok(observe_path(&self.path)?.owner)
    }
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        let _ = self.release();
    }
}

/// Inspect a project lock for bridge diagnostics.
pub fn inspect(dir: &Path) -> Result<LockOwner, LockError> {
    let path = dir.join(LOCK_FILE);
    Ok(observe_path(&path)?.owner)
}

/// Explicit administrative release. Live locks are never removed, even with
/// force. Dead or ambiguous records require force so the observed owner can be
/// returned to the caller as warning/audit data.
pub fn release_external(dir: &Path, force: bool) -> Result<(bool, Option<LockOwner>), LockError> {
    let path = dir.join(LOCK_FILE);
    let observed = match observe_path(&path) {
        Ok(v) => v,
        Err(LockError::NotFound(_)) => return Ok((false, None)),
        Err(e) => return Err(e),
    };
    if !force || observed.owner.status == OwnerStatus::Live {
        return Err(LockError::held(path, observed.owner));
    }
    let owner = observed.owner;
    if remove_observed(&path, &observed.raw)? {
        Ok((true, Some(owner)))
    } else {
        Err(LockError::Changed { path })
    }
}

fn create_record(path: &Path, record: &LockRecord) -> io::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let bytes = serde_json::to_vec(record).map_err(io::Error::other)?;
    if let Err(e) = file.write_all(&bytes).and_then(|_| file.sync_all()) {
        // No other owner can replace a path while our create_new file exists.
        let _ = std::fs::remove_file(path);
        return Err(e);
    }
    Ok(())
}

fn observe_path(path: &Path) -> Result<Observation, LockError> {
    let raw = match std::fs::read(path) {
        Ok(v) => v,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Err(LockError::NotFound(path.to_path_buf())),
        Err(e) => return Err(e.into()),
    };
    let metadata = std::fs::metadata(path).ok();
    let mtime_unix_ms = metadata.and_then(|m| m.modified().ok()).map(unix_ms);
    let value = parsed_value(&raw);
    let pid = value
        .as_ref()
        .and_then(|v| v.get("pid"))
        .and_then(|v| v.as_u64())
        .and_then(|v| u32::try_from(v).ok());
    let process_start_recorded = value
        .as_ref()
        .and_then(|v| v.get("process_start"))
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let version = value
        .as_ref()
        .and_then(|v| v.get("version"))
        .and_then(|v| v.as_u64())
        .and_then(|v| u32::try_from(v).ok());
    let token_present = value
        .as_ref()
        .and_then(|v| v.get("token"))
        .and_then(|v| v.as_str())
        .is_some_and(|v| !v.is_empty());
    let socket = value
        .as_ref()
        .and_then(|v| v.get("socket"))
        .and_then(|v| v.as_str())
        .map(PathBuf::from);
    let heartbeat_unix_ms = value
        .as_ref()
        .and_then(|v| v.get("heartbeat_unix_ms"))
        .and_then(|v| v.as_u64());
    let (status, detail) = match pid {
        Some(pid) => process_status(pid, process_start_recorded.as_deref()),
        None => (OwnerStatus::Ambiguous, Some("lock contains no valid PID".into())),
    };
    let socket_present = socket.as_ref().is_some_and(|p| p.exists());
    Ok(Observation {
        owner: LockOwner {
            pid,
            status,
            process_start: process_start_recorded,
            version,
            legacy: !token_present,
            token_present,
            socket,
            socket_present,
            heartbeat_unix_ms,
            mtime_unix_ms,
            detail: value.is_none().then(|| "lock contains invalid JSON".into()).or(detail),
        },
        raw,
    })
}

fn parsed_value(raw: &[u8]) -> Option<serde_json::Value> {
    serde_json::from_slice(raw).ok()
}

/// A second exact observation prevents a stale inspector from knowingly
/// deleting a replacement record. Acquisition after this removal still uses
/// create_new, so simultaneous reclaimers have exactly one winner.
fn remove_observed(path: &Path, expected: &[u8]) -> Result<bool, LockError> {
    let current = match std::fs::read(path) {
        Ok(v) => v,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(e.into()),
    };
    if current != expected {
        return Ok(false);
    }
    match std::fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e.into()),
    }
}

fn unix_ms(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}

fn process_status(pid: u32, recorded_start: Option<&str>) -> (OwnerStatus, Option<String>) {
    match read_process_start(Path::new("/proc"), pid) {
        Ok(actual) => match recorded_start {
            Some(expected) if expected != actual => (
                OwnerStatus::Dead,
                Some(format!("PID reused: start {actual}, lock records {expected}")),
            ),
            _ => (OwnerStatus::Live, None),
        },
        Err(e) if e.kind() == io::ErrorKind::NotFound => (OwnerStatus::Dead, Some("/proc entry for the PID is missing".into())),
        Err(e) => (OwnerStatus::Ambiguous, Some(format!("cannot verify /proc: {e}"))),
    }
}

fn process_start(pid: u32) -> Option<String> {
    read_process_start(Path::new("/proc"), pid).ok()
}

/// Linux `/proc/<pid>/stat` field 22. Kept independent from the real proc root
/// so parsing and PID-reuse behavior can be tested without process control.
fn read_process_start(proc_root: &Path, pid: u32) -> io::Result<String> {
    let text = std::fs::read_to_string(proc_root.join(pid.to_string()).join("stat"))?;
    let close = text.rfind(')').ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no ')' in stat"))?;
    let fields: Vec<&str> = text[close + 1..].split_whitespace().collect();
    // fields[0] is field 3 (state), therefore field 22 is index 19.
    let start = fields
        .get(19)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no starttime in stat"))?;
    start
        .parse::<u64>()
        .map(|v| v.to_string())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid starttime in stat"))
}

static TOKEN_COUNTER: AtomicU64 = AtomicU64::new(1);

fn ownership_token() -> String {
    let mut random = [0u8; 16];
    if File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut random)).is_ok() {
        return random.iter().map(|b| format!("{b:02x}")).collect();
    }
    // Portable fallback: uniqueness remains guaranteed within this process;
    // time, process identity and stack address add per-process entropy.
    let counter = TOKEN_COUNTER.fetch_add(1, Ordering::Relaxed);
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
    let addr = (&random as *const [u8; 16]) as usize;
    format!("{:032x}{:08x}{:016x}{:x}", now, std::process::id(), counter, addr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "phantom-lock-{name}-{}-{}",
            std::process::id(),
            TOKEN_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn dead_pid() -> u32 {
        // Linux pid_max is normally <= 4M; this value cannot name a process.
        u32::MAX
    }

    #[test]
    fn legacy_and_current_dead_locks_are_reclaimed() {
        for legacy in [true, false] {
            let dir = temp_dir(if legacy { "legacy-dead" } else { "current-dead" });
            let path = dir.join(LOCK_FILE);
            let value = if legacy {
                serde_json::json!({"pid": dead_pid()})
            } else {
                serde_json::json!({
                    "version": 1, "pid": dead_pid(), "process_start": "1",
                    "token": "old", "heartbeat_unix_ms": 1
                })
            };
            std::fs::write(&path, value.to_string()).unwrap();
            let acquired = LockGuard::acquire(&dir, None, false).unwrap();
            assert_eq!(acquired.reclaimed.as_ref().unwrap().status, OwnerStatus::Dead);
            drop(acquired);
            assert!(!path.exists());
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn live_foreign_lock_blocks_even_with_force() {
        let dir = temp_dir("live");
        let start = process_start(std::process::id());
        std::fs::write(
            dir.join(LOCK_FILE),
            serde_json::json!({
                "version": 1, "pid": std::process::id(), "process_start": start,
                "token": "different-guard", "heartbeat_unix_ms": 1
            })
            .to_string(),
        )
        .unwrap();
        for force in [false, true] {
            assert!(matches!(LockGuard::acquire(&dir, None, force), Err(LockError::Held { owner, .. }) if owner.status == OwnerStatus::Live));
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn force_releases_ambiguous_record_and_reports_it() {
        let dir = temp_dir("force-ambiguous");
        std::fs::write(dir.join(LOCK_FILE), b"not-json").unwrap();
        assert!(matches!(release_external(&dir, false), Err(LockError::Held { owner, .. }) if owner.status == OwnerStatus::Ambiguous));
        let (released, owner) = release_external(&dir, true).unwrap();
        assert!(released);
        assert_eq!(owner.unwrap().status, OwnerStatus::Ambiguous);
    }

    #[test]
    fn token_safe_release_does_not_remove_replacement() {
        let dir = temp_dir("token-release");
        let mut guard = LockGuard::acquire(&dir, None, false).unwrap().guard;
        let path = dir.join(LOCK_FILE);
        let mut replacement: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        replacement["token"] = serde_json::json!("replacement");
        std::fs::write(&path, replacement.to_string()).unwrap();
        assert!(!guard.release().unwrap());
        drop(guard);
        assert!(path.exists());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn two_atomic_contenders_have_one_winner() {
        let dir = temp_dir("race");
        let barrier = Arc::new(Barrier::new(3));
        let hold = Arc::new(Barrier::new(2));
        let (tx, rx) = std::sync::mpsc::channel();
        for _ in 0..2 {
            let dir = dir.clone();
            let barrier = Arc::clone(&barrier);
            let hold = Arc::clone(&hold);
            let tx = tx.clone();
            std::thread::spawn(move || {
                barrier.wait();
                match LockGuard::acquire(&dir, None, false) {
                    Ok(a) => {
                        tx.send(true).unwrap();
                        hold.wait();
                        drop(a);
                    }
                    Err(LockError::Held { .. }) => {
                        tx.send(false).unwrap();
                    }
                    Err(e) => panic!("unexpected lock result: {e}"),
                }
            });
        }
        barrier.wait();
        let a = rx.recv().unwrap();
        let b = rx.recv().unwrap();
        assert_ne!(a, b);
        hold.wait();
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn normal_drop_removes_owned_lock() {
        let dir = temp_dir("drop");
        let path = dir.join(LOCK_FILE);
        let acquired = LockGuard::acquire(&dir, None, false).unwrap();
        assert!(path.exists());
        drop(acquired);
        assert!(!path.exists());
    }

    #[test]
    fn proc_start_parser_is_isolated_from_real_proc() {
        let dir = temp_dir("proc");
        let pid = 42;
        std::fs::create_dir_all(dir.join(pid.to_string())).unwrap();
        let mut fields = vec!["0"; 52];
        fields[0] = "S"; // field 3
        fields[19] = "987654"; // field 22
        std::fs::write(dir.join(pid.to_string()).join("stat"), format!("42 (name with spaces) {}", fields.join(" "))).unwrap();
        assert_eq!(read_process_start(&dir, pid).unwrap(), "987654");
    }
}

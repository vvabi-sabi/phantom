//! Bridge protocol between the MCP process and the open editor:
//! newline-JSON over a Unix socket in a private runtime directory.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::project_lock::{self, LockOwner, OwnerStatus};
use crate::session::{ErrorData, Op};

#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("in the editor: {}", .0.message)]
    Remote(ErrorData),
    #[error("project lock file not found — no editor window open?")]
    NoLock,
    #[error("project lock belongs to {owner_summary}; socket={socket_present}")]
    OwnerUnavailable {
        owner: LockOwner,
        owner_summary: String,
        socket_present: bool,
    },
    #[error("project lock contains no socket; owner {owner_summary}")]
    NoSocket { owner: LockOwner, owner_summary: String },
    #[error("lock: {0}")]
    Lock(#[from] project_lock::LockError),
}

impl BridgeError {
    pub fn error_data(&self) -> ErrorData {
        match self {
            Self::Remote(data) => data.clone(),
            Self::Io(e) => ErrorData::new(
                "bridge_unavailable",
                self.to_string(),
                json!({ "kind": format!("{:?}", e.kind()).to_lowercase(), "cause": e.to_string() }),
            ),
            Self::Json(e) => ErrorData::new(
                "bridge_protocol",
                self.to_string(),
                json!({ "cause": e.to_string(), "category": format!("{:?}", e.classify()).to_lowercase() }),
            ),
            Self::NoLock => ErrorData::new(
                "bridge_unavailable",
                self.to_string(),
                json!({ "reason": "lock_not_found" }),
            ),
            Self::OwnerUnavailable { owner, socket_present, .. } => ErrorData::new(
                "bridge_unavailable",
                self.to_string(),
                json!({ "reason": "owner_unavailable", "owner": owner, "socket_present": socket_present }),
            ),
            Self::NoSocket { owner, .. } => ErrorData::new(
                "bridge_unavailable",
                self.to_string(),
                json!({ "reason": "socket_not_registered", "owner": owner }),
            ),
            Self::Lock(error) => crate::session::SessionError::Lock(match error {
                project_lock::LockError::Io(e) => {
                    project_lock::LockError::Io(std::io::Error::new(e.kind(), e.to_string()))
                }
                project_lock::LockError::Held { path, owner, owner_summary } => project_lock::LockError::Held {
                    path: path.clone(), owner: owner.clone(), owner_summary: owner_summary.clone(),
                },
                project_lock::LockError::Changed { path } => project_lock::LockError::Changed { path: path.clone() },
                project_lock::LockError::NotFound(path) => project_lock::LockError::NotFound(path.clone()),
            })
            .error_data(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BridgeResponse {
    pub id: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_data: Option<ErrorData>,
}

impl BridgeResponse {
    pub fn success(id: u64, result: Value) -> Self {
        Self { id, result: Some(result), error_data: None }
    }

    pub fn failure(id: u64, error_data: ErrorData) -> Self {
        Self { id, result: None, error_data: Some(error_data) }
    }
}

/// Private socket directory (0700).
pub fn runtime_dir() -> Result<PathBuf, BridgeError> {
    let base = std::env::var("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join(format!("phantom-{}", std::env::var("USER").unwrap_or_else(|_| "u".into()))));
    let dir = base.join("phantom-animate");
    std::fs::create_dir_all(&dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(dir)
}

/// Socket path for a project (from the canonical directory path).
pub fn socket_path(project_dir: &Path) -> Result<PathBuf, BridgeError> {
    let canon = project_dir.canonicalize().unwrap_or_else(|_| project_dir.to_path_buf());
    let hash = format!("{:x}", fxhash(canon.to_string_lossy().as_bytes()));
    Ok(runtime_dir()?.join(format!("project-{hash}.sock")))
}

fn fxhash(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in bytes {
        h = (h ^ *b as u64).wrapping_mul(0x100000001b3);
    }
    h
}

/// Bridge lock diagnostics returned before connecting to its socket.
#[derive(Clone, Debug)]
pub struct BridgeLock {
    pub socket: PathBuf,
    pub owner: LockOwner,
}

/// Read the shared lock record, including process/socket liveness diagnostics.
pub fn read_lock(project_dir: &Path) -> Result<BridgeLock, BridgeError> {
    let owner = match project_lock::inspect(project_dir) {
        Err(project_lock::LockError::NotFound(_)) => return Err(BridgeError::NoLock),
        other => other?,
    };
    let owner_summary = owner.summary();
    let socket = owner
        .socket
        .clone()
        .ok_or_else(|| BridgeError::NoSocket { owner: owner.clone(), owner_summary })?;
    if owner.status != OwnerStatus::Live || !owner.socket_present {
        return Err(BridgeError::OwnerUnavailable {
            socket_present: owner.socket_present,
            owner_summary: owner.summary(),
            owner,
        });
    }
    Ok(BridgeLock { socket, owner })
}

/// Request from a client; the answer goes back through the oneshot.
pub struct BridgeRequest {
    pub id: u64,
    pub op: Op,
    pub respond: std::sync::mpsc::Sender<Value>,
}

/// Starts the listener thread; requests arrive on the channel (the editor's
/// main thread executes them).
pub fn host(socket: &Path) -> Result<std::sync::mpsc::Receiver<BridgeRequest>, BridgeError> {
    let _ = std::fs::remove_file(socket);
    let listener = UnixListener::bind(socket)?;
    let (tx, rx) = std::sync::mpsc::channel::<BridgeRequest>();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            let tx = tx.clone();
            std::thread::spawn(move || handle_conn(stream, tx));
        }
    });
    Ok(rx)
}

fn handle_conn(stream: UnixStream, tx: std::sync::mpsc::Sender<BridgeRequest>) {
    let mut writer = match stream.try_clone() {
        Ok(w) => w,
        Err(_) => return,
    };
    let reader = BufReader::new(stream);
    for line in reader.lines() {
        let Ok(line) = line else { break };
        let parsed: Result<Value, _> = serde_json::from_str(&line);
        let response = match parsed {
            Ok(v) => {
                let id = v["id"].as_u64().unwrap_or(0);
                match serde_json::from_value::<Op>(v["op"].clone()) {
                    Ok(op) => {
                        let (rtx, rrx) = std::sync::mpsc::channel();
                        if tx.send(BridgeRequest { id, op, respond: rtx }).is_ok() {
                            rrx.recv().unwrap_or_else(|_| {
                                serde_json::to_value(BridgeResponse::failure(
                                    id,
                                    ErrorData::new(
                                        "bridge_unavailable",
                                        "the editor closed the connection",
                                        json!({ "reason": "editor_disconnected" }),
                                    ),
                                ))
                                .expect("bridge response")
                            })
                        } else {
                            break;
                        }
                    }
                    Err(e) => serde_json::to_value(BridgeResponse::failure(
                        id,
                        ErrorData::new(
                            "invalid_input",
                            "invalid bridge operation",
                            json!({ "cause": e.to_string() }),
                        ),
                    ))
                    .expect("bridge response"),
                }
            }
            Err(e) => serde_json::to_value(BridgeResponse::failure(
                0,
                ErrorData::new(
                    "invalid_input",
                    "invalid bridge JSON",
                    json!({ "cause": e.to_string() }),
                ),
            ))
            .expect("bridge response"),
        };
        if writer.write_all(response.to_string().as_bytes()).and_then(|_| writer.write_all(b"\n")).is_err() {
            break;
        }
    }
}

/// Bridge client (the `mcp --bridge` process).
pub struct BridgeClient {
    stream: UnixStream,
    next_id: u64,
}

impl BridgeClient {
    pub fn connect(project_dir: &Path) -> Result<Self, BridgeError> {
        let BridgeLock { socket, owner: _owner } = read_lock(project_dir)?;
        let stream = UnixStream::connect(&socket)?;
        Ok(Self { stream, next_id: 1 })
    }

    pub fn execute(&mut self, op: &Op) -> Result<Value, BridgeError> {
        let id = self.next_id;
        self.next_id += 1;
        let line = json!({ "id": id, "op": op }).to_string();
        self.stream.write_all(line.as_bytes())?;
        self.stream.write_all(b"\n")?;
        let mut reader = BufReader::new(&self.stream);
        let mut buf = String::new();
        reader.read_line(&mut buf)?;
        let response: BridgeResponse = serde_json::from_str(&buf)?;
        if response.id != id {
            return Err(BridgeError::Remote(ErrorData::new(
                "bridge_protocol",
                "bridge returned a response with the wrong id",
                json!({ "expected_id": id, "received_id": response.id }),
            )));
        }
        match (response.result, response.error_data) {
            (Some(result), None) => Ok(result),
            (None, Some(error)) => Err(BridgeError::Remote(error)),
            (result, error_data) => Err(BridgeError::Remote(ErrorData::new(
                "bridge_protocol",
                "bridge returned an ambiguous response",
                json!({ "has_result": result.is_some(), "has_error_data": error_data.is_some() }),
            ))),
        }
    }

    #[cfg(test)]
    pub(crate) fn from_stream(stream: UnixStream) -> Self {
        Self { stream, next_id: 1 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(1);

    #[test]
    fn read_lock_returns_owner_and_socket_diagnostics() {
        let dir = std::env::temp_dir().join(format!(
            "phantom-bridge-lock-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let socket = dir.join("bridge.sock");
        std::fs::write(&socket, b"diagnostic-placeholder").unwrap();
        let acquired = project_lock::LockGuard::acquire(&dir, Some(socket.clone()), false).unwrap();

        let lock = read_lock(&dir).unwrap();
        assert_eq!(lock.socket, socket);
        assert_eq!(lock.owner.pid, Some(std::process::id()));
        assert_eq!(lock.owner.status, OwnerStatus::Live);
        assert!(lock.owner.socket_present);

        std::fs::remove_file(&socket).unwrap();
        let err = read_lock(&dir).unwrap_err().to_string();
        assert!(err.contains(&std::process::id().to_string()), "{err}");
        assert!(err.contains("live"), "{err}");
        assert!(err.contains("socket=false"), "{err}");
        drop(acquired);
        let _ = std::fs::remove_dir_all(dir);
    }
}

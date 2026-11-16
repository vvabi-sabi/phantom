//! File-based project storage: a directory with a JSON manifest and assets.
//! Saving is atomic: assets first, then the manifest via a temp file and a
//! rename — a crash never leaves a manifest pointing at unwritten files.

use std::path::{Component, Path, PathBuf};

use phantom_core::{Project, SCHEMA_VERSION};

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("manifest not found: {path} — not a project directory; create one with: phantom-animate new")]
    NoManifest { path: PathBuf, #[source] source: std::io::Error },
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("manifest has a newer schema version {found} (supported: {SCHEMA_VERSION})")]
    NewerSchema { found: u32 },
    #[error("asset path escapes the project root: {0}")]
    PathEscape(PathBuf),
}

pub const MANIFEST: &str = "manifest.json";
pub const ASSETS_DIR: &str = "assets";

pub fn create_project(dir: &Path, name: &str) -> Result<Project, StorageError> {
    std::fs::create_dir_all(dir.join(ASSETS_DIR))?;
    let project = Project::new(name);
    save_project(dir, &project)?;
    Ok(project)
}

pub fn load_project(dir: &Path) -> Result<Project, StorageError> {
    let manifest = dir.join(MANIFEST);
    let text = std::fs::read_to_string(&manifest).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            StorageError::NoManifest { path: manifest.clone(), source: e }
        } else {
            StorageError::Io(e)
        }
    })?;
    // Check the version before a full parse: an unknown newer schema is
    // neither opened nor, worst case, overwritten.
    let head: serde_json::Value = serde_json::from_str(&text)?;
    let version = head["schema_version"].as_u64().unwrap_or(0) as u32;
    if version > SCHEMA_VERSION {
        return Err(StorageError::NewerSchema { found: version });
    }
    let project: Project = serde_json::from_str(&text)?;
    // Asset references must not escape the project root.
    for asset in project.assets.values() {
        check_relative(&asset.path)?;
    }
    Ok(project)
}

pub fn save_project(dir: &Path, project: &Project) -> Result<(), StorageError> {
    let text = serde_json::to_string_pretty(project)?;
    let tmp = dir.join(MANIFEST).with_extension("json.tmp");
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, dir.join(MANIFEST))?;
    Ok(())
}

/// Copy an external asset file into the project; returns the relative path.
pub fn import_file(dir: &Path, source: &Path) -> Result<PathBuf, StorageError> {
    let name = source
        .file_name()
        .ok_or_else(|| StorageError::PathEscape(source.to_path_buf()))?;
    let mut dest = PathBuf::from(ASSETS_DIR).join(name);
    // Name conflicts: suffix -1, -2, ...
    for i in 1.. {
        if !dir.join(&dest).exists() {
            break;
        }
        let stem = Path::new(name).file_stem().and_then(|s| s.to_str()).unwrap_or("asset");
        let ext = Path::new(name).extension().and_then(|s| s.to_str()).unwrap_or("");
        dest = PathBuf::from(ASSETS_DIR).join(format!("{stem}-{i}.{ext}"));
    }
    std::fs::copy(source, dir.join(&dest))?;
    Ok(dest)
}

/// Reject absolute paths and `..`.
fn check_relative(path: &Path) -> Result<(), StorageError> {
    if path.is_absolute()
        || path.components().any(|c| !matches!(c, Component::Normal(_) | Component::CurDir))
    {
        return Err(StorageError::PathEscape(path.to_path_buf()));
    }
    Ok(())
}

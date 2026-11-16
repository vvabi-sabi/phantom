//! Film: ordered clip references to scene ranges, straight cuts.

use serde::{Deserialize, Serialize};

use crate::id::{SceneId, ShotId};
use crate::time::{Frame, FrameRange};

/// Film clip: a reference to a scene range or to a scene shot.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FilmClip {
    pub id: ClipId,
    pub scene: SceneId,
    /// Scene frame range [start, end); when `shot = Some` it matches the
    /// shot's range (validated in the editing command).
    pub range: FrameRange,
    /// Scene shot; `None` — a legacy reference to a scene range. Audio
    /// maps through `range` in both cases: sound belongs to the scene and
    /// is not duplicated.
    #[serde(default)]
    pub shot: Option<ShotId>,
}

/// Film length in frames.
pub fn film_len(film: &[FilmClip]) -> Frame {
    film.iter().map(|c| c.range.len()).sum()
}

/// Single time-mapping function from film time to scene time.
/// Applies to both picture and sound. A shot clip maps into its own scene's
/// range — the scene's audio is not duplicated.
pub fn film_map(film: &[FilmClip], film_frame: Frame) -> Option<(SceneId, Frame)> {
    film_map_clip(film, film_frame).map(|(clip, frame)| (clip.scene, frame))
}

/// Mapping from film time to (clip, scene frame): gives access to the
/// clip's shot camera when rendering.
pub fn film_map_clip(film: &[FilmClip], film_frame: Frame) -> Option<(&FilmClip, Frame)> {
    if film_frame < 0 {
        return None;
    }
    let mut offset = 0;
    for clip in film {
        let len = clip.range.len();
        if film_frame < offset + len {
            return Some((clip, clip.range.start + (film_frame - offset)));
        }
        offset += len;
    }
    None
}

/// Film clip identifier.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct ClipId(pub u64);

impl std::fmt::Display for ClipId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u64> for ClipId {
    fn from(v: u64) -> Self {
        Self(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clip(id: u64, scene: u64, start: Frame, end: Frame) -> FilmClip {
        FilmClip { id: ClipId(id), scene: SceneId(scene), range: FrameRange::new(start, end).unwrap(), shot: None }
    }

    #[test]
    fn map_walks_cuts_without_drift() {
        let film = vec![clip(1, 10, 24, 72), clip(2, 11, 0, 10), clip(3, 10, 0, 24)];
        assert_eq!(film_len(&film), 48 + 10 + 24);
        assert_eq!(film_map(&film, 0), Some((SceneId(10), 24)));
        assert_eq!(film_map(&film, 47), Some((SceneId(10), 71)));
        assert_eq!(film_map(&film, 48), Some((SceneId(11), 0)));
        assert_eq!(film_map(&film, 57), Some((SceneId(11), 9)));
        assert_eq!(film_map(&film, 58), Some((SceneId(10), 0)));
        assert_eq!(film_map(&film, 81), Some((SceneId(10), 23)));
        assert_eq!(film_map(&film, 82), None);
        assert_eq!(film_map(&film, -1), None);
    }
}

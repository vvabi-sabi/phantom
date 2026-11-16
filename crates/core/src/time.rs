//! Editing time: integer frames at 24 fps, half-open ranges `[start, end)`.

pub const FPS: u32 = 24;

/// Frame number on the timeline.
pub type Frame = i64;

/// Half-open frame range `[start, end)`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub struct FrameRange {
    pub start: Frame,
    pub end: Frame,
}

impl FrameRange {
    pub fn new(start: Frame, end: Frame) -> Result<Self, crate::CoreError> {
        if end <= start {
            return Err(crate::CoreError::InvalidRange { start, end });
        }
        Ok(Self { start, end })
    }

    pub fn len(&self) -> Frame {
        self.end - self.start
    }

    pub fn contains(&self, frame: Frame) -> bool {
        frame >= self.start && frame < self.end
    }

    /// Clamps a frame into the range (for seek).
    pub fn clamp(&self, frame: Frame) -> Frame {
        frame.clamp(self.start, self.end - 1)
    }
}

/// Quantization "on twos": the pose updates 12 times/s, anchored to the scene start.
pub fn on_twos(frame: Frame, scene_start: Frame) -> Frame {
    scene_start + (frame - scene_start).div_euclid(2) * 2
}

/// Frame → seconds.
pub fn frame_to_secs(frame: Frame) -> f64 {
    frame as f64 / FPS as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_requires_positive_length() {
        assert!(FrameRange::new(0, 10).is_ok());
        assert!(FrameRange::new(10, 10).is_err());
        assert!(FrameRange::new(10, 0).is_err());
    }

    #[test]
    fn twos_anchor_to_scene_start() {
        assert_eq!(on_twos(0, 0), 0);
        assert_eq!(on_twos(1, 0), 0);
        assert_eq!(on_twos(2, 0), 2);
        assert_eq!(on_twos(5, 1), 5);
        assert_eq!(on_twos(6, 1), 5);
        assert_eq!(on_twos(7, 1), 7);
    }
}

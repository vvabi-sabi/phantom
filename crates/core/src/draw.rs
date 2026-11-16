//! Per-frame 2D layers: raster layer exposures in canvas coordinates.
//! The document stores strokes, not pixels; the raster is derived runtime data.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::time::Frame;

/// 2D layer identifier.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct LayerId(pub u64);

impl std::fmt::Display for LayerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u64> for LayerId {
    fn from(v: u64) -> Self {
        Self(v)
    }
}

/// Default layer canvas width/height (scaled to the output resolution).
pub const DEFAULT_CANVAS: (u32, u32) = (1920, 1080);

/// One brush or eraser stroke.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stroke {
    /// Points in canvas coordinates [0..w]×[0..h].
    pub points: Vec<[f32; 2]>,
    /// RGBA 0..1.
    pub color: [f32; 4],
    /// Width in canvas pixels.
    pub width: f32,
    /// Eraser: subtracts from existing strokes (destination-out).
    pub erase: bool,
}

/// Drawing on one frame: held until the next exposure.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Exposure {
    pub strokes: Vec<Stroke>,
}

/// Raster 2D layer drawn on top of the frame.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Layer2D {
    pub id: LayerId,
    pub name: String,
    /// Canvas size in pixels.
    pub canvas: (u32, u32),
    pub exposures: BTreeMap<Frame, Exposure>,
}

impl Layer2D {
    pub fn new(id: LayerId, name: impl Into<String>) -> Self {
        Self { id, name: name.into(), canvas: DEFAULT_CANVAS, exposures: BTreeMap::new() }
    }

    /// The exposure in effect at a frame: the last one at or before the frame.
    pub fn exposure_for_frame(&self, frame: Frame) -> Option<(&Frame, &Exposure)> {
        self.exposures.range(..=frame).next_back()
    }

    /// Onion skin neighbours: (previous, next) relative to the frame.
    pub fn onion(&self, frame: Frame) -> (Option<Frame>, Option<Frame>) {
        let prev = self.exposures.range(..frame).next_back().map(|(f, _)| *f);
        let next = self.exposures.range(frame + 1..).next().map(|(f, _)| *f);
        (prev, next)
    }
}

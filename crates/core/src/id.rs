//! Stable document identifiers. Not Bevy Entity IDs.

use serde::{Deserialize, Serialize};

macro_rules! define_id {
    ($name:ident) => {
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
        pub struct $name(pub u64);

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

define_id!(SceneId);
define_id!(PhantomId);
define_id!(AssetId);
define_id!(ObjectId);
define_id!(ShotId);
define_id!(ClipUseId);
define_id!(LightId);
define_id!(VfxId);

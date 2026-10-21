//! What the gateway does with a request Rust core does not serve.
//!
//! A core decline travels up as an error and rides on the error response as a [`Decline`]
//! extension. [`replay`] is the only place that acts on it: it hands the request, as the
//! client sent it, to the [`Fallback`].

mod forward;
mod replay;

pub use forward::{Forward, InvalidUpstream};
pub(crate) use replay::replay;

use litellm_core_utils::decline::Decline;

/// The reason a request with no native route carries to Python.
pub const NOT_NATIVE: Decline = Decline::new("route not native");

#[derive(Clone)]
pub enum Fallback {
    /// Standalone: an unserved path is a 404, a decline is the gateway's own 501.
    Reject,
    /// Next to the Python proxy: both go to Python unchanged.
    Forward(Forward),
}

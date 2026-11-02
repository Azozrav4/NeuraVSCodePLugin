//! Hardware backends for Neura.
//!
//! * `tensor` — the native CPU tensor engine (real kernels, used by the VM).
//! * `device` — device discovery and automatic placement heuristics.
//! * `cuda` / `rocm` — accelerator interfaces with honest availability checks.

pub mod cuda;
pub mod device;
pub mod rocm;
pub mod tensor;

pub use device::{choose_device, discover, DeviceInfo, DeviceKind, PlacementInput};
pub use tensor::{Storage, Tensor, TResult};

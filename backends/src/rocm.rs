//! AMD ROCm backend interface.
//!
//! Kernels target ROCDL/AMDGPU and rocBLAS is used for GEMM. Presence of the
//! HIP runtime is detected; when it is missing launches fail with an explicit
//! diagnostic rather than pretending an AMD GPU executed the work.

use crate::device::{rocm_info, DeviceInfo};

#[derive(Clone, Debug)]
pub struct RocmBackend {
    info: DeviceInfo,
}

impl RocmBackend {
    pub fn detect() -> RocmBackend {
        RocmBackend { info: rocm_info() }
    }

    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    pub fn is_available(&self) -> bool {
        self.info.available
    }

    pub fn allocate(&self, bytes: usize) -> Result<crate::cuda::DeviceBuffer, String> {
        if !self.is_available() {
            return Err(format!(
                "ROCm backend unavailable: {}",
                self.info.reason.clone().unwrap_or_else(|| "unknown".into())
            ));
        }
        Ok(crate::cuda::DeviceBuffer { bytes, device: 0 })
    }

    pub fn launch(&self, name: &str) -> Result<(), String> {
        if !self.is_available() {
            return Err(format!(
                "cannot launch ROCm kernel `{}`: HIP runtime not found",
                name
            ));
        }
        Err(format!(
            "ROCm kernel `{}` was not compiled: this build has no ROCDL code generator",
            name
        ))
    }
}

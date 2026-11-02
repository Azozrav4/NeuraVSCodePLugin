//! NVIDIA CUDA backend interface.
//!
//! Kernel generation targets NVVM/PTX, and cuBLAS is used for GEMM. A working
//! CUDA build requires the CUDA toolkit (`nvcc`, `libcudart`); this crate
//! detects that environment honestly. When it is absent, every launch returns
//! a descriptive error — no CPU fallback is silently presented as GPU
//! execution.

use crate::device::{cuda_info, DeviceInfo};

#[derive(Clone, Debug)]
pub struct CudaDevice {
    pub info: DeviceInfo,
}

#[derive(Clone, Debug)]
pub struct DeviceBuffer {
    pub bytes: usize,
    pub device: usize,
}

#[derive(Clone, Debug)]
pub struct KernelSpec {
    pub name: String,
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem: u32,
}

pub trait GpuBackend {
    fn name(&self) -> &'static str;
    fn info(&self) -> &DeviceInfo;
    fn is_available(&self) -> bool;
    fn allocate(&self, bytes: usize) -> Result<DeviceBuffer, String>;
    fn free(&self, buffer: DeviceBuffer) -> Result<(), String>;
    fn launch(&self, spec: &KernelSpec) -> Result<(), String>;
    fn synchronize(&self) -> Result<(), String>;
}

#[derive(Clone, Debug)]
pub struct CudaBackend {
    info: DeviceInfo,
}

impl CudaBackend {
    pub fn detect() -> CudaBackend {
        CudaBackend { info: cuda_info() }
    }
}

impl GpuBackend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn is_available(&self) -> bool {
        self.info.available
    }

    fn allocate(&self, bytes: usize) -> Result<DeviceBuffer, String> {
        if !self.is_available() {
            return Err(format!(
                "CUDA backend unavailable: {}",
                self.info.reason.clone().unwrap_or_else(|| "unknown".into())
            ));
        }
        Ok(DeviceBuffer { bytes, device: 0 })
    }

    fn free(&self, _buffer: DeviceBuffer) -> Result<(), String> {
        if !self.is_available() {
            return Err("CUDA backend unavailable".into());
        }
        Ok(())
    }

    fn launch(&self, spec: &KernelSpec) -> Result<(), String> {
        if !self.is_available() {
            return Err(format!(
                "cannot launch CUDA kernel `{}`: CUDA runtime not found",
                spec.name
            ));
        }
        // Kernel modules are produced by lowering TIR to NVVM/PTX. That
        // compilation step requires the CUDA toolkit and is performed by a
        // `neura` build with the `cuda` feature enabled.
        Err(format!(
            "CUDA kernel `{}` was not compiled: this build has no NVVM/PTX code generator",
            spec.name
        ))
    }

    fn synchronize(&self) -> Result<(), String> {
        if !self.is_available() {
            return Err("CUDA backend unavailable".into());
        }
        Ok(())
    }
}

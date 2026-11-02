//! Hardware discovery and automatic device selection.
//!
//! The CPU backend is always available. CUDA and ROCm are detected from the
//! presence of their runtime libraries; when they are missing this module
//! reports *unavailable* with the reason instead of pretending a GPU exists.

use std::path::Path;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DeviceKind {
    Cpu,
    Cuda,
    Rocm,
}

impl DeviceKind {
    pub fn name(self) -> &'static str {
        match self {
            DeviceKind::Cpu => "cpu",
            DeviceKind::Cuda => "cuda",
            DeviceKind::Rocm => "rocm",
        }
    }
}

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub kind: DeviceKind,
    pub name: String,
    pub available: bool,
    pub reason: Option<String>,
    pub memory_bytes: Option<usize>,
}

pub fn cpu_info() -> DeviceInfo {
    DeviceInfo {
        kind: DeviceKind::Cpu,
        name: format!("cpu:{}", std::env::consts::ARCH),
        available: true,
        reason: None,
        memory_bytes: None,
    }
}

fn find_lib(candidates: &[&str]) -> Option<String> {
    candidates
        .iter()
        .find(|p| Path::new(p).exists())
        .map(|p| p.to_string())
}

pub fn cuda_info() -> DeviceInfo {
    let lib = find_lib(&[
        "/usr/local/cuda/lib64/libcudart.so",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib64/libcuda.so.1",
        "/opt/cuda/lib64/libcudart.so",
    ]);
    match lib {
        Some(p) => DeviceInfo {
            kind: DeviceKind::Cuda,
            name: "cuda:0".into(),
            available: true,
            reason: Some(format!("runtime found at {}", p)),
            memory_bytes: None,
        },
        None => DeviceInfo {
            kind: DeviceKind::Cuda,
            name: "cuda:0".into(),
            available: false,
            reason: Some("no CUDA runtime library found; NVIDIA backend is not usable on this machine".into()),
            memory_bytes: None,
        },
    }
}

pub fn rocm_info() -> DeviceInfo {
    let lib = find_lib(&[
        "/opt/rocm/lib/libamdhip64.so",
        "/opt/rocm/lib64/libamdhip64.so",
        "/usr/lib/libamdhip64.so",
    ]);
    match lib {
        Some(p) => DeviceInfo {
            kind: DeviceKind::Rocm,
            name: "rocm:0".into(),
            available: true,
            reason: Some(format!("runtime found at {}", p)),
            memory_bytes: None,
        },
        None => DeviceInfo {
            kind: DeviceKind::Rocm,
            name: "rocm:0".into(),
            available: false,
            reason: Some("no ROCm runtime library found; AMD backend is not usable on this machine".into()),
            memory_bytes: None,
        },
    }
}

pub fn discover() -> Vec<DeviceInfo> {
    vec![cpu_info(), cuda_info(), rocm_info()]
}

/// Estimated work required before moving a tensor to an accelerator is worth
/// the transfer. These are deliberately conservative: the compiler must not
/// assume a GPU is always faster.
pub struct PlacementInput {
    pub elements: usize,
    pub estimated_flops: u64,
    pub bytes: usize,
}

pub fn choose_device(
    requested: &str,
    input: &PlacementInput,
    gpu_available: bool,
) -> DeviceKind {
    match requested {
        "cpu" => DeviceKind::Cpu,
        "cuda" => DeviceKind::Cuda,
        "rocm" => DeviceKind::Rocm,
        _ => {
            if !gpu_available {
                return DeviceKind::Cpu;
            }
            // Transfer cost dominates for small tensors; require at least
            // ~1 MiB of data and ~10 MFLOP before considering the GPU.
            if input.bytes >= 1 << 20 && input.estimated_flops >= 10_000_000 {
                DeviceKind::Cuda
            } else {
                DeviceKind::Cpu
            }
        }
    }
}

pub fn estimate_flops(op: &neura_ir::ops::TensorOp, shapes: &[Vec<usize>]) -> u64 {
    let numel = |s: &[usize]| s.iter().product::<usize>() as u64;
    match op {
        neura_ir::ops::TensorOp::MatMul if shapes.len() == 2 => {
            let a = &shapes[0];
            let b = &shapes[1];
            if a.len() < 2 || b.len() < 2 {
                return 0;
            }
            let m = a[a.len() - 2] as u64;
            let k = a[a.len() - 1] as u64;
            let n = b[b.len() - 1] as u64;
            2 * m * k * n
        }
        _ => shapes.iter().map(|s| numel(s)).max().unwrap_or(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_is_always_available() {
        assert!(discover().iter().any(|d| d.kind == DeviceKind::Cpu && d.available));
    }

    #[test]
    fn small_workloads_stay_on_cpu() {
        let small = PlacementInput { elements: 100, estimated_flops: 1_000, bytes: 400 };
        assert_eq!(choose_device("auto", &small, true), DeviceKind::Cpu);
        let big = PlacementInput { elements: 1_000_000, estimated_flops: 50_000_000, bytes: 8 << 20 };
        assert_eq!(choose_device("auto", &big, true), DeviceKind::Cuda);
    }

    #[test]
    fn explicit_requests_are_respected() {
        let small = PlacementInput { elements: 4, estimated_flops: 8, bytes: 16 };
        assert_eq!(choose_device("cpu", &small, true), DeviceKind::Cpu);
        assert_eq!(choose_device("cuda", &small, true), DeviceKind::Cuda);
        assert_eq!(choose_device("auto", &small, false), DeviceKind::Cpu);
    }
}

//! Device placement and independent-operation scheduling.
//!
//! Tensor operations are grouped into dependency levels: every operation in a
//! level can run concurrently. Each operation is also assigned a device using
//! the same conservative cost model the runtime uses: small transfers stay on
//! the CPU, and a GPU is only selected when it is present and the work is large
//! enough to amortise the copy.

use neura_backends::device::{
    choose_device, discover, estimate_flops, DeviceKind, PlacementInput,
};
use neura_ir::eir::{EirProgram, Inst, Reg};
use neura_ir::ops::TensorOp;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct OpPlacement {
    pub function: String,
    pub pc: usize,
    pub op: String,
    pub device: &'static str,
    pub elements: usize,
    pub flops: u64,
    pub bytes: usize,
    pub reason: &'static str,
}

#[derive(Clone, Debug, Default)]
pub struct FunctionSchedule {
    pub function: String,
    pub levels: Vec<Vec<usize>>,
}

impl FunctionSchedule {
    /// Maximum number of operations that can execute concurrently.
    pub fn width(&self) -> usize {
        self.levels.iter().map(|l| l.len()).max().unwrap_or(0)
    }
}

#[derive(Clone, Debug)]
pub struct SchedulingPlan {
    pub requested: &'static str,
    pub gpu_available: bool,
    pub placements: Vec<OpPlacement>,
    pub schedules: Vec<FunctionSchedule>,
    pub notes: Vec<String>,
}

impl SchedulingPlan {
    pub fn counts(&self) -> (usize, usize) {
        let mut cpu = 0;
        let mut gpu = 0;
        for p in &self.placements {
            if p.device == "cpu" {
                cpu += 1;
            } else {
                gpu += 1;
            }
        }
        (cpu, gpu)
    }
}

fn shape_of(program: &EirProgram, fi: usize, r: Reg) -> Option<(Vec<usize>, usize)> {
    let t = program.functions[fi].types.get(r as usize)?.tensor()?;
    let shape = t.static_shape()?;
    let numel: usize = shape.iter().product();
    Some((shape, numel))
}

pub fn plan(program: &EirProgram) -> SchedulingPlan {
    let devices = discover();
    let gpu_available = devices
        .iter()
        .any(|d| d.available && matches!(d.kind, DeviceKind::Cuda | DeviceKind::Rocm));
    let requested = program.policies.device.name();
    let mut out = SchedulingPlan {
        requested,
        gpu_available,
        placements: Vec::new(),
        schedules: Vec::new(),
        notes: Vec::new(),
    };

    for fi in 0..program.functions.len() {
        let f = &program.functions[fi];
        let mut producer: HashMap<Reg, usize> = HashMap::new();
        let mut depth: HashMap<usize, usize> = HashMap::new();
        let mut schedule = FunctionSchedule { function: f.name.clone(), levels: Vec::new() };

        for (pc, inst) in f.code.iter().enumerate() {
            if let Some(d) = inst.dst() {
                producer.insert(d, pc);
            }
            let (op_name, shapes, op) = match inst {
                Inst::TensorOp { args, op, .. } => {
                    let shapes: Vec<Vec<usize>> = args
                        .iter()
                        .filter_map(|r| shape_of(program, fi, *r).map(|(s, _)| s))
                        .collect();
                    (op.name(), shapes, Some(op.clone()))
                }
                Inst::FusedOp { args, .. } => {
                    let shapes: Vec<Vec<usize>> = args
                        .iter()
                        .filter_map(|r| shape_of(program, fi, *r).map(|(s, _)| s))
                        .collect();
                    ("fused".to_string(), shapes, None)
                }
                _ => continue,
            };
            let mut level = 0usize;
            let mut elements = 0usize;
            let mut bytes = 0usize;
            for r in inst.reads() {
                if let Some(p) = producer.get(&r) {
                    if *p < pc {
                        level = level.max(depth.get(p).copied().unwrap_or(0) + 1);
                    }
                }
                if let Some((_, numel)) = shape_of(program, fi, r) {
                    elements = elements.max(numel);
                    bytes += numel
                        * program.functions[fi]
                            .types
                            .get(r as usize)
                            .and_then(|t| t.tensor())
                            .map(|t| t.dtype.size())
                            .unwrap_or(4);
                }
            }
            if let Some(t) = f.types.get(inst.dst().unwrap_or(0) as usize).and_then(|t| t.tensor())
            {
                if let Some(shape) = t.static_shape() {
                    elements = elements.max(shape.iter().product());
                }
            }
            let flops = match &op {
                Some(op) => estimate_flops(op, &shapes),
                None => elements as u64,
            };
            let input = PlacementInput { elements, estimated_flops: flops, bytes };
            let device = choose_device(requested, &input, gpu_available);
            let reason = if requested != "auto" {
                "explicit device policy"
            } else if device == DeviceKind::Cpu {
                if !gpu_available {
                    "no supported GPU runtime detected"
                } else {
                    "below the transfer/memory threshold"
                }
            } else {
                "large enough to amortise the transfer"
            };
            out.placements.push(OpPlacement {
                function: f.name.clone(),
                pc,
                op: op_name,
                device: device.name(),
                elements,
                flops,
                bytes,
                reason,
            });
            depth.insert(pc, level);
            if schedule.levels.len() <= level {
                schedule.levels.resize(level + 1, Vec::new());
            }
            schedule.levels[level].push(pc);
        }
        out.schedules.push(schedule);
    }

    out.notes.push(format!(
        "{} tensor operations placed; GPU runtime {}",
        out.placements.len(),
        if gpu_available { "detected" } else { "not detected" }
    ));
    if !gpu_available {
        out.notes.push(
            "this build executes on the native CPU backend; the placement plan is what a build with CUDA/ROCm would use"
                .into(),
        );
    }
    out
}

/// Convenience used by tests and the CLI: which ops are eligible for the GPU.
pub fn gpu_eligible(placements: &[OpPlacement]) -> usize {
    placements.iter().filter(|p| p.device != "cpu").count()
}

/// Operation kind recorded for a placement (kept for report formatting).
pub fn describe_op(op: &TensorOp) -> String {
    op.name()
}

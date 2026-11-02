//! Last-use analysis and buffer planning.
//!
//! The plan assigns every tensor temporary to a buffer slot. Slots whose
//! lifetimes do not overlap share storage, which is what the runtime's memory
//! manager reports as a pooled reuse. Locals that participate in autograd, are
//! function parameters, or live across a loop back-edge are never dropped
//! early: gradients and the next iteration may still need them.

use neura_ir::eir::{EirProgram, Inst, Reg};
use neura_ir::types::Ty;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct BufferSlot {
    pub function: String,
    pub reg: Reg,
    pub bytes: usize,
    pub first: usize,
    pub last: usize,
    pub slot: usize,
    pub reused: bool,
}

#[derive(Clone, Debug, Default)]
pub struct MemoryPlan {
    pub function: String,
    pub buffers: Vec<BufferSlot>,
    pub peak_bytes: usize,
    pub shared_bytes: usize,
    pub drops_inserted: usize,
}

impl MemoryPlan {
    pub fn lives(&self) -> usize {
        self.buffers.len()
    }

    pub fn reused_buffers(&self) -> usize {
        self.buffers.iter().filter(|b| b.reused).count()
    }
}

/// Back-edges of loops, as `(start, end)` instruction ranges.
fn loop_regions(code: &[Inst]) -> Vec<(usize, usize)> {
    let mut regions = Vec::new();
    for (i, inst) in code.iter().enumerate() {
        let target = match inst {
            Inst::Jump(t) => Some(*t),
            Inst::JumpIfFalse { target, .. } | Inst::JumpIfTrue { target, .. } => Some(*target),
            _ => None,
        };
        if let Some(t) = target {
            if t <= i {
                regions.push((t, i));
            }
        }
    }
    regions
}

fn plan_buffers(program: &EirProgram, f_index: usize) -> Vec<BufferSlot> {
    let f = &program.functions[f_index];
    let mut last: HashMap<Reg, usize> = HashMap::new();
    for (i, inst) in f.code.iter().enumerate() {
        for r in inst.reads() {
            last.insert(r, i);
        }
    }
    let mut defs: HashMap<Reg, usize> = HashMap::new();
    let mut items: Vec<BufferSlot> = Vec::new();
    for (i, inst) in f.code.iter().enumerate() {
        let Some(d) = inst.dst() else { continue };
        defs.insert(d, i);
        let Some(ty) = f.types.get(d as usize) else { continue };
        let Some(tensor) = ty.tensor() else { continue };
        let Some(shape) = tensor.static_shape() else { continue };
        let bytes: usize = shape.iter().product::<usize>() * tensor.dtype.size();
        let end = last.get(&d).copied().unwrap_or(i).max(i);
        items.push(BufferSlot {
            function: f.name.clone(),
            reg: d,
            bytes,
            first: i,
            last: end,
            slot: 0,
            reused: false,
        });
    }
    // Greedy interval colouring: reuse the first slot whose last use ended.
    let mut slot_ends: Vec<usize> = Vec::new();
    for item in &mut items {
        if let Some(slot) = slot_ends.iter().position(|end| *end < item.first) {
            slot_ends[slot] = item.last;
            item.slot = slot;
            item.reused = true;
        } else {
            item.slot = slot_ends.len();
            slot_ends.push(item.last);
        }
    }
    items
}

fn peak_bytes(code_len: usize, buffers: &[BufferSlot]) -> usize {
    let mut peak = 0usize;
    for t in 0..code_len {
        let live: usize = buffers
            .iter()
            .filter(|b| b.first <= t && t <= b.last)
            .map(|b| b.bytes)
            .sum();
        peak = peak.max(live);
    }
    peak
}

fn insert_after(code: &[Inst], after: &HashMap<usize, Vec<Reg>>) -> Vec<Inst> {
    let extra: usize = after.values().map(|v| v.len()).sum();
    let total = code.len() + extra;
    let mut out = Vec::with_capacity(total);
    let mut map = vec![0usize; code.len() + 1];
    for (i, inst) in code.iter().enumerate() {
        map[i] = out.len();
        out.push(inst.clone());
        if let Some(regs) = after.get(&i) {
            for reg in regs {
                out.push(Inst::Drop(*reg));
            }
        }
    }
    map[code.len()] = total;
    let remap = |old: usize| if old >= code.len() { total } else { map[old] };
    for inst in &mut out {
        match inst {
            Inst::Jump(t) => *t = remap(*t),
            Inst::JumpIfFalse { target, .. } | Inst::JumpIfTrue { target, .. } => {
                *target = remap(*target)
            }
            Inst::IterNext { end, .. } => *end = remap(*end),
            _ => {}
        }
    }
    out
}

/// Analyse every function, update `last_use` metadata and optionally insert
/// early `Drop`s for locals whose lifetime is statically closed.
pub fn plan(program: &mut EirProgram, insert_drops: bool) -> Vec<MemoryPlan> {
    let mut plans = Vec::new();
    for fi in 0..program.functions.len() {
        let buffers = plan_buffers(program, fi);
        let f = &mut program.functions[fi];
        let regions = loop_regions(&f.code);

        let mut last: HashMap<Reg, usize> = HashMap::new();
        let mut defs: HashMap<Reg, usize> = HashMap::new();
        for (i, inst) in f.code.iter().enumerate() {
            for r in inst.reads() {
                last.insert(r, i);
            }
            if let Some(d) = inst.dst() {
                defs.insert(d, i);
            }
        }
        for (ri, local) in f.locals.iter_mut().enumerate() {
            local.last_use = last.get(&(ri as Reg)).copied();
        }

        let mut drops: HashMap<usize, Vec<Reg>> = HashMap::new();
        if insert_drops {
            for (ri, local) in f.locals.iter().enumerate() {
                if local.is_param || local.needs_autograd {
                    continue;
                }
                if !matches!(
                    local.ty,
                    Ty::Tensor(_) | Ty::List(_) | Ty::Struct(_) | Ty::Opaque(_)
                ) {
                    continue;
                }
                let reg = ri as Reg;
                let Some(&at) = last.get(&reg) else { continue };
                let Some(&def) = defs.get(&reg) else { continue };
                if matches!(f.code[at], Inst::Return(_) | Inst::Jump(_)) {
                    continue;
                }
                // Never drop a value created before a loop inside that loop:
                // the next iteration would read a cleared register.
                let crosses_loop = regions
                    .iter()
                    .any(|(s, e)| def < *s && *s <= at && at <= *e);
                if crosses_loop {
                    continue;
                }
                drops.entry(at).or_default().push(reg);
            }
        }
        let inserted: usize = drops.values().map(|v| v.len()).sum();
        if !drops.is_empty() {
            f.code = insert_after(&f.code, &drops);
        }
        let shared = buffers
            .iter()
            .filter(|b| b.reused)
            .map(|b| b.bytes)
            .sum();
        plans.push(MemoryPlan {
            function: f.name.clone(),
            peak_bytes: peak_bytes(f.code.len(), &buffers),
            shared_bytes: shared,
            drops_inserted: inserted,
            buffers,
        });
    }
    plans
}

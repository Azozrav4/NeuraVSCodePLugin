//! Precision planning.
//!
//! Mixed precision is opt-in (`--precision bf16|fp16`). When enabled, matrix
//! multiplies and convolutions whose shapes are fully static produce reduced
//! precision results; every other operation keeps the precision the program
//! declared. The default policy is `strict`, which performs no rewrite at all.

use neura_ir::eir::{EirProgram, Inst, PrecisionPolicy};
use neura_ir::ops::TensorOp;
use neura_ir::types::{DType, TensorTy, Ty};

#[derive(Clone, Debug)]
pub struct PrecisionChange {
    pub function: String,
    pub pc: usize,
    pub op: String,
    pub from: DType,
    pub to: DType,
}

#[derive(Clone, Debug)]
pub struct PrecisionPlan {
    pub policy: PrecisionPolicy,
    pub changes: Vec<PrecisionChange>,
    pub notes: Vec<String>,
}

fn target(policy: PrecisionPolicy) -> Option<DType> {
    match policy {
        PrecisionPolicy::Strict => None,
        PrecisionPolicy::MixedBf16 => Some(DType::BF16),
        PrecisionPolicy::MixedFp16 => Some(DType::F16),
    }
}

fn is_heavy(op: &TensorOp) -> bool {
    matches!(op, TensorOp::MatMul | TensorOp::Conv2D { .. } | TensorOp::Conv1D { .. })
}

fn eligible(ty: &TensorTy) -> bool {
    matches!(ty.dtype, DType::F32 | DType::F64) && ty.static_shape().is_some()
}

/// Rewrite eligible operations for the requested policy.
pub fn apply(program: &mut EirProgram, enabled: bool) -> PrecisionPlan {
    let policy = program.policies.precision;
    let mut plan = PrecisionPlan { policy, changes: Vec::new(), notes: Vec::new() };
    let Some(to) = target(policy) else {
        plan.notes.push("strict precision: no dtype was changed".into());
        return plan;
    };
    if !enabled {
        plan.notes.push("optimisation disabled: precision plan was not applied".into());
        return plan;
    }
    for f in &mut program.functions {
        for pc in 0..f.code.len() {
            let inst = f.code[pc].clone();
            let Inst::TensorOp { dst, op, args, dyn_dims, ty } = inst else {
                continue;
            };
            if !is_heavy(&op) || !eligible(&ty) {
                continue;
            }
            let inputs_static = args.iter().all(|r| {
                f.types
                    .get(*r as usize)
                    .and_then(|t| t.tensor())
                    .map(|t| t.static_shape().is_some() && t.dtype.is_float())
                    .unwrap_or(false)
            });
            if !inputs_static {
                continue;
            }
            let from = ty.dtype;
            let new_ty = TensorTy::new(to, ty.shape.clone());
            f.code[pc] = Inst::TensorOp {
                dst,
                op: op.clone(),
                args,
                dyn_dims,
                ty: new_ty.clone(),
            };
            f.types[dst as usize] = Ty::Tensor(new_ty);
            if let Some(local) = f.locals.get_mut(dst as usize) {
                if let Ty::Tensor(t) = &local.ty {
                    local.ty = Ty::Tensor(TensorTy::new(to, t.shape.clone()));
                }
            }
            plan.changes.push(PrecisionChange {
                function: f.name.clone(),
                pc,
                op: op.name(),
                from,
                to,
            });
        }
    }
    plan.notes.push(format!(
        "{} matrix/convolution kernels now produce {}; elementwise and reduction operations keep their declared dtype",
        plan.changes.len(),
        to.name()
    ));
    plan
}

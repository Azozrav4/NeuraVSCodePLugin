//! Elementwise kernel fusion.
//!
//! A chain such as `relu -> add -> gelu` is rewritten into a single
//! [`Inst::FusedOp`] carrying an expression tree. The runtime evaluates the
//! tree once per output element instead of materialising every intermediate
//! tensor, and the autograd engine can rebuild the same chain when gradients
//! are required.

use crate::autotuning::{Autotuner, TuneDecision};
use crate::graph::{remove_indices, use_counts};
use crate::PassOutcome;
use neura_ir::eir::{EirProgram, Inst, Reg};
use neura_ir::ops::{FusedExpr, FusedSpec, TensorOp};
use std::collections::{BTreeSet, HashMap};

#[derive(Clone)]
struct Fused {
    args: Vec<Reg>,
    expr: FusedExpr,
}

/// Rebuild `e` with every input index shifted by `by`.
fn shift(e: &FusedExpr, by: u32) -> FusedExpr {
    match e {
        FusedExpr::Input(i) => FusedExpr::Input(i + by),
        FusedExpr::Const(v) => FusedExpr::Const(*v),
        FusedExpr::Add(a, b) => FusedExpr::Add(Box::new(shift(a, by)), Box::new(shift(b, by))),
        FusedExpr::Sub(a, b) => FusedExpr::Sub(Box::new(shift(a, by)), Box::new(shift(b, by))),
        FusedExpr::Mul(a, b) => FusedExpr::Mul(Box::new(shift(a, by)), Box::new(shift(b, by))),
        FusedExpr::Div(a, b) => FusedExpr::Div(Box::new(shift(a, by)), Box::new(shift(b, by))),
        FusedExpr::Neg(a) => FusedExpr::Neg(Box::new(shift(a, by))),
        FusedExpr::ReLU(a) => FusedExpr::ReLU(Box::new(shift(a, by))),
        FusedExpr::GELU(a) => FusedExpr::GELU(Box::new(shift(a, by))),
        FusedExpr::Sigmoid(a) => FusedExpr::Sigmoid(Box::new(shift(a, by))),
        FusedExpr::Tanh(a) => FusedExpr::Tanh(Box::new(shift(a, by))),
        FusedExpr::Exp(a) => FusedExpr::Exp(Box::new(shift(a, by))),
        FusedExpr::Log(a) => FusedExpr::Log(Box::new(shift(a, by))),
        FusedExpr::Sqrt(a) => FusedExpr::Sqrt(Box::new(shift(a, by))),
    }
}

fn apply_op(op: &TensorOp, mut operands: Vec<FusedExpr>) -> Option<FusedExpr> {
    let b = operands.pop()?;
    let a = operands.pop();
    Some(match (op, a) {
        (TensorOp::Add, Some(a)) => FusedExpr::Add(Box::new(a), Box::new(b)),
        (TensorOp::Sub, Some(a)) => FusedExpr::Sub(Box::new(a), Box::new(b)),
        (TensorOp::Mul, Some(a)) => FusedExpr::Mul(Box::new(a), Box::new(b)),
        (TensorOp::Div, Some(a)) => FusedExpr::Div(Box::new(a), Box::new(b)),
        (TensorOp::Neg, None) => FusedExpr::Neg(Box::new(b)),
        (TensorOp::ReLU, None) => FusedExpr::ReLU(Box::new(b)),
        (TensorOp::GELU, None) => FusedExpr::GELU(Box::new(b)),
        (TensorOp::Sigmoid, None) => FusedExpr::Sigmoid(Box::new(b)),
        (TensorOp::Tanh, None) => FusedExpr::Tanh(Box::new(b)),
        (TensorOp::Exp, None) => FusedExpr::Exp(Box::new(b)),
        (TensorOp::Log, None) => FusedExpr::Log(Box::new(b)),
        (TensorOp::Sqrt, None) => FusedExpr::Sqrt(Box::new(b)),
        _ => return None,
    })
}

/// Fuse chains of elementwise tensor operations.
///
/// When `measure` is set, each multi-operation chain is handed to the
/// autotuner, which benchmarks the eager and fused implementations and can
/// reject a fusion that does not pay off.
pub fn fuse_elementwise(
    program: &mut EirProgram,
    tuner: &mut Autotuner,
    measure: bool,
) -> (PassOutcome, Vec<TuneDecision>) {
    let mut fused = 0usize;
    let mut removed_ops = 0usize;
    let mut decisions = Vec::new();
    for f in &mut program.functions {
        let counts = use_counts(&f.code);
        let mut producer: HashMap<Reg, usize> = HashMap::new();
        for (i, inst) in f.code.iter().enumerate() {
            if let Some(d) = inst.dst() {
                producer.insert(d, i);
            }
        }
        let mut values: HashMap<Reg, Fused> = HashMap::new();
        let mut drop: BTreeSet<usize> = BTreeSet::new();

        for i in 0..f.code.len() {
            let inst = f.code[i].clone();
            if let Inst::TensorOp { dst, op, args, dyn_dims, ty } = &inst {
                if op.is_elementwise() && dyn_dims.is_empty() {
                    let mut base: Vec<Reg> = Vec::new();
                    let mut operands: Vec<FusedExpr> = Vec::new();
                    let mut absorbed: Vec<Reg> = Vec::new();
                    for r in args {
                        let reuse = values.get(r).filter(|_| {
                            counts.get(r).copied().unwrap_or(0) == 1 && producer.contains_key(r)
                        });
                        if let Some(v) = reuse {
                            let start = base.len() as u32;
                            base.extend(v.args.iter().copied());
                            operands.push(shift(&v.expr, start));
                            absorbed.push(*r);
                        } else {
                            let idx = base.len() as u32;
                            base.push(*r);
                            operands.push(FusedExpr::Input(idx));
                        }
                    }
                    if let Some(expr) = apply_op(op, operands) {
                        let accept = if absorbed.is_empty() || !measure {
                            true
                        } else {
                            let mut shapes = Vec::new();
                            let mut complete = true;
                            for r in &base {
                                match f
                                    .types
                                    .get(*r as usize)
                                    .and_then(|t| t.tensor())
                                    .and_then(|t| t.static_shape())
                                {
                                    Some(s) => shapes.push(s),
                                    None => {
                                        complete = false;
                                        break;
                                    }
                                }
                            }
                            let d = if complete {
                                tuner.decide(&expr, &shapes)
                            } else {
                                tuner.decide(&expr, &[])
                            };
                            let ok = d.use_fused;
                            decisions.push(d);
                            ok
                        };
                        // A new definition invalidates expressions that used
                        // the destination register.
                        values.retain(|reg, v| reg != dst && !v.args.contains(dst));
                        if accept {
                            for r in &absorbed {
                                if let Some(pi) = producer.get(r) {
                                    if *pi != i {
                                        drop.insert(*pi);
                                    }
                                }
                                values.remove(r);
                            }
                            if !absorbed.is_empty() {
                                f.code[i] = Inst::FusedOp {
                                    dst: *dst,
                                    spec: FusedSpec { expr: expr.clone(), ty: ty.clone() },
                                    args: base.clone(),
                                    ty: ty.clone(),
                                };
                                fused += 1;
                                removed_ops += absorbed.len();
                            }
                            values.insert(*dst, Fused { args: base, expr });
                        } else {
                            let raw: Vec<FusedExpr> =
                                (0..args.len()).map(|k| FusedExpr::Input(k as u32)).collect();
                            if let Some(e) = apply_op(op, raw) {
                                values.insert(*dst, Fused { args: args.clone(), expr: e });
                            }
                        }
                        continue;
                    }
                }
                values.retain(|reg, v| reg != dst && !v.args.contains(dst));
                producer.insert(*dst, i);
                continue;
            }
            if let Some(d) = inst.dst() {
                values.retain(|reg, v| *reg != d && !v.args.contains(&d));
                producer.insert(d, i);
            }
        }
        if !drop.is_empty() {
            f.code = remove_indices(&f.code, &drop);
        }
    }
    (
        PassOutcome {
            name: "elementwise-fusion",
            changed: fused,
            detail: format!(
                "emitted {} fused kernels absorbing {} intermediate tensors",
                fused, removed_ops
            ),
        },
        decisions,
    )
}

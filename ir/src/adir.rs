//! ADIR — automatic-differentiation IR.
//!
//! The forward tensor graph (TIR) is projected into a differentiation graph:
//! every differentiable node records how many saved tensors its VJP needs,
//! whether it is linear, and its gradient rule. The runtime autograd engine
//! implements the same rules over concrete tensors; ADIR makes those rules
//! inspectable and lets the compiler validate gradient graphs before running.

use crate::ops::TensorOp;
use crate::tir::TirGraph;
use crate::types::TensorTy;
use std::fmt::Write as _;

#[derive(Clone, Debug)]
pub struct VjpRule {
    pub name: &'static str,
    /// Number of forward values that must be saved for the backward pass.
    pub num_saved: usize,
    /// Linear ops need no saved tensors beyond their inputs.
    pub linear: bool,
    /// Ops with a custom vector-Jacobian product (softmax, conv, ...).
    pub custom: bool,
}

pub fn vjp_rule(op: &TensorOp) -> VjpRule {
    use TensorOp::*;
    match op {
        Add | Sub | Neg | Sum { .. } | Mean { .. } | Reshape { .. } | Flatten
        | Transpose { .. } | BroadcastTo { .. } | Slice { .. } => VjpRule {
            name: "linear",
            num_saved: 0,
            linear: true,
            custom: false,
        },
        Mul | Div | MatMul | Tanh | Sigmoid | Exp | Log | Sqrt | GELU | Softmax { .. }
        | ReduceMax { .. } | ReduceMin { .. } | Dropout { .. } => VjpRule {
            name: "saved-inputs",
            num_saved: 1,
            linear: false,
            custom: matches!(op, Softmax { .. } | Mul | Div | Dropout { .. }),
        },
        ReLU => VjpRule { name: "relu-mask", num_saved: 1, linear: false, custom: false },
        Gather { .. } | Conv1D { .. } | Conv2D { .. } => VjpRule {
            name: "structural",
            num_saved: 1,
            linear: false,
            custom: true,
        },
    }
}

#[derive(Clone, Debug)]
pub struct AdNode {
    pub id: usize,
    pub op: TensorOp,
    pub inputs: Vec<usize>,
    pub ty: TensorTy,
    pub rule: VjpRule,
    /// Number of gradient buffers accumulated into this node.
    pub grad_accumulators: usize,
    /// Activation checkpointing candidate (large intermediate values).
    pub checkpoint_candidate: bool,
}

#[derive(Clone, Debug, Default)]
pub struct AdGraph {
    pub nodes: Vec<AdNode>,
    /// Nodes whose gradients are required (all differentiable outputs).
    pub roots: Vec<usize>,
}

impl AdGraph {
    pub fn from_tir(tir: &TirGraph) -> AdGraph {
        let mut nodes = Vec::new();
        for n in &tir.nodes {
            let rule = vjp_rule(&n.op);
            let elem_size = n.ty.dtype.size();
            let bytes = n.ty.numel().map(|x| x * elem_size).unwrap_or(0);
            nodes.push(AdNode {
                id: n.id,
                op: n.op.clone(),
                inputs: n.args.clone(),
                ty: n.ty.clone(),
                grad_accumulators: n.args.len().max(1),
                checkpoint_candidate: bytes >= 1 << 20 && !n.op.is_elementwise(),
                rule,
            });
        }
        let roots = tir.outputs.clone();
        AdGraph { nodes, roots }
    }

    pub fn report(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "; ADIR gradient graph ({} nodes)", self.nodes.len());
        for n in &self.nodes {
            let _ = writeln!(
                out,
                "adir.node %{} : {} = {} inputs=[{}] vjp={} saved={} checkpoint={}",
                n.id,
                n.ty,
                n.op.name(),
                n.inputs
                    .iter()
                    .map(|i| format!("%{}", i))
                    .collect::<Vec<_>>()
                    .join(", "),
                n.rule.name,
                n.rule.num_saved,
                n.checkpoint_candidate
            );
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules_are_classified() {
        assert!(vjp_rule(&TensorOp::Add).linear);
        assert!(!vjp_rule(&TensorOp::MatMul).linear);
        assert!(vjp_rule(&TensorOp::Softmax { axis: 1 }).custom);
    }
}

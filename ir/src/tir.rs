//! TIR — tensor dataflow graph.
//!
//! A TIR graph is a projection of the tensor operations of one NIR/EIR
//! function. Nodes preserve shape, element type, layout and ownership
//! metadata through every optimization pass. NAOE passes consume this graph
//! and report rewrites; the executable EIR is updated in lockstep.

use crate::eir::{EirFunction, Inst};
use crate::ops::{infer_shape, TensorOp};
use crate::types::{Dim, TensorTy, Ty};
use std::collections::HashMap;

pub type NodeId = usize;

#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    pub strides: Vec<usize>,
    pub contiguous: bool,
    pub alignment: usize,
}

impl Layout {
    pub fn contiguous(shape: &[Dim]) -> Layout {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            let n = shape[i + 1].concrete().unwrap_or(1);
            strides[i] = strides[i + 1] * n;
        }
        Layout { strides, contiguous: true, alignment: 16 }
    }
}

#[derive(Clone, Debug)]
pub struct TirNode {
    pub id: NodeId,
    pub op: TensorOp,
    pub args: Vec<NodeId>,
    pub ty: TensorTy,
    pub layout: Layout,
    pub reg: u32,
    pub inst_index: usize,
    /// Set by the memory planner: which buffer slot this value should reuse.
    pub buffer_slot: Option<usize>,
    /// Set when this node is needed by a backward pass.
    pub needed_for_grad: bool,
}

#[derive(Clone, Debug, Default)]
pub struct TirGraph {
    pub nodes: Vec<TirNode>,
    pub inputs: Vec<NodeId>,
    pub outputs: Vec<NodeId>,
    pub reg_to_node: HashMap<u32, NodeId>,
}

impl TirGraph {
    /// Build a tensor DAG from an executable function.
    pub fn from_eir(func: &EirFunction) -> Result<TirGraph, String> {
        let mut g = TirGraph::default();
        let mut types: HashMap<u32, Ty> = HashMap::new();
        for (idx, ty) in func.types.iter().enumerate() {
            types.insert(idx as u32, ty.clone());
        }
        for (inst_index, inst) in func.code.iter().enumerate() {
            match inst {
                Inst::TensorOp { dst, op, args, ty, .. } => {
                    let mut arg_nodes = Vec::new();
                    for a in args {
                        if let Some(n) = g.reg_to_node.get(a) {
                            arg_nodes.push(*n);
                            continue;
                        }
                        // Register was produced by a const/scalar: materialise
                        // it as a graph input with its static tensor type.
                        let leaf_ty = match types.get(a) {
                            Some(Ty::Tensor(t)) => t.clone(),
                            _ => TensorTy::new(ty.dtype, vec![]),
                        };
                        let id = g.nodes.len();
                        g.nodes.push(TirNode {
                            id,
                            op: TensorOp::BroadcastTo { shape: leaf_ty.shape.clone() },
                            args: vec![],
                            ty: leaf_ty.clone(),
                            layout: Layout::contiguous(&leaf_ty.shape),
                            reg: *a,
                            inst_index,
                            buffer_slot: None,
                            needed_for_grad: false,
                        });
                        g.reg_to_node.insert(*a, id);
                        g.inputs.push(id);
                        arg_nodes.push(id);
                    }
                    let input_tys: Vec<TensorTy> = arg_nodes
                        .iter()
                        .map(|n| g.nodes[*n].ty.clone())
                        .collect();
                    let out_ty = infer_shape(op, &input_tys)?;
                    let id = g.nodes.len();
                    g.nodes.push(TirNode {
                        id,
                        op: op.clone(),
                        args: arg_nodes,
                        ty: out_ty.clone(),
                        layout: Layout::contiguous(&out_ty.shape),
                        reg: *dst,
                        inst_index,
                        buffer_slot: None,
                        needed_for_grad: false,
                    });
                    g.reg_to_node.insert(*dst, id);
                    g.outputs.push(id);
                }
                Inst::FusedOp { dst, ty, .. } => {
                    let id = g.nodes.len();
                    g.nodes.push(TirNode {
                        id,
                        op: TensorOp::ReLU, // fused nodes are reported separately
                        args: vec![],
                        ty: ty.clone(),
                        layout: Layout::contiguous(&ty.shape),
                        reg: *dst,
                        inst_index,
                        buffer_slot: None,
                        needed_for_grad: false,
                    });
                    g.reg_to_node.insert(*dst, id);
                    g.outputs.push(id);
                }
                _ => {}
            }
        }
        Ok(g)
    }

    pub fn node_of_reg(&self, reg: u32) -> Option<&TirNode> {
        self.reg_to_node.get(&reg).map(|n| &self.nodes[*n])
    }

    pub fn uses_of(&self, node: NodeId) -> usize {
        self.nodes
            .iter()
            .filter(|n| n.args.contains(&node))
            .count()
            + self.outputs.iter().filter(|o| **o == node).count()
    }

    pub fn topo_order(&self) -> Vec<NodeId> {
        let mut order = Vec::new();
        let mut seen = vec![false; self.nodes.len()];
        fn visit(g: &TirGraph, n: NodeId, seen: &mut Vec<bool>, order: &mut Vec<NodeId>) {
            if seen[n] {
                return;
            }
            seen[n] = true;
            for a in &g.nodes[n].args {
                visit(g, *a, seen, order);
            }
            order.push(n);
        }
        for n in 0..self.nodes.len() {
            visit(self, n, &mut seen, &mut order);
        }
        order
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DType;

    #[test]
    fn layout_strides() {
        let l = Layout::contiguous(&[Dim::Static(2), Dim::Static(3), Dim::Static(4)]);
        assert_eq!(l.strides, vec![12, 4, 1]);
        assert!(l.contiguous);
    }

    #[test]
    fn dag_use_counts() {
        let mut g = TirGraph::default();
        g.nodes.push(TirNode {
            id: 0,
            op: TensorOp::ReLU,
            args: vec![],
            ty: TensorTy::new(DType::F32, vec![Dim::Static(2)]),
            layout: Layout::contiguous(&[Dim::Static(2)]),
            reg: 1,
            inst_index: 0,
            buffer_slot: None,
            needed_for_grad: false,
        });
        g.nodes.push(TirNode {
            id: 1,
            op: TensorOp::Add,
            args: vec![0],
            ty: TensorTy::new(DType::F32, vec![Dim::Static(2)]),
            layout: Layout::contiguous(&[Dim::Static(2)]),
            reg: 2,
            inst_index: 1,
            buffer_slot: None,
            needed_for_grad: false,
        });
        g.outputs.push(1);
        assert_eq!(g.uses_of(0), 1);
        assert_eq!(g.topo_order(), vec![0, 1]);
    }
}

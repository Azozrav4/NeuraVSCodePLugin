//! Reverse-mode automatic differentiation.
//!
//! Every tensor produced by an operation records its input nodes and a VJP
//! (vector-Jacobian product) rule. `backward` visits the reachable graph in
//! reverse topological order and accumulates gradients into the leaves.
//! Nothing here relies on an external autograd library.

use crate::value::{TRef, TNode};
use neura_backends::tensor::Tensor;
use neura_ir::ops::TensorOp;
use neura_ir::types::{Dim, DType};
use std::collections::HashSet;
use std::rc::Rc;

#[derive(Clone)]
pub enum Vjp {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Transpose { dim0: usize, dim1: usize },
    Reshape { shape: Vec<usize> },
    Sum { axes: Vec<usize>, keepdim: bool, input_shape: Vec<usize> },
    Mean { axes: Vec<usize>, keepdim: bool, input_shape: Vec<usize> },
    ReduceMax { axis: Option<usize>, keepdim: bool, input_shape: Vec<usize> },
    ReduceMin { axis: Option<usize>, keepdim: bool, input_shape: Vec<usize> },
    Slice { axis: usize, start: usize, end: usize, input_shape: Vec<usize> },
    ReLU,
    GELU,
    Sigmoid,
    Tanh,
    Exp,
    Log,
    Sqrt,
    Neg,
    Softmax { axis: usize },
    Conv2D { stride: (usize, usize), padding: (usize, usize) },
    Conv1D { stride: usize, padding: usize },
    Gather { axis: usize, input_shape: Vec<usize> },
    Dropout { mask: Tensor },
    Broadcast { input_shape: Vec<usize> },
}

fn cache_key(node: &TRef) -> usize {
    Rc::as_ptr(node) as usize
}

/// Apply a TIR operation, producing a tensor node with a gradient rule.
pub fn apply(op: &TensorOp, args: &[TRef], training: bool, seed: u64) -> Result<TRef, String> {
    use TensorOp as T;
    match op {
        T::Add | T::Sub | T::Mul | T::Div => {
            if args.len() != 2 {
                return Err(format!("{} expects 2 arguments", op.name()));
            }
            let out = {
                let a = args[0].borrow();
                let b = args[1].borrow();
                a.data.elementwise(op, &b.data)?
            };
            let vjp = match op {
                T::Add => Vjp::Add,
                T::Sub => Vjp::Sub,
                T::Mul => Vjp::Mul,
                _ => Vjp::Div,
            };
            Ok(TNode::derived(out, args.to_vec(), vjp))
        }
        T::MatMul => {
            if args.len() != 2 {
                return Err("matmul expects 2 arguments".into());
            }
            let out = {
                let a = args[0].borrow();
                let b = args[1].borrow();
                matmul_general(&a.data, &b.data)?
            };
            Ok(TNode::derived(out, args.to_vec(), Vjp::MatMul))
        }
        T::Transpose { dim0, dim1 } => {
            let out = args[0].borrow().data.transpose(*dim0, *dim1)?;
            Ok(TNode::derived(out, args.to_vec(), Vjp::Transpose { dim0: *dim0, dim1: *dim1 }))
        }
        T::Reshape { dims } => {
            let input = args[0].borrow();
            let out = input.data.reshape(dims)?;
            Ok(TNode::derived(
                out,
                args.to_vec(),
                Vjp::Reshape { shape: input.data.shape.clone() },
            ))
        }
        T::Flatten => {
            let input = args[0].borrow();
            let out = input.data.flatten();
            Ok(TNode::derived(
                out,
                args.to_vec(),
                Vjp::Reshape { shape: input.data.shape.clone() },
            ))
        }
        T::Slice { axis, start, end } => {
            let input = args[0].borrow();
            let out = input.data.slice(*axis, *start, *end)?;
            Ok(TNode::derived(
                out,
                args.to_vec(),
                Vjp::Slice {
                    axis: *axis,
                    start: *start,
                    end: *end,
                    input_shape: input.data.shape.clone(),
                },
            ))
        }
        T::Sum { axes, keepdim } => {
            let input = args[0].borrow();
            let out = input.data.reduce(op)?;
            Ok(TNode::derived(
                out,
                args.to_vec(),
                Vjp::Sum {
                    axes: axes.clone(),
                    keepdim: *keepdim,
                    input_shape: input.data.shape.clone(),
                },
            ))
        }
        T::Mean { axes, keepdim } => {
            let input = args[0].borrow();
            let out = input.data.reduce(op)?;
            Ok(TNode::derived(
                out,
                args.to_vec(),
                Vjp::Mean {
                    axes: axes.clone(),
                    keepdim: *keepdim,
                    input_shape: input.data.shape.clone(),
                },
            ))
        }
        T::ReduceMax { axis, keepdim } | T::ReduceMin { axis, keepdim } => {
            let input = args[0].borrow();
            let out = input.data.reduce(op)?;
            let vjp = if matches!(op, T::ReduceMax { .. }) {
                Vjp::ReduceMax { axis: *axis, keepdim: *keepdim, input_shape: input.data.shape.clone() }
            } else {
                Vjp::ReduceMin { axis: *axis, keepdim: *keepdim, input_shape: input.data.shape.clone() }
            };
            Ok(TNode::derived(out, args.to_vec(), vjp))
        }
        T::ReLU
        | T::Sigmoid
        | T::Tanh
        | T::Exp
        | T::Log
        | T::Sqrt
        | T::Neg => {
            let out = args[0].borrow().data.unary(op)?;
            let vjp = match op {
                T::ReLU => Vjp::ReLU,
                T::Sigmoid => Vjp::Sigmoid,
                T::Tanh => Vjp::Tanh,
                T::Exp => Vjp::Exp,
                T::Log => Vjp::Log,
                T::Sqrt => Vjp::Sqrt,
                _ => Vjp::Neg,
            };
            Ok(TNode::derived(out, args.to_vec(), vjp))
        }
        T::GELU => {
            let input = args[0].borrow();
            let out = input.data.unary(op)?;
            Ok(TNode::derived(out, args.to_vec(), Vjp::GELU))
        }
        T::Softmax { axis } => {
            let out = args[0].borrow().data.softmax(*axis)?;
            Ok(TNode::derived(out, args.to_vec(), Vjp::Softmax { axis: *axis }))
        }
        T::Conv2D { stride, padding } => {
            if args.len() != 2 {
                return Err("conv2d expects input and weight".into());
            }
            let out = {
                let a = args[0].borrow();
                let b = args[1].borrow();
                a.data.conv2d(&b.data, *stride, *padding)?
            };
            Ok(TNode::derived(
                out,
                args.to_vec(),
                Vjp::Conv2D { stride: *stride, padding: *padding },
            ))
        }
        T::Conv1D { stride, padding } => {
            if args.len() != 2 {
                return Err("conv1d expects input and weight".into());
            }
            let out = {
                let a = args[0].borrow();
                let b = args[1].borrow();
                a.data.conv1d(&b.data, *stride, *padding)?
            };
            Ok(TNode::derived(
                out,
                args.to_vec(),
                Vjp::Conv1D { stride: *stride, padding: *padding },
            ))
        }
        T::Gather { axis } => {
            if args.len() != 2 {
                return Err("gather expects data and indices".into());
            }
            let out = {
                let a = args[0].borrow();
                let b = args[1].borrow();
                a.data.gather(&b.data, *axis)?
            };
            let shape = args[0].borrow().data.shape.clone();
            Ok(TNode::derived(out, args.to_vec(), Vjp::Gather { axis: *axis, input_shape: shape }))
        }
        T::BroadcastTo { shape } => {
            let input = args[0].borrow();
            let target = resolve_dims(shape, &input.data.shape)?;
            let out = input.data.broadcast_to(&target)?;
            Ok(TNode::derived(
                out,
                args.to_vec(),
                Vjp::Broadcast { input_shape: input.data.shape.clone() },
            ))
        }
        T::Dropout { p } => {
            let input = args[0].borrow();
            let out = input.data.dropout(*p, training, seed);
            let mask = if training && *p > 0.0 {
                // Recover the exact mask: out / in where in != 0 would be
                // undefined for zeros, so build it from the same RNG state.
                dropout_mask(&input.data, *p, seed)
            } else {
                Tensor::ones(input.data.dtype, input.data.shape.clone())
            };
            Ok(TNode::derived(out, args.to_vec(), Vjp::Dropout { mask }))
        }
    }
}

fn dropout_mask(input: &Tensor, p: f64, seed: u64) -> Tensor {
    // Mirrors Tensor::dropout so the mask matches the forward pass exactly.
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let scale = 1.0 / (1.0 - p).max(1e-12);
    let vals: Vec<f64> = (0..input.numel())
        .map(|_| if next() < p { 0.0 } else { scale })
        .collect();
    Tensor::from_f64(input.dtype, input.shape.clone(), vals).unwrap()
}

fn resolve_dims(dims: &[Dim], current: &[usize]) -> Result<Vec<usize>, String> {
    let mut out = Vec::with_capacity(dims.len());
    for (i, d) in dims.iter().enumerate() {
        match d {
            Dim::Static(n) => out.push(*n),
            Dim::Unknown => out.push(*current.get(i).unwrap_or(&0)),
            Dim::Symbol(_) => return Err("cannot broadcast to a symbolic dimension at runtime".into()),
        }
    }
    Ok(out)
}

fn reshape_static(t: &Tensor, shape: &[usize]) -> Result<Tensor, String> {
    let dims: Vec<Dim> = shape.iter().map(|d| Dim::Static(*d)).collect();
    t.reshape(&dims)
}

pub fn matmul_general(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    let a_vec = a.rank() == 1;
    let b_vec = b.rank() == 1;
    let mut av = a.clone();
    let mut bv = b.clone();
    if a_vec {
        av = reshape_static(a, &[1, a.shape[0]])?;
    }
    if b_vec {
        bv = reshape_static(b, &[b.shape[0], 1])?;
    }
    let out = av.matmul(&bv)?;
    if a_vec && b_vec {
        return Tensor::from_f64(out.dtype, vec![], vec![out.f64_at(0)]);
    }
    if a_vec {
        let shape = out.shape[1..].to_vec();
        return Tensor::from_f64(out.dtype, shape, out.to_f64_vec());
    }
    if b_vec {
        let shape = out.shape[..out.rank() - 1].to_vec();
        return Tensor::from_f64(out.dtype, shape, out.to_f64_vec());
    }
    Ok(out)
}

fn sum_axes(t: &Tensor, axes: &[usize], keepdim: bool) -> Result<Tensor, String> {
    t.reduce(&TensorOp::Sum { axes: axes.to_vec(), keepdim })
}

fn unbroadcast(g: &Tensor, target: &[usize]) -> Result<Tensor, String> {
    let mut cur = g.clone();
    while cur.rank() > target.len() {
        cur = sum_axes(&cur, &[0], false)?;
    }
    for i in 0..target.len() {
        if target[i] == 1 && cur.shape[i] != 1 {
            cur = sum_axes(&cur, &[i], true)?;
        }
    }
    if cur.shape != target {
        cur = reshape_static(&cur, target)?;
    }
    Ok(cur)
}

fn expand_reduced(
    g: &Tensor,
    input_shape: &[usize],
    axes: &[usize],
    keepdim: bool,
) -> Result<Tensor, String> {
    let mut cur = g.clone();
    if !keepdim {
        let mut sorted = axes.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        for a in sorted {
            if a <= cur.rank() {
                let mut shape = cur.shape.clone();
                shape.insert(a, 1);
                cur = reshape_static(&cur, &shape)?;
            }
        }
    }
    if cur.shape != input_shape {
        cur = cur.broadcast_to(input_shape)?;
    }
    Ok(cur)
}

fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    a.elementwise(&TensorOp::Sub, b)
}

fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    a.elementwise(&TensorOp::Mul, b)
}

fn div(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    a.elementwise(&TensorOp::Div, b)
}

fn scale(t: &Tensor, s: f64) -> Result<Tensor, String> {
    t.binop_scalar(neura_ir::ops::BinOp::Mul, s)
}

fn mask_where(cond: &Tensor, t: &Tensor) -> Result<Tensor, String> {
    if cond.shape != t.shape {
        return Err("internal: mask shape mismatch".into());
    }
    let mut vals = Vec::with_capacity(t.numel());
    for i in 0..t.numel() {
        vals.push(if cond.f64_at(i) != 0.0 { t.f64_at(i) } else { 0.0 });
    }
    Tensor::from_f64(t.dtype, t.shape.clone(), vals)
}

fn transpose_last_two(t: &Tensor) -> Result<Tensor, String> {
    if t.rank() < 2 {
        return Err("transpose requires rank >= 2".into());
    }
    t.transpose(t.rank() - 2, t.rank() - 1)
}

fn gelu_grad(x: &Tensor, g: &Tensor) -> Result<Tensor, String> {
    let c = 0.797_884_560_802_865_4;
    let mut vals = Vec::with_capacity(x.numel());
    for i in 0..x.numel() {
        let xv = x.f64_at(i);
        let u = c * (xv + 0.044_715 * xv * xv * xv);
        let t = u.tanh();
        let du = c * (1.0 + 3.0 * 0.044_715 * xv * xv);
        let d = 0.5 * (1.0 + t) + 0.5 * xv * (1.0 - t * t) * du;
        vals.push(d * g.f64_at(i));
    }
    Tensor::from_f64(g.dtype, g.shape.clone(), vals)
}

fn scatter_slice(
    g: &Tensor,
    input_shape: &[usize],
    axis: usize,
    start: usize,
    end: usize,
) -> Result<Tensor, String> {
    let mut out = Tensor::zeros(g.dtype, input_shape.to_vec());
    let len = input_shape[axis];
    let outer: usize = input_shape[..axis].iter().product();
    let inner: usize = input_shape[axis + 1..].iter().product();
    let slen = end.saturating_sub(start).min(len.saturating_sub(start));
    for o in 0..outer {
        for i in 0..slen {
            for k in 0..inner {
                let dst = (o * len + start + i) * inner + k;
                let src = (o * slen + i) * inner + k;
                if dst < out.numel() && src < g.numel() {
                    let v = g.f64_at(src);
                    let mut vals = out.to_f64_vec();
                    vals[dst] = v;
                    out = Tensor::from_f64(out.dtype, out.shape.clone(), vals)?;
                }
            }
        }
    }
    Ok(out)
}

fn conv2d_vjp(
    kind: &Vjp,
    out: &Tensor,
    g: &Tensor,
    inputs: &[TRef],
    index: usize,
) -> Result<Tensor, String> {
    let (stride, padding) = match kind {
        Vjp::Conv2D { stride, padding } => (*stride, *padding),
        _ => unreachable!(),
    };
    let (x, w) = {
        let a = inputs[0].borrow();
        let b = inputs[1].borrow();
        (a.data.clone(), b.data.clone())
    };
    if index == 0 {
        let (n, c, h, wd) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
        let (o, _, kh, kw) = (w.shape[0], w.shape[1], w.shape[2], w.shape[3]);
        let (out_h, out_w) = (out.shape[2], out.shape[3]);
        let mut dx = vec![0.0f64; x.numel()];
        for ni in 0..n {
            for oi in 0..o {
                for y in 0..out_h {
                    for xx in 0..out_w {
                        let gv = g.f64_at(((ni * o + oi) * out_h + y) * out_w + xx);
                        for ci in 0..c {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let iy = (y * stride.0 + ky) as isize - padding.0 as isize;
                                    let ix = (xx * stride.1 + kx) as isize - padding.1 as isize;
                                    if iy < 0 || ix < 0 || iy >= h as isize || ix >= wd as isize {
                                        continue;
                                    }
                                    let wv = w.f64_at(((oi * c + ci) * kh + ky) * kw + kx);
                                    dx[((ni * c + ci) * h + iy as usize) * wd + ix as usize] += gv * wv;
                                }
                            }
                        }
                    }
                }
            }
        }
        Tensor::from_f64(x.dtype, x.shape.clone(), dx)
    } else {
        let (n, c, h, wd) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
        let (o, _, kh, kw) = (w.shape[0], w.shape[1], w.shape[2], w.shape[3]);
        let (out_h, out_w) = (out.shape[2], out.shape[3]);
        let mut dw = vec![0.0f64; w.numel()];
        for ni in 0..n {
            for oi in 0..o {
                for y in 0..out_h {
                    for xx in 0..out_w {
                        let gv = g.f64_at(((ni * o + oi) * out_h + y) * out_w + xx);
                        for ci in 0..c {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let iy = (y * stride.0 + ky) as isize - padding.0 as isize;
                                    let ix = (xx * stride.1 + kx) as isize - padding.1 as isize;
                                    if iy < 0 || ix < 0 || iy >= h as isize || ix >= wd as isize {
                                        continue;
                                    }
                                    let xv = x.f64_at(((ni * c + ci) * h + iy as usize) * wd + ix as usize);
                                    dw[((oi * c + ci) * kh + ky) * kw + kx] += gv * xv;
                                }
                            }
                        }
                    }
                }
            }
        }
        Tensor::from_f64(w.dtype, w.shape.clone(), dw)
    }
}

fn conv1d_vjp(
    kind: &Vjp,
    out: &Tensor,
    g: &Tensor,
    inputs: &[TRef],
    index: usize,
) -> Result<Tensor, String> {
    let (stride, padding) = match kind {
        Vjp::Conv1D { stride, padding } => (*stride, *padding),
        _ => unreachable!(),
    };
    let (x, w) = {
        let a = inputs[0].borrow();
        let b = inputs[1].borrow();
        (a.data.clone(), b.data.clone())
    };
    let (n, c, len) = (x.shape[0], x.shape[1], x.shape[2]);
    let (o, _, kw) = (w.shape[0], w.shape[1], w.shape[2]);
    let out_len = out.shape[2];
    if index == 0 {
        let mut dx = vec![0.0f64; x.numel()];
        for ni in 0..n {
            for oi in 0..o {
                for p in 0..out_len {
                    let gv = g.f64_at((ni * o + oi) * out_len + p);
                    for ci in 0..c {
                        for k in 0..kw {
                            let ip = (p * stride + k) as isize - padding as isize;
                            if ip < 0 || ip >= len as isize {
                                continue;
                            }
                            let wv = w.f64_at((oi * c + ci) * kw + k);
                            dx[(ni * c + ci) * len + ip as usize] += gv * wv;
                        }
                    }
                }
            }
        }
        Tensor::from_f64(x.dtype, x.shape.clone(), dx)
    } else {
        let mut dw = vec![0.0f64; w.numel()];
        for ni in 0..n {
            for oi in 0..o {
                for p in 0..out_len {
                    let gv = g.f64_at((ni * o + oi) * out_len + p);
                    for ci in 0..c {
                        for k in 0..kw {
                            let ip = (p * stride + k) as isize - padding as isize;
                            if ip < 0 || ip >= len as isize {
                                continue;
                            }
                            let xv = x.f64_at((ni * c + ci) * len + ip as usize);
                            dw[(oi * c + ci) * kw + k] += gv * xv;
                        }
                    }
                }
            }
        }
        Tensor::from_f64(w.dtype, w.shape.clone(), dw)
    }
}

fn gather_vjp(g: &Tensor, indices: &Tensor, input_shape: &[usize], axis: usize) -> Result<Tensor, String> {
    let mut out = Tensor::zeros(g.dtype, input_shape.to_vec());
    let mut vals = out.to_f64_vec();
    let outer: usize = input_shape[..axis].iter().product();
    let inner: usize = input_shape[axis + 1..].iter().product();
    let table_len = input_shape[axis];
    let idx_n = indices.numel();
    for o in 0..outer {
        for i in 0..idx_n {
            let pos = indices.i64_at(i);
            if pos < 0 || pos as usize >= table_len {
                return Err(format!("gather index {} out of range", pos));
            }
            for k in 0..inner {
                let gi = (o * idx_n + i) * inner + k;
                let dst = (o * table_len + pos as usize) * inner + k;
                vals[dst] += g.f64_at(gi);
            }
        }
    }
    out = Tensor::from_f64(g.dtype, input_shape.to_vec(), vals)?;
    Ok(out)
}

/// Compute the gradient for input `index` of an operation.
pub fn vjp(
    kind: &Vjp,
    out: &Tensor,
    g: &Tensor,
    inputs: &[TRef],
    index: usize,
) -> Result<Tensor, String> {
    match kind {
        Vjp::Add => unbroadcast(g, &inputs[index].borrow().data.shape),
        Vjp::Sub => {
            let u = unbroadcast(g, &inputs[index].borrow().data.shape)?;
            if index == 0 {
                Ok(u)
            } else {
                Ok(scale(&u, -1.0)?)
            }
        }
        Vjp::Mul => {
            let (a, b) = {
                let x = inputs[0].borrow();
                let y = inputs[1].borrow();
                (x.data.clone(), y.data.clone())
            };
            let other = if index == 0 { b } else { a };
            let target = inputs[index].borrow().data.shape.clone();
            unbroadcast(&mul(g, &other)?, &target)
        }
        Vjp::Div => {
            let (a, b) = {
                let x = inputs[0].borrow();
                let y = inputs[1].borrow();
                (x.data.clone(), y.data.clone())
            };
            let target = inputs[index].borrow().data.shape.clone();
            if index == 0 {
                unbroadcast(&div(g, &b)?, &target)
            } else {
                let num = mul(&mul(g, &a)?, &scale(&mul(&b, &b)?, -1.0)?)?;
                unbroadcast(&num, &target)
            }
        }
        Vjp::MatMul => {
            let (a, b) = {
                let x = inputs[0].borrow();
                let y = inputs[1].borrow();
                (x.data.clone(), y.data.clone())
            };
            let a_vec = a.rank() == 1;
            let b_vec = b.rank() == 1;
            let mut g2 = g.clone();
            if a_vec && b_vec {
                g2 = reshape_static(&g2, &[1, 1])?;
            } else if a_vec {
                g2 = reshape_static(&g2, &{
                    let mut s = vec![1];
                    s.extend_from_slice(&g2.shape);
                    s
                })?;
            } else if b_vec {
                g2 = reshape_static(&g2, &{
                    let mut s = g2.shape.clone();
                    s.push(1);
                    s
                })?;
            }
            if index == 0 {
                let bt = transpose_last_two(&b)?;
                let raw = crate::autograd::matmul_general(&g2, &bt)?;
                let mut grad = raw;
                if a_vec {
                    grad = reshape_static(&grad, &a.shape)?;
                } else if b_vec {
                    let mut target = a.shape.clone();
                    target.pop();
                    let mut s = grad.shape.clone();
                    if s.last() == Some(&1) {
                        s.pop();
                    }
                    grad = reshape_static(&grad, &s)?;
                    let _ = target;
                }
                unbroadcast(&grad, &a.shape)
            } else {
                let at = transpose_last_two(&a)?;
                let raw = crate::autograd::matmul_general(&at, &g2)?;
                let mut grad = raw;
                if b_vec {
                    grad = reshape_static(&grad, &b.shape)?;
                } else if a_vec {
                    let mut s = grad.shape.clone();
                    if s.first() == Some(&1) {
                        s.remove(0);
                    }
                    grad = reshape_static(&grad, &s)?;
                }
                unbroadcast(&grad, &b.shape)
            }
        }
        Vjp::Transpose { dim0, dim1 } => g.transpose(*dim0, *dim1),
        Vjp::Reshape { shape } => reshape_static(g, shape),
        Vjp::Sum { axes, keepdim, input_shape } => {
            expand_reduced(g, input_shape, axes, *keepdim)
        }
        Vjp::Mean { axes, keepdim, input_shape } => {
            let expanded = expand_reduced(g, input_shape, axes, *keepdim)?;
            let mut count = 1usize;
            for a in axes {
                count *= input_shape.get(*a).copied().unwrap_or(1);
            }
            scale(&expanded, 1.0 / count.max(1) as f64)
        }
        Vjp::ReduceMax { axis, keepdim, input_shape }
        | Vjp::ReduceMin { axis, keepdim, input_shape } => {
            let input = inputs[0].borrow().data.clone();
            let expanded = expand_reduced(g, input_shape, &axis.map(|a| vec![a]).unwrap_or_else(|| (0..input.rank()).collect()), *keepdim)?;
            let mask = equal_mask(&input, &expanded)?;
            mask_where(&mask, &expanded)
        }
        Vjp::Slice { axis, start, end, input_shape } => {
            scatter_slice(g, input_shape, *axis, *start, *end)
        }
        Vjp::ReLU => {
            let input = inputs[0].borrow().data.clone();
            let mask: Vec<f64> = (0..input.numel()).map(|i| if input.f64_at(i) > 0.0 { 1.0 } else { 0.0 }).collect();
            let m = Tensor::from_f64(g.dtype, g.shape.clone(), mask)?;
            mul(g, &m)
        }
        Vjp::GELU => {
            let input = inputs[0].borrow().data.clone();
            gelu_grad(&input, g)
        }
        Vjp::Sigmoid => {
            let d = mul(out, &sub(&Tensor::full(out.dtype, out.shape.clone(), 1.0), out)?)?;
            mul(g, &d)
        }
        Vjp::Tanh => {
            let d = sub(&Tensor::full(out.dtype, out.shape.clone(), 1.0), &mul(out, out)?)?;
            mul(g, &d)
        }
        Vjp::Exp => mul(g, out),
        Vjp::Log => {
            let input = inputs[0].borrow().data.clone();
            div(g, &input)
        }
        Vjp::Sqrt => {
            let d = mul(out, &Tensor::full(out.dtype, out.shape.clone(), 2.0))?;
            div(g, &d)
        }
        Vjp::Neg => scale(g, -1.0),
        Vjp::Softmax { axis } => {
            let y = out.clone();
            let gy = mul(g, &y)?;
            let s = sum_axes(&gy, &[*axis], true)?;
            let inner = sub(g, &s)?;
            mul(&y, &inner)
        }
        Vjp::Conv2D { .. } => conv2d_vjp(kind, out, g, inputs, index),
        Vjp::Conv1D { .. } => conv1d_vjp(kind, out, g, inputs, index),
        Vjp::Gather { axis, input_shape } => {
            let idx = inputs[1].borrow().data.clone();
            gather_vjp(g, &idx, input_shape, *axis)
        }
        Vjp::Dropout { mask } => mul(g, mask),
        Vjp::Broadcast { input_shape } => unbroadcast(g, input_shape),
    }
}

fn equal_mask(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.shape != b.shape {
        return Err("internal: mask shape mismatch".into());
    }
    let vals: Vec<f64> = (0..a.numel())
        .map(|i| if a.f64_at(i) == b.f64_at(i) { 1.0 } else { 0.0 })
        .collect();
    Tensor::from_f64(DType::F64, a.shape.clone(), vals)
}

fn accumulate(node: &TRef, g: Tensor) -> Result<(), String> {
    let dtype = node.borrow().data.dtype;
    let g = if g.dtype == dtype { g } else { g.cast(dtype) };
    let mut n = node.borrow_mut();
    match &n.grad {
        Some(existing) => {
            let sum = existing.elementwise(&TensorOp::Add, &g)?;
            n.grad = Some(sum);
        }
        None => n.grad = Some(g),
    }
    Ok(())
}

fn collect(root: &TRef, order: &mut Vec<TRef>, seen: &mut HashSet<usize>) {
    let key = cache_key(root);
    if !seen.insert(key) {
        return;
    }
    if !root.borrow().requires_grad {
        return;
    }
    let inputs = root.borrow().inputs.clone();
    for i in &inputs {
        collect(i, order, seen);
    }
    order.push(root.clone());
}

/// Run reverse-mode differentiation from `root`, accumulating into `.grad`.
pub fn backward(root: &TRef) -> Result<(), String> {
    if !root.borrow().requires_grad {
        return Err("backward() called on a tensor that does not require gradients".into());
    }
    let mut order = Vec::new();
    let mut seen = HashSet::new();
    collect(root, &mut order, &mut seen);

    let root_shape = root.borrow().data.shape.clone();
    let root_dtype = root.borrow().data.dtype;
    let seed = Tensor::full(root_dtype, root_shape, 1.0);
    accumulate(root, seed)?;

    for node in order.iter().rev() {
        let (g, inputs, vjp_kind, out_data) = {
            let n = node.borrow();
            match (&n.grad, &n.vjp) {
                (Some(g), Some(v)) => (g.clone(), n.inputs.clone(), v.clone(), n.data.clone()),
                _ => continue,
            }
        };
        for (i, inp) in inputs.iter().enumerate() {
            if !inp.borrow().requires_grad {
                continue;
            }
            let gi = vjp(&vjp_kind, &out_data, &g, &inputs, i)?;
            accumulate(inp, gi)?;
        }
    }
    Ok(())
}

/// Number of graph nodes reachable from `root` (used by inspect/profile output).
pub fn graph_size(root: &TRef) -> usize {
    let mut order = Vec::new();
    let mut seen = HashSet::new();
    collect(root, &mut order, &mut seen);
    order.len()
}

/// Clear gradients on the reachable graph.
pub fn zero_grad(root: &TRef) {
    let mut order = Vec::new();
    let mut seen = HashSet::new();
    collect(root, &mut order, &mut seen);
    for n in order {
        n.borrow_mut().grad = None;
    }
}

/// Gradient check by central finite differences for a scalar function.
pub fn numeric_grad(
    f: impl Fn(&Tensor) -> Result<Tensor, String>,
    x: &Tensor,
    eps: f64,
) -> Result<Tensor, String> {
    let mut vals = Vec::with_capacity(x.numel());
    for i in 0..x.numel() {
        let mut plus = x.to_f64_vec();
        plus[i] += eps;
        let mut minus = x.to_f64_vec();
        minus[i] -= eps;
        let fp = f(&Tensor::from_f64(x.dtype, x.shape.clone(), plus)?)?;
        let fm = f(&Tensor::from_f64(x.dtype, x.shape.clone(), minus)?)?;
        let sp = fp.scalar_f64().ok_or("gradient check needs a scalar output")?;
        let sm = fm.scalar_f64().ok_or("gradient check needs a scalar output")?;
        vals.push((sp - sm) / (2.0 * eps));
    }
    Tensor::from_f64(x.dtype, x.shape.clone(), vals)
}

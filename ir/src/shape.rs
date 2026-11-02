//! Shape algebra: broadcasting, matmul, reductions, convolution, reshape.

use crate::types::Dim;

pub fn broadcast_shapes(a: &[Dim], b: &[Dim]) -> Result<Vec<Dim>, String> {
    let rank = a.len().max(b.len());
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        let da = if i < rank - a.len() { Dim::Static(1) } else { a[i - (rank - a.len())].clone() };
        let db = if i < rank - b.len() { Dim::Static(1) } else { b[i - (rank - b.len())].clone() };
        out.push(broadcast_dim(&da, &db).map_err(|e| {
            format!(
                "cannot broadcast shapes [{}] and [{}]: {}",
                fmt_dims(a),
                fmt_dims(b),
                e
            )
        })?);
    }
    Ok(out)
}

pub fn broadcast_dim(a: &Dim, b: &Dim) -> Result<Dim, String> {
    use Dim::*;
    match (a, b) {
        (Static(1), other) | (other, Static(1)) => Ok(other.clone()),
        (Static(x), Static(y)) if x == y => Ok(Static(*x)),
        (Static(x), Static(y)) => Err(format!("dimension mismatch {} vs {}", x, y)),
        (Unknown, other) | (other, Unknown) => Ok(other.clone()),
        (Symbol(x), Symbol(y)) if x == y => Ok(Symbol(x.clone())),
        (Symbol(_), _) | (_, Symbol(_)) => Ok(Unknown),
    }
}

pub fn matmul_shape(a: &[Dim], b: &[Dim]) -> Result<(Vec<Dim>, Vec<Dim>), String> {
    if a.len() < 2 || b.len() < 2 {
        return Err("matmul requires both operands to have rank >= 2".into());
    }
    let m = a[a.len() - 2].clone();
    let k1 = a[a.len() - 1].clone();
    let k2 = b[b.len() - 2].clone();
    let n = b[b.len() - 1].clone();
    match (&k1, &k2) {
        (Dim::Static(x), Dim::Static(y)) if x != y => {
            return Err(format!("matmul inner dimension mismatch {} vs {}", x, y));
        }
        (Dim::Symbol(x), Dim::Symbol(y)) if x != y => {
            return Err(format!("matmul inner symbols differ: {} vs {}", x, y));
        }
        _ => {}
    }
    let batch = broadcast_shapes(&a[..a.len() - 2], &b[..b.len() - 2])?;
    let mut out = batch.clone();
    out.push(m);
    out.push(n);
    Ok((out, batch))
}

pub fn merge_dims(mut shape: Vec<Dim>, _batch: Vec<Dim>) -> Vec<Dim> {
    // `matmul_shape` already returns the full output shape; kept as a named
    // normalization point for backends that materialise batch dims separately.
    shape.shrink_to_fit();
    shape
}

pub fn reduce_shape(shape: &[Dim], axes: &[usize], keepdim: bool) -> Vec<Dim> {
    let mut out = Vec::new();
    for (i, d) in shape.iter().enumerate() {
        if axes.contains(&i) {
            if keepdim {
                out.push(Dim::Static(1));
            }
        } else {
            out.push(d.clone());
        }
    }
    out
}

pub fn conv_out_dim(input: &Dim, kernel: &Dim, stride: usize, padding: usize) -> Dim {
    match (input.concrete(), kernel.concrete()) {
        (Some(i), Some(k)) => {
            let s = stride.max(1);
            Dim::Static((i + 2 * padding).saturating_sub(k) / s + 1)
        }
        _ => Dim::Unknown,
    }
}

/// Resolve a reshape request against an input shape. At most one `-1`
/// (represented as `Dim::Unknown`) may appear.
pub fn resolve_reshape(input: &[Dim], dims: &[Dim]) -> Result<Vec<Dim>, String> {
    let mut out: Vec<Dim> = Vec::with_capacity(dims.len());
    let mut unknown_at: Option<usize> = None;
    let mut known: usize = 1;
    let total: Option<usize> = if input.iter().all(|d| d.concrete().is_some()) {
        Some(input.iter().map(|d| d.concrete().unwrap()).product())
    } else {
        None
    };
    for (i, d) in dims.iter().enumerate() {
        match d {
            Dim::Unknown => {
                if unknown_at.is_some() {
                    return Err("reshape may contain at most one inferred dimension (-1)".into());
                }
                unknown_at = Some(i);
                out.push(Dim::Unknown);
            }
            Dim::Static(n) => {
                known = known.saturating_mul(*n);
                out.push(Dim::Static(*n));
            }
            Dim::Symbol(s) => {
                out.push(Dim::Symbol(s.clone()));
            }
        }
    }
    if let (Some(idx), Some(total)) = (unknown_at, total) {
        if known > 0 && total % known == 0 {
            out[idx] = Dim::Static(total / known);
        } else {
            return Err(format!(
                "cannot reshape tensor with {} elements into {:?}",
                total, dims
            ));
        }
    }
    Ok(out)
}

fn fmt_dims(dims: &[Dim]) -> String {
    dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcasting_rules() {
        let a = vec![Dim::Static(4), Dim::Static(1), Dim::Static(8)];
        let b = vec![Dim::Static(3), Dim::Static(1)];
        let out = broadcast_shapes(&a, &b).unwrap();
        assert_eq!(
            out,
            vec![Dim::Static(4), Dim::Static(3), Dim::Static(8)]
        );
        assert!(broadcast_shapes(&[Dim::Static(2)], &[Dim::Static(3)]).is_err());
    }

    #[test]
    fn matmul_and_reshape() {
        let (out, _) = matmul_shape(&[Dim::Static(2), Dim::Static(3)], &[Dim::Static(3), Dim::Static(5)])
            .unwrap();
        assert_eq!(out, vec![Dim::Static(2), Dim::Static(5)]);
        let r = resolve_reshape(&[Dim::Static(2), Dim::Static(6)], &[Dim::Static(3), Dim::Unknown]).unwrap();
        assert_eq!(r, vec![Dim::Static(3), Dim::Static(4)]);
    }

    #[test]
    fn conv_shapes() {
        let out = crate::ops::infer_shape(
            &crate::ops::TensorOp::Conv2D { stride: (1, 1), padding: (0, 0) },
            &[
                crate::types::TensorTy::new(crate::types::DType::F32, vec![
                    Dim::Static(1),
                    Dim::Static(3),
                    Dim::Static(8),
                    Dim::Static(8),
                ]),
                crate::types::TensorTy::new(crate::types::DType::F32, vec![
                    Dim::Static(4),
                    Dim::Static(3),
                    Dim::Static(3),
                    Dim::Static(3),
                ]),
            ],
        )
        .unwrap();
        assert_eq!(out.shape, vec![Dim::Static(1), Dim::Static(4), Dim::Static(6), Dim::Static(6)]);
    }
}

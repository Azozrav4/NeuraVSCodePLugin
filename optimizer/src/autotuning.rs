//! Bounded autotuning for elementwise kernels.
//!
//! Two real implementations are benchmarked for each fused expression: the
//! eager sequence of tensor operations and a single fused pass over the output.
//! Results are cached, the number of measurements per compilation is bounded,
//! and chains whose shapes are dynamic are assumed to benefit from fusion
//! (which always removes intermediate buffers).

use neura_backends::tensor::Tensor;
use neura_ir::ops::{FusedExpr, TensorOp};
use neura_ir::types::DType;
use std::collections::HashMap;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct TuneDecision {
    pub use_fused: bool,
    pub measured: bool,
    pub eager_ns: u128,
    pub fused_ns: u128,
    pub elements: usize,
}

pub struct Autotuner {
    cache: HashMap<String, TuneDecision>,
    budget: usize,
    measurements: usize,
}

impl Default for Autotuner {
    fn default() -> Self {
        Autotuner::new(8)
    }
}

fn gen_values(n: usize) -> Vec<f64> {
    // Deterministic pseudo-random input in [-1, 1].
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.push((state % 2001) as f64 / 1000.0 - 1.0);
    }
    out
}

fn eager(expr: &FusedExpr, args: &[Tensor]) -> Result<Tensor, String> {
    Ok(match expr {
        FusedExpr::Input(i) => args
            .get(*i as usize)
            .cloned()
            .ok_or_else(|| format!("missing input {}", i))?,
        FusedExpr::Const(v) => Tensor::from_f64(DType::F32, vec![], vec![*v])?,
        FusedExpr::Add(a, b) => eager(a, args)?.elementwise(&TensorOp::Add, &eager(b, args)?)?,
        FusedExpr::Sub(a, b) => eager(a, args)?.elementwise(&TensorOp::Sub, &eager(b, args)?)?,
        FusedExpr::Mul(a, b) => eager(a, args)?.elementwise(&TensorOp::Mul, &eager(b, args)?)?,
        FusedExpr::Div(a, b) => eager(a, args)?.elementwise(&TensorOp::Div, &eager(b, args)?)?,
        FusedExpr::Neg(a) => eager(a, args)?.unary(&TensorOp::Neg)?,
        FusedExpr::ReLU(a) => eager(a, args)?.unary(&TensorOp::ReLU)?,
        FusedExpr::GELU(a) => eager(a, args)?.unary(&TensorOp::GELU)?,
        FusedExpr::Sigmoid(a) => eager(a, args)?.unary(&TensorOp::Sigmoid)?,
        FusedExpr::Tanh(a) => eager(a, args)?.unary(&TensorOp::Tanh)?,
        FusedExpr::Exp(a) => eager(a, args)?.unary(&TensorOp::Exp)?,
        FusedExpr::Log(a) => eager(a, args)?.unary(&TensorOp::Log)?,
        FusedExpr::Sqrt(a) => eager(a, args)?.unary(&TensorOp::Sqrt)?,
    })
}

fn element_at(t: &Tensor, out_idx: &[usize]) -> f64 {
    if t.rank() == 0 {
        return t.to_f64_vec().first().copied().unwrap_or(0.0);
    }
    let strides = t.strides();
    let offset = out_idx.len() - t.rank();
    let mut flat = 0usize;
    for d in 0..t.rank() {
        let i = if t.shape[d] == 1 { 0 } else { out_idx[offset + d] };
        flat += i * strides[d];
    }
    t.to_f64_vec().get(flat).copied().unwrap_or(0.0)
}

fn fused(expr: &FusedExpr, args: &[Tensor]) -> Result<Tensor, String> {
    let mut shape: Vec<usize> = Vec::new();
    for a in args {
        shape = Tensor::broadcast_pair(&shape, &a.shape)?;
    }
    let n: usize = shape.iter().product();
    let mut values = Vec::with_capacity(n);
    let mut idx = vec![0usize; shape.len()];
    for flat in 0..n {
        let mut rem = flat;
        for d in (0..shape.len()).rev() {
            idx[d] = rem % shape[d].max(1);
            rem /= shape[d].max(1);
        }
        let inputs: Vec<f64> = args.iter().map(|a| element_at(a, &idx)).collect();
        values.push(expr.eval(&inputs));
    }
    Tensor::from_f64(DType::F32, shape, values)
}

fn time<F: FnMut()>(reps: usize, mut f: F) -> u128 {
    let start = Instant::now();
    for _ in 0..reps {
        f();
    }
    start.elapsed().as_nanos().max(1)
}

impl Autotuner {
    pub fn new(budget: usize) -> Autotuner {
        Autotuner { cache: HashMap::new(), budget, measurements: 0 }
    }

    pub fn measurements(&self) -> usize {
        self.measurements
    }

    pub fn cached(&self) -> usize {
        self.cache.len()
    }

    /// Decide whether a fused kernel is preferable for this expression.
    ///
    /// The measurement is skipped for dynamic shapes and for very large or
    /// tiny tensors, where the answer is known and benchmarking would only
    /// slow compilation down.
    pub fn decide(&mut self, expr: &FusedExpr, input_shapes: &[Vec<usize>]) -> TuneDecision {
        let key = format!("{}:{:?}", expr, input_shapes);
        if let Some(d) = self.cache.get(&key) {
            return d.clone();
        }
        if input_shapes.is_empty() {
            let d = TuneDecision {
                use_fused: true,
                measured: false,
                eager_ns: 0,
                fused_ns: 0,
                elements: 0,
            };
            self.cache.insert(key, d.clone());
            return d;
        }
        let mut out_shape: Vec<usize> = Vec::new();
        for s in input_shapes {
            match Tensor::broadcast_pair(&out_shape, s) {
                Ok(v) => out_shape = v,
                Err(_) => {
                    let d = TuneDecision {
                        use_fused: true,
                        measured: false,
                        eager_ns: 0,
                        fused_ns: 0,
                        elements: 0,
                    };
                    self.cache.insert(key, d.clone());
                    return d;
                }
            }
        }
        let elements: usize = out_shape.iter().product();
        if elements < 16 || elements > 100_000 || self.measurements >= self.budget {
            let d = TuneDecision {
                use_fused: true,
                measured: false,
                eager_ns: 0,
                fused_ns: 0,
                elements,
            };
            self.cache.insert(key, d.clone());
            return d;
        }
        let args: Vec<Tensor> = input_shapes
            .iter()
            .map(|s| {
                let n: usize = s.iter().product();
                Tensor::from_f64(DType::F32, s.clone(), gen_values(n)).expect("valid shape")
            })
            .collect();
        let reps = (200_000 / elements.max(1)).clamp(1, 20);
        let eager_ns = time(reps, || {
            let _ = eager(expr, &args);
        });
        let fused_ns = time(reps, || {
            let _ = fused(expr, &args);
        });
        self.measurements += 1;
        let d = TuneDecision {
            use_fused: fused_ns <= eager_ns,
            measured: true,
            eager_ns: eager_ns / reps as u128,
            fused_ns: fused_ns / reps as u128,
            elements,
        };
        self.cache.insert(key, d.clone());
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measures_and_caches_a_chain() {
        let expr = FusedExpr::GELU(Box::new(FusedExpr::ReLU(Box::new(FusedExpr::Input(0)))));
        let mut tuner = Autotuner::new(4);
        let d = tuner.decide(&expr, &[vec![32, 32]]);
        assert!(d.measured);
        assert!(d.eager_ns > 0 && d.fused_ns > 0);
        assert_eq!(d.elements, 1024);
        let again = tuner.decide(&expr, &[vec![32, 32]]);
        assert_eq!(again.eager_ns, d.eager_ns);
        assert_eq!(tuner.cached(), 1);
        assert_eq!(tuner.measurements(), 1);
    }

    #[test]
    fn dynamic_shapes_skip_measurement() {
        let expr = FusedExpr::Exp(Box::new(FusedExpr::Input(0)));
        let mut tuner = Autotuner::new(4);
        let d = tuner.decide(&expr, &[]);
        assert!(d.use_fused);
        assert!(!d.measured);
        assert_eq!(tuner.measurements(), 0);
    }
}

//! Neura runtime.
//!
//! The compiler lowers Neura programs to the Neura Execution IR (EIR); this
//! crate executes that bytecode on the native CPU tensor engine and provides
//! the pieces the language's standard library is built from:
//!
//! * [`value`] — runtime values, tensor handles and reference-counted autograd nodes.
//! * [`autograd`] — reverse-mode automatic differentiation over the EIR op set.
//! * [`layers`] — neural network layers, forward passes, loss and the training loop.
//! * [`optim`] — SGD, Adam, AdamW and RMSProp implemented natively.
//! * [`data`] — CSV dataset loading, shuffling, splitting and batching.
//! * [`support`] — profiler, memory accounting, compilation and autotuning caches.
//! * [`vm`] — the EIR interpreter that ties everything together.
//!
//! Nothing here calls into Python or another ML framework.

pub mod autograd;
pub mod data;
pub mod layers;
pub mod optim;
pub mod support;
pub mod value;
pub mod vm;

pub use data::Dataset;
pub use layers::{Activation, Layer};
pub use optim::{OptKind, Optimizer};
pub use support::{AutotuneCache, CompilationCache, MemoryManager, MemoryStats, Profiler};
pub use value::{format_tensor, tensor, tensor_from_value, trainable, TRef, Value};
pub use vm::{run, RunOutcome, VmOptions};

#[cfg(test)]
mod tests {
    use super::*;
    use neura_backends::tensor::Tensor;
    use neura_ir::ops::TensorOp;
    use neura_ir::types::DType;

    fn scalar(v: f64) -> Tensor {
        Tensor::from_f64(DType::F64, vec![], vec![v]).unwrap()
    }

    /// The autodiff example from the language specification:
    /// `y = x * x + 3 * x` at `x = 2` has gradient `2x + 3 = 7`.
    #[test]
    fn autodiff_example_gradient_is_seven() {
        let x = trainable(scalar(2.0));
        let square = autograd::apply(&TensorOp::Mul, &[x.clone(), x.clone()], false, 1).unwrap();
        let three = tensor(scalar(3.0));
        let scaled = autograd::apply(&TensorOp::Mul, &[three, x.clone()], false, 1).unwrap();
        let y = autograd::apply(&TensorOp::Add, &[square, scaled], false, 1).unwrap();
        autograd::backward(&y).unwrap();
        let grad = x.borrow().grad.clone().expect("gradient was not produced");
        assert_eq!(grad.item().unwrap(), 7.0);
    }
}

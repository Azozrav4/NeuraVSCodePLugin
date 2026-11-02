//! Neura intermediate representations.
//!
//! Four conceptual layers are represented here:
//! * [`nir`]  — typed, ownership-annotated program IR (language level).
//! * [`tir`]  — tensor dataflow DAG with shape/dtype/layout metadata.
//! * [`adir`] — automatic-differentiation graph description.
//! * [`eir`]  — executable register bytecode consumed by the VM.
//!
//! [`mlir`] prints NIR/TIR using MLIR-compatible textual syntax (Neura dialects
//! plus `arith`/`func`/`scf`), so the output can be fed to `mlir-opt` when an
//! MLIR toolchain is available. No MLIR/LLVM implementation is re-created here.

pub mod adir;
pub mod eir;
pub mod mlir;
pub mod nir;
pub mod ops;
pub mod shape;
pub mod tir;
pub mod types;

pub use ops::{BinOp, TensorOp, UnOp};
pub use types::{DType, Dim, FnSig, TensorTy, Ty};

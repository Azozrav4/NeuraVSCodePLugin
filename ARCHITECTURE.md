# Neura Architecture

Neura is a compiled, statically typed language for numerical and machine-learning
work. This document describes the implemented pipeline and, where a dependency is
not available on the host, the exact boundary between what is implemented and what
is blocked.

## Pipeline

```
source (.neura)
  │  compiler::lexer      indentation-sensitive tokens
  │  compiler::parser     recursive-descent AST
  │  compiler::typecheck  semantic analysis, static types, shapes, library typing
  │  compiler::ownership  moves, borrows, last-use, autograd-aware lifetimes
  │  compiler::lowering   typed NIR -> register-machine EIR
  │  optimizer (NAOE)     DCE, folding, CSE, fusion, memory plan, placement,
  │                       precision policy, bounded autotuning
  │  runtime::vm          EIR interpreter with native CPU tensor kernels
  └─ artifact             `.neurabc` (EIR text) + optimization report (JSON)
```

The four IR layers are real data structures (`ir` crate):

* **NIR** (`ir::nir`) — typed tree with explicit `Let`/`Assign`/`If`/`While`/`For`,
  explicit `Move`/`Borrow`, and library calls resolved to canonical builtin names.
* **TIR** (`ir::tir`) — tensor DAG with layout, register, use counts, topo order and
  `needed_for_grad`.
* **ADIR** (`ir::adir`) — VJP rule table per `TensorOp`, gradient accumulators,
  saved-tensor and checkpoint metadata.
* **EIR** (`ir::eir`) — typed register machine, the only form the runtime executes.
  Every tensor instruction carries dtype, shape and requirement-graph metadata.

`ir::mlir` emits the TIR of a module as textual MLIR (`neura_tir` dialect plus
`func`/`arith`/`scf`). It is a textual, inspectable lowering target; MLIR itself is
**not** re-implemented and is not linked in this environment.

## Why there is no linked LLVM/MLIR yet

The host has no `llvm-config`, `mlir-opt`, CUDA or ROCm. Neura therefore:

1. compiles to EIR and executes it with the native Rust CPU tensor engine;
2. emits MLIR text from TIR for inspection and later `mlir-opt`/`llc` hand-off;
3. reports accelerator availability truthfully (`backends::device`), and returns
   explicit "no runtime / no code generator" errors from `cuda`/`rocm` instead of
   pretending to launch kernels.

A linked LLVM backend is an interface boundary, not a hidden substitution: when
LLVM is present the EIR→LLVM lowering replaces interpretation; until then
`neura build` produces the portable `.neurabc` artifact and says so.

## Tensor engine

`backends::tensor` is a dense row-major `Tensor` with `f64`/`i64`/`bool` storage.
Element types are *honest*: storing into `f16`, `bf16`, `i8` rounds/wraps to that
precision immediately. Kernels: elementwise arithmetic with broadcasting, batched
matmul, transpose, reshape, flatten, slice, reductions, softmax, conv1d/conv2d,
gather, dropout, cross-entropy, casting. Parallelism is available through Rayon for
elementwise maps and reductions; numerical results do not depend on thread count.

## Autograd

`runtime::autograd` is a tape/graph over `Rc<RefCell<TensorNode>>`:

* each node stores data, `requires_grad`, accumulated `grad`, input edges and the
  VJP rule that produced it;
* `backward` topologically sorts reachable nodes, seeds the root gradient with ones
  and accumulates vector-Jacobian products in reverse order;
* broadcasting is undone by sum-reduction to the input shape;
* `gather` uses same-rank index semantics (`out.shape == indices.shape`, with the
  coordinate on `axis` replaced by the index value); its VJP scatter-adds the
  upstream gradient at exactly those coordinates. Cross-entropy relies on this to
  select one target logit per row;
* layer forward passes are written with primitive ops, so gradients for Linear,
  GELU, LayerNorm, BatchNorm, Conv and Embedding are produced by the same engine;
* saved tensors are exactly the input `Rc`s held by the graph; `Drop` from EIR and
  Rust's reference counting release them after the last dependent backward node.

## Memory model

`compiler::ownership` enforces the static rules: one owner, multiple shared
borrows, an exclusive mutable borrow only on `mut` bindings, no move while
borrowed, no use after move, and branch/loop state merging. `last_use` is computed
per local during lowering and exposed on EIR locals; NAOE turns last uses into
`Drop` instructions and buffer-slot reuse. Autograd-required tensors are marked
`needs_autograd` so their last use is extended past the matching backward.

## Optimization (NAOE)

`optimizer` operates on EIR. Implemented passes (all report what they changed):

* constant folding and algebraic simplification of scalar operations;
* common subexpression elimination for pure tensor/scalar instructions;
* dead-code elimination with jump-target remapping;
* elementwise operator fusion into `FusedOp` with an inspectable `FusedExpr`;
* last-use `Drop` insertion plus a register-to-buffer-slot plan;
* device placement per instruction using `backends::device::choose_device`
  (explicit policy wins; `auto` compares size, estimated FLOPs and transfer cost);
* precision policy planning (`strict` by default; `mixed_bf16`/`mixed_fp16` only
  when requested, with numerical-error validation helpers);
* bounded autotuning with measured wall time and an in-process result cache.

Modes: `auto` (default), `aggressive`, `debug`, `off`. `debug` keeps all
instructions and records every decision; `off` executes EIR as lowered.

## Runtime

`runtime::vm` executes EIR: frames of registers, function calls, loops, methods,
library calls, tensors, and datasets. `runtime::memory` tracks live bytes, peak
usage, allocation/free counts and pool hits. `runtime::profiler` records per-op
execution counts and nanoseconds; `neura profile` prints the table. `runtime::cache`
stores compiled artifacts keyed by source hash, and `runtime::autotune` caches
measured kernel configurations.

## Backends

* **CPU** — implemented: native kernels, SIMD-friendly loops, Rayon parallelism.
* **CUDA / ROCm** — device discovery probes for runtime libraries; allocation and
  launch return explicit errors if the runtime or the NVVM/ROCDL code generator is
  absent. Interfaces (`GpuBackend`) are shared by both.

The language semantic layer never depends on a GPU, a Python installation, or a
vendor SDK.

## Packages and tooling

* `npack` — `init`, `install`, `build`, `run`, `test`. It parses `npack.toml`,
  resolves `path` dependencies recursively (rejecting cycles and version
  conflicts), resolves version requirements from the offline cache, writes a
  deterministic `npack.lock`, and refuses to build when the lock does not match
  the manifest. Registry downloads are reported as unavailable instead of faked.
  Dependency *symbols* are not yet importable: the compiler's module system covers
  the `neura.*` standard library only (see `LANGUAGE_SPEC.md` §9).
* `tools/formatter` — canonical, idempotent formatter driven by the AST: minimal
  parentheses by precedence, four-space blocks, `elif` reconstruction.
* `cli` — `check`, `build`, `run`, `inspect`, `graph`, `profile`,
  `optimize-report`, plus `fmt` and `version`.
* Diagnostics are structured (`severity`, stable code, span, notes, help) and
  serialisable to JSON (`Diagnostic::to_json`).

## Tests, examples and benchmarks

Tests live next to the code they cover, plus `cli/tests/e2e.rs`, which runs the
real `neura` binary and checks stdout/exit codes for: the required autograd
example (`7.0`), tensor ops, MLIR emission, `.neurabc` build→run round-trip,
diagnostics, formatter idempotence, and training on `examples/datasets/blobs.csv`
with save → load prediction parity.

```
cargo test --workspace
cargo run --release -p neura-runtime --example benchmarks
```

Benchmarks measure the host at run time (no stored numbers). On the development
machine (aarch64, release) they reported approximately: matmul 256³ f32
1.7 GFLOP/s, GELU 1M f32 86 Melem/s, softmax 128×1000 98 Melem/s, conv2d
4×8×32×32 0.46 GFLOP/s, and a 64-sample training step at ~6.3k samples/s. These
are unoptimised reference kernels, not vendor-BLAS figures.

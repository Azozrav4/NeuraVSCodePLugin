# Neura Roadmap

Milestones are ordered as required by the project brief. A milestone is only
marked complete when its implementation is wired into the toolchain and covered by
tests. The `Status` column reflects this repository, not intentions.

| Milestone | Scope | Status |
| --- | --- | --- |
| M0 | Workspace, architecture/spec/roadmap, crate boundaries | complete |
| M1 | Lexer, parser, AST, functions, variables, type checker, diagnostics | complete |
| M2 | Ownership, borrowing, lifetime rules, last-use analysis | complete |
| M3 | Tensor engine: dtypes, shapes, arithmetic, matmul, CPU kernels | complete |
| M4 | NIR/TIR/ADIR/EIR, lowering, MLIR text, runtime VM, portable artifact | complete |
| M5 | Reverse-mode autodiff, accumulation, gradient tests | complete |
| M6 | NVIDIA/AMD interfaces, discovery, memory manager boundary | interface-only (no SDK on host) |
| M7 | NAOE passes, device selection, precision policy, autotuning | complete |
| M8 | nn layers, optimizers, datasets, train/inference, serialization | complete |
| M9 | CLI, formatter, npack, profiler, tests, benchmarks, docs | complete (package imports limited, see below) |

## M0 — Foundation (complete)

* Cargo workspace: `ir`, `compiler`, `optimizer`, `backends`, `runtime`, `cli`,
  `tools/npack`, `tools/formatter`.
* `ARCHITECTURE.md`, `LANGUAGE_SPEC.md`, `ROADMAP.md`.

## M1 — Language foundation (complete)

* Indentation-sensitive lexer with bracket-aware newline suppression.
* Recursive-descent parser and AST for imports, structs, functions, control flow,
  slices, calls and keyword arguments.
* Type checker with inference, tensor shape algebra, library typing and shape
  refinement for `Linear`/`Sequential`.
* Structured diagnostics (human + JSON); 24 compiler tests.

## M2 — Ownership (complete)

* Move/use-after-move, borrow rules, branch/loop state merging, last-use index.
* Autograd-aware lifetime flag on EIR locals.
* Ownership tests.

## M3 — Tensor engine (complete)

* Dense tensor with f64/i64/bool storage and honest f16/bf16/i8 quantisation.
* Elementwise ops with broadcasting, batched matmul, transpose/reshape/slice,
  reductions, softmax, conv1d/conv2d, same-rank gather, dropout, cross-entropy.
* CPU tests for numerics, broadcasting, dtype honesty.

## M4 — IR and execution (complete)

* Four IR data structures; NIR→EIR lowering with spans, last uses and autograd
  flags; MLIR textual emission from TIR; GPU-neutral `TensorOp` metadata.
* EIR register-stack VM with traces, profiling and memory accounting.
* `.neurabc` artifacts are self-contained (header + canonical source) and
  hash-verified on load; `build`→`run` round-trip is covered by an e2e test.
* Remaining (blocked): native EIR→LLVM lowering, which needs an LLVM toolchain
  that is not installed; the interpreter path is the supported CPU backend.

## M5 — Autodiff (complete)

* VJP rules for every tensor op, accumulation, broadcasting undo, gradient checks.
* Reference-counted graph keeps saved tensors alive until their last dependent
  gradient; EIR `Drop` and `needs_autograd` extend lifetimes accordingly.
* `x = Tensor([2.0], requires_grad=true); y = x*x + 3*x; y.backward()` prints `7.0`
  through the full CLI pipeline.
* Cross-entropy picks target logits with same-rank gather and passes an analytic
  value/gradient test; a 2-layer model trains a separable problem to 4/4.

## M6 — GPU (interface-only)

* `GpuBackend` trait shared by CUDA and ROCm; device probing reports absence.
* Allocation/launch return explicit errors rather than fake results.

Blocked: NVVM/ROCDL code generation requires CUDA/ROCm toolchains that are not
installed. Next step when available: shared GPU tensor lowering, streams, async
copies, vendor BLAS hooks.

## M7 — NAOE (complete)

* Constant folding, algebraic simplification, CSE, DCE with jump remapping.
* Elementwise fusion with an inspectable fused expression tree.
* Last-use `Drop` insertion plus buffer-slot reuse; memory report.
* Device placement (`auto` compares size/FLOPs/transfer cost), precision policy
  with validation, bounded autotuning with measured times and a result cache.
* `neura optimize-report` prints measured before/after instruction counts.

## M8 — ML framework (complete)

* Layers: Linear, ReLU, GELU, Sigmoid, Tanh, Softmax, LayerNorm, BatchNorm,
  Dropout, Conv1D, Conv2D, Embedding, Sequential.
* Optimizers: SGD (momentum), Adam, AdamW, RMSProp.
* CSV datasets, deterministic batching/shuffling, `train`, `predict`, parameter
  save/load in a text parameter format.
* Verified end to end on `examples/train_classifier.neura`: loss falls from
  ~0.54 to <0.001 and accuracy reaches 1.0 within a few epochs on
  `examples/datasets/blobs.csv`; the reloaded model reproduces predictions.

## M9 — Tooling (complete)

* `neura` CLI: `check`, `build`, `run`, `inspect`, `graph`, `profile`,
  `optimize-report`, `fmt`, `version`, JSON output for agents.
* Canonical formatter (idempotent, precedence-aware, `elif` reconstruction),
  exposed through `neura fmt [--check]`.
* `npack`: `init`, `install`, `build`, `run`, `test`; manifest parsing, semver
  requirements (`^`, `~`, exact, `*`), recursive path-dependency resolution with
  cycle/conflict detection, deterministic lockfile, stale-lock rejection.
* Profiler, compilation cache, autotune cache, benchmark harness with measured
  CPU numbers (`runtime/examples/benchmarks.rs`), examples and `cli/tests/e2e.rs`.

Limitations (accurate, not simulated):

* Third-party package *imports* are not resolved by the compiler yet; the module
  system covers `neura.*` only. `npack` still resolves, locks and build-verifies
  dependencies.
* Registry downloads require network access, which this build does not use; a
  missing cached package is an explicit error.

## Engineering constraints

* No Python, PyTorch, tracing GC, or vendor SDK is required to build or run Neura.
* Numerical behaviour is explicit; precision changes require an explicit policy.
* Unavailable infrastructure is reported, never simulated.

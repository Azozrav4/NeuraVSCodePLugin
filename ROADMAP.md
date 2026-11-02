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
| M4 | NIR/TIR/ADIR/EIR, lowering, MLIR text, runtime VM, portable artifact | in progress |
| M5 | Reverse-mode autodiff, accumulation, checkpointing hooks | in progress |
| M6 | NVIDIA/AMD interfaces, discovery, memory manager boundary | interface-only (no SDK on host) |
| M7 | NAOE passes, device selection, precision policy, autotuning | in progress |
| M8 | nn layers, optimizers, datasets, train/inference, serialization | in progress |
| M9 | CLI, formatter, npack, profiler, tests, benchmarks, docs | in progress |

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
* Structured diagnostics (human + JSON), 24 compiler tests.

## M2 — Ownership (complete)

* Move/use-after-move, borrow rules, branch/loop state merging, last-use index.
* Autograd-aware lifetime flag on EIR locals.
* Ownership tests.

## M3 — Tensor engine (complete)

* Dense tensor with f64/i64/bool storage and honest f16/bf16/i8 quantisation.
* Elementwise ops with broadcasting, batched matmul, transpose/reshape/slice,
  reductions, softmax, conv1d/conv2d, gather, dropout, cross-entropy.
* CPU tests for numerics, broadcasting, dtype honesty.

## M4 — IR and execution (in progress)

Done: four IR data structures; NIR→EIR lowering with spans, last uses and
autograd flags; MLIR textual emission from TIR; GPU-neutral `TensorOp` metadata.

Remaining: full VM coverage tests, `.neurabc` artifact round-trip, EIR→LLVM when an
LLVM toolchain is available.

## M5 — Autodiff (in progress)

Done: ADIR rule table (`VjpRule`), tensor graph design, saved-tensor metadata.

Remaining: runtime VJP implementations for every tensor op, gradient-check tests
(finite differences), checkpointing execution, `needs_autograd` lifetime extension.

## M6 — GPU (interface-only)

* `GpuBackend` trait shared by CUDA and ROCm; device probing reports absence.
* Allocation/launch return explicit errors rather than fake results.

Blocked: NVVM/ROCDL code generation requires CUDA/ROCm toolchains that are not
installed. Next step when available: shared GPU tensor lowering, streams, async
copies, vendor BLAS hooks.

## M7 — NAOE (in progress)

Planned deliverables: constant folding, algebraic simplification, CSE, DCE with
jump remapping, elementwise fusion, last-use `Drop` + buffer plan, device placement,
precision policy with validation, bounded autotuning with cached measurements,
`optimize-report` output.

## M8 — ML framework (in progress)

Planned: layers (Linear, activations, softmax, LayerNorm, BatchNorm, Dropout,
Conv1D/2D, Embedding, Sequential), optimizers (SGD, Adam, AdamW, RMSProp), CSV
datasets and batching, train/predict APIs, parameter serialization.

## M9 — Tooling (in progress)

Planned: `neura` CLI (`check`, `build`, `run`, `inspect`, `graph`, `profile`,
`optimize-report`, `fmt`), canonical formatter, `npack` with reproducible manifest
and semver resolution, profiler, benchmark harness with honest CPU-only numbers,
examples and integration tests.

## Engineering constraints

* No Python, PyTorch, tracing GC, or vendor SDK is required to build or run Neura.
* Numerical behaviour is explicit; precision changes require an explicit policy.
* Unavailable infrastructure is reported, never simulated.

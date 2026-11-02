//! Honest CPU benchmarks for the tensor engine and the training loop.
//!
//! Every number printed here is measured on the machine running the binary —
//! nothing is hard-coded or extrapolated. Run with `--release` for meaningful
//! throughput figures:
//!
//! ```text
//! cargo run --release -p neura-runtime --example benchmarks
//! ```

use neura_backends::tensor::Tensor;
use neura_ir::ops::TensorOp;
use neura_ir::types::DType;
use neura_runtime::autograd;
use neura_runtime::layers::{self, Activation, Layer};
use neura_runtime::optim::{OptKind, Optimizer};
use neura_runtime::value::{tensor, trainable};
use std::time::Instant;

/// Best-of-three mean wall time per iteration, in seconds.
fn measure(mut f: impl FnMut(), iters: u32) -> f64 {
    f(); // warm-up
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let start = Instant::now();
        for _ in 0..iters {
            f();
        }
        let per_iter = start.elapsed().as_secs_f64() / iters as f64;
        best = best.min(per_iter);
    }
    best
}

fn report(name: &str, seconds: f64, detail: &str) {
    println!("{:<34} {:>10.3} ms   {}", name, seconds * 1e3, detail);
}

fn main() {
    println!(
        "Neura CPU benchmarks — arch={} profile={}",
        std::env::consts::ARCH,
        if cfg!(debug_assertions) { "debug" } else { "release" }
    );
    if cfg!(debug_assertions) {
        println!("note: debug builds are unoptimised; use --release for throughput");
    }
    println!();

    // ---- matmul -----------------------------------------------------------
    let n = 256;
    let a = Tensor::randn(vec![n, n], 1).cast(DType::F32);
    let b = Tensor::randn(vec![n, n], 2).cast(DType::F32);
    let seconds = measure(|| {
        let _ = a.matmul(&b).unwrap();
    }, 5);
    let flops = 2.0 * (n as f64).powi(3);
    report(
        &format!("matmul {}x{}x{} f32", n, n, n),
        seconds,
        &format!("{:.2} GFLOP/s", flops / seconds / 1e9),
    );

    // ---- elementwise ------------------------------------------------------
    let elements = 1 << 20;
    let x = Tensor::randn(vec![elements], 3).cast(DType::F32);
    let seconds = measure(|| {
        let _ = x.unary(&TensorOp::GELU).unwrap();
    }, 20);
    report(
        &format!("GELU {} f32 elements", elements),
        seconds,
        &format!("{:.1} Melem/s", elements as f64 / seconds / 1e6),
    );

    // ---- softmax ----------------------------------------------------------
    let rows = 128;
    let cols = 1000;
    let logits = Tensor::randn(vec![rows, cols], 4).cast(DType::F32);
    let seconds = measure(|| {
        let _ = logits.softmax(1).unwrap();
    }, 20);
    report(
        &format!("softmax {}x{} f32", rows, cols),
        seconds,
        &format!("{:.1} Melem/s", (rows * cols) as f64 / seconds / 1e6),
    );

    // ---- convolution ------------------------------------------------------
    let (batch, channels, height, width, out_channels, kernel) = (4, 8, 32, 32, 16, 3);
    let input = Tensor::randn(vec![batch, channels, height, width], 5).cast(DType::F32);
    let weight = Tensor::randn(vec![out_channels, channels, kernel, kernel], 6).cast(DType::F32);
    let seconds = measure(|| {
        let _ = input.conv2d(&weight, (1, 1), (1, 1)).unwrap();
    }, 5);
    let out_h = height - kernel + 1;
    let out_w = width - kernel + 1;
    let conv_flops =
        2.0 * (batch * out_channels * out_h * out_w * channels * kernel * kernel) as f64;
    report(
        &format!("conv2d {}x{}x{}x{} f32", batch, channels, height, width),
        seconds,
        &format!("{:.2} GFLOP/s", conv_flops / seconds / 1e9),
    );

    // ---- autograd training step ------------------------------------------
    let batch = 64;
    let features = Tensor::randn(vec![batch, 64], 7).cast(DType::F32);
    let labels = Tensor::from_i64(
        DType::I64,
        vec![batch],
        (0..batch).map(|i| (i % 10) as i64).collect(),
    )
    .unwrap();
    let model = Layer::sequential(vec![
        Layer::linear(64, 128, true, 11),
        Layer::Activation(Activation::ReLU),
        Layer::linear(128, 10, true, 12),
    ]);
    let mut optimizer = Optimizer::new(OptKind::Sgd { momentum: 0.9 }, 0.01, model.params());
    let x = tensor(features.clone());
    let y = tensor(labels);
    let step = || {
        optimizer.zero_grad();
        let logits = model.forward(&x, true, 0).unwrap();
        let loss = layers::cross_entropy(&logits, &y).unwrap();
        autograd::backward(&loss).unwrap();
        optimizer.step().unwrap();
    };
    let seconds = measure(step, 10);
    report(
        &format!("train step ({}x64 -> 128 -> 10)", batch),
        seconds,
        &format!("{:.0} samples/s", batch as f64 / seconds),
    );

    // ---- gradient check sanity -------------------------------------------
    let x = trainable(Tensor::from_f64(DType::F64, vec![3], vec![1.0, 2.0, 3.0]).unwrap());
    let y = {
        let square = autograd::apply(&TensorOp::Mul, &[x.clone(), x.clone()], true, 0).unwrap();
        autograd::apply(&TensorOp::Sum { axes: vec![0], keepdim: false }, &[square], true, 0).unwrap()
    };
    let start = Instant::now();
    autograd::backward(&y).unwrap();
    let analytic = x.borrow().grad.clone().unwrap();
    let elapsed = start.elapsed().as_secs_f64();
    report(
        "backward d(sum(x*x))/dx",
        elapsed,
        &format!(
            "grad = {:?} (expected [2.0, 4.0, 6.0])",
            analytic.to_f64_vec()
        ),
    );
}

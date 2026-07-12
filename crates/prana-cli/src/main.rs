//! Prana prototype driver.
//!
//! `prana demo`  — run a tiny transformer-style block through the graph.
//! `prana bench` — microbenchmark the Q8 matmul kernel (the decode hot path)
//!                 and report throughput plus the memory saved by quantization.
//!
//! This is not the Cactus engine; it is a proof that Prana's safe-Rust layering
//! (tensor -> kernels -> graph) composes into something that actually computes,
//! and that the quantized kernel is in a sane performance ballpark.

use std::time::Instant;

use prana_graph::Graph;
use prana_kernels::{matmul_f32, matmul_q8_f32, quantize_q8, Q8_BLOCK};
use prana_tensor::Tensor;

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "demo".to_string());
    match mode.as_str() {
        "demo" => demo(),
        "bench" => bench(),
        other => {
            eprintln!("unknown command '{other}'. use: prana [demo|bench]");
            std::process::exit(2);
        }
    }
}

/// Run one normalize -> project -> residual -> softmax block and print a summary.
fn demo() {
    println!("== Prana demo: one transformer-style block ==\n");
    let dim = 64;
    let mut g = Graph::new();
    let x = g.input(1);

    let n = g.rmsnorm(x, vec![1.0; dim], dim, 1e-5);
    let w: Vec<f32> = (0..dim * dim).map(|i| (i as f32 * 0.001).sin()).collect();
    let proj = g.matmul_f32(n, w, dim, dim);
    let res = g.add(proj, x);
    let out = g.softmax(res);

    let input: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.05).sin()).collect();
    g.set_input(x, Tensor::from_f32(vec![1, dim], &input));
    g.execute().expect("graph executed");

    let probs = g.output_f32(out).expect("f32 output");
    let argmax = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    println!("  input dim         : {dim}");
    println!("  output is a distribution over {} logits", probs.len());
    println!("  sum(probs)        : {:.6} (should be 1.0)", probs.iter().sum::<f32>());
    println!("  argmax index      : {argmax}");
    println!("\nAll layers ran with zero `unsafe` above the kernel boundary.");
}

/// Microbenchmark the decode-path matmul: a `[1 x k] * [n_rows x k]` projection,
/// dense f32 vs on-the-fly Q8 dequant, repeated to get a stable timing.
fn bench() {
    println!("== Prana microbenchmark: decode-path matmul ==\n");

    // Roughly the shape of one big projection in a ~1.5B model layer.
    let k = 2048;
    let n_rows = 2048;
    let iters = 200;

    let a: Vec<f32> = (0..k).map(|i| (i as f32 * 0.001).sin()).collect();
    let w: Vec<f32> = (0..n_rows * k).map(|i| (i as f32 * 0.0007).cos()).collect();
    let qm = quantize_q8(n_rows, k, &w);

    // Warm up + correctness sanity: quantized output should track dense.
    let dense0 = matmul_f32(&a, &w, 1, k, n_rows);
    let quant0 = matmul_q8_f32(&a, &qm, 1);
    let max_err = dense0
        .iter()
        .zip(&quant0)
        .map(|(d, q)| (d - q).abs())
        .fold(0f32, f32::max);
    // Relative to the RMS magnitude of the dense outputs — the honest way to
    // read quantization error on large-k dot products.
    let rms = (dense0.iter().map(|d| d * d).sum::<f32>() / dense0.len() as f32).sqrt();
    let rel_err = max_err / rms;

    let t0 = Instant::now();
    let mut sink = 0f32;
    for _ in 0..iters {
        let o = matmul_f32(&a, &w, 1, k, n_rows);
        sink += o[0];
    }
    let dense_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

    let t1 = Instant::now();
    for _ in 0..iters {
        let o = matmul_q8_f32(&a, &qm, 1);
        sink += o[0];
    }
    let quant_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;

    let flops = 2.0 * k as f64 * n_rows as f64; // one MAC = 2 flops
    let dense_gflops = flops / (dense_ms * 1e-3) / 1e9;
    let quant_gflops = flops / (quant_ms * 1e-3) / 1e9;

    let dense_bytes = w.len() * 4;
    let quant_bytes = qm.stored_bytes();

    println!("  problem           : [1 x {k}] * [{n_rows} x {k}]  (block={Q8_BLOCK})");
    println!("  threads           : {}", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    println!();
    println!("  dense  f32 matmul : {dense_ms:>7.3} ms/iter   {dense_gflops:>6.1} GFLOP/s");
    println!("  quant  q8  matmul : {quant_ms:>7.3} ms/iter   {quant_gflops:>6.1} GFLOP/s");
    println!();
    println!("  weight memory     : {} KiB f32  ->  {} KiB q8  ({:.2}x smaller)",
        dense_bytes / 1024, quant_bytes / 1024, dense_bytes as f64 / quant_bytes as f64);
    println!("  max abs error     : {max_err:.4}  ({:.2}% of output RMS)", rel_err * 100.0);
    println!("\n  (sink={sink:.3})  # keeps the optimizer from eliding the loops");
    println!();
    println!("  Honest reading: with the *scalar* kernel, Q8's win here is the");
    println!("  3.5x smaller weight footprint, not speed — the per-element i8->f32");
    println!("  cast doesn't autovectorize as well as the dense FMA, so on this");
    println!("  cache-resident size Q8 is compute-bound and slightly slower. The");
    println!("  speed win appears (a) at model sizes where weights spill L2/L3 and");
    println!("  bandwidth dominates, and (b) once the target-gated NEON/AVX tier");
    println!("  (same signatures) replaces the scalar dot products.");
}

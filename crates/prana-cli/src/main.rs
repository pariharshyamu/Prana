//! Prana prototype driver.
//!
//! `prana demo`  — run a tiny transformer-style block through the graph.
//! `prana bench` — microbenchmark the Q8 matmul kernel (the decode hot path)
//!                 and report throughput plus the memory saved by quantization.
//! `prana chat`  — drive the Phase 0 safe wrapper over the Cactus C ABI
//!                 (mock engine by default; real engine with `link-cactus`).
//! `prana run`   — generate text with a real model (llama2.c checkpoint
//!                 format, e.g. Karpathy's TinyStories models) running
//!                 entirely on Prana's safe Rust kernels.
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
        "teambench" => teambench(),
        "chat" => chat(),
        "run" => run(),
        other => {
            eprintln!("unknown command '{other}'. use: prana [demo|bench|chat|run]");
            std::process::exit(2);
        }
    }
}

/// Microbenchmark the team primitives: dispatch cost and per-barrier cost.
fn teambench() {
    use prana_kernels::run_team;
    let threads = prana_kernels::pool().threads;
    println!("== team primitives ({threads} threads) ==");

    // Dispatch: empty team run.
    let iters = 2000;
    let t0 = Instant::now();
    for _ in 0..iters {
        run_team(|_| {});
    }
    println!("  run_team dispatch : {:>8.2} µs", t0.elapsed().as_secs_f64() * 1e6 / iters as f64);

    // Barrier: one dispatch, many barriers inside.
    let barriers = 20_000usize;
    let t0 = Instant::now();
    run_team(|team| {
        for _ in 0..barriers {
            team.barrier();
        }
    });
    println!("  barrier           : {:>8.3} µs", t0.elapsed().as_secs_f64() * 1e6 / barriers as f64);

    // for_chunks with matmul-like chunk counts.
    let rounds = 5_000usize;
    let t0 = Instant::now();
    run_team(|team| {
        for _ in 0..rounds {
            team.for_chunks(threads * 4, |_| {});
        }
    });
    println!("  for_chunks(4/thr) : {:>8.3} µs", t0.elapsed().as_secs_f64() * 1e6 / rounds as f64);

    // serial section.
    let t0 = Instant::now();
    run_team(|team| {
        for _ in 0..rounds {
            team.serial(|| {});
        }
    });
    println!("  serial (empty)    : {:>8.3} µs", t0.elapsed().as_secs_f64() * 1e6 / rounds as f64);
}

/// Generate text with a real model. Usage:
/// `prana run [model.bin tokenizer.bin | model.gguf] [prompt] [--steps N] [--temp T] [--q8] [--chat]`
/// (GGUF files embed their tokenizer, so no tokenizer argument is needed.
/// `--chat` wraps the prompt in the ChatML template instruct models expect.)
fn run() {
    use prana_model::{checkpoint, gguf, Precision, Sampler, Tokenizer};
    use std::io::Write;

    let args: Vec<String> = std::env::args().skip(2).collect();
    let mut positional = Vec::new();
    let mut steps = 200usize;
    let mut temp = 0.8f32;
    let mut seed = 20260712u64;
    let mut precision = Precision::F32;
    let mut chat_mode = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--steps" => steps = it.next().and_then(|v| v.parse().ok()).unwrap_or(steps),
            "--temp" => temp = it.next().and_then(|v| v.parse().ok()).unwrap_or(temp),
            "--seed" => seed = it.next().and_then(|v| v.parse().ok()).unwrap_or(seed),
            "--q8" => precision = Precision::Q8,
            "--chat" => chat_mode = true,
            v => positional.push(v.to_string()),
        }
    }
    let model_path = positional.first().cloned().unwrap_or_else(|| "models/stories15M.bin".into());
    let is_gguf = model_path.ends_with(".gguf");
    // GGUF embeds its tokenizer, so positionals shift left by one.
    let tok_path = if is_gguf { None } else { Some(positional.get(1).cloned().unwrap_or_else(|| "models/tokenizer.bin".into())) };
    let prompt_idx = if is_gguf { 1 } else { 2 };
    let prompt = positional.get(prompt_idx).cloned().unwrap_or_else(|| "Once upon a time".into());

    println!("== Prana run: real-model inference on safe Rust kernels ==\n");
    let t_load = std::time::Instant::now();
    let (model, tokenizer): (_, Box<dyn prana_model::Tokenize>) = if is_gguf {
        match gguf::load(std::path::Path::new(&model_path), precision) {
            Ok((m, t)) => (m, Box::new(t)),
            Err(e) => {
                eprintln!("cannot load '{model_path}': {e}");
                std::process::exit(1);
            }
        }
    } else {
        let model = match checkpoint::load(std::path::Path::new(&model_path), precision) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("cannot load '{model_path}': {e}");
                eprintln!("hint: fetch the model first — see scripts/fetch-model.sh");
                std::process::exit(1);
            }
        };
        let tok_path = tok_path.unwrap();
        let tokenizer = Tokenizer::load(std::path::Path::new(&tok_path), model.config.vocab_size)
            .unwrap_or_else(|e| {
                eprintln!("cannot load tokenizer '{tok_path}': {e}");
                std::process::exit(1);
            });
        (model, Box::new(tokenizer))
    };

    let c = &model.config;
    println!(
        "  model             : {model_path}  (dim={} layers={} heads={} kv_heads={} vocab={} ctx={})",
        c.dim, c.n_layers, c.n_heads, c.n_kv_heads, c.vocab_size, c.seq_len
    );
    println!(
        "  precision         : {:?}  ({:.1} MiB projections + {:.1} MiB embedding table)",
        precision,
        model.projection_bytes() as f64 / (1024.0 * 1024.0),
        model.tok_emb.stored_bytes() as f64 / (1024.0 * 1024.0)
    );
    println!("  load time         : {:.2}s", t_load.elapsed().as_secs_f64());
    println!("  sampler           : temp={temp} seed={seed}   steps={steps}\n");
    println!("---");

    // ChatML is the template the Qwen-family instruct models are trained on.
    // The special markers encode verbatim via the tokenizer's control tokens.
    let prompt = if chat_mode {
        format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
             <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
    } else {
        prompt
    };

    // Completion mode echoes the prompt as it prefills; chat mode hides the
    // template and shows only the assistant's reply.
    let mut sampler = Sampler::new(temp, seed);
    let stats = prana_model::generate(&model, tokenizer.as_ref(), &prompt, steps, &mut sampler, |piece, is_prompt| {
        if !(chat_mode && is_prompt) {
            std::io::stdout().write_all(piece).ok();
            std::io::stdout().flush().ok();
        }
    });
    println!("\n---\n");
    println!(
        "  {} prompt + {} generated tokens in {:.2}s  ->  {:.1} tok/s",
        stats.prompt_tokens,
        stats.generated_tokens,
        stats.seconds,
        (stats.prompt_tokens + stats.generated_tokens) as f64 / stats.seconds
    );
    if let Some(report) = prana_model::timing_report() {
        print!("{report}");
    }
}

/// Drive the Phase 0 wrapper end-to-end: open a model handle over the C ABI,
/// stream a completion token by token, tokenize, and embed — all through the
/// safe API. With the default build this talks to the in-process mock engine;
/// built with `--features prana-cactus/link-cactus` it talks to the real
/// `libcactus_engine.a` through the identical code path.
fn chat() {
    use prana_cactus::{CompleteOptions, Message, Model};
    use std::io::Write;

    println!("== Prana chat: Phase 0 safe wrapper over the Cactus C ABI ==\n");
    println!("  engine            : {}", prana_cactus::engine_kind());

    let model_path = std::env::args().nth(2).unwrap_or_else(|| "mock://tiny-model".to_string());
    let prompt = std::env::args().nth(3).unwrap_or_else(|| "ping".to_string());

    let mut model = match Model::open(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("  failed to open '{model_path}': {e}");
            std::process::exit(1);
        }
    };
    println!("  model             : {model_path}");
    println!("  prompt            : {prompt}\n");

    print!("  streamed reply    : ");
    std::io::stdout().flush().ok();
    let mut n_tokens = 0u32;
    let reply = model
        .complete_streaming(
            &[Message::system("You are Prana."), Message::user(&prompt)],
            &CompleteOptions::default(),
            |tok, _id| {
                print!("{tok}");
                std::io::stdout().flush().ok();
                n_tokens += 1;
            },
        )
        .expect("completion");
    println!("\n  tokens streamed   : {n_tokens}");
    println!("  full response     : {reply}");

    let tokens = model.tokenize(&prompt).expect("tokenize");
    println!("  tokenize(prompt)  : {} tokens", tokens.len());
    let embedding = model.embed(&prompt, true).expect("embed");
    println!("  embed(prompt)     : {}-dim vector", embedding.len());
    println!("\n  Model handle is RAII — cactus_destroy runs on drop, even on panic.");
}

/// Run one full transformer block — attention sub-layer (norm -> QKV -> RoPE ->
/// causal attention -> residual) followed by an MLP sub-layer (norm -> up ->
/// down -> residual) — over a short token sequence, then print a summary.
fn demo() {
    println!("== Prana demo: one full transformer block (prefill, seq=8) ==\n");

    let seq = 8;
    let n_heads = 4;
    let n_kv_heads = 2; // grouped-query attention, like Llama-3 / Qwen2
    let head_dim = 16;
    let dim = n_heads * head_dim; // model width = 64
    let ffn = dim * 2; // MLP hidden width

    // Deterministic pseudo-weights so the demo is reproducible.
    let mat = |rows: usize, cols: usize, seed: f32| -> Vec<f32> {
        (0..rows * cols).map(|i| (i as f32 * seed).sin() * 0.1).collect()
    };

    let mut g = Graph::new();
    let x = g.input(seq);

    // --- Attention sub-layer ---
    let n1 = g.rmsnorm(x, vec![1.0; dim], dim, 1e-5);
    let q0 = g.matmul_f32(n1, mat(dim, dim, 0.0011), dim, dim);
    let k0 = g.matmul_f32(n1, mat(n_kv_heads * head_dim, dim, 0.0013), dim, n_kv_heads * head_dim);
    let v0 = g.matmul_f32(n1, mat(n_kv_heads * head_dim, dim, 0.0017), dim, n_kv_heads * head_dim);
    let q = g.rope(q0, n_heads, head_dim, 10000.0);
    let k = g.rope(k0, n_kv_heads, head_dim, 10000.0);
    let attn = g.attention(q, k, v0, n_heads, n_kv_heads, head_dim);
    let o_proj = g.matmul_f32(attn, mat(dim, dim, 0.0019), dim, dim);
    let res1 = g.add(o_proj, x);

    // --- MLP sub-layer ---
    let n2 = g.rmsnorm(res1, vec![1.0; dim], dim, 1e-5);
    let up = g.matmul_f32(n2, mat(ffn, dim, 0.0007), dim, ffn);
    let down = g.matmul_f32(up, mat(dim, ffn, 0.0009), ffn, dim);
    let out = g.add(down, res1);

    let input: Vec<f32> = (0..seq * dim).map(|i| (i as f32 * 0.01).sin()).collect();
    g.set_input(x, Tensor::from_f32(vec![seq, dim], &input));
    g.execute().expect("graph executed");

    let y = g.output_f32(out).expect("f32 output");
    let finite = y.iter().all(|v| v.is_finite());
    let rms = (y.iter().map(|v| v * v).sum::<f32>() / y.len() as f32).sqrt();

    println!("  config            : dim={dim}  heads={n_heads}  kv_heads={n_kv_heads} (GQA)  head_dim={head_dim}");
    println!("  sequence          : {seq} tokens (causal self-attention)");
    println!("  hidden state out  : [{seq} x {dim}] = {} values", y.len());
    println!("  all finite        : {finite}");
    println!("  output RMS        : {rms:.4}");
    println!("\n  Ops exercised: rmsnorm, matmul (Q/K/V/O/up/down), RoPE,");
    println!("  grouped-query causal attention, residual add.");
    println!("  All layers ran with zero `unsafe` above the kernel boundary.");
}

/// Microbenchmark the decode-path matmul: a `[1 x k] * [n_rows x k]` projection,
/// dense f32 vs integer Q8, repeated to get a stable timing.
/// `prana bench [k n_rows iters]` — pass e.g. `896 151936 20` to measure a
/// DRAM-streaming (cache-defeating) size instead of the cache-resident default.
fn bench() {
    println!("== Prana microbenchmark: decode-path matmul ==\n");

    // Default: roughly one big projection in a ~1.5B model layer.
    let arg = |i: usize, d: usize| {
        std::env::args().nth(2 + i).and_then(|v| v.parse().ok()).unwrap_or(d)
    };
    let k = arg(0, 2048);
    let n_rows = arg(1, 2048);
    let iters = arg(2, 200);

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

    // Q4_0 same shape (synthetic packed blocks): isolates the q4_0 dot
    // kernel against the q8 one at identical geometry.
    let q40_ms = if k % 32 == 0 {
        use prana_kernels::{matmul_kquant_f32, KQuantKind, KQuantMatrix};
        let blocks = n_rows * k / 32;
        let mut raw = vec![0u8; blocks * 18];
        for (i, b) in raw.chunks_exact_mut(18).enumerate() {
            b[0..2].copy_from_slice(&0x2e66u16.to_le_bytes()); // d ~ 0.1
            for (j, q) in b[2..].iter_mut().enumerate() {
                *q = ((i * 31 + j * 7) % 256) as u8;
            }
        }
        let m40 = KQuantMatrix::from_raw(n_rows, k, KQuantKind::Q40, raw);
        let t = Instant::now();
        for _ in 0..iters {
            let o = matmul_kquant_f32(&a, &m40, 1);
            sink += o[0];
        }
        Some(t.elapsed().as_secs_f64() * 1e3 / iters as f64)
    } else {
        None
    };

    let flops = 2.0 * k as f64 * n_rows as f64; // one MAC = 2 flops
    let dense_gflops = flops / (dense_ms * 1e-3) / 1e9;
    let quant_gflops = flops / (quant_ms * 1e-3) / 1e9;

    let dense_bytes = w.len() * 4;
    let quant_bytes = qm.stored_bytes();

    println!("  problem           : [1 x {k}] * [{n_rows} x {k}]  (block={Q8_BLOCK})");
    println!(
        "  simd tier         : {}",
        if prana_kernels::simd_vnni_available() {
            "AVX2+FMA+AVX-VNNI"
        } else if prana_kernels::simd_available() {
            "AVX2+FMA"
        } else {
            "scalar"
        }
    );
    println!("  threads           : {} (PRANA_THREADS overrides)", prana_kernels::pool().threads);
    println!();
    println!("  dense  f32 matmul : {dense_ms:>7.3} ms/iter   {dense_gflops:>6.1} GFLOP/s");
    println!("  quant  q8  matmul : {quant_ms:>7.3} ms/iter   {quant_gflops:>6.1} GFLOP/s");
    if let Some(ms) = q40_ms {
        println!("  quant  q4_0 matmul: {ms:>7.3} ms/iter   {:>6.1} GFLOP/s", flops / (ms * 1e-3) / 1e9);
    }
    println!();
    println!("  weight memory     : {} KiB f32  ->  {} KiB q8  ({:.2}x smaller)",
        dense_bytes / 1024, quant_bytes / 1024, dense_bytes as f64 / quant_bytes as f64);
    println!("  max abs error     : {max_err:.4}  ({:.2}% of output RMS)", rel_err * 100.0);
    println!("\n  (sink={sink:.3})  # keeps the optimizer from eliding the loops");
    println!();
    println!("  Reading: the quantized matmul runs *integer* dot products — the");
    println!("  activations are quantized to i8 per 32-group once per call, so the");
    println!("  inner loop is int8 x int8 (32 MACs per AVX2 maddubs+madd pair vs 8");
    println!("  f32 FMA lanes) and reads ~4x fewer weight bytes. It now wins on");
    println!("  both footprint and speed. Accuracy cost: ~0.1% of output RMS, the");
    println!("  same order as the weight quantization itself. End-to-end decode of");
    println!("  a real model is DRAM-bandwidth-bound, so the tok/s gain there is");
    println!("  smaller than this cache-resident kernel ratio suggests.");
}

//! A minimal define-then-run computation graph, the safe-Rust analogue of
//! `CactusGraph` (`cactus-graph/`).
//!
//! Cactus builds a graph of `void*`-typed buffers wired by integer node ids,
//! executes it, and exposes the result through `get_output(id)`. The whole
//! design leans on the caller not misusing raw ids or reading a buffer of the
//! wrong precision. Prana keeps the same *ergonomics* — build nodes, get back
//! opaque handles, `execute()`, read outputs — but every handle is a checked
//! index into a `Vec`, every op validates its input shapes, and a node can
//! never observe an unevaluated or wrongly-typed input because the type system
//! and `Option` enforce it.
//!
//! This is deliberately tiny (matmul, quantized-matmul, rmsnorm, add, softmax):
//! enough to run one transformer block's worth of ops end-to-end and prove the
//! layering, not a full op set.

use prana_kernels::{attention, matmul_f32, matmul_q8_f32, rmsnorm, rope, softmax, QuantMatrix};
use prana_tensor::Tensor;

/// Opaque handle to a graph node. Just a checked index — cannot be forged into
/// an out-of-range access because `NodeId` is only ever produced by the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeId(usize);

enum Op {
    /// Externally-supplied input; value set via `set_input`.
    Input,
    /// Dense weight matmul: `matmul(x, weight[n_rows x k])`.
    MatmulF32 { x: NodeId, weight: Vec<f32>, k: usize, n_rows: usize },
    /// Quantized weight matmul.
    MatmulQ8 { x: NodeId, weight: QuantMatrix },
    /// RMSNorm with per-feature weight.
    RmsNorm { x: NodeId, weight: Vec<f32>, dim: usize, eps: f32 },
    /// Elementwise add of two equal-shaped nodes.
    Add { a: NodeId, b: NodeId },
    /// Row-wise softmax (single row expected in the prototype).
    Softmax { x: NodeId },
    /// In-place rotary position embedding over `[seq, n_heads * head_dim]`.
    Rope { x: NodeId, n_heads: usize, head_dim: usize, theta_base: f32 },
    /// Causal (grouped-query) self-attention over Q/K/V nodes.
    Attention { q: NodeId, k: NodeId, v: NodeId, n_heads: usize, n_kv_heads: usize, head_dim: usize },
}

struct Node {
    op: Op,
    value: Option<Tensor>,
}

/// The graph itself: append-only list of nodes with cached values.
#[derive(Default)]
pub struct Graph {
    nodes: Vec<Node>,
    /// Number of tokens (rows) flowing through matmuls; set by the first input.
    n_tokens: usize,
}

/// Errors that are genuinely runtime conditions (shape mismatch, missing input),
/// surfaced as `Result` instead of Cactus's convention of logging + returning a
/// null/again-checked pointer.
#[derive(Debug, PartialEq, Eq)]
pub enum GraphError {
    UnsetInput(NodeId),
    ShapeMismatch { what: &'static str },
    NotF32,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, op: Op) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node { op, value: None });
        id
    }

    /// Declare an input node. `n_tokens` is the row count for the whole graph.
    pub fn input(&mut self, n_tokens: usize) -> NodeId {
        self.n_tokens = self.n_tokens.max(n_tokens);
        self.push(Op::Input)
    }

    pub fn matmul_f32(&mut self, x: NodeId, weight: Vec<f32>, k: usize, n_rows: usize) -> NodeId {
        self.push(Op::MatmulF32 { x, weight, k, n_rows })
    }

    pub fn matmul_q8(&mut self, x: NodeId, weight: QuantMatrix) -> NodeId {
        self.push(Op::MatmulQ8 { x, weight })
    }

    pub fn rmsnorm(&mut self, x: NodeId, weight: Vec<f32>, dim: usize, eps: f32) -> NodeId {
        self.push(Op::RmsNorm { x, weight, dim, eps })
    }

    pub fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.push(Op::Add { a, b })
    }

    pub fn softmax(&mut self, x: NodeId) -> NodeId {
        self.push(Op::Softmax { x })
    }

    /// Apply RoPE to a `[seq, n_heads * head_dim]` node (positions are the row
    /// indices of the current forward pass).
    pub fn rope(&mut self, x: NodeId, n_heads: usize, head_dim: usize, theta_base: f32) -> NodeId {
        self.push(Op::Rope { x, n_heads, head_dim, theta_base })
    }

    /// Causal self-attention. `q` is `[seq, n_heads*head_dim]`, `k`/`v` are
    /// `[seq, n_kv_heads*head_dim]`; `n_heads` must be a multiple of `n_kv_heads`.
    pub fn attention(
        &mut self,
        q: NodeId,
        k: NodeId,
        v: NodeId,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> NodeId {
        self.push(Op::Attention { q, k, v, n_heads, n_kv_heads, head_dim })
    }

    /// Bind a concrete tensor to an input node.
    pub fn set_input(&mut self, id: NodeId, value: Tensor) {
        self.nodes[id.0].value = Some(value);
    }

    /// Evaluate the whole graph in definition order. Because nodes are appended
    /// and only ever reference earlier ids, a single forward pass suffices — no
    /// topological sort or cycle handling needed for this prototype.
    pub fn execute(&mut self) -> Result<(), GraphError> {
        for i in 0..self.nodes.len() {
            if self.nodes[i].value.is_some() {
                continue; // inputs already bound
            }
            let value = self.eval_node(i)?;
            self.nodes[i].value = Some(value);
        }
        Ok(())
    }

    fn f32_of(&self, id: NodeId) -> Result<Vec<f32>, GraphError> {
        let t = self.nodes[id.0].value.as_ref().ok_or(GraphError::UnsetInput(id))?;
        t.as_f32().ok_or(GraphError::NotF32)
    }

    fn eval_node(&self, i: usize) -> Result<Tensor, GraphError> {
        match &self.nodes[i].op {
            Op::Input => Err(GraphError::UnsetInput(NodeId(i))),
            Op::MatmulF32 { x, weight, k, n_rows } => {
                let a = self.f32_of(*x)?;
                if a.len() != self.n_tokens * *k {
                    return Err(GraphError::ShapeMismatch { what: "matmul_f32 input" });
                }
                let out = matmul_f32(&a, weight, self.n_tokens, *k, *n_rows);
                Ok(Tensor::from_f32(vec![self.n_tokens, *n_rows], &out))
            }
            Op::MatmulQ8 { x, weight } => {
                let a = self.f32_of(*x)?;
                if a.len() != self.n_tokens * weight.cols {
                    return Err(GraphError::ShapeMismatch { what: "matmul_q8 input" });
                }
                let out = matmul_q8_f32(&a, weight, self.n_tokens);
                Ok(Tensor::from_f32(vec![self.n_tokens, weight.rows], &out))
            }
            Op::RmsNorm { x, weight, dim, eps } => {
                let a = self.f32_of(*x)?;
                if a.len() % dim != 0 {
                    return Err(GraphError::ShapeMismatch { what: "rmsnorm dim" });
                }
                let rows = a.len() / dim;
                let out = rmsnorm(&a, weight, rows, *dim, *eps);
                Ok(Tensor::from_f32(vec![rows, *dim], &out))
            }
            Op::Add { a, b } => {
                let va = self.f32_of(*a)?;
                let vb = self.f32_of(*b)?;
                if va.len() != vb.len() {
                    return Err(GraphError::ShapeMismatch { what: "add" });
                }
                let out: Vec<f32> = va.iter().zip(&vb).map(|(x, y)| x + y).collect();
                let shape = self.nodes[a.0].value.as_ref().unwrap().shape().to_vec();
                Ok(Tensor::from_f32(shape, &out))
            }
            Op::Softmax { x } => {
                let a = self.f32_of(*x)?;
                let out = softmax(&a);
                Ok(Tensor::from_f32(vec![a.len()], &out))
            }
            Op::Rope { x, n_heads, head_dim, theta_base } => {
                let mut a = self.f32_of(*x)?;
                if a.len() != self.n_tokens * n_heads * head_dim {
                    return Err(GraphError::ShapeMismatch { what: "rope input" });
                }
                rope(&mut a, self.n_tokens, *n_heads, *head_dim, *theta_base);
                Ok(Tensor::from_f32(vec![self.n_tokens, n_heads * head_dim], &a))
            }
            Op::Attention { q, k, v, n_heads, n_kv_heads, head_dim } => {
                let qv = self.f32_of(*q)?;
                let kv = self.f32_of(*k)?;
                let vv = self.f32_of(*v)?;
                let seq = self.n_tokens;
                if qv.len() != seq * n_heads * head_dim
                    || kv.len() != seq * n_kv_heads * head_dim
                    || vv.len() != seq * n_kv_heads * head_dim
                {
                    return Err(GraphError::ShapeMismatch { what: "attention qkv" });
                }
                let out = attention(&qv, &kv, &vv, seq, *n_heads, *n_kv_heads, *head_dim);
                Ok(Tensor::from_f32(vec![seq, n_heads * head_dim], &out))
            }
        }
    }

    /// Read a node's computed value (F32) after `execute`.
    pub fn output_f32(&self, id: NodeId) -> Result<Vec<f32>, GraphError> {
        self.f32_of(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prana_kernels::quantize_q8;

    #[test]
    fn runs_a_tiny_block() {
        // x -> rmsnorm -> matmul -> add(residual) -> softmax
        let dim = 32;
        let mut g = Graph::new();
        let x = g.input(1);

        let norm_w = vec![1.0f32; dim];
        let n = g.rmsnorm(x, norm_w, dim, 1e-5);

        let w: Vec<f32> = (0..dim * dim).map(|i| (i as f32 * 0.001).sin()).collect();
        let proj = g.matmul_f32(n, w, dim, dim);

        let res = g.add(proj, x);
        let out = g.softmax(res);

        g.set_input(x, Tensor::from_f32(vec![1, dim], &vec![0.5f32; dim]));
        g.execute().unwrap();

        let probs = g.output_f32(out).unwrap();
        assert_eq!(probs.len(), dim);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn quantized_matmul_node_works() {
        let dim = 32;
        let mut g = Graph::new();
        let x = g.input(1);
        let w: Vec<f32> = (0..dim * dim).map(|i| (i as f32 * 0.002).cos()).collect();
        let qm = quantize_q8(dim, dim, &w);
        let y = g.matmul_q8(x, qm);
        g.set_input(x, Tensor::from_f32(vec![1, dim], &vec![0.25f32; dim]));
        g.execute().unwrap();
        assert_eq!(g.output_f32(y).unwrap().len(), dim);
    }

    #[test]
    fn runs_an_attention_block() {
        // seq=4, 2 heads, head_dim=8 -> feature width 16. q/k/v come from three
        // projections of the same input; rope on q and k; then causal attention.
        let seq = 4;
        let n_heads = 2;
        let head_dim = 8;
        let width = n_heads * head_dim;

        let mut g = Graph::new();
        let x = g.input(seq);

        let proj = |seed: f32| -> Vec<f32> {
            (0..width * width).map(|i| (i as f32 * seed).sin()).collect()
        };
        let q0 = g.matmul_f32(x, proj(0.001), width, width);
        let k0 = g.matmul_f32(x, proj(0.002), width, width);
        let v0 = g.matmul_f32(x, proj(0.003), width, width);
        let q = g.rope(q0, n_heads, head_dim, 10000.0);
        let k = g.rope(k0, n_heads, head_dim, 10000.0);
        let attn = g.attention(q, k, v0, n_heads, n_heads, head_dim);
        let out = g.add(attn, x); // residual

        let input: Vec<f32> = (0..seq * width).map(|i| (i as f32 * 0.02).cos()).collect();
        g.set_input(x, Tensor::from_f32(vec![seq, width], &input));
        g.execute().unwrap();

        let y = g.output_f32(out).unwrap();
        assert_eq!(y.len(), seq * width);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn unset_input_is_an_error_not_a_crash() {
        let mut g = Graph::new();
        let x = g.input(1);
        let y = g.rmsnorm(x, vec![1.0; 4], 4, 1e-5);
        // Never bound x -> execute returns a typed error rather than UB.
        assert_eq!(g.execute(), Err(GraphError::UnsetInput(x)));
        let _ = y;
    }
}

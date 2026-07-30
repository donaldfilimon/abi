//! Model tensors, forward/backward, and sampling scratch.
//!
//! Ported from `src/features/nn/model.zig`.

use crate::types::{Activation, TrainReport};

/// Dense matrix stored row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    /// Flat data, `rows * cols` elements.
    pub data: Vec<f32>,
    /// Row count.
    pub rows: usize,
    /// Column count.
    pub cols: usize,
}

impl Matrix {
    /// Zero-initialized matrix.
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Borrow row `r`.
    #[must_use]
    pub fn row(&self, r: usize) -> &[f32] {
        let start = r * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Mutable borrow of row `r`.
    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        let start = r * self.cols;
        &mut self.data[start..start + self.cols]
    }

    /// `out = self * x` (matrix-vector product).
    pub fn mat_vec(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.cols);
        debug_assert_eq!(out.len(), self.rows);
        for (row, slot) in (0..self.rows).zip(out.iter_mut()) {
            *slot = dot(self.row(row), x);
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Trained (or freshly initialized) character-level model.
#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    /// Byte → vocab id (`-1` when absent).
    pub vocab: [i16; 256],
    /// Vocab id → byte.
    pub id_to_byte: Vec<u8>,
    /// Active vocabulary size.
    pub vocab_size: usize,
    /// Context length.
    pub seq_len: usize,
    /// Embedding width.
    pub embed_dim: usize,
    /// Hidden width.
    pub hidden: usize,
    /// Hidden activation.
    pub activation: Activation,
    /// Embedding table `[vocab_size, embed_dim]`.
    pub embed: Matrix,
    /// First affine `[hidden, seq_len * embed_dim]`.
    pub w1: Matrix,
    /// First bias.
    pub b1: Vec<f32>,
    /// Output affine `[vocab_size, hidden]`.
    pub w2: Matrix,
    /// Output bias.
    pub b2: Vec<f32>,
    /// Training corpus bytes.
    pub corpus: Vec<u8>,
    /// Last training report (zeros until trained).
    pub report: TrainReport,
}

impl Model {
    /// Input feature dimension (`seq_len * embed_dim`).
    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.seq_len * self.embed_dim
    }

    /// Fill `ctx` with vocab ids for the window starting at corpus position `p`,
    /// and return the target id immediately after the window.
    pub fn sample_at(&self, p: usize, ctx: &mut [usize]) -> usize {
        let n = self.corpus.len();
        for (s, slot) in ctx.iter_mut().enumerate().take(self.seq_len) {
            let byte = usize::from(self.corpus[(p + s) % n]);
            // Vocab entries for present bytes are non-negative by construction.
            *slot = usize::try_from(self.vocab[byte]).expect("present byte has a vocab id");
        }
        let target_byte = usize::from(self.corpus[(p + self.seq_len) % n]);
        usize::try_from(self.vocab[target_byte]).expect("present byte has a vocab id")
    }
}

/// Per-step scratch buffers.
#[derive(Debug, Clone)]
pub struct Scratch {
    /// Context window of vocab ids.
    pub ctx: Vec<usize>,
    /// Concatenated embeddings.
    pub x: Vec<f32>,
    /// Hidden activations.
    pub h: Vec<f32>,
    /// Output logits.
    pub logits: Vec<f32>,
    /// Softmax probabilities.
    pub probs: Vec<f32>,
    /// Logit gradients.
    pub dlogits: Vec<f32>,
    /// Hidden gradients.
    pub dh: Vec<f32>,
    /// Pre-activation hidden gradients.
    pub dhpre: Vec<f32>,
    /// Input gradients.
    pub dx: Vec<f32>,
}

impl Scratch {
    /// Allocate scratch sized for `model`.
    #[must_use]
    pub fn new(model: &Model) -> Self {
        let in_dim = model.input_dim();
        Self {
            ctx: vec![0; model.seq_len],
            x: vec![0.0; in_dim],
            h: vec![0.0; model.hidden],
            logits: vec![0.0; model.vocab_size],
            probs: vec![0.0; model.vocab_size],
            dlogits: vec![0.0; model.vocab_size],
            dh: vec![0.0; model.hidden],
            dhpre: vec![0.0; model.hidden],
            dx: vec![0.0; in_dim],
        }
    }
}

/// Gradient buffers matching model parameters.
#[derive(Debug, Clone)]
pub struct Grads {
    /// Embedding gradients.
    pub embed: Matrix,
    /// `w1` gradients.
    pub w1: Matrix,
    /// `b1` gradients.
    pub b1: Vec<f32>,
    /// `w2` gradients.
    pub w2: Matrix,
    /// `b2` gradients.
    pub b2: Vec<f32>,
}

impl Grads {
    /// Zero-initialized gradients.
    #[must_use]
    pub fn new(model: &Model) -> Self {
        Self {
            embed: Matrix::zeros(model.vocab_size, model.embed_dim),
            w1: Matrix::zeros(model.hidden, model.input_dim()),
            b1: vec![0.0; model.hidden],
            w2: Matrix::zeros(model.vocab_size, model.hidden),
            b2: vec![0.0; model.vocab_size],
        }
    }

    /// Clear every buffer.
    pub fn zero(&mut self) {
        self.embed.data.fill(0.0);
        self.w1.data.fill(0.0);
        self.b1.fill(0.0);
        self.w2.data.fill(0.0);
        self.b2.fill(0.0);
    }
}

/// Apply the activation.
#[must_use]
pub fn activate(act: Activation, v: f32) -> f32 {
    match act {
        Activation::Tanh => v.tanh(),
        Activation::Relu => v.max(0.0),
    }
}

/// Derivative of the activation at the *post*-activation value.
#[must_use]
pub fn activate_deriv(act: Activation, post: f32) -> f32 {
    match act {
        Activation::Tanh => 1.0 - post * post,
        Activation::Relu => {
            if post > 0.0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Numerically stable softmax into `out`.
pub fn softmax(logits: &[f32], out: &mut [f32]) {
    debug_assert_eq!(logits.len(), out.len());
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for (v, o) in logits.iter().zip(out.iter_mut()) {
        let e = (v - max).exp();
        *o = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for o in out {
        *o *= inv;
    }
}

/// Forward pass with NLL loss for `target`.
pub fn forward_loss(model: &Model, sc: &mut Scratch, target: usize) -> f32 {
    for s in 0..model.seq_len {
        let src = model.embed.row(sc.ctx[s]);
        let dest = &mut sc.x[s * model.embed_dim..(s + 1) * model.embed_dim];
        dest.copy_from_slice(src);
    }
    model.w1.mat_vec(&sc.x, &mut sc.h);
    for (hv, bv) in sc.h.iter_mut().zip(&model.b1) {
        *hv = activate(model.activation, *hv + bv);
    }
    model.w2.mat_vec(&sc.h, &mut sc.logits);
    for (lv, bv) in sc.logits.iter_mut().zip(&model.b2) {
        *lv += bv;
    }
    softmax(&sc.logits, &mut sc.probs);
    -sc.probs[target].max(1e-9).ln()
}

/// Forward pass without a target (fills `sc.probs` for greedy sampling).
pub fn forward_loss_no_target(model: &Model, sc: &mut Scratch) {
    for s in 0..model.seq_len {
        let src = model.embed.row(sc.ctx[s]);
        let dest = &mut sc.x[s * model.embed_dim..(s + 1) * model.embed_dim];
        dest.copy_from_slice(src);
    }
    model.w1.mat_vec(&sc.x, &mut sc.h);
    for (hv, bv) in sc.h.iter_mut().zip(&model.b1) {
        *hv = activate(model.activation, *hv + bv);
    }
    model.w2.mat_vec(&sc.h, &mut sc.logits);
    for (lv, bv) in sc.logits.iter_mut().zip(&model.b2) {
        *lv += bv;
    }
    softmax(&sc.logits, &mut sc.probs);
}

/// Accumulate parameter gradients for one example.
pub fn backward(model: &Model, g: &mut Grads, sc: &mut Scratch, target: usize) {
    sc.dlogits.copy_from_slice(&sc.probs);
    sc.dlogits[target] -= 1.0;

    for r in 0..model.vocab_size {
        let dl = sc.dlogits[r];
        g.b2[r] += dl;
        let gw2_row = g.w2.row_mut(r);
        for (gv, hv) in gw2_row.iter_mut().zip(&sc.h) {
            *gv += dl * hv;
        }
    }

    for c in 0..model.hidden {
        let mut s = 0.0_f32;
        for r in 0..model.vocab_size {
            s += model.w2.row(r)[c] * sc.dlogits[r];
        }
        sc.dh[c] = s;
    }
    for ((dp, dhv), hv) in sc.dhpre.iter_mut().zip(&sc.dh).zip(&sc.h) {
        *dp = dhv * activate_deriv(model.activation, *hv);
    }

    for r in 0..model.hidden {
        let dp = sc.dhpre[r];
        g.b1[r] += dp;
        let gw1_row = g.w1.row_mut(r);
        for (gv, xv) in gw1_row.iter_mut().zip(&sc.x) {
            *gv += dp * xv;
        }
    }

    for c in 0..model.input_dim() {
        let mut s = 0.0_f32;
        for r in 0..model.hidden {
            s += model.w1.row(r)[c] * sc.dhpre[r];
        }
        sc.dx[c] = s;
    }

    for s in 0..model.seq_len {
        let grow = g.embed.row_mut(sc.ctx[s]);
        let dx_slice = &sc.dx[s * model.embed_dim..(s + 1) * model.embed_dim];
        for (gv, dv) in grow.iter_mut().zip(dx_slice) {
            *gv += dv;
        }
    }
}

/// Mean NLL over the whole corpus.
#[allow(clippy::cast_precision_loss)] // corpus lengths for the demo stay far below f32 mantissa
pub fn eval_loss(model: &Model, sc: &mut Scratch) -> f32 {
    let mut total = 0.0_f32;
    for p in 0..model.corpus.len() {
        let target = model.sample_at(p, &mut sc.ctx);
        total += forward_loss(model, sc, target);
    }
    total / model.corpus.len() as f32
}

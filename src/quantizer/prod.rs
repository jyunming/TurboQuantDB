use nalgebra::DMatrix;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

use super::mse::MseQuantizer;
use super::qjl::QjlQuantizer;

/// TurboQuant_prod Quantizer structure
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ProdQuantizer {
    pub d: usize,
    pub b: usize,
    pub mse_quantizer: MseQuantizer,
    pub qjl_quantizer: QjlQuantizer,
}

impl ProdQuantizer {
    pub fn new(d: usize, b: usize, seed: u64) -> Self {
        assert!(b >= 2, "ProdQuantizer requires at least b=2");

        let mse_quantizer = MseQuantizer::new(d, b - 1, seed);
        // Use a different seed for the projection matrix to ensure independence
        let qjl_quantizer = QjlQuantizer::new(d, seed ^ 0xdeadbeef);

        Self {
            d,
            b,
            mse_quantizer,
            qjl_quantizer,
        }
    }

    /// Quantize a vector into (MSE indices, QJL bits, residual norm gamma)
    pub fn quantize(&self, x: &Array1<f64>) -> (Vec<usize>, Vec<i8>, f64) {
        let mut out = self.quantize_batch(&[x.clone()]);
        out.pop().unwrap_or_else(|| (Vec::new(), Vec::new(), 0.0))
    }

    /// Quantize a batch of vectors.
    /// Returns tuples of (MSE indices, QJL bits, gamma) in input order.
    pub fn quantize_batch(&self, xs: &[Array1<f64>]) -> Vec<(Vec<usize>, Vec<i8>, f64)> {
        let n = xs.len();
        if n == 0 {
            return Vec::new();
        }

        for x in xs {
            assert_eq!(x.len(), self.d);
        }

        let mut x_mat = DMatrix::zeros(self.d, n);
        for (col, x) in xs.iter().enumerate() {
            for row in 0..self.d {
                x_mat[(row, col)] = x[row];
            }
        }

        // MSE batch quantization.
        let mse_indices = self.mse_quantizer.quantize_batch(&x_mat);

        // Batch dequantize MSE result and compute residual/gamma.
        let x_tilde_mse_batch = self.mse_quantizer.dequantize_batch(&mse_indices);
        let mut r_mat = DMatrix::zeros(self.d, n);
        let mut gammas = vec![0.0f64; n];

        for col in 0..n {
            let mut gamma_sq = 0.0f64;
            for row in 0..self.d {
                let rv = x_mat[(row, col)] - x_tilde_mse_batch[(row, col)];
                r_mat[(row, col)] = rv;
                gamma_sq += rv * rv;
            }
            gammas[col] = gamma_sq.sqrt();
        }

        // QJL batch quantization on residuals.
        let qjl_all = self.qjl_quantizer.quantize_batch(&r_mat);

        mse_indices
            .into_iter()
            .zip(qjl_all)
            .zip(gammas)
            .map(|((idx, qjl), gamma)| (idx, qjl, gamma))
            .collect()
    }

    /// Dequantizes the components back into a real vector.
    pub fn dequantize(&self, idx: &[usize], qjl: &[i8], gamma: f64) -> Array1<f64> {
        // 1. MSE dequantization
        let x_tilde_mse = self.mse_quantizer.dequantize(idx);

        // 2. QJL dequantization
        let x_tilde_qjl = self.qjl_quantizer.dequantize(qjl, gamma);

        // 3. Combine: x_tilde = x_tilde_mse + x_tilde_qjl
        let mut result = Array1::zeros(self.d);
        for i in 0..self.d {
            result[i] = x_tilde_mse[i] + x_tilde_qjl[i];
        }

        result
    }
}

use nalgebra::DMatrix;
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// QJL Quantizer structure (1-bit inner product quantization on residual)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct QjlQuantizer {
    pub d: usize,
    pub projection: DMatrix<f64>,
}

impl QjlQuantizer {
    pub fn new(d: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        // S matrix is d x d, i.i.d N(0,1)
        let projection = crate::linalg::rotation::generate_projection_matrix(d, &mut rng);
        Self { d, projection }
    }

    /// Quantize residual vector into a bit vector.
    /// Return is represented as Vec<i8> containing +1 or -1.
    pub fn quantize(&self, r: &Array1<f64>) -> Vec<i8> {
        assert_eq!(r.len(), self.d);

        let mut r_mat = DMatrix::zeros(self.d, 1);
        for i in 0..self.d {
            r_mat[(i, 0)] = r[i];
        }

        self.quantize_batch(&r_mat)
            .into_iter()
            .next()
            .unwrap_or_default()
    }

    /// Quantize a batch of residual vectors arranged as a (d, n) matrix.
    pub fn quantize_batch(&self, rs: &DMatrix<f64>) -> Vec<Vec<i8>> {
        assert_eq!(rs.nrows(), self.d);

        let n = rs.ncols();
        if n == 0 {
            return Vec::new();
        }

        let s_r_batch = &self.projection * rs;
        let mut all_qjl = vec![vec![0i8; self.d]; n];

        for col in 0..n {
            for row in 0..self.d {
                all_qjl[col][row] = if s_r_batch[(row, col)] >= 0.0 { 1 } else { -1 };
            }
        }

        all_qjl
    }

    /// Dequantize QJL string.
    /// x_tilde = sqrt(pi / 2d) * gamma * S^T * qjl
    pub fn dequantize(&self, qjl: &[i8], gamma: f64) -> Array1<f64> {
        assert_eq!(qjl.len(), self.d);

        // Convert qjl to f64 nalgebra vector
        let mut qjl_nalg = DMatrix::zeros(self.d, 1);
        for i in 0..self.d {
            qjl_nalg[(i, 0)] = qjl[i] as f64;
        }

        let multiplier = (PI / (2.0 * self.d as f64)).sqrt() * gamma;

        // S^T * qjl
        let st_qjl = self.projection.transpose() * qjl_nalg;

        // Construct final array
        let mut result = Array1::zeros(self.d);
        for i in 0..self.d {
            result[i] = multiplier * st_qjl[(i, 0)];
        }

        result
    }
}

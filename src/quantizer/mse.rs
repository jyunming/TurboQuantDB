use nalgebra::DMatrix;
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use super::codebook::lloyd_max;

/// TurboQuant_mse Quantizer structure
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MseQuantizer {
    pub d: usize,
    pub b: usize,
    pub rotation: DMatrix<f64>,
    pub centroids: Vec<f64>,
}

impl MseQuantizer {
    /// Creates a new MseQuantizer with the specified dimension and bit-width.
    pub fn new(d: usize, b: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        // 1. Generate Rotation Matrix using Linalg module
        let rotation = crate::linalg::rotation::generate_random_rotation(d, &mut rng);

        // 2. Compute Lloyd-Max Codebook for beta distribution
        let num_points = 20_000;
        let centroids = lloyd_max(b, d, num_points);

        Self {
            d,
            b,
            rotation,
            centroids,
        }
    }

    /// Quantizes a single input vector.
    /// Input x: D-dimensional vector (should be unit length for optimal theoretical match).
    /// Output: b-bit indices for each coordinate.
    pub fn quantize(&self, x: &Array1<f64>) -> Vec<usize> {
        assert_eq!(x.len(), self.d);

        let mut x_mat = DMatrix::zeros(self.d, 1);
        for i in 0..self.d {
            x_mat[(i, 0)] = x[i];
        }

        self.quantize_batch(&x_mat)
            .into_iter()
            .next()
            .unwrap_or_default()
    }

    /// Quantizes a batch of vectors arranged as a (d, n) matrix.
    /// Returns n vectors of centroid indices, each with length d.
    pub fn quantize_batch(&self, xs: &DMatrix<f64>) -> Vec<Vec<usize>> {
        assert_eq!(xs.nrows(), self.d);

        let n = xs.ncols();
        if n == 0 {
            return Vec::new();
        }

        // y_batch = Pi * xs
        let y_batch = &self.rotation * xs;

        let mut all_indices = vec![vec![0usize; self.d]; n];
        for col in 0..n {
            for row in 0..self.d {
                let val = y_batch[(row, col)];
                all_indices[col][row] = self.nearest_centroid_index(val);
            }
        }

        all_indices
    }

    fn nearest_centroid_index(&self, val: f64) -> usize {
        let n = self.centroids.len();
        if n == 0 {
            return 0;
        }
        let pos = self.centroids.partition_point(|&c| c < val);
        if pos == 0 {
            0
        } else if pos >= n {
            n - 1
        } else {
            let lo = pos - 1;
            let hi = pos;
            if (val - self.centroids[lo]).abs() <= (self.centroids[hi] - val).abs() {
                lo
            } else {
                hi
            }
        }
    }

    /// Dequantizes a single index vector back to a real vector.
    pub fn dequantize(&self, indices: &[usize]) -> Array1<f64> {
        assert_eq!(indices.len(), self.d);
        let batch = vec![indices.to_vec()];
        let x_tilde_batch = self.dequantize_batch(&batch);

        let mut x_tilde = Array1::zeros(self.d);
        for i in 0..self.d {
            x_tilde[i] = x_tilde_batch[(i, 0)];
        }
        x_tilde
    }

    /// Dequantizes a batch of index vectors into a (d, n) matrix.
    pub fn dequantize_batch(&self, indices_batch: &[Vec<usize>]) -> DMatrix<f64> {
        let n = indices_batch.len();
        if n == 0 {
            return DMatrix::zeros(self.d, 0);
        }

        let mut y_tilde_batch = DMatrix::zeros(self.d, n);
        for (col, indices) in indices_batch.iter().enumerate() {
            assert_eq!(indices.len(), self.d);
            for row in 0..self.d {
                y_tilde_batch[(row, col)] = self.centroids[indices[row]];
            }
        }

        self.rotation.transpose() * y_tilde_batch
    }
}

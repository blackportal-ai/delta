use crate::devices::Device;
use ndarray::{Dim, IxDyn, IxDynImpl, Shape};
use rand::SeedableRng;
use rand::prelude::*;
use rand_distr::Distribution;
use std::ops::Index;

use super::{errors::ProjectionError, tensor_ops::Tensor};

type Result<T> = std::result::Result<T, ProjectionError>;

/// Generates a tensor filled with random values from a normal distribution.
///
/// # Arguments
///
/// * `shape` - The shape of the tensor to generate
/// * `seed` - Random seed for reproducibility
/// * `device` - Device to store the tensor on
///
/// # Returns
///
/// A tensor filled with random values from N(0,1)
fn stable_randn(shape: &[usize], seed: u64, device: &Device) -> Tensor {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let dist = rand_distr::Normal::new(0.0, 1.0).unwrap();

    let shape = Shape::from(IxDyn(shape));
    let mut tensor = Tensor::zeros(shape, device.clone());
    tensor.fill_random(&mut rng, &dist);
    tensor
}

/// Generates a new random seed from an existing seed.
///
/// # Arguments
///
/// * `seed` - The seed to generate a new seed from
/// * `adv` - The advancement parameter
///
/// # Returns
///
/// A new random seed
fn next_seed(seed: u64, adv: u64) -> u64 {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let nums: Vec<u64> = (0..adv).map(|_| rng.gen()).collect();
    *nums.last().unwrap()
}

const ADV_DEFAULT: u64 = 0xF;

/// Projects high-dimensional gradients to a lower-dimensional space.
///
/// This helps reduce memory and computation requirements while preserving
/// important gradient information during training.
#[allow(dead_code)]
pub struct GradientProjector {
    /// Target rank for the projected gradients
    rank: usize,
    /// Whether to print verbose output
    verbose: bool,
    /// Number of iterations between projection matrix updates
    update_proj_gap: usize,
    /// Scale factor for the projections
    scale: f32,
    /// Type of projection to use
    proj_type: ProjectionType,
    /// Current orthogonal projection matrix
    ortho_matrix: Option<Tensor>,
    /// Random seed for reproducibility
    seed: u64,
    /// Counter for SVD operations
    svd_count: usize,
}

/// Different types of gradient projection methods.
#[derive(Debug, Clone, Copy)]
pub enum ProjectionType {
    /// Standard projection based on matrix dimensions
    Standard,
    /// Reverse of standard projection
    ReverseStandard,
    /// Left projection only
    Left,
    /// Right projection only
    Right,
    /// Full rank projection (no dimension reduction)
    Full,
}

impl Tensor {
    /// Gets a mutable slice of the underlying data.
    fn data_mut(&mut self) -> &mut [f32] {
        self.data.as_slice_mut().unwrap()
    }

    /// Fills the tensor with random values from a distribution.
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator
    /// * `dist` - Distribution to sample from
    fn fill_random(&mut self, rng: &mut rand::rngs::StdRng, dist: &rand_distr::Normal<f32>) {
        let data = self.data_mut();
        for val in data.iter_mut() {
            *val = dist.sample(rng) as f32;
        }
    }
}

// TODO: Temporary hack
/// Wrapper to allow indexing into ndarray shapes.
struct ShapeWrapper<'a>(&'a Shape<Dim<IxDynImpl>>);

impl<'a> Index<usize> for ShapeWrapper<'a> {
    type Output = usize;
    fn index(&self, idx: usize) -> &Self::Output {
        &self.0.raw_dim()[idx]
    }
}

impl GradientProjector {
    /// Creates a new GradientProjector.
    ///
    /// # Arguments
    ///
    /// * `rank` - Target rank for projected gradients
    /// * `verbose` - Whether to print verbose output
    /// * `update_proj_gap` - Iterations between projection updates
    /// * `scale` - Scale factor for projections
    /// * `proj_type` - Type of projection to use
    /// * `seed` - Random seed for reproducibility
    pub fn new(
        rank: usize,
        verbose: bool,
        update_proj_gap: usize,
        scale: f32,
        proj_type: ProjectionType,
        seed: u64,
    ) -> Result<Self> {
        if rank == 0 {
            return Err(ProjectionError::InvalidRank("Rank must be positive".into()));
        }
        Ok(Self {
            rank,
            verbose,
            update_proj_gap,
            scale,
            proj_type,
            ortho_matrix: None,
            seed,
            svd_count: 0,
        })
    }

    /// Projects a gradient tensor to a lower rank.
    ///
    /// # Arguments
    ///
    /// * `full_rank_grad` - The full gradient tensor to project
    /// * `iter` - Current iteration number
    ///
    /// # Returns
    ///
    /// The projected gradient tensor
    pub fn project(&mut self, full_rank_grad: &Tensor, iter: usize) -> Result<Tensor> {
        match self.proj_type {
            ProjectionType::Standard => self.project_standard(full_rank_grad, iter),
            ProjectionType::ReverseStandard => self.project_reverse_standard(full_rank_grad, iter),
            ProjectionType::Left => self.project_left(full_rank_grad, iter),
            ProjectionType::Right => self.project_right(full_rank_grad, iter),
            ProjectionType::Full => Err(ProjectionError::InvalidProjectionType(
                "Full rank projection not implemented".into(),
            )),
        }
    }

    /// Updates the orthogonal projection matrix.
    ///
    /// # Arguments
    ///
    /// * `full_rank_grad` - Current gradient tensor
    /// * `proj_type` - Type of projection to use
    fn update_ortho_matrix(
        &mut self,
        full_rank_grad: &Tensor,
        proj_type: ProjectionType,
    ) -> Result<()> {
        self.ortho_matrix =
            Some(self.get_orthogonal_matrix(full_rank_grad, self.rank, proj_type, self.seed)?);
        self.seed = next_seed(self.seed, ADV_DEFAULT);
        Ok(())
    }

    /// Generates an orthogonal matrix for projection.
    ///
    /// # Arguments
    ///
    /// * `weights` - Current weights tensor
    /// * `rank` - Target rank
    /// * `proj_type` - Type of projection
    /// * `seed` - Random seed
    ///
    /// # Returns
    ///
    /// An orthogonal projection matrix
    fn get_orthogonal_matrix(
        &self,
        weights: &Tensor,
        rank: usize,
        proj_type: ProjectionType,
        seed: u64,
    ) -> Result<Tensor> {
        let shape_val = weights.shape();
        let shape = ShapeWrapper(&shape_val);
        let device = &weights.device;

        match proj_type {
            ProjectionType::Left => {
                if rank > shape[0] {
                    return Err(ProjectionError::InvalidRank(
                        "Rank cannot be larger than input dimension".into(),
                    ));
                }
                let proj = stable_randn(&[shape[0], rank], seed, device);
                Ok(proj.div_scalar(f32::sqrt(rank as f32)))
            }
            ProjectionType::Right => {
                if rank > shape[1] {
                    return Err(ProjectionError::InvalidRank(
                        "Rank cannot be larger than output dimension".into(),
                    ));
                }
                let proj = stable_randn(&[rank, shape[1]], seed, device);
                Ok(proj.div_scalar(f32::sqrt(rank as f32)))
            }
            _ => Err(ProjectionError::InvalidProjectionType(
                "Invalid projection type for orthogonal matrix generation".into(),
            )),
        }
    }

    /// Performs standard projection based on matrix dimensions.
    ///
    /// # Arguments
    ///
    /// * `full_rank_grad` - Gradient tensor to project
    /// * `iter` - Current iteration
    ///
    /// # Returns
    ///
    /// The projected gradient tensor
    fn project_standard(&mut self, full_rank_grad: &Tensor, iter: usize) -> Result<Tensor> {
        let shape_val = full_rank_grad.shape();
        let shape = ShapeWrapper(&shape_val);

        if shape[0] >= shape[1] {
            if self.ortho_matrix.is_none() || iter % self.update_proj_gap == 0 {
                self.update_ortho_matrix(full_rank_grad, ProjectionType::Right)?;
            }
            Ok(full_rank_grad.dot(
                &self
                    .ortho_matrix
                    .as_ref()
                    .ok_or(ProjectionError::UninitializedMatrix)?
                    .transpose(),
            ))
        } else {
            if self.ortho_matrix.is_none() || iter % self.update_proj_gap == 0 {
                self.update_ortho_matrix(full_rank_grad, ProjectionType::Left)?;
            }
            Ok(self
                .ortho_matrix
                .as_ref()
                .ok_or(ProjectionError::UninitializedMatrix)?
                .transpose()
                .dot(full_rank_grad))
        }
    }

    /// Performs reverse standard projection.
    ///
    /// # Arguments
    ///
    /// * `full_rank_grad` - Gradient tensor to project
    /// * `iter` - Current iteration
    ///
    /// # Returns
    ///
    /// The projected gradient tensor
    fn project_reverse_standard(&mut self, full_rank_grad: &Tensor, iter: usize) -> Result<Tensor> {
        let shape_val = full_rank_grad.shape();
        let shape = ShapeWrapper(&shape_val);
        if shape[0] >= shape[1] {
            if self.ortho_matrix.is_none() || iter % self.update_proj_gap == 0 {
                self.update_ortho_matrix(full_rank_grad, ProjectionType::Left)?;
            }
            Ok(self
                .ortho_matrix
                .as_ref()
                .ok_or(ProjectionError::UninitializedMatrix)?
                .transpose()
                .dot(full_rank_grad))
        } else {
            if self.ortho_matrix.is_none() || iter % self.update_proj_gap == 0 {
                self.update_ortho_matrix(full_rank_grad, ProjectionType::Right)?;
            }
            Ok(full_rank_grad.dot(
                &self
                    .ortho_matrix
                    .as_ref()
                    .ok_or(ProjectionError::UninitializedMatrix)?
                    .transpose(),
            ))
        }
    }

    /// Performs left-only projection.
    ///
    /// # Arguments
    ///
    /// * `full_rank_grad` - Gradient tensor to project
    /// * `iter` - Current iteration
    ///
    /// # Returns
    ///
    /// The projected gradient tensor
    fn project_left(&mut self, full_rank_grad: &Tensor, iter: usize) -> Result<Tensor> {
        if self.ortho_matrix.is_none() || iter % self.update_proj_gap == 0 {
            self.update_ortho_matrix(full_rank_grad, ProjectionType::Left)?;
        }
        Ok(self
            .ortho_matrix
            .as_ref()
            .ok_or(ProjectionError::UninitializedMatrix)?
            .transpose()
            .dot(full_rank_grad))
    }

    /// Performs right-only projection.
    ///
    /// # Arguments
    ///
    /// * `full_rank_grad` - Gradient tensor to project
    /// * `iter` - Current iteration
    ///
    /// # Returns
    ///
    /// The projected gradient tensor
    fn project_right(&mut self, full_rank_grad: &Tensor, iter: usize) -> Result<Tensor> {
        if self.ortho_matrix.is_none() || iter % self.update_proj_gap == 0 {
            self.update_ortho_matrix(full_rank_grad, ProjectionType::Right)?;
        }
        Ok(full_rank_grad.dot(
            &self.ortho_matrix.as_ref().ok_or(ProjectionError::UninitializedMatrix)?.transpose(),
        ))
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    /*
    // Constructor Tests
    test_new_with_valid_parameters
    test_new_with_zero_rank_returns_error
    test_new_with_negative_scale
    test_new_with_zero_update_proj_gap

    // Projection Type Tests
    test_standard_projection_tall_matrix
    test_standard_projection_wide_matrix
    test_reverse_standard_projection_tall_matrix
    test_reverse_standard_projection_wide_matrix
    test_left_projection
    test_right_projection
    test_full_projection_returns_error

    // Shape Tests
    test_left_projection_rank_exceeds_input_dim
    test_right_projection_rank_exceeds_output_dim
    test_projection_with_empty_tensor
    test_projection_with_single_element_tensor
    test_projection_with_rank_equal_to_dimension

    // Orthogonal Matrix Tests
    test_ortho_matrix_initialization
    test_ortho_matrix_update_frequency
    test_ortho_matrix_persistence_between_iterations
    test_ortho_matrix_different_seeds

    // Seed Generation Tests
    test_next_seed_deterministic
    test_next_seed_with_different_adv_values
    test_next_seed_sequence_uniqueness

    // Device Tests
    test_projection_same_device
    test_projection_different_device_error

    // Numerical Stability Tests
    test_projection_with_large_values
    test_projection_with_small_values
    test_projection_with_mixed_magnitude_values
    test_projection_numerical_precision

    // Error Handling Tests
    test_uninitialized_matrix_error
    test_invalid_projection_type_error
    test_incompatible_shape_error
    test_device_mismatch_error
    test_invalid_rank_error

    // Edge Cases
    test_projection_with_rank_one
    test_projection_with_max_rank
    test_projection_with_square_matrix
    test_projection_with_tall_skinny_matrix
    test_projection_with_short_wide_matrix

    // Memory Tests
    test_memory_usage_large_matrices
    test_memory_leak_multiple_projections
    test_memory_cleanup_after_error

    // Performance Tests
    test_projection_performance_large_matrix
    test_update_frequency_impact
    test_seed_generation_performance

    // Integration Tests
    test_multiple_projections_same_tensor
    test_projections_different_tensors
    test_projection_chain
    test_projection_with_gradient_descent
    */
}

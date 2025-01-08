//! BSD 3-Clause License
//!
//! Copyright (c) 2025, BlackPortal ○
//!
//! Redistribution and use in source and binary forms, with or without
//! modification, are permitted provided that the following conditions are met:
//!
//! 1. Redistributions of source code must retain the above copyright notice, this
//!    list of conditions and the following disclaimer.
//!
//! 2. Redistributions in binary form must reproduce the above copyright notice,
//!    this list of conditions and the following disclaimer in the documentation
//!    and/or other materials provided with the distribution.
//!
//! 3. Neither the name of the copyright holder nor the names of its
//!    contributors may be used to endorse or promote products derived from
//!    this software without specific prior written permission.
//!
//! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//! DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//! FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//! DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//! SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//! OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use ndarray::Ix;

use crate::activations::Activation;
use crate::common::Tensor;

/// A struct representing the Parametric Rectified Linear Unit (PReLU) activation function.
#[derive(Debug)]
pub struct PreluActivation {
    /// The learnable parameter `alpha` for the PReLU activation function.
    pub alpha: f32,
}

impl PreluActivation {
    /// Creates a new instance of `PreluActivation` with the specified `alpha` parameter.
    ///
    /// # Arguments
    ///
    /// * `alpha` - The parameter for the negative slope.
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }
}

impl Activation for PreluActivation {
    /// Applies PReLU activation to the input tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// The output tensor after applying PReLU activation.
    fn activate(&self, input: &Tensor) -> Tensor {
        let alpha = self.alpha;
        input.map(|x| if x > 0.0 { x } else { alpha * x })
    }

    /// Computes the derivative of PReLU activation for the input tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor representing the derivative of PReLU activation.
    fn derivative(&self, input: &Tensor) -> Tensor {
        let alpha = self.alpha;
        input.map(|x| if x > 0.0 { 1.0 } else { alpha })
    }

    /// Initializes the weights for the Softmax activation function.
    ///
    /// # Arguments
    ///
    /// * `input_units` - The number of input units.
    ///
    /// # Returns
    ///
    /// The initial weight value for the Softmax activation function.
    fn initialize(&self, input_units: Ix) -> f32 {
        (1.0 / input_units as f32).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{IxDyn, Shape};

    use super::*;

    #[test]
    fn test_prelu_activation() {
        let input = Tensor::new(vec![1.0, -2.0, 3.0, -4.0], Shape::from(IxDyn(&[2, 2])));
        let prelu = PreluActivation::new(0.1);
        let output = prelu.activate(&input);

        assert_eq!(output.data.iter().cloned().collect::<Vec<f32>>(), vec![1.0, -0.2, 3.0, -0.4]);
        assert_eq!(output.data.shape().to_vec(), vec![2, 2]);
    }

    #[test]
    fn test_prelu_derivative() {
        let input = Tensor::new(vec![1.0, -2.0, 3.0, -4.0], Shape::from(IxDyn(&[2, 2])));
        let prelu = PreluActivation::new(0.1);
        let derivative = prelu.derivative(&input);

        assert_eq!(derivative.data.iter().cloned().collect::<Vec<f32>>(), vec![1.0, 0.1, 1.0, 0.1]);
        assert_eq!(derivative.data.shape().to_vec(), vec![2, 2]);
    }

    #[test]
    fn test_prelu_initialize() {
        let prelu = PreluActivation::new(0.1);
        assert_eq!(prelu.initialize(10), 0.31622776);
    }
}

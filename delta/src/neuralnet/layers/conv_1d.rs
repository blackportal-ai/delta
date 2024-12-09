//! BSD 3-Clause License
//!
//! Copyright (c) 2024, Marcus Cvjeticanin, Chase Willden
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

use crate::activations::Activation;
use crate::common::{LayerError, Shape, Tensor};
use crate::neuralnet::layers::Layer;

/// A struct representing a 1D convolutional layer.
#[derive(Debug)]
pub struct Conv1D {
    name: String,
    filters: usize,
    kernel_size: usize,
    #[allow(dead_code)]
    stride: usize,
    #[allow(dead_code)]
    padding: usize,
    weights: Option<Tensor>,
    bias: Option<Tensor>,
    #[allow(dead_code)]
    activation: Option<Box<dyn Activation>>,
    trainable: bool,
    weights_grad: Option<Tensor>,
    bias_grad: Option<Tensor>,
    #[allow(dead_code)]
    input: Option<Tensor>,
}

impl Conv1D {
    /// Creates a new Conv1D layer.
    ///
    /// # Arguments
    ///
    /// * `filters` - The number of filters (output channels).
    /// * `kernel_size` - The size of the convolutional kernel.
    /// * `stride` - The stride of the convolution.
    /// * `padding` - The amount of zero-padding added to both sides of the input.
    /// * `activation` - The activation function to use.
    ///
    /// # Returns
    ///
    /// A new instance of the Conv1D layer.
    pub fn new<A: Activation + 'static>(
        filters: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        activation: Option<A>,
        trainable: bool,
    ) -> Self {
        Self {
            name: format!("Conv1D_{}", filters),
            filters,
            kernel_size,
            stride,
            padding,
            weights: None,
            bias: None,
            activation: activation.map(|act| Box::new(act) as Box<dyn Activation>),
            trainable,
            weights_grad: None,
            bias_grad: None,
            input: None,
        }
    }
}

impl Layer for Conv1D {
    /// Builds the layer with the given input shape.
    ///
    /// # Arguments
    ///
    /// * `input_shape` - The shape of the input tensor.
    fn build(&mut self, input_shape: Shape) -> Result<(), LayerError> {
        println!(
            "Building Conv1D layer with input shape: {:?} and filters: {}",
            input_shape, self.filters
        );
        let input_channels = input_shape.0.last().expect("Input shape must not be empty");
        self.weights = Some(Tensor::random(vec![
            self.filters,
            *input_channels,
            self.kernel_size,
        ]));
        self.bias = Some(Tensor::zeros(vec![self.filters]));
        Ok(())
    }

    /// Performs a forward pass through the layer.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// The output tensor.
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        let _ = input;
        unimplemented!()
    }

    /// Performs a backward pass through the layer.
    ///
    /// # Arguments
    ///
    /// * `_grad` - The gradient tensor.
    ///
    /// # Returns
    ///
    /// The gradient tensor with respect to the input.
    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let _ = grad;
        unimplemented!()
    }

    /// Returns the output shape of the layer.
    ///
    /// # Returns
    ///
    /// A `Shape` representing the output shape of the layer.
    fn output_shape(&self) -> Result<Shape, LayerError> {
        unimplemented!()
    }

    /// Returns the number of parameters in the layer.
    ///
    /// # Returns
    ///
    /// A `usize` representing the number of parameters in the layer.
    fn param_count(&self) -> Result<(usize, usize), LayerError> {
        let weights_count = self.weights.as_ref().map_or(0, |w| w.data.len());
        let bias_count = self.bias.as_ref().map_or(0, |b| b.data.len());
        Ok((weights_count, bias_count))
    }

    /// Returns the name of the layer.
    ///
    /// # Returns
    ///
    /// A `&str` representing the name of the layer.
    fn name(&self) -> &str {
        &self.name
    }

    /// Updates the weights of the layer using the given gradient and optimizer.
    ///
    /// # Arguments
    ///
    /// * `grad` - The gradient tensor.
    /// * `optimizer` - The optimizer to use.
    fn update_weights(
        &mut self,
        optimizer: &mut Box<dyn crate::optimizers::Optimizer>,
    ) -> Result<(), LayerError> {
        if !self.trainable {
            return Ok(());
        }

        // Update weights
        if let Some(ref weights_grad) = self.weights_grad {
            optimizer
                .step(self.weights.as_mut().unwrap(), weights_grad)
                .map_err(|e| LayerError::OptimizerError(e))?;
        }

        // Update bias
        if let Some(ref bias_grad) = self.bias_grad {
            optimizer
                .step(self.bias.as_mut().unwrap(), bias_grad)
                .map_err(|e| LayerError::OptimizerError(e))?;
        }

        // Clear gradients after update
        self.weights_grad = None;
        self.bias_grad = None;

        Ok(())
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::activations::relu::ReluActivation;
//
//     #[test]
//     fn test_conv1d_layer() {
//         let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5, 1]);
//         let mut conv1d_layer = Conv1D::new(2, 3, 1, 1, Some(ReluActivation::new()), true);
//         conv1d_layer
//             .build(Shape::new(vec![1, 5, 1]))
//             .expect("Failed to build layer");
//
//         let output = conv1d_layer.forward(&input).unwrap();
//
//         assert_eq!(output.shape(), &[1, 5, 2]);
//         assert_eq!(output.data.len(), 10);
//     }
//
//     #[test]
//     fn test_conv1d_layer_forward_pass() {
//         let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5, 1]);
//         let mut conv1d_layer = Conv1D::new(2, 3, 1, 1, Some(ReluActivation::new()), true);
//         conv1d_layer
//             .build(Shape::new(vec![1, 5, 1]))
//             .expect("Failed to build layer");
//
//         let output = conv1d_layer.forward(&input).unwrap();
//
//         assert_eq!(output.shape(), &[1, 5, 2]);
//         assert_eq!(output.data.len(), 10);
//     }
//
//     #[test]
//     fn test_conv1d_layer_backward_pass() {
//         let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5, 1]);
//         let mut conv1d_layer = Conv1D::new(2, 3, 1, 1, Some(ReluActivation::new()), true);
//         conv1d_layer.input = Some(input.clone());
//         conv1d_layer
//             .build(Shape::new(vec![1, 5, 1]))
//             .expect("Failed to build layer");
//
//         let grad = Tensor::new(
//             vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
//             vec![1, 5, 2],
//         );
//         let output = conv1d_layer.backward(&grad).unwrap();
//
//         assert_eq!(output.shape(), &[1, 5, 1]);
//         assert_eq!(output.data.len(), 5);
//     }
// }

// BSD 3-Clause License
//
// Copyright (c) 2025, BlackPortal ○
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use log::debug;
use ndarray::{ArrayD, Dimension, IxDyn, Shape, s};
use serde_json;

use crate::devices::Device;

use std::fmt::Debug;

use super::{
    activations::Activation, errors::LayerError, optimizers::Optimizer, tensor_ops::Tensor,
};

// A trait representing a neural network layer.
pub trait Layer: Debug {
    /// Builds the layer with the given input shape.
    ///
    /// # Arguments
    ///
    /// * `input_shape` - The shape of the input tensor.
    fn build(&mut self, input_shape: Shape<IxDyn>) -> Result<(), LayerError>;

    /// Performs the forward pass of the layer.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// The output tensor after applying the layer.
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, LayerError>;

    /// Performs the backward pass of the layer.
    ///
    /// # Arguments
    ///
    /// * `grad` - The gradient tensor from the next layer.
    ///
    /// # Returns
    ///
    /// The gradient tensor to be passed to the previous layer.
    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError>;

    /// Returns the output shape of the layer.
    ///
    /// # Returns
    ///
    /// A `Shape` representing the output shape of the layer.
    fn output_shape(&self) -> Result<Shape<IxDyn>, LayerError>;

    /// Returns the number of parameters in the layer.
    ///
    /// # Returns
    ///
    /// A tuple `(usize, usize)` representing the number of trainable and non-trainable parameters in the layer.
    fn param_count(&self) -> Result<(usize, usize), LayerError>;

    /// Returns the name of the layer.
    ///
    /// # Returns
    ///
    /// A `&str` representing the name of the layer.
    fn name(&self) -> &str;

    /// Sets the device for the layer.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to set for the layer.
    fn set_device(&mut self, _device: &Device);

    /// Returns the number of units in the layer.
    ///
    /// # Returns
    ///
    /// A `usize` representing the number of units in the layer. Default is 0.
    fn units(&self) -> usize {
        0
    }

    /// Updates the weights of the layer.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - The optimizer to use.
    fn update_weights(&mut self, optimizer: &mut Box<dyn Optimizer>) -> Result<(), LayerError>;

    /// Returns the weights of the layer as a serializable format.
    ///
    /// # Returns
    ///
    /// A `serde_json::Value` containing the layer's weights.
    fn get_weights(&self) -> serde_json::Value {
        serde_json::json!({})
    }

    /// Returns the layer's configuration as a serializable format.
    ///
    /// # Returns
    ///
    /// A `serde_json::Value` containing the layer's configuration.
    fn get_config(&self) -> serde_json::Value {
        serde_json::json!({})
    }

    /// Returns the type name of the layer.
    ///
    /// # Returns
    ///
    /// A `String` representing the type name of the layer.
    fn type_name(&self) -> String {
        std::any::type_name::<Self>().split("::").last().unwrap_or("Unknown").to_string()
    }
}

/// A struct representing the output of a layer.
#[derive(Debug)]
pub struct LayerOutput {
    /// The output tensor of the layer.
    pub output: Tensor,
    /// The gradients tensor of the layer.
    pub gradients: Tensor,
}

/// A dense (fully connected) layer.
#[derive(Debug)]
pub struct Dense {
    name: String,
    weights: Option<Tensor>,
    bias: Option<Tensor>,
    units: usize,
    activation: Option<Box<dyn Activation>>,
    trainable: bool,
    weights_grad: Option<Tensor>,
    bias_grad: Option<Tensor>,
    input: Option<Tensor>,
    device: Device,
}

impl Dense {
    /// Creates a new dense layer.
    ///
    /// # Arguments
    ///
    /// * `units` - The number of output units.
    /// * `activation` - The activation function to use.
    /// * `trainable` - Whether the layer is trainable.
    pub fn new<A: Activation + 'static>(
        units: usize,
        activation: Option<A>,
        trainable: bool,
    ) -> Self {
        Dense {
            name: format!("dense_{}", units),
            weights: None,
            bias: None,
            units,
            activation: activation.map(|a| Box::new(a) as Box<dyn Activation>),
            trainable,
            weights_grad: None,
            bias_grad: None,
            input: None,
            device: Device::default(),
        }
    }
}

impl Layer for Dense {
    /// Builds the layer with the given input shape.
    ///
    /// # Arguments
    ///
    /// * `input_shape` - The shape of the input tensor.
    fn build(&mut self, input_shape: Shape<IxDyn>) -> Result<(), LayerError> {
        debug!(
            "Building Dense layer with input shape: {:?} and units: {}",
            input_shape, self.units
        );

        let raw_dim = input_shape.raw_dim();
        let array_view = raw_dim.as_array_view();
        let input_units = array_view.last().ok_or(LayerError::InvalidInputShape)?;

        // Choose initialization strategy based on the activation function
        let stddev = if let Some(ref activation) = self.activation {
            activation.initialize(*input_units)
        } else {
            (1.0 / *input_units as f32).sqrt() // Xavier initialization for no activation
        };

        // Initialize weights using random normal distribution
        self.weights = Some(Tensor::random_normal(
            Shape::from(IxDyn(&[*input_units, self.units])),
            0.0,
            stddev,
        ));

        // Initialize bias to zeros
        self.bias = Some(Tensor::zeros(Shape::from(IxDyn(&[self.units])), self.device.clone()));

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
        let weights = self.weights.as_ref().ok_or(LayerError::UninitializedWeights)?;
        let bias = self.bias.as_ref().ok_or(LayerError::UninitializedBias)?;

        self.input = Some(input.clone());

        // Perform forward pass: Z = input · weights + bias
        let z = input.dot(weights).add(bias);

        // Apply activation if present
        let z =
            if let Some(ref activation) = self.activation { activation.activate(&z) } else { z };

        Ok(z)
    }

    /// Performs a backward pass through the layer.
    ///
    /// # Arguments
    ///
    /// * `grad` - The gradient tensor.
    ///
    /// # Returns
    ///
    /// The gradient tensor with respect to the input.
    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        // Ensure weights and input are initialized
        let weights = self.weights.as_ref().expect("Weights must be initialized");
        let input = self.input.as_ref().expect("Input must be initialized");

        // Calculate the gradient with respect to weights and bias
        let weights_grad = input.transpose().dot(grad);
        let bias_grad = grad.sum_along_axis(0);

        // Store the gradients
        if self.trainable {
            self.weights_grad = Some(weights_grad);
            self.bias_grad = Some(bias_grad);
        }

        // Calculate the gradient with respect to the input
        let input_grad = grad.dot(&weights.transpose());

        Ok(input_grad)
    }

    /// Returns the output shape of the layer.
    ///
    /// # Returns
    ///
    /// A `Shape` representing the output shape of the layer.
    fn output_shape(&self) -> Result<Shape<IxDyn>, LayerError> {
        let shape = Shape::from(IxDyn(&[self.units]));
        Ok(shape)
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

    /// Sets the device for the layer.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to set for the layer.
    fn set_device(&mut self, device: &Device) {
        self.device = device.clone();

        if let Some(ref mut weights) = self.weights {
            weights.device = device.clone();
        }
        if let Some(ref mut bias) = self.bias {
            bias.device = device.clone();
        }
        if let Some(ref mut input) = self.input {
            input.device = device.clone();
        }
    }

    /// Updates the weights of the layer using the given gradient and optimizer.
    ///
    /// # Arguments
    ///
    /// * `grad` - The gradient tensor.
    /// * `optimizer` - The optimizer to use.
    fn update_weights(&mut self, optimizer: &mut Box<dyn Optimizer>) -> Result<(), LayerError> {
        if !self.trainable {
            return Ok(());
        }

        // Update weights
        if let Some(ref weights_grad) = self.weights_grad {
            optimizer
                .step(self.weights.as_mut().unwrap(), weights_grad)
                .map_err(LayerError::OptimizerError)?;
        }

        if let Some(ref bias_grad) = self.bias_grad {
            optimizer
                .step(self.bias.as_mut().unwrap(), bias_grad)
                .map_err(LayerError::OptimizerError)?;
        }

        // Clear gradients after update
        self.weights_grad = None;
        self.bias_grad = None;

        Ok(())
    }

    fn get_weights(&self) -> serde_json::Value {
        serde_json::json!({
            "weights": self.weights.as_ref().map(|w| w.to_vec()),
            "bias": self.bias.as_ref().map(|b| b.to_vec())
        })
    }

    fn get_config(&self) -> serde_json::Value {
        serde_json::json!({
            "units": self.units,
            "trainable": self.trainable,
            "activation": self.activation.as_ref().map(|a| a.name())
        })
    }
}

/// A flatten layer that reshapes the input tensor to a 1D vector.
#[derive(Debug)]
pub struct Flatten {
    name: String,
    input_shape: Shape<IxDyn>,
}

impl Flatten {
    /// Creates a new flatten layer.
    ///
    /// # Arguments
    ///
    /// * `input_shape` - The shape of the input tensor.
    ///
    /// # Returns
    ///
    /// A new instance of the flatten layer.
    pub fn new(input_shape: Shape<IxDyn>) -> Self {
        Self { name: "Flatten".to_string(), input_shape }
    }
}

impl Layer for Flatten {
    /// Builds the layer with the given input shape.
    ///
    /// # Arguments
    ///
    /// * `input_shape` - The shape of the input tensor.
    fn build(&mut self, input_shape: Shape<IxDyn>) -> Result<(), LayerError> {
        self.input_shape = input_shape;
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
        let batch_size = input.data.shape()[0];
        let flattened_size = input.data.len() / batch_size;
        let reshaped = input.reshape(IxDyn(&[batch_size, flattened_size]));
        Ok(reshaped)
    }

    /// Performs a backward pass through the layer.
    ///
    /// # Arguments
    ///
    /// * `grad` - The gradient tensor.
    ///
    /// # Returns
    ///
    /// The gradient tensor with respect to the input.
    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let batch_size = grad.shape().raw_dim().as_array_view()[0];
        let new_shape = [batch_size]
            .iter()
            .chain(self.input_shape.raw_dim().as_array_view().iter())
            .cloned()
            .collect::<Vec<_>>();

        Ok(grad.reshape(IxDyn(&new_shape)))
    }

    /// Returns the output shape of the layer.
    ///
    /// # Returns
    ///
    /// A `Shape` representing the output shape of the layer.
    fn output_shape(&self) -> Result<Shape<IxDyn>, LayerError> {
        let shape =
            Shape::from(IxDyn(&[self.input_shape.raw_dim().as_array_view().iter().product()]));
        Ok(shape)
    }

    /// Returns the number of parameters in the layer.
    ///
    /// # Returns
    ///
    /// A `usize` representing the number of parameters in the layer.
    fn param_count(&self) -> Result<(usize, usize), LayerError> {
        Ok((0, 0))
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
    /// * `optimizer` - The optimizer to use.
    fn update_weights(&mut self, optimizer: &mut Box<dyn Optimizer>) -> Result<(), LayerError> {
        let _ = optimizer;
        // Do nothing
        Ok(())
    }

    /// Sets the device for the layer.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to set for the layer.
    fn set_device(&mut self, _device: &crate::devices::Device) {
        // Do nothing
    }
}

#[derive(Debug)]
pub struct Conv2D {
    name: String,
    filters: usize,
    kernel_size: (usize, usize),
    strides: (usize, usize),
    padding: (usize, usize),
    weights: Option<Tensor>,
    bias: Option<Tensor>,
    weights_grad: Option<Tensor>,
    bias_grad: Option<Tensor>,
    input: Option<Tensor>,
    device: Device,
    trainable: bool,
}

impl Conv2D {
    pub fn new(
        filters: usize,
        kernel_size: (usize, usize),
        strides: (usize, usize),
        padding: (usize, usize),
        trainable: bool,
    ) -> Self {
        Self {
            name: format!("conv2d_{}x{}", kernel_size.0, kernel_size.1),
            filters,
            kernel_size,
            strides,
            padding,
            weights: None,
            bias: None,
            weights_grad: None,
            bias_grad: None,
            input: None,
            device: Device::default(),
            trainable,
        }
    }

    fn im2col(
        input: &Tensor,
        kernel_size: (usize, usize),
        strides: (usize, usize),
        padding: (usize, usize),
    ) -> Tensor {
        let binding = input.shape();
        let input_shape = binding.raw_dim();
        let (batch_size, in_channels, in_height, in_width) =
            (input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
        let (kernel_height, kernel_width) = kernel_size;
        let (stride_height, stride_width) = strides;
        let (pad_height, pad_width) = padding;

        let out_height = (in_height + 2 * pad_height - kernel_height) / stride_height + 1;
        let out_width = (in_width + 2 * pad_width - kernel_width) / stride_width + 1;

        let shape =
            IxDyn(&[batch_size, out_height, out_width, in_channels, kernel_height, kernel_width]);

        let mut cols = ArrayD::zeros(shape);

        for b in 0..batch_size {
            for c in 0..in_channels {
                for h in 0..out_height {
                    for w in 0..out_width {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                // Compute indices with padding
                                let ih = (h * stride_height + kh) as isize - pad_height as isize;
                                let iw = (w * stride_width + kw) as isize - pad_width as isize;

                                // Check if the indices are within bounds
                                if ih >= 0
                                    && ih < in_height as isize
                                    && iw >= 0
                                    && iw < in_width as isize
                                {
                                    // Safe to cast to usize and access the input
                                    cols[[b, h, w, c, kh, kw]] =
                                        input.data[[b, c, ih as usize, iw as usize]];
                                } else {
                                    // Handle out-of-bounds indices (e.g., zero-padding)
                                    cols[[b, h, w, c, kh, kw]] = 0.0;
                                }
                            }
                        }
                    }
                }
            }
        }

        Tensor { data: cols.into_dyn(), device: input.device.clone() }
    }
}

impl Layer for Conv2D {
    fn build(&mut self, input_shape: Shape<IxDyn>) -> Result<(), LayerError> {
        let input_dims = input_shape.raw_dim();
        let in_channels = input_dims[1];

        let weight_shape = Shape::from(IxDyn(&[
            self.filters,
            in_channels,
            self.kernel_size.0,
            self.kernel_size.1,
        ]));

        self.weights = Some(Tensor::random_normal(weight_shape, 0.0, 0.01));
        self.bias = Some(Tensor::zeros(Shape::from(IxDyn(&[self.filters])), self.device.clone()));

        Ok(())
    }

    fn forward(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        let weights = self.weights.as_ref().ok_or(LayerError::UninitializedWeights)?;
        let bias = self.bias.as_ref().ok_or(LayerError::UninitializedBias)?;

        self.input = Some(input.clone());

        let cols = Self::im2col(input, self.kernel_size, self.strides, self.padding);
        let reshaped_cols = cols.reshape(IxDyn(&[
            cols.shape().raw_dim()[0] * cols.shape().raw_dim()[1] * cols.shape().raw_dim()[2],
            cols.shape().raw_dim()[3] * cols.shape().raw_dim()[4] * cols.shape().raw_dim()[5],
        ]));

        let reshaped_weights = weights.reshape(IxDyn(&[
            self.filters,
            weights.shape().raw_dim()[1]
                * weights.shape().raw_dim()[2]
                * weights.shape().raw_dim()[3],
        ]));

        let output = reshaped_cols.dot(&reshaped_weights.transpose());

        // reshape output to match the expected output dimensions
        let output_shape = self.output_shape()?;
        let output = output.reshape(output_shape.raw_dim().clone());

        // reshape bias to match output dimensions
        let reshaped_bias = bias.reshape(IxDyn(&[1, self.filters, 1, 1]));
        let output = output.add(&reshaped_bias);

        Ok(output)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let weights = self.weights.as_ref().ok_or(LayerError::UninitializedWeights)?;
        let input = self.input.as_ref().ok_or(LayerError::UninitializedInput)?;

        let input_shape = input.data.shape();
        if input_shape.len() != 4 {
            return Err(LayerError::InvalidInputShape);
        }
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let in_height = input_shape[2];
        let in_width = input_shape[3];

        // Compute output dimensions (as in the forward pass).
        let output_height =
            (in_height + 2 * self.padding.0 - self.kernel_size.0) / self.strides.0 + 1;
        let output_width =
            (in_width + 2 * self.padding.1 - self.kernel_size.1) / self.strides.1 + 1;

        // Initialize gradient tensors with the appropriate shapes.
        let mut weights_grad = Tensor::zeros(
            Shape::from(IxDyn(&[
                self.filters,
                in_channels,
                self.kernel_size.0,
                self.kernel_size.1,
            ])),
            self.device.clone(),
        );
        let mut bias_grad = Tensor::zeros(Shape::from(IxDyn(&[self.filters])), self.device.clone());
        let mut input_grad = Tensor::zeros(
            Shape::from(IxDyn(&[batch_size, in_channels, in_height, in_width])),
            self.device.clone(),
        );

        // Loop over each example in the batch.
        for b in 0..batch_size {
            // Loop over each filter.
            for f in 0..self.filters {
                // Loop over each spatial location in the output.
                for i in 0..output_height {
                    for j in 0..output_width {
                        // grad[b, f, i, j] is the scalar gradient from the next layer.
                        let grad_val = grad.data[[b, f, i, j]];
                        // Update bias gradient.
                        bias_grad.data[[f]] += grad_val;

                        // For each kernel element, compute its corresponding input index.
                        for kh in 0..self.kernel_size.0 {
                            for kw in 0..self.kernel_size.1 {
                                // Compute the "raw" input indices (without padding) for this output location.
                                let ih = i * self.strides.0 + kh;
                                let iw = j * self.strides.1 + kw;
                                // Adjust for padding.
                                let ih_padded = ih as isize - self.padding.0 as isize;
                                let iw_padded = iw as isize - self.padding.1 as isize;

                                // Only update if the indices fall within the actual input dimensions.
                                if ih_padded >= 0
                                    && ih_padded < in_height as isize
                                    && iw_padded >= 0
                                    && iw_padded < in_width as isize
                                {
                                    let ih_idx = ih_padded as usize;
                                    let iw_idx = iw_padded as usize;

                                    // For each channel, update the corresponding gradients.
                                    for c in 0..in_channels {
                                        // Weight gradient for filter f: add input[b, c, ih_idx, iw_idx] * grad_val.
                                        weights_grad.data[[f, c, kh, kw]] +=
                                            input.data[[b, c, ih_idx, iw_idx]] * grad_val;
                                        // Input gradient: add contribution from filter f weighted by the filter weight.
                                        input_grad.data[[b, c, ih_idx, iw_idx]] +=
                                            weights.data[[f, c, kh, kw]] * grad_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if self.trainable {
            self.weights_grad = Some(weights_grad);
            self.bias_grad = Some(bias_grad);
        }

        Ok(input_grad)
    }

    fn output_shape(&self) -> Result<Shape<IxDyn>, LayerError> {
        let binding = self.input.as_ref().ok_or(LayerError::UninitializedInput)?.shape();
        let input_shape = binding.raw_dim();

        let (in_height, in_width) = (input_shape[2], input_shape[3]);
        let (out_height, out_width) = (
            (in_height + 2 * self.padding.0 - self.kernel_size.0) / self.strides.0 + 1,
            (in_width + 2 * self.padding.1 - self.kernel_size.1) / self.strides.1 + 1,
        );

        println!("Input shape: {:?}", input_shape);

        // Return the output shape as `[batch_size, filters, out_height, out_width]`
        Ok(Shape::from(IxDyn(&[input_shape[0], self.filters, out_height, out_width])))
    }

    fn param_count(&self) -> Result<(usize, usize), LayerError> {
        let weights_count = self.weights.as_ref().map_or(0, |w| w.data.len());
        let bias_count = self.bias.as_ref().map_or(0, |b| b.data.len());
        Ok((weights_count, bias_count))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn set_device(&mut self, device: &Device) {
        self.device = device.clone();

        if let Some(ref mut weights) = self.weights {
            weights.device = device.clone();
        }
        if let Some(ref mut bias) = self.bias {
            bias.device = device.clone();
        }
        if let Some(ref mut input) = self.input {
            input.device = device.clone();
        }
    }

    fn update_weights(&mut self, optimizer: &mut Box<dyn Optimizer>) -> Result<(), LayerError> {
        if !self.trainable {
            return Ok(());
        }

        if let Some(ref weights_grad) = self.weights_grad {
            optimizer
                .step(self.weights.as_mut().unwrap(), weights_grad)
                .map_err(LayerError::OptimizerError)?;
        }

        if let Some(ref bias_grad) = self.bias_grad {
            optimizer
                .step(self.bias.as_mut().unwrap(), bias_grad)
                .map_err(LayerError::OptimizerError)?;
        }

        self.weights_grad = None;
        self.bias_grad = None;

        Ok(())
    }

    fn get_weights(&self) -> serde_json::Value {
        serde_json::json!({
            "weights": self.weights.as_ref().map(|w| w.to_vec()),
            "bias": self.bias.as_ref().map(|b| b.to_vec())
        })
    }

    fn get_config(&self) -> serde_json::Value {
        serde_json::json!({
            "filters": self.filters,
            "kernel_size": [self.kernel_size.0, self.kernel_size.1],
            "strides": [self.strides.0, self.strides.1],
            "padding": [self.padding.0, self.padding.1],
            "trainable": self.trainable,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::deep_learning::activations::ReluActivation;

    use super::*;

    #[test]
    fn test_dense_layer() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[1, 3])));
        let mut dense_layer = Dense::new(2, Some(ReluActivation::new()), true);
        dense_layer.build(Shape::from(IxDyn(&[1, 3]))).expect("Failed to build layer");

        let output = dense_layer.forward(&input).unwrap();

        assert_eq!(output.data.shape(), &[1, 2]);
        assert_eq!(output.data.len(), 2);
    }

    #[test]
    fn test_dense_layer_forward_pass() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[1, 3])));
        let mut dense_layer = Dense::new(2, Some(ReluActivation::new()), true);
        dense_layer.build(Shape::from(IxDyn(&[1, 3]))).expect("Failed to build layer");

        let output = dense_layer.forward(&input).unwrap();

        assert_eq!(output.data.shape(), &[1, 2]);
        assert_eq!(output.data.len(), 2);
    }

    #[test]
    fn test_dense_layer_backward_pass() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[1, 3])));
        let mut dense_layer = Dense::new(2, Some(ReluActivation::new()), true);
        dense_layer.input = Some(input.clone());
        dense_layer.build(Shape::from(IxDyn(&[1, 3]))).expect("Failed to build layer");

        let grad = Tensor::new(vec![1.0, 2.0], Shape::from(IxDyn(&[1, 2])));
        let output = dense_layer.backward(&grad).unwrap();

        assert_eq!(output.data.shape(), &[1, 3]);
        assert_eq!(output.data.len(), 3);
    }

    #[test]
    fn test_dense_layer_initialization() {
        let dense_layer = Dense::new(5, None::<ReluActivation>, true);
        assert_eq!(dense_layer.units, 5);
        assert!(dense_layer.weights.is_none());
        assert!(dense_layer.bias.is_none());
    }

    #[test]
    fn test_dense_layer_with_no_activation() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[1, 3])));
        let mut dense_layer = Dense::new(4, None::<ReluActivation>, true);
        dense_layer.build(Shape::from(IxDyn(&[1, 3]))).expect("Failed to build layer");

        let output = dense_layer.forward(&input).unwrap();

        assert_eq!(output.data.len(), 4);
        // Verify that the output is computed without activation.
        // (Exact values depend on random weight initialization.)
    }

    #[test]
    fn test_dense_layer_output_shape() {
        let dense_layer = Dense::new(10, Some(ReluActivation::new()), true);
        assert_eq!(
            dense_layer.output_shape().unwrap().raw_dim().as_array_view().to_vec(),
            vec![10]
        );
    }

    #[test]
    fn test_dense_layer_param_count() {
        let mut dense_layer = Dense::new(6, None::<ReluActivation>, true);
        dense_layer.build(Shape::from(IxDyn(&[1, 4]))).expect("Failed to build layer");

        let (weights_count, bias_count) = dense_layer.param_count().unwrap();
        assert_eq!(weights_count, 4 * 6); // 4 input units, 6 output units
        assert_eq!(bias_count, 6);
    }

    #[test]
    fn test_dense_layer_backward_with_no_trainable() {
        let mut dense_layer = Dense::new(4, None::<ReluActivation>, false);
        dense_layer.build(Shape::from(IxDyn(&[1, 3]))).expect("Failed to build layer");

        let input = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[1, 3])));
        dense_layer.input = Some(input);

        let grad = Tensor::new(vec![0.5, -0.5, 1.0, -1.0], Shape::from(IxDyn(&[1, 4])));
        let output_grad = dense_layer.backward(&grad).unwrap();

        // Ensure gradients are not stored when `trainable` is false.
        assert!(dense_layer.weights_grad.is_none());
        assert!(dense_layer.bias_grad.is_none());

        // Ensure output gradient is calculated.
        assert_eq!(output_grad.data.len(), 3);
    }

    #[test]
    fn test_dense_layer_with_zero_units() {
        let mut dense_layer = Dense::new(0, None::<ReluActivation>, true);
        dense_layer.build(Shape::from(IxDyn(&[1, 3]))).expect("Failed to build layer");

        // Ensure the layer initializes with zero units without crashing.
        assert_eq!(dense_layer.output_shape().unwrap().raw_dim().as_array_view().to_vec(), vec![0]);
        assert!(dense_layer.weights.is_some());
        assert!(dense_layer.bias.is_some());
    }

    #[test]
    fn test_dense_layer_with_large_input() {
        let input = Tensor::random(Shape::from(IxDyn(&[1000, 512])));
        let mut dense_layer = Dense::new(256, Some(ReluActivation::new()), true);
        dense_layer.build(Shape::from(IxDyn(&[1000, 512]))).expect("Failed to build layer");

        let output = dense_layer.forward(&input).unwrap();

        assert_eq!(output.data.shape(), &[1000, 256]);
        assert_eq!(output.data.len(), 1000 * 256);
    }

    #[test]
    fn test_flatten_new() {
        let input_shape = Shape::from(IxDyn(&[28, 28]));
        let flatten_layer = Flatten::new(input_shape.clone());

        assert_eq!(flatten_layer.name(), "Flatten");
        assert_eq!(flatten_layer.input_shape.raw_dim(), input_shape.raw_dim());
    }

    #[test]
    fn test_flatten_output_shape() {
        let input_shape = Shape::from(IxDyn(&[3, 4]));
        let flatten_layer = Flatten::new(input_shape.clone());

        let output_shape = flatten_layer.output_shape().unwrap();
        assert_eq!(output_shape.raw_dim().as_array_view().to_vec(), vec![12]);
    }

    #[test]
    fn test_flatten_param_count() {
        let flatten_layer = Flatten::new(Shape::from(IxDyn(&[10, 10])));
        let (trainable, non_trainable) = flatten_layer.param_count().unwrap();

        assert_eq!(trainable, 0);
        assert_eq!(non_trainable, 0);
    }

    #[test]
    fn test_conv2d_forward_pass() {
        let input = Tensor::random(Shape::from(IxDyn(&[1, 3, 32, 32])));
        let mut conv_layer = Conv2D::new(16, (3, 3), (1, 1), (1, 1), true);
        conv_layer.build(Shape::from(IxDyn(&[1, 3, 32, 32]))).expect("Failed to build layer");

        let output = conv_layer.forward(&input).unwrap();

        assert_eq!(output.shape().raw_dim().as_array_view().to_vec(), vec![1, 16, 32, 32]);
    }

    #[test]
    fn test_conv2d_backward_pass() {
        let input = Tensor::random(Shape::from(IxDyn(&[1, 3, 32, 32])));
        let mut conv_layer = Conv2D::new(16, (3, 3), (1, 1), (1, 1), true);
        conv_layer.build(Shape::from(IxDyn(&[1, 3, 32, 32]))).expect("Failed to build layer");

        let output = conv_layer.forward(&input).unwrap();
        let grad = Tensor::random(output.shape());
        let input_grad = conv_layer.backward(&grad).unwrap();

        // Verify the input gradient shape matches the input shape
        assert_eq!(input_grad.shape().raw_dim().as_array_view().to_vec(), vec![1, 3, 32, 32]);
    }

    #[test]
    fn test_conv2d_output_shape() {
        let mut conv_layer = Conv2D::new(16, (3, 3), (1, 1), (1, 1), true);
        let input_shape = Shape::from(IxDyn(&[1, 3, 32, 32]));
        conv_layer.build(input_shape.clone()).expect("Failed to build layer");

        // initialize the layer with the input shape
        conv_layer.input = Some(Tensor::random(input_shape));

        let output_shape = conv_layer.output_shape().unwrap();
        assert_eq!(output_shape.raw_dim().as_array_view().to_vec(), vec![1, 16, 32, 32]);
    }

    #[test]
    fn test_conv2d_param_count() {
        let mut conv_layer = Conv2D::new(16, (3, 3), (1, 1), (1, 1), true);
        conv_layer.build(Shape::from(IxDyn(&[1, 3, 32, 32]))).expect("Failed to build layer");

        let (weights_count, bias_count) = conv_layer.param_count().unwrap();
        assert_eq!(weights_count, 16 * 3 * 3 * 3);
        assert_eq!(bias_count, 16);
    }
}

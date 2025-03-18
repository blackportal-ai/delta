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

use ndarray::{Dimension, IxDyn, Shape};

use crate::devices::Device;

use super::{
    activations::Activation, errors::LayerError, layers::Layer, optimizers::Optimizer,
    tensor_ops::Tensor,
};

#[derive(Debug)]
pub struct Conv2D {
    filters: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    device: Device,
    activation: Option<Box<dyn Activation>>,
    weights: Option<Tensor>,
    bias: Option<Tensor>,
    input: Option<Tensor>,
    input_shape: Shape<IxDyn>,
    weights_gradient: Option<Tensor>,
    biases_gradient: Option<Tensor>,
    trainable: bool,
}

impl Conv2D {
    pub fn new<A: Activation + 'static>(
        filters: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        input_shape: Shape<IxDyn>,
        activation: Option<A>,
        trainable: bool,
    ) -> Self {
        Conv2D {
            filters,
            kernel_size,
            stride,
            padding,
            input_shape,
            activation: activation.map(|a| Box::new(a) as Box<dyn Activation>),
            device: Device::default(),
            weights: None,
            bias: None,
            input: None,
            weights_gradient: None,
            biases_gradient: None,
            trainable,
        }
    }
}

impl Layer for Conv2D {
    /// Builds the layer with the given input shape.
    ///
    /// # Arguments
    ///
    /// * `input_shape` - The shape of the input tensor.
    fn build(&mut self, input_shape: Shape<IxDyn>) -> Result<(), LayerError> {
        self.weights = Some(Tensor::random(Shape::from(IxDyn(&[
            self.filters,
            input_shape.raw_dim()[input_shape.raw_dim().to_owned().as_array_view().len() - 1],
            self.kernel_size,
            self.kernel_size,
        ]))));

        self.bias =
            Some(Tensor::zeros(Shape::from(IxDyn(&[1, 1, 1, self.filters])), self.device.clone()));

        self.input_shape = input_shape;

        Ok(())
    }

    /// Performs the forward pass of the layer.
    ///
    /// # Arguments
    ///
    /// * `input` - The input tensor.
    ///
    /// # Returns
    ///
    /// The output tensor after applying the layer.
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.input = Some(input.clone());

        let input_shape = input.shape();
        let batch_size = input_shape.raw_dim()[0];
        let input_height = input_shape.raw_dim()[1];
        let input_width = input_shape.raw_dim()[2];
        let input_channels = input_shape.raw_dim()[3];

        let output_width = (input_width + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let output_height = (input_height + 2 * self.padding - self.kernel_size) / self.stride + 1;

        let mut output = Tensor::zeros(
            Shape::from(IxDyn(&[batch_size, output_height, output_width, self.filters])),
            self.device.clone(),
        );

        if self.padding > 0 {
            self.input =
                Some(input.clone().pad(self.padding).expect("Failed to pad input tensor."));
        }

        for b in 0..batch_size {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    for f in 0..self.filters {
                        let mut sum = self.bias.as_ref().map_or(0.0, |b| b[(0, 0, 0, f)]);

                        for i in 0..self.kernel_size {
                            for j in 0..self.kernel_size {
                                for d in 0..input_channels {
                                    let padded_input_height = oh * self.stride + i;
                                    let padded_input_width = ow * self.stride + j;

                                    sum += input[(b, padded_input_height, padded_input_width, d)]
                                        * self.weights.as_ref().unwrap()[(f, d, i, j)];
                                }
                            }
                        }

                        output[(b, oh, ow, f)] = sum;
                    }
                }
            }
        }

        let output = if let Some(ref activation) = self.activation {
            activation.activate(&output)
        } else {
            output
        };

        Ok(output)
    }

    /// Performs the backward pass of the layer.
    ///
    /// # Arguments
    ///
    /// * `grad` - The gradient tensor from the next layer.
    ///
    /// # Returns
    ///
    /// The gradient tensor to be passed to the previous layer.
    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let padded_input = self.input.as_ref().expect("Input must be initialized.");
        let padded_input = padded_input
            .clone()
            .pad(self.padding)
            .expect("Failed to pad input tensor for backward pass.");

        let input_shape = padded_input.shape();
        let batch_size = input_shape.raw_dim()[0];
        let input_height = input_shape.raw_dim()[1];
        let input_width = input_shape.raw_dim()[2];
        let input_channels = input_shape.raw_dim()[3];

        let output_height = (input_height - self.kernel_size) / self.stride + 1;
        let output_width = (input_width - self.kernel_size) / self.stride + 1;

        let mut d_input = Tensor::zeros(
            Shape::from(IxDyn(&[batch_size, input_height, input_width, input_channels])),
            self.device.clone(),
        );

        let mut d_bias =
            Tensor::zeros(Shape::from(IxDyn(&[1, 1, 1, self.filters])), self.device.clone());

        let mut d_kernel = Tensor::zeros(
            Shape::from(IxDyn(&[self.filters, input_channels, self.kernel_size, self.kernel_size])),
            self.device.clone(),
        );

        for b in 0..batch_size {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    for f in 0..self.filters {
                        let output_gradient = grad[(b, oh, ow, f)];
                        d_bias[(0, 0, 0, f)] += output_gradient;

                        for i in 0..self.kernel_size {
                            for j in 0..self.kernel_size {
                                for d in 0..input_channels {
                                    let padded_input_height = oh * self.stride + i;
                                    let padded_input_width = ow * self.stride + j;

                                    d_kernel[(f, d, i, j)] += padded_input
                                        [(b, padded_input_height, padded_input_width, d)]
                                        * output_gradient;
                                    d_input[(b, padded_input_height, padded_input_width, d)] +=
                                        self.weights.as_ref().unwrap()[(f, d, i, j)]
                                            * output_gradient;
                                }
                            }
                        }
                    }
                }
            }
        }

        if self.trainable {
            self.weights_gradient = Some(d_kernel);
            self.biases_gradient = Some(d_bias);
        }

        Ok(d_input)
    }

    /// Returns the output shape of the layer.
    ///
    /// # Returns
    ///
    /// A `Shape` representing the output shape of the layer.
    fn output_shape(&self) -> Result<Shape<IxDyn>, LayerError> {
        println!("Shape of Conv2D Layer: {:?}", self.input_shape);
        let input_shape = self.input_shape.raw_dim();
        println!("Shape of Conv2D Layer DIM: {:?}", input_shape.as_array_view().to_vec());
        let output_height =
            (input_shape[1] + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let output_width = (input_shape[2] + 2 * self.padding - self.kernel_size) / self.stride + 1;

        Ok(Shape::from(IxDyn(&[
            input_shape[0], // batch size
            output_height,  // height
            output_width,   // width
            self.filters,   // channels
        ])))
    }

    /// Returns the number of parameters in the layer.
    ///
    /// # Returns
    ///
    /// A tuple `(usize, usize)` representing the number of trainable and non-trainable parameters in the layer.
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
        "Conv2D"
    }

    /// Sets the device for the layer.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to set for the layer.
    fn set_device(&mut self, device: &Device) {
        self.device = device.clone();
    }

    /// Updates the weights of the layer.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - The optimizer to use.
    fn update_weights(&mut self, optimizer: &mut Box<dyn Optimizer>) -> Result<(), LayerError> {
        if !self.trainable {
            return Ok(());
        }

        // Update weights
        if let Some(ref weights_gradient) = self.weights_gradient {
            optimizer
                .step(self.weights.as_mut().unwrap(), weights_gradient)
                .map_err(LayerError::OptimizerError)?;
        }

        if let Some(ref biases_gradient) = self.biases_gradient {
            optimizer
                .step(self.bias.as_mut().unwrap(), biases_gradient)
                .map_err(LayerError::OptimizerError)?;
        }

        // Clear gradients after update
        self.weights_gradient = None;
        self.biases_gradient = None;

        Ok(())
    }

    /// Returns the weights of the layer as a serializable format.
    ///
    /// # Returns
    ///
    /// A `serde_json::Value` containing the layer's weights.
    fn get_weights(&self) -> serde_json::Value {
        // convert Tensors to JSON
        let weights = self.weights.as_ref().map(|w| w.to_vec()).unwrap_or_else(|| vec![]);
        let bias = self.bias.as_ref().map(|b| b.to_vec()).unwrap_or_else(|| vec![]);

        serde_json::json!({
            "weights": weights,
            "bias": bias,
        })
    }

    fn get_config(&self) -> serde_json::Value {
        serde_json::json!({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "trainable": self.trainable,
            "weights": self.weights.as_ref().map(|w| w.to_vec()).unwrap_or_else(|| vec![]),
            "bias": self.bias.as_ref().map(|b| b.to_vec()).unwrap_or_else(|| vec![]),
            "weights_gradient": self.weights_gradient.as_ref().map(|w| w.to_vec()).unwrap_or_else(|| vec![]),
            "biases_gradient": self.biases_gradient.as_ref().map(|b| b.to_vec()).unwrap_or_else(|| vec![]),
        })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Dimension;

    use crate::deep_learning::activations::ReluActivation;

    use super::*;
    #[test]
    fn test_conv2d_forward_pass() {
        let input_shape = Shape::from(IxDyn(&[1, 3, 3, 1]));
        let input =
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], input_shape.clone());
        let mut conv2d_layer = Conv2D::new(
            1,                           // filters
            2,                           // kernel_size
            1,                           // stride
            0,                           // padding
            input_shape.clone(),         // input_shape
            Some(ReluActivation::new()), // activation
            true,                        // trainable
        );
        let _ = conv2d_layer.build(input_shape).unwrap();

        let output = conv2d_layer.forward(&input).unwrap();

        assert_eq!(output.shape().raw_dim().as_array_view().to_vec(), vec![1, 2, 2, 1]);
        assert_eq!(output.data.len(), 4);
    }

    #[test]
    fn test_conv2d_backward_pass() {
        let input_shape = Shape::from(IxDyn(&[1, 3, 3, 1]));
        let input =
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], input_shape.clone());
        let mut conv2d_layer = Conv2D::new(
            1,                           // filters
            2,                           // kernel_size
            1,                           // stride
            0,                           // padding
            input_shape.clone(),         // input_shape
            Some(ReluActivation::new()), // activation
            true,                        // trainable
        );
        let _ = conv2d_layer.build(input_shape).unwrap();

        let output = conv2d_layer.forward(&input).unwrap();
        let grad = Tensor::new(vec![1.0; output.data.len()], output.shape().clone());

        let d_input = conv2d_layer.backward(&grad).unwrap();

        assert_eq!(d_input.shape().raw_dim().as_array_view().to_vec(), vec![1, 3, 3, 1]);
        assert_eq!(d_input.data.len(), 9);
    }

    #[test]
    fn test_conv2d_get_weights() {
        let input_shape = Shape::from(IxDyn(&[1, 3, 3, 1]));
        let input =
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], input_shape.clone());
        let mut conv2d_layer = Conv2D::new(
            1,                           // filters
            2,                           // kernel_size
            1,                           // stride
            0,                           // padding
            input_shape.clone(),         // input_shape
            Some(ReluActivation::new()), // activation
            true,                        // trainable
        );
        let _ = conv2d_layer.build(input_shape).unwrap();

        let output = conv2d_layer.forward(&input).unwrap();
        let grad = Tensor::new(vec![1.0; output.data.len()], output.shape().clone());

        let _ = conv2d_layer.backward(&grad).unwrap();

        let weights = conv2d_layer.get_weights();

        assert!(weights.is_object());
    }

    #[test]
    fn test_conv2d_get_config() {
        let input_shape = Shape::from(IxDyn(&[1, 3, 3, 1]));
        let input =
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], input_shape.clone());
        let mut conv2d_layer = Conv2D::new(
            1,                           // filters
            2,                           // kernel_size
            1,                           // stride
            0,                           // padding
            input_shape.clone(),         // input_shape
            Some(ReluActivation::new()), // activation
            true,                        // trainable
        );
        let _ = conv2d_layer.build(input_shape).unwrap();

        let output = conv2d_layer.forward(&input).unwrap();
        let grad = Tensor::new(vec![1.0; output.data.len()], output.shape().clone());

        let _ = conv2d_layer.backward(&grad).unwrap();

        let config = conv2d_layer.get_config();

        println!("Conv2D Layer Config: {:?}", config);

        assert!(config.is_object());
    }
}

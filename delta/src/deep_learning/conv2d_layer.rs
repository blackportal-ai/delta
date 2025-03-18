use ndarray::{IxDyn, Shape};

use crate::devices::Device;

use super::{errors::LayerError, layers::Layer, tensor_ops::Tensor};

#[derive(Debug)]
pub struct Conv2D {
    filters: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    device: Device,
    weights: Option<Tensor>,
    bias: Option<Tensor>,
    input: Option<Tensor>,
    input_shape: Shape<IxDyn>,
    weights_gradient: Option<Tensor>,
    biases_gradient: Option<Tensor>,
    trainable: bool,
}

impl Conv2D {
    fn new(
        filters: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        input_shape: Shape<IxDyn>,
        device: Device,
        trainable: bool,
    ) -> Self {
        Conv2D {
            filters,
            kernel_size,
            stride,
            padding,
            input_shape,
            device: device.clone(),
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
    fn build(&mut self, input_shape: Shape<IxDyn>) -> Result<(), super::errors::LayerError> {
        self.weights = Some(Tensor::random(Shape::from(IxDyn(&[
            self.filters,
            input_shape.raw_dim()[3],
            self.kernel_size,
            self.kernel_size,
        ]))));

        self.bias =
            Some(Tensor::zeros(Shape::from(IxDyn(&[1, 1, 1, self.filters])), self.device.clone()));

        self.input_shape = input_shape;

        Ok(())
    }

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

        Ok(output)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, super::errors::LayerError> {
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

    fn output_shape(&self) -> Result<Shape<IxDyn>, super::errors::LayerError> {
        todo!()
    }

    fn param_count(&self) -> Result<(usize, usize), super::errors::LayerError> {
        let weights_count = self.weights.as_ref().map_or(0, |w| w.data.len());
        let bias_count = self.bias.as_ref().map_or(0, |b| b.data.len());
        Ok((weights_count, bias_count))
    }

    fn name(&self) -> &str {
        "Conv2D"
    }

    fn set_device(&mut self, device: &Device) {
        self.device = device.clone();
    }

    fn update_weights(
        &mut self,
        optimizer: &mut Box<dyn super::optimizers::Optimizer>,
    ) -> Result<(), super::errors::LayerError> {
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
}

#[cfg(test)]
mod tests {
    use ndarray::Dimension;

    use super::*;
    #[test]
    fn test_conv2d_forward_pass() {
        let input_shape = Shape::from(IxDyn(&[1, 3, 3, 1]));
        let input =
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], input_shape.clone());
        let mut conv2d_layer = Conv2D::new(
            1,                   // filters
            2,                   // kernel_size
            1,                   // stride
            0,                   // padding
            input_shape.clone(), // input_shape
            Device::Cpu,         // device
            true,                // trainable
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
            1,                   // filters
            2,                   // kernel_size
            1,                   // stride
            0,                   // padding
            input_shape.clone(), // input_shape
            Device::Cpu,         // device
            true,                // trainable
        );
        let _ = conv2d_layer.build(input_shape).unwrap();

        let output = conv2d_layer.forward(&input).unwrap();
        let grad = Tensor::new(vec![1.0; output.data.len()], output.shape().clone());

        let d_input = conv2d_layer.backward(&grad).unwrap();

        assert_eq!(d_input.shape().raw_dim().as_array_view().to_vec(), vec![1, 3, 3, 1]);
        assert_eq!(d_input.data.len(), 9);
    }
}

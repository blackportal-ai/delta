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

use super::{errors::OptimizerError, tensor_ops::Tensor};

use std::fmt::Debug;

/// A trait representing an optimizer for training neural networks.
pub trait Optimizer: Debug {
    /// Performs an optimization step using the given weights and gradients.
    ///
    /// # Arguments
    ///
    /// * `weights` - A mutable reference to the weights tensor.
    /// * `gradients` - A reference to the gradients tensor.
    fn step(&mut self, weights: &mut Tensor, gradients: &Tensor) -> Result<(), OptimizerError>;

    /// Sets the device for the optimizer.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to set for the optimizer.
    fn set_device(&mut self, device: &Device);
}

/// A struct representing the configuration for an optimizer.
#[derive(Debug)]
pub struct OptimizerConfig {
    /// The learning rate for the optimizer.
    pub learning_rate: f32,
}

/// The AdaDelta optimizer struct.
#[derive(Debug)]
pub struct AdaDelta {
    rho: f32,
    epsilon: f32,
    accumulated_gradients: Option<Tensor>,
    accumulated_updates: Option<Tensor>,
    device: Device,
}

impl AdaDelta {
    /// Creates a new AdaDelta optimizer with the given hyperparameters.
    ///
    /// # Arguments
    ///
    /// * `rho` - Decay rate for the moving average of gradients.
    /// * `epsilon` - Small value to avoid division by zero.
    pub fn new(rho: f32, epsilon: f32) -> Self {
        Self {
            rho,
            epsilon,
            accumulated_gradients: None,
            accumulated_updates: None,
            device: Device::default(),
        }
    }
}

impl Optimizer for AdaDelta {
    /// Performs an optimization step using the given gradients.
    ///
    /// # Arguments
    ///
    /// * `weights` - A mutable reference to the weights tensor.
    /// * `gradients` - A reference to the gradients tensor.
    fn step(&mut self, weights: &mut Tensor, gradients: &Tensor) -> Result<(), OptimizerError> {
        if gradients.shape().size() != weights.shape().size() {
            return Err(OptimizerError::IncompatibleGradientWeightShape(
                gradients.shape().raw_dim().as_array_view().to_vec(),
                weights.shape().raw_dim().as_array_view().to_vec(),
            ));
        }

        // Initialize accumulated gradients and updates if not already done
        if self.accumulated_gradients.is_none()
            || self.accumulated_gradients.as_ref().unwrap().shape().raw_dim()
                != weights.shape().raw_dim()
        {
            self.accumulated_gradients =
                Some(Tensor::zeros(weights.shape().clone(), self.device.clone()));
            self.accumulated_gradients = Some(
                self.accumulated_gradients
                    .as_mut()
                    .unwrap()
                    .to_device(self.device.clone())
                    .unwrap(),
            );
        }
        if self.accumulated_updates.is_none()
            || self.accumulated_updates.as_ref().unwrap().shape().raw_dim()
                != weights.shape().raw_dim()
        {
            self.accumulated_updates =
                Some(Tensor::zeros(weights.shape().clone(), self.device.clone()));
            self.accumulated_updates = Some(
                self.accumulated_updates.as_mut().unwrap().to_device(self.device.clone()).unwrap(),
            );
        }

        let accumulated_gradients = self.accumulated_gradients.as_mut().unwrap();
        let accumulated_updates = self.accumulated_updates.as_mut().unwrap();

        // Update accumulated gradients
        *accumulated_gradients = accumulated_gradients
            .mul_scalar(self.rho)
            .add(&gradients.pow(2.0).mul_scalar(1.0 - self.rho));

        // Compute the update value
        let rms_updates = accumulated_updates.sqrt().add_scalar(self.epsilon);
        let rms_gradients = accumulated_gradients.sqrt().add_scalar(self.epsilon);
        let update = Tensor {
            data: gradients.div(&rms_gradients).data * rms_updates.data,
            device: self.device.clone(),
        };

        // Update accumulated updates
        *accumulated_updates = accumulated_updates
            .mul_scalar(self.rho)
            .add(&update.pow(2.0).mul_scalar(1.0 - self.rho));

        // Apply the update to weights
        *weights -= update;

        Ok(())
    }

    /// Sets the device for the optimizer.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to set for the optimizer.
    fn set_device(&mut self, device: &Device) {
        self.device = device.clone();
    }
}

/// The AdaGrad optimizer struct.
#[derive(Debug)]
pub struct AdaGrad {
    learning_rate: f32,
    epsilon: f32,
    g_sum: Option<Tensor>,
    timestep: usize,
    device: Device,
}

impl AdaGrad {
    /// Creates a new AdaGrad optimizer with the given learning rate and epsilon.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for the optimizer.
    /// * `epsilon` - A small value to prevent division by zero.
    ///
    /// # Returns
    ///
    /// A new instance of the AdaGrad optimizer.
    pub fn new(learning_rate: f32, epsilon: f32) -> Self {
        Self { learning_rate, epsilon, g_sum: None, timestep: 0, device: Device::default() }
    }

    /// Resets the accumulated gradient sum (g_sum).
    pub fn reset(&mut self) {
        self.g_sum = None;
    }
}

impl Optimizer for AdaGrad {
    /// Performs an optimization step using the given gradients.
    ///
    /// # Arguments
    ///
    /// * `weights` - A mutable reference to the weights tensor.
    /// * `gradients` - A reference to the gradients tensor.
    fn step(&mut self, weights: &mut Tensor, gradients: &Tensor) -> Result<(), OptimizerError> {
        if self.learning_rate <= 0.0 {
            return Err(OptimizerError::InvalidLearningRate(
                "Learning rate must be greater than 0.".to_string(),
            ));
        }

        self.timestep += 1;

        // Initialize gradient sum if not already done
        if self.g_sum.is_none()
            || self.g_sum.as_ref().unwrap().shape().raw_dim() != weights.shape().raw_dim()
        {
            self.g_sum = Some(Tensor::zeros(weights.shape().clone(), self.device.clone()));
        }

        let g_sum = self.g_sum.as_mut().unwrap();

        // Ensure gradients match the weights' shape
        let processed_gradients = if gradients.shape().raw_dim().as_array_view().to_vec()
            == weights.shape().raw_dim().as_array_view().to_vec()
        {
            gradients.clone()
        } else if gradients.shape().raw_dim().ndim() <= weights.shape().raw_dim().ndim()
            && gradients
                .shape()
                .raw_dim()
                .as_array_view()
                .iter()
                .rev()
                .zip(weights.shape().raw_dim().as_array_view().iter().rev())
                .all(|(g, w)| *g == *w || *g == 1)
        {
            gradients.broadcast(weights.shape())
        } else {
            return Err(OptimizerError::IncompatibleGradientWeightShape(
                gradients.shape().raw_dim().as_array_view().to_vec(),
                weights.shape().raw_dim().as_array_view().to_vec(),
            ));
        };

        // Update gradient sum
        *g_sum = g_sum.add(&processed_gradients.pow(2.0));

        // Compute update
        let update = processed_gradients
            .div(&(g_sum.sqrt().add_scalar(self.epsilon)))
            .mul_scalar(self.learning_rate);

        *weights -= update;

        Ok(())
    }

    /// Sets the device for the optimizer.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to set for the optimizer.
    fn set_device(&mut self, device: &Device) {
        self.device = device.clone();
    }
}

/// A wrapper struct for a debuggable scheduler function.
#[allow(dead_code)]
struct DebuggableScheduler(Box<dyn Fn(usize) -> f32>);

impl Debug for DebuggableScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DebuggableScheduler")
    }
}

/// The Adam optimizer struct.
#[derive(Debug)]
pub struct Adam {
    #[allow(dead_code)]
    learning_rate: f32,
    scheduler: Option<DebuggableScheduler>,
    m: Option<Tensor>,
    v: Option<Tensor>,
    timestep: usize,
    device: Device,
}

impl Adam {
    /// Creates a new Adam optimizer with the given learning rate.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for the optimizer.
    ///
    /// # Returns
    ///
    /// A new instance of the Adam optimizer.
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            scheduler: None,
            m: None,
            v: None,
            timestep: 0,
            device: Device::default(),
        }
    }

    /// Sets the scheduler function for the Adam optimizer.
    ///
    /// # Arguments
    ///
    /// * `scheduler` - A function that takes an epoch number and returns a learning rate.
    pub fn set_scheduler<F>(&mut self, scheduler: F)
    where
        F: Fn(usize) -> f32 + 'static,
    {
        self.scheduler = Some(DebuggableScheduler(Box::new(scheduler)));
    }

    /// Initializes the moving averages for the Adam optimizer.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the weights tensor.
    fn initialize_moving_averages(&mut self, shape: &Vec<usize>) {
        if self.m.is_none()
            || self.m.as_ref().unwrap().shape().raw_dim().as_array_view().to_vec() != *shape
        {
            self.m = Some(Tensor::zeros(Shape::from(IxDyn(shape)), self.device.clone()));
        }
        if self.v.is_none()
            || self.v.as_ref().unwrap().shape().raw_dim().as_array_view().to_vec() != *shape
        {
            self.v = Some(Tensor::zeros(Shape::from(IxDyn(shape)), self.device.clone()));
        }
    }
}

impl Optimizer for Adam {
    /// Performs an optimization step using the given gradients.
    ///
    /// # Arguments
    ///
    /// * `weights` - A mutable reference to the weights tensor.
    /// * `gradients` - A reference to the gradients tensor.
    fn step(&mut self, weights: &mut Tensor, gradients: &Tensor) -> Result<(), OptimizerError> {
        if self.learning_rate <= 0.0 {
            return Err(OptimizerError::InvalidLearningRate(
                "Learning rate must be greater than 0.".to_string(),
            ));
        }

        self.timestep += 1;

        let weights_shape = weights.shape().raw_dim().as_array_view().to_vec();

        self.initialize_moving_averages(&weights_shape);

        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        // Ensure gradients match the weights' shape
        let processed_gradients = if gradients.shape().raw_dim().as_array_view().to_vec()
            == weights.shape().raw_dim().as_array_view().to_vec()
        {
            gradients.clone()
        } else if gradients.shape().raw_dim().ndim() <= weights.shape().raw_dim().ndim()
            && gradients
                .shape()
                .raw_dim()
                .as_array_view()
                .iter()
                .rev()
                .zip(weights.shape().raw_dim().as_array_view().iter().rev())
                .all(|(g, w)| *g == *w || *g == 1)
        {
            gradients.broadcast(weights.shape())
        } else {
            return Err(OptimizerError::IncompatibleGradientWeightShape(
                gradients.shape().raw_dim().as_array_view().to_vec(),
                weights.shape().raw_dim().as_array_view().to_vec(),
            ));
        };

        // Update moving averages
        *m = m.mul_scalar(0.9).add(&processed_gradients.mul_scalar(0.1));
        *v = v.mul_scalar(0.999).add(&processed_gradients.pow(2.0).mul_scalar(0.001));

        // Bias correction
        let bias_correction_1 = 1.0 - 0.9f32.powi(self.timestep as i32);
        let bias_correction_2 = 1.0 - 0.999f32.powi(self.timestep as i32);

        let m_hat = m.div_scalar(bias_correction_1);
        let v_hat = v.div_scalar(bias_correction_2);

        // Get learning rate
        let lr = self.scheduler.as_ref().map_or(self.learning_rate, |s| s.0(self.timestep));

        // Compute scaling factor based on max gradient magnitude
        let max_gradient = processed_gradients.data.iter().map(|g| g.abs()).fold(0.0, f32::max);

        let scaling_factor = if max_gradient > 10.0 { 10.0 / max_gradient } else { 1.0 };

        // Apply scaled learning rate
        let epsilon = 1e-8;
        let scaled_lr = lr * scaling_factor;
        let update = m_hat.div(&v_hat.sqrt().add_scalar(epsilon)).mul_scalar(scaled_lr);

        *weights -= update;

        Ok(())
    }

    /// Sets the device for the optimizer.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to set for the optimizer.
    fn set_device(&mut self, device: &Device) {
        self.device = device.clone();
    }
}

/// The Gradient Descent optimizer struct.
#[derive(Debug)]
pub struct GradientDescent {
    learning_rate: f32,
    device: Device,
}

impl GradientDescent {
    /// Creates a new Gradient Descent optimizer with the given learning rate.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for the optimizer.
    ///
    /// # Returns
    ///
    /// A new instance of the Gradient Descent optimizer.
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate, device: Device::default() }
    }
}

impl Optimizer for GradientDescent {
    /// Performs an optimization step using the given gradients.
    ///
    /// # Arguments
    ///
    /// * `weights` - A mutable reference to the weights tensor.
    /// * `gradients` - A reference to the gradients tensor.
    fn step(&mut self, weights: &mut Tensor, gradients: &Tensor) -> Result<(), OptimizerError> {
        if self.learning_rate <= 0.0 {
            return Err(OptimizerError::InvalidLearningRate(
                "Learning rate must be greater than 0.".to_string(),
            ));
        }

        // Ensure gradients match the weights' shape
        if gradients.shape().raw_dim().as_array_view().to_vec()
            != weights.shape().raw_dim().as_array_view().to_vec()
        {
            return Err(OptimizerError::IncompatibleGradientWeightShape(
                gradients.shape().raw_dim().as_array_view().to_vec(),
                weights.shape().raw_dim().as_array_view().to_vec(),
            ));
        }

        // Update weights
        let update = gradients.mul_scalar(self.learning_rate);
        *weights -= update;

        Ok(())
    }

    /// Sets the device for the optimizer.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to set for the optimizer.
    fn set_device(&mut self, device: &Device) {
        self.device = device.clone();
    }
}

/// Mini-Batch Gradient Descent optimizer.
#[derive(Debug)]
pub struct MiniBatchGD {
    #[allow(dead_code)]
    learning_rate: f32,
    device: Device,
}

impl MiniBatchGD {
    /// Creates a new Mini-Batch Gradient Descent optimizer with the given learning rate.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for the optimizer.
    ///
    /// # Returns
    ///
    /// A new instance of the MiniBatchGD optimizer.
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate, device: Device::default() }
    }
}

impl Optimizer for MiniBatchGD {
    /// Performs an optimization step using the given gradients.
    ///
    /// # Arguments
    ///
    /// * `weights` - A mutable reference to the weights tensor.
    /// * `gradients` - A reference to the gradients tensor.
    fn step(&mut self, weights: &mut Tensor, gradients: &Tensor) -> Result<(), OptimizerError> {
        if self.learning_rate <= 0.0 {
            return Err(OptimizerError::InvalidLearningRate(
                "Learning rate must be greater than 0.".to_string(),
            ));
        }

        // Ensure gradients match the weights' shape
        let processed_gradients = if gradients.shape().raw_dim().as_array_view().to_vec()
            == weights.shape().raw_dim().as_array_view().to_vec()
        {
            gradients.clone()
        } else if gradients.shape().raw_dim().ndim() <= weights.shape().raw_dim().ndim()
            && gradients
                .shape()
                .raw_dim()
                .as_array_view()
                .iter()
                .rev()
                .zip(weights.shape().raw_dim().as_array_view().iter().rev())
                .all(|(g, w)| *g == *w || *g == 1)
        {
            gradients.broadcast(weights.shape())
        } else {
            return Err(OptimizerError::IncompatibleGradientWeightShape(
                gradients.shape().raw_dim().as_array_view().to_vec(),
                weights.shape().raw_dim().as_array_view().to_vec(),
            ));
        };

        // Update weights
        *weights -= processed_gradients.mul_scalar(self.learning_rate);

        Ok(())
    }

    /// Sets the device for the optimizer.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to set for the optimizer.
    fn set_device(&mut self, device: &Device) {
        self.device = device.clone();
    }
}

/// The RMSProp optimizer struct.
#[derive(Debug)]
pub struct RMSProp {
    learning_rate: f32,
    decay_rate: f32,
    epsilon: f32,
    mean_square: Option<Tensor>,
    device: Device,
}

impl RMSProp {
    /// Creates a new RMSProp optimizer with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for the optimizer.
    /// * `decay_rate` - The decay rate for the moving average of squared gradients.
    /// * `epsilon` - A small value to prevent division by zero (must be > 0).
    ///
    /// # Returns
    ///
    /// A new instance of the RMSProp optimizer.
    pub fn new(learning_rate: f32, decay_rate: f32, epsilon: f32) -> Result<Self, OptimizerError> {
        if epsilon <= 0.0 {
            return Err(OptimizerError::InvalidEpsilon(
                "Epsilon must be greater than 0.".to_string(),
            ));
        }
        if learning_rate <= 0.0 {
            return Err(OptimizerError::InvalidLearningRate(
                "Learning rate must be greater than 0.".to_string(),
            ));
        }
        Ok(Self {
            learning_rate,
            decay_rate,
            epsilon,
            mean_square: None,
            device: Device::default(),
        })
    }
}

impl Optimizer for RMSProp {
    /// Performs an optimization step using the given gradients.
    ///
    /// # Arguments
    ///
    /// * `weights` - A mutable reference to the weights tensor.
    /// * `gradients` - A reference to the gradients tensor.
    fn step(&mut self, weights: &mut Tensor, gradients: &Tensor) -> Result<(), OptimizerError> {
        if weights.shape().raw_dim() != gradients.shape().raw_dim() {
            return Err(OptimizerError::IncompatibleGradientWeightShape(
                gradients.shape().raw_dim().as_array_view().to_vec(),
                weights.shape().raw_dim().as_array_view().to_vec(),
            ));
        }

        // Initialize mean square tensor if not already done
        if self.mean_square.is_none() {
            self.mean_square = Some(Tensor::zeros(weights.shape().clone(), self.device.clone()));
        }

        let mean_square = self.mean_square.as_mut().unwrap();
        let one_minus_decay = 1.0 - self.decay_rate;

        // Update mean square
        *mean_square = mean_square
            .mul_scalar(self.decay_rate)
            .add(&gradients.pow(2.0).mul_scalar(one_minus_decay));

        // Compute update
        let update = gradients
            .div(&mean_square.sqrt().add_scalar(self.epsilon))
            .mul_scalar(self.learning_rate);

        *weights -= update;

        Ok(())
    }

    /// Sets the device for the optimizer.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to set for the optimizer.
    fn set_device(&mut self, device: &Device) {
        self.device = device.clone();
    }
}

/// The Stochastic Gradient Descent (SGD) optimizer struct.
#[derive(Debug)]
pub struct SGD {
    learning_rate: f32,
    device: Device,
}

impl SGD {
    /// Creates a new SGD optimizer with the given learning rate.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for the optimizer.
    ///
    /// # Returns
    ///
    /// A new instance of the SGD optimizer.
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate, device: Device::default() }
    }
}

impl Optimizer for SGD {
    /// Performs an optimization step using the given gradients.
    ///
    /// # Arguments
    ///
    /// * `weights` - A mutable reference to the weights tensor.
    /// * `gradients` - A reference to the gradients tensor.
    fn step(&mut self, weights: &mut Tensor, gradients: &Tensor) -> Result<(), OptimizerError> {
        if self.learning_rate <= 0.0 {
            return Err(OptimizerError::InvalidLearningRate(
                "Learning rate must be greater than 0.".to_string(),
            ));
        }

        // Ensure gradients match the weights' shape
        if gradients.shape().raw_dim().as_array_view().to_vec()
            != weights.shape().raw_dim().as_array_view().to_vec()
        {
            return Err(OptimizerError::IncompatibleGradientWeightShape(
                gradients.shape().raw_dim().as_array_view().to_vec(),
                weights.shape().raw_dim().as_array_view().to_vec(),
            ));
        }

        // Update weights using the learning rate and gradients
        *weights -= gradients.mul_scalar(self.learning_rate);

        Ok(())
    }

    /// Sets the device for the optimizer.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to set for the optimizer.
    fn set_device(&mut self, device: &Device) {
        self.device = device.clone();
    }
}

/// The SGD with Momentum optimizer struct.
#[derive(Debug)]
pub struct SGDWithMomentum {
    #[allow(dead_code)]
    learning_rate: f32,
    #[allow(dead_code)]
    momentum: f32,
    velocity: Option<Tensor>,
    device: Device,
}

impl SGDWithMomentum {
    /// Creates a new SGDWithMomentum optimizer with the given learning rate and momentum.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for the optimizer.
    /// * `momentum` - The momentum factor.
    ///
    /// # Returns
    ///
    /// A new instance of the SGDWithMomentum optimizer.
    pub fn new(learning_rate: f32, momentum: f32) -> Self {
        Self { learning_rate, momentum, velocity: None, device: Device::default() }
    }
}

impl Optimizer for SGDWithMomentum {
    /// Performs an optimization step using the given gradients.
    ///
    /// # Arguments
    ///
    /// * `weights` - A mutable reference to the weights tensor.
    /// * `gradients` - A reference to the gradients tensor.
    fn step(&mut self, weights: &mut Tensor, gradients: &Tensor) -> Result<(), OptimizerError> {
        if self.learning_rate <= 0.0 {
            return Err(OptimizerError::InvalidLearningRate(
                "Learning rate must be greater than 0.".to_string(),
            ));
        }

        // Initialize velocity if not already done
        if self.velocity.is_none()
            || self.velocity.as_ref().unwrap().shape().raw_dim() != weights.shape().raw_dim()
        {
            self.velocity = Some(Tensor::zeros(weights.shape().clone(), self.device.clone()));
        }

        let velocity = self.velocity.as_mut().unwrap();

        // Ensure gradients match the weights' shape
        let processed_gradients = if gradients.shape().raw_dim() == weights.shape().raw_dim() {
            gradients.clone()
        } else if gradients.shape().raw_dim().ndim() <= weights.shape().raw_dim().ndim()
            && gradients
                .shape()
                .raw_dim()
                .as_array_view()
                .iter()
                .rev()
                .zip(weights.shape().raw_dim().as_array_view().iter().rev())
                .all(|(g, w)| *g == *w || *g == 1)
        {
            gradients.broadcast(weights.shape())
        } else {
            return Err(OptimizerError::IncompatibleGradientWeightShape(
                gradients.shape().raw_dim().as_array_view().to_vec(),
                weights.shape().raw_dim().as_array_view().to_vec(),
            ));
        };

        // Update velocity: v = momentum * v - learning_rate * gradients
        *velocity = velocity
            .mul_scalar(self.momentum)
            .sub(&processed_gradients.mul_scalar(self.learning_rate));

        // Update weights: w = w + v
        *weights += velocity.clone();

        Ok(())
    }

    /// Sets the device for the optimizer.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to set for the optimizer.
    fn set_device(&mut self, device: &Device) {
        self.device = device.clone();
    }
}

/// AdamW optimizer implementation.
/// AdamW is a variant of Adam that implements weight decay regularization.
#[derive(Debug)]
pub struct AdamW {
    /// Learning rate
    learning_rate: f32,
    /// Beta1 coefficient for first moment estimate
    beta1: f32,
    /// Beta2 coefficient for second moment estimate
    beta2: f32,
    /// Small constant for numerical stability
    epsilon: f32,
    /// Weight decay coefficient
    weight_decay: f32,
    /// First moment estimate
    m: Option<Tensor>,
    /// Second moment estimate
    v: Option<Tensor>,
    /// Time step
    t: usize,
    /// Device to use for computations
    device: Device,
}

impl AdamW {
    /// Creates a new AdamW optimizer.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Learning rate
    /// * `beta1` - Beta1 coefficient (default: 0.9)
    /// * `beta2` - Beta2 coefficient (default: 0.999)
    /// * `epsilon` - Small constant for numerical stability (default: 1e-8)
    /// * `weight_decay` - Weight decay coefficient (default: 0.01)
    pub fn new(
        learning_rate: f32,
        beta1: Option<f32>,
        beta2: Option<f32>,
        epsilon: Option<f32>,
        weight_decay: Option<f32>,
    ) -> Self {
        Self {
            learning_rate,
            beta1: beta1.unwrap_or(0.9),
            beta2: beta2.unwrap_or(0.999),
            epsilon: epsilon.unwrap_or(1e-8),
            weight_decay: weight_decay.unwrap_or(0.01),
            m: None,
            v: None,
            t: 0,
            device: Device::Cpu,
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, weights: &mut Tensor, gradients: &Tensor) -> Result<(), OptimizerError> {
        // Initialize momentum and velocity if not already done
        if self.m.is_none()
            || self.m.as_ref().unwrap().shape().raw_dim() != weights.shape().raw_dim()
        {
            self.m = Some(Tensor::zeros(weights.shape().clone(), self.device.clone()));
            self.v = Some(Tensor::zeros(weights.shape().clone(), self.device.clone()));
        }

        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        // Increment timestep
        self.t += 1;

        // Process gradients to match weight shape if needed
        let processed_gradients = if gradients.shape().raw_dim() == weights.shape().raw_dim() {
            gradients.clone()
        } else if gradients.shape().raw_dim().ndim() <= weights.shape().raw_dim().ndim()
            && gradients
                .shape()
                .raw_dim()
                .as_array_view()
                .iter()
                .rev()
                .zip(weights.shape().raw_dim().as_array_view().iter().rev())
                .all(|(g, w)| *g == *w || *g == 1)
        {
            gradients.broadcast(weights.shape())
        } else {
            return Err(OptimizerError::IncompatibleGradientWeightShape(
                gradients.shape().raw_dim().as_array_view().to_vec(),
                weights.shape().raw_dim().as_array_view().to_vec(),
            ));
        };

        // Update biased first moment estimate
        *m = m.mul_scalar(self.beta1).add(&processed_gradients.mul_scalar(1.0 - self.beta1));

        // Update biased second raw moment estimate
        *v = v
            .mul_scalar(self.beta2)
            .add(&processed_gradients.map(|x| x * x).mul_scalar(1.0 - self.beta2));

        // Compute bias-corrected first moment estimate
        let m_hat = m.div_scalar(1.0 - self.beta1.powi(self.t as i32));

        // Compute bias-corrected second raw moment estimate
        let v_hat = v.div_scalar(1.0 - self.beta2.powi(self.t as i32));

        // Apply weight decay
        *weights = weights.mul_scalar(1.0 - self.learning_rate * self.weight_decay);

        // Update weights
        *weights = weights
            .sub(&m_hat.div(&v_hat.sqrt().add_scalar(self.epsilon)).mul_scalar(self.learning_rate));

        Ok(())
    }

    fn set_device(&mut self, device: &Device) {
        self.device = device.clone();
    }
}

#[cfg(test)]
mod tests {
    use crate::deep_learning::utils::assert_almost_equal;

    use super::*;
    use ndarray::{IxDyn, Shape};

    /// Default constants for RMSProp optimizer
    const DEFAULT_DECAY_RATE: f32 = 0.9;
    const DEFAULT_EPSILON: f32 = 1e-8;

    #[test]
    fn test_adadelta_optimizer() {
        let mut optimizer = AdaDelta::new(0.9, 1e-6);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2, 0.3], Shape::from(IxDyn(&[3, 1])));

        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        let expected = vec![0.99999684, 1.999_996_8, 2.999_997];
        assert_almost_equal(&weights.data, &expected, 1e-4);
    }

    #[test]
    fn test_adadelta_optimizer_multiple_steps() {
        let mut optimizer = AdaDelta::new(0.9, 1e-6);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2, 0.3], Shape::from(IxDyn(&[3, 1])));

        for _ in 0..5 {
            optimizer.step(&mut weights, &gradients).expect("Failed to perform step");
        }

        let expected = vec![0.99997528, 1.999_975_3, 2.999_975_2];
        assert_almost_equal(&weights.data, &expected, 1e-4);
    }

    #[test]
    fn test_adadelta_optimizer_zero_gradients() {
        let mut optimizer = AdaDelta::new(0.9, 1e-6);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3])));
        let gradients = Tensor::new(vec![0.0, 0.0, 0.0], Shape::from(IxDyn(&[3])));

        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        let expected = vec![1.0, 2.0, 3.0];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_adadelta_optimizer_incompatible_shapes() {
        let mut optimizer = AdaDelta::new(0.9, 1e-6);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3])));
        let gradients = Tensor::new(vec![0.1, 0.2], Shape::from(IxDyn(&[2])));

        let result = optimizer.step(&mut weights, &gradients);

        assert!(result.is_err(), "Expected an error due to incompatible shapes");
        if let Err(OptimizerError::IncompatibleGradientWeightShape(g_shape, w_shape)) = result {
            assert_eq!(g_shape, vec![2]);
            assert_eq!(w_shape, vec![3]);
        } else {
            panic!("Unexpected error type");
        }
    }

    #[test]
    fn test_adagrad_optimizer() {
        let mut optimizer = AdaGrad::new(0.1, 1e-8);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], IxDyn(&[3]).into());
        let gradients = Tensor::new(vec![0.1, 0.2, 0.3], IxDyn(&[3]).into());

        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        let expected = vec![0.90000004, 1.9, 2.9];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_adagrad_optimizer_incompatible_shapes() {
        let mut optimizer = AdaGrad::new(0.1, 1e-8);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], IxDyn(&[3, 1]).into());
        let gradients = Tensor::new(vec![0.1, 0.2], IxDyn(&[2, 1]).into());

        let result = optimizer.step(&mut weights, &gradients);

        assert!(result.is_err(), "Expected an error due to incompatible shapes");

        if let Err(OptimizerError::IncompatibleGradientWeightShape(g_shape, w_shape)) = result {
            assert_eq!(g_shape, vec![2, 1]);
            assert_eq!(w_shape, vec![3, 1]);
        } else {
            panic!("Unexpected error type");
        }
    }

    #[test]
    fn test_adagrad_optimizer_reset() {
        let mut optimizer = AdaGrad::new(0.1, 1e-8);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], IxDyn(&[3]).into());
        let gradients = Tensor::new(vec![0.1, 0.2, 0.3], IxDyn(&[3]).into());

        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        optimizer.reset();

        assert!(optimizer.g_sum.is_none(), "g_sum was not reset");
    }

    #[test]
    fn test_adam_optimizer() {
        let mut optimizer = Adam::new(0.001);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2, 0.3], Shape::from(IxDyn(&[3, 1])));
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");
        let expected = vec![0.999, 1.999, 2.999];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_adam_optimizer_no_scheduler() {
        let mut optimizer = Adam::new(0.001);
        let mut weights = Tensor::new(vec![0.0, 0.0, 0.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2, 0.3], Shape::from(IxDyn(&[3, 1])));
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        let expected = vec![-0.0009999934, -0.0009999934, -0.0009999934];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_adam_optimizer_with_scheduler() {
        let mut optimizer = Adam::new(0.001);
        optimizer.set_scheduler(|_epoch| 0.05); // Set a fixed learning rate for simplicity
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.1, 0.1], Shape::from(IxDyn(&[3, 1])));

        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        let expected = vec![0.95000035, 1.9500003, 2.9500003];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_adam_optimizer_gradients_broadcasting() {
        let mut optimizer = Adam::new(0.001);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1], Shape::from(IxDyn(&[1, 1]))); // Broadcastable gradient
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");
        let expected = vec![0.999, 1.999, 2.999];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_adam_optimizer_incompatible_shapes() {
        let mut optimizer = Adam::new(0.001);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2], Shape::from(IxDyn(&[2, 1]))); // Mismatched shape
        let result = optimizer.step(&mut weights, &gradients);

        assert!(result.is_err(), "Expected an error due to incompatible shapes");

        if let Err(OptimizerError::IncompatibleGradientWeightShape(g_shape, w_shape)) = result {
            assert_eq!(g_shape, vec![2, 1]);
            assert_eq!(w_shape, vec![3, 1]);
        } else {
            panic!("Unexpected error type");
        }
    }

    #[test]
    fn test_adam_optimizer_step_multiple_times() {
        let mut optimizer = Adam::new(0.001);
        let mut weights = Tensor::new(vec![1.0, 1.0, 1.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2, 0.3], Shape::from(IxDyn(&[3, 1])));
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        let expected = vec![0.99700004, 0.99700004, 0.99700004];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_adam_optimizer_bias_correction() {
        let mut optimizer = Adam::new(0.001);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.1, 0.1], Shape::from(IxDyn(&[3, 1])));

        // Step without bias correction
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        // Correct timestep
        assert_eq!(optimizer.timestep, 1);

        // Bias correction factors should influence the next step
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");
        assert_eq!(optimizer.timestep, 2);
    }

    #[test]
    fn test_adam_optimizer_zero_gradients() {
        let mut optimizer = Adam::new(0.001);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.0, 0.0, 0.0], Shape::from(IxDyn(&[3, 1])));
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        let expected = vec![1.0, 2.0, 3.0];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_adam_optimizer_gradient_scaling() {
        let mut optimizer = Adam::new(0.001); // Initial learning rate
        let mut weights = Tensor::new(vec![1.0, 1.0, 1.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![20.0, 20.0, 20.0], Shape::from(IxDyn(&[3, 1]))); // High gradient values

        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        // Compute expected weights
        let scaling_factor: f32 = 10.0 / 20.0; // Scale learning rate
        let adjusted_lr = 0.001 * scaling_factor;
        let epsilon = 1e-8;
        let m = 0.0 + (1.0 - 0.9) * 20.0; // m after one step
        let v = 0.0 + (1.0 - 0.999) * (20.0 * 20.0); // v after one step
        let m_hat = m / (1.0 - 0.9); // Bias-corrected m
        let v_hat: f32 = v / (1.0 - 0.999); // Bias-corrected v
        let update = m_hat / (v_hat.sqrt() + epsilon) * adjusted_lr;

        let expected = vec![1.0 - update, 1.0 - update, 1.0 - update];

        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_adam_optimizer_small_gradients() {
        let mut optimizer = Adam::new(0.001);
        let mut weights = Tensor::new(vec![1.0_f32, 2.0_f32, 3.0_f32], Shape::from(IxDyn(&[3, 1])));
        let gradients =
            Tensor::new(vec![1e-7_f32, 1e-7_f32, 1e-7_f32], Shape::from(IxDyn(&[3, 1])));

        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        // Compute expected weights manually
        let learning_rate: f32 = 0.001;
        let beta1: f32 = 0.9;
        let beta2: f32 = 0.999;
        let epsilon: f32 = 1e-8;

        let mut m: f32 = 0.0; // First moment estimate
        let mut v: f32 = 0.0; // Second moment estimate

        let g: f32 = 1e-7; // Gradient value
        m = beta1 * m + (1.0 - beta1) * g; // Update first moment
        v = beta2 * v + (1.0 - beta2) * (g * g); // Update second moment

        let m_hat: f32 = m / (1.0 - beta1.powi(1)); // Bias-corrected first moment
        let v_hat: f32 = v / (1.0 - beta2.powi(1)); // Bias-corrected second moment

        let update: f32 = m_hat / (v_hat.sqrt() + epsilon) * learning_rate;

        let expected = vec![1.0 - update, 2.0 - update, 3.0 - update];

        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_gradient_descent_optimizer() {
        let mut optimizer = GradientDescent::new(0.01);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2, 0.3], Shape::from(IxDyn(&[3, 1])));
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");
        let expected = vec![0.999, 1.998, 2.997];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_gradient_descent_optimizer_incompatible_shapes() {
        let mut optimizer = GradientDescent::new(0.01);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2], Shape::from(IxDyn(&[2, 1]))); // Mismatched shape
        let result = optimizer.step(&mut weights, &gradients);

        assert!(result.is_err(), "Expected an error due to incompatible shapes");

        if let Err(OptimizerError::IncompatibleGradientWeightShape(g_shape, w_shape)) = result {
            assert_eq!(g_shape, vec![2, 1]);
            assert_eq!(w_shape, vec![3, 1]);
        } else {
            panic!("Unexpected error type");
        }
    }

    #[test]
    fn test_gradient_descent_optimizer_zero_gradients() {
        let mut optimizer = GradientDescent::new(0.01);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.0, 0.0, 0.0], Shape::from(IxDyn(&[3, 1])));
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        let expected = vec![1.0, 2.0, 3.0];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_gradient_descent_optimizer_multiple_steps() {
        let mut optimizer = GradientDescent::new(0.01);
        let mut weights = Tensor::new(vec![1.0, 1.0, 1.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.1, 0.1], Shape::from(IxDyn(&[3, 1])));

        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        let expected = vec![0.997, 0.997, 0.997];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_minibatch_gd_optimizer() {
        let mut optimizer = MiniBatchGD::new(0.01);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2, 0.3], Shape::from(IxDyn(&[3, 1])));

        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        let expected = vec![0.999, 1.998, 2.997];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_minibatch_gd_incompatible_shapes() {
        let mut optimizer = MiniBatchGD::new(0.01);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2], Shape::from(IxDyn(&[2, 1])));

        let result = optimizer.step(&mut weights, &gradients);

        assert!(result.is_err(), "Expected an error due to incompatible shapes");

        if let Err(OptimizerError::IncompatibleGradientWeightShape(g_shape, w_shape)) = result {
            assert_eq!(g_shape, vec![2, 1]);
            assert_eq!(w_shape, vec![3, 1]);
        } else {
            panic!("Unexpected error type");
        }
    }

    #[test]
    fn test_minibatch_gd_zero_gradients() {
        let mut optimizer = MiniBatchGD::new(0.01);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.0, 0.0, 0.0], Shape::from(IxDyn(&[3, 1])));

        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        let expected = vec![1.0, 2.0, 3.0];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_minibatch_gd_large_learning_rate() {
        let mut optimizer = MiniBatchGD::new(1.0);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.1, 0.1], Shape::from(IxDyn(&[3, 1])));

        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        let expected = vec![0.9, 1.9, 2.9];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_rmsprop_optimizer_multiple_steps() {
        let mut optimizer = RMSProp::new(0.01, DEFAULT_DECAY_RATE, DEFAULT_EPSILON)
            .expect("Failed to create optimizer");
        let mut weights = Tensor::new(vec![1.0, 1.0, 1.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2, 0.3], Shape::from(IxDyn(&[3, 1])));

        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        // Update expected values based on manual calculation or reference implementation
        let expected = vec![/* Recalculated values */];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_rmsprop_optimizer_incompatible_shapes() {
        let mut optimizer = RMSProp::new(0.01, DEFAULT_DECAY_RATE, DEFAULT_EPSILON)
            .expect("Failed to create optimizer");
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2], Shape::from(IxDyn(&[2, 1])));
        let result = optimizer.step(&mut weights, &gradients);

        assert!(result.is_err(), "Expected an error due to incompatible shapes");
    }

    #[test]
    fn test_rmsprop_optimizer_zero_gradients() {
        let mut optimizer = RMSProp::new(0.01, DEFAULT_DECAY_RATE, DEFAULT_EPSILON)
            .expect("Failed to create optimizer");
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.0, 0.0, 0.0], Shape::from(IxDyn(&[3, 1])));

        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        let expected = vec![1.0, 2.0, 3.0];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_rmsprop_invalid_epsilon() {
        let result = RMSProp::new(0.01, DEFAULT_DECAY_RATE, 0.0);
        assert!(result.is_err(), "Expected an error due to invalid epsilon");
    }

    #[test]
    fn test_rmsprop_invalid_learning_rate() {
        let result = RMSProp::new(-0.01, DEFAULT_DECAY_RATE, DEFAULT_EPSILON);
        assert!(result.is_err(), "Expected an error due to invalid learning rate");
    }

    #[test]
    fn test_sgd_optimizer() {
        let mut optimizer = SGD::new(0.01);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2, 0.3], Shape::from(IxDyn(&[3, 1])));
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");
        let expected = vec![0.999, 1.998, 2.997];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_sgd_optimizer_zero_gradients() {
        let mut optimizer = SGD::new(0.01);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.0, 0.0, 0.0], Shape::from(IxDyn(&[3, 1])));
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        let expected = vec![1.0, 2.0, 3.0];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_sgd_optimizer_incompatible_shapes() {
        let mut optimizer = SGD::new(0.01);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2], Shape::from(IxDyn(&[2, 1]))); // Mismatched shape
        let result = optimizer.step(&mut weights, &gradients);

        assert!(result.is_err(), "Expected an error due to incompatible shapes");

        if let Err(OptimizerError::IncompatibleGradientWeightShape(g_shape, w_shape)) = result {
            assert_eq!(g_shape, vec![2, 1]);
            assert_eq!(w_shape, vec![3, 1]);
        } else {
            panic!("Unexpected error type");
        }
    }

    #[test]
    fn test_sgd_optimizer_large_learning_rate() {
        let mut optimizer = SGD::new(1.0); // Large learning rate
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2, 0.3], Shape::from(IxDyn(&[3, 1])));
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");
        let expected = vec![0.9, 1.8, 2.7];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_sgd_with_momentum_optimizer() {
        let mut optimizer = SGDWithMomentum::new(0.01, 0.9);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2, 0.3], Shape::from(IxDyn(&[3, 1])));
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");
        let expected = vec![0.999, 1.998, 2.997];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_sgd_with_momentum_multiple_steps() {
        let mut optimizer = SGDWithMomentum::new(0.01, 0.9);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.1, 0.1], Shape::from(IxDyn(&[3, 1])));

        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");
        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        let expected = vec![0.9971, 1.9971, 2.9971];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_sgd_with_momentum_zero_gradients() {
        let mut optimizer = SGDWithMomentum::new(0.01, 0.9);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.0, 0.0, 0.0], Shape::from(IxDyn(&[3, 1])));

        optimizer.step(&mut weights, &gradients).expect("Failed to perform step");

        let expected = vec![1.0, 2.0, 3.0];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_sgd_with_momentum_incompatible_shapes() {
        let mut optimizer = SGDWithMomentum::new(0.01, 0.9);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let gradients = Tensor::new(vec![0.1, 0.2], Shape::from(IxDyn(&[2, 1])));
        let result = optimizer.step(&mut weights, &gradients);

        assert!(result.is_err(), "Expected an error due to incompatible shapes");

        if let Err(OptimizerError::IncompatibleGradientWeightShape(g_shape, w_shape)) = result {
            assert_eq!(g_shape, vec![2, 1]);
            assert_eq!(w_shape, vec![3, 1]);
        } else {
            panic!("Unexpected error type");
        }
    }

    // AdamW Tests

    #[test]
    fn test_adamw_initialization_defaults() {
        let optimizer = AdamW::new(0.001, None, None, None, None);
        assert_eq!(optimizer.learning_rate, 0.001);
        assert_eq!(optimizer.beta1, 0.9);
        assert_eq!(optimizer.beta2, 0.999);
        assert_eq!(optimizer.epsilon, 1e-8);
        assert_eq!(optimizer.weight_decay, 0.01);
        assert_eq!(optimizer.t, 0);
        assert!(optimizer.m.is_none());
        assert!(optimizer.v.is_none());
    }

    #[test]
    fn test_adamw_single_step() {
        let mut optimizer = AdamW::new(0.1, None, None, None, None);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3])));
        let gradients = Tensor::new(vec![0.1, 0.2, 0.3], Shape::from(IxDyn(&[3])));

        optimizer.step(&mut weights, &gradients).unwrap();

        // Expected values calculated with:
        // m_hat = grad * (1-beta1)/(1-beta1^t) = grad * 1.0
        // v_hat = grad^2 * (1-beta2)/(1-beta2^t) = grad^2 * 1.0
        // w = w * (1 - lr*weight_decay) - lr * m_hat/sqrt(v_hat + eps)
        let expected = vec![0.9989, 1.9978, 2.9967];
        assert_almost_equal(&weights.data, &expected, 1e-4);
    }

    #[test]
    fn test_adamw_multiple_steps() {
        let mut optimizer = AdamW::new(0.1, None, None, None, None);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3])));
        let gradients = Tensor::new(vec![0.1, 0.1, 0.1], Shape::from(IxDyn(&[3])));

        for _ in 0..3 {
            optimizer.step(&mut weights, &gradients).unwrap();
        }

        let expected = vec![0.9945, 1.9945, 2.9945];
        assert_almost_equal(&weights.data, &expected, 1e-4);
    }
    #[test]
    fn test_adamw_zero_gradients() {
        let mut optimizer = AdamW::new(0.1, None, None, None, None);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3])));
        let initial_weights = weights.clone();
        let gradients = Tensor::new(vec![0.0, 0.0, 0.0], Shape::from(IxDyn(&[3])));

        optimizer.step(&mut weights, &gradients).unwrap();

        // Only weight decay should affect the weights
        let expected: Vec<f32> = initial_weights
            .data
            .iter()
            .map(|w| w * (1.0 - optimizer.learning_rate * optimizer.weight_decay))
            .collect();
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_adamw_small_gradients() {
        let mut optimizer = AdamW::new(0.1, None, None, None, None);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3])));
        let gradients = Tensor::new(vec![1e-10, 1e-10, 1e-10], Shape::from(IxDyn(&[3])));

        optimizer.step(&mut weights, &gradients).unwrap();

        // Weights should still update due to weight decay, even with tiny gradients
        let expected = Tensor::new(vec![0.999, 1.999, 2.999], Shape::from(IxDyn(&[3])));
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_adamw_large_gradients() {
        let mut optimizer = AdamW::new(0.1, None, None, None, None);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3])));
        let gradients = Tensor::new(vec![1e6, 1e6, 1e6], Shape::from(IxDyn(&[3])));

        optimizer.step(&mut weights, &gradients).unwrap();

        // Large gradients should be scaled by adaptive learning rate
        // Values should remain finite due to second moment scaling
        assert!(weights.data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_adamw_weight_decay_effect() {
        let mut optimizer = AdamW::new(0.1, None, None, None, Some(0.1));
        let mut weights = Tensor::new(vec![1.0, 1.0, 1.0], Shape::from(IxDyn(&[3])));
        let gradients = Tensor::new(vec![0.0, 0.0, 0.0], Shape::from(IxDyn(&[3])));

        optimizer.step(&mut weights, &gradients).unwrap();

        // With zero gradients, only weight decay affects weights
        let expected = vec![0.99, 0.99, 0.99];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_adamw_zero_weight_decay() {
        let mut optimizer = AdamW::new(0.1, None, None, None, Some(0.0));
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3])));
        let initial_weights = weights.data.clone();
        let gradients = Tensor::new(vec![0.0, 0.0, 0.0], Shape::from(IxDyn(&[3])));

        optimizer.step(&mut weights, &gradients).unwrap();

        // With zero weight decay and zero gradients, weights should remain unchanged
        assert_almost_equal(&weights.data, &initial_weights.into_raw_vec_and_offset().0, 1e-6);
    }

    #[test]
    fn test_adamw_large_weight_decay() {
        let mut optimizer = AdamW::new(0.1, None, None, None, Some(1.0));
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3])));
        let gradients = Tensor::new(vec![0.0, 0.0, 0.0], Shape::from(IxDyn(&[3])));

        optimizer.step(&mut weights, &gradients).unwrap();

        // Large weight decay should significantly reduce weights
        let expected = vec![0.9, 1.8, 2.7];
        assert_almost_equal(&weights.data, &expected, 1e-6);
    }

    #[test]
    fn test_adamw_weight_decay_zero_gradients() {
        let mut optimizer = AdamW::new(0.1, None, None, None, Some(0.01));
        let mut weights = Tensor::new(vec![10.0, 10.0, 10.0], Shape::from(IxDyn(&[3])));
        let gradients = Tensor::new(vec![0.0, 0.0, 0.0], Shape::from(IxDyn(&[3])));

        // Run multiple steps to verify weight decay compounds correctly
        for _ in 0..5 {
            optimizer.step(&mut weights, &gradients).unwrap();
        }

        // Weights should decay exponentially
        let expected = vec![9.9511, 9.9511, 9.9511];
        assert_almost_equal(&weights.data, &expected, 1e-4);
    }

    #[test]
    fn test_adamw_incompatible_shapes() {
        let mut optimizer = AdamW::new(0.1, None, None, None, None);
        let mut weights = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3])));
        let gradients = Tensor::new(vec![1.0, 2.0], Shape::from(IxDyn(&[2])));

        let result = optimizer.step(&mut weights, &gradients);
        assert!(matches!(result, Err(OptimizerError::IncompatibleGradientWeightShape(_, _))));
    }

    #[test]
    fn test_adamw_negative_learning_rate() {
        let mut result = AdamW::new(-0.1, None, None, None, None);
        let mut weights = Tensor::new(vec![1.0], Shape::from(IxDyn(&[1])));
        let gradients = Tensor::new(vec![1.0], Shape::from(IxDyn(&[1])));

        let result = result.step(&mut weights, &gradients);
        assert!(matches!(result, Err(OptimizerError::InvalidLearningRate(_))));
    }

    #[test]
    fn test_adamw_invalid_beta1() {
        let cases = vec![1.1, -0.1, 0.0, 1.0];
        for beta1 in cases {
            let mut optimizer = AdamW::new(0.1, Some(beta1), None, None, None);
            let mut weights = Tensor::new(vec![1.0], Shape::from(IxDyn(&[1])));
            let gradients = Tensor::new(vec![1.0], Shape::from(IxDyn(&[1])));

            let result = optimizer.step(&mut weights, &gradients);
            assert!(matches!(result, Err(OptimizerError::InvalidBeta(_))));
        }
    }

    #[test]
    fn test_adamw_invalid_beta2() {
        let cases = vec![1.1, -0.1, 0.0, 1.0];
        for beta2 in cases {
            let mut optimizer = AdamW::new(0.1, None, Some(beta2), None, None);
            let mut weights = Tensor::new(vec![1.0], Shape::from(IxDyn(&[1])));
            let gradients = Tensor::new(vec![1.0], Shape::from(IxDyn(&[1])));

            let result = optimizer.step(&mut weights, &gradients);
            assert!(matches!(result, Err(OptimizerError::InvalidBeta(_))));
        }
    }

    #[test]
    fn test_adamw_negative_epsilon() {
        let mut optimizer = AdamW::new(0.1, None, None, Some(-1e-8), None);
        let mut weights = Tensor::new(vec![1.0], Shape::from(IxDyn(&[1])));
        let gradients = Tensor::new(vec![1.0], Shape::from(IxDyn(&[1])));

        let result = optimizer.step(&mut weights, &gradients);
        assert!(matches!(result, Err(OptimizerError::InvalidEpsilon(_))));
    }

    #[test]
    fn test_adamw_negative_weight_decay() {
        let mut optimizer = AdamW::new(0.1, None, None, None, Some(-0.01));
        let mut weights = Tensor::new(vec![1.0], Shape::from(IxDyn(&[1])));
        let gradients = Tensor::new(vec![1.0], Shape::from(IxDyn(&[1])));

        let result = optimizer.step(&mut weights, &gradients);
        assert!(matches!(result, Err(OptimizerError::InvalidWeightDecay(_))));
    }

    #[test]
    fn test_adamw_nan_gradients() {
        let mut optimizer = AdamW::new(0.1, None, None, None, None);
        let mut weights = Tensor::new(vec![1.0], Shape::from(IxDyn(&[1])));
        let gradients = Tensor::new(vec![f32::NAN], Shape::from(IxDyn(&[1])));

        let result = optimizer.step(&mut weights, &gradients);
        assert!(matches!(result, Err(OptimizerError::InvalidGradient(_))));
    }

    #[test]
    fn test_adamw_inf_weights() {
        let mut optimizer = AdamW::new(0.1, None, None, None, None);
        let mut weights = Tensor::new(vec![f32::INFINITY], Shape::from(IxDyn(&[1])));
        let gradients = Tensor::new(vec![1.0], Shape::from(IxDyn(&[1])));

        let result = optimizer.step(&mut weights, &gradients);
        assert!(matches!(result, Err(OptimizerError::InvalidWeight(_))));
    }

    #[test]
    fn test_adamw_bias_correction() {
        let mut optimizer = AdamW::new(0.1, Some(0.9), Some(0.999), None, None);
        let mut weights = Tensor::new(vec![1.0], Shape::from(IxDyn(&[1])));
        let gradients = Tensor::new(vec![0.1], Shape::from(IxDyn(&[1])));

        // First step - bias correction should have maximum effect
        optimizer.step(&mut weights, &gradients).unwrap();

        // m_hat = 0.1 * (1-0.9)/(1-0.9^1) = 0.1
        // v_hat = 0.01 * (1-0.999)/(1-0.999^1) = 0.01
        // weight_decay = 1.0 * (1 - 0.1*0.01) = 0.999
        // update = 0.1 * 0.1/sqrt(0.01 + 1e-8) = 0.1
        let expected_after_first = 0.899;
        assert_almost_equal(&weights.data, &vec![expected_after_first], 1e-3);

        // Multiple steps to test evolving bias correction
        for _ in 0..9 {
            optimizer.step(&mut weights, &gradients).unwrap();
        }

        // After 10 steps, bias correction should be significantly reduced
        // m_hat ≈ m/(1-0.9^10) ≈ m/0.651
        // v_hat ≈ v/(1-0.999^10) ≈ v/0.0096
        let expected_after_ten = 0.801;
        assert_almost_equal(&weights.data, &vec![expected_after_ten], 1e-3);
    }

    #[test]
    fn test_adamw_momentum_updates() {
        let mut optimizer = AdamW::new(0.1, Some(0.9), Some(0.999), None, None);
        let mut weights = Tensor::new(vec![1.0], Shape::from(IxDyn(&[1])));

        // Alternating gradients to test momentum behavior
        let grad1 = Tensor::new(vec![0.1], Shape::from(IxDyn(&[1])));
        let grad2 = Tensor::new(vec![-0.1], Shape::from(IxDyn(&[1])));

        optimizer.step(&mut weights, &grad1).unwrap();
        optimizer.step(&mut weights, &grad2).unwrap();
        optimizer.step(&mut weights, &grad1).unwrap();

        // Momentum should smooth out the oscillating gradients
        let m = optimizer.m.as_ref().unwrap();
        assert!(m.data[0].abs() < 0.1, "Momentum should dampen oscillations");
    }

    #[test]
    fn test_adamw_second_moment() {
        let mut optimizer = AdamW::new(0.1, Some(0.9), Some(0.999), None, None);
        let mut weights = Tensor::new(vec![1.0], Shape::from(IxDyn(&[1])));

        // Large gradient followed by small gradients
        let grad_large = Tensor::new(vec![1.0], Shape::from(IxDyn(&[1])));
        let grad_small = Tensor::new(vec![0.1], Shape::from(IxDyn(&[1])));

        optimizer.step(&mut weights, &grad_large).unwrap();
        optimizer.step(&mut weights, &grad_small).unwrap();

        // Second moment should retain memory of large gradient
        let v = optimizer.v.as_ref().unwrap();
        assert!(v.data[0] > 0.5, "Second moment should track gradient magnitude history");
    }

    #[test]
    fn test_adamw_precise_updates() {
        let mut optimizer = AdamW::new(0.1, Some(0.9), Some(0.999), Some(1e-8), Some(0.01));
        let mut weights = Tensor::new(vec![1.0], Shape::from(IxDyn(&[1])));
        let gradients = Tensor::new(vec![0.5], Shape::from(IxDyn(&[1])));

        // Manually calculated expected values after one step:
        // m = 0.5 * 0.1 = 0.05
        // v = 0.25 * 0.001 = 0.00025
        // m_hat = 0.05 / (1 - 0.9) = 0.5
        // v_hat = 0.00025 / (1 - 0.999) = 0.25
        // weight_decay = 1.0 * (1 - 0.1 * 0.01) = 0.999
        // update = 0.1 * 0.5 / sqrt(0.25 + 1e-8) = 0.1
        optimizer.step(&mut weights, &gradients).unwrap();

        let expected = 0.899; // 0.999 - 0.1
        assert_almost_equal(&weights.data, &vec![expected], 1e-6);
    }

    #[test]
    fn test_adamw_memory_cleanup() {
        let mut optimizer = AdamW::new(0.1, None, None, None, None);
        let shape = Shape::from(IxDyn(&[1000, 1000])); // 1M elements

        // Create and update large tensors
        let mut weights = Tensor::zeros(shape.clone(), Device::Cpu);
        let gradients = Tensor::ones(shape.clone(), Device::Cpu);

        optimizer.step(&mut weights, &gradients).unwrap();

        // Verify internal state is allocated
        assert!(optimizer.m.is_some());
        assert!(optimizer.v.is_some());

        // Change shape to trigger cleanup
        let new_shape = Shape::from(IxDyn(&[100, 100]));
        weights = Tensor::zeros(new_shape.clone(), Device::Cpu);
        let new_gradients = Tensor::ones(new_shape.clone(), Device::Cpu);

        optimizer.step(&mut weights, &new_gradients).unwrap();

        // Verify internal state was resized
        assert_eq!(
            optimizer.m.as_ref().unwrap().shape().raw_dim().as_array_view().to_vec(),
            new_shape.raw_dim().as_array_view().to_vec()
        );
        assert_eq!(
            optimizer.v.as_ref().unwrap().shape().raw_dim().as_array_view().to_vec(),
            new_shape.raw_dim().as_array_view().to_vec()
        );
    }

    #[test]
    fn test_adamw_changing_shapes() {
        let mut optimizer = AdamW::new(0.1, None, None, None, None);

        // Test with vector
        let mut weights = Tensor::ones(Shape::from(IxDyn(&[10])), Device::Cpu);
        let gradients = Tensor::ones(Shape::from(IxDyn(&[10])), Device::Cpu);
        optimizer.step(&mut weights, &gradients).unwrap();

        // Change to matrix
        weights = Tensor::ones(Shape::from(IxDyn(&[2, 5])), Device::Cpu);
        let gradients = Tensor::ones(Shape::from(IxDyn(&[2, 5])), Device::Cpu);
        optimizer.step(&mut weights, &gradients).unwrap();

        // Change to 3D tensor
        weights = Tensor::ones(Shape::from(IxDyn(&[2, 2, 2])), Device::Cpu);
        let gradients = Tensor::ones(Shape::from(IxDyn(&[2, 2, 2])), Device::Cpu);
        optimizer.step(&mut weights, &gradients).unwrap();

        // Verify optimizer handles shape changes correctly
        assert_eq!(
            optimizer.m.as_ref().unwrap().shape().raw_dim().as_array_view().to_vec(),
            weights.shape().raw_dim().as_array_view().to_vec()
        );
        assert_eq!(
            optimizer.v.as_ref().unwrap().shape().raw_dim().as_array_view().to_vec(),
            weights.shape().raw_dim().as_array_view().to_vec()
        );
    }
}

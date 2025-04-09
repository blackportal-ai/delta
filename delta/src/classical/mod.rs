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

pub mod algorithms;
pub mod losses;

use std::ops::SubAssign;

use losses::Loss;
use num_traits::Float;

use ndarray::{Array1, Array2, ScalarOperand};

/// Defines a common interface for classical machine learning models.
///
/// This trait outlines the basic methods that all classical ML models should implement,
/// providing a uniform way to instantiate, train, and use models for predictions.
pub trait Algorithm<T, L>
where
    T: Float,
    L: Loss<T>,
{
    /// Creates and returns a new instance of the model.
    ///
    /// This method should initialize the model with default parameters or learnable parameters
    /// set to initial values. The `Sized` constraint ensures that `Self` has a known size at compile time.
    fn new(loss_function: L) -> Self
    where
        Self: Sized;

    /// Trains the model using the provided data.
    ///
    /// This method adjusts the model's parameters based on the input data to minimize the error
    /// between predictions and actual values. The training process involves multiple iterations
    /// or epochs, with each iteration potentially updating the model's parameters.
    ///
    /// # Arguments
    ///
    /// * `x` - An `Array2<f64>` representing the input features where each row is a sample
    ///   and each column is a feature.
    /// * `y` - An `Array1<f64>` representing the target values or labels for each sample.
    /// * `learning_rate` - A `f64` specifying how much to adjust the model's parameters with
    ///   each iteration.
    /// * `epochs` - A `usize` indicating the number of training iterations.
    fn fit(&mut self, x: &Array2<T>, y: &Array1<T>, learning_rate: T, epochs: usize);

    /// Makes predictions using the trained model.
    ///
    /// This method applies the model's learned parameters to new or test data to generate predictions.
    ///
    /// # Arguments
    ///
    /// * `x` - An `Array2<f64>` representing the input features for which predictions are to be made.
    ///
    /// # Returns
    ///
    /// Returns an `Array1<f64>` where each element is the model's prediction for the corresponding
    /// input sample.
    fn predict(&self, x: &Array2<T>) -> Array1<T>;
}

/// Performs batch gradient descent to compute the gradients for weights and bias.
///
/// This function calculates the gradients for updating the model parameters in a linear regression
/// context. It computes the predictions based on the current weights and bias, then calculates
/// the gradients of the loss function with respect to these parameters.
///
/// # Arguments
///
/// * `x` - An `Array2<f64>` representing the input features where each row is a sample and each column
///   is a feature.
/// * `y` - An `Array1<f64>` representing the true labels or target values for each sample.
/// * `weights` - An `Array1<f64>` representing the current weights of the model.
/// * `bias` - A `f64` representing the current bias of the model.
///
/// # Returns
///
/// Returns a tuple where:
///   - The first element is an `Array1<f64>` representing the gradient for the weights.
///   - The second element is a `f64` representing the gradient for the bias.
fn batch_gradient_descent<T>(
    x: &Array2<T>,
    y: &Array1<T>,
    weights: &Array1<T>,
    bias: T,
) -> (Array1<T>, T)
where
    T: Float + ScalarOperand + SubAssign,
{
    let predictions = x.dot(weights) + bias;
    let m = T::from(x.shape()[0]).unwrap();

    let grad_weights = x.t().dot(&(predictions.clone() - y)) / m;
    let grad_bias = (predictions - y).sum() / m;

    (grad_weights, grad_bias)
}

/// Performs gradient descent for Logistic Regression.
///
/// This function computes the gradients for the weights and bias in the logistic
/// regression model using the sigmoid function applied to the predictions. The
/// gradients are calculated as the partial derivatives of the cost function with
/// respect to the model parameters.
///
/// The logistic regression model uses the sigmoid function to model the probability
/// of the positive class. The gradients are then used to update the model parameters
/// during training to minimize the cost (log loss).
///
/// # Parameters:
/// - `x`: A reference to an `Array2<f64>` representing the input data matrix,
///   where each row is a training example and each column is a feature.
/// - `y`: A reference to an `Array1<f64>` representing the actual labels (0 or 1)
///   for the training examples.
/// - `weights`: A reference to an `Array1<f64>` representing the model weights.
///   Each weight corresponds to a feature in the input data.
/// - `bias`: A `f64` value representing the model's bias term.
///
/// # Returns:
/// A tuple `(grad_weights, grad_bias)` where:
/// - `grad_weights`: An `Array1<f64>` representing the gradients of the weights.
/// - `grad_bias`: A `f64` value representing the gradient of the bias term.
fn logistic_gradient_descent<T>(
    x: &Array2<T>,
    y: &Array1<T>,
    weights: &Array1<T>,
    bias: T,
) -> (Array1<T>, T)
where
    T: Float + ScalarOperand,
{
    let predictions = x.dot(weights) + bias;
    let m = T::from(x.shape()[0]).unwrap();

    let sigmoid_preds = predictions.mapv(|x| T::one() / (T::one() + (-x).exp()));

    let grad_weights = x.t().dot(&(sigmoid_preds.clone() - y)) / m;
    let grad_bias = (sigmoid_preds - y).sum() / m;

    (grad_weights, grad_bias)
}

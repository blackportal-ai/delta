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

use std::iter::Sum;

use ndarray::{Array1, ScalarOperand};
use num_traits::Float;

/// A struct representing the Mean Squared Error (MSE) loss function.
pub struct MSE;

/// A struct representing the Cross-Entropy loss function.
pub struct CrossEntropy;

/// A trait for loss functions, which calculates the error between predictions and actual values.
pub trait Loss<T> {
    /// Calculates the loss value given the predicted values and the actual values.
    ///
    /// # Arguments
    ///
    /// * `predictions` - A reference to an `Array1<T>` containing the predicted values.
    /// * `actuals` - A reference to an `Array1<T>` containing the actual values.
    ///
    /// # Returns
    ///
    /// Returns a value of type `T`, representing the calculated loss.
    fn calculate(&self, predictions: &Array1<T>, actuals: &Array1<T>) -> T;
}

impl<T> Loss<T> for MSE
where
    T: Float,
{
    /// Calculates the Mean Squared Error (MSE) loss.
    ///
    /// The MSE is calculated as the mean of the squared differences between the predicted
    /// values and the actual values.
    ///
    /// # Arguments
    ///
    /// * `predictions` - A reference to an `Array1<T>` containing the predicted values.
    /// * `actuals` - A reference to an `Array1<T>` containing the actual values.
    ///
    /// # Returns
    ///
    /// Returns a value of type `T`, representing the calculated MSE.
    fn calculate(&self, predictions: &Array1<T>, actuals: &Array1<T>) -> T {
        let m = T::from(predictions.len()).unwrap();
        let diff = predictions - actuals;
        (diff.mapv(|x| x.powi(2)).sum()) / m
    }
}

impl<T> Loss<T> for CrossEntropy
where
    T: Float + ScalarOperand + Sum,
{
    /// Calculates the Cross-Entropy loss.
    ///
    /// The Cross-Entropy loss is calculated by summing the negative log-likelihood of
    /// the predicted probabilities for each class. This function clamps predicted values
    /// between a small epsilon value and 1 - epsilon to avoid issues with logarithms of zero.
    ///
    /// # Arguments
    ///
    /// * `predictions` - A reference to an `Array1<T>` containing the predicted values, typically probabilities.
    /// * `actuals` - A reference to an `Array1<T>` containing the actual values, typically binary (0 or 1).
    ///
    /// # Returns
    ///
    /// Returns a value of type `T`, representing the calculated Cross-Entropy loss.
    fn calculate(&self, predictions: &Array1<T>, actuals: &Array1<T>) -> T {
        let m = T::from(predictions.len()).unwrap();
        let epsilon = T::from(1e-15).unwrap();

        predictions
            .iter()
            .zip(actuals.iter())
            .map(|(p, y)| {
                let p_clamped = p.max(epsilon).min(T::one() - epsilon);
                -(*y * p_clamped.ln() + (T::one() - *y) * (T::one() - p_clamped).ln())
            })
            .sum::<T>()
            / m
    }
}

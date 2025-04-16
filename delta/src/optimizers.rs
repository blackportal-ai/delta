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

use ndarray::{Array1, Array2};

use crate::errors::OptimizerError;

pub trait Optimizer {
    fn compute_gradients(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        weights: &Array1<f64>,
        bias: f64,
    ) -> Result<(Array1<f64>, f64), OptimizerError>;
}

pub struct BatchGradientDescent;

impl Optimizer for BatchGradientDescent {
    fn compute_gradients(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        weights: &Array1<f64>,
        bias: f64,
    ) -> Result<(Array1<f64>, f64), OptimizerError> {
        if x.is_empty() || y.is_empty() {
            return Err(OptimizerError::EmptyInput);
        }

        if x.shape()[0] == 0 {
            return Err(OptimizerError::ZeroSamples);
        }

        if x.shape()[1] != weights.len() {
            return Err(OptimizerError::DimensionMismatch {
                expected: x.shape()[1],
                actual: weights.len(),
            });
        }

        if x.shape()[0] != y.len() {
            return Err(OptimizerError::DimensionMismatch {
                expected: x.shape()[0],
                actual: y.len(),
            });
        }

        if x.iter().any(|&v| !v.is_finite())
            || y.iter().any(|&v| !v.is_finite())
            || weights.iter().any(|&v| !v.is_finite())
            || !bias.is_finite()
        {
            return Err(OptimizerError::InvalidNumericValue);
        }

        let predictions = x.dot(weights) + bias;
        let errors = &predictions - y;
        let grad_weights = x.t().dot(&errors) / x.shape()[0] as f64;
        let grad_bias = errors.mean().ok_or(OptimizerError::NumericalInstability)?;

        if !grad_weights.iter().all(|&v| v.is_finite()) || !grad_bias.is_finite() {
            return Err(OptimizerError::NumericalInstability);
        }

        Ok((grad_weights, grad_bias))
    }
}

pub struct LogisticGradientDescent;

impl Optimizer for LogisticGradientDescent {
    fn compute_gradients(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        weights: &Array1<f64>,
        bias: f64,
    ) -> Result<(Array1<f64>, f64), OptimizerError> {
        if x.is_empty() || y.is_empty() {
            return Err(OptimizerError::EmptyInput);
        }

        if x.shape()[0] == 0 {
            return Err(OptimizerError::ZeroSamples);
        }

        if x.shape()[1] != weights.len() {
            return Err(OptimizerError::DimensionMismatch {
                expected: x.shape()[1],
                actual: weights.len(),
            });
        }

        if x.shape()[0] != y.len() {
            return Err(OptimizerError::DimensionMismatch {
                expected: x.shape()[0],
                actual: y.len(),
            });
        }

        if x.iter().any(|&v| !v.is_finite())
            || y.iter().any(|&v| !v.is_finite())
            || weights.iter().any(|&v| !v.is_finite())
            || !bias.is_finite()
        {
            return Err(OptimizerError::InvalidNumericValue);
        }

        let linear_output = x.dot(weights) + bias;
        let predictions = linear_output.mapv(|z| 1.0 / (1.0 + (-z).exp()));
        let errors = &predictions - y;
        let grad_weights = x.t().dot(&errors) / x.shape()[0] as f64;
        let grad_bias = errors.mean().ok_or(OptimizerError::NumericalInstability)?;

        if !grad_weights.iter().all(|&v| v.is_finite()) || !grad_bias.is_finite() {
            return Err(OptimizerError::NumericalInstability);
        }

        Ok((grad_weights, grad_bias))
    }
}

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

use ndarray::Array1;

use crate::errors::LossError;

pub trait LossFunction {
    fn calculate(&self, predictions: &Array1<f64>, actuals: &Array1<f64>)
    -> Result<f64, LossError>;
}

pub struct MSE;

impl LossFunction for MSE {
    fn calculate(
        &self,
        predictions: &Array1<f64>,
        actuals: &Array1<f64>,
    ) -> Result<f64, LossError> {
        if predictions.is_empty() || actuals.is_empty() {
            return Err(LossError::EmptyInput);
        }

        if predictions.len() != actuals.len() {
            return Err(LossError::DimensionMismatch {
                expected: predictions.len(),
                actual: actuals.len(),
            });
        }

        if predictions.iter().any(|&v| !v.is_finite()) || actuals.iter().any(|&v| !v.is_finite()) {
            return Err(LossError::InvalidNumericValue);
        }

        let diff = predictions - actuals;
        let mse = diff.mapv(|x| x * x).mean().ok_or(LossError::EmptyInput)?;
        Ok(mse)
    }
}

pub struct CrossEntropy;

impl LossFunction for CrossEntropy {
    fn calculate(
        &self,
        predictions: &Array1<f64>,
        actuals: &Array1<f64>,
    ) -> Result<f64, LossError> {
        if predictions.is_empty() || actuals.is_empty() {
            return Err(LossError::EmptyInput);
        }

        if predictions.len() != actuals.len() {
            return Err(LossError::DimensionMismatch {
                expected: predictions.len(),
                actual: actuals.len(),
            });
        }

        if predictions.iter().any(|&v| !v.is_finite()) || actuals.iter().any(|&v| !v.is_finite()) {
            return Err(LossError::InvalidNumericValue);
        }

        if predictions.iter().any(|&p| p < 0.0 || p > 1.0) {
            return Err(LossError::InvalidPredictionRange);
        }

        if actuals.iter().any(|&y| y != 0.0 && y != 1.0) {
            return Err(LossError::InvalidActualValue);
        }

        let epsilon = 1e-15;
        let clipped_preds = predictions.mapv(|x| x.max(epsilon).min(1.0 - epsilon));
        let log_loss = actuals
            .iter()
            .zip(clipped_preds.iter())
            .map(|(&y, &p)| -y * p.ln() - (1.0 - y) * (1.0 - p).ln())
            .sum::<f64>()
            / actuals.len() as f64;
        Ok(log_loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, array};

    #[test]
    fn test_mse_empty_input() {
        let loss = MSE;
        let predictions: Array1<f64> = Array1::zeros(0);
        let actuals = array![1.0];
        let result = loss.calculate(&predictions, &actuals);
        assert!(matches!(result, Err(LossError::EmptyInput)));
    }

    #[test]
    fn test_mse_dimension_mismatch() {
        let loss = MSE;
        let predictions = array![1.0, 2.0];
        let actuals = array![1.0, 2.0, 3.0];
        let result = loss.calculate(&predictions, &actuals);
        assert!(matches!(result, Err(LossError::DimensionMismatch { expected: 2, actual: 3 })));
    }

    #[test]
    fn test_mse_invalid_numeric_value() {
        let loss = MSE;
        let predictions = array![1.0, f64::NAN];
        let actuals = array![1.0, 2.0];
        let result = loss.calculate(&predictions, &actuals);
        assert!(matches!(result, Err(LossError::InvalidNumericValue)));
    }

    #[test]
    fn test_mse_valid_computation() {
        let loss = MSE;
        let predictions = array![1.0, 2.0, 3.0];
        let actuals = array![1.1, 2.1, 3.1];
        let result = loss.calculate(&predictions, &actuals);
        assert!(result.is_ok());
        let mse = result.unwrap();
        assert!((mse - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_cross_entropy_empty_input() {
        let loss = CrossEntropy;
        let predictions = Array1::zeros(0);
        let actuals = array![0.0];
        let result = loss.calculate(&predictions, &actuals);
        assert!(matches!(result, Err(LossError::EmptyInput)));
    }

    #[test]
    fn test_cross_entropy_dimension_mismatch() {
        let loss = CrossEntropy;
        let predictions = array![0.1, 0.9];
        let actuals = array![0.0, 1.0, 0.0];
        let result = loss.calculate(&predictions, &actuals);
        assert!(matches!(result, Err(LossError::DimensionMismatch { expected: 2, actual: 3 })));
    }

    #[test]
    fn test_cross_entropy_invalid_numeric_value() {
        let loss = CrossEntropy;
        let predictions = array![0.1, f64::INFINITY];
        let actuals = array![0.0, 1.0];
        let result = loss.calculate(&predictions, &actuals);
        assert!(matches!(result, Err(LossError::InvalidNumericValue)));
    }

    #[test]
    fn test_cross_entropy_invalid_prediction_range() {
        let loss = CrossEntropy;
        let predictions = array![0.1, 1.1];
        let actuals = array![0.0, 1.0];
        let result = loss.calculate(&predictions, &actuals);
        assert!(matches!(result, Err(LossError::InvalidPredictionRange)));
    }

    #[test]
    fn test_cross_entropy_invalid_actual_value() {
        let loss = CrossEntropy;
        let predictions = array![0.1, 0.9];
        let actuals = array![0.0, 2.0];
        let result = loss.calculate(&predictions, &actuals);
        assert!(matches!(result, Err(LossError::InvalidActualValue)));
    }

    #[test]
    fn test_cross_entropy_valid_computation() {
        let loss = CrossEntropy;
        let predictions = array![0.1, 0.9];
        let actuals = array![0.0, 1.0];
        let result = loss.calculate(&predictions, &actuals);
        assert!(result.is_ok());
        let log_loss = result.unwrap();
        assert!(log_loss > 0.0);
        assert!(log_loss.is_finite());
    }
}

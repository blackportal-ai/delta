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

use ndarray::{Array1, Array2, Axis};

use crate::errors::{LossError, ModelError, ScalerError};
use crate::losses::{CrossEntropy, LossFunction, MSE};
use crate::optimizers::{BatchGradientDescent, LogisticGradientDescent, Optimizer};
use crate::scalers::StandardScaler;

#[derive(Clone)]
pub struct TrainingMetrics {
    pub loss: f64,
    // pub accuracy: f64,
}

pub struct LinearRegressionBuilder {
    loss_function: Box<dyn LossFunction>,
    normalize: bool,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    optimizer: Box<dyn Optimizer>,
}

impl LinearRegressionBuilder {
    pub fn optimizer(mut self, optimizer: impl Optimizer + 'static) -> Self {
        self.optimizer = Box::new(optimizer);
        self
    }

    pub fn loss_function(mut self, loss_function: impl LossFunction + 'static) -> Self {
        self.loss_function = Box::new(loss_function);
        self
    }

    pub fn scaler(mut self, scaler: StandardScaler) -> Self {
        self.x_scaler = scaler.clone();
        self.y_scaler = scaler;
        self
    }

    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    pub fn build(self) -> LinearRegression {
        LinearRegression {
            weights: None,
            bias: 0.0,
            loss_function: self.loss_function,
            normalize: self.normalize,
            x_scaler: self.x_scaler,
            y_scaler: self.y_scaler,
            optimizer: self.optimizer,
            metrics: Vec::new(),
        }
    }
}

pub struct LinearRegression {
    weights: Option<Array1<f64>>,
    bias: f64,
    loss_function: Box<dyn LossFunction>,
    normalize: bool,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    optimizer: Box<dyn Optimizer>,
    metrics: Vec<TrainingMetrics>,
}

impl LinearRegression {
    pub fn new() -> LinearRegressionBuilder {
        LinearRegressionBuilder {
            loss_function: Box::new(MSE),
            normalize: true,
            x_scaler: StandardScaler::new(),
            y_scaler: StandardScaler::new(),
            optimizer: Box::new(BatchGradientDescent),
        }
    }

    pub fn fit(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        learning_rate: f64,
        epochs: usize,
    ) -> Result<(), ModelError> {
        if x.is_empty() || y.is_empty() {
            return Err(ModelError::Scaler(ScalerError::EmptyInput));
        }
        if x.shape()[0] != y.shape()[0] {
            return Err(ModelError::Scaler(ScalerError::DimensionMismatch {
                expected: x.shape()[0],
                actual: y.shape()[0],
            }));
        }

        let (x_scaled, y_scaled) = if self.normalize {
            let x_scaled = self.x_scaler.fit_transform(x)?;
            let y_2d = y.clone().insert_axis(Axis(1));
            let y_scaled_2d = self.y_scaler.fit_transform(&y_2d)?;
            let y_scaled = y_scaled_2d.remove_axis(Axis(1));
            (x_scaled, y_scaled)
        } else {
            (x.clone(), y.clone())
        };

        let (_, n_features) = x_scaled.dim();
        self.weights = Some(Array1::zeros(n_features));
        self.metrics.clear();

        for _ in 0..epochs {
            let predictions = self.predict_linear(&x_scaled);
            let loss = self.loss_function.calculate(&predictions, &y_scaled)?;

            let (grad_weights, grad_bias) = self.optimizer.compute_gradients(
                &x_scaled,
                &y_scaled,
                self.weights.as_ref().expect("Weights not initialized"),
                self.bias,
            )?;

            self.weights = Some(
                self.weights.take().expect("Weights not initialized")
                    - &(grad_weights * learning_rate),
            );
            self.bias -= grad_bias * learning_rate;

            self.metrics.push(TrainingMetrics { loss });
        }
        Ok(())
    }

    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, ModelError> {
        let x_scaled = if self.normalize { self.x_scaler.transform(x)? } else { x.clone() };
        let mut predictions = self.predict_linear(&x_scaled);
        if self.normalize {
            let pred_2d = predictions.clone().insert_axis(Axis(1));
            let pred_scaled_2d = self.y_scaler.inverse_transform(&pred_2d)?;
            predictions = pred_scaled_2d.remove_axis(Axis(1));
        }
        Ok(predictions)
    }

    pub fn calculate_loss(
        &self,
        predictions: &Array1<f64>,
        actuals: &Array1<f64>,
    ) -> Result<f64, LossError> {
        self.loss_function.calculate(predictions, actuals)
    }

    pub fn metrics(&self) -> &[TrainingMetrics] {
        &self.metrics
    }

    #[inline(always)]
    fn predict_linear(&self, x: &Array2<f64>) -> Array1<f64> {
        let weights = self.weights.as_ref().expect("Model not fitted");
        x.dot(weights) + self.bias
    }
}

pub struct LogisticRegressionBuilder {
    loss_function: Box<dyn LossFunction>,
    normalize: bool,
    x_scaler: StandardScaler,
    optimizer: Box<dyn Optimizer>,
}

impl LogisticRegressionBuilder {
    pub fn optimizer(mut self, optimizer: impl Optimizer + 'static) -> Self {
        self.optimizer = Box::new(optimizer);
        self
    }

    pub fn loss_function(mut self, loss_function: impl LossFunction + 'static) -> Self {
        self.loss_function = Box::new(loss_function);
        self
    }

    pub fn scaler(mut self, scaler: StandardScaler) -> Self {
        self.x_scaler = scaler;
        self
    }

    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    pub fn build(self) -> LogisticRegression {
        LogisticRegression {
            weights: Array1::zeros(0),
            bias: 0.0,
            loss_function: self.loss_function,
            normalize: self.normalize,
            x_scaler: self.x_scaler,
            optimizer: self.optimizer,
        }
    }
}

pub struct LogisticRegression {
    weights: Array1<f64>,
    bias: f64,
    loss_function: Box<dyn LossFunction>,
    normalize: bool,
    x_scaler: StandardScaler,
    optimizer: Box<dyn Optimizer>,
}

impl LogisticRegression {
    pub fn new() -> LogisticRegressionBuilder {
        LogisticRegressionBuilder {
            loss_function: Box::new(CrossEntropy),
            normalize: true,
            x_scaler: StandardScaler::new(),
            optimizer: Box::new(LogisticGradientDescent),
        }
    }

    pub fn fit(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        learning_rate: f64,
        epochs: usize,
    ) -> Result<(), ModelError> {
        if x.is_empty() || y.is_empty() {
            return Err(ModelError::Scaler(ScalerError::EmptyInput));
        }
        if x.shape()[0] != y.shape()[0] {
            return Err(ModelError::Scaler(ScalerError::DimensionMismatch {
                expected: x.shape()[0],
                actual: y.shape()[0],
            }));
        }
        if y.iter().any(|&v| v < 0.0 || v > 1.0) {
            return Err(ModelError::Scaler(ScalerError::DimensionMismatch {
                expected: 1,
                actual: 0,
            }));
        }

        let x_scaled = if self.normalize { self.x_scaler.fit_transform(x)? } else { x.clone() };
        if self.weights.len() == 0 {
            self.weights = Array1::zeros(x_scaled.shape()[1]);
        }

        for _ in 0..epochs {
            // These are commented out temporarily until we will store the predictions and loss for metrics
            // let linear_output = self.predict_linear(&x_scaled);
            // let predictions = self.sigmoid(&linear_output);
            // let _loss = self.loss_function.calculate(&predictions, y)?;

            let (grad_weights, grad_bias) =
                self.optimizer.compute_gradients(&x_scaled, y, &self.weights, self.bias)?;

            self.weights -= &(grad_weights * learning_rate);
            self.bias -= grad_bias * learning_rate;
        }
        Ok(())
    }

    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, ModelError> {
        let x_scaled = if self.normalize { self.x_scaler.transform(x)? } else { x.clone() };
        let linear_output = self.predict_linear(&x_scaled);
        Ok(self.sigmoid(&linear_output))
    }

    pub fn calculate_loss(
        &self,
        predictions: &Array1<f64>,
        actuals: &Array1<f64>,
    ) -> Result<f64, LossError> {
        self.loss_function.calculate(predictions, actuals)
    }

    #[inline(always)]
    fn predict_linear(&self, x: &Array2<f64>) -> Array1<f64> {
        x.dot(&self.weights) + self.bias
    }

    #[inline(always)]
    fn sigmoid(&self, z: &Array1<f64>) -> Array1<f64> {
        z.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::ModelError;
    use crate::losses::{CrossEntropy, MSE};
    use crate::optimizers::{BatchGradientDescent, LogisticGradientDescent};
    use ndarray::{Array1, Array2, array};

    #[test]
    fn linear_regression_fit_empty_input() {
        let mut model = LinearRegression::new().build();
        let x: Array2<f64> = Array2::zeros((0, 2));
        let y: Array1<f64> = Array1::zeros(0);
        let result = model.fit(&x, &y, 0.01, 10);
        assert!(matches!(result, Err(ModelError::Scaler(ScalerError::EmptyInput))));
    }

    #[test]
    fn linear_regression_fit_no_features() {
        let mut model = LinearRegression::new().build();
        let x: Array2<f64> = Array2::zeros((2, 0));
        let y = array![1.0, 2.0];
        let result = model.fit(&x, &y, 0.01, 10);
        assert!(matches!(result, Err(ModelError::Scaler(ScalerError::EmptyInput))));
    }

    #[test]
    fn linear_regression_fit_dimension_mismatch() {
        let mut model = LinearRegression::new().build();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0, 3.0];
        let result = model.fit(&x, &y, 0.01, 10);
        assert!(matches!(
            result,
            Err(ModelError::Scaler(ScalerError::DimensionMismatch { expected: 2, actual: 3 }))
        ));
    }

    #[test]
    fn linear_regression_predict_not_fitted() {
        let model = LinearRegression::new().build();
        let x = array![[1.0, 2.0]];
        let result = model.predict(&x);
        assert!(matches!(result, Err(ModelError::Scaler(ScalerError::NotFitted))));
    }

    #[test]
    fn linear_regression_predict_dimension_mismatch() {
        let mut model = LinearRegression::new().build();
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let y_train = array![1.0, 2.0];
        model.fit(&x_train, &y_train, 0.01, 10).unwrap();
        let x_test = array![[1.0, 2.0, 3.0]];
        let result = model.predict(&x_test);
        assert!(matches!(
            result,
            Err(ModelError::Scaler(ScalerError::DimensionMismatch { expected: 2, actual: 3 }))
        ));
    }

    #[test]
    fn linear_regression_fit_predict() {
        let mut model = LinearRegression::new()
            .optimizer(BatchGradientDescent)
            .loss_function(MSE)
            .normalize(false)
            .build();
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![2.0, 4.0, 6.0];
        model.fit(&x, &y, 0.01, 100).unwrap();
        let predictions = model.predict(&x).unwrap();
        for (p, &t) in predictions.iter().zip(y.iter()) {
            assert!((p - t).abs() < 1.0);
        }
    }

    #[test]
    fn linear_regression_calculate_loss() {
        let model = LinearRegression::new().build();
        let predictions = array![1.0, 2.0, 3.0];
        let actuals = array![1.1, 2.1, 3.1];
        let loss = model.calculate_loss(&predictions, &actuals).unwrap();
        assert!((loss - 0.01).abs() < 1e-6);
    }

    #[test]
    fn logistic_regression_fit_empty_input() {
        let mut model = LogisticRegression::new().build();
        let x: Array2<f64> = Array2::zeros((0, 2));
        let y: Array1<f64> = Array1::zeros(0);
        let result = model.fit(&x, &y, 0.01, 10);
        assert!(matches!(result, Err(ModelError::Scaler(ScalerError::EmptyInput))));
    }

    #[test]
    fn logistic_regression_fit_no_features() {
        let mut model = LogisticRegression::new().build();
        let x: Array2<f64> = Array2::zeros((2, 0));
        let y = array![0.0, 1.0];
        let result = model.fit(&x, &y, 0.01, 10);
        assert!(matches!(result, Err(ModelError::Scaler(ScalerError::EmptyInput))));
    }

    #[test]
    fn logistic_regression_fit_dimension_mismatch() {
        let mut model = LogisticRegression::new().build();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0, 0.0];
        let result = model.fit(&x, &y, 0.01, 10);
        assert!(matches!(
            result,
            Err(ModelError::Scaler(ScalerError::DimensionMismatch { expected: 2, actual: 3 }))
        ));
    }

    #[test]
    fn logistic_regression_fit_invalid_labels() {
        let mut model = LogisticRegression::new().build();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 2.0];
        let result = model.fit(&x, &y, 0.01, 10);
        assert!(matches!(
            result,
            Err(ModelError::Scaler(ScalerError::DimensionMismatch { expected: 1, actual: 0 }))
        ));
    }

    #[test]
    fn logistic_regression_predict_not_fitted() {
        let model = LogisticRegression::new().build();
        let x = array![[1.0, 2.0]];
        let result = model.predict(&x);
        assert!(matches!(result, Err(ModelError::Scaler(ScalerError::NotFitted))));
    }

    #[test]
    fn logistic_regression_predict_dimension_mismatch() {
        let mut model = LogisticRegression::new().build();
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let y_train = array![0.0, 1.0];
        model.fit(&x_train, &y_train, 0.01, 10).unwrap();
        let x_test = array![[1.0, 2.0, 3.0]];
        let result = model.predict(&x_test);
        assert!(matches!(
            result,
            Err(ModelError::Scaler(ScalerError::DimensionMismatch { expected: 2, actual: 3 }))
        ));
    }

    #[test]
    fn logistic_regression_fit_predict() {
        let mut model = LogisticRegression::new()
            .optimizer(LogisticGradientDescent)
            .loss_function(CrossEntropy)
            .normalize(false)
            .build();
        let x = array![[0.0], [1.0]];
        let y = array![0.0, 1.0];
        model.fit(&x, &y, 0.1, 100).unwrap();
        let predictions = model.predict(&x).unwrap();
        assert!(predictions[0] < 0.5);
        assert!(predictions[1] > 0.5);
    }

    #[test]
    fn logistic_regression_calculate_loss() {
        let model = LogisticRegression::new().build();
        let predictions = array![0.1, 0.9];
        let actuals = array![0.0, 1.0];
        let loss = model.calculate_loss(&predictions, &actuals).unwrap();
        assert!(loss > 0.0);
    }
}

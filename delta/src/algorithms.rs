// BSD 3-Clause License
//
// Copyright (c) 2025, BlackPortal
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

use std::collections::HashMap;

use ndarray::{Array1, Array2, Axis};

use crate::errors::{LossError, ModelError, PreprocessingError};
use crate::losses::{CrossEntropy, LossFunction, MSE};
use crate::optimizers::{BatchGradientDescent, LogisticGradientDescent, Optimizer};
use crate::preprocessors::StandardScaler;

// Re-export KNNMode variants for cleaner syntax
pub use self::KNNMode::{Classification, Regression};

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
            return Err(ModelError::Preprocessing(PreprocessingError::EmptyInput));
        }
        if x.shape()[0] != y.shape()[0] {
            return Err(ModelError::Preprocessing(PreprocessingError::DimensionMismatch {
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

        for _ in 0..epochs {
            // These are commented out temporarily until we will store the predictions and loss for metrics
            // let predictions = self.predict_linear(&x_scaled);
            // let _loss = self.loss_function.calculate(&predictions, &y_scaled)?;

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
            return Err(ModelError::Preprocessing(PreprocessingError::EmptyInput));
        }
        if x.shape()[0] != y.shape()[0] {
            return Err(ModelError::Preprocessing(PreprocessingError::DimensionMismatch {
                expected: x.shape()[0],
                actual: y.shape()[0],
            }));
        }
        if y.iter().any(|&v| v < 0.0 || v > 1.0) {
            return Err(ModelError::Preprocessing(PreprocessingError::DimensionMismatch {
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

#[derive(Clone, Copy)]
pub enum KNNMode {
    Classification,
    Regression,
}

pub struct KNNBuilder {
    k: usize,
    normalize: bool,
    x_scaler: StandardScaler,
    mode: KNNMode,
}

impl KNNBuilder {
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    pub fn scaler(mut self, scaler: StandardScaler) -> Self {
        self.x_scaler = scaler;
        self
    }

    pub fn mode(mut self, mode: KNNMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn build(self) -> KNN {
        KNN {
            x_train: None,
            y_train: None,
            k: self.k,
            normalize: self.normalize,
            x_scaler: self.x_scaler,
            mode: self.mode,
        }
    }
}

pub struct KNN {
    x_train: Option<Array2<f64>>,
    y_train: Option<Array1<f64>>,
    k: usize,
    normalize: bool,
    x_scaler: StandardScaler,
    mode: KNNMode,
}

impl KNN {
    pub fn new() -> KNNBuilder {
        KNNBuilder {
            k: 3,
            normalize: true,
            x_scaler: StandardScaler::new(),
            mode: KNNMode::Classification,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), ModelError> {
        if x.ncols() == 0 {
            return Err(ModelError::Preprocessing(PreprocessingError::NoFeatures));
        }
        if x.is_empty() || y.is_empty() {
            return Err(ModelError::Preprocessing(PreprocessingError::EmptyInput));
        }
        if x.shape()[0] != y.shape()[0] {
            return Err(ModelError::Preprocessing(PreprocessingError::DimensionMismatch {
                expected: x.shape()[0],
                actual: y.shape()[0],
            }));
        }
        if self.k == 0 || self.k > x.shape()[0] {
            return Err(ModelError::Preprocessing(PreprocessingError::InvalidParameter));
        }

        let x_scaled = if self.normalize { self.x_scaler.fit_transform(x)? } else { x.clone() };

        self.x_train = Some(x_scaled);
        self.y_train = Some(y.clone());
        Ok(())
    }

    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, ModelError> {
        let x_train = self.x_train.as_ref().ok_or(ModelError::Preprocessing(PreprocessingError::NotFitted))?;
        let y_train = self.y_train.as_ref().ok_or(ModelError::Preprocessing(PreprocessingError::NotFitted))?;

        if x.is_empty() {
            return Err(ModelError::Preprocessing(PreprocessingError::EmptyInput));
        }
        if x.ncols() != x_train.ncols() {
            return Err(ModelError::Preprocessing(PreprocessingError::DimensionMismatch {
                expected: x_train.ncols(),
                actual: x.ncols(),
            }));
        }

        let x_scaled = if self.normalize {
            self.x_scaler.transform(x).map_err(ModelError::Preprocessing)?
        } else {
            x.clone()
        };

        let mut predictions = Array1::zeros(x_scaled.nrows());
        for (i, row) in x_scaled.axis_iter(Axis(0)).enumerate() {
            let distances = Array1::from_iter(x_train.axis_iter(Axis(0)).map(|v| {
                let squared_sum = ndarray::Zip::from(v).and(row).fold(0.0, |acc, &v_i, &row_i| {
                    let diff = v_i - row_i;
                    acc + diff * diff
                });
                squared_sum.sqrt()
            }));

            let mut indices: Vec<(usize, f64)> =
                distances.iter().enumerate().map(|(i, &d)| (i, d)).collect();
            indices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let k_indices = indices.iter().take(self.k).map(|&(idx, _)| idx);

            match self.mode {
                KNNMode::Classification => {
                    // Collect class labels of k nearest neighbors as integers
                    let mut class_counts = HashMap::new();
                    for idx in k_indices {
                        let label = y_train[idx];
                        // Optional: Validate labels for Iris (comment out for general use)
                        if label != 0.0 && label != 1.0 && label != 2.0 {
                            return Err(ModelError::Preprocessing(PreprocessingError::InvalidParameter));
                        }
                        let label_int = label as usize; // Safe for 0.0, 1.0, 2.0
                        *class_counts.entry(label_int).or_insert(0) += 1;
                    }

                    let (predicted_class_int, _) = class_counts
                        .into_iter()
                        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
                        .unwrap_or((0, 0)); // Default to class 0 if no votes
                    predictions[i] = predicted_class_int as f64;
                }
                KNNMode::Regression => {
                    // Average the labels for regression
                    let mean = k_indices.map(|idx| y_train[idx]).sum::<f64>() / self.k as f64;
                    predictions[i] = mean;
                }
            }
        }
        Ok(predictions)
    }

    pub fn calculate_accuracy(&self, predictions: &Array1<f64>, y_test: &Array1<f64>) -> f64 {
        let correct = predictions
            .iter()
            .zip(y_test.iter())
            .filter(|(&pred, &true_label)| pred == true_label)
            .count();
        correct as f64 / predictions.len() as f64
    }
}


pub struct NaiveBayesBuilder {
    alpha: f64,
    normalize: bool,
    x_scaler: StandardScaler,
}

impl NaiveBayesBuilder {
    pub fn new() -> Self {
        NaiveBayesBuilder {
            alpha: 1.0,
            normalize: false,
            x_scaler: StandardScaler::new(),
        }
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    pub fn scaler(mut self, scaler: StandardScaler) -> Self {
        self.x_scaler = scaler;
        self
    }

    pub fn build(self) -> NaiveBayes {
        if self.alpha <= 0.0 {
            panic!("Alpha must be positive");
        }
        NaiveBayes {
            alpha: self.alpha,
            class_priors: HashMap::new(),
            feature_log_likelihoods: HashMap::new(),
            vocab_size: 0,
            classes: Vec::new(),
            normalize: self.normalize,
            x_scaler: self.x_scaler,
        }
    }
}

pub struct NaiveBayes {
    alpha: f64,
    class_priors: HashMap<i32, f64>, 
    feature_log_likelihoods: HashMap<i32, Array1<f64>>, 
    vocab_size: usize, 
    classes: Vec<i32>,
    normalize: bool,
    x_scaler: StandardScaler,
}

impl NaiveBayes {
    pub fn fit(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        _learning_rate: f64,
        _epochs: usize,
    ) -> Result<(), ModelError> {


        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            return Err(ModelError::Preprocessing(PreprocessingError::InvalidParameter));
        }

        if x.ncols() == 0 || x.is_empty() || y.is_empty() {
            return Err(ModelError::Preprocessing(if x.ncols() == 0 {
                PreprocessingError::NoFeatures
            } else {
                PreprocessingError::EmptyInput
            }));
        }

        if x.shape()[0] != y.shape()[0] {
            return Err(ModelError::Preprocessing(PreprocessingError::DimensionMismatch {
                expected: x.shape()[0],
                actual: y.shape()[0],
            }));
        }

        if x.iter().any(|&v| v < 0.0) {
            return Err(ModelError::Preprocessing(PreprocessingError::InvalidParameter));
        }

        let x_scaled = if self.normalize {
            match self.x_scaler.fit_transform(x) {
                Ok(scaled) => scaled,
                Err(e) => return Err(ModelError::Preprocessing(e)),
            }
        } else {
            x.clone()
        };

        if x_scaled.ncols() == 0 {
            return Err(ModelError::Preprocessing(PreprocessingError::NoFeatures));
        }

        let unique_classes: Vec<i32> = y
            .iter()
            .map(|&v| v.round() as i32)
            .collect::<std::collections::HashSet<i32>>()
            .into_iter()
            .collect();
        if unique_classes.is_empty() {
            return Err(ModelError::Preprocessing(PreprocessingError::InvalidParameter));
        }

        self.classes = unique_classes;
        self.vocab_size = x.ncols();

        let n_samples = x.shape()[0] as f64;

        for &class in &self.classes {
            let class_count = y.iter().filter(|&&v| (v.round() as i32) == class).count() as f64;
            self.class_priors.insert(class, (class_count / n_samples).ln());
        }

        for &class in &self.classes {
            let class_indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|(_, &v)| (v.round() as i32) == class)
                .map(|(i, _)| i)
                .collect();
            let class_x = x_scaled.select(Axis(0), &class_indices);

            if class_x.ncols() == 0 {
                return Err(ModelError::Preprocessing(PreprocessingError::NoFeatures));
            }

            let raw_counts = class_x.sum_axis(Axis(0));
            let feature_counts: Array1<f64> = &raw_counts + self.alpha;
            let total_count = raw_counts.sum() + self.alpha * self.vocab_size as f64;


            let log_likelihoods = feature_counts.mapv(|v| (v / total_count).ln());
            self.feature_log_likelihoods.insert(class, log_likelihoods);
        }

        Ok(())
    }

    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, ModelError> {
        if self.classes.is_empty() {
            return Err(ModelError::Preprocessing(PreprocessingError::NotFitted));
        }
        if x.is_empty() {
            return Err(ModelError::Preprocessing(PreprocessingError::EmptyInput));
        }
        if x.ncols() != self.vocab_size {
            return Err(ModelError::Preprocessing(PreprocessingError::DimensionMismatch {
                expected: self.vocab_size,
                actual: x.ncols(),
            }));
        }
        if x.iter().any(|&v| v < 0.0) {
            return Err(ModelError::Preprocessing(PreprocessingError::InvalidParameter));
        }

        let x_scaled = if self.normalize {
            self.x_scaler.transform(x).map_err(ModelError::Preprocessing)?
        } else {
            x.clone()
        };

        let mut predictions = Array1::zeros(x_scaled.nrows());

        for (i, row) in x_scaled.axis_iter(Axis(0)).enumerate() {
            let mut max_log_prob = f64::NEG_INFINITY;
            let mut predicted_class = self.classes[0];

            for &class in &self.classes {
                let prior = self
                    .class_priors
                    .get(&class)
                    .ok_or(ModelError::Preprocessing(PreprocessingError::InvalidParameter))?;
                let likelihoods = self
                    .feature_log_likelihoods
                    .get(&class)
                    .ok_or(ModelError::Preprocessing(PreprocessingError::InvalidParameter))?;
                let log_prob = prior + row.dot(likelihoods);

                if log_prob > max_log_prob {
                    max_log_prob = log_prob;
                    predicted_class = class;
                }
            }

            predictions[i] = predicted_class as f64;
        }

        Ok(predictions)
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
        assert!(matches!(result, Err(ModelError::Preprocessing(PreprocessingError::EmptyInput))));
    }

    #[test]
    fn linear_regression_fit_no_features() {
        let mut model = LinearRegression::new().build();
        let x: Array2<f64> = Array2::zeros((2, 0));
        let y = array![1.0, 2.0];
        let result = model.fit(&x, &y, 0.01, 10);
        assert!(matches!(result, Err(ModelError::Preprocessing(PreprocessingError::EmptyInput))));
    }

    #[test]
    fn linear_regression_fit_dimension_mismatch() {
        let mut model = LinearRegression::new().build();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0, 3.0];
        let result = model.fit(&x, &y, 0.01, 10);
        assert!(matches!(
            result,
            Err(ModelError::Preprocessing(PreprocessingError::DimensionMismatch { expected: 2, actual: 3 }))
        ));
    }

    #[test]
    fn linear_regression_predict_not_fitted() {
        let model = LinearRegression::new().build();
        let x = array![[1.0, 2.0]];
        let result = model.predict(&x);
        assert!(matches!(result, Err(ModelError::Preprocessing(PreprocessingError::NotFitted))));
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
            Err(ModelError::Preprocessing(PreprocessingError::DimensionMismatch { expected: 2, actual: 3 }))
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
        assert!(matches!(result, Err(ModelError::Preprocessing(PreprocessingError::EmptyInput))));
    }

    #[test]
    fn logistic_regression_fit_no_features() {
        let mut model = LogisticRegression::new().build();
        let x: Array2<f64> = Array2::zeros((2, 0));
        let y = array![0.0, 1.0];
        let result = model.fit(&x, &y, 0.01, 10);
        assert!(matches!(result, Err(ModelError::Preprocessing(PreprocessingError::EmptyInput))));
    }

    #[test]
    fn logistic_regression_fit_dimension_mismatch() {
        let mut model = LogisticRegression::new().build();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0, 0.0];
        let result = model.fit(&x, &y, 0.01, 10);
        assert!(matches!(
            result,
            Err(ModelError::Preprocessing(PreprocessingError::DimensionMismatch { expected: 2, actual: 3 }))
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
            Err(ModelError::Preprocessing(PreprocessingError::DimensionMismatch { expected: 1, actual: 0 }))
        ));
    }

    #[test]
    fn logistic_regression_predict_not_fitted() {
        let model = LogisticRegression::new().build();
        let x = array![[1.0, 2.0]];
        let result = model.predict(&x);
        assert!(matches!(result, Err(ModelError::Preprocessing(PreprocessingError::NotFitted))));
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
            Err(ModelError::Preprocessing(PreprocessingError::DimensionMismatch { expected: 2, actual: 3 }))
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

    #[test]
    fn knn_fit_predict() {
        let mut knn = KNN::new().k(3).normalize(false).mode(KNNMode::Regression).build();
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];
        knn.fit(&x, &y).unwrap();
        let x_test = array![[2.5, 3.5]];
        let predictions = knn.predict(&x_test).unwrap();
        assert!((predictions[0] - 2.0).abs() < 1e-3); // Average of y[1,2,3] ≈ (2.0 + 3.0 + 1.0) / 3
    }

    #[test]
    fn knn_invalid_k() {
        let mut knn = KNN::new().k(5).build();
        let x = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![1.0, 2.0];
        let result = knn.fit(&x, &y);
        assert!(matches!(result, Err(ModelError::Preprocessing(PreprocessingError::InvalidParameter))));
    }

    #[test]
    fn knn_empty_input() {
        let mut knn = KNN::new().build();
        let x: Array2<f64> = Array2::zeros((0, 2));
        let y: Array1<f64> = Array1::zeros(0);
        let result = knn.fit(&x, &y);
        assert!(matches!(result, Err(ModelError::Preprocessing(PreprocessingError::EmptyInput))));
    }

    #[test]
    fn knn_no_features() {
        let mut knn = KNN::new().build();
        let x: Array2<f64> = Array2::zeros((2, 0));
        let y = array![1.0, 2.0];
        let result = knn.fit(&x, &y);
        assert!(matches!(result, Err(ModelError::Preprocessing(PreprocessingError::NoFeatures))));
    }

    #[test]
    fn knn_dimension_mismatch() {
        let mut knn = KNN::new().build();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0, 3.0];
        let result = knn.fit(&x, &y);
        assert!(matches!(
            result,
            Err(ModelError::Preprocessing(PreprocessingError::DimensionMismatch { expected: 2, actual: 3 }))
        ));
    }

    #[test]
    fn knn_not_fitted() {
        let knn = KNN::new().build();
        let x = array![[1.0, 2.0]];
        let result = knn.predict(&x);
        assert!(matches!(result, Err(ModelError::Preprocessing(PreprocessingError::NotFitted))));
    }

    #[test]
    fn knn_predict_dimension_mismatch() {
        let mut knn = KNN::new().k(1).build();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];
        knn.fit(&x, &y).unwrap();
        let x_test = array![[1.0, 2.0, 3.0]];
        let result = knn.predict(&x_test);
        assert!(matches!(
            result,
            Err(ModelError::Preprocessing(PreprocessingError::DimensionMismatch { expected: 2, actual: 3 }))
        ));
    }
    
    #[test]
    fn naive_bayes_fit_empty_input() {
        let mut nb = NaiveBayesBuilder::new().build();
        let x: Array2<f64> = Array2::zeros((0, 2));
        let y: Array1<f64> = Array1::zeros(0);
        let result = nb.fit(&x, &y, 0.0, 1);
        assert!(matches!(
            result,
            Err(ModelError::Preprocessing(PreprocessingError::EmptyInput))
        ));
    }
    
    
    #[test]
    fn naive_bayes_fit_no_features() {
        let mut nb = NaiveBayesBuilder::new().build();
        let x: Array2<f64> = Array2::zeros((2, 0));
        let y = array![0.0, 1.0];
        let result = nb.fit(&x, &y, 0.0, 1);
        assert!(matches!(
            result,
            Err(ModelError::Preprocessing(PreprocessingError::NoFeatures))
        ));
    }
    
    #[test]
    fn naive_bayes_fit_dimension_mismatch() {
        let mut nb = NaiveBayesBuilder::new().build();
        let x = array![[1.0, 0.0], [0.0, 2.0]];
        let y = array![0.0, 1.0, 0.0];
        let result = nb.fit(&x, &y, 0.0, 1);
        assert!(matches!(
            result,
            Err(ModelError::Preprocessing(PreprocessingError::DimensionMismatch {
                expected: 2,
                actual: 3
            }))
        ));
    }
    
    #[test]
    fn naive_bayes_fit_invalid_features() {
        let mut nb = NaiveBayesBuilder::new().build();
        let x = array![[1.0, -1.0], [0.0, 2.0]];
        let y = array![0.0, 1.0];
        let result = nb.fit(&x, &y, 0.0, 1);
        assert!(matches!(
            result,
            Err(ModelError::Preprocessing(PreprocessingError::InvalidParameter))
        ));
    }
    
    #[test]
    fn naive_bayes_predict_not_fitted() {
        let nb = NaiveBayesBuilder::new().build();
        let x = array![[1.0, 0.0]];
        let result = nb.predict(&x);
        assert!(matches!(
            result,
            Err(ModelError::Preprocessing(PreprocessingError::NotFitted))
        ));
    }
    
    #[test]
    fn naive_bayes_predict_dimension_mismatch() {
        let mut nb = NaiveBayesBuilder::new().build();
        let x_train = array![[1.0, 0.0], [0.0, 2.0]];
        let y_train = array![0.0, 1.0];
        nb.fit(&x_train, &y_train, 0.0, 1).unwrap();
        let x_test = array![[1.0, 0.0, 0.0]];
        let result = nb.predict(&x_test);
        assert!(matches!(
            result,
            Err(ModelError::Preprocessing(PreprocessingError::DimensionMismatch {
                expected: 2,
                actual: 3
            }))
        ));
    }
    
    #[test]
    fn naive_bayes_fit_predict() {
        let mut nb = NaiveBayesBuilder::new().normalize(false).alpha(1.0).build();
        let x = array![[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [2.0, 0.0]];
        let y = array![0.0, 1.0, 0.0, 2.0];
        nb.fit(&x, &y, 0.0, 1).unwrap();
        let predictions = nb.predict(&x).unwrap();
        assert_eq!(predictions.len(), 4);
        // Check if predictions are valid class labels
        for &pred in predictions.iter() {
            assert!(y.iter().any(|&label| label == pred));
        }
    }
    
    #[test]
    fn naive_bayes_multi_class() {
        let mut nb = NaiveBayesBuilder::new().normalize(false).alpha(1.0).build();
        let x = array![
            [2.0, 0.0, 1.0],
            [0.0, 3.0, 0.0],
            [1.0, 1.0, 2.0],
            [3.0, 0.0, 1.0],
            [0.0, 2.0, 1.0]
        ];
        let y = array![0.0, 1.0, 2.0, 0.0, 1.0];
        nb.fit(&x, &y, 0.0, 1).unwrap();
        let predictions = nb.predict(&x).unwrap();
        assert_eq!(predictions.len(), 5);
        // Check if predictions are valid class labels
        for &pred in predictions.iter() {
            assert!(y.iter().any(|&label| label == pred));
        }
    }
}

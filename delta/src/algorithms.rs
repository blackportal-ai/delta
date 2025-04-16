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

use std::fmt::Debug;
use std::ops::SubAssign;

use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive};

use crate::{
    batch_gradient_descent, logistic_gradient_descent,
    scalers::{Scaler, StandardScaler},
};

use super::losses::Loss;

pub trait Algorithm<T, L>
where
    T: Float,
    L: Loss<T>,
{
    fn new(loss_function: L) -> Self
    where
        Self: Sized;

    fn fit(&mut self, x: &Array2<T>, y: &Array1<T>, learning_rate: T, epochs: usize);

    fn predict(&self, x: &Array2<T>) -> Array1<T>;
}

pub struct LinearRegression<T, L, S>
where
    T: Float,
    L: Loss<T>,
    S: Scaler<T>,
{
    weights: Option<Array1<T>>,
    bias: T,
    loss_function: L,
    normalize: bool,
    x_scaler: S,
    y_scaler: S,
}

impl<T, L, S> LinearRegression<T, L, S>
where
    T: Float + ScalarOperand + Debug + FromPrimitive,
    L: Loss<T>,
    S: Scaler<T>,
{
    pub fn new(loss_function: L, normalize: bool, x_scaler: S, y_scaler: S) -> Self {
        LinearRegression {
            weights: None,
            bias: T::zero(),
            loss_function,
            normalize,
            x_scaler,
            y_scaler,
        }
    }
}

impl<T, L> LinearRegression<T, L, StandardScaler<T>>
where
    T: Float + ScalarOperand + SubAssign + Debug + FromPrimitive,
    L: Loss<T>,
{
    pub fn new_with_defaults(loss_function: L) -> Self {
        Self::new(loss_function, true, StandardScaler::new(), StandardScaler::new())
    }
}

impl<T, L, S> LinearRegression<T, L, S>
where
    T: Float + ScalarOperand,
    L: Loss<T>,
    S: Scaler<T>,
{
    pub fn calculate_loss(&self, predictions: &Array1<T>, actuals: &Array1<T>) -> T {
        self.loss_function.calculate(predictions, actuals)
    }
}

impl<T, L> Algorithm<T, L> for LinearRegression<T, L, StandardScaler<T>>
where
    T: Float + ScalarOperand + SubAssign + Debug + FromPrimitive,
    L: Loss<T>,
{
    fn new(loss_function: L) -> Self {
        LinearRegression::new_with_defaults(loss_function)
    }

    fn fit(&mut self, x: &Array2<T>, y: &Array1<T>, learning_rate: T, epochs: usize) {
        if x.is_empty() || y.is_empty() {
            panic!("Input data cannot be empty");
        }
        if x.shape()[0] != y.shape()[0] {
            panic!("Number of samples in x and y must match");
        }

        let (x_scaled, y_scaled) = if self.normalize {
            let x_scaled = self.x_scaler.fit_transform(x);
            let y_2d = y.clone().insert_axis(Axis(1));
            let y_scaled = self.y_scaler.fit_transform(&y_2d).remove_axis(Axis(1));
            (x_scaled, y_scaled)
        } else {
            (x.clone(), y.clone())
        };

        let (_, n_features) = x_scaled.dim();
        self.weights = Some(Array1::zeros(n_features));

        for _epoch in 0..epochs {
            let predictions = self.predict(&x_scaled);
            let _loss = self.calculate_loss(&predictions, &y_scaled);

            let (grad_weights, grad_bias) = batch_gradient_descent(
                &x_scaled,
                &y_scaled,
                self.weights.as_ref().expect("Weights not initialized"),
                self.bias,
            );

            self.weights = Some(
                self.weights.take().expect("Weights not initialized")
                    - &(grad_weights * learning_rate),
            );
            self.bias -= grad_bias * learning_rate;
        }
    }

    fn predict(&self, x: &Array2<T>) -> Array1<T> {
        let x_scaled = if self.normalize { self.x_scaler.transform(x) } else { x.clone() };
        let predictions_scaled =
            x_scaled.dot(self.weights.as_ref().expect("Weights not initialized")) + self.bias;
        if self.normalize {
            let pred_2d = predictions_scaled.insert_axis(Axis(1));
            self.y_scaler.inverse_transform(&pred_2d).remove_axis(Axis(1))
        } else {
            predictions_scaled
        }
    }
}

pub struct LogisticRegression<T, L, S>
where
    T: Float,
    L: Loss<T>,
    S: Scaler<T>,
{
    weights: Array1<T>,
    bias: T,
    loss_function: L,
    normalize: bool,
    x_scaler: S,
}

impl<T, L, S> LogisticRegression<T, L, S>
where
    T: Float + ScalarOperand + Debug + FromPrimitive,
    L: Loss<T>,
    S: Scaler<T>,
{
    pub fn new(loss_function: L, normalize: bool, x_scaler: S) -> Self {
        LogisticRegression {
            weights: Array1::zeros(0),
            bias: T::zero(),
            loss_function,
            normalize,
            x_scaler,
        }
    }
}

impl<T, L> LogisticRegression<T, L, StandardScaler<T>>
where
    T: Float + ScalarOperand + SubAssign + Debug + FromPrimitive,
    L: Loss<T>,
{
    pub fn new_with_defaults(loss_function: L) -> Self {
        Self::new(loss_function, true, StandardScaler::new())
    }
}

impl<T, L, S> LogisticRegression<T, L, S>
where
    T: Float + ScalarOperand,
    L: Loss<T>,
    S: Scaler<T>,
{
    pub fn calculate_loss(&self, predictions: &Array1<T>, actuals: &Array1<T>) -> T {
        self.loss_function.calculate(predictions, actuals)
    }

    fn sigmoid(&self, linear_output: Array1<T>) -> Array1<T> {
        linear_output.mapv(|x| T::one() / (T::one() + (-x).exp()))
    }

    fn predict_linear(&self, x: &Array2<T>) -> Array1<T> {
        x.dot(&self.weights) + self.bias
    }

    pub fn calculate_accuracy(&self, predictions: &Array1<T>, actuals: &Array1<T>) -> f64
    where
        T: Float,
    {
        let binary_predictions =
            predictions.mapv(|x| if x >= T::from(0.5).unwrap() { T::one() } else { T::zero() });
        let matches = binary_predictions
            .iter()
            .zip(actuals.iter())
            .filter(|(pred, actual)| (**pred - **actual).abs() < T::epsilon())
            .count();
        matches as f64 / actuals.len() as f64
    }
}

impl<T, L> Algorithm<T, L> for LogisticRegression<T, L, StandardScaler<T>>
where
    T: Float + ScalarOperand + SubAssign + Debug + FromPrimitive,
    L: Loss<T>,
{
    fn new(loss_function: L) -> Self {
        LogisticRegression::new_with_defaults(loss_function)
    }

    fn fit(&mut self, x: &Array2<T>, y: &Array1<T>, learning_rate: T, epochs: usize) {
        if x.is_empty() || y.is_empty() {
            panic!("Input data cannot be empty");
        }
        if x.shape()[0] != y.shape()[0] {
            panic!("Number of samples in x and y must match");
        }
        if y.iter().any(|&v| v < T::zero() || v > T::one()) {
            panic!("LogisticRegression expects binary labels (0 or 1)");
        }

        let x_scaled = if self.normalize { self.x_scaler.fit_transform(x) } else { x.clone() };
        if self.weights.len() == 0 {
            self.weights = Array1::zeros(x_scaled.shape()[1]);
        }

        for _ in 0..epochs {
            let linear_output = self.predict_linear(&x_scaled);
            let _predictions = self.sigmoid(linear_output);
            let (grad_weights, grad_bias) =
                logistic_gradient_descent(&x_scaled, y, &self.weights, self.bias);

            self.weights -= &(grad_weights * learning_rate);
            self.bias -= grad_bias * learning_rate;
        }
    }

    fn predict(&self, x: &Array2<T>) -> Array1<T> {
        let x_scaled = if self.normalize { self.x_scaler.transform(x) } else { x.clone() };
        let linear_output = self.predict_linear(&x_scaled);
        self.sigmoid(linear_output)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};
    use num_traits::Float;

    use super::{LinearRegression, LogisticRegression};
    use crate::{
        algorithms::Algorithm,
        losses::{CrossEntropy, MSE},
    };

    #[test]
    fn test_linear_regression_fit_predict() {
        let x_data = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y_data = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0]);

        let mut model = LinearRegression::new_with_defaults(MSE);

        let learning_rate = 0.1;
        let epochs = 1000;
        model.fit(&x_data, &y_data, learning_rate, epochs);

        let new_data = Array2::from_shape_vec((2, 1), vec![5.0, 6.0]).unwrap();
        let predictions = model.predict(&new_data);

        assert!((predictions[0] - 10.0).abs() < 0.1);
        assert!((predictions[1] - 12.0).abs() < 0.1);
    }

    #[test]
    fn test_linear_regression_calculate_loss() {
        let predictions = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0]);
        let actuals = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0]);

        let model = LinearRegression::new_with_defaults(MSE);

        let loss = model.calculate_loss(&predictions, &actuals);
        assert!(loss.abs() < 1e-6, "Loss should be close to 0, got: {}", loss);
    }

    #[test]
    fn test_linear_regression_multi_feature() {
        let x_data =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.5, 2.0, 1.0, 3.0, 1.5, 4.0, 2.0]).unwrap();
        let y_data = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0]);
        let mut model = LinearRegression::new_with_defaults(MSE);
        model.fit(&x_data, &y_data, 0.1, 100);
        let predictions = model.predict(&x_data);
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    #[should_panic(expected = "Number of samples in x and y must match")]
    fn test_linear_regression_invalid_shapes() {
        let x_data = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y_data = Array1::from_vec(vec![2.0, 4.0, 6.0]);
        let mut model = LinearRegression::new_with_defaults(MSE);
        model.fit(&x_data, &y_data, 0.1, 10);
    }

    #[test]
    fn test_logistic_regression_fit_predict() {
        let x_data = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y_data = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut model = LogisticRegression::new_with_defaults(CrossEntropy);

        let learning_rate = 0.1;
        let epochs = 1000;
        model.fit(&x_data, &y_data, learning_rate, epochs);

        let new_data = Array2::from_shape_vec((2, 1), vec![1.5, 3.5]).unwrap();
        let predictions = model.predict(&new_data);

        assert!(predictions[0] >= 0.0 && predictions[0] <= 1.0);
        assert!(predictions[1] >= 0.0 && predictions[1] <= 1.0);
    }

    #[test]
    fn test_logistic_regression_calculate_loss() {
        let predictions = Array1::from_vec(vec![0.1, 0.2, 0.7, 0.9]);
        let actuals = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let model = LogisticRegression::new_with_defaults(CrossEntropy);

        let loss = model.calculate_loss(&predictions, &actuals);
        assert!(loss > 0.0, "Loss should be positive, got: {}", loss);
    }

    #[test]
    fn test_logistic_regression_calculate_accuracy() {
        let predictions = Array1::from_vec(vec![0.1, 0.8, 0.3, 0.7]);
        let actuals = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]);

        let model = LogisticRegression::new_with_defaults(CrossEntropy);

        let accuracy = model.calculate_accuracy(&predictions, &actuals);
        assert!((accuracy - 0.5).abs() < 1e-6, "Accuracy should be 0.5, got: {}", accuracy);
    }

    #[test]
    fn test_logistic_regression_multi_feature() {
        let x_data =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.5, 2.0, 1.0, 3.0, 1.5, 4.0, 2.0]).unwrap();
        let y_data = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let mut model = LogisticRegression::new_with_defaults(CrossEntropy);
        model.fit(&x_data, &y_data, 0.1, 100);
        let predictions = model.predict(&x_data);
        assert!(predictions.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    #[should_panic(expected = "LogisticRegression expects binary labels")]
    fn test_logistic_regression_invalid_labels() {
        let x_data = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y_data = Array1::from_vec(vec![0.0, 2.0, 1.0, 3.0]);
        let mut model = LogisticRegression::new_with_defaults(CrossEntropy);
        model.fit(&x_data, &y_data, 0.1, 10);
    }
}

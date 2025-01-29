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
use std::{collections::HashSet, ops::SubAssign};
use std::collections::HashMap;

use libm::log2;
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::{Float, FromPrimitive};

use super::{Algorithm, batch_gradient_descent, logistic_gradient_descent, losses::Loss};

/// A struct for performing linear regression.
///
/// # Generics
/// - `T`: The type of data, must implement `num_traits::Float` and `ndarray::ScalarOperand`.
/// - `L`: The type of the loss function, must implement `Loss<T>`.
pub struct LinearRegression<T, L>
where
    T: Float,
    L: Loss<T>,
{
    weights: Array1<T>,
    bias: T,
    loss_function: L,
}

impl<T, L> LinearRegression<T, L>
where
    T: Float + ScalarOperand,
    L: Loss<T>,
{
    /// Calculates the loss between predictions and actual values.
    ///
    /// # Arguments
    /// - `predictions`: Predicted values as a 1D array.
    /// - `actuals`: Actual values as a 1D array.
    ///
    /// # Returns
    /// The calculated loss as a value of type `T`.
    pub fn calculate_loss(&self, predictions: &Array1<T>, actuals: &Array1<T>) -> T {
        self.loss_function.calculate(predictions, actuals)
    }
}

impl<T, L> Algorithm<T, L> for LinearRegression<T, L>
where
    T: Float + ScalarOperand + SubAssign,
    L: Loss<T>,
{
    /// Creates a new `LinearRegression` instance with the given loss function and optimizer.
    ///
    /// # Arguments
    /// - `loss_function`: The loss function to use.
    /// - `optimizer`: The optimizer to use.
    ///
    /// # Returns
    /// A new instance of `LinearRegression`.
    fn new(loss_function: L) -> Self {
        LinearRegression { weights: Array1::zeros(1), bias: T::zero(), loss_function }
    }

    /// Fits the model to the given data using batch gradient descent.
    ///
    /// # Arguments
    /// - `x`: The input features as a 2D array.
    /// - `y`: The target values as a 1D array.
    /// - `learning_rate`: The learning rate for gradient descent.
    /// - `epochs`: The number of iterations for gradient descent.
    fn fit(&mut self, x: &Array2<T>, y: &Array1<T>, learning_rate: T, epochs: usize) {
        for _ in 0..epochs {
            let predictions = self.predict(x);
            let _loss = self.calculate_loss(&predictions, y);

            let (grad_weights, grad_bias) = batch_gradient_descent(x, y, &self.weights, self.bias);

            self.weights -= &(grad_weights * learning_rate);
            self.bias -= grad_bias * learning_rate;
        }
    }

    /// Predicts target values for the given input features.
    ///
    /// # Arguments
    /// - `x`: The input features as a 2D array.
    ///
    /// # Returns
    /// Predicted values as a 1D array.
    fn predict(&self, x: &Array2<T>) -> Array1<T> {
        x.dot(&self.weights) + self.bias
    }
}

/// A struct for performing logistic regression.
///
/// # Generics
/// - `T`: The type of data, must implement `num_traits::Float` and `ndarray::ScalarOperand`.
/// - `L`: The type of the loss function, must implement `Loss<T>`.
pub struct LogisticRegression<T, L>
where
    T: Float,
    L: Loss<T>,
{
    weights: Array1<T>,
    bias: T,
    loss_function: L,
}

impl<T, L> LogisticRegression<T, L>
where
    T: Float + ScalarOperand,
    L: Loss<T>,
{
    /// Calculates the loss between predictions and actual values.
    ///
    /// # Arguments
    /// - `predictions`: Predicted probabilities as a 1D array.
    /// - `actuals`: Actual values as a 1D array.
    ///
    /// # Returns
    /// The calculated loss as a value of type `T`.
    pub fn calculate_loss(&self, predictions: &Array1<T>, actuals: &Array1<T>) -> T {
        self.loss_function.calculate(predictions, actuals)
    }

    // TODO: we should create generics for Activation
    // /// Applies the sigmoid function to the given linear output.
    ///
    /// # Arguments
    /// - `linear_output`: A 1D array of linear outputs.
    ///
    /// # Returns
    /// A 1D array of sigmoid-transformed values.
    fn sigmoid(&self, linear_output: Array1<T>) -> Array1<T> {
        linear_output.mapv(|x| T::one() / (T::one() + (-x).exp()))
    }

    /// Predicts the output of a logistic regression model.
    ///
    /// This function performs the linear regression prediction by calculating the dot product
    /// of the input features `x` and the model weights, and then adding the bias term.
    ///
    /// # Parameters
    /// - `x`: A 2D array of input features (`Array2<T>`). Each row represents a feature vector for a single sample.
    ///
    /// # Returns
    /// - An `Array1<T>` containing the predicted values for each sample.
    fn predict_linear(&self, x: &Array2<T>) -> Array1<T> {
        x.dot(&self.weights) + self.bias
    }

    /// Calculates the accuracy of a binary classification model.
    ///
    /// This function compares the predicted values with the actual values, considering a threshold of 0.5
    /// to determine binary classifications. It returns the proportion of correct predictions, where predictions
    /// that differ from the actual values by less than `T::epsilon()` are considered correct.
    ///
    /// # Parameters
    /// - `predictions`: A 1D array (`Array1<T>`) containing the model's predicted values.
    /// - `actuals`: A 1D array (`Array1<T>`) containing the actual ground truth values.
    ///
    /// # Returns
    /// - A `f64` representing the accuracy, calculated as the ratio of correct predictions to total samples.
    ///
    /// # Constraints
    /// - `T` must implement `num_traits::Float` so that numerical operations like comparisons and arithmetic can be performed.
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

impl<T, L> Algorithm<T, L> for LogisticRegression<T, L>
where
    T: Float + ScalarOperand + SubAssign,
    L: Loss<T>,
{
    /// Creates a new `LogisticRegression` model.
    ///
    /// This constructor initializes a logistic regression model with a given loss function.
    /// The weights are initialized to zeros, and the bias is set to `T::zero()`.
    ///
    /// # Parameters
    /// - `loss_function`: The loss function to use for model training.
    ///
    /// # Returns
    /// - A new `LogisticRegression` instance with initialized weights and bias.
    fn new(loss_function: L) -> Self {
        LogisticRegression { weights: Array1::zeros(1), bias: T::zero(), loss_function }
    }

    /// Fits the logistic regression model to the training data using gradient descent.
    ///
    /// This function trains the model by iteratively updating the weights and bias to minimize
    /// the loss function. It performs a specified number of epochs of gradient descent with
    /// a given learning rate.
    ///
    /// # Parameters
    /// - `x`: A 2D array of input features (`Array2<T>`), where each row represents a sample.
    /// - `y`: A 1D array of target labels (`Array1<T>`) corresponding to the input samples.
    /// - `learning_rate`: The learning rate used in gradient descent.
    /// - `epochs`: The number of iterations (epochs) to run the gradient descent.
    fn fit(&mut self, x: &Array2<T>, y: &Array1<T>, learning_rate: T, epochs: usize) {
        for _ in 0..epochs {
            let linear_output = self.predict_linear(x);
            let predictions = self.sigmoid(linear_output.clone());
            let _loss = self.calculate_loss(&predictions, y);

            let _errors = &predictions - y;

            let (grad_weights, grad_bias) =
                logistic_gradient_descent(x, y, &self.weights, self.bias);

            self.weights -= &(grad_weights * learning_rate);
            self.bias -= grad_bias * learning_rate;
        }
    }

    /// Makes predictions using the logistic regression model.
    ///
    /// This function first calculates the linear output by applying the learned weights and bias,
    /// and then applies the sigmoid function to obtain the predicted probabilities.
    ///
    /// # Parameters
    /// - `x`: A 2D array of input features (`Array2<T>`), where each row represents a sample.
    ///
    /// # Returns
    /// - A 1D array (`Array1<T>`) containing the predicted probabilities for each sample.
    fn predict(&self, x: &Array2<T>) -> Array1<T> {
        let linear_output = self.predict_linear(x);
        self.sigmoid(linear_output)
    }
}

/// TreeNode can be a `Internal` or `Leaf` at any given moment
/// 

#[derive(Debug)]
enum TreeNode<T, P> {
    Internal { 
        feature: Option<usize>,
        threshold: Option<T>,
        left: Option<Box<TreeNode<T, P>>>,
        right: Option<Box<TreeNode<T, P>>> 
    },
    Leaf { 
        prediction: P,
        indices: Array1<usize>
    },
} 
impl<T, P> TreeNode<T, P> {
    fn new_empty() -> Box<TreeNode<T, P>>{
        Box::new(TreeNode::Internal { 
            feature: None,
            threshold: None,
            left: None,
            right: None 
        })
    }
}

/// A struct for performing Decision Tree classification.
///
/// # Generics
/// - `T`: The type of data, must implement `num_traits::Float` and `ndarray::ScalarOperand`.
/// - `L`: The type of the loss function, must implement `Loss<T>`.
pub struct DecisionTreeClassifier<T, L>
where
    T: Float+ Debug,
    L: Loss<T>,
{
    max_depth: usize,
    min_loss: f64,
    _loss_function: L,
    data_x: Option<Array2<T>>,
    data_y: Option<Array1<usize>>,
    root: Option<TreeNode<T, usize>>,
}

impl<T, L> DecisionTreeClassifier<T, L>
where
    T: Float + FromPrimitive + Debug,
    L: Loss<T>,
{
    /// Creates a new `DecisionTree` instance.
    ///
    /// # Arguments
    /// - `max_depth`: The maximum depth of the tree.
    /// - `loss_function`: The loss function to use.
    ///S
    /// # Returns
    /// A new instance of `DecisionTree`.
    pub fn new(max_depth: usize, min_loss: f64, loss_function: L) -> Self {
        DecisionTreeClassifier { 
            max_depth, 
            min_loss,
            _loss_function: loss_function, 
            data_x: None,
            data_y: None,
            root: None 
        }
    }

    /// Recursively splits the data based on the best feature and threshold.
    fn build_tree(&mut self, node: &mut TreeNode<T, usize>, indices: Array1<usize>, depth: usize) {
        // println!("TOP: depth: {}, node {:?}", depth, node);
        // println!("indices len : {}", indices.len());
        if depth >= self.max_depth || indices.shape()[0] <= 1 {
            
            let prediction = self.calculate_leaf_prediction(&indices).unwrap();
            
            // Update this node to Leaf Node
            *node = TreeNode::Leaf { prediction, indices};
            // println!("BOTTOM: depth: {}, node {:?}", depth, node);
            return;
        }

        // Check If Pure If yes then assign this node as leaf 
        let data_y_ref = self.data_y.as_ref().unwrap();
        let classes: HashSet<_> = indices.iter()
            .map(|&idx| data_y_ref[idx])
            .collect();
        let mut class_counts: HashMap<usize, usize> = classes
            .iter()
            .map(|class| (*class, 0))
            .collect();
        for &idx in indices.iter() {
            let class = data_y_ref[idx];
            *class_counts.get_mut(&class).unwrap() += 1;
        }
        let loss = Self::calculate_entropy(&class_counts);

        if loss <= self.min_loss {
            let prediction = self.calculate_leaf_prediction(&indices).unwrap();
            
            // Update this node to Leaf Node
            *node = TreeNode::Leaf { prediction, indices};
            // println!("BOTTOM: depth: {}, node {:?}", depth, node);
            return;
        }
        
        
        // Main Decision Tree Algorithm 
        let (best_feature, best_threshold) = self.find_best_split(&indices);
        let (index_left, index_right) = self.split_data(indices, best_feature, best_threshold);
        
        // Initialize the Left and Right Nodes
        let mut left_node = TreeNode::new_empty();
        let mut right_node = TreeNode::new_empty();
        
        // Recursively build left and right subtrees
        self.build_tree(&mut left_node, index_left, depth + 1);
        self.build_tree(&mut right_node, index_right, depth + 1);

        // Update the current Node
        *node = TreeNode::Internal { 
            feature: Some(best_feature),
            threshold: Some(best_threshold),
            left: Some(left_node),
            right: Some(right_node)
        };
        
        // println!("BOTTOM: depth: {}, node {:?}", depth, node);
    }
    
    fn calculate_leaf_prediction(&self, indices: &Array1<usize>) -> Option<usize> {
        let mut frequency_map = HashMap::new();
        for &val in indices {
            *frequency_map.entry(self.data_y.as_ref().unwrap()[val]).or_insert(0) += 1;
        }
        frequency_map.into_iter().max_by_key(|&(_, count)| count).map(|(value, _)| value)
    }

    fn find_best_split(&self, indices: &Array1<usize>) -> (usize, T) {
        
        let (num_rows, num_features) = (indices.shape()[0], self.data_x.as_ref().unwrap().shape()[1]);

        
        let mut best_loss = f64::MAX;// Replace this with the loss of parent node
        let mut best_threshold_idx = 0_usize;
        let mut best_feature_idx = 0_usize;
        
        let data_y_ref = self.data_y.as_ref().unwrap();
        let data_x_ref = self.data_x.as_ref().unwrap();
        
        for feature in 0..num_features {
            
            // Create a 2D array with index and feature values
            let mut feature_values: Vec<(usize, T)> = indices
            .iter()
                .map(|&idx| (idx, self.data_x.as_ref().unwrap()[[idx, feature]]))
                .collect();

            // Sort by feature value
            feature_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Extract sorted indices
            let sorted_indices: Vec<usize> = feature_values.iter().map(|&(idx, _)| idx).collect();

            // Initialize a hashmap to store class distributions
            let subset_classes: HashSet<_> = sorted_indices.iter()
                .map(|&idx| data_y_ref[idx])
                .collect();
            
            let mut class_counts_left: HashMap<usize, usize> = subset_classes
                .iter()
                .map(|class| (*class, 0))
                .collect();

            let mut class_counts_right: HashMap<usize, usize> = subset_classes
                .iter()
                .map(|class| (*class, 0))
                .collect();

            for &idx in sorted_indices.iter() {
                let class = data_y_ref[idx];
                *class_counts_right.get_mut(&class).unwrap() += 1;
            }
            
            // Find the best Threshold among all the threshold
            for (idx, &threshold_idx) in sorted_indices.iter().enumerate() {

                // Left sunbet have < and right have >=
                let class = data_y_ref[threshold_idx];
                if idx == 0 {
                    *class_counts_left.get_mut(&class).unwrap() += 1_usize;
                    *class_counts_right.get_mut(&class).unwrap() -= 1_usize;
                    continue;
                }
                // Find the Entropy for current Threshold
                let loss_left = Self::calculate_entropy(&class_counts_left);
                let loss_right = Self::calculate_entropy(&class_counts_right);

                // Finding overall loss of split
                let total_loss = loss_left * ((idx) as f64)/(num_rows as f64) + 
                                    loss_right * ((num_rows-idx) as f64)/(num_rows as f64);
                
                // Upating the best loss and threshold
                if total_loss < best_loss {
                    best_loss = total_loss;
                    best_feature_idx = feature;
                    best_threshold_idx = threshold_idx;
                }
                
                // Update Class Counts
                *class_counts_left.get_mut(&class).unwrap() += 1_usize;
                *class_counts_right.get_mut(&class).unwrap() -= 1_usize;

            }
        }
        (best_feature_idx, data_x_ref[[best_threshold_idx, best_feature_idx]])
    }
    
    fn calculate_entropy(class_counts: &HashMap<usize, usize>) -> f64 {
        let subset_size= class_counts.values().sum::<usize>() as f64;
        let entropy = class_counts
            .into_iter()
            .map(|(_, &count)|{
                if count == 0_usize {
                    0 as f64
                }
                else {
                    let p = (count as f64) / subset_size;
                    (-p) * log2(p)
                }
            }).sum::<f64>();     

        entropy
    }


    /// Splits the data given threshold and feature
    /// 
    /// `Moves the data and returns the splitted data.`
    fn split_data(
        &self,
        indices: Array1<usize>,
        feature: usize,
        threshold: T,
    ) -> (Array1<usize>, Array1<usize>) {
        
        let data_x_ref = self.data_x.as_ref().unwrap();
        let (left_indices, right_indices): (Vec<_>, Vec<_>) = indices
            .iter()
            .partition(|&&idx| data_x_ref[[idx, feature]] < threshold);
        // println!("Left-Indices Len: {}, Right-Indices Len: {}", left_indices.len(), right_indices.len());
        (Array1::from(left_indices), Array1::from(right_indices))
    }
}

impl<T, L> Algorithm<T, L> for DecisionTreeClassifier<T, L>
where
    T: Float + FromPrimitive + Debug,
    L: Loss<T>,
{
    fn new(loss_function: L) -> Self {
        DecisionTreeClassifier { 
            max_depth: 10,
            min_loss: 1e-6,
            _loss_function: loss_function,
            data_x: None,
            data_y: None, 
            root: None 
        }
    }

    fn fit(&mut self, x: &Array2<T>, y: &Array1<T>, _learning_rate: T, _epochs: usize) {
        self.data_x = Some(x.clone());
        // Type cast classes to usize
        self.data_y = Some(y.mapv(|x| x.to_usize().unwrap()));

        let mut root = TreeNode::new_empty();
        let index = Array1::from_iter(0..x.shape()[0]);
        self.build_tree(&mut root, index, 0);
        self.root = Some(*root);
    }

    fn predict(&self, x: &Array2<T>) -> Array1<T> {
        // Implement the prediction logic by traversing the tree

        let mut predictions = Vec::with_capacity(x.shape()[0]);

        if let Some(root) = &self.root {
            for i in 0..x.shape()[0] {
                let mut current = root;
                loop {
                    match current {
                        TreeNode::Leaf { prediction, .. } => {
                            predictions.push(*prediction);
                            break;
                        }
                        TreeNode::Internal { feature, threshold, left, right, .. } => {
                            let value = x[[i, feature.unwrap()]];
                            current = if value < threshold.unwrap() {
                                left.as_ref().unwrap()
                            } else {
                                right.as_ref().unwrap()
                            };
                        }
                    }
                }
            }
        }
        Array1::from_iter(predictions.into_iter().map(|x| T::from_usize(x).unwrap()))

    }
}



#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};
    use num_traits::Float;

    use crate::classical_ml::{
        Algorithm,
        algorithms::{LinearRegression, LogisticRegression},
        losses::{CrossEntropy, MSE},
    };

    use super::DecisionTreeClassifier;

    #[test]
    fn test_linear_regression_fit_predict() {
        let x_data = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y_data = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0]);

        let mut model = LinearRegression::new(MSE);

        let learning_rate = 0.1;
        let epochs = 1000;
        model.fit(&x_data, &y_data, learning_rate, epochs);

        let new_data = Array2::from_shape_vec((2, 1), vec![5.0, 6.0]).unwrap();
        let predictions = model.predict(&new_data);

        assert!((predictions[0] - 10.0).abs() < 1e-2);
        assert!((predictions[1] - 12.0).abs() < 1e-2);
    }

    #[test]
    fn test_linear_regression_calculate_loss() {
        let predictions = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0]);
        let actuals = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0]);

        let model = LinearRegression::new(MSE);

        let loss = model.calculate_loss(&predictions, &actuals);
        assert!(loss.abs() < 1e-6, "Loss should be close to 0, got: {}", loss);
    }

    #[test]
    fn test_logistic_regression_fit_predict() {
        let x_data = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y_data = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut model = LogisticRegression::new(CrossEntropy);

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

        let model = LogisticRegression::new(CrossEntropy);

        let loss = model.calculate_loss(&predictions, &actuals);
        assert!(loss > 0.0, "Loss should be positive, got: {}", loss);
    }

    #[test]
    fn test_logistic_regression_calculate_accuracy() {
        let predictions = Array1::from_vec(vec![0.1, 0.8, 0.3, 0.7]);
        let actuals = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]);
        
        let model = LogisticRegression::new(CrossEntropy);
        
        let accuracy = model.calculate_accuracy(&predictions, &actuals);
        assert!((accuracy - 0.5).abs() < 1e-6, "Accuracy should be 0.5, got: {}", accuracy);
    }

    #[test]
    fn test_iris_data_loading() {
        let (train, _) = linfa_datasets::iris()
                                .split_with_ratio(0.8);

        println!("X =  {:?}", train.records());
        println!("y =  {:?}", train.targets());
        
    }
    #[test]
    fn test_decision_tree_fit_predict() {
        let (train, test) = linfa_datasets::iris()
            .split_with_ratio(0.8);

        // Convert train data to ndarray format
        let x_train = Array2::from_shape_vec(
            (train.records().nrows(), train.records().ncols()),
            train.records().to_owned().into_raw_vec()
        ).unwrap();

        let y_train = Array1::from_shape_vec(
            train.targets().len(),
            train.targets().iter().map(|&x| x as f64).collect()
        ).unwrap();

        // Convert test data to ndarray format
        let x_test = Array2::from_shape_vec(
            (test.records().nrows(), test.records().ncols()),
            test.records().to_owned().into_raw_vec()
        ).unwrap();

        let y_test = Array1::from_shape_vec(
            test.targets().len(),
            test.targets().iter().map(|&x| x as f64).collect()
        ).unwrap();

        let mut model = DecisionTreeClassifier::new(10, 1e-6, MSE);
        model.fit(&x_train, &y_train, 0.1, 100);

        let predictions = model.predict(&x_test);
        

        // Calculate and print the accuracy
        let correct_predictions = predictions.iter()
            .zip(y_test.iter())
            .filter(|(&pred, &actual)| (pred - actual).abs() < 1e-6)
            .count();
        let accuracy = correct_predictions as f64 / y_test.len() as f64;
        println!("Test accuracy: {:.2}%", accuracy * 100.0);

    }
}

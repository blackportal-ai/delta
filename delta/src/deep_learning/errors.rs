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

use std::fmt;

/// A type alias for results returned by core operations.
pub type Result<T> = std::result::Result<T, CoreError>;

/// An enumeration of possible core errors.
#[derive(Debug)]
pub enum CoreError {
    /// Indicates an invalid shape error.
    InvalidShape,
    /// Indicates a gradient mismatch error.
    GradientMismatch,
    /// Represents other types of errors with a message.
    Other(String),
}

/// Errors that can occur in the Optimizer.
#[derive(Debug)]
pub enum OptimizerError {
    /// Error when learning rate is not set or is invalid.
    InvalidLearningRate(String),
    /// Error when the gradient and weight shapes are incompatible.
    IncompatibleGradientWeightShape(Vec<usize>, Vec<usize>),
    /// Error when epsilon is not set or is invalid.
    InvalidEpsilon(String),
    /// Error when beta parameter is invalid.
    InvalidBeta(String),
    /// Error when weight decay parameter is invalid.
    InvalidWeightDecay(String),
    /// Error when gradient contains invalid values (NaN or Inf).
    InvalidGradient(String),
    /// Error when weight contains invalid values (NaN or Inf).
    InvalidWeight(String),
    /// Error when shapes don't match
    ShapeMismatch(String),
}

/// An enumeration of possible errors that can occur in a model.
#[derive(Debug)]
pub enum ModelError {
    /// Error indicating that the optimizer is missing.
    MissingOptimizer,
    /// Error indicating that the loss function is missing.
    MissingLossFunction,
    /// Error related to the dataset, with a message describing the issue.
    DatasetError(String),
    /// Error that occurs during training, with a message describing the issue.
    TrainingError(String),
    /// Error related to a specific layer in the model.
    LayerError(LayerError),
    /// Error related to a device, with a message describing the issue.
    DeviceError(String),
}

/// Errors that can occur in the Dense layer.
#[derive(Debug)]
pub enum LayerError {
    /// Error when weights are not initialized.
    UninitializedWeights,
    /// Error when bias is not initialized.
    UninitializedBias,
    /// Error when input is not set for backward pass.
    UninitializedInput,
    /// Error when input is not set.
    MissingInput,
    /// Error when the input shape is invalid.
    InvalidInputShape,
    /// Error when an optimizer error occurs.
    OptimizerError(OptimizerError),
}

impl fmt::Display for OptimizerError {
    /// Formats the `OptimizerError` for display purposes.
    ///
    /// # Arguments
    ///
    /// * `f` - The formatter.
    ///
    /// # Returns
    ///
    /// A `fmt::Result` indicating the success or failure of the formatting operation.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizerError::InvalidLearningRate(s) => write!(f, "{}", s),
            OptimizerError::IncompatibleGradientWeightShape(g, w) => {
                write!(f, "Gradient shape {:?} is incompatible with weight shape {:?}", g, w)
            }
            OptimizerError::InvalidEpsilon(s) => write!(f, "{}", s),
            OptimizerError::InvalidBeta(s) => write!(f, "{}", s),
            OptimizerError::InvalidWeightDecay(s) => write!(f, "{}", s),
            OptimizerError::InvalidGradient(s) => write!(f, "{}", s),
            OptimizerError::InvalidWeight(s) => write!(f, "{}", s),
            OptimizerError::ShapeMismatch(s) => write!(f, "Shape mismatch: {}", s),
        }
    }
}

impl fmt::Display for ModelError {
    /// Formats the `ModelError` for display purposes.
    ///
    /// # Arguments
    ///
    /// * `f` - The formatter.
    ///
    /// # Returns
    ///
    /// A `fmt::Result` indicating the success or failure of the formatting operation.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::MissingOptimizer => write!(f, "Optimizer must be set before training"),
            ModelError::MissingLossFunction => {
                write!(f, "Loss function must be set before training")
            }
            ModelError::DatasetError(msg) => write!(f, "Dataset error: {}", msg),
            ModelError::TrainingError(msg) => write!(f, "Training error: {}", msg),
            ModelError::LayerError(err) => write!(f, "Layer error: {}", err),
            ModelError::DeviceError(msg) => write!(f, "Device error: {}", msg),
        }
    }
}

impl fmt::Display for LayerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerError::UninitializedWeights => write!(f, "Weights must be initialized"),
            LayerError::UninitializedBias => write!(f, "Bias must be initialized"),
            LayerError::UninitializedInput => write!(f, "Input must be initialized"),
            LayerError::MissingInput => write!(f, "Input must be set"),
            LayerError::OptimizerError(err) => write!(f, "Optimizer error: {}", err),
            LayerError::InvalidInputShape => write!(f, "Invalid input shape"),
        }
    }
}

impl std::error::Error for LayerError {}
impl std::error::Error for OptimizerError {}
impl std::error::Error for ModelError {}

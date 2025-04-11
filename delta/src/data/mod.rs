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

pub mod csv;
pub mod error;

pub use csv::{CsvHeadersLoader, CsvLoader};

use ndarray::{Array1, Array2};

/// A trait for loading data from files into feature matrices and target vectors.
///
/// This trait defines a generic interface for loading data from various file formats
/// into a standardized format suitable for machine learning tasks. Implementors must
/// provide a `load` method that reads a file from a given path and returns a 2D feature
/// array (`Array2<f64>`) and a 1D target array (`Array1<f64>`). The trait is generic
/// over an associated error type, allowing each implementation to define its own specific
/// errors.
///
/// # Associated Types
/// - `Error`: The error type returned by the `load` method, which must implement
///   `std::error::Error` and have a `'static` lifetime.
///
/// # Methods
/// - `load`: Loads data from the specified file path into features and targets.
///
/// # Notes
/// - The feature array represents the input data with shape `(n_rows, n_features)`.
/// - The target array represents the output data with length `n_rows`.
/// - Implementors are responsible for defining how data is parsed and structured.
pub trait DataLoader {
    /// Loads data from a file into a feature matrix and target vector.
    ///
    /// # Parameters
    /// - `path`: The path to the data file, accepting any type that implements `AsRef<Path>`.
    ///
    /// # Returns
    /// A `Result` containing:
    /// - On success: A tuple `(features, targets)` where `features` is an `Array2<f64>`
    ///   of shape `(n_rows, n_features)` and `targets` is an `Array1<f64>` of length `n_rows`.
    /// - On error: An error of type `Self::Error` specific to the implementation.
    fn load<P: AsRef<std::path::Path>>(path: P) -> Result<(Array2<f64>, Array1<f64>), Self::Error>;

    /// The error type returned by the `load` method.
    type Error: std::error::Error + 'static;
}

/// Loads data from a file using a specified `DataLoader` implementation.
///
/// This generic helper function provides a convenient way to load data by delegating
/// to a type that implements the `DataLoader` trait. It takes a file path and returns
/// the parsed data as a 2D feature array (`Array2<f64>`) and a 1D target array (`Array1<f64>`),
/// with errors propagated from the underlying loader implementation.
///
/// # Type Parameters
/// - `T`: The type implementing `DataLoader`, determining the specific loading behavior
///   and error type.
/// - `P`: The path type, constrained to implement `AsRef<Path>`.
///
/// # Parameters
/// - `path`: The path to the data file.
///
/// # Returns
/// A `Result` containing:
/// - On success: A tuple `(features, targets)` where `features` is an `Array2<f64>`
///   of shape `(n_rows, n_features)` and `targets` is an `Array1<f64>` of length `n_rows`.
/// - On error: An error of type `T::Error`, specific to the `DataLoader` implementation.
///
/// # Notes
/// - The exact data format and error conditions depend on the `DataLoader` implementation
///   (e.g., `CsvLoader` for CSV files).
pub fn load_data<T: DataLoader, P: AsRef<std::path::Path>>(
    path: P,
) -> Result<(Array2<f64>, Array1<f64>), T::Error> {
    T::load(path)
}

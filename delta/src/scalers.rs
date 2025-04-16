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

use crate::errors::ScalerError;

#[derive(Clone)]
pub struct StandardScaler {
    mean: Option<Array1<f64>>,
    std: Option<Array1<f64>>,
}

impl StandardScaler {
    pub fn new() -> Self {
        StandardScaler { mean: None, std: None }
    }

    pub fn fit_transform(&mut self, x: &Array2<f64>) -> Result<Array2<f64>, ScalerError> {
        if x.ncols() == 0 {
            return Err(ScalerError::NoFeatures);
        }

        if x.is_empty() {
            return Err(ScalerError::EmptyInput);
        }

        self.mean = Some(x.mean_axis(Axis(0)).ok_or_else(|| {
            ScalerError::ArrayOperation(ndarray::ShapeError::from_kind(
                ndarray::ErrorKind::IncompatibleShape,
            ))
        })?);

        self.std = Some(
            x.var_axis(Axis(0), 0.0).mapv(|v| v.sqrt()).mapv(|s| if s < 1e-10 { 1.0 } else { s }),
        );

        let mean = self.mean.as_ref().unwrap();
        let std = self.std.as_ref().unwrap();
        Ok((x - mean) / std)
    }

    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, ScalerError> {
        let mean = self.mean.as_ref().ok_or(ScalerError::NotFitted)?;
        let std = self.std.as_ref().ok_or(ScalerError::NotFitted)?;

        if x.is_empty() {
            return Err(ScalerError::EmptyInput);
        }

        if x.ncols() != mean.len() {
            return Err(ScalerError::DimensionMismatch { expected: mean.len(), actual: x.ncols() });
        }

        Ok((x - mean) / std)
    }

    pub fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, ScalerError> {
        let mean = self.mean.as_ref().ok_or(ScalerError::NotFitted)?;
        let std = self.std.as_ref().ok_or(ScalerError::NotFitted)?;

        if x.is_empty() {
            return Err(ScalerError::EmptyInput);
        }

        if x.ncols() != mean.len() {
            return Err(ScalerError::DimensionMismatch { expected: mean.len(), actual: x.ncols() });
        }

        Ok(x * std + mean)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn standard_scaler_new() {
        let scaler = StandardScaler::new();
        assert!(scaler.mean.is_none());
        assert!(scaler.std.is_none());
    }

    #[test]
    fn standard_scaler_fit_transform_empty_input() {
        let mut scaler = StandardScaler::new();
        let x: Array2<f64> = Array2::zeros((0, 2));
        let result = scaler.fit_transform(&x);
        assert!(matches!(result, Err(ScalerError::EmptyInput)));
    }

    #[test]
    fn standard_scaler_fit_transform_no_features() {
        let mut scaler = StandardScaler::new();
        let x: Array2<f64> = Array2::zeros((2, 0));
        let result = scaler.fit_transform(&x);
        assert!(matches!(result, Err(ScalerError::NoFeatures)));
    }

    #[test]
    fn standard_scaler_fit_transform_valid() {
        let mut scaler = StandardScaler::new();
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let result = scaler.fit_transform(&x);
        assert!(result.is_ok());
        let scaled = result.unwrap();
        assert_eq!(scaled.shape(), [3, 2]);
        assert!(scaler.mean.is_some());
        assert!(scaler.std.is_some());
        let mean = scaler.mean.unwrap();
        let std = scaler.std.unwrap();
        assert!((mean[0] - 3.0).abs() < 1e-10);
        assert!((mean[1] - 4.0).abs() < 1e-10);
        assert!(std.iter().all(|&s| s > 0.0));
    }

    #[test]
    fn standard_scaler_transform_not_fitted() {
        let scaler = StandardScaler::new();
        let x = array![[1.0, 2.0]];
        let result = scaler.transform(&x);
        assert!(matches!(result, Err(ScalerError::NotFitted)));
    }

    #[test]
    fn standard_scaler_transform_empty_input() {
        let mut scaler = StandardScaler::new();
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        scaler.fit_transform(&x_train).unwrap();
        let x_test: Array2<f64> = Array2::zeros((0, 2));
        let result = scaler.transform(&x_test);
        assert!(matches!(result, Err(ScalerError::EmptyInput)));
    }

    #[test]
    fn standard_scaler_transform_dimension_mismatch() {
        let mut scaler = StandardScaler::new();
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        scaler.fit_transform(&x_train).unwrap();
        let x_test = array![[1.0, 2.0, 3.0]];
        let result = scaler.transform(&x_test);
        assert!(matches!(result, Err(ScalerError::DimensionMismatch { expected: 2, actual: 3 })));
    }

    #[test]
    fn standard_scaler_transform_valid() {
        let mut scaler = StandardScaler::new();
        let x_train = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        scaler.fit_transform(&x_train).unwrap();
        let x_test = array![[2.0, 3.0]];
        let result = scaler.transform(&x_test);
        assert!(result.is_ok());
        let transformed = result.unwrap();
        assert_eq!(transformed.shape(), [1, 2]);
        assert!(transformed.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn standard_scaler_inverse_transform_not_fitted() {
        let scaler = StandardScaler::new();
        let x = array![[1.0, 2.0]];
        let result = scaler.inverse_transform(&x);
        assert!(matches!(result, Err(ScalerError::NotFitted)));
    }

    #[test]
    fn standard_scaler_inverse_transform_empty_input() {
        let mut scaler = StandardScaler::new();
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        scaler.fit_transform(&x_train).unwrap();
        let x_test: Array2<f64> = Array2::zeros((0, 2));
        let result = scaler.inverse_transform(&x_test);
        assert!(matches!(result, Err(ScalerError::EmptyInput)));
    }

    #[test]
    fn standard_scaler_inverse_transform_dimension_mismatch() {
        let mut scaler = StandardScaler::new();
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        scaler.fit_transform(&x_train).unwrap();
        let x_test = array![[1.0, 2.0, 3.0]];
        let result = scaler.inverse_transform(&x_test);
        assert!(matches!(result, Err(ScalerError::DimensionMismatch { expected: 2, actual: 3 })));
    }

    #[test]
    fn standard_scaler_inverse_transform_valid() {
        let mut scaler = StandardScaler::new();
        let x_train = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let scaled = scaler.fit_transform(&x_train).unwrap();
        let result = scaler.inverse_transform(&scaled);
        assert!(result.is_ok());
        let unscaled = result.unwrap();
        assert_eq!(unscaled.shape(), [3, 2]);
        for (orig, unscaled) in x_train.iter().zip(unscaled.iter()) {
            assert!((orig - unscaled).abs() < 1e-10);
        }
    }
}

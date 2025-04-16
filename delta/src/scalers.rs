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
        if x.is_empty() {
            return Err(ScalerError::EmptyInput);
        }

        if x.ncols() == 0 {
            return Err(ScalerError::NoFeatures);
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

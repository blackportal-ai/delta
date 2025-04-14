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
use num_traits::{Float, FromPrimitive};

pub trait Scaler<T: Float> {
    fn fit(&mut self, x: &Array2<T>);

    fn transform(&self, x: &Array2<T>) -> Array2<T>;

    fn inverse_transform(&self, x: &Array2<T>) -> Array2<T>;

    fn fit_transform(&mut self, x: &Array2<T>) -> Array2<T> {
        self.fit(x);
        self.transform(x)
    }
}

pub struct StandardScaler<T: Float> {
    mean: Option<Array1<T>>,
    std: Option<Array1<T>>,
}

impl<T: Float + FromPrimitive> StandardScaler<T> {
    pub fn new() -> Self {
        StandardScaler { mean: None, std: None }
    }
}

impl<T: Float + FromPrimitive> Scaler<T> for StandardScaler<T> {
    fn fit(&mut self, x: &Array2<T>) {
        let mean = x.mean_axis(Axis(0)).unwrap();
        let std = x.var_axis(Axis(0), T::one()).mapv(|v| v.sqrt());
        self.mean = Some(mean);
        self.std = Some(std.mapv(|s| if s == T::zero() { T::one() } else { s }));
    }

    fn transform(&self, x: &Array2<T>) -> Array2<T> {
        let mean = self.mean.as_ref().expect("Scaler not fitted");
        let std = self.std.as_ref().expect("Scaler not fitted");
        (x - mean) / std
    }

    fn inverse_transform(&self, x: &Array2<T>) -> Array2<T> {
        let mean = self.mean.as_ref().expect("Scaler not fitted");
        let std = self.std.as_ref().expect("Scaler not fitted");
        x * std + mean
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_standard_scaler_fit_transform() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mut scaler = StandardScaler::new();
        let x_scaled = scaler.fit_transform(&x);

        // Check mean ~ 0 and std ~ 1 per column
        let mean = x_scaled.mean_axis(Axis(0)).unwrap();
        let std = x_scaled.var_axis(Axis(0), 1.0).mapv(|v| v.sqrt());
        for &m in mean.iter() {
            assert!((m.abs() < 1e-10), "Mean should be ~0, got {}", m);
        }
        for &s in std.iter() {
            assert!((s - 1.0).abs() < 1e-10, "Std should be ~1, got {}", s);
        }
    }

    #[test]
    fn test_standard_scaler_inverse_transform() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mut scaler = StandardScaler::new();
        let x_scaled = scaler.fit_transform(&x);
        let x_restored = scaler.inverse_transform(&x_scaled);

        // Check restored data matches original
        for (orig, restored) in x.iter().zip(x_restored.iter()) {
            assert!((orig - restored).abs() < 1e-10, "Restored value differs");
        }
    }

    #[test]
    fn test_standard_scaler_zero_variance() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]).unwrap();
        let mut scaler = StandardScaler::new();
        let x_scaled = scaler.fit_transform(&x);

        // Check constant columns are unchanged (std=1 fallback)
        assert_eq!(x_scaled.column(0), Array1::from_vec(vec![0.0, 0.0, 0.0]));
    }
}

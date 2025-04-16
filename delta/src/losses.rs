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

use std::iter::Sum;

use ndarray::{Array1, ScalarOperand};
use num_traits::Float;

#[derive(Clone)]
pub struct MSE;

#[derive(Clone)]
pub struct CrossEntropy;

#[derive(Clone)]
pub struct MAE;

pub trait Loss<T> {
    fn calculate(&self, predictions: &Array1<T>, actuals: &Array1<T>) -> T;
}

impl<T> Loss<T> for MSE
where
    T: Float,
{
    fn calculate(&self, predictions: &Array1<T>, actuals: &Array1<T>) -> T {
        let m = T::from(predictions.len()).unwrap();
        let diff = predictions - actuals;
        (diff.mapv(|x| x.powi(2)).sum()) / m
    }
}

impl<T> Loss<T> for CrossEntropy
where
    T: Float + ScalarOperand + Sum,
{
    fn calculate(&self, predictions: &Array1<T>, actuals: &Array1<T>) -> T {
        let m = T::from(predictions.len()).unwrap();
        let epsilon = T::from(1e-15).unwrap();

        predictions
            .iter()
            .zip(actuals.iter())
            .map(|(p, y)| {
                let p_clamped = p.max(epsilon).min(T::one() - epsilon);
                -(*y * p_clamped.ln() + (T::one() - *y) * (T::one() - p_clamped).ln())
            })
            .sum::<T>()
            / m
    }
}

impl<T> Loss<T> for MAE
where
    T: Float,
{
    fn calculate(&self, predictions: &Array1<T>, actuals: &Array1<T>) -> T {
        let m = T::from(predictions.len()).unwrap();
        let diff = predictions - actuals;
        (diff.mapv(|x| x.abs()).sum()) / m
    }
}

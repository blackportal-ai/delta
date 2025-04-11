use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive};

pub struct Normalization<T: Float> {
    mean: Option<Array1<T>>,
    std: Option<Array1<T>>,
}

impl<T> Normalization<T>
where
    T: Float + FromPrimitive + std::fmt::Debug + ScalarOperand,
{
    pub fn new() -> Self {
        Normalization { mean: None, std: None }
    }

    // Adapt to 2D data (e.g., x_train)
    pub fn adapt_2d(&mut self, data: &Array2<T>) {
        self.mean = Some(data.mean_axis(Axis(0)).expect("Failed to compute mean"));
        let std = data.std_axis(Axis(0), T::zero());
        self.std = Some(std.mapv(|s| if s.is_zero() { T::from(1.0).unwrap() } else { s }));
    }

    // Adapt to 1D data (e.g., y_train)
    pub fn adapt_1d(&mut self, data: &Array1<T>) {
        let mean = data.mean().expect("Failed to compute mean");
        let std = data.std(T::zero());
        self.mean = Some(Array1::from_elem(1, mean));
        self.std =
            Some(Array1::from_elem(1, if std.is_zero() { T::from(1.0).unwrap() } else { std }));
    }

    // Transform 2D data
    pub fn transform_2d(&self, data: &Array2<T>) -> Array2<T> {
        let mean = self.mean.as_ref().expect("Normalizer not adapted");
        let std = self.std.as_ref().expect("Normalizer not adapted");
        (data - mean) / std
    }

    // Transform 1D data
    pub fn transform_1d(&self, data: &Array1<T>) -> Array1<T> {
        let mean = self.mean.as_ref().expect("Normalizer not adapted")[0];
        let std = self.std.as_ref().expect("Normalizer not adapted")[0];
        (data - mean) / std
    }

    // Inverse transform 1D data (for predictions)
    pub fn inverse_transform_1d(&self, data: &Array1<T>) -> Array1<T> {
        let mean = self.mean.as_ref().expect("Normalizer not adapted")[0];
        let std = self.std.as_ref().expect("Normalizer not adapted")[0];
        (data * std) + mean
    }
}

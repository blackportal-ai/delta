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

use std::io::Cursor;
use std::ops::{AddAssign, Index, IndexMut, Mul, Range, SubAssign};

use image::{GenericImageView, ImageReader};
use ndarray::{Array, ArrayD, Axis, Dimension, Ix2, IxDyn, Shape};
use rand::{Rng, thread_rng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

use crate::devices::Device;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::devices::osx_metal::{
    tensor_add_metal, tensor_divide_metal, tensor_map_max_metal, tensor_matmul_metal,
    tensor_power_metal, tensor_subtract_metal,
};

/// A struct representing a tensor.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// The dataset of the tensor stored as an n-dimensional array.
    pub data: ArrayD<f32>,
    pub device: Device,
}

impl Tensor {
    /// Creates a new tensor.
    ///
    /// # Arguments
    ///
    /// * `dataset` - A vector of dataset.
    /// * `shape` - A vector representing the shape of the tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    pub fn new(data: Vec<f32>, shape: Shape<IxDyn>) -> Self {
        Self {
            data: Array::from_shape_vec(shape, data).expect("Invalid shape for dataset"),
            device: Device::default(),
        }
    }

    /// Creates a tensor filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - A vector representing the shape of the tensor.
    ///
    /// # Returns
    ///
    /// A tensor filled with zeros.
    pub fn zeros(shape: Shape<IxDyn>, device: Device) -> Self {
        Self { data: Array::zeros(shape), device }
    }

    /// Creates a tensor filled with ones.
    ///
    /// # Arguments
    ///
    /// * `shape` - A vector representing the shape of the tensor.
    ///
    /// # Returns
    ///
    /// A tensor filled with ones.
    pub fn ones(shape: Shape<IxDyn>, device: Device) -> Self {
        Self { data: Array::ones(shape), device }
    }

    /// Creates a tensor filled with random values.
    ///
    /// # Arguments
    ///
    /// * `shape` - A vector representing the shape of the tensor.
    ///
    /// # Returns
    ///
    /// A tensor filled with random values.
    pub fn random(shape: Shape<IxDyn>) -> Self {
        let mut rng = thread_rng();
        let data: Vec<f32> = (0..shape.size()).map(|_| rng.gen::<f32>()).collect(); // Use size() method
        Self {
            data: Array::from_shape_vec(shape, data).expect("Invalid shape for random dataset"),
            device: Device::default(),
        }
    }

    /// Adds two tensors element-wise.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to add.
    ///
    /// # Returns
    ///
    /// A new tensor containing the result of the addition.
    pub fn add(&self, other: &Tensor) -> Tensor {
        // Check device compatibility
        match &self.device {
            Device::Cpu => Tensor { data: &self.data + &other.data, device: self.device.clone() },
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Device::Metal { device, queue } => {
                // Perform Metal addition
                tensor_add_metal(self, other, device, queue)
                    .expect("Failed to perform addition on Metal device")
            }
        }
    }

    /// Gets the maximum value in the tensor.
    ///
    /// # Returns
    ///
    /// The maximum value in the tensor.
    pub fn max(&self) -> f32 {
        *self.data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
    }

    /// Calculates the mean of the tensor.
    ///
    /// # Returns
    ///
    /// The mean of the tensor.
    pub fn mean(&self) -> f32 {
        self.data.mean().unwrap_or(0.0)
    }

    /// Reshapes the tensor to a new shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The new shape.
    ///
    /// # Returns
    ///
    /// A new tensor with the reshaped dataset.
    pub fn reshape(&self, shape: IxDyn) -> Tensor {
        Tensor {
            data: self
                .data
                .view() // Prevent cloning
                .into_shape_with_order(shape)
                .expect("Invalid shape for reshape")
                .to_owned(), // Only create a new allocation if necessary
            device: self.device.clone(),
        }
    }

    /// Applies a function to each element of the tensor.
    ///
    /// # Arguments
    ///
    /// * `f` - The function to apply.
    ///
    /// # Returns
    ///
    /// A new tensor with the result of applying the function.
    pub fn map<F>(&self, f: F) -> Tensor
    where
        F: Fn(f32) -> f32,
    {
        // Create a new array by applying the function `f` to each element of `self.dataset`
        let new_data = self.data.mapv(f);

        Tensor { data: new_data, device: self.device.clone() }
    }

    /// Slices the tensor along the specified indices.
    ///
    /// # Arguments
    ///
    /// * `indices` - A vector of ranges for slicing along each axis.
    ///
    /// # Returns
    ///
    /// A new tensor containing the sliced dataset.
    pub fn slice(&self, indices: Vec<Range<usize>>) -> Tensor {
        let slices: Vec<_> = indices.iter().map(|r| r.clone().into()).collect();
        let view = self.data.slice(slices.as_slice());
        Tensor { data: view.to_owned(), device: self.device.clone() }
    }

    /// Performs matrix multiplication between two tensors.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor.
    ///
    /// # Returns
    ///
    /// A new tensor containing the result of the matrix multiplication.
    pub fn dot(&self, other: &Tensor) -> Tensor {
        // Ensure both tensors have at least 2 dimensions for matrix multiplication
        if self.data.ndim() < 2 || other.data.ndim() < 2 {
            panic!("Both tensors must have at least 2 dimensions for matmul");
        }

        // Extract the last two dimensions for matrix multiplication
        let self_2d = self
            .data
            .view()
            .into_dimensionality::<Ix2>()
            .expect("Self tensor must be 2D for matmul");
        let other_2d = other
            .data
            .view()
            .into_dimensionality::<Ix2>()
            .expect("Other tensor must be 2D for matmul");

        // Tensor { data: self_2d.dot(&other_2d).into_dyn(), device: self.device.clone() }
        match &self.device {
            Device::Cpu => {
                Tensor { data: self_2d.dot(&other_2d).into_dyn(), device: self.device.clone() }
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Device::Metal { device, queue } => tensor_matmul_metal(
                &Tensor { data: self_2d.to_owned().into_dyn(), device: self.device.clone() },
                &Tensor { data: other_2d.to_owned().into_dyn(), device: self.device.clone() },
                device,
                queue,
            )
            .expect("Failed to perform matrix multiplication on Metal device"),
        }
    }

    /// Transposes the tensor by swapping axes.
    ///
    /// # Returns
    ///
    /// A new tensor containing the transposed dataset.
    ///
    /// # Panics
    ///
    /// This method assumes the tensor is at least 2D.
    pub fn transpose(&self) -> Tensor {
        let ndim = self.data.ndim();
        if ndim < 2 {
            panic!("Cannot transpose a tensor with less than 2 dimensions");
        }

        // Create a transposed array by reversing the axes
        let axes: Vec<usize> = (0..ndim).rev().collect();
        Tensor { data: self.data.clone().permuted_axes(axes), device: self.device.clone() }
    }

    /// Gets the shape of the tensor.
    ///
    /// # Returns
    ///
    /// The shape of the tensor as `Shape<IxDyn>`.
    pub fn shape(&self) -> Shape<IxDyn> {
        IxDyn(self.data.shape()).into()
    }

    /// Permutes the axes of the tensor.
    ///
    /// # Arguments
    ///
    /// * `axes` - A vector representing the new order of axes.
    ///
    /// # Returns
    ///
    /// A new tensor with the permuted axes.
    pub fn permute(&self, axes: Vec<usize>) -> Tensor {
        Tensor { data: self.data.clone().permuted_axes(axes), device: self.device.clone() }
    }

    /// Sums the tensor along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to sum along.
    ///
    /// # Returns
    ///
    /// A new tensor containing the summed dataset.
    pub fn sum_along_axis(&self, axis: usize) -> Tensor {
        let sum = self.data.sum_axis(Axis(axis));
        Tensor { data: sum, device: self.device.clone() }
    }

    /// Multiplies the tensor by a scalar value.
    ///
    /// # Arguments
    ///
    /// * `amount` - The scalar value to multiply the tensor by.
    pub fn mul_scalar(&self, amount: f32) -> Tensor {
        let data: Vec<f32> = self
            .data
            .as_slice()
            .expect("Tensor data must be contiguous")
            .par_iter()
            .map(|&x| x * amount)
            .collect();

        let shape = self.data.shape();
        Tensor {
            data: Array::from_shape_vec(IxDyn(shape), data).expect("Invalid shape"),
            device: self.device.clone(),
        }
    }

    /// Raises the tensor to a power.
    ///
    /// # Arguments
    ///
    /// * `amount` - The power to raise the tensor to.
    pub fn pow(&self, amount: f32) -> Tensor {
        match &self.device {
            Device::Cpu => {
                let data: Vec<f32> = self
                    .data
                    .as_slice()
                    .expect("Tensor data must be contiguous")
                    .par_iter()
                    .map(|&x| x.powf(amount))
                    .collect();

                let shape = self.data.shape();
                Tensor {
                    data: Array::from_shape_vec(IxDyn(shape), data).expect("Invalid shape"),
                    device: self.device.clone(),
                }
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Device::Metal { device, queue } => tensor_power_metal(self, amount, device, queue)
                .expect("Failed to perform power operation on Metal device"),
        }
    }

    /// Divides the tensor by a scalar value.
    ///
    /// # Arguments
    ///
    /// * `amount` - The scalar value to divide the tensor by.
    pub fn div_scalar(&self, amount: f32) -> Tensor {
        let data: Vec<f32> = self
            .data
            .as_slice()
            .expect("Tensor data must be contiguous")
            .par_iter()
            .map(|&x| x / amount)
            .collect();

        let shape = self.data.shape();
        Tensor {
            data: Array::from_shape_vec(IxDyn(shape), data).expect("Invalid shape"),
            device: self.device.clone(),
        }
    }

    /// Computes the square root of each element in the tensor.
    ///
    /// # Returns
    ///
    /// A new tensor containing the square roots of the elements.
    pub fn sqrt(&self) -> Tensor {
        let data: Vec<f32> = self
            .data
            .as_slice()
            .expect("Tensor data must be contiguous")
            .par_iter()
            .map(|&x| x.sqrt())
            .collect();

        let shape = self.data.shape();
        Tensor {
            data: Array::from_shape_vec(IxDyn(shape), data).expect("Invalid shape"),
            device: self.device.clone(),
        }
    }

    /// Adds a scalar value to each element in the tensor.
    ///
    /// # Arguments
    ///
    /// * `amount` - The scalar value to add to each element.
    ///
    /// # Returns
    ///
    /// A new tensor containing the result of the addition.
    pub fn add_scalar(&self, amount: f32) -> Tensor {
        let data: Vec<f32> = self
            .data
            .as_slice()
            .expect("Tensor data must be contiguous")
            .par_iter()
            .map(|&x| x + amount)
            .collect();

        let shape = self.data.shape();
        Tensor {
            data: Array::from_shape_vec(IxDyn(shape), data).expect("Invalid shape"),
            device: self.device.clone(),
        }
    }

    /// Divides each element in the tensor.
    ///
    /// # Arguments
    ///
    /// * `amount` - The scalar value to divide each element by.
    ///
    /// # Returns
    ///
    /// A new tensor containing the result of the division.
    pub fn div(&self, other: &Tensor) -> Tensor {
        match &self.device {
            Device::Cpu => Tensor { data: &self.data / &other.data, device: self.device.clone() },
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Device::Metal { device, queue } => tensor_divide_metal(self, other, device, queue)
                .expect("Failed to perform division on Metal device"),
        }
    }

    /// Applies a threshold to each element in the tensor.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The threshold value to apply.
    ///
    /// # Returns
    ///
    /// A new tensor containing the result of the threshold operation.
    pub fn map_max(&self, threshold: f32) -> Tensor {
        match &self.device {
            Device::Cpu => {
                let data: Vec<f32> = self
                    .data
                    .as_slice()
                    .expect("Tensor data must be contiguous")
                    .par_iter()
                    .map(|&x| x.max(threshold))
                    .collect();

                let shape = self.data.shape();
                Tensor {
                    data: Array::from_shape_vec(IxDyn(shape), data).expect("Invalid shape"),
                    device: self.device.clone(),
                }
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Device::Metal { device, queue } => tensor_map_max_metal(self, threshold, device, queue)
                .expect("Failed to perform map_max operation on Metal device"),
        }
    }

    /// Flattens the tensor into a 1D array.
    ///
    /// # Returns
    ///
    /// A new tensor containing the flattened dataset.
    pub fn flatten(&self) -> Tensor {
        let shape = IxDyn(&[self.data.len()]);
        Tensor {
            data: self.data.clone().into_shape_with_order(shape).unwrap(),
            device: self.device.clone(),
        }
    }

    /// Computes the mean along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to compute the mean along.
    ///
    /// # Returns
    ///
    /// A new tensor containing the mean dataset.
    pub fn mean_axis(&self, axis: usize) -> Tensor {
        let mean = self.data.mean_axis(Axis(axis)).expect("Failed to calculate mean");
        Tensor { data: mean, device: self.device.clone() }
    }

    /// Broadcasts the tensor to a target shape.
    ///
    /// # Arguments
    ///
    /// * `target_shape` - The target shape to broadcast to.
    ///
    /// # Returns
    ///
    /// A new tensor with the broadcasted shape.
    ///
    /// # Panics
    ///
    /// Panics if the current shape cannot be broadcasted to the target shape.
    pub fn broadcast(&self, target_shape: Shape<IxDyn>) -> Tensor {
        let shape_binding = self.shape();
        let self_shape = shape_binding.raw_dim();
        let ndim_self = self_shape.ndim();
        let ndim_target = target_shape.raw_dim().ndim();

        // Pad the current shape with leading 1s to match the target dimensions
        let mut padded_shape = vec![1; ndim_target - ndim_self];
        padded_shape.extend(self_shape.slice());

        // Validate compatibility for broadcasting
        for (self_dim, target_dim) in padded_shape.iter().zip(target_shape.raw_dim().slice()) {
            if *self_dim != *target_dim && *self_dim != 1 {
                panic!("Cannot broadcast shape {:?} to {:?}", self_shape, target_shape);
            }
        }

        // Perform the broadcasting
        let broadcasted_data = self
            .data
            .broadcast(target_shape.raw_dim().clone()) // Dereference to get Dim<IxDynImpl>
            .expect("Broadcast failed")
            .to_owned();

        Tensor { data: broadcasted_data, device: self.device.clone() }
    }

    /// Normalizes the tensor to a specified range.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum value of the range.
    /// * `max` - The maximum value of the range.
    ///
    /// # Returns
    ///
    /// A new tensor containing the normalized dataset.
    pub fn normalize(&self, min: f32, max: f32) -> Tensor {
        let current_min = self.data.iter().cloned().fold(f32::INFINITY, f32::min);
        let current_max = self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        if current_min == current_max {
            return Tensor::zeros(self.shape(), self.device.clone());
        }

        let normalized_data =
            self.data.mapv(|x| (x - current_min) / (current_max - current_min) * (max - min) + min);

        Tensor { data: normalized_data, device: self.device.clone() }
    }

    /// Adds noise to the tensor.
    ///
    /// # Arguments
    ///
    /// * `noise_level` - The level of noise to add.
    pub fn add_noise(&mut self, noise_level: f32) {
        let mut rng = thread_rng();
        self.data.mapv_inplace(|value| {
            let noise: f32 = rng.gen_range(-noise_level..noise_level);
            value + noise
        });
    }

    /// Reduces the tensor along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to reduce along.
    ///
    /// # Returns
    ///
    /// A new tensor containing the reduced dataset.
    pub fn reduce_sum(&self, axis: usize) -> Tensor {
        let sum = self.data.sum_axis(Axis(axis));
        Tensor { data: sum, device: self.device.clone() }
    }

    /// Gets the index of the maximum value along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to find the maximum along.
    ///
    /// # Returns
    ///
    /// A new tensor containing the indices of the maximum values.
    ///
    /// # Panics
    ///
    /// Panics if the axis is out of bounds.
    pub fn argmax(&self, axis: usize) -> Tensor {
        // Ensure the axis is valid
        if axis >= self.data.ndim() {
            panic!("Axis {} is out of bounds for tensor with shape {:?}", axis, self.shape());
        }

        // Compute the indices of the maximum values along the specified axis
        let max_indices = self
            .data
            .map_axis(Axis(axis), |subview| {
                subview
                    .indexed_iter()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(index, _)| index)
                    .unwrap()
            })
            .into_dyn();

        Tensor { data: max_indices.mapv(|x| x as f32), device: self.device.clone() }
    }

    /// Takes elements from the tensor according to the given indices.
    ///
    /// # Arguments
    ///
    /// * `indices` - A vector of indices to take.
    ///
    /// # Returns
    ///
    /// A new tensor containing the selected elements.
    pub fn take(&self, indices: &[usize]) -> Tensor {
        let mut data =
            Vec::with_capacity(indices.len() * self.data.len() / self.shape().raw_dim()[0]);
        let stride = self.data.len() / self.shape().raw_dim()[0];

        for &idx in indices {
            let start = idx * stride;
            let end = start + stride;
            data.extend_from_slice(&self.data.as_slice().unwrap()[start..end]);
        }

        let mut new_shape: Vec<usize> = self.shape().raw_dim().as_array_view().to_vec();
        new_shape[0] = indices.len();
        let shape = Shape::from(IxDyn(&new_shape));

        Tensor::new(data, shape)
    }

    /// Converts the tensor dataset to a vector.
    ///
    /// # Returns
    ///
    /// A vector containing the tensor dataset in row-major order.
    pub fn to_vec(&self) -> Vec<f32> {
        self.data.as_slice().unwrap_or(&[]).to_vec()
    }

    /// Creates a tensor from image bytes.
    ///
    /// # Arguments
    ///
    /// * `image_bytes` - The bytes of the image.
    ///
    /// # Returns
    ///
    /// A `Tensor` containing the image pixel dataset in the shape `(height, width, channels)`.
    pub fn from_image_bytes(image_bytes: Vec<u8>) -> Result<Self, String> {
        // Decode the image from bytes
        let image = ImageReader::new(Cursor::new(image_bytes))
            .with_guessed_format()
            .map_err(|e| format!("Failed to read image: {}", e))?
            .decode()
            .map_err(|e| format!("Failed to decode image: {}", e))?;

        // Get image dimensions and pixel dataset
        let (width, height) = image.dimensions();
        let pixel_data = image.to_rgba8().into_raw(); // Convert to RGBA and flatten the pixel dataset

        // Construct the Tensor with shape (height, width, 4)
        Ok(Tensor::new(
            pixel_data.iter().map(|&x| x as f32 / 255.0).collect(), // Normalize pixel values
            Shape::from(IxDyn(&[height as usize, width as usize, 4])), // Shape (H, W, C)
        ))
    }

    /// Stacks multiple tensors along a new axis.
    ///
    /// # Arguments
    ///
    /// * `tensors` - A slice of tensors to stack.
    ///
    /// # Returns
    ///
    /// A new tensor containing the stacked tensors.
    ///
    /// # Panics
    ///
    /// Panics if the tensors do not have the same shape.
    pub fn stack(tensors: &[Tensor]) -> Result<Tensor, String> {
        if tensors.is_empty() {
            return Err("Cannot stack an empty list of tensors.".to_string());
        }

        // Create a longer-lived binding for the shape
        let shape_binding = tensors[0].shape();
        let first_shape = shape_binding.raw_dim();

        for tensor in tensors {
            if tensor.shape().raw_dim() != first_shape {
                return Err(format!(
                    "All tensors must have the same shape. Expected {:?}, got {:?}",
                    first_shape,
                    tensor.shape().raw_dim()
                ));
            }
        }

        // Stack tensors along a new axis
        let stacked_data =
            ndarray::stack(Axis(0), &tensors.iter().map(|t| t.data.view()).collect::<Vec<_>>())
                .map_err(|e| e.to_string())?;

        Ok(Tensor { data: stacked_data.into_dyn(), device: tensors[0].device.clone() })
    }

    /// Splits the tensor into two parts at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index at which to split the tensor.
    ///
    /// # Returns
    ///
    /// A tuple containing the two resulting tensors.
    pub fn split_at(&self, index: usize) -> (Tensor, Tensor) {
        let shape_binding = self.shape();
        let shape = shape_binding.raw_dim();

        // Ensure the tensor has at least one dimension
        assert!(shape.ndim() > 0, "Tensor must have at least one dimension");

        // Ensure the index is within the bounds of the tensor's first dimension
        assert!(index <= shape[0], "Index out of bounds for tensor split");

        // Ensure the tensor has at least two dimensions for slicing
        assert!(shape.ndim() >= 2, "Tensor must have at least two dimensions for slicing");

        println!("shape: {:?}", shape);

        // Create dynamic slice patterns based on shape dimensions
        let mut slice1 = vec![ndarray::Slice::new(0, Some(index.try_into().unwrap()), 1)];
        let mut slice2 = vec![ndarray::Slice::new(index.try_into().unwrap(), None, 1)];

        // Add full slices (..) for remaining dimensions
        for _ in 1..shape.ndim() {
            slice1.push(ndarray::Slice::from(..));
            slice2.push(ndarray::Slice::from(..));
        }

        let data1 = self.data.slice_each_axis(|ax| slice1[ax.axis.0]).to_owned().into_dyn();
        let data2 = self.data.slice_each_axis(|ax| slice2[ax.axis.0]).to_owned().into_dyn();

        (
            Tensor { data: data1, device: self.device.clone() },
            Tensor { data: data2, device: self.device.clone() },
        )
    }

    /// Creates a tensor filled with random values sampled from a normal distribution.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `mean` - The mean of the normal distribution.
    /// * `stddev` - The standard deviation of the normal distribution.
    ///
    /// # Returns
    ///
    /// A new tensor filled with random values sampled from the normal distribution.
    pub fn random_normal(shape: Shape<IxDyn>, mean: f32, stddev: f32) -> Self {
        let normal = Normal::new(mean, stddev).expect("Failed to create normal distribution");
        let mut rng = thread_rng();

        // Generate random values from the normal distribution
        let data: Vec<f32> = (0..shape.size()).map(|_| normal.sample(&mut rng)).collect();

        // Create a tensor from the generated data
        Tensor {
            data: Array::from_shape_vec(shape, data).expect("Invalid shape for random dataset"),
            device: Device::default(),
        }
    }

    /// Subtracts another tensor from this tensor element-wise.
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to subtract.
    ///
    /// # Returns
    ///
    /// A new tensor containing the element-wise subtraction result.
    ///
    /// # Panics
    ///
    /// Panics if the shapes are incompatible and cannot be broadcast.
    pub fn sub(&self, other: &Tensor) -> Tensor {
        // Attempt broadcasting
        let broadcasted_other = other
            .data
            .broadcast(self.data.raw_dim())
            .expect("Shapes are incompatible for broadcasting");

        // Perform the element-wise subtraction
        match &self.device {
            Device::Cpu => {
                Tensor { data: &self.data - &broadcasted_other, device: self.device.clone() }
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Device::Metal { device, queue } => tensor_subtract_metal(
                self,
                &Tensor {
                    data: broadcasted_other.to_owned().into_dyn(),
                    device: self.device.clone(),
                },
                device,
                queue,
            )
            .expect("Failed to perform subtraction on Metal device"),
        }
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().copied().sum()
    }

    /// Transfers the tensor to the specified device.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to transfer the tensor to.
    ///
    /// # Returns
    ///
    /// A new tensor on the specified device.
    pub fn to_device(&mut self, device: Device) -> Result<Self, String> {
        self.device = device.clone();
        Ok(self.clone())
    }

    /// Pads the tensor with zeros on all sides by creating a new tensor with all zeros
    /// and filling the center with the original tensor data.
    ///
    /// # Arguments
    /// * `padding` - The number of zeros to add to each side of the tensor.
    ///
    /// # Returns
    /// * A new tensor with the specified padding.
    pub fn pad(&mut self, padding: usize) -> Result<Self, String> {
        if padding == 0 {
            return Ok(self.clone());
        }

        let input_shape = self.shape();
        let input_shape = input_shape.raw_dim();
        let batch_size = input_shape[0];
        let height = input_shape[1];
        let width = input_shape[2];
        let channels = input_shape[3];

        let padded_width = width + 2 * padding;
        let padded_height = width + 2 * padding;

        let mut padded_tensor = Tensor::zeros(
            Shape::from(IxDyn(&[batch_size, padded_height, padded_width, channels])),
            self.device.clone(),
        );

        for b in 0..batch_size {
            for h in 0..height {
                for w in 0..width {
                    for c in 0..channels {
                        padded_tensor[(b, h + padding, w + padding, c)] = self[(b, h, w, c)];
                    }
                }
            }
        }

        Ok(padded_tensor)
    }
}

impl Index<(usize, usize, usize, usize)> for Tensor {
    type Output = f32;
    fn index(&self, index: (usize, usize, usize, usize)) -> &Self::Output {
        let flat = self.data.as_slice().expect("Tensor data must be contiguous");
        let (b, i, j, f) = index;
        let dims = self.data.shape();
        let flat_index = b * dims[1] * dims[2] * dims[3] + i * dims[2] * dims[3] + j * dims[3] + f;
        &flat[flat_index]
    }
}

impl IndexMut<(usize, usize, usize, usize)> for Tensor {
    fn index_mut(&mut self, index: (usize, usize, usize, usize)) -> &mut Self::Output {
        // Copy the shape into an owned vector so we no longer hold an immutable borrow
        let dims: Vec<usize> = self.data.shape().to_vec();
        let flat = self.data.as_slice_mut().expect("Tensor data must be contiguous");
        let (b, i, j, f) = index;
        let flat_index = b * dims[1] * dims[2] * dims[3] + i * dims[2] * dims[3] + j * dims[3] + f;
        &mut flat[flat_index]
    }
}

impl SubAssign for Tensor {
    /// Subtracts another tensor from the current tensor in-place.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The tensor to subtract from the current tensor.
    fn sub_assign(&mut self, rhs: Self) {
        self.data -= &rhs.data;
    }
}

impl AddAssign for Tensor {
    /// Adds another tensor to the current tensor in-place.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The tensor to add to the current tensor.
    fn add_assign(&mut self, rhs: Self) {
        self.data += &rhs.data;
    }
}

impl Default for Tensor {
    /// Creates a new tensor with default values.
    ///
    /// # Returns
    ///
    /// A new tensor with default values.
    fn default() -> Self {
        Self::zeros(Shape::from(IxDyn(&[1, 1])), Device::default())
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    /// Multiplies two tensors.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The tensor to multiply with the current tensor.
    ///
    /// # Returns
    ///
    /// A new tensor containing the result of the multiplication.
    fn mul(self, rhs: Self) -> Self::Output {
        self.dot(&rhs)
    }
}

impl PartialEq for Tensor {
    /// Checks if two tensors are equal.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to compare with.
    ///
    /// # Returns
    ///
    /// `true` if the tensors are equal, `false` otherwise.
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

#[cfg(test)]
mod tests {
    use ndarray::IxDyn;

    use super::*;

    #[test]
    fn test_new() {
        let data = vec![1.0, 2.0, 3.0];
        let shape = Shape::from(IxDyn(&[3, 1]));
        let tensor = Tensor::new(data, shape);
        assert_eq!(tensor.data.shape(), &[3, 1]);
    }

    #[test]
    fn test_zeros() {
        let shape = Shape::from(IxDyn(&[2, 3]));
        let tensor = Tensor::zeros(shape, Device::Cpu);
        assert_eq!(tensor.data.shape(), &[2, 3]);
    }

    #[test]
    fn test_random() {
        let shape = Shape::from(IxDyn(&[2, 3]));
        let tensor = Tensor::random(shape);
        assert_eq!(tensor.data.shape(), &[2, 3]);
    }

    #[test]
    fn test_add() {
        let tensor1 = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3, 1])));
        let tensor2 = Tensor::new(vec![4.0, 5.0, 6.0], Shape::from(IxDyn(&[3, 1])));
        let result = tensor1.add(&tensor2);
        assert_eq!(result.data.shape(), &[3, 1]);
    }

    #[test]
    fn test_max() {
        let data = vec![1.0, 2.0, 3.0];
        let tensor = Tensor::new(data, Shape::from(IxDyn(&[3, 1])));
        assert_eq!(tensor.max(), 3.0);
    }

    #[test]
    fn test_mean() {
        let data = vec![1.0, 2.0, 3.0];
        let tensor = Tensor::new(data, Shape::from(IxDyn(&[3, 1])));
        assert_eq!(tensor.mean(), 2.0);
    }

    #[test]
    fn test_reshape() {
        let data = vec![1.0, 2.0, 3.0];
        let tensor = Tensor::new(data, Shape::from(IxDyn(&[3, 1])));
        let reshaped = tensor.reshape(IxDyn(&[1, 3]));
        assert_eq!(reshaped.data.shape(), &[1, 3]);
    }

    #[test]
    fn test_map() {
        let data = vec![1.0, 2.0, 3.0];
        let tensor = Tensor::new(data, Shape::from(IxDyn(&[3, 1])));
        let mapped = tensor.map(|x| x * 2.0);
        assert_eq!(mapped.data.shape(), &[3, 1]);
    }

    #[test]
    fn test_slice() {
        let data = vec![1.0, 2.0, 3.0];
        let tensor = Tensor::new(data, Shape::from(IxDyn(&[3, 1])));
        let sliced = tensor.slice(vec![0..2, 0..1]);
        assert_eq!(sliced.data.shape(), &[2, 1]);
    }

    #[test]
    fn test_matmul() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let tensor1 = Tensor::new(data1, Shape::from(IxDyn(&[2, 2])));
        let data2 = vec![5.0, 6.0, 7.0, 8.0];
        let tensor2 = Tensor::new(data2, Shape::from(IxDyn(&[2, 2])));
        let result = tensor1.dot(&tensor2);
        assert_eq!(result.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_transpose() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, Shape::from(IxDyn(&[2, 2])));
        let transposed = tensor.transpose();
        assert_eq!(transposed.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_shape() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, Shape::from(IxDyn(&[2, 2])));
        assert_eq!(tensor.shape().raw_dim().as_array_view().to_vec(), vec![2, 2]);
    }

    #[test]
    fn test_permute() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, Shape::from(IxDyn(&[2, 2])));
        let permuted = tensor.permute(vec![1, 0]);
        assert_eq!(permuted.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_sum_along_axis() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, Shape::from(IxDyn(&[2, 2])));
        let summed = tensor.sum_along_axis(1);
        assert_eq!(summed.data.shape(), &[2]);
    }

    #[test]
    fn test_mul_scalar() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, Shape::from(IxDyn(&[2, 2])));
        let multiplied = tensor.mul_scalar(2.0);
        assert_eq!(multiplied.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_pow() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, Shape::from(IxDyn(&[2, 2])));
        let powered = tensor.pow(2.0);
        assert_eq!(powered.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_div_scalar() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, Shape::from(IxDyn(&[2, 2])));
        let divided = tensor.div_scalar(2.0);
        assert_eq!(divided.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_sqrt() {
        let data = vec![1.0, 4.0, 9.0, 16.0];
        let tensor = Tensor::new(data, Shape::from(IxDyn(&[2, 2])));
        let sqrted = tensor.sqrt();
        assert_eq!(sqrted.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_add_scalar() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, Shape::from(IxDyn(&[2, 2])));
        let added = tensor.add_scalar(2.0);
        assert_eq!(added.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_div() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let tensor1 = Tensor::new(data1, Shape::from(IxDyn(&[2, 2])));
        let data2 = vec![2.0, 4.0, 6.0, 8.0];
        let tensor2 = Tensor::new(data2, Shape::from(IxDyn(&[2, 2])));
        let divided = tensor1.div(&tensor2);
        assert_eq!(divided.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_flatten() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, Shape::from(IxDyn(&[2, 2])));
        let flattened = tensor.flatten();
        assert_eq!(flattened.data.shape(), &[4]);
    }

    #[test]
    fn test_mean_axis() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, Shape::from(IxDyn(&[2, 2])));
        let meaned = tensor.mean_axis(1);
        assert_eq!(meaned.data.shape(), &[2]);
    }

    #[test]
    fn test_broadcast() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, Shape::from(IxDyn(&[2, 2])));
        let broadcasted = tensor.broadcast(Shape::from(IxDyn(&[2, 2, 2])));
        assert_eq!(broadcasted.data.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_normalize() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, Shape::from(IxDyn(&[2, 2])));
        let normalized = tensor.normalize(0.0, 1.0);
        assert_eq!(normalized.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_default() {
        let tensor = Tensor::default();
        assert_eq!(tensor.data.shape(), &[1, 1]);
    }

    #[test]
    fn test_add_noise() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut tensor = Tensor::new(data, Shape::from(IxDyn(&[2, 2])));
        tensor.add_noise(0.1);
        assert_eq!(tensor.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_argmax() {
        let data = vec![1.0, 3.0, 2.0, 4.0, 5.0, 0.0];
        let tensor = Tensor::new(data, Shape::from(IxDyn(&[2, 3])));

        let argmax = tensor.argmax(1);

        assert_eq!(argmax.data.shape(), &[2]);
        assert_eq!(argmax.data.iter().cloned().collect::<Vec<f32>>(), vec![1.0, 1.0]);
    }

    #[test]
    fn test_mul_operator() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let tensor1 = Tensor::new(data1, Shape::from(IxDyn(&[2, 2])));
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let tensor2 = Tensor::new(data2, Shape::from(IxDyn(&[2, 2])));
        let result = tensor1 * tensor2;
        assert_eq!(result.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_stack() {
        let tensor1 = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3])));
        let tensor2 = Tensor::new(vec![4.0, 5.0, 6.0], Shape::from(IxDyn(&[3])));
        let stacked = Tensor::stack(&[tensor1, tensor2]).unwrap();
        assert_eq!(stacked.shape().raw_dim().as_array_view().to_vec(), vec![2, 3]);
    }

    #[test]
    fn test_split_at() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, Shape::from(IxDyn(&[2, 2])));
        let (tensor1, tensor2) = tensor.split_at(1);
        assert_eq!(tensor1.data.shape(), &[1, 2]);
        assert_eq!(tensor2.data.shape(), &[1, 2]);
    }

    #[test]
    fn test_random_normal() {
        let tensor = Tensor::random_normal(Shape::from(IxDyn(&[2, 2])), 0.0, 1.0);
        assert_eq!(tensor.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_sub() {
        let tensor1 = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from(IxDyn(&[3])));
        let tensor2 = Tensor::new(vec![4.0, 5.0, 6.0], Shape::from(IxDyn(&[3])));
        let result = tensor1.sub(&tensor2);
        assert_eq!(result.data.shape(), &[3]);
    }
}

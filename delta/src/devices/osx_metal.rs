//! BSD 3-Clause License
//!
//! Copyright (c) 2024, The Delta Project Δ
//!
//! Redistribution and use in source and binary forms, with or without
//! modification, are permitted provided that the following conditions are met:
//!
//! 1. Redistributions of source code must retain the above copyright notice, this
//!    list of conditions and the following disclaimer.
//!
//! 2. Redistributions in binary form must reproduce the above copyright notice,
//!    this list of conditions and the following disclaimer in the documentation
//!    and/or other materials provided with the distribution.
//!
//! 3. Neither the name of the copyright holder nor the names of its
//!    contributors may be used to endorse or promote products derived from
//!    this software without specific prior written permission.
//!
//! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//! DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//! FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//! DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//! SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//! OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

pub use metal;
use ndarray::{Array, IxDyn, Shape};

use crate::common::Tensor;

/// Transfers the tensor to a Metal device.
///
/// # Arguments
///
/// * `metal_device` - The Metal device to transfer the tensor to.
///
/// # Returns
///
/// A new tensor with data stored on the Metal device.
pub fn to_device_metal(
    tensor: &Tensor,
    metal_device: &metal::Device,
    _queue: &metal::CommandQueue,
) -> Result<metal::Buffer, String> {
    // Create a Metal buffer for the tensor's data
    let tensor_size = tensor.data.len() * size_of::<f32>();
    let buffer =
        metal_device.new_buffer(tensor_size as u64, metal::MTLResourceOptions::StorageModeShared);

    // Copy the tensor's data into the Metal buffer
    unsafe {
        std::ptr::copy_nonoverlapping(
            tensor.data.as_slice().unwrap().as_ptr(),
            buffer.contents() as *mut f32,
            tensor.data.len(),
        );
    }

    Ok(buffer)
}

/// Transfers the tensor's data back from a Metal buffer to the CPU.
///
/// # Arguments
///
/// * `buffer` - The Metal buffer containing the tensor's data.
///
/// # Returns
///
/// A new `Tensor` instance with the data transferred from the Metal device.
pub fn from_device_metal(buffer: &metal::Buffer, shape: Shape<IxDyn>) -> Tensor {
    // Create a vector to hold the data
    let tensor_size = buffer.length() as usize / size_of::<f32>();
    let mut data = vec![0.0; tensor_size];

    // Copy the data from the Metal buffer
    unsafe {
        std::ptr::copy_nonoverlapping(
            buffer.contents() as *const f32,
            data.as_mut_ptr(),
            tensor_size,
        );
    }

    // Create a new tensor from the data
    Tensor::new(data, shape)
}

/// Creates a Metal device and command queue.
///
/// # Returns
///
/// A tuple containing the Metal device and command queue.
pub fn get_device_and_queue_metal() -> (metal::Device, metal::CommandQueue) {
    let device = metal::Device::system_default().expect("no device found");
    let queue = device.new_command_queue();
    (device, queue)
}

/// Creates a Metal compute pipeline state.
///
/// # Arguments
///
/// * `device` - The Metal device to create the pipeline state on.
/// * `shader_name` - The name of the Metal shader to use.
///
/// # Returns
///
/// A new `metal::ComputePipelineState` instance.
pub fn create_compute_pipeline(
    device: &metal::Device,
    shader_name: &str,
    function_name: &str,
) -> Result<metal::ComputePipelineState, String> {
    let library = device
        .new_library_with_source(shader_name, &metal::CompileOptions::new())
        .map_err(|e| format!("Failed to compile Metal shader: {:?}", e))?;
    let kernel = library
        .get_function(function_name, None)
        .map_err(|e| format!("Failed to get kernel function: {:?}", e))?;
    device
        .new_compute_pipeline_state_with_function(&kernel)
        .map_err(|e| format!("Failed to create compute pipeline: {:?}", e))
}

fn execute_tensor_operation_metal(
    tensor1: &Tensor,
    tensor2: &Tensor,
    operation: &str, // Name of the Metal shader function
    device: &metal::Device,
    queue: &metal::CommandQueue,
) -> Result<Tensor, String> {
    // Check if the tensors have the same shape
    // if tensor1.data.shape() != tensor2.data.shape() {
    //     return Err(format!(
    //         "Tensors must have the same shape, got {:?} and {:?}",
    //         tensor1.data.shape(),
    //         tensor2.data.shape()
    //     ));
    // }

    // Load shader and create compute pipeline
    let shader_source = include_str!("metal_shaders/tensor_ops.metal");
    let library = device
        .new_library_with_source(shader_source, &metal::CompileOptions::new())
        .map_err(|e| format!("Failed to compile Metal shader: {:?}", e))?;
    let kernel = library
        .get_function(operation, None)
        .map_err(|e| format!("Failed to get kernel function: {:?}", e))?;
    let pipeline_descriptor = metal::ComputePipelineDescriptor::new();
    pipeline_descriptor.set_compute_function(Some(&kernel));
    let pipeline_state = device
        .new_compute_pipeline_state(&pipeline_descriptor)
        .map_err(|e| format!("Failed to create compute pipeline: {:?}", e))?;

    // Ensure the tensor data is contiguous and properly aligned
    let tensor1_data = tensor1.data.as_slice().unwrap();
    let tensor2_data = tensor2.data.as_slice().unwrap();

    // Calculate buffer sizes
    let tensor1_buffer_size = (tensor1_data.len() * std::mem::size_of::<f32>()) as u64;
    let tensor2_buffer_size = (tensor2_data.len() * std::mem::size_of::<f32>()) as u64;

    // Create aligned buffers for input tensors
    let input1_buffer = device.new_buffer_with_data(
        tensor1_data.as_ptr() as *const _,
        tensor1_buffer_size,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let input2_buffer = device.new_buffer_with_data(
        tensor2_data.as_ptr() as *const _,
        tensor2_buffer_size,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Create an output buffer with the same size as the input tensors
    let output_buffer =
        device.new_buffer(tensor1_buffer_size, metal::MTLResourceOptions::StorageModeShared);

    // Create a buffer for tensor length
    let tensor_length = tensor1_data.len() as u32;
    let length_buffer = device.new_buffer_with_data(
        &tensor_length as *const u32 as *const _,
        size_of::<u32>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Check for buffer alignment and size correctness
    assert_eq!(input1_buffer.length(), tensor1_buffer_size);
    assert_eq!(input2_buffer.length(), tensor2_buffer_size);
    assert_eq!(output_buffer.length(), tensor1_buffer_size);

    // Create command buffer and encoder
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(input1_buffer.as_ref()), 0);
    encoder.set_buffer(1, Some(input2_buffer.as_ref()), 0);
    encoder.set_buffer(2, Some(output_buffer.as_ref()), 0);
    encoder.set_buffer(3, Some(length_buffer.as_ref()), 0);

    // Dispatch threads
    let tensor_length = tensor1.data.len() as u64;

    let threadgroup_size = metal::MTLSize {
        width: 256, // Typical value for threadgroup size
        height: 1,
        depth: 1,
    };

    // Compute the threadgroup count based on the actual data size
    let threadgroup_count = metal::MTLSize {
        width: ((tensor_length + threadgroup_size.width - 1) / threadgroup_size.width) as u64,
        height: 1,
        depth: 1,
    };
    assert!(
        threadgroup_count.width * threadgroup_size.width >= tensor_length,
        "Thread group size mismatch!"
    );

    // Dispatch threads
    encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
    encoder.end_encoding();

    // Commit and wait for execution
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Copy result back to CPU
    let output_data = unsafe {
        std::slice::from_raw_parts(output_buffer.contents() as *const f32, tensor1.data.len())
    }
    .to_vec();

    Ok(Tensor {
        data: Array::from_shape_vec(tensor1.shape(), output_data).unwrap(),
        device: tensor1.device.clone(),
    })
}

pub fn tensor_add_metal(
    tensor1: &Tensor,
    tensor2: &Tensor,
    device: &metal::Device,
    queue: &metal::CommandQueue,
) -> Result<Tensor, String> {
    // println!("tensor_add_metal");
    execute_tensor_operation_metal(tensor1, tensor2, "tensor_add", device, queue)
}

pub fn tensor_subtract_metal(
    tensor1: &Tensor,
    tensor2: &Tensor,
    device: &metal::Device,
    queue: &metal::CommandQueue,
) -> Result<Tensor, String> {
    // println!("tensor_subtract_metal");
    execute_tensor_operation_metal(tensor1, tensor2, "tensor_subtract", device, queue)
}

pub fn tensor_multiply_metal(
    tensor1: &Tensor,
    tensor2: &Tensor,
    device: &metal::Device,
    queue: &metal::CommandQueue,
) -> Result<Tensor, String> {
    // println!("tensor_multiply_metal");
    execute_tensor_operation_metal(tensor1, tensor2, "tensor_multiply", device, queue)
}

pub fn tensor_divide_metal(
    tensor1: &Tensor,
    tensor2: &Tensor,
    device: &metal::Device,
    queue: &metal::CommandQueue,
) -> Result<Tensor, String> {
    // println!("tensor_divide_metal");
    execute_tensor_operation_metal(tensor1, tensor2, "tensor_divide", device, queue)
}

#[cfg(test)]
mod tests {
    use ndarray::{IxDyn, Shape};

    use super::*;
    use crate::common::Tensor;

    #[test]
    fn test_tensor_add_metal() {
        let tensor1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::from(IxDyn(&[2, 2])));
        let tensor2 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], Shape::from(IxDyn(&[2, 2])));
        let device = metal::Device::system_default().unwrap();
        let queue = device.new_command_queue();
        let result = tensor_add_metal(&tensor1, &tensor2, &device, &queue).unwrap();
        assert_eq!(result.data.shape(), &[2, 2]);
        assert_eq!(
            result.data,
            Tensor::new(vec![6.0, 8.0, 10.0, 12.0], Shape::from(IxDyn(&[2, 2]))).data
        );
    }

    #[test]
    fn test_tensor_subtract_metal() {
        let tensor1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::from(IxDyn(&[2, 2])));
        let tensor2 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], Shape::from(IxDyn(&[2, 2])));
        let device = metal::Device::system_default().unwrap();
        let queue = device.new_command_queue();
        let result = tensor_subtract_metal(&tensor1, &tensor2, &device, &queue).unwrap();
        assert_eq!(result.data.shape(), &[2, 2]);
        assert_eq!(
            result.data,
            Tensor::new(vec![-4.0, -4.0, -4.0, -4.0], Shape::from(IxDyn(&[2, 2]))).data
        );
    }

    #[test]
    fn test_tensor_multiply_metal() {
        let tensor1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::from(IxDyn(&[2, 2])));
        let tensor2 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], Shape::from(IxDyn(&[2, 2])));
        let device = metal::Device::system_default().unwrap();
        let queue = device.new_command_queue();
        let result = tensor_multiply_metal(&tensor1, &tensor2, &device, &queue).unwrap();
        assert_eq!(result.data.shape(), &[2, 2]);
        assert_eq!(
            result.data,
            Tensor::new(vec![5.0, 12.0, 21.0, 32.0], Shape::from(IxDyn(&[2, 2]))).data
        );
    }

    #[test]
    fn test_tensor_divide_metal() {
        let tensor1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::from(IxDyn(&[2, 2])));
        let tensor2 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], Shape::from(IxDyn(&[2, 2])));
        let device = metal::Device::system_default().unwrap();
        let queue = device.new_command_queue();
        let result = tensor_divide_metal(&tensor1, &tensor2, &device, &queue).unwrap();
        assert_eq!(result.data.shape(), &[2, 2]);
        assert_eq!(
            result.data,
            Tensor::new(vec![0.2, 0.33333334, 0.42857146, 0.5], Shape::from(IxDyn(&[2, 2]))).data
        );
    }
}

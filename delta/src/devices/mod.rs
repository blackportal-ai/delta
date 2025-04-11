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

#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod osx_metal;

/// Represents a computational device that can be utilized for machine learning operations.
///
/// # Variants
///
/// - `Cpu`: The central processing unit (CPU), available on all platforms.
///
/// - `Metal`: Represents a Metal-based device for macOS. This variant is only available
///   when compiled on macOS with the `metal` feature enabled. It includes:
///     - `device`: A `metal::Device` object representing the Metal device.
///     - `queue`: A `metal::CommandQueue` used for issuing commands to the device.
///
/// # Future Work
///
/// The following device types are planned for future support but are currently not implemented:
///
/// - `Cuda`: Support for NVIDIA GPUs using the CUDA API.
/// - `OpenCL`: Cross-platform GPU support using OpenCL.
/// - `OpenCLCuda`: A specialized OpenCL implementation for NVIDIA GPUs.
/// - `Vulkan`: Cross-platform GPU support using the Vulkan API.
/// - `DirectX12`: Windows-based GPU support via DirectX 12.
/// - `Sycl`: Intel's SYCL API for heterogeneous computing.
/// - `Tpu`: Support for Google's TPUs (Tensor Processing Units).
/// - `WebGpu`: Lightweight GPU computing in browsers or embedded environments using WebGPU.
///
/// These variants are currently commented out in the codebase and will be introduced as the
/// framework evolves to support additional hardware platforms.
#[derive(Debug, Clone)]
pub enum Device {
    /// The central processing unit (CPU).
    Cpu,

    /// A Metal-based device for macOS.
    ///
    /// This variant includes:
    /// - `device`: A Metal device object.
    /// - `queue`: A command queue for issuing commands to the device.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal { device: metal::Device, queue: metal::CommandQueue },
    // Placeholder for future device types.
    // Cuda,
    // OpenCL,
    // OpenCLCuda,
    // Vulkan,
    // DirectX12,
    // Sycl,
    // Tpu,
    // WebGpu,
}

impl Default for Device {
    fn default() -> Self {
        Self::Cpu
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "CPU"),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Device::Metal { .. } => write!(f, "Metal"),
        }
    }
}

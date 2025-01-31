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

use std::collections::HashSet;
use std::fs;
use std::fs::File;
use std::future::Future;
use std::io::Read;
use std::path::Path;
use std::pin::Pin;

use flate2::read::GzDecoder;
use log::debug;
use ndarray::{IxDyn, Shape};
use tar::Archive;

use crate::deep_learning::dataset::{Dataset, DatasetOps};
use crate::deep_learning::tensor_ops::Tensor;
use crate::devices::Device;
use crate::get_workspace_dir;

/// A struct representing the CIFAR-100 dataset.
pub struct Cifar100Dataset {
    train: Option<Dataset>,
    test: Option<Dataset>,
    val: Option<Dataset>,
}

impl Cifar100Dataset {
    const CIFAR100_URL: &'static str = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz";
    const CIFAR100_TRAIN_FILE: &'static str = "train.bin";
    const CIFAR100_TEST_FILE: &'static str = "test.bin";
    const CIFAR100_IMAGE_SIZE: usize = 32;
    const CIFAR100_NUM_CLASSES: usize = 100;
    const TRAIN_EXAMPLES: usize = 50_000;
    const TEST_EXAMPLES: usize = 10_000;

    /// Downloads and extracts the CIFAR-100 dataset.
    ///
    /// This function downloads the CIFAR-100 dataset from the specified URL
    /// and extracts it to the cache directory.
    async fn download_and_extract() {
        let workspace_dir = get_workspace_dir();
        let cache_path = format!("{}/.cache/dataset/cifar100/", workspace_dir.display());
        let tarball_path = format!("{}cifar-100-binary.tar.gz", cache_path);

        if !Path::new(&tarball_path).exists() {
            debug!("Downloading CIFAR-100 dataset from {}", Self::CIFAR100_URL);
            let response =
                reqwest::get(Self::CIFAR100_URL).await.expect("Failed to download CIFAR-100");
            let data = response.bytes().await.expect("Failed to read CIFAR-100 dataset");
            fs::create_dir_all(cache_path.clone()).unwrap();
            fs::write(&tarball_path, data).unwrap();
        }

        let tar_gz = File::open(&tarball_path).unwrap();
        let tar = GzDecoder::new(tar_gz);
        let mut archive = Archive::new(tar);
        let mut seen_files = HashSet::new();

        for entry in archive.entries().unwrap() {
            let mut entry = entry.unwrap();
            let path = entry.path().unwrap().into_owned();
            let file_name = path.file_name().unwrap().to_string_lossy().to_string();

            if path.is_dir() || path.extension().is_none_or(|ext| ext != "bin") {
                continue;
            }

            if seen_files.insert(file_name.clone()) {
                let full_path = format!("{}{}", cache_path, file_name);
                if let Some(parent) = Path::new(&full_path).parent() {
                    fs::create_dir_all(parent).unwrap();
                }
                entry.unpack(&full_path).unwrap();
                debug!("Unarchived file: {}", full_path);
            }
        }
    }

    /// Parses a CIFAR-100 binary file and returns the images and labels as vectors of `f32`.
    ///
    /// Each record in CIFAR-100 is structured as follows:
    /// `[fine_label(1 byte), coarse_label(1 byte), 32x32x3 image(3072 bytes)]`.
    /// The coarse label is ignored, and only the fine label is used for a one-hot label vector.
    ///
    /// # Arguments
    /// * `file_path` - The path to the CIFAR-100 binary file.
    /// * `num_examples` - The number of examples in the file.
    ///
    /// # Returns
    /// A tuple containing:
    /// - `Vec<f32>`: Flattened and normalized image data.
    /// - `Vec<f32>`: One-hot-encoded labels for each example.
    fn parse_file(file_path: &str, num_examples: usize) -> (Vec<f32>, Vec<f32>) {
        let mut file = File::open(file_path).expect("Failed to open CIFAR-100 file");
        let mut buffer =
            vec![0u8; 1 + 1 + Self::CIFAR100_IMAGE_SIZE * Self::CIFAR100_IMAGE_SIZE * 3];

        let mut images =
            vec![0.0; num_examples * Self::CIFAR100_IMAGE_SIZE * Self::CIFAR100_IMAGE_SIZE * 3];
        let mut labels = vec![0.0; num_examples * Self::CIFAR100_NUM_CLASSES];

        for i in 0..num_examples {
            file.read_exact(&mut buffer).expect("Failed to read CIFAR-100 example");
            let fine_label = buffer[0] as usize;

            labels[i * Self::CIFAR100_NUM_CLASSES + fine_label] = 1.0; // One-hot encode

            for (j, &pixel) in buffer[2..].iter().enumerate() {
                images[i * Self::CIFAR100_IMAGE_SIZE * Self::CIFAR100_IMAGE_SIZE * 3 + j] =
                    pixel as f32 / 255.0; // Normalize to [0, 1]
            }
        }

        (images, labels)
    }

    /// Loads the CIFAR-100 dataset from a specific file (train or test).
    ///
    /// # Arguments
    /// * `file` - The file name to load (e.g., `train.bin`, `test.bin`).
    /// * `total_examples` - The total number of examples in the specified file.
    ///
    /// # Returns
    /// A `Dataset` containing the loaded images and labels.
    fn load_data(file: &str, total_examples: usize) -> Dataset {
        let (images, labels) = Self::parse_file(
            &format!("{}/.cache/dataset/cifar100/{}", get_workspace_dir().display(), file),
            total_examples,
        );

        Dataset::new(
            Tensor::new(
                images,
                Shape::from(IxDyn(&[
                    total_examples,
                    Self::CIFAR100_IMAGE_SIZE,
                    Self::CIFAR100_IMAGE_SIZE,
                    3,
                ])),
            ),
            Tensor::new(labels, Shape::from(IxDyn(&[total_examples, Self::CIFAR100_NUM_CLASSES]))),
        )
    }

    /// Splits the training data into training and validation datasets.
    ///
    /// # Arguments
    ///
    /// * `validation_split` - The fraction of the training data to use for validation.
    ///
    /// # Returns
    ///
    /// A tuple containing the training and validation datasets.
    fn split_train_validation(&mut self, validation_split: f32) {
        if let Some(train_data) = &self.train {
            let total_samples = train_data.inputs.shape().raw_dim()[0];
            let validation_size = (total_samples as f32 * validation_split).round() as usize;
            let train_size = total_samples - validation_size;

            let (train_inputs, val_inputs) = train_data.inputs.split_at(train_size);
            let (train_labels, val_labels) = train_data.labels.split_at(train_size);

            self.train = Some(Dataset::new(train_inputs, train_labels));
            self.val = Some(Dataset::new(val_inputs, val_labels));
        } else {
            panic!("Training dataset not loaded!");
        }
    }
}

impl DatasetOps for Cifar100Dataset {
    type LoadFuture = Pin<Box<dyn Future<Output = Self> + Send>>;

    /// Loads the training dataset for CIFAR-100.
    ///
    /// # Returns
    /// A future that resolves to the `Cifar100Dataset` with the training dataset loaded.
    fn load_train() -> Self::LoadFuture {
        Box::pin(async {
            Self::download_and_extract().await;
            let train_data = Self::load_data(Self::CIFAR100_TRAIN_FILE, Self::TRAIN_EXAMPLES);
            Cifar100Dataset { train: Some(train_data), test: None, val: None }
        })
    }

    /// Loads the test dataset.
    ///
    /// # Returns
    /// A future that resolves to the `Cifar100Dataset` with the test dataset loaded.
    fn load_test() -> Self::LoadFuture {
        Box::pin(async {
            Self::download_and_extract().await;
            let test_data = Self::load_data(Self::CIFAR100_TEST_FILE, Self::TEST_EXAMPLES);
            Cifar100Dataset { train: None, test: Some(test_data), val: None }
        })
    }

    fn load_val() -> Self::LoadFuture {
        Box::pin(async {
            Self::download_and_extract().await;
            let train_data = Self::load_data(Self::CIFAR100_TRAIN_FILE, Self::TRAIN_EXAMPLES);
            let mut dataset = Cifar100Dataset { train: Some(train_data), test: None, val: None };
            dataset.split_train_validation(0.2);
            dataset
        })
    }

    /// Normalizes the dataset.
    ///
    /// # Arguments
    /// * `min` - The minimum value for normalization.
    /// * `max` - The maximum value for normalization.
    fn normalize(&mut self, min: f32, max: f32) {
        let _ = max;
        let _ = min;
        todo!()
    }

    /// Adds noise to the dataset.
    ///
    /// # Arguments
    /// * `noise_level` - The level of noise to add.
    fn add_noise(&mut self, noise_level: f32) {
        let _ = noise_level;
        todo!()
    }

    /// Gets the number of examples in the dataset.
    ///
    /// # Returns
    /// The number of examples in the dataset.
    fn len(&self) -> usize {
        if let Some(ref train) = self.train {
            train.inputs.data.shape()[0]
        } else if let Some(ref test) = self.test {
            test.inputs.data.shape()[0]
        } else {
            0
        }
    }

    /// Gets a batch of dataset from the dataset.
    ///
    /// # Arguments
    /// * `batch_idx` - The index of the batch to get.
    /// * `batch_size` - The size of the batch to get.
    ///
    /// # Returns
    /// A tuple containing the input and label tensors for the batch.
    fn get_batch(&self, batch_idx: usize, batch_size: usize) -> (Tensor, Tensor) {
        let dataset = match (self.train.as_ref(), self.test.as_ref()) {
            (Some(train), _) => train,
            (_, Some(test)) => test,
            _ => panic!("Dataset not loaded!"),
        };

        let total_samples = dataset.inputs.shape().raw_dim()[0];
        let start_idx = batch_idx * batch_size;
        let end_idx = start_idx + batch_size;

        if start_idx >= total_samples {
            panic!("Batch index {} out of range. Total samples: {}", batch_idx, total_samples);
        }

        let adjusted_end_idx = end_idx.min(total_samples);

        let inputs_batch = dataset.inputs.slice(vec![
            start_idx..adjusted_end_idx,
            0..Self::CIFAR100_IMAGE_SIZE,
            0..Self::CIFAR100_IMAGE_SIZE,
            0..3,
        ]);

        let labels_batch =
            dataset.labels.slice(vec![start_idx..adjusted_end_idx, 0..Self::CIFAR100_NUM_CLASSES]);

        (inputs_batch, labels_batch)
    }

    /// Calculates the loss between the predicted outputs and the true targets.
    ///
    /// # Arguments
    /// * `outputs` - The predicted outputs from the model (logits or probabilities).
    /// * `targets` - The true target values (one-hot encoded).
    ///
    /// # Returns
    /// The calculated loss as a `f32` value.
    fn loss(&self, outputs: &Tensor, targets: &Tensor) -> f32 {
        let outputs_data = outputs.data.clone();
        let targets_data = targets.data.clone();

        let batch_size = targets.shape().raw_dim()[0];
        let num_classes = targets.shape().raw_dim()[1];

        let mut loss = 0.0;

        for i in 0..batch_size {
            for j in 0..num_classes {
                let target = targets_data[i * num_classes + j];
                let predicted = outputs_data[i * num_classes + j].max(1e-15);
                loss -= target * predicted.ln();
            }
        }

        loss / batch_size as f32
    }

    /// Calculates the gradient of the loss with respect to the predicted outputs.
    ///
    /// # Arguments
    /// * `outputs` - The predicted outputs from the model (probabilities).
    /// * `targets` - The true target values (one-hot encoded).
    ///
    /// # Returns
    /// A `Tensor` containing the gradients of the loss with respect to the outputs.
    fn loss_grad(&self, outputs: &Tensor, targets: &Tensor) -> Tensor {
        let outputs_data = outputs.data.iter().cloned().collect::<Vec<f32>>();
        let targets_data = targets.data.iter().cloned().collect::<Vec<f32>>();

        let batch_size = targets.shape().raw_dim()[0];
        let num_classes = targets.shape().raw_dim()[1];
        assert_eq!(
            outputs.shape().raw_dim(),
            targets.shape().raw_dim(),
            "Outputs and targets must have the same shape"
        );

        let mut grad_data = vec![0.0; batch_size * num_classes];

        for i in 0..batch_size {
            for j in 0..num_classes {
                let target = targets_data[i * num_classes + j];
                let predicted = outputs_data[i * num_classes + j];
                grad_data[i * num_classes + j] = (predicted - target) / batch_size as f32;
            }
        }

        Tensor::new(grad_data, outputs.shape().clone())
    }

    /// Shuffles the dataset.
    fn shuffle(&mut self) {}

    /// Clones the dataset.
    ///
    /// # Returns
    /// A new `Cifar100Dataset` instance with cloned data references.
    fn clone(&self) -> Self {
        Self { train: self.train.clone(), test: self.test.clone(), val: self.val.clone() }
    }

    /// Transfers the dataset to the specified device.
    ///
    /// # Arguments
    /// * `device` - The device to transfer the dataset to.
    ///
    /// # Returns
    /// A `Result` containing the dataset on the specified device.
    fn to_device(&mut self, device: Device) -> Result<(), String> {
        if let Some(train) = self.train.as_mut() {
            train.to_device(&device);
        }

        if let Some(test) = self.test.as_mut() {
            test.to_device(&device);
        }

        if let Some(val) = self.val.as_mut() {
            val.to_device(&device);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Dimension;
    use serial_test::serial;

    use super::*;

    fn setup() {
        let workspace_dir = get_workspace_dir();
        let cache_path = format!("{}/.cache/dataset/cifar100", workspace_dir.display());
        if Path::new(&cache_path).exists() {
            fs::remove_dir_all(&cache_path).expect("Failed to delete cache directory");
        }
    }

    #[tokio::test]
    #[serial]
    async fn test_download_and_extract() {
        setup();
        Cifar100Dataset::download_and_extract().await;
        let workspace_dir = get_workspace_dir();
        let cache_path = format!(
            "{}/.cache/dataset/cifar100/{}",
            workspace_dir.display(),
            Cifar100Dataset::CIFAR100_TRAIN_FILE
        );
        assert!(
            Path::new(&cache_path).exists(),
            "CIFAR-100 dataset should be downloaded and extracted"
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_parse_file() {
        setup();
        Cifar100Dataset::download_and_extract().await;

        let workspace_dir = get_workspace_dir();
        let cache_path = format!(
            "{}/.cache/dataset/cifar100/{}",
            workspace_dir.display(),
            Cifar100Dataset::CIFAR100_TRAIN_FILE
        );

        let (images, labels) = Cifar100Dataset::parse_file(&cache_path, 50000);
        assert_eq!(images.len(), 50000 * 32 * 32 * 3);
        assert_eq!(labels.len(), 50000 * 100);
    }

    #[tokio::test]
    #[serial]
    async fn test_load_data() {
        setup();
        Cifar100Dataset::download_and_extract().await;

        let dataset = Cifar100Dataset::load_data(Cifar100Dataset::CIFAR100_TRAIN_FILE, 50000);
        assert_eq!(dataset.inputs.shape().raw_dim().as_array_view().to_vec(), &[50000, 32, 32, 3]);
        assert_eq!(dataset.labels.shape().raw_dim().as_array_view().to_vec(), &[50000, 100]);
    }

    #[tokio::test]
    #[serial]
    async fn test_load_train() {
        let dataset = Cifar100Dataset::load_train().await;
        assert!(dataset.train.is_some(), "Training dataset should be loaded");
    }

    #[tokio::test]
    #[serial]
    async fn test_load_test() {
        let dataset = Cifar100Dataset::load_test().await;
        assert!(dataset.test.is_some(), "Test dataset should be loaded");
    }
}

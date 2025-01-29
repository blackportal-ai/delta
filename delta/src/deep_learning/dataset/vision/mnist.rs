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

use std::fs::File;
use std::future::Future;
use std::io::{self, Read};
use std::path::Path;
use std::pin::Pin;

use flate2::read::GzDecoder;
use log::debug;
use ndarray::{IxDyn, Shape};
use rand::seq::SliceRandom;
use reqwest;
use tokio::fs as async_fs;

use crate::deep_learning::dataset::{Dataset, DatasetOps};
use crate::deep_learning::tensor_ops::Tensor;
use crate::get_workspace_dir;

/// A struct representing the MNIST dataset.
pub struct MnistDataset {
    train: Option<Dataset>,
    test: Option<Dataset>,
    val: Option<Dataset>,
}

impl MnistDataset {
    const MNIST_URL: &'static str = "https://storage.googleapis.com/cvdf-datasets/mnist";
    const MNIST_TRAIN_DATA_FILENAME: &'static str = "train-images-idx3-ubyte.gz";
    const MNIST_TRAIN_LABELS_FILENAME: &'static str = "train-labels-idx1-ubyte.gz";
    const MNIST_TEST_DATA_FILENAME: &'static str = "t10k-images-idx3-ubyte.gz";
    const MNIST_TEST_LABELS_FILENAME: &'static str = "t10k-labels-idx1-ubyte.gz";
    const MNIST_IMAGE_SIZE: usize = 28;
    const MNIST_NUM_CLASSES: usize = 10;
    const TRAIN_EXAMPLES: usize = 60_000;
    const TEST_EXAMPLES: usize = 10_000;

    /// Asynchronously loads the MNIST dataset.
    ///
    /// This function downloads and parses the MNIST dataset, either for training or testing,
    /// depending on the `is_train` parameter.
    ///
    /// # Arguments
    ///
    /// * `is_train` - A boolean indicating whether to load the training dataset (`true`) or the test dataset (`false`).
    ///
    /// # Returns
    ///
    /// A `Result` containing the loaded `Dataset` or an error message if the operation fails.
    async fn load_data(is_train: bool) -> Result<Dataset, String> {
        let (data_filename, labels_filename, num_examples) = if is_train {
            (
                Self::MNIST_TRAIN_DATA_FILENAME,
                Self::MNIST_TRAIN_LABELS_FILENAME,
                Self::TRAIN_EXAMPLES,
            )
        } else {
            (Self::MNIST_TEST_DATA_FILENAME, Self::MNIST_TEST_LABELS_FILENAME, Self::TEST_EXAMPLES)
        };

        let data_bytes = Self::get_bytes_data(data_filename).await?;
        let labels_bytes = Self::get_bytes_data(labels_filename).await?;

        let data = Self::parse_images(&data_bytes, num_examples)?;
        let labels = Self::parse_labels(&labels_bytes, num_examples)?;

        Ok(Dataset::new(data, labels))
    }

    /// Parse the images from the MNIST dataset
    ///
    /// # Arguments
    /// * `dataset` - The dataset to parse
    /// * `num_images` - The number of images to parse
    ///
    /// # Returns
    /// A tensor containing the parsed images
    fn parse_images(data: &[u8], num_images: usize) -> Result<Tensor, String> {
        if data.len() < 16 {
            return Err("Invalid MNIST image dataset file: too short".into());
        }

        let image_data = &data[16..];
        let num_pixels = Self::MNIST_IMAGE_SIZE * Self::MNIST_IMAGE_SIZE;
        let mut tensor_data = vec![0.0; num_images * num_pixels];

        for i in 0..num_images {
            let start = i * num_pixels;
            let end = start + num_pixels;

            if end > image_data.len() {
                return Err("Image dataset file is incomplete".into());
            }

            for (j, &pixel) in image_data[start..end].iter().enumerate() {
                tensor_data[i * num_pixels + j] = pixel as f32 / 255.0; // Normalize to [0, 1]
            }
        }

        Ok(Tensor::new(
            tensor_data,
            Shape::from(IxDyn(&[num_images, Self::MNIST_IMAGE_SIZE, Self::MNIST_IMAGE_SIZE, 1])),
        ))
    }

    /// Parse the labels from the MNIST dataset
    ///
    /// # Arguments
    /// * `dataset` - The dataset to parse
    /// * `num_labels` - The number of labels to parse
    ///
    /// # Returns
    /// A tensor containing the parsed labels
    fn parse_labels(data: &[u8], num_labels: usize) -> Result<Tensor, String> {
        if data.len() < 8 {
            return Err("Invalid MNIST label dataset file: too short".into());
        }

        let label_data = &data[8..];
        let mut tensor_data = vec![0.0; num_labels * Self::MNIST_NUM_CLASSES];

        for (i, &label) in label_data.iter().enumerate().take(num_labels) {
            if label as usize >= Self::MNIST_NUM_CLASSES {
                return Err(format!("Invalid label value: {}", label));
            }
            tensor_data[i * Self::MNIST_NUM_CLASSES + label as usize] = 1.0;
        }

        Ok(Tensor::new(tensor_data, Shape::from(IxDyn(&[num_labels, Self::MNIST_NUM_CLASSES]))))
    }

    /// Download and decompress a file from the MNIST dataset
    ///
    /// # Arguments
    /// * `filename` - The name of the file to download
    ///
    /// # Returns
    /// A vector containing the decompressed dataset
    async fn get_bytes_data(filename: &str) -> Result<Vec<u8>, String> {
        let workspace_dir = get_workspace_dir();
        let file_path = format!("{}/.cache/dataset/mnist/{}", workspace_dir.display(), filename);

        if Path::new(&file_path).exists() {
            debug!("Using cached file: {}", &file_path);
            return Self::decompress_gz(&file_path).map_err(|e| e.to_string());
        }

        let url = format!("{}/{}", Self::MNIST_URL, filename);
        debug!("Downloading MNIST dataset from {}", &url);

        let response = reqwest::get(&url).await.map_err(|e| e.to_string())?;
        if !response.status().is_success() {
            return Err(format!(
                "Request failed with status: {} for URL: {}",
                response.status(),
                url
            ));
        }
        let compressed_data = response.bytes().await.map_err(|e| e.to_string())?;

        async_fs::create_dir_all(format!("{}/.cache/dataset/mnist", workspace_dir.display()))
            .await
            .map_err(|e| e.to_string())?;
        async_fs::write(&file_path, &compressed_data).await.map_err(|e| e.to_string())?;

        Self::decompress_gz(&file_path).map_err(|e| e.to_string())
    }

    /// Decompresses a gzip file.
    ///
    /// This function opens a gzip file, decompresses its contents, and returns the decompressed data as a `Vec<u8>`.
    ///
    /// # Arguments
    ///
    /// * `file_path` - A string slice that holds the path to the gzip file.
    ///
    /// # Returns
    ///
    /// A `Result` containing a vector of bytes with the decompressed data, or an `io::Error` if the operation fails.
    fn decompress_gz(file_path: &str) -> io::Result<Vec<u8>> {
        let file = File::open(file_path)?;
        let mut decoder = GzDecoder::new(file);
        let mut decompressed_data = Vec::new();
        decoder.read_to_end(&mut decompressed_data)?;
        debug!("Decompressed file: {}", file_path);
        Ok(decompressed_data)
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

impl DatasetOps for MnistDataset {
    type LoadFuture = Pin<Box<dyn Future<Output = MnistDataset> + Send>>;

    /// Loads the training dataset.
    ///
    /// # Returns
    ///
    /// A future that resolves to the `MnistDataset` with the MNIST dataset loaded.
    fn load_train() -> Self::LoadFuture {
        Box::pin(async {
            match MnistDataset::load_data(true).await {
                Ok(train_data) => MnistDataset { train: Some(train_data), test: None, val: None },
                Err(err) => panic!("Failed to load train dataset: {}", err),
            }
        })
    }

    /// Loads the test dataset.
    ///
    /// # Returns
    ///
    /// A future that resolves to the `MnistDataset` with the MNIST dataset loaded.
    fn load_test() -> Self::LoadFuture {
        Box::pin(async {
            match MnistDataset::load_data(false).await {
                Ok(test_data) => MnistDataset { train: None, test: Some(test_data), val: None },
                Err(err) => panic!("Failed to load test dataset: {}", err),
            }
        })
    }

    /// Loads the validation dataset.
    ///
    /// # Returns
    ///
    /// A future that resolves to the `MnistDataset` with the MNIST dataset loaded.
    fn load_val() -> Self::LoadFuture {
        Box::pin(async {
            match MnistDataset::load_data(true).await {
                Ok(train_data) => {
                    let mut dataset =
                        MnistDataset { train: Some(train_data), test: None, val: None };
                    dataset.split_train_validation(0.2);
                    dataset
                }
                Err(err) => panic!("Failed to load train dataset: {}", err),
            }
        })
    }

    /// Normalizes the dataset.
    ///
    /// # Arguments
    ///
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
    ///
    /// * `noise_level` - The level of noise to add.
    fn add_noise(&mut self, noise_level: f32) {
        let _ = noise_level;
        todo!()
    }

    /// Returns the number of samples in the dataset.
    ///
    /// # Returns
    ///
    /// The number of samples in the dataset.
    fn len(&self) -> usize {
        self.train
            .as_ref()
            .or(self.test.as_ref())
            .map(|ds| ds.inputs.shape().raw_dim()[0])
            .unwrap_or(0)
    }

    /// Get a batch of dataset from the dataset
    ///
    /// # Arguments
    /// * `batch_idx` - The index of the batch to get
    /// * `batch_size` - The size of the batch to get
    ///
    /// # Returns
    ///
    /// A tuple containing the input and label tensors for the batch
    fn get_batch(&self, batch_idx: usize, batch_size: usize) -> (Tensor, Tensor) {
        // Determine which dataset to use: train or test
        let dataset = match (self.train.as_ref(), self.test.as_ref()) {
            (Some(train), _) => train,          // Use the train dataset if available
            (_, Some(test)) => test,            // Otherwise, use the test dataset
            _ => panic!("Dataset not loaded!"), // Panic if neither dataset is loaded
        };

        // Get the total number of samples in the dataset
        let total_samples = dataset.inputs.shape().raw_dim()[0];

        // Calculate the start and end indices for the batch
        let start_idx = batch_idx * batch_size;
        let end_idx = start_idx + batch_size;

        // Ensure the start index is within range
        if start_idx >= total_samples {
            panic!("Batch index {} out of range. Total samples: {}", batch_idx, total_samples);
        }

        // Adjust the end index if it exceeds the total samples
        let adjusted_end_idx = end_idx.min(total_samples);

        // Slice the input tensor for the batch
        let inputs_batch = dataset.inputs.slice(vec![
            start_idx..adjusted_end_idx, // Batch range along the sample dimension
            0..28,                       // Full range for the image height
            0..28,                       // Full range for the image width
            0..1,                        // Full range for the channels (grayscale)
        ]);

        // Slice the label tensor for the batch
        let labels_batch = dataset.labels.slice(vec![
            start_idx..adjusted_end_idx, // Batch range along the sample dimension
            0..10,                       // Full range for the classes (one-hot encoding)
        ]);

        // Return the inputs and labels for the batch
        (inputs_batch, labels_batch)
    }

    /// Calculates the loss between the predicted outputs and the true targets.
    ///
    /// # Arguments
    ///
    /// * `outputs` - The predicted outputs from the model (logits or probabilities).
    /// * `targets` - The true target values (one-hot encoded).
    ///
    /// # Returns
    ///
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
    ///
    /// * `outputs` - The predicted outputs from the model (probabilities).
    /// * `targets` - The true target values (one-hot encoded).
    ///
    /// # Returns
    ///
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
    fn shuffle(&mut self) {
        let shuffle_data = |dataset: &mut Dataset| {
            let num_samples = dataset.inputs.shape().raw_dim()[0];
            let mut indices: Vec<usize> = (0..num_samples).collect();
            indices.shuffle(&mut rand::thread_rng());

            dataset.inputs = dataset.inputs.take(&indices);
            dataset.labels = dataset.labels.take(&indices);
        };

        if let Some(train) = &mut self.train {
            shuffle_data(train);
        }

        if let Some(test) = &mut self.test {
            shuffle_data(test);
        }
    }

    /// Clones the dataset.
    ///
    /// # Returns
    ///
    /// A new `MnistDataset` instance that is a clone of the current instance.
    fn clone(&self) -> Self {
        Self { train: self.train.clone(), test: self.test.clone(), val: self.val.clone() }
    }

    /// Transfers the dataset to the specified device.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to transfer the dataset to.
    ///
    /// # Returns
    ///
    /// A `Result` containing the dataset on the specified device.
    fn to_device(&mut self, device: crate::devices::Device) -> Result<(), String> {
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
    use std::fs;

    use serial_test::serial;
    use tokio::runtime::Runtime;

    use super::*;

    fn setup() {
        let workspace_dir = get_workspace_dir();
        let cache_path = format!("{}/.cache/dataset/mnist", workspace_dir.display());
        if Path::new(&cache_path).exists() {
            fs::remove_dir_all(&cache_path).expect("Failed to delete cache directory");
        }
    }

    #[test]
    #[serial]
    fn test_download_and_extract() {
        setup();
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            let _ = MnistDataset::load_data(true).await;
            let workspace_dir = get_workspace_dir();
            let cache_path = format!("{}/.cache/dataset/mnist", workspace_dir.display());
            assert!(
                Path::new(&cache_path).exists(),
                "MNIST dataset should be downloaded and extracted"
            );
        });
    }

    // #[tokio::test]
    // async fn test_parse_images() {
    //     // Ensure the dataset is downloaded before parsing
    //     let _ = MnistDataset::get_bytes_data(MnistDataset::MNIST_TRAIN_DATA_FILENAME)
    //         .await
    //         .expect("Failed to get image data");

    //     let data_bytes = MnistDataset::get_bytes_data(MnistDataset::MNIST_TRAIN_DATA_FILENAME)
    //         .await
    //         .expect("Failed to get image data");

    //     let images =
    //         MnistDataset::parse_images(&data_bytes, 60000).expect("Failed to parse images");
    //     assert_eq!(images.data.len(), 60000 * 28 * 28, "Images should have the correct length");
    // }

    #[tokio::test]
    #[serial]
    async fn test_parse_labels() {
        let labels_bytes = MnistDataset::get_bytes_data(MnistDataset::MNIST_TRAIN_LABELS_FILENAME)
            .await
            .expect("Failed to get label data");

        let labels =
            MnistDataset::parse_labels(&labels_bytes, 60000).expect("Failed to parse labels");

        assert_eq!(labels.data.len(), 60000 * 10, "Labels should have the correct length");
    }

    #[tokio::test]
    #[serial]
    async fn test_load_train() {
        let dataset = MnistDataset::load_train().await;
        assert!(dataset.train.is_some(), "Training dataset should be loaded");
    }

    #[tokio::test]
    #[serial]
    async fn test_load_test() {
        let dataset = MnistDataset::load_test().await;
        assert!(dataset.test.is_some(), "Test dataset should be loaded");
    }
}

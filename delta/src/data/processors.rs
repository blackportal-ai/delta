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

use csv::ReaderBuilder;
use log::{debug, error, info};
use rand::seq::SliceRandom;
use reqwest::blocking::get;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::mpsc::{Sender, channel};
use std::thread::{self, JoinHandle};

/// Configuration for CSV dataset processing
#[derive(Debug)]
pub struct CsvDatasetConfig {
    url: String,
    has_headers: bool,
    feature_col: usize,
    target_col: usize,
    train_ratio: f64,
    output_dir: PathBuf,
    target_mapping: Option<fn(&str) -> Result<u8, Box<dyn Error + Send + Sync + 'static>>>,
    num_threads: usize,
}

impl CsvDatasetConfig {
    pub fn new(url: &str, has_headers: bool, feature_col: usize, target_col: usize) -> Self {
        CsvDatasetConfig {
            url: url.to_string(),
            has_headers,
            feature_col,
            target_col,
            train_ratio: 0.7,
            output_dir: PathBuf::from("."),
            target_mapping: None,
            num_threads: thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
        }
    }

    pub fn with_headers(mut self, has_headers: bool) -> Self {
        self.has_headers = has_headers;
        self
    }

    pub fn with_train_ratio(mut self, ratio: f64) -> Self {
        self.train_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    pub fn with_output_dir(mut self, dir: PathBuf) -> Self {
        self.output_dir = dir;
        self
    }

    pub fn with_target_mapping(
        mut self,
        mapping: fn(&str) -> Result<u8, Box<dyn Error + Send + Sync + 'static>>,
    ) -> Self {
        self.target_mapping = Some(mapping);
        self
    }

    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads.max(1);
        self
    }
}

fn spawn_writer_thread(
    train_path: PathBuf,
    test_path: PathBuf,
    rx: std::sync::mpsc::Receiver<(f64, u8, bool)>,
) -> JoinHandle<Result<(), Box<dyn Error + Send + Sync + 'static>>> {
    debug!("Spawning writer thread for paths: {:?}, {:?}", train_path, test_path);
    thread::spawn(move || match File::create(&train_path) {
        Ok(train_file) => match File::create(&test_path) {
            Ok(test_file) => {
                let mut train_writer = BufWriter::new(train_file);
                let mut test_writer = BufWriter::new(test_file);

                while let Ok((feature, target, is_train)) = rx.recv() {
                    let writer = if is_train { &mut train_writer } else { &mut test_writer };
                    if let Err(e) = writeln!(writer, "{},{}", feature, target) {
                        error!("Failed to write to file: {:?}", e);
                        return Err(Box::new(e) as Box<dyn Error + Send + Sync + 'static>);
                    }
                }
                train_writer.flush()?;
                test_writer.flush()?;
                Ok(())
            }
            Err(e) => {
                error!("Failed to create test file {:?}: {:?}", test_path, e);
                Err(Box::new(e) as Box<dyn Error + Send + Sync + 'static>)
            }
        },
        Err(e) => {
            error!("Failed to create train file {:?}: {:?}", train_path, e);
            Err(Box::new(e) as Box<dyn Error + Send + Sync + 'static>)
        }
    })
}

fn process_csv_chunk(
    data: Vec<(f64, u8)>,
    tx: Sender<(f64, u8, bool)>,
    is_train: bool,
) -> JoinHandle<()> {
    thread::spawn(move || {
        for (feature, target) in &data {
            if let Err(e) = tx.send((*feature, *target, is_train)) {
                error!("Failed to send data to writer: {:?}", e);
                return;
            }
        }
    })
}

/// Process a CSV dataset with multithreading
pub fn process_csv_dataset(
    config: CsvDatasetConfig,
) -> Result<(usize, usize), Box<dyn Error + Send + Sync + 'static>> {
    info!("Starting CSV dataset processing with config: {:?}", config);

    // Ensure output directory exists
    fs::create_dir_all(&config.output_dir)?;

    // Download data
    debug!("Downloading data from {}", config.url);
    let response = get(&config.url)?;
    let text = response.text()?;

    // Parse CSV
    let mut rdr = ReaderBuilder::new().has_headers(config.has_headers).from_reader(text.as_bytes());
    let mut rows: Vec<(f64, u8)> = Vec::new();
    for (i, result) in rdr.records().enumerate() {
        match result {
            Ok(record) => match record[config.feature_col].parse::<f64>() {
                Ok(feature) => {
                    let target = if let Some(mapping) = config.target_mapping {
                        mapping(&record[config.target_col])?
                    } else {
                        record[config.target_col].parse()?
                    };
                    rows.push((feature, target));
                }
                Err(e) => {
                    error!("Failed to parse feature in row {}: {:?}", i, e);
                    return Err(Box::new(e));
                }
            },
            Err(e) => {
                error!("Failed to read CSV record at row {}: {:?}", i, e);
                return Err(Box::new(e));
            }
        }
    }
    debug!("Parsed {} rows", rows.len());

    // Shuffle and split
    let mut rng = rand::thread_rng();
    rows.shuffle(&mut rng);
    let total_rows = rows.len();
    let train_size = (total_rows as f64 * config.train_ratio).round() as usize;
    let (train_data, test_data) = rows.split_at(train_size);
    debug!("Split data: train_size={}, test_size={}", train_size, test_data.len());

    // Create channel and spawn threads
    let (tx, rx) = channel();
    let train_path = config.output_dir.join("train_data.csv");
    let test_path = config.output_dir.join("test_data.csv");
    let writer_handle = spawn_writer_thread(train_path, test_path, rx);

    let chunk_size = (total_rows / config.num_threads).max(1);
    let mut handles = Vec::new();

    for chunk in train_data.chunks(chunk_size) {
        let tx = tx.clone();
        let chunk = chunk.to_vec();
        handles.push(process_csv_chunk(chunk, tx, true));
    }

    for chunk in test_data.chunks(chunk_size) {
        let tx = tx.clone();
        let chunk = chunk.to_vec();
        handles.push(process_csv_chunk(chunk, tx, false));
    }

    drop(tx);

    // Wait for threads to complete
    for (i, handle) in handles.into_iter().enumerate() {
        if let Err(e) = handle.join() {
            error!("Processing thread {} panicked: {:?}", i, e);
        }
    }

    match writer_handle.join() {
        Ok(result) => result?,
        Err(e) => {
            error!("Writer thread panicked: {:?}", e);
            return Err("Writer thread panicked".into());
        }
    }

    info!("CSV dataset processing completed successfully");
    Ok((train_size, total_rows - train_size))
}

#[cfg(test)]
mod tests {
    use super::*;
    use env_logger;

    fn breast_cancer_mapping(value: &str) -> Result<u8, Box<dyn Error + Send + Sync + 'static>> {
        match value {
            "M" => Ok(1),
            "B" => Ok(0),
            _ => Err("Invalid diagnosis value".into()),
        }
    }

    #[test]
    fn test_process_csv_breast_cancer() {
        let _ =
            env_logger::builder().is_test(true).filter_level(log::LevelFilter::Debug).try_init();

        info!("Starting test_process_csv_breast_cancer");
        let config = CsvDatasetConfig::new(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
            false,
            2,
            1,
        )
        .with_target_mapping(breast_cancer_mapping)
        .with_output_dir(PathBuf::from("./test_output"))
        .with_threads(2);

        let result = process_csv_dataset(config);
        if let Err(e) = &result {
            error!("Test failed with error: {:?}", e);
        }
        assert!(result.is_ok(), "Process CSV dataset failed: {:?}", result.err());
        let (train_size, test_size) = result.unwrap();
        debug!("Test result: train_size={}, test_size={}", train_size, test_size);
        assert!(train_size > 0, "Train size should be greater than 0");
        assert!(test_size > 0, "Test size should be greater than 0");
        info!("Test completed successfully");
    }
}

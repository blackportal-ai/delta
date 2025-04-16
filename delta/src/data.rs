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

use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use super::errors::CsvError;

pub struct CsvLoader;
pub struct CsvHeadersLoader;

pub trait DataLoader {
    fn load<P: AsRef<std::path::Path>>(path: P) -> Result<(Array2<f64>, Array1<f64>), Self::Error>;

    type Error: std::error::Error + 'static;
}

pub fn load_data<T: DataLoader, P: AsRef<std::path::Path>>(
    path: P,
) -> Result<(Array2<f64>, Array1<f64>), T::Error> {
    T::load(path)
}

fn load_csv_common<P: AsRef<Path>>(
    path: P,
    has_headers: bool,
) -> Result<(Array2<f64>, Array1<f64>), CsvError> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let mut rdr =
        csv::ReaderBuilder::new().has_headers(has_headers).flexible(true).from_reader(file);

    // Parse all fields as strings initially
    let mut data: Vec<Vec<String>> = Vec::new();
    for (i, result) in rdr.records().enumerate() {
        let record = result?;
        let row: Vec<String> = record.iter().map(|s| s.to_string()).collect();
        if i > 0 && row.len() != data[0].len() {
            return Err(CsvError::InconsistentColumns {
                row: i + 1,
                actual: row.len(),
                expected: data[0].len(),
            });
        }
        data.push(row);
    }

    let n_rows = data.len();
    if n_rows == 0 {
        return Err(CsvError::EmptyFile);
    }
    let n_cols = data[0].len();
    if n_cols < 2 {
        return Err(CsvError::InsufficientColumns);
    }

    // Identify categorical columns (non-numeric, except target)
    let mut is_categorical = vec![false; n_cols];
    for col in 0..n_cols - 1 {
        // Exclude target
        is_categorical[col] =
            data.iter().any(|row| !row[col].is_empty() && row[col].parse::<f64>().is_err());
    }

    // Encode data
    let mut feature_data: Vec<Vec<f64>> = Vec::with_capacity(n_rows);
    let mut target_data: Vec<f64> = Vec::with_capacity(n_rows);
    let mut encoders: Vec<HashMap<String, f64>> = vec![HashMap::new(); n_cols];

    for row in data.iter() {
        let mut feature_row = Vec::with_capacity(n_cols - 1);
        for col in 0..n_cols - 1 {
            let value = &row[col];
            if is_categorical[col] {
                // Label encode categorical, impute missing with "missing"
                let encoder = &mut encoders[col];
                let imputed_value =
                    if value.is_empty() { "missing".to_string() } else { value.clone() };
                let next_id = encoder.len() as f64;
                let encoded = *encoder.entry(imputed_value).or_insert(next_id);
                feature_row.push(encoded);
            } else {
                // Parse numeric, impute missing with 0.0
                let num = if value.is_empty() {
                    0.0
                } else {
                    value.parse::<f64>().map_err(|e| CsvError::InvalidNumeric {
                        value: value.clone(),
                        row: feature_data.len() + 1,
                        source: e,
                    })?
                };
                feature_row.push(num);
            }
        }
        feature_data.push(feature_row);

        // Parse target (always numeric), impute missing with 0.0
        let target_value = &row[n_cols - 1];
        let target = if target_value.is_empty() {
            0.0
        } else {
            target_value.parse::<f64>().map_err(|e| CsvError::InvalidTarget {
                value: target_value.clone(),
                row: target_data.len() + 1,
                source: e,
            })?
        };
        target_data.push(target);
    }

    // Convert to arrays
    let feature_data_flat: Vec<f64> = feature_data.into_iter().flatten().collect();
    let features = Array2::from_shape_vec((n_rows, n_cols - 1), feature_data_flat)?;
    let targets = Array1::from_vec(target_data);

    Ok((features, targets))
}

impl DataLoader for CsvLoader {
    type Error = CsvError;

    fn load<P: AsRef<Path>>(path: P) -> Result<(Array2<f64>, Array1<f64>), Self::Error> {
        load_csv_common(path, false)
    }
}

impl DataLoader for CsvHeadersLoader {
    type Error = CsvError;

    fn load<P: AsRef<Path>>(path: P) -> Result<(Array2<f64>, Array1<f64>), Self::Error> {
        load_csv_common(path, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::load_data;
    use ndarray::array;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_temp_csv(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("Failed to create temp file");
        file.write_all(content.as_bytes()).expect("Failed to write to temp file");
        file.flush().expect("Failed to flush temp file");
        file
    }

    #[test]
    fn test_load_numeric_no_headers() {
        let csv_content = "1.0,2.0\n3.0,4.0\n5.0,6.0\n";
        let temp_file = create_temp_csv(csv_content);

        let (features, targets) =
            load_data::<CsvLoader, _>(temp_file.path()).expect("Failed to load CSV");

        let expected_features = array![[1.0], [3.0], [5.0]];
        let expected_targets = array![2.0, 4.0, 6.0];

        assert_eq!(features, expected_features, "Features do not match");
        assert_eq!(targets, expected_targets, "Targets do not match");
    }

    #[test]
    fn test_load_categorical_no_headers() {
        let csv_content = "1.0,male,0\n2.0,female,1\n3.0,male,0\n";
        let temp_file = create_temp_csv(csv_content);

        let (features, targets) =
            load_data::<CsvLoader, _>(temp_file.path()).expect("Failed to load CSV");

        let expected_features = array![[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]];
        let expected_targets = array![0.0, 1.0, 0.0];

        assert_eq!(features, expected_features, "Features do not match");
        assert_eq!(targets, expected_targets, "Targets do not match");
    }

    #[test]
    fn test_load_headers_with_categoricals() {
        let csv_content = "age,gender,target\n25,male,0\n30,female,1\n35,male,0\n";
        let temp_file = create_temp_csv(csv_content);

        let (features, targets) =
            load_data::<CsvHeadersLoader, _>(temp_file.path()).expect("Failed to load CSV");

        let expected_features = array![[25.0, 0.0], [30.0, 1.0], [35.0, 0.0]];
        let expected_targets = array![0.0, 1.0, 0.0];

        assert_eq!(features, expected_features, "Features do not match");
        assert_eq!(targets, expected_targets, "Targets do not match");
    }

    #[test]
    fn test_load_empty_file() {
        let csv_content = "";
        let temp_file = create_temp_csv(csv_content);

        let result = load_data::<CsvLoader, _>(temp_file.path());
        assert!(matches!(result, Err(CsvError::EmptyFile)));
    }

    #[test]
    fn test_load_single_column() {
        let csv_content = "1.0\n2.0\n3.0\n";
        let temp_file = create_temp_csv(csv_content);

        let result = load_data::<CsvLoader, _>(temp_file.path());
        assert!(matches!(result, Err(CsvError::InsufficientColumns)));
    }

    #[test]
    fn test_load_missing_numeric_value() {
        let csv_content = "1.0,,0\n2.0,4.0,1\n";
        let temp_file = create_temp_csv(csv_content);

        let (features, targets) =
            load_data::<CsvLoader, _>(temp_file.path()).expect("Failed to load CSV");

        let expected_features = array![[1.0, 0.0], [2.0, 4.0]];
        let expected_targets = array![0.0, 1.0];

        assert_eq!(features, expected_features, "Features do not match");
        assert_eq!(targets, expected_targets, "Targets do not match");
    }

    #[test]
    fn test_load_missing_categorical_value() {
        let csv_content = "1.0,male,0\n2.0,,1\n3.0,female,0\n";
        let temp_file = create_temp_csv(csv_content);

        let (features, targets) =
            load_data::<CsvLoader, _>(temp_file.path()).expect("Failed to load CSV");

        let expected_features = array![[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]];
        let expected_targets = array![0.0, 1.0, 0.0];

        assert_eq!(features, expected_features, "Features do not match");
        assert_eq!(targets, expected_targets, "Targets do not match");
    }

    #[test]
    fn test_load_missing_target() {
        let csv_content = "1.0,male,0\n2.0,female,\n3.0,male,1\n";
        let temp_file = create_temp_csv(csv_content);

        let (features, targets) =
            load_data::<CsvLoader, _>(temp_file.path()).expect("Failed to load CSV");

        let expected_features = array![[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]];
        let expected_targets = array![0.0, 0.0, 1.0];

        assert_eq!(features, expected_features, "Features do not match");
        assert_eq!(targets, expected_targets, "Targets do not match");
    }

    #[test]
    fn test_load_invalid_target() {
        let csv_content = "1.0,male,invalid\n2.0,female,1\n";
        let temp_file = create_temp_csv(csv_content);

        let result = load_data::<CsvLoader, _>(temp_file.path());
        assert!(matches!(result, Err(CsvError::InvalidTarget { value, .. }) if value == "invalid"));
    }

    #[test]
    fn test_load_inconsistent_columns() {
        let csv_content = "1.0,male,0\n2.0,female,1,extra\n";
        let temp_file = create_temp_csv(csv_content);

        let result = load_data::<CsvLoader, _>(temp_file.path());
        assert!(
            matches!(result, Err(CsvError::InconsistentColumns { row, actual, expected }) if row == 2 && actual == 4 && expected == 3)
        );
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = load_data::<CsvLoader, _>("nonexistent.csv");
        assert!(matches!(result, Err(CsvError::FileOpen(_))));
    }

    #[test]
    fn test_load_all_missing_categorical() {
        let csv_content = "1.0,,0\n2.0,,1\n3.0,,0\n";
        let temp_file = create_temp_csv(csv_content);
        let (features, targets) =
            load_data::<CsvLoader, _>(temp_file.path()).expect("Failed to load CSV");
        let expected_features = array![[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]; // "missing" = 0.0
        let expected_targets = array![0.0, 1.0, 0.0];
        assert_eq!(features, expected_features);
        assert_eq!(targets, expected_targets);
    }
}

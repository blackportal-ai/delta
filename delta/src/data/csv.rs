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

use crate::data::DataLoader;
use ndarray::{Array1, Array2};
use std::error::Error;
use std::fs::File;
use std::path::Path;

pub struct CsvLoader;
pub struct CsvHeadersLoader;

fn load_csv_common<P: AsRef<Path>>(
    path: P,
    has_headers: bool,
) -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let mut rdr =
        csv::ReaderBuilder::new().has_headers(has_headers).flexible(true).from_reader(file);

    let mut data: Vec<Vec<f64>> = Vec::new();
    for (i, result) in rdr.records().enumerate() {
        let record = result?;
        let row: Vec<f64> = record
            .iter()
            .map(|field| field.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Row {}: {}", i + 1, e))?;
        if i > 0 && row.len() != data[0].len() {
            return Err(format!(
                "Inconsistent column count: row {} has {} columns, expected {}",
                i + 1,
                row.len(),
                data[0].len()
            )
            .into());
        }
        data.push(row);
    }

    let n_rows = data.len();
    if n_rows == 0 || data[0].is_empty() {
        return Err("Empty CSV file or invalid data".into());
    }
    let n_cols = data[0].len();
    if n_cols < 2 {
        return Err("CSV must have at least one feature and one target column".into());
    }

    let feature_data: Vec<f64> = data.iter().flat_map(|row| row[..n_cols - 1].to_vec()).collect();
    let target_data: Vec<f64> = data.iter().map(|row| row[n_cols - 1]).collect();

    let features = Array2::from_shape_vec((n_rows, n_cols - 1), feature_data)?;
    let targets = Array1::from_vec(target_data);

    Ok((features, targets))
}

impl DataLoader for CsvLoader {
    fn load<P: AsRef<Path>>(path: P) -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>> {
        load_csv_common(path, false)
    }
}

impl DataLoader for CsvHeadersLoader {
    fn load<P: AsRef<Path>>(path: P) -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>> {
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
    fn test_load_default_no_headers() {
        let csv_content = "1.0,2.0\n2.0,4.0\n3.0,6.0\n4.0,8.0\n";
        let temp_file = create_temp_csv(csv_content);

        let (features, targets) =
            load_data::<CsvLoader, _>(temp_file.path()).expect("Failed to load CSV");

        let expected_features = array![[1.0], [2.0], [3.0], [4.0]];
        let expected_targets = array![2.0, 4.0, 6.0, 8.0];

        assert_eq!(features, expected_features, "Features do not match");
        assert_eq!(targets, expected_targets, "Targets do not match");
    }

    #[test]
    fn test_load_multi_feature_columns() {
        let csv_content = "1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0\n";
        let temp_file = create_temp_csv(csv_content);

        let (features, targets) =
            load_data::<CsvLoader, _>(temp_file.path()).expect("Failed to load CSV");

        let expected_features = array![[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]];
        let expected_targets = array![3.0, 6.0, 9.0];

        assert_eq!(features, expected_features, "Features do not match");
        assert_eq!(targets, expected_targets, "Targets do not match");
    }

    #[test]
    fn test_load_empty_file() {
        let csv_content = "";
        let temp_file = create_temp_csv(csv_content);

        let result = load_data::<CsvLoader, _>(temp_file.path());
        assert!(result.is_err(), "Loading empty file should fail");
        if let Err(e) = result {
            assert_eq!(e.to_string(), "Empty CSV file or invalid data", "Unexpected error message");
        }
    }

    #[test]
    fn test_load_single_column() {
        let csv_content = "1.0\n2.0\n3.0\n";
        let temp_file = create_temp_csv(csv_content);

        let result = load_data::<CsvLoader, _>(temp_file.path());
        assert!(result.is_err(), "Loading single-column CSV should fail");
        if let Err(e) = result {
            assert_eq!(
                e.to_string(),
                "CSV must have at least one feature and one target column",
                "Unexpected error message"
            );
        }
    }

    #[test]
    fn test_load_invalid_numeric_data() {
        let csv_content = "1.0,2.0\nabc,4.0\n3.0,6.0\n";
        let temp_file = create_temp_csv(csv_content);

        let result = load_data::<CsvLoader, _>(temp_file.path());
        assert!(result.is_err(), "Loading invalid numeric data should fail");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Row 2: invalid float literal"),
                "Unexpected error: {}",
                e
            );
        }
    }

    #[test]
    fn test_load_inconsistent_column_count() {
        let csv_content = "1.0,2.0\n2.0,4.0,5.0\n3.0,6.0\n";
        let temp_file = create_temp_csv(csv_content);

        let result = load_data::<CsvLoader, _>(temp_file.path());
        assert!(result.is_err(), "Loading inconsistent column count should fail");
        if let Err(e) = result {
            assert!(
                e.to_string().contains("Inconsistent column count: row 2"),
                "Unexpected error: {}",
                e
            );
        }
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = load_data::<CsvLoader, _>("nonexistent.csv");
        assert!(result.is_err(), "Loading nonexistent file should fail");
        if let Err(e) = result {
            assert!(e.to_string().contains("No such file or directory"), "Unexpected error: {}", e);
        }
    }

    #[test]
    fn test_load_single_row() {
        let csv_content = "1.0,2.0\n";
        let temp_file = create_temp_csv(csv_content);

        let (features, targets) =
            load_data::<CsvLoader, _>(temp_file.path()).expect("Failed to load CSV");

        let expected_features = array![[1.0]];
        let expected_targets = array![2.0];

        assert_eq!(features, expected_features, "Features do not match");
        assert_eq!(targets, expected_targets, "Targets do not match");
    }

    #[test]
    fn test_load_headers() {
        let csv_content = "x,y\n1.0,2.0\n2.0,4.0\n3.0,6.0\n";
        let temp_file = create_temp_csv(csv_content);

        let (features, targets) =
            load_data::<CsvHeadersLoader, _>(temp_file.path()).expect("Failed to load CSV");

        let expected_features = array![[1.0], [2.0], [3.0]];
        let expected_targets = array![2.0, 4.0, 6.0];

        assert_eq!(features, expected_features, "Features do not match");
        assert_eq!(targets, expected_targets, "Targets do not match");
    }
}

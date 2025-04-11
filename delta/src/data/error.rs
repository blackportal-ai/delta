use thiserror::Error;

#[derive(Error, Debug)]
pub enum DataError {
    #[error("CSV error: {0}")]
    Csv(#[from] CsvError),
}

#[derive(Error, Debug)]
pub enum CsvError {
    #[error("Failed to open file: {0}")]
    FileOpen(#[from] std::io::Error),

    #[error("CSV file is empty")]
    EmptyFile,

    #[error("CSV must have at least one feature and one target column")]
    InsufficientColumns,

    #[error("Inconsistent column count: row {row} has {actual} columns, expected {expected}")]
    InconsistentColumns { row: usize, actual: usize, expected: usize },

    #[error("Invalid numeric value '{value}' at row {row}: {source}")]
    InvalidNumeric { value: String, row: usize, source: std::num::ParseFloatError },

    #[error("Invalid target value '{value}' at row {row}: {source}")]
    InvalidTarget { value: String, row: usize, source: std::num::ParseFloatError },

    #[error("Failed to shape data into array: {0}")]
    ArrayShape(#[from] ndarray::ShapeError),

    #[error("Failed to parse CSV: {0}")]
    CsvParse(#[from] csv::Error),
}

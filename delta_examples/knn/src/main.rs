use deltaml::{
    algorithms::{Classification, KNN},
    data::{CsvHeadersLoader, load_data},
    preprocessors::StandardScaler,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let train_path = concat!(env!("CARGO_MANIFEST_DIR"), "/train_data.csv");
    let test_path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data.csv");

    // Load the Iris dataset (already prepared with splits)
    let (x_train, y_train) = load_data::<CsvHeadersLoader, _>(train_path)
        .expect("Failed to load train_data.csv");
    let (x_test, y_test) =
        load_data::<CsvHeadersLoader, _>(test_path).expect("Failed to load test_data.csv");

    // Instantiate the model
    let mut model = KNN::new()
        .k(3) // Use 3 nearest neighbors
        .scaler(StandardScaler::new())
        .normalize(true) // Standardize features for distance-based KNN
        .mode(Classification)
        .build();

    // Train the model
    model.fit(&x_train, &y_train)?;

    // Make predictions with the trained model
    let predictions = model.predict(&x_test)?;

    println!("Predictions for new data (probabilities): {:?}", predictions);

    let accuracy = model.calculate_accuracy(&predictions, &y_test);
    println!("Test Accuracy: {:.6}", accuracy);

    Ok(())
}

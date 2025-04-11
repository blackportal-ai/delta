use deltaml::{
    classical::{Algorithm, Normalization, algorithms::LinearRegression, losses::MSE},
    data::{CsvHeadersLoader, load_data},
    ndarray::Axis,
};

#[tokio::main]
async fn main() {
    // Load training data (California Housing Prices dataset)
    let (x_train, y_train) = load_data::<CsvHeadersLoader, _>("../train_data.csv")
        .expect("Failed to load train_data.csv");

    // Normalize x_train (features)
    let mut x_normalizer = Normalization::new();
    x_normalizer.adapt_2d(&x_train);
    let x_train_scaled = x_normalizer.transform_2d(&x_train);

    // Normalize y_train (target: median_house_value)
    let mut y_normalizer = Normalization::new();
    y_normalizer.adapt_1d(&y_train);
    let y_train_scaled = y_normalizer.transform_1d(&y_train);

    // Instantiate the model with Mean Squared Error loss
    let mut model = LinearRegression::new(MSE);

    // Train the model on scaled data
    let learning_rate = 0.01; // Can keep this now with normalization
    let epochs = 1000;
    model.fit(&x_train_scaled, &y_train_scaled, learning_rate, epochs);

    // Load and normalize test data
    let (x_test, y_test) =
        load_data::<CsvHeadersLoader, _>("../test_data.csv").expect("Failed to load test_data.csv");
    let x_test_scaled = x_normalizer.transform_2d(&x_test);

    // Make predictions and inverse-transform to original scale
    let predictions_scaled = model.predict(&x_test_scaled);
    let predictions = y_normalizer.inverse_transform_1d(&predictions_scaled);

    println!("Predictions for test data: {:?}", predictions);

    // Calculate test loss (MSE) on original scale
    let test_loss = model.calculate_loss(&predictions, &y_test);
    println!("Test Loss (MSE) after training: {:.6}", test_loss);
}

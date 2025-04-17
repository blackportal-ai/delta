use deltaml::{
    algorithms::LinearRegression,
    data::{CsvHeadersLoader, load_data},
    losses::MSE,
    optimizers::BatchGradientDescent,
    preprocessors::StandardScaler,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let train_path = concat!(env!("CARGO_MANIFEST_DIR"), "/train_data.csv");
    let test_path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data.csv");

    // Using California Housing Dataset (already prepared with splits)
    let (x_train, y_train) = load_data::<CsvHeadersLoader, _>(train_path)
        .expect("Failed to load train_data.csv");
    let (x_test, y_test) =
        load_data::<CsvHeadersLoader, _>(test_path).expect("Failed to load test_data.csv");

    // Instantiate the model
    let mut model = LinearRegression::new()
        .optimizer(BatchGradientDescent)
        .loss_function(MSE)
        .scaler(StandardScaler::new())
        .normalize(true)
        .build();

    // Train the model
    let learning_rate = 0.01;
    let epochs = 1000;
    model.fit(&x_train, &y_train, learning_rate, epochs)?;

    // Make predictions with the trained model
    let predictions = model.predict(&x_test)?;

    println!("Predictions for new data: {:?}", predictions);

    // Calculate log loss for the test data
    let test_loss = model.calculate_loss(&predictions, &y_test)?;
    println!("Test Loss after training: {:.6}", test_loss);

    Ok(())
}

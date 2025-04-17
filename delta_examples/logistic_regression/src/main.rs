use deltaml::{
    algorithms::LogisticRegression,
    data::{CsvLoader, load_data},
    losses::CrossEntropy,
    optimizers::LogisticGradientDescent,
    preprocessors::StandardScaler,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Using Breast Cancer Wisconsin (Diagnostic) dataset from UCI (already prepared with splits)
    let (x_train, y_train) =
        load_data::<CsvLoader, _>("../train_data.csv").expect("Failed to load train_data.csv");
    let (x_test, y_test) =
        load_data::<CsvLoader, _>("../test_data.csv").expect("Failed to load test_data.csv");

    // Instantiate the model
    let mut model = LogisticRegression::new()
        .optimizer(LogisticGradientDescent)
        .loss_function(CrossEntropy)
        .scaler(StandardScaler::new())
        .normalize(true)
        .build();

    // Train the model
    let learning_rate = 0.01;
    let epochs = 1000;
    model.fit(&x_train, &y_train, learning_rate, epochs)?;

    // Make predictions with the trained model
    let predictions = model.predict(&x_test)?;

    println!("Predictions for new data (probabilities): {:?}", predictions);

    // Calculate log loss for the test data
    let test_loss = model.calculate_loss(&predictions, &y_test)?;
    println!("Test Loss after training: {:.6}", test_loss);

    // Calculate accuracy
    // Will be implemented in the next version of Delta
    // let accuracy = model.calculate_accuracy(&predictions, &y_test);
    // println!("Accuracy: {:.2}%", accuracy * 100.0);

    Ok(())
}

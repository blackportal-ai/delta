use deltaml::{
    classical::{Algorithm, algorithms::LogisticRegression, losses::CrossEntropy},
    data::{CsvLoader, load_data},
};

#[tokio::main]
async fn main() {
    // Using Breast Cancer Wisconsin (Diagnostic) dataset from UCI (already prepared with splits)
    let (x_train, y_train) =
        load_data::<CsvLoader, _>("../train_data.csv").expect("Failed to load train_data.csv");

    // Instantiate the model
    let mut model = LogisticRegression::new_with_defaults(CrossEntropy);

    // Train the model
    let learning_rate = 0.01;
    let epochs = 1000;
    model.fit(&x_train, &y_train, learning_rate, epochs);

    // Make predictions with the trained model
    let (x_test, y_test) =
        load_data::<CsvLoader, _>("../test_data.csv").expect("Failed to load test_data.csv");

    let predictions = model.predict(&x_test);

    println!("Predictions for new data (probabilities): {:?}", predictions);

    // Calculate log loss for the test data
    let test_loss = model.calculate_loss(&predictions, &y_test);
    println!("Test Loss after training: {:.6}", test_loss);

    // Calculate accuracy
    let accuracy = model.calculate_accuracy(&predictions, &y_test);
    println!("Accuracy: {:.2}%", accuracy * 100.0);
}

use deltaml::{
    classical::{Algorithm, algorithms::LinearRegression, losses::MSE},
    data::{CsvHeadersLoader, load_data},
};

#[tokio::main]
async fn main() {
    // Using California Housing Prices dataset from Kaggle
    let (x_train, y_train) = load_data::<CsvHeadersLoader, _>("../train_data.csv")
        .expect("Failed to load train_data.csv");
    println!("x_train shape: {:?}", x_train.dim());

    // Instantiate the model with Mean Squared Error loss
    let mut model = LinearRegression::new(MSE);

    // Train the model
    let learning_rate = 0.01;
    let epochs = 1000;
    model.fit(&x_train, &y_train, learning_rate, epochs);

    // Load test data
    let (x_test, y_test) =
        load_data::<CsvHeadersLoader, _>("../test_data.csv").expect("Failed to load test_data.csv");

    // Make predictions
    let predictions = model.predict(&x_test);

    println!("Predictions for test data: {:?}", predictions);

    // Calculate test loss (MSE)
    let test_loss = model.calculate_loss(&predictions, &y_test);
    println!("Test Loss (MSE) after training: {:.6}", test_loss);
}

use deltaml::{
    classical_ml::{Algorithm, algorithms::LinearRegression, losses::MSE},
    ndarray::{Array1, Array2},
};

#[tokio::main]
async fn main() {
    // Create some dummy data for example purposes
    let x_data = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y_data = Array1::from_vec(vec![2.0, 4.0, 5.0, 4.0, 5.0]);

    // Instantiate the model
    let mut model = LinearRegression::new(MSE);

    // Train the model
    let learning_rate = 0.01;
    let epochs = 1000;
    model.fit(&x_data, &y_data, learning_rate, epochs);

    // Make predictions with the trained model
    let new_data = Array2::from_shape_vec((3, 1), vec![2.5, 3.5, 4.5]).unwrap();
    let predictions = model.predict(&new_data);

    println!("Predictions for new data: {:?}", predictions);

    // Calculate accuracy or loss for the test data for demonstration
    let test_loss = model.calculate_loss(&model.predict(&x_data), &y_data);
    println!("Test Loss after training: {:.6}", test_loss);
}

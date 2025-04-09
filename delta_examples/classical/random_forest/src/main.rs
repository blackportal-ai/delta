use deltaml::{
    classical::{Algorithm, algorithms::RandomForest, losses::MSE},
    ndarray::{Array1, Array2},
};

#[tokio::main]
async fn main() {
    // create feature data
    let x_data = Array2::from_shape_vec(
        (4, 4),
        vec![
            25.0,     // 25 years old
            30.0,     // 30 years old
            22.0,     // 22 years old
            40.0,     // 40 years old
            0.0,      // male
            1.0,      // female
            0.0,      // male
            1.0,      // female
            1.0,      // engineer
            2.0,      // doctor
            3.0,      // student
            4.0,      // manager
            60000.0,  // 60000.0 salary
            80000.0,  // 80000.0 salary
            20000.0,  // 20000.0 salary
            100000.0, // 100000.0 salary
        ],
    )
    .unwrap();

    let y_data = Array1::from_vec(vec![1.0, 1.0, 0.0, 1.0]); // high, high, low, high

    // Instantiate the model
    let mut model = RandomForest::new(MSE);

    // Train the model
    let learning_rate = 0.01;
    let epochs = 1000;
    model.fit(&x_data, &y_data, learning_rate, epochs);

    // Make predictions with the trained model
    let new_data = Array2::from_shape_vec((4, 1), vec![28.0, 1.0, 1.0, 70000.0]).unwrap();
    let predictions = model.predict(&new_data);

    println!("Predictions for new data: {:?}", predictions);

    // Calculate accuracy or loss for the test data for demonstration
    let test_loss = model.calculate_loss(&model.predict(&x_data), &y_data);
    println!("Test Loss after training: {:.6}", test_loss);
}

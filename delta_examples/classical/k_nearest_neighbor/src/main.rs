use deltaml::{
    classical::{Algorithm, algorithms::KNearestNeighbors, losses::CrossEntropy},
    ndarray::{Array1, Array2},
};

#[tokio::main]
async fn main() {
    let x_data =
        Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 6.0, 5.0]).unwrap();

    let y_data = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

    let mut model = KNearestNeighbors::new(5, CrossEntropy);

    let learning_rate = 0.01;
    let epochs = 1000;
    model.fit(&x_data, &y_data, learning_rate, epochs);

    let new_data = Array2::from_shape_vec((2, 2), vec![1.5, 2.5, 5.0, 4.5]).unwrap();
    let predictions = model.predict(&new_data);

    println!("Predictions for new data: {:?}", predictions);

    let test_loss = model.calculate_loss(&model.predict(&x_data), &y_data);
    println!("Test Loss after training: {:.6}", test_loss);
}

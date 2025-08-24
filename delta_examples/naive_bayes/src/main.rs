use deltaml::{
    algorithms::{NaiveBayesBuilder},
    data::{CsvLoader, load_data},
};
use ndarray::{Array1, s};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let train_path = concat!(env!("CARGO_MANIFEST_DIR"), "/train_data.csv");
    let test_path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data.csv");

    // Load data (features and labels are separated by load_data)
    let (x_train, y_train) = load_data::<CsvLoader, _>(train_path)?;
    let (x_test, y_test) = load_data::<CsvLoader, _>(test_path)?;

    // Instantiate the model
    let mut model = NaiveBayesBuilder::new()
        .alpha(1.0)
        .normalize(false)
        .build();

    // Train the model
    model.fit(&x_train, &y_train, 0.0, 1)?;

    // Make predictions
    let predictions = model.predict(&x_test)?;

    // Calculate accuracy
    let accuracy = accuracy(&y_test, &predictions);
    println!("Accuracy on test set: {:.2}%", accuracy * 100.0);

    // Optional: Print sample predictions
    println!("Sample predictions: {:?}", &predictions.slice(s![0..5]));

    Ok(())
}

// Custom accuracy function
fn accuracy(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| (t - p).abs() < 1e-5)
        .count();
    correct as f64 / y_true.len() as f64
}
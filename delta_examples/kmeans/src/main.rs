use deltaml::clustering::kmeans::{InitMethod, KMeans};
use deltaml::ndarray::{Array1, Array2, Axis};
use linfa_datasets::iris;
use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the Iris dataset (150 samples, 4 features)
    let dataset = iris();
    let records_linfa = dataset.records.clone();
    let (n_samples, n_features) = records_linfa.dim();

    // Convert the dataset to a 2D array
    let flat: Vec<f64> = records_linfa.iter().cloned().collect();
    let data: Array2<f64> = Array2::from_shape_vec((n_samples, n_features), flat)?;

    // Configure and run KMeans
    let mut km = KMeans::new()
        .n_clusters(3)
        .init_method(InitMethod::KMeansPlusPlus)
        .n_init(10)
        .max_iterations(300)
        .tolerance(1e-4)
        .random_state(Some(42))
        .build();

    km.fit(&data)?;
    let labels: Array1<usize> = km.labels().unwrap().clone();
    let centroids: Array2<f64> = km.centroids().unwrap().clone();

    // Create a drawing area
    let backend = BitMapBackend::new("iris_kmeans.png", (800, 600));
    let root = backend.into_drawing_area();
    root.fill(&WHITE)?;

    // Build chart
    let x_range = data.column(0).iter().cloned().reduce(f64::min).unwrap()
        ..data.column(0).iter().cloned().reduce(f64::max).unwrap();
    let y_range = data.column(1).iter().cloned().reduce(f64::min).unwrap()
        ..data.column(1).iter().cloned().reduce(f64::max).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("KMeans Clustering on Iris", ("sans-serif", 30))
        .margin(20)
        .build_cartesian_2d(x_range, y_range)?;

    chart.configure_mesh().draw()?;

    // Define colors for clusters
    let palette = [RED, BLUE, GREEN];

    // Plot data points by cluster
    for k in 0..3 {
        let points: Vec<(f64, f64)> = data
            .rows()
            .into_iter()
            .zip(labels.iter())
            .filter(|(_, &lab)| lab == k)
            .map(|(row, _)| (row[0], row[1]))
            .collect();

        chart.draw_series(
            points.iter().map(|&(x, y)| Circle::new((x, y), 6, palette[k].filled())),
        )?;
    }

    // Plot centroids
    chart.draw_series(
        centroids
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(k, row)| Cross::new((row[0], row[1]), 15, palette[k].stroke_width(2))),
    )?;

    // Save the figure
    root.present()?;
    println!("Clustered Iris plot saved to 'iris_kmeans.png'");
    Ok(())
}

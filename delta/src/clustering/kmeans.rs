// BSD 3-Clause License
//
// Copyright (c) 2025, BlackPortal ○
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//! k-means clustering implemented in Rust, inspired by scikit-learn
use crate::errors::{ModelError, PreprocessingError};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

/// Method to initialize cluster centroids.
///
/// - `Random`: choose k observations uniformly at random as initial centers.
/// - `KMeansPlusPlus`: select initial centers spread out via weighted probability to improve convergence.
#[derive(Clone, Copy)]
pub enum InitMethod {
    /// Random selection of initial centers
    Random,
    /// k-means++ smart initialization
    KMeansPlusPlus,
}

/// Builder for configuring and creating a `KMeans` instance.
pub struct KMeansBuilder {
    n_clusters: usize,
    init_method: InitMethod,
    n_init: usize,
    max_iterations: usize,
    tolerance: f64,
    random_state: Option<u64>,
}

/// Builder for configuring and creating a `KMeans` instance.
///
/// # Examples
/// ```rust
/// use deltaml::clustering::kmeans::{KMeans, InitMethod};
/// let mut km = KMeans::new()
///     .n_clusters(3)
///     .init_method(InitMethod::KMeansPlusPlus)
///     .max_iterations(300)
///     .tolerance(1e-4)
///     .random_state(Some(42))
///     .build();
/// ```
///
impl KMeansBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        KMeansBuilder {
            n_clusters: 3,
            init_method: InitMethod::KMeansPlusPlus,
            n_init: 10,
            max_iterations: 100,
            tolerance: 1e-4,
            random_state: None,
        }
    }

    /// Set the number of clusters to form.
    pub fn n_clusters(mut self, n_clusters: usize) -> Self {
        self.n_clusters = n_clusters;
        self
    }

    /// Set the maximum number of iterations per run.
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the convergence tolerance.
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Choose the centroid initialization method.
    pub fn init_method(mut self, init_method: InitMethod) -> Self {
        self.init_method = init_method;
        self
    }

    /// Set the number of times the algorithm will be run with different seeds.
    pub fn n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init;
        self
    }

    /// Set an optional random seed for reproducibility.
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Finalize the configuration and create a `KMeans` instance.
    pub fn build(self) -> KMeans {
        KMeans {
            n_clusters: self.n_clusters,
            init_method: self.init_method,
            n_init: self.n_init,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            random_state: self.random_state,
            centroids: None,
            labels: None,
            inertia: None,
            best_n_iter: None,
        }
    }
}

/// k-means clustering algorithm.
pub struct KMeans {
    n_clusters: usize,
    init_method: InitMethod,
    n_init: usize,
    max_iterations: usize,
    tolerance: f64,
    random_state: Option<u64>,
    centroids: Option<Array2<f64>>,
    labels: Option<Array1<usize>>,
    inertia: Option<f64>,
    best_n_iter: Option<usize>,
}

/// k-means clustering estimator.
///
/// After fitting, retrieve:
/// - `centroids()`: final cluster centers
/// - `labels()`: cluster assignment for each sample
/// - `inertia()`: sum of squared distances to nearest centroid
/// - `n_iter()`: iterations run in the best initialization
impl KMeans {
    /// Begin configuring a new KMeans estimator.
    pub fn new() -> KMeansBuilder {
        KMeansBuilder::new()
    }

    /// Fit the model to data `x` of shape (n_samples, n_features).
    ///
    /// Basic algorithm:
    /// 1) Initialize centroids (Random or KMeans++).
    /// 2) Loop up to `max_iterations`:
    ///    a) **Assignment step**: assign each sample to nearest centroid.
    ///    b) **Update step**: recompute centroids as mean of assigned samples.
    ///    c) **Convergence check**: compute centroid shift and break if below `tolerance`.
    /// 3) Repeat for `n_init` runs and keep the best by inertia (sum of squared distances).
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<(), ModelError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        if x.is_empty() {
            return Err(ModelError::Preprocessing(PreprocessingError::EmptyInput));
        }
        if self.n_clusters == 0 || self.n_clusters > n_samples {
            return Err(ModelError::Preprocessing(PreprocessingError::InvalidParameter));
        }

        let mut best_inertia = f64::MAX;
        let mut best_centroids = Array2::<f64>::zeros((self.n_clusters, n_features));
        let mut best_labels = Array1::<usize>::zeros(n_samples);
        let mut best_iter = 0;

        for run in 0..self.n_init {
            // 0) Initialize RNG with optional seed
            let seed = self.random_state.unwrap_or_else(|| rand::rng().random());
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(run as u64));

            // 1) Initialize centroids
            let mut centroids = match self.init_method {
                InitMethod::Random => {
                    let mut idx: Vec<usize> = (0..n_samples).collect();
                    idx.shuffle(&mut rng);
                    let mut c = Array2::<f64>::zeros((self.n_clusters, n_features));
                    for (i, &j) in idx.iter().take(self.n_clusters).enumerate() {
                        c.row_mut(i).assign(&x.row(j));
                    }
                    c
                }
                InitMethod::KMeansPlusPlus => {
                    // Simplest KMeans++: first center random, others weighted by distance squared
                    let mut c = Array2::<f64>::zeros((self.n_clusters, n_features));
                    c.row_mut(0).assign(&x.row(rng.random_range(0..n_samples)));
                    for i in 1..self.n_clusters {
                        let dist2: Vec<f64> = x
                            .axis_iter(ndarray::Axis(0))
                            .map(|row| {
                                c.axis_iter(ndarray::Axis(0))
                                    .map(|cen| (&row - &cen).mapv(|v| v * v).sum())
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .unwrap()
                            })
                            .collect();
                        let sum: f64 = dist2.iter().sum();
                        let mut r = rng.random::<f64>() * sum;
                        let mut next = 0;
                        for (j, &d2) in dist2.iter().enumerate() {
                            r -= d2;
                            if r <= 0.0 {
                                next = j;
                                break;
                            }
                        }
                        c.row_mut(i).assign(&x.row(next));
                    }
                    c
                }
            };

            let mut labels = Array1::<usize>::zeros(n_samples);
            let mut iters = 0;
            let mut inertia_run = 0.0;

            // 2) Iterative refinement
            for it in 0..self.max_iterations {
                // a) Assignment step
                for i in 0..n_samples {
                    let row = x.row(i);
                    let (best_k, dist) = (0..self.n_clusters)
                        .map(|k| (&row - &centroids.row(k)).mapv(|v| v * v).sum())
                        .enumerate()
                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                        .unwrap();
                    labels[i] = best_k;
                    inertia_run += dist;
                }

                // b) Update step
                let mut new_c = Array2::<f64>::zeros((self.n_clusters, n_features));
                let mut counts = vec![0usize; self.n_clusters];
                for i in 0..n_samples {
                    let k = labels[i];
                    let updated_row = new_c.row(k).to_owned() + x.row(i).to_owned();
                    new_c.row_mut(k).assign(&updated_row);
                    counts[k] += 1;
                }
                for k in 0..self.n_clusters {
                    if counts[k] > 0 {
                        new_c.row_mut(k).mapv_inplace(|v| v / (counts[k] as f64));
                    }
                }

                // c) Convergence check
                let shift = (&centroids - &new_c).mapv(|v| v * v).sum().sqrt();
                centroids = new_c;
                iters = it + 1;
                if shift <= self.tolerance {
                    break;
                }
            }

            // 3) Track best run by inertia
            if inertia_run < best_inertia {
                best_inertia = inertia_run;
                best_centroids = centroids.clone();
                best_labels = labels.clone();
                best_iter = iters;
            }
        }
        self.centroids = Some(best_centroids);
        self.labels = Some(best_labels);
        self.inertia = Some(best_inertia);
        self.best_n_iter = Some(best_iter);
        Ok(())
    }

    /// Predict cluster labels for new data `x` of shape (n_samples, n_features).
    ///
    /// Returns an error if the model has not been fitted.
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<usize>, ModelError> {
        let centroids = self
            .centroids
            .as_ref()
            .ok_or(ModelError::Preprocessing(PreprocessingError::NotFitted))?;

        let n_samples = x.nrows();
        let mut labels = Array1::<usize>::zeros(n_samples);

        for i in 0..n_samples {
            let point = x.row(i);
            let (mut best_centroid, mut min_dist) = (0, f64::MAX);
            for centroid in 0..centroids.nrows() {
                let center = centroids.row(centroid);
                let dist = (&point - &center).mapv(|d| d * d).sum();
                if dist < min_dist {
                    min_dist = dist;
                    best_centroid = centroid;
                }
            }
            labels[i] = best_centroid;
        }

        Ok(labels)
    }

    /// Get final cluster centers after fitting.
    pub fn centroids(&self) -> Option<&Array2<f64>> {
        self.centroids.as_ref()
    }
    /// Get labels of each sample after fitting.
    pub fn labels(&self) -> Option<&Array1<usize>> {
        self.labels.as_ref()
    }

    /// Sum of squared distances to nearest centroid (inertia) after fitting.
    pub fn inertia(&self) -> Option<f64> {
        self.inertia
    }

    /// Number of iterations run in the best initialization.
    pub fn n_iter(&self) -> Option<usize> {
        self.best_n_iter
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_simple_clusters() {
        let data = array![[0.0, 0.0], [0.1, 0.1], [9.0, 9.0], [9.1, 9.1]];
        let mut kmeans = KMeans::new().n_clusters(2).max_iterations(100).tolerance(1e-4).build();
        kmeans.fit(&data).unwrap();
        let labels = kmeans.labels().unwrap();
        // Check that the first two points are in one cluster and the last two in another
        assert!(labels.iter().take(2).all(|&l| l == labels[0]));
        assert!(labels.iter().skip(2).all(|&l| l == labels[2]));
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_predict_near_centroids() {
        let data = array![[1.0, 2.0], [1.1, 2.1], [8.0, 9.0], [8.1, 9.1]];
        let mut kmeans = KMeans::new().n_clusters(2).build();
        kmeans.fit(&data).unwrap();
        let labels = kmeans.labels().unwrap();
        let new_pts = array![[1.05, 2.05], [8.05, 9.05]];
        let preds = kmeans.predict(&new_pts).unwrap();
        // Check that the predicted labels match the expected labels
        assert_eq!(preds[0], labels[0]);
        assert_eq!(preds[1], labels[2]);
    }

    #[test]
    fn test_predict_not_fitted() {
        let kmeans = KMeans::new().build();
        let data = array![[0.0, 0.0]];
        assert!(kmeans.predict(&data).is_err());
    }

    #[test]
    fn test_inertia_non_negative() {
        let data = array![[0.0, 0.0], [0.1, 0.1], [9.0, 9.0], [9.1, 9.1]];
        let mut km = KMeans::new().n_clusters(2).n_init(2).max_iterations(5).build();
        km.fit(&data).unwrap();
        let inertia = km.inertia().unwrap();
        assert!(inertia >= 0.0);
    }

    #[test]
    fn test_n_iter_bounds() {
        let data = array![[0.0, 0.0], [9.0, 9.0]];
        let mut km = KMeans::new().n_clusters(2).max_iterations(10).build();
        km.fit(&data).unwrap();
        let it = km.n_iter().unwrap();
        assert!(it >= 1 && it <= 10);
    }

    #[test]
    fn test_all_same_cluster_when_k1() {
        let data = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let mut km = KMeans::new().n_clusters(1).build();
        km.fit(&data).unwrap();
        let labels = km.labels().unwrap();
        assert!(labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_invalid_k() {
        let data = array![[1.0, 2.0]];
        let mut kmeans = KMeans::new().n_clusters(2).build();
        assert!(kmeans.fit(&data).is_err());
    }
}

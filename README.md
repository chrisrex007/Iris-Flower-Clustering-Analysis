
# 🌸 Iris Flower Clustering Analysis

This project explores various clustering algorithms applied to the classic **Iris flower dataset**, showcasing a full machine learning pipeline from data loading and visualization to dimensionality reduction, clustering, evaluation, and result comparison.

<br/>

## 📚 Overview

The goal of this project is to demonstrate **unsupervised learning** techniques by discovering natural groupings within the dataset without using label information during model training.

We implement and compare the following clustering algorithms:

- **K-Means Clustering**
- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
- **Agglomerative Hierarchical Clustering**
- **Gaussian Mixture Models (GMM)**

Each method is evaluated based on cluster validity metrics and compared with the true species labels for insight into performance.

<br/>

## 📁 Project Structure

```
Machine_Learning.ipynb     # Main notebook with all code and visualizations
README.md                  # Project documentation
```

<br/>

## 🛠️ Technologies Used

- Python 3.x
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

<br/>

## 📊 Workflow

1. **Dataset Loading and Exploration**  
   Load Iris dataset using `scikit-learn` and explore class distributions and feature relationships.

2. **Preprocessing**  
   Apply multiple scaling techniques (StandardScaler, MinMaxScaler, Normalizer), primarily using StandardScaler for clustering.

3. **Dimensionality Reduction**  
   Visualize feature space using:
   - PCA (Principal Component Analysis)
   - LDA (Linear Discriminant Analysis)
   - NMF (Non-negative Matrix Factorization)
   - SVD (Truncated SVD)
   - t-SNE (for final cluster visualization)

4. **Clustering Algorithms**  
   Apply and compare 4 different clustering algorithms on the processed data.

5. **Evaluation**  
   Use clustering metrics:
   - Silhouette Score
   - Davies-Bouldin Index
   - Calinski-Harabasz Index
   - Adjusted Rand Index
   - Normalized Mutual Information

6. **Visualization**  
   - Visualize clusters using t-SNE projections.
   - Bar plots for comparing performance metrics across models.

7. **Final Analysis**  
   - Normalize metrics and compute an overall performance score.
   - Identify and justify the best clustering method.

<br/>

## 📈 Results

- Performance metrics are compared and visualized.
- The best model is selected based on a normalized average score.
- Visualizations reveal clear separations between species in reduced feature spaces.

<br/>

## 🧠 Conclusion

This project demonstrates how clustering algorithms can reveal hidden structure in data. Although clustering is unsupervised, its results align well with actual class labels in the Iris dataset—making it an excellent case study for learning and experimenting with machine learning concepts.

<br/>

## 🚀 How to Run

1. Clone the repository or download the `.ipynb` file.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:
   ```bash
   jupyter notebook Machine_Learning.ipynb
   ```

> Optionally, use a platform like **Google Colab** to run the notebook without local setup.

<br/>

## 📌 Notes

- All visualizations are optimized for clarity.
- Warnings related to Seaborn styling (future deprecations) are resolved.
- The notebook is modular, well-commented, and ready for extension.

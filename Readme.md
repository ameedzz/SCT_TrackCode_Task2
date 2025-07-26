
# 🛍️ Mall Customer Segmentation using K-Means Clustering

This project performs customer segmentation using **K-Means Clustering** on the popular **Mall Customers** dataset. The aim is to group customers into distinct clusters based on features like **Age**, **Annual Income**, and **Spending Score**, helping businesses understand customer behavior and target segments better.

---

## 📁 Files Included

* `Mall_Customers.csv` – Original dataset with customer demographic and spending data.
* `Mall_Customers_Clustered.csv` – Output file with an added `Cluster` column indicating assigned clusters.
* `mall_kmeans_clustering.py` – Python script for clustering and visualization.
* `README.md` – This documentation file.

---

## Lets see How It Works

1. **Data Loading**
   Load customer data from a CSV file.

2. **Feature Selection**
   Focus on:

   * Age
   * Annual Income (k\$)
   * Spending Score (1–100)

3. **Standardization**
   Use `StandardScaler` to normalize the features.

4. **Finding Optimal Clusters**
   Apply the **Elbow Method** to determine the best number of clusters (k).

5. **K-Means Clustering**
   Perform clustering with the selected `k` value.

6. **Save Output**
   Save the clustered dataset to a new CSV file.

7. **3D Visualization**
   Visualize the clusters using a 3D scatter plot.

---

## 📊 Visualization

A 3D plot illustrates how the customers are grouped based on their:

* Age
* Income
* Spending behavior

Clusters are color-coded using `viridis` colormap.

---

## 📚 What I Learned

* How to preprocess data using `StandardScaler`
* How to apply KMeans clustering in `scikit-learn`
* Elbow method for selecting the number of clusters
* 3D plotting using `matplotlib` for better visual insights


---

## 💻 Running  the Project

```bash
pip install pandas scikit-learn matplotlib
python mall_kmeans_clustering.py

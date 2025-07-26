import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1 : Load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)  # Clean data if needed
    return df

# Step 2: Feature scaling
def scale_features(df, feature_cols):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    return scaled

# Step 3: Find optimal number of clusters using the elbow method
def plot_elbow(scaled_features):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method For Optimal K')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Step 4: Apply KMeans clustering
def apply_kmeans(scaled_features, df, k=5):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(scaled_features)
    return df

# Step 5: 3D Visualization of clusterr
def plot_clusters_3d(df):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        df['Age'],
        df['Annual Income (k$)'],
        df['Spending Score (1-100)'],
        c=df['Cluster'],
        cmap='viridis',
        s=60
    )

    ax.set_title("3D Cluster Plot: Age vs Income vs Spending")
    ax.set_xlabel("Age")
    ax.set_ylabel("Annual Income (k$)")
    ax.set_zlabel("Spending Score (1-100)")
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    FILE_PATH = "Mall_Customers.csv"
    FEATURES = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

    df = load_data(FILE_PATH)
    scaled = scale_features(df, FEATURES)

    # Elbow plot to choose optimal k
    plot_elbow(scaled)

    # Choose k = 5 (using on elbow)
    df_clustered = apply_kmeans(scaled, df.copy(), k=5)
    df_clustered.to_csv("Mall_Customers_Clustered.csv", index=False)
    print(" Clustered data saved to 'Mall_Customers_Clustered.csv'")

    # Showing sample of clustered data
    print("\n Clustered Data Preview!!:")
    print(df_clustered.head())

    # 3D Cluster Visualization
    plot_clusters_3d(df_clustered)

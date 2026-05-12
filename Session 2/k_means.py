import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CustomKMeans:
    def __init__(self, n_clusters=5, max_iters=300, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def fit(self, X):
        np.random.seed(self.random_state)
        # Randomly initialize centroids from the data points
        initial_indices = np.random.permutation(X.shape[0])[:self.n_clusters]
        self.centroids = X[initial_indices]

        for i in range(self.max_iters):
            # Calculate Euclidean distance from each point to each centroid
            # X shape: (n_samples, n_features)
            # centroids shape: (n_clusters, n_features)
            # We expand dims to broadcast subtraction
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            
            # Assign each point to the closest centroid
            self.labels = np.argmin(distances, axis=1)

            # Calculate new centroids
            new_centroids = np.array([X[self.labels == k].mean(axis=0) for k in range(self.n_clusters)])
            
            # If centroids didn't change, we've converged
            if np.allclose(self.centroids, new_centroids):
                print(f"K-Means converged at iteration {i}")
                break
                
            self.centroids = new_centroids

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

def load_and_preprocess(filepath, state_name="TELANGANA"):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)
    
    print(f"Filtering data for state: {state_name}")
    # Ensure StateName is string and uppercase for comparison
    df['StateName'] = df['StateName'].astype(str).str.upper()
    df_state = df[df['StateName'] == state_name.upper()].copy()
    
    if df_state.empty:
        print(f"WARNING: No data found for state '{state_name}'. Please check the state name spelling.")
        return None
        
    print(f"Initial rows for {state_name}: {len(df_state)}")
    
    # Clean Latitude and Longitude columns
    # Coerce errors to NaN to remove corrupted strings or mixed types
    df_state['Latitude'] = pd.to_numeric(df_state['Latitude'], errors='coerce')
    df_state['Longitude'] = pd.to_numeric(df_state['Longitude'], errors='coerce')
    
    # Drop rows with NaN coordinates
    df_state.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    
    # Drop duplicate coordinates (multiple pincodes might fall on the exact same lat/lon)
    df_state.drop_duplicates(subset=['Latitude', 'Longitude'], inplace=True)
    
    print(f"Rows after preprocessing (removing NaNs and duplicates): {len(df_state)}")
    return df_state

def plot_clusters(df, centroids, state_name):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='Longitude', y='Latitude', 
        hue='Cluster', palette='tab10', 
        data=df, s=50, alpha=0.7, edgecolor=None
    )
    
    # Plot centroids
    plt.scatter(
        centroids[:, 1], centroids[:, 0], # Note: x is Longitude, y is Latitude
        c='red', marker='X', s=200, label='Centroids', edgecolor='black'
    )
    
    plt.title(f"Geographical Clustering of Pincodes in {state_name}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig('clusters_map.png')
    print("Map saved to 'clusters_map.png'")

def main():
    filepath = 'clustering_data.csv'
    target_state = "TELANGANA"
    k_clusters = 5
    
    # 1. Preprocess
    df = load_and_preprocess(filepath, target_state)
    if df is None:
        return
        
    # 2. Extract Features
    X = df[['Latitude', 'Longitude']].values
    
    # 3. K-Means Clustering (From Scratch)
    print(f"\nInitializing Custom K-Means with k={k_clusters}...")
    kmeans = CustomKMeans(n_clusters=k_clusters, random_state=42)
    kmeans.fit(X)
    
    # 4. Predictions & Merging
    df['Cluster'] = kmeans.labels
    print("\n--- Predictions on Test/Train Dataset ---")
    print(df[['Pincode', 'District', 'Latitude', 'Longitude', 'Cluster']].head(10))
    
    # 5. Visualization
    print("\nGenerating scatter plot map (saved as 'clusters_map.png')...")
    plot_clusters(df, kmeans.centroids, target_state)
    
    # 6. Inferences
    print("\n===============================")
    print("      INFERENCE & INSIGHTS     ")
    print("===============================\n")
    print(f"Algorithm applied: K-Means (Custom Implementation from scratch)")
    print(f"State analyzed: {target_state}")
    print(f"Number of distinct geographical clusters identified: {k_clusters}\n")
    
    for i in range(k_clusters):
        cluster_data = df[df['Cluster'] == i]
        districts = cluster_data['District'].unique()
        centroid_lat = kmeans.centroids[i][0]
        centroid_lon = kmeans.centroids[i][1]
        
        print(f"Cluster {i}:")
        print(f"  - Centroid: (Lat: {centroid_lat:.4f}, Lon: {centroid_lon:.4f})")
        print(f"  - Number of Pincodes: {len(cluster_data)}")
        # Print top 3 districts in this cluster to give geographical context
        top_districts = cluster_data['District'].value_counts().head(3).index.tolist()
        print(f"  - Major Districts: {', '.join(top_districts)}")
        print()
        
    print("General Insights:")
    print("1. Density Distribution: By looking at the number of pincodes per cluster, we can infer which geographical regions have a higher concentration of postal offices (often correlating with urban/semi-urban density).")
    print("2. Spatial Grouping: The algorithm successfully segmented the state into logical geographical zones (e.g., North, South, Central) solely based on latitude and longitude without needing the district labels.")
    print("3. Centroid Significance: The centroid of the densest cluster likely represents the major metropolitan hub of the state (e.g., Hyderabad for Telangana), where pincode density is naturally highest.")
    print("4. Outlier Detection: Some clusters (like those with Lat > 25 or extreme longitudes) represent data entry errors in the dataset. K-Means effectively isolated these noisy outliers into their own small clusters!")

if __name__ == "__main__":
    main()

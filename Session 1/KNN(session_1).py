import numpy as np

# --- Step 1: Data Preparation & Encoding ---

data = [
    [150, 7.0, 1, 'Apple'],
    [120, 6.5, 0, 'Banana'],
    [180, 7.5, 2, 'Orange'],
    [155, 7.2, 1, 'Apple'],
    [110, 6.0, 0, 'Banana'],
    [190, 7.8, 2, 'Orange'],
    [145, 7.1, 1, 'Apple'],
    [115, 6.3, 0, 'Banana']
]

# Mapping strings to integers
label_to_idx = {'Apple': 0, 'Banana': 1, 'Orange': 2}
idx_to_label = {0: 'Apple', 1: 'Banana', 2: 'Orange'}

# Convert to NumPy arrays
X = np.array([row[:3] for row in data], dtype=float)
y = np.array([label_to_idx[row[3]] for row in data])

# --- Step 2 & 3: KNN Classifier Class ---

class KNN:
    def __init__(self, k=3):
        """
        Initialize with number of neighbors.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        KNN 'Learning' is just storing the training data.
        """
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, p1, p2):
        """
        Calculates the straight-line distance between two points.
        Formula: sqrt(sum((x - y)^2))
        """
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def predict_one(self, x):
        """
        Predict the label for a single sample.
        """
        # 1. Calculate distances from x to all training points
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # 2. Get indices of the k smallest distances
        k_indices = np.argsort(distances)[:self.k]
        
        # 3. Extract the labels of those k neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # 4. Majority Vote: Find the most frequent label
        counts = np.bincount(k_nearest_labels)
        return np.argmax(counts)

    def predict(self, X_test):
        """
        Predict labels for an array of test samples.
        """
        return np.array([self.predict_one(x) for x in X_test])


# --- Step 4 & 5: Testing and Evaluation ---

def run_test():
    test_data = np.array([
        [118, 6.2, 0],  # Expected: Banana
        [160, 7.3, 1],  # Expected: Apple
        [185, 7.7, 2]   # Expected: Orange
    ])

    # Initialize model
    k_value = 3
    knn = KNN(k=k_value)
    knn.fit(X, y)

    # Perform prediction
    predictions = knn.predict(test_data)

    print(f"--- KNN Classification (k={k_value}) ---")
    for i, pred in enumerate(predictions):
        label = idx_to_label[pred]
        print(f"Test Sample {i+1} Features: {test_data[i]} -> Predicted: {label}")

    # Bonus: Simple Accuracy Checker
    expected_labels = ['Banana', 'Apple', 'Orange']
    predicted_labels = [idx_to_label[p] for p in predictions]
    
    correct = sum(1 for p, e in zip(predicted_labels, expected_labels) if p == e)
    accuracy = (correct / len(expected_labels)) * 100
    print(f"\nAccuracy: {accuracy:.1f}%")

if __name__ == "__main__":
    run_test()

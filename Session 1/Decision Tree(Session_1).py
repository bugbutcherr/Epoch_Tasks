import numpy as np

# --- Step 1: Dataset Preparation & Encoding ---

data = [
    [12.0, 1.5, 1, 'Wine'],
    [5.0, 2.0, 0, 'Beer'],
    [40.0, 0.0, 1, 'Whiskey'],
    [13.5, 1.2, 1, 'Wine'],
    [4.5, 1.8, 0, 'Beer'],
    [38.0, 0.1, 1, 'Whiskey'],
    [11.5, 1.7, 1, 'Wine'],
    [5.5, 2.3, 0, 'Beer']
]

# Manual Encoding
label_map = {'Beer': 0, 'Wine': 1, 'Whiskey': 2}
reverse_map = {0: 'Beer', 1: 'Wine', 2: 'Whiskey'}

X = np.array([[row[0], row[1], row[2]] for row in data], dtype=float)
y = np.array([label_map[row[3]] for row in data])

feature_names = ['Alcohol', 'Sugar', 'Color']

# --- Core Functions: Gini & Entropy ---

def calculate_gini(labels):
    """Computes Gini Impurity: 1 - sum(squared probabilities)"""
    if len(labels) == 0: return 0
    counts = np.bincount(labels)
    probs = counts / len(labels)
    return 1.0 - np.sum(probs**2)

def calculate_entropy(labels):
    """Bonus: Computes Information Entropy: -sum(p * log2(p))"""
    if len(labels) == 0: return 0
    counts = np.bincount(labels)
    probs = counts[counts > 0] / len(labels)
    return -np.sum(probs * np.log2(probs))

# --- Step 4: Node and Tree Implementation ---

class Node:
    """A container for tree branches and leaves."""
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # If this is not None, it's a leaf node

class DecisionTree:
    def __init__(self, criterion='gini', max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None

    def _get_impurity(self, labels):
        if self.criterion == 'gini':
            return calculate_gini(labels)
        return calculate_entropy(labels)

    def _best_split(self, X, y):
        """Step 3: Loops through features/thresholds to find the best cut."""
        best_gain = -1
        split_idx, split_thresh = None, None
        
        n_samples, n_features = X.shape
        parent_impurity = self._get_impurity(y)

        for feat_idx in range(n_features):
            # Test all unique values in the feature as potential thresholds
            thresholds = np.unique(X[:, feat_idx])
            for thresh in thresholds:
                left_mask = X[:, feat_idx] <= thresh
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Weighted impurity of children
                n_l, n_r = np.sum(left_mask), np.sum(right_mask)
                imp_l, imp_r = self._get_impurity(y[left_mask]), self._get_impurity(y[right_mask])
                child_impurity = (n_l / n_samples) * imp_l + (n_r / n_samples) * imp_r

                # Information Gain
                gain = parent_impurity - child_impurity

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh

        return split_idx, split_thresh

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        """Step 4: Recursive Tree Building."""
        num_samples, num_features = X.shape
        unique_labels = np.unique(y)

        # Base Cases: Pure node, Max depth, or no samples
        if len(unique_labels) == 1 or depth >= self.max_depth or num_samples < 2:
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)

        # Find best split
        feat_idx, thresh = self._best_split(X, y)
        
        if feat_idx is None:
            return Node(value=np.bincount(y).argmax())

        # Recursion
        left_mask = X[:, feat_idx] <= thresh
        right_mask = ~left_mask
        
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(feat_idx, thresh, left_child, right_child)

    def predict(self, X):
        """Step 5: Prediction Logic."""
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def print_tree(self, node=None, depth=0):
        """Bonus: Pretty print the tree."""
        if node is None: node = self.root
        
        if node.value is not None:
            print("  " * depth + f"Predict: {reverse_map[node.value]}")
            return

        print("  " * depth + f"[{feature_names[node.feature_index]} <= {node.threshold}]")
        self.print_tree(node.left, depth + 1)
        print("  " * depth + f"[{feature_names[node.feature_index]} > {node.threshold}]")
        self.print_tree(node.right, depth + 1)

# --- Step 6: Evaluation ---

# Initialize and train
clf = DecisionTree(criterion='gini', max_depth=3)
clf.fit(X, y)

# Predictions on training data
train_preds = clf.predict(X)
accuracy = np.mean(train_preds == y)
print(f"Training Accuracy: {accuracy * 100:.1f}%\n")

# Testing on requested samples
test_data = np.array([
    [6.0, 2.1, 0],   # Expected: Beer
    [39.0, 0.05, 1], # Expected: Whiskey
    [13.0, 1.3, 1]   # Expected: Wine
])

print("Test Data Predictions:")
test_results = clf.predict(test_data)
expected = ['Beer', 'Whiskey', 'Wine']

for i, pred in enumerate(test_results):
    print(f"Sample {i+1}: Predicted = {reverse_map[pred]}, Expected = {expected[i]}")

print("\nTree Structure Visualized:")
clf.print_tree()

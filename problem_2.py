from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# === 1. Load the Iris dataset ===
iris = load_iris()
X, y = iris.data, iris.target

# === 2. Train/Test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 3. Train Decision Tree Classifier ===
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# === 4. Predictions and Accuracy ===
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# === 5. Plot the Decision Tree ===
plt.figure(figsize=(12, 8))
plot_tree(
    clf,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)
plt.title("Decision Tree for Iris Dataset")
plt.savefig("outputs/decision_tree_iris_problem_2.png")
plt.close()

print("Decision tree plot saved as 'decision_tree_iris_problem_2.png'")

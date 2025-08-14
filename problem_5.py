import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# === 1. Load Iris dataset ===
iris = load_iris()
X = iris.data
y = iris.target

# === 2. Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# === 3. Train classifier ===
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)

# === 4. Predict ===
y_pred = clf.predict(X_test)

# === 5. Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)

# === 6. Plot ===
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression (Iris Dataset)")
plt.savefig("outputs/confusion_matrix_problem_5.png")
plt.close()

print("Confusion matrix plot saved as 'confusion_matrix_problem_5.png' in the outputs folder")

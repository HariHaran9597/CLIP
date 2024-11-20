from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assume you have 10 classes for simplicity.
# Generate mock true labels and high-accuracy predictions.

# Step 1: Create true labels and dummy predictions
n_samples = 100  # Adjust as needed
n_classes = 10

# True labels and dummy predictions (set high similarity to simulate accuracy)
true_labels = np.random.randint(0, n_classes, n_samples)
predicted_labels = true_labels.copy()  # Start with perfect accuracy

# Introduce a few incorrect predictions to make it realistic
for i in range(int(0.1 * n_samples)):  # Simulate 90% accuracy
    predicted_labels[i] = (predicted_labels[i] + 1) % n_classes

# Step 2: Compute and print classification metrics
print("Classification Report:")
print(classification_report(true_labels, predicted_labels))

# Step 3: Confusion Matrix Visualization
conf_matrix = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(n_classes), yticklabels=range(n_classes))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Step 4: Print overall accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Simulated Accuracy: {accuracy * 100:.2f}%")

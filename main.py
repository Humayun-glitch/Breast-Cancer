import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# Import and load dataset (CSV File)
data = pd.read_csv('cancer_data.csv')

# Drop 'id' and 'Unnamed: 32' columns
data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

# Convert 'M' (Malignant) and 'B' (Benign) to 1 and 0
data['diagnosis'] = (data['diagnosis'] == 'M').astype(int)

# Extract features and target
X = data.drop('diagnosis', axis=1).values
y = data['diagnosis'].values.reshape(-1, 1)

# Implement polynomial features
def add_polynomial_features(X, degree=2):
    X_poly = X[:, 1:]  # Exclude the bias term
    for d in range(2, degree + 1):
        X_poly = np.concatenate((X_poly, X[:, 1:]**d), axis=1)
    return np.concatenate((X[:, [0]], X_poly), axis=1)

X_poly = add_polynomial_features(X, degree=2)

# Implement min-max normalization
def minmax_normalization(X):
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    return (X - min_vals) / (max_vals - min_vals)

X_scaled = minmax_normalization(X_poly)

# Logistic regression loss function and gradient descent algorithm
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss_function(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    return (-1 / m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    losses = []
    for _ in range(iterations):
        h = sigmoid(X @ theta)
        gradient = X.T @ (h - y) / m
        theta -= learning_rate * gradient
        loss = loss_function(X, y, theta)
        losses.append(loss)
    return theta, losses

# Add bias term to X_scaled
X_scaled = np.concatenate((np.ones((X_scaled.shape[0], 1)), X_scaled), axis=1)

# Initialize parameters and perform gradient descent
theta_initial = np.zeros((X_scaled.shape[1], 1))
learning_rate = 0.01
iterations = 1000

theta_final, losses = gradient_descent(X_scaled, y, theta_initial, learning_rate, iterations)


# Plot learning curve for training data
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, iterations + 1), np.squeeze(losses))
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Learning Curve')

# Generate test predictions
X_test_poly = add_polynomial_features(X, degree=2)
X_test_scaled = minmax_normalization(X_test_poly)
X_test_scaled = np.concatenate((np.ones((X_test_scaled.shape[0], 1)), X_test_scaled), axis=1)

y_pred = (sigmoid(X_test_scaled @ theta_final) >= 0.5).astype(int)

# Confusion Matrix
conf_matrix = np.zeros((2, 2))
for true_label, pred_label in zip(y.flatten(), y_pred.flatten()):
    conf_matrix[true_label, pred_label] += 1

# Evaluation Metrics
TP, FP, FN, TN = conf_matrix.flatten()
precision = TP / (TP + FP)
recall = TP / (TP + FN)
accuracy = (TP + TN) / (TP + FP + FN + TN)
f1_score = 2 * (precision * recall) / (precision + recall)

# Learning rates to try
learning_rates = [0.0001, 0.0005, 0.0007, 0.001, 0.005, 0.01, 0.05, 0.09, 0.1, 0.4, 0.7]

# Lists to store learning rates and associated F1 scores
learning_rate_list = []
f1_score_list = []

# Loop over different learning rates
for learning_rate in learning_rates:
    # Initialize parameters and perform gradient descent
    theta_initial = np.zeros((X_scaled.shape[1], 1))
    iterations = 1000

    theta_final, _ = gradient_descent(X_scaled, y, theta_initial, learning_rate, iterations)

    # Generate test predictions
    X_test_poly = add_polynomial_features(X, degree=2)
    X_test_scaled = minmax_normalization(X_test_poly)
    X_test_scaled = np.concatenate((np.ones((X_test_scaled.shape[0], 1)), X_test_scaled), axis=1)

    y_pred = (sigmoid(X_test_scaled @ theta_final) >= 0.5).astype(int)

    # Compute F1 score
    conf_matrix = np.zeros((2, 2))
    for true_label, pred_label in zip(y.flatten(), y_pred.flatten()):
        conf_matrix[true_label, pred_label] += 1

    TP, FP, FN, TN = conf_matrix.flatten()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Store results in lists
    learning_rate_list.append(learning_rate)
    f1_score_list.append(f1_score)

    # Plot learning curve for test data
    plt.plot(learning_rate_list, f1_score_list, marker='o')
    plt.xscale('log')  # Set x-axis to a logarithmic scale for better visualization
    plt.xlabel('Learning Rate')
    plt.ylabel('F1 Score')
    plt.title('Test Learning Curve')

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


print("Confusion Matrix:")
print(conf_matrix)
print("\nPrecision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
print("F1 Score:", f1_score)
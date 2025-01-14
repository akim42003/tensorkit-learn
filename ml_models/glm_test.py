import tensor_slow as ts
from glm import GLM
from optimizers.loss_functions import Binary_CELoss
from link_functions import LogitLink
import math

# Parameters for dataset
num_samples = 100  # Number of samples
num_features = 5   # Number of features

# Generate synthetic input data (features)
X_data = []
for i in range(num_samples):
    for j in range(num_features):
        X_data.append((i + 1) * (j + 1))  # Structured feature scaling

X = ts.Tensor.from_values([num_samples, num_features], X_data)

# Generate weights for linear combination
true_weights = [5, -3, 2, 7, -5]  # True weights for synthetic data
true_weights_tensor = ts.Tensor.from_values([num_features, 1], true_weights)

# Compute linear predictions and convert to probabilities with sigmoid
linear_predictions = X.matmul(true_weights_tensor)  # Shape: [num_samples, 1]
sigmoid = lambda x: 1 / (1 + math.exp(-x))

# Compute probabilities and create binary labels
y_data = []
for i in range(num_samples):
    prob = sigmoid(linear_predictions[[i, 0]])  # Access element (i, 0)
    y_data.append(1.0 if prob >= 0.5 else 0.0)

y = ts.Tensor.from_values([num_samples, 1], y_data)

# Initialize GLM with LogitLink and Binary Cross Entropy Loss
link_function = LogitLink()
loss_function = Binary_CELoss()
model = GLM(num_features=num_features, link_function=link_function, loss_function=loss_function)

# Train the model
print("Training the GLM on a synthetic dataset...")
model.train(X, y, learning_rate=0.0001, max_iterations=5, tolerance=1e-6)

# Print the trained coefficients
print("Trained coefficients:")
model.get_coefficients().print()

# Compare trained coefficients with true weights
print("True weights:")
true_weights_tensor.print()

import tensor_slow as ts

# Helper functions for scalar multiplication, generating indices, and tensor dot product:

def generate_indices(shape):
    """Recursively generate all index lists for a tensor of the given shape."""
    if not shape:
        return [[]]
    result = []
    for i in range(shape[0]):
        for sub in generate_indices(shape[1:]):
            result.append([i] + sub)
    return result

def scalar_multiply(t, scalar):
    """Multiply every element of tensor 't' by the given scalar."""
    shape = t.getShape()
    indices = generate_indices(shape)
    new_values = []
    for idx in indices:
        new_values.append(t[list(idx)] * scalar)
    return ts.Tensor.from_values(shape, new_values)

def tensor_dot(t1, t2):
    """
    Computes the dot product between two tensors of the same shape by summing
    the products of corresponding elements.
    """
    shape = t1.getShape()
    if shape != t2.getShape():
        raise ValueError("Shapes must match for tensor_dot")
    indices = generate_indices(shape)
    total = 0.0
    for idx in indices:
        total += t1[list(idx)] * t2[list(idx)]
    return total

class SVM:
    def __init__(self, learning_rate=0.01, lambda_reg=0.01, epochs=1000):
        self.lr = learning_rate
        self.lambda_reg = lambda_reg
        self.epochs = epochs
        self.weights = None  # Will be initialized in fit()
        self.bias = 0        # Scalar bias

    def compute_loss(self, X, y):
        """Computes hinge loss plus L2 regularization."""
        n_samples = X.getShape()[0]
        # Create a bias tensor of shape [n_samples, 1]
        bias_tensor = ts.Tensor.from_values([n_samples, 1], [self.bias] * n_samples)
        pred = X.matmul(self.weights).tplus(bias_tensor)
        margins = y.tminus(pred)
        hinge_loss = margins.clamp(0, float("inf"))
        # reg_loss: (lambda/2)*||w||^2
        reg_loss = tensor_dot(self.weights, self.weights) * (self.lambda_reg / 2)
        # Sum all elements of hinge_loss. Here, create a ones tensor matching hinge_loss shape.
        ones_tensor = ts.Tensor.from_values([n_samples, 1], [1.0] * n_samples)
        total_hinge = tensor_dot(hinge_loss, ones_tensor)
        # Return the total loss as a scalar (hinge loss sum + reg_loss)
        return total_hinge + reg_loss

    def fit(self, X, y):
        """Trains the SVM using gradient descent."""
        n_samples, n_features = X.getShape()
        self.weights = ts.Tensor.zeros([n_features, 1])  # Initialize weights as zeros

        for epoch in range(self.epochs):
            for i in range(n_samples):
                # Manually extract row i as a tensor of shape [1, n_features]
                xi_values = [X[i, j] for j in range(n_features)]
                xi = ts.Tensor.from_values([1, n_features], xi_values)
                yi = y[i, 0]  # Extract scalar label (float)

                # Convert bias to a 1x1 tensor
                bias_tensor = ts.Tensor.from_values([1, 1], [self.bias])
                # Compute prediction: xi @ weights + bias
                pred = xi.matmul(self.weights).tplus(bias_tensor)
                # Multiply the prediction by the scalar label using our helper
                margin_tensor = scalar_multiply(pred, yi)
                margin = margin_tensor[0, 0]  # Extract scalar value

                if margin >= 1:
                    # Correct classification: update weights with regularization only
                    self.weights = self.weights.tminus(scalar_multiply(self.weights, self.lr * self.lambda_reg))
                else:
                    # Misclassified: update weights and bias
                    # Transpose xi to convert its shape from [1, n_features] to [n_features, 1]
                    update_term = scalar_multiply(xi.Tp(), yi * self.lr)
                    self.weights = self.weights.tminus(
                        scalar_multiply(self.weights, self.lr * self.lambda_reg).tminus(update_term)
                    )
                    self.bias -= self.lr * yi

            if epoch % 100 == 0:
                loss = self.compute_loss(X, y)
                print(f"Epoch {epoch}: Loss = {loss}")

    def predict(self, X):
        """Predicts class labels (-1 or 1)."""
        n_samples = X.getShape()[0]
        bias_tensor = ts.Tensor.from_values([n_samples, 1], [self.bias] * n_samples)
        pred = X.matmul(self.weights).tplus(bias_tensor)
        return pred.clamp(-1, 1)

    def decision_function(self, X):
        """Computes the raw margin scores."""
        n_samples = X.getShape()[0]
        bias_tensor = ts.Tensor.from_values([n_samples, 1], [self.bias] * n_samples)
        return X.matmul(self.weights).tplus(bias_tensor)

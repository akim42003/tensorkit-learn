import tensor_slow as ts

# Helper functions:

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
    """
    Multiply every element of tensor 't' by the given scalar.
    Returns a new Tensor.
    """
    shape = t.getShape()
    indices = generate_indices(shape)
    new_values = []
    for idx in indices:
        new_values.append(t[list(idx)] * scalar)
    return ts.Tensor.from_values(shape, new_values)

def elementwise_multiply(t1, t2):
    """
    Multiply two tensors elementwise.
    Both tensors must have the same shape.
    """
    shape = t1.getShape()
    if shape != t2.getShape():
        raise ValueError("Shapes must match for elementwise multiplication")
    indices = generate_indices(shape)
    new_values = []
    for idx in indices:
        new_values.append(t1[list(idx)] * t2[list(idx)])
    return ts.Tensor.from_values(shape, new_values)

def tensor_dot(t1, t2):
    """
    Compute the dot product between two tensors of the same shape
    by summing the products of corresponding elements.
    Returns a scalar.
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
    def __init__(self, learning_rate=0.0001, lambda_reg=0.01, epochs=1000):
        self.lr = learning_rate
        self.lambda_reg = lambda_reg
        self.epochs = epochs
        self.weights = None  # Will be initialized in fit()
        self.bias = 0        # Scalar bias

    def compute_loss(self, X, y):
        """
        Computes total loss = sum_i max(0, 1 - y_i * (x_i · w + b))
        + (lambda/2)*||w||^2.
        """
        n_samples = X.getShape()[0]
        # Create bias tensor of shape [n_samples, 1]
        bias_tensor = ts.Tensor.from_values([n_samples, 1], [self.bias] * n_samples)
        pred = X.matmul(self.weights).tplus(bias_tensor)
        # Compute elementwise product: y * pred.
        prod = elementwise_multiply(y, pred)
        # Create a ones tensor for each sample: shape [n_samples, 1]
        ones_tensor = ts.Tensor.from_values([n_samples, 1], [1.0] * n_samples)
        # Compute margins: 1 - (y * pred)
        margins = ones_tensor.tminus(prod)
        hinge_loss = margins.clamp(0, float("inf"))
        total_hinge = 0.0
        for idx in generate_indices(hinge_loss.getShape()):
            total_hinge += hinge_loss[list(idx)]
        reg_loss = tensor_dot(self.weights, self.weights) * (self.lambda_reg / 2)
        return total_hinge + reg_loss

    def fit(self, X, y):
        """
        Trains the SVM using gradient descent.
        For each sample, if y * (x·w + b) >= 1, the gradient is lambda * w;
        otherwise, it is lambda * w - y * x.
        """
        n_samples, n_features = X.getShape()
        self.weights = ts.Tensor.zeros([n_features, 1])  # Initialize weights as zeros

        for epoch in range(self.epochs):
            for i in range(n_samples):
                # Extract row i as tensor of shape [1, n_features]
                xi_values = [X[i, j] for j in range(n_features)]
                xi = ts.Tensor.from_values([1, n_features], xi_values)
                yi = y[i, 0]  # Extract scalar label

                # Create bias tensor for this sample
                bias_tensor = ts.Tensor.from_values([1, 1], [self.bias])
                pred = xi.matmul(self.weights).tplus(bias_tensor)
                # Compute margin: y_i * (x_i · w + b)
                margin_tensor = scalar_multiply(pred, yi)
                margin = margin_tensor[0, 0]  # Extract scalar

                if margin >= 1:
                    # Correct classification: gradient = lambda * w
                    self.weights = self.weights.tminus(scalar_multiply(self.weights, self.lr * self.lambda_reg))
                else:
                    # Misclassified: gradient = lambda * w - y_i * x_i
                    # We need to update weights: subtract lr * gradient.
                    # To get y_i * x_i with shape [n_features, 1], transpose xi.
                    update_term = scalar_multiply(xi.Tp(), yi * self.lr)
                    self.weights = self.weights.tminus(
                        scalar_multiply(self.weights, self.lr * self.lambda_reg).tminus(update_term)
                    )
                    self.bias -= self.lr * yi

            if epoch % 100 == 0:
                loss = self.compute_loss(X, y)
                print(f"Epoch {epoch}: Loss = {loss}")

    def predict(self, X):
        """Predicts class labels (-1 or 1) for each sample in X."""
        n_samples = X.getShape()[0]
        bias_tensor = ts.Tensor.from_values([n_samples, 1], [self.bias] * n_samples)
        pred = X.matmul(self.weights).tplus(bias_tensor)
        return pred.clamp(-1, 1)

    def decision_function(self, X):
        """Computes the raw margin scores for X."""
        n_samples = X.getShape()[0]
        bias_tensor = ts.Tensor.from_values([n_samples, 1], [self.bias] * n_samples)
        return X.matmul(self.weights).tplus(bias_tensor)

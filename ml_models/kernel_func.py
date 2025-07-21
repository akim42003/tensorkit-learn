import tensor_slow as ts
import math

# We'll reuse generate_indices from before for elementwise operations
def generate_indices(shape):
    """Recursively generate all index lists for a tensor of the given shape."""
    if not shape:
        return [[]]
    result = []
    for i in range(shape[0]):
        for sub in generate_indices(shape[1:]):
            result.append([i] + sub)
    return result

class KernelFunctions:
    @staticmethod
    def linear(X, Y):
        """
        Computes the linear kernel matrix between tensors X and Y.
        That is, K = X @ Y^T.
        """
        # Assuming Y.Tp() returns the transpose of Y
        return X.matmul(Y.Tp())

    @staticmethod
    def polynomial(X, Y, degree=3, coef0=1):
        """
        Computes the polynomial kernel matrix:
            K(x, y) = (coef0 + x·y)^degree
        """
        # First compute the dot product matrix K = X @ Y^T
        K = X.matmul(Y.Tp())
        # We need to add coef0 to every element of K.
        shape = K.getShape()
        total_elements = 1
        for dim in shape:
            total_elements *= dim
        # Create a list of constant values equal to coef0.
        coef_list = [coef0] * total_elements
        coef_tensor = ts.Tensor.from_values(shape, coef_list)
        # Elementwise add coef_tensor to K:
        # Here, since your Tensor class doesn't overload the + operator,
        # we can use the provided tplus method if it expects another Tensor.
        K_plus = K.tplus(coef_tensor)
        # Now raise each element to the given degree.
        indices = generate_indices(shape)
        new_values = []
        for idx in indices:
            new_values.append(math.pow(K_plus[list(idx)], degree))
        return ts.Tensor.from_values(shape, new_values)

    @staticmethod
    def rbf(X, Y, gamma=0.1):
        """
        Computes the RBF (Gaussian) kernel matrix:
            K(x, y) = exp(-gamma * ||x - y||^2)
        """
        # Get shapes
        n_samples_X, n_features = X.getShape()
        n_samples_Y, _ = Y.getShape()
        K_values = []
        # For each pair of rows in X and Y, compute the squared Euclidean distance.
        for i in range(n_samples_X):
            for j in range(n_samples_Y):
                # Manually extract row i from X and row j from Y
                xi = [X[i, k] for k in range(n_features)]
                yj = [Y[j, k] for k in range(n_features)]
                # Compute squared Euclidean distance
                sq_dist = sum((xi[k] - yj[k])**2 for k in range(n_features))
                K_values.append(math.exp(-gamma * sq_dist))
        return ts.Tensor.from_values([n_samples_X, n_samples_Y], K_values)

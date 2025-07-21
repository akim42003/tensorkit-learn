import tensor_slow as ts
import math
from kernel_func import KernelFunctions  # our KernelFunctions class from before

class SVM:
    def __init__(self, C=1.0, kernel="linear", degree=3, coef0=1, gamma=0.1, epochs=1000, learning_rate=0.001):
        """
        C: Regularization parameter (upper bound on alpha)
        kernel: Kernel type ("linear", "polynomial", "rbf")
        degree, coef0, gamma: Kernel parameters for polynomial/RBF kernels
        epochs, learning_rate: Optimization parameters for updating alpha
        """
        self.C = C
        self.kernel_type = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.epochs = epochs
        self.lr = learning_rate

        # These will be set during training:
        self.alpha = None  # Lagrange multipliers (a Tensor of shape [n_samples, 1])
        self.bias = 0
        self.X_train = None  # Keep a copy of training inputs
        self.y_train = None  # and training labels

        # Select kernel function from KernelFunctions
        if kernel == "linear":
            self.kernel_func = KernelFunctions.linear
        elif kernel == "polynomial":
            self.kernel_func = lambda X, Y: KernelFunctions.polynomial(X, Y, degree, coef0)
        elif kernel == "rbf":
            self.kernel_func = lambda X, Y: KernelFunctions.rbf(X, Y, gamma)
        else:
            raise ValueError("Unsupported kernel type")

    def compute_kernel_matrix(self, X):
        """
        Computes the kernel matrix between the training data and X.
        If X is the training set, returns K of shape [n_samples, n_samples].
        """
        return self.kernel_func(self.X_train, X)

    def compute_objective(self):
        """
        Computes the dual objective:
        """
        n_samples = self.y_train.getShape()[0]
        # Sum of alphas:
        sum_alpha = 0.0
        for i in range(n_samples):
            sum_alpha += self.alpha[i, 0]
        # Double sum term:
        double_sum = 0.0
        # Precompute kernel matrix K_train
        K_train = self.compute_kernel_matrix(self.X_train)
        for i in range(n_samples):
            for j in range(n_samples):
                double_sum += self.alpha[i, 0] * self.alpha[j, 0] * self.y_train[i, 0] * self.y_train[j, 0] * K_train[i, j]
        return sum_alpha - 0.5 * double_sum

    def fit(self, X, y):
        """
        Trains the kernel SVM using a simple gradient ascent on the dual objective.
        (For a complete implementation, one must also enforce the equality constraint. Here we use a simple approach for learning.)
        """
        n_samples, _ = X.getShape()
        # Save training data for later use in prediction.
        self.X_train = X
        self.y_train = y

        self.alpha = ts.Tensor.zeros([n_samples, 1])
        # Simple gradient ascent loop
        for epoch in range(self.epochs):

            K_train = self.compute_kernel_matrix(X)  # shape: [n_samples, n_samples]
            for i in range(n_samples):
                grad = 1.0
                for j in range(n_samples):
                    grad -= self.alpha[j, 0] * y[i, 0] * y[j, 0] * K_train[i, j]
                # Update α_i using gradient ascent (and clip to [0, C])
                new_alpha = self.alpha[i, 0] + self.lr * grad
                if new_alpha < 0:
                    new_alpha = 0
                elif new_alpha > self.C:
                    new_alpha = self.C
                self.alpha[i, 0] = new_alpha
            # (Optional) Update bias using KKT conditions
            # For simplicity, we set bias to 0 here.
            self.bias = 0
            if epoch % 100 == 0:
                obj = self.compute_objective()
                print(f"Epoch {epoch}: Dual Objective = {obj}")

    def decision_function(self, X):
        """
        Computes the decision function for each input sample in X:
        """
        K_test = self.kernel_func(self.X_train, X)  # shape: [n_train, n_test]
        n_train, n_test = K_test.getShape()
        # Compute the sum for each test sample:
        f_vals = [0.0] * n_test
        for j in range(n_test):
            total = 0.0
            for i in range(n_train):
                total += self.alpha[i, 0] * self.y_train[i, 0] * K_test[i, j]
            f_vals[j] = total + self.bias
        # Return f_vals as a Tensor of shape [n_test, 1]
        return ts.Tensor.from_values([n_test, 1], f_vals)

    def predict(self, X):
        """
        Predicts class labels (-1 or 1) based on the decision function.
        """
        f = self.decision_function(X)
        n_test, _ = f.getShape()
        predictions = []
        for i in range(n_test):
            # Using a simple threshold at 0:
            predictions.append(1 if f[i, 0] >= 0 else -1)
        return ts.Tensor.from_values([n_test, 1], predictions)

# Generalized Linear Models (GLM)
import tensor_slow as ts
from optimizers.loss_functions import MSELoss, Binary_CELoss


class GLM:
    def __init__(self, num_features, link_function, loss_function):
        """
        Initialize the GLM.
        Args:
            num_features: Number of input features.
            link_function: Link function to transform the linear predictor.
            loss_function: Loss function to compute the error.
        """
        self.coefficients = ts.Tensor.from_values([num_features, 1], [0.0] * num_features)  # Initialize weights
        self.link_function = link_function
        self.loss_function = loss_function

    def forward(self, X):
        """
        Perform the forward pass.
        Args:
            X: Input tensor of shape (num_samples, num_features).
        Returns:
            Predictions (mu) after applying the link function's inverse.
        """
        eta = X.matmul(self.coefficients)  # Linear predictor: eta = X * coefficients
        mu = self.link_function.inverse(eta)  # Apply inverse link function
        return mu

    def compute_loss(self, X, y):
        """
        Compute the loss for the given inputs and targets.
        Args:
            X: Input tensor.
            y: Target tensor.
        Returns:
            Loss value.
        """
        predictions = self.forward(X)
        return self.loss_function.forward(predictions, y)

    def train(self, X, y, learning_rate=0.01, epochs=100):
        """
        Train the model using gradient descent.

        Args:
            X: Input tensor of shape (num_samples, num_features).
            y: Target tensor of shape (num_samples, 1).
            learning_rate: Learning rate for gradient descent.
            epochs: Number of epochs to train for.
        """
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)

            # Compute gradients
            gradients = self.loss_function.backward(predictions, y)

            # Compute updates
            updates = X.Tp().matmul(gradients)  # Shape: [num_features, 1]

            # Dynamically determine the shape of updates
            updates_shape = [X.getShape()[1], 1]  # [num_features, 1]

            # Create a tensor filled with learning_rate, matching the shape of updates
            ones_matrix = ts.Tensor.from_values(updates_shape, [learning_rate] * (updates_shape[0] * updates_shape[1]))

            # Add scaled ones_matrix to updates
            updates = updates.tplus(ones_matrix)

            # Update coefficients
            self.coefficients = self.coefficients.tminus(updates)

            # Print loss for debugging
            current_loss = self.compute_loss(X, y)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {current_loss}")



    def get_coefficients(self):
        """
        Get the trained coefficients.
        Returns:
            Tensor of coefficients.
        """
        return self.coefficients

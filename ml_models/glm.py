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

    def train(self, X, y, learning_rate, max_iterations, tolerance):
        """
        Train the model using an iterative solver with stability fixes.
        """
        epsilon = 1e-6  # Small value to prevent division/log issues
        previous_loss = float('inf')

        for iteration in range(max_iterations):
            # Forward pass
            predictions = self.forward(X)
            predictions = predictions.clamp(epsilon, 1 - epsilon)
            # Compute gradients
            gradients = self.loss_function.backward(predictions, y)
            # gradients.print()

            # Gradient clipping for stability
            gradients = gradients.clamp(-1.0, 1.0)
            # gradients.print()

            # Compute updates
            updates = X.Tp().matmul(gradients)  # Shape: [num_features, 1]
            # updates.print()
            # Scale updates by learning rate
            updates = updates.scalar_multiply(learning_rate)

            # Update coefficients
            self.coefficients = self.coefficients.tminus(updates)

            # Compute current loss
            current_loss = self.compute_loss(X, y)

            # Check for NaN values
            # if ts.isnan(current_loss):
            #     print("Encountered NaN loss. Stopping training.")
            #     break

            # Check for convergence
            if abs(previous_loss - current_loss) < tolerance:
                print(f"Converged after {iteration + 1} iterations. Final Loss: {current_loss}")
                break

            previous_loss = current_loss

            # Print loss for debugging
            print(f"Iteration {iteration + 1}, Loss: {current_loss}")

        else:
            print(f"Reached maximum iterations ({max_iterations}) without full convergence.")



    def get_coefficients(self):
        """
        Get the trained coefficients.
        Returns:
            Tensor of coefficients.
        """
        return self.coefficients

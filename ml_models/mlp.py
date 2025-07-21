import mlp_cpp
import tensor_slow as ts


class MLP:

    def __init__(self):
        """Initialize empty MLP."""
        self.model = mlp_cpp.MLP()
        self.loss_function = None

    def add_layer(self, input_dim, output_dim, activation='relu', init_method='xavier'):
        """
        Add a layer to the network.

        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            activation: Activation function ('relu', 'sigmoid', 'tanh', 'linear')
            init_method: Weight initialization ('xavier', 'he')
        """
        # Create activation function
        if activation == 'relu':
            act_fn = mlp_cpp.ReLU()
        elif activation == 'sigmoid':
            act_fn = mlp_cpp.Sigmoid()
        elif activation == 'tanh':
            act_fn = mlp_cpp.Tanh()
        elif activation == 'linear':
            act_fn = mlp_cpp.Linear()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

        self.model.add_layer(input_dim, output_dim, act_fn, init_method)

    def build(self, layer_dims, activations=None, loss_function=None, init_method='xavier'):
        """
        Build the entire network at once.

        Args:
            layer_dims: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
            activations: List of activation functions for each layer (excluding input)
            loss_function: Loss function instance
            init_method: Weight initialization method
        """
        self.loss_function = loss_function

        # Default activations: ReLU for hidden layers, linear for output
        if activations is None:
            activations = ['relu'] * (len(layer_dims) - 2) + ['linear']

        # Add layers
        for i in range(len(layer_dims) - 1):
            activation = activations[i] if i < len(activations) else 'relu'
            self.add_layer(layer_dims[i], layer_dims[i+1], activation, init_method)

    def forward(self, X):
        """
        Forward pass through the network.

        Args:
            X: Input tensor

        Returns:
            Output tensor
        """
        return self.model.forward(X)

    def predict(self, X):
        """Alias for forward pass."""
        return self.forward(X)

    def fit(self, X, y, epochs=100, learning_rate=0.01, batch_size=None, verbose=True):
        """
        Train the MLP.

        Args:
            X: Input data tensor
            y: Target data tensor
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size (None for full batch)
            verbose: Print progress
        """
        if self.loss_function is None:
            raise ValueError("Loss function not set. Use build() or set loss_function manually.")

        n_samples = X.getShape()[0]

        if batch_size is None:
            batch_size = n_samples

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            # Process in batches
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)

                # Get batch (simplified - assumes data is already in tensor format)
                # For a more robust implementation, you'd want proper batching
                batch_X = X  # Simplified for now
                batch_y = y  # Simplified for now

                # Forward pass
                predictions = self.forward(batch_X)

                # Compute loss
                loss = self.loss_function.forward(predictions, batch_y)
                if isinstance(loss, float):
                    epoch_loss += loss
                else:
                    epoch_loss += self._tensor_to_float(loss)
                n_batches += 1

                # Backward pass
                loss_grad = self.loss_function.backward(predictions, batch_y)
                self.model.backward(loss_grad, learning_rate)

            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / n_batches
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    def predict_classes(self, X, threshold=0.5):
        """
        Predict classes for binary classification.

        Args:
            X: Input data tensor
            threshold: Classification threshold

        Returns:
            Class predictions (0 or 1)
        """
        probs = self.forward(X)

        # Convert probabilities to class predictions
        shape = probs.getShape()
        predictions_data = []

        # Iterate through all elements
        total_size = 1
        for dim in shape:
            total_size *= dim

        for i in range(total_size):
            indices = self._flat_to_indices(i, shape)
            prob = probs[indices]
            predictions_data.append(1.0 if prob > threshold else 0.0)

        return ts.Tensor.from_values(shape, predictions_data)

    def get_num_layers(self):
        """Get number of layers in the network."""
        return self.model.num_layers()

    def get_layer(self, index):
        """Get a specific layer."""
        return self.model.get_layer(index)

    def _tensor_to_float(self, tensor):
        """Convert single-element tensor to float."""
        return tensor[[0]]

    def _flat_to_indices(self, flat_idx, shape):
        """Convert flat index to multi-dimensional indices."""
        indices = []
        for dim in reversed(shape):
            indices.append(flat_idx % dim)
            flat_idx //= dim
        return list(reversed(indices))


# Convenience functions for quick network creation
def create_classifier(input_dim, hidden_dims, num_classes=1, loss_function=None):
    """
    Create a classification MLP.

    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions
        num_classes: Number of output classes (1 for binary, >1 for multi-class)
        loss_function: Loss function instance

    Returns:
        MLP instance
    """
    from optimizers.loss_functions import Binary_CELoss

    layer_dims = [input_dim] + hidden_dims + [num_classes]
    activations = ['relu'] * len(hidden_dims) + ['sigmoid' if num_classes == 1 else 'linear']

    if loss_function is None:
        loss_function = Binary_CELoss()

    mlp = MLP()
    mlp.build(layer_dims, activations, loss_function)
    return mlp


def create_regressor(input_dim, hidden_dims, output_dim=1, loss_function=None):
    """
    Create a regression MLP.

    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions
        output_dim: Number of output dimensions
        loss_function: Loss function instance

    Returns:
        MLP instance
    """
    from optimizers.loss_functions import MSELoss

    layer_dims = [input_dim] + hidden_dims + [output_dim]
    activations = ['relu'] * len(hidden_dims) + ['linear']

    if loss_function is None:
        loss_function = MSELoss()

    mlp = MLP()
    mlp.build(layer_dims, activations, loss_function)
    return mlp

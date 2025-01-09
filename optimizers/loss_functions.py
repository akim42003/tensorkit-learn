import tensor_slow as ts

class MSELoss:
    """Mean Squared Error Loss"""

    def __init__(self):
        pass

    def forward(self, predictions, targets):
        if predictions.getShape() != targets.getShape():
            raise ValueError("Predictions and targets must have the same shape")

        error = predictions.tminus(targets)
        squared_error_sum = 0.0
        num_elements = predictions.getShape()[0]
        for i in range(num_elements):
            squared_error_sum += error[[i]] ** 2

        mse = squared_error_sum / num_elements
        return mse

    def backward(self, predictions, targets):
        if predictions.getShape() != targets.getShape():
            raise ValueError("Predictions and targets must have the same shape")

        error = predictions.tminus(targets)
        num_elements = predictions.getShape()[0]

        # Compute gradient for each element
        grad = ts.Tensor([num_elements])  # Shape of the gradient matches predictions
        for i in range(num_elements):
            grad[[i]] = 2 * error[[i]] / num_elements

        return grad

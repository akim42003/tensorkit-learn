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

        # Iterate through all elements in the tensor
        shape = predictions.getShape()
        num_elements = 1
        for dim in shape:
            num_elements *= dim  # Compute total number of elements

        if len(shape) == 1:  # 1D tensor
            for i in range(shape[0]):
                squared_error_sum += error[[i]] ** 2
        elif len(shape) == 2:  # 2D tensor
            for i in range(shape[0]):
                for j in range(shape[1]):
                    squared_error_sum += error[[i, j]] ** 2

        mse = squared_error_sum / num_elements
        return mse

    def backward(self, predictions, targets):
        if predictions.getShape() != targets.getShape():
            raise ValueError("Predictions and targets must have the same shape")

        error = predictions.tminus(targets)
        shape = predictions.getShape()
        num_elements = 1
        for dim in shape:
            num_elements *= dim  # Compute total number of elements

        grad = ts.Tensor(shape)  # Gradient tensor with the same shape as predictions

        if len(shape) == 1:  # 1D tensor
            for i in range(shape[0]):
                grad[[i]] = 2 * error[[i]] / num_elements
        elif len(shape) == 2:  # 2D tensor
            for i in range(shape[0]):
                for j in range(shape[1]):
                    grad[[i, j]] = 2 * error[[i, j]] / num_elements

        return grad

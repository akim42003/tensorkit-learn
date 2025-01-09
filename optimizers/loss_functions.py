import tensor_slow as ts
import math


class Binary_CELoss:
    """Binary Cross-Entropy Loss"""

    def __init__(self) -> None:
        pass

    def forward(self, predictions, targets):
        """
        Compute the forward pass for binary cross-entropy loss.

        Args:
            predictions (Tensor): Predicted probabilities (0 <= predictions <= 1).
            targets (Tensor): Ground truth binary labels (0 or 1).

        Returns:
            float: Scalar value representing the BCE loss.
        """
        if predictions.getShape() != targets.getShape():
            raise ValueError("Predictions and targets must have the same shape")
        
        shape = predictions.getShape()
        num_elements = 1
        for dim in shape:
            num_elements *= dim  # Compute total number of elements

        bce_loss_sum = 0.0
        if len(shape) == 1:  # 1D tensor
            for i in range(shape[0]):
                pred = predictions[[i]]
                target = targets[[i]]
                bce_loss_sum += -(
                    target * math.log(max(pred, 1e-10)) +
                    (1 - target) * math.log(max(1 - pred, 1e-10))
                )
        elif len(shape) == 2:  # 2D tensor
            for i in range(shape[0]):
                for j in range(shape[1]):
                    pred = predictions[[i, j]]
                    target = targets[[i, j]]
                    bce_loss_sum += -(
                        target * math.log(max(pred, 1e-10)) +
                        (1 - target) * math.log(max(1 - pred, 1e-10))
                    )
        
        return bce_loss_sum / num_elements

    def backward(self, predictions, targets):
        """
        Compute the gradient of the binary cross-entropy loss with respect to predictions.

        Args:
            predictions (Tensor): Predicted probabilities (0 <= predictions <= 1).
            targets (Tensor): Ground truth binary labels (0 or 1).

        Returns:
            Tensor: Gradient Tensor with the same shape as predictions.
        """
        if predictions.getShape() != targets.getShape():
            raise ValueError("Predictions and targets must have the same shape")
        
        shape = predictions.getShape()
        num_elements = 1
        for dim in shape:
            num_elements *= dim  # Compute total number of elements

        grad = ts.Tensor(shape)  # Gradient tensor with the same shape as predictions

        if len(shape) == 1:  # 1D tensor
            for i in range(shape[0]):
                pred = predictions[[i]]
                target = targets[[i]]
                grad[[i]] = -(target / max(pred, 1e-10) - (1 - target) / max(1 - pred, 1e-10)) / num_elements
        elif len(shape) == 2:  # 2D tensor
            for i in range(shape[0]):
                for j in range(shape[1]):
                    pred = predictions[[i, j]]
                    target = targets[[i, j]]
                    grad[[i, j]] = -(target / max(pred, 1e-10) - (1 - target) / max(1 - pred, 1e-10)) / num_elements

        return grad

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

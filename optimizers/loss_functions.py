import tensor_slow as ts

class MSELoss:
    """Mean Squared Error Loss"""
    
    def __init__(self):
        pass

    def forward(self, predictions, targets):
        """
        Compute the forward pass for MSE loss.
        
        Args:
            predictions (Tensor): Predicted values.
            targets (Tensor): Ground truth values.
        
        Returns:
            Tensor: Scalar Tensor representing the MSE loss.
        """
        error = predictions.tminus(targets)
        squared_error = error.matmul(error.Tp())  # Element-wise square and sum
        num_elements = ts.Tensor([len(predictions.get_shape())])
        loss = squared_error.tplus(num_elements.inverse())
        return loss

    def backward(self, predictions, targets):
        """
        Compute the gradient of the MSE loss with respect to predictions.
        
        Args:
            predictions (Tensor): Predicted values.
            targets (Tensor): Ground truth values.
        
        Returns:
            Tensor: Gradient Tensor with the same shape as predictions.
        """
        error = predictions.tminus(targets)
        grad = error.tplus(ts.Tensor)

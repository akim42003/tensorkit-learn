import tensor_slow as ts
from loss_functions import MSELoss

def test_mse_loss_forward():
    """Test the forward pass of MSELoss."""
    predictions = ts.Tensor([1.0, 2.0, 3.0])  # Example predictions
    targets = ts.Tensor([1.5, 2.5, 3.5])      # Example ground truth

    mse_loss = MSELoss()
    loss = mse_loss.forward(predictions, targets)
    
    # Expected loss: mean((1.0-1.5)^2 + (2.0-2.5)^2 + (3.0-3.5)^2) = 0.25
    expected_loss = 0.25
    assert abs(loss - expected_loss) < 1e-6, f"Expected {expected_loss}, got {loss}"

def test_mse_loss_backward():
    """Test the backward pass of MSELoss."""
    predictions = ts.Tensor([1.0, 2.0, 3.0])  # Example predictions
    targets = ts.Tensor([1.5, 2.5, 3.5])      # Example ground truth

    mse_loss = MSELoss()
    grad = mse_loss.backward(predictions, targets)
    
    # Expected gradient: 2 * (predictions - targets) / N
    expected_grad = ts.Tensor([-0.3333, -0.3333, -0.3333])  # Manually computed
    for g, e in zip(grad.data, expected_grad.data):
        assert abs(g - e) < 1e-6, f"Expected gradient {e}, got {g}"

def run_tests():
    test_mse_loss_forward()
    test_mse_loss_backward()
    print("All tests passed!")

if __name__ == "__main__":
    run_tests()

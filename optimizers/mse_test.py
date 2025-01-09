import tensor_slow as ts
from loss_functions import MSELoss

def test_mse_forward():
    print("\n=== MSE Forward Test ===")
    predictions = ts.Tensor([3])  # Shape: (3,)
    targets = ts.Tensor([3])      # Shape: (3,)

    # Assigning values
    predictions[[0]] = 1.0
    predictions[[1]] = 2.0
    predictions[[2]] = 3.0

    targets[[0]] = 1.5
    targets[[1]] = 2.5
    targets[[2]] = 3.5

    mse_loss = MSELoss()
    loss = mse_loss.forward(predictions, targets)

    print(f"Predictions: {[predictions[[i]] for i in range(3)]}")
    print(f"Targets: {[targets[[i]] for i in range(3)]}")
    print(f"Computed Loss: {loss}")

    expected_loss = 0.25  # Precomputed value
    assert abs(loss - expected_loss) < 1e-6, f"Expected {expected_loss}, got {loss}"


def test_mse_backward():
    print("\n=== MSE Backward Test ===")
    predictions = ts.Tensor([3])  # Shape: (3,)
    targets = ts.Tensor([3])      # Shape: (3,)

    # Assigning values
    predictions[[0]] = 1.0
    predictions[[1]] = 2.0
    predictions[[2]] = 3.0

    targets[[0]] = 1.5
    targets[[1]] = 2.5
    targets[[2]] = 3.5

    mse_loss = MSELoss()
    grad = mse_loss.backward(predictions, targets)

    print(f"Predictions: {[predictions[[i]] for i in range(3)]}")
    print(f"Targets: {[targets[[i]] for i in range(3)]}")
    print(f"Gradient: {[grad[[i]] for i in range(3)]}")

    expected_grad = [-0.3333, -0.3333, -0.3333]  # Precomputed values
    for i in range(3):
        assert abs(grad[[i]] - expected_grad[i]) < 1e-4, f"Expected {expected_grad[i]}, got {grad[[i]]}"


def run_mse_tests():
    test_mse_forward()
    test_mse_backward()
    print("\nAll MSE tests passed!")


if __name__ == "__main__":
    run_mse_tests()

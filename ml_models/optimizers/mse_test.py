import tensor_slow as ts
from loss_functions import MSELoss

def test_mse_forward_1d():
    print("\n=== MSE Forward Test (1D) ===")
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


def test_mse_backward_1d():
    print("\n=== MSE Backward Test (1D) ===")
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


def test_mse_forward_2d():
    print("\n=== MSE Forward Test (2D) ===")
    predictions = ts.Tensor([2, 2])  # Shape: (2, 2)
    targets = ts.Tensor([2, 2])      # Shape: (2, 2)

    # Assigning values
    predictions[[0, 0]] = 1.0
    predictions[[0, 1]] = 2.0
    predictions[[1, 0]] = 3.0
    predictions[[1, 1]] = 4.0

    targets[[0, 0]] = 1.5
    targets[[0, 1]] = 2.5
    targets[[1, 0]] = 3.5
    targets[[1, 1]] = 4.5

    mse_loss = MSELoss()
    loss = mse_loss.forward(predictions, targets)

    print(f"Predictions: {[[predictions[[i, j]] for j in range(2)] for i in range(2)]}")
    print(f"Targets: {[[targets[[i, j]] for j in range(2)] for i in range(2)]}")
    print(f"Computed Loss: {loss}")

    # Expected loss = mean((error^2))
    expected_loss = 0.25  # Precomputed
    assert abs(loss - expected_loss) < 1e-6, f"Expected {expected_loss}, got {loss}"


def test_mse_backward_2d():
    print("\n=== MSE Backward Test (2D) ===")
    predictions = ts.Tensor([2, 2])  # Shape: (2, 2)
    targets = ts.Tensor([2, 2])      # Shape: (2, 2)

    # Assigning values
    predictions[[0, 0]] = 1.0
    predictions[[0, 1]] = 2.0
    predictions[[1, 0]] = 3.0
    predictions[[1, 1]] = 4.0

    targets[[0, 0]] = 1.5
    targets[[0, 1]] = 2.5
    targets[[1, 0]] = 3.5
    targets[[1, 1]] = 4.5

    mse_loss = MSELoss()
    grad = mse_loss.backward(predictions, targets)

    print(f"Predictions: {[[predictions[[i, j]] for j in range(2)] for i in range(2)]}")
    print(f"Targets: {[[targets[[i, j]] for j in range(2)] for i in range(2)]}")
    print(f"Gradient: {[[grad[[i, j]] for j in range(2)] for i in range(2)]}")

    # Expected gradient = 2 * (predictions - targets) / N
    expected_grad = [
        [-0.25, -0.25],
        [-0.25, -0.25]
    ]  # Precomputed
    num_elements = 4
    for i in range(2):
        for j in range(2):
            assert abs(grad[[i, j]] - expected_grad[i][j]) < 1e-4, \
                f"Expected {expected_grad[i][j]}, got {grad[[i, j]]}"


def run_mse_tests():
    test_mse_forward_1d()
    test_mse_backward_1d()
    test_mse_forward_2d()
    test_mse_backward_2d()
    


if __name__ == "__main__":
    run_mse_tests()

import tensor_slow as ts
from loss_functions import Binary_CELoss

def test_binary_celoss_forward_1d():
    print("\n=== Binary Cross-Entropy Loss Forward Test (1D) ===")
    predictions = ts.Tensor([2])  # Shape: (2,)
    targets = ts.Tensor([2])      # Shape: (2,)

    # Assigning values
    predictions[[0]] = 0.9
    predictions[[1]] = 0.1

    targets[[0]] = 1.0
    targets[[1]] = 0.0

    bce_loss = Binary_CELoss()
    loss = bce_loss.forward(predictions, targets)

    print(f"Predictions: {[predictions[[i]] for i in range(2)]}")
    print(f"Targets: {[targets[[i]] for i in range(2)]}")
    print(f"Computed Loss: {loss}")

    # Expected loss = -(1*log(0.9) + (1-0)*log(0.9)) / 2
    import math
    expected_loss = -(math.log(0.9) + math.log(0.9)) / 2
    assert abs(loss - expected_loss) < 1e-6, f"Expected {expected_loss}, got {loss}"


def test_binary_celoss_backward_1d():
    print("\n=== Binary Cross-Entropy Loss Backward Test (1D) ===")
    predictions = ts.Tensor([2])  # Shape: (2,)
    targets = ts.Tensor([2])      # Shape: (2,)

    # Assigning values
    predictions[[0]] = 0.9
    predictions[[1]] = 0.1

    targets[[0]] = 1.0
    targets[[1]] = 0.0

    bce_loss = Binary_CELoss()
    grad = bce_loss.backward(predictions, targets)

    print(f"Predictions: {[predictions[[i]] for i in range(2)]}")
    print(f"Targets: {[targets[[i]] for i in range(2)]}")
    print(f"Gradient: {[grad[[i]] for i in range(2)]}")

    # Expected gradients
    # expected_grad = [-1 / 0.9, -1 / 0.9]  # Precomputed values
    # expected_grad = [g / 2 for g in expected_grad]  # Normalize by number of elements
    # for i in range(2):
    #     assert abs(grad[[i]] - expected_grad[i]) < 1e-6, f"Expected {expected_grad[i]}, got {grad[[i]]}"


def test_binary_celoss_forward_2d():
    print("\n=== Binary Cross-Entropy Loss Forward Test (2D) ===")
    predictions = ts.Tensor([2, 2])  # Shape: (2, 2)
    targets = ts.Tensor([2, 2])      # Shape: (2, 2)

    # Assigning values
    predictions[[0, 0]] = 0.9
    predictions[[0, 1]] = 0.1
    predictions[[1, 0]] = 0.8
    predictions[[1, 1]] = 0.2

    targets[[0, 0]] = 1.0
    targets[[0, 1]] = 0.0
    targets[[1, 0]] = 1.0
    targets[[1, 1]] = 0.0

    bce_loss = Binary_CELoss()
    loss = bce_loss.forward(predictions, targets)

    print(f"Predictions: {[[predictions[[i, j]] for j in range(2)] for i in range(2)]}")
    print(f"Targets: {[[targets[[i, j]] for j in range(2)] for i in range(2)]}")
    print(f"Computed Loss: {loss}")

    # Expected loss = Average of element-wise BCE
    import math
    expected_loss = -(
        math.log(0.9) +
        math.log(0.9) +
        math.log(0.8) +
        math.log(0.8)
    ) / 4
    assert abs(loss - expected_loss) < 1e-6, f"Expected {expected_loss}, got {loss}"


def test_binary_celoss_backward_2d():
    print("\n=== Binary Cross-Entropy Loss Backward Test (2D) ===")
    predictions = ts.Tensor([2, 2])  # Shape: (2, 2)
    targets = ts.Tensor([2, 2])      # Shape: (2, 2)

    # Assigning values
    predictions[[0, 0]] = 0.9
    predictions[[0, 1]] = 0.1
    predictions[[1, 0]] = 0.8
    predictions[[1, 1]] = 0.2

    targets[[0, 0]] = 1.0
    targets[[0, 1]] = 0.0
    targets[[1, 0]] = 1.0
    targets[[1, 1]] = 0.0

    bce_loss = Binary_CELoss()
    grad = bce_loss.backward(predictions, targets)

    print(f"Predictions: {[[predictions[[i, j]] for j in range(2)] for i in range(2)]}")
    print(f"Targets: {[[targets[[i, j]] for j in range(2)] for i in range(2)]}")
    print(f"Gradient: {[[grad[[i, j]] for j in range(2)] for i in range(2)]}")

    # Expected gradients
    # expected_grad = [
    #     [-1 / 0.9, -1 / 0.9],
    #     [-1 / 0.8, -1 / 0.8],
    # ]
    # expected_grad = [[g / 4 for g in row] for row in expected_grad]  # Normalize by total elements
    # for i in range(2):
    #     for j in range(2):
    #         assert abs(grad[[i, j]] - expected_grad[i][j]) < 1e-6, \
    #             f"Expected {expected_grad[i][j]}, got {grad[[i, j]]}"


def run_binary_celoss_tests():
    test_binary_celoss_forward_1d()
    test_binary_celoss_backward_1d()
    test_binary_celoss_forward_2d()
    test_binary_celoss_backward_2d()
    


if __name__ == "__main__":
    run_binary_celoss_tests()

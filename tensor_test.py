import tensor_slow

def test_tensor_creation():
    t = tensor_slow.Tensor([2, 3])
    t[[0, 0]] = 1.0
    t[[0, 1]] = 2.0
    t[[1, 0]] = 3.0
    t[[1, 1]] = 4.0

    print("Tensor creation and value assignment test:")
    t.print()


def test_tensor_addition():
    t1 = tensor_slow.Tensor([2, 2])
    t2 = tensor_slow.Tensor([2, 2])

    t1[[0, 0]] = 1.0
    t1[[0, 1]] = 2.0
    t1[[1, 0]] = 3.0
    t1[[1, 1]] = 4.0

    t2[[0, 0]] = 5.0
    t2[[0, 1]] = 6.0
    t2[[1, 0]] = 7.0
    t2[[1, 1]] = 8.0

    t3 = t1.tplus(t2)
    print("Tensor addition test:")
    t3.print()


def test_tensor_subtraction():
    t1 = tensor_slow.Tensor([2, 2])
    t2 = tensor_slow.Tensor([2, 2])

    t1[[0, 0]] = 10.0
    t1[[0, 1]] = 20.0
    t1[[1, 0]] = 30.0
    t1[[1, 1]] = 40.0

    t2[[0, 0]] = 1.0
    t2[[0, 1]] = 2.0
    t2[[1, 0]] = 3.0
    t2[[1, 1]] = 4.0

    t3 = t1.tminus(t2)
    print("Tensor subtraction test:")
    t3.print()


def test_tensor_transpose():
    t = tensor_slow.Tensor([2, 3])
    t[[0, 0]] = 1.0
    t[[0, 1]] = 2.0
    t[[0, 2]] = 3.0
    t[[1, 0]] = 4.0
    t[[1, 1]] = 5.0
    t[[1, 2]] = 6.0

    t_transposed = t.Tp()
    print("Tensor transpose test:")
    print("Original tensor:")
    t.print()
    print("Transposed tensor:")
    t_transposed.print()


def test_tensor_matmul():
    t1 = tensor_slow.Tensor([2, 3])
    t2 = tensor_slow.Tensor([3, 2])

    t1[[0, 0]] = 1.0
    t1[[0, 1]] = 2.0
    t1[[0, 2]] = 3.0
    t1[[1, 0]] = 4.0
    t1[[1, 1]] = 5.0
    t1[[1, 2]] = 6.0

    t2[[0, 0]] = 7.0
    t2[[0, 1]] = 8.0
    t2[[1, 0]] = 9.0
    t2[[1, 1]] = 10.0
    t2[[2, 0]] = 11.0
    t2[[2, 1]] = 12.0

    t3 = t1.matmul(t2)
    print("Matrix multiplication test:")
    print("Tensor 1:")
    t1.print()
    print("Tensor 2:")
    t2.print()
    print("Result:")
    t3.print()


def test_tensor_inverse():
    t = tensor_slow.Tensor([2, 2])
    t[[0, 0]] = 4.0
    t[[0, 1]] = 7.0
    t[[1, 0]] = 2.0
    t[[1, 1]] = 6.0

    t_inverse = t.inverse()
    print("Matrix inversion test:")
    print("Original tensor:")
    t.print()
    print("Inverse tensor:")
    t_inverse.print()


def test_tensor_indexing():
    print("Tensor Indexing Test")

    # Create a 2x3 tensor
    t = tensor_slow.Tensor([2, 3])

    # Set values using indexing
    t[[0, 0]] = 1.0
    t[[0, 1]] = 2.0
    t[[0, 2]] = 3.0
    t[[1, 0]] = 4.0
    t[[1, 1]] = 5.0
    t[[1, 2]] = 6.0

    print(str(t[[0,1]]) + "\n")

if __name__ == "__main__":
    test_tensor_creation()
    test_tensor_indexing()
    test_tensor_addition()
    test_tensor_subtraction()
    test_tensor_transpose()
    test_tensor_matmul()
    test_tensor_inverse()

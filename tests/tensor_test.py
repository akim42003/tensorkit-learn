import tensor_slow

def test_tensor_creation():
    t = tensor_slow.Tensor.from_values([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    print("Tensor creation and value assignment test:")
    t.print()


def test_tensor_addition():
    t1 = tensor_slow.Tensor.from_values([2, 2], [1.0, 2.0, 3.0, 4.0])
    t2 = tensor_slow.Tensor.from_values([2, 2], [5.0, 6.0, 7.0, 8.0])

    t3 = t1.tplus(t2)
    print("Tensor addition test:")
    t3.print()


def test_tensor_subtraction():
    t1 = tensor_slow.Tensor.from_values([2, 2], [10.0, 20.0, 30.0, 40.0])
    t2 = tensor_slow.Tensor.from_values([2, 2], [1.0, 2.0, 3.0, 4.0])

    t3 = t1.tminus(t2)
    print("Tensor subtraction test:")
    t3.print()


def test_tensor_transpose():
    t = tensor_slow.Tensor.from_values([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    t_transposed = t.Tp()
    print("Tensor transpose test:")
    print("Original tensor:")
    t.print()
    print("Transposed tensor:")
    t_transposed.print()


def test_tensor_matmul():
    t1 = tensor_slow.Tensor.from_values([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    t2 = tensor_slow.Tensor.from_values([3, 2], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

    t3 = t1.matmul(t2)
    print("Matrix multiplication test:")
    print("Tensor 1:")
    t1.print()
    print("Tensor 2:")
    t2.print()
    print("Result:")
    t3.print()


def test_tensor_inverse():
    t = tensor_slow.Tensor.from_values([2, 2], [4.0, 7.0, 2.0, 6.0])
    t_inverse = t.inverse()
    print("Matrix inversion test:")
    print("Original tensor:")
    t.print()
    print("Inverse tensor:")
    t_inverse.print()


def test_tensor_indexing():
    print("Tensor Indexing Test")
    t = tensor_slow.Tensor.from_values([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    print(f"Value at [0, 1]: {t[[0, 1]]}\n")

def test_tensor_division():
    t1 = tensor_slow.Tensor.from_values([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    t2 = tensor_slow.Tensor.from_values([2, 3], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

    t3 = t1.divide(t2)
    print("Element wise division test")
    t3.print()

def test_tensor_log():
    t1 = tensor_slow.Tensor.from_values([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    t_log = t1.log()
    print("Element wise log test")
    t_log.print()

def test_tensor_exp():
    t1 = tensor_slow.Tensor.from_values([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    t_exp = t1.exp()
    print("Element wise exp test")
    t_exp.print()

if __name__ == "__main__":
    test_tensor_creation()
    test_tensor_indexing()
    test_tensor_addition()
    test_tensor_subtraction()
    test_tensor_transpose()
    test_tensor_matmul()
    test_tensor_inverse()
    test_tensor_division()
    test_tensor_log()
    test_tensor_exp()
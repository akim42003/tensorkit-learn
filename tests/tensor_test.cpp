#include "tensor.h"
#include <iostream>

int main() {
    // Create a tensor filled with zeros
    Tensor t1 = Tensor::zeros({2, 3});
    std::cout << "Tensor filled with zeros:\n";
    t1.print();

    // Create a tensor filled with ones
    Tensor t2 = Tensor::ones({3, 3});
    std::cout << "Tensor filled with ones:\n";
    t2.print();

    // Create a tensor with specific values
    Tensor t3 = Tensor::from_values({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    std::cout << "Tensor from values:\n";
    t3.print();

    return 0;
}

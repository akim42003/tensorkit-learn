#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <stdexcept>

class Tensor {
private:
    std::vector<float> data;
    std::vector<size_t> shape;
    std::vector<size_t> strides;

    void computeStrides();

public:
    // Constructor
    Tensor(const std::vector<size_t>& shape);

    // Element Access
    float& operator()(const std::vector<size_t>& indices);
    const float& operator()(const std::vector<size_t>& indices) const;

    // Get Shape
    const std::vector<size_t>& getShape() const;

    // Print Tensor
    void print() const;

    // Arithmetic Operations
    Tensor tplus(const Tensor& other);
    Tensor tminus(const Tensor& other);

    // Dot Product
    float dot(Tensor& other) const;

    // Matrix Operations
    Tensor matmul(const Tensor& other) const;
    Tensor Tp() const;

    // Inverse
    Tensor inverse() const;

};

#endif // TENSOR_H

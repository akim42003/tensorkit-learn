#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>
#include <cmath>

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
    Tensor tminus(const Tensor& other) const;
    Tensor elementwise_multiply(const Tensor& other) const;
    Tensor scalar_multiply(float scalar) const;

    // Dot Product
    float dot(Tensor& other) const;

    // Matrix Operations
    Tensor matmul(const Tensor& other) const;
    Tensor Tp() const;

    //Element wise operations
    Tensor log() const; //natural log
    Tensor exp() const; 
    Tensor divide(const Tensor& other) const;

    //clamp values

    Tensor clamp(float min_value, float max_value) const;

    // Inverse
    Tensor inverse() const;

    // Static Methods for Tensor Creation
    static Tensor zeros(const std::vector<size_t>& shape);
    static Tensor ones(const std::vector<size_t>& shape);
    static Tensor from_values(const std::vector<size_t>& shape, const std::vector<float>& values);
};

#endif // TENSOR_H

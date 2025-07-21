#include "tensor.h" // Include the header file for the Tensor class
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <limits>

// Constructor
Tensor::Tensor(const std::vector<size_t>& shape) : shape(shape) {
    size_t totalSize = 1;
    for (size_t dim : shape) {
        totalSize *= dim;
    }
    data.resize(totalSize, 0); // Initialize with zeros
    computeStrides();
}

// Compute strides for efficient indexing
void Tensor::computeStrides() {
    strides.resize(shape.size());
    size_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

// Access element with multi-dimensional index
float& Tensor::operator()(const std::vector<size_t>& indices) {
    if (indices.size() != shape.size()) {
        throw std::invalid_argument("Invalid number of indices");
    }
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        offset += indices[i] * strides[i];
    }
    return data[offset];
}

const float& Tensor::operator()(const std::vector<size_t>& indices) const {
    if (indices.size() != shape.size()) {
        throw std::invalid_argument("Invalid number of indices");
    }
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        offset += indices[i] * strides[i];
    }
    return data[offset];
}

// Get the shape of the tensor
const std::vector<size_t>& Tensor::getShape() const {
    return shape;
}

// Print the tensor (for debugging)
void Tensor::print() const {
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << data[i] << " ";
        if ((i + 1) % shape.back() == 0) {
            std::cout << "\n";
        }
    }
    std::cout << "\n";
}

// Overload + for tensor addition
Tensor Tensor::tplus(const Tensor& other) {
    if (shape != other.shape) {
        throw std::invalid_argument("Shapes for addition must match");
    }
    Tensor result(shape);
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

// Overload - for tensor subtraction
Tensor Tensor::tminus(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Shapes for subtraction must match");
    }
    Tensor result(shape);
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

// Element-wise multiplication
Tensor Tensor::elementwise_multiply(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Shapes must match for element-wise multiplication");
    }
    Tensor result(shape);
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

// Scalar multiplication
Tensor Tensor::scalar_multiply(float scalar) const {
    Tensor result(shape);
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = data[i] * scalar;
    }
    return result;
}

// Element wise operations: Logarithm
Tensor Tensor::log() const {
    Tensor result(shape);
    const float epsilon = 1e-6;  // Small constant to prevent log(0) or log(negative)
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] <= 0.0f) {
            result.data[i] = std::log(epsilon);  // Clamp to epsilon
        } else {
            result.data[i] = std::log(data[i]);
        }
    }
    return result;
}

// Element-wise exponential
Tensor Tensor::exp() const {
    Tensor result(shape);
    const float max_exp = 88.0f;  // Clamp to prevent overflow
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] > max_exp) {
            result.data[i] = std::exp(max_exp);
        } else {
            result.data[i] = std::exp(data[i]);
        }
    }
    return result;
}

// Element wise division
Tensor Tensor::divide(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Shapes must match for element-wise division");
    }
    Tensor result(shape);
    const float epsilon = 1e-6;  // Small constant to prevent division by zero
    for (size_t i = 0; i < data.size(); ++i) {
        if (other.data[i] == 0.0f) {
            result.data[i] = data[i] / epsilon;  // Replace zero denominator with epsilon
        } else {
            result.data[i] = data[i] / other.data[i];
        }
    }
    return result;
}

// Clamp values
Tensor Tensor::clamp(float min_value, float max_value) const {
    Tensor result(shape);
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] < min_value) {
            result.data[i] = min_value;
        } else if (data[i] > max_value) {
            result.data[i] = max_value;
        } else {
            result.data[i] = data[i];
        }
    }
    return result;
}

// Dot product for vectors
float Tensor::dot(Tensor& other) const {
    if (shape.size() != 1 || other.shape.size() != 1) {
        throw std::invalid_argument("Dot product is defined for vectors only");
    }
    if (shape[0] != other.shape[0]) {
        throw std::invalid_argument("Vectors must have the same length for dot product");
    }

    float result = 0;
    for (size_t i = 0; i < data.size(); i++) {
        result += data[i] * other.data[i];
    }

    return result;
}

// Matrix multiplication (lowkey deprecated)
Tensor Tensor::matmul(const Tensor& other) const {
    if (shape.size() != 2 || other.shape.size() != 2) {
        throw std::invalid_argument("Matrix multiplication is defined for 2D tensors only");
    }
    if (shape[1] != other.shape[0]) {
        throw std::invalid_argument("Incompatible dimensions for matrix multiplication");
    }

    Tensor result({shape[0], other.shape[1]});

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < other.shape[1]; ++j) {
            float sum = 0.0;
            for (size_t k = 0; k < shape[1]; ++k) {
                sum += data[i * strides[0] + k * strides[1]] * other.data[k * other.strides[0] + j * other.strides[1]];
            }
            result.data[i * result.strides[0] + j * result.strides[1]] = sum;
        }
    }

    return result;
}

// 2D Matrix Transposition
Tensor Tensor::Tp() const {
    if (shape.size() != 2) {
        throw std::invalid_argument("Transpose is defined only for 2D matrices");
    }

    Tensor result({shape[1], shape[0]});
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            result.data[j * result.strides[0] + i * result.strides[1]] =
                data[i * strides[0] + j * strides[1]];
        }
    }

    return result;
}

// Matrix Inversion
Tensor Tensor::inverse() const {
    if (shape.size() != 2 || shape[0] != shape[1]) {
        throw std::invalid_argument("Matrices are nonsingular if and only if they are square");
    }
    size_t n = shape[0];
    Tensor augmented({n, 2 * n});

    // Create the augmented matrix [A | I]
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            augmented.data[i * augmented.strides[0] + j * augmented.strides[1]] =
                data[i * strides[0] + j * strides[1]];
        }
        augmented.data[i * augmented.strides[0] + (n + i) * augmented.strides[1]] = 1.0;
    }

    // Gaussian elimination
    const float tolerance = 1e-6;  // Small tolerance for near-zero values
    for (size_t i = 0; i < n; ++i) {
        size_t pivot_index = i * augmented.strides[0] + i * augmented.strides[1];
        if (std::abs(augmented.data[pivot_index]) < tolerance) {
            throw std::runtime_error("Matrix is singular or near-singular and cannot be inverted");
        }

        float pivot = augmented.data[pivot_index];
        for (size_t j = 0; j < 2 * n; ++j) {
            augmented.data[i * augmented.strides[0] + j * augmented.strides[1]] /= pivot;
        }

        for (size_t k = 0; k < n; ++k) {
            if (k == i) continue;
            float factor = augmented.data[k * augmented.strides[0] + i * augmented.strides[1]];
            for (size_t j = 0; j < 2 * n; ++j) {
                augmented.data[k * augmented.strides[0] + j * augmented.strides[1]] -=
                    factor * augmented.data[i * augmented.strides[0] + j * augmented.strides[1]];
            }
        }
    }

    Tensor inverse({n, n});
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            inverse.data[i * inverse.strides[0] + j * inverse.strides[1]] =
                augmented.data[i * augmented.strides[0] + (n + j) * augmented.strides[1]];
        }
    }

    return inverse;
}

// Static method to create a tensor filled with zeros
Tensor Tensor::zeros(const std::vector<size_t>& shape) {
    return Tensor(shape);
}

// Static method to create a tensor filled with ones
Tensor Tensor::ones(const std::vector<size_t>& shape) {
    Tensor t(shape);
    for (auto& val : t.data) {
        val = 1.0f;
    }
    return t;
}

// Static method to create a tensor from a list of values
Tensor Tensor::from_values(const std::vector<size_t>& shape, const std::vector<float>& values) {
    size_t total_size = 1;
    for (size_t dim : shape) {
        total_size *= dim;
    }
    if (values.size() != total_size) {
        throw std::invalid_argument("Values size must match the total size of the tensor");
    }

    Tensor t(shape);
    t.data = values;
    return t;
}

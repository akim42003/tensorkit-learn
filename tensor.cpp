#include <iostream>
#include <vector>
#include <stdexcept> 

using namespace std;
class Tensor {

    private: 
        vector<float>data;
        vector<size_t>shape;
        vector<size_t>strides;
    // Compute strides for efficient indexing
        void computeStrides() {
            strides.resize(shape.size());
            size_t stride = 1;
            for (int i = shape.size() - 1; i >= 0; --i) {
                strides[i] = stride;
                stride *= shape[i];
            }
        }
    public: 
        Tensor(const vector<size_t>& shape) : shape(shape) {
            size_t totalSize = 1;
            for (size_t dim : shape) {
                totalSize *= dim;
            }
            data.resize(totalSize, 0); // Initialize with zeros
            computeStrides();
    }
        // Access element with multi-dimensional index
        float& operator()(const vector<size_t>& indices) {
            if (indices.size() != shape.size()) {
                throw invalid_argument("Invalid number of indices");
            }
            size_t offset = 0;
            for (size_t i = 0; i < indices.size(); ++i) {
                offset += indices[i] * strides[i];
            }
            return data[offset];
        }
        // Get the shape of the tensor
        const vector<size_t>& getShape() const {
            return shape;
        }

        // Print the tensor (for debugging)
        void print() const {
            for (size_t i = 0; i < data.size(); ++i) {
                cout << data[i] << " ";
                if ((i + 1) % shape.back() == 0) {
                    cout << "\n";
                }
            }
            cout << "\n";
        }
        // overload + for tensor addition
        Tensor operator+(const Tensor& other) {
            if (shape != other.shape){
                throw invalid_argument("Shapes for addition must match");
            }
            Tensor result(shape);
            for (size_t i = 0; i < data.size(); i++){
                result.data[i] = data[i] + other.data[i];
            }
            return result;
        }
        //overload - for tensor subtraction
        Tensor operator-(const Tensor& other) {
            if (shape != other.shape){
                throw invalid_argument("Shapes for subtraction must match");
            }
            Tensor result(shape);
            for (size_t i = 0; i <data.size(); i++){
                result.data[i] = data[i] - other.data[i];
            }
            return result;
        }
        //dot product for vectors 
        float dot(Tensor& other) const{
            if (shape.size() != 1 || other.shape.size() != 1) {
                throw std::invalid_argument("Dot product is defined for vectors only");
            }
            if (shape[0] != other.shape[0]) {
                throw std::invalid_argument("Vectors must have the same length for dot product");
            }

            float result = 0;

            for (size_t i = 0; i < data.size(); i++){
                result += data[i]* other.data[i];
            }

            return result;
        }

        // matrix multiplication 
        Tensor matmul(const Tensor& other) const {
            if (shape.size() != 2 || other.shape.size() != 2) {
                throw std::invalid_argument("Matrix multiplication is defined for 2D tensors only");
            }
            if (shape[1] != other.shape[0]) {
                throw std::invalid_argument("Incompatible dimensions for matrix multiplication");
            }

            Tensor result({shape[0], other.shape[1]});

            // Perform the multiplication
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < other.shape[1]; ++j) {
                    float sum = 0.0;
                    for (size_t k = 0; k < shape[1]; ++k) {
                        // Compute flattened indices manually
                        size_t thisIndex = i * strides[0] + k * strides[1];
                        size_t otherIndex = k * other.strides[0] + j * other.strides[1];

                        sum += data[thisIndex] * other.data[otherIndex];
                    }
                    // Set result using its strides
                    size_t resultIndex = i * result.strides[0] + j * result.strides[1];
                    result.data[resultIndex] = sum;
                }
            }

            return result;
        }
        //2D Matrix Transposition (will add higher dimensions at a later date)
        Tensor Tp() const {
            if (shape.size() != 2){
                throw invalid_argument ("Matrices must be 2D");
            }

            Tensor result({shape[1], shape[0]});

            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    // Access elements directly using data and strides
                    size_t originalIndex = i * strides[0] + j * strides[1];
                    size_t transposedIndex = j * result.strides[0] + i * result.strides[1];
                    result.data[transposedIndex] = data[originalIndex];
                }
            }

            return result;
        }

        Tensor inverse() const {
            if (shape.size() != 2 || shape[0] != shape[1]){
                throw invalid_argument("Matrices are nonsingular if and only if they are square");
            }
            size_t n = shape[0];
            Tensor augmented({n, 2 * n});

            // Create the augmented matrix [A | I]
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    // Access `data` directly
                    augmented.data[i * augmented.strides[0] + j * augmented.strides[1]] = 
                        data[i * strides[0] + j * strides[1]];
                }
                augmented.data[i * augmented.strides[0] + (n + i) * augmented.strides[1]] = 1.0; // Identity matrix
            }

            // Perform Gaussian elimination
            for (size_t i = 0; i < n; ++i) {
                // Ensure the pivot element is non-zero
                size_t pivot_index = i * augmented.strides[0] + i * augmented.strides[1];
                if (augmented.data[pivot_index] == 0.0) {
                    // Swap with a row below that has a non-zero pivot
                    bool swapped = false;
                    for (size_t k = i + 1; k < n; ++k) {
                        size_t swap_index = k * augmented.strides[0] + i * augmented.strides[1];
                        if (augmented.data[swap_index] != 0.0) {
                            for (size_t j = 0; j < 2 * n; ++j) {
                                std::swap(
                                    augmented.data[i * augmented.strides[0] + j * augmented.strides[1]],
                                    augmented.data[k * augmented.strides[0] + j * augmented.strides[1]]
                                );
                            }
                            swapped = true;
                            break;
                        }
                    }
                    if (!swapped) {
                        throw std::runtime_error("Matrix is singular and cannot be inverted");
                    }
                }

                // Normalize the pivot row
                float pivot = augmented.data[pivot_index];
                for (size_t j = 0; j < 2 * n; ++j) {
                    augmented.data[i * augmented.strides[0] + j * augmented.strides[1]] /= pivot;
                }

                // Eliminate other rows
                for (size_t k = 0; k < n; ++k) {
                    if (k == i) continue;
                    float factor = augmented.data[k * augmented.strides[0] + i * augmented.strides[1]];
                    for (size_t j = 0; j < 2 * n; ++j) {
                        augmented.data[k * augmented.strides[0] + j * augmented.strides[1]] -=
                            factor * augmented.data[i * augmented.strides[0] + j * augmented.strides[1]];
                    }
                }
            }

            // Extract the inverse matrix [I | A^-1]
            Tensor inverse({n, n});
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    inverse.data[i * inverse.strides[0] + j * inverse.strides[1]] =
                        augmented.data[i * augmented.strides[0] + (n + j) * augmented.strides[1]];
                }
            }

            return inverse;
        }


};

int main() {
    Tensor tensor1({2, 2});  // Create a 2x3 tensor

    tensor1({0, 0}) = 1.0;  
    tensor1({0, 1}) = 2.0;
    // tensor1({0, 2}) = 3.0;
    tensor1({1, 0}) = 4.0;
    tensor1({1, 1}) = 5.0;
    // tensor1({1, 2}) = 6.0;

    Tensor tensor2({3,2});

    tensor2({0,0}) = 2;
    tensor2({0,1}) = 1;
    tensor2({1,0}) = 0;
    tensor2({1,1}) = 0;
    tensor2({2,0}) = 1;
    tensor2({2,1}) = 2;


    // Tensor eg_tensors = tensor1 - tensor2;

    // eg_tensors.print();
//     cout << "Tensor contents:\n";
//     tensor.print();  // Print the tensor

//     cout << "Element at (1, 2): " << tensor({1,2}) << "\n";

    // Tensor tensor1({3});
    
    // tensor1({0}) = 1;
    // tensor1({1}) = 2;
    // tensor1({2}) = 0;

    // Tensor tensor2({3});

    // tensor2({0}) = 2;
    // tensor2({1}) = 2;
    // tensor2({2}) = 10;

    // float dot_result = tensor1.dot(tensor2);

    // cout << dot_result << endl;

    // Tensor multiplied_tensor = tensor1.matmul(tensor2);
    // tensor1.print();
    // tensor2.print();
    // multiplied_tensor.print();

    // Tensor transposed_matrix = tensor1.Tp();

    // transposed_matrix.print();

    // Compute the inverse
    Tensor inverseMatrix = tensor1.inverse();

    // Print the inverse matrix
    cout << "Inverse Matrix:\n";
    inverseMatrix.print();

    return 0;


}

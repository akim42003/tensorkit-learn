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
};

int main() {
    Tensor tensor({2, 3});  // Create a 2x3 tensor

    tensor({0, 0}) = 1.0;  // Set values
    tensor({0, 1}) = 2.0;
    tensor({0, 2}) = 3.0;
    tensor({1, 0}) = 4.0;
    tensor({1, 1}) = 5.0;
    tensor({1, 2}) = 6.0;

    cout << "Tensor contents:\n";
    tensor.print();  // Print the tensor

    cout << "Element at (1, 2): " << tensor({1,2}) << "\n";

    return 0;
}

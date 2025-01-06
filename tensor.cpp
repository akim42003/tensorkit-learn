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

};

int main() {
    // Tensor tensor1({2, 3});  // Create a 2x3 tensor

    // tensor1({0, 0}) = 1.0;  // Set values
    // tensor1({0, 1}) = 2.0;
    // tensor1({0, 2}) = 3.0;
    // tensor1({1, 0}) = 4.0;
    // tensor1({1, 1}) = 5.0;
    // tensor1({1, 2}) = 6.0;

    // Tensor tensor2({2,3});

    // tensor2({0,0}) = 2;
    // tensor2({0,1}) = 1;
    // tensor2({0,2}) = 0;
    // tensor2({1,0}) = 0;
    // tensor2({1,1}) = 1;
    // tensor2({1,2}) = 2;


    // Tensor eg_tensors = tensor1 - tensor2;

    // eg_tensors.print();
//     cout << "Tensor contents:\n";
//     tensor.print();  // Print the tensor

//     cout << "Element at (1, 2): " << tensor({1,2}) << "\n";

    Tensor tensor1({3});
    
    tensor1({0}) = 1;
    tensor1({1}) = 2;
    tensor1({2}) = 0;

    Tensor tensor2({3});

    tensor2({0}) = 2;
    tensor2({1}) = 2;
    tensor2({2}) = 10;

    float dot_result = tensor1.dot(tensor2);

    cout << dot_result << endl;

    return 0;


}

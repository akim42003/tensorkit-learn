#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h> // Required for py::self
#include "../tensor_cpp/tensor.h" // Include your Tensor class header

namespace py = pybind11;

PYBIND11_MODULE(tensor_slow, m) {
    py::class_<Tensor>(m, "Tensor")
        // Constructor
        .def(py::init<const std::vector<size_t>&>(), "Create a tensor with the given shape")

        // Get shape of the tensor
        .def("getShape", &Tensor::getShape, "Get the shape of the tensor")

        // Access elements
        .def("__getitem__", [](const Tensor& t, const std::vector<size_t>& indices) {
            return t(indices); // Indexing (read access)
        })
        .def("__setitem__", [](Tensor& t, const std::vector<size_t>& indices, float value) {
            t(indices) = value; // Indexing (write access)
        })

        // Print the tensor
        .def("print", &Tensor::print, "Print the tensor data")

        // Tensor operations
        .def("matmul", &Tensor::matmul, "Perform matrix multiplication")
        .def("Tp", &Tensor::Tp, "Transpose the tensor (2D only)")
        .def("inverse", &Tensor::inverse, "Calculate the inverse of the tensor (2D square matrices only)")

        // // Tensor arithmetic
        // .def("__add__", [](const Tensor& a, const Tensor& b) {
        //     return a.tplus(b); // Tensor addition
        // })
        // .def("__sub__", [](const Tensor& a, const Tensor& b) {
        //     return a.tminus(b); // Tensor subtraction
        // })

        // Add other overloads as needed
        .def("tplus", &Tensor::tplus, "Explicit tensor addition")
        .def("tminus", &Tensor::tminus, "Explicit tensor subtraction")
        .def("elementwise_multiply", &Tensor::elementwise_multiply, "Element-wise multiplication")
        .def("scalar_multiply", &Tensor::scalar_multiply, "Scalar multiplication")

        // Element wise operations

        .def("log", &Tensor::log, "logarithm on tensor")
        .def("exp", &Tensor::exp, "exponent on tensor")
        .def("divide", &Tensor::divide, "division by element")
        //clamp values
        .def("clamp", &Tensor::clamp, "clamp values")

        //initializations (zeros, ones, from values)
        .def_static("zeros", &Tensor::zeros, "Initialize tensor with zeros")

        .def_static("from_values", &Tensor::from_values, "Initialize tensor via matrix construction");
}

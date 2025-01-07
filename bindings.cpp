#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor.h" // Include your Tensor class header

namespace py = pybind11;

PYBIND11_MODULE(tensor_slow, m) {
    py::class_<Tensor>(m, "Tensor")
        // Constructor
        .def(py::init<const std::vector<size_t>&>(), "Create a tensor with the given shape")

        // Get shape of the tensor
        .def("getShape", &Tensor::getShape, "Get the shape of the tensor")

        // Access elements
        .def("__getitem__", [](const Tensor& t, const std::vector<size_t>& indices) {
            return t(indices); // Indexing
        })
        .def("__setitem__", [](Tensor& t, const std::vector<size_t>& indices, float value) {
            t(indices) = value; // Setting value
        })

        // Print the tensor
        .def("print", &Tensor::print, "Print the tensor data")

        // Tensor operations
        .def("matmul", &Tensor::matmul, "Perform matrix multiplication")
        .def("Tp", &Tensor::Tp, "Transpose the tensor (2D only)")
        .def("inverse", &Tensor::inverse, "Calculate the inverse of the tensor (2D square matrices only)");
}

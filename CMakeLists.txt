cmake_minimum_required(VERSION 3.14)
project(TensorPythonBindings)

set(CMAKE_CXX_STANDARD 14)

# Add pybind11
find_package(pybind11 REQUIRED)

# Create the Python module
pybind11_add_module(tensor_slow bindings.cpp tensor.cpp)

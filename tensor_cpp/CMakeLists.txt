cmake_minimum_required(VERSION 3.14)
project(TensorPythonBindings)

# Use C++14 standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify where to find pybind11
set(pybind11_DIR "/Users/alex/tensorkit-learn/venv/lib/python3.11/site-packages/pybind11/share/cmake/pybind11")

# Add pybind11 package
find_package(pybind11 REQUIRED)

# Add the Python module
pybind11_add_module(tensor_slow bindings.cpp tensor.cpp)

# Optional: Add additional include directories if necessary
# include_directories(${PROJECT_SOURCE_DIR}/include)

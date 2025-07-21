cmake_minimum_required(VERSION 3.14)
project(TensorAndLinkFunctions)

# Use C++11 standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify where to find pybind11
set(pybind11_DIR "/Users/alex/tensorkit-learn/venv/lib/python3.11/site-packages/pybind11/share/cmake/pybind11")

# Add pybind11 package
find_package(pybind11 REQUIRED)

# Optional: Add include directories (if needed)
include_directories(${PROJECT_SOURCE_DIR})

# # Tensor Module
# pybind11_add_module(tensor_slow lf_bindings.cpp tensor.cpp)

# Link Functions Module
pybind11_add_module(link_functions link_functions.cpp /Users/alex/tensorkit-learn/tensor_cpp/tensor.cpp lf_bindings.cpp)

# Link Libraries (if required)
target_link_libraries(link_functions PRIVATE pybind11::module)

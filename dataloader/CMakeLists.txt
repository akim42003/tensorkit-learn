cmake_minimum_required(VERSION 3.14)
project(DataloaderBindings)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure CMake finds the correct Python version
set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
# Specify where to find pybind11
set(pybind11_DIR "/Users/alex/tensorkit-learn/venv/lib/python3.11/site-packages/pybind11/share/cmake/pybind11")

# Find pybind11
find_package(pybind11 REQUIRED)

# Include project directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add the dataloader module
pybind11_add_module(dataloader
    dl_bindings.cpp 
    dataloader.cpp 
    /Users/alex/tensorkit-learn/tensor_cpp/tensor.cpp
)

# Ensure the target exists before setting properties
if (TARGET dataloader)
    set_target_properties(dataloader PROPERTIES PREFIX "")
else()
    message(FATAL_ERROR "dataloaderlib target was not created correctly.")
endif()

# Link libraries (if needed)
target_link_libraries(dataloader PRIVATE pybind11::module)

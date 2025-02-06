cmake_minimum_required(VERSION 3.14)
project(TensorKitBindings)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure CMake finds the correct Python version
set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Specify where to find pybind11
set(pybind11_DIR "/Users/alex/tensorkit-learn/venv/lib/python3.11/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

# Include project directories
include_directories(${PROJECT_SOURCE_DIR}/include)
# Get site-packages directory dynamically
execute_process(COMMAND python3 -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE SITE_PACKAGES_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

# Set the output directory for compiled bindings (avoid manual moving)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${SITE_PACKAGES_DIR})

#  Tensor Module
pybind11_add_module(tensor_slow 
    bindings.cpp 
    /Users/alex/tensorkit-learn/tensor_cpp/tensor.cpp
)
target_link_libraries(tensor_slow PRIVATE pybind11::module)

#  DataLoader Module
pybind11_add_module(dataloader
    dl_bindings.cpp
    /Users/alex/tensorkit-learn/dataloader/dataloader.cpp
    /Users/alex/tensorkit-learn/tensor_cpp/tensor.cpp
)
target_link_libraries(dataloader PRIVATE pybind11::module)

#  Link Functions Module
pybind11_add_module(link_functions
    /Users/alex/tensorkit-learn/ml_models/link_functions/link_functions.cpp
    /Users/alex/tensorkit-learn/tensor_cpp/tensor.cpp
    lf_bindings.cpp
)
target_link_libraries(link_functions PRIVATE pybind11::module)

# Ensure targets are properly created
foreach(target tensor_slow dataloader link_functions)
    if (TARGET ${target})
        set_target_properties(${target} PROPERTIES PREFIX "")
    else()
        message(FATAL_ERROR "${target} target was not created correctly.")
    endif()
endforeach()

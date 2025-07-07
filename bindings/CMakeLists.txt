cmake_minimum_required(VERSION 3.14)
project(TensorKitBindings)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure CMake finds the correct Python version
set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find pybind11 - try multiple methods for portability
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    # Try to find pybind11 via Python
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE pybind11_FIND_RESULT
    )
    
    if(pybind11_FIND_RESULT EQUAL 0)
        set(pybind11_DIR ${pybind11_DIR})
        find_package(pybind11 REQUIRED)
    else()
        message(FATAL_ERROR "pybind11 not found. Please install it with: pip install pybind11")
    endif()
endif()

# Include project directories
include_directories(${PROJECT_SOURCE_DIR}/include)
# Get site-packages directory dynamically
execute_process(COMMAND python3 -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE SITE_PACKAGES_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

# Set the output directory for compiled bindings (avoid manual moving)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${SITE_PACKAGES_DIR})

# Define source directories relative to project root
set(TENSOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../tensor_cpp)
set(DATALOADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../dataloader)
set(LINK_FUNCTIONS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../ml_models/link_functions)
set(MLP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../ml_models/mlp)

#  Tensor Module
pybind11_add_module(tensor_slow 
    bindings.cpp 
    ${TENSOR_DIR}/tensor.cpp
)
target_link_libraries(tensor_slow PRIVATE pybind11::module)

#  DataLoader Module
pybind11_add_module(dataloader
    dl_bindings.cpp
    ${DATALOADER_DIR}/dataloader.cpp
    ${TENSOR_DIR}/tensor.cpp
)
target_link_libraries(dataloader PRIVATE pybind11::module)

#  Link Functions Module
pybind11_add_module(link_functions
    lf_bindings.cpp
    ${LINK_FUNCTIONS_DIR}/link_functions.cpp
    ${TENSOR_DIR}/tensor.cpp
)
target_link_libraries(link_functions PRIVATE pybind11::module)

#  MLP Module
pybind11_add_module(mlp_cpp
    mlp_bindings.cpp
    ${MLP_DIR}/mlp.cpp
    ${TENSOR_DIR}/tensor.cpp
)
target_link_libraries(mlp_cpp PRIVATE pybind11::module)

# Ensure targets are properly created
foreach(target tensor_slow dataloader link_functions mlp_cpp)
    if (TARGET ${target})
        set_target_properties(${target} PROPERTIES PREFIX "")
    else()
        message(FATAL_ERROR "${target} target was not created correctly.")
    endif()
endforeach()

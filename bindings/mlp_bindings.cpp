#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../ml_models/mlp/mlp.h"

namespace py = pybind11;

PYBIND11_MODULE(mlp_cpp, m) {
    m.doc() = "Multi-Layer Perceptron C++ implementation";
    
    // Activation base class
    py::class_<Activation, std::shared_ptr<Activation>>(m, "Activation")
        .def("forward", &Activation::forward)
        .def("backward", &Activation::backward)
        .def("name", &Activation::name);
    
    // ReLU activation
    py::class_<ReLU, Activation, std::shared_ptr<ReLU>>(m, "ReLU")
        .def(py::init<>());
    
    // Sigmoid activation  
    py::class_<Sigmoid, Activation, std::shared_ptr<Sigmoid>>(m, "Sigmoid")
        .def(py::init<>());
    
    // Tanh activation
    py::class_<Tanh, Activation, std::shared_ptr<Tanh>>(m, "Tanh")
        .def(py::init<>());
    
    // Linear activation
    py::class_<Linear, Activation, std::shared_ptr<Linear>>(m, "Linear")
        .def(py::init<>());
    
    // Layer class
    py::class_<Layer, std::shared_ptr<Layer>>(m, "Layer")
        .def(py::init<int, int, std::shared_ptr<Activation>, const std::string&>(),
             py::arg("input_dim"),
             py::arg("output_dim"),
             py::arg("activation") = std::make_shared<Linear>(),
             py::arg("init_method") = "xavier")
        .def("forward", &Layer::forward)
        .def("backward", &Layer::backward)
        .def("get_weights", &Layer::get_weights)
        .def("get_bias", &Layer::get_bias)
        .def("get_input_dim", &Layer::get_input_dim)
        .def("get_output_dim", &Layer::get_output_dim);
    
    // MLP class
    py::class_<MLP>(m, "MLP")
        .def(py::init<>())
        .def("add_layer", &MLP::add_layer,
             py::arg("input_dim"),
             py::arg("output_dim"),
             py::arg("activation") = std::make_shared<ReLU>(),
             py::arg("init_method") = "xavier")
        .def("forward", &MLP::forward)
        .def("backward", &MLP::backward)
        .def("num_layers", &MLP::num_layers)
        .def("get_layer", &MLP::get_layer);
}
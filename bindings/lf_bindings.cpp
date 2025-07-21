#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../ml_models/link_functions/link_functions.h"

namespace py = pybind11;

PYBIND11_MODULE(link_functions, m) {
    m.doc() = "Python bindings for LinkFunction and derived classes";

    // Base class: LinkFunction
    py::class_<LinkFunction, std::shared_ptr<LinkFunction>>(m, "LinkFunction")
        .def("__call__", &LinkFunction::operator(), py::arg("mu"), "Apply the link function g(mu) = eta")
        .def("inverse", &LinkFunction::inverse, py::arg("eta"), "Apply the inverse link function g^-1(eta) = mu");

    // Derived class: IdentityLink
    py::class_<IdentityLink, LinkFunction, std::shared_ptr<IdentityLink>>(m, "IdentityLink")
        .def(py::init<>(), "Identity link function (g(mu) = mu)");

    // Derived class: LogitLink
    py::class_<LogitLink, LinkFunction, std::shared_ptr<LogitLink>>(m, "LogitLink")
        .def(py::init<>(), "Logit link function (g(mu) = log(mu / (1 - mu)))");

    // Derived class: LogLink
    py::class_<LogLink, LinkFunction, std::shared_ptr<LogLink>>(m, "LogLink")
        .def(py::init<>(), "Log link function (g(mu) = log(mu))");
}

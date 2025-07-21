#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../tensor_cpp/tensor.h"
#include "../dataloader/dataloader.h"

namespace py = pybind11;

PYBIND11_MODULE(dataloader, m) {
    py::class_<DataLoader>(m, "DataLoader")
        .def(py::init<const std::string&, const std::vector<size_t>&, size_t, bool>(),
            
             py::arg("file_path"), py::arg("tensor_shape"), py::arg("batch_size") = 32, py::arg("shuffle") = true)
        .def("load_data", &DataLoader::loadData, "Load data from a CSV file",
            
             py::arg("file_path"), py::arg("tensor_shape"))
        .def("get_next_batch", &DataLoader::getNextBatch, "Fetch the next batch of data",
             
             py::arg("start_index"))
        .def("iterate_dataset", &DataLoader::iterateDataset, "Iterate through the dataset");
}

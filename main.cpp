#include <pybind11/pybind11.h>
#include "classifier.h"

namespace py = pybind11;

PYBIND11_MODULE(tensorrt_cls_pybind, m) {
    py::class_<ClassifierConfig>(m, "ClassifierConfig")
        .def(py::init<>())
        .def_readwrite("model_file_path", &ClassifierConfig::m_model_file_path);

    py::class_<Classifier>(m, "Classifier")
        .def(py::init<ClassifierConfig>())
        .def("inference", &Classifier::inference);
}

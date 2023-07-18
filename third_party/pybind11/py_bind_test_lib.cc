// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.

#include <pybind11/pybind11.h>

namespace py = pybind11;

int add(int i, int j) { return i + j; }

PYBIND11_MODULE(py_bind_test_lib, m) { m.def("add", &add, "A function which adds two numbers"); }

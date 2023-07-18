CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python
WORKSPACE ?= /workspace/tao-tf1

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
INCLUDE_DIR := ${PWD}
CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++14 -I${INCLUDE_DIR}
LDFLAGS = -shared ${TF_LFLAGS}

multiple_polyline_to_polygon_srcs = $(wildcard ./*.cc)
multiple_polyline_to_polygon_lib = ../op_multiple_polyline_to_polygon.so

multiple_polyline_to_polygon_op: $(multiple_polyline_to_polygon_lib)
$(multiple_polyline_to_polygon_lib): $(multiple_polyline_to_polygon_srcs)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

clean:
	rm -rf ${multiple_polyline_to_polygon_lib}
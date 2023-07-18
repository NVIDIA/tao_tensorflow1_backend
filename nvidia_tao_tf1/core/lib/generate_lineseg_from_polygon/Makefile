CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python
WORKSPACE ?= /workspace/tao-tf1

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++14
LDFLAGS = -shared ${TF_LFLAGS}

generate_lineseg_from_polygon_srcs = $(wildcard ./*.cc)
generate_lineseg_from_polygon_lib = ../op_generate_lineseg_from_polygon.so

generate_lineseg_from_polygon_op: $(generate_lineseg_from_polygon_lib)
$(generate_lineseg_from_polygon_lib): $(generate_lineseg_from_polygon_srcs)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

clean:
	rm -rf ${generate_lineseg_from_polygon_lib}
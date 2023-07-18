CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python
WORKSPACE ?= /workspace/tao-tf1

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++14
LDFLAGS = -shared ${TF_LFLAGS}

merge_polylines_srcs = $(wildcard ./*.cc)
merge_polylines_lib = ../op_merge_polylines.so

merge_polylines_op: $(merge_polylines_lib)

$(merge_polylines_lib): $(merge_polylines_srcs)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

clean:
	rm -rf ${merge_polylines_lib}
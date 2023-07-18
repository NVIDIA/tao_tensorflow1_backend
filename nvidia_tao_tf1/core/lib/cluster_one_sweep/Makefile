CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python
WORKSPACE ?= /workspace/tao-tf1

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++14
LDFLAGS = -shared ${TF_LFLAGS}

cluster_one_sweep_srcs = $(wildcard ./*.cc)
cluster_one_sweep_lib = ../op_cluster_one_sweep.so

cluster_one_sweep_op: $(cluster_one_sweep_lib)

$(cluster_one_sweep_lib): $(cluster_one_sweep_srcs)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

clean:
	rm -rf ${cluster_one_sweep_lib}
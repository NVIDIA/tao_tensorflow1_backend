CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python
WORKSPACE ?= /workspace/tao-tf1

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++14
LDFLAGS = -shared ${TF_LFLAGS}

values_and_count_to_sparse_tensor_srcs = $(wildcard ./*.cc)
values_and_count_to_sparse_tensor_lib = ../op_values_and_count_to_sparse_tensor.so

values_and_count_to_sparse_tensor_op: $(values_and_count_to_sparse_tensor_lib)

$(values_and_count_to_sparse_tensor_lib): $(values_and_count_to_sparse_tensor_srcs)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

# CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++14
# LDFLAGS = -shared ${TF_LFLAGS}

clean:
	rm -rf ${values_and_count_to_sparse_tensor_lib}
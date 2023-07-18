CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python
WORKSPACE ?= /workspace/tao-tf1

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++14
LDFLAGS = -shared ${TF_LFLAGS}
INCLUDE = -I${WORKSPACE}/third_party/json_cpp

json_arrays_to_tensor_srcs = $(wildcard ./*.cc)
json_arrays_to_tensor_lib = ../op_json_arrays_to_tensor.so

json_arrays_to_tensor_op: $(json_arrays_to_tensor_lib)

$(json_arrays_to_tensor_lib): $(json_arrays_to_tensor_srcs)
	$(CXX) $(CFLAGS) ${INCLUDE} -o $@ $^ ${LDFLAGS}

# CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++14
# LDFLAGS = -shared ${TF_LFLAGS}

clean:
	rm -rf ${json_arrays_to_tensor_lib}
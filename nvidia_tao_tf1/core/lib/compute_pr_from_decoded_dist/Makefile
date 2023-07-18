CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python
WORKSPACE ?= /workspace/tao-tf1
INCLUDE_DIR = ${WORKSPACE}

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
IFLAGS := -I${INCLUDE_DIR}

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++14
LDFLAGS = -shared ${TF_LFLAGS}

compute_pr_from_decoded_dist_srcs = $(wildcard ./*.cc)
compute_pr_from_decoded_dist_lib = ../op_compute_pr_from_decoded_dist.so

compute_pr_from_decoded_dist_op: $(compute_pr_from_decoded_dist_lib)
$(compute_pr_from_decoded_dist_lib): $(compute_pr_from_decoded_dist_srcs)
	$(CXX) $(CFLAGS) $(IFLAGS) -o $@ $^ ${LDFLAGS}

# CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++14
# LDFLAGS = -shared ${TF_LFLAGS}

clean:
	rm -rf ${compute_pr_from_decoded_dist_lib}
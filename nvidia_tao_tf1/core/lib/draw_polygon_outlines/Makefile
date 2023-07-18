CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python
WORKSPACE ?= /workspace/tao-tf1

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++14
LDFLAGS = -shared ${TF_LFLAGS}
INCLUDE = -I${WORKSPACE}/third_party/json_cpp

draw_polygon_outlines_srcs = $(wildcard ./*.cc)
draw_polygon_outlines_lib = ../op_draw_polygon_outlines.so

draw_polygon_outlines_op: $(draw_polygon_outlines_lib)

$(draw_polygon_outlines_lib): $(draw_polygon_outlines_srcs)
	$(CXX) $(CFLAGS) ${INCLUDE} -o $@ $^ ${LDFLAGS}

# CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++14
# LDFLAGS = -shared ${TF_LFLAGS}

clean:
	rm -rf ${draw_polygon_outlines_lib}
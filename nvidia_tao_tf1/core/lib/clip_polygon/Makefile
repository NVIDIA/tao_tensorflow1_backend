CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python
WORKSPACE ?= /workspace/tao-tf1

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++14
LDFLAGS = -shared ${TF_LFLAGS}
INCLUDE = "-I${WORKSPACE}/third_party/clipper"

clip_polygon_srcs = $(wildcard ./*.cc)
clip_polygon_lib = ../op_clip_polygon.so
clipper_src = ${WORKSPACE}/third_party/clipper/clipper.cpp
clipper_lib = ${WORKSPACE}/third_party/clipper/clipper.o

clip_polygon_op: $(clip_polygon_lib) 
$(clip_polygon_lib): $(clip_polygon_srcs) ${clipper_lib}
	$(CXX) $(CFLAGS) $(INCLUDE) -o $@ $^ ${LDFLAGS}

${clipper_lib}: ${clipper_src}
	$(CXX) $(CFLAGS) -std=c++14 ${INCLUDE} -o $@ $^ ${LDFLAGS}

clean:
	rm -rf ${clip_polygon_lib} ${clipper_lib}
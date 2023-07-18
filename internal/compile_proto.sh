
#!/bin/bash

# Generate _pb2.py file for respective model type

apt-get install -y protobuf-compiler

protoc nvidia_tao_tf1/cv/$1/proto/*.proto --python_out=.

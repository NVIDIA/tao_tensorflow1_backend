all: build install

build:
	rm -rf **/__pycache__
	python3 setup.py bdist_wheel

clean:
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

install: build
	pip3 install dist/nvidia_tao_tf1*.whl

uninstall:
	pip3 uninstall -y nvidia-tao-tf1


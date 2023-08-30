# TFLite convertor

This tool helps convert a Keras model trained by the TAO Toolkit TF1 container to a TFLite model.

<!-- vscode-markdown-toc -->
* [Installation](#Installation)
* [Running the converter](#Runningtheconverter)

<!-- vscode-markdown-toc-config
	numbering=false
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

## <a name='Installation'></a>Installation

Follow the steps below, to setup the required environment for the converter.

1. Create a conda env

    ```sh
    conda create -n tflite_convert python=3.6
    ```

2. Activate the environment and install the required dependencies

    ```sh
    conda activate tflite_convert
    pip install -r requirements-pip.txt
    ```

3. Add the TAO_TF2_ROOT directory to your pythonpath variable.

    ```sh
    export PYTHONPATH=${PWD}/../../:$PYTHONPATH
    ```

## <a name='Runningtheconverter'></a>Running the converter

The sample usage for the converter

```sh
usage: export_tflite [-h] [--model_file MODEL_FILE] [--key KEY] [--output_file OUTPUT_FILE]

Export keras models to tflite.

optional arguments:
  -h, --help            show this help message and exit
  --model_file MODEL_FILE
                        Path to a model file.
  --key KEY             Key to load the model.
  --output_file OUTPUT_FILE
                        Path to the output model file.
```

Sample command to run the tflite converter.

```sh
python export_tflite.py --model_file /path/to/model.[tlt/hdf5] \
                        --output_file /path/to/model.tflite \
                        --key $KEY
```

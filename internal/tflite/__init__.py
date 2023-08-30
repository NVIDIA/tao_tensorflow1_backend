"""Init module to run the tflite conversion."""


import os

# Apply patch to keras if only the model is loaded with native keras
# and not tensorflow.keras.
if os.getenv("TF_KERAS", "0") != "1":
    import third_party.keras.mixed_precision as MP
    import third_party.keras.tensorflow_backend as TFB

    MP.patch()
    TFB.patch()

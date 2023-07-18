# MagLev Keras

This module contains files for adding functionality to keras, for forking on monkey-patching.

Whenever you import Maglev, it will patch keras.

Updating the `monkey.patch` file can be done by changing the target Keras's `__init__.py`
and running `diff -Naur keras/__init__.py keras/__init__.py2 > monkey.patch`.
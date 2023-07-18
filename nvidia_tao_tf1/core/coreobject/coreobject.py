# Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
r"""TAOObject APIs.

The :any:`TAOObject` is a baseclass meant to enable subclasses to easily
and automatically serialize and deserialize their constructor arguments to a Python dict.

Because two-way serialization between a dict and many serialization languages is trivial
(e.g. YAML, JSON), the :any:`TAOObject` allows for serialization of your objects to such
languages, and deserialization (a.k.a. 'building') from them.

The :any:`TAOObject` does so by saving the input arguments to your object. The input arguments
can be
of any type that can be serializated to a dict (e.g. int, float, string, list, tuple, dict, etc),
or a :any:`TAOObject` itself. Amongst types not supported are; any object not inheriting from
:any:`TAOObject`, including ``namedtuple``, ``OrderedDict``, etc.

The :any:`TAOObject` is currently used by several apps [#f1]_ to automate their spec-to-code
conversion.

Using the :any:`TAOObject`
-----------------------------

* Inherit from  :any:`TAOObject`
* Add the :any:`save_args` around your ``__init__`` method.
* Pass on ``**kwargs`` in your ``__init__`` method.
* Call your parent object with ``**kwargs``.

Example
-------

.. code-block:: python

    from nvidia_tao_tf1.core.coreobject import deserialize_tao_object
    from nvidia_tao_tf1.core.coreobject import TAOObject
    from nvidia_tao_tf1.core.coreobject import save_args

    class MyTAOObjectChild(TAOObject):
        @save_args
        def __init__(self, my_arg, **kwargs):
            super(MyTAOObjectChild, self).__init__(**kwargs)
            self.my_arg = my_arg

    # Instantate your object.
    o = MyTAOObjectChild(my_arg=1337)

    # Serialize this TAOObject.
    d = o.serialize()  # `d` is of type dict.
    o.to_yaml('spec.yaml')  # Writes `o` to a YAML file.
    o.to_json('spec.json')  # Writes `o` to a JSON file.

    # Deserialize
    o_from_d = deserialize_tao_object(d)  # `o` is of type MyTAOObjectChild.

    print(o_from_d.my_arg)  # Output: ``1337``.

.. [#f1] `Python Constructor Spec (TAOObject) Design Doc
    <https://docs.google.com/document/d/1K2jZ02a7fFq2a1cWDbUkUhbAs2pd5RKd6EdOTrMnB3U>`_.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from collections import Hashable
import inspect
from io import open  # Python 2/3 compatibility. pylint: disable=W0622
import logging
import types

import google.protobuf.message
import simplejson as json
from six import text_type, with_metaclass
import yaml

from nvidia_tao_tf1.core.coreobject import protobuf_to_dict

logger = logging.getLogger(__name__)

# Dictionary of registered classes.
_MLOBJECT_REGISTRY = {}

_CLASS_NAME_KEY = "__class_name__"
_FUNCTION_NAME_KEY = "__function_name__"
_IGNORE_KEYS = {_CLASS_NAME_KEY, _FUNCTION_NAME_KEY}


def deserialize_tao_object(data):
    """
    Deserialize a (child of) a :class:`.TAOObject`.

    The deserialization of *any* child class of this object is done in three steps.

    (1) Retrieving the actual MagLev object (pointer) through the value of the ``class_name`` key.
    (2) Initialization of the object through passing in the ``config`` of the class directly and
        entirely by converting it to keyword arguments to the initializer.

    Args:
        data (dict): A serialized structure containing the information to deserialize any child of
            TAOObject entirely.
    Returns:
        :class:`.TAOObject`: Any child object of a :class:`.TAOObject` that has been
            deserialized.
    """
    is_class = _CLASS_NAME_KEY in data
    is_function = _FUNCTION_NAME_KEY in data

    if is_class is is_function:
        raise ValueError(
            "Exactly one of {} or {} must be present in data.".format(
                _CLASS_NAME_KEY, _FUNCTION_NAME_KEY
            )
        )

    _KEY = _CLASS_NAME_KEY if is_class else _FUNCTION_NAME_KEY

    name_handle = data[_KEY]
    if name_handle not in _MLOBJECT_REGISTRY:
        raise ValueError(
            "Trying to deserialize object / function of class `{}`, but it was not found "
            "in the registry. This can typically be solved by making sure the top-level script "
            "you are running has in its 'import tree' imported all the modules and classes "
            "you need.".format(name_handle)
        )
    if isinstance(_MLOBJECT_REGISTRY[name_handle], list):
        if len(_MLOBJECT_REGISTRY[name_handle]) > 1:
            message = (
                "Found multiple class / function names matching '{}', please update your input "
                "configuration's __class_name__ and / or __function_name__ entries to be in "
                "the format 'module.lib.package.MyClass'. Candidates include: ".format(
                    name_handle
                )
            )
            message += ", ".join(
                [_get_registry_key(entry) for entry in _MLOBJECT_REGISTRY[name_handle]]
            )

            raise ValueError(message)
    kwargs = _get_kwargs(data)

    def _deserialize_recursively(value):
        """Recursively deserializes modulusobjects."""
        if isinstance(value, dict):
            if _CLASS_NAME_KEY in value or _FUNCTION_NAME_KEY in value:
                value = deserialize_tao_object(value)
            else:
                value = {
                    key: _deserialize_recursively(val) for key, val in value.items()
                }
        elif isinstance(value, list):
            for i in range(len(value)):
                value[i] = _deserialize_recursively(value[i])
        return value

    for key in kwargs:
        if key in _IGNORE_KEYS:
            continue
        kwargs[key] = _deserialize_recursively(value=kwargs[key])

    obj_ptr = _MLOBJECT_REGISTRY[name_handle]
    if isinstance(obj_ptr, list):
        obj_ptr = obj_ptr[0]  # We already checked for single entry above.

    if isinstance(obj_ptr, types.FunctionType):
        tao_object = obj_ptr
    else:
        tao_object = obj_ptr(**kwargs)

    return tao_object


def register_tao_object(candidate):
    """
    Register a tao object class or function.

    Args:
        candidate (class or function): Class or function to register.
    """
    registry_key = _get_registry_key(candidate)
    if registry_key not in _MLOBJECT_REGISTRY:
        _MLOBJECT_REGISTRY[registry_key] = candidate
    else:
        raise RuntimeError(
            "Conflicting module and class names: %s already exists in the TAOObject registry. "
            "This probably means you are overriding a class definition that you should have left "
            "untouched." % registry_key
        )

    # For backwards compatibility, since most class names were _already_ unique across the 'default'
    # NAME_SPACE (attribute since removed), we can register them by a shorter handle, and raise
    # an error at deserialization time if multiple definitions are found.
    obj_name = candidate.__name__
    if obj_name not in _MLOBJECT_REGISTRY:
        _MLOBJECT_REGISTRY[obj_name] = []
    _MLOBJECT_REGISTRY[obj_name].append(candidate)


def register_tao_function(func):
    """Decorator that registers a function.

    Args:
        func (function): Function to be added to the registry.

    Returns:
        func (function).
    """
    # Functions can be registered the same way as classes.
    register_tao_object(func)

    return func


def _get_registry_key(candidate):
    # In practice, would look something like: modulus.some_lib.MyClass.
    return ".".join([candidate.__module__, candidate.__name__])


class MetaTAOObject(type):
    """
    TAOObject meta class.

    See references to ``with_metaclass(MetaTAOObject, object)``.
    An instance of the meta class is created whenever a :class:`.TAOObject` class
    is defined. In the ``__init__`` method of the meta-class we register the newly
    defined class into a global registry of :class:`.TAOObject` descendants.

    Args:
        name (str): name of the newly defined class
        bases (list): list of base classes
        attr (dict): dictionary of attributes
    """

    def __init__(cls, name, bases, attr):
        """init function."""
        type.__init__(cls, name, bases, attr)

        register_tao_object(cls)


class TAOObject(with_metaclass(MetaTAOObject, object)):
    """
    Core TAOObject class.

    A Maglev Object is independently serializable and deserializable, and thus should contain a
    variable configuration obtainable by get_config().
    """

    def serialize(self):
        """
        Serialization of the object.

        Returns:
            data (dict): A structure containing serialized information about this object
        """
        data = {_CLASS_NAME_KEY: _get_registry_key(self.__class__)}
        data.update(self.get_config())
        return data

    def get_config(self):
        """
        Obtain this object's configuration.

        Returns:
            dict: A structure containing all information to configure this object.
        """
        if hasattr(self, "_modulus_config"):
            # self._modulus_config is automatically populated by @save_args decorator.
            # _config seems to be a common member variable name for teams,
            # using _modulus_config to prevent name collisions.
            return self._modulus_config
        return {}

    def to_json(self, filename=None, indent=4, sort_keys=True, **kwargs):
        """
        Serialize graph and write to json.

        Args:
            filename (str): If supplied, it will write a json file to the cwd.
            **kwargs: Keyword arguments that will passed on to `json.dumps()`.
        Returns:
            data (str): Serialized object (json).
        """
        data = json.dumps(
            self.serialize(), indent=indent, sort_keys=sort_keys, **kwargs
        )
        if filename:
            with open(filename, "w") as f:
                f.write(text_type(data))
        return data

    def to_yaml(
        self, filename=None, encoding="utf-8", default_flow_style=False, **kwargs
    ):
        """
        Serialize graph and write to yaml.

        Args:
            filename (str): If supplied, it will write a yaml file to the cwd.
            **kwargs: Keyword arguments that will passed on to `yaml.dump()`.
        Returns:
            data (str): Serialized object (yaml).
        """
        data = yaml.safe_dump(
            self.serialize(),
            encoding=encoding,
            default_flow_style=default_flow_style,
            **kwargs
        )
        if filename:
            with open(filename, "wb") as f:
                f.write(data)
        return data

    def __eq__(self, b):
        """
        Loose check if two TAOObjects are equal.

        This function just compares the serialized form of the Objects
        for equality. Derived classes should implement stricter checks
        based on their implementation.

        Args:
            b (TAOObject): Instance of TAOObject class to compare with.

        Returns:
            true if objects' serialized representation match, false otherwise.
        """
        if not isinstance(b, TAOObject):
            return False
        return self.serialize() == b.serialize()

    def __hash__(self):
        """Computes hash based on serialized dict contents."""
        d = self.serialize()
        return sum(
            [
                hash(key) + (hash(value) if isinstance(value, Hashable) else 0)
                for key, value in d.items()
            ]
        )


class AbstractMetaTAOObject(ABCMeta, MetaTAOObject):
    """Used multiple inheritance to forge two meta classes for building AbstractTAOObject."""

    pass


class AbstractTAOObject(with_metaclass(AbstractMetaTAOObject, TAOObject)):
    """TAOObject abstract base class for interfaces to inherit from."""

    pass


try:
    _BUILTIN_PYTHON_TYPES = (
        int,
        str,
        unicode,
        bool,
        long,
        float,
        type(None),
        set,
    )  # noqa
except NameError:
    # Python 3.
    _BUILTIN_PYTHON_TYPES = (int, str, bytes, bool, float, type(None), set)


def _recursively_serialize(val):
    """Serializes a value, recursing through iterable structures (list or tuple)."""
    # Loop and serialize.
    if isinstance(val, (list, tuple)):
        val = [_recursively_serialize(v) for v in val]
    elif isinstance(val, dict):
        val = {key: _recursively_serialize(value) for key, value in val.items()}
    elif isinstance(val, TAOObject):
        val = val.serialize()
    elif isinstance(val, types.FunctionType):
        val = {_FUNCTION_NAME_KEY: _get_registry_key(val)}
    elif isinstance(val, google.protobuf.message.Message):
        val = protobuf_to_dict.protobuf_to_modulus_dict(
            val,
            add_class_metadata=True,
            overwrite_package_with_name="default",
            overwrite_class_with_name="TAOKeywordObject",
        )
    return val


def save_args(func):
    """
    Decorator to put around initializer of TAOObject to save arguments.

    A decorator that saves named arguments (not ``*args`` or ``**kwargs``)
    into an object's config field.

    This decorator is meant to wrap the ``__init__`` method of a
    :class:`nvidia_tao_tf1.core.coreobject.TAOObject` class.

    Typical usage would be to define a class as::

        class TAOObjectChild(TAOObject):
            @save_args
            def __init__(self, arg1, *args, **kwargs):
                super(TAOObjectChild, self).__init__(*args, **kwargs)

    Then on a call to this class's ``serialize`` method, the returned
    config will include the value of ``arg1`` that was specified on
    instantiation. If ``arg1`` is a :class:`nvidia_tao_tf1.core.coreobject.TAOObject` the argument is
    serialized recursively.
    """

    def wrapper(*args, **kwargs):
        arg_spec = inspect.getargspec(func)  # pylint: disable=W1505

        # Check if we are missing any required arguments so we can throw a descriptive error.
        # Remove the ones with defaults. Defaults are counted from the end.
        n_defaults = 0 if arg_spec.defaults is None else len(arg_spec.defaults)
        required_args = arg_spec.args[:-n_defaults]
        # Remove the ones that have been provided as non-keyword args.
        required_args = required_args[len(args) :]
        # Remove the ones that have been given as kwargs
        required_args = list(set(required_args).difference(set(kwargs.keys())))
        if required_args:
            raise TypeError(
                "{} missing required arguments: {}.".format(args[0], required_args)
            )

        call_args = inspect.getcallargs(func, *args, **kwargs)  # pylint: disable=W1505
        f = func(*args, **kwargs)
        obj = call_args[arg_spec.args[0]]

        if not isinstance(obj, TAOObject):
            raise ValueError("This decorator only supports TAOObject instances.")
        if not hasattr(obj, "_modulus_config"):
            obj._modulus_config = {}

        def process_arg(arg, arg_map):
            val = _recursively_serialize(arg_map[arg])
            if arg in obj._modulus_config:
                if obj._modulus_config[arg] != val:
                    raise ValueError(
                        "argument '%s' is already in config "
                        "and values are different." % arg
                    )
            obj._modulus_config[arg] = val

        # Loop over object arguments (ignore first argument `self`).
        for arg in arg_spec.args[1:]:
            process_arg(arg, call_args)

        # process other kwargs
        if arg_spec.keywords:
            keywords = arg_spec.keywords
            other_kwargs = call_args[keywords].keys()
            for arg in other_kwargs:
                process_arg(arg, call_args[keywords])

        return f

    return wrapper


def _get_kwargs(data):
    return {key: val for key, val in data.items() if key not in _IGNORE_KEYS}

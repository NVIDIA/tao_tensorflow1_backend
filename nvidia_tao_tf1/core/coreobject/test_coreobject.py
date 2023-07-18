# Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import json
from parameterized import parameterized
import pytest
import yaml

from nvidia_tao_tf1.core.coreobject import deserialize_tao_object, TAOObject, save_args
from nvidia_tao_tf1.core.coreobject.test_fixtures import (
    get_duplicated_tao_object_child_class,
    MY_RETURN_VALUE,
    my_test_function,
)


class TAOObjectChild(TAOObject):
    @save_args
    def __init__(
        self, arg1, arg2="default_arg2_val", *args, **kwargs
    ):  # pylint: disable=W1113
        super(TAOObjectChild, self).__init__(*args, **kwargs)
        self.arg1 = arg1
        self.arg2 = arg2


class TAOObjectGrandChild(TAOObjectChild):
    @save_args
    def __init__(
        self, arg3, arg4="default_arg4_val", *args, **kwargs
    ):  # pylint: disable=W1113
        super(TAOObjectGrandChild, self).__init__(*args, **kwargs)


class TAOObjectGrandChildWithSameArg(TAOObjectChild):
    @save_args
    def __init__(
        self, arg1, arg3, arg4="default_arg4_val", *args, **kwargs
    ):  # pylint: disable=W1113
        super(TAOObjectGrandChildWithSameArg, self).__init__(arg1, *args, **kwargs)


class TAOObjectGrandChildWithSameArgAndDifferentValue(TAOObjectChild):
    @save_args
    def __init__(self, arg1, arg2, *args, **kwargs):
        super(TAOObjectGrandChildWithSameArgAndDifferentValue, self).__init__(
            arg1, **kwargs
        )


class NonTAOObjectChild(object):
    @save_args
    def __init__(
        self, arg1, arg2="default_arg2_val", *args, **kwargs
    ):  # pylint: disable=W1113
        super(NonTAOObjectChild, self).__init__(*args, **kwargs)


class TestTAOObjectSaveArgs(object):
    """Test class for @save_args decorator on TAOObject."""

    def test_child(self):
        a = TAOObjectChild("arg1_val")
        data = a.serialize()
        assert "arg1" in data
        assert data["arg1"] == "arg1_val"
        assert "arg2" in data
        assert data["arg2"] == "default_arg2_val"

    def test_child_arg_is_maglev_obj(self):
        a = TAOObjectChild("arg1_val", arg2="arg2_val")
        b = TAOObjectChild(arg1=a)
        data = b.serialize()
        assert "arg1" in data
        b_arg1 = data["arg1"]
        # The child was defined in this module.
        assert b_arg1["__class_name__"] == __name__ + "." + "TAOObjectChild"
        assert b_arg1["arg1"] == "arg1_val"
        assert b_arg1["arg2"] == "arg2_val"
        assert "arg2" in data
        assert data["arg2"] == "default_arg2_val"

    def test_error_on_non_maglev_obj(self):
        with pytest.raises(ValueError):
            NonTAOObjectChild("arg1_val")

    def test_missing_required_argument_error(self):
        with pytest.raises(TypeError):
            TAOObjectChild()
        with pytest.raises(TypeError):
            TAOObjectChild(arg2="arg2_val")
        with pytest.raises(TypeError):
            TAOObjectGrandChild(arg3="arg3_val")

    def test_grand_child(self):
        a = TAOObjectGrandChild(arg1="arg1_val", arg2="arg2_val", arg3="arg3_val")
        data = a.serialize()
        assert "arg1" in data
        assert data["arg1"] == "arg1_val"
        assert "arg2" in data
        assert data["arg2"] == "arg2_val"
        assert "arg3" in data
        assert data["arg3"] == "arg3_val"
        assert "arg4" in data
        assert data["arg4"] == "default_arg4_val"

    def test_grand_child_with_same_arg(self):
        a = TAOObjectGrandChildWithSameArg(arg1="arg1_val", arg3="arg3_val")
        data = a.serialize()
        assert "arg1" in data
        assert data["arg1"] == "arg1_val"
        assert "arg2" in data
        assert data["arg2"] == "default_arg2_val"
        assert "arg3" in data
        assert data["arg3"] == "arg3_val"
        assert "arg4" in data
        assert data["arg4"] == "default_arg4_val"

    def test_grand_child_with_same_arg_and_different_value(self):
        with pytest.raises(ValueError):
            TAOObjectGrandChildWithSameArgAndDifferentValue(
                arg1="arg1_val", arg2="arg2_val"
            )

    def test_object_in_object(self, val_a="val_a"):
        a = TAOObjectChild(arg1=val_a)
        b = TAOObjectChild(arg1=a)
        data = b.serialize()
        assert data["arg1"]["arg1"] == val_a

    def test_object_list_in_arg(self, val_a="val_a", val_b="val_b"):
        a = TAOObjectChild(arg1=val_a)
        b = TAOObjectChild(arg1=val_b)
        c = TAOObjectChild(arg1=[a, b])
        data = c.serialize()
        assert len(data["arg1"]) == 2
        assert data["arg1"][0]["arg1"] == val_a
        assert data["arg1"][1]["arg1"] == val_b

    def test_eq_with_maglev_object_same_arg(self, val_a="val_a"):
        a = TAOObjectChild(arg1=val_a)
        b = TAOObjectChild(arg1=val_a)
        assert a == b

    def test_eq_with_maglev_object_diff_arg(self):
        a = TAOObjectChild(arg1="val_a")
        b = TAOObjectChild(arg1="val_b")
        assert not a == b

    def test_eq_with_not_maglev_object(self, val_a="val_a"):
        a = TAOObjectChild(arg1=val_a)
        b = "string_value"
        assert not a == b


class TestTAOObject(object):
    """Test expected behaviour of TAOObject class."""

    @parameterized.expand(
        [
            (1,),  # int
            (1.337,),  # float
            ("arg1_val",),  # string
            ([1337, "1337", 1337],),  # list
            ({"foo", 1337},),  # dict
            (my_test_function,),  # function.
            ({"my_func": my_test_function},),  # Dict with a function as a value.
        ]
    )
    def test_value_serialization(self, value):
        """Test proper serialization of different values and types."""
        # create instance
        o = TAOObjectChild(value)
        # serialize
        d = o.serialize()
        # de-serialize
        o = deserialize_tao_object(d)
        assert isinstance(o, TAOObjectChild)
        assert o.arg1 == value

    def test_serialization_json(self):
        # create instance
        o = TAOObjectChild("arg1val_json")
        # serialize
        s = o.to_json()
        # make sure serialized string is a valid json
        d = json.loads(s)
        # de-serialize
        o = deserialize_tao_object(d)
        assert isinstance(o, TAOObjectChild)
        assert o.arg1 == "arg1val_json"

    def test_serialization_yaml(self):
        # create instance
        o = TAOObjectChild("arg1val_yaml")
        # serialize
        s = o.to_yaml()
        # make sure serialized string is a valid json
        d = yaml.load(s)
        # de-serialize
        o = deserialize_tao_object(d)
        assert isinstance(o, TAOObjectChild)
        assert o.arg1 == "arg1val_yaml"

    def test_serialization_recursive(self):
        # create instances
        o1 = TAOObjectChild("arg1val_recursive")
        o2 = TAOObjectChild(o1)
        # serialize
        d = o2.serialize()
        # de-serialize
        o = deserialize_tao_object(d)
        assert isinstance(o, TAOObjectChild)
        assert isinstance(o.arg1, TAOObjectChild)
        assert o.arg1.arg1 == "arg1val_recursive"

    def test_serialization_recursive_json(self):
        # create instances
        o1 = TAOObjectChild("arg1val_recursive_json")
        o2 = TAOObjectChild(o1)
        # serialize
        s = o2.to_json()
        # make sure serialized string is a valid json
        d = json.loads(s)
        # de-serialize
        o = deserialize_tao_object(d)
        assert isinstance(o, TAOObjectChild)
        assert isinstance(o.arg1, TAOObjectChild)
        assert o.arg1.arg1 == "arg1val_recursive_json"

    def test_serialization_list_yaml(self):
        # create instance
        o1 = TAOObjectChild(arg1="o1")
        o2 = TAOObjectChild(arg1="o2")
        o = TAOObjectChild(arg1=[o1, o2])
        # serialize
        s = o.to_yaml()
        # make sure serialized string is a valid yaml.
        d = yaml.load(s)
        # de-serialize
        o = deserialize_tao_object(d)
        assert isinstance(o, TAOObjectChild)
        assert isinstance(o.arg1[0], TAOObjectChild)
        assert isinstance(o.arg1[1], TAOObjectChild)
        assert o.arg1[0].arg1 == "o1"
        assert o.arg1[1].arg1 == "o2"

    @pytest.mark.parametrize("file_format", ["yaml", "json"])
    def test_serialization_dict(self, file_format):
        """Test serializing a dict whose values are TAOObject instances."""
        o1 = TAOObjectChild(arg1="o1")
        o2 = TAOObjectChild(arg1="o2")
        o = TAOObjectChild(arg1={"kwarg1": o1, "kwarg2": o2})
        if file_format == "yaml":
            s = o.to_yaml()
            d = yaml.load(s)
            o = deserialize_tao_object(d)
        elif file_format == "json":
            s = o.to_json()
            d = json.loads(s)
            o = deserialize_tao_object(d)
        assert isinstance(o, TAOObjectChild)
        assert isinstance(o.arg1["kwarg1"], TAOObjectChild)
        assert isinstance(o.arg1["kwarg2"], TAOObjectChild)
        assert o.arg1["kwarg1"].arg1 == "o1"
        assert o.arg1["kwarg2"].arg1 == "o2"

    @pytest.mark.parametrize("file_format", ["yaml", "json"])
    def test_multiple_levels_of_recursion(self, file_format):
        """Test with multiple levels of recursion TAOObject instances."""
        o1 = TAOObjectChild(arg1="o1")
        o2 = TAOObjectChild(arg1="o2")
        o = TAOObjectChild(arg1={"kwarg1": {"kwarg1_1": o1}}, arg2=[[o2]])
        if file_format == "yaml":
            s = o.to_yaml()
            d = yaml.load(s)
            o = deserialize_tao_object(d)
        elif file_format == "json":
            s = o.to_json()
            d = json.loads(s)
            o = deserialize_tao_object(d)
        assert isinstance(o, TAOObjectChild)
        assert isinstance(o.arg1["kwarg1"]["kwarg1_1"], TAOObjectChild)
        assert isinstance(o.arg2[0][0], TAOObjectChild)
        assert o.arg1["kwarg1"]["kwarg1_1"].arg1 == "o1"
        assert o.arg2[0][0].arg1 == "o2"

    def test_function_serialization(self):
        """Test serialization of function."""
        o1 = TAOObjectChild(arg1="o1")
        o2 = TAOObjectChild(
            arg1={"bogus_dict_key": my_test_function}, arg2=[None, my_test_function]
        )
        o3 = TAOObjectChild(arg1=o1, arg2=o2)

        s = o3.serialize()

        o4 = deserialize_tao_object(s)
        # Now try __call__'ing the appropriate objects.
        assert o4.arg2.arg1["bogus_dict_key"]() == MY_RETURN_VALUE
        assert o4.arg2.arg2[1]() == MY_RETURN_VALUE

    def test_class_name_short_name(self):
        """Test that deserializing with the class name alone works."""
        obj = TAOObjectChild(arg1="o1")
        s = obj.serialize()
        s["__class_name__"] = "TAOObjectChild"
        deserialize_tao_object(s)

    def test_name_collisions(self):
        """Test that an appropriate error is raised when there are name collisions."""
        # Force the registration of another class with the same name as `TAOObjectChild` defined
        # above.
        get_duplicated_tao_object_child_class()

        obj = TAOObjectChild(arg1="rebel")
        s = obj.serialize()
        # Change the "__class_name__" field to the short hand to set up the collision.
        s["__class_name__"] = "TAOObjectChild"
        with pytest.raises(ValueError) as excinfo:
            deserialize_tao_object(s)
        assert (
            "Found multiple class / function names matching 'TAOObjectChild'"
            in str(excinfo.value)
        )

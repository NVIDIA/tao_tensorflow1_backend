# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

from nvidia_tao_tf1.core.decorators.arg_scope import add_arg_scope, arg_scope


class FooBarClass(object):
    def get_foo_bar(self):
        return self.foo, self.bar


class ScopedClass(FooBarClass):
    @add_arg_scope
    def __init__(self, foo=None, bar=None):
        self.foo = foo
        self.bar = bar


class UnscopedClass(FooBarClass):
    def __init__(self, foo=None, bar=None):
        self.foo = foo
        self.bar = bar


class MethodClass(FooBarClass):
    @add_arg_scope
    def scoped_set_foo_bar(self, foo=None, bar=None):
        self.foo = foo
        self.bar = bar
        return self

    def unscoped_set_foo_bar(self, foo=None, bar=None):
        self.foo = foo
        self.bar = bar
        return self


@add_arg_scope
def scoped_func(foo=None, bar=None):
    return foo, bar


def unscoped_func(foo=None, bar=None):
    return foo, bar


foo_arg = "spam"
bar_arg = "eggs"


@pytest.mark.parametrize("foo_arg", [None, "spam"])
@pytest.mark.parametrize("bar_arg", [None, "eggs"])
def test_init_scope(foo_arg, bar_arg):
    """Test argument scoping on (initializers of) objects."""

    # Test unscoped case
    assert ScopedClass().get_foo_bar() == (None, None)
    assert UnscopedClass().get_foo_bar() == (None, None)

    kwargs = {}
    if foo_arg:
        kwargs.update({"foo": foo_arg})
    if bar_arg:
        kwargs.update({"bar": bar_arg})

    with arg_scope([ScopedClass], **kwargs):
        assert ScopedClass().get_foo_bar() == (foo_arg, bar_arg)
        assert UnscopedClass().get_foo_bar() == (None, None)
        # Test we can still override
        assert ScopedClass(foo="dog").get_foo_bar() == ("dog", bar_arg)
        assert ScopedClass(foo="dog", bar="cat").get_foo_bar() == ("dog", "cat")

    # Test unscoped case again to make sure the previous scoping has reset
    assert ScopedClass().get_foo_bar() == (None, None)
    assert UnscopedClass().get_foo_bar() == (None, None)


@pytest.mark.parametrize("foo_arg", [None, "spam"])
@pytest.mark.parametrize("bar_arg", [None, "eggs"])
def test_method_scope(foo_arg, bar_arg):
    """Test argument scoping on methods."""

    # Test unscoped case
    assert MethodClass().scoped_set_foo_bar().get_foo_bar() == (None, None)
    assert MethodClass().unscoped_set_foo_bar().get_foo_bar() == (None, None)

    kwargs = {}
    if foo_arg:
        kwargs.update({"foo": foo_arg})
    if bar_arg:
        kwargs.update({"bar": bar_arg})

    with arg_scope([MethodClass.scoped_set_foo_bar], **kwargs):
        assert MethodClass().scoped_set_foo_bar().get_foo_bar() == (foo_arg, bar_arg)
        assert MethodClass().unscoped_set_foo_bar().get_foo_bar() == (None, None)
        # Test we can still override
        assert MethodClass().scoped_set_foo_bar(foo="dog").get_foo_bar() == (
            "dog",
            bar_arg,
        )
        assert MethodClass().unscoped_set_foo_bar(
            foo="dog", bar="cat"
        ).get_foo_bar() == ("dog", "cat")

    # Test unscoped case again to make sure the previous scoping has reset
    assert MethodClass().scoped_set_foo_bar().get_foo_bar() == (None, None)
    assert MethodClass().unscoped_set_foo_bar().get_foo_bar() == (None, None)


@pytest.mark.parametrize("foo_arg", [None, "spam"])
@pytest.mark.parametrize("bar_arg", [None, "eggs"])
def test_function_scope(foo_arg, bar_arg):
    """Test argument scoping on functions."""

    # Test unscoped state
    assert scoped_func() == (None, None)
    assert unscoped_func() == (None, None)

    kwargs = {}
    if foo_arg:
        kwargs.update({"foo": foo_arg})
    if bar_arg:
        kwargs.update({"bar": bar_arg})

    with arg_scope([scoped_func], **kwargs):
        assert scoped_func() == (foo_arg, bar_arg)
        assert unscoped_func() == (None, None)
        # Test if we can override
        assert scoped_func(foo="dog") == ("dog", bar_arg)
        assert scoped_func(foo="dog", bar="cat") == ("dog", "cat")

    # Test unscoped state again to make sure the previous scope was reset
    assert scoped_func() == (None, None)
    assert unscoped_func() == (None, None)


def test_kwargs():
    """Test if we can pass in arbitrary kwargs, even if not explicitly defined."""

    @add_arg_scope
    def scoped_func_kwarg(**kwargs):
        return kwargs

    expected = {"foo": "spam", "bar": "eggs"}
    with arg_scope([scoped_func_kwarg], **expected):
        assert scoped_func_kwarg() == expected


def test_multiple_scope_targets():
    """Test that we can scope multiple targets with the same kwargs."""

    @add_arg_scope
    def scoped_func_a(foo=None, a=None):
        return foo

    @add_arg_scope
    def scoped_func_b(foo=None, b=None):
        return foo

    with arg_scope([scoped_func_a, scoped_func_b], foo=foo_arg):
        assert scoped_func_a() == foo_arg
        assert scoped_func_b() == foo_arg


def test_unscoped_scope_fail():
    """Test verification if the target has scoping when requested."""
    with pytest.raises(ValueError):
        with arg_scope([unscoped_func], foo=foo_arg):
            pass

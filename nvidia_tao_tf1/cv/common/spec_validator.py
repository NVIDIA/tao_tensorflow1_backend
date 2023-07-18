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
"""Spec validator to validate experiment spec."""

import operator
import sys
from google.protobuf.pyext._message import MessageMapContainer, ScalarMapContainer
import six


def eval_str(s):
    """If s is a string, return the eval results. Else return itself."""

    if isinstance(s, six.string_types):
        if len(s) > 0:
            return eval(s)
        return None
    return s


def length(s):
    """same as len(eval(s))."""
    return len(eval(s))


class ValueChecker:
    """Class to wrap the op and print info for value check."""

    def __init__(self, comp_op, limit, func=None, func_info=""):
        """Init."""
        self.comp_op = comp_op
        self.limit = limit
        self.func = func
        self.func_info = func_info

    def __call__(self):
        """Call."""
        return self.comp_op, self.limit, self.func, self.func_info


def a_in_b(a, b):
    """Same as a in b."""
    return (a in b)


def a_mod_b(a, b):
    """Check if a is divisible by b."""
    return operator.mod(a, b) == 0


operator_dict = {">": operator.gt,
                 "=": operator.eq,
                 "<": operator.lt,
                 "!=": operator.ne,
                 ">=": operator.ge,
                 "<=": operator.le,
                 "in": a_in_b,
                 "%": a_mod_b}


def check_has(value_name, input_value, checker):
    """Function to check if a value is set."""
    comp_op_name, limit, _, _ = checker()
    comp_op = operator_dict[comp_op_name]
    return comp_op(input_value, limit)


def check_value(value_name, input_value, checker_list):
    """Function to check if a value is within the limitation."""
    for checker in checker_list:
        comp_op_name, limit, func, func_info = checker()
        comp_op = operator_dict[comp_op_name]
        if func:
            try:
                value = func(input_value)
            except SyntaxError:
                print("Experiment Spec Setting Error: " +
                      "{} can not be parsed correct. ".format(value_name) +
                      "Wrong value: {}".format(input_value))
                sys.exit(1)
        else:
            value = input_value

        if isinstance(value, list):
            for item in value:
                if isinstance(item, list):
                    new_vc = ValueChecker(comp_op_name, limit)
                    check_value(value_name, item, [new_vc])
                else:
                    if limit == "":
                        error_info = (
                            "Experiment Spec Setting Error: " + func_info +
                            "{} should be set. ".format(value_name))
                    else:
                        error_info = (
                            "Experiment Spec Setting Error: " + func_info +
                            "{} should be {} {}. Wrong value: {}".format(value_name,
                                                                         comp_op_name,
                                                                         limit,
                                                                         item))
                    assert comp_op(item, limit), error_info
        else:
            if limit == "":
                error_info = \
                    ("Experiment Spec Setting Error: " + func_info +
                     "{} should be set.".format(value_name))
            else:
                error_info = \
                    ("Experiment Spec Setting Error: " + func_info +
                     "{} should {} {}. Wrong value: {}".format(value_name,
                                                               comp_op_name,
                                                               limit,
                                                               value))

            assert comp_op(value, limit), error_info


class SpecValidator:
    """Validator for spec check."""

    def __init__(self, required_msg_dict, value_checker_dict, option_checker_dict=None):
        """Init."""
        self.required_msg_dict = required_msg_dict
        self.value_checker_dict = value_checker_dict
        if option_checker_dict is None:
            self.option_checker_dict = {}
        else:
            self.option_checker_dict = option_checker_dict

    def validate(self, spec, required_msg):
        """Recursively validate experiment spec protobuf."""

        def spec_validator(spec, required_msg=None):
            """
            Spec validate function.

            spec: protobuf spec.
            required_msg: The names of the required messages in the spec.
            """

            if required_msg is None:
                required_msg = []
            try:
                for desc in spec.DESCRIPTOR.fields:
                    value = getattr(spec, desc.name)
                    if desc.type == desc.TYPE_MESSAGE:
                        if desc.name in required_msg:
                            if desc.label == desc.LABEL_REPEATED:
                                assert len(value) > 0, \
                                    "{} should be set in experiment spec file.".format(desc.name)
                                # @TODO(tylerz): to skip ScalarMapContainer check
                                # because it is handled by protobuf.
                                if isinstance(value, ScalarMapContainer):
                                    continue
                            else:
                                assert spec.HasField(desc.name), \
                                    "{} should be set in experiment spec file.".format(desc.name)

                        if desc.name in self.required_msg_dict:
                            required_msg_next = self.required_msg_dict[desc.name]
                        else:
                            required_msg_next = []

                        if desc.label == desc.LABEL_REPEATED:
                            # @vpraveen: skipping scalar map containers because
                            # this is handled by protobuf internally.
                            if isinstance(value, ScalarMapContainer):
                                continue
                            if isinstance(value, MessageMapContainer):
                                for item in value:
                                    spec_validator(value[item], required_msg=required_msg_next)
                            else:
                                for item in value:
                                    spec_validator(spec=item, required_msg=required_msg_next)
                        else:
                            # Check if the message exists.
                            if spec.HasField(desc.name):
                                spec_validator(spec=value, required_msg=required_msg_next)
                    else:
                        # If the parameter is optional and it is not set,
                        # then we skip the check_value.
                        if desc.name in self.option_checker_dict:
                            if not check_has(desc.name, value,
                                             self.option_checker_dict[desc.name]):
                                continue

                        if desc.name in self.value_checker_dict:
                            value_checker = self.value_checker_dict[desc.name]
                        else:
                            continue

                        if desc.label == desc.LABEL_REPEATED:
                            for item in value:
                                check_value(desc.name, item, value_checker)
                        else:
                            check_value(desc.name, value, value_checker)
            except AttributeError:
                print("failed for spec: {}, type(spec): {}".format(spec, type(spec)))
                sys.exit(-1)
        spec_validator(spec, required_msg)

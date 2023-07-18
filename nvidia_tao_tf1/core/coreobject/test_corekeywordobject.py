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


import unittest

from nvidia_tao_tf1.core.coreobject import deserialize_tao_object, TAOObject, save_args
from nvidia_tao_tf1.core.coreobject.corekeywordobject import TAOKeywordObject
from nvidia_tao_tf1.core.coreobject.coreobject import _get_kwargs
from nvidia_tao_tf1.core.coreobject.test_data import addressbook_pb2


class TAOAddressBookBuilder(TAOObject):
    @save_args
    def __init__(self, proto_obj, *args, **kwargs):  # pylint: disable=W1113
        super(TAOAddressBookBuilder, self).__init__(*args, **kwargs)
        self.addressBook = proto_obj


class TestTAOKeywordObject(unittest.TestCase):
    def test_corekeywordobject_params(self):
        a = TAOKeywordObject(arg1="1", arg2="2", arg3="3")
        self.assertTrue(hasattr(a, "arg1"))
        self.assertTrue(hasattr(a, "arg2"))
        self.assertTrue(hasattr(a, "arg3"))
        self.assertEqual(a.arg1, "1")
        self.assertEqual(a.arg2, "2")
        self.assertEqual(a.arg3, "3")

    def test_corekeywordobject_serialization(self):
        a = TAOKeywordObject(arg1="1", arg2="2", arg3="3")
        data = a.serialize()
        config = _get_kwargs(data)
        self.assertIn("arg1", config.keys())
        self.assertFalse(
            set(["arg1", "arg2", "arg3"]).difference(set(_get_kwargs(data).keys()))
        )

    def _get_protobuf_for_test(self):
        # Create a protobuf object.
        addressbook = addressbook_pb2.MaglevAddressBook()
        person = addressbook.people.add()
        person.id = 1234
        person.full_name = "John Doe"
        person.email = "jdoe@example.com"
        phone = person.phones.add()
        phone.number = "555-4321"
        phone.type = addressbook_pb2.MaglevPerson.HOME
        return addressbook

    def test_protobuf_serialize(self):
        addressbook = self._get_protobuf_for_test()
        a = TAOAddressBookBuilder(addressbook)
        data = a.serialize()
        config = _get_kwargs(data)
        self.assertIn("proto_obj", config)
        proto_obj = config["proto_obj"]
        config = _get_kwargs(proto_obj)
        self.assertIn("people", config)
        people = config["people"]
        self.assertIsInstance(people, list)

    def test_protobuf_deserialize(self):
        addressbook = self._get_protobuf_for_test()
        a = TAOAddressBookBuilder(addressbook)
        data = a.serialize()
        o = deserialize_tao_object(data)
        self.assertIsInstance(o, TAOAddressBookBuilder)
        self.assertIsInstance(o.addressBook, TAOKeywordObject)
        ab = o.addressBook
        self.assertTrue(ab.people)
        self.assertIsInstance(ab.people[0], TAOKeywordObject)
        person = ab.people[0]
        self.assertEqual(person.full_name, "John Doe")
        self.assertIsInstance(person.phones[0], TAOKeywordObject)
        phone_number = person.phones[0]
        self.assertEqual(phone_number.number, "555-4321")
        self.assertEqual(phone_number.type, addressbook_pb2.MaglevPerson.HOME)

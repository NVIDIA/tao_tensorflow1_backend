import json
import unittest

from parameterized import parameterized

from nvidia_tao_tf1.core.coreobject.protobuf_to_dict.protobuf_to_dict import (
    _CLASS_NAME_KEY,
    _PACKAGE_KEY,
    dict_to_protobuf,
    protobuf_to_dict,
)
from nvidia_tao_tf1.core.coreobject.protobuf_to_dict.tests.sample_pb2 import (
    extDouble,
    extString,
    MessageOfTypes,
    NestedExtension,
)


class Test(unittest.TestCase):
    @parameterized.expand(
        [([True], "default"), ([True], None), ([False], "default"), ([False], None)]
    )
    def test_basics(self, add_class_metadata, overwrite_package_with_name):
        m = self.populate_MessageOfTypes()
        d = protobuf_to_dict(
            m,
            add_class_metadata=add_class_metadata,
            overwrite_package_with_name=overwrite_package_with_name,
        )
        print("overwrite_package_with_name: %s" % overwrite_package_with_name)
        self.compare(
            m,
            d,
            ["nestedRepeated"],
            add_class_metadata=add_class_metadata,
            overwrite_package_with_name=overwrite_package_with_name,
        )

        m2 = dict_to_protobuf(MessageOfTypes, d)
        assert m == m2

    @parameterized.expand(
        [([True], "default"), ([True], None), ([False], "default"), ([False], None)]
    )
    def test_use_enum_labels(self, add_class_metadata, overwrite_package_with_name):
        m = self.populate_MessageOfTypes()
        d = protobuf_to_dict(
            m,
            use_enum_labels=True,
            add_class_metadata=add_class_metadata,
            overwrite_package_with_name=overwrite_package_with_name,
        )

        self.compare(
            m,
            d,
            ["enm", "enmRepeated", "nestedRepeated"],
            add_class_metadata=add_class_metadata,
            overwrite_package_with_name=overwrite_package_with_name,
        )
        assert d["enm"] == "C"
        assert d["enmRepeated"] == ["A", "C"]

        m2 = dict_to_protobuf(MessageOfTypes, d)
        assert m == m2

        d["enm"] = "MEOW"
        with self.assertRaises(KeyError):
            dict_to_protobuf(MessageOfTypes, d)

        d["enm"] = "A"
        d["enmRepeated"] = ["B"]
        dict_to_protobuf(MessageOfTypes, d)

        d["enmRepeated"] = ["CAT"]
        with self.assertRaises(KeyError):
            dict_to_protobuf(MessageOfTypes, d)

    @parameterized.expand(
        [([True], "default"), ([True], None), ([False], "default"), ([False], None)]
    )
    def test_repeated_enum(self, add_class_metadata, overwrite_package_with_name):
        m = self.populate_MessageOfTypes()
        d = protobuf_to_dict(
            m,
            use_enum_labels=True,
            add_class_metadata=add_class_metadata,
            overwrite_package_with_name=overwrite_package_with_name,
        )
        self.compare(
            m,
            d,
            ["enm", "enmRepeated", "nestedRepeated"],
            add_class_metadata=add_class_metadata,
            overwrite_package_with_name=overwrite_package_with_name,
        )
        assert d["enmRepeated"] == ["A", "C"]

        m2 = dict_to_protobuf(MessageOfTypes, d)
        assert m == m2

        d["enmRepeated"] = ["MEOW"]
        with self.assertRaises(KeyError):
            dict_to_protobuf(MessageOfTypes, d)

    @parameterized.expand(
        [([True], "default"), ([True], None), ([False], "default"), ([False], None)]
    )
    def test_nested_repeated(self, add_class_metadata, overwrite_package_with_name):
        m = self.populate_MessageOfTypes()
        m.nestedRepeated.extend(
            [MessageOfTypes.NestedType(req=str(i)) for i in range(10)]
        )

        d = protobuf_to_dict(
            m,
            add_class_metadata=add_class_metadata,
            overwrite_package_with_name=overwrite_package_with_name,
        )
        self.compare(
            m,
            d,
            exclude=["nestedRepeated"],
            add_class_metadata=add_class_metadata,
            overwrite_package_with_name=overwrite_package_with_name,
        )
        if not add_class_metadata:
            assert d["nestedRepeated"] == [{"req": str(i)} for i in range(10)]
        else:
            if overwrite_package_with_name is not None:
                assert d["nestedRepeated"] == [
                    {
                        "req": str(i),
                        _CLASS_NAME_KEY: "NestedType",
                        _PACKAGE_KEY: overwrite_package_with_name,
                    }
                    for i in range(10)
                ]
            else:
                assert d["nestedRepeated"] == [
                    {
                        "req": str(i),
                        _CLASS_NAME_KEY: "NestedType",
                        _PACKAGE_KEY: "tests.MessageOfTypes",
                    }
                    for i in range(10)
                ]

        m2 = dict_to_protobuf(MessageOfTypes, d)
        assert m == m2

    @parameterized.expand(
        [([True], "default"), ([True], None), ([False], "default"), ([False], None)]
    )
    def test_reverse(self, add_class_metadata, overwrite_package_with_name):
        m = self.populate_MessageOfTypes()
        m2 = dict_to_protobuf(
            MessageOfTypes,
            protobuf_to_dict(
                m,
                add_class_metadata=add_class_metadata,
                overwrite_package_with_name=overwrite_package_with_name,
            ),
        )
        assert m == m2
        m2.dubl = 0
        assert m2 != m

    @parameterized.expand(
        [([True], "default"), ([True], None), ([False], "default"), ([False], None)]
    )
    def test_incomplete(self, add_class_metadata, overwrite_package_with_name):
        m = self.populate_MessageOfTypes()
        d = protobuf_to_dict(
            m,
            add_class_metadata=add_class_metadata,
            overwrite_package_with_name=overwrite_package_with_name,
        )
        d.pop("dubl")
        m2 = dict_to_protobuf(MessageOfTypes, d)
        assert m2.dubl == 0
        assert m != m2

    @parameterized.expand(
        [([True], "default"), ([True], None), ([False], "default"), ([False], None)]
    )
    def test_pass_instance(self, add_class_metadata, overwrite_package_with_name):
        m = self.populate_MessageOfTypes()
        d = protobuf_to_dict(
            m,
            add_class_metadata=add_class_metadata,
            overwrite_package_with_name=overwrite_package_with_name,
        )
        d["dubl"] = 1
        m2 = dict_to_protobuf(m, d)
        assert m is m2
        assert m.dubl == 1

    @parameterized.expand(
        [([True], "default"), ([True], None), ([False], "default"), ([False], None)]
    )
    def test_strict(self, add_class_metadata, overwrite_package_with_name):
        m = self.populate_MessageOfTypes()
        d = protobuf_to_dict(
            m,
            add_class_metadata=add_class_metadata,
            overwrite_package_with_name=overwrite_package_with_name,
        )
        d["meow"] = 1
        with self.assertRaises(KeyError):
            m2 = dict_to_protobuf(MessageOfTypes, d)
        m2 = dict_to_protobuf(MessageOfTypes, d, strict=False)
        assert m == m2

    def populate_MessageOfTypes(self):
        m = MessageOfTypes()
        m.dubl = 1.7e308
        m.flot = 3.4e038
        m.i32 = 2 ** 31 - 1  # 2147483647 #
        m.i64 = 2 ** 63 - 1  # 0x7FFFFFFFFFFFFFFF
        m.ui32 = 2 ** 32 - 1
        m.ui64 = 2 ** 64 - 1
        m.si32 = -1 * m.i32
        m.si64 = -1 * m.i64
        m.f32 = m.i32
        m.f64 = m.i64
        m.sf32 = m.si32
        m.sf64 = m.si64
        m.bol = True
        m.strng = "string"
        m.byts = b"\n\x14\x1e"
        assert len(m.byts) == 3, len(m.byts)
        m.nested.req = "req"
        m.enm = MessageOfTypes.C  # @UndefinedVariable
        m.enmRepeated.extend([MessageOfTypes.A, MessageOfTypes.C])
        m.range.extend(range(10))
        return m

    def compare(
        self,
        m,
        d,
        exclude=None,
        add_class_metadata=False,
        overwrite_package_with_name=None,
    ):
        i = 0
        exclude = ["byts", "nested", _CLASS_NAME_KEY] + (exclude or [])
        for i, field in enumerate(
            MessageOfTypes.DESCRIPTOR.fields
        ):  # @UndefinedVariable
            if field.name not in exclude:
                assert field.name in d, field.name
                assert d[field.name] == getattr(m, field.name), (
                    field.name,
                    d[field.name],
                )
        assert i > 0
        assert len(m.byts) == 3, len(m.bytes)
        print(d["nested"])
        if add_class_metadata:
            if overwrite_package_with_name is not None:
                assert d["nested"] == {
                    "req": m.nested.req,
                    _CLASS_NAME_KEY: "NestedType",
                    _PACKAGE_KEY: overwrite_package_with_name,
                }
            else:
                assert d["nested"] == {
                    "req": m.nested.req,
                    _CLASS_NAME_KEY: "NestedType",
                    _PACKAGE_KEY: "tests.MessageOfTypes",
                }
        else:
            assert d["nested"] == {"req": m.nested.req}

    @parameterized.expand(
        [([True], "default"), ([True], None), ([False], "default"), ([False], None)]
    )
    def test_extensions(self, add_class_metadata, overwrite_package_with_name):
        m = MessageOfTypes()

        primitives = {extDouble: 123.4, extString: "string", NestedExtension.extInt: 4}

        for key, value in primitives.items():
            m.Extensions[key] = value
        m.Extensions[NestedExtension.extNested].req = "nested"

        # Confirm compatibility with JSON serialization
        res = json.loads(
            json.dumps(
                protobuf_to_dict(
                    m,
                    add_class_metadata=add_class_metadata,
                    overwrite_package_with_name=overwrite_package_with_name,
                )
            )
        )
        assert "___X" in res
        exts = res["___X"]
        assert set(exts.keys()) == {
            str(f.number) for f, _ in m.ListFields() if f.is_extension
        }
        for key, value in primitives.items():
            assert exts[str(key.number)] == value
        assert exts[str(NestedExtension.extNested.number)]["req"] == "nested"

        deser = dict_to_protobuf(MessageOfTypes, res)
        assert deser
        for key, value in primitives.items():
            assert deser.Extensions[key] == m.Extensions[key]
        req1 = deser.Extensions[NestedExtension.extNested].req
        req2 = m.Extensions[NestedExtension.extNested].req
        assert req1 == req2

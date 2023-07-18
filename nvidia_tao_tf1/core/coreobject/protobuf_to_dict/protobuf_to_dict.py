"""Class which converts protobuf objects to python dicts."""
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message
import six


EXTENSION_CONTAINER = "___X"
_CLASS_NAME_KEY = "__class_name__"
_PACKAGE_KEY = "__name_space__"
_DEFAULT_PACKAGE_NAME = "default"
_PROTOBUF_KEY = "proto_obj"

_IGNORE_FIELDS = {EXTENSION_CONTAINER, _PACKAGE_KEY, _CLASS_NAME_KEY}


TYPE_CALLABLE_MAP = {
    FieldDescriptor.TYPE_DOUBLE: float,
    FieldDescriptor.TYPE_FLOAT: float,
    FieldDescriptor.TYPE_INT32: int,
    FieldDescriptor.TYPE_INT64: int if six.PY3 else six.integer_types[1],
    FieldDescriptor.TYPE_UINT32: int,
    FieldDescriptor.TYPE_UINT64: int if six.PY3 else six.integer_types[1],
    FieldDescriptor.TYPE_SINT32: int,
    FieldDescriptor.TYPE_SINT64: int if six.PY3 else six.integer_types[1],
    FieldDescriptor.TYPE_FIXED32: int,
    FieldDescriptor.TYPE_FIXED64: int if six.PY3 else six.integer_types[1],
    FieldDescriptor.TYPE_SFIXED32: int,
    FieldDescriptor.TYPE_SFIXED64: int if six.PY3 else six.integer_types[1],
    FieldDescriptor.TYPE_BOOL: bool,
    FieldDescriptor.TYPE_STRING: six.text_type,
    FieldDescriptor.TYPE_BYTES: six.binary_type,
    FieldDescriptor.TYPE_ENUM: int,
}


def _repeated(type_callable):
    return lambda value_list: [type_callable(value) for value in value_list]


def _enum_label_name(field, value):
    return field.enum_type.values_by_number[int(value)].name


def protobuf_to_modulus_dict(
    pb,
    use_enum_labels=False,
    add_class_metadata=False,
    overwrite_package_with_name=None,
    overwrite_class_with_name=None,
):
    """Recursively populate a dictionary with a protobuf object.

    Args:
        pb (google.protobuf.Message): a protobuf message class, or an protobuf instance
        use_enum_labels (bool): True if enums should be represented as their string labels,
            False for their ordinal value.
        add_class_metadata (bool): True to add class names and package names to the dictionary.
        overwrite_package_with_name (string): If set, will use the value as the package name
            for all objects. Only used if add_class_metadata is True.
        overwrite_class_with_name (string): If set, will use the value as the class name
            for all objects. Only used if add_class_metadata is True.
    """
    type_callable_map = TYPE_CALLABLE_MAP
    result_dict = {}
    extensions = {}
    if add_class_metadata:
        result_dict[_CLASS_NAME_KEY] = pb.DESCRIPTOR.name
        if overwrite_class_with_name:
            result_dict[_CLASS_NAME_KEY] = overwrite_class_with_name
        # Overwrite the package name if we have a value to overwrite with.
        if overwrite_package_with_name is not None:
            result_dict[_PACKAGE_KEY] = overwrite_package_with_name

        else:
            result_dict[_PACKAGE_KEY] = pb.DESCRIPTOR.full_name.rpartition(".")[0]

    # Hack to get around the problem of ListFields not returning default fields (see
    # https://github.com/universe-proton/universe-topology/issues/1). Hack = use
    # pb.DESCRIPTOR.fields_by_name.items() instead of pb.ListFields().
    for field_name, field in pb.DESCRIPTOR.fields_by_name.items():
        value = getattr(pb, field_name)
        type_callable = _get_field_value_adaptor(
            pb,
            field,
            type_callable_map,
            use_enum_labels,
            add_class_metadata=add_class_metadata,
            overwrite_package_with_name=overwrite_package_with_name,
            message_type_func=protobuf_to_modulus_dict,
            overwrite_class_with_name=overwrite_class_with_name,
        )
        if field.label == FieldDescriptor.LABEL_REPEATED:
            type_callable = _repeated(type_callable)

        if field.is_extension:
            extensions[str(field.number)] = type_callable(value)
            continue

        result_dict[field.name] = type_callable(value)

    if extensions:
        result_dict[EXTENSION_CONTAINER] = extensions
    return result_dict


def protobuf_to_dict(
    pb,
    use_enum_labels=False,
    add_class_metadata=False,
    overwrite_package_with_name=None,
    overwrite_class_with_name=None,
):
    """Recursively populate a dictionary with a protobuf object.

    Args:
        pb (google.protobuf.Message): a protobuf message class, or an protobuf instance
        use_enum_labels (bool): True if enums should be represented as their string labels,
            False for their ordinal value.
        add_class_metadata (bool): True to add class names and package names to the dictionary.
        overwrite_package_with_name (string): If set, will use the value as the package name
            for all objects. Only used if add_class_metadata is True.
    """
    type_callable_map = TYPE_CALLABLE_MAP
    result_dict = {}
    extensions = {}
    if add_class_metadata:
        result_dict[_CLASS_NAME_KEY] = pb.DESCRIPTOR.name
        if overwrite_class_with_name:
            result_dict[_CLASS_NAME_KEY] = overwrite_class_with_name
        # Overwrite the package name if we have a value to overwrite with.
        if overwrite_package_with_name is not None:
            result_dict[_PACKAGE_KEY] = overwrite_package_with_name
        else:
            result_dict[_PACKAGE_KEY] = pb.DESCRIPTOR.full_name.rpartition(".")[0]

    for field, value in pb.ListFields():
        type_callable = _get_field_value_adaptor(
            pb,
            field,
            type_callable_map,
            use_enum_labels,
            add_class_metadata=add_class_metadata,
            overwrite_package_with_name=overwrite_package_with_name,
        )
        if field.label == FieldDescriptor.LABEL_REPEATED:
            type_callable = _repeated(type_callable)

        if field.is_extension:
            extensions[str(field.number)] = type_callable(value)
            continue

        result_dict[field.name] = type_callable(value)

    if extensions:
        result_dict[EXTENSION_CONTAINER] = extensions

    return result_dict


def _get_field_value_adaptor(
    pb,
    field,
    type_callable_map,
    use_enum_labels=False,
    add_class_metadata=False,
    overwrite_package_with_name=None,
    message_type_func=protobuf_to_dict,
    overwrite_class_with_name=None,
):
    if field.type == FieldDescriptor.TYPE_MESSAGE:
        # recursively encode protobuf sub-message
        return lambda pb: message_type_func(
            pb,
            use_enum_labels=use_enum_labels,
            add_class_metadata=add_class_metadata,
            overwrite_package_with_name=overwrite_package_with_name,
            overwrite_class_with_name=overwrite_class_with_name,
        )

    if use_enum_labels and field.type == FieldDescriptor.TYPE_ENUM:
        return lambda value: _enum_label_name(field, value)

    if field.type in type_callable_map:
        return type_callable_map[field.type]

    raise TypeError(
        "Field %s.%s has unrecognised type id %d"
        % (pb.__class__.__name__, field.name, field.type)
    )


REVERSE_TYPE_CALLABLE_MAP = {}


def dict_to_protobuf(pb_klass_or_instance, values, strict=True):
    """Populates a protobuf model from a dictionary.

    Args:
        pb_klass_or_instance (google.protobuf.Message): a protobuf message class, or
            an protobuf instance
        pb_klass_or_instance (google.prorobuf.Message): a type or instance of a subclass
            of google.protobuf.message.Message
        values (dict): a dictionary of values. Repeated and nested values are
            fully supported.
        type_callable_map (dict): a mapping of protobuf types to callables for setting
            values on the target instance.
        strict (bool): complain if keys in the map are not fields on the message.
    """
    if isinstance(pb_klass_or_instance, Message):
        instance = pb_klass_or_instance
    else:
        instance = pb_klass_or_instance()
    return _dict_to_protobuf(instance, values, REVERSE_TYPE_CALLABLE_MAP, strict)


def _get_field_mapping(pb, dict_value, strict):
    field_mapping = []
    for key, value in dict_value.items():
        if key in _IGNORE_FIELDS:
            continue
        if key not in pb.DESCRIPTOR.fields_by_name:
            if strict:
                raise KeyError("%s does not have a field called %s" % (pb, key))
            continue
        field_mapping.append(
            (pb.DESCRIPTOR.fields_by_name[key], value, getattr(pb, key, None))
        )

    for ext_num, ext_val in dict_value.get(EXTENSION_CONTAINER, {}).items():
        try:
            ext_num = int(ext_num)
        except ValueError:
            raise ValueError("Extension keys must be integers.")
        if ext_num not in pb._extensions_by_number:
            if strict:
                raise KeyError(
                    "%s does not have a extension with number %s. Perhaps you forgot to import it?"
                    % (pb, ext_num)
                )
            continue
        ext_field = pb._extensions_by_number[ext_num]
        pb_val = None
        pb_val = pb.Extensions[ext_field]
        field_mapping.append((ext_field, ext_val, pb_val))

    return field_mapping


def _dict_to_protobuf(pb, value, type_callable_map, strict):
    fields = _get_field_mapping(pb, value, strict)

    for field, input_value, pb_value in fields:
        if field.label == FieldDescriptor.LABEL_REPEATED:
            for item in input_value:
                if field.type == FieldDescriptor.TYPE_MESSAGE:
                    m = pb_value.add()
                    _dict_to_protobuf(m, item, type_callable_map, strict)
                elif field.type == FieldDescriptor.TYPE_ENUM and isinstance(item, str):
                    pb_value.append(_string_to_enum(field, item))
                else:
                    pb_value.append(item)
            continue
        if field.type == FieldDescriptor.TYPE_MESSAGE:
            _dict_to_protobuf(pb_value, input_value, type_callable_map, strict)
            continue

        if field.type in type_callable_map:
            input_value = type_callable_map[field.type](input_value)

        if field.is_extension:
            pb.Extensions[field] = input_value
            continue

        if field.type == FieldDescriptor.TYPE_ENUM and isinstance(input_value, str):
            input_value = _string_to_enum(field, input_value)

        setattr(pb, field.name, input_value)

    return pb


def _string_to_enum(field, input_value):
    enum_dict = field.enum_type.values_by_name
    try:
        input_value = enum_dict[input_value].number
    except KeyError:
        raise KeyError(
            "`%s` is not a valid value for field `%s`" % (input_value, field.name)
        )
    return input_value

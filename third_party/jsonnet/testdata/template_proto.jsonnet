local data = import "third_party/jsonnet/testdata/import.proto.txt";

{
    "foo": data.bar,
}

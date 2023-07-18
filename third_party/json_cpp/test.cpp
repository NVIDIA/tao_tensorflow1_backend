// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

#include <json.hpp>
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include "gtest/gtest.h"

using json = nlohmann::json;

//Run with --test_output=all to see standard output:
//dazel test //third_party/json_cpp:json_compile_test --test_output=all

std::string global_testfile_path= "";

namespace {

class jsonTest : public ::testing::Test {
    protected:
        jsonTest() {}
        virtual ~jsonTest() {}
        virtual void SetUp() {}
        virtual void TearDown() {}

};

TEST(dwVideoTest, ParseJSONFromFile) {
    std::ifstream json_test_file (global_testfile_path);
    std::string json_test_string
         {std::istreambuf_iterator<char> (json_test_file), std::istreambuf_iterator<char>()};

    ASSERT_TRUE(json_test_string.length()!=0) << "Failed to read test file!";

    std::cout << "Successfully found JSON file. Contents:" << std::endl;
    std::cout << json_test_string << std::endl;
    json j_testfile = json::parse(json_test_string);

    //Check that JSON object contains correct field values:
    ASSERT_TRUE(j_testfile["field_1"] ==  "test_result_1");
    ASSERT_TRUE(j_testfile["field_2"]["subfield_1"] == "subfield_1_result");
    ASSERT_FLOAT_EQ(j_testfile["field_2"]["subfield_2"], 1.234);
}
} // namespace

int main(int argc, char **argv) {
    //Note! We are using a workaround to fine the test file test_file.json
    //by passing its location as a parameter to this test code. This is
    //specified in the BUILD file. argv[1] contains test_file.json path:
    global_testfile_path = std::string(argv[1]);

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

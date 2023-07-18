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

import os
from tempfile import TemporaryFile
import pytest

from nvidia_tao_tf1.encoding import encoding

test_byte_key = b"Valid-Byte-Object-Key"
other_byte_key = b"Different-Valid-Byte-Object-Key"
test_file_size = 2001

_CHUNK_SIZE = 1024  # bytes
__NONCE_SIZE = 16   # bytes -- 128 bit


# encode function error testing
@pytest.mark.parametrize("encoding_file_size, encoding_password, \
                         expected_result",
                         [(test_file_size, 123, TypeError),
                          (test_file_size, 123.456, TypeError),
                          (test_file_size, [], TypeError),
                          (test_file_size, True, TypeError)])
def test_encode_failures(encoding_file_size, encoding_password,
                         expected_result):
    with pytest.raises(expected_result):
        with TemporaryFile() as tmp_random_test_file:
            with TemporaryFile() as tmp_encoding_target_file:
                tmp_random_test_file.write(os.urandom(encoding_file_size))
                tmp_random_test_file.seek(0)
                encoding.encode(tmp_random_test_file, tmp_encoding_target_file,
                                encoding_password)


# decode function error testing
@pytest.mark.parametrize("encoding_file_size, encoding_password, \
                         expected_result",
                         [(test_file_size, 123, TypeError),
                          (test_file_size, 123.456, TypeError),
                          (test_file_size, [], TypeError),
                          (test_file_size, True, TypeError),
                          (0, test_byte_key, ValueError),
                          (__NONCE_SIZE-1, test_byte_key, ValueError)])
def test_decode_failures(encoding_file_size, encoding_password,
                         expected_result):
    with pytest.raises(expected_result):
        with TemporaryFile() as temp_encoded_file:
            with TemporaryFile() as tmp_decoding_target_file:
                temp_encoded_file.write(os.urandom(encoding_file_size))
                temp_encoded_file.seek(0)
                encoding.decode(temp_encoded_file, tmp_decoding_target_file,
                                encoding_password)


# Validation testing
@pytest.mark.parametrize("encoding_file_size, encoding_password",
                         # No payload, encoded file is only the nonce value
                         [(0, test_byte_key),
                          # 1 partial iteration
                          (_CHUNK_SIZE-1, test_byte_key),
                          # 2 complete iterations, no partial chunk
                          ((_CHUNK_SIZE*2), test_byte_key),
                          # 3 iterations, including a partial 3rd chunk
                          ((_CHUNK_SIZE*2)+1, test_byte_key)])
def test_validate_encode_decode_of_random_file(encoding_file_size,
                                               encoding_password):
    with TemporaryFile() as tmp_random_test_file,\
         TemporaryFile() as tmp_encoding_target_file:
        with TemporaryFile() as tmp_decoding_target_file,\
             TemporaryFile() as bad_key_target_file:
            random_payload = os.urandom(encoding_file_size)
            tmp_random_test_file.write(random_payload)
            tmp_random_test_file.seek(0)
            encoding.encode(tmp_random_test_file, tmp_encoding_target_file,
                            encoding_password)
            tmp_encoding_target_file.seek(0)
            encoding.decode(tmp_encoding_target_file, tmp_decoding_target_file,
                            encoding_password)
            tmp_encoding_target_file.seek(0)
            tmp_decoding_target_file.seek(0)

            assert tmp_decoding_target_file.read() == random_payload

            if encoding_file_size > 0:
                encoding.decode(tmp_encoding_target_file, bad_key_target_file,
                                other_byte_key)
                bad_key_target_file.seek(0)

                assert bad_key_target_file.read() != random_payload

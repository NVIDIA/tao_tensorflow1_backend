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

"""TAO Toolkit file encoding APIs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from eff_tao_encryption.tao_codec import encrypt_stream, decrypt_stream


def encode(input_stream, output_stream, encode_key):
    """Encode a stream using a byte object key as the encoding 'password'.

    Derives the 256 bit key using PBKDF2 password stretching and encodes using
    AES in CTR mode 1024 bytes at a time. Includes the raw initialization value
    -- 'nonce' -- as the first item in the encoded ciphertext.

    NOTE: this function does not close the input or output stream.
    The stream is duck-typed, as long as the object can be read N bits at a
    time via read(N) and written to using write(BYTES_OBJECT).

    Args:
        input_stream (open readable binary io stream): Input stream, typically
            an open file. Data read from this stream is encoded and written to
            the output_stream.
        output_stream (open writable binary io stream): Writable output stream
            where the encoded data of the input_file is written to.
        encode_key (bytes): byte text representing the encoding password.
            b"yourkey"

    Returns:
        Nothing.

    Raises:
        TypeError: if the encode_key is not a byte object

    """
    # Validate encode_key input and then derive the key.
    if not isinstance(encode_key, bytes):
        try:
            # if this is str, let's try to encode it into bytes.
            encode_key = str.encode(encode_key)
        except Exception:
            raise TypeError("encode_key must be passed as a byte object")

    encrypt_stream(
        input_stream, output_stream,
        encode_key, encryption=True, rewind=False
    )


def decode(input_stream, output_stream, encode_key):
    """Decode a stream using byte object key as the decoding 'password'.

    Derives the 256 bit key using PBKDF2 password stretching and decodes
    a stream encoded using AES in CTR mode by the above encode function.
    Processes 1024 bytes at a time and uses the 'nonce' value included at
    the beginning of the cipher text input stream.

    NOTE: This function does not delete the encoded cipher text.

    Args:
        input_stream (open readable binary io stream): Encoded input stream,
            typically an open file. Data read from this stream is decoded and
            written to the output_stream.
        output_stream (open writable binary io stream): Writable output stream
            where the decoded data of the input_file is written to.
        encode_key (bytes): byte text representing the encoding password.
            b"yourkey".

    Returns:
        Nothing.

    Raises:
        TypeError: if the encode_key is not a byte object
        ValueError: if a valid nonce value can't be read from the given file.

    """
    # Validate encode_key input and derive the key.
    if not isinstance(encode_key, bytes):
        try:
            # if this is str, let's try to encode it into bytes.
            encode_key = str.encode(encode_key)
        except Exception:
            raise TypeError("encode_key must be passed as a byte object")

    decrypt_stream(
        input_stream, output_stream,
        encode_key, encryption=True, rewind=False
    )

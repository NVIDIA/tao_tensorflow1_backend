// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.

#include "image_loader_function.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "turbojpeg.h"

/*
 * @brief transpose a hwc buffer to a chw buffer.
 *
 *
 * @param src_buffer: the source buffer to transpose.
 * @param des_buffer: the buffer to keep the transposed data.
 * @param height: the image height.
 * @param width: the image width.
 * @param channel: the image channel.
 */
template <typename T>
void transpose_hwc_to_chw(T* src_buffer, T* des_buffer, size_t height, size_t width,
                          size_t channel) {
    if (src_buffer == nullptr) {
        throw std::invalid_argument("src_buffer is null");
    }

    if (des_buffer == nullptr) {
        throw std::invalid_argument("des_buffer is null");
    }

    for (size_t c = 0; c < channel; c++) {
        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                size_t hwc_pos = channel * width * h + channel * w + c;
                size_t chw_pos = c * height * width + h * width + w;
                des_buffer[chw_pos] = src_buffer[hwc_pos];
            }
        }
    }
}

/*
 * @brief read a single image from a file and store it in a buffer.
 *
 * Note: All filesystem related operations are put here.
 *
 * @param path: the image path.
 * @param buffer: the pointer to a buffer that is supposed to keep the loaded image.
        if the buffer is nullptr, allocate the buffer.
 * @param length: the length of the image, -1 to bypass image length check.
 * @param verbose: verbose logging mode.
 *
 * @return image_size: the image size.
 */
size_t read_image(const std::string& path, char** buffer, const int length, bool verbose) {
    std::fstream image_file(path, std::ios::in | std::ios::binary);

    if (!image_file.is_open()) {
        // TODO(weich): It seems that the exception thrown by std::invalid_argument() can't be
        // caught by tf.test.TestCase. Needs to investigate.
        throw std::invalid_argument("File " + path + "open failed " + strerror(errno));
    }

    int image_size;

    if (!image_file.seekg(0, std::ios::end) || ((image_size = image_file.tellg()) < 0) ||
        !image_file.seekg(0, std::ios::beg)) {
        throw std::invalid_argument("Cannot determine the size of file: " + path);
    }

    // Note, for image that is decoded (e.g, jpeg), length of the image file is not
    // the same as the size of the image after being decoded. We bypass length check for
    // decoded images.
    if (length != -1) {
        if (image_size != length) {
            throw std::invalid_argument("The image size " + std::to_string(image_size) +
                                        "is less than the length to read " +
                                        std::to_string(length) + "for image " + path);
        }
    }

    // Note, for image that is decoded, the buffer is allocated until
    // the length of the image file is known. The caller should
    // handle the release of the buffer.
    if (*buffer == nullptr) {
        *buffer = new char[image_size];

        if (*buffer == nullptr) {
            throw std::invalid_argument("Memory allocation fails for image size " +
                                        std::to_string(image_size));
        }
    }

    if (!image_file.read(*buffer, image_size)) {
        delete buffer;
        throw std::invalid_argument("File reads " + std::to_string(image_file.gcount()) +
                                    " bytes was not as expected as " + std::to_string(image_size) +
                                    " for image " + path + strerror(errno));
    }

    image_file.close();

    return image_size;
}

void fp16_image_loader(const std::string& path, char* buffer, const size_t length, bool verbose) {
    // For fp16, just read it as a binary file.
    read_image(path, &buffer, length, verbose);
}

void jpeg_image_loader(const std::string& path, char* buffer, const size_t length, bool verbose) {
    // For jpeg, first read the image file, then decode and transpose it.
    char* image_buffer = nullptr;
    size_t jpeg_size = read_image(path, &image_buffer, -1, verbose);

    std::unique_ptr<uint8_t[]> jpeg_buf(reinterpret_cast<uint8_t*>(image_buffer));

    tjhandle tj_instance = nullptr;
    if ((tj_instance = tjInitDecompress()) == nullptr) {
        throw std::invalid_argument("Error initializing jpeg decompressor.");
    }

    int in_subsamp, in_colorspace, width, height;
    if (tjDecompressHeader3(tj_instance, jpeg_buf.get(), jpeg_size, &width, &height, &in_subsamp,
                            &in_colorspace) < 0) {
        tjDestroy(tj_instance);
        throw std::invalid_argument("Error reading JPEG header of file " + path);
    }

    int channel = tjPixelSize[TJPF_RGB];

    size_t jpeg_image_size = width * height * channel;

    // For jpeg sizeof(uint8_t) = 1
    if (jpeg_image_size != length) {
        tjDestroy(tj_instance);
        throw std::invalid_argument("Error jpeg image size" + std::to_string(length) +
                                    "is not expected" + path);
    }

    // The reason to have another buffer is to transpose it from hwc to chw.
    std::unique_ptr<uint8_t[]> transpose_buffer = std::make_unique<uint8_t[]>(jpeg_image_size);

    if (tjDecompress2(tj_instance, jpeg_buf.get(), jpeg_size, transpose_buffer.get(), width, 0,
                      height, TJPF_RGB, TJFLAG_FASTUPSAMPLE | TJFLAG_FASTDCT) < 0) {
        tjDestroy(tj_instance);
        throw std::invalid_argument("Error decompressing JPEG image file " + path);
    }

    // Cleanup
    if (tj_instance) {
        // TODO(weich): this might not be cleaned up correctly if errors are thrown.
        tjDestroy(tj_instance);
    }

    // Transpose the HWC to CHW and copy to buffer.
    transpose_hwc_to_chw(transpose_buffer.get(), reinterpret_cast<uint8_t*>(buffer), height, width,
                         channel);
}

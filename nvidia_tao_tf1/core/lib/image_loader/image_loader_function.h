// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.

#ifndef _IMAGE_LOADER_FUNCTION_H_
#define _IMAGE_LOADER_FUNCTION_H_

#include <string>
/*
 * @brief Image loading function for fp16 image.
 *
 * @param path: the image path.
 * @param buffer: the buffer to keep the loaded image, usually the returned tensor
 * (tensor->flat<>().data()).
 * @param length: the length of the image.
 * @param verbose: verbose logging mode.
 */
void fp16_image_loader(const std::string& path, char* buffer, const size_t length, bool verbose);

/*
 * @brief Image loading function for jpeg.
 *
 * @param path: the image path.
 * @param buffer: the buffer to keep the loaded image, usually the returned tensor
 * (tensor->flat<>().data()).
 * @param length: the length of the image.
 * @param verbose: verbose logging mode.
 */
void jpeg_image_loader(const std::string& path, char* buffer, const size_t length, bool verbose);

#endif

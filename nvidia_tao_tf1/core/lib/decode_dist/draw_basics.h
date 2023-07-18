// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

#ifndef DRAW_BASICS_H_
#define DRAW_BASICS_H_

#include <algorithm>
#include <climits>
#include <cmath>
#include <map>
#include <vector>

#include <cfloat>
#include <fstream>
#include <iostream>
#include <string>
#include "nvidia_tao_tf1/core/lib/decode_dist/color.h"
#include "nvidia_tao_tf1/core/lib/generate_lineseg_from_polygon/point.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __device__
#else
#define CUDA_HOSTDEV
#endif

CUDA_HOSTDEV inline void set_dots(int* output, const int h, const int w, const int x, const int y,
                                  const int radius, const int r, const int g, const int b) {
    // this will be ugly but doesn't matter.
    for (int dx = -radius; dx <= radius; dx++) {
        for (int dy = -radius; dy <= radius; dy++) {
            if (dx * dx + dy * dy <= radius * radius) {
                int yp = y + dy;
                int xp = x + dx;
                if (yp >= 0 && yp < h && xp >= 0 && xp < w) {
#ifdef __CUDACC__
                    atomicExch(&output[yp * w * 3 + xp * 3], r);
                    atomicExch(&output[yp * w * 3 + xp * 3 + 1], g);
                    atomicExch(&output[yp * w * 3 + xp * 3 + 2], b);
#else
                    output[yp * w * 3 + xp * 3] = r;
                    output[yp * w * 3 + xp * 3 + 1] = g;
                    output[yp * w * 3 + xp * 3 + 2] = b;
#endif
                }
            }
        }
    }
}

// Bresenham Algorithm
CUDA_HOSTDEV inline void draw_line(int* output, const int h, const int w, const float input_x1,
                                   const float input_y1, const float input_x2, const float input_y2,
                                   int radius, const int r, const int g, const int b) {
    int x1, x2, y1, y2;
#ifdef __CUDACC__
    x1 = static_cast<int>(floorf(0.5f + input_x1));
    x2 = static_cast<int>(floorf(0.5f + input_x2));
    y1 = static_cast<int>(floorf(0.5f + input_y1));
    y2 = static_cast<int>(floorf(0.5f + input_y2));
#else
    x1 = std::floor(0.5f + input_x1);
    x2 = std::floor(0.5f + input_x2);
    y1 = std::floor(0.5f + input_y1);
    y2 = std::floor(0.5f + input_y2);
#endif
    int x, y;
    int dx, dy;
    int incx, incy;
    int balance;

    if (x2 >= x1) {
        dx = x2 - x1;
        incx = 1;
    } else {
        dx = x1 - x2;
        incx = -1;
    }

    if (y2 >= y1) {
        dy = y2 - y1;
        incy = 1;
    } else {
        dy = y1 - y2;
        incy = -1;
    }
    x = x1;
    y = y1;

    if (dx >= dy) {
        dy <<= 1;
        balance = dy - dx;
        dx <<= 1;

        while (x != x2) {
            set_dots(output, h, w, x, y, radius, r, g, b);
            if (balance >= 0) {
                y += incy;
                balance -= dx;
            }
            balance += dy;
            x += incx;
        }
        set_dots(output, h, w, x, y, radius, r, g, b);
    } else {
        dx <<= 1;
        balance = dx - dy;
        dy <<= 1;

        while (y != y2) {
            set_dots(output, h, w, x, y, radius, r, g, b);
            if (balance >= 0) {
                x += incx;
                balance -= dy;
            }
            balance += dx;
            y += incy;
        }
        set_dots(output, h, w, x, y, radius, r, g, b);
    }
}

// Bresenham Algorithm
CUDA_HOSTDEV inline void draw_line(int* output, const int h, const int w, const float input_x1,
                                   const float input_y1, const float input_x2, const float input_y2,
                                   int radius, const Color& color) {
    draw_line(output, h, w, input_x1, input_y1, input_x2, input_y2, radius, color.r_, color.g_,
              color.b_);
}

CUDA_HOSTDEV inline void set_dots(int* output, const int h, const int w, const Point<float> pt,
                                  const int radius, const Color& color) {
    int x = static_cast<int>(pt.x_);
    int y = static_cast<int>(pt.y_);
    set_dots(output, h, w, x, y, radius, color.r_, color.g_, color.b_);
}

CUDA_HOSTDEV inline void set_dots(int* output, const int h, const int w, const Point<float> pt,
                                  const int radius, const int r, const int g, const int b) {
    int x = static_cast<int>(pt.x_);
    int y = static_cast<int>(pt.y_);
    set_dots(output, h, w, x, y, radius, r, g, b);
}

CUDA_HOSTDEV inline void draw_line(int* output, const int h, const int w,
                                   const Point<float>& input1, const Point<float>& input2,
                                   const int radius, const Color& color) {
    int input_x1 = std::floor(0.5f + input1.x_);
    int input_x2 = std::floor(0.5f + input2.x_);
    int input_y1 = std::floor(0.5f + input1.y_);
    int input_y2 = std::floor(0.5f + input2.y_);
    draw_line(output, h, w, input_x1, input_y1, input_x2, input_y2, radius, color);
}

CUDA_HOSTDEV inline void draw_line(int* output, const int h, const int w,
                                   const Point<float>& input1, const Point<float>& input2,
                                   const int radius, const int r, const int g, const int b) {
    int input_x1 = std::floor(0.5f + input1.x_);
    int input_x2 = std::floor(0.5f + input2.x_);
    int input_y1 = std::floor(0.5f + input1.y_);
    int input_y2 = std::floor(0.5f + input2.y_);
    draw_line(output, h, w, input_x1, input_y1, input_x2, input_y2, radius, r, g, b);
}

#endif /* DRAW_BASICS_H_ */

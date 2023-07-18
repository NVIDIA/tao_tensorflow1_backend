// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

#ifndef DRAW_CHARACTERS_H_
#define DRAW_CHARACTERS_H_

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
#include "nvidia_tao_tf1/core/lib/decode_dist/draw_basics.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __device__
#else
#define CUDA_HOSTDEV
#endif

class DrawCharacters {
 public:
    static constexpr int W = 10;
    static constexpr int H = 20;
    static constexpr int CHAR_MIN_SPACE = 4;
    DrawCharacters() {
        base_vertex_[0] = Point<float>(0, 0);
        base_vertex_[1] = Point<float>(W, 0);
        base_vertex_[2] = Point<float>(W, W);
        base_vertex_[3] = Point<float>(W, H);
        base_vertex_[4] = Point<float>(0, H);
        base_vertex_[5] = Point<float>(0, W);
        // 0 ----- 1
        // |       |
        // 5 ----- 2
        // |       |
        // 4 ----- 3
    }

    float getSpace(const float scale) { return CHAR_MIN_SPACE * scale; }
    float getSpacePerDigit(const float scale) {
        int space = getSpace(scale);
        return W * scale + space;
    }
    Point<float> draw(int* output_img, const int h, const int w, const float percentage,
                      const Point<float>& offset, const float scale, const int thickness,
                      const Color& color) {
        int i_percentage = percentage * 10000.0f + 0.5f;
        int space = getSpace(scale);
        float space_per_digit = getSpacePerDigit(scale);
        Point<float> abs_offset = offset;
        int denom = 10000;
        for (int i = 0; i < 5; i++) {
            int j = i_percentage / denom;
            if (j > 0 || i > 0) {
                drawDigit(output_img, h, w, j, abs_offset, scale, thickness, color);
            }

            abs_offset.x_ += space_per_digit;
            if (i == 2) {
                drawDot(output_img, h, w, abs_offset, scale, thickness + 1, color);
                abs_offset.x_ += space;
            }
            i_percentage -= denom * j;
            denom /= 10;
        }
        abs_offset.x_ -= offset.x_;
        abs_offset.x_ += space;
        abs_offset.y_ = scale * base_vertex_[3].y_ + space;
        // size
        return abs_offset;
    }
    Point<float> drawPrecision(int* output_img, const int height, const int width,
                               const float percentage, const Point<float>& offset,
                               const float scale, const int thickness, const Color& color) {
        int space = getSpace(scale);

        Point<float> size1 = drawP(output_img, height, width, offset, scale, thickness, color);
        Point<float> newoffset = offset;
        newoffset.x_ += size1.x_ + space;
        drawColon(output_img, height, width, newoffset, scale, thickness, color);
        Point<float> size2 =
            draw(output_img, height, width, percentage, newoffset, scale, thickness, color);
        // size
        size1.x_ += size2.x_;
        size1.y_ += space;
        return size1;
    }
    Point<float> drawRecall(int* output_img, const int height, const int width,
                            const float percentage, const Point<float>& offset, const float scale,
                            const int thickness, const Color& color) {
        int space = getSpace(scale);

        Point<float> size1 = drawR(output_img, height, width, offset, scale, thickness, color);
        Point<float> newoffset = offset;
        newoffset.x_ += size1.x_ + space;
        drawColon(output_img, height, width, newoffset, scale, thickness, color);
        Point<float> size2 =
            draw(output_img, height, width, percentage, newoffset, scale, thickness, color);
        // size
        size1.x_ += size2.x_;
        size1.y_ += space;
        return size1;
    }

    Point<float> drawP(int* output_img, const int height, const int width,
                       const Point<float>& offset, const float scale, const int thickness,
                       const Color& color) {
        Point<float> pt_dir = base_vertex_[1] - base_vertex_[0];
        Point<float> newpt1 = base_vertex_[0] + pt_dir * 0.8f;
        Point<float> newpt4 = base_vertex_[5] + pt_dir * 0.8f;

        pt_dir = base_vertex_[2] - base_vertex_[1];
        Point<float> newpt2 = base_vertex_[1] + pt_dir * 0.25f;
        Point<float> newpt3 = base_vertex_[1] + pt_dir * 0.75f;
        draw_line(output_img, height, width, base_vertex_[0] * scale + offset,
                  newpt1 * scale + offset, thickness, color);
        draw_line(output_img, height, width, newpt1 * scale + offset, newpt2 * scale + offset,
                  thickness, color);
        draw_line(output_img, height, width, newpt2 * scale + offset, newpt3 * scale + offset,
                  thickness, color);
        draw_line(output_img, height, width, newpt3 * scale + offset, newpt4 * scale + offset,
                  thickness, color);
        draw_line(output_img, height, width, newpt4 * scale + offset,
                  base_vertex_[5] * scale + offset, thickness, color);
        drawLine(output_img, height, width, 0, 4, offset, scale, thickness, color);

        return Point<float>(W * scale, H * scale);
    }

    Point<float> drawR(int* output_img, const int height, const int width,
                       const Point<float>& offset, const float scale, const int thickness,
                       const Color& color) {
        drawP(output_img, height, width, offset, scale, thickness, color);
        Point<float> pt_dir = base_vertex_[2] - base_vertex_[5];
        pt_dir = pt_dir * 0.8f;
        Point<float> newpt1 = base_vertex_[5] + pt_dir;
        pt_dir = base_vertex_[3] - base_vertex_[2];
        pt_dir = pt_dir * 0.25f;
        Point<float> newpt2 = base_vertex_[2] + pt_dir;

        draw_line(output_img, height, width, newpt1 * scale + offset, newpt2 * scale + offset,
                  thickness, color);
        draw_line(output_img, height, width, newpt2 * scale + offset,
                  base_vertex_[3] * scale + offset, thickness, color);
        return Point<float>(W * scale, H * scale);
    }

    Point<float> drawClassId(int* output_img, const int height, const int width,
                             const Point<float>& offset, const float scale, const int thickness,
                             const Color& color, const int class_id) {
        drawLine(output_img, height, width, 0, 1, offset, scale, thickness, color);
        drawLine(output_img, height, width, 0, 4, offset, scale, thickness, color);
        drawLine(output_img, height, width, 4, 3, offset, scale, thickness, color);

        int space = getSpace(scale);

        Point<float> newoffset = offset;
        newoffset.x_ += W * scale + space;

        if (class_id == -1) {
            drawA(output_img, height, width, newoffset, scale, thickness, color);
        } else {
            drawDigit(output_img, height, width, class_id, newoffset, scale, thickness, color);
        }

        return Point<float>(W * scale * 2 + space * 2, H * scale);
    }

    Point<float> drawA(int* output_img, const int height, const int width,
                       const Point<float>& offset, const float scale, const int thickness,
                       const Color& color) {
        drawLine(output_img, height, width, 0, 1, offset, scale, thickness, color);
        drawLine(output_img, height, width, 5, 2, offset, scale, thickness, color);
        drawLine(output_img, height, width, 0, 4, offset, scale, thickness, color);
        drawLine(output_img, height, width, 1, 3, offset, scale, thickness, color);
        return Point<float>(W * scale, H * scale);
    }

 protected:
    void drawDigit(int* output_img, const int height, const int width, const int digit,
                   const Point<float>& offset, const float scale, const int thickness,
                   const Color& color) {
        switch (digit) {
            case 0:
                drawLine(output_img, height, width, 0, 1, offset, scale, thickness, color);
                drawLine(output_img, height, width, 1, 3, offset, scale, thickness, color);
                drawLine(output_img, height, width, 3, 4, offset, scale, thickness, color);
                drawLine(output_img, height, width, 4, 0, offset, scale, thickness, color);
                break;
            case 1:
                drawLine(output_img, height, width, 1, 3, offset, scale, thickness, color);
                break;
            case 2:
                drawLine(output_img, height, width, 0, 1, offset, scale, thickness, color);
                drawLine(output_img, height, width, 1, 2, offset, scale, thickness, color);
                drawLine(output_img, height, width, 2, 5, offset, scale, thickness, color);
                drawLine(output_img, height, width, 5, 4, offset, scale, thickness, color);
                drawLine(output_img, height, width, 4, 3, offset, scale, thickness, color);
                break;
            case 3:
                drawLine(output_img, height, width, 0, 1, offset, scale, thickness, color);
                drawLine(output_img, height, width, 5, 2, offset, scale, thickness, color);
                drawLine(output_img, height, width, 4, 3, offset, scale, thickness, color);
                drawLine(output_img, height, width, 1, 3, offset, scale, thickness, color);
                break;
            case 4:
                drawLine(output_img, height, width, 0, 5, offset, scale, thickness, color);
                drawLine(output_img, height, width, 5, 2, offset, scale, thickness, color);
                drawLine(output_img, height, width, 1, 3, offset, scale, thickness, color);
                break;
            case 5:
                drawLine(output_img, height, width, 0, 1, offset, scale, thickness, color);
                drawLine(output_img, height, width, 0, 5, offset, scale, thickness, color);
                drawLine(output_img, height, width, 5, 2, offset, scale, thickness, color);
                drawLine(output_img, height, width, 2, 3, offset, scale, thickness, color);
                drawLine(output_img, height, width, 4, 3, offset, scale, thickness, color);
                break;
            case 6:
                drawLine(output_img, height, width, 0, 1, offset, scale, thickness, color);
                drawLine(output_img, height, width, 0, 4, offset, scale, thickness, color);
                drawLine(output_img, height, width, 5, 2, offset, scale, thickness, color);
                drawLine(output_img, height, width, 2, 3, offset, scale, thickness, color);
                drawLine(output_img, height, width, 4, 3, offset, scale, thickness, color);
                break;
            case 7:
                drawLine(output_img, height, width, 5, 0, offset, scale, thickness, color);
                drawLine(output_img, height, width, 0, 1, offset, scale, thickness, color);
                drawLine(output_img, height, width, 1, 3, offset, scale, thickness, color);
                break;
            case 8:
                drawLine(output_img, height, width, 0, 1, offset, scale, thickness, color);
                drawLine(output_img, height, width, 1, 3, offset, scale, thickness, color);
                drawLine(output_img, height, width, 3, 4, offset, scale, thickness, color);
                drawLine(output_img, height, width, 4, 0, offset, scale, thickness, color);
                drawLine(output_img, height, width, 5, 2, offset, scale, thickness, color);
                break;
            case 9:
                drawLine(output_img, height, width, 5, 0, offset, scale, thickness, color);
                drawLine(output_img, height, width, 0, 1, offset, scale, thickness, color);
                drawLine(output_img, height, width, 1, 3, offset, scale, thickness, color);
                drawLine(output_img, height, width, 5, 2, offset, scale, thickness, color);
                break;
            default:
                break;
        }
    }

 protected:
    void drawLine(int* output_img, const int height, const int width, const int index1,
                  const int index2, const Point<float>& offset, const float scale,
                  const int thickness, const Color& color) {
        draw_line(output_img, height, width, base_vertex_[index1] * scale + offset,
                  base_vertex_[index2] * scale + offset, thickness, color);
    }
    void drawDot(int* output_img, const int height, const int width, const Point<float>& offset,
                 const float scale, const int thickness, const Color& color) {
        set_dots(output_img, height, width, base_vertex_[4] * scale + offset, thickness, color);
    }
    void drawColon(int* output_img, const int height, const int width, const Point<float>& offset,
                   const float scale, const int thickness, const Color& color) {
        Point<float> middle1(0.5f * W, 0.25f * H);
        Point<float> middle2(0.5f * W, 0.75f * H);
        set_dots(output_img, height, width, middle1 * scale + offset, thickness + 1, color);
        set_dots(output_img, height, width, middle2 * scale + offset, thickness + 1, color);
    }
    Point<float> base_vertex_[6];
};

#endif /* DRAW_CHARACTERS_H_ */

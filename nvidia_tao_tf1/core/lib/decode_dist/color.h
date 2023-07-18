// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

#ifndef COLOR_H_
#define COLOR_H_

#include <algorithm>

struct Color {
    uint8_t r_, g_, b_;
    Color(uint8_t r, uint8_t g, uint8_t b) : r_(r), g_(g), b_(b) {}
    Color() : r_(0), g_(0), b_(0) {}
};

class ColorList {
    static constexpr int MAX_NUM_COLOR = 20;

 public:
    ColorList() {
        color_list_[0] = Color(81, 81, 0);
        color_list_[1] = Color(255, 0, 0);
        color_list_[2] = Color(60, 180, 75);
        color_list_[3] = Color(255, 255, 25);
        color_list_[4] = Color(0, 200, 255);
        color_list_[5] = Color(245, 130, 48);
        color_list_[6] = Color(145, 30, 180);
        color_list_[7] = Color(70, 240, 240);
        color_list_[8] = Color(240, 50, 230);
        color_list_[9] = Color(210, 245, 60);
        color_list_[10] = Color(250, 190, 190);
        color_list_[11] = Color(0, 128, 128);
        color_list_[12] = Color(230, 190, 255);
        color_list_[13] = Color(170, 110, 40);
        color_list_[14] = Color(255, 250, 200);
        color_list_[15] = Color(128, 0, 0);
        color_list_[16] = Color(170, 255, 195);
        color_list_[17] = Color(128, 128, 0);
        color_list_[18] = Color(255, 215, 180);
        color_list_[19] = Color(0, 0, 0);
    }
    Color get(const int index) {
        if (index < MAX_NUM_COLOR) {
            return color_list_[index];
        } else {
            return Color(0, 0, 0);
        }
    }
    Color color_list_[MAX_NUM_COLOR];
    static inline int get_max_size() { return MAX_NUM_COLOR; }
};
#endif /* COLOR_H_ */

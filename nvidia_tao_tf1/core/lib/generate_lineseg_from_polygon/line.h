// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.

#ifndef _LINE_HPP_
#define _LINE_HPP_

#include <cmath>     // for abs, max, atan2, M_PI
#include <iostream>  // for ostream
#include <vector>    // for vector

#include "point.h"

template <typename T>
class Line2 {
 public:
    Point2<T> top_;
    Point2<T> bottom_;
    float dx_dy_;               // slope
    float angle_;               // angle_
    int class_id_;              // class id
    int cluster_id_;            // instance_id_
    float angle_bottom_;        // angle_ at the bottom_ segments
    float angle_top_;           // angle_ at the top_ segments.
    float width_left_top_;      // left width of top point to the boundary
    float width_right_top_;     // right width of top point to the boundary
    float width_left_bottom_;   // left width of bottom point to the boundary
    float width_right_bottom_;  // left width of bottom point to the boundary

    // default constructor
    Line2() {
        class_id_ = -1;
        cluster_id_ = 0;
        width_left_top_ = 0.0;
        width_right_top_ = 0.0;
        width_left_bottom_ = 0.0;
        width_right_bottom_ = 0.0;
    }  // do nothing

    Line2(const T x1, const T y1, const T x2, const T y2, const int class_id)
        : Line2(Point2<T>(x1, y1), Point2<T>(x2, y2), class_id) {}

    Line2(const Point2<T> &start, const Point2<T> &end, const int class_id,
          const float width_left_top = 0.0, const float width_right_top = 0.0,
          const float width_left_bottom = 0.0, const float width_right_bottom = 0.0,
          const int cluster_id = 0) {
        top_ = end;
        bottom_ = start;
        calculate_angle();
        class_id_ = class_id;
        cluster_id_ = cluster_id;
        width_left_top_ = width_left_top;
        width_right_top_ = width_right_top;
        width_left_bottom_ = width_left_bottom;
        width_right_bottom_ = width_right_bottom;
    }

    // sort point in y direction
    bool operator<(const Line2<T> &other) const { return top_.y < other.top_.y; }
    static float average_angles(const float rad1, const float rad2) {
        float sum_sin = std::sin(rad1) + std::sin(rad2);
        float sum_cos = std::cos(rad1) + std::cos(rad2);
        return std::atan2(sum_sin, sum_cos);
    }

    // calculate the angle of the line.
    // update/initialize angle values for each line segment.
    void calculate_angle() {
        // --------------->x                        //
        // |        \                               //
        // |         \                              //
        // |        ->\                             //
        // | angle_ (   \                            //
        // | ---------top_-------                    //
        // |             \    ) atan2 (negative)    //
        // |              \ <-                      //
        // |               \                        //
        // |                \                       //
        // y                 bottom_                 //

        // angle_ = -atan2(bottom_.y - top_.y, bottom_.x - top_.x); // in (-PI, PI)

        // ------------------------>x           //
        // |            /                       //
        // |           / <-  angle_ = atan2 + PI //
        // |          /    )                    //
        // |         top_------------            //
        // |        /   ) atan2 (negative)      //
        // |       / <--                        //
        // |      /                             //
        // |     /                              //
        // y  bottom_                            //

        angle_ = std::atan2(bottom_.y - top_.y, bottom_.x - top_.x);  // in (-PI, PI)

        angle_top_ = angle_;
        angle_bottom_ = angle_;
    }

    friend std::ostream &operator<<(std::ostream &os, const Line2<T> &line) {
        os << line.bottom_ << " -> " << line.top_
           << "D:" << point_to_point_distance(line.top_, line.bottom_) << "\t";
        return os;
    }
};
typedef Line2<int> Line2i;
typedef Line2<float> Line2f;

// compare
template <typename T>
inline bool compare_line(const Line2<T> a, const Line2<T> b) {
    const T dy = a.top_.y - b.top_.y;
    return std::abs(dy) > EPSILON ? dy < 0 : a.top_.x < b.top_.x;
}

template <typename T>
inline float angle_between_two_lines(const Line2<T> &line1, const Line2<T> &line2,
                                     bool inDegree = false) {
    const float angle_difference = std::abs(line1.angle_ - line2.angle_);
    float r = std::fmod(angle_difference, M_PI * 2.0f);
    if (r < -M_PI) {
        r += M_PI * 2;
    }
    if (r >= M_PI) {
        r -= M_PI * 2;
    }
    if (inDegree) {
        r *= 180 / M_PI;
    }
    return r;
}

#endif  // _LINE_HPP_

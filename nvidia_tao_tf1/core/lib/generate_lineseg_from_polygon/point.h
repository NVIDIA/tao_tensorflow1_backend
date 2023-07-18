// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

#ifndef _POINT_HPP_
#define _POINT_HPP_

#include <cmath>     // for abs
#include <iostream>  // for ostream
#include <vector>    // for vector

// epsilon for division
const float EPSILON = 1e-6f;

template <typename T>
class Point {
 public:
    Point(T x, T y) {
        x_ = x;
        y_ = y;
    }
    Point() : Point(static_cast<T>(0), static_cast<T>(0)) {}
    Point operator*(T alpha) {
        Point ret = *this;
        ret.x_ *= alpha;
        ret.y_ *= alpha;
        return ret;
    }
    Point operator+(const Point<T> &pt) {
        Point ret = (*this);
        ret.x_ += pt.x_;
        ret.y_ += pt.y_;
        return ret;
    }
    Point operator-(const Point<T> &pt) {
        Point ret = (*this);
        ret.x_ -= pt.x_;
        ret.y_ -= pt.y_;
        return ret;
    }
    T x_;
    T y_;
};

template <typename T>
class Point2 {
 public:
    T x;
    T y;

    Point2() : x(0), y(0) {}
    Point2(T _x, T _y) : x(_x), y(_y) {}
    Point2(const Point2<T> &p) : x(p.x), y(p.y) {}

    bool is_nonnegative() { return (x >= 0 && y >= 0); }
    // sort point in y direction
    bool operator<(const Point2<T> &other) const { return y < other.y; }

    // comparison
    bool operator==(const Point2<T> &other) const {
        return (std::abs(x - other.x) < EPSILON) && (std::abs(y - other.y) < EPSILON);
    }
    // comparison
    bool operator!=(const Point2<T> &other) const { return !((*this) == other); }

    friend std::ostream &operator<<(std::ostream &os, const Point2<T> &point) {
        os << "(" << point.x << ", " << point.y << ")";
        return os;
    }
};
typedef Point2<int> Point2i;
typedef Point2<float> Point2f;

// addition
template <typename T>
Point2<T> operator+(const Point2<T> &p1, const Point2<T> &p2) {
    return Point2<T>(p1.x + p2.x, p1.y + p2.y);
}

// subtraction
template <typename T>
Point2<T> operator-(const Point2<T> &p1, const Point2<T> &p2) {
    return Point2<T>(p1.x - p2.x, p1.y - p2.y);
}

// multiplication
template <typename T>
Point2<T> operator*(const T scale, const Point2<T> &point) {
    return Point2<T>(scale * point.x, scale * point.y);
}

template <typename T>
Point2<T> operator*(const Point2<T> &point, const T scale) {
    return Point2<T>(scale * point.x, scale * point.y);
}

// middle point
template <typename T>
inline Point2<T> mid_point(const Point2<T> a, const Point2<T> b) {
    return Point2<T>(0.5f * (a.x + b.x), 0.5f * (a.y + b.y));
}

template <typename T>
inline bool cos_val(const Point2<T> a, const Point2<T> b, T *cosVal) {
    T mag_a;
    T mag_b;
    bool result = cos_val(a, b, cosVal, &mag_a, &mag_b);
    (void)mag_a;
    (void)mag_b;
    return result;
}

template <typename T>
inline bool cos_val(const Point2<T> a, const Point2<T> b, T *cosVal, T *mag_a, T *mag_b) {
    *mag_a = std::sqrt(a.x * a.x + a.y * a.y);
    *mag_b = std::sqrt(b.x * b.x + b.y * b.y);
    bool result = false;
    if ((*mag_a) > FLT_MIN && (*mag_b) > FLT_MIN) {
        *cosVal = a.x * b.x + a.y * b.y;
        *cosVal /= ((*mag_a) * (*mag_b));
        result = true;
    }
    return result;
}

template <typename T>
inline float norm2(T x, T y) {
    return std::sqrt(x * x + y * y);
}

template <typename T>
inline float norm2(const Point2<T> &p) {
    return norm2(p.x, p.y);
}

// calculate the distance between two points
inline float point_to_point_distance(const float x1, const float y1, const float x2,
                                     const float y2) {
    const float dx = x2 - x1;
    const float dy = y2 - y1;
    return norm2(dx, dy);
}

// calculate the distance between two points
template <typename T>
inline float point_to_point_distance(const Point2<T> &p1, const Point2<T> &p2) {
    return point_to_point_distance(p1.x, p1.y, p2.x, p2.y);
}

#endif  // _POINT_HPP_

#pragma once

// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.

/*
Basic 2D Point/Vector implementation
*/
#include <cmath>

template <typename T>
struct Point2 {
    T x;
    T y;
};

template <typename T>
Point2<T> operator+(const Point2<T> &a, const Point2<T> &b) {
    return Point2<T>{a.x + b.x, a.y + b.y};
}

template <typename T>
Point2<T> operator-(const Point2<T> &a, const Point2<T> &b) {
    return Point2<T>{a.x - b.x, a.y - b.y};
}

template <typename T>
Point2<T> operator*(const T &v, const Point2<T> &a) {
    return Point2<T>{a.x * v, a.y * v};
}

template <typename T>
Point2<T> operator/(const Point2<T> &a, const T &v) {
    return Point2<T>{a.x / v, a.y / v};
}

template <typename T>
T norm(const Point2<T> &a) {
    return std::sqrt(a.x * a.x + a.y * a.y);
}

template <typename T>
Point2<T> perpendicular(const Point2<T> &a) {
    return Point2<T>{a.y, -a.x};
}

template <typename T>
T distance(const Point2<T> &a, const Point2<T> &b) {
    return std::sqrt(std::pow((a.x - b.x), 2) + std::pow((a.y - b.y), 2));
}

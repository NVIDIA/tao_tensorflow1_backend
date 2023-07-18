// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

#ifndef _POINT_HPP_
#define _POINT_HPP_

#include <vector>

template <typename T>
class PointN {
 public:
    std::vector<T> x_;
    int n_dims_;
    explicit PointN(int _n_dims) {
        for (int i = 0; i < _n_dims; i++) x_.push_back(0);
        n_dims_ = _n_dims;
    }
    explicit PointN(std::vector<T> _x) : x_(_x), n_dims_(static_cast<int>(_x.size())) {}
    PointN(std::vector<T> _x, int _n_dims) : x_(_x), n_dims_(_n_dims) {}
    PointN(const PointN<T> &p) : x_(p.x_), n_dims_(p.n_dims_) {}
};
typedef PointN<int> PointNi;
typedef PointN<float> PointNf;

// Addition.
template <typename T>
PointN<T> operator+(const PointN<T> &p1, const PointN<T> &p2) {
    PointN<T> res = PointN<T>(p1);
    for (int i = 0; i < p1.n_dims_; i++) {
        res.x_[i] = p1.x_[i] + p2.x_[i];
    }
    return res;
}

// Subtraction.
template <typename T>
PointN<T> operator-(const PointN<T> &p1, const PointN<T> &p2) {
    PointN<T> res = PointN<T>(p1);
    for (int i = 0; i < p1.n_dims_; i++) {
        res.x_[i] = p1.x_[i] - p2.x_[i];
    }
    return res;
}

// Multiplication.
template <typename T>
PointN<T> operator*(const T scale, const PointN<T> &p1) {
    PointN<T> res = PointN<T>(p1);
    for (int i = 0; i < p1.n_dims_; i++) {
        res.x_[i] = p1.x_[i] * scale;
    }
    return res;
}

// Multiplication.
template <typename T>
PointN<T> operator*(const PointN<T> &p1, const T scale) {
    PointN<T> res = PointN<T>(p1);
    for (int i = 0; i < p1.n_dims_; i++) {
        res.x_[i] = p1.x_[i] * scale;
    }
    return res;
}

// L2-norm.
template <typename T>
inline float norm2(const PointN<T> &p) {
    float res = 0.;
    for (int i = 0; i < p.n_dims_; i++) {
        res += p.x_[i] * p.x_[i];
    }
    return std::sqrt(res);
}

// Calculate the distance between two points.
template <typename T>
inline float point_to_point_distance(const PointN<T> &p1, const PointN<T> &p2) {
    return norm2(p1 - p2);
}

#endif  // _POINT_HPP_

// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.

#ifndef _POLYGON_HPP_
#define _POLYGON_HPP_

#include <algorithm>  // for min
#include <cassert>    // for assert
#include <cmath>
#include <iostream>  // for ostream
#include <limits>    // for float limits
#include <utility>   // for pair
#include <vector>    // for vector
#include "line.h"
#include "point.h"
#include "polyfit.h"

template <typename T>
class Polygon2 {
 protected:
    class _point3 {
     public:
        _point3() : x(0), y(0), z(0) {}
        explicit _point3(const Point2<T>& p) : x(p.x), y(p.y), z(1) {}
        _point3(T a, T b, T c) : x(a), y(b), z(c) {}
        T x;
        T y;
        T z;
    };

 public:
    static constexpr float TURNING_DEGREE = 100;
    static constexpr float WEIRD_ANGLE_DIFF_IN_LINE = 150;
    static constexpr int MAX_WEIRD_ANGLE_COUNT_IN_LINE = 0;
    static constexpr float COS_VAL_TO_DEFINE_ALIGNED = 0.2;
    static constexpr float VERTICAL_CHANGE_TO_DEFINE_ALIGHED = 0.15;
    static constexpr float MIN_DISTANCE_BETWEEN_VERTICES = 4.0f;
    static constexpr float MAX_DISTANCE_BETWEEN_VERTICES = 100.0f;
    static constexpr float VERTEX_ANGLE_OFFSET = 10.0f;
    static constexpr int HORIZONTAL_STEP = 1;
    static constexpr int HORIZONTAL_MAX_NUM_SCAN_LINES = 20;
    static constexpr int MAX_SCAN_LINES = 20;
    static constexpr float MIN_GAP_FROM_Y = 20.0f;
    static constexpr int MIN_POINTS_TO_FIT = 3;
    static constexpr float RATIO_OF_TRIMMED_END = 0.05f;
    enum CheckIntersectionOption { CIO_NORMAL = 0, CIO_BOTTOM = 1, CIO_TOP = 2 };
    Polygon2() {
        processed_ = false;
        class_id_ = -1;
        width_ = -1;
        height_ = -1;
    }
    // add vertices.
    void add_vertex(const T x, const T y) {
        Point2<T> curr(x, y);
        bool add = true;
        raw_vertices_.push_back(curr);
        if (ordered_vertices_.size() > 0) {
            const Point2<T> prev = ordered_vertices_.back();
            float dist = point_to_point_distance(prev, curr);
            if (dist < MIN_DISTANCE_BETWEEN_VERTICES) {
                add = false;
            } else if (dist > MAX_DISTANCE_BETWEEN_VERTICES) {  // break it...
                Point2<T> mid = mid_point(prev, curr);
                ordered_vertices_.push_back(mid);
            }
        }
        if (add) {
            ordered_vertices_.push_back(curr);
        }
    }

    // Refernce: http://mathworld.wolfram.com/PolygonArea.html
    bool is_counterclockwise(const std::vector<Point2<T>>& vertices) {
        Point2<T> prev = vertices[0];
        float signed_area = 0.0f;
        for (size_t i = 1; i < vertices.size(); i++) {
            Point2<T> current = vertices[i];
            signed_area += static_cast<float>(prev.x * current.y - current.x * prev.y);
            prev = current;
        }
        Point2<T> first_point = vertices[0];
        signed_area += static_cast<float>(prev.x * first_point.y - first_point.x * prev.y);
        return signed_area > 0.0f;
    }

    bool skip_conversion_to_polygon_process(const int cluster_id) {
        if (processed_ == false) {
            centerline_segments_.clear();
            // In Actual images, we are using the flipped coordinates.
            // (i.e going down vertically is positive for y coordinate)
            // Therefore the return images should have counterclocwise
            // if we chose `clockwise` for above defintion.
            // So below we reverse the defintion to make it "clockwise" in our images.
            if (processed_ == false && ordered_vertices_.size() >= 3) {
                if (!is_counterclockwise(ordered_vertices_)) {
                    std::reverse(ordered_vertices_.begin(), ordered_vertices_.end());
                }
                update_linesegment_from_vertices(ordered_vertices_, cluster_id,
                                                 &centerline_segments_, !is_polyline_);
                processed_ = true;
            }
        }
        return processed_;
    }

    // Width Encoding
    bool get_width(const Point2<T>& bottom, const Point2<T>& top,
                   const std::vector<Line2<T>>& linesegments, bool isBottom, T* width_left,
                   T* width_right) {
        bool result_right = false;
        bool result_left = false;

        float min_width_left = std::numeric_limits<float>::max();
        float min_width_right = std::numeric_limits<float>::max();

        _point3 lineSeg;
        get_line(bottom, top, &lineSeg);

        // find the normal line through point
        _point3 normal_line;
        if (isBottom) {
            get_normal_line_passing(lineSeg, bottom, &normal_line);
        } else {
            get_normal_line_passing(lineSeg, top, &normal_line);
        }

        // iterate over all the linesegments of a polygon
        _point3 intersection;
        for (const Line2<T>& line : linesegments) {
            _point3 polygonLineSeg;
            get_line(line.bottom_, line.top_, &polygonLineSeg);

            // check for intersection of the normal line with polygon linesegment
            if (cross_product(normal_line, polygonLineSeg, &intersection)) {
                Point2<T> pt_intersection = Point2<T>(intersection.x, intersection.y);
                bool proper_intersection =
                    check_proper_intersection(line.bottom_, line.top_, pt_intersection);

                // Compute width left and right
                if (proper_intersection) {
                    if (!isBottom) {
                        float z_val = (bottom.x - top.x) * (pt_intersection.y - top.y) -
                                      (bottom.y - top.y) * (pt_intersection.x - top.x);
                        float dist_val = point_to_point_distance(top, pt_intersection);
                        // for top point, cross product with right point gives negative value
                        if (z_val < 0) {
                            if (dist_val < min_width_right) {
                                min_width_right = dist_val;
                                result_right = true;
                            }
                        } else {
                            if (dist_val < min_width_left) {
                                min_width_left = dist_val;
                                result_left = true;
                            }
                        }
                    } else {
                        float z_val = (top.x - bottom.x) * (pt_intersection.y - bottom.y) -
                                      (top.y - bottom.y) * (pt_intersection.x - bottom.x);
                        float dist_val = point_to_point_distance(bottom, pt_intersection);
                        // for bottom point, cross product with right point gives positive value
                        if (z_val > 0) {
                            if (dist_val < min_width_right) {
                                min_width_right = dist_val;
                                result_right = true;
                            }
                        } else {
                            if (dist_val < min_width_left) {
                                min_width_left = dist_val;
                                result_left = true;
                            }
                        }
                    }
                }
            }
        }

        // if left is found and right is not found or vice versa
        // use the found width for both
        if (result_left && !result_right) {
            *width_left = static_cast<T>(min_width_left);
            *width_right = static_cast<T>(min_width_left);
            return true;
        } else if (!result_left && result_right) {
            *width_right = static_cast<T>(min_width_right);
            *width_left = static_cast<T>(min_width_right);
            return true;
        } else if (result_left && result_right) {
            *width_left = static_cast<T>(min_width_left);
            *width_right = static_cast<T>(min_width_right);
            return true;
        }

        return false;
    }

    void get_boundary_encoding(const Point2<T>& bottom, const Point2<T>& top,
                               const std::vector<Line2<T>>& linesegments, T* width_left_top,
                               T* width_right_top, T* width_left_bottom, T* width_right_bottom) {
        bool result_top =
            get_width(bottom, top, linesegments, false, width_left_top, width_right_top);
        bool result_bottom =
            get_width(bottom, top, linesegments, true, width_left_bottom, width_right_bottom);

        // if top is found and bottom is not found or vice versa
        // use the found width for both
        if (result_top && !result_bottom) {
            *width_left_bottom = *width_left_top;
            *width_right_bottom = *width_right_top;
        }

        if (result_bottom && !result_top) {
            *width_left_top = *width_left_bottom;
            *width_right_top = *width_right_bottom;
        }
    }

    void update_ys() {
        float tmp_min = FLT_MAX;
        float tmp_max = -1.0;
        for (size_t i = 0; i < raw_vertices_.size(); ++i) {
            if (raw_vertices_[i].y > tmp_max) {
                tmp_max = raw_vertices_[i].y;
            }
            if (raw_vertices_[i].y < tmp_min) {
                tmp_min = raw_vertices_[i].y;
            }
        }
        max_y_ = static_cast<T>(tmp_max);
        min_y_ = static_cast<T>(tmp_min);
    }

    void generate_scan_y(const float miny, const float maxy, const float space_y, const int num_ys,
                         std::vector<float>* output) {
        for (int i = 0; i < num_ys; i++) {
            float y = MIN(static_cast<float>(miny + i * space_y), static_cast<float>(maxy));
            if (num_ys - i - 1 >= 0) {
                (*output)[num_ys - i - 1] = y;
            }
        }
    }

    void update_final_pts() {
        if (scanned_pts_.size() != scan_line_y_.size()) {
            printf(" Number of keys in polygon does not match %d vs %d \n",
                   static_cast<int>(scanned_pts_.size()), static_cast<int>(scan_line_y_.size()));
            //  skip if happens the above.
            return;
        } else {
            width_left_.clear();
            width_right_.clear();
            float sum_y1 = 0.0f;
            float sum_y2 = 0.0f;
            float sum_y3 = 0.0f;
            float sum_y4 = 0.0f;
            float sum_x1 = 0.0f;
            float sum_x1y1 = 0.0f;
            float sum_x1y2 = 0.0f;
            float sum_x2 = 0.0f;
            float count = 0.0f;
            size_t maps_size = scanned_pts_.size();
            float actual_gap = max_y_ - min_y_;
            size_t start_index =
                static_cast<size_t>(MAX(std::floor(RATIO_OF_TRIMMED_END * maps_size), 0));
            start_index = actual_gap > MIN_GAP_FROM_Y ? start_index : 0;
            size_t end_index = static_cast<size_t>(
                MIN(maps_size - std::ceil(RATIO_OF_TRIMMED_END * maps_size), maps_size));
            end_index = actual_gap > MIN_GAP_FROM_Y ? end_index : maps_size;
            for (size_t i = start_index; i < end_index; ++i) {
                int sz_vec = scanned_pts_[i].second.size();
                if (sz_vec == 1) {
                    centerline_vertices_.push_back(scanned_pts_[i].second[0]);
                    width_left_.push_back(0.0f);
                    width_right_.push_back(0.0f);
                    float x = scanned_pts_[i].second[0].x;
                    float y = scanned_pts_[i].second[0].y;
                    get_quad_sum_input(x, y, &sum_y1, &sum_y2, &sum_y3, &sum_y4, &sum_x1, &sum_x1y1,
                                       &sum_x1y2, &sum_x2, &count);
                } else if (sz_vec != 2) {
                    continue;
                } else {
                    Point2<T> mid_pt =
                        mid_point(scanned_pts_[i].second[0], scanned_pts_[i].second[1]);
                    float dist_to_right = 0.0f;
                    float dist_to_left = 0.0f;
                    if (scanned_pts_[i].second[0].x > scanned_pts_[i].second[1].x) {
                        dist_to_right = point_to_point_distance(mid_pt, scanned_pts_[i].second[0]);
                        dist_to_left = point_to_point_distance(mid_pt, scanned_pts_[i].second[1]);
                    } else {
                        dist_to_left = point_to_point_distance(mid_pt, scanned_pts_[i].second[0]);
                        dist_to_right = point_to_point_distance(mid_pt, scanned_pts_[i].second[1]);
                    }
                    mid_pt.x = static_cast<int>(mid_pt.x) + 0.0;
                    mid_pt.y = static_cast<int>(mid_pt.y) + 0.0;
                    float x = mid_pt.x;
                    float y = mid_pt.y;
                    get_quad_sum_input(x, y, &sum_y1, &sum_y2, &sum_y3, &sum_y4, &sum_x1, &sum_x1y1,
                                       &sum_x1y2, &sum_x2, &count);
                    centerline_vertices_.push_back(mid_pt);
                    width_left_.push_back(dist_to_left);
                    width_right_.push_back(dist_to_right);
                }
            }
            fit_quad(&coeffs_, count, sum_y1, sum_y2, sum_y3, sum_y4, sum_x1, sum_x1y1, sum_x1y2,
                     sum_x2);
        }
    }

    void generate_scanned_pts(const std::vector<float>& scany, const int sz) {
        for (int i = 0; i < sz; ++i) {
            float val_y = scany[i];
            int val_y_int = static_cast<int>(val_y + 0.5);

            _point3 horizontal_line;
            horizontal_line.x = 0.0;
            horizontal_line.y = 1;
            horizontal_line.z = -scany[i];
            // iterate over all the linesegments of a polygon.
            _point3 intersection;
            int current_intersection_count = 0;
            scan_line_y_.push_back(val_y_int);
            std::vector<Point2<T>> intersection_pts;
            for (const Line2<T>& line : sorted_segments_) {
                _point3 polygonLineSeg;
                get_line(line.bottom_, line.top_, &polygonLineSeg);
                float ym = MIN(line.bottom_.y, line.top_.y);
                float yM = MAX(line.bottom_.y, line.top_.y);
                if (val_y < ym || val_y > yM) {
                    continue;
                }
                if (cross_product(horizontal_line, polygonLineSeg, &intersection)) {
                    Point2<T> pt_intersection = Point2<T>(intersection.x, intersection.y);
                    bool proper_intersection =
                        check_proper_intersection(line.bottom_, line.top_, pt_intersection);
                    if (proper_intersection) {
                        current_intersection_count++;
                        intersection_pts.push_back(pt_intersection);
                    }
                }
            }
            scanned_pts_.emplace_back(val_y_int, intersection_pts);
        }
        update_final_pts();
    }

    void process_each_polygon(float space_y = static_cast<float>(HORIZONTAL_STEP),
                              int max_num_ys = HORIZONTAL_MAX_NUM_SCAN_LINES) {
        update_ys();
        if (max_y_ > min_y_) {
            float tmp_spacey = (max_y_ - min_y_) / HORIZONTAL_MAX_NUM_SCAN_LINES;
            float space_y = tmp_spacey > HORIZONTAL_STEP + 0.5f ? tmp_spacey : HORIZONTAL_STEP;
            space_y = round(space_y + 0.5f);
            if (space_y <= 0) {
                printf(" space_y should be positive but got %f, set to 1.0\n", space_y);
                space_y = 1.0;
            }
            int num_ys = static_cast<int>((max_y_ + 0.5 - min_y_) / space_y);
            if (num_ys > 0) {
                centerline_vertices_.clear();
                std::vector<float> scan_y(num_ys);
                generate_scan_y(min_y_, max_y_, space_y, num_ys, &scan_y);
                int size_y = scan_y.size();
                if (size_y <= 0) {
                    printf(
                        " horizontal lines size is Not positive, something is wrong, miny=%f "
                        "maxy=%f \n",
                        min_y_, max_y_);
                } else {
                    generate_scanned_pts(scan_y, size_y);
                    scan_y.clear();
                }
            }
        }
    }

    void update_linesegment_from_vertices(const std::vector<Point2<T>>& vertices,
                                          const int cluster_id, std::vector<Line2<T>>* linesegments,
                                          const bool connect = false) {
        linesegments->clear();
        int total_filtered = vertices.size();
        Point2<T> prev = vertices[0];
        for (int j = 1; j < total_filtered; j++) {
            Point2<T> pt = vertices[j];
            if (cluster_id >= 0) {
                linesegments->push_back(Line2<T>(prev, pt, class_id_, static_cast<T>(0.0f),
                                                 static_cast<T>(0.0f), static_cast<T>(0.0f),
                                                 static_cast<T>(0.0f), cluster_id));
            } else {
                linesegments->push_back(Line2<T>(prev, pt, class_id_));
            }
            prev = pt;
        }
        // connect last and first point as closed polygon.
        if (connect) {
            linesegments->push_back(Line2<T>(
                vertices[total_filtered - 1], vertices[0], class_id_, static_cast<T>(0.0f),
                static_cast<T>(0.0f), static_cast<T>(0.0f), static_cast<T>(0.0f), cluster_id));
        }
    }

    bool skip_extract_centerlines(const int cluster_id) {
        // for valid polyline type, minim number of vertices should be at least 2.
        // such as point1 and point2 to have a line.
        if (processed_ == false && ordered_vertices_.size() >= 2) {
            centerline_segments_.clear();
            update_linesegment_from_vertices(ordered_vertices_, cluster_id, &centerline_segments_,
                                             false);
            processed_ = true;
        }
        return processed_;
    }

    float get_x_based_on_coeff(const Vector3f coeff, const float y) {
        return coeff[0] + coeff[1] * y + coeff[2] * y * y;
    }

    void update_line_angle_via_coeffcients(Line2<T>* line, const Point2<T>& prev,
                                           const Point2<T>& current) {
        if (coeffs_[2] != 0.0f) {
            line->angle_top_ = std::atan2(-1.0f, 2 * coeffs_[2] * prev.y + coeffs_[1]);
            line->angle_bottom_ = std::atan2(-1.0f, 2 * coeffs_[2] * current.y + coeffs_[1]);
            line->angle_ = Line2<T>::average_angles(line->angle_top_, line->angle_bottom_);
        } else {
            float angle = std::atan2((1.0f - coeffs_[0]) / coeffs_[1], 1.0f);
            line->angle_ = angle;
            line->angle_top_ = angle;
            line->angle_bottom_ = angle;
        }
    }

    size_t convert_ordered_vertices_to_line_segments(const std::vector<Point2<T>>& ordered_vertices,
                                                     const int cluster_id,
                                                     std::vector<Line2<T>>* line_segments) {
        centerline_segments_.clear();
        // convert ordered vertices to line segments
        // points are odered from bottom to top.
        int gap_from_bottom = std::ceil(max_y_ - ordered_vertices.front().y);
        int gap_from_top = std::ceil(ordered_vertices.back().y - min_y_);
        int start_index = MAX(gap_from_bottom, 0);
        int end_index = MAX(gap_from_top, 0);

        T y1 = ordered_vertices.front().y + gap_from_bottom;
        T x1 = ordered_vertices.front().x + gap_from_bottom * (2.0f * coeffs_[2] * y1 + coeffs_[0]);
        Point2<T> prev(x1, y1);
        for (int i = 0; i < start_index; ++i) {
            T y2 = prev.y - 1;
            T x2 = prev.x - (2.0f * coeffs_[2] * y2 + coeffs_[0]);
            Point2<T> current(x2, y2);
            Line2<T> line(Line2<T>(prev, current, class_id_, static_cast<T>(width_left_[0]),
                                   static_cast<T>(width_right_[0]), static_cast<T>(width_left_[0]),
                                   static_cast<T>(width_right_[0]), cluster_id));
            update_line_angle_via_coeffcients(&line, prev, current);
            centerline_segments_.push_back(line);
            prev = current;
        }

        prev = ordered_vertices[0];
        for (size_t i = 1; i < ordered_vertices.size(); i++) {
            Point2<T> current = ordered_vertices[i];
            Line2<T> line(Line2<T>(prev, current, class_id_, static_cast<T>(width_left_[i]),
                                   static_cast<T>(width_right_[i]), static_cast<T>(width_left_[i]),
                                   static_cast<T>(width_right_[i]), cluster_id));
            centerline_segments_.push_back(line);
            prev = current;
        }

        prev = ordered_vertices.back();
        for (int i = 0; i < end_index; ++i) {
            T y2 = prev.y - 1;
            T x2 = prev.x - (2.0f * coeffs_[2] * y2 + coeffs_[0]);
            Point2<T> current(x2, y2);
            Line2<T> line(Line2<T>(prev, current, class_id_, static_cast<T>(width_left_.back()),
                                   static_cast<T>(width_right_.back()),
                                   static_cast<T>(width_left_.back()),
                                   static_cast<T>(width_right_.back()), cluster_id));
            centerline_segments_.push_back(line);
            prev = current;
        }
        return centerline_segments_.size();
    }

    // extract the centerlines.
    bool extract_centerlines(const int cluster_id) {
        if (processed_ == false) {
            if (extract_via_horizontal_scanline_ == true && raw_vertices_.size() >= 3) {
                update_linesegment_from_vertices(raw_vertices_, -1, &sorted_segments_, true);
                process_each_polygon();
                if (centerline_vertices_.size() > 0) {
                    update_linesegment_from_vertices(centerline_vertices_, cluster_id,
                                                     &centerline_segments_before_, false);
                    convert_ordered_vertices_to_line_segments(centerline_vertices_, cluster_id,
                                                              &centerline_segments_before_);
                    processed_ = true;
                }
            }
            if (extract_via_horizontal_scanline_ == false && ordered_vertices_.size() >= 3) {
                find_turning_and_remove_close_points();
                if (refine_line_segments(cluster_id)) {
                    processed_ = true;
                }
            }
        }
        return processed_;
    }
    // detect v shape polygons.
    bool is_v_shape_polygon() { return detect_v_polygon_helper(); }

    // save v shape vertices (inside Point2, outside Point2).
    const std::vector<Point2<T>>& get_v_shape_vertices() const { return v_shape_vertices_; }

    void set_dimension(int width, int height) {
        width_ = width;
        height_ = height;
    }

    void set_class_id(const int class_id) { class_id_ = class_id; }

    void set_is_polyline(const bool is_polyline) { is_polyline_ = is_polyline; }
    void set_extract_via_horizontal_scanline(const bool extract_via_horizontal_scanline) {
        extract_via_horizontal_scanline_ = extract_via_horizontal_scanline;
    }
    int get_class_id() const { return class_id_; }

    size_t num_vertices() const { return ordered_vertices_.size(); }
    size_t raw_num_vertices() const { return raw_vertices_.size(); }

    const std::vector<Point2<T>>& get_ordered_vertices() const { return ordered_vertices_; }

    const std::vector<Line2<T>>& get_centerlines() const { return centerline_segments_; }

    friend std::ostream& operator<<(std::ostream& os, const Polygon2<T>& polygon) {
        // number of vertices.
        os << polygon.ordered_vertices_.size();

        // for each vertex point.
        for (const auto& point : polygon.ordered_vertices_) {
            os << std::endl << point.x << "\t" << point.y;
        }
        return os;
    }

 protected:
    float get_angle_pt_360(Point2<T> pt1) const {
        // Get the angle of a point/vector ranging from 0 to 360.
        float cos_ang = pt1.x / norm2(pt1);
        float sin_ang = pt1.y / norm2(pt1);
        float ang = std::acos(cos_ang);
        if (sin_ang > 0)
            ang = ang / M_PI * 180;
        else
            ang = 360 - ang / M_PI * 180;
        return ang;
    }

    float get_angle_turn_360(Point2<T> pt1, Point2<T> pt2, Point2<T> pt3) const {
        // Get the angle of turns based on the angle of each line segment.
        // return a float number between 0 and 360.
        Point2<T> vec12 = pt2 - pt1;
        Point2<T> vec23 = pt3 - pt2;
        float ang12 = get_angle_pt_360(vec12);
        float ang23 = get_angle_pt_360(vec23);
        float angle_diff = ang23 - ang12;
        if (angle_diff < 0) angle_diff += 360;
        return angle_diff;
    }

    float convert_angle_from_360_to_180(float angle_diff) const {
        // Get the angle of turns based on the angle of each line segment.
        // return a float number between 0 and 180.
        if (angle_diff > 180)
            return 360 - angle_diff;
        else
            return angle_diff;
    }

    int get_turn_left_or_right(float angle_diff) const {
        // Get the direction of turns based on the angle of each line segment.
        // return 0 means straight or 180 turn. return 1 means left turn.
        // return -1 means right turn.
        if (angle_diff > 180) return 1;
        if (angle_diff < 180) return -1;
        return 0;
    }

    bool get_turn_up_or_down(Point2<T> pt1, Point2<T> pt2, Point2<T> pt3, Point2<T> pt4) const {
        // Get the forward/backward change of four consecutive points.
        // Return true if there is such a change.
        return (pt1.y < pt2.y && pt3.y > pt4.y) || (pt1.y > pt2.y && pt3.y < pt4.y);
    }

    bool check_abnormal(const std::vector<Point2<T>>& pt_index) const {
        // Check abnormal cases when one arm is significantly longer than the other.
        const float thre_arm_length_ratio = 3.;
        const float thre_midpoint_distance_ratio = 5.;

        Point2<T> mid_pt1 = mid_point(pt_index[0], pt_index[2]);
        Point2<T> mid_pt2 = mid_point(pt_index[1], pt_index[3]);

        float dist1 = point_to_point_distance(pt_index[0], pt_index[2]);
        float dist2 = point_to_point_distance(pt_index[1], pt_index[3]);
        float dist3 = point_to_point_distance(mid_pt1, mid_pt2);

        float dist_arm11 = point_to_point_distance(mid_pt1, pt_index[1]);
        float dist_arm12 = point_to_point_distance(mid_pt1, pt_index[3]);
        float dist_arm21 = point_to_point_distance(mid_pt2, pt_index[0]);
        float dist_arm22 = point_to_point_distance(mid_pt2, pt_index[2]);

        if ((dist2 / dist3 > thre_arm_length_ratio) || (dist1 / dist3 > thre_arm_length_ratio) ||
            (dist_arm11 / dist_arm12 > thre_midpoint_distance_ratio) ||
            (dist_arm21 / dist_arm22 > thre_midpoint_distance_ratio) ||
            (dist_arm12 / dist_arm11 > thre_midpoint_distance_ratio) ||
            (dist_arm22 / dist_arm21 > thre_midpoint_distance_ratio))
            return false;

        return true;
    }

    bool detect_v_polygon_helper() {
        // Detecting whether the polygon if v-shape or not.
        // At the same time, if it's v-shape, update the v-shape vertices.
        // \                                 /
        //  \                               /
        //    Here are the points to record
        std::vector<bool> updown;
        std::vector<float> angs_pre;
        std::vector<float> angs_post;
        std::vector<int> turn_pre;
        std::vector<int> turn_post;
        std::vector<int> turn_dir;
        std::vector<Point2<T>> pt_index;

        int vertices_per_polygon = ordered_vertices_.size();

        if (vertices_per_polygon > 4) {
            int c_start = 0;
            if (point_to_point_distance(ordered_vertices_[0],
                                        ordered_vertices_[vertices_per_polygon - 1]) <
                MIN_DISTANCE_BETWEEN_VERTICES)
                c_start = 1;

            for (int c = c_start; c < vertices_per_polygon; c++) {
                Point2<T> cur_pt = ordered_vertices_[c];
                Point2<T> prev_pt;
                Point2<T> next_pt;
                Point2<T> nextnext_pt;

                if (c == c_start)
                    prev_pt = ordered_vertices_[vertices_per_polygon - 1];
                else
                    prev_pt = ordered_vertices_[c - 1];

                if (c == vertices_per_polygon - 1)
                    next_pt = ordered_vertices_[c_start];
                else
                    next_pt = ordered_vertices_[c + 1];

                if (c == vertices_per_polygon - 2)
                    nextnext_pt = ordered_vertices_[c_start];
                else if (c == vertices_per_polygon - 1)
                    nextnext_pt = ordered_vertices_[c_start + 1];
                else
                    nextnext_pt = ordered_vertices_[c + 2];

                float angle_turn = get_angle_turn_360(prev_pt, cur_pt, next_pt);
                int cur_turn = get_turn_left_or_right(angle_turn);
                float cur_angle = convert_angle_from_360_to_180(angle_turn);
                bool cur_updown_change = get_turn_up_or_down(prev_pt, cur_pt, next_pt, nextnext_pt);

                updown.push_back(cur_updown_change);
                angs_pre.push_back(cur_angle);
                turn_pre.push_back(cur_turn);

                if (c > c_start) {
                    angs_post.push_back(cur_angle);
                    turn_post.push_back(cur_turn);
                }
            }
            angs_post.push_back(angs_pre[0]);
            turn_post.push_back(turn_pre[0]);

            int idx = 0;
            int sum_turns = 0;
            int n_corner = 0;
            bool is_corner = false;
            bool is_corner_pre = false;

            while (idx < vertices_per_polygon - c_start) {
                if (updown[idx] == true) {
                    if ((angs_post[idx] > 120) || (angs_pre[idx] > 120) ||
                        (angs_pre[idx] <= 120 && angs_post[idx] <= 120 &&
                         angs_pre[idx] + angs_post[idx] > 160 && turn_pre[idx] == turn_post[idx])) {
                        if (is_corner_pre == false) {
                            is_corner = true;
                            n_corner++;

                            if (angs_post[idx] > 120) {
                                sum_turns += turn_post[idx];
                                turn_dir.push_back(turn_post[idx]);
                            } else {
                                sum_turns += turn_pre[idx];
                                turn_dir.push_back(turn_pre[idx]);
                            }

                            if (angs_post[idx] > angs_pre[idx]) {
                                Point2<T> tmp_pt =
                                    ordered_vertices_[(idx + 1) % (vertices_per_polygon - c_start) +
                                                      c_start];
                                pt_index.push_back(tmp_pt);
                            } else {
                                Point2<T> tmp_pt =
                                    ordered_vertices_[(idx) % (vertices_per_polygon - c_start) +
                                                      c_start];
                                pt_index.push_back(tmp_pt);
                            }
                        }
                    }
                }
                idx++;
                is_corner_pre = is_corner;
                is_corner = false;
            }

            if ((n_corner == 4) && (sum_turns == 2 || sum_turns == -2)) {
                int i_corner = 0;

                for (int i = 0; i < 4; i++) {
                    if ((sum_turns < 0 && turn_dir[i] == 1) ||
                        (sum_turns > 0 && turn_dir[i] == -1)) {
                        Point2<T> curr = pt_index[i];
                        v_shape_vertices_.push_back(curr);
                        i_corner = i;
                    }
                }

                if (i_corner < 2) {
                    Point2<T> curr = pt_index[i_corner + 2];
                    v_shape_vertices_.push_back(curr);
                } else {
                    Point2<T> curr = pt_index[i_corner - 2];
                    v_shape_vertices_.push_back(curr);
                }

                return check_abnormal(pt_index);
            }
        }

        return false;
    }

    bool closing_required(const Line2<T>& first, const Line2<T>& end) {
        bool yes = false;
        _point3 out;
        _point3 firstL;
        get_line(first.bottom_, first.top_, &firstL);
        _point3 endL;
        get_line(end.bottom_, end.top_, &endL);
        if (cross_product(firstL, endL, &out)) {
            T dist = point_to_point_distance(first.bottom_, Point2<T>(out.x, out.y));
            if (dist > 20) {
                yes = true;
            }
        } else {
            yes = true;
        }
        return yes;
    }

    void correct_order(const std::vector<Line2<T>>& line_segments,
                       std::vector<Line2<T>>* corrected) {
        // make sure overall bottom up.
        (*corrected).clear();
        int vertical_change = line_segments.front().bottom_.y - line_segments.back().top_.y;
        Point2<T> cxcy(width_ * 0.5, height_ * 0.5);
        bool change_required = false;
        if (vertical_change < 0) {
            change_required = true;
        }
        if (change_required) {
            // NOT OK, reverse orders
            int total_lines = line_segments.size();
            for (int m = 0; m < total_lines; m++) {
                Line2<T> line = line_segments[total_lines - 1 - m];
                Point2<T> tmp = line.bottom_;
                line.bottom_ = line.top_;
                line.top_ = tmp;
                line.calculate_angle();
                (*corrected).push_back(line);
            }
        } else {
            (*corrected) = line_segments;
        }
    }

    bool preprocess_vertices(std::vector<Line2<T>>* linesegments, bool is_polygon = false) {
        linesegments->clear();
        bool result = true;
        // check: we need at least 3 vertices to construct a polygon
        if (ordered_vertices_.size() < 3) {
            result = false;
        }
        if (result) {
            Point2<T> prevPts = ordered_vertices_[0];
            T shortest = std::numeric_limits<T>::max() * 0.8f;
            T second_shortest = shortest + 1;
            for (int i = 1; i < static_cast<int>(ordered_vertices_.size()); i++) {
                Point2<T> curPts = ordered_vertices_[i];
                T dist = point_to_point_distance(prevPts, curPts);
                if (dist < shortest) {
                    second_shortest = shortest;
                    shortest = dist;
                } else if (dist < second_shortest) {
                    second_shortest = dist;
                }
            }

            // check turning points....
            if (ordered_vertices_.size() < 3) {
                result = false;
            }
            if (result) {
                bool breakLongLines = false;
                if (ordered_vertices_.size() < 7) {
                    breakLongLines = true;
                }
                // ordered_vertices_=ordered_vertices_;
                int total_filtered = ordered_vertices_.size();
                Point2<T> prev = ordered_vertices_[0];
                for (int j = 1; j < total_filtered; j++) {
                    Point2<T> pt = ordered_vertices_[j];
                    if (breakLongLines) {
                        T dist = point_to_point_distance(prev, pt);
                        if (dist > second_shortest * 1.2) {
                            Point2<T> midpt = mid_point(prev, pt);
                            linesegments->push_back(Line2<T>(prev, midpt, class_id_));
                            linesegments->push_back(Line2<T>(midpt, pt, class_id_));
                        } else {
                            linesegments->push_back(Line2<T>(prev, pt, class_id_));
                        }
                    } else {
                        linesegments->push_back(Line2<T>(prev, pt, class_id_));
                    }
                    prev = pt;
                }
                // check if the first line and the last line could be intersected near to the
                // polygons.
                if ((linesegments->size() >= 2 && is_polyline_ == false &&
                     closing_required(linesegments->front(), linesegments->back()))) {
                    linesegments->push_back(Line2<T>(prev, ordered_vertices_[0], class_id_));
                } else {
                    if (is_polygon) {
                        linesegments->push_back(Line2<T>(prev, ordered_vertices_[0], class_id_));
                    }
                }
                result = true;
            }
        }
        return result;
    }
    T get_dist_ratio(const Line2<T>& A, const Line2<T>& B) {
        T dist1 = point_to_point_distance(A.bottom_, A.top_);
        T dist2 = point_to_point_distance(B.bottom_, B.top_);
        T ratio = 1;
        if (dist1 > dist2) {
            ratio = std::abs(dist1 - dist2) / dist1;
        } else {
            ratio = std::abs(dist1 - dist2) / dist2;
        }
        return ratio;
    }
    bool find_turning_and_remove_close_points() {
        // TODO(minoop): break down this.
        // run this function only once.
        centerline_segments_.clear();
        bool result = true;

        std::vector<Line2<T>> linesegments;

        result = preprocess_vertices(&linesegments);
        if (result) {
            // using three lines to check turning points.
            int total_seg = linesegments.size();
            Line2<T> prevLine;
            std::vector<float> angles;
            std::vector<uint8_t> vertex_turn;
            for (int i = 0; i < total_seg; i++) {
                Line2<T> curLine = linesegments[i];

                if (i == 0) {
                    prevLine = linesegments[total_seg - 1];
                }
                Line2<T> nextLine;
                if (i + 1 < total_seg) {
                    nextLine = linesegments[i + 1];
                } else {
                    nextLine = linesegments[0];
                }
                // check prevLine and curLine.
                float angDiff1 = std::abs(angle_between_two_lines(curLine, nextLine, true)) +
                                 VERTEX_ANGLE_OFFSET;
                // check prevLine and nextLine.
                float angDiff2 = std::abs(angle_between_two_lines(prevLine, nextLine, true));
                if (angDiff1 > angDiff2) {
                    angles.push_back(angDiff1);
                    vertex_turn.push_back(255);
                } else {
                    angles.push_back(angDiff2);
                    vertex_turn.push_back(0);
                }
                prevLine = curLine;
            }
            // find local maxima......
            int total_angles = angles.size();
            float first_max = 0;
            float first_index = 0;
            bool first_vertex_turn = false;
            float second_max = 0;
            float second_index = 0;
            bool second_vertex_turn = false;
            std::vector<int> turning_indices;
            std::vector<int> vertex_turn_indicator;
            for (int i = 0; i < total_angles; i++) {
                int prev_i = i - 1;
                if (i == 0) {
                    prev_i = total_angles - 1;
                }
                int next_i = i + 1;
                if (i == total_angles - 1) {
                    next_i = 0;
                }
                float prevA = angles[prev_i];
                float currA = angles[i];
                float nextA = angles[next_i];

                if (currA > prevA && currA > nextA && currA > TURNING_DEGREE) {  // local maxima
                    // pick the best two local maxima.
                    if (currA > first_max) {
                        second_max = first_max;
                        second_index = first_index;
                        second_vertex_turn = first_vertex_turn;
                        first_max = currA;
                        first_index = i;
                        if (vertex_turn[i] > 0) {
                            first_vertex_turn = true;
                        } else {
                            first_vertex_turn = false;
                        }
                    } else if (currA > second_max) {
                        second_max = currA;
                        second_index = i;
                        if (vertex_turn[i] > 0) {
                            second_vertex_turn = true;
                        } else {
                            second_vertex_turn = false;
                        }
                    }
                }
            }
            if (first_max > 0 && second_max > 0) {
                // we need to preserver drawing order.
                if (first_index < second_index) {
                    turning_indices.push_back(first_index);
                    vertex_turn_indicator.push_back(first_vertex_turn);
                    turning_indices.push_back(second_index);
                    vertex_turn_indicator.push_back(second_vertex_turn);
                } else {
                    turning_indices.push_back(second_index);
                    vertex_turn_indicator.push_back(second_vertex_turn);
                    turning_indices.push_back(first_index);
                    vertex_turn_indicator.push_back(first_vertex_turn);
                }
            }

            if (turning_indices.size() == 2) {
                turning_indices.push_back(turning_indices.front());
                vertex_turn_indicator.push_back(vertex_turn_indicator.front());

                std::vector<std::vector<Line2<T>>> vector_of_lines;
                int total_turning = turning_indices.size();
                int max_len = 0;
                int max_index = -1;
                for (int j = 0; j < total_turning - 1; j++) {
                    // from turning_indices[j] to turning_indices[j+1].
                    std::vector<Line2<T>> lines;
                    bool vertex_turn_end = vertex_turn_indicator[j + 1];

                    int start_index = turning_indices[j];
                    int end_index = turning_indices[j + 1];
                    int curIndex = start_index + 1;
                    curIndex %= total_seg;
                    while (curIndex != end_index) {
                        Line2<T> curLine = linesegments[curIndex];
                        lines.push_back(curLine);
                        curIndex++;
                        curIndex %= total_seg;
                    }
                    if (vertex_turn_end > 0) {
                        lines.push_back(linesegments[end_index]);
                    }
                    if (lines.size() > 0) {
                        vector_of_lines.push_back(lines);
                        if (static_cast<int>(lines.size()) > max_len) {
                            max_len = lines.size();
                            max_index = vector_of_lines.size();
                            max_index -= 1;
                        }
                    }
                }

                // do normal intersection between all max lines with the rest.
                if (max_index != -1) {
                    std::vector<Line2<T>> best;
                    std::vector<Line2<T>> rest;
                    for (int j = 0; j < static_cast<int>(vector_of_lines.size()); j++) {
                        if (j == max_index) {
                            // make sure overall bottom up.
                            correct_order(vector_of_lines[j], &best);
                        } else {
                            rest.insert(rest.end(), vector_of_lines[j].begin(),
                                        vector_of_lines[j].end());
                        }
                    }
                    std::vector<Point2<T>> ptslist;
                    refine_line_segments(&best);
                    int best_length = best.size();
                    for (int m = 0; m < best_length; m++) {
                        Point2<T> bottom = best[m].bottom_;
                        Point2<T> top = best[m].top_;
                        _point3 basisline;
                        if (get_line(bottom, top, &basisline)) {
                            _point3 bottomLine;
                            get_normal_line_passing(best[m], true, &bottomLine);

                            _point3 topLine;
                            get_normal_line_passing(best[m], false, &topLine);

                            Point2<T> selectedT(-1, -1);
                            T minDistanceT = std::numeric_limits<T>::max();
                            Point2<T> selectedB(-1, -1);
                            T minDistanceB = std::numeric_limits<T>::max();

                            for (int j = 0; j < static_cast<int>(rest.size()); j++) {
                                Point2<T> bottom2 = rest[j].bottom_;
                                Point2<T> top2 = rest[j].top_;
                                _point3 line;
                                get_line(bottom2, top2, &line);

                                // compute intersection with top1 and make sure top1.y is within
                                // [top2.y bottom2.y].
                                if (m == best_length - 1) {
                                    check_intersection(topLine, top, line, bottom2, top2, CIO_TOP,
                                                       &selectedT, &minDistanceT);

                                    check_intersection(bottomLine, bottom, line, bottom2, top2,
                                                       CIO_NORMAL, &selectedB, &minDistanceB);
                                } else if (m == 0) {
                                    check_intersection(bottomLine, bottom, line, bottom2, top2,
                                                       CIO_BOTTOM, &selectedB, &minDistanceB);
                                } else {
                                    check_intersection(bottomLine, bottom, line, bottom2, top2,
                                                       CIO_NORMAL, &selectedB, &minDistanceB);
                                }

                                // compute intersection with bottom1 and make sure bottom1.y is
                                // within [top2.y bottom2.y].
                            }
                            if (minDistanceB < std::numeric_limits<T>::max()) {
                                ptslist.push_back(mid_point(bottom, selectedB));
                            }

                            if (minDistanceT < std::numeric_limits<T>::max()) {
                                ptslist.push_back(mid_point(top, selectedT));
                            }

                        } else {
                            printf("No valid line is found.\n");
                            assert(0);
                        }
                    }

                    if (ptslist.size() > 0) {
                        Point2<T> prev = ptslist[0];
                        for (size_t i = 1; i < ptslist.size(); i++) {
                            Point2<T> current = ptslist[i];
                            if (point_to_point_distance(prev, current) >=
                                MIN_DISTANCE_BETWEEN_VERTICES) {
                                float width_left_top = 0;
                                float width_right_top = 0;
                                float width_left_bottom = 0;
                                float width_right_bottom = 0;

                                get_boundary_encoding(prev, current, linesegments, &width_left_top,
                                                      &width_right_top, &width_left_bottom,
                                                      &width_right_bottom);

                                centerline_segments_.push_back(Line2<T>(
                                    prev, current, class_id_, width_left_top, width_right_top,
                                    width_left_bottom, width_right_bottom));

                                prev = current;
                            }
                        }
                    }
                }
            }

            result = true;
        }
        return result;
    }

 protected:
    bool process_linesegments(const int cluster_id) {
        bool result = false;
        if (centerline_segments_.size() > 0) {
            for (size_t i = 0; i < centerline_segments_.size(); i++) {
                centerline_segments_[i].cluster_id_ = cluster_id;
                centerline_segments_[i].angle_ = -1000;
                centerline_segments_[i].angle_bottom_ = -1000;
                centerline_segments_[i].angle_top_ = -1000;
            }
            result = true;
        }
        return result;
    }
    bool refine_line_segments(std::vector<Line2<T>>* linesegments, const int cluster_id = -1) {
        bool result = false;
        size_t num_linesegments = (*linesegments).size();
        // If there is only one line segment, update the cluster id and return.
        if (num_linesegments == 1) {
            Line2<T>& current = (*linesegments)[0];
            current.cluster_id_ = cluster_id;
            result = true;
        } else if (num_linesegments > 1) {
            int weird_angle_count = 0;
            for (size_t i = 1; i < num_linesegments; i++) {
                Line2<T>& before = (*linesegments)[i - 1];
                Line2<T>& after = (*linesegments)[i];
                // check wrong labels.
                T diff_angle = angle_between_two_lines(before, after, true);
                if (diff_angle > WEIRD_ANGLE_DIFF_IN_LINE) {
                    weird_angle_count++;
                }
                if (cluster_id >= 0) {
                    if (i == 1) {
                        before.cluster_id_ = cluster_id;
                    }
                    after.cluster_id_ = cluster_id;
                }

                float avg_angle = Line2<T>::average_angles(before.angle_, after.angle_);
                before.angle_top_ = avg_angle;
                after.angle_bottom_ = avg_angle;
            }
            if (weird_angle_count <= MAX_WEIRD_ANGLE_COUNT_IN_LINE) {
                result = true;
            }
        }
        return result;
    }
    bool refine_line_segments(const int cluster_id) {
        return refine_line_segments(&centerline_segments_, cluster_id);
    }
    bool get_line(const Point2<T>& pt1, const Point2<T>& pt2, _point3* line) {
        _point3 a(pt1);
        _point3 b(pt2);
        return cross_product(a, b, line, false);
    }

    void get_normal_line_passing(const Line2<T>& line, bool bottom, _point3* normalLine) {
        Point2<T> pt = line.top_;
        (*normalLine).x = sin(line.angle_top_ + 0.5f * M_PI);
        (*normalLine).y = -cos(line.angle_top_ + 0.5f * M_PI);
        if (bottom) {
            pt = line.bottom_;
            (*normalLine).x = sin(line.angle_bottom_ + 0.5f * M_PI);
            (*normalLine).y = -cos(line.angle_bottom_ + 0.5f * M_PI);
        }
        (*normalLine).z = -(*normalLine).x * pt.x - (*normalLine).y * pt.y;
    }

    void get_normal_line_passing(const _point3& line, const Point2<T>& pt, _point3* normalLine) {
        (*normalLine).x = -line.y;
        (*normalLine).y = line.x;
        (*normalLine).z = line.y * pt.x - line.x * pt.y;
    }
    bool cross_product(const _point3& a, const _point3& b, _point3* out, bool checkZ = true) {
        (*out).x = a.y * b.z - a.z * b.y;
        (*out).y = a.z * b.x - a.x * b.z;
        (*out).z = a.x * b.y - a.y * b.x;
        bool result = true;
        if (checkZ == true) {
            if (std::abs((*out).z) > FLT_MIN) {
                (*out).x /= (*out).z;
                (*out).y /= (*out).z;
                (*out).z = 1;
            } else {
                result = false;
            }
        }
        return result;
    }

    bool check_proper_intersection(const Point2<T>& bottom, const Point2<T>& top,
                                   const Point2<T>& pt_intersection) {
        bool proper_intersection = false;
        Point2<T> vec1 = pt_intersection - bottom;
        Point2<T> vec2 = pt_intersection - top;
        T mag1 = norm2(vec1);
        T mag2 = norm2(vec2);
        T mag_total = norm2(top.x - bottom.x, top.y - bottom.y);
        // this is just to check if the intersection is on the line or not...
        if (std::abs(mag1 + mag2 - mag_total) < 3) {
            proper_intersection = true;
        }

        return proper_intersection;
    }

    void check_intersection(const _point3& scan_line, const Point2<T>& basis_point_for_scanline,
                            const _point3& line, const Point2<T>& bottom, const Point2<T>& top,
                            CheckIntersectionOption bottom_or_top_end_point,
                            Point2<T>* selected_pts, T* minDistance) {
        // compute intersection with top1 and make sure top1.y is within [top2.y bottom2.y].
        _point3 intersection;
        if (cross_product(scan_line, line, &intersection)) {
            Point2<T> pt_intersection = Point2<T>(intersection.x, intersection.y);
            bool proper_intersection = false;
            if (bottom_or_top_end_point > 0) {
                proper_intersection = true;
            } else {  // if margin is zero we will make sure intersection is on the line....
                proper_intersection = check_proper_intersection(bottom, top, pt_intersection);
            }

            if (proper_intersection) {
                // distance from intersection to normal line.
                T dist = point_to_point_distance(basis_point_for_scanline, pt_intersection);
                if (bottom_or_top_end_point > 0) {
                    // distance from intersection to top point in the other line.
                    T dist_to_top = point_to_point_distance(basis_point_for_scanline, top);
                    // distance from intersection to bottom point in the other line.
                    T dist_to_bottom = point_to_point_distance(basis_point_for_scanline, bottom);

                    if (bottom_or_top_end_point == CIO_BOTTOM) {
                        dist += std::min(dist_to_top, dist_to_bottom);
                    } else {
                        dist = std::min(dist_to_top, dist_to_bottom);
                        if (dist_to_top < dist_to_bottom) {
                            pt_intersection = top;
                        } else {
                            pt_intersection = bottom;
                        }
                    }
                }

                if (dist < *minDistance) {
                    *selected_pts = pt_intersection;
                    *minDistance = dist;
                }
            }
        } else if (bottom_or_top_end_point == CIO_TOP) {
            T dist_to_top = point_to_point_distance(basis_point_for_scanline, top);
            // distance from intersection to bottom point in the other line.
            T dist_to_bottom = point_to_point_distance(basis_point_for_scanline, bottom);

            T dist = std::min(dist_to_top, dist_to_bottom);
            if (dist < *minDistance) {
                if (dist_to_top < dist_to_bottom) {
                    *selected_pts = top;
                } else {
                    *selected_pts = bottom;
                }
                *minDistance = dist;
            }
        }
    }
    int class_id_;   // class ID.
    int attribute_;  // attribute.
    bool processed_;
    bool is_polyline_;
    bool extract_via_horizontal_scanline_;
    std::vector<Point2<T>> v_shape_vertices_;           // V shape splitting vertices.
    std::vector<Point2<T>> ordered_vertices_;           // ordered vertices.
    std::vector<Line2<T>> centerline_segments_;         // centerline segments.
    std::vector<Line2<T>> centerline_segments_before_;  // centerline segments before reordering.
    std::vector<Point2<T>> raw_vertices_;               // unprocessed raw vertices from input.
    std::vector<Point2<T>> centerline_vertices_;        // centerline vertices.
    std::vector<Line2<T>> sorted_segments_;             // sorted segments.
    std::vector<float> width_left_;                     // width to left edges.
    std::vector<float> width_right_;                    // width to right edges.
    Vector3f coeffs_;
    std::vector<int> scan_line_y_;
    std::vector<std::pair<int, std::vector<Point2<T>>>> scanned_pts_;

    int width_;
    int height_;
    T min_y_;
    T max_y_;
};

typedef Polygon2<int> Polygon2i;
typedef Polygon2<float> Polygon2f;

#endif  // _POLYGON_HPP_

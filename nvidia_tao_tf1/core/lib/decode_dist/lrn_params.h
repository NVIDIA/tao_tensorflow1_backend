// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#ifndef LRN_PARAMS_H_
#define LRN_PARAMS_H_

#include <stdio.h>
#include <algorithm>

namespace lineregressordecoder {
typedef enum {
    LRN_VANILLA_ENCODING = 0,
    LRN_VANILLA_DIST2D_ENCODING = 1,
    LRN_LREDGE_ENCODING = 2,
    LRN_LREDGE_DIST2D_ENCODING = 3,
    LRN_TOTAL_ENCODING_TYPE_COUNT
} LRNEncodingType;

class LRNEncodingParam {
 public:
    static constexpr int8_t INVALID = -1;

    LRNEncodingParam() {
        encoding_dist_2D_ = false;
        setToVanilla();
    }
    explicit LRNEncodingParam(LRNEncodingType type) { set(type); }
    /**
     * @brief set LRNEncodingType to let the module know how to decode.
     * @param type LRNEncodingType the list will grow.
     */
    void set(LRNEncodingType type) {
        encoding_type_ = type;
        switch (type) {
            case LRN_VANILLA_ENCODING:
                setToVanilla();
                break;
            case LRN_LREDGE_ENCODING:
                setToEightClassLREdge();
                break;
            case LRN_LREDGE_DIST2D_ENCODING:
                setToEightClassLREdgeDist2D();
                break;
            case LRN_VANILLA_DIST2D_ENCODING:
            case LRN_TOTAL_ENCODING_TYPE_COUNT:
            default:
                setToVanilla();
                break;
        }
    }
    /**
     * @brief Returns channel locations for dx.
     * @return
     */
    int8_t getDXPosition() { return dx_position_; }
    /**
     * @brief Returns channel locations for dy.
     * @return
     */
    int8_t getDYPosition() { return dy_position_; }
    /**
     * @brief Returns channel locations for negative dx.
     * @return
     */
    int8_t getNDXPosition() { return ndx_position_; }
    /**
     * @brief Returns channel locations for negative dy.
     * @return
     */
    int8_t getNDYPosition() { return ndy_position_; }
    /**
     * @brief Returns channel locations for direction encoding by cosine.
     * @return
     */
    int8_t getDirCosPosition() { return dir_cos_position_; }
    /**
     * @brief Returns channel locations for direction encoding by sine.
     * @return
     */
    int8_t getDirSinPosition() { return dir_sin_position_; }
    /**
     * @brief Returns channel locations for bit encoding location.
     * @return
     */
    int8_t getBitStartPosition() { return bit_start_position_; }
    /**
     * @brief Returns channel locations for left edge width.
     * @return
     */
    int8_t getLeftWidthStartPosition() { return edge_width_left_position_; }
    /**
     * @brief Returns channel locations for right edge width.
     * @return
     */
    int8_t getRightWidthStartPosition() { return edge_width_right_position_; }

    /**
     * @brief Returns magnitude position.
     * @return
     */
    int8_t getMagnitudePosition() { return magnitude_position_; }

    /**
     * @brief Returns normal cosine position.
     * @return
     */
    int8_t getNormalCosPosition() { return normal_cos_position_; }

    /**
     * @brief Returns normal sine position.
     * @return
     */
    int8_t getNormalSinePosition() { return normal_sine_position_; }

    /**
     * @brief Returns if encoder is dist2D.
     * @return
     */
    bool isEncoderDist2D() { return encoding_dist_2D_; }

 protected:
    /**
     * LRN_LREDGE_ENCODING specification.
     */
    void setToEightClassLREdge() {
        mask_position_ = 0;
        dx_position_ = 1;
        dy_position_ = 2;
        ndx_position_ = 3;
        ndy_position_ = 4;
        dir_cos_position_ = 5;
        dir_sin_position_ = 6;
        edge_width_left_position_ = 7;
        edge_width_right_position_ = 8;
        bit_start_position_ = 9;
        encoding_type_ = LRN_LREDGE_ENCODING;

        magnitude_position_ = INVALID;
        normal_cos_position_ = INVALID;
        normal_sine_position_ = INVALID;
        encoding_dist_2D_ = false;
    }

    /**
     * LRN_LREDGE_DIST2D_ENCODING specification.
     */
    void setToEightClassLREdgeDist2D() {
        mask_position_ = 0;
        magnitude_position_ = 1;
        normal_cos_position_ = 2;
        normal_sine_position_ = 3;
        dir_cos_position_ = 4;
        dir_sin_position_ = 5;
        edge_width_left_position_ = 6;
        edge_width_right_position_ = 7;
        bit_start_position_ = 8;
        encoding_type_ = LRN_LREDGE_DIST2D_ENCODING;

        dx_position_ = INVALID;
        dy_position_ = INVALID;
        ndx_position_ = INVALID;
        ndy_position_ = INVALID;
        encoding_dist_2D_ = true;
    }
    /**
     * LRN_VANILLA_ENCODING specification.
     */
    void setToVanilla() {
        mask_position_ = 0;
        dx_position_ = 1;
        dy_position_ = 2;
        ndx_position_ = 3;
        ndy_position_ = 4;
        dir_cos_position_ = 5;
        dir_sin_position_ = 6;
        edge_width_left_position_ = INVALID;
        edge_width_right_position_ = INVALID;
        bit_start_position_ = 7;
        encoding_type_ = LRN_VANILLA_ENCODING;

        magnitude_position_ = INVALID;
        normal_cos_position_ = INVALID;
        normal_sine_position_ = INVALID;
        encoding_dist_2D_ = false;
    }
    // 2D dist transform encoding option.
    int8_t magnitude_position_;
    int8_t normal_cos_position_;
    int8_t normal_sine_position_;

    int8_t mask_position_;
    // vx, vy encoding option.
    int8_t dx_position_;
    int8_t dy_position_;
    int8_t ndx_position_;
    int8_t ndy_position_;
    // Common variables for directions, left edge width, right edge width.
    int8_t dir_cos_position_;
    int8_t dir_sin_position_;
    int8_t edge_width_left_position_;
    int8_t edge_width_right_position_;
    int8_t bit_start_position_;

    bool encoding_dist_2D_;
    LRNEncodingType encoding_type_;
};

}  // namespace lineregressordecoder
#endif /* LRN_PARAMS_H_ */

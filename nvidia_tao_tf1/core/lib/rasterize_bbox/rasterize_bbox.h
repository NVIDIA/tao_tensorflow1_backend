// Copyright (c) 2016-2019, NVIDIA CORPORATION.  All rights reserved.
// This file contains common code that can be used between .cc and _gpu.cu.cc
// versions of the rasterize_bbox_op file.
#ifndef _RASTERIZE_BBOX_H_
#define _RASTERIZE_BBOX_H_

#include <algorithm>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#define MAX_NUM_GRADIENTS 16

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#ifdef __CUDACC__
#define CUDA_HOSTDEV __device__
#else
#define CUDA_HOSTDEV
#endif

struct vec2 {
    float x;
    float y;
};

struct vec3 {
    float x;
    float y;
    float z;
};

struct Bbox {
    int32_t image_id;
    int32_t class_id;
    float matrix[3 * 3];
    vec3 gradients[MAX_NUM_GRADIENTS];
    vec2 coverage_radii;
    uint32_t flags;
    float sort_value;
};

static __inline__ CUDA_HOSTDEV void _RasterizeBboxKernel(int x, int y, const Bbox* bboxes,
                                                         const uint32_t gradient_flags,
                                                         int num_bboxes, int width, int height,
                                                         int num_classes, int num_gradients,
                                                         int num_outputs, float* out) {
    // Code common to CPU and GPU kernel
    float px = x;  // Pixel coordinates within image
    float py = y;

    float pg[MAX_NUM_GRADIENTS];
    for (int i = 0; i < num_gradients; i++) {
        pg[i] = 0.0f;
    }
    float max_coverage = 0.0f;

    // bboxes are presorted in back to front order
    for (int i = 0; i < num_bboxes; i++) {
        const float* mat = bboxes[i].matrix;
        bool draw_ellipse = (bboxes[i].flags & 1) != 0;
        float cov_radius_x = 1.0f / bboxes[i].coverage_radii.x;
        float cov_radius_y = 1.0f / bboxes[i].coverage_radii.y;

        // supersample deadzone and coverage areas
        float a = 0.0f;  // deadzone coverage
        float c = 0.0f;  // shape coverage
        const int num_samples = 4;
        const float oo_num_samples = 1.0f / static_cast<float>(num_samples);
        const float sample_cov = 1.0f / static_cast<float>(num_samples * num_samples);
        float fxs = px + 0.5f * oo_num_samples;
        float fy = py + 0.5f * oo_num_samples;

        for (int psy = 0; psy < num_samples; psy++, fy += oo_num_samples) {
            float fx = fxs;
            for (int psx = 0; psx < num_samples; psx++, fx += oo_num_samples) {
                // transform pixel to unit square coordinate system
                float tx = fx * mat[0] + fy * mat[3] + mat[6];
                float ty = fx * mat[1] + fy * mat[4] + mat[7];
                float txc = tx * cov_radius_x;
                float tyc = ty * cov_radius_y;

                if (draw_ellipse) {
                    if (txc * txc + tyc * tyc < 1.0f)
                        c += sample_cov;                            // inside cov area, increment c
                    if (tx * tx + ty * ty < 1.0f) a += sample_cov;  // inside deadzone, increment a
                } else {                                            // draw rectangle
                    if (fabsf(txc) < 1.0f && fabsf(tyc) < 1.0f)
                        c += sample_cov;  // inside cov area, increment c
                    if (fabsf(tx) < 1.0f && fabsf(ty) < 1.0f)
                        a += sample_cov;  // inside deadzone, increment a
                }
            }
        }

        // if the new deadzone has larger coverage than the current maximum, clear the pixel
        if ((a >= max_coverage) && (a > 0.f)) {
            max_coverage = 0.0f;
            for (int j = 0; j < num_gradients; j++) {
                pg[j] = 0.0f;
            }
        }
        // if the new bbox has larger coverage than the current maximum, replace the pixel
        if ((c >= max_coverage) && (c > 0.f)) {
            max_coverage = c;
            for (int j = 0; j < num_gradients; j++) {
                // compute gradient value at current pixel coordinates
                vec3 gmat = bboxes[i].gradients[j];
                float g = px * gmat.x + py * gmat.y + gmat.z;
                // optionally multiply gradient by pixel coverage to smooth edges
                bool cov_mult = ((gradient_flags >> j) & 1) != 0;
                if (cov_mult) g *= c;
                pg[j] = g;
            }
        }
    }
    // write result to output
    int loc = y * width + x;
    for (int j = 0; j < num_gradients; j++) {
        out[j * width * height + loc] = pg[j];
    }
}

// Common class which would be shared by CPU/GPU versions of code
// Any functions which are either CPU or GPU specific are declared as virtual
// and it is upto specific implementations of those files to extend these as
// they see fit
class _RasterizeBboxOp : public OpKernel {
 protected:  // to be accessed in derived classes
    int num_classes;
    bool verbose_;

 public:
    explicit _RasterizeBboxOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("num_classes", &num_classes));
        OP_REQUIRES(context, num_classes >= 1,
                    errors::InvalidArgument("num_classes must be > 0, got ", num_classes));
        OP_REQUIRES_OK(context, context->GetAttr("verbose", &verbose_));
    }

    struct {
        bool operator()(Bbox a, Bbox b) {
            if (a.image_id == b.image_id) {
                if (a.class_id == b.class_id) return a.sort_value < b.sort_value;
                return a.class_id < b.class_id;
            }
            return a.image_id < b.image_id;
        }
    } sortBboxes;

    void ReadInput(int* value, OpKernelContext* context, int input, const char* name) {
        *value = -1;
        const Tensor& input_tensor = context->input(input);
        int num_dims = input_tensor.shape().dims();
        OP_REQUIRES(context, (num_dims == 0) || (num_dims == 1),
                    errors::InvalidArgument(name, " shape should be 1D, got ", num_dims));
        if (num_dims) {
            OP_REQUIRES(context, input_tensor.shape().dim_size(0) == 1,
                        errors::InvalidArgument(name, " dim0 should be of size 1 got ",
                                                input_tensor.shape().dim_size(0)));
        }
        *value = input_tensor.flat<int32_t>().data()[0];
    }

    virtual void ComputeArch(OpKernelContext* context, float* output_images,
                             const std::vector<Bbox>& bboxes, uint32_t combined_gradient_flags,
                             const std::vector<int>& class_indices, int num_images,
                             int output_height, int output_width, int num_gradients,
                             int num_outputs) = 0;

    void Compute(OpKernelContext* context) override {
        int num_images;
        int num_gradients;
        int output_height;
        int output_width;

        // Grab the input tensors
        ReadInput(&num_images, context, 0, "num_images");
        ReadInput(&num_gradients, context, 1, "num_gradients");
        ReadInput(&output_height, context, 2, "output_height");
        ReadInput(&output_width, context, 3, "output_width");

        OP_REQUIRES(context, num_images >= 1,
                    errors::InvalidArgument("num_images must be > 0, got ", num_images));
        OP_REQUIRES(context, num_gradients >= 1 && num_gradients <= MAX_NUM_GRADIENTS,
                    errors::InvalidArgument("num_gradients must be >= 1 and <= ", MAX_NUM_GRADIENTS,
                                            ", got ", num_gradients));
        OP_REQUIRES(context, output_height >= 1,
                    errors::InvalidArgument("output_height must be > 0, got ", output_height));
        OP_REQUIRES(context, output_width >= 1,
                    errors::InvalidArgument("output_width must be > 0, got ", output_width));

        // Check bboxes_per_image shape.
        const Tensor& bboxes_per_image_tensor = context->input(4);
        OP_REQUIRES(context, bboxes_per_image_tensor.shape().dims() == 1,
                    errors::InvalidArgument("bboxes_per_image shape should be 1D, got ",
                                            bboxes_per_image_tensor.shape().dims()));
        OP_REQUIRES(context, bboxes_per_image_tensor.shape().dim_size(0) == num_images,
                    errors::InvalidArgument("bboxes_per_image dim0 should be of size ", num_images,
                                            ", got ", bboxes_per_image_tensor.shape().dim_size(0)));
        const int32_t* bboxes_per_image_ptr = bboxes_per_image_tensor.flat<int32_t>().data();

        // Check bbox_class_ids shape.
        const Tensor& bbox_class_ids_tensor = context->input(5);
        int num_total_bboxes = bbox_class_ids_tensor.shape().dim_size(0);
        OP_REQUIRES(context, bbox_class_ids_tensor.shape().dims() == 1,
                    errors::InvalidArgument("bbox_class_ids shape should be 1D, got ",
                                            bbox_class_ids_tensor.shape().dims()));
        const int32_t* bbox_class_ids_ptr = bbox_class_ids_tensor.flat<int32_t>().data();

        // Check bbox_matrices shape.
        const Tensor& bbox_matrices_tensor = context->input(6);
        if (num_total_bboxes) {
            OP_REQUIRES(context, bbox_matrices_tensor.shape().dims() == 3,
                        errors::InvalidArgument("bbox_matrices shape should be 3D, got ",
                                                bbox_matrices_tensor.shape().dims()));
            OP_REQUIRES(
                context, bbox_matrices_tensor.shape().dim_size(0) == num_total_bboxes,
                errors::InvalidArgument("bbox_matrices dim0 should be of size ", num_total_bboxes,
                                        ", got ", bbox_matrices_tensor.shape().dim_size(0)));
            OP_REQUIRES(context, bbox_matrices_tensor.shape().dim_size(1) == 3,
                        errors::InvalidArgument("bbox_matrices dim1 should be of size 3, got ",
                                                bbox_matrices_tensor.shape().dim_size(1)));
            OP_REQUIRES(context, bbox_matrices_tensor.shape().dim_size(2) == 3,
                        errors::InvalidArgument("bbox_matrices dim2 should be of size 3, got ",
                                                bbox_matrices_tensor.shape().dim_size(2)));
        }
        const float* bbox_matrices_ptr = bbox_matrices_tensor.flat<float>().data();

        // Check bbox_gradients shape.
        const Tensor& bbox_gradients_tensor = context->input(7);
        if (num_total_bboxes) {
            OP_REQUIRES(context, bbox_gradients_tensor.shape().dims() == 3,
                        errors::InvalidArgument("bbox_gradients shape should be 3D, got ",
                                                bbox_gradients_tensor.shape().dims()));
            OP_REQUIRES(
                context, bbox_gradients_tensor.shape().dim_size(0) == num_total_bboxes,
                errors::InvalidArgument("bbox_gradients dim0 should be of size ", num_total_bboxes,
                                        ", got ", bbox_gradients_tensor.shape().dim_size(0)));
            OP_REQUIRES(
                context, bbox_gradients_tensor.shape().dim_size(1) == num_gradients,
                errors::InvalidArgument("bbox_gradients dim1 should be of size ", num_gradients,
                                        ", got ", bbox_gradients_tensor.shape().dim_size(1)));
            OP_REQUIRES(context, bbox_gradients_tensor.shape().dim_size(2) == 3,
                        errors::InvalidArgument("bbox_gradients dim2 should be of size 3, got ",
                                                bbox_gradients_tensor.shape().dim_size(2)));
        }
        const float* bbox_gradients_ptr = bbox_gradients_tensor.flat<float>().data();

        // Check bbox_coverage_radii shape.
        const Tensor& bbox_coverage_radii_tensor = context->input(8);
        if (num_total_bboxes) {
            OP_REQUIRES(context, bbox_coverage_radii_tensor.shape().dims() == 2,
                        errors::InvalidArgument("bbox_coverage_radii shape should be 2D, got ",
                                                bbox_coverage_radii_tensor.shape().dims()));
            OP_REQUIRES(context, bbox_coverage_radii_tensor.shape().dim_size(0) == num_total_bboxes,
                        errors::InvalidArgument("bbox_coverage_radii dim0 should be of size ",
                                                num_total_bboxes, ", got ",
                                                bbox_coverage_radii_tensor.shape().dim_size(0)));
            OP_REQUIRES(
                context, bbox_coverage_radii_tensor.shape().dim_size(1) == 2,
                errors::InvalidArgument("bbox_coverage_radii dim1 should be of size 2, got ",
                                        bbox_coverage_radii_tensor.shape().dim_size(1)));
        }
        const float* bbox_coverage_radii_ptr = bbox_coverage_radii_tensor.flat<float>().data();

        // Check bbox_flags shape.
        const Tensor& bbox_flags_tensor = context->input(9);
        if (num_total_bboxes) {
            OP_REQUIRES(context, bbox_flags_tensor.shape().dims() == 1,
                        errors::InvalidArgument("bbox_flags shape should be 1D, got ",
                                                bbox_flags_tensor.shape().dims()));
            OP_REQUIRES(
                context, bbox_flags_tensor.shape().dim_size(0) == num_total_bboxes,
                errors::InvalidArgument("bbox_flags dim0 should be of size ", num_total_bboxes,
                                        ", got ", bbox_flags_tensor.shape().dim_size(0)));
        }
        const uint8_t* bbox_flags_ptr = bbox_flags_tensor.flat<uint8>().data();

        // Check bbox_sort_value shape.
        const Tensor& bbox_sort_values_tensor = context->input(10);
        if (num_total_bboxes) {
            OP_REQUIRES(context, bbox_sort_values_tensor.shape().dims() == 1,
                        errors::InvalidArgument("bbox_sort_values shape should be 1D, got ",
                                                bbox_sort_values_tensor.shape().dims()));
            OP_REQUIRES(context, bbox_sort_values_tensor.shape().dim_size(0) == num_total_bboxes,
                        errors::InvalidArgument("bbox_sort_values dim0 should be of size ",
                                                num_total_bboxes, ", got ",
                                                bbox_sort_values_tensor.shape().dim_size(0)));
        }
        const float* bbox_sort_values_ptr = bbox_sort_values_tensor.flat<float>().data();

        // Check gradient_flags shape.
        const Tensor& gradient_flags_tensor = context->input(11);
        auto gradient_flags = gradient_flags_tensor.flat<uint8_t>();
        OP_REQUIRES(context, gradient_flags_tensor.shape().dims() == 1,
                    errors::InvalidArgument("gradient_flags shape should be 1D, got ",
                                            gradient_flags_tensor.shape().dims()));
        OP_REQUIRES(context, gradient_flags_tensor.shape().dim_size(0) == num_gradients,
                    errors::InvalidArgument("gradient_flags dim0 should be of size ", num_gradients,
                                            ", got ", gradient_flags_tensor.shape().dim_size(0)));

        // Collect gradient flags to one uint32.
        const uint8_t* gradient_flags_ptr = gradient_flags.data();
        uint32_t combined_gradient_flags = 0u;
        for (int i = 0; i < num_gradients; i++) {
            combined_gradient_flags |= (gradient_flags_ptr[i] & 1u) << i;
        }

        // Create an output tensor.
        TensorShape output_shape(
            {num_images, num_classes, num_gradients, output_height, output_width});
        if (verbose_) {
            for (int i = 0; i < output_shape.dims(); i++) {
                LOG(INFO) << "output dim " << i << " size = " << output_shape.dim_size(i);
            }
        }
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

        // Populate bboxes array.
        std::vector<Bbox> bboxes(num_total_bboxes);
        for (int i = 0; i < num_total_bboxes; i++) {
            int cl = bbox_class_ids_ptr[i];
            OP_REQUIRES(context, cl >= 0 && cl < num_classes,
                        errors::InvalidArgument("bboxes class ID out of range [0, ", num_classes,
                                                "[, got", cl));
            bboxes[i].class_id = cl;
            for (int j = 0; j < 3 * 3; j++) {
                bboxes[i].matrix[j] = bbox_matrices_ptr[i * 3 * 3 + j];
            }
            for (int j = 0; j < num_gradients; j++) {
                bboxes[i].gradients[j].x = bbox_gradients_ptr[(i * num_gradients + j) * 3 + 0];
                bboxes[i].gradients[j].y = bbox_gradients_ptr[(i * num_gradients + j) * 3 + 1];
                bboxes[i].gradients[j].z = bbox_gradients_ptr[(i * num_gradients + j) * 3 + 2];
            }
            bboxes[i].coverage_radii.x = bbox_coverage_radii_ptr[i * 2 + 0];
            bboxes[i].coverage_radii.y = bbox_coverage_radii_ptr[i * 2 + 1];
            bboxes[i].flags = bbox_flags_ptr[i];
            bboxes[i].sort_value = bbox_sort_values_ptr[i];
        }

        // Set bbox image_ids.
        int bb = 0;
        for (int i = 0; i < num_images; i++) {
            for (int j = 0; j < bboxes_per_image_ptr[i]; j++, bb++) {
                if (bb < num_total_bboxes)  // Avoid crash.
                    bboxes[bb].image_id = i;
            }
        }

        OP_REQUIRES(context, bb == num_total_bboxes,
                    errors::InvalidArgument("The sum of bboxes in bboxes_per_image does not match "
                                            "the number of bboxes in bbox_class_ids array ( ",
                                            bb, " != ", num_total_bboxes, ")"));

        // Sort bboxes by a) image id b) class id c) supplied sort value.
        std::sort(bboxes.begin(), bboxes.end(), sortBboxes);

        int num_outputs = num_images * num_classes;
        if (verbose_) {
            LOG(INFO) << "num_outputs = " << num_outputs << " num_bboxes = " << bboxes.size();
        }

        // Compute where each image&class is within the sorted bbox array.
        int bbox_base = 0;
        std::vector<int> class_indices(num_outputs * 2);
        for (int im = 0, i = 0; im < num_images; im++) {
            int num_bboxes = bboxes_per_image_ptr[im];
            int curr_bb = bbox_base;
            for (int cl = 0; cl < num_classes; cl++, i++) {
                int bbox_base_of_this_class = curr_bb;
                int num_bboxes_of_this_class = 0;
                while (curr_bb < bbox_base + num_bboxes) {
                    if (bboxes[curr_bb].class_id != cl) break;
                    curr_bb++;
                    num_bboxes_of_this_class++;
                }
                class_indices[i * 2 + 0] = bbox_base_of_this_class;
                class_indices[i * 2 + 1] = num_bboxes_of_this_class;
            }
            bbox_base += num_bboxes;
        }

        // Hand off computation to Arch (GPU/CPU) specific functions.
        float* output_ptr = output_tensor->flat<float>().data();
        ComputeArch(context, output_ptr, bboxes, combined_gradient_flags, class_indices, num_images,
                    output_height, output_width, num_gradients, num_outputs);

        if (verbose_) {
            LOG(INFO) << "done";
        }
    }
};  // end __RasterizeBboxOp class

#endif  // _RASTERIZE_BBOX_H_

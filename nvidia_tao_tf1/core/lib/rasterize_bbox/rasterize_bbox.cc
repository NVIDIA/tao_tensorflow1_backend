// Copyright (c) 2016-2019, NVIDIA CORPORATION.  All rights reserved.
#undef EIGEN_USE_GPU

// Bring in common code
#include "rasterize_bbox.h"

REGISTER_OP("RasterizeBbox")
    .Input("num_images: int32")
    .Input("num_gradients: int32")
    .Input("image_height: int32")
    .Input("image_width: int32")
    .Input("bboxes_per_image: int32")
    .Input("bbox_class_ids: int32")
    .Input("bbox_matrices: float")
    .Input("bbox_gradients: float")
    .Input("bbox_coverage_radii: float")
    .Input("bbox_flags: uint8")
    .Input("bbox_sort_values: float")
    .Input("gradient_flags: uint8")
    .Output("output_image: float")
    .Attr("num_classes: int")
    .Attr("verbose: bool = false")
    .SetShapeFn([](::shape_inference::InferenceContext* c) {
        // Inherit the shape from the attributes
        int num_classes;
        TF_RETURN_IF_ERROR(c->GetAttr("num_classes", &num_classes));
        std::vector<::shape_inference::DimensionHandle> dims;
        dims.push_back(c->UnknownDim());          // N
        dims.push_back(c->MakeDim(num_classes));  // C
        dims.push_back(c->UnknownDim());          // G
        dims.push_back(c->UnknownDim());          // H
        dims.push_back(c->UnknownDim());          // W
        c->set_output(0, c->MakeShape(dims));
        return Status::OK();
    })
    .Doc(R"doc(
           BBox rasterizer.
           Bboxes are drawn in the order they are specified, ie. a later bbox is drawn on top of
             an earlier one.
           num_images: 1D tensor with length of 1 that describes the number of output images
             (= batch size N). The value must be >= 1.
           num_gradients: 1D tensor with length of 1 that describes the number of output gradients
             G. The value must be >= 1.
           image_height: 1D tensor with length of 1 that describes the height H of output images.
             The value must be >= 1.
           image_width: 1D tensor with length of 1 that describes the width W of output images. The
             value must be >= 1.
           bboxes_per_image: 1D int32 tensor of length N. Contains number of bboxes in each image.
           bbox_class_ids: 1D int32 tensor of length B (=total number of bboxes to draw). Contains
             a class ID for each bbox. Class ID must be a monotonically increasing value within
             each image.
           bbox_matrices: 3D float32 tensor of shape (B,3,3). Contains a 3x3 row major matrix that
             specifies the shape of each bbox to be drawn. The third column of the matrix is
             implicitly taken to be [0,0,1] (ie. the actual values in that column are ignored).
             In rectangle drawing mode, pixel coordinates form a row vector P=[px,py,1] that is
             multiplied by the matrix M from the right: Q = P M. The resulting coordinates Q that
             end up within the unit square around the origin (ie. Q is within [-1,1] range) are
             considered to be inside deadzone, and the Q that satisfy |Q.x| < coverage_radii.x AND
             |Q.y| < coverage_radii.y are considered to be inside coverage zone. Pixels inside
             coverage zone are drawn with coverage value 1.0, and pixels outside coverage zone but
             inside deadzone are drawn with 0.0.
             In ellipse mode, the unit square is replaced by the unit circle. Pixels inside
             coverage zone satisfy (Q.x/coverage_radii.x)^2 + (Q.y/coverage_radii.y)^2 < 1.
           bbox_gradients: 3D float32 tensor of shape (B,G,3). Contains three gradient coefficients
             A, B, and C for each bbox and gradient. Used for computing a gradient value based on
             pixel coordinates using the gradient function A*px+B*py+C.
           bbox_coverage_radii: 2D float32 tensor of shape (B, 2).
           bbox_flags: 1D uint8 tensor of length B. Contains per bbox flags. Currently the only
             supported flags choose between rectangle mode (=0) and ellipse mode (=1).
           bbox_sort_values: 1D float32 tensor of length B. Contains bbox sort values that
             define bbox drawing order within each image and class (the order is ascending:
             the bbox with the smallest sort value is drawn first). In case of equal values (eg.
             all zeros), bboxes are drawn in the input order.
           gradient_flags: 1D uint8 tensor of length G. Contains per gradient flags. Currently the
             only supported flag chooses whether a particular gradient value should be multiplied
             by coverage value or not.
           output_image: 5D tensor with shape (N, C, G, H, W).
           num_classes: integer attribute that describes the number of output classes C. The value
             must be >= 1.
           verbose: bool for printing out debugging info.
           )doc");

class RasterizeBboxOp : public _RasterizeBboxOp {
 public:
    explicit RasterizeBboxOp(OpKernelConstruction* context) : _RasterizeBboxOp(context) {}

    // CPU specific Computation code
    void ComputeArch(OpKernelContext* context, float* output_images,
                     const std::vector<Bbox>& bboxes, uint32_t combined_gradient_flags,
                     const std::vector<int>& class_indices, int num_images, int output_height,
                     int output_width, int num_gradients, int num_outputs) override {
        if (verbose_) {
            LOG(INFO) << "running CPU kernel";
        }

        for (int c = 0; c < num_outputs; c++) {
            int im = c / num_classes;
            int cl = c % num_classes;
            int bbox_base = class_indices[c * 2 + 0];
            int num_bboxes = class_indices[c * 2 + 1];
            int o = im * num_classes + cl;

            const Bbox* bboxes_data = bboxes.data() + bbox_base;

            int wxh = output_width * output_height;
            float* out_data = output_images + o * num_gradients * wxh;

            for (int y = 0; y < output_height; y++) {
                for (int x = 0; x < output_width; x++) {
                    _RasterizeBboxKernel(x, y, bboxes_data, combined_gradient_flags, num_bboxes,
                                         output_width, output_height, num_classes, num_gradients,
                                         num_outputs, out_data);
                }
            }
        }
    }
};

#pragma message("Registering CPU kernel")
REGISTER_KERNEL_BUILDER(Name("RasterizeBbox").Device(DEVICE_CPU), RasterizeBboxOp);

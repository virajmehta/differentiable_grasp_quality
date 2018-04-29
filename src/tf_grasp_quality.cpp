#include "grasp_quality.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "third_party/eigen3/Eigen/Core"
#include "Eigen/Geometry"
#include "ClpSimplex.hpp"
#include <vector>
#include <cmath>
#include <random>
#include <cstring>

using namespace tensorflow;

typedef ::tensorflow::shape_inference::ShapeHandle ShapeHandle;

REGISTER_OP("GraspQuality")
    //.Attr("grasps_per_shape: int = 10")
    .Attr("num_grasp_perturbations: int = 10")
    .Attr("mu: float = 0.5")
    .Attr("num_cone_edges: int = 11") // for best results, must be prime
    .Input("grasps: float32")
    .Input("vertices: float32")
    .Input("triangles: int32")
    .Input("vertex_normals: float32")
    .Output("grasps_out: float32")
    .Output("vertices_out: float32")
    .Output("triangles_out: int32")
    .Output("quality: float32")
    .Output("triangle_nums: int32")
    .Output("lp_vars: float32")
    .Output("wrenches: float32")
    .Output("centroids: float32")
    .Output("perturbations: float32")
    .Output("vertex_normals_out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ShapeHandle grasps; //batch, ngrasps,6
      ShapeHandle vertices; //batch, maxvertices,3
      ShapeHandle triangles; //batch, maxtris, 3
      ShapeHandle normals; //batch, maxvertices, 3
      ::tensorflow::shape_inference::DimensionHandle tmp;
      ShapeHandle out_qual; //batch, ngrasps
      ShapeHandle out_tri_num;//batch, ngrasps, nsamples, 2
      ShapeHandle out_lp_vars;//batch, ngrasps, nsamples, (7 + num_cone_edges * 2)
      ShapeHandle out_wrenches;//batch, ngrasps, nsamples, (num_cone_edges * 2), 6
      ShapeHandle out_centroids;//batch, 3
      ShapeHandle out_perts;
      //batch, ngrasps, nSamples, kGraspDim
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &grasps));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &vertices));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &triangles));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 3, &normals));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(grasps, 2), kGraspDim, &tmp));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(vertices, 2), 3, &tmp));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(triangles, 2), 3, &tmp));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(normals, 2), 3, &tmp));
      int num_samples;
      int num_edges;
      TF_RETURN_IF_ERROR(c->GetAttr("num_grasp_perturbations", &num_samples));
      TF_RETURN_IF_ERROR(c->GetAttr("num_cone_edges", &num_edges));
      c->Subshape(grasps, 0, 2, &out_qual);
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      c->set_output(2, c->input(2));
      c->set_output(3, out_qual);
      ShapeHandle tri_end = c->MakeShape({num_samples, 2});
      TF_RETURN_IF_ERROR(c->Concatenate(out_qual, tri_end, &out_tri_num));
      c->set_output(4, out_tri_num);
      ShapeHandle lp_end = c->MakeShape({num_samples, (7 + num_edges * 2)});
      TF_RETURN_IF_ERROR(c->Concatenate(out_qual, lp_end, &out_lp_vars));
      c->set_output(5, out_lp_vars);
      ShapeHandle wrench_end = c->MakeShape({num_samples, (num_edges * 2), 6});
      TF_RETURN_IF_ERROR(c->Concatenate(out_qual, wrench_end, &out_wrenches));
      c->set_output(6, out_wrenches);
      ShapeHandle pert_end = c->MakeShape({num_samples, kGraspDim});
      TF_RETURN_IF_ERROR(c->Concatenate(out_qual, pert_end, &out_perts));
      c->set_output(8, out_perts);
      c->Subshape(grasps, 0, 1, &out_qual);
      ShapeHandle cent_end = c->MakeShape({3});
      TF_RETURN_IF_ERROR(c->Concatenate(out_qual, cent_end, &out_centroids));
      c->set_output(7, out_centroids);
      c->set_output(9, c->input(3));
      return Status::OK();
    });

REGISTER_OP("GraspQualityHelper")
    .Input("grasps: float32")
    .Input("vertices: float32")
    .Input("triangles: int32")
    .Input("quality: float32")
    .Input("triangle_nums: int32")
    .Input("lp_vars: float32")
    .Input("wrenches: float32")
    .Input("centroids: float32")
    .Input("perturbations: float32")
    .Input("vertex_normals: float32")
    .Output("quality_out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ShapeHandle grasps; //batch, ngrasps,6
      ShapeHandle vertices; //batch, maxvertices, 3
      ShapeHandle triangles; //batch, maxtriangles, 3
      ShapeHandle quality;
      ShapeHandle triangle_nums; //batch, ngrasps, nperts, 2
      ShapeHandle lp_vars; //batch, ngrasps, nperts, 2 * conevertices + 7
      ShapeHandle wrenches; //batch, ngrasps, nperts, 2 * conevertices, 6
      ShapeHandle centroids; //batch, 3
      ShapeHandle perturbations; //batch, ngrasps, nperts, 6
      ShapeHandle normals; //batch, maxtriangles, 3
      c->set_output(0, c->input(3));

      ::tensorflow::shape_inference::DimensionHandle tmp;

      //batch, ngrasps, nSamples, kGraspDim
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &grasps));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &vertices));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &triangles));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 4, &triangle_nums));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 4, &lp_vars));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 5, &wrenches));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 2, &centroids));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 4, &perturbations));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(9), 3, &normals));

      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(grasps, 2), kGraspDim, &tmp));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(vertices, 2), 3, &tmp));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(triangles, 2), 3, &tmp));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(triangle_nums, 3), 2, &tmp));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(wrenches, 4), 6, &tmp));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(centroids, 1), 3, &tmp));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(perturbations, 3), kGraspDim, &tmp));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(normals, 2), 3, &tmp));
      return Status::OK();
    });

REGISTER_OP("GraspQualGrad")
    .Attr("mu: float = 0.5")
    .Input("grasps: float32")
    .Input("vertices: float32")
    .Input("triangles: int32")
    .Input("quality: float32")
    .Input("triangle_nums: int32")
    .Input("lp_vars: float32")
    .Input("wrenches: float32")
    .Input("centroids: float32")
    .Input("perturbations: float32")
    .Input("vertex_normals: float32")
    .Output("gradient: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ShapeHandle grasps; //batch, ngrasps,6
      ShapeHandle vertices; //batch, maxvertices, 3
      ShapeHandle triangles; //batch, maxtriangles, 3
      ShapeHandle quality;
      ShapeHandle triangle_nums; //batch, ngrasps, nperts, 2
      ShapeHandle lp_vars; //batch, ngrasps, nperts, 2 * conevertices + 7
      ShapeHandle wrenches; //batch, ngrasps, nperts, 2 * conevertices, 6
      ShapeHandle centroids; //batch, 3
      ShapeHandle perturbations; //batch, ngrasps, nperts, 6
      ShapeHandle normals;  //batch, maxtriangles, 3
      c->set_output(0, c->input(0));

      ::tensorflow::shape_inference::DimensionHandle tmp;

      //batch, ngrasps, nSamples, kGraspDim
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &grasps));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &vertices));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &triangles));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 4, &triangle_nums));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 4, &lp_vars));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 5, &wrenches));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 2, &centroids));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 4, &perturbations));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(9), 3, &normals));

      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(grasps, 2), kGraspDim, &tmp));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(vertices, 2), 3, &tmp));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(triangles, 2), 3, &tmp));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(triangle_nums, 3), 2, &tmp));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(wrenches, 4), 6, &tmp));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(centroids, 1), 3, &tmp));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(perturbations, 3), kGraspDim, &tmp));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(normals, 2), 3, &tmp));
      return Status::OK();
    });




class GraspQualityOp : public OpKernel {
  public:
    explicit GraspQualityOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context,
                       context->GetAttr("num_grasp_perturbations",
                       &num_grasp_perturbations_));
        OP_REQUIRES(context, num_grasp_perturbations_ > 0,
               errors::InvalidArgument("Need num_grasp_perturbations > 0, got ",
               num_grasp_perturbations_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("mu",
                       &mu_));
        OP_REQUIRES(context, mu_ > 0,
               errors::InvalidArgument("Need mu > 0, got ",
               mu_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("num_cone_edges",
                       &num_cone_edges_));
        OP_REQUIRES(context, num_cone_edges_ > 0,
               errors::InvalidArgument("Need num_cone_edges > 0, got ",
               num_cone_edges_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& grasps = context->input(0);
        const Tensor& vertices = context->input(1);
        const Tensor& triangles = context->input(2);
        const Tensor& normals = context->input(3);
        TensorShape shape = TensorShape(grasps.shape());
        int batch_size = shape.dim_size(0);
        this->batch_size_ = batch_size;
        int grasps_per_batch = shape.dim_size(1);
        int max_vertices = vertices.shape().dim_size(1);
        int max_triangles = triangles.shape().dim_size(1);
        shape.RemoveDim(2);
        TensorShape tri_shape;
        int32 tri_dims[] = {batch_size, grasps_per_batch,
                                    this->num_grasp_perturbations_, 2};
        TensorShapeUtils::MakeShape(tri_dims, 4, &tri_shape);
        TensorShape lp_shape;
        int32 lp_dims[] = {batch_size, grasps_per_batch,
            this->num_grasp_perturbations_, 7 + this->num_cone_edges_ * 2};
        TensorShapeUtils::MakeShape(lp_dims, 4, &lp_shape);
        TensorShape wrench_shape;
        int32 wrench_dims[] = {batch_size, grasps_per_batch,
            this->num_grasp_perturbations_, this->num_cone_edges_ * 2, 6};
        TensorShapeUtils::MakeShape(wrench_dims, 5, &wrench_shape);
        TensorShape cent_shape;
        int32 cent_dims[] = {batch_size, 3};
        TensorShapeUtils::MakeShape(cent_dims, 2, &cent_shape);
        TensorShape pert_shape;
        int32 pert_dims[] = {batch_size, grasps_per_batch,
                            this->num_grasp_perturbations_, kGraspDim};
        TensorShapeUtils::MakeShape(pert_dims, 4, &pert_shape);

        Tensor* grasp_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, grasps.shape(),
                                                        &grasp_tensor));
        if(!grasp_tensor->CopyFrom(grasps, grasps.shape())) {
            return;
        }
        Tensor* vertex_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, vertices.shape(),
                                                        &vertex_tensor));
        if(!vertex_tensor->CopyFrom(vertices, vertices.shape())) {
            return;
        }
        Tensor* triangle_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(2, triangles.shape(),
                                                        &triangle_tensor));
        if(!triangle_tensor->CopyFrom(triangles, triangles.shape())) {
            return;
        }
        Tensor* normal_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(9, normals.shape(),
                                                        &normal_tensor));
        if(!normal_tensor->CopyFrom(normals, normals.shape())) {
            return;
        }
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(3, shape,
                                                         &output_tensor));
        Tensor* tri_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(4, tri_shape,
                                                         &tri_tensor));
        TriTensor tri_e = tri_tensor->tensor<int, 4>();
        Tensor* lp_tensor= NULL;
        OP_REQUIRES_OK(context, context->allocate_output(5, lp_shape,
                                                         &lp_tensor));
        LpTensor lp_e = lp_tensor->tensor<float, 4>();
        Tensor* wrench_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(6, wrench_shape,
                                                         &wrench_tensor));
        WrenchTensor wrench_e = wrench_tensor->tensor<float, 5>();
        Tensor* cent_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(7, cent_shape,
                                                         &cent_tensor));
        CenTensor cent_e = cent_tensor->tensor<float, 2>();
        Tensor* pert_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(8, pert_shape,
                                                         &pert_tensor));
        PertTensor pert_e = pert_tensor->tensor<float, 4>();

        TensorMap flat_output = output_tensor->flat<float>();
        ConstTensorMap grasp_list = grasps.flat_inner_dims<float>();
        float grasp[kGraspDim];
        for (int b_num = 0; b_num < batch_size; ++b_num) {
            const MapN3 b_vertices(&(vertices.flat<float>().data()[
                                                    b_num * max_vertices *3]),
                                                    max_vertices, 3);
            const IntMapN3 b_triangles(&(triangles.flat<int>().data()[
                                                  b_num * max_triangles * 3]),
                                                  max_triangles, 3);
            const MapN3 b_normals(&(normals.flat<float>().data()[
                                                    b_num * max_vertices *3]),
                                                    max_vertices, 3);
            int num_triangles = find_num_triangles(b_triangles);
            const Eigen::Vector3f centroid = compute_centroid(b_vertices,
                                                            b_triangles);
            for (int i = 0; i < 3; ++i) {
                cent_e(b_num, i) = centroid(i);
            }
            for (int g_num = 0; g_num < grasps_per_batch; ++g_num) {
                int i = b_num * grasps_per_batch + g_num;
                for (int j = 0; j < kGraspDim; ++j) {
                    grasp[j] = grasp_list(i, j);
                }
                handle_one_grasp(grasp, b_vertices, b_triangles, b_normals,
                                      flat_output, this->num_grasp_perturbations_,
                                      this->num_cone_edges_, this->mu_, i,
                                      b_num, g_num, centroid, num_triangles,
                                      tri_e, lp_e, wrench_e, pert_e);
            }
        }
    }

  private:
    int num_grasp_perturbations_;
    float mu_;
    int num_cone_edges_;
    int batch_size_;

};

REGISTER_KERNEL_BUILDER(Name("GraspQuality").Device(DEVICE_CPU),GraspQualityOp);


class GraspQualityHelperOp : public OpKernel {
  public:
    explicit GraspQualityHelperOp(OpKernelConstruction* context) : OpKernel(context) {
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& quality = context->input(3);
        Tensor* quality_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, quality.shape(),
                                                        &quality_tensor));
        if(!quality_tensor->CopyFrom(quality, quality.shape())) {
            return;
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("GraspQualityHelper").Device(DEVICE_CPU),GraspQualityHelperOp);


class GraspQualGradOp : public OpKernel {
  public:
    explicit GraspQualGradOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context,
                       context->GetAttr("mu",
                       &mu_));
        OP_REQUIRES(context, mu_ > 0,
               errors::InvalidArgument("Need mu > 0, got ",
               mu_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& grasps = context->input(0);
        const Tensor& vertices = context->input(1);
        const Tensor& triangles = context->input(2);
        const Tensor& triangle_nums = context->input(4);
        const Tensor& lp_vars = context->input(5);
        const Tensor& wrenches = context->input(6);
        const Tensor& centroids = context->input(7);
        const Tensor& perturbations = context->input(8);
        const Tensor& normals = context->input(9);
        Tensor* grad_tensor = NULL; // batch_size * grasps_per_batch * 6
        OP_REQUIRES_OK(context, context->allocate_output(0, grasps.shape(),
                                                        &grad_tensor));
        int batch_size = grasps.dim_size(0);
        int grasps_per_batch = grasps.dim_size(1);
        int num_perturbations = perturbations.dim_size(2);
        int max_vertices  = vertices.dim_size(1);
        int max_triangles = triangles.dim_size(1);
        int lp_dim = lp_vars.dim_size(3);
        int facets = wrenches.dim_size(3);
        for (int b_num = 0; b_num < batch_size; ++b_num) {
            const MapN3 b_vertices(&(vertices.flat<float>().data()[
                                                    b_num * max_vertices *3]),
                                                    max_vertices, 3);
            const IntMapN3 b_triangles(&(triangles.flat<int>().data()[
                                                  b_num * max_triangles * 3]),
                                                  max_triangles, 3);
            const MapN3 b_normals(&(normals.flat<float>().data()[
                                                  b_num * max_vertices * 3]),
                                                  max_vertices, 3);
            const Map3 b_centroid(&(centroids.flat<float>().data()[
                                        b_num * 3]), 3);


            for (int g_num = 0; g_num < grasps_per_batch; ++g_num) {
                int grasp_num = (b_num * grasps_per_batch) + g_num;
                int wrenches_start = grasp_num * facets * num_perturbations * 6;
                const IntMapN2 tri_nums(&(triangle_nums.flat<int>().data()[
                                grasp_num * num_perturbations * 2]), num_perturbations, 2);
                const MapNN lp(&(lp_vars.flat<float>().data()[
                                grasp_num * num_perturbations * lp_dim]), 
                        num_perturbations, lp_dim);
                const SubWrenchTensor wrench(&(wrenches.flat<float>().data()[wrenches_start]), 
                        num_perturbations,facets, 6);
                const MapN6 perts(&(perturbations.flat<float>().data()[
                                ((b_num * grasps_per_batch) + g_num) *
                            num_perturbations * 6]), num_perturbations, 6);

                float grasp[6];
                for (int j = 0; j < kGraspDim; ++j) {
                    grasp[j] = grasps.flat<float>()((b_num * grasps_per_batch +
                            g_num) * kGraspDim + j);
                }
                Map6 grads(&(grad_tensor->flat<float>().data()[
                                ((b_num * grasps_per_batch) + g_num) * 6]));
                handle_one_grasp_grad(b_vertices, b_triangles, b_centroid, b_normals,
                            tri_nums, lp, wrench, perts, grads, this->mu_, grasp);
            }

        }
    }
   private:
    float mu_;
};

REGISTER_KERNEL_BUILDER(Name("GraspQualGrad").Device(DEVICE_CPU),GraspQualGradOp);


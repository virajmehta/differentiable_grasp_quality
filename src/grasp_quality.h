/* tf_grasp_quality.h: header file STRICTLY FOR TESTING
 * the tensorflow grasp quality op
 * only exposes test main.
 * In general you'd want to compile the op as a shared library and use it in a
 * tensorflow graph.
 * See tf_grasp_quality.py for a usage example as we use the file and define
 * gradients there.
 *
 * Viraj Mehta, 2017
 */

#ifndef TF_GRASP_QUALITY_H
#define TF_GRASP_QUALITY_H

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "third_party/eigen3/Eigen/Core"

static const int kGraspDim = 6;

typedef Eigen::Vector3f Vec3;
typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Matrix33;
typedef Eigen::Matrix<float, 1, kGraspDim, Eigen::RowMajor> RowVec6;
typedef Eigen::Matrix<float, 6, 1, Eigen::ColMajor> Vec6;
typedef Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> MatrixN3;
typedef Eigen::Matrix<float, 3, Eigen::Dynamic, Eigen::RowMajor> Matrix3N;
typedef Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> IntMatrixN3;
typedef Eigen::Matrix<float, Eigen::Dynamic, 6, Eigen::RowMajor> MatrixN6;
typedef Eigen::Matrix<float, Eigen::Dynamic, 7, Eigen::RowMajor> MatrixN7;
typedef Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor> MatrixN2;
typedef Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> IntMatrixN2;
typedef Eigen::Matrix<float, 6, 6, Eigen::RowMajor> Matrix66;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixNN;
typedef Eigen::Map<const Eigen::Vector3f> Map3;
typedef Eigen::Map<Eigen::Matrix<float, 6, 1, Eigen::ColMajor>> Map6;
typedef Eigen::TensorMap<Eigen::Tensor<int, 4, Eigen::RowMajor, long int>, 16> TriTensor;
typedef Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor, long int>, 16> LpTensor;
typedef Eigen::TensorMap<Eigen::Tensor<float, 5, Eigen::RowMajor, long int>, 16> WrenchTensor;
typedef Eigen::TensorMap<Eigen::Tensor<const float, 3, Eigen::RowMajor, long int>, 16> SubWrenchTensor;
typedef Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor, long int>, 16> CenTensor;
typedef Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor, long int>, 16> PertTensor;
typedef Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor, long int>, 16> GradTensor;
typedef Eigen::Map<const MatrixN3> MapN3;
typedef Eigen::Map<const IntMatrixN3> IntMapN3;
typedef Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long int>, 16> TensorMap;
typedef Eigen::TensorMap<Eigen::Tensor<const float, 2, 1, long int>, 16> ConstTensorMap;
typedef Eigen::Map<const MatrixN2> MapN2;
typedef Eigen::Map<const IntMatrixN2> IntMapN2;
typedef Eigen::Map<const MatrixN6> MapN6;
typedef Eigen::Map<const MatrixNN> MapNN;

Vec3 compute_centroid(const MatrixN3& vertices, const IntMapN3& triangles);

int find_num_triangles(const IntMapN3& triangles);

void handle_one_grasp(float grasp[kGraspDim],
                      const MapN3& vertices,
                      const IntMapN3& triangles,
                      const MapN3& normals,
                      TensorMap& output_tensor,
                      int num_grasp_perturbations,
                      int num_cone_edges,
                      float mu,
                      int index,
                      int batch_num,
                      int grasp_num,
                      const Eigen::Vector3f& centroid,
                      int num_triangles,
                      TriTensor& tri_e,
                      LpTensor& lp_e,
                      WrenchTensor& wrench_e,
                      PertTensor& pert_e);

void handle_one_grasp_grad (const MapN3& vertices,
                            const IntMapN3& triangles,
                            const Map3& centroid,
                            const MapN3& normals,
                            const IntMapN2& tri_nums,
                            const MapNN& lp_vars,
                            const SubWrenchTensor& wrenches,
                            const MapN6& perts,
                            Map6& grads,
                            float mu,
                            float grasp[6]);
#endif

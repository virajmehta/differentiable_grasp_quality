/* Grasp Quality
 * This is the core of the differentiable grasp quality op.
 * Viraj Mehta -- 2017
 */
#include "grasp_quality.h"
#include "Eigen/Geometry"
#include "ClpSimplex.hpp"
#include <vector>
#include <cmath>
#include <random>
#include <cstring>

static const bool JACDEBUG = false;
static const bool DEBUG = false;
static const float PI = 3.141592;
static const float kEpsilon = 1e-6;
static const float kBigEpsilon = 1e-1;
static const float kInfinity = 1e10;
static const float kVariance = 0.05;
static const float kNegSol = -1.0;
static const int kNotFound = -1;
static const int kNotFoundPos = -2;
static const int kNotFoundNeg = -3;
static const float kMaxError = 16.0;
static const int kSimplexVerts = 7;
static const float Q[kSimplexVerts][kGraspDim] = {
    {0.2182, 0.2582, 0.3162, 0.4082, 0.5774, 1.0},
    {0.2182, 0.2582, 0.3162, 0.4082, 0.5774, -1.0},
    {0.2182, 0.2582, 0.3162, 0.4082, -1.155, 0.0},
    {0.2182, 0.2582, 0.3162, -1.225, 0.0, 0.0},
    {0.2182, 0.2582, -1.265, 0.0, 0.0, 0.0},
    {0.2182, -1.291, 0.0, 0.0, 0.0, 0.0},
    {-1.309, 0.0, 0.0, 0.0, 0.0, 0.0}
};

Vec3 compute_centroid(const MatrixN3& vertices, const IntMapN3& triangles) {
    /* Performs smart centroid computation by weighting each vertex by the sum
     * of the areas of the triangles it is in */
    int n_vertices = vertices.rows();
    Eigen::VectorXf weights(n_vertices, 1);
    float total = 0;
    for (int i = 0; i < n_vertices; ++i) {
        weights(i) = 0;
    }
    for (int i = 0; i < triangles.rows(); ++i) {
        Vec3 v0 = vertices.row(triangles(i, 0));
        Vec3 v1 = vertices.row(triangles(i, 1));
        Vec3 v2 = vertices.row(triangles(i, 2));
        Vec3 v0v1 = v1 - v0;
        Vec3 v0v2 = v2 - v0;
        float area = 0.5 * (v0v1.cross(v0v2).norm());
        weights(triangles(i, 0)) += area;
        weights(triangles(i, 1)) += area;
        weights(triangles(i, 2)) += area;
        total += area;
    }
    total *= 3;
    MatrixN3 weighted_vertices(n_vertices, 3);
    for (int i = 0; i < n_vertices; ++i) {
        weighted_vertices(i,0) = vertices(i,0) * weights(i);
        weighted_vertices(i,1) = vertices(i,1) * weights(i);
        weighted_vertices(i,2) = vertices(i,2) * weights(i);
    }
    Vec3 sum = weighted_vertices.colwise().sum();
    sum /= total;
    return sum;
}

static inline void subtract_3_vector(const float a[3], const float b[3],
                                                                float* c){
    // a - b = c
    c[0] = a[0] - b[0];
    c[1] = a[1] - b[1];
    c[2] = a[2] - b[2];
}


static inline void cross_product(const float a[3], const float b[3], float* c){
    //a X b = c
    c[0] = a[1] * b[2] - b[1] * a[2];
    c[1] = b[0] * a[2] - a[0] * b[2];
    c[2] = a[0] * b[1] - b[0] * a[1];
}

static inline float dot_product(const float a[3], const float b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static inline float l2_norm_3_vector(const float input[3]) {
    return sqrt(input[0] * input[0] + input[1] * input[1] +
                                    input[2] * input[2]);
}

static Vec3 compute_triangle_normal(const IntMapN3& triangles, int idx,
                      float u, float v, const MapN3& normals, bool normalize) {
    float w = 1 - u - v;
    const Map3 n0(normals.row(triangles(idx, 0)).data());
    const Map3 n1(normals.row(triangles(idx, 1)).data());
    const Map3 n2(normals.row(triangles(idx, 2)).data());
    Vec3 normal = w * n0 + u * n1 + v * n2;
    if (normalize) {
        normal = normal / normal.norm();
    }
    return normal;
}

int find_num_triangles(const IntMapN3& triangles) {
    for (int i = 0; i < triangles.rows(); ++i) {
        if (triangles(i,0) == 0 && triangles(i,1)==0 && triangles(i,2) == 0) {
            return i;
        }
    }
    return triangles.rows();
}

static inline void copy_eigen_3_vec_array(const MapN3& tensor, int row,
                                   float* dst) {
    for (int i = 0; i < 3; ++i) {
        dst[i] = tensor(row, i);
    }
}

static Eigen::Vector3f contact_point(float u, float v, float v0[3],
                                           float v1[3], float v2[3]) {
    float v0v1[3];
    float v0v2[3];
    subtract_3_vector(v1, v0, v0v1);
    subtract_3_vector(v2, v0, v0v2);
    Eigen::Vector3f intersection;
    intersection(0) = v0[0] + u * v0v1[0] + v * v0v2[0];
    intersection(1) = v0[1] + u * v0v1[1] + v * v0v2[1];
    intersection(2) = v0[2] + u * v0v1[2] + v * v0v2[2];
    return intersection;

}


static MatrixN3 compute_friction_cone(const Eigen::Vector3f& normal, float mu,
                                                int num_vertices){
    Eigen::Vector3f indep;
    indep << 1.0,1.0,1.0;
    Vec3 perp = normal.cross(indep);
    if (perp(0) < kEpsilon && perp(1) < kEpsilon && perp(2) < kEpsilon) {
        // incase indep was not independent
        indep << 2.0,1.0,1.0;
        perp = normal.cross(indep);
    }
    //perp is guaranteed to be perpendicular to normal
    perp *= mu / perp.norm();
    Vec3 force = perp + normal;
    Eigen::AngleAxisf rotation = Eigen::AngleAxisf((2.0 / num_vertices) * PI, normal);
    MatrixN3 cone(num_vertices, 3);
    for (int i =0; i < num_vertices; ++i) {
        cone(i, 0) = force(0);
        cone(i, 1) = force(1);
        cone(i, 2) = force(2);
        force = rotation * force; // rotate around normal axis to get a cone
    }
    return -1 * cone; // forces point into the contact point
}

/* Code adapted from scratchapixel.org */
static bool ray_triangle_intersect(const float point[3], const float direction[3],
                        const float v0[3], const float v1[3], const float v2[3],
                        float& t, float& u, float& v) {
    float v0v1[3];
    subtract_3_vector(v1, v0, v0v1);
    float v0v2[3];
    subtract_3_vector(v2, v0, v0v2);
    float pvec[3];
    cross_product(direction, v0v2, pvec);
    float det = dot_product(v0v1, pvec);
    if (fabs(det) < kEpsilon) {
        return false;
    }
    float invdet = 1 / det;
    float tvec[3];
    subtract_3_vector(point, v0, tvec);
    u = dot_product(tvec, pvec) * invdet;
    if (u < 0 || u > 1) {
        return false;
    }
    float qvec[3];
    cross_product(tvec, v0v1, qvec);
    v = dot_product(direction, qvec) * invdet;
    if (v < 0 || u + v > 1) {
        return false;
    }

    t = dot_product(v0v2, qvec) * invdet;
    return true;
}

static MatrixN6 compute_primitive_wrenches(const Eigen::Vector3f& normal,
                                       const Eigen::Vector3f& neg_normal,
                                       const Eigen::Vector3f& contact_point,
                                       const Eigen::Vector3f& neg_contact_point,
                                       const Eigen::Vector3f& centroid,
                                       float mu,
                                       int num_vertices) {
    // TODO: maybe add noise to mu
    MatrixN3 cone = compute_friction_cone(normal, mu, num_vertices);
    MatrixN3 neg_cone = compute_friction_cone(neg_normal, mu, num_vertices);
    Eigen::Vector3f r = contact_point - centroid;
    Eigen::Vector3f r_neg = neg_contact_point - centroid;
    MatrixN6 wrenches(2 * num_vertices, 6);
    // computing torques
    for (int i = 0; i < num_vertices; ++i) {
        Eigen::Vector3f force = cone.row(i).transpose();
        Eigen::Vector3f torque = r.cross(force);
        for (int j = 0; j < 3; ++j) {
            wrenches(i, j) = force(j);
            wrenches(i, j + 3) = torque(j);
        }
    }
    for (int i = 0; i < num_vertices; ++i) {
        Eigen::Vector3f neg_force = neg_cone.row(i).transpose();
        Eigen::Vector3f neg_torque = r_neg.cross(neg_force);
        for (int j = 0; j < 3; ++j) {
            wrenches(i + num_vertices, j) = neg_force(j);
            wrenches(i + num_vertices, j + 3) = neg_torque(j);
        }
    }
    return wrenches;
}

static void two_point_to_center_orientation(float* grasp) {
    float c[3];
    c[0] = (grasp[0] + grasp[3]) / 2;
    c[1] = (grasp[1] + grasp[4]) / 2;
    c[2] = (grasp[2] + grasp[5]) / 2;
    float d[3];
    d[0] = grasp[3] - grasp[0];
    d[1] = grasp[4] - grasp[1];
    d[2] = grasp[5] - grasp[2];
    float norm = l2_norm_3_vector(d);
    grasp[0] = c[0];
    grasp[1] = c[1];
    grasp[2] = c[2];
    grasp[3] = d[0] / norm;
    grasp[4] = d[1] / norm;
    grasp[5] = d[2] / norm;
}

static bool co_hull_contains_zero(MatrixN6& wrenches) {
    /*
    AKA
    minimize (over x): 1
    s.t.     Ax = P
             x^T * [1] = 1
             x[i] >= 0  \forall i
    */
    int width = wrenches.rows();
    double* rows[6];
    ClpSimplex model;
    model.setLogLevel(0);
    model.resize(0, width);

    // initialize top part of A
    int* rowIndex = new int[width];
    double* ones = new double[width];
    for (int i = 0; i < width; ++i) {
        rowIndex[i] = i;
        ones[i] = 1;
    }
    for (int row = 0; row < 6; ++row) {
        rows[row] = new double[width];
        for (int i = 0; i < width; ++i) rows[row][i] = wrenches(i, row);
        model.addRow(width, rowIndex, rows[row], 0., 0.);
    }
    model.addRow(width, rowIndex, ones, 1., 1.);
    for (int i = 0; i < width; ++i) {
        model.setColumnLower(i, 0.);
        model.setColumnUpper(i, COIN_DBL_MAX);
    }

    model.primal();
    for (int i = 0; i < 6; ++i) delete[] rows[i];
    delete[] rowIndex;
    delete[] ones;

    return model.status() == 0;  // 0 is their code for optimal
}

static float compute_q_neg(const MatrixN6& wrenches,
                           int batch_num,
                           int grasp_num,
                           int pert_num,
                           LpTensor& lp_e) {
    int N = wrenches.rows();
    int width = N + 1;
    float max_q = -1 * kInfinity;
    int max_q_index = -1;
    double* rows[6];
    int* rowIndex = new int[width];
    double* ones = new double[width];
    for (int i = 0; i < width; ++i) {
        rowIndex[i] = i;
        ones[i] = i == 0 ? 0 : 1;
    }
    for (int row = 0; row < 6; ++row) {
        rows[row] = new double[width];
        for (int i = 0; i < width; ++i) {
            rows[row][i] = i == 0 ? 0 :  wrenches(i - 1, row);
        }
    }
    for (int i = 0; i < kSimplexVerts; ++i) {
        ClpSimplex model;
        model.setLogLevel(0);
        model.resize(0, width);
        model.setObjectiveCoefficient(0, -1.0);
        for (int row = 0; row < 6; ++row) {
            rows[row][0] = -1 * Q[i][row];
            model.addRow(width, rowIndex, rows[row], 0., 0.);
        }
        model.addRow(width, rowIndex, ones, 1.,1.);
        for (int i = 0; i < width; ++i) {
            model.setColumnLower(i, 0.);
            model.setColumnUpper(i, COIN_DBL_MAX);
        }
        model.primal();
        float val = model.objectiveValue();
        if (val > max_q && model.status() == 0) {
            max_q = val;
            max_q_index = i;
            const double* col = model.getColSolution();
            for (int i = 0; i < width; ++i) {
                lp_e(batch_num, grasp_num, pert_num, i) = col[i];
            }
        }
    }
    lp_e(batch_num, grasp_num, pert_num, N + 5) = max_q_index;
    lp_e(batch_num, grasp_num, pert_num, N + 6) = kNegSol;
    if (max_q_index == -1) {
        if (DEBUG) std::cout << "no neg lp solutions" << std::endl;
        max_q = kMaxError;
    }
    for (int i = 0; i < 6; ++i) delete[] rows[i];
    delete[] rowIndex;
    delete[] ones;
    return max_q;
}

static float compute_q_pos(const MatrixN6& wrenches,
                           int batch_num,
                           int grasp_num,
                           int pert_num,
                           LpTensor& lp_e) {
    int N = wrenches.rows();
    int width = N + kSimplexVerts;
    ClpSimplex model;
    model.setLogLevel(0);
    // size of matrix is 7 * (7 + N)
    model.resize(0, width);
    // we minimize over just rho
    for (int i = 0; i < kSimplexVerts; ++i) {
        model.setObjectiveCoefficient(i, 1.0);
    }
    double* rows[6];
    int* rowIndex = new int[width];
    double* ones = new double[width];
    for (int i = 0; i < width; ++i) {
        rowIndex[i] = i;
        ones[i] = i < kSimplexVerts ? 0 : 1;
    }
    //horizontal bounds
    for (int row = 0; row < 6; ++row) {
        rows[row] = new double[width];
        for (int i = 0; i < width; ++i) {
            rows[row][i] = i < kSimplexVerts ? -1 * Q[i][row] : wrenches(i -
                                                    kSimplexVerts, row);
        }
        model.addRow(width, rowIndex, rows[row], 0., 0.);
    }
    model.addRow(width, rowIndex, ones, 1.,1.);
    // vertical bounds are default
    for (int i = 0; i < width; ++i) {
        model.setColumnLower(i, 0.);
        model.setColumnUpper(i, COIN_DBL_MAX);
    }

    model.primal();
    if (model.status() != 0) {
        lp_e(batch_num, grasp_num, pert_num, width - 1) = kNegSol;
        lp_e(batch_num, grasp_num, pert_num, width - 2) = kMaxError;
        if (DEBUG) {
            std::cout << "q pos infeasible" << std::endl;
        }
        for (int i = 0; i < 6; ++i) delete[] rows[i];
        delete[] rowIndex;
        delete[] ones;
        return kMaxError;
    }
    const double* col = model.getColSolution();
    for (int i = 0; i < width; ++i) {
        lp_e(batch_num, grasp_num, pert_num, i) = col[i];
    }
    for (int i = 0; i < 6; ++i) delete[] rows[i];
    delete[] rowIndex;
    delete[] ones;
    return model.objectiveValue();
}

static int intersect(float point[3],
                      float direction[3],
                      const MapN3& vertices,
                      const IntMapN3& triangles,
                      int num_triangles,
                      int& pos_tri_index,
                      int& neg_tri_index,
                      float& u,
                      float& v,
                      float& u_neg,
                      float& v_neg) {
    float t = 0;
    bool found = false;
    float t_neg = 0;
    bool found_neg = false;

    float u_tmp;
    float v_tmp;
    float t_tmp = 0;
    float v0[3];
    float v1[3];
    float v2[3];
    for (int tri_index = 0; tri_index < num_triangles; ++tri_index) {
        copy_eigen_3_vec_array(vertices, triangles(tri_index, 0), v0);
        copy_eigen_3_vec_array(vertices, triangles(tri_index, 1), v1);
        copy_eigen_3_vec_array(vertices, triangles(tri_index, 2), v2);
        if (ray_triangle_intersect(point, direction,
                                v0, v1, v2, t_tmp, u_tmp, v_tmp)) {
            if (DEBUG) {
                std::cout << "intersection found at triangle " << tri_index <<
                                        std::endl;
                std::cout << "u: " << u_tmp << " v: " << v_tmp << " t: " <<
                                t_tmp << std::endl;
            }
            float intersection[3];
            for (int i =0; i < 3; ++i) {
                intersection[i] = (1 - u_tmp - v_tmp) * v0[i] + u_tmp * v1[i] +
                                                v_tmp * v2[i];
            }
            float pointint[3];
            subtract_3_vector(intersection, point, pointint);
            if (DEBUG) {
                std::cout << "intersection: " << intersection[0] << " " <<
                    intersection[1] << " " << intersection[2] << std::endl;
            }
            if (dot_product(pointint, direction) > 0) {
                found = true;
                if (DEBUG) {
                    std::cout << "positive!" << std::endl;
                }
                if (fabs(t_tmp) >= t) {
                    u = u_tmp;
                    v = v_tmp;
                    t = fabs(t_tmp);
                    pos_tri_index = tri_index;
                }
            } else {
                found_neg = true;
                if (DEBUG) {
                    std::cout << "positive!" << std::endl;
                }
                if (fabs(t_tmp) >= t_neg) {
                    u_neg = u_tmp;
                    v_neg = v_tmp;
                    t_neg = fabs(t_tmp);
                    neg_tri_index = tri_index;
                }
            }
        }
    }
    if (found && found_neg) {
        return 0; // normal case
    } else if (found) {
        return 1; // need to give a gradient towards postive direction
    } else if (found_neg) {
        return 2; // need to give a gradient towards negative direction
    } else {
        return 3; // no intersection, need gradient toward centroid or spin
    }
}

static float eval_one_pert(float grasp[kGraspDim],
                           const MapN3& vertices,
                           const IntMapN3& triangles,
                           const MapN3& normals,
                           int num_triangles,
                           const Eigen::RowVector3f centroid,
                           float mu,
                           int num_cone_edges,
                           int batch_num,
                           int grasp_num,
                           int pert_num,
                           TriTensor& tri_e,
                           LpTensor& lp_e,
                           WrenchTensor& wrench_e) {
    float u = 0, v = 0, u_neg = 0 , v_neg = 0;
    int pos_tri_index = kNotFound, neg_tri_index = kNotFound;
    int found = intersect(grasp, &(grasp[3]), vertices, triangles,
                            num_triangles, pos_tri_index, neg_tri_index,
                            u, v, u_neg, v_neg);
    if (found == 3) {
        tri_e(batch_num, grasp_num, pert_num, 0) = kNotFound;
        tri_e(batch_num, grasp_num, pert_num, 1) = kNotFound;
        if (DEBUG)
            std::cout << "no intersection found" << std::endl;
        return kMaxError;
    } else if (found == 2) { // negative
        tri_e(batch_num, grasp_num, pert_num, 0) = kNotFoundNeg;
        tri_e(batch_num, grasp_num, pert_num, 1) = kNotFoundNeg;
        if (DEBUG)
            std::cout << "only negative intersection found" << std::endl;
        return kMaxError;
    } else if (found == 1) { // positive
        tri_e(batch_num, grasp_num, pert_num, 0) = kNotFoundPos;
        tri_e(batch_num, grasp_num, pert_num, 1) = kNotFoundPos;
        if (DEBUG)
            std::cout << "only positive intersection found" << std::endl;
        return kMaxError;
    }
    tri_e(batch_num, grasp_num, pert_num, 0) = pos_tri_index;
    tri_e(batch_num, grasp_num, pert_num, 1) = neg_tri_index;
    float v0[3];
    float v1[3];
    float v2[3];
    float nv0[3];
    float nv1[3];
    float nv2[3];
    copy_eigen_3_vec_array(vertices, triangles(pos_tri_index, 0), v0);
    copy_eigen_3_vec_array(vertices, triangles(pos_tri_index, 1), v1);
    copy_eigen_3_vec_array(vertices, triangles(pos_tri_index, 2), v2);
    copy_eigen_3_vec_array(vertices, triangles(neg_tri_index, 0), nv0);
    copy_eigen_3_vec_array(vertices, triangles(neg_tri_index, 1), nv1);
    copy_eigen_3_vec_array(vertices, triangles(neg_tri_index, 2), nv2);
    Eigen::Vector3f normal = compute_triangle_normal(triangles,
                                    pos_tri_index, u, v, normals, true);
    Eigen::Vector3f neg_normal = compute_triangle_normal(triangles,
                                 neg_tri_index, u_neg, v_neg, normals, true);
    Eigen::Vector3f contact = contact_point(u,v,v0,v1,v2);
    Eigen::Vector3f neg_contact = contact_point(u_neg,v_neg,nv0,nv1,nv2);
    MatrixN6 wrenches = compute_primitive_wrenches(normal, neg_normal,
        contact, neg_contact, centroid, mu, num_cone_edges);
    for (int i = 0; i < num_cone_edges*2; ++i) {
        for (int j = 0; j < 6; ++j) {
            wrench_e(batch_num, grasp_num, pert_num, i, j) = wrenches(i,j);
        }
    }
    float q;
    if (co_hull_contains_zero(wrenches)) {
        q = compute_q_neg(wrenches, batch_num, grasp_num,
                                                          pert_num, lp_e);
    }
    else {
        q = compute_q_pos(wrenches, batch_num, grasp_num,
                                                    pert_num, lp_e);
    }
    if (q == kMaxError) {
        tri_e(batch_num, grasp_num, pert_num, 0) = kNotFound;
        tri_e(batch_num, grasp_num, pert_num, 1) = kNotFound;
    }
    return q;
}

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
                      PertTensor& pert_e) {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, kVariance);
    two_point_to_center_orientation(grasp);
    float new_grasp[kGraspDim];
    float grasp_quality_score = 0;
    for (int i = 0;  i < kGraspDim; ++i) {
        new_grasp[i] = grasp[i];
    }
    for (int i = 0; i < num_grasp_perturbations; ++i) {
        grasp_quality_score += eval_one_pert(new_grasp,
                                             vertices,
                                             triangles,
                                             normals,
                                             num_triangles,
                                             centroid,
                                             mu,
                                             num_cone_edges,
                                             batch_num,
                                             grasp_num,
                                             i,
                                             tri_e,
                                             lp_e,
                                             wrench_e);
        for (int j = 0; j < kGraspDim; ++j) {
            pert_e(batch_num, grasp_num, i, j) = new_grasp[j];
            new_grasp[j] = grasp[j] + distribution(generator);
        }
    }
    output_tensor(index) = grasp_quality_score / num_grasp_perturbations;
}


static inline std::vector<int> get_nonzero_indices(const MapNN& lp_vars, int row,
                                                               bool is_negative) {
    std::vector<int> indices;
    int end = is_negative ? lp_vars.cols() - kSimplexVerts + 1: lp_vars.cols();
    for (int i = 0; i < end; ++i) {
        if (fabs(lp_vars(row, i)) > kEpsilon) {
            indices.push_back(i);
        }
    }
    return indices;
}

static Matrix66 get_wrench_jac(int pert,
                               const MapN3& vertices,
                               const IntMapN3& triangles,
                               const Map3& centroid,
                               const MapN3& normals,
                               const IntMapN2& tri_nums,
                               const MapNN& lp_vars,
                               const SubWrenchTensor& wrenches,
                               const MapN6& perts,
                               Vec6& grads,
                               std::vector<int>& nonzero,
                               int r,
                               bool isNegDistance,
                               float mu) {
    int num_wrenches = wrenches.dimension(1);
    int num_vertices = num_wrenches/2;
    if (DEBUG) {
        std::cout << "num wrenches: " << num_wrenches << std::endl;
        std::cout << "num vertices: " << num_vertices << std::endl;
    }
    // any lp grasp index less than this comes from the cone of the pos grasp
    Matrix66 jac = Matrix66::Zero();
    Map3 v0_pos = Map3(vertices.row(triangles(tri_nums(pert, 0),
                                                0)).data());
    Map3 v1_pos = Map3(vertices.row(triangles(tri_nums(pert, 0),
                                                1)).data());
    Map3 v2_pos = Map3(vertices.row(triangles(tri_nums(pert, 0),
                                                2)).data());
    Map3 v0_neg = Map3(vertices.row(triangles(tri_nums(pert, 1),
                                                0)).data());
    Map3 v1_neg = Map3(vertices.row(triangles(tri_nums(pert, 1),
                                                1)).data());
    Map3 v2_neg = Map3(vertices.row(triangles(tri_nums(pert, 1),
                                                2)).data());
    if (JACDEBUG) {
        std::cout<< "v0pos: "<<std::endl << v0_pos<<std::endl;
        std::cout<< "v1pos: "<< std::endl << v1_pos <<std::endl;
        std::cout<< "v2pos: "<<std::endl <<v2_pos << std::endl;
        std::cout<< "v0neg: "<<std::endl << v0_neg << std::endl;
        std::cout<< "v1neg: "<< std::endl << v1_neg << std::endl;
        std::cout<< "v2neg: "<<std::endl <<v2_neg << std::endl;
    }
    Vec3 v0v1_pos = v1_pos - v0_pos;
    Vec3 v0v2_pos = v2_pos - v0_pos;
    Vec3 v0v1_neg = v1_neg - v0_neg;
    Vec3 v0v2_neg = v2_neg - v0_neg;
    Map3 center = Map3(&(perts.data()[6 * pert]));
    Map3 dir_pos = Map3(&(perts.data()[6 * pert + 3]));
    Vec3 dir_neg = -1 * dir_pos;
    if(JACDEBUG) {
        std::cout << "v0v1_pos: " <<std::endl << v0v1_pos<<std::endl;
        std::cout << "v0v2_pos: " << std::endl << v0v2_pos << std::endl;
        std::cout << "v0v1_neg: " <<std::endl << v0v1_neg<<std::endl;
        std::cout << "v0v2_neg: " << std::endl << v0v2_neg << std::endl;
        std::cout << "dir_pos: " << std::endl << dir_pos << std::endl;
        std::cout << "dir_neg: " << std::endl << dir_neg << std::endl;
        std::cout << "center: " << std::endl << center << std::endl;
    }
    //See Moller-Trombore
    Vec3 pvec_pos = dir_pos.cross(v0v2_pos);
    float det_pos = v0v1_pos.dot(pvec_pos);
    Vec3 tvec_pos = center - v0_pos;
    float u_pos = tvec_pos.dot(pvec_pos) / det_pos;
    Vec3 qvec_pos = tvec_pos.cross(v0v1_pos);
    float v_pos = dir_pos.dot(qvec_pos) / det_pos;
    Vec3 ddet_pos = v0v2_pos.cross(v0v1_pos);
    if (JACDEBUG) {
        std::cout << "intersection_pos" << std::endl << v0_pos + u_pos *
        v0v1_pos + v_pos * v0v2_pos << std::endl;
        std::cout << "pvec_pos" << std::endl << pvec_pos << std::endl;
        std::cout << "det_pos: " << det_pos << std::endl;
        std::cout << "tvec_pos: "<<std::endl << tvec_pos << std::endl;
        std::cout << "u_pos: " << u_pos << std::endl;
        std::cout << "qvec_pos: " << qvec_pos << std::endl;
        std::cout << "v_pos: " << v_pos << std::endl;
        std::cout << "ddet_pos: " << ddet_pos << std::endl;
    }

    Vec3 du_dcenter_pos = pvec_pos / det_pos;
    Vec3 du_ddir_pos = (v0v2_pos.cross(tvec_pos) * det_pos -
                tvec_pos.dot(pvec_pos) * ddet_pos) / (det_pos * det_pos);
    Vec3 dv_dcenter_pos = (1 / det_pos) * v0v1_pos.cross(dir_pos);
    Vec3 dv_ddir_pos = (det_pos * qvec_pos - dir_pos.dot(qvec_pos) * ddet_pos)/
                            (det_pos * det_pos);


    Vec3 pvec_neg = dir_neg.cross(v0v2_neg);
    float det_neg = v0v1_neg.dot(pvec_neg);
    Vec3 tvec_neg = center - v0_neg;
    float u_neg = tvec_neg.dot(pvec_neg) / det_neg;
    Vec3 qvec_neg = tvec_neg.cross(v0v1_neg);
    float v_neg = dir_neg.dot(qvec_neg) / det_neg;
    Vec3 ddet_neg = v0v2_neg.cross(v0v1_neg);
    if (JACDEBUG) {
        std::cout << "intersection_neg" << std::endl << v0_neg + u_neg *
        v0v1_neg + v_neg * v0v2_neg << std::endl;
        std::cout << "pvec_neg" << std::endl << pvec_neg << std::endl;
        std::cout << "det_neg: " << det_neg << std::endl;
        std::cout << "tvec_neg: "<<std::endl << tvec_neg << std::endl;
        std::cout << "u_neg: " << u_neg << std::endl;
        std::cout << "qvec_neg: " << qvec_neg << std::endl;
        std::cout << "v_neg: " << v_neg << std::endl;
        std::cout << "ddet_neg: " << ddet_neg << std::endl;
    }

    Vec3 du_dcenter_neg = pvec_neg / det_neg;
    Vec3 du_ddir_neg = (v0v2_neg.cross(tvec_neg) * det_neg -
                tvec_neg.dot(pvec_neg) * ddet_neg) / (det_neg * det_neg);
    Vec3 dv_dcenter_neg = (1 / det_neg) * v0v1_neg.cross(dir_neg);
    Vec3 dv_ddir_neg = (det_neg * qvec_neg - dir_neg.dot(qvec_neg) * ddet_neg)/
                            (det_neg * det_neg);

    Vec3 radius_pos = (v0_pos + (u_pos * v0v1_pos) + (v_pos *
                        v0v2_pos)) - centroid;
    Vec3 radius_neg = (v0_neg + (u_neg * v0v1_neg) + (v_neg *
                        v0v2_neg)) - centroid;
    Vec3 normal_pos = compute_triangle_normal(triangles,
                    tri_nums(pert,0), u_pos, v_pos, normals, false);
    float norm_pos =  normal_pos.norm();
    normal_pos = normal_pos / norm_pos;
    Vec3 normal_neg = compute_triangle_normal(triangles,
                    tri_nums(pert,1), u_neg, v_neg, normals, false);
    float norm_neg = normal_neg.norm();
    normal_neg = normal_neg / norm_neg;
    if (JACDEBUG) {
        std::cout << "normal pos:" << std::endl << normal_pos << std::endl;
        std::cout << "normal neg:" << std::endl << normal_neg << std::endl;
    }
    Vec3 dnormal_du_pos = (Matrix33::Identity() - normal_pos *
        normal_pos.transpose()) *
        (normals.row(triangles(tri_nums(pert,0),1)).transpose() -
        normals.row(triangles(tri_nums(pert,0),0)).transpose()) / norm_pos;
    Vec3 dnormal_dv_pos = (Matrix33::Identity() - normal_pos *
        normal_pos.transpose()) *
        (normals.row(triangles(tri_nums(pert,0),2)).transpose() -
        normals.row(triangles(tri_nums(pert,0),0)).transpose()) / norm_pos;
    Vec3 dnormal_du_neg = (Matrix33::Identity() - normal_neg *
        normal_neg.transpose()) *
        (normals.row(triangles(tri_nums(pert,1),1)).transpose() -
        normals.row(triangles(tri_nums(pert,1),0)).transpose()) / norm_neg;
    Vec3 dnormal_dv_neg = (Matrix33::Identity() - normal_neg *
        normal_neg.transpose()) *
        (normals.row(triangles(tri_nums(pert,1),2)).transpose() -
        normals.row(triangles(tri_nums(pert,1),0)).transpose()) / norm_neg;
    if (DEBUG) {
        std::cout << "positive normal 0" <<
        std::endl<<normals.row(triangles(tri_nums(pert, 0), 0))<<std::endl;
        std::cout << "positive normal 1" <<
        std::endl<<normals.row(triangles(tri_nums(pert, 0), 1))<<std::endl;
        std::cout << "positive normal 2" <<
        std::endl<<normals.row(triangles(tri_nums(pert, 0), 2))<<std::endl;
        std::cout << "negative normal 0" <<
        std::endl<<normals.row(triangles(tri_nums(pert, 1), 0))<<std::endl;
        std::cout << "negative normal 1" <<
        std::endl<<normals.row(triangles(tri_nums(pert, 1), 1))<<std::endl;
        std::cout << "negative normal 2" <<
        std::endl<<normals.row(triangles(tri_nums(pert, 1), 2))<<std::endl;
        std::cout<<"dnormal/dupos: " <<std::endl<<dnormal_du_pos<<std::endl;
        std::cout<<"dnormal/duneg: " <<std::endl<<dnormal_du_neg<<std::endl;
        std::cout<<"dnormal/dvpos: " <<std::endl<<dnormal_dv_pos<<std::endl;
        std::cout<<"dnormal/dvneg: " <<std::endl<<dnormal_dv_neg<<std::endl;
    }
    Matrix3N df_du_pos(3, num_vertices);
    Matrix3N df_dv_pos(3, num_vertices);
    Matrix3N df_du_neg(3, num_vertices);
    Matrix3N df_dv_neg(3, num_vertices);

    Vec3 n_p = normal_pos;
    Vec3 b_p;
    b_p << 1, 1, 1;
    Vec3 perp_p = n_p.cross(b_p);
    if (perp_p(0) < kEpsilon && perp_p(1) < kEpsilon && perp_p(2) < kEpsilon) {
        // incase indep was not independent
        b_p << 2.0,1.0,1.0;
        perp_p = n_p.cross(b_p);
    }
    float p_norm = perp_p.norm();
    perp_p /= p_norm;
    if (DEBUG) {
        std::cout << "perp_p:" << std::endl<< perp_p << std::endl;
    }
    df_du_pos.col(0) = dnormal_du_pos + mu * (((Matrix33::Identity() - perp_p *
                perp_p.transpose()) / p_norm) * (dnormal_du_pos.cross(b_p)));
    df_dv_pos.col(0) = dnormal_dv_pos + mu * (((Matrix33::Identity() - perp_p *
                perp_p.transpose()) / p_norm) * (dnormal_dv_pos.cross(b_p)));
    Vec3 n_n = normal_neg;
    Vec3 b_n;
    b_n << 1, 1, 1;
    Vec3 perp_n = n_n.cross(b_n);
    if (perp_n(0) < kEpsilon && perp_n(1) < kEpsilon && perp_n(2) < kEpsilon) {
        // incase indep was not independent
        b_n << 2.0,1.0,1.0;
        perp_n = n_n.cross(b_n);
    }
    float n_norm = perp_n.norm();
    perp_n /= n_norm;
    if (DEBUG) {
        std::cout << "perp_n:" << std::endl<< perp_n << std::endl;
    }
    df_du_neg.col(0) = dnormal_du_neg + mu * (((Matrix33::Identity() - perp_n *
                perp_n.transpose()) / n_norm) * (dnormal_du_neg.cross(b_n)));
    df_dv_neg.col(0) = dnormal_dv_neg + mu * (((Matrix33::Identity() - perp_p *
                perp_p.transpose()) / n_norm) * (dnormal_dv_neg.cross(b_n)));

    

    float angle = (2.0 / num_vertices) * PI;
    float n_cos = 1-cos(angle);
    float s = sin(angle);

    MatrixN3 dRdn0_p(3, 3);
    dRdn0_p <<  2*n_p(0)*n_cos, n_p(1)*n_cos, n_p(2)*n_cos,
              n_p(1)*n_cos,   0,            -s,
              n_p(2)*n_cos,   s,            0;

    MatrixN3 dRdn1_p(3, 3);
    dRdn1_p << 0,            n_p(0)*n_cos,   s,
             n_p(0)*n_cos, 2*n_p(1)*n_cos, n_p(2)*n_cos,
             -s,           n_p(2)*n_cos,   0;

    MatrixN3 dRdn2_p(3, 3);
    dRdn2_p << 0 ,           -s,           n_p(0)*n_cos, 
             s,            0,            n_p(1)*n_cos,
             n_p(0)*n_cos, n_p(1)*n_cos, 2*n_p(2)*n_cos;
    
    MatrixN3 dRdn0_n(3, 3);
    dRdn0_n <<  2*n_n(0)*n_cos, n_n(1)*n_cos, n_n(2)*n_cos,
              n_n(1)*n_cos,   0,            -s,
              n_n(2)*n_cos,   s,            0;

    MatrixN3 dRdn1_n(3, 3);
    dRdn1_n << 0,            n_n(0)*n_cos,   s,
             n_n(0)*n_cos, 2*n_n(1)*n_cos,   n_n(2)*n_cos,
             -s,           n_n(2)*n_cos,   0;

    MatrixN3 dRdn2_n(3, 3);
    dRdn2_n << 0 ,           -s,           n_n(0)*n_cos, 
             s,            0,            n_n(1)*n_cos,
             n_n(0)*n_cos, n_n(1)*n_cos, 2*n_n(2)*n_cos;

    Eigen::AngleAxisf rotation_pos = Eigen::AngleAxisf((2.0 / num_vertices) * PI, normal_pos);
    Eigen::AngleAxisf rotation_neg = Eigen::AngleAxisf((2.0 / num_vertices) * PI, normal_neg);

    for (int i = 1; i < num_vertices; ++i) {
        Map3 f_n_1_pos(&(wrenches.data()[(pert * num_wrenches + (i - 1)) *6]));
        Map3 f_n_1_neg(&(wrenches.data()[(pert * num_wrenches + num_vertices +
                                                            (i - 1)) * 6]));
        Matrix33 dR_du_pos = dRdn0_p * dnormal_du_pos(0) + dRdn1_p *
                            dnormal_du_pos(1) + dRdn2_p * dnormal_du_pos(2);
        Matrix33 dR_dv_pos = dRdn0_p * dnormal_dv_pos(0) + dRdn1_p *
                            dnormal_dv_pos(1) + dRdn2_p * dnormal_dv_pos(2);
        Matrix33 dR_du_neg = dRdn0_n * dnormal_du_neg(0) + dRdn1_n *
                            dnormal_du_neg(1) + dRdn2_n * dnormal_du_neg(2);
        Matrix33 dR_dv_neg = dRdn0_n * dnormal_dv_neg(0) + dRdn1_n *
                            dnormal_dv_neg(1) + dRdn2_n * dnormal_dv_neg(2);
        df_du_pos.col(i) = -1 * dR_du_pos  * f_n_1_pos + rotation_pos *
                                                        df_du_pos.col(i - 1);
        df_dv_pos.col(i) = -1 * dR_dv_pos * f_n_1_pos + rotation_pos *
                                                        df_dv_pos.col(i - 1);
        df_du_neg.col(i) = -1 * dR_du_neg * f_n_1_neg + rotation_neg *
                                                        df_du_neg.col(i - 1);
        df_dv_neg.col(i) = -1 * dR_dv_neg * f_n_1_neg + rotation_neg *
                                                        df_dv_neg.col(i - 1);
    }
    if (DEBUG) {
        std::cout<< " rot computed " <<std::endl;
    }

    
    df_du_pos *= -1;
    df_dv_pos *= -1;
    df_du_neg *= -1;
    df_dv_neg *= -1;
    
    for (size_t j = r; j < nonzero.size(); ++j) { // positive lp vars
        int wrench_num = nonzero[j] - (isNegDistance ? 1 : kSimplexVerts);
        Vec3 f = Map3(&(wrenches.data()[(pert * num_wrenches + wrench_num) * 6]));
        Vec3 df_du;
        Vec3 df_dv;
        if (wrench_num < num_vertices) {
            df_du =  df_du_pos.col(wrench_num);
            df_dv = df_dv_pos.col(wrench_num);
        }
        else {
            df_du = df_du_neg.col(wrench_num - num_vertices);
            df_dv = df_dv_neg.col(wrench_num - num_vertices);
        }
        Matrix66 wrench_jac = Matrix66::Zero();
        if (DEBUG) {
            std::cout << "df_du: " << df_du << std::endl;
            std::cout << "df_dv: " << df_dv << std::endl;
            std::cout << "force "<< wrench_num<<": " << f << std::endl;
            std::cout << "nonzero[j]: "<<nonzero[j]<<std::endl;
            if (isNegDistance) std::cout << "is neg distance"<< std::endl;
        }

        /*
        Vec3 df_du = df_dn * dnormal_du_pos; //maybe? NOPE
        Vec3 df_dv = df_dn * dnormal_dv_pos; 
        */
        for (size_t i = 0; i < kGraspDim; ++i) { //cols in Jacobian matrix
            if (wrench_num < num_vertices) { // positive grasp point
                //Vec3 Y = (f - normal_pos).cross(normal_pos);
                //Vec3 dd_du = dnormal_du_pos.cross(Y) * mu /
                //                                dnormal_du_pos.norm();
                //Vec3 dd_dv = dnormal_dv_pos.cross(Y) * mu /
                //                                dnormal_dv_pos.norm();

                if (i < 3) { // center
                    Vec3 df_dcenter = df_du * du_dcenter_pos(i) + df_dv *dv_dcenter_pos(i);
                    for (int k = 0; k < 3; ++k) {
                        wrench_jac(k, i) = df_dcenter(k);
                    }
                    Vec3 dx_dcenter = v0v1_pos * du_dcenter_pos(i) + v0v2_pos *
                                                        dv_dcenter_pos(i);
                    Vec3 dt_dcenter = (dx_dcenter.cross(f) + radius_pos.cross(df_dcenter));
                    for (int k = 0; k < 3; ++k) {
                        wrench_jac(3 + k, i) = dt_dcenter(k);
                    }
                } else { // direction
                    Vec3 df_ddir = df_du * du_ddir_pos(i - 3) + df_dv * dv_ddir_pos(i - 3);
                    for (int k = 0; k < 3; ++k) {
                        wrench_jac(k, i) = df_ddir(k);
                    }
                    Vec3 dx_ddir = v0v1_pos * du_ddir_pos(i - 3) + v0v2_pos
                                                        * dv_ddir_pos(i - 3);
                    Vec3 dt_ddir = (dx_ddir.cross(f) + radius_pos.cross(df_ddir));
                    for (int k = 0; k < 3; ++k) {
                        wrench_jac(3 + k, i) = dt_ddir(k);
                    }
                }
            } else {  // negative grasp point
                /*
                Vec3 Y = (f - normal_neg).cross(normal_neg);
                Vec3 dd_du = dnormal_du_neg.cross(Y) * mu /
                                                dnormal_du_neg.norm();
                Vec3 dd_dv = dnormal_dv_neg.cross(Y) * mu /
                                                dnormal_dv_neg.norm();
                Vec3 df_du = dnormal_du_neg + dd_du;
                Vec3 df_dv = dnormal_dv_neg + dd_dv;
                */
                if (i < 3) {  // center
                    Vec3 df_dcenter = df_du * du_dcenter_neg(i) + df_dv *
                                                       dv_dcenter_neg(i);
                    for (int k = 0; k < 3; ++k) {
                        wrench_jac(k, i) += df_dcenter(k);
                    }
                    Vec3 dx_dcenter = v0v1_neg * du_dcenter_neg(i) +
                                            v0v2_neg * dv_dcenter_neg(i);
                    Vec3 dt_dcenter = (dx_dcenter.cross(f) + radius_neg.cross(df_dcenter));
                    for (int k = 0; k < 3; ++k) {
                        wrench_jac(3 + k, i) = dt_dcenter(k);
                    }
                } else { // direction
                    Vec3 df_ddir = df_du * du_ddir_neg(i - 3) + df_dv *
                                                       dv_ddir_neg(i - 3);
                    for (int k = 0; k < 3; ++k) {
                        wrench_jac(k, i) = df_ddir(k);
                    }
                    Vec3 dx_ddir = v0v1_neg * du_ddir_neg(i - 3) + v0v2_neg
                                                        * dv_ddir_neg(i - 3);
                    Vec3 dt_ddir = (dx_ddir.cross(f) + radius_neg.cross(df_ddir));
                    for (int k = 0; k < 3; ++k) {
                        wrench_jac(3 + k, i) = dt_ddir(k);
                    }
                }
            }
        }
        jac += wrench_jac * lp_vars(pert, nonzero[j]);
    }
    return jac;
}

static RowVec6 make_e(int dim) {
    RowVec6 e;
    for (int i = 0; i < dim; ++i) {
        e(i) = 1;
    }
    for (int i = dim; i < 6; ++i) {
        e(i) = 0;
    }
    return e;
}

static Matrix66 make_D(std::vector<int>& nonzero,
           const SubWrenchTensor& wrenches, int pert, int r, bool isNeg) {
    // here r is overloaded. If isNeg, then r is k* from the paper and is
    // then set to 1, the first index of the ai in nonzero
    // otherwise it is already set to the first index of the ai
    Matrix66 D;
    if (isNeg) {
        for (int j = 0; j < 6; ++j) {
            D(j, 0) = -1 * Q[r][j];
        }
        r = 1;
        for (size_t i = r; i < nonzero.size() - 1; ++i) {
            for (size_t j = 0; j < 6; ++j) {
                D(j, i) = wrenches(pert, nonzero[nonzero.size() -
                                                    1] - 1, int(j))
                 - wrenches(size_t(pert), size_t(nonzero[i] - 1), int(j));
            }
        }
    } else {
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < 6; ++j) {
                D(j, i) = Q[nonzero[i]][j];
            }
        }
        for (size_t i = r; i < nonzero.size() - 1; ++i) {
            for (size_t j = 0; j < 6; ++j) {
                D(j, i) = wrenches(pert, nonzero[nonzero.size() -
                                                    1] - kSimplexVerts, int(j))
                        - wrenches(pert, nonzero[i] - kSimplexVerts, int(j));
            }
        }
    }
    return D;
}

// returns 1 if diff, 0 if not
static int handle_one_pert_pos(int pert, const MapN3& vertices, const MapN3& normals,
    const IntMapN3& triangles, const Map3& centroid, const IntMapN2& tri_nums,
    const MapNN& lp_vars, const SubWrenchTensor& wrenches,
                                const MapN6& perts, Vec6& grads, float mu) {
    std::vector<int> nonzero = get_nonzero_indices(lp_vars, pert, false);
    if (nonzero.size() != 7){
        return 0;
    }
    int r = 0;
    int s = 0;
    for (size_t i = 0; i < nonzero.size(); ++i) {
        if (nonzero[i] < kSimplexVerts) {
            r += 1;
        } else {
            s += 1;
        }
    }
    Matrix66 D = make_D(nonzero, wrenches, pert, r, false);
    Matrix66 D_inv = D.inverse();
    if (DEBUG) {
        std::cout<<"nonzero: "<<std::endl;
        for (size_t i = 0; i < nonzero.size(); ++i) std::cout<<nonzero[i] <<" ";
        std::cout<<std::endl;
        std::cout<<"D: "<<D<<std::endl;
        std::cout<<"D_INV: "<<D_inv<<std::endl;
    }
    RowVec6 e = make_e(r);
    Matrix66 jac = get_wrench_jac(pert, vertices, triangles, centroid,
         normals, tri_nums, lp_vars, wrenches, perts, grads, nonzero, r,
         false, mu);
    RowVec6 eDinv = e * D_inv;
    if (DEBUG) {
        std::cout <<"jac: "<<jac<<std::endl;
        std::cout <<"e :" << e<< std::endl;
        std::cout << "e * Dinv:"<<std::endl<< eDinv<<std::endl;
    }
    for (int i = 0; i < 6; ++i) {
        grads(i) += eDinv * jac.col(i);
    }
    return 1;
}

// returns 1 if diff, 0 if not
static int handle_one_pert_neg(int pert, const MapN3& vertices, const MapN3& normals,
    const IntMapN3& triangles, const Map3& centroid, const IntMapN2& tri_nums,
    const MapNN& lp_vars, const SubWrenchTensor& wrenches,
                                const MapN6& perts, Vec6& grads, float mu) {
    std::vector<int> nonzero = get_nonzero_indices(lp_vars, pert, true);
    if (DEBUG) {
        std::cout << "nonzero:"<<std::endl;
        for (size_t i = 0; i < nonzero.size(); ++i) std::cout<<nonzero[i] <<" ";
    }
    if (nonzero.size() != 7){
        return 0;
    }
    Matrix66 D = make_D(nonzero, wrenches, pert,
                            lp_vars(pert, lp_vars.cols() - 2), true);
    Matrix66 D_inv = D.inverse();
    if (DEBUG) {
        std::cout<<"D: "<<D<<std::endl;
        std::cout<<"D_INV: "<<D_inv<<std::endl;
    }
    RowVec6 e;
    e << 1,0,0,0,0,0;
    Matrix66 jac = get_wrench_jac(pert, vertices, triangles, centroid,
         normals, tri_nums, lp_vars, wrenches, perts, grads, nonzero, 1,
         true, mu);
    RowVec6 eDinv = e * D_inv;
    if (DEBUG) {
        std::cout <<"jac: "<<jac<<std::endl;
        std::cout <<"e :" << e<< std::endl;
        std::cout << "e * Dinv:"<<std::endl<< eDinv<<std::endl;
    }
    for (int i = 0; i < 6; ++i) {
        grads(i) += eDinv * jac.col(i);
    }
    return 1;
}

static void make_error_grad(Vec6& axisgrads, const MapN6& perts,
                        const Map3& centroid, int pert, int error_type) {
    if (error_type == kNotFound) {
        const Map3 grasp_center(perts.row(pert).data());
        Vec3 offset = grasp_center - centroid;
        if (offset.norm() < kBigEpsilon) {
            std::default_random_engine generator;        
            std::uniform_real_distribution<float> dist(-1.0,1.0);
            for (int i = 0; i < 3; ++i) {
                offset(i) = dist(generator);
            }
            offset /= offset.norm();
            for (int i = 0; i < 3; ++i) {
                axisgrads(i + 3) += offset(i);
            }
            return;
        } else {
            offset /= offset.norm();
            if(DEBUG) {
                std::cout << "offset:" << std::endl << offset <<std::endl;
            }
            for (int i = 0; i < 3; ++i) {
                axisgrads(i) += offset(i);
            }
        }
    } else if (error_type == kNotFoundPos) {
        const Map3 dir_pos(perts.row(pert).data() + 3);
        for (int i = 0; i < 3; ++i) {
            axisgrads(i) -= dir_pos(i); // need grad opposite of dir
        }
    } else if (error_type ==  kNotFoundNeg) {
        const Map3 dir_pos(perts.row(pert).data() + 3);
        for (int i = 0; i < 3; ++i) {
            axisgrads(i) += dir_pos(i); // need grad same as pos dir
        }
    }
}

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
                            float grasp[6]) {
    Vec6 axisgrads;
    for (int i = 0; i < 6; ++i) {
      grads(i) = 0;
      axisgrads(i) = 0;
    }
    int last_lp = lp_vars.cols() - 1;
    int num_perts = 0;
    if (DEBUG) {
        std::cout<<"grasp: "<<std::endl;
        for (int i = 0; i < 6; ++i) std::cout <<grasp[i];
        std::cout<<std::endl;
    }
    int total_perts = perts.rows();
    for (int pert = 0; pert  < total_perts; ++pert) {
        if (DEBUG) {
            std::cout <<"lp vars: "<<std::endl;
            std::cout<<lp_vars.row(pert) <<std::endl;
            std::cout <<"subwrench: " << wrenches <<std::endl;
        }
        if (tri_nums(pert, 0) == kNotFound || tri_nums(pert, 0) == kNotFoundPos
                                        || tri_nums(pert, 0) == kNotFoundNeg) {
            make_error_grad(axisgrads, perts, centroid, pert, tri_nums(pert, 0));
            num_perts += 1;
        }
        else if (lp_vars(pert, last_lp) == kNegSol) {
            num_perts += handle_one_pert_neg(pert, vertices, normals,
            triangles, centroid, tri_nums, lp_vars, wrenches, perts, axisgrads,
                                                                            mu);
        }
        else {
            num_perts += handle_one_pert_pos(pert, vertices, normals,
            triangles, centroid, tri_nums, lp_vars, wrenches, perts, axisgrads,
                                                                            mu);
        }
    }
    if (num_perts > 0) {
        axisgrads /= num_perts;
    } else {
        for (int i = 0; i < 6; ++i) {
            grads(i) = 0;
        }
        //std::cout << "no differentiable perts" << std::endl;
    }
    if (DEBUG)
        std::cout << "axis grads: " << std::endl << axisgrads<<std::endl;
    Map3 dir(&(perts.row(0).data()[3]));
    float difference[3];
    subtract_3_vector(&grasp[3], grasp, difference);
    Matrix33 normgrad = (Matrix33::Identity() - dir * dir.transpose()) /
                                l2_norm_3_vector(difference);

    for (int i = 0; i < 3; ++i) {
        grads(i) += axisgrads(i) / 2.0;
        grads(i + 3) += axisgrads(i) / 2.0;
    }
    Map3 ddir(&(axisgrads.data()[3]));
    if(DEBUG) {
        std::cout << "ddir" << std::endl << ddir << std::endl;
    }
    Vec3 dp = normgrad.transpose() * ddir;
    if(DEBUG) {
        std::cout << "dp: " << std::endl << dp << std::endl;
    }
    for (int i = 0; i < 3; ++i) {
        grads(i) -= dp(i);
        grads(i + 3) += dp(i);
    }

}

static int test_compute_centroid() {
    std::cout << "testing compute centroid" << std::endl;
    MatrixN3 vertices(4, 3);
    vertices << 0,0,0,
                1,0,0,
                0,1,0,
                -1,0,0;
    MapN3 vertices_map(vertices.data(), 4, 3);
    IntMatrixN3 triangles (2, 3);
    triangles << 0, 1, 2,
                 0, 2, 3;
    IntMapN3 tri_map(triangles.data(), 2, 3);
    Vec3 centroid = compute_centroid(vertices_map, tri_map);
    Vec3 true_centroid;
    true_centroid << 0, 1/3., 0;
    bool success = true;
    for (int i = 0; i < 3; ++i) {
        success = success && fabs(centroid(i) - true_centroid(i)) < kEpsilon;
    }
    if (success) {
        return 1;
    } else {
        std::cout << "compute centroid failed" << std::endl;
        std::cout << "true centroid: "<< std::endl<<true_centroid<<std::endl;
        std::cout << "output centroid:" << std::endl << centroid<<std::endl;
        return 0;
    }
}

static int test_subtract_3_vector() {
    std::cout << "testing subtract_3_vector" << std::endl;
    float a[3] = {0, 8.5, 19.4};
    float b[3] = {4, 4, 4};
    float c[3];
    subtract_3_vector(a, b, c);
    if (fabs(c[0] + 4) < kEpsilon && fabs(c[1] - 4.5) < kEpsilon && fabs(c[2] -
                                                        15.4) < kEpsilon) {
        return 1;
    } else {
        std::cout << "subtract_3_vector failed" << std::endl;
        return 0;
    }
}

static int test_cross_product() {
    std::cout << "testing cross product" << std::endl;
    float a[3] = {0, 8.5, 19.4};
    float b[3] = {4, 4, 4};
    float c[3];
    float d[3] = {-43.6, 77.6, -34.};
    cross_product(a, b, c);
    if (fabs(c[0] - d[0]) < kEpsilon && fabs(c[1] - d[1]) < kEpsilon && fabs(c[2] -
                                                        d[2]) < kEpsilon) {
        return 1;
    } else {
        std::cout << "cross product failed" << std::endl;
        return 0;
    }
}

static int test_l2_norm_3_vector() {
    std::cout << "testing norm" << std::endl;
    float true_norm = 21.46765008099;
    float a[3] = {-3.5, 8.5, 19.4};
    float norm = l2_norm_3_vector(a);
    if (fabs(norm - true_norm) < kEpsilon) {
        return 1;
    } else {
        std::cout << "norm failed" << std::endl;
        return 0;
    }
}

static int test_compute_triangle_normal() {
    std::cout << "testing triangle normal" << std::endl;
    IntMatrixN3 triangles(5, 3);
    triangles << 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6;
    MatrixN3 normals(7, 3);
    normals <<  0, 0, 1,
                0, 1, 0,
                1, 0, 0,
                0.7071, 0, 0.7071,
                0.7071, 0.7071, 0,
                0, 0.7071, 0.7071,
                0, -0.7071, -0.7071;
    float u = 0.25;
    float v = 0.35;
    IntMapN3 tri_map(triangles.data(), 5, 3);
    MapN3 normal_map(normals.data(), 7, 3);
    Vec3 normal = compute_triangle_normal(tri_map, 0, u, v, normal_map, true);
    Vec3 true_normal;
    true_normal << 0.35, 0.25, 0.4;
    true_normal /= true_normal.norm();
    bool success = fabs(normal(0) - true_normal(0)) < kEpsilon;
    success = success && fabs(normal(1) - true_normal(1)) < kEpsilon;
    success = success && fabs(normal(2) - true_normal(2)) < kEpsilon;
    if (success) {
        return 1;
    } else {
        std::cout << "triangle normal failed" << std::endl;
        std::cout << "triangles:" << std::endl;
        std::cout << triangles << std::endl;
        std::cout << "normals:" << std::endl;
        std::cout << normals << std::endl;
        std::cout << "u: "<< u<< std::endl;
        std::cout << "v: "<< v<< std::endl;
        std::cout << "triangle 0 normal: " << normal << std::endl;
        std::cout << "gt triangle 0 normal: " << true_normal << std::endl;

        return 0;
    }
}

static int test_find_num_triangles() {
    std::cout << "testing find num triangles" << std::endl;
    IntMatrixN3 triangles(5, 3);
    triangles << 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6; 
    IntMapN3 tri_map(triangles.data(), 5, 3);
    bool success = find_num_triangles(tri_map) == 5;
    triangles(1, 0) = 0;
    triangles(1, 1) = 0;
    triangles(1, 2) = 0;
    success = success && find_num_triangles(tri_map) == 1;
    if (success) {
        return 1;
    } else {
        std::cout << "triangle nums failed" << std::endl;
        return 0;
    }
}

static int test_contact_point() {
    std::cout << "testing contact point" << std::endl;
    float u = 0.5;
    float v = 0.5;
    float v0[3] = {0, 0, 0};
    float v1[3] = {2, 0, 0};
    float v2[3] = {0, 2, 0};
    Eigen::Vector3f pt = contact_point(u, v, v0, v1, v2);
    bool success = fabs(pt(0) - 1) < kEpsilon;
    success = success && fabs(pt(1) - 1) < kEpsilon;
    success = success && fabs(pt(2) - 0) < kEpsilon;
    if (success) {
        return 1;
    } else {
        std::cout << "contact point failed" << std::endl;
        return 0;
    }
}

static int test_compute_friction_cone() {
    std::cout << "testing compute_friction_cone" << std::endl;
    Vec3 normal;
    normal << 0, 0, 1;
    float mu = 0.7;
    int num_vertices = 5;
    MatrixN3 cone = compute_friction_cone(normal, mu, num_vertices);
    MatrixN3 true_cone(5, 3);
    true_cone << -0.49497, 0.49497, 1,
                 -0.62370,-0.31779, 1,
                 0.10950, -0.69137, 1,
                 0.69137, -0.10950, 1,
                 0.31779, 0.623689, 1;
    true_cone = true_cone * -1;
    bool success = true;
    for (int row = 0; row < 5; ++row) {
        for (int col = 0; col < 3; ++col) {
            success = success && fabs (cone(row, col) - true_cone(row, col)) <
                                                    kBigEpsilon;
        }
    }
    if (success) {
        return 1;
    } else {
        std::cout << "friction cone failed" << std::endl;
        std::cout << "cone: " << cone  << std::endl;
        std::cout << "true cone: " << true_cone  << std::endl;
        return 0;
    }
}

static int test_ray_triangle_intersection() {
    std::cout << "testing ray_triangle intersection" << std::endl;
    float point[3] = {0,0,0};
    float direction[3] = {0,0,1};
    float v0[3] = {-0.5, -0.5, 1};
    float v1[3] = {0.5, -0.5, 1};
    float v2[3] = {-0.5, 1, 1};
    float t = 0;
    float u = 0;
    float v = 0;
    bool success = ray_triangle_intersect(point, direction, v0, v1, v2, t, u,
                                                                          v);
    success = success && fabs(u - 0.5) < kEpsilon;
    success = success && fabs(v - 1/3.) < kEpsilon;
    if (success) {
        return 1;
    } else {
        std::cout << "ray-triangle intersection failed" << std::endl;
        return 0;
    }
}

static int test_compute_primitive_wrenches() {
    std::cout << "testing conpute primitive wrenches" << std::endl;
    Vec3 normal;
    normal << 0,0,1;
    Vec3 neg_normal;
    neg_normal << 0,0,-1;
    Vec3 contact_point;
    contact_point << 0,0,1;
    Vec3 neg_contact_point;
    neg_contact_point << 0,0,-1;
    Vec3 centroid;
    centroid << 0,0,0;
    float mu = 0.7;
    int num_vertices = 5;
    MatrixN3 true_cone(5, 3);
    true_cone << -0.49497, 0.49497, 1,
                 -0.62370,-0.31779, 1,
                 0.10950, -0.69137, 1,
                 0.69137, -0.10950, 1,
                 0.31779, 0.623689, 1;
    true_cone *= -1;
    MatrixN3 neg_true_cone(5, 3);
    neg_true_cone << -0.49497, 0.49497, 1,
                 0.31779, 0.623689, 1,
                 0.69137, -0.10950, 1,
                 0.10950, -0.69137, 1,
                 -0.62370,-0.31779, 1;
    MatrixN3 torques(5, 3);
    torques << 0.49497, 0.49497, 0,
               -0.31779, 0.6237, 0,
               -0.69137, -0.1095, 0,
               -0.1095, -0.69157, 0,
               0.623689, -0.31779, 0;
    MatrixN3 neg_torques(5, 3);
    neg_torques << 0.49497, 0.49497, 0,
               0.623689, -0.31779, 0,
               -0.1095, -0.69157, 0,
               -0.69137, -0.1095, 0,
               -0.31779, 0.6237, 0;
    MatrixN6 true_wrenches_pos(5, 6);
    true_wrenches_pos << true_cone, torques;
    MatrixN6 true_wrenches_neg(5, 6);
    true_wrenches_neg << neg_true_cone, neg_torques;
    MatrixN6 true_wrenches(10, 6);
    true_wrenches << true_wrenches_pos,
                     true_wrenches_neg;
    MatrixN6 wrenches = compute_primitive_wrenches(normal, neg_normal,
        contact_point, neg_contact_point, centroid, mu, num_vertices);
    bool success = true;
    for (int row = 0; row < wrenches.rows(); ++row) {
        for (int col = 0; col < wrenches.cols(); ++col) {
            success = success && fabs(wrenches(row, col) - true_wrenches(row,
                                            col)) < kBigEpsilon;
            if (!success) std::cout << row << col << std::endl;
        }
    }
    if (success) {
        return 1;
    } else {
        std::cout << "primitive wrenches failed" << std::endl;
        std::cout << "true wrenches" << true_wrenches << std::endl;
        std::cout << "output wrenches" << wrenches << std::endl;
        return 0;
    }
}

static int test_two_point_to_center_orientation() {
    std::cout << "testing grasp conversion" << std::endl;
    float input_grasp[6] = {1, 1, 1, -1, -1, -1};
    float grasp[6] = {1, 1, 1, -1, -1, -1};
    float true_grasp[6] = {0, 0, 0, -0.57735, -0.57735, -0.57735};
    bool success = true;
    two_point_to_center_orientation(grasp);
    for (int i  = 0; i < 6; ++i) {
        success = success && fabs(grasp[i] - true_grasp[i]) < kBigEpsilon;
    }
    if (success) {
        return 1;
    } else {
        std::cout << "grasp conversion failed" << std::endl;
        std::cout << "input grasp" << input_grasp << std::endl;
        for (int i = 0; i < 6; ++i)  std::cout << input_grasp[i] << std::endl;
        std::cout << "true grasp" << true_grasp << std::endl;
        for (int i = 0; i < 6; ++i)  std::cout << true_grasp[i] << std::endl;
        std::cout << "output grasp" << grasp << std::endl;
        for (int i = 0; i < 6; ++i)  std::cout << grasp[i] << std::endl;
        return 0;
    }

}

static int test_co_hull_contains_zero() {
    std::cout << "testing co_hull_contains_zero" << std::endl;

    MatrixN6 wrenches(8, 6);
    wrenches << Q[0][0], Q[0][1], Q[0][2], Q[0][3], Q[0][4], Q[0][5],
                Q[1][0], Q[1][1], Q[1][2], Q[1][3], Q[1][4], Q[1][5],
                Q[2][0], Q[2][1], Q[2][2], Q[2][3], Q[2][4], Q[2][5],
                Q[3][0], Q[3][1], Q[3][2], Q[3][3], Q[3][4], Q[3][5],
                Q[4][0], Q[4][1], Q[4][2], Q[4][3], Q[4][4], Q[4][5],
                Q[5][0], Q[5][1], Q[5][2], Q[5][3], Q[5][4], Q[5][5],
                Q[6][0], Q[6][1], Q[6][2], Q[6][3], Q[6][4], Q[6][5],
                1, 1, 1, 1, 1, 1;
    bool success = co_hull_contains_zero(wrenches);
    MatrixN6 new_wrenches(4,6);
    new_wrenches << 1,1,1,1,1,1,
                2,0,2,2,2,2,
                0,3,3,3,3,3,
                4,1,0,3,2,1;
    success = success && !co_hull_contains_zero(new_wrenches);
    if (success) {
        return 1;
    } else {
        std::cout << "co_hull_contains_zero failed" << std::endl;
        std::cout << "first hull" << std::endl << wrenches << std::endl;
        std::cout << "second hull" << std::endl << new_wrenches << std::endl;
        return 0;
    }
}


static int test_compute_q_neg() {
    std::cout << "testing compute_q_neg" << std::endl;
    int num_cone_edges = 4;
    int batch_num = 0;
    int grasp_num = 0;
    int pert_num = 0;
    Eigen::Tensor<float, 4, Eigen::RowMajor> lp(1, 1, 1, num_cone_edges * 2 + 7);
    LpTensor lp_e(lp.data(), 1, 1, 1, num_cone_edges * 2 + 7);
    MatrixN6 wrenches(8, 6);
    wrenches << 0.212, -1.298, 0.0, 0.0, 0.0, 0.001,
                10 * Q[0][0], 10 * Q[0][1], 10 * Q[0][2], 10 * Q[0][3], 10 * Q[0][4], 10 * Q[0][5],
                10 * Q[1][0], 10 * Q[1][1], 10 * Q[1][2], 10 * Q[1][3], 10 * Q[1][4], 10 * Q[1][5],
                10 * Q[2][0], 10 * Q[2][1], 10 * Q[2][2], 10 * Q[2][3], 10 * Q[2][4], 10 * Q[2][5],
                10 * Q[3][0], 10 * Q[3][1], 10 * Q[3][2], 10 * Q[3][3], 10 * Q[3][4], 10 * Q[3][5],
                10 * Q[4][0], 10 * Q[4][1], 10 * Q[4][2], 10 * Q[4][3], 10 * Q[4][4], 10 * Q[4][5],
                10 * Q[6][0], 10 * Q[6][1], 10 * Q[6][2], 10 * Q[6][3], 10 * Q[6][4], 10 * Q[6][5],
                9 * Q[0][0], 9 * Q[0][1], 9 * Q[0][2], 9 * Q[0][3], 9 * Q[0][4], 9 * Q[0][5];
    float q = compute_q_neg(wrenches,  batch_num, grasp_num,
                                pert_num, lp_e);
    bool success = fabs(q + 1) < kBigEpsilon; // we know it's not exactly 1 but
                                             //  it should be close
    success = success && lp_e(0, 0, 0, num_cone_edges * 2 + 6) == kNegSol;
    success = success && lp_e(0, 0, 0, num_cone_edges * 2 + 5) == 5;
    if (success) {
        return 1;
    } else {
        std::cout << "compute Q neg failed" << std::endl;
        std::cout << "Q: " << q << std::endl;
        std::cout << "wrenches" << std::endl << wrenches << std::endl;
        std::cout << "lp vars" << std::endl << lp_e << std::endl;
        return 0;
    }
}

static int test_compute_q_pos() {
    std::cout << "testing compute_q_pos" << std::endl;
    int num_cone_edges = 4;
    int batch_num = 0;
    int grasp_num = 0;
    int pert_num = 0;
    Eigen::Tensor<float, 4, Eigen::RowMajor> lp(1, 1, 1, num_cone_edges * 2 + 7);
    LpTensor lp_e(lp.data(), 1, 1, 1, num_cone_edges * 2 + 7);
    MatrixN6 wrenches(8, 6);
   wrenches << -1.4,0.4,0,0,0,0,
                -1.4, 0, 0, 0, 0, 0,
                -1.4, 0.2, 0, 0, 0, 0,
                -1.4, 0.2, 0.2, 0, 0, 0,
                -1.4, 0.2, 0.2, 0.2, 0, 0,
                -1.4, 0.2, 0.2, 0.2, 0, 0,
                -1.4, 0.2, 0.2, 0.2, 0.2, 0,
                -1.4, 0.2, 0.2, 0.2, 0.2, 0.2;
    float q = compute_q_pos(wrenches, batch_num, grasp_num,
                                pert_num, lp_e);
    bool success = fabs(q - 1) < kBigEpsilon; // we know it's not exactly 1 but
                                             //  it should be close
    if (success) {
        return 1;
    } else {
        std::cout << "compute Q pos failed" << std::endl;
        std::cout << "Q: " << q << std::endl;
        std::cout << "wrenches" << std::endl << wrenches << std::endl;
        std::cout << "lp vars" << std::endl << lp_e << std::endl;
        return 0;
    }
}

static int test_intersect() {
    std::cout << "testing intersect" << std::endl;
    IntMatrixN3 triangles(3, 3);
    triangles << 0, 1, 2,3,4,5, 6, 7, 8;
    IntMapN3 tmap(triangles.data(), 3, 3);
    MatrixN3 vertices(9, 3);
    vertices <<  -10, 11, 12,
                -11, 12, 13,
                -10, 11, 13,
                -1, -1, 1,
                -1, 1, 1,
                1, -1, 1,
                -1, -1, -1,
                -1, 1, -1,
                1, -1, -1;
    MapN3 vmap(vertices.data(), 9, 3);
    float point[3] = {0 ,0,0};
    float direction[3] = {0,0,1};
    int num_triangles = 3;
    int pos_tri_index = -1;
    int neg_tri_index = -1;
    float u = 0;
    float v = 0;
    float u_neg = 0;
    float v_neg = 0;
    float true_u = 0.5;
    float true_v = 0.5;
    float true_u_neg = 0.5;
    float true_v_neg = 0.5;
    bool success = intersect(point, direction, vmap, tmap,
        num_triangles, pos_tri_index, neg_tri_index, u, v, u_neg, v_neg) == 0;
    
    success = success && fabs(u - true_u) < kEpsilon;
    success = success && fabs(v - true_v) < kEpsilon;
    success = success && fabs(u_neg - true_u_neg) < kEpsilon;
    success = success && fabs(v_neg - true_v_neg) < kEpsilon;
    success = success && pos_tri_index == 1;
    success = success && neg_tri_index == 2;

    if (success) {
        return 1;
    } else {
        std::cout << "intersection failed" << std::endl;
        std::cout << "triangles:" << std::endl;
        std::cout << triangles << std::endl;
        std::cout << "vertices:" << std::endl;
        std::cout << vertices << std::endl;
        std::cout << "u: "<< u<< std::endl;
        std::cout << "v: "<< v<< std::endl;
        std::cout << "u_neg: "<< u_neg<< std::endl;
        std::cout << "v_neg: "<< v_neg<< std::endl;
        std::cout << "pos_tri_index: " << pos_tri_index << std::endl;
        std::cout << "neg_tri_index: " << neg_tri_index << std::endl;
        return 0;
    }
}

static int test_get_nonzero_indices() {
    std::cout << "testing get_nonzero_indices" << std::endl;
    Eigen::Matrix<float, 2, 9, Eigen::RowMajor> lp_vars;
    lp_vars<< 0,1,2,3,0,4,5,0,1,
              1,0,0,1,2,3,4,5,0;
    MapNN lp(lp_vars.data(), 2, 9);
    std::vector<int> nonzero = get_nonzero_indices(lp, 0, false);
    bool success = nonzero.size() == 6;
    success = nonzero[0] == 1 && nonzero[1] == 2 && nonzero[2] == 3 &&
              nonzero[3] == 5 && nonzero[4] == 6 && nonzero[8] && success;
    nonzero = get_nonzero_indices(lp, 1, true);
    success = success && nonzero.size() == 1 && nonzero[0] == 0;
    if (success) {
        return 1;
    } else {
        std::cout << "get_nonzero_indices failed" << std::endl;
        return 0;
    }
}

static int test_make_e() {
    std::cout << "testing make_e" << std::endl;
    RowVec6 e = make_e(1);
    RowVec6 true_e;
    true_e << 1,0,0,0,0,0;
    bool success = true;
    for (int i = 0; i < 6; ++i) {
        success = success && e(i) == true_e(i);
    }
    e = make_e(5);
    true_e << 1,1,1,1,1,0;
    for (int i = 0; i < 6; ++i) {
        success = success && e(i) == true_e(i);
    }
    if (success) {
        return 1;
    } else {
        std::cout << "make_e failed" << std::endl;
        return 0;
    }
}

static int test_make_D() {
    std::cout << "testing make_D" << std::endl;
    std::vector<int> nonzero = {0, 1, 3, 7, 8, 9, 10};
    int r = 3;
    int pert = 0;
    Eigen::Tensor<float, 3, Eigen::RowMajor, long int> wrench(1, 4, 6);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 6; ++j) {
            wrench(0, i, j) = i;
        }
    }
    SubWrenchTensor wrenches(wrench.data(), 1, 4, 6);
    Matrix66 D = make_D(nonzero, wrenches, pert, r, false);
    Matrix66 true_D;
    for (int i = 0; i < 6; ++i) {
        true_D(i, 0) = Q[0][i];
        true_D(i, 1) = Q[1][i];
        true_D(i, 2) = Q[3][i];
        true_D(i, 3) = wrenches(pert, 3, i) - wrenches(pert, 0, i);
        true_D(i, 4) = wrenches(pert, 3, i) - wrenches(pert, 1, i);
        true_D(i, 5) = wrenches(pert, 3, i) - wrenches(pert, 2, i);
    }
    bool success = true;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            success = success && fabs(D(i, j) - true_D(i, j)) < kEpsilon;
        }
    }
    Eigen::Tensor<float, 3, Eigen::RowMajor, long int> wrench2(1, 6, 6);
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            wrench2(0, i, j) = i;
        }
    }
    int r2 = 4;
    std::vector<int> nonzero2 = {0,1,2,3,4,5,6};
    SubWrenchTensor wrenches2(wrench2.data(), 1, 6, 6);
    Matrix66 D2 = make_D(nonzero2, wrenches2, pert, r2, true);
    Matrix66 true_D2;
    for (int i = 0; i < 6; ++i) {
        true_D2(i, 0) = -1 * Q[r2][i];
        true_D2(i, 1) = wrenches2(pert, 5, i) - wrenches2(pert, 0, i);
        true_D2(i, 2) = wrenches2(pert, 5, i) - wrenches2(pert, 1, i);
        true_D2(i, 3) = wrenches2(pert, 5, i) - wrenches2(pert, 2, i);
        true_D2(i, 4) = wrenches2(pert, 5, i) - wrenches2(pert, 3, i);
        true_D2(i, 5) = wrenches2(pert, 5, i) - wrenches2(pert, 4, i);
    }
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            success = success && fabs(D2(i, j) - true_D2(i, j)) < kEpsilon;
        }
    }
    if (success) {
        return 1;
    } else {
        std::cout << "make D failed" << std::endl;
        std::cout << "wrenches" << std::endl;
        std::cout << wrench << std::endl;
        std::cout << "D: " << std::endl << D <<std::endl;
        std::cout << "True D: " << std::endl << true_D << std::endl;
        std::cout << "second wrenches" << std::endl;
        std::cout << wrench2 << std::endl;
        std::cout << "D2: " << std::endl << D2 <<std::endl;
        std::cout << "True D2: " << std::endl << true_D2 << std::endl;
        return 0;
    }
}

static int test_make_error_grad() {
    std::cout << "testing make_error_grad" << std::endl;
    Vec6 axisgrads;
    for (int i = 0; i < 6; ++i) {
        axisgrads(i) = 0;
    }
    MatrixN6 perts(1, 6);
    perts << 1, 1, 1, 0, 1, 2;
    MapN6 pert_map(perts.data(), 1, 6);
    Vec3 ctr;
    ctr << 1, -1, 1;
    Map3 centroid(ctr.data());
    int pert = 0;
    make_error_grad(axisgrads, pert_map, centroid, pert, kNotFound);
    Vec6 true_grad;
    true_grad << 0, 1, 0, 0, 0, 0;
    bool success = true;
    for (int i = 0; i < 6; ++i) {
        success = success && fabs(axisgrads(i) - true_grad(i)) < kEpsilon;
    }
    if (success) {
        return 1;
    } else {
        std::cout << "make error grad failed" << std::endl;
        std::cout << "grad: "<<std::endl<<axisgrads<<std::endl;
        std::cout << "true grad: " << std::endl << true_grad << std::endl;
        return 0;
    }
}


int main(int argc, char* argv[]) {
    std::cout << "starting" << std::endl;
    int passed = 0;
    passed += test_compute_centroid();
    passed += test_subtract_3_vector();
    passed += test_cross_product();
    passed += test_l2_norm_3_vector();
    passed += test_compute_triangle_normal();
    passed += test_find_num_triangles();
    passed += test_contact_point();
    passed += test_compute_friction_cone();
    passed += test_ray_triangle_intersection();
    passed += test_compute_primitive_wrenches();
    passed += test_two_point_to_center_orientation();
    passed += test_co_hull_contains_zero();
    passed += test_compute_q_neg();
    passed += test_compute_q_pos();
    passed += test_intersect();
    passed += test_get_nonzero_indices();
    passed += test_make_e();
    passed += test_make_D();
    passed += test_make_error_grad();
    std::cout << "19 tests conducted" << std::endl;
    std::cout << passed << " passed" << std::endl;
}



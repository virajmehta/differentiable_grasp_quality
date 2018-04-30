import tensorflow as tf
from tensorflow.python.framework import ops
try:
    gq_module = tf.load_op_library('tf_grasp_quality_so.so')
except Exception as e:
    gq_module = tf.load_op_library('tf_grasp_quality_so.so')
_mu = None
def grasp_quality(grasps, vertices, triangles, normals,
num_grasp_perturbations=1,
                 mu=0.5, num_cone_edges=11):
    '''
input:
    grasps : batch_size * #grasps per object * 6
    vertices : batch_size * #max_num_vertices * 3
    triangles : batch_size * #max_num_triangles * 3
returns:
    quality : batch_size * #grasps_per_object
    '''
    _mu = mu
    grasps2, vertices2, triangles2, quality, triangle_nums, lp_vars, wrenches, centroids, perturbations, normals2 = \
                        gq_module.grasp_quality(grasps, vertices, triangles,
                        normals,
                        num_grasp_perturbations=num_grasp_perturbations,
                        mu=mu,
                        num_cone_edges=num_cone_edges)
    quality2 = gq_module.grasp_quality_helper(grasps2, vertices2, triangles,
        quality, triangle_nums, lp_vars, wrenches, centroids, perturbations,
        normals2);
    return quality2

@ops.RegisterGradient("GraspQuality")
def _grasp_quality_grad(op, d_grasps, d_vertices, d_triangles, d_quality,
    d_triangle_nums, d_lp_vars, d_wrenches, d_centroids, d_perturbations, d_normals):
    return [d_grasps, None, None, None]

@ops.RegisterGradient("GraspQualityHelper")
def _grasp_quality_helper_grad(op, grads):

    grasps = op.inputs[0]
    vertices = op.inputs[1]
    triangles = op.inputs[2]
    quality = op.inputs[3]
    triangle_nums = op.inputs[4]
    lp_vars = op.inputs[5]
    wrenches = op.inputs[6]
    centroids = op.inputs[7]
    perturbations = op.inputs[8]
    normals = op.inputs[9]
    grad = gq_module.grasp_qual_grad(grasps, vertices, triangles, quality,
                triangle_nums, lp_vars, wrenches, centroids, perturbations,
                normals, mu=_mu)
    return [grad, None, None, None, None, None, None, None, None, None]

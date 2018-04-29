import os
import numpy as np
import tensorflow as tf
import tf_grasp_quality as gq
import _init_paths
from ipdb import set_trace as db
from lib.data_io import read_mesh, category_model_id_pair
from lib.config import cfg

def compute_vertex_normals(vertices, triangles):
    vertex_normals = np.zeros(vertices.shape)
    for i in range(triangles.shape[0]):
        v0 = vertices[triangles[i, 0], :]
        v1 = vertices[triangles[i, 1], :]
        v2 = vertices[triangles[i, 2], :]
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        normal = np.cross(v0v1, v0v2) * 100
        vertex_normals[triangles[i,0], :] += normal
        vertex_normals[triangles[i,1], :] += normal
        vertex_normals[triangles[i,2], :] += normal
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    for i in range(vertices.shape[0]):
        if norms[i] == 0:
            continue
        vertex_normals[i, :] /= norms[i]
    return vertex_normals

def evaluate_grasps(cat, model_id):
    grasp_fn = cfg.DIR.GRASP_PATH%(cat, model_id)
    try:
        grasps = np.load(grasp_fn)
    except:
        return
    mesh = read_mesh(cfg.DIR.MODEL_PATH%(cat, model_id))
    if mesh is None:
        return
    vertices, triangles, _ = mesh
    normals = compute_vertex_normals(vertices, triangles)
    normals = np.expand_dims(normals, 0)
    vertices = np.expand_dims(vertices, 0)
    triangles = np.expand_dims(triangles, 0)
    grasps_e = np.expand_dims(grasps, 0)
    with tf.Session() as sess:
        grasps_ = tf.placeholder(tf.float32, shape=grasps_e.shape, name='grasps')
        triangles_ = tf.placeholder(tf.int32, shape=(1,
                      triangles.shape[1], 3), name='triangles')
        vertices_ = tf.placeholder(tf.float32, shape=(1,
                      vertices.shape[1], 3), name='vertices')
        normals_ = tf.placeholder(tf.float32, shape=(1,
                      vertices.shape[1], 3), name='normals')
        quality = gq.grasp_quality(grasps_, vertices_, triangles_, normals_)
        feed_dict = {triangles_: triangles, vertices_: vertices,
                       normals_: normals, grasps_: grasps_e}
        qualities = sess.run([quality], feed_dict=feed_dict)[0]
    qualities = qualities[0,:]
    order = np.argsort(qualities)
    grasps = grasps[order,:]
    qualities = qualities[order]
    np.save(grasp_fn, grasps)
    root = os.path.dirname(grasp_fn)
    try:
        qual_fn = os.path.join(root, 'qualities.npy')
        np.save(qual_fn, qualities)
    except Exception as e:
        print(e)
    print('grasps evaluated for cat %s model %s'%(cat, model_id))

def main():
    for category, model_id in category_model_id_pair(dataset_portion=[0,1]):
        evaluate_grasps(category, model_id)
if __name__=='__main__':
    main()

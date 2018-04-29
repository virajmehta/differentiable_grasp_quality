import os
import sys
import numpy as np
import tensorflow as tf
from tf_grasp_quality import grasp_quality
from ipdb import set_trace as db

kEpsilon = 1e-5
kMaxIterations = 1000 #probably way too big
kLearningRate = 0.1
output_grasp_dir = 'test_data/outputs/'
log_dir = output_grasp_dir + 'logdir/'


def gq_gradient_descent(vertices_data, triangles_data, normals_data,
                                                        all_grasps):
    all_out_grasps = np.zeros((0, kMaxIterations + 1, 6))
    in_qualities = []
    out_qualities = []
    for g in range(len(all_grasps)):
        out_grasps = np.zeros((kMaxIterations+ 1, 6))
        grasps_data = all_grasps[g:g+1, :]
        out_grasps[0, :] = grasps_data[0,:]
        grasps_data = np.expand_dims(grasps_data, 0)
        with tf.Session() as sess:
            # build input variables
            grasps_init = tf.constant(grasps_data, dtype=tf.float32)
            grasps = tf.get_variable('grasps', dtype=tf.float32,
                                                    initializer=grasps_init)
            triangles = tf.placeholder(tf.int32, shape=(1,
                          triangles_data.shape[1], 3), name='triangles')
            vertices = tf.placeholder(tf.float32, shape=(1,
                          vertices_data.shape[1], 3), name='vertices')
            normals = tf.placeholder(tf.float32, shape=(1,
                          vertices_data.shape[1], 3), name='normals')
            quality = grasp_quality(grasps, vertices, triangles, normals)

            # setup tf optimization
            train_step = tf.train.AdagradOptimizer(kLearningRate).minimize(quality)
            #train_step = tf.train.GradientDescentOptimizer(kLearningRate).minimize(quality)

            # setup loss logging
            quality_log = tf.squeeze(quality)
            summary = tf.summary.scalar('quality_log', quality_log)
            summary = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(log_dir)

            # get ready to go
            original_score = None
            tf.global_variables_initializer().run()
            min_loss = 17   # higher than worst
            min_loss_index = -1
            for i in range(kMaxIterations):
                feed_dict = {triangles: triangles_data, vertices: vertices_data,
                                                          normals: normals_data}
                grasps_out, quality_out, _, summary_out = sess.run([grasps,
                            quality, train_step, summary], feed_dict=feed_dict)
                train_writer.add_summary(summary_out, i)
                if quality_out[0,0] < min_loss:
                    min_loss = quality_out[0,0]
                    min_loss_index = i
                if i % 10 == 0:
                    print('descent step %d loss: %f'%(i, quality_out[0,0]))
                out_grasps[i+1,:] = grasps_out[0,0,:]
                if quality_out[0,0] < kEpsilon:
                    print('descent step %d loss: %f'%(i, quality_out[0,0]))
                    out_grasps[-1,:] = grasps_out[0,0,:]
                    out_qualities.append(quality_out[0,0])
                    break
                if i == 0:
                    in_qualities.append(quality_out[0,0])
                    original_score = quality_out[0,0]
                if i == kMaxIterations - 1:
                    out_grasps[min_loss_index + 1:-1] = 0
                    out_grasps[-1,:] = out_grasps[min_loss_index, :]
                    out_qualities.append(min_loss)
            train_writer.close()
            print('grasp: ', g)
            print('original_score', original_score)
            print('final_score', min_loss)
        out_grasps = np.expand_dims(out_grasps, 0)
        all_out_grasps = np.concatenate((all_out_grasps, out_grasps))
        tf.reset_default_graph()
    return all_out_grasps, in_qualities, out_qualities 

def load_input(name):
    vertices_path = 'test_data/%s_vertices.npy'%name
    triangles_path = 'test_data/%s_triangles.npy'%name
    normals_path = 'test_data/%s_normals.npy'%name
    grasps_path = 'test_data/%s_grasps.npy'%name
    vertices_data = np.load(vertices_path)
    num_vertices = vertices_data.shape[0]
    triangles_data = np.load(triangles_path)
    triangles_data = triangles_data.astype(int)
    num_triangles = triangles_data.shape[0]
    normals_data = np.load(normals_path)
    all_grasps = np.load(grasps_path)
    return vertices_data, triangles_data, normals_data, all_grasps


def process_input(vertices_data, triangles_data, normals_data, grasps_data):
    centroid = np.mean(vertices_data, axis=0, keepdims=True)
    vertices_data -= centroid
    centroid2 = np.array([centroid[0,i] for i in range(3) for _ in range(2)])
    centroid2 = np.expand_dims(centroid2, 0)
    grasps_data -= centroid2
    max_norm = np.max(np.linalg.norm(vertices_data, axis=1))
    vertices_data /= max_norm
    grasps_data /= max_norm
    vertices = np.expand_dims(vertices_data, 0)
    triangles = np.expand_dims(triangles_data, 0)
    normals = np.expand_dims(normals_data, 0)
    return max_norm, centroid, vertices, triangles, normals, grasps_data

def process_output(grasps, scale, offset):
    grasps *= scale
    offset = np.array([offset[0,i] for i in range(3) for _ in range(2)])
    for i in range(grasps.shape[0]):
        for j in range(grasps.shape[1]):
            if grasps[i,j,0] != 0 or grasps[i,j,1] != 0 or grasps[i,j,2] != 0\
              or grasps[i,j,3] != 0 or grasps[i,j,4] != 0 or grasps[i,j,5] != 0:
                grasps[i,j,:] += offset
    return grasps


def save_grasps(grasps, name, cat=""):
    all_grasp_name = os.path.join(output_grasp_dir, cat, '%s_grasps_path'%(name))
    np.save(all_grasp_name, grasps)
    io_grasp_name = os.path.join(output_grasp_dir, cat, '%s_grasps_io'%(name))
    first_grasps = grasps[:,0,:]
    last_grasps = grasps[:,-1,:]
    io_grasps = np.concatenate((first_grasps, last_grasps))
    np.save(io_grasp_name, io_grasps)



if __name__ == '__main__':
    vertices_data, triangles_data, normals_data, grasps_data = load_input(
                                                                   sys.argv[1])
    scale, centroid, vertices, triangles, normals, grasps = process_input(
                      vertices_data, triangles_data, normals_data, grasps_data)
    out_grasps_raw, iq, oq = gq_gradient_descent(vertices, triangles, normals, grasps)
    out_grasps_processed = process_output(out_grasps_raw, scale, centroid)
    save_grasps(out_grasps_processed, sys.argv[1])

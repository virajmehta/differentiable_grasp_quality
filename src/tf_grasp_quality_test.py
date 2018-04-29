import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gradient_checker
from tf_grasp_quality import grasp_quality
from ipdb import set_trace as db

vertices_path = 'test_data/teapot_vertices.npy'
triangles_path = 'test_data/teapot_triangles.npy'
normals_path = 'test_data/teapot_normals.npy'
grasps_path = 'test_data/teapot_grasps.npy'

gq_module = tf.load_op_library('tf_grasp_quality_so.so')
class GQTest(tf.test.TestCase):
    def testGQ(self):
    # need to setup mesh, grasps
        vertices = np.load(vertices_path)
        triangles = np.load(triangles_path)
        normals = np.load(normals_path)
        grasps = np.load(grasps_path)
        vertices = np.expand_dims(vertices, 0)
        triangles = np.expand_dims(triangles, 0)
        grasps = np.expand_dims(grasps, 0)
        print(grasps)
        normals = np.expand_dims(normals, 0)
        with self.test_session():
            result = grasp_quality(grasps, vertices, triangles, normals)
            print('evaluating')
            out = result.eval()
            print(out)

    def testGQGrad(self):
        vertices_data = np.load(vertices_path)
        num_vertices = vertices_data.shape[0]
        triangles_data = np.load(triangles_path)
        triangles_data = triangles_data.astype(int)
        num_triangles = triangles_data.shape[0]
        normals_data = np.load(normals_path)
        all_grasps = np.load(grasps_path)
        #all_grasps = all_grasps[2:3, :]
        vertices_data = np.expand_dims(vertices_data, 0)
        triangles_data = np.expand_dims(triangles_data, 0)
        normals_data = np.expand_dims(normals_data, 0)
        for g in range(len(all_grasps)):
            grasps_data = all_grasps[g:g+1]
            grasps_data = np.expand_dims(grasps_data, 0)
            with self.test_session():
               grasps = tf.placeholder(tf.float32, shape=grasps_data.shape)
               triangles = tf.placeholder(tf.int32, shape=(1, num_triangles, 3))
               vertices = tf.placeholder(tf.float32, shape=(1, num_vertices, 3))
               normals = tf.placeholder(tf.float32, shape=(1, num_vertices, 3))
               result = grasp_quality(grasps, vertices, triangles, normals)
               feed_dict = {triangles: triangles_data, vertices: vertices_data,
                                normals: normals_data}
               print('evaluating gradients')
               t_grad, n_grad =tf.test.compute_gradient(grasps, 
                                                        grasps_data.shape, 
                                                        result, 
                                                        result.get_shape().as_list(),
                                                        x_init_value=grasps_data,
                                                        extra_feed_dict=feed_dict)
               print('t_grad')
               print(t_grad)
               print('n_grad')
               print(n_grad)
               db()

if __name__ == '__main__':
    z = GQTest()
    z.testGQ()
    db()
    z.testGQGrad()
#    tf.test.main()

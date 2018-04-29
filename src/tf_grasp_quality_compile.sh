TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

echo $TF_INC
COIN_INC='/cvgl2/u/virajm/DeformNet_tf/tf_ops/coin-Clp/include/coin/'

export PKG_CONFIG_PATH='/cvgl2/u/virajm/DeformNet_tf/tf_ops/coin-Clp/lib/pkgconfig/'
echo $PKG_CONFIG_PATH

set -e
echo 'g++'
echo 'compiling gq obj'
g++ -std=c++11 -Wall -shared grasp_quality.cpp -o grasp_quality.o  $(pkg-config --libs clp) -fPIC -I $TF_INC -I $COIN_INC -O2
echo 'compiling tf_gq obj'
g++ -std=c++11 -Wall -shared tf_grasp_quality.cpp grasp_quality.o -o tf_grasp_quality_so.so -fPIC -I $TF_INC -I $COIN_INC -O2
echo 'compiling gq_test executable'
g++ -std=c++11 -Wall grasp_quality.cpp -o grasp_quality_test $(pkg-config --libs clp) -I $TF_INC -I $COIN_INC -g -O0
#g++ -std=c++11 -c tf_grasp_quality_test.cpp -o tf_grasp_quality_test.o
#g++ -std=c++11 -o tf_grasp_quality_test -o tf_grasp_quality_test.o tf_grasp_quality_so.so -I $TF_INC

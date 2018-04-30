# Differentiable Grasp Quality Tensorflow Op (dGQ)
### Viraj Mehta

This is a differentiable grasp quality operation based on an extension from the paper Synthesis of
force-closure grasps on 3-D objects based on the Q distance.

For a general summary of the mathematics behind this see the below paper.

Zhu, Xiangyang, and Jun Wang. "Synthesis of force-closure grasps on 3-D objects based on the Q distance." IEEE Transactions on robotics and Automation 19.4 (2003): 669-679.

The primary contribution of this piece of software is to extend the mathematics of the previous article to the general setting of a triangular mesh.

## Install instructions:
### Dependencies:
* [Tensorflow](tensorflow.org) (It's a tensorflow op so it needs the bones). You should be able to straightforwardly install this according to the instructions on the website. This has been tested with version 1.8 but most recent versions should be fine.

* [Clp](https://projects.coin-or.org/Clp) (for linear programming). This is more of a pain, but only because it isn't as well-documented. In a separate directory ($COINDIR from here on out), install coin-Clp according to the fairly straightforward instructions on the website.
### Installation:
1. Setting up to compile: Edit the `COIN_INC` variable in `src/tf_grasp_quality_compile.sh` to point to the include directory in your Clp installation. My path is there as an example.
2. Compile and verify: Making sure you've got your TF virtual environment active, `cd` into `src` and run `./tf_grasp_quality_compile.sh`. I'll probably make this a Makefile eventually, but for now, I'm lazy. This should be clean and give you a bunch of files in various states of library and executableness. As a sanity check, run `./grasp_quality_test`. This should also be clean.
3. Link the dynamic library for Clp: Navigate to the `activate` file of your virtual env and edit it to include the following line

  ```
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COINDIR/lib/
  ```
  beneath the line that says
  ```
  export VIRTUAL_ENV
  ```

  For reference, mine looks like
  ```
  export VIRTUAL_ENV
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvgl2/u/virajm/lib/coin-Clp/lib/
  ```

  This again is probably some kind of Unix bastardization that there are best practices for. At minimum, you should probably remove this from your environment variable when the env is deactivated. I have that, but I'm not sure it actually matters. If someone wants to tell me how to do this in a more smart way, please do that.

Congrats, you're done with installation!

## Usage

The paradigm of use of this operation is if you have triangular mesh(es) and grasps specified by two 3d points that you want to compute the quality of and then have a derivative of grasp position with respect to quality. For example, if you had a neural net that took some input and predicted the finger positions of a gripper, you could train it to give better grasps by passing the grasps and ground-truth meshes into this operation and get the qualities of the grasps and their derivatives in a way that TF can automatically maximize the quality.

There is example usage in `src/tf_grasp_quality_test.py`.

First, you need to load the module in your TF script by
```
dgq_module = tf.load_op_library('tf_grasp_quality_so.so')
```
You're welcome to move the library to wherever you wanna use it in your code, but there can sometimes be weirdness with also importing the `grasp_quality.o` and other files. There are definitely better resources for debugging linkage issues, so good luck with that. You're also welcome to point your code at the directory the libraries currently live in. (I was really planning on a bin directory but alas, maybe later)

You'll also need to use the python code in `tf_grasp_quality.py`. You can import it from the file like in typical python. Feel free to move files to where they're convenient and hack whatever needs to be hacked.

Finally, the interface:
The function defined is `grasp_quality`, a Tensorflow function that takes some arguments:

* `Grasps`: an object batch size by number of grasps per object by 6 Tensor
* `Vertices`: an object batch size by a constant maximum number of vertices per object by 3 Tensor
* `Triangles`: an object batch size by a constant maximumm number of triangles per object by 3 Tensor

and returns the `Quality`, a batch size * grasps per object Tensor. There is a registered gradient so Tensorflow is able to backpropagate starting at this function.

## Issues

Maybe I should use the GitHub issues system to manage the issues with this piece of software, but I figured these need to be more visible than that, because boy does this software have issues. This is by no means an exhaustive list, and the software suffers in quality due to being written by someone who hadn't written a Tensorflow operation and then wrote the longest one I can find anywhere.

One issue is that dGQ doesn't handle concavity of meshes well. Since the ray-tracing necessary to make the fingers land on triangles of the mesh doesn't have a great way to find where in the mesh the grasp was intended to be put, I chose the heuristic of finding the farthest triangles incident on the ray of the grasp and assuming this was where the grasp was supposed to be. This has obvious issues. Consider grasping the lip of a mug for example.

A possible solution will be made obvious now.

Another issue is that dGQ doesn't have anything handling finger width. If we could know the max / min grip width of the hand, we could toss out irrelevant triangles and make sure the grasps are realistic. This is doable in principle, but will be challenging to implement. This solves the previous issue as then we'll only consider the relevant triangles and concavitiy will be irrelevant. But it's unclear what to do for a gradient in this case.

Another issue is that the code is really convoluted because I wanted to cache certain expensive computations from the forward pass for the gradient computation. If someone who know TF/C++ better wants to suggest a better way to do this than passing a bunch of intermediate tensors between the forward and backward parts of the op, let me know.

Another issue is that the gradient is just not the same as the numerical gradient. Through much anguish and sleeplessness, I found that this was due to the discontinuous nature of mesh triangles in the numerical gradient computation, and not a huge deal.

Finally, this code is currently pretty committed to the idea of a parallel-jaw gripper. The mathematics is more general and I think it would be a valuable contribution to generalize this to arbitrary fingers. There are a couple problems with this though. First, it's hard to parametrize a multiple-finger grasp. Second, multi-finger grippers have more complicated restrictions to their configuration. I am even more unsure how to either enforce these or pass gradients in these situations, though the math still works in their absence.

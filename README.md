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
1 Setting up to compile: Edit the `COIN_INC` variable in `src/tf_grasp_quality_compile.sh` to point to the include directory in your Clp installation. My path is there as an example.
2 Compile and verify: Making sure you've got your TF virtual environment active, `cd` into `src` and run `./tf_grasp_quality_compile.sh`. I'll probably make this a Makefile eventually, but for now, I'm lazy. This should be clean and give you a bunch of files in various states of library and executableness. As a sanity check, run `./grasp_quality_test`. This should also be clean.
3 Link the dynamic library for Clp: Navigate to the `activate` file of your virtual env and edit it to include the following line

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





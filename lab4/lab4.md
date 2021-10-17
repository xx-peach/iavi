# <center> Lab 4 for IAVI: Projector-Camera-Based Stereo Vision

<center>
    Group Number: 3 <br/>
    Group Members: 林政楷、谢涛、陈蔚
</center>

## Contents

[toc]



## Introduction

This report is about the fourth lab of the course **Intelligent Acquisition of Visual Information** in fall. 2021.

This topic of this lab is to do projector-camera-based stereo vision and two possible ways are provided:

- Approach #1: 2 Cameras + 1 Projector;
- Approach #2: 1 Camera   + 1 Projector;

We've chosen the Approach #2 and finish the lab with the help of the paper and application got from website [http://mesh.brown.edu/calibration/].

To begin with, the most important steps of this lab includes:

1. Carefully calibrate the camera-projector system;
2. Project a pattern (the pattern we choose is `Gray Code`) and find correspondence between the pattern and the pixels;
3. Find the Depth map and 3D point cloud;

which would be discussed in details in the following sections.



## Environment

The applications and packages used in this lab are:

- Pylon Viewer for GUI to capture images;
- `Python` with `opencv` to generate patterns;
- **Dual Basler Dart Machine Vision USB-3 Color Camera**;
- Application from [http://mesh.brown.edu/calibration/] to do calibration and reconstruct;



## Experiment Principles

### Camera calibration

The methods and steps to do camera calibration is same as what we do in the previous labs.

We could use the `opencv` functions to find the corners of the chessboard and then do calibration.

[这里可以放一张软件里生成的标出了角点的图片？（雾）]

### Decoding

Decoding is the process of finding the correspondence between the image pixels and patterns.

The main concepts of this step is to use the bright and dark pattern projected by the projector to encode the  positions.

[这里可以放一下我们生成的格雷码图片？（雾）]

For example, if a position in the image undergoes a sequence of *bright-dark-bright-bright-dark-bright*, the code of the position is $101101  = 45$.

[这里可以放一张软件里生成的那个五颜六色的图片？（雾）]

### Projector calibration

The mathematical model used to describe projector in this lab is actually the same as that of a camera. Thus, we could use the same method to calibrate the projector too.

The main difficulty to be solved is how to find the coordinates of the chessboard corners in the projector coordinate. Considering that there is a matrix to describe the transformation from the camera coordinate to the projector coordinate, so we could reduce it into an optimization problem as follows:
$$
\hat{H} = \arg \min_{H} \sum_{\forall p} || q - Hp ||^2,
$$
where $H \in \R^{3\times 3}$ is the transformation matrix,  $p = [x, y, 1]^T$ in the image coordinates, $q = [col, row, 1]^T$ in the view of projector.

### Stereo calibration

Stereo calibration means finding the relative rotation and translation between projector and camera, which have been realized in the previous labs.

Similarly, we could use the `stereoCalibrate()` function from `opencv` to get the result.



## Process of Experiment





## Thoughts





## Appendix: Source Code


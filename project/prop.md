# <center>Make 2D Pictures Alive</center>

## 1 Introduction

长久以来，我们在日常生活中认识并接触到的图片总是二维的，也就是我们熟知的`RGB`图像。在将三维空间中的点投射至二维成像平面的过程中，每个实际坐标点的深度信息就丢失了，这也正是我们觉得拍出来的照片不够“立体”的原因之一。为了保留这些原始信息，我们可以拍摄所谓的`RGB-D`图像，即由一个普通的`RGB`三通道彩色图像和与之对应的`Depth`图像组成。

根据我们小组的调查了解，在图像深度信息捕获的研究领域中，较为成熟的技术有双目立体匹配。通过双目相机拍摄同一场景的左、右两幅视点图像，进而运用立体匹配匹配算法得到视差图，最终获取拍摄图像的深度信息。有了图像的深度信息，我们就可以对图像画面中的物体进行分层处理，得到物体间的遮挡关系，并尝试恢复被遮挡的画面。在将上述信息综合到图像处理中，我们就有能力根据有限张图片甚至一张图片，在一定角度范围内还原出不同视角所能看到的画面，从而让我们的图片更加的立体生动。而当我们还原出不同视角的图片之后，我们就有了通过拍摄几张不同位置的照片来还原出一个类似连续移动相机拍摄的视频效果。



## 2 Related Works

### 2.1 binocular stereo vision measure depth

1. Distance Measurement System Based on Binocular Stereo Vision[<sup>[1]</sup>](#refer-anchor-1);
2. A Real-Time Range Finding System with Binocular Stereo Vision[<sup>[2]</sup>](#refer-anchor-2);

### 2.2 deep learning measure depth

1. Monocular Depth Estimation Based On Deep Learning: An Overview[<sup>[3]</sup>](#refer-anchor-3);

2. Unsupervised Learning of Depth and Ego-Motion from Monocular Video Using 3D Geometric Constraints[<sup>[4]</sup>](#refer-anchor-4);
3. Depth Prediction Without the Sensors: Leveraging Structure for Unsupervised Learning from Monocular Videos[<sup>[5]</sup>](#refer-anchor-5);

### 2.3 3D Reconstruction

1. 3D Photography using Context-aware Layered Depth Inpainting[<sup>[6]</sup>](#refer-anchor-6);



## 3 Our Approach

经过初步的讨论，我们小组考虑有以下可能用到以下的方法：

1. 通过不同相机的摄影结果完成深度估计，用于后续分析；
2. 根据不同角度摄制的图片，计算得到相应变换的矩阵，实现遮挡部分图像的生成；
3. 尝试另一种办法：通过构建神经网路得到深度信息，并生成各个视角的图像；



## 4 Basic Methods

### 4.1 Depth Estimation

1. 可以使用光学计算的方法，双目视觉是模拟人类视觉原理，使用计算机被动感知距离的方法。从两个或者多个点观察一个物体，获取在不同视角下的图像，根据图像之间像素的匹配关系，通过三角测量原理计算出像素之间的偏移来获取物体的三维信息：

   <img src="https://pic1.zhimg.com/80/v2-e2fe1b96e23369a907195c14c569c240_720w.jpg" alt="img" style="zoom: 50%;" />

   + 这其中涉及到两张图片的像素点匹配问题，这一点我们现在想到的主要方法有：

     + 通过 opencv 的特征点匹配算法来实现；

     + 先对图片做语义分割，将图像分成一些部分，然后将每个部分视为一个点来做匹配；

2. 可以使用神经网络的方式来拟合图像的深度信息；

### 4.2 Background Image Recovery

1. 线性插值来恢复中间图像；
2. 神经网络插值来恢复中间图像；



## 5 Time Line

以下是项目的大致时间线：

1. 9.29：提交项目开题报告，确定研究方向；
2. 10.3：搜集相关资料，了解目前已有的实现方法与原理；
3. 10.7：复现能够找到的各种已有方法；
4. 10.15：使用双目相机完成一个Demo，实现较小范围的生成，得到初步成果；
5. 10.20：搭建轨道模拟多个相机，实现较大范围的生成；
6. 10.28：优化算法与演示效果；
7. 11.3：完成实验报告与展示；



## 6 Reference

<div id="refer-anchor-1"></div>[1] Liu, Zhengzhen, and Tianding Chen. "Distance measurement system based on binocular stereo vision." *2009 International Joint Conference on Artificial Intelligence*. IEEE, 2009.

<div id="refer-anchor-2"></div>[2] Lai, Xiao-bo, Hai-shun Wang, and Yue-hong Xu. "A real-time range finding system with binocular stereo vision." *International Journal of Advanced Robotic Systems* 9.1 (2012): 26.

<div id="refer-anchor-3"></div>[3] Zhao, Chaoqiang, et al. "Monocular depth estimation based on deep learning: An overview." *Science China Technological Sciences* (2020): 1-16.

<div id="refer-anchor-4"></div>[4] Mahjourian, Reza, Martin Wicke, and Anelia Angelova. "Unsupervised learning of depth and ego-motion from monocular video using 3d geometric constraints." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2018.

<div id="refer-anchor-5"></div>[5] Casser, Vincent, et al. "Depth prediction without the sensors: Leveraging structure for unsupervised learning from monocular videos." Proceedings of the AAAI conference on artificial intelligence. Vol. 33. No. 01. 2019.

<div id="refer-anchor-6"></div>[6] Meng-Li Shi, Shih-Yang Su, Johannes Kopf, Jia-Bin Huang. “3D Photography Using Context-Aware Layered Depth Inpainting” 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.


<div class="cover" style="page-break-after:always;font-family:方正公文仿宋;width:100%;height:100%;border:none;margin: 0 auto;text-align:center;">
    <div style="width:60%;margin: 0 auto;height:0;padding-bottom:10%;">
        </br>
        <img src="https://gitee.com/Keldos-Li/picture/raw/master/img/%E6%A0%A1%E5%90%8D-%E9%BB%91%E8%89%B2.svg" alt="校名" style="width:100%;"/>
    </div>
    </br></br></br></br></br>
    <div style="width:60%;margin: 0 auto;height:0;padding-bottom:40%;">
        <img src="https://gitee.com/Keldos-Li/picture/raw/master/img/%E6%A0%A1%E5%BE%BD-%E9%BB%91%E8%89%B2.svg" alt="校徽" style="width:100%;"/>
	</div>
    </br></br></br></br></br></br></br></br>
    <span style="font-family:华文黑体Bold;text-align:center;font-size:20pt;margin: 10pt auto;line-height:30pt;">Intellectual Acquisiton and 3D Reconstruction<br/> Base on Smart Car</span>
    <p style="text-align:center;font-size:14pt;margin: 0 auto">Course Project</p>
    </br>
    </br>
    <table style="border:none;text-align:center;width:72%;font-family:仿宋;font-size:14px; margin: 0 auto;">
    <tbody style="font-family:方正公文仿宋;font-size:12pt;">
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">题　　目</td>
    		<td style="width:2%">：</td> 
            <td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">Intellectual Acquisition and 3D Reconstruction Base on Smart Car</td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">上课时间</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">周三上午345节</td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">授课教师</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">吴鸿智</td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">姓　　名</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">林政楷、谢涛、陈蔚</td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">学　　号</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">3190104811、0000000000、3190100925</td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">组　　别</td>
    		<td style="width:%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">Group 3</td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">日　　期</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">2021年11月6日</td>     </tr>
    </tbody>              
    </table>
</div>

<div STYLE="page-break-after: always;"></div>

# Intellectual Acquisition and 3D Reconstruction<br/>Base on Smart Car



<div style="width:80px;float:left;"><b>Abstract:</b></div> 
<div style="overflow:hidden;">With the development of camera system and computer vision, tons of methods to reconstruct the 3D shape of some object based on the given photos have merged. This article has come up with as well as realise a smart-car-based system to automatically take pictures and get the reconstruction result.</div>

<div style="width:80px;float:left;"><b>Key Words:</b></div> 
<div style="overflow:hidden;">Automatic System; Smart Car Control; </div>



## Introduction

Recently years have viewed a enormous surge of the 3D reconstruction tools. Through using computer vision tools like `COLMAP` and so on, users are blessed with the ability to reconstruct the 3D shape of the object from even some photos took by their own phones.

However, such reconstruction may suffer from low accuracy and high error rate because the photos taken are not so accurate themselves and the parameters of the camera is hard to know. 

On the contrary, this article introduce a method of using the smart car to automatically acquire high quality photos from various perspective and finally achieve relatively good reconstruction results.



## Related Works





## Experiment Principles

### Smart Car Control



### Path Determination

In order to reconstruct an object into a 3D digital model, we need to take pictures of the target from different angles. And the best way to do this is to keep the object stay in the center of each photo so that we can achieve a high quality result. The simplest method to complete this task is to drive around the object in a perfect circle and make the camera focus on the target at the first place. However, due to some limits on the accuracy of hardware, it's impractical to let the car drive around in a standard circle. So our final solution is to let the car walk a rectangle around and keep a reasonable distance from the object so that the photos can cover every aspect of the object. Although it simplifies the controlling algorithm of the walking part, it also leads to another tricky problem: how to make sure that the object always stays in the frame while moving? If we enlarge the rectangle, we will lose a lot of details due to the larger distance. If we want to stay close to the target, then we must find a way to make the hold of camera adjust itself to keep the object in the center. And that brings us to the next part of our method. 

### Object Recognition and Tracking during 

As mention before, we need an algorithm for the holder of camera to adjust itself to force the camera to face directly towards the object. The first method popped in our head is to use object detection algorithm. We can use `Tensorflow` or `Pytorch` to build up a `CNN` so that it can mark all the object in one frame. After that, what we need to do is to adjust the holder according to the relative position of our target to the center of frame. But soon after designing and loading such a neural network to the micro-computer of this car and testing, our group have found this idea was impractical: first, the performance of the micro-computer on this car in running this neural network is rather pool, it can only process quite few pictures given a limited time and that contradicts to our original purpose that is to make the whole process faster. The other reason is that it can only recognize some certain targets that have been used to train the network, we don't want to limit our method to a finite region of objects. 

Finally we work out a substitution that turns out quite efficient and accurate, that is to track some conspicuous features of our target which can be distinguished from the background. In this project we decided to let the holder to trace the unique color of our target while moving. This idea sounds a little easy and cheap, but it may be the optimal choice under such a limited computation force and needs of speed. In order to implement this method, there are some steps we need to follow:

1.  Pick up a certain color of the object and set up the lower and upper bounds of the chosen color in `HSV` format. In our project we mostly use red.
2. After the acquisition of pictures, we need to mark out the region with target color in the given picture. We can apply the thresholds defined in step one to design the corresponding mask.
3. From the results of step 2, we need to work out the position of target region. Our group decides to use the center of the minimum enclosing circle of the area to stand for its position in the picture. Since there could be multiple areas with the same color, we choose the largest continuous region and use `cv2.findContours` and `cv2.minEnclosingCenter` to get the result.
4. With the coordinate of the center of our target in the image plane, now we can set our camera facing towards the object. This can be done by calculating `PID` pulses and sending them to different motors of the camera holder. 

At the beginning of our project, we decided to make intermittent stops with predefined intervals while moving. Again this contradicts to our general purpose to make it faster and sometimes the object can easily slip out of the frame if the step size is too large between two intervals. So we adopt `multi-thread programming` techniques in our method. We use two threads to perform different tasks so that now both parts of movement and camera adjustment can be carried out concurrently.



### High Quality Photo Acquisition

Since we introduce the `multi-threading programming` technique to our algorithm in the last part, it's a natural thinking to add one more thread for capturing images. But the movements of both car and camera could cause motion blur. So we have to stop the car and camera before taking pictures. However, that wasn't enough to complete the task of image acquisition. Our group has encountered another tricky problem in this part. We found that sometimes we couldn't get the expected result from the camera and some pictures doesn't match the scene of the position from which it had been taken. In other words, a 'delay effect' of image caption has been observed. After looking up the some document of `Opencv` and analyzing the retrieved images, we inferred the problem lies in the function `cv2.VideoCapture`. In order to support the continuous playing of real-time video, the camera will capture images in a certain rate and store them in a buffer. And since we only use a small amount of images to support our algorithm, those unused images won't be flushed and keep accumulating so that new images couldn't get into the buffer. Thus we can't retrieve the result we want as a result of this delay. So in order to fix this problem, again we use a multi-thread solution. We create another thread that is to keep reading pictures from the buffer so that those unused images can be consumed in time and as a consequence the blocking of new incoming images can be avoided.

### 3D Reconstruction





## Equipment and Device

### Parameters about the Smart Car

The parameters of the components used in this project is listed below:

- Raspberry 4B
  - 4G Memory
- LM25965 circuit to provide power
- TB6612FNG circuit to drive the motor
- SG90 motor to drive the camera
- 480*600 USB camera 

### Experiment Steps

The project has follows the steps described below.

1. Buy the components of the smart car
2. Assemble them
2. Install operating system for the Raspberry
3. Try to control the motion of car
4. Complete color detection and auto-tracking part
5. Take pictures and do 3D reconstruction



## Experiment Result

### The Smart Car

The smart car is assembled from the parts bought from the website.

<img src="./imgs/1.jpg" alt="car1" style="zoom:5%;" /><img src="./imgs/5.jpg" alt="car1" style="zoom:5%;" /><img src="./imgs/9.jpg" alt="car1" style="zoom:5%;" />

### Photos Acquired

Using the methods described in Chapter 3, a series of photos could be grabbed by the smart-car-camera system.

The photos would be from all perspective, here are two examples.

<img src="./imgs/12.jpg" alt="car1" style="zoom:45%;" /><img src="./imgs/13.jpg" alt="car1" style="zoom:45%;" />

### Reconstruction Results





## Further Works





## References

 


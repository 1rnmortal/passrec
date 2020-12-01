## opencv-python project  

python3.7

##### Summary  

  	使用opencv-pyhon从航拍俯视角度识别行人
  	计算行人过斑马线的轨迹，绘制出s-t图



#####  Implement

​	  先使用opencv去除背景，轮廓匹配识别出头顶，计算出每个轮廓的唯一值，跟踪这个轮廓在每一帧	  的位置并记录下来，即刻绘制出图像
​	  处理视频时以 帧为t,像素为s

##### LIb

​	**opencv-python** **3.4.728**  

##### 	numpy  1.19.3
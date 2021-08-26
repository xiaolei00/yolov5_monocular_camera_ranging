# yolov5_monocular_camera_ranging
This repository is a project of monocular camera ranging, which object detection frame is yolov5. 
This project deal with real-time video. It just shows you a video directly which contains the type of object, the confidence of object and the distance from network camera to object. <br>
![result picture](https://github.com/xiaol-arch/yolov5_monocular_camera_ranging/blob/main/article_pic/02.jpg)
# yolov5
I just use the pretrained model of yolov5 directly.The detail of yolov5 is available here [YOLOv5 🚀 Vision AI ⭐](https://github.com/ultralytics/yolov5)   
The referenced project version is v5.0.The version of project does'nt matter. 
# quick start examples
## install
python>=3.6 and pytorch>= 1.7:   
`$ git clone https://github.com/xiaol-arch/yolov5_monocular_camera_ranging`<br>
`$ cd yolov5_monocular_camera_ranging`<br>
`$ pip install -r requirements.txt`   
## inference   
Runing following instruction, you can get a resultant video.<br>
This instruction is for video<br>
`$ python video.py`<br>
This instruction is for webcam<br>
`$ python distance.py`<br>
This intruction is for webcam which can track stuff<br>
`$ python track.py`
#  to do 
+ Modify issues of accuracy of ranging in program
+ Improve the README documentation
+ Add speed measurement to the program

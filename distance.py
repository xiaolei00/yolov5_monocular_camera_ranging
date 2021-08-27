import cv2
import queue
import threading
import sys
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import plot_one_box
import random
import numpy as np
from models.experimental import attempt_load
from estimateDistanceUtil import *
import ffmpeg
import socket
import struct

q = queue.Queue()
flag_queue = queue.Queue()

# init
def init():
    FILE = Path(__file__).absolute()
    sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

    device = torch.device('cuda:0')
    half = device.type != True  # half precision only supported on CUDA

    model = attempt_load('yolov5s.pt', map_location=device)  # load FP32 model
    imgsz = check_img_size(640, s=model.stride.max())  # check img_size

    if half:
        model.half()  # to FP16
    cudnn.benchmark = True  # set True to speed up constant image size inference

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    img01 = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img01.half() if half else img01) if device.type != 'cpu' else None  # run once
    return device, half, model, names, colors

def receive_dahua():
    print("start Receive, the webcam is dahua");
    cap = cv2.VideoCapture("rtsp://192.168.1.22:8554/hl_cam", cv2.CAP_FFMPEG);
    ret, frame = cap.read();
    q.put(frame);
    while ret:
        ret, frame = cap.read();
        q.put(frame);

def receive_huilian():
    print("start Receive, the webcam is huilian");
    webcam_path = "rtsp://192.168.2.22:8554/hl_cam";
    probe = ffmpeg.probe(webcam_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    r_frame_rate = video_stream['r_frame_rate'];
    avg_frame_rate = video_stream['avg_frame_rate'];
    print(f'width-----{width}')
    print(f'height----{height}')
    print(f'r_frame_rate----{r_frame_rate}')
    print(f'avg_frame_rate----{avg_frame_rate}')
    process = (
        ffmpeg
            .input(webcam_path)
            .output('pipe:', format='rawvideo', pix_fmt='bgr24', vframes=1e18)
            .run_async(pipe_stdout=True)
    );
    while True:
        in_bytes = process.stdout.read(width * height * 3);
        if not in_bytes:
            break;
        in_frame = (
            np
                .frombuffer(in_bytes, np.uint8)
                .reshape([height, width, 3])
        );
        q.put(in_frame);

def display_video(device, half, model, names, colors):
    print("start displaying");
    window_name = "192.168.2.22";
    cv2.namedWindow(window_name, flags = cv2.WINDOW_AUTOSIZE);
    # count = 0
    while True:
        if q.empty() != True:
            frame = q.get();
            imgs = [frame];
            img, pred = predict_img(imgs, device, half, model);
            im0, flag = draw_img_info(pred, imgs, img, names, colors);
            cv2.imshow(window_name, im0)
            cv2.waitKey(1)
            for data in flag:
                flag = data
                if flag == 1:
                    socket_flag[65] = 0x01
                else:
                    socket_flag[65] = 0x00
                flag_queue.put(socket_flag)
            # cv2.imwrite(f'bbb/{count}.jpg', v1)

def predict_img(imgs, device, half, model):
    img = [letterbox(x, new_shape=640, auto=True)[0] for x in imgs];
    # Stack
    img = np.stack(img, 0);
    # Convert
    img = img[:, :, :, ::-1].transpose(0, 3, 1, 2);  # BGR to RGB, to bsx3x416x416
    img = np.ascontiguousarray(img);

    img = torch.from_numpy(img).to(device);
    img = img.half() if half else img.float();  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0   # torch.Size([1, 3, 480, 640])

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.3, 0.45, classes=[0, 1, 2, 3, 5, 6, 7], agnostic=False)
    return img, pred

def draw_img_info(pred, imgs, img, names, colors):
    flag = []
    for i, det in enumerate(pred):  # detections per image
        s, im0 =  '%g: ' % i, imgs[i].copy()
        s += '%gx%g ' % img.shape[2:]  # print string
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                distance = 0
                if names[int(cls)] == 'person':
                    object_width_in_frame = int(xyxy[2]) - int(xyxy[0])
                    distance = distance_finder(focal_length_person, object_width_in_frame, KNOWN_PERSON_WIDTH);
                elif names[int(cls)] == 'bus':
                    object_width_in_frame = int(xyxy[2]) - int(xyxy[0]);
                    distance = distance_finder(focal_length_person, object_width_in_frame, KNOWN_BUS_WIDTH);
                elif names[int(cls)] == 'car':
                    object_width_in_frame = int(xyxy[2]) - int(xyxy[0]);
                    distance = distance_finder(focal_length_person, object_width_in_frame, KNOWN_CAR_WIDTH);
                elif names[int(cls)] == 'motorcycle':
                    object_width_in_frame = int(xyxy[2]) - int(xyxy[0]);
                    distance = distance_finder(focal_length_person, object_width_in_frame, KNOWN_MOTORCYCLE_WIDTH);
                if distance < 5:
                    flag.append(1)
                else:
                    flag.append(0)
                label = f'{names[int(cls)]} {conf:.2f} {distance:.3f}m'
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
        else:
            flag.append(3)
    return im0, flag

def ref_img_information(img_path, device, half, model):
    imgs = [cv2.imread(img_path)]
    img, pred = predict_img(imgs, device, half, model)
    focal_length_found = 0
    for i, det in enumerate(pred):  # detections per image
        im0 = imgs[i].copy()
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                if names[int(cls)] == 'person':
                    ref_image_object_width = int(xyxy[2]) - int(xyxy[0])
                    focal_length_found = focal_length(ref_image_object_width, KNOWN_PRESON_DISTANCE, KNOWN_PERSON_WIDTH)
                elif (names[int(cls)] == 'bus') and (round(float(conf), 2) == 0.61):
                    ref_image_object_width = (int(xyxy[2]) - 400) - int(xyxy[0])
                    focal_length_found = focal_length(ref_image_object_width, KNOWN_BUS_DISTANCE, KNOWN_BUS_WIDTH)
                elif (names[int(cls)] == 'car') and (round(float(conf), 2) == 0.47):
                    ref_image_object_width = (int(xyxy[2]) - 30) - int(xyxy[0])
                    focal_length_found = focal_length(ref_image_object_width, KNOWN_CAR_DISTANCE, KNOWN_CAR_WIDTH)
                elif (names[int(cls)] == 'motorcycle'):
                    ref_image_object_width = (int(xyxy[2])) - int(xyxy[0])
                    focal_length_found = focal_length(ref_image_object_width, KNOWN_MOTORCYCLE_DISTANCE, KNOWN_MOTORCYCLE_WIDTH)
    return focal_length_found

def speed_finder(covered_distance, time_token):
    speed = covered_distance /time_token;
    return speed;

def average_finder(complete_list, average_of_items):
    # finding the length of list.
    length_of_list = len(complete_list);

    # calculating tne number items to find the average of
    selected_item = length_of_list -average_of_items;

    # getting the list most recent items of list to find average of
    selected_items_list = complete_list[selected_item:];

    # finding the average
    average = sum(selected_items_list) / len(selected_items_list);

    return average;

def get_focal_length(device, half, model):
    bus_img_path = 'data/images/Ref_bus.jpg'
    car_img_path = 'data/images/Ref_car.jpg'
    motorcycle_img_path = 'data/images/Ref_motorcycle.jpg'
    person_img_path = 'data/images/Ref_person.png'

    focal_length_bus = ref_img_information(bus_img_path, device, half, model)
    focal_length_car = ref_img_information(car_img_path, device, half, model)
    focal_length_motorcycle = ref_img_information(motorcycle_img_path, device, half, model)
    focal_length_person = ref_img_information(person_img_path, device, half, model)
    return focal_length_bus, focal_length_car, focal_length_motorcycle, focal_length_person

def get_socket_flag():
    mycheck = 0
    flag = [0x23, 0x23, 0x23, 0x23, 0x4a, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00,
            0x01, 0x00, 0x00, 0x00, 0xFF, 0xCC, 0x0B, 0x01]
    for i in range(20, 65):
        flag.append(0x00)
    for i in range(65, 74):
        # 此处是我计算出来的值
        if i == 65:
            flag.append(0x01)
        elif i == 71:
            for j in range(0, 71):
                mycheck ^= flag[j]
            flag.append(mycheck)
        elif i == 72:
            flag.append(0xEE)
        elif i == 73:
            flag.append(0xDD)
        else:
            flag.append(0x00)
    return flag

def socket_send():
    time.sleep(2)
    host = '192.168.1.92'
    port = 6220
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 定义socket类型，网络通信，TCP
    s.connect((host, port))
    while True:
        if flag_queue.empty() != True:
            socket_flag = flag_queue.get();
            data = struct.pack("%dB" % (len(socket_flag)), *socket_flag)
            print(data)
            s.sendall(data)
            time.sleep(0.05)
    s.close()  # 关闭连接

device, half, model, names, colors = init()
focal_length_bus, focal_length_car, focal_length_motorcycle, focal_length_person = get_focal_length(device, half, model)
socket_flag = get_socket_flag()

if __name__ == '__main__':
    cond = threading.Condition();
    p1 = threading.Thread(target = receive_huilian);
    p2 = threading.Thread(target = display_video, args=(device, half, model, names, colors));
    p3 = threading.Thread(target=socket_send);

    p1.start();
    p2.start();
    p3.start();
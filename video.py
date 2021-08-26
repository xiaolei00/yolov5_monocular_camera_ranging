import cv2
import sys
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
import random
import numpy as np
from models.experimental import attempt_load
from estimateDistanceUtil import *

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

def predict_img(imgs, device, half, model):
    img = [letterbox(x, new_shape=640, auto=True)[0] for x in imgs]
    # Stack
    img = np.stack(img, 0)
    # Convert
    img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0   # torch.Size([1, 3, 480, 640])

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.25, 0.45, classes=[0, 1, 2, 3, 5, 6, 7], agnostic=False)
    return img, pred

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
                elif (names[int(cls)] == 'bus') and (round(float(conf), 2)==0.61):
                    ref_image_object_width = (int(xyxy[2]) - 400) - int(xyxy[0])
                    focal_length_found = focal_length(ref_image_object_width, KNOWN_BUS_DISTANCE, KNOWN_BUS_WIDTH)
                elif (names[int(cls)] == 'car') and (round(float(conf), 2)==0.47):
                    ref_image_object_width = (int(xyxy[2]) - 30) - int(xyxy[0])
                    focal_length_found = focal_length(ref_image_object_width, KNOWN_CAR_DISTANCE, KNOWN_CAR_WIDTH)
                elif (names[int(cls)]=='motorcycle'):
                    ref_image_object_width = (int(xyxy[2])) - int(xyxy[0])
                    focal_length_found = focal_length(ref_image_object_width, KNOWN_MOTORCYCLE_DISTANCE, KNOWN_MOTORCYCLE_WIDTH)
    return focal_length_found

if __name__ == '__main__':
    device, half, model, names, colors = init()
    bus_img_path = 'data/images/Ref_bus.jpg'
    car_img_path = 'data/images/Ref_car.jpg'
    motorcycle_img_path = 'data/images/Ref_motorcycle.jpg'
    person_img_path = 'data/images/Ref_person.png'

    focal_length_bus = ref_img_information(bus_img_path, device, half, model)
    focal_length_car = ref_img_information(car_img_path, device, half, model)
    focal_length_motorcycle = ref_img_information(motorcycle_img_path, device, half, model)
    focal_length_person = ref_img_information(person_img_path, device, half, model)

    video_path = 'data/video/testVideo.mp4'
    save_path = 'data/video/videoResult.mp4'
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f'Failed to open {video_path}'
    # get video information
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    ret, frame = cap.read()
    while(ret):
        imgs = [frame]
        img, pred = predict_img([frame], device, half, model)
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
                        distance = distance_finder(focal_length_person, object_width_in_frame, KNOWN_PERSON_WIDTH)
                    elif names[int(cls)] == 'bus':
                        object_width_in_frame = int(xyxy[2]) - int(xyxy[0])
                        distance = distance_finder(focal_length_person, object_width_in_frame, KNOWN_BUS_WIDTH)
                    elif names[int(cls)] == 'car':
                        object_width_in_frame = int(xyxy[2]) - int(xyxy[0])
                        distance = distance_finder(focal_length_person, object_width_in_frame, KNOWN_CAR_WIDTH)
                    elif names[int(cls)] == 'motorcycle':
                        object_width_in_frame = int(xyxy[2]) - int(xyxy[0])
                        distance = distance_finder(focal_length_person, object_width_in_frame, KNOWN_MOTORCYCLE_WIDTH)
                    label = f'{names[int(cls)]} {conf:.2f} {distance:.3f}m'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
        cv2.imshow('a', im0)
        cv2.waitKey(20)
        # transform frame to video
        vid_writer.write(im0)
        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()




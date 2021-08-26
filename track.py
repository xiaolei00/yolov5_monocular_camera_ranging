import cv2
import queue
import threading
import sys
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

from utils.augmentations import letterbox
from utils.downloads import attempt_download
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import plot_one_box
import random
import numpy as np
from models.experimental import attempt_load
from estimateDistanceUtil import *
import ffmpeg

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort


q = queue.Queue()
parameter = dict()

# init
def init():
    FILE = Path(__file__).absolute()
    sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

    deep_sort = 'deep_sort_pytorch/configs/deep_sort.yaml';
    deep_sort_weights = 'deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7';
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(deep_sort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

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
    return device, half, model, names, colors, deepsort

def receive_video():
    print("start Receive, the input is video");
    video_path = 'data/video/testVideo.mp4'
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f'Failed to open {video_path}'
    # get video information
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    parameter['fps'] = fps
    parameter['w'] = w
    parameter['h'] = h
    ret, frame = cap.read()
    while (ret):
        q.put(frame)
        ret, frame = cap.read()
    cap.release()

def receive_huilian():
    print('start Receive, the webcam is huilian');
    webcam_path = 'rtsp://192.168.1.22:8554/hl_cam';
    probe = ffmpeg.probe(webcam_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
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

def display_video_result(device, half, model, names, colors):
    print("start displaying");
    window_name = "video";
    cv2.namedWindow(window_name, flags=cv2.WINDOW_AUTOSIZE);

    save_path = 'data/video/videoResultTrack.mp4'
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), parameter['fps'], (parameter['w'], parameter['h']))
    while q.empty() != True:
        frame = q.get();
        imgs = [frame];
        img, pred = predict_img(imgs, device, half, model);
        v1 = draw_img_info(pred, imgs, img, names, colors);
        cv2.imshow(window_name, v1);
        cv2.waitKey(1)
        # transform frame to video
        vid_writer.write(v1)
    cv2.destroyAllWindows()

def display_webcam_result(device, half, model, names, colors):
    print("start displaying");
    window_name = "192.168.1.22";
    cv2.namedWindow(window_name, flags = cv2.WINDOW_AUTOSIZE);

    save_path = 'data/video/videoResultWebcam.mp4'
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30,
                                 (1024, 768))
    while True:
        if q.empty() != True:
            frame = q.get();
            imgs = [frame];
            img, pred = predict_img(imgs, device, half, model);
            v1 = draw_img_info(pred, imgs, img, names, colors);
            cv2.imshow(window_name, v1);
            vid_writer.write(v1);
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;

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
    pred = non_max_suppression(pred, 0.4, 0.45, classes=[0, 1, 2, 3, 5, 6, 7], agnostic=False)
    return img, pred

def draw_img_info(pred, imgs, img, names, colors):
    for i, det in enumerate(pred):  # detections per image
        s, im0 =  '%g: ' % i, imgs[i].copy()
        s += '%gx%g ' % img.shape[2:]  # print string
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # 这是deepsort的部分
            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]
            # pass detections to deepsort
            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss, im0)

            # draw boxes for visualization
            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, confs)):
                    distance = 0
                    bboxes = output[0:4]
                    id = output[4]
                    cls = output[5]
                    c = int(cls)  # integer class

                    if names[c] == 'person':
                        object_width_in_frame = int(bboxes[2]) - int(bboxes[0])
                        distance = distance_finder(focal_length_person, object_width_in_frame, KNOWN_PERSON_WIDTH);
                    elif names[c] == 'bus':
                        object_width_in_frame = int(bboxes[2]) - int(bboxes[0]);
                        distance = distance_finder(focal_length_person, object_width_in_frame, KNOWN_BUS_WIDTH);
                    elif names[c] == 'car':
                        object_width_in_frame = int(bboxes[2]) - int(bboxes[0]);
                        distance = distance_finder(focal_length_person, object_width_in_frame, KNOWN_CAR_WIDTH);
                    elif names[c] == 'motorcycle':
                        object_width_in_frame = int(bboxes[2]) - int(bboxes[0]);
                        distance = distance_finder(focal_length_person, object_width_in_frame, KNOWN_MOTORCYCLE_WIDTH);

                    index = id - 1
                    list_distance[index].append(distance)
                    distance_in_meter = average_finder(list_distance[index], 2)
                    if initial_distance[index] != 0:
                        change_in_distance[index] = initial_distance[index] - distance_in_meter
                        change_in_time[index] = time.time() - initial_time[index]
                        speed = speed_finder(change_in_distance[index], change_in_time[index])
                        list_speed[index].append(speed)
                        average_speed = average_finder(list_speed[index], 10)
                        if average_speed < 0:
                            average_speed = average_speed * -1
                    else:
                        average_speed = 0
                    initial_distance[index] = distance_in_meter
                    initial_time[index] = time.time()

                    label = f'{id} {names[c]} {conf:.2f} {distance:.3f}m {average_speed:.2f}m/s'
                    color = compute_color_for_id(id)
                    plot_one_box(bboxes, im0, label=label, color=color, line_thickness=2)
    return im0


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

def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

# speed parameter init
initial_time = [0 for i in range(99999)];
initial_distance = [0 for i in range(99999)];
change_in_time = [0 for i in range(99999)];
change_in_distance = [0 for i in range(99999)];

list_distance = [[] for i in range(99999)]
list_speed = [ [] for i in range(99999)]

device, half, model, names, colors, deepsort = init();
focal_length_bus, focal_length_car, focal_length_motorcycle, focal_length_person = get_focal_length(device, half, model);

if __name__ == '__main__':
    p1 = threading.Thread(target=receive_huilian);
    p2 = threading.Thread(target=display_webcam_result, args=(device, half, model, names, colors));

    p1.start();
    p2.start();

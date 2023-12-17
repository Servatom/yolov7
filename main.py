import argparse
import torch
from queue import Queue
import threading
import time
import logging
from utils.general import strip_optimizer
import requests
import base64
from io import BytesIO
import cv2
from detect import detect
from scripts import timeIt
import pyttsx3

LOGGER = logging.getLogger("CaptureEngine")
logging.basicConfig(level=logging.INFO)

results_queue = Queue()
image_queue = Queue()


class YOLOv7Client:
    opt = argparse.ArgumentParser()
    def __init__(self):
        @timeIt
        def infer(self, filename, results_queue):
            det = detect(save_img=False)
            if len(det):
                for *xyxy, conf, cls in reversed(det):
                    class_name = self.model.names[int(cls)]  # Get class name
                    results_queue.put(class_name)
        
        @timeIt
        def infer_api(self, img_str, results_queue):
            resp = requests.post(
                self.url,
                data=img_str,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            data = resp.json()
            if len(data["predictions"]) == 0:
                result = "No object detected"
            else:
                result = data["predictions"][0]["class"]
            results_queue.put(result)


class AudioEngine:
  def __init__(self):
    self.engine = pyttsx3.init()
    self.engine.setProperty('rate', 150)
    self.engine.setProperty('volume', 1.0)

  def say(self, message):
    self.engine.say(message)
    self.engine.runAndWait()

    
def audio_person():
  engine = AudioEngine()
  # check results queue and if a string is there say the string and pop it
  while True:
    if not results_queue.empty():
      result = results_queue.get()
      engine.say(result)
      print("Finished audio")

def capture_queue_filler(picam2):
    while True:
        img_str = picam2.get_image()
        image_queue.put(img_str)

def inference_manager(opt):
    yoloClient = YOLOv7Client()
    yoloClient.opt = opt

    while True:
        if not image_queue.empty():
            # spawn a thread to call rf.infer_api
            img_str = image_queue.get()
            thread = threading.Thread(target=yoloClient.infer, args=(results_queue,))
            thread.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    # inference_manager(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    # client = YOLOv7Client()
    # client.opt = opt

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect(opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt)
    # # start audio thread
    # audio_thread = threading.Thread(target=audio_person)
    # audio_thread.start()

    # # start image capture thread
    # image_thread = threading.Thread(target=capture_queue_filler)
    # image_thread.start()
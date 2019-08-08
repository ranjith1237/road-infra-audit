import cv2
import cv2
import torch
import json
from model import SCNN
from utils.prob2lines import getLane
from utils.transforms import *
import time
from collections import defaultdict

from lane_marking import segment_lanes

class Detector:

    def __init__(self,image):
        self.img = image
    
    def detect_TrafficSign(self):
        pass
    
    def detect_LaneMarking(self,frame_num,preloaded_params):
        segment_lanes(frame_num, self.img,preloaded_params)
        


def preload_lanemarking(weights_path):
    net = SCNN(pretrained=False)
    mean=(0.3598, 0.3653, 0.3662)
    std=(0.2573, 0.2663, 0.2756)
    transform = Compose(Resize((800, 288)), ToTensor(),
                    Normalize(mean=mean, std=std))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dict = torch.load(weights_path, map_location='cpu')
    net.load_state_dict(save_dict['net'])
    net.eval()
    net.to(device)
    return {
            'net':net,
            'transform':transform,
            'device':device
        }


if __name__ == "__main__":
    videoPath="../../road_infra/SCNN_Pytorch/2019-04-08-10-49-25_f.mp4"
    weights_path="/Neutron6/ranjith.reddy/Road-Infrastructure/weights/vgg_SCNN_DULR_w9.pth"
    preloaded_params = preload_lanemarking(weights_path)
    vid = cv2.VideoCapture(videoPath)
    fps = vid.get(cv2.CAP_PROP_FPS)
    print("FPS:{}".format(fps))
    frame_num=0
    while vid.isOpened():
        frame_num+=1
        ret, image = vid.read()
        detector = Detector(image)
        detector.detect_LaneMarking(frame_num,preloaded_params)
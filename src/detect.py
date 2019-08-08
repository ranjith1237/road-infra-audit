import cv2
import cv2
import torch
import json

#lane marking SCNN
from model import SCNN
from utils.prob2lines import getLane
from utils.transforms import *
import time
from collections import defaultdict

#traffic signs darknet yolo
from darknet import Darknet



from traffic_sign import traffic_detector
from street_light import streetlight_detector
from lane_marking import segment_lanes

class Detector:

    def __init__(self,image):
        self.img = image
    
    def detect_TrafficSign(self,frame_num,preloaded_params):
        traffic_detector(frame_num,self.img,preloaded_params)
    
    def detect_Streetlight(self,frame_num,preloaded_params):
        streetlight_detector(frame_num,self.img,preloaded_params)

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

def preload_trafficsigns(cfgfile,weightsfile):
    reso = 416
    num_classes = 5

    confidence = 0.5
    nms_thesh = 0.4


    CUDA = torch.cuda.is_available()

    print("Loading network.....")
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = reso
    inp_dim = int(model.net_info["height"])

    if CUDA:
        model.cuda()

    model.eval()
    classes_gtsrb = ['Speed Limit 20','Speed Limit 30','Speed Limit 40','Speed Limit 50','Speed Limit 60','Speed Limit 80','Speed Limit 100','Stop','No Entry','Compulsory Ahead Only','Compulsory Keep Left','Compulsory Keep Right','Compulsory Turn Right Ahead','Compulsory Turn Left Ahead','Compulsory Ahead or Turn Left','Compulsory Ahead or Turn Right','Give Way','Pedestrian Crossings','Hump or Rough Roads','Narrow Road Ahead','Roundabout','School Ahead','Red Light','Green Light','Person Light','Yellow Light','Not a Sign']
    classes_gtsrb[26]
    return {
            'CUDA':CUDA,
            'model':model,
            'reso':reso,
            'num_classes':num_classes,
            'confidence':confidence,
            'nms_thesh':nms_thesh,
            'inp_dim':inp_dim,
            'classes_gtsrb':classes_gtsrb
        }

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

def preload_trafficsigns(cfgfile,weightsfile):
    reso = 416
    num_classes = 5

    confidence = 0.5
    nms_thesh = 0.4


    CUDA = torch.cuda.is_available()

    print("Loading network.....")
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = reso
    inp_dim = int(model.net_info["height"])

    if CUDA:
        model.cuda()

    model.eval()
    classes_gtsrb = ['Speed Limit 20','Speed Limit 30','Speed Limit 40','Speed Limit 50','Speed Limit 60','Speed Limit 80','Speed Limit 100','Stop','No Entry','Compulsory Ahead Only','Compulsory Keep Left','Compulsory Keep Right','Compulsory Turn Right Ahead','Compulsory Turn Left Ahead','Compulsory Ahead or Turn Left','Compulsory Ahead or Turn Right','Give Way','Pedestrian Crossings','Hump or Rough Roads','Narrow Road Ahead','Roundabout','School Ahead','Red Light','Green Light','Person Light','Yellow Light','Not a Sign']
    classes_gtsrb[26]
    return {
            'CUDA':CUDA,
            'model':model,
            'reso':reso,
            'num_classes':num_classes,
            'confidence':confidence,
            'nms_thesh':nms_thesh,
            'inp_dim':inp_dim,
            'classes_gtsrb':classes_gtsrb
        }

def preload_streetlights(cfgfile,weightsfile):
    reso = 416
    num_classes = 5

    confidence = 0.5
    nms_thesh = 0.4


    CUDA = torch.cuda.is_available()

    print("Loading network.....")
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = reso
    inp_dim = int(model.net_info["height"])

    if CUDA:
        model.cuda()

    model.eval()
    classes_gtsrb = ['Speed Limit 20','Speed Limit 30','Speed Limit 40','Speed Limit 50','Speed Limit 60','Speed Limit 80','Speed Limit 100','Stop','No Entry','Compulsory Ahead Only','Compulsory Keep Left','Compulsory Keep Right','Compulsory Turn Right Ahead','Compulsory Turn Left Ahead','Compulsory Ahead or Turn Left','Compulsory Ahead or Turn Right','Give Way','Pedestrian Crossings','Hump or Rough Roads','Narrow Road Ahead','Roundabout','School Ahead','Red Light','Green Light','Person Light','Yellow Light','Not a Sign']
    classes_gtsrb[26]
    return {
            'CUDA':CUDA,
            'model':model,
            'reso':reso,
            'num_classes':num_classes,
            'confidence':confidence,
            'nms_thesh':nms_thesh,
            'inp_dim':inp_dim,
            'classes_gtsrb':classes_gtsrb
        }
        


if __name__ == "__main__":
    videoPath="/Neutron6/ranjith.reddy/traffic_signs/pytorch-yolo-v3/2019-07-03-13-25-01/Video/capture.mp4"
    weights_path="/Neutron6/ranjith.reddy/Road-Infrastructure/weights/vgg_SCNN_DULR_w9.pth"
    preloaded_params_lanes = preload_lanemarking(weights_path)
    cfgfile_trafficsigns = '/Neutron6/ranjith.reddy/traffic_signs/tad_yolov3_5.cfg'
    weightsfile_yolotraffic = '/Neutron6/ranjith.reddy/traffic_signs/tad_yolov3_5_6000.weights'
    preloaded_params_signs=preload_trafficsigns(cfgfile_trafficsigns,weightsfile_yolotraffic)
    cfgfile_streetlights = '/Neutron6/ranjith.reddy/Road-Infrastructure/weights/streetlights.cfg'
    weightsfile_yolostreetlight = '/Neutron6/ranjith.reddy/Road-Infrastructure/weights/streetlights_best.weights'
    preloaded_params_lights=preload_streetlights(cfgfile_streetlights,weightsfile_yolostreetlight)
    vid = cv2.VideoCapture(videoPath)
    fps = vid.get(cv2.CAP_PROP_FPS)
    print("FPS:{}".format(fps))
    frame_num=0
    while vid.isOpened():
        frame_num+=1
        ret, image = vid.read()
        detector = Detector(image)
        detector.detect_LaneMarking(frame_num,preloaded_params_lanes)
        detector.detect_TrafficSign(frame_num,preloaded_params_signs)
        detector.detect_Streetlight(frame_num,preloaded_params_lights)
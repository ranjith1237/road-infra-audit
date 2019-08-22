import os
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
from sign_attributes import classify_attributes
import torchvision
from torchvision import models
import pdb
class Detector:
    def __init__(self,image,out_path):
        self.img = image
        self.out_path = out_path
        self.street_light_frames = os.path.join(self.out_path,'street_light_frames')
        self.traffic_sign_frames = os.path.join(self.out_path,'traffic_sign_frames')
        self.lane_mark_frames = os.path.join(self.out_path,'lane_mark_frames')
        try:
            os.mkdir(self.street_light_frames)
        except:
            print("street light folder already present")
        try:
            os.mkdir(self.traffic_sign_frames)
        except:
            print("traffic sign folder already present")
        try:
            os.mkdir(self.lane_mark_frames)
        except:
            print("lane marking folder already present")
        
    def get_attribute_name(self,sign_attribute_pred):
        if sign_attribute_pred==0:
            return "Faded"
        elif sign_attribute_pred==1:
            return "Normal"
        elif sign_attribute_pred==2:
            return "Rusted"
        return "Normal"
    
    def detect_TrafficSign(self,frame_num,preloaded_params,preloaded_params_attributes):
        attribute_data={}
        preloaded_params['out_path']=self.out_path
        preloaded_params['traffic_sign_frames']=self.traffic_sign_frames
        tracked_objects = traffic_detector(frame_num,self.img,preloaded_params)
        if tracked_objects is not None:
            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                try:
                    print(int(x1),int(x2),int(y1),int(y2),frame_num)
                    img_crop = self.img[int(y1):int(y2),int(x1):int(x2),:]
                    sign_attribute_pred = self.classify_SignAttribute(frame_num,img_crop,preloaded_params_attributes)
                    attr_name = self.get_attribute_name(sign_attribute_pred)
                    with open(self.out_path+'/attribute_data.json','r') as fp:
                        attribute_data = json.load(fp)
                    with open(self.out_path+'/attribute_data.json','w') as fp:
                        object_detected_id = int(obj_id)
                        if object_detected_id not in attribute_data:
                            attribute_data[int(obj_id)]=[[int(x1), int(y1),int(x2),int(y2),frame_num,attr_name]]
                        else:
                            attribute_data[int(obj_id)].append([int(x1), int(y1),int(x2),int(y2),frame_num,attr_name])
                        json.dump(attribute_data,fp)
                    cv2.rectangle(self.img, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 4)
                    cv2.putText(self.img, attr_name, (int(x2), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (178,34,34), 3)
                except Exception as e:
                    print(e)

    def classify_SignAttribute(self,frame_num,img_crop,preloaded_params):
        preloaded_params['out_path']=self.out_path
        return classify_attributes(frame_num,img_crop,preloaded_params)
    
    def detect_Streetlight(self,frame_num,preloaded_params):
        preloaded_params['out_path']=self.out_path
        preloaded_params['street_light_frames']=self.street_light_frames
        streetlight_detector(frame_num,self.img,preloaded_params)

    def detect_LaneMarking(self,frame_num,preloaded_params):
        preloaded_params['out_path']=self.out_path
        preloaded_params['lane_mark_frames']=self.lane_mark_frames
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
    num_classes = 1

    confidence = 0.4
    nms_thesh = 0.2


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
        
def preload_TrafficSignsAttributes():
    vgg_s = models.vgg11(pretrained=True)
    classifier = torch.nn.Sequential(torch.nn.Linear(512, 256, bias = True)
                                        , torch.nn.ReLU()
                                        , torch.nn.Dropout(p=0.5)
                                        , torch.nn.Linear(in_features=256, out_features=128, bias=True)
                                        , torch.nn.ReLU()
                                        , torch.nn.Dropout(p=0.5)
                                        , torch.nn.Linear(in_features=128, out_features=64, bias=True)
                                        , torch.nn.ReLU()
                                        , torch.nn.Dropout(p=0.5)
                                        , torch.nn.Linear(in_features=64, out_features=3, bias=True))

    trnscm = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.Resize((48,48))
                                            #,rgb2gray
                                            #,torchvision.transforms.Grayscale(num_output_channels=1)
                                            , torchvision.transforms.ToTensor() 
                                            ,torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406]
                                                                            , std=[0.229, 0.224, 0.225])])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = vgg_s.features
    classifier = classifier
    net.load_state_dict(torch.load('../weights/net_FE_sd.pt'))
    net = net.to(device)
    classifier.load_state_dict(torch.load('../weights/clf_loaded_sd.pt'))
    classifier = classifier.to(device)
    return {
            'classifier':classifier,
            'net':net,
            'trnscm':trnscm,
            'device':device
    }


if __name__ == "__main__":
    folder_path = "/Neutron6/ranjith.reddy/traffic_signs/pytorch-yolo-v3/Captures/2019-07-27-09-27-21"
    _id = folder_path.split('/')[-1]
    videoPath=os.path.join(folder_path,"Video/capture.mp4")
    out_path = "/Neutron6/ranjith.reddy/Road-Infrastructure/experiments"
    out_path = os.path.join(out_path,_id)
    try:
        os.mkdir(out_path)
    except Exception as e:
        print(e)
    print("the folder has been created===> ",out_path)
    with open(out_path+'/attribute_data.json','w') as f1:
        json.dump({},f1)
    with open(out_path+"/traffic_sign.json","w") as f2:
        json.dump({},f2)
    with open(out_path+"/street_light.json","w") as f3:
        json.dump({},f3)
    with open(out_path+"/coverage_score.json","w") as f4:
        json.dump({},f4)
    
    weights_path="/Neutron6/ranjith.reddy/Road-Infrastructure/weights/vgg_SCNN_DULR_w9.pth"
    preloaded_params_lanes = preload_lanemarking(weights_path)
    cfgfile_trafficsigns = '/Neutron6/ranjith.reddy/traffic_signs/tad_yolov3_5.cfg'
    weightsfile_yolotraffic = '/Neutron6/ranjith.reddy/traffic_signs/tad_yolov3_5_6000.weights'
    preloaded_params_signs=preload_trafficsigns(cfgfile_trafficsigns,weightsfile_yolotraffic)
    preloaded_params_attributes = preload_TrafficSignsAttributes()
    cfgfile_streetlights = '/Neutron6/ranjith.reddy/Road-Infrastructure/weights/streetlights.cfg'
    weightsfile_yolostreetlight = '/Neutron6/ranjith.reddy/Road-Infrastructure/weights/streetlights_best.weights'
    preloaded_params_lights=preload_streetlights(cfgfile_streetlights,weightsfile_yolostreetlight)
    vid = cv2.VideoCapture(videoPath)
    fps = vid.get(cv2.CAP_PROP_FPS)
    print("FPS:{}".format(fps))
    frame_num=0
    image=np.zeros((10,10,3)) # dummy initilization
    detector = Detector(image,out_path)
    while vid.isOpened():
        frame_num+=1
        print("frame num*** ",frame_num)
        ret, image = vid.read()
        if image is None:
            continue
        detector.img = image
        detector.detect_TrafficSign(frame_num,preloaded_params_signs,preloaded_params_attributes)
        detector.detect_Streetlight(frame_num,preloaded_params_lights)
        detector.detect_LaneMarking(frame_num,preloaded_params_lanes)

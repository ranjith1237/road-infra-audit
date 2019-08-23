import torch
import torchvision
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import shutil
import numpy as np
import time
from torchvision import models

def rgb2gray(image):
    return image.convert('L')

def classify_attributes(frame_num,img,preloaded_params):
    print(frame_num)
    trnscm = preloaded_params['trnscm']
    net = preloaded_params['net']
    classifier = preloaded_params['classifier']
    device = preloaded_params['device']
    img = trnscm(img)

    with torch.no_grad():
        net.eval()
        classifier.eval()
        features= img.to(device)
        features=features.unsqueeze(0)
        inputs = net(features)
        inputs=inputs.reshape(-1, 512)
        outputs = classifier(inputs)
        #outputs = net(features)
        _, predicted = torch.max(outputs.data, 1)
    print("predicted attribute ===>",predicted)
    return predicted

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

if __name__ == '__main__':
    preloaded_params_attributes = preload_TrafficSignsAttributes()
    classify_attributes(img,preloaded_params_attributes)
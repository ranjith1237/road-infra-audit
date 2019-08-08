import cv2
import torch
import json
from model import SCNN
from utils.prob2lines import getLane
from utils.transforms import *
from collections import defaultdict

def segment_lanes(count,img,preloaded_params):
    net=preloaded_params['net']
    transform=preloaded_params['transform']
    device=preloaded_params['device']
    def a():
        return []
    
    d = defaultdict(a)
    x = transform(img)[0]
    x.unsqueeze_(0)
    x=x.to(device)
    seg_pred, exist_pred = net(x)[:2]
    seg_pred = seg_pred.detach().cpu().numpy()
    exist_pred = exist_pred.detach().cpu().numpy()
    seg_pred = seg_pred[0]
    img = cv2.resize(img, (800, 288))
    lane_img = np.zeros_like(img)
    color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8')
    coord_mask = np.argmax(seg_pred, axis=0)
    for i in range(0, 4):
        if exist_pred[0, i] > 0.5:
            lane_img[coord_mask == (i + 1)] = color[i]
    img = cv2.addWeighted(src1=lane_img, alpha=0.8, src2=img, beta=1., gamma=0.)
    grayImage = cv2.cvtColor(lane_img, cv2.COLOR_BGR2GRAY)
    coverage_score = np.sum(grayImage!=0)/(np.sum(grayImage!=0)+np.sum(grayImage==0))
    print("coverage score --> ",coverage_score)
    d[count].append(coverage_score)
    #cv2.imwrite("result/img{}.jpg".format(count), img)
    print("Frame Number {}".format(count))
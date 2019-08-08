import h5py
f = h5py.File('/Neutron6/ranjith.reddy/traffic_signs/model_22_classes_batch8.h5', 'r')
print(f.attrs.get('keras_version'))
import keras
import warnings
warnings.filterwarnings("ignore")
keras.__version__
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
from keras.models import load_model
classification_model = load_model('/Neutron6/ranjith.reddy/traffic_signs/model_22_classes_batch8.h5')
import numpy as np
import json
from skimage import io, color, exposure, transform
import pdb

import torch 
import torch.nn as nn
from torch.autograd import Variable
import cv2
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import random
import pickle as pkl
import argparse
from sort import *
from bbox import get_abs_coord

def preprocess_img(img):
    # Histogram normalization in y
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)
    IMG_SIZE = 48
    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]
    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
    # roll color axis to axis 0
    img = np.rollaxis(img,-1)
    return img
def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim
def write(x, img, cls_id,classes_gtsrb):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    if cls == 26:
        return img, c1,c2
    else:
        label = classes_gtsrb[int(cls_id)]
        colors = (255,0,0)
        #cv2.rectangle(img, c1, c2,color, 10)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        #c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        #cv2.rectangle(img, c1, c2,color, -1)
        #cv2.putText(img, label, (c1[0], c1[1] + 30), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 1);
        return img, c1, c2, cls_id
def detect_sign(file_name,confidence,inp_dim,CUDA,model,num_classes,nms_thesh,classes_gtsrb):
    frame_url = file_name
    frame = cv2.imread(frame_url)
    try:
        b,g,r = cv2.split(frame)       # get b,g,r
        frame_rgb = cv2.merge([r,g,b])     # switch it to rgb
    except:
        return None,"fsdaf"
    img, orig_im, dim = prep_image(frame, inp_dim)
    sign = True
    im_dim = torch.FloatTensor(dim).repeat(1,2)                        

    if CUDA:
        im_dim = im_dim.cuda()
        img = img.cuda()
    with torch.no_grad():
        output = model(Variable(img), CUDA)
    output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
    
    if type(output) == int:
        print('no prediction observed')
    else: 
        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)

        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2

        output[:,1:5] /= scaling_factor
        gtsrb_labels = np.zeros(output.shape[0])
        _signs_ = []
        _outputs_ = []
        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])

            if output[i][-1].round() == 0:
                y1,y2,x1,x2 =output[i][1].int(),output[i][3].int(),output[i][2].int(),output[i][4].int()
                img = frame_rgb[x1:x2,y1:y2]
                if img.shape[0]==0 and img.shape[1]==0 :
                	return None, frame_rgb
                out_vector = output[i][1:].cpu().numpy()
                try:
                    processed_img = np.array(preprocess_img(img))
                    processed_img_uint = np.transpose((processed_img*255).astype(np.uint8),(1,2,0))
                    processed_img_batch = np.expand_dims(processed_img,axis=0)
                    gtsrb_labels[i] = 0 #classification_model.predict_classes(processed_img_batch)
                    output[i][-1] = gtsrb_labels[i]
                    out_vector[-1] = gtsrb_labels[i]
                    frame_rgb, c1, c2, cls = write(output[i], frame_rgb, gtsrb_labels[i],classes_gtsrb)
                    out_vector[0] = c1[0]
                    out_vector[1] = c1[1]
                    out_vector[2] = c2[0]
                    out_vector[3] = c2[1]
                    if (int(c1[0]) == 0 and int(c1[1]) == 0) or (int(c2[1]) == 0 and int(c2[1]) == 0):
                        return None, frame_rgb
                except Exception as e:
                    print(e)
                _outputs_.append(out_vector)
                print(_outputs_)
            else:
                output[i][-1] = output[i][-1] + 21
        if _outputs_ == []:
            return None, frame_rgb
        return np.array(_outputs_), frame_rgb



def traffic_detector(frame_num,img,preloaded_params):
    num_classes = preloaded_params['num_classes']
    confidence = preloaded_params['confidence']
    nms_thesh = preloaded_params['nms_thesh']
    CUDA = preloaded_params['CUDA']
    model = preloaded_params['model']
    model.net_info["height"] = preloaded_params['reso']
    inp_dim = preloaded_params['inp_dim']
    classes_gtsrb = preloaded_params['classes_gtsrb']
    videopath="/Neutron6/ranjith.reddy/traffic_signs/pytorch-yolo-v3/2019-07-03-13-25-01/Video/capture.mp4"
    classes = ["traffic sign", "traffic sign"]
    from collections import defaultdict
    def a():
        return []
    d = defaultdict(a)

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    # initialize Sort object and video capture

    vid = cv2.VideoCapture(videopath)
    print("FPS {}".format(vid.get(cv2.CAP_PROP_FPS)))
    mot_tracker = Sort() 
    i = frame_num
    frame = img
    cv2.imwrite('test.jpg', frame)
    try:
        detections, img = detect_sign('test.jpg',confidence,inp_dim,CUDA,model,num_classes,nms_thesh,classes_gtsrb)
        if detections is not None:
            tracked_objects = mot_tracker.update(detections)

            unique_labels = np.unique(detections[:, -1])
            n_cls_preds = len(unique_labels)
            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                try:
                    color_ = colors[int(obj_id) % len(colors)]
                    color_ = [ij * 255 for ij in color_]
                    cls = classes_gtsrb[int(cls_pred)]
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color_, 4)
                    cv2.putText(img, "Traffic Sign", (int(x2), int(y2) + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (178,34,34), 3)
                    d[i].append(int(obj_id))
                    with open("gps_frames.json","w") as f:
                        json.dump(d,f)
                except Exception as e:
                    print(e)
        fig=plt.figure(figsize=(12, 8))
        plt.title("Video Stream {}".format(i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('test_new_route/img{}.jpg'.format(i), img)
    except Exception as e:
        print(e)



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
            'inp_dim':inp_dim
        }
        

if __name__ == "__main__":
    cfgfile = '/Neutron6/ranjith.reddy/traffic_signs/tad_yolov3_5.cfg'
    weightsfile = '/Neutron6/ranjith.reddy/traffic_signs/tad_yolov3_5_6000.weights'
    preload_trafficsigns(cfgfile,weightsfile)
    print("@@@@@@@@@")
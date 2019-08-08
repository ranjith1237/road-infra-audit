import torch
import torchvision
#import torch.Ttr
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import shutil
import numpy as np
import time
#import pdb
from torchvision import models
#pdb.set_trace()
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


def rgb2gray(image):
    return image.convert('L')
trnscm = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.Resize((48,48))
                                         #,rgb2gray
                                         #,torchvision.transforms.Grayscale(num_output_channels=1)
                                         , torchvision.transforms.ToTensor() 
                                         ,torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406]
                                                                           , std=[0.229, 0.224, 0.225])])
#batch_size =64

#train_data = torchvision.datasets.ImageFolder('./Augmented_data/Train/', transform=trnscm)
#validation_data = torchvision.datasets.ImageFolder('./Augmented_data/Validation/', transform=trnscm)
#test_data = torchvision.datasets.ImageFolder('./Augmented_data/Test/', transform=trnscm)

#train_data_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True)
#test_data_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=True)
#validation_data_loader = torch.utils.data.DataLoader(validation_data,batch_size=batch_size,shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#num_epochs = 1000
net = vgg_s.features
classifier = classifier
net.load_state_dict(torch.load('../../weights/net_FE_sd.pt'))
net = net.to(device)
classifier.load_state_dict(torch.load('../../weights/clf_loaded_sd.pt'))
classifier = classifier.to(device)
#net_loaded = torch.load('net_FE.pt', map_location='cpu')
#clf_loaded = torch.load('classifier.pt', map_location='cpu')
#torch.save(net_loaded.state_dict(), 'net_FE_sd.pt')
#torch.save(clf_loaded.state_dict(), 'clf_loaded_sd.pt')
img=cv2.imread('test.jpg')
cv2.imwrite('saved.jpg', img)
img = trnscm(img)
#criterion= torch.nn.CrossEntropyLoss()
#params=list(net.parameters())+list(classifier.parameters())
#optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=1e-6)
#test_accuracies=[]
#train_accuracies=[]

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
print(predicted)
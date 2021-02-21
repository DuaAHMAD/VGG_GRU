#!/usr/bin/python
# encoding: utf-8

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
from PIL import Image
from utils import *
from torchvision import transforms



class listTESTDataset(Dataset):

#     def __init__(self, root, length = None, shuffle=True,  train=False, dataset = None ,debug = False):
    def __init__(self, root, length = None, shuffle=False, dataset = None ,debug = False):
        with open(root, 'r') as file:
            self.lines = file.readlines()

        if shuffle:
            random.shuffle(self.lines)

        self.nSamples  = len(self.lines[:60]) if debug  else len(self.lines)

#         self.train = train
        self.length = length
        self.dataset = dataset


    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()

        img, label_index, participant = self.load_test_data_label(imgpath)
        label_index = torch.LongTensor([label_index])

        return (img ,label_index, participant)
   
    
    def load_test_data_label(self, imgpath, filter_size=16, stride=16):

        data_transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

#         classes = self.get_classes()

        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        img_path = imgpath.split(" ")[0]
        par= img_path.split("/")
        participant = par[-1]
        label_path = imgpath.split(" ")[1]
        label_index = int(label_path)
#         print("Participant:", participant)
#         print("image label", label_path)
#         label_index = int(classes.index(label_path))

        video_length = len(os.listdir(img_path))
        imgs = os.listdir(img_path)
        imgs.sort()

        output = []
        if video_length >= filter_size:
            for i in range(0, video_length, stride):
#                 print(i)
                if i + filter_size <= video_length:
                    inputs = []
                    frames_subset = imgs[i: i+filter_size]
#                     print(frames_subset)
                    for frame in frames_subset:
#                         print(frames_subset.index(frame))
                        frame = os.path.join(img_path, frame)
                        img = Image.open(frame)
                        inputs.append(data_transforms(img).unsqueeze(0))
                    output_subset = torch.cat(inputs).unsqueeze(0)
                    output.append(output_subset)

        else:
            inputs = []
            for k in range(filter_size):
                if k+1 <= video_length:
                    frame = os.path.join(img_path, imgs[k])
                    img = Image.open(frame)
                    inputs.append(data_transforms(img).unsqueeze(0))
                else:
                    frame = os.path.join(img_path, imgs[video_length-1])
                    img = Image.open(frame)
                    inputs.append(data_transforms(img).unsqueeze(0))
            output_subset = torch.cat(inputs).unsqueeze(0)
            output.append(output_subset)

        output = torch.cat(output)

        return output, label_index, participant
    

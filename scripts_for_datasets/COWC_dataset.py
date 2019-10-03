from __future__ import print_function, division
import os
import torch
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose,
    BboxParams, RandomCrop, Normalize, Resize, VerticalFlip
)

from albumentations.pytorch import ToTensor
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class COWCDataset(Dataset):
  def __init__(self, root, image_height, image_width, transform = None):
    self.root = root
    #take all under same folder for train and test split.
    self.transform = transform
    self.image_height = image_height
    self.image_width = image_width
    #sort all images for indexing, filter out check.jpgs
    self.imgs = list(sorted(set(glob.glob(root_dir+"*.jpg")) - set(glob.glob(root_dir+"*check.jpg"))))
    self.annotation = list(sorted(glob.glob(root_dir+"*.txt")))

  def __getitem__(self, idx):
    #get the paths
    img_path = os.path.join(self.root, self.imgs[idx])
    annotation_path = os.path.join(self.root, self.annotation[idx])
    img = cv2.imread(img_path,1) #read color image height*width*channel=3
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #get the bounding box
    boxes = list()
    label_car_type = list()
    with open(annotation_path) as f:
      for line in f:
        values = (line.split())
        obj_class = int(values[0])
        #get coordinates withing height width range
        x = float(values[1])*self.image_width
        y = float(values[2])*self.image_height
        width = float(values[3])*self.image_width
        height = float(values[4])*self.image_height
        #creating bounding boxes that would not touch the image edges
        x_min = 1 if x - width/2 <= 0 else int(x - width/2)
        x_max = 255 if x + width/2 >= 256 else int(x + width/2)
        y_min = 1 if y - height/2 <= 0 else int(y - height/2)
        y_max = 255 if y + height/2 >= 256 else int(y + height/2)

        boxes.append([x_min, y_min, x_max, y_max])
        label_car_type.append(obj_class)
        labels = np.ones(len(boxes)) # all are cars
        #create dictionary to access the values
        target = {}
        target['image'] = img
        target['bboxes'] = boxes
        target['labels'] = labels
        target['label_car_type'] = label_car_type
        target['idx'] = idx

    if self.transform is None:
      #convert to tensor
      target = self.convert_to_tensor(**target)
      return target

    #transform
    else:
      target = {}
      target['image'] = img
      target['bboxes'] = boxes
      target['labels'] = labels
      target['label_car_type'] = label_car_type
      target['idx'] = idx

      transformed = self.transform(**target)
      #print(transformed['image'], transformed['bboxes'], transformed['labels'], transformed['idx'])
      target = self.convert_to_tensor(**transformed)

      return target

  def __len__(self):
    return len(self.imgs)

  def convert_to_tensor(self, **target):
      #convert to tensor
      target['image'] = torch.from_numpy(target['image'].transpose((2, 0, 1)))
      target['bboxes'] = torch.as_tensor(target['bboxes'], dtype=torch.int64)
      target['labels'] = torch.ones(len(target['bboxes']), dtype=torch.int64)
      target['label_car_type'] = torch.as_tensor(target['label_car_type'], dtype=torch.int64)
      target['image_id'] = torch.tensor([target['idx']])

      return target

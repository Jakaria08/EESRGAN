from __future__ import print_function, division
import os
import torch
import numpy as np
import glob
import cv2
from torch.utils.data import Dataset, DataLoader

class COWCFRCNNDataset(Dataset):
  def __init__(self, root=None, image_height=256, image_width=256, transforms = None):
    self.root = root
    #take all under same folder for train and test split.
    self.transforms = transforms
    self.image_height = image_height
    self.image_width = image_width
    #sort all images for indexing, filter out check.jpgs
    self.imgs = list(sorted(set(glob.glob(self.root+"*.jpg") +
                    glob.glob(self.root+"*.png")) -
                    set(glob.glob(self.root+"*check.jpg"))))
    self.annotation = list(sorted(glob.glob(self.root+"*.txt")))

  def __getitem__(self, idx):
    #get the paths
    img_path = os.path.join(self.root, self.imgs[idx])
    annotation_path = os.path.join(self.root, self.annotation[idx])
    img = cv2.imread(img_path,1) #read color image height*width*channel=3
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #get the bounding box
    boxes = list()
    with open(annotation_path) as f:
        for line in f:
            values = (line.split())
            if "\ufeff" in values[0]:
              values[0] = values[0][-1]
            #get coordinates withing height width range
            x = float(values[1])*self.image_width
            y = float(values[2])*self.image_height
            width = float(values[3])*self.image_width
            height = float(values[4])*self.image_height
            #creating bounding boxes that would not touch the image edges
            x_min = 1 if x - width/2 <= 0 else int(x - width/2)
            x_max = self.image_width-1 if x + width/2 >= self.image_width-1 else int(x + width/2)
            y_min = 1 if y - height/2 <= 0 else int(y - height/2)
            y_max = self.image_height-1 if y + height/2 >= self.image_height-1 else int(y + height/2)

            x_min = int(x_min)
            x_max = int(x_max)
            y_min = int(y_min)
            y_max = int(y_max)

            boxes.append([x_min, y_min, x_max, y_max])
            #print(boxes)

    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    # there is only one class
    labels = torch.ones((len(boxes),), dtype=torch.int64)
    image_id = torch.tensor([idx])
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    # suppose all instances are not crowd
    iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
    #create dictionary to access the values
    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    target["image_id"] = image_id
    target["area"] = area
    target["iscrowd"] = iscrowd

    if self.transforms is not None:
      img, target = self.transforms(img, target)

    #return img, target
    return img, target, annotation_path

  def __len__(self):
    return len(self.imgs)

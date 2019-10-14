import json
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def visualize_bbox(img, bbox, class_id, class_idx_to_name, color=BOX_COLOR, thickness=2):
    #receives data from visualize() method
    print(np.shape(bbox))
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    class_name = class_idx_to_name[class_id]
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,TEXT_COLOR, lineType=cv2.LINE_AA)
    return img

def visualize(annotations, category_id_to_name):
    '''
    receives single image and its properties

    To use this method make sure use data loader with collate function
    and use batch size as 1

    collate function in data loader will create this format as a list:
    This is the desired format: 'bboxes': [tensor([[255,   0, 256,   1]])],
    'labels': [tensor([1])], 'label_car_type': [tensor([0])], 'idx': [1670]
    '''
    img = annotations['image'].squeeze().numpy().transpose(1,2,0).copy()
    annotations['labels'] = annotations['labels'][0].numpy()
    length = len(annotations['labels'])
    if length == 1:
        img = visualize_bbox(img, annotations['bboxes'][0].squeeze().numpy(), int(annotations['labels'].squeeze()), category_id_to_name)
    else:
        for idx, bbox in enumerate(annotations['bboxes'][0].numpy()):
            img = visualize_bbox(img, bbox, annotations['labels'][idx], category_id_to_name)
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    print(img.shape)
    x = random.randint(0,1000)
    cv2.imwrite(str(x)+'test.png', img)

def calculate_mean_std(data_loader):
    '''
    receivdes a data loader object to calcute std, mean for 3 channel image dataset
    From: https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/2
    check this method using collate function in data DataLoader
    test it before using
    '''
    mean = 0
    std = 0
    nb_samples = 0
    for batch_idx, dataset_dict in enumerate(data_loader):
        batch_samples = dataset_dict['image'].size(0)# check this line for correctness
        imgs = dataset_dict['image'].double().view(batch_samples, dataset_dict['image'].size(1), -1)
        #print(imgs.size())
        mean += imgs.mean(2).sum(0)
        std += imgs.std(2).sum(0)
        nb_samples += batch_samples
    print(nb_samples)
    mean /= nb_samples
    std /= nb_samples
    mean /= 255
    std /= 255
    print(mean,std)
    return mean, std

def collate_fn(batch):
    '''
    Image have a different number of objects, we need a collate function
    (to be passed to the DataLoader).
    '''
    target = {}
    target['object'] = list()
    target['image'] = list()
    target['bboxes'] = list()
    target['labels'] = list()
    target['label_car_type'] = list()
    target['idx'] = list()

    for b in batch:
        target['object'].append(b['object'])
        target['image'].append(b['image'])
        target['bboxes'].append(b['bboxes'])
        target['labels'].append(b['labels'])
        target['label_car_type'].append(b['label_car_type'])
        target['idx'].append(b['idx'])

    target['image'] = torch.stack(target['image'], dim=0)

    return target

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

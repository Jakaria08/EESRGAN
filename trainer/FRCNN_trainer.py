'''
quick and dirty test, need to change later
'''
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, utils
from detection.engine import train_one_epoch, evaluate

class COWCFRCNNTrainer:
    """
    Trainer class
    """
    def __init__(self, config, data_loader, valid_data_loader=None):
        self.config = config
        self.data_loader = data_loader

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        n_gpu = torch.cuda.device_count()
        self.device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        self.train_size = int(math.ceil(self.data_loader.length / int(config['data_loader']['args']['batch_size'])))
        self.total_iters = int(config['train']['niter'])
        self.total_epochs = int(math.ceil(self.total_iters / self.train_size))
        print(self.total_epochs)
        self.model = ESRGAN_EESN.ESRGAN_EESN_Model(config,self.device)


    def train(self):

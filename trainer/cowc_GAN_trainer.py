import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, visualize_bbox, visualize
'''
python train.py -c config_GAN.json
'''

class COWCGANTrainer:
    """
    Trainer class
    """
    def __init__(self, config, data_loader, valid_data_loader=None):
        self.config = config
        self.data_loader = data_loader

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None

    def train(self):
        '''
        Training logic for an epoch

        for visualization use the following code (use batch size = 1):
        '''
        category_id_to_name = {1: 'car'}
        for dataset_dict in self.data_loader:
            print(batch_idx)
            print(dataset_dict['idx'])
            if dataset_dict['idx'].squeeze() == 10:
                print(dataset_dict)
                visualize(dataset_dict, category_id_to_name) #--> see this method in util

        #image size: torch.Size([10, 3, 256, 256]) if batch_size = 10

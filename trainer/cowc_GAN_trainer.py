import numpy as np
import torch
import math
import model.ESRGANModel as ESRGAN
from scripts_for_datasets import COWCDataset, COWCGANDataset
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
        n_gpu = torch.cuda.device_count()
        self.device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        self.train_size = int(math.ceil(self.data_loader.length / int(config['data_loader']['args']['batch_size'])))
        self.total_iters = int(config['train']['niter'])
        self.total_epochs = int(math.ceil(self.total_iters / self.train_size))
        print(self.total_epochs)
        self.model = ESRGAN.ESRGANModel(config,self.device)


    def train(self):
        '''
        Training logic for an epoch
        for visualization use the following code (use batch size = 1):

        category_id_to_name = {1: 'car'}
        for batch_idx, dataset_dict in enumerate(self.data_loader):
            if dataset_dict['idx'][0] == 10:
                print(dataset_dict)
                visualize(dataset_dict, category_id_to_name) #--> see this method in util

        #image size: torch.Size([10, 3, 256, 256]) if batch_size = 10
        '''
        pass

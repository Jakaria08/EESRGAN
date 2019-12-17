import logging
import numpy as np
import torch
import math
import os
import model.ESRGANModel as ESRGAN
import model.ESRGAN_EESN_FRCNN_Model as ESRGAN_EESN
from scripts_for_datasets import COWCDataset, COWCGANDataset
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, visualize_bbox, visualize, calculate_psnr, save_img, tensor2img, mkdir

logger = logging.getLogger('base')
'''
python train.py -c config_GAN.json
modified from ESRGAN repo
'''

class COWCGANFrcnnTrainer:
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
        self.model = ESRGAN_EESN.ESRGAN_EESN_FRCNN_Model(config,self.device)

    def test(self):
        for _, test_data in enumerate(self.data_loader):
            #print(val_data)
            image, target = test_data
            img_name = os.path.splitext(os.path.basename(image['LQ_path'][0]))[0]
            img_dir = "/home/jakaria/Super_Resolution/Filter_Enhance_Detect/saved/"

            self.model.feed_data(image, target)
            self.model.test()

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
        logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    self.data_loader.length, self.train_size))
        logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    self.total_epochs, self.total_iters))
        # tensorboard logger
        if self.config['use_tb_logger'] and 'debug' not in self.config['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='saved/tb_logger/' + self.config['name'])

        current_step = 16001
        start_epoch = 43

        #### training
        logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
        for epoch in range(start_epoch, self.total_epochs + 1):
            for _, (image, targets) in enumerate(self.data_loader):
                current_step += 1
                if current_step > self.total_iters:
                    break
                #### update learning rate
                self.model.update_learning_rate(current_step, warmup_iter=self.config['train']['warmup_iter'])

                #### training
                self.model.feed_data(image, targets)
                self.model.optimize_parameters(current_step)

                #### log
                if current_step % self.config['logger']['print_freq'] == 0:
                    logs = self.model.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                        epoch, current_step, self.model.get_current_learning_rate())
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        # tensorboard logger
                        if self.config['use_tb_logger'] and 'debug' not in self.config['name']:
                            tb_logger.add_scalar(k, v, current_step)

                    logger.info(message)

                # validation
                if current_step % self.config['train']['val_freq'] == 0:
                    self.model.test(self.valid_data_loader)

                #### save models and training states
                if current_step % self.config['logger']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    self.model.save(current_step)
                    self.model.save_training_state(epoch, current_step)


        logger.info('Saving the final model.')
        self.model.save('latest')
        logger.info('End of training.')

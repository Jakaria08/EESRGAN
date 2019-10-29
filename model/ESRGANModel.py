from collections import OrderedDict
import torch
import torch.nn as nn
import model.model as networks
import model.lr_scheduler as lr_scheduler
from model.loss import GANLoss

class ESRGANModel:
    def __init__(self, config):
        print(config['train'])

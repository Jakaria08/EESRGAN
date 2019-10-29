from collections import OrderedDict
import torch
import torch.nn as nn
import model.model as networks
import model.lr_scheduler as lr_scheduler
from model.loss import GANLoss

class ESRGANModel:
    def __init__(self, config, device):
        self.configG = config['network_G']
        self.configD = config['network_D']
        self.device = device
        self.netG = model.RRDBNet(in_nc=self.configG['in_nc'], out_nc=self.configG['out_nc'],
                                    nf=self.configG['nf'], nb=self.configG['nb'])
        self.netG = self.netG.to(self.device)
        self.netD = model.Discriminator_VGG_128(in_nc=self.configD['in_nc'], nf=self.configD['nf'])
        self.netD = self.netD.to(self.device)

        self.netG.train()
        self.netD.train()

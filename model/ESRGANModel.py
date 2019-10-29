from collections import OrderedDict
import torch
import torch.nn as nn
import model.model as model
import model.lr_scheduler as lr_scheduler
from model.loss import GANLoss

class ESRGANModel:
    def __init__(self, config, device):
        self.configG = config['network_G']
        self.configD = config['network_D']
        self.configT = config['train']
        self.device = device
        self.netG = model.RRDBNet(in_nc=self.configG['in_nc'], out_nc=self.configG['out_nc'],
                                    nf=self.configG['nf'], nb=self.configG['nb'])
        self.netG = self.netG.to(self.device)
        self.netD = model.Discriminator_VGG_128(in_nc=self.configD['in_nc'], nf=self.configD['nf'])
        self.netD = self.netD.to(self.device)

        self.netG.train()
        self.netD.train()
        print(float(self.configT['pixel_weight']))
        # G pixel loss
        if float(self.configT['pixel_weight']) > 0.0:
            l_pix_type = self.configT['pixel_criterion']
            if l_pix_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif l_pix_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
            self.l_pix_w = train_opt['pixel_weight']
        else:
            self.cri_pix = None

        # G feature loss
        print(train_opt['feature_weight']+1)
        if train_opt['feature_weight'] > 0:
            l_fea_type = train_opt['feature_criterion']
            if l_fea_type == 'l1':
                self.cri_fea = nn.L1Loss().to(self.device)
            elif l_fea_type == 'l2':
                self.cri_fea = nn.MSELoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
            self.l_fea_w = train_opt['feature_weight']
        else:
            self.cri_fea = None
        if self.cri_fea:  # load VGG perceptual loss
            self.netF = model.VGGFeatureExtractor(feature_layer=34,
                                          use_input_norm=True, device=self.device)
            self.netF = self.netF.to(self.device)
            self.netF.eval()

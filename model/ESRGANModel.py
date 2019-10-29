from collections import OrderedDict
import torch
import torch.nn as nn
import model.model as model
import model.lr_scheduler as lr_scheduler
from model.loss import GANLoss

class ESRGANModel:
    def __init__(self, config, device):
        self.optimizers = []
        self.configG = config['network_G']
        self.configD = config['network_D']
        self.configT = config['train']
        self.configO = config['optimizer']['args']
        self.configS = config['lr_scheduler']
        self.device = device
        self.netG = model.RRDBNet(in_nc=self.configG['in_nc'], out_nc=self.configG['out_nc'],
                                    nf=self.configG['nf'], nb=self.configG['nb'])
        self.netG = self.netG.to(self.device)
        self.netD = model.Discriminator_VGG_128(in_nc=self.configD['in_nc'], nf=self.configD['nf'])
        self.netD = self.netD.to(self.device)

        self.netG.train()
        self.netD.train()
        #print(self.configT['pixel_weight'])
        # G pixel loss
        if self.configT['pixel_weight'] > 0.0:
            l_pix_type = self.configT['pixel_criterion']
            if l_pix_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif l_pix_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
            self.l_pix_w = self.configT['pixel_weight']
        else:
            self.cri_pix = None

        # G feature loss
        #print(self.configT['feature_weight']+1)
        if self.configT['feature_weight'] > 0:
            l_fea_type = self.configT['feature_criterion']
            if l_fea_type == 'l1':
                self.cri_fea = nn.L1Loss().to(self.device)
            elif l_fea_type == 'l2':
                self.cri_fea = nn.MSELoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
            self.l_fea_w = self.configT['feature_weight']
        else:
            self.cri_fea = None
        if self.cri_fea:  # load VGG perceptual loss
            self.netF = model.VGGFeatureExtractor(feature_layer=34,
                                          use_input_norm=True, device=self.device)
            self.netF = self.netF.to(self.device)
            self.netF.eval()

        # GD gan loss
        self.cri_gan = GANLoss(self.configT['gan_type'], 1.0, 0.0).to(self.device)
        self.l_gan_w = self.configT['gan_weight']
        # D_update_ratio and D_init_iters
        self.D_update_ratio = self.configT['D_update_ratio'] if self.configT['D_update_ratio'] else 1
        self.D_init_iters = self.configT['D_init_iters'] if self.configT['D_init_iters'] else 0


        # optimizers
        # G
        wd_G = self.configO['weight_decay_G'] if self.configO['weight_decay_G'] else 0
        optim_params = []
        for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            if v.requires_grad:
                optim_params.append(v)

        self.optimizer_G = torch.optim.Adam(optim_params, lr=self.configO['lr_G'],
                                            weight_decay=wd_G,
                                            betas=(self.configO['beta1_G'], self.configO['beta2_G']))
        self.optimizers.append(self.optimizer_G)
        # D
        wd_D = self.configO['weight_decay_D'] if self.configO['weight_decay_D'] else 0
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.configO['lr_D'],
                                            weight_decay=wd_D,
                                            betas=(self.configO['beta1_D'], self.configO['beta2_D']))
        self.optimizers.append(self.optimizer_D)

        # schedulers
        if self.configS['type'] == 'MultiStepLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepLR_Restart(optimizer, self.configS['args']['lr_steps'],
                                                     restarts=self.configS['args']['restarts'],
                                                     weights=self.configS['args']['restart_weights'],
                                                     gamma=self.configS['args']['lr_gamma'],
                                                     clear_state=False))
        elif train_opt['type'] == 'CosineAnnealingLR_Restart':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingLR_Restart(
                        optimizer, self.configS['args']['T_period'], eta_min=self.configS['args']['eta_min'],
                        restarts=self.configS['args']['restarts'], weights=self.configS['args']['restart_weights']))
        else:
            raise NotImplementedError('MultiStepLR learning rate scheme is enough.')
        print(self.configS['args']['restarts'])
        self.log_dict = OrderedDict()

    #self.print_network()  # print network
    #self.load()  # load G and D if needed

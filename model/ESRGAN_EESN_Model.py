import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import model.model as model
import model.lr_scheduler as lr_scheduler
from model.loss import GANLoss
from .gan_base_model import BaseModel
from torch.nn.parallel import DataParallel

logger = logging.getLogger('base')
# Taken from ESRGAN BASICSR repository and modified
class ESRGANModel(BaseModel):
    def __init__(self, config, device):
        super(ESRGANModel, self).__init__(config, device)
        self.configG = config['network_G']
        self.configD = config['network_D']
        self.configT = config['train']
        self.configO = config['optimizer']['args']
        self.configS = config['lr_scheduler']
        self.device = device
        #Generator
        self.netG = model.RRDBNet(in_nc=self.configG['in_nc'], out_nc=self.configG['out_nc'],
                                    nf=self.configG['nf'], nb=self.configG['nb'])
        self.netG = self.netG.to(self.device)
        self.netG = DataParallel(self.netG)
        #descriminator
        self.netD = model.Discriminator_VGG_128(in_nc=self.configD['in_nc'], nf=self.configD['nf'])
        self.netD = self.netD.to(self.device)
        self.netD = DataParallel(self.netD)

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
            self.netF = DataParallel(self.netF)
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
        elif self.configS['type'] == 'CosineAnnealingLR_Restart':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingLR_Restart(
                        optimizer, self.configS['args']['T_period'], eta_min=self.configS['args']['eta_min'],
                        restarts=self.configS['args']['restarts'], weights=self.configS['args']['restart_weights']))
        else:
            raise NotImplementedError('MultiStepLR learning rate scheme is enough.')
        print(self.configS['args']['restarts'])
        self.log_dict = OrderedDict()

        self.print_network()  # print network
        self.load()  # load G and D if needed
    '''
    The main repo did not use collate_fn and image read has different flags
    and also used np.ascontiguousarray()
    Might change my code if problem happens
    '''
    def feed_data(self, data):
        self.var_L = data['image_lq'].to(self.device)
        self.var_H = data['image'].to(self.device)
        input_ref = data['ref'] if 'ref' in data else data['image']
        self.var_ref = input_ref.to(self.device)

    def optimize_parameters(self, step):
        #Generator
        for p in self.netD.parameters():
            p.requires_grad = False
        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)

        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix: #pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                l_g_total += l_g_pix
            if self.cri_fea: # feature loss
                real_fea = self.netF(self.var_H).detach() #don't want to backpropagate this, need proper explanation
                fake_fea = self.netF(self.fake_H) #In netF normalize=False, check it
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                l_g_total += l_g_fea

            pred_g_fake = self.netD(self.fake_H)
            if self.configT['gan_type'] == 'gan':
                l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
            elif self.configT['gan_type'] == 'ragan':
                pred_d_real = self.netD(self.var_ref).detach()
                l_g_gan = self.l_gan_w * (
                self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
            l_g_total += l_g_gan

            l_g_total.backward()
            self.optimizer_G.step()

        #descriminator
        for p in self.netD.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()
        l_d_total = 0
        pred_d_real = self.netD(self.var_ref)
        pred_d_fake = self.netD(self.fake_H.detach()) #to avoid BP to Generator
        if self.configT['gan_type'] == 'gan':
            l_d_real = self.cri_gan(pred_d_real, True)
            l_d_fake = self.cri_gan(pred_d_fake, False)
            l_d_total = l_d_real + l_d_fake
        elif self.configT['gan_type'] == 'ragan':
            l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
            l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
            l_d_total = (l_d_real + l_d_fake) / 2

        l_d_total.backward()
        self.optimizer_D.step()

        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
            if self.cri_fea:
                self.log_dict['l_g_fea'] = l_g_fea.item()
            self.log_dict['l_g_gan'] = l_g_gan.item()

        self.log_dict['l_d_real'] = l_d_real.item()
        self.log_dict['l_d_fake'] = l_d_fake.item()
        self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
        self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

        # Discriminator
        s, n = self.get_network_description(self.netD)
        if isinstance(self.netD, nn.DataParallel) or isinstance(self.netD,
                                                                DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                             self.netD.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netD.__class__.__name__)

        logger.info('Network D structure: {}, with parameters: {:,d}'.format(
            net_struc_str, n))
        logger.info(s)

        if self.cri_fea:  # F, Perceptual Network
            s, n = self.get_network_description(self.netF)
            if isinstance(self.netF, nn.DataParallel) or isinstance(
                    self.netF, DistributedDataParallel):
                net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                 self.netF.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netF.__class__.__name__)

            logger.info('Network F structure: {}, with parameters: {:,d}'.format(
                net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.config['path']['pretrain_model_G']
        if load_path_G:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.config['path']['strict_load'])
        load_path_D = self.config['path']['pretrain_model_D']
        if load_path_D:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, self.config['path']['strict_load'])

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)

import torch
import functools
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import utils as mutil
import kornia
import logging
from functools import partial
from dataclasses import dataclass
from collections import OrderedDict
from base import BaseModel

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

#Code for Tiny Net#########################
'''
Resnet architecture code from here:https://github.com/FrancescoSaverioZuppichini/ResNet
Need to move the classe for the resnet block and layers to a different file like 'blocks.py'
and also need to move the activation funtions and conv3x3 to utils
'''
'''
Image should be square in size, otherwise will fail in "GlobalAttentionSPP class"
Will fix it later##########
'''
'''
auto padding class
'''
class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size
#create func with default kernel_size and bias using partial
conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)

#dict of activation function
def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=False)], #sometimes you will see inplace gradient calculation operation error, just change it to false
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=False)],
        ['selu', nn.SELU(inplace=False)],
        ['none', nn.Identity()]
    ])[activation]
'''
Basic residual block, extension is in later parts of the Code
'''
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()
#block, then residual connection from the previous input
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels
'''
extension of the residual block
'''
class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None


    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels
'''
create conv-batchnorm stack
'''
def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))
'''
Resnet Basic block
'''
class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )
'''
create a ResNet layer with resnet basic blocks
'''
class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

class TinyNet(nn.Module):
    """
    Introduced in this paper: . Pang, C. Li, J. Shi, Z. Xu and H. Feng,
    "$\mathcal{R}^2$ -CNN: Fast Tiny Object Detection in Large-Scale Remote Sensing Images,"
    in IEEE Transactions on Geoscience and Remote Sensing, vol. 57, no. 8, pp. 5512-5524,
    Aug. 2019. doi: 10.1109/TGRS.2019.2899955

    TinyNet composed by increasing different layers with increasing features.
    Similar to ResNet
    """
    def __init__(self, in_channels=3, blocks_sizes=[12, 18, 36, 48, 72], depths=[2,2,3,2],
                 activation='relu', block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()

        self.blocks_sizes = blocks_sizes
        #creating the first layer of tiny net with two conv
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, blocks_sizes[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation_func(activation),
            nn.Conv2d(blocks_sizes[0], blocks_sizes[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(blocks_sizes[0]),
            activation_func(activation),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        #next layers are same as resnet
        self.in_out_block_sizes = list(zip(blocks_sizes[1:], blocks_sizes[2:]))
        self.blocks = nn.ModuleList([
            ResNetLayer(blocks_sizes[0], blocks_sizes[1], n=depths[0], activation=activation,
                        block=block,*args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion,
                          out_channels, n=n, activation=activation,
                          block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])]
        ])
        #last conv layer with relu
        self.lastconv = nn.Sequential(
        nn.UpsamplingBilinear2d(scale_factor=2),
        nn.Conv2d(in_channels=blocks_sizes[-1], out_channels=blocks_sizes[-2],
                  kernel_size=3, padding=1),
        nn.BatchNorm2d(blocks_sizes[-2]),
        activation_func(activation))


    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            third_last_layer = x
            x = block(x)
        x = self.lastconv(x)
        x += third_last_layer #skip connection for the last layer
        return x

class GlobalAttentionSPP(nn.Module):
  '''
  Spatial pyramid pooling for global attention Block
  '''
  def __init__(self):
    super().__init__()
    self.conv = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=1) #1*1 convolution
    self.SPP1x1 = nn.AdaptiveMaxPool2d(1)
    self.SPP2x2 = nn.AdaptiveMaxPool2d(2)
    self.SPP4x4 = nn.AdaptiveMaxPool2d(4)
    self.tinynet = TinyNet()
    '''
    calculation: https://discuss.pytorch.org/t/inferred-padding-size-for-nn-convtranspose2d/12141
    doing for image size 256, now it is 32, so upsample each pyramid to 32
    2xp1 = 2*input + 1, 3xp1 = 3*input + 1, p1 = input + 1, p2 = input + 2
    p3 = input + 3 and m1 = input - 1

    Too many parameters in these layers, might consider just upsampling to see the performance,
    if similar omit these layers. Actually omit these and using Bilinear interpolation
    But the image of paper named it deconv, is misleading, in the description it is interpolation.

    self.Deconv_2xp1 = nn.ConvTranspose2d(in_channels=48,out_channels=48,kernel_size=3,stride=2)
    self.Deconv_3xp1 = nn.ConvTranspose2d(in_channels=48,out_channels=48,kernel_size=3,stride=3)
    self.Deconv_p1 = nn.ConvTranspose2d(in_channels=48,out_channels=48,kernel_size=2)
    self.Deconv_p2 = nn.ConvTranspose2d(in_channels=48,out_channels=48,kernel_size=3)
    self.Deconv_p3 = nn.ConvTranspose2d(in_channels=48,out_channels=48,kernel_size=4)
    self.Deconv_m1 = nn.ConvTranspose2d(in_channels=48,out_channels=48,kernel_size=2,padding=1)

    x1 = self.Deconv_p1(self.Deconv_2xp1(self.Deconv_2xp1(self.Deconv_2xp1(
        self.Deconv_2xp1(self.SPP1x1(x))))))
    x2 = self.Deconv_m1(self.Deconv_3xp1(self.Deconv_2xp1(self.Deconv_2xp1(self.SPP2x2(x)))))
    x4 = self.Deconv_p2(self.Deconv_p3(self.Deconv_3xp1(self.Deconv_2xp1(self.SPP4x4(x)))))
    x = x + x1 + x2 + x4
    '''
  def forward(self,x):
    x = self.tinynet(x)
    Upscale_factor = x.size()[2] # image must be square in size
    x1 = nn.UpsamplingBilinear2d(Upscale_factor)
    x1 = x1(self.SPP1x1(x))
    x2 = nn.UpsamplingBilinear2d(Upscale_factor)
    x2 = x2(self.SPP2x2(x))
    x4 = nn.UpsamplingBilinear2d(Upscale_factor)
    x4 = x4(self.SPP4x4(x))
    x = x + x1 + x2 + x4
    x = self.conv(x)
    return x
'''
Classifer for clearing out unwanted image patches without  target objects.
'''
class ImagePatchClassifier(BaseModel):
  def __init__(self):
    super().__init__()
    self.globalAttendtionBlock = GlobalAttentionSPP()
    self.firstLayer = nn.Sequential(
        nn.Conv2d(in_channels=48, out_channels=128, #get the kernel size from the last layer of Global attention block
                  kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        activation_func('relu'))
    self.secondLayer = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=256,
                  kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        activation_func('relu'))
    self.conv1x1 = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1) #1*1 convolution
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, x):
    x = self.globalAttendtionBlock(x)
    x = self.firstLayer(x)
    x = self.secondLayer(x)
    Pool_kernel = x.size()[2] # image must be square in size
    Pool = nn.AvgPool2d(kernel_size=Pool_kernel)
    x = Pool(x)
    x = self.conv1x1(x)
    x = torch.squeeze(x) # get a array of two elements
    #x = self.softmax(x) #Cross entropy loss contain softmax, so skip it if use that
    return x

'''
RRDB Generator -- Taken from https://github.com/xinntao/BasicSR/blob/master/codes/models/modules/RRDBNet_arch.py
This is the G for Proposed GAN
'''
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = mutil.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out

'''
VGG descriminator (D) for ESRGAN
Taken from: https://github.com/xinntao/BasicSR/blob/master/codes/models/modules/discriminator_vgg_arch.py
'''
class Discriminator_VGG_128(nn.Module):
    def __init__(self, in_nc, nf):
        super(Discriminator_VGG_128, self).__init__()
        # [64, 128, 128]
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        # [64, 64, 64]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        # [128, 32, 32]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        # [256, 16, 16]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [512, 8, 8]
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)

        self.linear1 = nn.Linear(2048 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))

        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))

        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out


class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=False,#data already normalized, need checking
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        if self.use_input_norm:
            mean = torch.Tensor([0.3442, 0.3708, 0.3476]).view(1, 3, 1, 1).to(device)# for cowc dataset
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor([0.1232, 0.1230, 0.1284]).view(1, 3, 1, 1).to(device)# for cowc dataset
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        # Assume input range is [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output

'''
Create EESN from this paper: https://ieeexplore.ieee.org/document/8677274 EEGAN - Edge Enhanced GAN
'''

'''
Begining conv layer
'''

'''
Starting layer before Dense-Mask Branch
'''
class BeginEdgeConv(nn.Module):
    def __init__(self):
        super(BeginEdgeConv, self).__init__()

        self.conv_layer1 = nn.Conv2d(3, 64, 3, 1, 1, bias=True)
        self.conv_layer2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_layer3 = nn.Conv2d(64, 128, 3, 2, 1, bias=True)
        self.conv_layer4 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.conv_layer5 = nn.Conv2d(128, 256, 3, 2, 1, bias=True)
        self.conv_layer6 = nn.Conv2d(256, 64, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv_layer1, self.conv_layer2, self.conv_layer3,
                            self.conv_layer4, self.conv_layer5, self.conv_layer6], 0.1)

    def forward(self, x):
      x = self.lrelu(self.conv_layer1(x))
      x = self.lrelu(self.conv_layer2(x))
      x = self.lrelu(self.conv_layer3(x))
      x = self.lrelu(self.conv_layer4(x))
      x = self.lrelu(self.conv_layer5(x))
      x = self.lrelu(self.conv_layer6(x))

      return x

'''
Dense sub branch
'''

class EESNRRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(EESNRRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = mutil.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        #fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        #fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.lrelu(self.conv_last(self.lrelu(self.HRconv(fea))))

        return out

'''
Second: Mask Branch of two Dense-Mask branch
'''
class MaskConv(nn.Module):
    def __init__(self):
        super(MaskConv, self).__init__()

        self.conv_layer1 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_layer2 = nn.Conv2d(64, 128, 3, 1, 1, bias=True)
        self.conv_layer3 = nn.Conv2d(128, 256, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv_layer1, self.conv_layer2, self.conv_layer3], 0.1)

    def forward(self, x):
      x = self.lrelu(self.conv_layer1(x))
      x = self.lrelu(self.conv_layer2(x))
      x = self.lrelu(self.conv_layer3(x))
      x = torch.sigmoid(x)

      return x

'''
Final conv layer on Edge Enhanced network
'''
class FinalConv(nn.Module):
    def __init__(self):
        super(FinalConv, self).__init__()

        self.upconv1 = nn.Conv2d(256, 128, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(128, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.upconv1(F.interpolate(x, scale_factor=2, mode='nearest')))
        x = self.lrelu(self.upconv2(F.interpolate(x, scale_factor=2, mode='nearest')))
        x = self.conv_last(self.lrelu(self.HRconv(x)))

        return x

'''
Only EESN
'''
class EESN(nn.Module):
  def __init__(self):
    super(EESN, self).__init__()
    self.beginEdgeConv = BeginEdgeConv() #  Output 64*64*64 input 3*64*64
    self.denseNet = EESNRRDBNet(64, 256, 64, 5) # RRDB densenet with 64 in kernel, 256 out kernel and 64 intermediate kernel, output: 256*64*64
    self.maskConv = MaskConv() # Output 256*64*64
    self.finalConv = FinalConv() # Output 3*256*256

  def forward(self, x):
    x_lap = kornia.laplacian(x, 3) # see kornia laplacian kernel
    x1 = self.beginEdgeConv(x_lap)
    x2 = self.denseNet(x1)
    x3 = self.maskConv(x1)
    x4 = x3*x2 + x2
    x_learned_lap = self.finalConv(x4)

    return x_learned_lap, x_lap
'''
combined EESN
'''

class ESRGAN_EESN(nn.Module):
  def __init__(self, in_nc, out_nc, nf, nb):
    super(ESRGAN_EESN, self).__init__()
    self.netRG = RRDBNet(in_nc, out_nc, nf, nb)
    self.netE = EESN()

  def forward(self, x):
    x_base = self.netRG(x) # add bicubic according to the implementation by author but not stated in the paper
    x5, x_lap = self.netE(x_base) # EESN net
    x_sr = x5 + x_base - x_lap

    return x_base, x_sr, x5, x_lap

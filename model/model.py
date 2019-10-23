import torch
import torch.nn as nn
import torch.nn.functional as F
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
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
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

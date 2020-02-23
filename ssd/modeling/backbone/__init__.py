from ssd.modeling import registry
from .vgg import vgg
from .mobilenet import MobileNetV2
from .efficient_net import EfficientNet

def build_backbone():
    return vgg()

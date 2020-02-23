from ssd.modeling import registry
from .box_head import SSDBoxHead


def build_box_head():
    return SSDBoxHead()

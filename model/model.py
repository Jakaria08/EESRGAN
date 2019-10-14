import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.Identity = nn.Identity()
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

class TinyNet(BaseModel):
    '''
    Introduced in this paper: . Pang, C. Li, J. Shi, Z. Xu and H. Feng,
    "$\mathcal{R}^2$ -CNN: Fast Tiny Object Detection in Large-Scale Remote Sensing Images,"
    in IEEE Transactions on Geoscience and Remote Sensing, vol. 57, no. 8, pp. 5512-5524,
    Aug. 2019. doi: 10.1109/TGRS.2019.2899955
    '''
    def __init__(self, num_classes=2):
        super().__init__()

    def forward(self, x):
        pass

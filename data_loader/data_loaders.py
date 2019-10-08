from torchvision import datasets, transforms
from base import BaseDataLoader
from scripts_for_datasets import COWCDataset
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose,
    BboxParams, RandomCrop, Normalize, Resize, VerticalFlip
)

from albumentations.pytorch import ToTensor


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class COWCDataLoader(BaseDataLoader):
    """
    COWC data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        #data transformation
        #According to this link: https://discuss.pytorch.org/t/normalization-of-input-image/34814/8
        #satellite image 0.5 is good otherwise calculate mean and std for the whole dataset.
        data_transforms = Compose([
            Resize(256, 256),
            HorizontalFlip(),
            OneOf([
                    IAAAdditiveGaussianNoise(),
                    GaussNoise(),
                ], p=0.2),
            OneOf([
                    CLAHE(clip_limit=2),
                    IAASharpen(),
                    IAAEmboss(),
                    RandomBrightnessContrast(),
                ], p=0.3),
                HueSaturationValue(p=0.3),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
        ], p=0.5,
            bbox_params=BboxParams(
             format='pascal_voc',
             min_area=0,
             min_visibility=0,
             label_fields=['labels'])
        )
        self.data_dir = data_dir
        self.dataset = COWCDataset(self.data_dir, transform = data_transforms)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

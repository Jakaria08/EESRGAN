'''
quick and dirty test, need to change later
'''
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torch.nn.parallel import DataParallel
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
from detection.engine import train_one_epoch, evaluate
from detection.utils import collate_fn
from scripts_for_datasets import COWCFRCNNDataset
from detection.transforms import ToTensor, RandomHorizontalFlip, Compose

class COWCFRCNNTrainer:
    """
    Trainer class
    """
    def __init__(self, config):
        self.config = config

        n_gpu = torch.cuda.device_count()
        self.device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')

    def get_transform(self, train):
        transforms = []
        # converts the image, a PIL image, into a PyTorch Tensor
        transforms.append(ToTensor())
        if train:
            # during training, randomly flip the training images
            # and ground-truth for data augmentation
            transforms.append(RandomHorizontalFlip(0.5))
        return Compose(transforms)

    def data_loaders(self):
        # use our dataset and defined transformations
        dataset = COWCFRCNNDataset(root=self.config['data_loader']['args']['data_dir_GT'],
                    transforms=self.get_transform(train=True))
        dataset_test = COWCFRCNNDataset(root=self.config['data_loader']['args']['data_dir_Valid'],
                         transforms=self.get_transform(train=False))

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=4,
            collate_fn=collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=collate_fn)

        return data_loader, data_loader_test

    def save_model(self, network, network_labe, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(self.config['path']['FRCNN_model'], save_filename)

        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def train(self):
        # load a model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # replace the classifier with a new one, that has
        # num_classes which is user-defined
        num_classes = 2  # 1 class (car) + background
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        model.to(self.device)

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)

        # and a learning rate scheduler which decreases the learning rate by
        # 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        data_loader, data_loader_test = self.data_loaders()
        # let's train it for 10 epochs
        num_epochs = 10000

        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, self.device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            if epoch % 10 == 0:
                evaluate(model, data_loader_test, device=self.device)
            if epoch % 500 == 0:
                self.save_model(model, 'FRCNN', epoch)

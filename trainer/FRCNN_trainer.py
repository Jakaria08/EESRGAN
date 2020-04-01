'''
quick and dirty test, need to change later
'''
import torch
import torch.nn as nn
import numpy as np
import torchvision
import os
import cv2
from collections import OrderedDict
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
from detection.engine import train_one_epoch, evaluate_base
from detection.utils import collate_fn
from scripts_for_datasets import COWCFRCNNDataset
from detection.transforms import ToTensor, RandomHorizontalFlip, Compose
from matplotlib import pyplot as plt
from utils import tensor2img

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
        dataset = COWCFRCNNDataset(root=self.config['path']['data_dir_LR_train'],
                    image_height=64, image_width=64, transforms=self.get_transform(train=True))
        dataset_test = COWCFRCNNDataset(root=self.config['path']['data_dir_Valid'],
                         image_height=64, image_width=64, transforms=self.get_transform(train=False))
        dataset_test_SR = COWCFRCNNDataset(root=self.config['path']['data_dir_SR'],
                         transforms=self.get_transform(train=False))
        dataset_test_SR_combined = COWCFRCNNDataset(root=self.config['path']['data_dir_SR_combined'],
                         transforms=self.get_transform(train=False))
        dataset_test_E_SR_1 = COWCFRCNNDataset(root=self.config['path']['data_dir_E_SR_1'],
                         transforms=self.get_transform(train=False))
        dataset_test_E_SR_2 = COWCFRCNNDataset(root=self.config['path']['data_dir_E_SR_2'],
                         transforms=self.get_transform(train=False))
        dataset_test_E_SR_3 = COWCFRCNNDataset(root=self.config['path']['data_dir_E_SR_3'],
                         transforms=self.get_transform(train=False))
        dataset_test_F_SR = COWCFRCNNDataset(root=self.config['path']['data_dir_F_SR'],
                         transforms=self.get_transform(train=False))
        dataset_test_Bic = COWCFRCNNDataset(root=self.config['path']['data_dir_Bic'],
                         transforms=self.get_transform(train=False))

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=4,
            collate_fn=collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=collate_fn)

        data_loader_test_SR = torch.utils.data.DataLoader(
            dataset_test_SR, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=collate_fn)

        data_loader_test_SR_combined = torch.utils.data.DataLoader(
            dataset_test_SR_combined, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=collate_fn)

        data_loader_test_E_SR_1 = torch.utils.data.DataLoader(
            dataset_test_E_SR_1, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=collate_fn)

        data_loader_test_E_SR_2 = torch.utils.data.DataLoader(
            dataset_test_E_SR_2, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=collate_fn)

        data_loader_test_E_SR_3 = torch.utils.data.DataLoader(
            dataset_test_E_SR_3, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=collate_fn)

        data_loader_test_F_SR = torch.utils.data.DataLoader(
            dataset_test_F_SR, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=collate_fn)

        data_loader_test_Bic = torch.utils.data.DataLoader(
            dataset_test_Bic, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=collate_fn)

        return data_loader, data_loader_test, data_loader_test_SR, data_loader_test_SR_combined, \
                data_loader_test_E_SR_1, data_loader_test_E_SR_2, data_loader_test_E_SR_3, \
                data_loader_test_F_SR, data_loader_test_Bic

    def save_model(self, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(self.config['path']['FRCNN_model'], save_filename)

        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_model(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)
        print("model_loaded")

    '''
    Draw boxes on the test images
    '''
    def draw_detection_boxes(self, new_class_conf_box, file_path):
        source_image_path = os.path.join(self.config['path']['data_dir_Bic_valid'], os.path.splitext(os.path.basename(file_path))[0]+'.jpg')
        dest_image_path = os.path.splitext(file_path)[0]+'.jpg'
        #print(dest_image_path)
        image = cv2.imread(source_image_path,1)
        #print(new_class_conf_box)
        #print(len(new_class_conf_box))
        for i in range(len(new_class_conf_box)):
            clas,con,x1,y1,x2,y2 = new_class_conf_box[i]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 4)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, 'Car: '+ str((int(con*100))) + '%', ((x1)+5, (y1)+8), font, 0.2,(0,255,0),1,cv2.LINE_AA)

        cv2.imwrite(dest_image_path, image)

    '''
    for generating test boxes
    '''
    def get_prediction(self, model, images, annotation_path, threshold=0.5):
        new_class_conf_box = list()
        image = list(img.to(self.device) for img in images)
        outputs = model(image)
        file_path = os.path.join(self.config['path']['Test_Result_LR_LR_COWC'], os.path.basename(annotation_path))
        #print(file_path)
        pred_class = [i for i in list(outputs[0]['labels'].detach().cpu().numpy())] # Get the Prediction Score
        text_boxes = [ [i[0], i[1], i[2], i[3] ] for i in list(outputs[0]['boxes'].detach().cpu().numpy())] # Bounding boxes
        pred_score = list(outputs[0]['scores'].detach().cpu().numpy())
        #print(pred_score)
        for i in range(len(text_boxes)):
            if pred_score[i]<0.8:
                continue
            new_class_conf_box.append([pred_class[i], pred_score[i], int(text_boxes[i][0]*4), int(text_boxes[i][1]*4), int(text_boxes[i][2]*4), int(text_boxes[i][3]*4)])
        self.draw_detection_boxes(new_class_conf_box, file_path)
        new_class_conf_box1 = np.matrix(new_class_conf_box)
        #print(new_class_conf_box)
        if(len(new_class_conf_box))>0:
            np.savetxt(file_path, new_class_conf_box1, fmt="%i %1.3f %i %i %i %i")

    #get test results
    def test(self):

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

        self.load_model(self.config['path']['pretrain_model_FRCNN_LR_LR'], model)

        _, data_loader_test, data_loader_test_SR, data_loader_test_SR_combined, \
                data_loader_test_E_SR_1, data_loader_test_E_SR_2, data_loader_test_E_SR_3, \
                 data_loader_test_F_SR, data_loader_test_Bic = self.data_loaders()

        print("test lenghts of the data loaders.............")
        print(len(data_loader_test))
        model.eval()
        i = 0
        print("Detection started........")
        for image, targets, annotation_path in data_loader_test:
            annotation_path = ''.join(annotation_path)
            self.get_prediction(model, image, annotation_path)
            #evaluate_base(model, data_loader_test_Bic, device=self.device)
            i=i+1
            print(i)
        print("successfully generated the results!")
        '''
        print(len(data_loader_test_SR))
        print(len(data_loader_test_SR_combined))
        print(len(data_loader_test_E_SR_1))
        print(len(data_loader_test_E_SR_2))
        print(len(data_loader_test_E_SR_3))
        print(len(data_loader_test_F_SR))
        print(len(data_loader_test_Bic))
        print("test HR images..............................")
        evaluate_base(model, data_loader_test, device=self.device)
        print("test SR images..............................")
        evaluate_base(model, data_loader_test_SR, device=self.device)
        print("test SR combined images..............................")
        evaluate_base(model, data_loader_test_SR_combined, device=self.device)
        print("test Enhanced SR 1 images.....................")
        evaluate_base(model, data_loader_test_E_SR_1, device=self.device)
        print("test Enhanced SR 2 images.....................")
        evaluate_base(model, data_loader_test_E_SR_2, device=self.device)
        print("test Enhanced SR 3 images.....................")
        evaluate_base(model, data_loader_test_E_SR_3, device=self.device)
        print("test Final SR images.........................")
        evaluate_base(model, data_loader_test_F_SR, device=self.device)
        print("test Bicubic images..........................")
        evaluate_base(model, data_loader_test_Bic, device=self.device)
        '''
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
        #self.load_model(self.config['path']['pretrain_model_FRCNN_LR_LR'], model)

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)

        # and a learning rate scheduler which decreases the learning rate by
        # 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        data_loader, data_loader_test, _, _, _, _, _, _, _ = self.data_loaders()
        # let's train it for 10 epochs
        num_epochs = 1000

        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, self.device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate_base(model, data_loader_test, device=self.device)
            if epoch % 1 == 0:
                self.save_model(model, 'FRCNN_LR_LR', epoch)

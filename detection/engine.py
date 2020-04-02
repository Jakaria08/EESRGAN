import math
import sys
import time
import torch
import os
import numpy as np
import cv2

import torchvision.models.detection.mask_rcnn

from .coco_utils import get_coco_api_from_dataset, get_coco_api_from_dataset_base
from .coco_eval import CocoEvaluator
from .utils import MetricLogger, SmoothedValue, warmup_lr_scheduler, reduce_dict
from utils import tensor2img


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #print(images)
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

'''
Draw boxes on the test images
'''
def draw_detection_boxes(new_class_conf_box, config, file_name, image):
    source_image_path = os.path.join(config['path']['output_images'], file_name, file_name+'_112000_final_SR.png')
    dest_image_path = os.path.join(config['path']['Test_Result_SR'], file_name+'.png')
    image = cv2.imread(source_image_path, 1)
    #print(new_class_conf_box)
    #print(len(new_class_conf_box))
    for i in range(len(new_class_conf_box)):
        clas,con,x1,y1,x2,y2 = new_class_conf_box[i]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, 'Car: '+ str((int(con*100))) + '%', (x1+5, y1+8), font, 0.2,(0,255,0),1,cv2.LINE_AA)

    cv2.imwrite(dest_image_path, image)

'''
for generating test boxes
'''
def get_prediction(outputs, file_path, config, file_name, image, threshold=0.5):
    new_class_conf_box = list()
    pred_class = [i for i in list(outputs[0]['labels'].detach().cpu().numpy())] # Get the Prediction Score
    text_boxes = [ [i[0], i[1], i[2], i[3] ] for i in list(outputs[0]['boxes'].detach().cpu().numpy())] # Bounding boxes
    pred_score = list(outputs[0]['scores'].detach().cpu().numpy())
    #print(pred_score)
    for i in range(len(text_boxes)):
        new_class_conf_box.append([pred_class[i], pred_score[i], int(text_boxes[i][0]), int(text_boxes[i][1]), int(text_boxes[i][2]), int(text_boxes[i][3])])
    draw_detection_boxes(new_class_conf_box, config, file_name, image)
    new_class_conf_box1 = np.matrix(new_class_conf_box)
    #print(new_class_conf_box)
    if(len(new_class_conf_box))>0:
        np.savetxt(file_path, new_class_conf_box1, fmt="%i %1.3f %i %i %i %i")


@torch.no_grad()
def evaluate_save(model_G, model_FRCNN, data_loader, device, config):
    i = 0
    print("Detection started........")
    for image, targets in data_loader:
        image['image_lq'] = image['image_lq'].to(device)

        _, img, _, _ = model_G(image['image_lq'])
        img_count = img.size()[0]
        images = [img[i] for i in range(img_count)]
        outputs = model_FRCNN(images)
        file_name = os.path.splitext(os.path.basename(image['LQ_path'][0]))[0]
        file_path = os.path.join(config['path']['Test_Result_SR'], file_name+'.txt')
        i=i+1
        print(i)
        img = img[0].detach()[0].float().cpu()
        img = tensor2img(img)
        get_prediction(outputs, file_path, config, file_name, img)
    print("successfully generated the results!")

'''
This evaluate method is changed to pass the generator network and evalute
the FRCNN with generated SR images
'''
@torch.no_grad()
def evaluate(model_G, model_FRCNN, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    #model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model_FRCNN)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image['image_lq'] = image['image_lq'].to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        _, image, _, _ = model_G(image['image_lq'])
        img_count = image.size()[0]
        image = [image[i] for i in range(img_count)]
        outputs = model_FRCNN(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

@torch.no_grad()
def evaluate_base(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset_base(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)
        #print(outputs)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

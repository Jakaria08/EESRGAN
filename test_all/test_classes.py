import pytest
import os
import shutil
import torch
from parse_config import ConfigParser
from utils import read_json, write_json
from scripts_for_datasets import COWCDataset
# run tests
# python -m pytest test_all/
# python -m pytest test_all/ -s ==> to see print statements
class TestCOWCDataset():

    def test_image_annot_equality(self):
        # Test code for init method
        # Testing the dataset size and similarity
        config = read_json('config.json')
        config = ConfigParser(config)
        data_dir = config['data_loader']['args']['data_dir']
        shutil.rmtree("./saved")#removing /saved directory, everytime created by ConfigParser
        a = COWCDataset(data_dir)
        for img, annot in zip(a.imgs, a.annotation):
            if os.path.splitext(img)[0] != os.path.splitext(annot)[0]:
                print("problem")
        print(len(a.annotation))
        assert len(a.imgs) == len(a.annotation), "NOT equal"

    def test_zero_annotation(self):
        # Test for checking number of image without bounding box
        config = read_json('config.json')
        config = ConfigParser(config)
        data_dir = config['data_loader']['args']['data_dir']
        shutil.rmtree("./saved")#removing /saved directory, everytime created by ConfigParser
        a = COWCDataset(data_dir)
        zero_annotation = 0
        for i in range(len(a.annotation)):
            zero_annotation_get =  a[i]
            if zero_annotation_get['object'].item() == 0:
                 zero_annotation += 1
        print("Number of image withot bbox: "+str(zero_annotation))
        #use assert zero_annotation == 0 if all image contain bounding box
        assert zero_annotation != 0, "Image exists with bounding box"

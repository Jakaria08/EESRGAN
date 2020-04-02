# EESRGAN
## Training
`python train.py -c config_GAN.json`
## Testing
`python test.py -c config_GAN.json`
## Edit the JSON File
The directory of the following JSON file is needed to be changed according to the user directory. For details see [config_GAN.json](https://github.com/Jakaria08/EESRGAN/blob/master/config_GAN.json) and pretrained weights are uploaded in [google drive](https://drive.google.com/drive/folders/15xN_TKKTUpQ5EVdZWJ2aZUa4Y-u-Mt0f?usp=sharing)
```yaml
{
    "data_loader": {
        "type": "COWCGANFrcnnDataLoader",
        "args":{
            "data_dir_GT": "/Directory for High-Resolution Ground Truth images/",
            "data_dir_LQ": "/Directory for 4x downsampled Low-Resolution images from the above High-Resolution images/"
        }
    },

    "path": {
        "models": "saved/save_your_model_in_this_directory/",
        "pretrain_model_G": "Pretrained_model_path_for_train_test/164000_G.pth",
        "pretrain_model_D": "Pretrained_model_path_for_train_test/164000_G.pth",
        "pretrain_model_FRCNN": "Pretrained_model_path_for_train_test/164000_G.pth",
        "data_dir_Valid": "/Low_resoluton_test_validation_image_directory/"
    }
}

```
## Paper
Find the priprints of the related paper on [preprints.org](https://www.preprints.org/manuscript/202003.0313/v1), [arxiv.org](https://arxiv.org/abs/2003.09085) and [researchgate.net](https://www.researchgate.net/publication/340095015_Small-Object_Detection_in_Remote_Sensing_Images_with_End-to-End_Edge-Enhanced_GAN_and_Object_Detector_Network).
## Related Repository
Some code segments are based on [ESRGAN](https://github.com/xinntao/BasicSR)
## Citation
`@article{rabbi2020small,`\
  `title={Small-Object Detection in Remote Sensing Images with End-to-End Edge-Enhanced GAN and Object Detector Network},`\
  `author={Rabbi, Jakaria and Ray, Nilanjan and Schubert, Matthias and Chowdhury, Subir and Chao, Dennis},`\
  `journal={arXiv preprint arXiv:2003.09085},`\
  `year={2020}`\
`}` 

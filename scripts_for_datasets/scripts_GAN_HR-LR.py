'''
This script is taken from https://github.com/xinntao/BasicSR/blob/master/codes/scripts/generate_mod_LR_bic.py
Need to understand, but for now using for generating Low res ang bicubic upscale images.
Can be done using much simpler code.
This code is useful for variable size of images but my images are square, so don't Need
this now but useful for later use cases.
'''
import os
import sys
import cv2
import numpy as np
import glob
import shutil

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import imresize_np
except ImportError:
    pass


def generate_mod_LR_bic():
    # set parameters
    up_scale = 4
    mod_scale = 4
    # set data dir
    # directory structure on sunray server pc
    # Need to change later when refactoring, code cleaning and testing.
    sourcedir = '/home/jakaria/Super_Resolution/Datasets/COWC/DetectionPatches_256x256/Potsdam_ISPRS'
    savedir = '/home/jakaria/Super_Resolution/Datasets/COWC/DetectionPatches_256x256/Potsdam_ISPRS'

    saveHRpath = os.path.join(savedir, 'HR', 'x' + str(mod_scale))
    saveLRpath = os.path.join(savedir, 'LR', 'x' + str(up_scale))
    saveBicpath = os.path.join(savedir, 'Bic', 'x' + str(up_scale))

    if not os.path.isdir(sourcedir):
        print('Error: No source data found')
        exit(0)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    if not os.path.isdir(os.path.join(savedir, 'HR')):
        os.mkdir(os.path.join(savedir, 'HR'))
    if not os.path.isdir(os.path.join(savedir, 'LR')):
        os.mkdir(os.path.join(savedir, 'LR'))
    if not os.path.isdir(os.path.join(savedir, 'Bic')):
        os.mkdir(os.path.join(savedir, 'Bic'))

    if not os.path.isdir(saveHRpath):
        os.mkdir(saveHRpath)
    else:
        print('It will cover ' + str(saveHRpath))

    if not os.path.isdir(saveLRpath):
        os.mkdir(saveLRpath)
    else:
        print('It will cover ' + str(saveLRpath))

    if not os.path.isdir(saveBicpath):
        os.mkdir(saveBicpath)
    else:
        print('It will cover ' + str(saveBicpath))

    filepaths = [f for f in os.listdir(sourcedir) if f.endswith('.jpg') and not f.endswith('check.jpg')]
    num_files = len(filepaths)

    # prepare data with augementation
    for i in range(num_files):
        filename = filepaths[i]
        print('No.{} -- Processing {}'.format(i, filename))
        # read image
        image = cv2.imread(os.path.join(sourcedir, filename))

        width = int(np.floor(image.shape[1] / mod_scale))
        height = int(np.floor(image.shape[0] / mod_scale))
        # modcrop
        if len(image.shape) == 3:
            image_HR = image[0:mod_scale * height, 0:mod_scale * width, :]
        else:
            image_HR = image[0:mod_scale * height, 0:mod_scale * width]
        # LR
        image_LR = imresize_np(image_HR, 1 / up_scale, True)
        # bic
        image_Bic = imresize_np(image_LR, up_scale, True)

        cv2.imwrite(os.path.join(saveHRpath, filename), image_HR)
        cv2.imwrite(os.path.join(saveLRpath, filename), image_LR)
        cv2.imwrite(os.path.join(saveBicpath, filename), image_Bic)

def copy_folder_name_for_valid_image():
    Dir_HR = "/home/jakaria/Super_Resolution/Datasets/COWC/DetectionPatches_256x256/Potsdam_ISPRS/HR/x4/"
    Dir_Bic = "/home/jakaria/Super_Resolution/Datasets/COWC/DetectionPatches_256x256/Potsdam_ISPRS/Bic/x4/"
    Dir_LR = "/home/jakaria/Super_Resolution/Datasets/COWC/DetectionPatches_256x256/Potsdam_ISPRS/LR/x4/"

    for file in glob.glob("/home/jakaria/Super_Resolution/Filter_Enhance_Detect/saved_ESRGAN/val_images/*/"):
        img_file = os.path.basename(file[:-1]+'.jpg')
        txt_file = os.path.basename(file[:-1]+'.txt')

        #print(file)
        sourceH = os.path.join(Dir_HR,img_file)
        destinationH = os.path.join(Dir_HR, 'valid_img', img_file)
        shutil.move(sourceH, destinationH)

        sourceHtxt = os.path.join(Dir_HR,txt_file)
        destinationHtxt = os.path.join(Dir_HR, 'valid_img', txt_file)
        shutil.move(sourceHtxt, destinationHtxt)

        sourceB = os.path.join(Dir_Bic,img_file)
        destinationB = os.path.join(Dir_Bic, 'valid_img', img_file)
        shutil.move(sourceB, destinationB)

        sourceBtxt = os.path.join(Dir_Bic,txt_file)
        destinationBtxt = os.path.join(Dir_Bic, 'valid_img', txt_file)
        shutil.move(sourceBtxt, destinationBtxt)

        sourceL = os.path.join(Dir_LR,img_file)
        destinationL = os.path.join(Dir_LR, 'valid_img', img_file)
        shutil.move(sourceL, destinationL)

        sourceLtxt = os.path.join(Dir_LR,txt_file)
        destinationLtxt = os.path.join(Dir_LR, 'valid_img', txt_file)
        shutil.move(sourceLtxt, destinationLtxt)

def merge_edge():
    dir = "/home/jakaria/Super_Resolution/Filter_Enhance_Detect/saved/val_images/*/*"
    img_final_SR = sorted(glob.glob(dir+'_160000_final_SR.png'))
    img_lap = sorted(glob.glob(dir+'_160000_lap.png'))
    mean = np.array([0.3442, 0.3708, 0.3476])
    std = np.array([0.1232, 0.1230, 0.1284])

    for i, j in zip(img_final_SR, img_lap):
        #print(i+'____'+j)
        img_final_SR_1 = cv2.imread(i) / 255
        img_lap_1 = cv2.imread(j) / 255

        img_final_SR_1 = np.clip(img_final_SR_1, 0, 1)
        img_lap_1 = np.clip(img_lap_1, 0, 1)

        img_final_SR_1 = (img_final_SR_1 - mean) / std
        img_lap_1 = (img_lap_1 - mean) / std

        img_final_SR_enhanced = img_final_SR_1 + img_lap_1
        img_final_SR_enhanced = std * img_final_SR_enhanced + mean
        img_final_SR_enhanced = np.clip(img_final_SR_enhanced, 0, 1)

        img_final_SR_enhanced = (img_final_SR_enhanced * 255.0).round().astype(np.uint8)

        folder_name = os.path.dirname(i)
        file_name = os.path.basename(folder_name)
        img_path = os.path.join(folder_name,file_name+'_160000_img_final_SR_enhanced.png')
        #print('_____'+img_path)

        #img_final_SR_enhanced = cv2.cvtColor(img_final_SR_enhanced, cv2.COLOR_BGR2RGB)
        cv2.imwrite(img_path, img_final_SR_enhanced)

if __name__ == "__main__":
    #generate_mod_LR_bic()
    #copy_folder_name_for_valid_image()
    merge_edge()

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
import kornia

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import imresize_np, calculate_psnr, calculate_ssim
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
    img_SR = sorted(glob.glob(dir+'_20000_SR.png'))
    img_lap = sorted(glob.glob(dir+'_20000_lap.png'))
    img_lap_learned = sorted(glob.glob(dir+'_20000_lap_learned.png'))
    mean = np.array([0.3442, 0.3708, 0.3476])
    std = np.array([0.1232, 0.1230, 0.1284])

    for i, j, k in zip(img_SR, img_lap, img_lap_learned):
        #print(i+'____'+j)
        img_SR_1 = cv2.imread(i) / 255
        img_lap_1 = cv2.imread(j) / 255
        img_lap_learned_1 = cv2.imread(k) / 255

        img_SR_1 = np.clip(img_SR_1, 0, 1)
        img_lap_1 = np.clip(img_lap_1, 0, 1)
        img_lap_learned_1 = np.clip(img_lap_learned_1, 0, 1)

        img_SR_1 = (img_SR_1 - mean) / std
        img_lap_1 = (img_lap_1 - mean) / std
        img_lap_learned_1 = (img_lap_learned_1 - mean) / std

        #img_final_SR_enhanced = img_SR_1 + (img_lap_1 - img_lap_learned_1)
        img_final_SR_enhanced = img_SR_1 + (img_lap_learned_1 - img_lap_1)
        img_final_SR_enhanced = std * img_final_SR_enhanced + mean
        img_final_SR_enhanced = np.clip(img_final_SR_enhanced, 0, 1)

        img_final_SR_enhanced = (img_final_SR_enhanced * 255.0).round().astype(np.uint8)

        folder_name = os.path.dirname(i)
        file_name = os.path.basename(folder_name)
        img_path = os.path.join(folder_name,file_name+'_20000_img_final_SR_enhanced.png')
        #print('_____'+img_path)

        #img_final_SR_enhanced = cv2.cvtColor(img_final_SR_enhanced, cv2.COLOR_BGR2RGB)
        cv2.imwrite(img_path, img_final_SR_enhanced)

def calculate_psnr_ssim():
    dir = "/home/jakaria/Super_Resolution/Filter_Enhance_Detect/saved/val_images/*/*"
    bicubic_DIR = "/home/jakaria/Super_Resolution/Datasets/COWC/DetectionPatches_256x256/Potsdam_ISPRS/Bic/x4/valid_img/*"
    img_GT = sorted(glob.glob(dir+'_20000_GT.png'))
    img_final_SR_enhanced = sorted(glob.glob(dir+'_20000_img_final_SR_enhanced.png'))
    img_final_SR = sorted(glob.glob(dir+'_20000_final_SR.png'))
    img_SR = sorted(glob.glob(dir+'_20000_SR.png'))
    img_Bic = sorted(glob.glob(bicubic_DIR+'.jpg'))


    psnr_enhanced = 0
    psnr_final = 0
    psnr_SR = 0
    psnr_Bic = 0

    ssim_enhanced = 0
    ssim_final = 0
    ssim_SR = 0
    ssim_Bic = 0

    total = len(img_SR)
    print(total)

    i = 0

    for im_gt, im_enhanced, im_final, im_SR, im_Bic in zip(img_GT, img_final_SR_enhanced,
                                                    img_final_SR, img_SR, img_Bic):
        print(os.path.basename(im_gt)+'--', os.path.basename(im_enhanced)+'--',
        os.path.basename(im_final)+'--', os.path.basename(im_SR)+'--',
        os.path.basename(im_Bic))

        image_gt = cv2.imread(im_gt)
        image_enhanced = cv2.imread(im_enhanced)
        image_final = cv2.imread(im_final)
        image_SR = cv2.imread(im_SR)
        image_Bic = cv2.imread(im_Bic)

        psnr_enhanced += calculate_psnr(image_gt, image_enhanced)
        psnr_final += calculate_psnr(image_gt, image_final)
        psnr_SR += calculate_psnr(image_gt, image_SR)
        psnr_Bic += calculate_psnr(image_gt, image_Bic)

        ssim_enhanced += calculate_ssim(image_gt, image_enhanced)
        ssim_final += calculate_ssim(image_gt, image_final)
        ssim_SR += calculate_ssim(image_gt, image_SR)
        ssim_Bic += calculate_ssim(image_gt, image_Bic)

        i += 1
        print(i)

    avg_psnr_enhanced, avg_psnr_final, avg_psnr_SR, avg_psnr_Bic = (psnr_enhanced / total,
                                                                      psnr_final / total,
                                                                      psnr_SR / total,
                                                                      psnr_Bic / total)

    avg_ssim_enhanced, avg_ssim_final, avg_ssim_SR, avg_ssim_Bic = (ssim_enhanced / total,
                                                                      ssim_final / total,
                                                                      ssim_SR / total,
                                                                      ssim_Bic / total)

    text_file = open("/home/jakaria/Super_Resolution/Filter_Enhance_Detect/saved/Output.txt", "a")
    print("Enhanced PSNR: %4.2f" % avg_psnr_enhanced)
    text_file.write("Enhanced PSNR: %4.2f \n" % avg_psnr_enhanced)
    print("Final PSNR: %4.2f"%avg_psnr_final)
    text_file.write("Final PSNR: %4.2f \n"%avg_psnr_final)
    print("SR PSNR: %4.2f"%avg_psnr_SR)
    text_file.write("SR PSNR: %4.2f \n"%avg_psnr_SR)
    print("Bic PSNR: %4.2f"%avg_psnr_Bic)
    text_file.write("Bic PSNR: %4.2f \n"%avg_psnr_Bic)

    print("Enhanced SSIM: %5.4f"%avg_ssim_enhanced)
    text_file.write("Enhanced SSIM: %5.4f \n"%avg_ssim_enhanced)
    print("Final SSIM: %5.4f"%avg_ssim_final)
    text_file.write("Final SSIM: %5.4f \n"%avg_ssim_final)
    print("SR SSIM: %5.4f"%avg_ssim_SR)
    text_file.write("SR SSIM: %5.4f \n"%avg_ssim_SR)
    print("Bic SSIM: %5.4f"%avg_ssim_Bic)
    text_file.write("Bic SSIM: %5.4f \n"%avg_ssim_Bic)
    text_file.close()

def calculate_psnr_ssim_ESRGAN():
    dir = "/home/jakaria/Super_Resolution/Filter_Enhance_Detect/saved_ESRGAN/val_images/*/*"
    HR_DIR = "/home/jakaria/Super_Resolution/Datasets/COWC/DetectionPatches_256x256/Potsdam_ISPRS/HR/x4/valid_img/"
    img_SR = sorted(glob.glob(dir+'_300000.png'))

    psnr_SR = 0
    ssim_SR = 0

    total = len(img_SR)
    print(total)

    i = 0

    for im_SR in img_SR:
        print(os.path.basename(im_SR)+'--')
        im_gt = os.path.basename(im_SR)
        im_gt = im_gt.rsplit('_', 1)[0]+".jpg"
        im_gt = os.path.join(HR_DIR, im_gt)
        print(im_gt)

        image_SR = cv2.imread(im_SR)
        image_SR = cv2.cvtColor(image_SR, cv2.COLOR_BGR2RGB)
        cv2.imwrite(im_SR, image_SR)

        image_gt = cv2.imread(im_gt)
        image_SR = cv2.imread(im_SR)

        psnr_SR += calculate_psnr(image_gt, image_SR)
        ssim_SR += calculate_ssim(image_gt, image_SR)

        i += 1
        print(i)

    avg_psnr_SR = psnr_SR / total
    avg_ssim_SR = ssim_SR / total

    text_file = open("/home/jakaria/Super_Resolution/Filter_Enhance_Detect/saved_ESRGAN/Output.txt", "a")
    print("SR PSNR: %4.2f"%avg_psnr_SR)
    text_file.write("SR PSNR: %4.2f \n"%avg_psnr_SR)
    print("SR SSIM: %5.4f"%avg_ssim_SR)
    text_file.write("SR SSIM: %5.4f \n"%avg_ssim_SR)

def separate_generated_image_for_test():
    dir = "/home/jakaria/Super_Resolution/Filter_Enhance_Detect/saved/val_images/*/*"
    dir_ESRGAN = "/home/jakaria/Super_Resolution/Filter_Enhance_Detect/saved_EEGAN_separate/val_images/*/*"
    dir_save = "/home/jakaria/Super_Resolution/Filter_Enhance_Detect/saved/"

    img_final_SR = sorted(glob.glob(dir+'_20000_final_SR.png'))
    img_SR = sorted(glob.glob(dir+'_20000_SR.png'))
    img_enhanced_SR = sorted(glob.glob(dir_ESRGAN+'_400000_img_final_SR_enhanced.png'))

    for im_final_SR, im_SR, im_enhanced_SR in zip(img_final_SR, img_SR, img_enhanced_SR):
        image_final_SR = cv2.imread(im_final_SR)
        image_SR = cv2.imread(im_SR)
        image_enhanced_SR = cv2.imread(im_enhanced_SR)

        image_final_SR = cv2.flip(image_final_SR, 0)
        image_SR = cv2.flip(image_SR, 0)
        image_enhanced_SR = cv2.flip(image_enhanced_SR, 0)

        final_SR_Dir = os.path.basename(im_final_SR)
        final_SR_Dir = final_SR_Dir.rsplit('_', 3)[0]+".png"
        final_SR_Dir = os.path.join(dir_save, 'final_SR_images', final_SR_Dir)
        cv2.imwrite(final_SR_Dir, image_final_SR)

        SR_Dir = os.path.basename(im_SR)
        SR_Dir = SR_Dir.rsplit('_', 2)[0]+".png"
        SR_Dir = os.path.join(dir_save, 'SR_images', SR_Dir)
        cv2.imwrite(SR_Dir, image_SR)

        enhanced_SR_Dir = os.path.basename(im_enhanced_SR)
        enhanced_SR_Dir = enhanced_SR_Dir.rsplit('_', 5)[0]+".png"
        enhanced_SR_Dir = os.path.join(dir_save, 'enhanced_SR_images', enhanced_SR_Dir)
        cv2.imwrite(enhanced_SR_Dir, image_enhanced_SR)


if __name__ == "__main__":
    #generate_mod_LR_bic()
    #copy_folder_name_for_valid_image()
    #merge_edge()
    #calculate_psnr_ssim()
    #calculate_psnr_ssim_ESRGAN() #not working expected, use the other methods.
    separate_generated_image_for_test()

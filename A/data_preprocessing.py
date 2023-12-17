
# -*- encoding: utf-8 -*-
'''
@File    :   data_preprocessing.py
@Time    :   2023/12/16 19:46:34
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0134 Applied Machine Learning Systems
@Author  :   Wenrui Li
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file includes all data preprocessing procedures for task A. 
        Notice that there are lots of comments here as trials and experiments for comparison. 
        Most of the results are explained and visualized in the report.
'''

# here put the import lib
import os
import cv2
import random
import numpy as np
from medmnist import PathMNIST
import matplotlib.pyplot as plt
from medmnist import PneumoniaMNIST  # cannot be used on gpu but can be used on my cpu, maybe because of package version
from imblearn.over_sampling import SMOTE

from utils import visual4label

"""
    In this project, two kinds of datasets are used. 
    The first one is the npz format download from the website with provided PneumoniaMNIST and PathMNIST package.
    The second one is the dataset saved as .png format with the commands running in terminal.
    Detailed dataset deployment can be seen in README.md and Github link.
"""
# # To download dataset in npz format from MeMNIST website, can use the sentence below
# dataset1 = PneumoniaMNIST(split='train',download=True,root="Datasets/")

# # how to save the dataset as png figure and csv (reference from MeMNIST)
# python -m medmnist save --flag=pneumoniamnist --postfix=png --folder=Datasets/ --root=Datasets/  

'''
description: This function is used for histogram equalization and comparison of CLAHE method.
param {*} path: path of raw dataset
param {*} f: filename
return {*}: original image, image after histogram equalization, image after CLAHE
'''
def histogram_equalization(path, f):
    
    img = cv2.imread(os.path.join(path,f),0)
    equ = cv2.equalizeHist(img)

    # create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(img)

    return img, equ, cl

'''
description: This function is used for Sobel operation.
param {*} imgEqu: image after histogram equalization
return {*}: image after Sobel operation
'''
def sobel(imgEqu):
    SobelX = cv2.Sobel(imgEqu, cv2.CV_16S, 1, 0)  
    SobelY = cv2.Sobel(imgEqu, cv2.CV_16S, 0, 1)  
    absX = cv2.convertScaleAbs(SobelX)  
    absY = cv2.convertScaleAbs(SobelY) 
    SobelXY = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)  
    imgSobel = np.uint8(cv2.normalize(SobelXY, None, 0, 255, cv2.NORM_MINMAX))

    passivation = imgSobel*0.3 + imgEqu  # sobel add on image after histogram equalization
    imgPas = np.uint8(cv2.normalize(passivation, None, 0, 255, cv2.NORM_MINMAX))
    return imgPas

'''
description: This function is used for Gamma correction.
param {*} imgPas: image after Sobel operation
return {*}: image after Gamma correction
'''
def gammaCorrection(imgPas):
    epsilon = 1e-5  
    gamma = np.power(imgPas + epsilon, 0.5)
    imgGamma = np.uint8(cv2.normalize(gamma, None, 0, 255, cv2.NORM_MINMAX))
    return imgGamma

'''
description: This function is used for data augmentation with rotation operation.
return {*}:  image after rotation along center for 45 degree.
'''
# rotation
def rotation(img):
    h, w, c = img.shape
    M = cv2.getRotationMatrix2D((w/2,h/2),45,1) 
    imgRot = cv2.warpAffine(img,M,(w,h))  
    return imgRot

'''
description: This function is used for data augmentation with width and height shifts.
return {*}: shifted image
'''
# width shift & height shift 
def shift(img):
    h, w, c = img.shape
    H = np.float32([[1,0,5],
                    [0,1,5]])
    imgShift = cv2.warpAffine(img,H,(w,h))
    return imgShift

'''
description: This function is used for data augmentation with shearing.
return {*}: image after shearing
'''
# shear
def shear(img):
    h, w, c = img.shape
    pts1 = np.float32([[0, 0],[0, h-1],[w-1, 0]])
    pts2 = np.float32([[0, 0],[5, h-5],[w-5, 5]])
    M = cv2.getAffineTransform(pts1,pts2)
    imgShear = cv2.warpAffine(img,M,(w,h))
    return imgShear

'''
description: This function is used for data augmentation with zooming operation.
return {*}: resized image with half of the width and height.
'''
# zoom
def zoom(img):
    h, w, c = img.shape
    imgZoom = cv2.resize(img,(int(0.5*w),int(0.5*h)),interpolation=cv2.INTER_CUBIC)
    keepSize = np.zeros((h,w,c),np.uint8)
    keepSize[:int(0.5*h), :int(0.5*w), :] = imgZoom
    return keepSize

'''
description: This function is used for data augmentation with horizontal flip.
return {*}: flipped image
'''
# horizontal flip
def horizontalFlip(img):
    imgFlip = cv2.flip(img, 1) 
    return imgFlip

'''
description: This function is used for data augmentation with five techniques.
param {*} path: preprocessed data path
param {*} length_1: number of label 1
param {*} length_0: number of label 0
param {*} pre_file_0: specific files: all train images, all validation images or all test images
param {*} mode: train, validation or test
return {*}: new length of the whole dataset after data augmentation
'''
def data_augmentation(path, length_1, length_0, pre_file_0, mode=None):
    imgs = []  # augmented images

    # sampling from minority class
    flip_index = random.sample([i for i in range(length_0)],int((length_1-length_0)/5)) 
    for i in flip_index:
        img = cv2.imread(os.path.join(path,f'{pre_file_0[i]}'))
        imgFlip = horizontalFlip(img)
        imgs.append(imgFlip)

    shear_index = random.sample([i for i in range(length_0)],int((length_1-length_0)/5))
    for i in shear_index:
        img = cv2.imread(os.path.join(path,f'{pre_file_0[i]}'))
        imgShear = shear(img)
        imgs.append(imgShear)
    
    zoom_index = random.sample([i for i in range(length_0)],int((length_1-length_0)/5))
    for i in zoom_index:
        img = cv2.imread(os.path.join(path,f'{pre_file_0[i]}'))
        imgZoom = zoom(img)
        imgs.append(imgZoom)
    
    shift_index = random.sample([i for i in range(length_0)],int((length_1-length_0)/5))
    for i in shift_index:
        img = cv2.imread(os.path.join(path,f'{pre_file_0[i]}'))
        imgShift = shift(img)
        imgs.append(imgShift)
    
    rot_index = random.sample([i for i in range(length_0)],int((length_1-length_0-len(imgs))))
    for i in rot_index:
        img = cv2.imread(os.path.join(path,f'{pre_file_0[i]}'))
        imgRot = rotation(img)
        imgs.append(imgRot)
    
    for index,i in enumerate(imgs):
        cv2.imwrite(os.path.join(path,f'{mode}{length_0+length_1+index}_0.png'),i)

    return length_1+length_0+len(imgs)


'''
description: This function is used for showing basic data descriptions,
        like number of labels of each class, number of train/validation/test images.
param {*} npz: whether download npz data, if you don't use provided backup project and want to check the project
        from original dataset, please use the argument here. Detailed guideline is shown in README.md and Github link.
return {*}: npz data
'''
def load_data_log4A(npz):
    if npz == True:
        download_dataset = PneumoniaMNIST(split='train',download=True,root="Datasets/")
    data = np.load("Datasets/pneumoniamnist.npz")
    print(f"Train data length: {len(data['train_images'])}, label 0: {np.count_nonzero(data['train_labels'].flatten() == 0)}, label 1: {np.count_nonzero(data['train_labels'].flatten() == 1)}")
    print(f"Validation data length: {len(data['val_images'])}, label 0: {np.count_nonzero(data['val_labels'].flatten() == 0)}, label 1: {np.count_nonzero(data['val_labels'].flatten() == 1)}")                                                               
    print(f"Test data length: {len(data['test_images'])}, label 0: {np.count_nonzero(data['test_labels'].flatten() == 0)}, label 1: {np.count_nonzero(data['test_labels'].flatten() == 1)}")
    visual4label("A",data)  # label distribution
    return data


'''
description: This function is a conclusion for all data preprocessing procedures.
param {*} raw_path: raw dataset path
return {*}: size of train/validation/test dataset after all data preprocessing procedures.
'''
def data_preprocess4A(raw_path):
    print("Start preprocessing data......")
    data = load_data_log4A()
    raw_file=os.listdir(raw_path)

    # data preprocessing
    os.makedirs('Outputs/pneumoniamnist/preprocessed_data', exist_ok=True)
    for index,f in enumerate(raw_file):
        if not os.path.isfile(os.path.join(raw_path,f)):
            continue
        else:
            img, equ, cl = histogram_equalization(raw_path, f)
            imgPas = sobel(equ)
            imgGamma = gammaCorrection(imgPas)
            cv2.imwrite(os.path.join('Outputs/pneumoniamnist/preprocessed_data',f'{f}'),imgGamma)

    # data augmentation
    pre_path = 'Outputs/pneumoniamnist/preprocessed_data'
    pre_file = os.listdir(pre_path)
    pre_file_test_0 = []
    pre_file_train_0 = []
    pre_file_val_0 = []

    for index,f in enumerate(pre_file): # construct individual train/validation/test files aggregation
            if not os.path.isfile(os.path.join(pre_path,f)):
                continue
            else:
                if (f.split("_")[1][0] == "0"):
                    if "test" in f:
                        pre_file_test_0.append(f)
                    elif "train" in f:
                        pre_file_train_0.append(f)
                    elif "val" in f:
                        pre_file_val_0.append(f)
    
    new_train_length = data_augmentation(pre_path,len(data['train_images'])-len(pre_file_train_0),len(pre_file_train_0), pre_file_train_0,mode="train")
    new_test_length = data_augmentation(pre_path,len(data['test_images'])-len(pre_file_test_0),len(pre_file_test_0), pre_file_test_0,mode="test")
    new_val_length = data_augmentation(pre_path, len(data['val_images'])-len(pre_file_val_0),len(pre_file_val_0), pre_file_val_0,mode="val")

    print("Finish preprocessing data.")
    return new_train_length, new_test_length, new_val_length


"""
    Experiments for comparison of CLAHE, Laplacian, Sobel operation, Gamma correction, etc.
    This part is for experiment and will not be included in the committed code
"""
# Here I compare the performance for with or without each step of laplacian, sobel, gamma
# the most suitable one is sobel + gamma

# # lapacian: used to emphasize the details for original image
# img = cv2.imread("Datasets/pneumoniamnist/test298_1.png",0)
# img, equ, cl1 = histogram_equalization("test298_1.png")
# plt.subplot(1,3,1)
# plt.title("Original")
# plt.hist(img.ravel(),256,[0,256])
# plt.subplot(1,3,2)
# plt.title("Histogram equalization")
# plt.hist(equ.ravel(),256,[0,256])
# plt.subplot(1,3,3)
# plt.title("CLAHE")
# plt.hist(cl1.ravel(),256,[0,256])
# plt.savefig('Outputs/histogram_example.png')

# # laplacian
# kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]],np.int8)
# laplacian = cv2.filter2D(equ, ddepth=-1, kernel=kernel)
# imglaplacian = np.uint8(cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX))
# addlap = equ+imglaplacian
# addlap = np.uint8(cv2.normalize(addlap, None, 0, 255, cv2.NORM_MINMAX))

# # sobel
# SobelX = cv2.Sobel(equ, cv2.CV_16S, 1, 0)  # 计算 x 轴方向
# SobelY = cv2.Sobel(equ, cv2.CV_16S, 0, 1)  # 计算 y 轴方向
# absX = cv2.convertScaleAbs(SobelX)  # 转回 uint8
# absY = cv2.convertScaleAbs(SobelY)  # 转回 uint8
# SobelXY = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)  # 用绝对值近似平方根
# imgSobel = np.uint8(cv2.normalize(SobelXY, None, 0, 255, cv2.NORM_MINMAX))

# passivation1 = imgSobel*0.3 + equ  # sobel add on image after histogram equalization
# passivation2 = imgSobel*0.3 + addlap  # sobel add on image after laplacian
# imgPassi1 = np.uint8(cv2.normalize(passivation1, None, 0, 255, cv2.NORM_MINMAX))
# imgPassi2 = np.uint8(cv2.normalize(passivation2, None, 0, 255, cv2.NORM_MINMAX))

# # gamma
# epsilon = 1e-5  
# Gamma1 = np.power(imgPassi1 + epsilon, 0.5)
# Gamma2 = np.power(imgPassi2 + epsilon, 0.5)
# imgGamma1 = np.uint8(cv2.normalize(Gamma1, None, 0, 255, cv2.NORM_MINMAX))
# imgGamma2 = np.uint8(cv2.normalize(Gamma2, None, 0, 255, cv2.NORM_MINMAX))

# res = np.hstack((img,equ,cl1,addlap,imgPassi1,imgPassi2,imgGamma1,imgGamma2))  # stacking images side-by-side
# cv2.imwrite(os.path.join('Outputs/','res.png'),res)
# cv2.imshow("result",res)
# cv2.waitKey(0)

"""
    This part is the implementation of SMOTE introduced in report. 
    Since it's not used in the final processing procedures and just trials for comparison, 
    it's commented and not included in committed code.
"""
# #smote
# path = 'Datasets/pneumoniamnist'
# raw_file=os.listdir(path)
# os.makedirs('Outputs/pneumoniamnist/smote_data', exist_ok=True)
# # print(file_list)   

# for index,f in enumerate(raw_file):
#         # print(os.path.join(path,f))
#         if not os.path.isfile(os.path.join(path,f)):
#             continue
#         else:
#             img, equ, cl = histogram_equalization(f)
#             imgPas = sobel(equ)
#             imgGamma = gammaCorrection(imgPas)
#             # res = np.hstack((img,equ,cl,imgPas,imgGamma))  # stacking images side-by-side
#             cv2.imwrite(os.path.join('Outputs/pneumoniamnist/smote_data',f'{f}'),imgGamma)


# path = 'Outputs/pneumoniamnist/smote_data'
# pre_file=os.listdir(path)
# pre_file_test = []
# pre_file_train = []
# pre_file_val = []

# for index,f in enumerate(pre_file):
#         # print(os.path.join(path,f))
#         if not os.path.isfile(os.path.join(path,f)):
#             continue
#         else:
#             img = cv2.imread(os.path.join(path,f))
#             if "test" in f:
#                 pre_file_test.append(img)
#             elif "train" in f:
#                 pre_file_train.append(img)
#             elif "val" in f:
#                 pre_file_val.append(img)

# def data_smote(prefile, mode=None):
#     smote=SMOTE(random_state=0)
#     s_x, s_y=smote.fit_resample(np.array(prefile).reshape(np.array(prefile).shape[0], 28*28*3), data[f'{mode}_labels'].ravel())
#     s_x = s_x.reshape(s_x.shape[0], 28, 28, 3)
#     # experiments show that the smote data was appended after original data
#     for index,i in enumerate(s_x[np.array(prefile).shape[0]:,:,:,:]):
#         cv2.imwrite(os.path.join(path,f'{mode}{np.array(prefile).shape[0]+index}_0.png'),i)
#     return len(s_y)

# new_train_length = data_smote(pre_file_train,mode="train")
# new_test_length = data_smote(pre_file_test,mode="test")
# new_val_length = data_smote(pre_file_val,mode="val")
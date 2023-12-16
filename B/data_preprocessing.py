# -*- encoding: utf-8 -*-
'''
@File    :   data_preprocessing.py
@Time    :   2023/12/16 22:28:04
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
# to download dataset in npz format from MeMNIST website, can use the sentence below
# dataset2 = PathMNIST(split='train',download=True,root="Datasets/")

# how to save the dataset as png figure and csv (reference from MeMNIST)
# python -m medmnist save --flag=pathmnist --postfix=png --folder=Datasets/ --root=Datasets/

'''
description: This function is used for histogram equalization and comparison of CLAHE method.
param {*} path: path of raw dataset
param {*} f: filename
return {*}: original image, image after histogram equalization, image after CLAHE
'''
def histogram_equalization(path,f):
    img = cv2.imread(os.path.join(path,f))
    equ = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # convert into HSV channels for histogram equalization
    equ[:, :, 2] = cv2.equalizeHist(equ[:, :, 2])  # only do equalization for channel V for contrastness
    equ = cv2.cvtColor(equ, cv2.COLOR_HSV2RGB)

    cl = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl[:, :, 2] = clahe.apply(cl[:, :, 2])
    cl = cv2.cvtColor(cl, cv2.COLOR_HSV2RGB)

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
    imgShift = cv2.warpAffine(img,H,(w,h)) #需要图像、变换矩阵、变换后的大小
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
description:  This function is used for data augmentation with five techniques.
param {*} data: npz data
param {*} path: preprocessed data path
param {*} pre_file: all preprocessed files
param {*} clength: size of train, validation or test dataset
param {*} label: specific label of 9 class
param {*} mode: train, validation or test
return {*}: new length of the whole dataset after data augmentation
'''
def data_augmentation(data, path, pre_file,clength, label, mode=None):

    pre_file_phase = []

    # specific dataset with specific label
    for index,f in enumerate(pre_file):
        if not os.path.isfile(os.path.join(path,f)):
            continue
        else:
            if mode in f and (f.split("_")[1][0] == label):
                    pre_file_phase.append(f)
    
    count = []
    for i in range(9):
        count.append(np.count_nonzero(data[f'{mode}_labels'].flatten() == i))
    length_0 = count[int(label)]  # number of samples for current class
    length_1 = max(count) # class with most samples in 9 types

    
    imgs = [] # augmented images
    # sampling from minority class
    flip_index = random.sample([i for i in range(length_0)],int((length_1-length_0)/5))
    for i in flip_index:
        img = cv2.imread(os.path.join(path,f'{pre_file_phase[i]}'))
        imgFlip = horizontalFlip(img)
        imgs.append(imgFlip)

    shear_index = random.sample([i for i in range(length_0)],int((length_1-length_0)/5))
    for i in shear_index:
        img = cv2.imread(os.path.join(path,f'{pre_file_phase[i]}'))
        imgShear = shear(img)
        imgs.append(imgShear)
    
    zoom_index = random.sample([i for i in range(length_0)],int((length_1-length_0)/5))
    for i in zoom_index:
        img = cv2.imread(os.path.join(path,f'{pre_file_phase[i]}'))
        imgZoom = zoom(img)
        imgs.append(imgZoom)
    
    shift_index = random.sample([i for i in range(length_0)],int((length_1-length_0)/5))
    for i in shift_index:
        img = cv2.imread(os.path.join(path,f'{pre_file_phase[i]}'))
        imgShift = shift(img)
        imgs.append(imgShift)
    
    rot_index = random.sample([i for i in range(length_0)],int((length_1-length_0)-len(imgs)))  # all left
    for i in rot_index:
        img = cv2.imread(os.path.join(path,f'{pre_file_phase[i]}'))
        imgRot = rotation(img)
        imgs.append(imgRot)
    
    for index,i in enumerate(imgs):
        cv2.imwrite(os.path.join(path,f'{mode}{clength+index}_{label}.png'),i)

    return clength+len(imgs)


'''
description: This function is used for showing basic data descriptions,
        like number of labels of each class, number of train/validation/test images.
param {*} npz: whether download npz data, if you don't use provided backup project and want to check the project
        from original dataset, please use the argument here. Detailed guideline is shown in README.md and Github link.
return {*}: npz data
'''
def load_data_log4B(npz):
    if npz == True:
        download_dataset = PathMNIST(split='train',download=True,root="Datasets/")
    data = np.load('Datasets/pathmnist.npz')

    train_label = {f"label {i}":np.count_nonzero(data['train_labels'].flatten() == i) for i in range(9)}
    val_label = {f"label {i}":np.count_nonzero(data['val_labels'].flatten() == i) for i in range(9)}
    test_label = {f"label {i}":np.count_nonzero(data['test_labels'].flatten() == i) for i in range(9)}

    print(f"Train data length: {len(data['train_images'])}")
    print(train_label)
    print(f"Validation data length: {len(data['val_images'])}")
    print(val_label)
    print(f"Test data length: {len(data['test_images'])}")
    print(test_label)
    visual4label("B",data)  # label distribution

    return data

'''
description: This function is a conclusion for all data preprocessing procedures.
param {*} raw_path: raw dataset path
return {*}: size of train/validation/test dataset after all data preprocessing procedures.
'''
def data_preprocess4B(raw_path):
    print("Start preprocessing data......")
    data = load_data_log4B()
    raw_file=os.listdir(raw_path)

    # data preprocessing
    os.makedirs('Outputs/pathmnist/preprocessed_data', exist_ok=True)
    for index,f in enumerate(raw_file):
        if not os.path.isfile(os.path.join(raw_path,f)):
            continue
        else:
            img, equ, cl = histogram_equalization(raw_path, f)
            imgPas = sobel(equ)
            imgGamma = gammaCorrection(imgPas)
            cv2.imwrite(os.path.join('Outputs/pathmnist/preprocessed_data',f'{f}'),imgGamma)

    # data augmentation
    pre_path = 'Outputs/pathmnist/preprocessed_data'
    pre_file=os.listdir(pre_path)
    new_train_length = len(data["train_images"])
    new_test_length = len(data["test_images"])
    new_val_length = len(data["val_images"])
    
    for i in range(9):  # for each class type
        new_train_length = data_augmentation(data, pre_path, pre_file,new_train_length,str(i),mode="train")
        new_test_length = data_augmentation(data, pre_path, pre_file,new_test_length,str(i), mode="test")
        new_val_length = data_augmentation(data, pre_path, pre_file,new_val_length, str(i), mode="val")

    print("Finish preprocessing data.")
    return new_train_length, new_test_length, new_val_length


"""
    Experiments for comparison of histogram equalization with single/double channel, CLAHE and RGB/HSV format.
    This part is for experiment and will not be included in the committed code
"""
# experiment 1: histogram equalization
# this part is for experiment and will not be included in the committed code
# in this part, I compare the effect of building histogram equalization on single channel and multiple channel
# finally I choose the single channel for brightness
# notice that I convert the image into HSV since the result of simple histogram equalization for RGB will change the color to a great extent
# consideirng the balance of color and as close to the original figure as possible
# only equalize one single channel of 2
# clahe or equ need to see subsequent processing

# img = cv2.imread('Datasets/pathmnist/test0_8.png')

# # comparison
# # equ without convert into HSV
# equ1 = cv2.imread('Datasets/pathmnist/test0_8.png')
# equ1[:, :, 0] = cv2.equalizeHist(equ1[:, :, 0])
# equ1[:, :, 1] = cv2.equalizeHist(equ1[:, :, 1])
# equ1[:, :, 2] = cv2.equalizeHist(equ1[:, :, 2])

# # CLAHE without convert into HSV
# cl1 = cv2.imread('Datasets/pathmnist/test0_8.png')
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl1[:, :, 0] = clahe.apply(cl1[:, :, 0])
# cl1[:, :, 1] = clahe.apply(cl1[:, :, 1])
# cl1[:, :, 2] = clahe.apply(cl1[:, :, 2])

# # equ convert into HSV and equalize two channels
# equ2 = cv2.imread('Datasets/pathmnist/test0_8.png')
# equ2 = cv2.cvtColor(equ2, cv2.COLOR_RGB2HSV)
# equ2[:, :, 1] = cv2.equalizeHist(equ2[:, :, 1])
# equ2[:, :, 2] = cv2.equalizeHist(equ2[:, :, 2])
# equ2 = cv2.cvtColor(equ2, cv2.COLOR_HSV2RGB)

# # CLAHE convert into HSV and equalize two channels
# cl2 = cv2.imread('Datasets/pathmnist/test0_8.png')
# cl2 = cv2.cvtColor(cl2, cv2.COLOR_RGB2HSV)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl2[:, :, 1] = clahe.apply(cl2[:, :, 1])
# cl2[:, :, 2] = clahe.apply(cl2[:, :, 2])
# cl2 = cv2.cvtColor(cl2, cv2.COLOR_HSV2RGB)

# # equ convert into HSV and equalize one channel (select)
# equ3 = cv2.imread('Datasets/pathmnist/test0_8.png')
# equ3 = cv2.cvtColor(equ3, cv2.COLOR_RGB2HSV)
# equ3[:, :, 2] = cv2.equalizeHist(equ3[:, :, 2])
# equ3 = cv2.cvtColor(equ3, cv2.COLOR_HSV2RGB)

# # CLAHE convert into HSV and equalize one channel 
# cl3 = cv2.imread('Datasets/pathmnist/test0_8.png')
# cl3 = cv2.cvtColor(cl3, cv2.COLOR_RGB2HSV)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl3[:, :, 2] = clahe.apply(cl3[:, :, 2])
# cl3 = cv2.cvtColor(cl3, cv2.COLOR_HSV2RGB)

# res = np.hstack((img,equ1,cl1,equ2,cl2, equ3, cl3))  # stacking images side-by-side
# cv2.imwrite(os.path.join('Outputs/','res.png'),res)
# cv2.imshow("result",res)
# cv2.waitKey(0)


# chans = cv2.split(img)   
# colors = ("b", "g", "r")
# plt.subplot(1,2,1)
# plt.title('calcHist before equalization')
# plt.xlabel("Bins")
# plt.ylabel("Pixels Num")

# for (chan, color) in zip(chans, colors):
#     hist = cv2.calcHist([chan], [0], None, [256], [0,256])
#     print(color)
#     plt.plot(hist, color=color)
#     plt.xlim([0, 256])

# # the selected one
# chans = cv2.split(equ3)   
# colors = ("b", "g", "r")
# plt.subplot(1,2,2)
# plt.title('calcHist after equalization')
# plt.xlabel("Bins")
# plt.ylabel("Pixels Num")

# for (chan, color) in zip(chans, colors):
#     hist = cv2.calcHist([chan], [0], None, [256], [0,256])
#     print(color)
#     plt.plot(hist, color=color)
#     plt.xlim([0, 256])

# plt.show()
# plt.savefig('Outputs/histogram_example_task B.png')

"""
    Experiments for comparison of CLAHE, Laplacian, Sobel operation, Gamma correction, etc.
    This part is for experiment and will not be included in the committed code
"""
# experiment 2: selection for sobel/lapacian/gamma correction
# result：sobel+gamma
# laplacian may lead to lots of noise
# clahe lost some information of dark region to separate apart
# laplacian
# img, equ, cl = histogram_equalization("test0_8.png")
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

# # laplacian
# kernel_cl = np.array([[0,1,0],[1,-4,1],[0,1,0]],np.int8)
# laplacian_cl = cv2.filter2D(cl, ddepth=-1, kernel=kernel)
# imglaplacian_cl = np.uint8(cv2.normalize(laplacian_cl, None, 0, 255, cv2.NORM_MINMAX))
# addlap_cl = cl+imglaplacian_cl
# addlap_cl = np.uint8(cv2.normalize(addlap_cl, None, 0, 255, cv2.NORM_MINMAX))

# # sobel
# SobelX_cl = cv2.Sobel(cl, cv2.CV_16S, 1, 0)  # 计算 x 轴方向
# SobelY_cl= cv2.Sobel(cl, cv2.CV_16S, 0, 1)  # 计算 y 轴方向
# absX_cl = cv2.convertScaleAbs(SobelX_cl)  # 转回 uint8
# absY_cl = cv2.convertScaleAbs(SobelY_cl)  # 转回 uint8
# SobelXY_cl = cv2.addWeighted(absX_cl, 0.5, absY_cl, 0.5, 0)  # 用绝对值近似平方根
# imgSobel_cl = np.uint8(cv2.normalize(SobelXY_cl, None, 0, 255, cv2.NORM_MINMAX))

# passivation1_cl = imgSobel*0.3 + cl  # sobel add on image after histogram equalization
# passivation2_cl = imgSobel*0.3 + addlap_cl  # sobel add on image after laplacian
# imgPassi1_cl = np.uint8(cv2.normalize(passivation1_cl, None, 0, 255, cv2.NORM_MINMAX))
# imgPassi2_cl = np.uint8(cv2.normalize(passivation2_cl, None, 0, 255, cv2.NORM_MINMAX))

# # gamma
# epsilon = 1e-5  
# Gamma1_cl = np.power(imgPassi1_cl + epsilon, 0.5)
# Gamma2_cl = np.power(imgPassi2_cl + epsilon, 0.5)
# imgGamma1_cl = np.uint8(cv2.normalize(Gamma1_cl, None, 0, 255, cv2.NORM_MINMAX))
# imgGamma2_cl = np.uint8(cv2.normalize(Gamma2_cl, None, 0, 255, cv2.NORM_MINMAX))

# res = np.hstack((img,equ,cl,addlap,imgPassi1,imgPassi2,imgGamma1,imgGamma2, addlap_cl,imgPassi1_cl,imgPassi2_cl,imgGamma1_cl,imgGamma2_cl))  # stacking images side-by-side
# cv2.imwrite(os.path.join('Outputs/','res.png'),res)
# cv2.imshow("result",res)
# cv2.waitKey(0)

"""
    This part is the implementation of SMOTE introduced in report. 
    Since it's not used in the final processing procedures and just trials for comparison, 
    it's commented and not included in committed code.
"""
# # smote
# path = 'Datasets/pathmnist'
# raw_file=os.listdir(path)
# os.makedirs('Outputs/pathmnist/smote_data', exist_ok=True)
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
#             cv2.imwrite(os.path.join('Outputs/pathmnist/smote_data',f'{f}'),imgGamma)


# path = 'Outputs/pathmnist/smote_data'
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
# print(len(pre_file_train))

# # notice that the order of each dataset in npz and images is same
# # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# # to sample the data, we can either use data augmentation or SMOTE algortihm, here try SMOTE as well to compare the performance

# def data_smote(prefile, mode=None):
#     smote=SMOTE(random_state=0)
#     print(np.array(prefile).reshape(np.array(prefile).shape[0], 28*28*3).shape)
#     print(data[f'{mode}_labels'].ravel().shape)
#     s_x, s_y=smote.fit_resample(np.array(prefile).reshape(np.array(prefile).shape[0], 28*28*3), data[f'{mode}_labels'].ravel())
#     s_x = s_x.reshape(s_x.shape[0], 28, 28, 3)
#     # experiments show that the smote data was appended after original data
#     count = []
#     for i in range(9):
#         count.append(np.count_nonzero(data[f'{mode}_labels'].flatten() == i))
#     pos = np.array(prefile).shape[0]
#     print(pos)
#     for label in range(9):
#         print(pos)
#         for index,i in enumerate(s_x[pos:pos+(max(count)-count[int(label)]),:,:,:]):
#             cv2.imwrite(os.path.join(path,f'{mode}{pos+index}_{label}.png'),i)
#         pos = pos+(max(count)-count[int(label)])

#     return len(s_y)

# new_train_length = data_smote(pre_file_train,mode="train")
# new_test_length = data_smote(pre_file_test,mode="test")
# new_val_length = data_smote(pre_file_val,mode="val")

# I don't think smote is a good method to balance the dataset, because the result of smote
# includes figures with lots of noise, it's not so clear also, it also contains some strange color pixels

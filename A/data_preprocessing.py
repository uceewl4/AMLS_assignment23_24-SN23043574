from medmnist import PneumoniaMNIST
from medmnist import PathMNIST
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
from imblearn.over_sampling import SMOTE

# to download dataset in npz format from MeMNIST website, can use the sentence below
# dataset1 = PneumoniaMNIST(split='train',download=True,root="Datasets/")

# how to save the dataset as png figure and csv (reference from MeMNIST)
# python -m medmnist save --flag=pneumoniamnist --postfix=png --folder=Datasets/ --root=Datasets/
# python -m medmnist save --flag=pathmnist --postfix=png --folder=Datasets/ --root=Datasets/
# dataset2 = PathMNIST(split='train',download=True,root="Datasets/")

# load the npz dataset and read it through numpy
data = np.load('/Users/anlly/Desktop/ucl/Applied Machine Learning Systems-I/AMLS assignment/AMLS_assignment23_24-SN23043574/Datasets/pneumoniamnist.npz')
print(f"Train data length: {len(data['train_images'])}, label 0: {np.count_nonzero(data['train_labels'].flatten() == 0)}, label 1: {np.count_nonzero(data['train_labels'].flatten() == 1)}")
print(f"Validation data length: {len(data['val_images'])}, label 0: {np.count_nonzero(data['val_labels'].flatten() == 0)}, label 1: {np.count_nonzero(data['val_labels'].flatten() == 1)}")                                                               
print(f"Test data length: {len(data['test_images'])}, label 0: {np.count_nonzero(data['test_labels'].flatten() == 0)}, label 1: {np.count_nonzero(data['test_labels'].flatten() == 1)}")
# # print(data['train_images'][0:2,:,:])
# # print(data['train_images'][0,:,:].shape)  # 28x28


# histogram equalization
# there are two ways for histogram equalization, one is use traditional, second is use CLAHE where CLAHE can help avoid 
# the loss of information due to over-brightness after histogram equalization, which is an adaptive histogram equalization.
path = 'Datasets/pneumoniamnist'
raw_file=os.listdir(path)
os.makedirs('Outputs/pneumoniamnist/preprocessed_data', exist_ok=True)
# print(file_list)   

def histogram_equalization(f):
    
    img = cv2.imread(os.path.join(path,f),0)
    equ = cv2.equalizeHist(img)

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(img)

    return img, equ, cl

def sobel(imgEqu):
    SobelX = cv2.Sobel(imgEqu, cv2.CV_16S, 1, 0)  # 计算 x 轴方向
    SobelY = cv2.Sobel(imgEqu, cv2.CV_16S, 0, 1)  # 计算 y 轴方向
    absX = cv2.convertScaleAbs(SobelX)  # 转回 uint8
    absY = cv2.convertScaleAbs(SobelY)  # 转回 uint8
    SobelXY = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)  # 用绝对值近似平方根
    imgSobel = np.uint8(cv2.normalize(SobelXY, None, 0, 255, cv2.NORM_MINMAX))

    passivation = imgSobel*0.3 + imgEqu  # sobel add on image after histogram equalization
    imgPas = np.uint8(cv2.normalize(passivation, None, 0, 255, cv2.NORM_MINMAX))
    return imgPas

def gammaCorrection(imgPas):
    epsilon = 1e-5  
    gamma = np.power(imgPas + epsilon, 0.5)
    imgGamma = np.uint8(cv2.normalize(gamma, None, 0, 255, cv2.NORM_MINMAX))
    return imgGamma

for index,f in enumerate(raw_file):
        # print(os.path.join(path,f))
        if not os.path.isfile(os.path.join(path,f)):
            continue
        else:
            img, equ, cl = histogram_equalization(f)
            imgPas = sobel(equ)
            imgGamma = gammaCorrection(imgPas)
            # res = np.hstack((img,equ,cl,imgPas,imgGamma))  # stacking images side-by-side
            cv2.imwrite(os.path.join('Outputs/pneumoniamnist/preprocessed_data',f'{f}'),imgGamma)


# here I compare the performance for with or without each step of laplacian, sobel, gamma
# the most suitable one is sobel + gamma

# # this part is for experiment and will not be included in the committed code
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

# rotation
def rotation(img):
    h, w, c = img.shape
    M = cv2.getRotationMatrix2D((w/2,h/2),45,1) # 第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
    imgRot = cv2.warpAffine(img,M,(w,h))  # 第三个参数：变换后的图像大小
    return imgRot

# width shift & height shift  1/5
def shift(img):
    h, w, c = img.shape
    H = np.float32([[1,0,5],
                    [0,1,5]])
    imgShift = cv2.warpAffine(img,H,(w,h)) #需要图像、变换矩阵、变换后的大小
    return imgShift

# shear
def shear(img):
    h, w, c = img.shape

    pts1 = np.float32([[0, 0],[0, h-1],[w-1, 0]])
    pts2 = np.float32([[0, 0],[5, h-5],[w-5, 5]])
    M = cv2.getAffineTransform(pts1,pts2)
    #第三个参数：变换后的图像大小
    imgShear = cv2.warpAffine(img,M,(w,h))
    return imgShear

# zoom
def zoom(img):
    # img = cv2.imread('Outputs/preprocessed_data/test0_1.png')
    h, w, c = img.shape
    imgZoom = cv2.resize(img,(int(0.5*w),int(0.5*h)),interpolation=cv2.INTER_CUBIC)
    keepSize = np.zeros((h,w,c),np.uint8)
    keepSize[:int(0.5*h), :int(0.5*w), :] = imgZoom
    return keepSize

# horizontal flip
def horizontalFlip(img):
    # img = cv2.imread('Outputs/preprocessed_data/test0_1.png')
    imgFlip = cv2.flip(img, 1) # 水平翻转
    return imgFlip


path = 'Outputs/pneumoniamnist/preprocessed_data'
pre_file=os.listdir(path)
pre_file_test_0 = []
pre_file_train_0 = []
pre_file_val_0 = []

for index,f in enumerate(pre_file):
        # print(os.path.join(path,f))
        if not os.path.isfile(os.path.join(path,f)):
            continue
        else:
            if (f.split("_")[1][0] == "0"):
                if "test" in f:
                    pre_file_test_0.append(f)
                elif "train" in f:
                    pre_file_train_0.append(f)
                elif "val" in f:
                    pre_file_val_0.append(f)

def data_augmentation(length_1, length_0, pre_file_0, mode=None):
    imgs = []

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


new_train_length = data_augmentation(len(data['train_images'])-len(pre_file_train_0),len(pre_file_train_0), pre_file_train_0,mode="train")
new_test_length = data_augmentation(len(data['test_images'])-len(pre_file_test_0),len(pre_file_test_0), pre_file_test_0,mode="test")
new_val_length = data_augmentation(len(data['val_images'])-len(pre_file_val_0),len(pre_file_val_0), pre_file_val_0,mode="val")

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

# notice that the order of each dataset in npz and images is same
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# to sample the data, we can either use data augmentation or SMOTE algortihm, here try SMOTE as well to compare the performance

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

# # I don't think smote is a good method to balance the dataset, because the result of smote
# # includes figures with lots of noise, it's not so clear also

# for the shape of figures:
# order is same in npz and images
# shape of npz: 28x28x1
# shape of images both before and after data preprocessing: 28x28x3
# model may need to convert the images into gray, depends after
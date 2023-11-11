from medmnist import PneumoniaMNIST
from medmnist import PathMNIST
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

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
# print(data['train_images'][0:2,:,:])
# print(data['train_images'][0,:,:].shape)  # 28x28


# histogram equalization
# there are two ways for histogram equalization, one is use traditional, second is use CLAHE where CLAHE can help avoid 
# the loss of information due to over-brightness after histogram equalization, which is an adaptive histogram equalization.
path = 'Datasets/pneumoniamnist'
file_list=os.listdir(path)
os.makedirs('Outputs/preprocessed_data', exist_ok=True)
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

for index,f in enumerate(file_list):
        # print(os.path.join(path,f))
        if not os.path.isfile(os.path.join(path,f)):
            continue
        else:
            img, equ, cl = histogram_equalization(f)
            imgPas = sobel(equ)
            imgGamma = gammaCorrection(imgPas)
            # res = np.hstack((img,equ,cl,imgPas,imgGamma))  # stacking images side-by-side
            cv2.imwrite(os.path.join('Outputs/preprocessed_data',f'{f}'),imgGamma)



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
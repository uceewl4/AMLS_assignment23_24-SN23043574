from medmnist import PneumoniaMNIST
from medmnist import PathMNIST
import numpy as np
import matplotlib.pyplot as plt

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
print(data['train_images'][0:2,:,:])
print(data['train_images'][0,:,:].shape)  # 28x28

# for i in data['train_images']:
#     plt.plot(i)
#     plt.show()



# histogram equalization
import numpy as np
import cv2

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# img = cv2.imread('Datasets/pathmnist/test0_8.png',0)
# equ = cv2.equalizeHist(img)
# equ = cv2.cvtColor(equ, cv2.COLOR_YUV2BGR)
# print(equ)
# # res = np.hstack((img,equ)) #stacking images side-by-side
# # os.makedirs('Outputs/', exist_ok=True)
# # cv2.imwrite(os.path.join('Outputs/','res.png'),res)

# img2 = cv2.imread('Datasets/pathmnist/test0_8.png',0)

# # create a CLAHE object (Arguments are optional).
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl1 = clahe.apply(img2)
# cl1 = cv2.cvtColor(cl1, cv2.COLOR_YUV2BGR)
# res = np.hstack((img,cl1,equ))#stacking images side-by-side
# os.makedirs('Outputs/', exist_ok=True)
# cv2.imwrite(os.path.join('Outputs/','res.png'),res)


# img = cv2.imread('Datasets/pathmnist/test0_8.png',0)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
# img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
# # equ = cv2.equalizeHist(img)
# equ = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
# print(equ)
# # res = np.hstack((img,equ)) #stacking images side-by-side
# # os.makedirs('Outputs/', exist_ok=True)
# # cv2.imwrite(os.path.join('Outputs/','res.png'),res)

# img2 = cv2.imread('Datasets/pathmnist/test0_8.png',0)
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)
# # create a CLAHE object (Arguments are optional).
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl1 = clahe.apply(img2)
# equalized_img = cv2.cvtColor(cl1, cv2.COLOR_YCrCb2BGR)
# res = np.hstack((img,cl1,equ))#stacking images side-by-side

# os.makedirs('Outputs/', exist_ok=True)
# cv2.imwrite(os.path.join('Outputs/','res.png'),res)


# equalize the histogram of the Y channel


# convert back to RGB color-space from YCrCb


image = cv2.imread('Datasets/pathmnist/test0_8.png')

# convert image from RGB to HSV
img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_hsv[:, :, 2] = clahe.apply(img_hsv[:, :, 2])
# Histogram equalisation on the V-channel
# img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])

# convert image back from HSV to RGB
image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

cv2.imshow("equalizeHist", image)
cv2.waitKey(0)


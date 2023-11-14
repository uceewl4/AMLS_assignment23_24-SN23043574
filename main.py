import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from A.models.baselines import Baselines as A_baselines
from B.models.baselines import Baselines as B_baselines
from utils import visual4cm
# # data = np.load('/Users/anlly/Desktop/ucl/Applied Machine Learning Systems-I/AMLS assignment/AMLS_assignment23_24-SN23043574/Datasets/pneumoniamnist.npz')
# # print(f"Train data length: {len(data['train_images'])}, label 0: {np.count_nonzero(data['train_labels'].flatten() == 0)}, label 1: {np.count_nonzero(data['train_labels'].flatten() == 1)}")
# # print(f"Validation data length: {len(data['val_images'])}, label 0: {np.count_nonzero(data['val_labels'].flatten() == 0)}, label 1: {np.count_nonzero(data['val_labels'].flatten() == 1)}")                                                               
# # print(f"Test data length: {len(data['test_images'])}, label 0: {np.count_nonzero(data['test_labels'].flatten() == 0)}, label 1: {np.count_nonzero(data['test_labels'].flatten() == 1)}")
# # Generally the ratio of train:val:test should be 3:1:1, here first use the dataset and no need for train test split


# visualization, result are all separate from model

# path = 'Outputs/pneumoniamnist/preprocessed_data'
# file=os.listdir(path)
# Xtest = []
# ytest = []
# Xtrain = []
# ytrain = []
# Xval = []
# yval = []

# for index,f in enumerate(file):
#         if not os.path.isfile(os.path.join(path,f)):
#             continue
#         else:
#             img = cv2.imread(os.path.join(path,f))
#             imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#             if "test" in f:
#                 Xtest.append(imgGray)
#                 ytest.append(f.split("_")[1][0])
#             elif "train" in f:
#                 Xtrain.append(imgGray)
#                 ytrain.append(f.split("_")[1][0])
#             elif "val" in f:
#                 Xval.append(imgGray)
#                 yval.append(f.split("_")[1][0])

# n,h,w = np.array(Xtrain).shape
# Xtrain = np.array(Xtrain).reshape(n,h*w) # need to reshape gray picture into two-dimensional ones
# Xval = np.array(Xval).reshape(len(Xval),h*w)
# Xtest = np.array(Xtest).reshape(len(Xtest),h*w)

# # logistic regression
# # LR = A_baselines("LR")  # lbfgs will have warning about max iteration limitation
# # LR.train(Xtrain, ytrain, Xval, yval, gridSearch=False)
# # pred_train, pred_val, pred_test = LR.test(Xtrain, ytrain, Xval, yval, Xtest)
# # acc = accuracy_score(np.array(ytest).astype(int), pred_test.astype(int))
# # pre = precision_score(np.array(ytest).astype(int), pred_test.astype(int))
# # recall = recall_score(np.array(ytest).astype(int), pred_test.astype(int))
# # f1 = f1_score(np.array(ytest).astype(int), pred_test.astype(int))
# # print("acc: {%.4f}, pre: {%.4f}, rec: {%.4f}, f1: {%.4f}" % ((acc, pre, recall, f1)))

# # knn = A_baselines("KNN")  
# # res = knn.train(Xtrain, ytrain, Xval, yval, gridSearch=False)
# # pred_train, pred_val, pred_test = knn.test(Xtrain, ytrain, Xval, yval, Xtest)
# # acc = accuracy_score(np.array(ytest).astype(int), pred_test.astype(int))
# # pre = precision_score(np.array(ytest).astype(int), pred_test.astype(int))
# # recall = recall_score(np.array(ytest).astype(int), pred_test.astype(int))
# # f1 = f1_score(np.array(ytest).astype(int), pred_test.astype(int))
# # print("acc: {%.4f}, pre: {%.4f}, rec: {%.4f}, f1: {%.4f}" % ((acc, pre, recall, f1)))

# # svm = A_baselines("SVM")  
# # svm.train(Xtrain, ytrain, Xval, yval, gridSearch=False)
# # pred_train, pred_val, pred_test = svm.test(Xtrain, ytrain, Xval, yval, Xtest)
# # acc = accuracy_score(np.array(ytest).astype(int), pred_test.astype(int))
# # pre = precision_score(np.array(ytest).astype(int), pred_test.astype(int))
# # recall = recall_score(np.array(ytest).astype(int), pred_test.astype(int))
# # f1 = f1_score(np.array(ytest).astype(int), pred_test.astype(int))
# # print("acc: {%.4f}, pre: {%.4f}, rec: {%.4f}, f1: {%.4f}" % ((acc, pre, recall, f1)))

# dt = A_baselines("DT")  
# dt.train(Xtrain, ytrain, Xval, yval, gridSearch=False)
# pred_train, pred_val, pred_test = dt.test(Xtrain, ytrain, Xval, yval, Xtest)
# acc = accuracy_score(np.array(ytest).astype(int), pred_test.astype(int))
# pre = precision_score(np.array(ytest).astype(int), pred_test.astype(int))
# recall = recall_score(np.array(ytest).astype(int), pred_test.astype(int))
# f1 = f1_score(np.array(ytest).astype(int), pred_test.astype(int))
# print("acc: {%.4f}, pre: {%.4f}, rec: {%.4f}, f1: {%.4f}" % ((acc, pre, recall, f1)))
# visual4cm(np.array(ytest).astype(int), pred_test.astype(int))

# # nb = A_baselines("NB")  
# # nb.train(Xtrain, ytrain, Xval, yval, gridSearch=False)
# # pred_train, pred_val, pred_test = nb.test(Xtrain, ytrain, Xval, yval, Xtest)
# # acc = accuracy_score(np.array(ytest).astype(int), pred_test.astype(int))
# # pre = precision_score(np.array(ytest).astype(int), pred_test.astype(int))
# # recall = recall_score(np.array(ytest).astype(int), pred_test.astype(int))
# # f1 = f1_score(np.array(ytest).astype(int), pred_test.astype(int))
# # print("acc: {%.4f}, pre: {%.4f}, rec: {%.4f}, f1: {%.4f}" % ((acc, pre, recall, f1)))

# # rf = A_baselines("RF")  
# # rf.train(Xtrain, ytrain, Xval, yval, gridSearch=False)
# # pred_train, pred_val, pred_test = rf.test(Xtrain, ytrain, Xval, yval, Xtest)
# # acc = accuracy_score(np.array(ytest).astype(int), pred_test.astype(int))
# # pre = precision_score(np.array(ytest).astype(int), pred_test.astype(int))
# # recall = recall_score(np.array(ytest).astype(int), pred_test.astype(int))
# # f1 = f1_score(np.array(ytest).astype(int), pred_test.astype(int))
# # print("acc: {%.4f}, pre: {%.4f}, rec: {%.4f}, f1: {%.4f}" % ((acc, pre, recall, f1)))

# # abc = A_baselines("ABC")  
# # abc.train(Xtrain, ytrain, Xval, yval, gridSearch=False)
# # pred_train, pred_val, pred_test = abc.test(Xtrain, ytrain, Xval, yval, Xtest)
# # acc = accuracy_score(np.array(ytest).astype(int), pred_test.astype(int))
# # pre = precision_score(np.array(ytest).astype(int), pred_test.astype(int))
# # recall = recall_score(np.array(ytest).astype(int), pred_test.astype(int))
# # f1 = f1_score(np.array(ytest).astype(int), pred_test.astype(int))
# # print("acc: {%.4f}, pre: {%.4f}, rec: {%.4f}, f1: {%.4f}" % ((acc, pre, recall, f1)))

path = 'Outputs/pathmnist/preprocessed_data'
file=os.listdir(path)
Xtest = []
ytest = []
Xtrain = []
ytrain = []
Xval = []
yval = []

for index,f in enumerate(file):
        if not os.path.isfile(os.path.join(path,f)):
            continue
        else:
            img = cv2.imread(os.path.join(path,f))
            if "test" in f:
                Xtest.append(img)
                ytest.append(f.split("_")[1][0])
            elif "train" in f:
                Xtrain.append(img)
                ytrain.append(f.split("_")[1][0])
            elif "val" in f:
                Xval.append(img)
                yval.append(f.split("_")[1][0])

n,h,w,c = np.array(Xtrain).shape
Xtrain = np.array(Xtrain).reshape(n,h*w*c) # need to reshape gray picture into two-dimensional ones
Xval = np.array(Xval).reshape(len(Xval),h*w*c)
Xtest = np.array(Xtest).reshape(len(Xtest),h*w*c)

# logistic regression


knn = B_baselines("KNN")  
res = knn.train(Xtrain, ytrain, Xval, yval, gridSearch=False)
pred_train, pred_val, pred_test = knn.test(Xtrain, ytrain, Xval, yval, Xtest)
acc = accuracy_score(np.array(ytest).astype(int), pred_test.astype(int))
pre = precision_score(np.array(ytest).astype(int), pred_test.astype(int))
recall = recall_score(np.array(ytest).astype(int), pred_test.astype(int))
f1 = f1_score(np.array(ytest).astype(int), pred_test.astype(int))
print("acc: {%.4f}, pre: {%.4f}, rec: {%.4f}, f1: {%.4f}" % ((acc, pre, recall, f1)))

# svm = A_baselines("SVM")  
# svm.train(Xtrain, ytrain, Xval, yval, gridSearch=False)
# pred_train, pred_val, pred_test = svm.test(Xtrain, ytrain, Xval, yval, Xtest)
# acc = accuracy_score(np.array(ytest).astype(int), pred_test.astype(int))
# pre = precision_score(np.array(ytest).astype(int), pred_test.astype(int))
# recall = recall_score(np.array(ytest).astype(int), pred_test.astype(int))
# f1 = f1_score(np.array(ytest).astype(int), pred_test.astype(int))
# print("acc: {%.4f}, pre: {%.4f}, rec: {%.4f}, f1: {%.4f}" % ((acc, pre, recall, f1)))

# dt = A_baselines("DT")  
# dt.train(Xtrain, ytrain, Xval, yval, gridSearch=False)
# pred_train, pred_val, pred_test = dt.test(Xtrain, ytrain, Xval, yval, Xtest)
# acc = accuracy_score(np.array(ytest).astype(int), pred_test.astype(int))
# pre = precision_score(np.array(ytest).astype(int), pred_test.astype(int))
# recall = recall_score(np.array(ytest).astype(int), pred_test.astype(int))
# f1 = f1_score(np.array(ytest).astype(int), pred_test.astype(int))
# print("acc: {%.4f}, pre: {%.4f}, rec: {%.4f}, f1: {%.4f}" % ((acc, pre, recall, f1)))
# visual4cm(np.array(ytest).astype(int), pred_test.astype(int))

# nb = A_baselines("NB")  
# nb.train(Xtrain, ytrain, Xval, yval, gridSearch=False)
# pred_train, pred_val, pred_test = nb.test(Xtrain, ytrain, Xval, yval, Xtest)
# acc = accuracy_score(np.array(ytest).astype(int), pred_test.astype(int))
# pre = precision_score(np.array(ytest).astype(int), pred_test.astype(int))
# recall = recall_score(np.array(ytest).astype(int), pred_test.astype(int))
# f1 = f1_score(np.array(ytest).astype(int), pred_test.astype(int))
# print("acc: {%.4f}, pre: {%.4f}, rec: {%.4f}, f1: {%.4f}" % ((acc, pre, recall, f1)))

# rf = A_baselines("RF")  
# rf.train(Xtrain, ytrain, Xval, yval, gridSearch=False)
# pred_train, pred_val, pred_test = rf.test(Xtrain, ytrain, Xval, yval, Xtest)
# acc = accuracy_score(np.array(ytest).astype(int), pred_test.astype(int))
# pre = precision_score(np.array(ytest).astype(int), pred_test.astype(int))
# recall = recall_score(np.array(ytest).astype(int), pred_test.astype(int))
# f1 = f1_score(np.array(ytest).astype(int), pred_test.astype(int))
# print("acc: {%.4f}, pre: {%.4f}, rec: {%.4f}, f1: {%.4f}" % ((acc, pre, recall, f1)))

# abc = A_baselines("ABC")  
# abc.train(Xtrain, ytrain, Xval, yval, gridSearch=False)
# pred_train, pred_val, pred_test = abc.test(Xtrain, ytrain, Xval, yval, Xtest)
# acc = accuracy_score(np.array(ytest).astype(int), pred_test.astype(int))
# pre = precision_score(np.array(ytest).astype(int), pred_test.astype(int))
# recall = recall_score(np.array(ytest).astype(int), pred_test.astype(int))
# f1 = f1_score(np.array(ytest).astype(int), pred_test.astype(int))
# print("acc: {%.4f}, pre: {%.4f}, rec: {%.4f}, f1: {%.4f}" % ((acc, pre, recall, f1)))







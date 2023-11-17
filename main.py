import os
import cv2
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from A.models.baselines import Baselines as A_Baselines
from A.models.CNN import CNN as A_CNN
from A.models.DenseNet201 import DenseNet201 as A_DenseNet201
from A.models.InceptionV3 import InceptionV3 as A_InceptionV3
from A.models.MLP import MLP as A_MLP
from A.models.MobileNetV2 import MobileNetV2 as A_MobileNetV2
from A.models.ResNet50 import ResNet50 as A_ResNet50
from A.models.VGG16 import VGG16 as A_VGG16
from B.models.baselines import Baselines as B_Baselines
from B.models.CNN import CNN as B_CNN
from B.models.DenseNet201 import DenseNet201 as B_DenseNet201
from B.models.InceptionV3 import InceptionV3 as B_InceptionV3
from B.models.MLP import MLP as B_MLP
from B.models.MobileNetV2 import MobileNetV2 as B_MobileNetV2
from B.models.ResNet50 import ResNet50 as B_ResNet50
from B.models.VGG16 import VGG16 as B_VGG16
from utils import get_metrics, hyperpara_selection

from utils import visual4cm

import tensorflow as tf

# # data = np.load('/Users/anlly/Desktop/ucl/Applied Machine Learning Systems-I/AMLS assignment/AMLS_assignment23_24-SN23043574/Datasets/pneumoniamnist.npz')
# # print(f"Train data length: {len(data['train_images'])}, label 0: {np.count_nonzero(data['train_labels'].flatten() == 0)}, label 1: {np.count_nonzero(data['train_labels'].flatten() == 1)}")
# # print(f"Validation data length: {len(data['val_images'])}, label 0: {np.count_nonzero(data['val_labels'].flatten() == 0)}, label 1: {np.count_nonzero(data['val_labels'].flatten() == 1)}")                                                               
# # print(f"Test data length: {len(data['test_images'])}, label 0: {np.count_nonzero(data['test_labels'].flatten() == 0)}, label 1: {np.count_nonzero(data['test_labels'].flatten() == 1)}")
# # Generally the ratio of train:val:test should be 3:1:1, here first use the dataset and no need for train test split


def load_data(task, path, method, batch_size=None):
    file=os.listdir(path)
    Xtest, ytest, Xtrain, ytrain, Xval, yval = [],[],[],[],[],[]
       
    for index,f in enumerate(file):
        if not os.path.isfile(os.path.join(path,f)):
            continue
        else:
            img = cv2.imread(os.path.join(path,f))
            if task == "A" and method in ["LR","KNN","SVM","DT","NB","RF","ABC"]: 
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            if "test" in f:
                Xtest.append(img)
                ytest.append(f.split("_")[1][0])
            elif "train" in f:
                Xtrain.append(img)
                ytrain.append(f.split("_")[1][0])
            elif "val" in f:
                Xval.append(img)
                yval.append(f.split("_")[1][0])

    if method in ["LR","KNN","SVM","DT","NB","RF","ABC"]: # baselines
        if task == "A":
            n,h,w = np.array(Xtrain).shape
            Xtrain = np.array(Xtrain).reshape(n,h*w) # need to reshape gray picture into two-dimensional ones
            Xval = np.array(Xval).reshape(len(Xval),h*w)
            Xtest = np.array(Xtest).reshape(len(Xtest),h*w)
        elif task == "B":
            n,h,w,c = np.array(Xtrain).shape
            Xtrain = np.array(Xtrain).reshape(n,h*w*c) # need to reshape gray picture into two-dimensional ones
            Xval = np.array(Xval).reshape(len(Xval),h*w*c)
            Xtest = np.array(Xtest).reshape(len(Xtest),h*w*c)
        
        return Xtrain,ytrain,Xval,yval,Xtest,ytest

    else:
        n,h,w,c = np.array(Xtrain).shape
        Xtrain = np.array(Xtrain)
        Xval = np.array(Xval)
        Xtest = np.array(Xtest)

        if method in ["CNN","MLP"]:
            train_ds = tf.data.Dataset.from_tensor_slices(
                (Xtrain, np.array(ytrain).astype(int))).batch(batch_size)
            val_ds = tf.data.Dataset.from_tensor_slices((Xval, np.array(yval).astype(int))).batch(batch_size)
            test_ds = tf.data.Dataset.from_tensor_slices((Xtest, np.array(ytest).astype(int))).batch(batch_size)
            normalization_layer = tf.keras.layers.Rescaling(1./255)
            train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
            val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
            test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
            return train_ds, val_ds, test_ds
        else:
            return Xtrain,ytrain,Xval,yval,Xtest,ytest

def load_model(task, method):
        if "CNN" in method:
            model = A_CNN() if task == "A" else B_CNN()
        elif "DenseNet201" in method:
            model = A_DenseNet201(method) if task == "A" else B_DenseNet201(method)
        elif "InceptionV3" in method:
            model = A_InceptionV3(method) if task == "A" else B_InceptionV3(method)
        elif "MLP" in method:
            model = A_MLP() if task == "A" else B_MLP()
        elif "MobileNetV2" in method:
            model = A_MobileNetV2(method) if task == "A" else B_MobileNetV2(method)
        elif "ResNet50" in method:
            model = A_ResNet50(method) if task == "A" else B_ResNet50(method)
        elif "VGG16" in method:
            model = A_VGG16(method) if task == "A" else B_VGG16(method)
        else:
            model = A_Baselines(method) if task == "A" else B_Baselines(method)

        return model

if __name__ == '__main__':

    # argument processing
    parser = argparse.ArgumentParser(description='Argparse')
    parser.add_argument('--task',type=str, default = "A",required=True,help="")
    parser.add_argument('--method',type=str, default="", required=True,help='age of the programmer')
    parser.add_argument('--batch_size',type=int, default=32,help='age of the programmer')
    parser.add_argument('--epochs',type=int, default=10,help='age of the programmer')
    args = parser.parse_args()
    task = args.task
    method = args.method
    print(f"Method: {method} Task: {task}.")
     
    # data processing (haven't decide how to present yet)


    # load data
    print("Start loading data......")
    if task == "A":
        path = 'Outputs/pneumoniamnist/preprocessed_data'
    else:
        path = 'Outputs/pathmnist/preprocessed_data'
        
    if ("LR" in method) or ("KNN" in method) or ("SVM" in method) or ("DT" in method) \
        or ("NB" in method) or ("RF" in method) or ("ABC" in method):
        Xtrain, ytrain, Xtest, ytest, Xval, yval = load_data(task,path,method)
    elif method in ["CNN","MLP"]:
        train_ds, val_ds, test_ds = load_data(task,path,method,batch_size=args.batch_size)
    print("Load data successfully.")
    
    # model selection
    # didn't consider individual pre-trained currently
    print("Start loading model......")
    model = load_model(task, method)
    print("Load model successfully.")
    
    if method in ["LR","KNN","SVM","DT","NB","RF","ABC"]:  
        if method in ["KNN","DT","RF","ABC"]:
            cv_results_ = model.train(Xtrain, ytrain, Xval, yval, gridSearch=True)
        else:
            model.train(Xtrain, ytrain, Xval, yval)
        
        pred_train, pred_val, pred_test = model.test(Xtrain, ytrain, Xval, yval, Xtest)

        res = {"train_res":get_metrics(ytrain, pred_train),
               "val_res":get_metrics(yval, pred_val),
               "test_res":get_metrics(ytest, pred_test)}
        for i in res.items():
            print(i)

    elif method in ["MLP","CNN"]:
        res = model.train(model, train_ds, val_ds, args.epochs)
        test_res = model.test(model, test_ds)
        res.update(test_res)
        for i in res:
            print(i)
    
    elif (("VGG16" in method) or ("ResNet50" in method) or ("DenseNet201" in method) \
        or ("MobileNetV2" in method) or ("InceptionV3" in method)):
        model.train(model, Xtrain, ytrain)
        model.test(model, Xtest, ytest)
        

    # visualization
    if method in ["KNN","DT","RF","ABC"]:
        hyperpara_selection(method, cv_results_["mean_test_score"])

   


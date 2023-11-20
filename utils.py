# cm
# tree/decision boundary
# cross validation average curve with different k
# auc roc comparison
import cv2
import tensorflow as tf
import os
import numpy as np
from sklearn import tree
from sklearn.metrics import confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, ConfusionMatrixDisplay,auc
import matplotlib.colors as mcolors
from A.models.baselines import Baselines as A_Baselines
from A.models.CNN import CNN as A_CNN
from A.models.DenseNet201 import DenseNet201 as A_DenseNet201
from A.models.EnsembleNet import EnsembleNet as A_EnsembleNet
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

        if method in ["CNN","MLP","EnsembleNet"]:
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
            model = A_CNN(task,method) if task == "A" else B_CNN(task, method)
        elif "DenseNet201" in method:
            model = A_DenseNet201(method) if task == "A" else B_DenseNet201(method)
        elif "InceptionV3" in method:
            model = A_InceptionV3(method) if task == "A" else B_InceptionV3(method)
        elif "MLP" in method:
            model = A_MLP(task,method) if task == "A" else B_MLP(task, method)
        elif "MobileNetV2" in method:
            model = A_MobileNetV2(method) if task == "A" else B_MobileNetV2(method)
        elif "ResNet50" in method:
            model = A_ResNet50(method) if task == "A" else B_ResNet50(method)
        elif "VGG16" in method:
            model = A_VGG16(method) if task == "A" else B_VGG16(method)
        elif method == "EnsembleNet":
            model = A_EnsembleNet() if task == "A" else None
        else:
            model = A_Baselines(method) if task == "A" else B_Baselines(method)

        return model


def visual4cm(task, method, ytrain, yval, ytest, train_pred, val_pred, test_pred):
    """
    This function is used for visualizing confusion matrix for modelling experiments.
    :param true: true value/labels
    :param predict: predict value/labels
    """
    cms = {"train":confusion_matrix(ytrain, train_pred),
           "val":confusion_matrix(yval, val_pred),
           "test":confusion_matrix(ytest,test_pred)}
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey='row')

    for index,mode in enumerate(["train","val","test"]):
        disp = ConfusionMatrixDisplay(cms[mode],
                                    display_labels=set(ytrain))
        disp.plot(ax=axes[index])
        disp.ax_.set_title(mode)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        if index!=0:
            disp.ax_.set_ylabel('')

    fig.text(0.45, 0.05, 'Predicted label', ha='center')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    fig.colorbar(disp.im_, ax=axes)

    if not os.path.exists("Outputs/images/confusion_matrix/"):
        os.makedirs("Outputs/images/confusion_matrix/") 
    fig.savefig(f'Outputs/images/confusion_matrix/{method}_task{task}.png')
    


def visual4auc(task, method, ytrain, yval, ytest, train_pred, val_pred, test_pred):
    """
    This function is used for visualizing AUROC curve.
    :param label_dict: predict labels of various methods
    :param class_dict: true labels of various methods
    :param name: name of output picture (name of the method)
    """
    rocs = {"train":roc_curve(np.array(ytrain).astype(int), train_pred.astype(int), pos_label=1, drop_intermediate=True),
           "val":roc_curve(np.array(yval).astype(int), val_pred.astype(int), pos_label=1, drop_intermediate=True),
           "test":roc_curve(np.array(ytest).astype(int), test_pred.astype(int), pos_label=1, drop_intermediate=True)}
    colors = list(mcolors.TABLEAU_COLORS.keys())
    plt.figure(figsize=(10,6))
    for index, mode in enumerate(["train","val","test"]):
        plt.plot(rocs[mode][0], rocs[mode][1], lw=1, label="{}(AUC={:.3f})".format(mode, auc(rocs[mode][0], rocs[mode][1])),
                 color=mcolors.TABLEAU_COLORS[colors[index]])  # draw each one
    plt.plot([0, 1], [0, 1], "--", lw=1, color="grey")
    plt.axis("square")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False Positive Rate", fontsize=10)
    plt.ylabel("True Positive Rate", fontsize=10)
    plt.title(f"ROC Curve for {method}", fontsize=10)
    plt.legend(loc="lower right", fontsize=5)
    if not os.path.exists("Outputs/images/roc_curve/"):
        os.makedirs("Outputs/images/roc_curve/") 
    plt.savefig(f'Outputs/images/roc_curve/{method}_task{task}.png')

def visual4tree(task, method, model):
    """
    This function is used for visualizing decision tree for decision tree family models.
    :param model: the tree to be visualized (i.e. DecisionTreeClassifier/DecisionTreeRegressor)
    :param method: the method which produce the tree (i.e. uniclass_DT)
    :param title: attributes used for splitting tree nodes
    """
    plt.figure(figsize=(100, 15))
    class_names = ["pneumonia","non-pneumonia"] if task == "A" else []
    tree.plot_tree(model, class_names=class_names, filled=True,
                       rounded=True,
                       fontsize=5)
    if not os.path.exists("Outputs/images/trees/"):
        os.makedirs("Outputs/images/trees/") 
    plt.savefig(f'Outputs/images/trees/{method}_task{task}.png')


def get_metrics(task, y,pred):
    average = "binary" if task == "A" else "macro"
    result = {
        "acc":round(accuracy_score(np.array(y).astype(int), pred.astype(int))*100,4),
        "pre":round(precision_score(np.array(y).astype(int), pred.astype(int), average=average)*100,4),
        "rec":round(recall_score(np.array(y).astype(int), pred.astype(int),average=average)*100,4),
        "f1":round(f1_score(np.array(y).astype(int), pred.astype(int),average=average)*100,4)

    }
    return result


def hyperpara_selection(task,method, scores):
    plt.figure(figsize=(8, 5))
    plt.plot(scores, c="g", marker='D', markersize=5)
    plt.xlabel("Params")
    plt.ylabel("Accuracy")
    plt.title(f"Params for {method}")
    if not os.path.exists("Outputs/images/hyperpara_selection/"):
        os.makedirs("Outputs/images/hyperpara_selection/") 
    plt.savefig(f'Outputs/images/hyperpara_selection/{method}_task{task}.png')

def visual4label(task, data):
    fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(6, 3), subplot_kw=dict(aspect="equal"),dpi=600)
    for index,mode in enumerate(["train","val","test"]):
        pie_data = [np.count_nonzero(data[f'{mode}_labels'].flatten() == i) for i in range(len(set(data[f'{mode}_labels'].flatten().tolist())))]
        labels = [f"label {i}" for i in sorted(list(set(data[f'{mode}_labels'].flatten().tolist())))]
        wedges, texts, autotexts = ax[index].pie(pie_data, autopct=lambda pct: f"{pct:.2f}%\n({int(np.round(pct/100.*np.sum(pie_data))):d})",
                                        textprops=dict(color="w"))
        if index == 2:
            ax[index].legend(wedges, labels,
                    loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1))
        size = 6 if task == "A" else 3
        plt.setp(autotexts, size=size, weight="bold")
        ax[index].set_title(mode)
    plt.tight_layout()
    if not os.path.exists("Outputs/images/"):
        os.makedirs("Outputs/images/") 
    fig.savefig(f'Outputs/images/label_distribution_task{task}.png')




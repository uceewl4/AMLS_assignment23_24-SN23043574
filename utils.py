# cm
# tree/decision boundary
# cross validation average curve with different k
# auc roc comparison
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def visual4cm(true, pred):
    """
    This function is used for visualizing confusion matrix for modelling experiments.
    :param true: true value/labels
    :param predict: predict value/labels
    """
    cm = confusion_matrix(true, pred)
    plt.matshow(cm, cmap=plt.cm.Reds)
    plt.colorbar()
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[j, i], xy=(i, j), horizontalalignment="center",
                         verticalalignment="center")
    plt.xlabel("Predict label")
    plt.ylabel("Actual label")
    plt.show()


# def visual4auc(label_dict, class_dict, name):
#     """
#     This function is used for visualizing AUROC curve.
#     :param label_dict: predict labels of various methods
#     :param class_dict: true labels of various methods
#     :param name: name of output picture (name of the method)
#     """
#     colors = list(mcolors.TABLEAU_COLORS.keys())
#     for index, i in enumerate(class_dict.keys()):
#         fpr, tpr, thre = roc_curve(label_dict[i], class_dict[i], pos_label=1, drop_intermediate=True)
#         plt.plot(fpr, tpr, lw=1, label="{}(AUC={:.3f})".format(i, auc(fpr, tpr)),
#                  color=mcolors.TABLEAU_COLORS[colors[index]])  # draw each one
#     plt.plot([0, 1], [0, 1], "--", lw=1, color="grey")
#     plt.axis("square")
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     plt.xlabel("False Positive Rate", fontsize=10)
#     plt.ylabel("True Positive Rate", fontsize=10)
#     plt.title("ROC Curve", fontsize=10)
#     plt.legend(loc="lower right", fontsize=5)
#     plt.savefig(f"C:/FinalProject/outputs/{name}.png".format(name))
#     plt.show()


def get_metrics(y,pred):
    result = {
        "acc":round(accuracy_score(np.array(y).astype(int), pred.astype(int))*100,4),
        "pre":round(precision_score(np.array(y).astype(int), pred.astype(int))*100,4),
        "rec":round(recall_score(np.array(y).astype(int), pred.astype(int))*100,4),
        "f1":round(f1_score(np.array(y).astype(int), pred.astype(int))*100,4)

    }
    return result


def hyperpara_selection(method, scores):
    # knn
    plt.figure(figsize=(8, 5))
    plt.plot(scores, c="g", marker='D', markersize=5)
    plt.xlabel("Params")
    plt.ylabel("Accuracy")
    plt.title(f"Params for {method}")
    if not os.path.exists("Outputs/images/hyperpara_selection/"):
        os.makedirs("Outputs/images/hyperpara_selection/") 
    plt.savefig(f'Outputs/images/hyperpara_selection/{method}.png')


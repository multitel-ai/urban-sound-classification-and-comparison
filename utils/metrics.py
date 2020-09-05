#%%
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
import pytorch_lightning.metrics.functional as plm
import torch


def auprc(y_true, y_scores):
    """ Compute AUPRC for 1 class
        Args:
            y_true (np.array): one hot encoded labels
            y_scores (np.array): model prediction
        Return:
            auc (float): the Area Under the Recall Precision curve
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    recall = np.concatenate((np.array([1.0]), recall, np.array([0.0])))
    precision = np.concatenate((np.array([0.0]), precision, np.array([1.0])))
    return auc(recall, precision)


def auprc_pytorch(y_true, y_scores):
    """ Compute AUPRC for 1 class
        Args:
            y_true (np.array): one hot encoded labels
            y_scores (np.array): model prediction
        Return:
            auc (float): the Area Under the Recall Precision curve
    """
    device = y_true.device
    precision, recall, thresholds = plm.precision_recall_curve(y_scores, y_true)
    # To ensure the curve goes from (0,1) to (1,0) included
    recall = torch.cat((torch.ones([1], device=device), recall, torch.zeros([1], device=device)))
    precision = torch.cat((torch.zeros([1], device=device), precision, torch.ones([1], device=device)))

    return plm.auc(recall, precision)


def compute_macro_auprc_pytorch(y_true, y_scores, return_auprc_per_class=False):
    num_classes = y_true.size()[-1]
    auprc_scores = torch.zeros([num_classes])
    for i in range(num_classes):
        auprc_scores[i] = auprc_pytorch(y_true[:, i], y_scores[:, i])
    # nanmean to ignore nan for borderline cases
    auprc_macro = auprc_scores[~torch.isnan(auprc_scores)].mean()
    if return_auprc_per_class:
        return auprc_scores, auprc_macro
    else:
        return auprc_macro


def compute_micro_auprc_pytorch(y_true, y_scores):
    """ Compute micro AUPRC
        Args:
            y_true (np.array): one hot encoded labels
            y_scores (np.array): model prediction
        Return:
            auprc_macro (float): the micro AUPRC
    """
    auprc_micro = auprc_pytorch(torch.flatten(y_true), torch.flatten(y_scores))
    return auprc_micro


def accuracy_pytorch(y_true, y_scores):
    """ Compute Accuracy
        Args:
            y_true (np.array): one hot encoded labels
            y_scores (np.array): model prediction
        Return:
            auprc_macro (float): the accuracy
    """
    auprc_micro = plm.accuracy(torch.round(torch.flatten(y_scores)), torch.flatten(y_true), 2)
    return auprc_micro


def micro_F1_pytorch(y_true, y_scores):
    """ Compute Accuracy
        Args:
            y_true (np.array): one hot encoded labels
            y_scores (np.array): model prediction
        Return:
            auprc_macro (float): the accuracy
    """
    f1 = plm.f1_score(torch.round(torch.flatten(y_scores)), torch.flatten(y_true), 2)
    return f1


def mean_average_precision_pytorch(y_true, y_scores):
    num_classes = y_true.size()[-1]
    ap_scores = torch.zeros([num_classes])
    for i in range(num_classes):
        ap_scores[i] = plm.average_precision(y_scores[:, i], y_true[:, i])
    # nanmean to ignore nan for borderline cases
    mean_average_precision = ap_scores[~torch.isnan(ap_scores)].mean()
    return mean_average_precision


def binary_confusion_matrix(y_true, y_scores):
    TN, FP, FN, TP = confusion_matrix(y_true, y_scores).ravel()
    return TN, FP, FN, TP


def compute_macro_auprc(y_true, y_scores, return_auprc_per_class=False):
    """ Compute macro AUPRC
        Args:
            y_true (np.array): one hot encoded labels
            y_scores (np.array): model prediction
        Return:
            auprc_macro (float): the macro AUPRC
    """
    _, num_classes = y_true.shape
    auprc_scores = [auprc(y_true[:, i], y_scores[:, i]) for i in range(num_classes)]
    # nanmean to ignore nan for borderline cases
    auprc_macro = np.nanmean(np.array(auprc_scores))
    if return_auprc_per_class:
        return auprc_scores, auprc_macro
    else:
        return auprc_macro


def compute_micro_auprc(y_true, y_scores):
    """ Compute micro AUPRC
        Args:
            y_true (np.array): one hot encoded labels
            y_scores (np.array): model prediction
        Return:
            auprc_macro (float): the micro AUPRC
    """
    auprc_micro = auprc(y_true.flatten(), y_scores.flatten())
    return auprc_micro


def compute_micro_F1(y_true, y_scores):
    """ Compute micro F1 @ 0.5
        Args:
            y_true (np.array): one hot encoded labels
            y_scores (np.array): model prediction
        Return:
            micro_F1 (float): the micro F1
    """
    micro_F1 = f1_score(y_true.flatten(), np.around(y_scores).flatten())
    return micro_F1


def accuracy(y_true, y_scores):
    """ Compute Accuracy
        Args:
            y_true (np.array): one hot encoded labels
            y_scores (np.array): model prediction
        Return:
            auprc_macro (float): the accuracy
    """
    accuracy = accuracy_score(np.argmax(y_true, -1), np.argmax(y_scores, -1))
    return accuracy


def mean_average_precision(y_true, y_scores):
    _, num_classes = y_true.shape
    ap_scores = [average_precision_score(y_true[:, i], y_scores[:, i]) for i in range(num_classes)]
    # nanmean to ignore nan for borderline cases
    mean_average_precision = np.nanmean(np.array(ap_scores))
    return mean_average_precision


if __name__ == "__main__":
    pred = torch.rand(50)
    actual = torch.FloatTensor(
        [
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
        ]
    )
    print(auprc_pytorch(actual, pred))
    pred_numpy = pred.numpy()
    actual_numpy = actual.numpy()
    print(auprc(actual_numpy, pred_numpy))

# %%

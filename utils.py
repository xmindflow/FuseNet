import numpy as np
import cv2
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_image(img_file, img_size):
    im = cv2.imread(img_file)
    im = cv2.resize(im, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32')/255.]))
    
    return data


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    
    
def create_mask(pred, GT):
    
    kernel = np.ones((5, 5), np.uint8) 
    dilated_GT = cv2.dilate(GT, kernel, iterations = 4)

    mult = pred * GT        
    unique, count = np.unique(mult[mult !=0], return_counts=True)
    cls= unique[np.argmax(count)]
    
    lesion = np.where(pred==cls, 1, 0) * dilated_GT
    
    return lesion


def dice_metric(A, B):
    intersect = np.sum(A * B)
    fsum = np.sum(A)
    ssum = np.sum(B)
    dice = (2 * intersect ) / (fsum + ssum)
    
    return dice    


def hm_metric(A, B):
    intersection = A * B
    union = np.logical_or(A, B)
    hm_score = (np.sum(union) - np.sum(intersection)) / np.sum(union)
    
    return hm_score


def xor_metric(A, GT):
    intersection = A * GT
    union = np.logical_or(A, GT)
    xor_score = (np.sum(union) - np.sum(intersection)) / np.sum(GT)
    
    return xor_score
import numpy
import torch
import numpy as np
import cv2
# from hausdorff import hausdorff_distance
import torchvision.transforms as transforms
import surface_distance as surfdist

def evaluate_func(predicts, labels):
    N = predicts.shape[0]

    predicts = predicts.view(N, -1)
    labels = labels.view(N, -1)

    TP = labels * predicts
    FP = predicts - TP
    FN = labels - TP

    precision = torch.sum(TP, dim=1) / torch.sum(TP + FP, dim=1)
    precision = precision.sum() / N

    recall = torch.sum(TP, dim=1) / torch.sum(TP + FN, dim=1)
    recall = recall.sum() / N

    dice = (2 * torch.sum(TP, dim=1)) / ((2 * torch.sum(TP, dim=1)) + torch.sum(FN, dim=1) + torch.sum(FP, dim=1))
    dice = dice.sum() / N

    return dice, precision, recall

def surface_dist(predict, label):
    N = predict.shape[0]

    predict = predict.squeeze(0)
    label = label.squeeze(0)
    predict = predict.cpu().numpy()
    label = label.cpu().numpy()
    predict = np.array(predict).astype(bool)
    label = np.array(label).astype(bool)

    sd = surfdist.compute_surface_distances(label, predict, spacing_mm=(1.0, 1.0, 1.0))
    avg_surf_dist = surfdist.compute_average_surface_distance(sd)
    hd_dist_95 = surfdist.compute_robust_hausdorff(sd, 95)
    avg_surf_dist = np.array(avg_surf_dist, dtype=float)

    asd = avg_surf_dist / N
    hd_dist_95 = hd_dist_95 / N
    asd = torch.tensor(asd)
    hd_dist_95 = torch.tensor(hd_dist_95)

    return asd, hd_dist_95



import torch
import cv2
import numpy as np
from torch.nn.modules.loss import _Loss

class BSLoss(_Loss):

    def __init__(self, *args, **kwargs):
        super(BSLoss, self).__init__()

    def forward(self, prediction, ground_truth, alpha):
        bs_loss = boundary_sensitive_loss(prediction, ground_truth, alpha)
        return bs_loss

class BSL_LC(_Loss):
    def __init__(self, *args, **kwargs):
        super(BSL_LC, self).__init__()

    def forward(self, prediction, ground_truth, alpha, beta):
        bs_loss = boundary_sensitive_loss(prediction, ground_truth, alpha)
        lc = location_constraint(prediction, ground_truth)
        bs_col = beta*bs_loss + (1-beta)*0.000001 * lc
        return bs_col

def location_constraint(prediction, label):
    N = prediction.shape[0]
    # pos_sum = torch.sum(label, 3)
    # pos_sum = torch.sum(pos_sum, 2)
    x = torch.abs((torch.sum(prediction, 2) - torch.sum(label, 2)))
    y = torch.abs((torch.sum(prediction, 3) - torch.sum(label, 3)))
    x = x.view(N, -1)
    y = y.view(N, -1)
    x = x.sum() / N
    y = y.sum() / N
    loss = (x + y) / 2
    return loss

#################################################################################
# new boundary dice loss 2: concern the inside/outside boundary of GT and Pred
def get_boundary(img):
    """
    Get dilated edge image of input image:
    Input: [1, 1, H, W] Tensor or [1, H, W]
    """
    # 1. Convert to numpy
    img = img.detach().cpu().squeeze(0).squeeze(0).numpy()


    img = np.array(img * 255)
    img = img.astype('uint8')


    # 2. Get edge image
    edge_img = cv2.Canny(img, 100, 200)
    _, edge_img = cv2.threshold(edge_img, 127, 255, cv2.THRESH_BINARY)

    # 3. Dilate the edge image
    kernel = np.ones((2, 2), np.uint8)
    edge_dialte_img = cv2.dilate(edge_img, kernel, iterations=2)  # adjust manually
    _, edge_dialte_img = cv2.threshold(edge_dialte_img, 127, 255, cv2.THRESH_BINARY)

    # 4. normalization [0, 255] -> [0, 1]
    edge_dialte_img = edge_dialte_img / 255

    return edge_dialte_img

def boundary_sensitive_loss(prediction, label, alpha, eps=1e-5):
    """
    New boundary dice loss 2: concern the inside/outside boundary of GT and prediction
    Input:
        prediction: [0, 1]
        label: {0, 1}
        label_edge: {0, 1}
    """
    N, C, H, W = prediction.size()
    w_edge = alpha
    w_true = 1 - alpha
    w_bk = w_true

    # ------------------------------------------------------------------------
    # 1. Extract edge images of predictions
    predict_edge = None
    gt_edge = None
    for bs_id in range(N):
        pred = prediction[bs_id, :, :, :]
        gt = label[bs_id, :, :, :]
        pred_edge_img = get_boundary(pred)
        gt_edge_img = get_boundary(gt)
        pred_edge_img_tensor = torch.from_numpy(pred_edge_img).unsqueeze(0).unsqueeze(0)
        gt_edge_img_tensor = torch.from_numpy(gt_edge_img).unsqueeze(0).unsqueeze(0)
        if bs_id == 0:
            predict_edge = pred_edge_img_tensor
            gt_edge = gt_edge_img_tensor
        else:
            predict_edge = torch.cat((predict_edge, pred_edge_img_tensor), dim=0)
            gt_edge = torch.cat((gt_edge, pred_edge_img_tensor), dim=0)

    # ------------------------------------------------------------------------
    # 2. Flatten data [B, 1, H, W] -> [B, H x W], same as pred, label and edge
    prediction = prediction.contiguous().view(N, -1)
    label = label.contiguous().view(N, -1)
    label_edge = gt_edge.contiguous().view(N, -1)
    predict_edge = predict_edge.contiguous().view(N, -1)

    # ------------------------------------------------------------------------
    # 3. TP, FP, FN TN
    TP = prediction * label
    FP = prediction - TP
    FN = label - TP

    # ------------------------------------------------------------------------
    # 4. Weight FN
    FN_in_boundary = FN * label_edge
    FN_out_boundary = FN * predict_edge

    FN_boundary_intersection = FN_in_boundary * FN_out_boundary
    FN_out_boundary = FN_out_boundary - FN_boundary_intersection

    FN_gt = FN - (FN_in_boundary + FN_out_boundary)

    FN = FN_in_boundary * w_edge + FN_out_boundary * w_edge + FN_gt * w_true

    # ------------------------------------------------------------------------
    # 5. Weight FP
    FP_in_boundary = FP * predict_edge
    FP_out_boundary = FP * label_edge

    FP_boundary_intersection = FP_in_boundary * FP_out_boundary
    FP_out_boundary = FP_out_boundary - FP_boundary_intersection

    FP_bk = FP - (FP_in_boundary + FP_out_boundary)

    FP = FP_in_boundary * w_edge + FP_out_boundary * w_edge + FP_bk * w_bk

    # ------------------------------------------------------------------------
    # 6. Loss
    loss = (2 * torch.sum(TP, dim=1) + eps) / (2 * torch.sum(TP, dim=1) +
                                               torch.sum(FP, dim=1) + torch.sum(FN, dim=1) + eps)

    loss = 1 - loss.sum() / N

    return loss
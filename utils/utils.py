import os
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import auc
from skimage.measure import label, regionprops


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def embedding_concat(x, y):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = x.new_zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


def compute_pro_retrieval_metrics(scores, gt_mask):
    """
    calculate segmentation AUPRO, from https://github.com/YoungGod/DFR
    """
    max_step = 1000
    expect_fpr = 0.3  # default 30%
    max_th = scores.max()
    min_th = scores.min()
    delta = (max_th - min_th) / max_step
    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(scores, dtype=np.bool)
    for step in range(max_step):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[scores <= thred] = 0
        binary_score_maps[scores > thred] = 1
        pro = []  # per region overlap
        iou = []  # per image iou
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map 
        for i in range(len(binary_score_maps)):    # for i th image
            # pro (per region level)
            label_map = label(gt_mask[i], connectivity=2)
            props = regionprops(label_map)
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox    # find the bounding box of an anomaly region 
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # cropped_mask = gt_mask[i][x_min:x_max, y_min:y_max]   # bug!
                cropped_mask = prop.filled_image    # corrected!
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)
            # iou (per image level)
            intersection = np.logical_and(binary_score_maps[i], gt_mask[i]).astype(np.float32).sum()
            union = np.logical_or(binary_score_maps[i], gt_mask[i]).astype(np.float32).sum()
            if gt_mask[i].any() > 0:    # when the gt have no anomaly pixels, skip it
                iou.append(intersection / union)
        # against steps and average metrics on the testing data
        ious_mean.append(np.array(iou).mean())
        #print("per image mean iou:", np.array(iou).mean())
        ious_std.append(np.array(iou).std())
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr for pro-auc
        gt_masks_neg = ~gt_mask
        fpr = np.logical_and(gt_masks_neg, binary_score_maps).sum() / gt_masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)
    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)
    ious_mean = np.array(ious_mean)
    ious_std = np.array(ious_std)
    # best per image iou
    best_miou = ious_mean.max()
    #print(f"Best IOU: {best_miou:.4f}")
    # default 30% fpr vs pro, pro_auc
    idx = fprs <= expect_fpr  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]    
    pix_pro_auc = auc(fprs_selected, pros_mean_selected)
    
    return pix_pro_auc


def compute_aupr_retrieval_metrics(
    predicted_masks,
    ground_truth_masks,
    include_optimal_threshold_rates=False,
):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.
    Args:
        predicted_masks: [list of np.arrays or np.array] [NxHxW] Contains
                               generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    pred_mask = copy.deepcopy(predicted_masks)
    gt_mask = copy.deepcopy(ground_truth_masks)
    num = 200
    out = {}

    if pred_mask is None or gt_mask is None:
        for key in out:
            out[key].append(float('nan'))
    else:
        fprs, tprs = [], []
        precisions, f1s = [], []
        gt_mask = np.array(gt_mask, np.uint8)

        t = (gt_mask == 1)
        f = ~t
        n_true = t.sum()
        n_false = f.sum()
        th_min = pred_mask.min() - 1e-8
        th_max = pred_mask.max() + 1e-8
        pred_gt = pred_mask[t]
        th_gt_min = pred_gt.min()
        th_gt_max = pred_gt.max()

        '''
        Using scikit learn to compute pixel au_roc results in a memory error since it tries to store the NxHxW float score values.
        To avoid this, we compute the tp, fp, tn, fn at equally spaced thresholds in the range between min of predicted 
        scores and maximum of predicted scores
        '''
        percents = np.linspace(100, 0, num=num // 2)
        th_gt_per = np.percentile(pred_gt, percents)
        th_unif = np.linspace(th_gt_max, th_gt_min, num=num // 2)
        thresholds = np.concatenate([th_gt_per, th_unif, [th_min, th_max]])
        thresholds = np.flip(np.sort(thresholds))

        if n_true == 0 or n_false == 0:
            raise ValueError("gt_submasks must contains at least one normal and anomaly samples")

        for th in thresholds:
            p = (pred_mask > th).astype(np.uint8)
            p = (p == 1)
            fp = (p & f).sum()
            tp = (p & t).sum()

            fpr = fp / n_false
            tpr = tp / n_true
            if tp + fp > 0:
                prec = tp / (tp + fp)
            else:
                prec = 1.0
            if prec > 0. and tpr > 0.:
                f1 = (2 * prec * tpr) / (prec + tpr)
            else:
                f1 = 0.0
            fprs.append(fpr)
            tprs.append(tpr)
            precisions.append(prec)
            f1s.append(f1)

        roc_auc = auc(fprs, tprs)
        roc_auc = round(roc_auc, 4)
        pr_auc = auc(tprs, precisions)
        pr_auc = round(pr_auc, 4)
        out['roc_auc'] = (roc_auc)
        out['pr_auc'] = (pr_auc)
        out['fpr'] = (fprs)
        out['tpr'] = (tprs)
        out['precision'] = (precisions)
        out['f1'] = (f1s)
        out['thresholds'] = (thresholds)

    return pr_auc

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())

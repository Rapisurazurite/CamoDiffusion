import os
import time

import cv2
import numba
import numpy as np
from tqdm import tqdm
from utils.metrics import Emeasure, Smeasure, WeightedFmeasure, _cal_mae
from utils.metrics import _prepare_data
from tqdm.contrib.concurrent import thread_map, process_map  # or thread_map, process_map

SM = Smeasure()
WFM = WeightedFmeasure()


@numba.jit(nopython=True)
def generate_parts_numel_combinations(fg_fg_numel, fg_bg_numel, pred_fg_numel, pred_bg_numel, gt_fg_numel, gt_size):
    bg_fg_numel = gt_fg_numel - fg_fg_numel
    bg_bg_numel = pred_bg_numel - bg_fg_numel

    parts_numel = [fg_fg_numel, fg_bg_numel, bg_fg_numel, bg_bg_numel]

    mean_pred_value = pred_fg_numel / gt_size
    mean_gt_value = gt_fg_numel / gt_size

    demeaned_pred_fg_value = 1 - mean_pred_value
    demeaned_pred_bg_value = 0 - mean_pred_value
    demeaned_gt_fg_value = 1 - mean_gt_value
    demeaned_gt_bg_value = 0 - mean_gt_value

    combinations = [
        (demeaned_pred_fg_value, demeaned_gt_fg_value),
        (demeaned_pred_fg_value, demeaned_gt_bg_value),
        (demeaned_pred_bg_value, demeaned_gt_fg_value),
        (demeaned_pred_bg_value, demeaned_gt_bg_value),
    ]
    return parts_numel, combinations


def cal_em_with_cumsumhistogram(pred: np.ndarray, gt: np.ndarray, gt_fg_numel, gt_size) -> np.ndarray:
    pred = (pred * 255).astype(np.uint8)
    bins = np.linspace(0, 256, 257)
    fg_fg_hist, _ = np.histogram(pred[gt], bins=bins)
    fg_bg_hist, _ = np.histogram(pred[~gt], bins=bins)
    fg_fg_numel_w_thrs = np.cumsum(np.flip(fg_fg_hist), axis=0)
    fg_bg_numel_w_thrs = np.cumsum(np.flip(fg_bg_hist), axis=0)

    fg___numel_w_thrs = fg_fg_numel_w_thrs + fg_bg_numel_w_thrs
    bg___numel_w_thrs = gt_size - fg___numel_w_thrs

    if gt_fg_numel == 0:
        enhanced_matrix_sum = bg___numel_w_thrs
    elif gt_fg_numel == gt_size:
        enhanced_matrix_sum = fg___numel_w_thrs
    else:
        parts_numel_w_thrs, combinations = generate_parts_numel_combinations(
            fg_fg_numel=fg_fg_numel_w_thrs,
            fg_bg_numel=fg_bg_numel_w_thrs,
            pred_fg_numel=fg___numel_w_thrs,
            pred_bg_numel=bg___numel_w_thrs,
            gt_fg_numel=gt_fg_numel,
            gt_size=gt_size,
        )

        results_parts = np.empty(shape=(4, 256), dtype=np.float64)
        for i, (part_numel, combination) in enumerate(zip(parts_numel_w_thrs, combinations)):
            align_matrix_value = (
                    2
                    * (combination[0] * combination[1])
                    / (combination[0] ** 2 + combination[1] ** 2 + np.spacing(1))
            )
            enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
            results_parts[i] = enhanced_matrix_value * part_numel
        enhanced_matrix_sum = results_parts.sum(axis=0)

    em = enhanced_matrix_sum / (gt_size - 1 + np.spacing(1))
    return em


def measure_mea(mask_name):
    mask_path = os.path.join(mask_root, mask_name)
    pred_path = os.path.join(pred_root, mask_name)
    gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    def cal_sm(self, pred: np.ndarray, gt: np.ndarray):
        sm = self.cal_sm(pred, gt)
        return sm

    def cal_em(pred: np.ndarray, gt: np.ndarray):
        # Here we do not use EM() class to avoid multiple process conflict
        gt_fg_numel = np.count_nonzero(gt)
        gt_size = gt.shape[0] * gt.shape[1]
        changeable_em = cal_em_with_cumsumhistogram(pred, gt, gt_fg_numel, gt_size)
        return changeable_em

    def cal_wfm(self, pred: np.ndarray, gt: np.ndarray):
        if np.all(~gt):
            wfm = 0
        else:
            wfm = self.cal_wfm(pred, gt)
        return wfm

    pred, gt = _prepare_data(pred=pred, gt=gt)
    sm = cal_sm(SM, pred, gt)
    changeable_em = cal_em(pred, gt)
    wfm = cal_wfm(WFM, pred, gt)
    mae = _cal_mae(pred, gt)
    return sm, changeable_em, wfm, mae


def eval(mask_path='./Dataset/TestDataset',
         pred_path='./results',
         dataset_name='COD10K'):
    global mask_root, pred_root
    # for dataset in ['COD10K', 'CAMO', 'CHAMELEON']:
    for dataset in [dataset_name]:
        mask_root = os.path.join(mask_path, dataset, 'GT')
        pred_root = os.path.join(pred_path, dataset)
        mask_name_list = sorted(os.listdir(mask_root))

        res = process_map(measure_mea, mask_name_list, max_workers=8, chunksize=4)

        sms = [x[0] for x in res]
        changeable_ems = [x[1] for x in res]
        wfms = [x[2] for x in res]
        maes = [x[3] for x in res]

        results = {
            "Smeasure": np.mean(np.array(sms, dtype=np.float64)),
            "wFmeasure": np.mean(np.array(wfms, dtype=np.float64)),
            "MAE": np.mean(np.array(maes, dtype=np.float64)),
            "meanEm": np.mean(np.array(changeable_ems, dtype=np.float64), axis=0).mean(),
            "maxEm": np.mean(np.array(changeable_ems, dtype=np.float64), axis=0).max(),
        }
        print(dataset_name, ":", results)
        return results

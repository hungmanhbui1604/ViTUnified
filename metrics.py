import numpy as np
from sklearn.metrics import roc_curve, auc

def compute_recog_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict:
    # fmr, fnmr, tar
    fmr, tar, thrs = roc_curve(labels, scores, pos_label=1)
    fnmr = 1.0 - tar

    asc_idx = np.argsort(thrs)
    thrs = thrs[asc_idx]
    fmr = fmr[asc_idx]
    tar = tar[asc_idx]
    fnmr = fnmr[asc_idx]

    # EER
    diff = np.abs(fmr - fnmr)
    eer_idx = int(np.argmin(diff))
    eer = float((fmr[eer_idx] + fnmr[eer_idx]) / 2.0)
    eer_thr = float(thrs[eer_idx])

    # AUC
    auc_roc = float(auc(fmr, tar))

    # TAR@FAR
    tar_at_far = {}
    for far_target in (0.1, 0.01, 0.001):
        mask = fmr <= far_target
        tar_at_far[far_target] = float(tar[mask].max()) if mask.any() else 0.0

    return {
        "thresholds": thrs.tolist(),
        "FMR": fmr.tolist(),
        "FNMR": fnmr.tolist(),
        "TAR": tar.tolist(),
        "EER": eer,
        "EER_threshold": eer_thr,
        "AUC": auc_roc,
        "TAR@FAR=0.1": tar_at_far[0.1],
        "TAR@FAR=0.01": tar_at_far[0.01],
        "TAR@FAR=0.001": tar_at_far[0.001],
    }
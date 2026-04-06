import numpy as np
from sklearn.metrics import roc_curve

def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    # --- Sanity checks ---
    assert scores.shape == labels.shape, "scores and labels must have same shape"
    assert set(np.unique(labels)) <= {0, 1}, "labels must be 0/1"

    # --- Compute ROC ---
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr

    # --- Find EER via interpolation ---
    diff = fpr - fnr

    # find first index where sign changes
    idx1_candidates = np.where(diff >= 0)[0]
    if len(idx1_candidates) == 0:
        # fallback (rare edge case)
        idx = np.argmin(np.abs(diff))
        eer = (fpr[idx] + fnr[idx]) / 2.0
        return float(eer), float(thresholds[idx])

    idx1 = idx1_candidates[0]
    idx0 = max(idx1 - 1, 0)

    # points (x0, y0) and (x1, y1)
    x0, y0 = fpr[idx0], fnr[idx0]
    x1, y1 = fpr[idx1], fnr[idx1]

    # linear interpolation
    denom = (x1 - x0) - (y1 - y0)
    if denom == 0:
        t = 0.0
    else:
        t = (y0 - x0) / denom

    eer = x0 + t * (x1 - x0)
    eer_threshold = thresholds[idx0] + t * (thresholds[idx1] - thresholds[idx0])

    return float(eer), float(eer_threshold)


def compute_roc(
    scores: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 2000,
) -> dict:

    genuine_scores = scores[labels == 0]   # higher = more similar
    impostor_scores = scores[labels == 1]

    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)

    fmr_list, fnmr_list = [], []

    for thr in thresholds:
        # FMR  = False Match Rate  = impostors accepted (score >= thr)
        fmr  = float((impostor_scores >= thr).mean())
        # FNMR = False Non-Match Rate = genuines rejected (score < thr)
        fnmr = float((genuine_scores  <  thr).mean())
        fmr_list.append(fmr)
        fnmr_list.append(fnmr)

    fmr_arr  = np.array(fmr_list)
    fnmr_arr = np.array(fnmr_list)
    tar_arr  = 1.0 - fnmr_arr          # TAR = True Accept Rate = 1 - FNMR

    # EER: point where FMR ≈ FNMR
    diff     = np.abs(fmr_arr - fnmr_arr)
    eer_idx  = int(np.argmin(diff))
    eer      = float((fmr_arr[eer_idx] + fnmr_arr[eer_idx]) / 2.0)
    eer_thr  = float(thresholds[eer_idx])

    # AUC via trapezoidal rule (FAR on x-axis, TAR on y-axis)
    # sort by increasing FAR for a well-defined integral
    sort_idx = np.argsort(fmr_arr)
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    auc = float(_trapz(tar_arr[sort_idx], fmr_arr[sort_idx]))

    # TAR at fixed FAR operating points
    tar_at_far = {}
    for far_target in (0.01, 0.001, 0.0001):
        # find thresholds where FMR <= far_target, pick highest TAR among them
        mask = fmr_arr <= far_target
        if mask.any():
            tar_at_far[far_target] = float(tar_arr[mask].max())
        else:
            tar_at_far[far_target] = 0.0

    return {
        "thresholds"  : thresholds.tolist(),
        "FMR"         : fmr_arr.tolist(),
        "FNMR"        : fnmr_arr.tolist(),
        "TAR"         : tar_arr.tolist(),
        "EER"         : eer,
        "EER_threshold": eer_thr,
        "AUC"         : auc,
        "TAR@FAR=0.01"   : tar_at_far[0.01],
        "TAR@FAR=0.001"  : tar_at_far[0.001],
        "TAR@FAR=0.0001" : tar_at_far[0.0001],
    }
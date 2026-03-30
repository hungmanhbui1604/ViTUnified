import numpy as np

def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    genuine_scores = scores[labels == 0]
    impostor_scores = scores[labels == 1]

    thresholds = np.linspace(scores.min(), scores.max(), 1000)
    min_diff = float("inf")
    eer = 1.0
    best_thr = thresholds[0]

    for thr in thresholds:
        far = (impostor_scores >= thr).mean()
        frr = (genuine_scores < thr).mean()
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2.0
            best_thr = thr

    return float(eer), float(best_thr)
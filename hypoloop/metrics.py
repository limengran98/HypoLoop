"""Task-native prediction metrics and group-aware aggregation."""
import numpy as np


def mse(prediction, target, ids=None):
    mask = np.isfinite(target)
    return float(np.mean((np.asarray(prediction)[mask]-np.asarray(target)[mask])**2))


def mae(prediction, target, ids=None):
    mask = np.isfinite(target)
    return float(np.mean(np.abs(np.asarray(prediction)[mask]-np.asarray(target)[mask])))


def rmse(prediction, target, ids=None):
    return float(np.sqrt(mse(prediction, target)))


def relative_l2(prediction, target, ids=None):
    p, y = np.asarray(prediction), np.asarray(target)
    return float(np.mean(np.linalg.norm(p-y, axis=1)/np.maximum(np.linalg.norm(y, axis=1), 1e-12)))


def abs_rel(prediction, target, ids=None):
    p, y = np.asarray(prediction), np.asarray(target)
    valid = np.isfinite(y) & (y > 0)
    return float(np.mean([np.mean(np.abs(a[m]-b[m])/b[m]) for a, b, m in zip(p, y, valid) if m.any()]))


def average_precision(prediction, target):
    p, y = np.asarray(prediction).ravel(), np.asarray(target).ravel()
    observed = np.isfinite(y)
    p, y = p[observed], y[observed] > 0
    if not y.any():
        return 0.0
    order = np.argsort(-p, kind="stable")
    p, y = p[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(p)), len(p)-1]
    tp = np.cumsum(y)[ends]
    recall = tp / y.sum()
    precision = tp / (ends+1)
    return float(np.sum(np.diff(np.r_[0., recall])*precision))


def f1(prediction, target):
    p, y = np.asarray(prediction).ravel() > 0, np.asarray(target).ravel() > 0
    tp = np.count_nonzero(p & y)
    return float(2*tp/max(np.count_nonzero(p)+np.count_nonzero(y), 1))


def grouped_folds(ids, groups, maximum=3, seed=17):
    """Partition complete groups without using targets."""
    groups = np.asarray([groups[g] for g in ids])
    unique = np.unique(groups)
    count = min(maximum, len(unique))
    if count < 2:
        raise ValueError("Validation requires at least two experimental groups.")
    parts = np.array_split(np.random.default_rng(seed).permutation(unique), count)
    result = []
    for part in parts:
        valid = np.flatnonzero(np.isin(groups, part))
        train = np.flatnonzero(~np.isin(groups, part))
        result.append((train, valid))
    return result

import numpy as np
import pandas as pd


def normalize_targets(labels):
    if isinstance(labels, pd.DataFrame):
        if labels.shape[1] == 1:
            return labels.iloc[:, 0]
        return labels.to_numpy()
    return labels


def normalize_feature_row(row):
    if isinstance(row, pd.Series):
        return row.to_numpy(dtype=float).tolist()
    return np.asarray(row, dtype=float).tolist()


def normalize_state_targets(labels):
    if labels is None:
        raise ValueError("Labels cannot be None.")

    if isinstance(labels, pd.DataFrame):
        if labels.shape[1] == 1:
            return [np.asarray(value) for value in labels.iloc[:, 0].tolist()]
        return [row.to_numpy(dtype=float) for _, row in labels.iterrows()]

    if isinstance(labels, pd.Series):
        return [np.asarray(value) for value in labels.tolist()]

    array = np.asarray(labels, dtype=object)
    if array.ndim == 1:
        return [np.asarray(value) for value in array.tolist()]
    return [np.asarray(value, dtype=float) for value in array]


def normalize_sample_targets(labels):
    if labels is None:
        raise ValueError("Labels cannot be None.")

    if isinstance(labels, pd.DataFrame):
        if labels.shape[1] == 1:
            return [np.asarray(value) for value in labels.iloc[:, 0].tolist()]
        return [row.to_numpy(dtype=float) for _, row in labels.iterrows()]

    if isinstance(labels, pd.Series):
        return [np.asarray(value) for value in labels.tolist()]

    array = np.asarray(labels, dtype=object)
    if array.ndim == 0:
        return [np.asarray(array.item())]
    if array.ndim == 1:
        return [np.asarray(value) for value in array.tolist()]
    return [np.asarray(value, dtype=float) for value in array]

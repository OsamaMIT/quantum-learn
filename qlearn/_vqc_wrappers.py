import inspect

import numpy as np
import pandas as pd

from ._backends import DEFAULT_BACKEND, load_backend_attr
from ._utils import normalize_targets


def _call_hook(hook, primary_arg, **context):
    if hook is None:
        return primary_arg

    try:
        signature = inspect.signature(hook)
    except (TypeError, ValueError):
        return hook(primary_arg)

    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return hook(primary_arg, **context)

    accepted_context = {
        name: value for name, value in context.items() if name in signature.parameters
    }
    return hook(primary_arg, **accepted_context)


def _normalize_numeric_matrix(values):
    if isinstance(values, pd.DataFrame):
        return values.to_numpy(dtype=float)
    if isinstance(values, pd.Series):
        return values.to_numpy(dtype=float).reshape(-1, 1)

    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


class BaseVariationalQuantumWrapper:
    def __init__(
        self,
        *,
        backend=DEFAULT_BACKEND,
        vqc=None,
        target_encoder=None,
        prediction_decoder=None,
        fit_kwargs=None,
    ):
        self.backend = backend
        self.vqc = vqc
        self.vqc_ = None
        self.target_encoder = target_encoder
        self.prediction_decoder = prediction_decoder
        self.fit_kwargs = dict(fit_kwargs or {})

    def _create_vqc(self):
        vqc_class = load_backend_attr(self.backend, "VariationalQuantumCircuit")
        return vqc_class()

    def _get_vqc(self):
        if self.vqc_ is None:
            self.vqc_ = self.vqc or self._create_vqc()
        return self.vqc_

    def _ensure_is_fitted(self):
        if self.vqc_ is None:
            raise ValueError("Model has not been trained yet.")

    def default_target_encoder(self, labels, features=None):
        return labels

    def default_prediction_decoder(self, outputs, features=None):
        return outputs

    def default_fit_kwargs(self, features, labels):
        return {}

    def encode_targets(self, labels, features=None):
        if self.target_encoder is None:
            return self.default_target_encoder(labels, features=features)
        return _call_hook(
            self.target_encoder,
            labels,
            features=features,
            wrapper=self,
        )

    def decode_predictions(self, outputs, features=None):
        if self.prediction_decoder is None:
            return self.default_prediction_decoder(outputs, features=features)
        return _call_hook(
            self.prediction_decoder,
            outputs,
            features=features,
            wrapper=self,
        )

    def fit(self, features, labels, **kwargs):
        if features is None or labels is None:
            raise ValueError("Features and labels cannot be None.")

        default_kwargs = self.default_fit_kwargs(features, labels)
        encoded_targets = self.encode_targets(labels, features=features)

        fit_kwargs = dict(default_kwargs)
        fit_kwargs.update(self.fit_kwargs)
        fit_kwargs.update(kwargs)

        vqc = self._get_vqc()
        fit_result = vqc.fit(features, encoded_targets, **fit_kwargs)
        if fit_result is not None:
            self.vqc_ = fit_result
        return self

    def train(self, *args, **kwargs):
        return self.fit(*args, **kwargs)

    def predict_raw(self, features, **kwargs):
        self._ensure_is_fitted()
        return self.vqc_.predict(features, **kwargs)

    def predict(self, features, **kwargs):
        raw_outputs = self.predict_raw(features, **kwargs)
        return self.decode_predictions(raw_outputs, features=features)


class _ClassMetadataMixin:
    def _update_classes(self, labels):
        normalized = normalize_targets(labels)
        array = np.asarray(normalized, dtype=object)
        if array.ndim == 1 and all(np.ndim(value) == 0 for value in array.tolist()):
            self.classes_ = np.unique(array)
        else:
            self.classes_ = None

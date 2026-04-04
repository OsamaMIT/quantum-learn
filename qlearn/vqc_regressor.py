import numpy as np

from ._backends import DEFAULT_BACKEND
from ._vqc_wrappers import BaseVariationalQuantumWrapper, _call_hook, _normalize_numeric_matrix


class VariationalQuantumRegressor(BaseVariationalQuantumWrapper):
    def __init__(
        self,
        *,
        backend=DEFAULT_BACKEND,
        vqc=None,
        target_encoder=None,
        prediction_decoder=None,
        fit_kwargs=None,
    ):
        super().__init__(
            backend=backend,
            vqc=vqc,
            target_encoder=target_encoder,
            prediction_decoder=prediction_decoder,
            fit_kwargs=fit_kwargs,
        )
        self.n_targets_ = None
        self.target_min_ = None
        self.target_max_ = None

    def _resolve_target_layout(self, features, labels, kwargs=None):
        requested_kwargs = dict(self.fit_kwargs)
        if kwargs:
            requested_kwargs.update(kwargs)

        matrix = _normalize_numeric_matrix(labels)
        self.n_targets_ = matrix.shape[1]
        self.target_min_ = matrix.min(axis=0)
        self.target_max_ = matrix.max(axis=0)

        requested_qubits = requested_kwargs.get("n_qubits")
        minimum_qubits = self.n_targets_
        if requested_qubits is not None and requested_qubits < minimum_qubits:
            raise ValueError(
                "n_qubits is too small for the number of regression targets. "
                f"Need at least {minimum_qubits} qubits for {self.n_targets_} targets."
            )
        return max(len(features.columns), requested_qubits or 0, minimum_qubits)

    def default_fit_kwargs(self, features, labels):
        total_qubits = self._resolve_target_layout(features, labels)
        return {
            "n_qubits": total_qubits,
            "measurement": "expectation_z",
            "measurement_wires": list(range(self.n_targets_)),
            "loss": "mse",
        }

    def default_target_encoder(self, labels, features=None):
        matrix = _normalize_numeric_matrix(labels)
        scale = self.target_max_ - self.target_min_
        safe_scale = np.where(scale == 0, 1.0, scale)
        normalized = 2.0 * ((matrix - self.target_min_) / safe_scale) - 1.0
        normalized[:, scale == 0] = 0.0
        return normalized

    def fit(self, features, labels, **kwargs):
        self._resolve_target_layout(features, labels, kwargs=kwargs)
        return super().fit(features, labels, **kwargs)

    def default_prediction_decoder(self, outputs, features=None):
        predictions = np.asarray(outputs, dtype=float)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        predictions = predictions[:, : self.n_targets_]
        scale = self.target_max_ - self.target_min_
        restored = 0.5 * (predictions + 1.0) * scale + self.target_min_
        restored[:, scale == 0] = self.target_min_[scale == 0]

        if self.n_targets_ == 1:
            return restored[:, 0]
        return restored

    def decode_predictions(self, outputs, features=None):
        if self.prediction_decoder is None:
            return self.default_prediction_decoder(outputs, features=features)
        return _call_hook(
            self.prediction_decoder,
            outputs,
            features=features,
            wrapper=self,
        )

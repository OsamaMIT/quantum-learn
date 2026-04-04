import math

import numpy as np

from ._backends import DEFAULT_BACKEND
from ._utils import normalize_targets
from ._vqc_wrappers import BaseVariationalQuantumWrapper, _ClassMetadataMixin, _call_hook


class VariationalQuantumClassifier(_ClassMetadataMixin, BaseVariationalQuantumWrapper):
    def __init__(
        self,
        *,
        backend=DEFAULT_BACKEND,
        vqc=None,
        target_encoder=None,
        prediction_decoder=None,
        probability_decoder=None,
        fit_kwargs=None,
    ):
        super().__init__(
            backend=backend,
            vqc=vqc,
            target_encoder=target_encoder,
            prediction_decoder=prediction_decoder,
            fit_kwargs=fit_kwargs,
        )
        self.classes_ = None
        self.class_to_index_ = None
        self.n_output_qubits_ = None
        self.output_dimension_ = None
        self.probability_decoder = probability_decoder

    def _resolve_output_layout(self, features, labels, kwargs=None):
        if self.classes_ is None:
            raise ValueError(
                "Default classification encoding requires one-dimensional labels."
            )

        requested_kwargs = dict(self.fit_kwargs)
        if kwargs:
            requested_kwargs.update(kwargs)

        requested_qubits = requested_kwargs.get("n_qubits")
        minimum_output_qubits = max(1, math.ceil(math.log2(len(self.classes_))))

        if requested_qubits is not None and requested_qubits < minimum_output_qubits:
            raise ValueError(
                "n_qubits is too small for the number of classes. "
                f"Need at least {minimum_output_qubits} qubits for {len(self.classes_)} classes."
            )

        total_qubits = max(
            len(features.columns),
            requested_qubits or 0,
            minimum_output_qubits,
        )
        self.n_output_qubits_ = minimum_output_qubits
        self.output_dimension_ = 2 ** self.n_output_qubits_
        self.class_to_index_ = {
            label: index for index, label in enumerate(self.classes_.tolist())
        }
        return total_qubits

    def default_fit_kwargs(self, features, labels):
        total_qubits = self._resolve_output_layout(features, labels)
        return {
            "n_qubits": total_qubits,
            "measurement": "probabilities",
            "measurement_wires": list(range(self.n_output_qubits_)),
            "loss": "cross_entropy",
        }

    def default_target_encoder(self, labels, features=None):
        normalized = normalize_targets(labels)
        values = np.asarray(normalized, dtype=object).tolist()
        encoded = []
        for value in values:
            if value not in self.class_to_index_:
                raise ValueError(f"Unknown class label '{value}' encountered.")
            vector = np.zeros(self.output_dimension_, dtype=float)
            vector[self.class_to_index_[value]] = 1.0
            encoded.append(vector)
        return encoded

    def _normalize_class_probabilities(self, outputs):
        raw_outputs = np.asarray(outputs, dtype=float)
        if raw_outputs.ndim == 1:
            raw_outputs = raw_outputs.reshape(1, -1)

        class_probabilities = raw_outputs[:, : len(self.classes_)]
        class_probabilities = np.clip(class_probabilities, 0.0, None)
        totals = class_probabilities.sum(axis=1, keepdims=True)
        zero_totals = totals.squeeze(axis=1) == 0
        if np.any(zero_totals):
            class_probabilities[zero_totals] = 1.0 / len(self.classes_)
            totals = class_probabilities.sum(axis=1, keepdims=True)
        return class_probabilities / totals

    def fit(self, features, labels, **kwargs):
        self._update_classes(labels)
        self._resolve_output_layout(features, labels, kwargs=kwargs)
        return super().fit(features, labels, **kwargs)

    def default_prediction_decoder(self, outputs, features=None):
        probabilities = self._normalize_class_probabilities(outputs)
        indices = probabilities.argmax(axis=1)
        return self.classes_[indices]

    def decode_predictions(self, outputs, features=None):
        if self.prediction_decoder is None:
            return self.default_prediction_decoder(outputs, features=features)
        return _call_hook(
            self.prediction_decoder,
            outputs,
            classes=self.classes_,
            features=features,
            wrapper=self,
        )

    def predict_proba(self, features, **kwargs):
        raw_outputs = self.predict_raw(features, **kwargs)
        if self.probability_decoder is None:
            return self._normalize_class_probabilities(raw_outputs)
        return _call_hook(
            self.probability_decoder,
            raw_outputs,
            classes=self.classes_,
            features=features,
            wrapper=self,
        )

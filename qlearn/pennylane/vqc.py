import inspect

import pennylane as qml
from pennylane import numpy as np

from qlearn.backends.base import VariationalQuantumCircuitBase
from qlearn._utils import normalize_feature_row, normalize_sample_targets, normalize_state_targets


class VariationalQuantumCircuit(VariationalQuantumCircuitBase):
    def __init__(self):
        self.params = None
        self.ansatz = None
        self.measurement = "state"
        self.measurement_wires = None
        self.loss = "fidelity"
        self.n_qubits = None

    @staticmethod
    def default_ansatz(features, params, n_qubits):
        for i in range(n_qubits):
            qml.Rot(
                features[i] * params[i][0],
                features[i] * params[i][1],
                features[i] * params[i][2],
                wires=i,
            )
            if i < n_qubits - 1:
                qml.CNOT(wires=[i, i + 1])

    def _resolve_measurement_wires(self, n_qubits, measurement_wires=None):
        if measurement_wires is None:
            return list(range(n_qubits))
        return list(measurement_wires)

    def _invoke_measurement(self, measurement, n_qubits, measurement_wires):
        if measurement in (None, "state"):
            return qml.state()
        if measurement == "probabilities":
            return qml.probs(wires=measurement_wires)
        if measurement == "expectation_z":
            return [qml.expval(qml.PauliZ(wire)) for wire in measurement_wires]
        if callable(measurement):
            try:
                signature = inspect.signature(measurement)
            except (TypeError, ValueError):
                return measurement()

            kwargs = {}
            if "n_qubits" in signature.parameters:
                kwargs["n_qubits"] = n_qubits
            if "measurement_wires" in signature.parameters:
                kwargs["measurement_wires"] = measurement_wires
            return measurement(**kwargs)
        raise ValueError(
            "Measurement must be 'state', 'probabilities', 'expectation_z', or a callable."
        )

    def _resolve_loss(self, loss):
        if loss in (None, "fidelity"):
            return self._fidelity_loss
        if loss == "mse":
            return self._mse_loss
        if loss in {"cross_entropy", "categorical_cross_entropy"}:
            return self._cross_entropy_loss
        if callable(loss):
            return loss
        raise ValueError(
            "Loss must be 'fidelity', 'mse', 'cross_entropy', or a callable."
        )

    def _prepare_targets(self, labels, loss):
        if loss in (None, "fidelity"):
            return normalize_state_targets(labels)
        return normalize_sample_targets(labels)

    @staticmethod
    def _fidelity_loss(output, target):
        state0 = qml.math.dm_from_state_vector(output)
        state1 = qml.math.dm_from_state_vector(target)
        return 1 - qml.math.fidelity(state0, state1)

    @staticmethod
    def _mse_loss(output, target):
        output_array = qml.math.reshape(qml.math.asarray(output), (-1,))
        target_array = qml.math.reshape(qml.math.asarray(target), (-1,))
        return qml.math.mean((output_array - target_array) ** 2)

    @staticmethod
    def _cross_entropy_loss(output, target):
        output_array = qml.math.reshape(qml.math.asarray(output), (-1,))
        target_array = qml.math.reshape(qml.math.asarray(target), (-1,))
        output_array = qml.math.clip(output_array, 1e-9, 1.0)
        return -qml.math.sum(target_array * qml.math.log(output_array))

    def forward(
        self,
        features,
        params,
        n_qubits,
        device,
        ansatz=None,
        measurement="state",
        measurement_wires=None,
        diff_method="backprop",
    ):
        resolved_wires = self._resolve_measurement_wires(
            n_qubits,
            measurement_wires=measurement_wires,
        )

        @qml.qnode(device, diff_method=diff_method)
        def circuit(features, params):
            if ansatz is not None:
                ansatz(features, params, n_qubits)
            else:
                self.default_ansatz(features, params, n_qubits)
            return self._invoke_measurement(measurement, n_qubits, resolved_wires)

        return circuit(features, params)

    def generator(self, features, params, n_qubits, device, ansatz=None, diff_method="backprop"):
        return self.forward(
            features,
            params,
            n_qubits,
            device,
            ansatz=ansatz,
            measurement="state",
            measurement_wires=None,
            diff_method=diff_method,
        )

    def fit(
        self,
        features,
        labels,
        params=None,
        batch_size=32,
        epochs=1,
        n_qubits=None,
        ansatz=None,
        optimizer=None,
        device=None,
        diff_method="backprop",
        measurement="state",
        measurement_wires=None,
        loss="fidelity",
        **_unused,
    ):
        if features is None or labels is None:
            raise ValueError("Features and labels cannot be None.")

        if n_qubits is None:
            n_qubits = len(features.columns)

        if device is None:
            device = qml.device("default.qubit", wires=n_qubits)

        if optimizer is None:
            optimizer = qml.AdamOptimizer(stepsize=0.05)

        if params is None:
            params = np.random.randn(n_qubits, 3)
        else:
            params = np.array(params, requires_grad=True)

        feature_rows = [normalize_feature_row(row) for _, row in features.iterrows()]
        targets = self._prepare_targets(labels, loss)
        if len(feature_rows) != len(targets):
            raise ValueError("Features and labels must contain the same number of rows.")

        self.ansatz = ansatz
        self.measurement = measurement
        self.measurement_wires = self._resolve_measurement_wires(
            n_qubits,
            measurement_wires=measurement_wires,
        )
        self.loss = loss
        self.n_qubits = n_qubits

        loss_fn = self._resolve_loss(loss)

        for _ in range(epochs):
            for start_idx in range(0, len(feature_rows), batch_size):
                batch_end = min(start_idx + batch_size, len(feature_rows))
                batch_features = feature_rows[start_idx:batch_end]
                batch_targets = targets[start_idx:batch_end]

                def cost_function(current_params):
                    batch_loss = 0
                    for feature_row, target in zip(batch_features, batch_targets):
                        output = self.forward(
                            feature_row,
                            current_params,
                            n_qubits,
                            device,
                            ansatz=ansatz,
                            measurement=measurement,
                            measurement_wires=self.measurement_wires,
                            diff_method=diff_method,
                        )
                        batch_loss += loss_fn(output, target)
                    return qml.math.mean(batch_loss / len(batch_features))

                params = optimizer.step(cost_function, params)

        self.params = params
        return self

    def train(self, *args, **kwargs):
        return self.fit(*args, **kwargs)

    def predict(
        self,
        features,
        n_qubits=None,
        device=None,
        diff_method="backprop",
        measurement=None,
        measurement_wires=None,
        **_unused,
    ):
        if self.params is None:
            raise ValueError("Model has not been trained yet.")

        resolved_n_qubits = n_qubits or self.n_qubits or len(features.columns)
        if device is None:
            device = qml.device("default.qubit", wires=resolved_n_qubits)

        resolved_measurement = self.measurement if measurement is None else measurement
        resolved_wires = (
            self.measurement_wires
            if measurement_wires is None
            else self._resolve_measurement_wires(
                resolved_n_qubits,
                measurement_wires=measurement_wires,
            )
        )

        predictions = []
        for _, row in features.iterrows():
            feature_row = normalize_feature_row(row)
            output = self.forward(
                feature_row,
                self.params,
                resolved_n_qubits,
                device,
                ansatz=self.ansatz,
                measurement=resolved_measurement,
                measurement_wires=resolved_wires,
                diff_method=diff_method,
            )
            predictions.append(output)
        return predictions

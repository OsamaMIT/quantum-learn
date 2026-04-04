import pennylane as qml
from pennylane import numpy as np

from qlearn.backends.base import VariationalQuantumCircuitBase
from qlearn._utils import normalize_feature_row, normalize_state_targets


class VariationalQuantumCircuit(VariationalQuantumCircuitBase):
    def __init__(self):
        self.params = None
        self.ansatz = None

    def generator(self, features, params, n_qubits, device, ansatz=None, diff_method="backprop"):
        @qml.qnode(device, diff_method=diff_method)
        def circuit(features, params):
            if ansatz is not None:
                return ansatz(features, params, n_qubits)

            for i in range(n_qubits):
                qml.Rot(
                    features[i] * params[i][0],
                    features[i] * params[i][1],
                    features[i] * params[i][2],
                    wires=i,
                )
                if i < n_qubits - 1:
                    qml.CNOT(wires=[i, i + 1])
            return qml.state()

        return circuit(features, params)

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
        targets = normalize_state_targets(labels)

        if len(feature_rows) != len(targets):
            raise ValueError("Features and labels must contain the same number of rows.")

        self.ansatz = ansatz

        def fidelity_loss(output, target):
            state0 = qml.math.dm_from_state_vector(output)
            state1 = qml.math.dm_from_state_vector(target)
            return 1 - qml.math.fidelity(state0, state1)

        for _ in range(epochs):
            for start_idx in range(0, len(feature_rows), batch_size):
                batch_end = min(start_idx + batch_size, len(feature_rows))
                batch_features = feature_rows[start_idx:batch_end]
                batch_targets = targets[start_idx:batch_end]

                def cost_function(current_params):
                    batch_loss = 0
                    for feature_row, target in zip(batch_features, batch_targets):
                        output = self.generator(
                            feature_row,
                            current_params,
                            n_qubits,
                            device,
                            ansatz=ansatz,
                            diff_method=diff_method,
                        )
                        batch_loss += fidelity_loss(output, target)
                    return qml.math.mean(batch_loss / len(batch_features))

                params = optimizer.step(cost_function, params)

        self.params = params
        return self

    def train(self, *args, **kwargs):
        return self.fit(*args, **kwargs)

    def predict(self, features, n_qubits=None, device=None, diff_method="backprop", **_unused):
        if self.params is None:
            raise ValueError("Model has not been trained yet.")

        if n_qubits is None:
            n_qubits = len(features.columns)
        if device is None:
            device = qml.device("default.qubit", wires=n_qubits)

        predictions = []
        for _, row in features.iterrows():
            feature_row = normalize_feature_row(row)
            output = self.generator(
                feature_row,
                self.params,
                n_qubits,
                device,
                ansatz=self.ansatz,
                diff_method=diff_method,
            )
            predictions.append(output)
        return predictions

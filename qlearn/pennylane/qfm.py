import warnings

import pandas as pd
import pennylane as qml

from qlearn.backends.base import QuantumFeatureMapBase


class QuantumFeatureMap(QuantumFeatureMapBase):
    @staticmethod
    def default_feature_map(data, qubits):
        for i in range(min(len(data), qubits)):
            value = data.iloc[i].item()
            qml.RY(value, wires=i)
            qml.RX(value, wires=i)

        for i in range(qubits):
            qml.Hadamard(wires=i)

        for i in range(1, min(len(data), qubits)):
            value = data.iloc[i].item()
            qml.CRY(value, wires=[i - 1, i])
            qml.CRZ(value, wires=[i - 1, i])

        for i in range(qubits - 1, 0, -1):
            for j in range(i - 1, -1, -1):
                qml.CNOT(wires=[i, j])

        for i in range(min(len(data), qubits)):
            value = data.iloc[i].item()
            qml.RX(value * 0.8, wires=i)
            qml.RZ(value * 1.2, wires=i)

        for i in range(min(len(data), qubits)):
            value = data.iloc[i].item()
            qml.RY(value, wires=i)
            qml.RZ(value, wires=i)

    def transform(self, data=None, feature_map=None, qubits=None, device=None, **_unused):
        if data is None:
            raise ValueError("Data cannot be None.")

        if qubits is None:
            qubits = len(data.columns)
            warnings.warn(
                "The number of qubits required is not specified, by default the number of columns in the data will be used."
            )

        if device is None:
            device = qml.device("default.qubit", wires=qubits)

        if feature_map is None:
            feature_map = QuantumFeatureMap.default_feature_map
            warnings.warn(
                "No feature map is specified, by default a general purpose feature map will be used. It is recommended to use a custom feature map fit for the data."
            )
        elif isinstance(feature_map, str):
            if feature_map == "default":
                feature_map = QuantumFeatureMap.default_feature_map
            else:
                raise ValueError(
                    "Feature map must be a function or a string containing a valid feature map name."
                )
        elif not callable(feature_map):
            raise ValueError(
                "Feature map must be a function or a string containing a valid feature map name."
            )

        @qml.qnode(device)
        def quantum_circuit(row):
            feature_map(row, qubits)
            return [qml.expval(qml.PauliZ(i)) for i in range(qubits)]

        transformed_data = [quantum_circuit(row) for _, row in data.iterrows()]
        return pd.DataFrame(
            transformed_data,
            index=data.index,
            columns=[f"Qubit_{i}" for i in range(qubits)],
        )

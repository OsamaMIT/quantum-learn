import pennylane as qml
from pennylane import numpy as np
import pandas as pd

import warnings

class QuantumFeatureMap:

    def default_feature_map(data, qubits): # Try to find some universal ansatz that works as a general best fit
        # Layer 1: Initial rotations
        for i in range(min(len(data), qubits)):
            qml.RY(data.iloc[i].item(), wires=i)
            qml.RX(data.iloc[i].item(), wires=i)

        # Superposition Layer: After initial rotations
        for i in range(qubits):
            qml.Hadamard(wires=i)

        # Layer 2: Controlled rotations for added non-linearity
        for i in range(1, min(len(data), qubits)):
            qml.CRY(data.iloc[i].item(), wires=[i - 1, i])
            qml.CRZ(data.iloc[i].item(), wires=[i - 1, i])

        # # Layer 3: Reverse entanglement
        for i in range(qubits - 1, 0, -1):
            for j in range(i - 1, -1, -1):
                qml.CNOT(wires=[i, j])

        # Layer 4: More parameterized rotations
        for i in range(min(len(data), qubits)):
            qml.RX(data.iloc[i].item() * 0.8, wires=i)
            qml.RZ(data.iloc[i].item() * 1.2, wires=i)

        # Layer 6: Final rotations for classification resolution
        for i in range(min(len(data), qubits)):
            qml.RY(data.iloc[i].item(), wires=i)
            qml.RZ(data.iloc[i].item(), wires=i)



    def transform(self, data=None, feature_map=None, qubits=None, device=None):
        "Transforms classical data using a quantum feature map in Pennylane"

        if data is None:
            raise ValueError("Data cannot be None.")
        
        if qubits is None:
            qubits = len(data.columns)
            warnings.warn("The number of qubits required is not specified, by default the number of columns in the data will be used.")

        if device is None:
            device = qml.device("default.qubit", wires=qubits)

        if feature_map is None:
            feature_map = QuantumFeatureMap.default_feature_map # Try to find some universal ansatz that works as a best fit
            warnings.warn("No feature map is specified, by default a general purpose feature map will be used. It is recommended to use a custom feature map fit for the data.")
        elif type(feature_map) == str:
            feature_map = feature_map
        elif type(feature_map) != function:
            pass


        @qml.qnode(device) # Assigns circuit to device
        def quantum_circuit(data):
            feature_map(data, qubits) # Applies feature map to the qubits in the circuit
            return [qml.expval(qml.PauliZ(i)) for i in range(qubits)] # Returns measurements from the circuit

        # Iterate through the dataframe 'data' and transform each row using the quantum circuit
        transformed_data = []
        for index, row in data.iterrows():
            transformed_data.append(quantum_circuit(row))

        # Convert the transformed data into a DataFrame with the same index and new column names
        transformed_df = pd.DataFrame(transformed_data, index=data.index, columns=[f"Qubit_{i}" for i in range(qubits)])
        return transformed_df
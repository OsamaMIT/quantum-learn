import pennylane as qml
from pennylane import numpy as np
import pandas as pd


class VariationalQuantumCircuit:
    def __init__(self):
        self.params = None
        self.ansatz = None

    def generator(self, features, params, n_qubits, device, ansatz=None):
        @qml.qnode(device, diff_method="backprop")
        def circuit(features, params):
            if ansatz != None:
                return ansatz(features, params, n_qubits)
            else:
                # Default ansatz
                for i in range(n_qubits):
                    qml.Rot(features[i] * params[i][0],
                            features[i] * params[i][1],
                            features[i] * params[i][2],
                            wires=i)
                    if i < n_qubits - 1:
                        qml.CNOT(wires=[i, i + 1])
                return qml.state()
        return circuit(features, params)

    def train(self, features, labels, params=None, batch_size=32, epochs=1, n_qubits=None, device=None, ansatz=None,
                optimizer=qml.AdamOptimizer(stepsize=0.05)):

        if n_qubits == None:
            n_qubits = len(features.columns)

        if device == None:
            device = qml.device("default.qubit", wires=n_qubits)

        if params == None:
            params = np.random.randn(n_qubits, 3)

        self.ansatz = ansatz
        data = pd.concat([features, labels], axis=1)

        def fidelity_loss(output, target):
            state0 = qml.math.dm_from_state_vector(output)
            state1 = qml.math.dm_from_state_vector(target)
            error = 1 - qml.math.fidelity(state0, state1)
            return error

        def learn():
            params = np.random.randn(n_qubits, 3)
            print(params)
            for epoch in range(epochs):
                costs = []
                for start_idx in range(0, len(data), batch_size):
                    end_idx = min(start_idx + batch_size, len(data))
                    batch_data = data.iloc[start_idx:end_idx]

                    def cost_function(params):
                        total_loss = 0
                        for _, row in batch_data.iterrows():
                            total_loss += fidelity_loss(
                                self.generator([row[feature] for feature in features.columns], params, n_qubits, device, ansatz),
                                [row[label] for label in labels.columns]
                            )
                        return qml.math.mean(total_loss / len(batch_data))

                    params, cost = optimizer.step_and_cost(cost_function, params)
                    costs.append(cost)

                print(f'Epoch {epoch+1}, mean cost: {np.mean(costs)}')
                print(f'Params: {params}')
            return params

        self.params = learn()

    def predict(self, features, n_qubits=None, device=None):
        if n_qubits == None:
            n_qubits = len(features.columns)

        if device == None:
            device = qml.device("default.qubit", wires=n_qubits)

        predictions = []
        for _, row in features.iterrows():
            # Fix the fact that params is not being passed from training when predicting
            output = self.generator([row[feature] for feature in features.columns], params=self.params,
                                    n_qubits=n_qubits, device=device, ansatz=self.ansatz) 
            predictions.append(output)
        return predictions


# def simple_quantum_dataset():
#     # For a 2-qubit system, a valid state vector has 4 elements.
#     # Here we encode:
#     # 0 -> |00 = [1, 0, 0, 0]
#     # 1 -> |11 = [0, 0, 0, 1]
#     def encode(label):
#         return np.array([1, 0, 0, 0]) if label == 0 else np.array([0, 0, 0, 1])
    
#     data = pd.DataFrame({
#         "feature1": [0, 1, 0, 1],
#         "feature2": [0, 0, 1, 1],
#         "label": [encode(l) for l in [0, 1, 1, 0]]
#     })
#     return data


# data = simple_quantum_dataset()

# if __name__ == "__main__":
# # This is an example of how to use the VariationalQuantumCircuit class with the simple_quantum_dataset
#     features = data[["feature1", "feature2"]]
#     labels = data[["label"]]
#     vqc = VariationalQuantumCircuit()
#     vqc.train(features, labels, epochs=10)
#     predictions = vqc.predict(features)
#     print(f'Predictions: {predictions}')


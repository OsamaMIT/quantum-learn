from abc import ABC, abstractmethod


class QuantumFeatureMapBase(ABC):
    @abstractmethod
    def transform(self, data=None, feature_map=None, qubits=None, **runtime_options):
        """Transform classical features into quantum features."""


class VariationalQuantumCircuitBase(ABC):
    @abstractmethod
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
        **runtime_options,
    ):
        """Train the circuit parameters."""

    @abstractmethod
    def predict(self, features, n_qubits=None, **runtime_options):
        """Run inference with the trained circuit."""

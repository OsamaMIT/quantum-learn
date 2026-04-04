from .classification import HybridClassification
from .clustering import HybridClustering
from .regression import HybridRegression
from ._backends import DEFAULT_BACKEND, load_backend_attr

__all__ = [
    "VariationalQuantumCircuit",
    "QuantumFeatureMap",
    "HybridClassification",
    "HybridClustering",
    "HybridRegression",
]


def __getattr__(name):
    if name in {"QuantumFeatureMap", "VariationalQuantumCircuit"}:
        return load_backend_attr(DEFAULT_BACKEND, name)
    raise AttributeError(f"module 'qlearn' has no attribute '{name}'")

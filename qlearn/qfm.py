from ._backends import DEFAULT_BACKEND, load_backend_attr

__all__ = ["QuantumFeatureMap"]


def __getattr__(name):
    if name == "QuantumFeatureMap":
        return load_backend_attr(DEFAULT_BACKEND, name)
    raise AttributeError(f"module 'qlearn.qfm' has no attribute '{name}'")

from ._backends import DEFAULT_BACKEND, load_backend_attr

__all__ = ["VariationalQuantumCircuit"]


def __getattr__(name):
    if name == "VariationalQuantumCircuit":
        return load_backend_attr(DEFAULT_BACKEND, name)
    raise AttributeError(f"module 'qlearn.vqc' has no attribute '{name}'")

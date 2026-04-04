import importlib


DEFAULT_BACKEND = "pennylane"

_BACKEND_INSTALL_HINTS = {
    "pennylane": "Install the Pennylane backend with `pip install quantum-learn[pennylane]`.",
    "qiskit": "Install the Qiskit backend with `pip install quantum-learn[qiskit]`.",
}


def _normalize_backend_name(backend):
    if backend is None:
        return DEFAULT_BACKEND

    name = str(backend).strip().lower()
    if name in {"default", "qml"}:
        return DEFAULT_BACKEND
    if name not in _BACKEND_INSTALL_HINTS:
        raise ValueError(
            f"Unknown backend '{backend}'. Expected one of: {', '.join(sorted(_BACKEND_INSTALL_HINTS))}."
        )
    return name


def load_backend_module(backend):
    backend_name = _normalize_backend_name(backend)
    module_name = f"qlearn.{backend_name}"

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name in {backend_name, module_name, "pennylane", "qiskit"}:
            hint = _BACKEND_INSTALL_HINTS[backend_name]
            raise ImportError(
                f"The '{backend_name}' backend is not available. {hint}"
            ) from exc
        raise


def load_backend_attr(backend, attr_name):
    module = load_backend_module(backend)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ImportError(
            f"The '{backend}' backend does not provide '{attr_name}'."
        ) from exc

# quantum-learn

[![PyPI Version](https://img.shields.io/pypi/v/quantum-learn.svg)](https://pypi.org/project/quantum-learn/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/OsamaMIT/quantum-learn/blob/main/LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/quantum-learn.svg)](https://pypi.org/project/quantum-learn/)

`quantum-learn` is a small quantum machine learning library with backend-specific APIs for Pennylane and Qiskit, plus hybrid estimators that can use either backend.

## Features

- Backend-specific entry points such as `qlearn.pennylane.QuantumFeatureMap` and `qlearn.qiskit.QuantumFeatureMap`
- Hybrid estimators for classification, regression, and clustering
- Optional backend dependencies so importing `qlearn` does not require every quantum framework
- A default top-level backend for users who want a simple `qlearn.QuantumFeatureMap()` entry point

## Installation

Base install:

```bash
pip install quantum-learn
```

Install with a specific backend:

```bash
pip install "quantum-learn[pennylane]"
pip install "quantum-learn[qiskit]"
```

Install both backends:

```bash
pip install "quantum-learn[all]"
```

Install from source:

```bash
git clone https://github.com/OsamaMIT/quantum-learn.git
cd quantum-learn
pip install -e ".[pennylane]"
```

## Usage

Use a backend directly:

```python
from qlearn.pennylane import QuantumFeatureMap

qfm = QuantumFeatureMap()
transformed = qfm.transform(data)
```

Use a hybrid estimator with an explicit backend:

```python
from qlearn import HybridClassification

model = HybridClassification(backend="qiskit")
model.fit(features, labels)
predictions = model.predict(features)
```

The top-level `qlearn.QuantumFeatureMap` and `qlearn.VariationalQuantumCircuit` resolve to the default backend, which is currently Pennylane.

## API Notes

- `fit()` is the primary training method. `train()` remains available as an alias for compatibility.
- `HybridClustering.predict()` now requires a fitted model. Use `fit_predict()` when you want clustering in a single call.
- `qlearn.qiskit.VariationalQuantumCircuit` is not exported yet because it is not implemented.

## Documentation
For tutorials, examples, and details on the classes, check out the [quantum-learn documentation](https://quantum-learn.readthedocs.io/en/latest/).

## Dependencies
If you're working with the source, the required dependencies can be installed by

```bash
pip install -r requirements.txt
```

## Planned Features
- Implement quantum kernel methods
- Implement categorical feature maps

## Contributing
Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch (feature-branch)
3. Commit your changes and open a pull request

## License

This project is licensed under the MIT License. See `LICENSE` for details.

# quantum-learn

[![PyPI Version](https://img.shields.io/pypi/v/quantum-learn.svg)](https://pypi.org/project/quantum-learn/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/OsamaMIT/quantum-learn/blob/main/LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/quantum-learn.svg)](https://pypi.org/project/quantum-learn/)

`quantum-learn` is a quantum machine learning library with backend-specific APIs for Pennylane and Qiskit, plus higher-level estimators and VQC wrappers for common workflows.

## Features

- Backend-specific entry points such as `qlearn.pennylane.QuantumFeatureMap` and `qlearn.qiskit.QuantumFeatureMap`
- A generic `VariationalQuantumCircuit` with configurable measurement and loss settings
- Task-oriented `VariationalQuantumClassifier` and `VariationalQuantumRegressor` wrappers with sensible defaults
- Hybrid estimators for classification, regression, and clustering on top of quantum feature maps
- Optional backend dependencies so importing `qlearn` does not require every quantum framework
- A default top-level backend for users who want a simple `qlearn.QuantumFeatureMap()` or `qlearn.VariationalQuantumCircuit()` entry point

## Installation

Base install:

```bash
pip install quantum-learn
```

The base install is enough to import the package and use backend-independent utilities, but any quantum execution requires at least one backend extra.

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

## Backend Model

- `qlearn.pennylane` exposes the implemented Pennylane backend.
- `qlearn.qiskit` currently exposes the Qiskit `QuantumFeatureMap`.
- The top-level `qlearn.QuantumFeatureMap` and `qlearn.VariationalQuantumCircuit` resolve to the default backend, which is currently Pennylane.
- `qlearn.qiskit.VariationalQuantumCircuit` is intentionally not exported yet because it is not implemented.

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

model = HybridClassification(backend="pennylane")
model.fit(features, labels)
predictions = model.predict(features)
```

Use clustering with a sklearn-style workflow:

```python
from qlearn import HybridClustering

clusterer = HybridClustering(backend="pennylane")
labels = clusterer.fit_predict(features, n_clusters=3)
```

Use a task wrapper on top of the shared VQC:

```python
from qlearn import VariationalQuantumClassifier, VariationalQuantumRegressor

classifier = VariationalQuantumClassifier()
classifier.fit(features, labels)
predictions = classifier.predict(features)
probabilities = classifier.predict_proba(features)

regressor = VariationalQuantumRegressor()
regressor.fit(features, targets)
values = regressor.predict(features)
```

Or use the generic VQC directly with explicit output and loss choices:

```python
from qlearn import VariationalQuantumCircuit

vqc = VariationalQuantumCircuit()
vqc.fit(
    features,
    labels,
    measurement="probabilities",
    loss="cross_entropy",
)
raw_outputs = vqc.predict(features)
```

You can also override the task defaults by supplying custom target encoders, prediction decoders, probability decoders, ansatz functions, or fit-time VQC options.

## API Notes

- `fit()` is the primary training method. `train()` remains available as an alias for compatibility.
- `HybridClustering.predict()` now requires a fitted model. Use `fit_predict()` when you want clustering in a single call.
- `VariationalQuantumClassifier` defaults to probability outputs and cross-entropy training, with automatic label encoding and `predict_proba()`.
- `VariationalQuantumRegressor` defaults to expectation-value outputs and MSE training, with automatic target scaling and inverse-scaling at prediction time.
- `VariationalQuantumCircuit` now supports configurable `measurement`, `measurement_wires`, and `loss` values in `fit()`.
- Both wrappers still accept custom target encoders and decoders when you want full control.

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

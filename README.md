# Quantum-Learn

[![PyPI Version](https://img.shields.io/pypi/v/quantum-learn.svg)](https://pypi.org/project/quantum-learn/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/OsamaMIT/quantum-learn/blob/main/LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/quantum-learn.svg)](https://pypi.org/project/quantum-learn/)

**Quantum-Learn** is an open-source Python library that simplifies **Quantum Machine Learning (QML)** using **PennyLane**. Inspired by **scikit-learn**, it provides a high-level interface for creating and training **Variational Quantum Circuits (VQCs)** with ease.

## Features

- **Simple API** for training quantum models  
- Currently only supports **Variational Quantum Circuits (VQC)**  
- Works with **PennyLane**, **scikit-learn**, and standard ML tools  
- All aspects of a VQC are customizable

## Installation

Quantum-Learn requires **Python 3.6+**. Install it via pip:

```bash
pip install quantum-learn
```

Or install from source:

```bash
git clone https://github.com/OsamaMIT/quantum-learn.git
cd quantum-learn
pip install .
```

## Quick Start
### Train a Quantum Model
```python
Copy
Edit
import pennylane as qml
import pandas as pd
from quantum_learn import VariationalQuantumCircuit

# Create a sample dataset
features = pd.DataFrame({
    "feature1": [0, 1],
    "feature2": [1, 0]
})
labels = pd.DataFrame({
    "label": [
        [1, 0, 0, 0],  # Encoded quantum state for class 0
        [0, 0, 0, 1]   # Encoded quantum state for class 1
    ]
})

# Initialize and train the model
vqc = VariationalQuantumCircuit()
vqc.train(features, labels, epochs=5)

# Make predictions
predictions = vqc.predict(features)
print(predictions)
```

## Documentation
For detailed usage and examples, check out the Quantum-Learn Documentation.

## Roadmap
- Implement quantum kernel methods
- Improve compatibility with scikit-learn
- Implement more abstract classes for applied QML tasks (similar to fastai)

## Contributing
Contributions are welcome! To contribute:

Fork the repository
Create a new branch (feature-branch)
Commit your changes and open a pull request

## License
This project is licensed under the MIT License. See the LICENSE file for details.

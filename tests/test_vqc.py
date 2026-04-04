import unittest

import numpy as np
import pandas as pd

import qlearn
from helpers import backend_available

HAS_PENNYLANE = backend_available("pennylane")
HAS_QISKIT = backend_available("qiskit")


def simple_quantum_dataset():
    def encode(label):
        return np.array([1, 0, 0, 0]) if label == 0 else np.array([0, 0, 0, 1])

    return pd.DataFrame(
        {
            "feature1": [0, 1, 0, 1],
            "feature2": [0, 0, 1, 1],
            "label": [encode(label) for label in [0, 1, 1, 0]],
        }
    )


class TestTopLevelVariationalQuantumCircuit(unittest.TestCase):
    @unittest.skipUnless(HAS_PENNYLANE, "pennylane backend not available")
    def test_default_backend_class_available(self):
        variational_quantum_circuit = getattr(qlearn, "VariationalQuantumCircuit")
        self.assertTrue(callable(variational_quantum_circuit))

    @unittest.skipIf(HAS_PENNYLANE, "pennylane backend is installed")
    def test_default_backend_class_missing_dependency(self):
        with self.assertRaises(ImportError) as exc:
            getattr(qlearn, "VariationalQuantumCircuit")
        self.assertIn("quantum-learn[pennylane]", str(exc.exception))


@unittest.skipUnless(HAS_PENNYLANE, "pennylane backend not available")
class TestPennylaneVariationalQuantumCircuit(unittest.TestCase):
    def setUp(self):
        from qlearn.pennylane import VariationalQuantumCircuit

        data = simple_quantum_dataset()
        self.features = data[["feature1", "feature2"]]
        self.labels = data[["label"]]
        self.vqc = VariationalQuantumCircuit()

    def test_fit_returns_self(self):
        returned = self.vqc.fit(self.features, self.labels, epochs=2)
        self.assertIs(returned, self.vqc)
        self.assertIsNotNone(self.vqc.params)
        self.assertEqual(self.vqc.params.shape[0], len(self.features.columns))

    def test_train_alias(self):
        returned = self.vqc.train(self.features, self.labels, epochs=2)
        self.assertIs(returned, self.vqc)

    def test_predict(self):
        self.vqc.fit(self.features, self.labels, epochs=2)
        predictions = self.vqc.predict(self.features)
        self.assertEqual(len(predictions), len(self.features))

    def test_predict_without_training(self):
        with self.assertRaises(ValueError):
            self.vqc.predict(self.features)

    def test_invalid_features(self):
        with self.assertRaises(ValueError):
            self.vqc.fit(None, self.labels, epochs=2)

    def test_respects_initial_params_shape(self):
        initial_params = np.zeros((2, 3))
        self.vqc.fit(self.features, self.labels, params=initial_params, epochs=1)
        self.assertEqual(self.vqc.params.shape, initial_params.shape)


class TestQiskitVariationalQuantumCircuit(unittest.TestCase):
    @unittest.skipUnless(HAS_QISKIT, "qiskit backend not available")
    def test_qiskit_vqc_not_exported(self):
        import qlearn.qiskit as qiskit_backend

        self.assertFalse(hasattr(qiskit_backend, "VariationalQuantumCircuit"))


if __name__ == "__main__":
    unittest.main()

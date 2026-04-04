import unittest
import warnings

import pandas as pd

import qlearn
from helpers import backend_available

HAS_PENNYLANE = backend_available("pennylane")
HAS_QISKIT = backend_available("qiskit")


def simple_qfm_dataset():
    return pd.DataFrame(
        {
            "feature1": [0.1, 0.2, 0.3],
            "feature2": [0.4, 0.5, 0.6],
        }
    )


class TestTopLevelQuantumFeatureMap(unittest.TestCase):
    def test_import_is_lazy(self):
        self.assertTrue(hasattr(qlearn, "HybridClassification"))

    @unittest.skipUnless(HAS_PENNYLANE, "pennylane backend not available")
    def test_default_backend_class_available(self):
        quantum_feature_map = getattr(qlearn, "QuantumFeatureMap")
        self.assertTrue(callable(quantum_feature_map))

    @unittest.skipIf(HAS_PENNYLANE, "pennylane backend is installed")
    def test_default_backend_class_missing_dependency(self):
        with self.assertRaises(ImportError) as exc:
            getattr(qlearn, "QuantumFeatureMap")
        self.assertIn("quantum-learn[pennylane]", str(exc.exception))


@unittest.skipUnless(HAS_PENNYLANE, "pennylane backend not available")
class TestPennylaneQuantumFeatureMap(unittest.TestCase):
    def setUp(self):
        from qlearn.pennylane import QuantumFeatureMap

        self.qfm = QuantumFeatureMap()
        self.data = simple_qfm_dataset()

    def test_transform(self):
        transformed = self.qfm.transform(self.data, qubits=2)
        self.assertIsInstance(transformed, pd.DataFrame)
        self.assertEqual(list(transformed.columns), ["Qubit_0", "Qubit_1"])

    def test_transform_no_data(self):
        with self.assertRaises(ValueError):
            self.qfm.transform(None, qubits=2)

    def test_transform_qubits_default_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            transformed = self.qfm.transform(self.data)
        self.assertIsInstance(transformed, pd.DataFrame)
        self.assertTrue(any("number of qubits" in str(item.message).lower() for item in caught))

    def test_transform_invalid_feature_map_name(self):
        with self.assertRaises(ValueError):
            self.qfm.transform(self.data, feature_map="nope", qubits=2)

    def test_default_feature_map_callable(self):
        transformed = self.qfm.transform(
            self.data,
            feature_map=self.qfm.default_feature_map,
            qubits=2,
        )
        self.assertIsInstance(transformed, pd.DataFrame)


@unittest.skipUnless(HAS_QISKIT, "qiskit backend not available")
class TestQiskitQuantumFeatureMap(unittest.TestCase):
    def setUp(self):
        from qlearn.qiskit import QuantumFeatureMap

        self.qfm = QuantumFeatureMap()
        self.data = simple_qfm_dataset()

    def test_transform(self):
        transformed = self.qfm.transform(self.data, qubits=2)
        self.assertIsInstance(transformed, pd.DataFrame)
        self.assertEqual(list(transformed.columns), ["Qubit_0", "Qubit_1"])

    def test_feature_map_sets_qc_attribute(self):
        from qiskit import QuantumCircuit

        def feature_map(row, qubits):
            qc = QuantumCircuit(qubits)
            qc.h(0)
            feature_map.qc = qc

        transformed = self.qfm.transform(self.data, feature_map=feature_map, qubits=2)
        self.assertIsInstance(transformed, pd.DataFrame)

    def test_feature_map_returns_qc(self):
        from qiskit import QuantumCircuit

        def feature_map(row, qubits):
            qc = QuantumCircuit(qubits)
            qc.h(0)
            return qc

        transformed = self.qfm.transform(self.data, feature_map=feature_map, qubits=2)
        self.assertIsInstance(transformed, pd.DataFrame)

    def test_feature_map_wrong_signature(self):
        def feature_map(row):
            return row

        with self.assertRaises(TypeError):
            self.qfm.transform(self.data, feature_map=feature_map, qubits=2)

    def test_feature_map_invalid_return(self):
        def feature_map(row, qubits):
            return None

        with self.assertRaises(TypeError):
            self.qfm.transform(self.data, feature_map=feature_map, qubits=2)


if __name__ == "__main__":
    unittest.main()

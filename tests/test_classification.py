import unittest

import pandas as pd
from sklearn.svm import SVC

from helpers import IdentityFeatureMap, backend_available
from qlearn import HybridClassification


def simple_quantum_dataset():
    return pd.DataFrame(
        {
            "feature1": [0, 1, 0, 1],
            "feature2": [0, 0, 1, 1],
            "label": [0, 1, 1, 0],
        }
    )


class TestClassification(unittest.TestCase):
    def setUp(self):
        data = simple_quantum_dataset()
        self.features = data[["feature1", "feature2"]]
        self.labels = data[["label"]]
        self.classifier = HybridClassification(transformer=IdentityFeatureMap())

    def test_fit_returns_self(self):
        returned = self.classifier.fit(self.features, self.labels)
        self.assertIs(returned, self.classifier)
        self.assertIsNotNone(self.classifier.model)

    def test_train_alias(self):
        returned = self.classifier.train(self.features, self.labels)
        self.assertIs(returned, self.classifier)

    def test_predict(self):
        self.classifier.fit(self.features, self.labels)
        predictions = self.classifier.predict(self.features)
        self.assertEqual(len(predictions), len(self.features))

    def test_predict_without_training(self):
        with self.assertRaises(ValueError):
            self.classifier.predict(self.features)

    def test_custom_model(self):
        model = SVC(kernel="linear", random_state=123)
        self.classifier.fit(self.features, self.labels, model=model)
        self.assertIs(self.classifier.model, model)

    def test_invalid_backend_name(self):
        classifier = HybridClassification(backend="unknown-backend")
        with self.assertRaises(ValueError):
            classifier.fit(self.features, self.labels)

    @unittest.skipIf(backend_available("pennylane"), "pennylane is installed")
    def test_missing_default_backend_raises_helpful_error(self):
        classifier = HybridClassification()
        with self.assertRaises(ImportError) as exc:
            classifier.fit(self.features, self.labels)
        self.assertIn("quantum-learn[pennylane]", str(exc.exception))


if __name__ == "__main__":
    unittest.main()

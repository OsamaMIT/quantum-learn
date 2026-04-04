import unittest

import pandas as pd
from sklearn.linear_model import Lasso

from helpers import IdentityFeatureMap, backend_available
from qlearn import HybridRegression


def simple_regression_dataset():
    return pd.DataFrame(
        {
            "feature1": [0, 1, 2, 3],
            "feature2": [1, 2, 3, 4],
            "label": [0.0, 1.0, 2.0, 3.0],
        }
    )


class TestRegression(unittest.TestCase):
    def setUp(self):
        data = simple_regression_dataset()
        self.features = data[["feature1", "feature2"]]
        self.labels = data[["label"]]
        self.regressor = HybridRegression(transformer=IdentityFeatureMap())

    def test_fit_returns_self(self):
        returned = self.regressor.fit(self.features, self.labels)
        self.assertIs(returned, self.regressor)
        self.assertIsNotNone(self.regressor.model)

    def test_train_alias(self):
        returned = self.regressor.train(self.features, self.labels)
        self.assertIs(returned, self.regressor)

    def test_predict(self):
        self.regressor.fit(self.features, self.labels)
        predictions = self.regressor.predict(self.features)
        self.assertEqual(len(predictions), len(self.features))

    def test_predict_without_training(self):
        with self.assertRaises(ValueError):
            self.regressor.predict(self.features)

    def test_custom_model(self):
        model = Lasso(random_state=123)
        self.regressor.fit(self.features, self.labels, model=model)
        self.assertIs(self.regressor.model, model)

    def test_invalid_backend_name(self):
        regressor = HybridRegression(backend="unknown-backend")
        with self.assertRaises(ValueError):
            regressor.fit(self.features, self.labels)

    @unittest.skipIf(backend_available("pennylane"), "pennylane is installed")
    def test_missing_default_backend_raises_helpful_error(self):
        regressor = HybridRegression()
        with self.assertRaises(ImportError) as exc:
            regressor.fit(self.features, self.labels)
        self.assertIn("quantum-learn[pennylane]", str(exc.exception))


if __name__ == "__main__":
    unittest.main()

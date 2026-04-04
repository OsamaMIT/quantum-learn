import unittest

import numpy as np
import pandas as pd

from helpers import IdentityFeatureMap, backend_available
from qlearn import HybridClustering


def simple_dataset():
    return pd.DataFrame(
        {
            "feature1": [0, 1, 0, 1, 2],
            "feature2": [0, 0, 1, 1, 2],
        }
    )


class TestClustering(unittest.TestCase):
    def setUp(self):
        self.data = simple_dataset()
        self.clusterer = HybridClustering(transformer=IdentityFeatureMap())

    def test_fit_returns_self(self):
        returned = self.clusterer.fit(self.data, n_clusters=2)
        self.assertIs(returned, self.clusterer)
        self.assertIsNotNone(self.clusterer.model)

    def test_train_alias(self):
        returned = self.clusterer.train(self.data, n_clusters=2)
        self.assertIs(returned, self.clusterer)

    def test_fit_predict(self):
        clusters = self.clusterer.fit_predict(self.data, n_clusters=2)
        self.assertEqual(len(clusters), len(self.data))
        self.assertIsInstance(clusters, np.ndarray)

    def test_fit_predict_no_clusters(self):
        clusters = self.clusterer.fit_predict(self.data, n_clusters=0)
        self.assertEqual(len(clusters), len(self.data))
        self.assertIsInstance(clusters, np.ndarray)

    def test_predict_after_fit(self):
        self.clusterer.fit(self.data, n_clusters=2)
        clusters = self.clusterer.predict(self.data)
        self.assertEqual(len(clusters), len(self.data))

    def test_predict_without_training(self):
        with self.assertRaises(ValueError):
            self.clusterer.predict(self.data)

    @unittest.skipIf(backend_available("pennylane"), "pennylane is installed")
    def test_missing_default_backend_raises_helpful_error(self):
        clusterer = HybridClustering()
        with self.assertRaises(ImportError) as exc:
            clusterer.fit(self.data, n_clusters=2)
        self.assertIn("quantum-learn[pennylane]", str(exc.exception))


if __name__ == "__main__":
    unittest.main()

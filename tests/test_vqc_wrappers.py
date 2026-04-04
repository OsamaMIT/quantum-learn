import unittest

import numpy as np
import pandas as pd

from qlearn import VariationalQuantumClassifier, VariationalQuantumRegressor


class DummyVQC:
    def __init__(self, outputs=None):
        self.outputs = outputs or []
        self.fit_features = None
        self.fit_labels = None
        self.fit_kwargs = None
        self.predict_features = None
        self.predict_kwargs = None

    def fit(self, features, labels, **kwargs):
        self.fit_features = features
        self.fit_labels = labels
        self.fit_kwargs = kwargs
        return self

    def predict(self, features, **kwargs):
        self.predict_features = features
        self.predict_kwargs = kwargs
        return self.outputs


def simple_dataset():
    return pd.DataFrame(
        {
            "feature1": [0.1, 0.2, 0.3],
            "feature2": [0.4, 0.5, 0.6],
            "label": ["a", "b", "a"],
            "target": [1.0, 2.0, 3.0],
        }
    )


class TestVariationalQuantumClassifier(unittest.TestCase):
    def setUp(self):
        self.data = simple_dataset()
        self.features = self.data[["feature1", "feature2"]]
        self.labels = self.data[["label"]]

    def test_default_fit_is_usable_without_custom_hooks(self):
        dummy_vqc = DummyVQC()
        classifier = VariationalQuantumClassifier(vqc=dummy_vqc, fit_kwargs={"epochs": 3})

        returned = classifier.fit(self.features, self.labels)

        self.assertIs(returned, classifier)
        self.assertEqual(classifier.classes_.tolist(), ["a", "b"])
        self.assertEqual(dummy_vqc.fit_kwargs["epochs"], 3)
        self.assertEqual(dummy_vqc.fit_kwargs["measurement"], "probabilities")
        self.assertEqual(dummy_vqc.fit_kwargs["loss"], "cross_entropy")
        self.assertEqual(dummy_vqc.fit_kwargs["n_qubits"], 2)
        np.testing.assert_array_equal(dummy_vqc.fit_labels[0], np.array([1.0, 0.0]))
        np.testing.assert_array_equal(dummy_vqc.fit_labels[1], np.array([0.0, 1.0]))

    def test_default_predict_and_predict_proba(self):
        dummy_vqc = DummyVQC(outputs=[[0.7, 0.3], [0.2, 0.8], [0.55, 0.45]])
        classifier = VariationalQuantumClassifier(vqc=dummy_vqc)
        classifier.fit(self.features, self.labels)

        probabilities = classifier.predict_proba(self.features)
        predictions = classifier.predict(self.features)

        self.assertEqual(probabilities.shape, (3, 2))
        np.testing.assert_array_equal(predictions, np.array(["a", "b", "a"], dtype=object))
        self.assertIs(dummy_vqc.predict_features, self.features)

    def test_predict_proba_renormalizes_unused_basis_states(self):
        dummy_vqc = DummyVQC(outputs=[[0.2, 0.3, 0.5, 0.0]])
        classifier = VariationalQuantumClassifier(vqc=dummy_vqc)
        classifier.fit(self.features, self.labels)

        probabilities = classifier.predict_proba(self.features)

        np.testing.assert_allclose(probabilities, np.array([[0.4, 0.6]]))

    def test_custom_hooks_override_defaults(self):
        dummy_vqc = DummyVQC(outputs=[0.1, 0.9, 0.2])

        def encode_targets(labels):
            return [f"encoded:{label}" for label in labels.iloc[:, 0].tolist()]

        def decode_outputs(outputs, classes=None):
            return [classes[1] if value > 0.5 else classes[0] for value in outputs]

        classifier = VariationalQuantumClassifier(
            vqc=dummy_vqc,
            target_encoder=encode_targets,
            prediction_decoder=decode_outputs,
        )
        classifier.fit(self.features, self.labels)

        predictions = classifier.predict(self.features)

        self.assertEqual(
            dummy_vqc.fit_labels,
            ["encoded:a", "encoded:b", "encoded:a"],
        )
        self.assertEqual(predictions, ["a", "b", "a"])

    def test_train_alias(self):
        dummy_vqc = DummyVQC()
        classifier = VariationalQuantumClassifier(vqc=dummy_vqc)

        returned = classifier.train(self.features, self.labels)

        self.assertIs(returned, classifier)

    def test_predict_raw_before_fit_raises(self):
        classifier = VariationalQuantumClassifier()
        with self.assertRaises(ValueError):
            classifier.predict_raw(self.features)

    def test_invalid_backend_name_raises_on_fit(self):
        classifier = VariationalQuantumClassifier(backend="unknown-backend")
        with self.assertRaises(ValueError):
            classifier.fit(self.features, self.labels)

    def test_raises_when_classes_exceed_available_states(self):
        labels = pd.DataFrame({"label": ["a", "b", "c", "d", "e"]})
        features = pd.DataFrame({"feature1": [0, 1, 2, 3, 4]})
        classifier = VariationalQuantumClassifier(vqc=DummyVQC(), fit_kwargs={"n_qubits": 2})

        with self.assertRaises(ValueError):
            classifier.fit(features, labels)


class TestVariationalQuantumRegressor(unittest.TestCase):
    def setUp(self):
        self.data = simple_dataset()
        self.features = self.data[["feature1", "feature2"]]
        self.targets = self.data[["target"]]

    def test_default_fit_scales_targets_and_sets_regression_defaults(self):
        dummy_vqc = DummyVQC()
        regressor = VariationalQuantumRegressor(vqc=dummy_vqc, fit_kwargs={"epochs": 5})

        returned = regressor.fit(self.features, self.targets)

        self.assertIs(returned, regressor)
        self.assertEqual(dummy_vqc.fit_kwargs["epochs"], 5)
        self.assertEqual(dummy_vqc.fit_kwargs["measurement"], "expectation_z")
        self.assertEqual(dummy_vqc.fit_kwargs["loss"], "mse")
        self.assertEqual(dummy_vqc.fit_kwargs["measurement_wires"], [0])
        np.testing.assert_allclose(dummy_vqc.fit_labels, np.array([[-1.0], [0.0], [1.0]]))

    def test_default_predict_inverse_scales_outputs(self):
        dummy_vqc = DummyVQC(outputs=[[-1.0], [0.0], [1.0]])
        regressor = VariationalQuantumRegressor(vqc=dummy_vqc)
        regressor.fit(self.features, self.targets)

        predictions = regressor.predict(self.features)

        np.testing.assert_allclose(predictions, np.array([1.0, 2.0, 3.0]))

    def test_custom_decoder_overrides_default(self):
        dummy_vqc = DummyVQC(outputs=[1.25, 2.5, 3.75])
        regressor = VariationalQuantumRegressor(
            vqc=dummy_vqc,
            target_encoder=lambda labels: labels,
            prediction_decoder=lambda outputs: [round(value, 1) for value in outputs],
        )
        regressor.fit(self.features, self.targets)

        predictions = regressor.predict(self.features)

        self.assertEqual(predictions, [1.2, 2.5, 3.8])

    def test_multi_output_regression_defaults(self):
        targets = pd.DataFrame(
            {
                "target1": [1.0, 2.0, 3.0],
                "target2": [10.0, 20.0, 30.0],
            }
        )
        dummy_vqc = DummyVQC(outputs=[[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        regressor = VariationalQuantumRegressor(vqc=dummy_vqc)
        regressor.fit(self.features, targets)

        predictions = regressor.predict(self.features)

        np.testing.assert_allclose(
            predictions,
            np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
        )

    def test_train_alias(self):
        dummy_vqc = DummyVQC()
        regressor = VariationalQuantumRegressor(vqc=dummy_vqc)

        returned = regressor.train(self.features, self.targets)

        self.assertIs(returned, regressor)

    def test_predict_raw_before_fit_raises(self):
        regressor = VariationalQuantumRegressor()
        with self.assertRaises(ValueError):
            regressor.predict_raw(self.features)

    def test_invalid_backend_name_raises_on_fit(self):
        regressor = VariationalQuantumRegressor(backend="unknown-backend")
        with self.assertRaises(ValueError):
            regressor.fit(self.features, self.targets)


if __name__ == "__main__":
    unittest.main()

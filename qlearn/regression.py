from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor

from ._backends import DEFAULT_BACKEND, load_backend_attr
from ._utils import normalize_targets


class HybridRegression:
    def __init__(
        self,
        model=None,
        backend=DEFAULT_BACKEND,
        transformer=None,
        transform_kwargs=None,
    ):
        self.model = model
        self.backend = backend
        self.transformer = transformer
        self.transform_kwargs = dict(transform_kwargs or {})

    def _get_transformer(self):
        if self.transformer is None:
            transformer_class = load_backend_attr(self.backend, "QuantumFeatureMap")
            self.transformer = transformer_class()
        return self.transformer

    def _transform(self, features, **transform_kwargs):
        kwargs = dict(self.transform_kwargs)
        kwargs.update(transform_kwargs)
        transformer = self._get_transformer()
        return transformer.transform(features, **kwargs)

    def fit(self, features, labels, model=None, random_state=42, **transform_kwargs):
        selected_model = model or self.model
        if selected_model is None:
            if len(features) < 100000:
                selected_model = Lasso(random_state=random_state)
            else:
                selected_model = SGDRegressor(random_state=random_state)

        targets = normalize_targets(labels)
        transformed_features = self._transform(features, **transform_kwargs)
        selected_model.fit(transformed_features, targets)

        self.model = selected_model
        return self

    def train(self, features, labels, model=None, random_state=42, **transform_kwargs):
        return self.fit(
            features,
            labels,
            model=model,
            random_state=random_state,
            **transform_kwargs,
        )

    def predict(self, features, **transform_kwargs):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        transformed_features = self._transform(features, **transform_kwargs)
        return self.model.predict(transformed_features)

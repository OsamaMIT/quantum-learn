import warnings

from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import MiniBatchKMeans

from ._backends import DEFAULT_BACKEND, load_backend_attr


class HybridClustering:
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

    def _transform(self, data, **transform_kwargs):
        kwargs = dict(self.transform_kwargs)
        kwargs.update(transform_kwargs)
        transformer = self._get_transformer()
        return transformer.transform(data, **kwargs)

    def _select_model(self, data, n_clusters=0, model=None, random_state=42):
        selected_model = model or self.model
        if selected_model is not None:
            return selected_model

        if n_clusters == 0:
            if len(data) > 10000:
                warnings.warn(
                    "MeanShift is recommended for datasets with fewer than 10,000 samples. "
                    "Clustering without a fixed number of clusters may be slow for larger datasets."
                )
            return MeanShift()

        if len(data) < 10000:
            return KMeans(n_clusters=n_clusters, random_state=random_state)
        return MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state)

    def fit(self, data, n_clusters=0, model=None, random_state=42, **transform_kwargs):
        selected_model = self._select_model(
            data,
            n_clusters=n_clusters,
            model=model,
            random_state=random_state,
        )
        transformed_data = self._transform(data, **transform_kwargs)
        selected_model.fit(transformed_data)
        self.model = selected_model
        return self

    def train(self, data, n_clusters=0, model=None, random_state=42, **transform_kwargs):
        return self.fit(
            data,
            n_clusters=n_clusters,
            model=model,
            random_state=random_state,
            **transform_kwargs,
        )

    def fit_predict(
        self,
        data,
        n_clusters=0,
        model=None,
        random_state=42,
        **transform_kwargs,
    ):
        selected_model = self._select_model(
            data,
            n_clusters=n_clusters,
            model=model,
            random_state=random_state,
        )
        transformed_data = self._transform(data, **transform_kwargs)
        labels = selected_model.fit_predict(transformed_data)
        self.model = selected_model
        return labels

    def predict(self, data, **transform_kwargs):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        if not hasattr(self.model, "predict"):
            raise AttributeError(
                "The fitted clustering model does not support predict(). Use fit_predict() instead."
            )

        transformed_data = self._transform(data, **transform_kwargs)
        return self.model.predict(transformed_data)

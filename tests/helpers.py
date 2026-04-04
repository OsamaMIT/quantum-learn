import importlib.util


class IdentityFeatureMap:
    def transform(self, data, **_kwargs):
        return data.copy()


def backend_available(module_name):
    return importlib.util.find_spec(module_name) is not None

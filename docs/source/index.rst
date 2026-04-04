quantum-learn documentation
===========================

`quantum-learn` provides backend-specific quantum APIs and higher-level hybrid estimators that can be configured with either Pennylane or Qiskit feature maps.

Features
--------

- Backend-specific APIs such as ``qlearn.pennylane.QuantumFeatureMap`` and ``qlearn.qiskit.QuantumFeatureMap``
- Hybrid estimators for classification, regression, and clustering
- Optional backend dependencies so importing ``qlearn`` does not require every quantum framework

Quickstart
----------

Install a backend before using its quantum classes:

.. code-block:: bash

   pip install "quantum-learn[pennylane]"

Use a hybrid estimator with an explicit backend:

.. code-block:: python

   from qlearn import HybridClassification

   clf = HybridClassification(backend="pennylane")
   clf.fit(features, labels)
   predictions = clf.predict(features)

Use a backend directly:

.. code-block:: python

   from qlearn.qiskit import QuantumFeatureMap

   qfm = QuantumFeatureMap()
   transformed = qfm.transform(features)

API Reference
-------------

.. toctree::

   api

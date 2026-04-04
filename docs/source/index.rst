quantum-learn documentation
===========================

`quantum-learn` provides backend-specific quantum APIs and higher-level estimators built on top of quantum feature maps and variational quantum circuits.

Features
--------

- Backend-specific APIs such as ``qlearn.pennylane.QuantumFeatureMap`` and ``qlearn.qiskit.QuantumFeatureMap``
- A generic ``VariationalQuantumCircuit`` with configurable measurements and losses
- Hybrid estimators for classification, regression, and clustering
- Task-oriented VQC wrappers with built-in defaults for classification and regression
- Optional backend dependencies so importing ``qlearn`` does not require every quantum framework

Quickstart
----------

Install a backend before using its quantum classes:

.. code-block:: bash

   pip install "quantum-learn[pennylane]"

The base install does not include a quantum execution backend. Use a backend extra whenever you want to run quantum feature maps or variational circuits.

Backend structure
-----------------

- ``qlearn.pennylane`` exposes the implemented Pennylane backend.
- ``qlearn.qiskit`` currently exposes the Qiskit ``QuantumFeatureMap``.
- Top-level classes such as ``qlearn.QuantumFeatureMap`` and ``qlearn.VariationalQuantumCircuit`` resolve to the default backend, which is currently Pennylane.

Use a hybrid estimator with an explicit backend:

.. code-block:: python

   from qlearn import HybridClassification

   clf = HybridClassification(backend="pennylane")
   clf.fit(features, labels)
   predictions = clf.predict(features)

Use clustering with a sklearn-style workflow:

.. code-block:: python

   from qlearn import HybridClustering

   clusterer = HybridClustering(backend="pennylane")
   labels = clusterer.fit_predict(features, n_clusters=3)

Use a backend directly:

.. code-block:: python

   from qlearn.qiskit import QuantumFeatureMap

   qfm = QuantumFeatureMap()
   transformed = qfm.transform(features)

Use a VQC wrapper with built-in defaults:

.. code-block:: python

   from qlearn import VariationalQuantumClassifier

   classifier = VariationalQuantumClassifier()
   classifier.fit(features, labels)
   predictions = classifier.predict(features)
   probabilities = classifier.predict_proba(features)

Use the generic VQC directly when you want explicit control over outputs and loss:

.. code-block:: python

   from qlearn import VariationalQuantumCircuit

   vqc = VariationalQuantumCircuit()
   vqc.fit(
       features,
       labels,
       measurement="probabilities",
       loss="cross_entropy",
   )
   raw_outputs = vqc.predict(features)

API Reference
-------------

.. toctree::

   api

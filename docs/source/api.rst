API Reference
=============

Package Root
------------

The root package exposes the default backend wrappers and the backend-independent estimator classes.

.. automodule:: qlearn
   :members:
   :undoc-members:
   :show-inheritance:

Default Backend Wrappers
------------------------

The top-level modules resolve to the default backend, which is currently Pennylane.

.. automodule:: qlearn.qfm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qlearn.vqc
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qlearn.vqc_classifier
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qlearn.vqc_regressor
   :members:
   :undoc-members:
   :show-inheritance:

These wrappers sit on top of the generic variational circuit API:

- ``qlearn.vqc``: default-backend ``VariationalQuantumCircuit``
- ``qlearn.vqc_classifier``: automatic class encoding plus probability-based decoding
- ``qlearn.vqc_regressor``: automatic target scaling plus inverse-scaling at prediction time

Hybrid Estimators
-----------------

.. automodule:: qlearn.classification
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qlearn.clustering
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qlearn.regression
   :members:
   :undoc-members:
   :show-inheritance:

Pennylane Backend
-----------------

.. automodule:: qlearn.pennylane.qfm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qlearn.pennylane.vqc
   :members:
   :undoc-members:
   :show-inheritance:

Qiskit Backend
--------------

.. automodule:: qlearn.qiskit.qfm
   :members:
   :undoc-members:
   :show-inheritance:

The Qiskit ``VariationalQuantumCircuit`` is not included in the public API yet because it is not implemented.

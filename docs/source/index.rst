.. You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

quantum-learn documentation
===========================

Welcome to the **quantum-learn** documentation! quantum-learn is a quantum machine learning library that bridges classical machine learning with quantum feature mapping, offering abstract hybrid models for classification, regression, clustering, and variational quantum circuits. This documentation is designed to help you quickly get started and explore the API of Quantum-Learn.

**Features**
--------
*Pure Quantum*

- **Variational Quantum Circuits**: Build and train quantum circuits with customizable ansätze.

*Hybrid Quantum*

- **Quantum Feature Mapping**: Transform your classical data into a quantum feature space.
- Abstracted Hybrid-Quantum models that build on scikit-learn, including:
   - **Hybrid Regression**
   - **Hybrid Classification**
   - **Hybrid Clustering**

**API Reference**
-------------
For detailed API documentation, you can refer to one of the pages below:

.. toctree::

   api


**Quickstart**
----------
For a quick start, you can import the library as follows:

.. code-block:: python

    from qlearn import HybridClassification, HybridRegression, HybridClustering, VariationalQuantumCircuit, QuantumFeatureMap

    # Example for hybrid classification:
    clf = HybridClassification()
    clf.train(features, labels) # Where <features> and <labels> are dataframes containing your training data
    predictions = clf.predict(features)
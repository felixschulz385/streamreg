.. filepath: /scicore/home/meiera/schulz0022/projects/streamreg/docs/api.rst

API Reference
=============

This page documents the public API for StreamReg.

High-Level Estimators
---------------------

These classes provide the main user interface for regression estimation.

OLS
~~~

.. autoclass:: streamreg.api.OLS
   :members:
   :undoc-members:
   :show-inheritance:

   **Main Interface for OLS Estimation**
   
   The OLS class provides ordinary least squares estimation with support for:
   
   * Streaming/chunked data processing for large datasets
   * R-style formula interface with transformations
   * Cluster-robust standard errors (one-way and two-way)
   * Feature engineering (polynomials, interactions)
   * Multiple output formats
   
   **Example:**
   
   .. code-block:: python
   
      from streamreg.api import OLS
      
      model = OLS(formula="y ~ x1 + x2 + I(x1^2)")
      model.fit(data, cluster=['country', 'year'])
      print(model.summary())
      model.results_.save("output/my_analysis")

TwoSLS
~~~~~~

.. autoclass:: streamreg.api.TwoSLS
   :members:
   :undoc-members:
   :show-inheritance:

   **Instrumental Variables Estimation**
   
   The TwoSLS class provides two-stage least squares estimation for causal inference
   when endogeneity is present.
   
   **Example:**
   
   .. code-block:: python
   
      from streamreg.api import TwoSLS
      
      model = TwoSLS(
          formula="y ~ x1 + x2 | z1 + z2 + z3",
          endogenous=['x1']
      )
      model.fit(data, cluster='country')
      print(model.summary(stage='all'))

Results
-------

RegressionResults
~~~~~~~~~~~~~~~~~

.. autoclass:: streamreg.results.RegressionResults
   :members:
   :undoc-members:
   :show-inheritance:

   **Standardized Results Container**
   
   This class stores all regression results and provides methods for:
   
   * Summary tables and statistics
   * Coefficient extraction and confidence intervals
   * Saving results in multiple formats (JSON, CSV, LaTeX, etc.)
   * Serialization and export

Data Interface
--------------

StreamData
~~~~~~~~~~

.. autoclass:: streamreg.data.StreamData
   :members:
   :undoc-members:
   :show-inheritance:

   **Unified Data Interface**
   
   StreamData provides a consistent interface for different data sources:
   
   * Pandas DataFrame (in-memory)
   * Single parquet file
   * Partitioned parquet datasets
   
   Handles chunking and parallel processing transparently.

DatasetInfo
~~~~~~~~~~~

.. autoclass:: streamreg.data.DatasetInfo
   :members:
   :undoc-members:
   :show-inheritance:

   **Dataset Metadata**
   
   Contains information about dataset structure, columns, and source type.

Formula Parsing
---------------

FormulaParser
~~~~~~~~~~~~~

.. autoclass:: streamreg.formula.FormulaParser
   :members:
   :undoc-members:
   :show-inheritance:

   **R-style Formula Parser**
   
   Parses formulas with support for:
   
   * Basic terms: ``y ~ x1 + x2``
   * Interactions: ``x1:x2`` or ``x1*x2``
   * Polynomials: ``I(x^2)`` or ``poly(x, 3)``
   * Instruments: ``y ~ x1 | z1 + z2``
   * Intercept control: ``y ~ x - 1``

Feature Engineering
-------------------

FeatureTransformer
~~~~~~~~~~~~~~~~~~

.. autoclass:: streamreg.transforms.FeatureTransformer
   :members:
   :undoc-members:
   :show-inheritance:

   **Feature Engineering Pipeline**
   
   Applies transformations including:
   
   * Polynomial terms
   * Interaction terms
   * Custom transformations
   * Predicted value substitution (for 2SLS)

Output Formatters
-----------------

These classes handle saving results in different formats.

BaseFormatter
~~~~~~~~~~~~~

.. autoclass:: streamreg.output.BaseFormatter
   :members:
   :undoc-members:
   :show-inheritance:

JSONFormatter
~~~~~~~~~~~~~

.. autoclass:: streamreg.output.JSONFormatter
   :members:
   :undoc-members:
   :show-inheritance:

   Saves results as machine-readable JSON with all statistics and metadata.

CSVFormatter
~~~~~~~~~~~~

.. autoclass:: streamreg.output.CSVFormatter
   :members:
   :undoc-members:
   :show-inheritance:

   Exports coefficient table as CSV for spreadsheet analysis.

SummaryFormatter
~~~~~~~~~~~~~~~~

.. autoclass:: streamreg.output.SummaryFormatter
   :members:
   :undoc-members:
   :show-inheritance:

   Creates formatted text report with all regression statistics.

LaTeXFormatter
~~~~~~~~~~~~~~

.. autoclass:: streamreg.output.LaTeXFormatter
   :members:
   :undoc-members:
   :show-inheritance:

   Generates publication-ready LaTeX tables.

DiagnosticsFormatter
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: streamreg.output.DiagnosticsFormatter
   :members:
   :undoc-members:
   :show-inheritance:

   Detailed diagnostics including confidence intervals and cluster information.

READMEFormatter
~~~~~~~~~~~~~~~

.. autoclass:: streamreg.output.READMEFormatter
   :members:
   :undoc-members:
   :show-inheritance:

   Creates Markdown README with analysis overview.

ConfigFormatter
~~~~~~~~~~~~~~~

.. autoclass:: streamreg.output.ConfigFormatter
   :members:
   :undoc-members:
   :show-inheritance:

   Saves configuration snapshot for reproducibility.

Low-Level Estimators
--------------------

Advanced users can use these classes directly for custom workflows.

OnlineRLS
~~~~~~~~~

.. autoclass:: streamreg.estimators.ols.OnlineRLS
   :members:
   :undoc-members:
   :show-inheritance:

   **Low-Level Online RLS**
   
   Implements recursive least squares with cluster-robust covariance.
   Most users should use the high-level OLS class instead.

Online2SLS
~~~~~~~~~~

.. autoclass:: streamreg.estimators.iv.Online2SLS
   :members:
   :undoc-members:
   :show-inheritance:

   **Low-Level 2SLS Implementation**
   
   Implements two-stage least squares using composition of OnlineRLS models.
   Most users should use the high-level TwoSLS class instead.

Orchestrators
-------------

These classes manage parallel processing workflows.

ParallelOLSOrchestrator
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: streamreg.estimators.ols.ParallelOLSOrchestrator
   :members:
   :undoc-members:
   :show-inheritance:

   **Parallel OLS Orchestration**
   
   Manages parallel processing of partitioned datasets for OLS estimation.

TwoSLSOrchestrator
~~~~~~~~~~~~~~~~~~

.. autoclass:: streamreg.estimators.iv.TwoSLSOrchestrator
   :members:
   :undoc-members:
   :show-inheritance:

   **Two-Pass 2SLS Orchestration**
   
   Manages two-pass estimation for instrumental variables regression.

Utility Functions
-----------------

Convenience functions for quick analyses.

Convenience Functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: streamreg.api.ols

   **Quick OLS Estimation**
   
   Convenience function that fits an OLS model and returns results in one call.
   
   .. code-block:: python
   
      from streamreg.api import ols
      
      results = ols("y ~ x1 + x2", data, cluster='country')
      print(results.summary())

.. autofunction:: streamreg.api.twosls

   **Quick 2SLS Estimation**
   
   Convenience function that fits a 2SLS model and returns results in one call.
   
   .. code-block:: python
   
      from streamreg.api import twosls
      
      results = twosls(
          "y ~ x1 | z1 + z2", 
          data, 
          endogenous=['x1'],
          cluster='country'
      )
      print(results.summary())
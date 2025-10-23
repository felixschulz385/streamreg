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

   **Parameters:**
   
   - **formula** (str): R-style formula (e.g., "y ~ x1 + x2 + I(x1^2)")
   - **alpha** (float): Regularization parameter for numerical stability (default: 1e-3)
   - **forget_factor** (float): Forgetting factor for online learning (default: 1.0)
   - **chunk_size** (int): Chunk size for processing large datasets (default: 10000)
   - **n_workers** (int): Number of parallel workers (auto-detected if None)
   - **show_progress** (bool): Show progress bar during fitting (default: True)
   - **se_type** (str): Standard error type - 'stata', 'HC0', 'HC1', 'HC2', or 'HC3' (default: 'stata')

   **Fit Method:**
   
   .. automethod:: fit
   
      :param data: Data source (str, Path, DataFrame, or StreamData)
      :param cluster: Cluster variable(s) for robust standard errors (str or list of str)
      :param query: Pandas query string to filter data (e.g., "year >= 2000 and country == 'USA'")
      :type query: str, optional
      :return: Fitted model
      :rtype: OLS
      
      The query parameter allows efficient chunk-level filtering:
      
      * Applied to each chunk as it's loaded from disk
      * Uses pandas `.query()` syntax
      * Memory-efficient for large datasets
      * Supports numeric comparisons, string matching, list membership, and boolean logic
      
      Examples::
      
          model.fit(data, query="year >= 2000")
          model.fit(data, query="country == 'USA' and year >= 2000")
          model.fit(data, query="gdp > 10000 or population < 1000000")
          model.fit(data, query="country.isin(['USA', 'CAN', 'MEX'])")

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

   **Parameters:**
   
   - **formula** (str): R-style formula with instruments (e.g., "y ~ x1 + x2 | z1 + z2")
   - **endogenous** (list): List of endogenous variables (defaults to all features if None)
   - **alpha** (float): Regularization parameter (default: 1e-3)
   - **forget_factor** (float): Forgetting factor (default: 1.0)
   - **chunk_size** (int): Chunk size for processing (default: 10000)
   - **n_workers** (int): Number of parallel workers (auto-detected if None)
   - **show_progress** (bool): Show progress bar (default: True)
   - **se_type** (str): Standard error type - 'stata', 'HC0', 'HC1', 'HC2', or 'HC3' (default: 'stata')

   **Fit Method:**
   
   .. automethod:: fit
   
      :param data: Data source (str, Path, DataFrame, or StreamData)
      :param cluster: Cluster variable(s) for robust standard errors (str or list of str)
      :param query: Pandas query string to filter data
      :type query: str, optional
      :return: Fitted model
      :rtype: TwoSLS
      
      The query parameter works identically to OLS::
      
          model.fit(data, query="year >= 2000")
          model.fit(data, query="developed == True and year >= 2000")

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

   **Parameters:**
   
   - **data**: Data source (str, Path, or DataFrame)
   - **chunk_size** (int): Size of chunks for iteration (default: 10000)
   - **query**: Pandas query string to filter data
   - **type query**: str, optional

   **Query Parameter:**
   
   The query parameter enables memory-efficient filtering:
      
      * Validated against a sample when StreamData is initialized
      * Applied to each chunk during iteration
      * Raises ValueError if query syntax is invalid
      
      Examples::
      
          # Filter on initialization
          data = StreamData("large_file.parquet", query="year >= 2000")
          
          # Iterate over filtered chunks
          for chunk in data.iter_chunks():
              # Process filtered data
              pass

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

   **Parameters:**
   
   - **formula** (str): R-style formula (e.g., "y ~ x1 + x2 + I(x1^2)")
   - **data**: Data source (str, Path, DataFrame, or StreamData)
   - **cluster**: Cluster variable(s) for robust standard errors
   - **query**: Pandas query string to filter data
   - **type query**: str, optional
   - **se_type** (str): Standard error type (default: 'stata')
   - **kwargs: Additional arguments passed to OLS
   - **return**: Fitted results
   - **rtype**: RegressionResults
   
   **Example:**
   
   .. code-block:: python
   
      results = ols(
          formula="y ~ x1 + x2",
          data="data.parquet",
          cluster='country',
          query="year >= 2000"
      )

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

   **Parameters:**
   
   - **formula** (str): R-style formula with instruments (e.g., "y ~ x1 + x2 | z1 + z2")
   - **data**: Data source (str, Path, DataFrame, or StreamData)
   - **endogenous** (list): Endogenous variables
   - **cluster**: Cluster variable(s) for robust standard errors
   - **query**: Pandas query string to filter data
   - **type query**: str, optional
   - **se_type** (str): Standard error type (default: 'stata')
   - **kwargs: Additional arguments passed to TwoSLS
   - **return**: Fitted results
   - **rtype**: RegressionResults
   
   **Example:**
   
   .. code-block:: python
   
      results = twosls(
          formula="y ~ x1 | z1 + z2",
          data="data.parquet",
          endogenous=['x1'],
          query="developed == True"
      )

Query Functionality
-------------------

StreamReg supports efficient chunk-level filtering using the `query` parameter, which accepts pandas `.query()` syntax. This allows you to filter large datasets without loading the entire dataset into memory.

**How it works:**

- For DataFrames: The query is applied once at initialization.
- For Parquet files: The query is applied to each chunk as it is loaded. If possible, simple queries are converted to filter pushdown for fastparquet.
- For partitioned datasets: The query is applied to each partition and chunk.

**Usage Example:**

.. code-block:: python

    # Filter by year
    model.fit(data, query="year >= 2000")

    # Filter by multiple conditions
    model.fit(data, query="year >= 2000 and country == 'USA'")

    # Complex boolean logic
    model.fit(data, query="(gdp > 10000 or population < 1000000) and year >= 1990")

    # Use pandas string methods
    model.fit(data, query="country.isin(['USA', 'CAN', 'MEX'])")

    # Numeric comparisons
    model.fit(data, query="temperature > 20 and temperature < 30")

**Supported Syntax:**

- Numeric comparisons: `year >= 2000`
- String comparisons: `country == 'USA'`
- List membership: `country.isin(['USA', 'CAN', 'MEX'])`
- Boolean combinations: `(year >= 2000) & (country == 'USA')`
- Missing values: `temperature.notna()`
- Complex expressions: `(year >= 2000) & (temperature > 15) & country.isin(['USA', 'CAN'])`

**Notes:**

- Use `&` for AND, `|` for OR, `~` for NOT (not `and`, `or`, `not`)
- String values must use quotes: `"country == 'USA'"`
- Use parentheses for complex expressions: `"(a > 5) & (b < 10)"`
- Query is validated on a sample before processing
- Invalid queries raise `ValueError` with descriptive message
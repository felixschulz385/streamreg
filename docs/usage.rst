.. filepath: /scicore/home/meiera/schulz0022/projects/growth-and-temperature/docs/usage.rst

Usage Guide
===========

This guide covers common workflows for using StreamReg to estimate regression models
on large datasets.

Basic OLS Regression
--------------------

Simple Linear Model
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from gnt.analysis.streamreg.api import OLS
    
    # Create model
    model = OLS(formula="gdp_growth ~ temperature + rainfall")
    
    # Fit on data
    model.fit("data/climate_data.parquet")
    
    # View results
    print(model.summary())
    print(f"R²: {model.r_squared_:.4f}")
    print(f"N: {model.n_obs_:,}")

With Cluster-Robust Standard Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One-way clustering (e.g., by country):

.. code-block:: python

    model = OLS(formula="y ~ x1 + x2")
    model.fit(data, cluster='country')
    print(model.summary())

Two-way clustering (e.g., by country and year):

.. code-block:: python

    model = OLS(formula="y ~ x1 + x2")
    model.fit(data, cluster=['country', 'year'])
    print(model.summary())

Feature Engineering with Formulas
----------------------------------

Polynomial Terms
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Quadratic term
    model = OLS(formula="y ~ temperature + I(temperature^2)")
    
    # Cubic polynomial
    model = OLS(formula="y ~ temp + I(temp^2) + I(temp^3)")
    
    # Alternative syntax
    model = OLS(formula="y ~ poly(temperature, 3)")

Interaction Terms
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Interaction with main effects (equivalent to x1 + x2 + x1:x2)
    model = OLS(formula="y ~ x1 * x2")
    
    # Interaction without main effects
    model = OLS(formula="y ~ x1 + x2 + x1:x2")
    
    # Multiple interactions
    model = OLS(formula="y ~ x1 * x2 * x3")

Removing Intercept
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    model = OLS(formula="y ~ x1 + x2 - 1")
    model.fit(data)

Two-Stage Least Squares (2SLS)
-------------------------------

Basic IV Estimation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from gnt.analysis.streamreg.api import TwoSLS
    
    # Specify instruments after |
    model = TwoSLS(
        formula="gdp_growth ~ temperature + rainfall | historical_temp + elevation",
        endogenous=['temperature']  # Which variables are endogenous
    )
    
    model.fit(data, cluster='country')
    
    # View first stage results
    first_stage = model.summary(stage='first')
    for stage_name, summary in first_stage.items():
        print(f"\n{stage_name}")
        print(summary)
    
    # View second stage results
    print("\nSecond Stage:")
    print(model.summary(stage='second'))

Multiple Endogenous Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    model = TwoSLS(
        formula="y ~ endog1 + endog2 + exog1 | instr1 + instr2 + instr3",
        endogenous=['endog1', 'endog2']
    )
    model.fit(data)

Checking Instrument Strength
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    model.fit(data)
    
    # Access first stage diagnostics
    for i, fs_result in enumerate(model.results_.first_stage_results):
        print(f"\nFirst Stage {i+1}:")
        print(f"  F-statistic (overall): {fs_result.f_statistic:.2f}")
        
        # IV-specific F-statistic for instrument strength
        if 'iv_f_statistic' in fs_result.metadata:
            iv_f = fs_result.metadata['iv_f_statistic']
            print(f"  F-statistic (instruments): {iv_f:.2f}")
            
            # Rule of thumb: F > 10 indicates strong instruments
            if iv_f < 10:
                print("  WARNING: Weak instruments detected!")

Working with Different Data Sources
------------------------------------

From DataFrame
~~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    
    df = pd.read_csv("small_data.csv")
    
    model = OLS(formula="y ~ x1 + x2")
    model.fit(df)

From Single Parquet File
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    model = OLS(formula="y ~ x1 + x2")
    model.fit("data/large_dataset.parquet")

From Partitioned Parquet Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Automatically detects and processes all partitions
    model = OLS(formula="y ~ x1 + x2", n_workers=8)
    model.fit("data/partitioned_dataset/")

Controlling Processing Parameters
----------------------------------

Chunk Size
~~~~~~~~~~

.. code-block:: python

    # Smaller chunks for memory-constrained systems
    model = OLS(formula="y ~ x", chunk_size=5000)
    model.fit(data)

Number of Workers
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Explicit worker count
    model = OLS(formula="y ~ x", n_workers=4)
    model.fit(data)
    
    # Auto-detect from SLURM environment
    model = OLS(formula="y ~ x", n_workers=None)
    model.fit(data)

Progress Reporting
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Show progress bar (default)
    model = OLS(formula="y ~ x", show_progress=True)
    model.fit(data)
    
    # Silent mode
    model = OLS(formula="y ~ x", show_progress=False)
    model.fit(data)

Saving and Exporting Results
-----------------------------

Save All Formats
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Saves JSON, CSV, LaTeX, summary report, README, diagnostics
    model.fit(data)
    model.results_.save("output/my_analysis", spec_name="baseline")

Save Specific Formats
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Only save JSON and CSV
    model.results_.save(
        "output/my_analysis",
        spec_name="baseline",
        formats=['json', 'csv']
    )

Individual Format Methods
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pathlib import Path
    
    output_dir = Path("output/my_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual formats
    model.results_.save_json(output_dir)
    model.results_.save_csv(output_dir)
    model.results_.save_latex(output_dir)
    model.results_.save_summary(output_dir, spec_config={'description': 'My analysis'})

Accessing Results Programmatically
-----------------------------------

Coefficients and Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Access properties
    print("Coefficients:", model.coef_)
    print("Standard errors:", model.se_)
    print("R²:", model.r_squared_)
    print("N observations:", model.n_obs_)
    
    # Get specific coefficient
    coef_info = model.results_.get_coefficient('temperature')
    print(f"Temperature coef: {coef_info['coefficient']:.4f}")
    print(f"P-value: {coef_info['p_value']:.4f}")
    
    # Confidence intervals
    ci_lower, ci_upper = model.results_.get_confidence_interval('temperature', alpha=0.05)
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

Full Results Object
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    results = model.results_
    
    # Convert to dictionary for JSON serialization
    results_dict = results.to_dict()
    
    # Access DataFrame summary
    summary_df = results.summary()
    
    # Filter significant coefficients
    significant = summary_df[summary_df['p_value'] < 0.05]
    print(significant)

Making Predictions
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import pandas as pd
    
    # New data
    new_data = pd.DataFrame({
        'temperature': [20, 25, 30],
        'rainfall': [100, 150, 200]
    })
    
    # Make predictions
    predictions = model.predict(new_data)
    print("Predictions:", predictions)

Configuration-Based Workflow
-----------------------------

For reproducible research, use YAML configuration files:

.. code-block:: yaml

    # config/analysis.yaml
    analyses:
      online_rls:
        defaults:
          alpha: 0.001
          chunk_size: 10000
          cluster_type: two_way
        
        specifications:
          baseline:
            description: "Baseline climate-growth relationship"
            formula: "gdp_growth ~ temperature + rainfall + I(temperature^2)"
            data_source: "data/climate_growth.parquet"
            cluster1_col: "country"
            cluster2_col: "year"

Run from command line::

    python -m gnt.analysis.entrypoint online_rls -s baseline -o output/

Complete Example: Climate-Growth Analysis
------------------------------------------

.. code-block:: python

    from gnt.analysis.streamreg.api import OLS, TwoSLS
    
    # 1. OLS with polynomials and interactions
    ols_model = OLS(
        formula="gdp_growth ~ temperature * rainfall + I(temperature^2)",
        chunk_size=10000,
        n_workers=8
    )
    
    ols_model.fit(
        "data/climate_growth.parquet",
        cluster=['country', 'year']
    )
    
    print("OLS Results:")
    print(ols_model.summary())
    
    # Save OLS results
    ols_model.results_.save("output/climate_ols", spec_name="polynomial")
    
    # 2. IV estimation for causal inference
    iv_model = TwoSLS(
        formula="gdp_growth ~ temperature + rainfall | historical_temp + elevation",
        endogenous=['temperature'],
        n_workers=8
    )
    
    iv_model.fit(
        "data/climate_growth.parquet",
        cluster='country'
    )
    
    print("\n2SLS First Stage:")
    for name, fs in iv_model.summary(stage='first').items():
        print(f"\n{name}")
        print(fs)
    
    print("\n2SLS Second Stage:")
    print(iv_model.summary(stage='second'))
    
    # Save IV results
    iv_model.results_.save("output/climate_iv", spec_name="instrumented")
    
    # 3. Compare results
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"OLS Temperature Coef: {ols_model.results_.get_coefficient('temperature')['coefficient']:.4f}")
    print(f"2SLS Temperature Coef: {iv_model.results_.get_coefficient('temperature_hat')['coefficient']:.4f}")

Troubleshooting Common Issues
------------------------------

Memory Errors
~~~~~~~~~~~~~

If you encounter memory errors:

.. code-block:: python

    # Reduce chunk size
    model = OLS(formula="y ~ x", chunk_size=5000)
    model.fit(data)
    
    # Or reduce number of workers
    model = OLS(formula="y ~ x", n_workers=2)
    model.fit(data)

Convergence Issues
~~~~~~~~~~~~~~~~~~

If coefficients seem unstable:

.. code-block:: python

    # Increase regularization
    model = OLS(formula="y ~ x", alpha=1e-2)
    model.fit(data)
    
    # Check condition number in logs

Missing Data Warnings
~~~~~~~~~~~~~~~~~~~~~

StreamReg automatically removes rows with missing values:

.. code-block:: python

    # Check how many observations were used
    model.fit(data)
    print(f"Used {model.n_obs_:,} observations")
    
    # To see warnings about missing data, set logging level
    import logging
    logging.basicConfig(level=logging.DEBUG)

Cluster Warnings
~~~~~~~~~~~~~~~~

If you see warnings about small clusters:

.. code-block:: python

    # Check cluster diagnostics
    model.fit(data, cluster='country')
    diag = model.results_.cluster_diagnostics
    print(f"Clusters: {diag['dim1']['n_clusters']}")
    print(f"Min size: {diag['dim1']['min_size']}")
    print(f"Warnings: {diag['dim1']['warnings']}")
    
    # Consider different clustering level if needed
    model.fit(data, cluster='region')  # Larger clusters

Performance Tips
~~~~~~~~~~~~~~~~

For best performance:

1. **Use partitioned parquet files** for large datasets
2. **Match workers to CPU cores**: ``n_workers=16`` for 16-core machine
3. **Tune chunk size** based on available memory: 10,000-50,000 rows per chunk
4. **Use SSD storage** for faster I/O
5. **Pre-filter data** to only required columns

.. code-block:: python

    # Example optimized setup
    model = OLS(
        formula="y ~ x1 + x2",
        chunk_size=20000,  # Based on memory
        n_workers=16,       # Match CPU cores
        show_progress=True
    )
    model.fit("data/partitioned/")

Data Format Requirements
~~~~~~~~~~~~~~~~~~~~~~~~

StreamReg expects:

* **Numeric columns** for features and target
* **Any type** for cluster variables (strings, integers, etc.)
* **Finite values** (NaN, Inf, -Inf are removed automatically)

.. code-block:: python

    # Check your data
    import pandas as pd
    
    df = pd.read_parquet("data/sample.parquet")
    print(df.dtypes)  # Check column types
    print(df.isna().sum())  # Check missing values
    print(df.describe())  # Check for outliers/infinite values

Formula Syntax Reference
------------------------

Quick reference for formula syntax:

.. code-block:: python

    # Basic
    "y ~ x1 + x2"                    # Linear model
    "y ~ x1 + x2 - 1"                # No intercept
    
    # Transformations
    "y ~ x + I(x^2)"                 # Quadratic
    "y ~ poly(x, 3)"                 # Cubic polynomial
    "y ~ x1 + x2 + I(x1*x2)"        # Interaction
    "y ~ x1 * x2"                    # Main effects + interaction
    "y ~ x1:x2"                      # Interaction only
    
    # Instrumental variables
    "y ~ x1 + x2 | z1 + z2"         # 2SLS with instruments
    "y ~ x1 | z1 + z2 + z3"         # Overidentified model
    
    # Complex formulas
    "y ~ x1 + I(x1^2) + x2 * x3 | z1 + z2 - 1"

API Quick Reference
-------------------

Common methods and properties:

.. code-block:: python

    # Create model
    model = OLS(formula="y ~ x")
    
    # Fit with options
    model.fit(data, cluster=['dim1', 'dim2'])
    
    # Access results
    model.coef_              # Coefficients
    model.se_                # Standard errors
    model.n_obs_             # Number of observations
    model.r_squared_         # R-squared
    model.results_           # Full results object
    
    # Get summary
    model.summary()          # DataFrame summary
    
    # Make predictions
    model.predict(X_new)     # Predictions
    
    # Save results
    model.results_.save(
        "output/",
        spec_name="baseline",
        formats=['json', 'latex', 'summary']
    )
    
    # Access specific coefficients
    coef_info = model.results_.get_coefficient('x1')
    ci_lower, ci_upper = model.results_.get_confidence_interval('x1')
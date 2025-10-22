.. filepath: /scicore/home/meiera/schulz0022/projects/growth-and-temperature/docs/installation.rst

Installation
============

Requirements
------------

**Python Version**

StreamReg requires **Python 3.8 or later**. We recommend Python 3.9 or 3.10 for best performance.

**Core Dependencies**

StreamReg depends on the following packages:

* **numpy** >= 1.20.0 - Numerical computations
* **pandas** >= 1.3.0 - Data manipulation
* **scipy** >= 1.7.0 - Statistical functions
* **pyarrow** >= 5.0.0 - Parquet file support
* **tqdm** >= 4.60.0 - Progress bars
* **pyyaml** >= 5.4.0 - Configuration files

**Optional Dependencies**

For development and documentation:

* **pytest** >= 6.0.0 - Testing framework
* **pytest-cov** >= 2.12.0 - Coverage reports
* **sphinx** >= 4.0.0 - Documentation generation
* **sphinx_rtd_theme** >= 1.0.0 - ReadTheDocs theme

Installation Methods
---------------------

From Source (Recommended for Development)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Clone the repository::

    git clone https://github.com/your-org/growth-and-temperature.git
    cd growth-and-temperature

2. Create a virtual environment (recommended)::

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install in development mode::

    pip install -e .

This will install the package in editable mode, so changes to the code are immediately available.

4. (Optional) Install development dependencies::

    pip install -e ".[dev]"

From PyPI (When Available)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once published to PyPI, you can install directly::

    pip install streamreg-gnt

Verifying Installation
----------------------

Test your installation:

.. code-block:: python

    # Test basic imports
    import streamreg
    from streamreg.api import OLS, TwoSLS
    from streamreg.data import StreamData
    
    print("✓ StreamReg installed successfully")
    
    # Test with sample data
    import pandas as pd
    import numpy as np
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    x = np.random.randn(n)
    y = 2*x + np.random.randn(n)*0.5
    df = pd.DataFrame({'y': y, 'x': x})
    
    # Fit simple model
    model = OLS(formula="y ~ x")
    model.fit(df)
    
    print(f"✓ Coefficient: {model.coef_[1]:.3f} (expected ~2.0)")
    print(f"✓ R²: {model.r_squared_:.3f}")

If this runs without errors, your installation is working correctly.

Setting up for HPC Environments
--------------------------------

SLURM Integration
~~~~~~~~~~~~~~~~~

StreamReg automatically detects SLURM environment variables for optimal resource usage:

* ``SLURM_CPUS_PER_TASK`` - CPUs allocated to the task
* ``SLURM_NTASKS`` - Number of tasks
* ``SLURM_JOB_CPUS_PER_NODE`` - CPUs per node

Example SLURM submission script::

    #!/bin/bash
    #SBATCH --job-name=streamreg_analysis
    #SBATCH --output=logs/streamreg_%j.out
    #SBATCH --error=logs/streamreg_%j.err
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=16
    #SBATCH --mem=64G
    #SBATCH --time=04:00:00
    #SBATCH --partition=shared
    
    # Load modules
    module load python/3.9
    module load gcc/9.3.0
    
    # Activate environment
    source ~/venv/streamreg/bin/activate
    
    # Run analysis
    python scripts/run_analysis.py --config config/baseline.yaml

Performance Tuning
~~~~~~~~~~~~~~~~~~

For HPC systems:

1. **Match workers to cores**: Use ``n_workers`` equal to ``SLURM_CPUS_PER_TASK``
2. **Adjust chunk size** based on memory: ``chunk_size = total_memory_mb / (n_workers * 10)``
3. **Use local storage** if available: Copy data to ``$TMPDIR`` for faster I/O

Example configuration:

.. code-block:: python

    import os
    
    # Auto-detect resources
    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    mem_per_worker = 4096  # MB
    chunk_size = mem_per_worker * 2  # rows per chunk
    
    model = OLS(
        formula="y ~ x1 + x2",
        chunk_size=chunk_size,
        n_workers=n_workers
    )

Troubleshooting Installation
-----------------------------

ImportError: No module named 'pyarrow'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install pyarrow manually::

    pip install pyarrow>=5.0.0

NumPy Compatibility Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you see NumPy warnings or errors, try upgrading::

    pip install --upgrade numpy>=1.20.0

Pandas Version Conflicts
~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure compatible pandas version::

    pip install pandas>=1.3.0

For Apple Silicon (M1/M2) Macs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some packages may need to be installed via conda::

    conda install numpy pandas scipy pyarrow
    pip install streamreg-gnt --no-deps

Permission Errors
~~~~~~~~~~~~~~~~~

If you encounter permission errors, use::

    pip install --user -e .

Path Issues
~~~~~~~~~~~

If imports fail, ensure the package is in your Python path::

    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

Updating StreamReg
------------------

If installed from source::

    cd growth-and-temperature
    git pull
    pip install -e .

If installed from PyPI::

    pip install --upgrade streamreg-gnt

Uninstallation
--------------

To remove StreamReg::

    pip uninstall streamreg-gnt

Or if installed in development mode::

    pip uninstall streamreg
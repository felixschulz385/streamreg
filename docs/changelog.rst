.. filepath: /scicore/home/meiera/schulz0022/projects/growth-and-temperature/docs/changelog.rst

Changelog
=========

Version 0.1.0 (2026-20-22)
--------------------------

Initial release of StreamReg.

**Features:**

* Online OLS estimation with cluster-robust standard errors
* Two-stage least squares (2SLS) for instrumental variable estimation
* R-style formula interface with transformations
* Parallel processing for partitioned parquet datasets
* Multiple output formats (JSON, CSV, LaTeX, text reports)
* Comprehensive cluster diagnostics
* Support for one-way and two-way clustering
* Feature engineering (polynomials, interactions, custom transformations)
* Unified StreamData interface for different data sources
* F-statistics for overall model significance and instrument strength

**Documentation:**

* Complete ReadTheDocs documentation
* Usage guide with examples
* API reference
* Output formatting guide

**Infrastructure:**

* YAML-based configuration system
* Command-line entrypoint for reproducible analyses
* HPC environment detection (SLURM)
* Progress reporting and logging

Future Releases
---------------

Planned features for future versions:

* Bootstrap standard errors
* Additional output formats (Stata, R)
* Support for categorical variables

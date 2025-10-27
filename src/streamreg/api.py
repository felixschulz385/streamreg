import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, List, Literal, Tuple
from pathlib import Path
import logging

from streamreg.data import StreamData
from streamreg.results import RegressionResults
from streamreg.formula import FormulaParser
from streamreg.transforms import FeatureTransformer
from streamreg.estimators.ols import OnlineRLS, DaskOLSEstimator

logger = logging.getLogger(__name__)


class OLS:
    """
    Ordinary Least Squares estimator with streaming/online capabilities.
    
    Usage:
    ------
    >>> model = OLS(formula="y ~ x1 + x2 + I(x1^2)", se_type='HC1')
    >>> model.fit(data, cluster=['country', 'year'])
    >>> print(model.summary())
    >>> coeffs = model.coef_
    
    >>> # Filter data with query (automatically optimized)
    >>> model = OLS(formula="y ~ x1 + x2")
    >>> model.fit(data, query="year >= 2000 and country == 'USA'")
    """
    
    def __init__(
        self,
        formula: str,
        alpha: float = 1e-3,
        forget_factor: float = 1.0,
        chunk_size: int = 10000,
        n_workers: Optional[int] = None,
        show_progress: bool = True,
        se_type: Literal['stata', 'HC0', 'HC1'] = 'stata'
    ):
        """
        Initialize OLS estimator.
        
        Parameters:
        -----------
        formula : str
            R-style formula (e.g., "y ~ x1 + x2 + I(x1^2)")
        alpha : float
            Regularization parameter for numerical stability
        forget_factor : float
            Forgetting factor (1.0 = no forgetting)
        chunk_size : int
            Chunk size for processing large datasets
        n_workers : int, optional
            Number of parallel workers (auto-detected if None)
        show_progress : bool
            Show progress bar during fitting
        se_type : str
            Standard error type: 'stata', 'HC0', 'HC1'
            - 'stata': Default STATA-style correction: (N/(N-1)) * ((NT-1)/(NT-K))
            - 'HC0': White's heteroskedasticity-consistent estimator with no correction
            - 'HC1': Degrees-of-freedom correction: NT/(NT-K)
        """
        self.formula = formula
        self.alpha = alpha
        self.forget_factor = forget_factor
        self.chunk_size = chunk_size
        self.n_workers = n_workers
        self.show_progress = show_progress
        self.se_type = se_type
        
        # Parse formula
        self._parser = FormulaParser.parse(formula)
        
        if self._parser.instruments:
            raise ValueError(
                "Formula contains instruments. Use TwoSLS for IV estimation."
            )
        
        # Results (populated after fit)
        self._results = None
        self._rls_model = None
        self._cluster_type = 'classical'
    
    def fit(
        self,
        data: Union[str, Path, pd.DataFrame, StreamData],
        cluster: Optional[Union[str, List[str]]] = None,
        query: Optional[str] = None,
        demean: Optional[Union[str, List[str]]] = None  # New parameter
    ) -> 'OLS':
        """
        Fit the OLS model with efficient data filtering.
        
        Parameters:
        -----------
        data : str, Path, DataFrame, or StreamData
            Data source
        cluster : str or list of str, optional
            Cluster variable(s) for robust standard errors
        query : str, optional
            Pandas query string to filter data
        demean : str or list of str, optional
            Grouping variable(s) for demeaning (e.g., 'country' or ['country', 'year'])
        
        Returns:
        --------
        self : OLS
            Fitted model
        """
        # Setup data with filtering
        if not isinstance(data, StreamData):
            data = StreamData(data, chunk_size=self.chunk_size, query=query)
        elif query is not None:
            logger.warning("Query parameter ignored when data is already a StreamData object")

        # Extract dataset root for means storage
        dataset_root = self._extract_dataset_root(data)

        # If demeaning is specified inside the formula (new syntax), prefer that unless explicit demean arg provided
        except_vars: Optional[List[str]] = None
        if getattr(self._parser, "demean_groups", None):
            if demean is not None:
                logger.info("Demean specified both in formula and via 'demean' argument: explicit argument takes precedence")
            else:
                # Use groups parsed from formula
                demean = self._parser.demean_groups
                except_vars = self._parser.demean_except

        # Apply demeaning if requested
        extra_columns = None
        if demean:
            data = self._apply_demeaning(data, demean, dataset_root, except_vars)
            # Collect demeaning group columns to ensure they are loaded
            if isinstance(demean, str):
                extra_columns = [demean]
            else:
                extra_columns = [col for sublist in demean for col in (sublist if isinstance(sublist, list) else [sublist])]

        # Validate columns
        required_cols = [self._parser.target] + self._parser.features
        if cluster:
            cluster_cols = [cluster] if isinstance(cluster, str) else cluster
            required_cols.extend(cluster_cols)
        data.validate_columns(required_cols)
        
        # Determine cluster type
        self._cluster_type = 'classical'
        cluster1_col = None
        cluster2_col = None
        
        if cluster:
            if isinstance(cluster, str):
                self._cluster_type = 'one_way'
                cluster1_col = cluster
            elif len(cluster) == 2:
                self._cluster_type = 'two_way'
                cluster1_col, cluster2_col = cluster
            else:
                raise ValueError("cluster must be string or list of 2 strings")
        
        # Create feature engineering config
        feature_engineering = None
        if self._parser.transformations:
            feature_engineering = {'transformations': self._parser.transformations}
        
        # Setup feature transformation
        from streamreg.transforms import FeatureTransformer
        
        if feature_engineering or self._parser.has_intercept:
            transformer = FeatureTransformer.from_config(
                feature_engineering or {'transformations': []},
                self._parser.features,
                add_intercept=self._parser.has_intercept
            )
            n_features = transformer.get_n_features()
            feature_names = transformer.get_feature_names()
        else:
            n_features = len(self._parser.features)
            feature_names = self._parser.features.copy()
        
        # Use DaskOLSEstimator for efficient out-of-memory computation
        estimator = DaskOLSEstimator(
            dask_df=data._dask_df,  # Access the internal Dask DataFrame
            feature_cols=self._parser.features,
            target_col=self._parser.target,
            cluster1_col=cluster1_col,
            cluster2_col=cluster2_col,
            add_intercept=self._parser.has_intercept,
            alpha=self.alpha,
            se_type=self.se_type,
            feature_transformer=transformer if (feature_engineering or self._parser.has_intercept) else None,
            n_workers=self.n_workers
        )
        
        self._rls_model = estimator.fit(verbose=True)
        
        # Store results
        self._results = self._create_results()
        
        return self
    
    def _extract_dataset_root(self, data: StreamData) -> Optional[Path]:
        """Extract dataset root directory for means storage."""
        if data.info.source_type in ['parquet', 'partitioned']:
            return data.info.source_path.parent if data.info.source_type == 'parquet' else data.info.source_path
        return None
    
    def _apply_demeaning(self, data: StreamData, group_cols: Union[str, List[str]],
                        dataset_root: Optional[Path], except_vars: Optional[List[str]] = None) -> StreamData:
        """Apply demeaning transformation.

        Parameters
        ----------
        data : StreamData
            The data object to compute means from
        group_cols : str or list
            Group columns specification (single or list of lists)
        dataset_root : Path or None
            Root directory (for caching means)
        except_vars : list of str, optional
            Variables to exclude from demeaning
        """
        from streamreg.demean import DemeanComputer, DemeanTransformer

        if isinstance(group_cols, str):
            group_cols = [group_cols]

        # Compute means (uses LMDB cache)
        computer = DemeanComputer(dataset_root=dataset_root)

        # Variables to demean: all features + target
        demean_vars = self._parser.features + [self._parser.target]

        # Remove except_vars from demean_vars if provided
        if except_vars:
            demean_vars = [v for v in demean_vars if v not in except_vars]

        # Determine if we need sequential demeaning (multiple levels)
        sequential = len(group_cols) > 1
        
        if sequential:
            # Compute means separately for each level (cached in LMDB)
            group_means = computer.compute_means_sequential(
                data,
                variables=demean_vars,
                group_levels=group_cols,
                query=data.query
            )
            logger.info(f"Using sequential demeaning with {len(group_cols)} levels")
        else:
            # Single level: use standard combined key (cached in LMDB)
            group_means = computer.compute_means(
                data,
                variables=demean_vars,
                group_cols=group_cols,
                query=data.query
            )

        # Create transformer - now with dataset_root instead of means
        # Workers will query LMDB on-demand
        transformer = DemeanTransformer(
            dataset_root=dataset_root,  # Pass root, not means
            group_cols=group_cols,
            demean_vars=demean_vars,
            sequential=sequential,
            query=data.query  # Pass query for cache key
        )

        # Apply transformation via StreamData.with_transform()
        return data.with_transform(transformer.transform)
    
    def _create_results(self) -> RegressionResults:
        """Create standardized results object."""
        se = self._rls_model.get_standard_errors(self._cluster_type)
        t_stats = self._rls_model.theta / se
        
        from scipy import stats
        p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
        
        if self._cluster_type == 'classical':
            cov = self._rls_model.get_covariance_matrix()
        else:
            cov = self._rls_model.get_cluster_robust_covariance(self._cluster_type)
        
        # Get F-statistic
        f_stat, df_model, df_resid = self._rls_model.get_f_statistic()
        f_pval = self._rls_model.get_f_pvalue()
        
        cluster_diag = None
        if self._cluster_type != 'classical':
            cluster_diag = {}
            if self._cluster_type == 'one_way':
                cluster_diag['dim1'] = self._rls_model.diagnose_cluster_structure(
                    self._rls_model.cluster_stats, "Cluster1"
                )
            elif self._cluster_type == 'two_way':
                cluster_diag['dim1'] = self._rls_model.diagnose_cluster_structure(
                    self._rls_model.cluster_stats, "Cluster1"
                )
                cluster_diag['dim2'] = self._rls_model.diagnose_cluster_structure(
                    self._rls_model.cluster2_stats, "Cluster2"
                )
        
        return RegressionResults(
            coefficients=self._rls_model.theta,
            std_errors=se,
            feature_names=self._rls_model.get_feature_names(),
            n_obs=self._rls_model.n_obs,
            n_features=self._rls_model.n_features,
            rss=self._rls_model.rss,
            r_squared=self._rls_model.get_r_squared(),
            adj_r_squared=self._rls_model.get_adjusted_r_squared(),
            t_statistics=t_stats,
            p_values=p_values,
            f_statistic=f_stat,
            f_pvalue=f_pval,
            df_model=df_model,
            df_resid=df_resid,
            model_type='ols',
            cluster_type=self._cluster_type,
            covariance_matrix=cov,
            cluster_diagnostics=cluster_diag
        )
    
    def summary(self) -> pd.DataFrame:
        """
        Get regression summary table.
        
        Returns:
        --------
        DataFrame with coefficients, standard errors, t-stats, and p-values
        """
        if self._results is None:
            raise ValueError("Model must be fitted before calling summary()")
        
        return self._results.summary()
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : DataFrame or ndarray
            Features to predict on
        
        Returns:
        --------
        predictions : ndarray
        """
        if self._rls_model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        if isinstance(X, pd.DataFrame):
            X = X[self._parser.features].values
        
        # Apply feature transformation
        transformer = FeatureTransformer.from_config(
            {'transformations': self._parser.transformations or []},
            self._parser.features,
            add_intercept=self._parser.has_intercept
        )
        X_transformed = transformer.transform(X, self._parser.features)
        
        return self._rls_model.predict(X_transformed)
    
    def save_results(
        self,
        output_dir: str,
        spec_name: Optional[str] = None,
        spec_config: Optional[Dict[str, Any]] = None,
        full_config: Optional[Dict[str, Any]] = None,
        formats: Optional[List[str]] = None
    ) -> Path:
        """
        Save regression results to disk.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save results
        spec_name : str, optional
            Name for this specification (used in subdirectory)
        spec_config : dict, optional
            Specification configuration (for documentation)
        full_config : dict, optional
            Full configuration snapshot
        formats : list of str, optional
            Formats to save. Options: 'summary', 'csv', 'json', 'latex', 'readme', 'diagnostics'
            Default: all formats
        
        Returns:
        --------
        run_dir : Path
            Directory where results were saved
        """
        if self._results is None:
            raise ValueError("Model must be fitted before saving results")
        
        return self._results.save(
            output_dir=output_dir,
            spec_name=spec_name,
            spec_config=spec_config,
            full_config=full_config,
            formats=formats
        )
    
    # Scikit-learn style properties
    @property
    def coef_(self) -> np.ndarray:
        """Coefficient estimates."""
        if self._results is None:
            raise ValueError("Model must be fitted first")
        return self._results.coefficients
    
    @property
    def se_(self) -> np.ndarray:
        """Standard errors."""
        if self._results is None:
            raise ValueError("Model must be fitted first")
        return self._results.std_errors
    
    @property
    def n_obs_(self) -> int:
        """Number of observations."""
        if self._results is None:
            raise ValueError("Model must be fitted first")
        return self._results.n_obs
    
    @property
    def r_squared_(self) -> float:
        """R-squared."""
        if self._results is None:
            raise ValueError("Model must be fitted first")
        return self._results.r_squared
    
    @property
    def results_(self) -> RegressionResults:
        """Full results object."""
        if self._results is None:
            raise ValueError("Model must be fitted first")
        return self._results


class TwoSLS:
    """
    Two-Stage Least Squares estimator with streaming/online capabilities.
    
    Usage:
    ------
    >>> model = TwoSLS(formula="y ~ x1 + x2 | z1 + z2", endogenous=['x1'], se_type='HC1')
    >>> model.fit(data, cluster='country')
    >>> print(model.summary())
    
    >>> # Filter data with query (automatically optimized)
    >>> model = TwoSLS(formula="y ~ x1 + x2 | z1 + z2", endogenous=['x1'])
    >>> model.fit(data, query="year >= 2000 and developed == True")
    """
    
    def __init__(
        self,
        formula: str,
        endogenous: Optional[List[str]] = None,
        alpha: float = 1e-3,
        forget_factor: float = 1.0,
        chunk_size: int = 10000,
        n_workers: Optional[int] = None,
        show_progress: bool = True,
        se_type: Literal['stata', 'HC0', 'HC1'] = 'stata'
    ):
        """
        Initialize 2SLS estimator.
        
        Parameters:
        -----------
        formula : str
            R-style formula with instruments (e.g., "y ~ x1 + x2 | z1 + z2")
        endogenous : list of str, optional
            List of endogenous variables. If None, all features are endogenous
        alpha : float
            Regularization parameter
        forget_factor : float
            Forgetting factor
        chunk_size : int
            Chunk size for processing
        n_workers : int, optional
            Number of parallel workers
        show_progress : bool
            Show progress bar
        se_type : str
            Standard error type: 'stata', 'HC0', 'HC1'
        """
        self.formula = formula
        self.endogenous = endogenous
        self.alpha = alpha
        self.forget_factor = forget_factor
        self.chunk_size = chunk_size
        self.n_workers = n_workers
        self.show_progress = show_progress
        self.se_type = se_type
        
        # Parse formula
        self._parser = FormulaParser.parse(formula)
        
        if not self._parser.instruments:
            raise ValueError(
                "2SLS requires instruments. Use format: y ~ x1 + x2 | z1 + z2"
            )
        
        # Results (populated after fit)
        self._results = None
        self._twosls_model = None
        self._cluster_type = 'classical'
        self._endog_cols = endogenous or self._parser.features
        self._exog_cols = [f for f in self._parser.features if f not in self._endog_cols]
    
    def fit(
        self,
        data: Union[str, Path, pd.DataFrame, StreamData],
        cluster: Optional[Union[str, List[str]]] = None,
        query: Optional[str] = None
    ) -> 'TwoSLS':
        """
        Fit the 2SLS model with efficient data filtering.
        
        Parameters:
        -----------
        data : str, Path, DataFrame, or StreamData
            Data source
        cluster : str or list of str, optional
            Cluster variable(s) for robust standard errors
        query : str, optional
            Pandas query string to filter data.
            For DataFrames: Applied once at initialization.
            For Parquet: Automatically converted to filter pushdown when possible.
            Examples:
            - "year >= 2000"
            - "country == 'USA' and year >= 2000"
            - "country.isin(['USA', 'CAN', 'MEX'])"
        
        Returns:
        --------
        self : TwoSLS
            Fitted model
        """
        from streamreg.estimators.iv import TwoSLSOrchestrator
        
        # Setup data with filtering
        if not isinstance(data, StreamData):
            data = StreamData(data, chunk_size=self.chunk_size, query=query)
        elif query is not None:
            logger.warning("Query parameter ignored when data is already a StreamData object")
        
        # Validate columns
        required_cols = [self._parser.target] + self._parser.features + self._parser.instruments
        if cluster:
            cluster_cols = [cluster] if isinstance(cluster, str) else cluster
            required_cols.extend(cluster_cols)
        data.validate_columns(required_cols)
        
        # Determine cluster type
        self._cluster_type = 'classical'
        cluster1_col = None
        cluster2_col = None
        
        if cluster:
            if isinstance(cluster, str):
                self._cluster_type = 'one_way'
                cluster1_col = cluster
            elif len(cluster) == 2:
                self._cluster_type = 'two_way'
                cluster1_col, cluster2_col = cluster
        
        # Feature engineering config
        feature_engineering = {'endogenous': self._endog_cols}
        if self._parser.transformations:
            feature_engineering['transformations'] = self._parser.transformations
        
        # Create orchestrator with StreamData object (now works for all data types)
        orchestrator = TwoSLSOrchestrator(
            data=data,  # Pass StreamData object, not path
            endog_cols=self._endog_cols,
            exog_cols=self._exog_cols,
            instr_cols=self._parser.instruments,
            target_col=self._parser.target,
            cluster1_col=cluster1_col,
            cluster2_col=cluster2_col,
            add_intercept=self._parser.has_intercept,
            alpha=self.alpha,
            chunk_size=self.chunk_size,
            n_workers=self.n_workers,
            show_progress=self.show_progress,
            verbose=True,
            feature_engineering=feature_engineering
        )
        
        self._twosls_model = orchestrator.fit()
        
        # Store results
        self._results = self._create_results()
        
        return self
    
    def _create_results(self) -> RegressionResults:
        """Create standardized results object."""
        # Convert first stage
        first_stage_results = []
        for fs_model in self._twosls_model.first_stage_models:
            se = fs_model.get_standard_errors(self._cluster_type)
            t_stats = fs_model.theta / se
            
            from scipy import stats
            p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
            
            # Get F-statistic for first stage
            f_stat, df_model, df_resid = fs_model.get_f_statistic()
            f_pval = fs_model.get_f_pvalue()
            
            # Add IV-specific F-statistic if available
            iv_f_stat = fs_model.iv_f_statistic
            iv_f_df = fs_model.iv_f_df
            
            fs_result = RegressionResults(
                coefficients=fs_model.theta,
                std_errors=se,
                feature_names=fs_model.get_feature_names(),
                n_obs=fs_model.n_obs,
                n_features=fs_model.n_features,
                rss=fs_model.rss,
                r_squared=fs_model.get_r_squared(),
                adj_r_squared=fs_model.get_adjusted_r_squared(),
                t_statistics=t_stats,
                p_values=p_values,
                f_statistic=f_stat,
                f_pvalue=f_pval,
                df_model=df_model,
                df_resid=df_resid,
                model_type='first_stage',
                cluster_type=self._cluster_type,
                covariance_matrix=fs_model.get_cluster_robust_covariance(self._cluster_type)
                    if self._cluster_type != 'classical' else fs_model.get_covariance_matrix()
            )
            
            # Add IV F-statistic to metadata
            if iv_f_stat is not None:
                fs_result.metadata['iv_f_statistic'] = iv_f_stat
                fs_result.metadata['iv_f_df'] = iv_f_df
            
            first_stage_results.append(fs_result)
        
        # Convert second stage
        ss_model = self._twosls_model.second_stage
        se = ss_model.get_standard_errors(self._cluster_type)
        t_stats = ss_model.theta / se
        
        from scipy import stats
        p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
        
        # Get F-statistic for second stage
        f_stat, df_model, df_resid = ss_model.get_f_statistic()
        f_pval = ss_model.get_f_pvalue()
        
        result = RegressionResults(
            coefficients=ss_model.theta,
            std_errors=se,
            feature_names=ss_model.get_feature_names(),
            n_obs=ss_model.n_obs,
            n_features=ss_model.n_features,
            rss=ss_model.rss,
            r_squared=ss_model.get_r_squared(),
            adj_r_squared=ss_model.get_adjusted_r_squared(),
            t_statistics=t_stats,
            p_values=p_values,
            f_statistic=f_stat,
            f_pvalue=f_pval,
            df_model=df_model,
            df_resid=df_resid,
            model_type='2sls',
            cluster_type=self._cluster_type,
            covariance_matrix=ss_model.get_cluster_robust_covariance(self._cluster_type)
                if self._cluster_type != 'classical' else ss_model.get_covariance_matrix(),
            first_stage_results=first_stage_results
        )
        result.metadata['endogenous_variables'] = self._endog_cols
        
        return result
    
    def summary(self, stage: str = 'second') -> pd.DataFrame:
        """
        Get regression summary.
        
        Parameters:
        -----------
        stage : str
            'first', 'second', or 'all'
        
        Returns:
        --------
        DataFrame or dict of DataFrames
        """
        if self._results is None:
            raise ValueError("Model must be fitted before calling summary()")
        
        if stage == 'second':
            return self._results.summary()
        elif stage == 'first':
            return {
                f"First Stage {i+1} ({self._endog_cols[i]})": fs.summary()
                for i, fs in enumerate(self._results.first_stage_results)
            }
        elif stage == 'all':
            return {
                **{f"First Stage {i+1} ({self._endog_cols[i]})": fs.summary()
                   for i, fs in enumerate(self._results.first_stage_results)},
                "Second Stage": self._results.summary()
            }
        else:
            raise ValueError("stage must be 'first', 'second', or 'all'")
    
    def save_results(
        self,
        output_dir: str,
        spec_name: Optional[str] = None,
        spec_config: Optional[Dict[str, Any]] = None,
        full_config: Optional[Dict[str, Any]] = None,
        formats: Optional[List[str]] = None
    ) -> Path:
        """
        Save regression results to disk.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save results
        spec_name : str, optional
            Name for this specification (used in subdirectory)
        spec_config : dict, optional
            Specification configuration (for documentation)
        full_config : dict, optional
            Full configuration snapshot
        formats : list of str, optional
            Formats to save. Options: 'summary', 'csv', 'json', 'latex', 'readme', 'diagnostics'
            Default: all formats
        
        Returns:
        --------
        run_dir : Path
            Directory where results were saved
        """
        if self._results is None:
            raise ValueError("Model must be fitted before saving results")
        
        return self._results.save(
            output_dir=output_dir,
            spec_name=spec_name,
            spec_config=spec_config,
            full_config=full_config,
            formats=formats
        )
    
    @property
    def coef_(self) -> np.ndarray:
        """Second stage coefficient estimates."""
        if self._results is None:
            raise ValueError("Model must be fitted first")
        return self._results.coefficients
    
    @property
    def results_(self) -> RegressionResults:
        """Full results object."""
        if self._results is None:
            raise ValueError("Model must be fitted first")
        return self._results
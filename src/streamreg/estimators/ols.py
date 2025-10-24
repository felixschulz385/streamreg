import time
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Literal
from dataclasses import dataclass
import warnings
import logging
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
from contextlib import contextmanager
from joblib import Parallel, delayed
import fastparquet as fp
from streamreg.data import ChunkTask as DataChunkTask

logger = logging.getLogger(__name__)

# Constants for magic numbers
DEFAULT_ALPHA = 1e-3
DEFAULT_BATCH_SIZE = 1000
DEFAULT_CHUNK_SIZE = 10000
MIN_FILE_SIZE_BYTES = 1024
MAX_REGULARIZATION_MULTIPLIER = 10
CONDITION_NUMBER_THRESHOLD = 1e12
MIN_CLUSTER_SIZE = 5
MIN_VALID_DATA_RATIO = 0.001
FUTURE_TIMEOUT_SECONDS = 60
PERIODIC_COLLECTION_MULTIPLIER = 2

def _default_cluster_stats(n_features):
    """Create default cluster stats dictionary - needed for pickling."""
    return {
        'X_sum': np.zeros(n_features),
        'residual_sum': 0.0,
        'count': 0,
        'XtX': np.zeros((n_features, n_features)),
        'X_residual_sum': np.zeros(n_features),
        'Xy': np.zeros(n_features)
    }


@dataclass
class ChunkTask:
    """Configuration for processing a single chunk."""
    chunk_id: int
    partition_idx: int
    feature_cols: List[str]
    target_col: str
    cluster1_col: Optional[str]
    cluster2_col: Optional[str]
    add_intercept: bool
    n_features: int
    alpha: float
    feature_engineering_config: Optional[Dict[str, Any]]

# Define WorkerChunkSpec alias expected by the rest of the module
WorkerChunkSpec = ChunkTask


@dataclass
class ChunkResult:
    """Results from processing a single chunk."""
    chunk_id: int
    partition_idx: int
    XtX: np.ndarray
    Xty: np.ndarray
    n_obs: int
    sum_y: float
    sum_y_squared: float
    cluster_stats: Dict
    cluster2_stats: Dict
    intersection_stats: Dict
    success: bool = True
    error: Optional[str] = None


class LinAlgHelper:
    """Helper class for common linear algebra operations with error handling."""
    
    # Add class variable to track if warning was shown
    _regularization_warned = False
    
    @staticmethod
    def safe_solve(A: np.ndarray, b: np.ndarray, alpha: float, 
                   multiplier: int = MAX_REGULARIZATION_MULTIPLIER) -> np.ndarray:
        """Safely solve linear system with fallback regularization."""
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Only log the first time this happens per session
            if not LinAlgHelper._regularization_warned:
                logger.info("Using regularization for numerical stability (this message shown once)")
                LinAlgHelper._regularization_warned = True
            regularized_A = A + alpha * multiplier * np.eye(A.shape[0])
            return np.linalg.solve(regularized_A, b)
    
    @staticmethod
    def safe_inv(A: np.ndarray, use_pinv: bool = False) -> np.ndarray:
        """Safely invert matrix with fallback to pseudo-inverse."""
        try:
            return np.linalg.inv(A)
        except np.linalg.LinAlgError:
            if use_pinv:
                # Don't log this - it's expected in some cases
                return np.linalg.pinv(A)
            raise
    
    @staticmethod
    def check_condition_number(A: np.ndarray, threshold: float = CONDITION_NUMBER_THRESHOLD) -> Tuple[bool, float]:
        """Check if matrix is well-conditioned."""
        try:
            eigvals = np.linalg.eigvalsh(A)
            cond = eigvals.max() / max(eigvals.min(), 1e-10)
            return cond < threshold, cond
        except np.linalg.LinAlgError:
            return False, float('inf')


class ClusterStatsAggregator:
    """Handles cluster statistics aggregation and merging."""
    
    def __init__(self, n_features: int):
        self.n_features = n_features
    
    def create_empty_stats(self) -> Dict:
        """Create empty statistics dictionary."""
        return _default_cluster_stats(self.n_features)
    
    def update_stats(self, stats_dict: Dict, cluster_ids: np.ndarray,
                    X: np.ndarray, y: np.ndarray, errors: np.ndarray) -> None:
        """Update cluster statistics in place."""
        if cluster_ids is None:
            return
        
        if isinstance(cluster_ids, list):
            cluster_ids = np.array(cluster_ids)
        
        unique_clusters = np.unique(cluster_ids)
        for cluster_id in unique_clusters:
            mask = cluster_ids == cluster_id
            if not np.any(mask):
                continue
            
            Xc, ec, yc = X[mask], errors[mask], y[mask]
            
            if cluster_id not in stats_dict:
                stats_dict[cluster_id] = self.create_empty_stats()
            
            stats = stats_dict[cluster_id]
            stats['X_sum'] += np.sum(Xc, axis=0)
            stats['residual_sum'] += np.sum(ec)
            stats['count'] += Xc.shape[0]
            stats['XtX'] += Xc.T @ Xc
            stats['X_residual_sum'] += (Xc * ec.reshape(-1, 1)).sum(axis=0)
            stats['Xy'] += Xc.T @ yc
    
    def merge_stats(self, source: Dict, target: Dict) -> None:
        """Merge source statistics into target."""
        for cluster_id, stats in source.items():
            if cluster_id in target:
                target[cluster_id]['X_sum'] += stats['X_sum']
                target[cluster_id]['residual_sum'] += stats['residual_sum']
                target[cluster_id]['count'] += stats['count']
                target[cluster_id]['XtX'] += stats['XtX']
                target[cluster_id]['X_residual_sum'] += stats['X_residual_sum']
                target[cluster_id]['Xy'] += stats['Xy']
            else:
                target[cluster_id] = {
                    'X_sum': stats['X_sum'].copy(),
                    'residual_sum': stats['residual_sum'],
                    'count': stats['count'],
                    'XtX': stats['XtX'].copy(),
                    'X_residual_sum': stats['X_residual_sum'].copy(),
                    'Xy': stats['Xy'].copy()
                }


class OnlineRLS:
    """
    Online Recursive Least Squares with cluster-robust standard errors.
    Handles large datasets that don't fit in memory.
    """

    def __init__(self, n_features: int, alpha: float = DEFAULT_ALPHA, 
                 forget_factor: float = 1.0, batch_size: int = DEFAULT_BATCH_SIZE, 
                 feature_names: Optional[List[str]] = None,
                 se_type: Literal['stata', 'HC0', 'HC1'] = 'stata'):
        """Initialize Online RLS."""
        self.n_features = n_features
        self.alpha = alpha
        self.forget_factor = forget_factor
        self.batch_size = batch_size
        self.se_type = se_type
        
        self.feature_names = feature_names or [f"feature_{i}" for i in range(n_features)]
        
        # Initialize parameter estimates
        self.theta = np.zeros(n_features)
        self.P = (1.0 / alpha) * np.eye(n_features)
        
        # Sufficient statistics
        self.XtX = alpha * np.eye(n_features)
        self.Xty = np.zeros(n_features)
        
        # Model statistics
        self.n_obs = 0
        self.rss = 0.0
        self.sum_y = 0.0
        self.sum_y_squared = 0.0
        
        # IV-specific statistics
        self.iv_f_statistic = None  # Special F-stat for instrument strength
        self.iv_f_df = None         # (df_instr, df_resid)
        
        # Cluster statistics
        self.cluster_stats = {}
        self.cluster2_stats = {}
        self.intersection_stats = {}
        
        # Helper instances
        self._linalg = LinAlgHelper()
        self._cluster_aggregator = ClusterStatsAggregator(n_features)

    def _validate_and_clean_data(self, X: np.ndarray, y: np.ndarray, 
                                cluster1: Optional[np.ndarray] = None,
                                cluster2: Optional[np.ndarray] = None) -> Tuple:
        """Validate and clean input data."""
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
        
        if X.shape[0] == 0 or y.shape[0] == 0:
            logger.debug("Empty input data")
            return X, y, cluster1, cluster2
        
        # Validate finite values
        valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        n_invalid = (~valid_mask).sum()
        
        if n_invalid > 0:
            n_total = len(valid_mask)
            if n_invalid == n_total:
                logger.debug("No valid observations in chunk")
            elif n_invalid > n_total * 0.1:
                logger.debug(f"Removed {n_invalid}/{n_total} ({n_invalid/n_total*100:.1f}%) invalid observations")
            
            X = X[valid_mask]
            y = y[valid_mask]
            cluster1 = cluster1[valid_mask] if cluster1 is not None else None
            cluster2 = cluster2[valid_mask] if cluster2 is not None else None
        
        return X, y, cluster1, cluster2

    def partial_fit(self, X: np.ndarray, y: np.ndarray, 
                   cluster1: Optional[np.ndarray] = None,
                   cluster2: Optional[np.ndarray] = None) -> 'OnlineRLS':
        """Update RLS estimates with new batch of data."""
        X, y, cluster1, cluster2 = self._validate_and_clean_data(X, y, cluster1, cluster2)
        
        if X.shape[0] == 0:
            return self
        
        # Process in batches if needed
        if X.shape[0] <= self.batch_size:
            self._update_vectorized(X, y, cluster1, cluster2)
        else:
            for i in range(0, X.shape[0], self.batch_size):
                end_idx = min(i + self.batch_size, X.shape[0])
                self._update_vectorized(
                    X[i:end_idx], y[i:end_idx],
                    cluster1[i:end_idx] if cluster1 is not None else None,
                    cluster2[i:end_idx] if cluster2 is not None else None
                )
        
        return self
    
    def _update_vectorized(self, X: np.ndarray, y: np.ndarray,
                          cluster1: Optional[np.ndarray] = None,
                          cluster2: Optional[np.ndarray] = None) -> None:
        """Vectorized RLS update for a batch of observations."""
        n_batch = X.shape[0]
        
        # Update sufficient statistics
        self.XtX += X.T @ X
        self.Xty += X.T @ y
        self.n_obs += n_batch
        self.sum_y += np.sum(y)
        self.sum_y_squared += np.sum(y**2)
        
        # Solve for parameters with fallback
        self.theta = self._linalg.safe_solve(self.XtX, self.Xty, self.alpha)
        
        # Update precision matrix
        self.P = self._linalg.safe_inv(self.XtX, use_pinv=True)
        
        # Compute residuals for this batch
        errors = y - X @ self.theta
        
        # Update cluster statistics using aggregator
        self._cluster_aggregator.update_stats(self.cluster_stats, cluster1, X, y, errors)
        self._cluster_aggregator.update_stats(self.cluster2_stats, cluster2, X, y, errors)
        
        if cluster1 is not None and cluster2 is not None:
            intersection_ids = np.array([f"{cluster1[i]}_{cluster2[i]}" for i in range(len(cluster1))])
            self._cluster_aggregator.update_stats(self.intersection_stats, intersection_ids, X, y, errors)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return np.atleast_2d(X) @ self.theta
    
    def get_covariance_matrix(self) -> np.ndarray:
        """Get parameter covariance matrix (non-robust)."""
        dof = max(1, self.n_obs - self.n_features)
        sigma2 = self.rss / dof
        return sigma2 * self.P
    
    def get_cluster_robust_covariance(self, cluster_type: str = 'one_way') -> np.ndarray:
        """Compute cluster-robust covariance matrix."""
        # Validate se_type is compatible with clustering
        if self.se_type in ['HC2', 'HC3']:
            raise ValueError(f"se_type='{self.se_type}' is incompatible with clustering. "
                            f"Use 'stata', 'HC0', or 'HC1' with cluster_type='{cluster_type}'.")
            
        if cluster_type == 'one_way':
            return self._compute_cluster_covariance(self.cluster_stats)
        elif cluster_type == 'two_way':
            V1 = self._compute_cluster_covariance(self.cluster_stats)
            V2 = self._compute_cluster_covariance(self.cluster2_stats)
            V_int = self._compute_cluster_covariance(self.intersection_stats)
            return V1 + V2 - V_int
        else:
            raise ValueError("cluster_type must be 'one_way' or 'two_way'")
    
    def _compute_cluster_covariance(self, stats_dict: Dict) -> np.ndarray:
        """Compute cluster-robust covariance from cluster statistics."""
        XtX_inv = self.P
        
        # Check cluster count
        n_clusters = len([s for s in stats_dict.values() if s['count'] > 0])
        if n_clusters <= 1:
            warnings.warn("Insufficient clusters for robust standard errors, using classical")
            return self.get_covariance_matrix()
        
        # Note: HC2/HC3 are not implemented for cluster-robust standard errors
        # because they require cluster-level leverage adjustments which are incompatible 
        # with the standard cluster-robust variance estimator.
        
        # Standard meat matrix computation
        meat = np.zeros((self.n_features, self.n_features))
        for cluster_id, stats in stats_dict.items():
            # Use the pre-computed X_residual_sum which is the sum of X_i * e_i for the cluster
            # This is the correct score vector for cluster-robust standard errors
            score_vector = stats['X_residual_sum']
            meat += np.outer(score_vector, score_vector)
        
        # Check and regularize if needed
        is_well_conditioned, cond_number = self._linalg.check_condition_number(meat)
        if not is_well_conditioned:
            logger.warning(f"Meat matrix ill-conditioned (cond={cond_number:.2e}), regularizing")
            meat = self._regularize_meat_matrix(meat)
        
        # Apply corrections based on SE type
        correction = self._compute_small_sample_correction(n_clusters)
        
        # Sandwich estimator
        V = correction * XtX_inv @ meat @ XtX_inv
        V = (V + V.T) / 2  # Ensure symmetry
        
        # Validate result
        self._validate_covariance_matrix(V)
        
        return V
    
    def _regularize_meat_matrix(self, meat: np.ndarray) -> np.ndarray:
        """Regularize ill-conditioned meat matrix using eigenvalue floor."""
        eigvals, eigvecs = np.linalg.eigh(meat)
        eigval_floor = eigvals.max() * 1e-10
        eigvals_reg = np.maximum(eigvals, eigval_floor)
        return eigvecs @ np.diag(eigvals_reg) @ eigvecs.T
    
    def _compute_small_sample_correction(self, n_clusters: int) -> float:
        """Compute small sample correction factor based on SE type."""
        N = n_clusters
        NT = self.n_obs
        K = self.n_features
        
        if self.se_type == 'stata':
            # STATA: (N/(N-1)) * ((NT-1)/(NT-K))
            # Matches Stata's formula: (G/(G-1)) * ((N-1)/(N-K))
            correction = (N / (N - 1)) * ((NT - 1) / (NT - K))
        
        elif self.se_type == 'HC0':
            # HC0: No correction
            correction = 1.0
        
        elif self.se_type == 'HC1':
            # HC1: NT/(NT-K)
            correction = NT / (NT - K)
        
        else:
            # Default to stata
            correction = (N / (N - 1)) * ((NT - 1) / (NT - K))
        
        return correction
    
    def _validate_covariance_matrix(self, V: np.ndarray) -> None:
        """Validate covariance matrix for negative variances."""
        diag_V = np.diag(V)
        if np.any(diag_V < 0):
            n_negative = (diag_V < 0).sum()
            logger.error(f"Negative variances detected: {n_negative}/{len(diag_V)}")
    
    def get_standard_errors(self, cluster_type: str = 'classical') -> np.ndarray:
        """Get standard errors."""
        # Validate se_type is compatible with clustering
        if cluster_type != 'classical' and self.se_type in ['HC2', 'HC3']:
            raise ValueError(f"se_type='{self.se_type}' is incompatible with cluster_type='{cluster_type}'. "
                            f"Use 'stata', 'HC0', or 'HC1' with clustering.")
            
        # Ensure RSS is up-to-date before computing covariance
        self.rss = float(self.sum_y_squared - self.theta @ self.Xty)
        
        if cluster_type == 'classical':
            cov_matrix = self.get_covariance_matrix()
        else:
            cov_matrix = self.get_cluster_robust_covariance(cluster_type)
        
        # Get diagonal and ensure minimum value for stability
        diag_values = np.diag(cov_matrix)
        # Use a small positive value based on the maximum diagonal value
        min_se_value = np.sqrt(np.finfo(float).eps) * max(np.max(diag_values), self.alpha)
        
        # Take square root of non-negative values with minimum threshold
        return np.sqrt(np.maximum(diag_values, min_se_value))
    
    def diagnose_cluster_structure(self, cluster_stats: Dict, cluster_name: str = "Cluster") -> Dict[str, Any]:
        """Diagnose cluster structure and return diagnostic statistics."""
        if not cluster_stats:
            return {
                "n_clusters": 0,
                "min_size": 0,
                "max_size": 0,
                "mean_size": 0,
                "median_size": 0,
                "warnings": ["No clusters found"]
            }
        
        cluster_sizes = [stats['count'] for stats in cluster_stats.values()]
        n_clusters = len(cluster_sizes)
        
        diagnostics = {
            "n_clusters": n_clusters,
            "min_size": int(np.min(cluster_sizes)),
            "max_size": int(np.max(cluster_sizes)),
            "mean_size": float(np.mean(cluster_sizes)),
            "median_size": float(np.median(cluster_sizes)),
            "total_obs": sum(cluster_sizes),
            "warnings": []
        }
        
        # Check for issues
        if n_clusters < 10:
            diagnostics["warnings"].append(
                f"Few clusters ({n_clusters}). Cluster-robust SEs may be unreliable with <10 clusters."
            )
        
        if diagnostics["min_size"] < MIN_CLUSTER_SIZE:
            diagnostics["warnings"].append(
                f"Small clusters detected (min={diagnostics['min_size']}). May lead to imprecise estimates."
            )
        
        if diagnostics["max_size"] > 10 * diagnostics["mean_size"]:
            diagnostics["warnings"].append(
                f"Unbalanced clusters (max/mean={diagnostics['max_size']/diagnostics['mean_size']:.1f})."
            )
        
        return diagnostics

    def summary(self, cluster_type: str = 'classical', **kwargs) -> pd.DataFrame:
        """Get regression summary."""
        if cluster_type != 'classical':
            self._log_cluster_report(cluster_type)
        
        se = self.get_standard_errors(cluster_type)
        t_stats = self.theta / se
        
        from scipy import stats
        p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
        
        return pd.DataFrame({
            'coefficient': self.theta,
            'std_error': se,
            't_statistic': t_stats,
            'p_value': p_values
        }, index=self.feature_names)
    
    def _log_cluster_report(self, cluster_type: str) -> None:
        """Log comprehensive cluster diagnostics report."""
        report_lines = [
            "",
            "=" * 80,
            f"CLUSTER-ROBUST STANDARD ERRORS: {cluster_type.upper()}",
            "=" * 80
        ]
        
        if cluster_type == 'one_way':
            diag = self.diagnose_cluster_structure(self.cluster_stats, "Cluster")
            report_lines.extend([
                f"Cluster Variable: {diag['n_clusters']} clusters",
                f"  Size Range: {diag['min_size']:,} - {diag['max_size']:,}",
                f"  Mean Size: {diag['mean_size']:.1f}",
                f"  Median Size: {diag['median_size']:.1f}",
                f"  Total Observations: {diag['total_obs']:,}"
            ])
            
            if diag['warnings']:
                report_lines.append("")
                report_lines.append("WARNINGS:")
                for warning in diag['warnings']:
                    report_lines.append(f"  ⚠ {warning}")
        
        elif cluster_type == 'two_way':
            diag1 = self.diagnose_cluster_structure(self.cluster_stats, "Cluster1")
            diag2 = self.diagnose_cluster_structure(self.cluster2_stats, "Cluster2")
            diag_int = self.diagnose_cluster_structure(self.intersection_stats, "Intersection")
            
            report_lines.extend([
                f"Dimension 1: {diag1['n_clusters']} clusters",
                f"  Size Range: {diag1['min_size']:,} - {diag1['max_size']:,}",
                f"  Mean Size: {diag1['mean_size']:.1f}",
                f"  Median Size: {diag1['median_size']:.1f}",
                "",
                f"Dimension 2: {diag2['n_clusters']} clusters",
                f"  Size Range: {diag2['min_size']:,} - {diag2['max_size']:,}",
                f"  Mean Size: {diag2['mean_size']:.1f}",
                f"  Median Size: {diag2['median_size']:.1f}",
                "",
                f"Intersection: {diag_int['n_clusters']} unique clusters",
                f"  Total Observations: {diag1['total_obs']:,}"
            ])
            
            all_warnings = diag1['warnings'] + diag2['warnings']
            if all_warnings:
                report_lines.append("")
                report_lines.append("WARNINGS:")
                for warning in set(all_warnings):  # Remove duplicates
                    report_lines.append(f"  ⚠ {warning}")
        
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Log as single multi-line message
        logger.info("\n".join(report_lines))

    def get_total_sum_squares(self) -> float:
        """Compute the total sum of squares (TSS)."""
        if self.n_obs <= 1:
            return 0.0
        mean_y = self.sum_y / self.n_obs
        return self.sum_y_squared - self.n_obs * (mean_y ** 2)
        
    def get_r_squared(self) -> float:
        """Calculate R-squared."""
        # Recompute RSS from full data using sufficient statistics
        # RSS = y'y - 2*theta'X'y + theta'X'X*theta
        #     = y'y - theta'X'y  (since theta solves X'X*theta = X'y)
        self.rss = float(self.sum_y_squared - self.theta @ self.Xty)
        
        tss = self.get_total_sum_squares()
        if tss <= 0:
            return 0.0
        return max(0.0, 1.0 - (self.rss / tss))
        
    def get_adjusted_r_squared(self) -> float:
        """Calculate adjusted R-squared."""
        if self.n_obs <= self.n_features:
            return 0.0
        r_squared = self.get_r_squared()
        n = self.n_obs
        k = self.n_features
        return 1.0 - ((1.0 - r_squared) * (n - 1) / (n - k))
    
    def get_f_statistic(self) -> Tuple[float, int, int]:
        """
        Calculate F-statistic for overall model significance.
        
        Returns:
        --------
        f_stat : float
            F-statistic value
        df_model : int
            Degrees of freedom for model (k - 1, excluding intercept if present)
        df_resid : int
            Degrees of freedom for residuals (n - k)
        """
        if self.n_obs <= self.n_features:
            return 0.0, 0, 0
        
        # Ensure RSS is up-to-date
        self.rss = float(self.sum_y_squared - self.theta @ self.Xty)
        
        # Determine if intercept is included (first feature named 'intercept')
        has_intercept = (self.feature_names[0].lower() == 'intercept' if self.feature_names else False)
        
        # Degrees of freedom
        k = self.n_features
        n = self.n_obs
        df_model = k - 1 if has_intercept else k  # Exclude intercept from model df
        df_resid = n - k
        
        if df_model <= 0 or df_resid <= 0:
            return 0.0, df_model, df_resid
        
        # Calculate F-statistic
        r_squared = self.get_r_squared()
        if r_squared >= 1.0:
            return float('inf'), df_model, df_resid
        
        f_stat = (r_squared / df_model) / ((1 - r_squared) / df_resid)
        
        return f_stat, df_model, df_resid
    
    def get_f_pvalue(self) -> float:
        """Get p-value for F-statistic."""
        from scipy import stats
        
        f_stat, df_model, df_resid = self.get_f_statistic()
        
        if f_stat == 0.0 or df_model <= 0 or df_resid <= 0:
            return 1.0
        
        if f_stat == float('inf'):
            return 0.0
        
        return 1 - stats.f.cdf(f_stat, df_model, df_resid)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names used in the model."""
        return self.feature_names.copy()

    def merge_statistics(self, other: 'OnlineRLS') -> None:
        """Merge statistics from another OnlineRLS instance."""
        # Merge sufficient statistics
        self.XtX += other.XtX - self.alpha * np.eye(self.n_features)
        self.Xty += other.Xty
        self.n_obs += other.n_obs
        self.rss += other.rss
        self.sum_y += other.sum_y
        self.sum_y_squared += other.sum_y_squared
        
        # Recompute parameters
        self.theta = self._linalg.safe_solve(self.XtX, self.Xty, self.alpha)
        self.P = self._linalg.safe_inv(self.XtX, use_pinv=True)
        
        # Merge cluster statistics using aggregator
        self._cluster_aggregator.merge_stats(other.cluster_stats, self.cluster_stats)
        self._cluster_aggregator.merge_stats(other.cluster2_stats, self.cluster2_stats)
        self._cluster_aggregator.merge_stats(other.intersection_stats, self.intersection_stats)


class ChunkWorker:
    """Worker class for processing a single chunk."""
    
    @staticmethod
    def process_chunk(payload: Union[pd.DataFrame, DataChunkTask], spec: WorkerChunkSpec) -> ChunkResult:
        """Process a single chunk of data."""
        try:
            if isinstance(payload, DataChunkTask):
                chunk_df = ChunkWorker._load_parquet_chunk(payload)
            else:
                chunk_df = payload
            return ChunkWorker._process_chunk_impl(chunk_df, spec)
        except Exception as e:
            logger.error(f"Chunk {spec.chunk_id} failed: {str(e)}")
            return ChunkWorker._empty_result(spec, str(e))
    
    @staticmethod
    def _process_chunk_impl(chunk_df: pd.DataFrame, spec: WorkerChunkSpec) -> ChunkResult:
        """Implementation of chunk processing."""
        X, y = ChunkWorker._extract_data(chunk_df, spec)
        if X.shape[0] == 0:
            return ChunkWorker._empty_result(spec, "Empty after validation")
        X = ChunkWorker._apply_transformations(X, spec)
        XtX, Xty, theta = ChunkWorker._compute_statistics(X, y, spec.alpha, spec.n_features)
        errors = y - X @ theta
        cluster_stats = ChunkWorker._compute_all_cluster_stats(chunk_df, X, y, errors, spec)
        return ChunkResult(
            chunk_id=spec.chunk_id,
            partition_idx=spec.partition_idx,
            XtX=XtX,
            Xty=Xty,
            n_obs=X.shape[0],
            sum_y=float(np.sum(y)),
            sum_y_squared=float(np.sum(y**2)),
            **cluster_stats
        )
    
    @staticmethod
    def _extract_data(chunk_df: pd.DataFrame, spec: WorkerChunkSpec) -> Tuple[np.ndarray, np.ndarray]:
        """Extract and validate X and y from chunk."""
        X = chunk_df[spec.feature_cols].values
        y = chunk_df[spec.target_col].values
        
        # Convert to numeric dtypes
        try:
            X = X.astype(np.float32)
            y = y.astype(np.float32)
        except (ValueError, TypeError) as e:
            logger.error(f"Cannot convert features/target to numeric: {e}")
            return np.array([]).reshape(0, len(spec.feature_cols)), np.array([])
        
        # Validate finite values
        valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        
        # Handle empty mask gracefully
        if len(valid_mask) == 0 or valid_mask.sum() == 0:
            return np.array([]).reshape(0, len(spec.feature_cols)), np.array([])
        
        # Check if too many invalid observations
        valid_ratio = valid_mask.sum() / len(valid_mask)
        if valid_ratio < MIN_VALID_DATA_RATIO:
            return np.array([]).reshape(0, len(spec.feature_cols)), np.array([])
        
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        # Check if we need to append instruments (for 2SLS second stage)
        if spec.feature_engineering_config and '_extra_input_columns' in spec.feature_engineering_config:
            extra_cols = spec.feature_engineering_config['_extra_input_columns']
            # Make sure extra columns exist in the chunk
            missing_cols = [col for col in extra_cols if col not in chunk_df.columns]
            if missing_cols:
                logger.warning(f"Instruments not in chunk (query may have filtered them): {missing_cols}")
            else:
                try:
                    Z = chunk_df[extra_cols].values[valid_mask]
                    Z = Z.astype(np.float32)
                    # Only append if all instrument values are finite
                    if np.isfinite(Z).all():
                        X_valid = np.column_stack([X_valid, Z])
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not load instruments: {e}")
        
        return X_valid, y_valid
    
    @staticmethod
    def _apply_transformations(X: np.ndarray, spec: WorkerChunkSpec) -> np.ndarray:
        """Apply feature engineering transformations."""
        if spec.feature_engineering_config or spec.add_intercept:
            from streamreg.transforms import FeatureTransformer
            fe_config = spec.feature_engineering_config.copy() if spec.feature_engineering_config else {}
            fe_config.pop('_extra_input_columns', None)
            fe_config.pop('_base_feature_count', None)
            
            # Handle case where config might be a list (for backward compatibility)
            if isinstance(fe_config, list):
                fe_config = {'transformations': fe_config}
            
            # Extract transformations
            transformations = fe_config.get('transformations', []) if isinstance(fe_config, dict) else []
            
            transformer = FeatureTransformer.from_config(
                {'transformations': transformations},
                spec.feature_cols,
                add_intercept=spec.add_intercept
            )
            return transformer.transform(X, spec.feature_cols)
        elif spec.add_intercept:
            intercept = np.ones((X.shape[0], 1), dtype=np.float32)
            return np.column_stack([intercept, X])
        return X
    
    @staticmethod
    def _compute_statistics(X: np.ndarray, y: np.ndarray, alpha: float, 
                          n_features: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute sufficient statistics."""
        # Handle empty arrays
        if X.shape[0] == 0 or X.shape[0] < n_features:
            XtX = alpha * np.eye(n_features)
            Xty = np.zeros(n_features)
            theta = np.zeros(n_features)
            return XtX, Xty, theta

        XtX = X.T @ X
        Xty = X.T @ y
        
        # Add regularization for numerical stability
        XtX_reg = XtX + alpha * np.eye(n_features)
        
        # Solve using a robust approach
        try:
            theta = np.linalg.solve(XtX_reg, Xty)
        except np.linalg.LinAlgError:
            # Fall back to more stable pseudo-inverse
            theta = np.linalg.pinv(XtX_reg) @ Xty
            
        return XtX, Xty, theta
    
    @staticmethod
    def _compute_all_cluster_stats(chunk_df: pd.DataFrame, X: np.ndarray, y: np.ndarray,
                                  errors: np.ndarray, spec: WorkerChunkSpec) -> Dict:
        """Compute all cluster statistics."""
        valid_mask_features = np.isfinite(chunk_df[spec.feature_cols].values.astype(np.float32)).all(axis=1)
        valid_mask_target = np.isfinite(chunk_df[spec.target_col].values.astype(np.float32))
        valid_mask = valid_mask_features & valid_mask_target
        
        cluster1 = chunk_df[spec.cluster1_col].values[valid_mask] if spec.cluster1_col else None
        cluster2 = chunk_df[spec.cluster2_col].values[valid_mask] if spec.cluster2_col else None
        
        aggregator = ClusterStatsAggregator(spec.n_features)
        
        cluster_stats_dict = {}
        cluster2_stats_dict = {}
        intersection_stats_dict = {}
        
        if cluster1 is not None:
            aggregator.update_stats(cluster_stats_dict, cluster1, X, y, errors)
        
        if cluster2 is not None:
            aggregator.update_stats(cluster2_stats_dict, cluster2, X, y, errors)
        
        if cluster1 is not None and cluster2 is not None:
            intersection_ids = np.array([f"{cluster1[i]}_{cluster2[i]}" for i in range(len(cluster1))])
            aggregator.update_stats(intersection_stats_dict, intersection_ids, X, y, errors)
        
        return {
            'cluster_stats': cluster_stats_dict,
            'cluster2_stats': cluster2_stats_dict,
            'intersection_stats': intersection_stats_dict
        }
    
    @staticmethod
    def _empty_result(spec: WorkerChunkSpec, error: str) -> ChunkResult:
        """Create empty result for failed chunk."""
        return ChunkResult(
            chunk_id=spec.chunk_id,
            partition_idx=spec.partition_idx,
            XtX=spec.alpha * np.eye(spec.n_features),
            Xty=np.zeros(spec.n_features),
            n_obs=0,
            sum_y=0.0,
            sum_y_squared=0.0,
            cluster_stats={},
            cluster2_stats={},
            intersection_stats={},
            success=False,
            error=error
        )

    @staticmethod
    def _load_parquet_chunk(task: DataChunkTask) -> pd.DataFrame:
        """Materialize a parquet task payload into a DataFrame."""
        import pyarrow.parquet as pq
        
        try:
            # Use PyArrow for efficient row group reading
            pq_file = pq.ParquetFile(task.file_path)
            
            # Determine columns to read
            columns = task.columns if task.columns else None
            
            if task.row_groups:
                # Read specific row groups using PyArrow's efficient method
                if len(task.row_groups) == 1:
                    # Single row group - use read_row_group directly
                    table = pq_file.read_row_group(
                        task.row_groups[0],
                        columns=columns,
                        use_threads=True,
                        use_pandas_metadata=False
                    )
                    df = table.to_pandas()
                else:
                    # Multiple row groups - read and concatenate
                    tables = [
                        pq_file.read_row_group(
                            rg,
                            columns=columns,
                            use_threads=True,
                            use_pandas_metadata=False
                        )
                        for rg in task.row_groups
                    ]
                    import pyarrow as pa
                    combined_table = pa.concat_tables(tables)
                    df = combined_table.to_pandas()
            else:
                # Read entire file or specific row range
                table = pq_file.read(columns=columns, use_threads=True)
                df = table.to_pandas()
            
            # Apply row slicing if specified
            if task.row_start is not None or task.row_end is not None:
                start = task.row_start or 0
                end = task.row_end or len(df)
                df = df.iloc[start:end]
            
            # Apply query filter if filters weren't applied and query exists
            if (not task.filters) and task.query:
                df = df.query(task.query)
            
            return df
            
        except Exception as e:
            logger.warning(f"PyArrow read failed ({e}), falling back to fastparquet")
            # Fallback to fastparquet
            parquet_file = fp.ParquetFile(task.file_path)
            kwargs: Dict[str, Any] = {}
            if task.columns is not None:
                kwargs['columns'] = task.columns
            if task.filters:
                kwargs['filters'] = task.filters
            if task.row_groups:
                df = parquet_file.to_pandas(row_groups=task.row_groups, **kwargs)
            else:
                df = parquet_file.to_pandas(**kwargs)
            if task.row_start is not None or task.row_end is not None:
                start = task.row_start or 0
                end = task.row_end or len(df)
                df = df.iloc[start:end]
            if (not task.filters) and task.query:
                df = df.query(task.query)
            return df


class ParallelOLSOrchestrator:
    """Orchestrates parallel OLS estimation using unified StreamData interface."""
    
    def __init__(self, data, feature_cols: List[str], target_col: str,
                 cluster1_col: Optional[str] = None, cluster2_col: Optional[str] = None,
                 add_intercept: bool = True, n_features: int = None,
                 transformed_feature_names: List[str] = None, alpha: float = DEFAULT_ALPHA,
                 chunk_size: int = DEFAULT_CHUNK_SIZE, n_workers: Optional[int] = None,
                 show_progress: bool = True, verbose: bool = True,
                 feature_engineering: Optional[Dict[str, Any]] = None,
                 se_type: Literal['stata', 'HC0', 'HC1'] = 'stata'):
        """Initialize orchestrator."""
        self.data = data
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.cluster1_col = cluster1_col
        self.cluster2_col = cluster2_col
        self.add_intercept = add_intercept
        self.n_features = n_features
        self.transformed_feature_names = transformed_feature_names
        self.alpha = alpha
        self.chunk_size = chunk_size
        self.n_workers = n_workers or self._get_optimal_workers()
        self.show_progress = show_progress
        self.verbose = verbose
        self.feature_engineering = feature_engineering
        self.se_type = se_type
    
    def fit(self) -> OnlineRLS:
        """Execute parallel fitting."""
        start_time = time.time()
        
        logger.info(f"Starting parallel OLS: {self.n_workers} workers, data_type={self.data.info.source_type}, se_type={self.se_type}")
        
        # Initialize main RLS
        main_rls = OnlineRLS(
            n_features=self.n_features,
            alpha=self.alpha,
            feature_names=self.transformed_feature_names,
            se_type=self.se_type
        )
        
        # Get required columns
        load_cols = self._get_required_columns()
        
        # Process chunks based on n_workers setting
        if self.n_workers > 1:
            # Parallel processing using joblib
            results = self._process_chunks_joblib(load_cols)
        else:
            # Sequential processing
            results = self._process_chunks_sequential(load_cols)
        
        # Merge results
        self._merge_results_into_model(results, main_rls)
        
        # Report
        self._log_completion(start_time, results, main_rls)
        
        return main_rls
    
    def _get_required_columns(self) -> List[str]:
        """Get list of required columns to load."""
        cols = self.feature_cols + [self.target_col]
        if self.cluster1_col:
            cols.append(self.cluster1_col)
        if self.cluster2_col:
            cols.append(self.cluster2_col)
        
        # Add extra columns if specified (e.g., instruments for 2SLS second stage)
        if self.feature_engineering and '_extra_input_columns' in self.feature_engineering:
            extra_cols = self.feature_engineering['_extra_input_columns']
            cols.extend(extra_cols)
        
        return cols
    
    def _process_chunks_sequential(self, load_cols: List[str]) -> List[ChunkResult]:
        """Process all chunks sequentially (n_workers=1)."""
        results = []
        pbar = self._create_progress_bar() if self.show_progress else None
        for chunk_id, payload in self._iter_chunk_payloads(load_cols):
            partition_idx = getattr(payload, 'partition_idx', 0) or 0
            spec = self._create_worker_spec(chunk_id, partition_idx)
            result = ChunkWorker.process_chunk(payload, spec)
            results.append(result)
            if pbar is not None:
                if pbar.n >= pbar.total:
                    pbar.total = pbar.n + 1
                pbar.update(1)
                self._update_progress_postfix(pbar, results)
        if pbar is not None:
            pbar.total = len(results)
            pbar.refresh()
            pbar.close()
        return results

    def _process_chunks_joblib(self, load_cols: List[str]) -> List[ChunkResult]:
        """Process all chunks in parallel using joblib."""
        chunk_inputs = []
        for chunk_id, payload in self._iter_chunk_payloads(load_cols):
            partition_idx = getattr(payload, 'partition_idx', 0) or 0
            spec = self._create_worker_spec(chunk_id, partition_idx)
            chunk_inputs.append((payload, spec))
        n_chunks = len(chunk_inputs)
        import sys
        backend = 'threading' if sys.gettrace() is not None else 'loky'
        if self.show_progress:
            from tqdm.auto import tqdm
            generator = Parallel(
                n_jobs=self.n_workers,
                backend=backend,
                verbose=0,
                return_as='generator'
            )(
                delayed(ChunkWorker.process_chunk)(payload, spec)
                for payload, spec in chunk_inputs
            )
            results = list(tqdm(generator, total=n_chunks, desc="Processing chunks", unit="chunks"))
        else:
            results = Parallel(
                n_jobs=self.n_workers,
                backend=backend,
                verbose=0
            )(
                delayed(ChunkWorker.process_chunk)(payload, spec)
                for payload, spec in chunk_inputs
            )
        return results
    
    def _merge_results_into_model(self, results: List[ChunkResult], main_rls: OnlineRLS) -> None:
        """Merge chunk results into main model."""
        for result in results:
            if not result.success or result.n_obs == 0:
                continue
            
            temp_rls = OnlineRLS(n_features=self.n_features, alpha=self.alpha, se_type=self.se_type)
            temp_rls.XtX = result.XtX
            temp_rls.Xty = result.Xty
            temp_rls.n_obs = result.n_obs
            temp_rls.sum_y = result.sum_y
            temp_rls.sum_y_squared = result.sum_y_squared
            temp_rls.cluster_stats = result.cluster_stats
            temp_rls.cluster2_stats = result.cluster2_stats
            temp_rls.intersection_stats = result.intersection_stats
            temp_rls.theta = LinAlgHelper.safe_solve(temp_rls.XtX, temp_rls.Xty, self.alpha)
            
            main_rls.merge_statistics(temp_rls)
    
    def _log_completion(self, start_time: float, results: List[ChunkResult], main_rls: OnlineRLS) -> None:
        """Log completion statistics."""
        elapsed = time.time() - start_time
        success_count = sum(1 for r in results if r.success)
        
        logger.info(
            f"Completed in {elapsed:.1f}s: {success_count}/{len(results)} chunks, "
            f"{main_rls.n_obs:,} obs, R²={main_rls.get_r_squared():.4f}"
        )
        
        if success_count < len(results) * 0.5:
            logger.warning(f"Low success rate: {success_count/len(results)*100:.1f}%")
    
    def _create_progress_bar(self) -> tqdm:
        """Create progress bar with correct total chunks."""
        if self.data.info.source_type == 'dataframe':
            # For DataFrame, use filtered row count
            total_rows = self.data.info.n_rows
            estimated_chunks = max(1, (total_rows + self.chunk_size - 1) // self.chunk_size)
            return tqdm(total=estimated_chunks, desc="Processing chunks", unit="chunks")
        else:
            # For parquet/partitioned, use estimated total, will adjust at end
            total_rows = self.data.info.n_rows
            estimated_chunks = max(1, total_rows // self.chunk_size)
            return tqdm(total=estimated_chunks, desc="Processing chunks", unit="chunks")

    def _create_worker_spec(self, chunk_id: int, partition_idx: int = 0) -> WorkerChunkSpec:
        """Create worker configuration for a chunk."""
        return WorkerChunkSpec(
            chunk_id=chunk_id,
            partition_idx=partition_idx,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            cluster1_col=self.cluster1_col,
            cluster2_col=self.cluster2_col,
            add_intercept=self.add_intercept,
            n_features=self.n_features,
            alpha=self.alpha,
            feature_engineering_config=self.feature_engineering
        )

    def _iter_chunk_payloads(self, load_cols: List[str]):
        """Yield chunk payloads (DataFrame or parquet task) with chunk ids."""
        if self.data.info.source_type == 'dataframe':
            for idx, chunk_df in enumerate(self.data.iter_chunks(columns=load_cols)):
                yield idx, chunk_df
        else:
            yield from self.data.iter_tasks(requested_columns=load_cols)
    
    def _get_optimal_workers(self) -> int:
        """Determine optimal number of workers."""
        import os
        
        # Try to get from environment first
        n_workers = os.environ.get('STREAMREG_N_WORKERS')
        if n_workers:
            try:
                return int(n_workers)
            except ValueError:
                pass
        
        # Use CPU count
        try:
            cpu_count = mp.cpu_count()
            # Use at most 80% of CPUs to avoid overloading
            return max(1, int(cpu_count * 0.8))
        except NotImplementedError:
            return 4  # Reasonable default
    
    def _update_progress_postfix(self, pbar: tqdm, results: List[ChunkResult]) -> None:
        """Update progress bar postfix with current statistics."""
        if not results:
            return
        
        success_count = sum(1 for r in results if r.success)
        total_obs = sum(r.n_obs for r in results if r.success)
        
        pbar.set_postfix({
            'success': f"{success_count}/{len(results)}",
            'obs': f"{total_obs:,}"
        })
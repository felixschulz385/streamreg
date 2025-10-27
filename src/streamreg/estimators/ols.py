import time
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.distributed import progress
from typing import Dict, List, Tuple, Optional, Literal, Any
import warnings
import logging

logger = logging.getLogger(__name__)

# Constants
DEFAULT_ALPHA = 1e-3
CONDITION_NUMBER_THRESHOLD = 1e12
MIN_CLUSTER_SIZE = 5

def _default_cluster_stats(n_features):
    """Create default cluster stats dictionary."""
    return {
        'X_sum': np.zeros(n_features),
        'residual_sum': 0.0,
        'count': 0,
        'XtX': np.zeros((n_features, n_features)),
        'X_residual_sum': np.zeros(n_features),
        'Xy': np.zeros(n_features)
    }


class LinAlgHelper:
    """Helper class for common linear algebra operations with error handling."""
    
    _regularization_warned = False
    
    @staticmethod
    def safe_solve(A: np.ndarray, b: np.ndarray, alpha: float, 
                   multiplier: int = 10) -> np.ndarray:
        """Safely solve linear system with fallback regularization."""
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
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


# Optimized helper functions outside the class
def _fast_update_stats(stats_dict: Dict, cluster_ids,
                       X: np.ndarray, y: np.ndarray, errors: np.ndarray,
                       n_features: int) -> None:
    """Optimized cluster statistics update with NaN handling."""
    if cluster_ids is None or len(cluster_ids) == 0:
        return
    
    import pandas as pd
    
    # Convert cluster_ids to numpy array for consistent handling
    cluster_ids = np.asarray(cluster_ids)
    
    # Handle 2D array (intersection case with shape (n, 2))
    if cluster_ids.ndim == 2:
        # Use pandas isna for robust NaN detection across all types
        valid_mask = ~pd.isna(cluster_ids).any(axis=1)
        
        if not np.any(valid_mask):
            return
        
        cluster_ids = cluster_ids[valid_mask]
        X = X[valid_mask]
        y = y[valid_mask]
        errors = errors[valid_mask]
        
        # Hash the pairs into unique integers for fast grouping
        if cluster_ids.dtype.kind in ['U', 'S', 'O']:  # String types
            cluster_hash = np.array([hash(tuple(row)) for row in cluster_ids])
        else:
            # For numeric types, create compound key
            # Use safer conversion for potential floats
            try:
                c1 = cluster_ids[:, 0].astype(np.int64)
                c2 = cluster_ids[:, 1].astype(np.int64)
                max_val = int(np.max(c2)) + 1
                cluster_hash = c1 * max_val + c2
            except (ValueError, OverflowError):
                # Fallback to hashing for very large values
                cluster_hash = np.array([hash(tuple(row)) for row in cluster_ids])
        
        unique_hashes, inverse_indices = np.unique(cluster_hash, return_inverse=True)
        
        # Map back to original tuple format for dictionary keys
        hash_to_tuple = {}
        for i, h in enumerate(unique_hashes):
            idx = np.where(cluster_hash == h)[0][0]
            hash_to_tuple[h] = tuple(cluster_ids[idx])
        
        for i, h in enumerate(unique_hashes):
            mask = inverse_indices == i
            
            if not np.any(mask):
                continue
            
            Xc = X[mask]
            ec = errors[mask]
            yc = y[mask]
            
            cluster_id = hash_to_tuple[h]
            
            if cluster_id not in stats_dict:
                stats_dict[cluster_id] = {
                    'X_sum': Xc.sum(axis=0),
                    'residual_sum': ec.sum(),
                    'count': len(Xc),
                    'XtX': Xc.T @ Xc,
                    'X_residual_sum': Xc.T @ ec,
                    'Xy': Xc.T @ yc
                }
            else:
                stats = stats_dict[cluster_id]
                stats['X_sum'] += Xc.sum(axis=0)
                stats['residual_sum'] += ec.sum()
                stats['count'] += len(Xc)
                stats['XtX'] += Xc.T @ Xc
                stats['X_residual_sum'] += Xc.T @ ec
                stats['Xy'] += Xc.T @ yc
    else:
        # 1D array (regular cluster case)
        if cluster_ids.ndim > 1:
            cluster_ids = cluster_ids.ravel()
        
        # Use pandas isna for robust NaN detection
        valid_mask = ~pd.isna(cluster_ids)
        
        if not np.any(valid_mask):
            return
        
        cluster_ids = cluster_ids[valid_mask]
        X = X[valid_mask]
        y = y[valid_mask]
        errors = errors[valid_mask]
        
        unique_clusters, inverse_indices = np.unique(cluster_ids, return_inverse=True)
        
        for i, cluster_id in enumerate(unique_clusters):
            mask = inverse_indices == i
            
            if not np.any(mask):
                continue
            
            Xc = X[mask]
            ec = errors[mask]
            yc = y[mask]
            
            if cluster_id not in stats_dict:
                stats_dict[cluster_id] = {
                    'X_sum': Xc.sum(axis=0),
                    'residual_sum': ec.sum(),
                    'count': len(Xc),
                    'XtX': Xc.T @ Xc,
                    'X_residual_sum': Xc.T @ ec,
                    'Xy': Xc.T @ yc
                }
            else:
                stats = stats_dict[cluster_id]
                stats['X_sum'] += Xc.sum(axis=0)
                stats['residual_sum'] += ec.sum()
                stats['count'] += len(Xc)
                stats['XtX'] += Xc.T @ Xc
                stats['X_residual_sum'] += Xc.T @ ec
                stats['Xy'] += Xc.T @ yc
                
def _fast_merge_stats(source: Dict, target: Dict) -> None:
    """Optimized statistics merge."""
    for cluster_id, stats in source.items():
        if cluster_id in target:
            t = target[cluster_id]
            t['X_sum'] += stats['X_sum']
            t['residual_sum'] += stats['residual_sum']
            t['count'] += stats['count']
            t['XtX'] += stats['XtX']
            t['X_residual_sum'] += stats['X_residual_sum']
            t['Xy'] += stats['Xy']
        else:
            # Shallow copy is fine - numpy arrays are references
            target[cluster_id] = stats

class ClusterStatsAggregator:
    """Handles cluster statistics aggregation and merging."""
    
    def __init__(self, n_features: int):
        self.n_features = n_features
    
    def create_empty_stats(self) -> Dict:
        """Create empty statistics dictionary."""
        return {
            'X_sum': np.zeros(self.n_features, dtype=np.float32),
            'residual_sum': 0.0,
            'count': 0,
            'XtX': np.zeros((self.n_features, self.n_features), dtype=np.float32),
            'X_residual_sum': np.zeros(self.n_features, dtype=np.float32),
            'Xy': np.zeros(self.n_features, dtype=np.float32)
        }
    
    def update_stats(self, stats_dict: Dict, cluster_ids: np.ndarray,
                     X: np.ndarray, y: np.ndarray, errors: np.ndarray) -> None:
        """Update cluster statistics in place."""
        _fast_update_stats(stats_dict, cluster_ids, X, y, errors, self.n_features)
    
    def merge_stats(self, source: Dict, target: Dict) -> None:
        """Merge source statistics into target."""
        _fast_merge_stats(source, target)


class OnlineRLS:
    """
    Online Recursive Least Squares with cluster-robust standard errors.
    """

    def __init__(self, n_features: int, alpha: float = DEFAULT_ALPHA, 
                 feature_names: Optional[List[str]] = None,
                 se_type: Literal['stata', 'HC0', 'HC1'] = 'stata'):
        """Initialize Online RLS."""
        self.n_features = n_features
        self.alpha = alpha
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
        
        # Cluster statistics
        self.cluster_stats = {}
        self.cluster2_stats = {}
        self.intersection_stats = {}
        
        # Helper instances
        self._linalg = LinAlgHelper()
        self._cluster_aggregator = ClusterStatsAggregator(n_features)

    def _update(self, X: np.ndarray, y: np.ndarray,
                          cluster1: Optional[np.ndarray] = None,
                          cluster2: Optional[np.ndarray] = None) -> None:
        """Update for a batch of observations."""
        n_batch = len(X)
        
        self.XtX += X.T @ X
        self.Xty += X.T @ y
        self.n_obs += n_batch
        self.sum_y += y.sum()
        self.sum_y_squared += (y ** 2).sum()
        
        self.theta = self._linalg.safe_solve(self.XtX, self.Xty, self.alpha)
        self.P = self._linalg.safe_inv(self.XtX, use_pinv=True)
        
        errors = y - X @ self.theta
        
        self._cluster_aggregator.update_stats(self.cluster_stats, cluster1, X, y, errors)
        self._cluster_aggregator.update_stats(self.cluster2_stats, cluster2, X, y, errors)
        
        if cluster1 is not None and cluster2 is not None:
            intersection_ids = np.char.add(np.char.add(cluster1.astype(str), '_'), cluster2.astype(str))
            self._cluster_aggregator.update_stats(self.intersection_stats, intersection_ids, X, y, errors)

    def get_covariance_matrix(self) -> np.ndarray:
        """Get parameter covariance matrix (non-robust)."""
        dof = max(1, self.n_obs - self.n_features)
        sigma2 = self.rss / dof
        return sigma2 * self.P
    
    def get_cluster_robust_covariance(self, cluster_type: str = 'one_way') -> np.ndarray:
        """Compute cluster-robust covariance matrix."""
        if self.se_type in ['HC2', 'HC3']:
            raise ValueError(f"se_type='{self.se_type}' is incompatible with clustering.")
            
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
        
        active_clusters = [s for s in stats_dict.values() if s['count'] > 0]
        n_clusters = len(active_clusters)
        
        if n_clusters <= 1:
            warnings.warn("Insufficient clusters for robust standard errors, using classical")
            return self.get_covariance_matrix()
        
        # Vectorized meat computation
        score_vectors = np.array([stats['X_residual_sum'] for stats in active_clusters])
        meat = score_vectors.T @ score_vectors
        
        is_well_conditioned, cond_number = self._linalg.check_condition_number(meat)
        if not is_well_conditioned:
            logger.warning(f"Meat matrix ill-conditioned (cond={cond_number:.2e}), regularizing")
            meat = self._regularize_meat_matrix(meat)
        
        correction = self._compute_small_sample_correction(n_clusters)
        
        V = correction * (XtX_inv @ meat @ XtX_inv)
        V = 0.5 * (V + V.T)
        
        self._validate_covariance_matrix(V)
        
        return V
    
    def _regularize_meat_matrix(self, meat: np.ndarray) -> np.ndarray:
        """Regularize ill-conditioned meat matrix."""
        eigvals, eigvecs = np.linalg.eigh(meat)
        eigval_floor = eigvals.max() * 1e-10
        eigvals_reg = np.maximum(eigvals, eigval_floor)
        return eigvecs @ np.diag(eigvals_reg) @ eigvecs.T
    
    def _compute_small_sample_correction(self, n_clusters: int) -> float:
        """Compute small sample correction factor."""
        N = n_clusters
        NT = self.n_obs
        K = self.n_features
        
        if self.se_type == 'stata':
            correction = (N / (N - 1)) * ((NT - 1) / (NT - K))
        elif self.se_type == 'HC0':
            correction = 1.0
        elif self.se_type == 'HC1':
            correction = NT / (NT - K)
        else:
            correction = (N / (N - 1)) * ((NT - 1) / (NT - K))
        
        return correction
    
    def _validate_covariance_matrix(self, V: np.ndarray) -> None:
        """Validate covariance matrix."""
        diag_V = np.diag(V)
        if np.any(diag_V < 0):
            n_negative = (diag_V < 0).sum()
            logger.error(f"Negative variances detected: {n_negative}/{len(diag_V)}")
    
    def get_standard_errors(self, cluster_type: str = 'classical') -> np.ndarray:
        """Get standard errors."""
        if cluster_type != 'classical' and self.se_type in ['HC2', 'HC3']:
            raise ValueError(f"se_type='{self.se_type}' incompatible with cluster_type='{cluster_type}'.")
            
        self.rss = float(self.sum_y_squared - self.theta @ self.Xty)
        
        if cluster_type == 'classical':
            cov_matrix = self.get_covariance_matrix()
        else:
            cov_matrix = self.get_cluster_robust_covariance(cluster_type)
        
        diag_values = np.diag(cov_matrix)
        min_se_value = np.sqrt(np.finfo(float).eps) * max(np.max(diag_values), self.alpha)
        
        return np.sqrt(np.maximum(diag_values, min_se_value))
    
    def diagnose_cluster_structure(self, cluster_stats: Dict, cluster_name: str = "Cluster") -> Dict[str, Any]:
        """Diagnose cluster structure."""
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
        
        if n_clusters < 10:
            diagnostics["warnings"].append(
                f"Few clusters ({n_clusters}). Cluster-robust SEs may be unreliable."
            )
        
        if diagnostics["min_size"] < MIN_CLUSTER_SIZE:
            diagnostics["warnings"].append(
                f"Small clusters detected (min={diagnostics['min_size']})."
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
        """Log cluster diagnostics report."""
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
                f"  Total Observations: {diag['total_obs']:,}"
            ])
            
            if diag['warnings']:
                report_lines.append("")
                report_lines.append("WARNINGS:")
                for warning in diag['warnings']:
                    report_lines.append(f"  ⚠ {warning}")
        
        elif cluster_type == 'two_way':
            diag1 = self.diagnose_cluster_structure(self.cluster_stats)
            diag2 = self.diagnose_cluster_structure(self.cluster2_stats)
            diag_int = self.diagnose_cluster_structure(self.intersection_stats)
            
            report_lines.extend([
                f"Dimension 1: {diag1['n_clusters']} clusters",
                f"  Size Range: {diag1['min_size']:,} - {diag1['max_size']:,}",
                "",
                f"Dimension 2: {diag2['n_clusters']} clusters",
                f"  Size Range: {diag2['min_size']:,} - {diag2['max_size']:,}",
                "",
                f"Intersection: {diag_int['n_clusters']} unique clusters"
            ])
        
        report_lines.append("=" * 80)
        logger.info("\n".join(report_lines))

    def get_total_sum_squares(self) -> float:
        """Compute total sum of squares."""
        if self.n_obs <= 1:
            return 0.0
        mean_y = self.sum_y / self.n_obs
        return self.sum_y_squared - self.n_obs * (mean_y ** 2)
        
    def get_r_squared(self) -> float:
        """Calculate R-squared."""
        self.rss = self.sum_y_squared - self.theta @ self.Xty
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
        """Calculate F-statistic for overall model significance."""
        if self.n_obs <= self.n_features:
            return 0.0, 0, 0
        
        self.rss = self.sum_y_squared - self.theta @ self.Xty
        
        has_intercept = (self.feature_names[0].lower() == 'intercept' if self.feature_names else False)
        
        k = self.n_features
        n = self.n_obs
        df_model = k - 1 if has_intercept else k
        df_resid = n - k
        
        if df_model <= 0 or df_resid <= 0:
            return 0.0, df_model, df_resid
        
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
        """Get feature names."""
        return self.feature_names.copy()


def _transform_partition_with_transformer(partition_df, feature_cols, feature_names, target_col, 
                                         cluster1_col, cluster2_col, transformer):
    """Module-level function for partition transformation with feature transformer."""
    X = partition_df[feature_cols].values
    X_transformed = transformer.transform(X, feature_cols)
    # Create new DataFrame with transformed features
    transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=partition_df.index
    )
    # Add target and cluster columns
    transformed_df[target_col] = partition_df[target_col].values
    if cluster1_col:
        transformed_df[cluster1_col] = partition_df[cluster1_col].values
    if cluster2_col:
        transformed_df[cluster2_col] = partition_df[cluster2_col].values
    return transformed_df


def _compute_sufficient_stats_chunk(partition_df, feature_names, target_col, n_features, alpha):
    """Module-level function for computing sufficient statistics on a chunk."""
    X = partition_df[feature_names].values.astype(float, copy=False)
    y = partition_df[target_col].values.astype(float, copy=False)
    
    valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[valid_mask]
    y = y[valid_mask]
    
    if len(X) == 0:
        return {
            'XtX': alpha * np.eye(n_features),
            'Xty': np.zeros(n_features),
            'n_obs': 0,
            'sum_y': 0.0,
            'sum_y_squared': 0.0
        }
    
    return {
        'XtX': X.T @ X,
        'Xty': X.T @ y,
        'n_obs': len(X),
        'sum_y': y.sum(),
        'sum_y_squared': (y ** 2).sum()
    }


def _combine_sufficient_stats(stats_list, n_features, alpha):
    """Module-level function for combining sufficient statistics."""
    if stats_list.empty:
        return {
            'XtX': alpha * np.eye(n_features),
            'Xty': np.zeros(n_features),
            'n_obs': 0,
            'sum_y': 0.0,
            'sum_y_squared': 0.0
        }
    
    return {
        'XtX': sum(s['XtX'] for s in stats_list),
        'Xty': sum(s['Xty'] for s in stats_list),
        'n_obs': sum(s['n_obs'] for s in stats_list),
        'sum_y': sum(s['sum_y'] for s in stats_list),
        'sum_y_squared': sum(s['sum_y_squared'] for s in stats_list)
    }


def _compute_cluster_stats_chunk(partition_df, feature_names, target_col, cluster1_col, 
                                cluster2_col, theta, n_features):
    """Module-level function for computing cluster statistics on a chunk."""
    # Extract only needed columns to reduce memory
    cols_needed = feature_names + [target_col]
    if cluster1_col:
        cols_needed.append(cluster1_col)
    if cluster2_col:
        cols_needed.append(cluster2_col)
    
    partition_df = partition_df[cols_needed]
    
    X = partition_df[feature_names].values.astype(np.float32, copy=False)
    y = partition_df[target_col].values.astype(np.float32, copy=False)
    
    valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    
    if not np.any(valid_mask):
        return {'cluster1': {}, 'cluster2': {}, 'intersection': {}}
    
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Compute residuals once
    errors = y - X @ theta
    
    cluster1_stats = {}
    cluster2_stats = {}
    intersection_stats = {}
    
    # Process cluster1
    if cluster1_col:
        cluster1 = partition_df[cluster1_col].values[valid_mask]
        _fast_update_stats(cluster1_stats, cluster1, X, y, errors, n_features)
    
    # Process cluster2
    if cluster2_col:
        cluster2 = partition_df[cluster2_col].values[valid_mask]
        _fast_update_stats(cluster2_stats, cluster2, X, y, errors, n_features)
    
    # Process intersection - create 2D array
    if cluster1_col and cluster2_col:
        if 'cluster1' not in locals():
            cluster1 = partition_df[cluster1_col].values[valid_mask]
        if 'cluster2' not in locals():
            cluster2 = partition_df[cluster2_col].values[valid_mask]
        
        # Stack into 2D array (n_samples, 2)
        intersection_ids = np.column_stack([cluster1, cluster2])
        _fast_update_stats(intersection_stats, intersection_ids, X, y, errors, n_features)
    
    return {
        'cluster1': cluster1_stats,
        'cluster2': cluster2_stats,
        'intersection': intersection_stats
    }


def _combine_cluster_stats(stats_list):
    """Module-level function for combining cluster statistics."""
    if stats_list.empty:
        return {'cluster1': {}, 'cluster2': {}, 'intersection': {}}
    
    cluster1_combined = {}
    cluster2_combined = {}
    intersection_combined = {}
    
    # Direct merge without intermediate aggregator object
    for stats in stats_list:
        _fast_merge_stats(stats['cluster1'], cluster1_combined)
        _fast_merge_stats(stats['cluster2'], cluster2_combined)
        _fast_merge_stats(stats['intersection'], intersection_combined)
    
    return {
        'cluster1': cluster1_combined,
        'cluster2': cluster2_combined,
        'intersection': intersection_combined
    }


class DaskOLSEstimator:
    """
    Efficient OLS estimation using Dask DataFrames for out-of-memory computation.
    
    Computes sufficient statistics (X'X, X'y) using Dask aggregations,
    then uses OnlineRLS for parameter estimation and standard errors.
    """
    
    def __init__(self, 
                 dask_df: dd.DataFrame,
                 feature_cols: List[str],
                 target_col: str,
                 cluster1_col: Optional[str] = None,
                 cluster2_col: Optional[str] = None,
                 add_intercept: bool = True,
                 alpha: float = DEFAULT_ALPHA,
                 se_type: Literal['stata', 'HC0', 'HC1'] = 'stata',
                 feature_transformer=None,
                 client=None,
                 n_workers: Optional[int] = None):
        """
        Initialize Dask OLS estimator.
        
        Parameters:
        -----------
        dask_df : dd.DataFrame
            Dask DataFrame (lazy) with features and target
        feature_cols : list of str
            Column names for features
        target_col : str
            Column name for target variable
        cluster1_col : str, optional
            First clustering variable
        cluster2_col : str, optional
            Second clustering variable (for two-way clustering)
        add_intercept : bool
            Whether to add intercept
        alpha : float
            Regularization parameter
        se_type : str
            Standard error type ('stata', 'HC0', 'HC1')
        feature_transformer : FeatureTransformer, optional
            Transformer for feature engineering
        client : dask.distributed.Client, optional
            Dask client for distributed computation. If None, creates LocalCluster
        n_workers : int, optional
            Number of workers for LocalCluster (only used if client is None)
        """
        self.dask_df = dask_df
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.cluster1_col = cluster1_col
        self.cluster2_col = cluster2_col
        self.add_intercept = add_intercept
        self.alpha = alpha
        self.se_type = se_type
        self.feature_transformer = feature_transformer
        
        # Setup Dask client
        self._client = client
        self._client_created = False
        self._setup_client(n_workers)
        
        # Determine feature names after transformation
        self.feature_names = self._get_feature_names()
        self.n_features = len(self.feature_names)
    
    def _setup_client(self, n_workers: Optional[int]):
        """Setup Dask client, creating LocalCluster if needed."""
        if self._client is None:
            from dask.distributed import Client, LocalCluster
            import os
            
            # Determine number of workers
            if n_workers is None:
                n_workers = max(1, os.cpu_count() - 1)
            
            logger.info(f"Creating LocalCluster with {n_workers} workers")
            
            cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=1,
                processes=True,
                silence_logs=logging.WARNING  # Changed from ERROR to WARNING to reduce clutter
            )
            self._client = Client(cluster)
            self._client_created = True
            
            # Reduce Dask cluster logging clutter
            logging.getLogger('distributed').setLevel(logging.WARNING)
            logging.getLogger('distributed.scheduler').setLevel(logging.WARNING)
            logging.getLogger('distributed.worker').setLevel(logging.WARNING)
            logging.getLogger('distributed.nanny').setLevel(logging.WARNING)
            
            logger.info(f"Dashboard: {self._client.dashboard_link}")
        
        # Calculate optimal split_every based on worker count
        n_workers_actual = len(self._client.scheduler_info()['workers'])
        # Use 2x workers as split_every for good tree reduction parallelism
        self._split_every = max(4, min(n_workers_actual * 2, 32))
        
        logger.debug(f"Using split_every={self._split_every} for {n_workers_actual} workers")
    
    def _cleanup_client(self):
        """Cleanup client if we created it."""
        if self._client_created and self._client is not None:
            logger.debug("Closing LocalCluster")
            self._client.close()
            self._client = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self._cleanup_client()
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names after transformations."""
        names = self.feature_cols.copy()
        
        if self.feature_transformer:
            # Let transformer handle naming
            names = self.feature_transformer.get_feature_names()
        elif self.add_intercept:
            names = ['intercept'] + names
        
        return names
    
    def fit(self, verbose: bool = True) -> OnlineRLS:
        """
        Fit OLS model using Dask aggregations.
        
        Returns:
        --------
        OnlineRLS model with estimated parameters and cluster statistics
        """
        start_time = time.time()
        
        if verbose:
            logger.info(f"Starting Dask OLS estimation: {self.n_features} features, se_type={self.se_type}")
        
        # Prepare data: select columns and transform
        df_work = self._prepare_dask_df()
        
        # Compute sufficient statistics using Dask aggregations
        stats = self._compute_sufficient_statistics(df_work, verbose)
        
        # Create OnlineRLS model and populate with statistics
        model = OnlineRLS(
            n_features=self.n_features,
            alpha=self.alpha,
            feature_names=self.feature_names,
            se_type=self.se_type
        )
        
        # Set sufficient statistics
        model.XtX = stats['XtX']
        model.Xty = stats['Xty']
        model.n_obs = stats['n_obs']
        model.sum_y = stats['sum_y']
        model.sum_y_squared = stats['sum_y_squared']
        
        # Compute parameters
        model.theta = LinAlgHelper.safe_solve(model.XtX, model.Xty, self.alpha)
        model.P = LinAlgHelper.safe_inv(model.XtX, use_pinv=True)
        
        # Compute cluster statistics if needed
        if self.cluster1_col or self.cluster2_col:
            cluster_stats = self._compute_cluster_statistics(df_work, model.theta, verbose)
            model.cluster_stats = cluster_stats['cluster1']
            model.cluster2_stats = cluster_stats['cluster2']
            model.intersection_stats = cluster_stats['intersection']
        
        elapsed = time.time() - start_time
        
        if verbose:
            logger.info(
                f"Completed in {elapsed:.1f}s: {model.n_obs:,} obs, "
                f"R²={model.get_r_squared():.4f}"
            )
        
        return model
    
    def _prepare_dask_df(self) -> dd.DataFrame:
        """Prepare Dask DataFrame with transformations."""
        # Select required columns
        required_cols = self.feature_cols + [self.target_col]
        if self.cluster1_col:
            required_cols.append(self.cluster1_col)
        if self.cluster2_col:
            required_cols.append(self.cluster2_col)
        
        df = self.dask_df[required_cols]
        
        # Drop rows with NaN/inf in features or target
        df = df[df[self.feature_cols + [self.target_col]].notnull().all(axis=1)]
        
        # Apply feature transformation
        if self.feature_transformer:
            # Use module-level function for deterministic hashing
            df = df.map_partitions(
                _transform_partition_with_transformer,
                self.feature_cols,
                self.feature_names,
                self.target_col,
                self.cluster1_col,
                self.cluster2_col,
                self.feature_transformer,
                meta=pd.DataFrame(columns=self.feature_names + 
                                         [self.target_col] + 
                                         ([self.cluster1_col] if self.cluster1_col else []) +
                                         ([self.cluster2_col] if self.cluster2_col else []))
            )
        
        elif self.add_intercept:
            # Add intercept column
            df = df.assign(intercept=1.0)
            # Reorder columns to put intercept first
            feature_cols_with_intercept = ['intercept'] + self.feature_cols
            df = df[feature_cols_with_intercept + [self.target_col] + 
                   ([self.cluster1_col] if self.cluster1_col else []) +
                   ([self.cluster2_col] if self.cluster2_col else [])]
        
        return df
    
    def _compute_sufficient_statistics(self, df: dd.DataFrame, verbose: bool) -> Dict:
        """Compute X'X, X'y using Dask reduction for optimal performance."""
        if verbose:
            logger.info("Computing sufficient statistics...")
        
        # Use reduction for tree-based aggregation with optimized split_every
        reduction_result = df.reduction(
            chunk=_compute_sufficient_stats_chunk,
            aggregate=_combine_sufficient_stats,
            combine=_combine_sufficient_stats,
            chunk_kwargs={
                'feature_names': self.feature_names,
                'target_col': self.target_col,
                'n_features': self.n_features,
                'alpha': self.alpha
            },
            aggregate_kwargs={
                'n_features': self.n_features,
                'alpha': self.alpha
            },
            combine_kwargs={
                'n_features': self.n_features,
                'alpha': self.alpha
            },
            meta=object,
            split_every=self._split_every
        )
        
        result = reduction_result.persist()
        if verbose:
            progress(result)
        
        return result.compute()

    def _compute_cluster_statistics(self, df: dd.DataFrame, theta: np.ndarray, verbose: bool) -> Dict:
        """Compute cluster-robust statistics using Dask reduction for optimal performance."""
        if verbose:
            logger.info("Computing cluster statistics...")
        
        # Use reduction for tree-based aggregation with optimized split_every
        reduction_result = df.reduction(
            chunk=_compute_cluster_stats_chunk,
            aggregate=_combine_cluster_stats,
            combine=_combine_cluster_stats,
            chunk_kwargs={
                'feature_names': self.feature_names,
                'target_col': self.target_col,
                'cluster1_col': self.cluster1_col,
                'cluster2_col': self.cluster2_col,
                'theta': theta,
                'n_features': self.n_features
            },
            meta=object,
            split_every=self._split_every
        )
        
        result = reduction_result.persist()
        if verbose:
            progress(result)
        
        return result.compute()
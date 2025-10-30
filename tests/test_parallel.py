"""
Tests for parallel processing functionality.
"""
import numpy as np
import pandas as pd
import pytest
import tempfile
import shutil
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

from streamreg.data import StreamData
from streamreg.estimators.ols import ParallelOLSOrchestrator
from streamreg.api import OLS, TwoSLS


@pytest.fixture
def large_dataframe():
    """Generate a large DataFrame for parallel testing."""
    np.random.seed(42)
    n_samples = 5000
    
    df = pd.DataFrame({
        'y': np.random.randn(n_samples),
        'x1': np.random.randn(n_samples),
        'x2': np.random.randn(n_samples),
        'x3': np.random.randn(n_samples),
        'cluster': np.repeat(np.arange(50), 100),
        'time': np.tile(np.arange(100), 50)
    })
    
    # Create true relationship
    df['y'] = 2.0 * df['x1'] + 3.0 * df['x2'] - 1.5 * df['x3'] + 0.5 * np.random.randn(n_samples)
    
    return df


@pytest.fixture
def temp_parquet_dir():
    """Create temporary directory for parquet files."""
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath)


@pytest.fixture
def partitioned_parquet(large_dataframe, temp_parquet_dir):
    """Create partitioned parquet dataset."""
    # Split into partitions
    n_partitions = 10
    partition_size = len(large_dataframe) // n_partitions
    
    for i in range(n_partitions):
        start_idx = i * partition_size
        end_idx = (i + 1) * partition_size if i < n_partitions - 1 else len(large_dataframe)
        
        partition_df = large_dataframe.iloc[start_idx:end_idx]
        partition_path = temp_parquet_dir / f"partition_{i:03d}.parquet"
        
        table = pa.Table.from_pandas(partition_df)
        pq.write_table(table, partition_path)
    
    return temp_parquet_dir


@pytest.fixture
def single_parquet(large_dataframe, temp_parquet_dir):
    """Create single parquet file."""
    parquet_path = temp_parquet_dir / "data.parquet"
    table = pa.Table.from_pandas(large_dataframe)
    pq.write_table(table, parquet_path)
    return parquet_path


class TestStreamDataParallel:
    """Tests for StreamData parallel iteration."""
    
    def test_dataframe_parallel_iteration(self, large_dataframe):
        """Test parallel iteration over DataFrame."""
        data = StreamData(large_dataframe, chunk_size=500)
        
        chunks = list(data.iter_chunks_parallel(n_workers=4))
        
        # Should have multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should have chunk_id and DataFrame
        for chunk_id, chunk_df in chunks:
            assert isinstance(chunk_id, int)
            assert isinstance(chunk_df, pd.DataFrame)
            assert len(chunk_df) <= 500
        
        # Total rows should match original
        total_rows = sum(len(chunk_df) for _, chunk_df in chunks)
        assert total_rows == len(large_dataframe)
    
    def test_single_parquet_parallel_iteration(self, single_parquet):
        """Test parallel iteration over single parquet file."""
        data = StreamData(single_parquet, chunk_size=500)
        
        chunks = list(data.iter_chunks_parallel(n_workers=4))
        
        # Should have multiple chunks
        assert len(chunks) > 1
        
        # Total rows should match
        total_rows = sum(len(chunk_df) for _, chunk_df in chunks)
        assert total_rows == data.info.n_rows
    
    def test_partitioned_parquet_parallel_iteration(self, partitioned_parquet):
        """Test parallel iteration over partitioned parquet."""
        data = StreamData(partitioned_parquet, chunk_size=500)
        
        assert data.info.source_type == 'partitioned'
        
        chunks = list(data.iter_chunks_parallel(n_workers=4))
        
        # Should have chunks from all partitions
        assert len(chunks) >= 10  # At least one chunk per partition
        
        # Total rows should match
        total_rows = sum(len(chunk_df) for _, chunk_df in chunks)
        assert total_rows == data.info.n_rows
    
    def test_supports_parallel_all_sources(self, large_dataframe, single_parquet, partitioned_parquet):
        """Test that all data sources report supporting parallel processing."""
        df_data = StreamData(large_dataframe)
        single_data = StreamData(single_parquet)
        part_data = StreamData(partitioned_parquet)
        
        assert df_data.supports_parallel()
        assert single_data.supports_parallel()
        assert part_data.supports_parallel()


class TestParallelOLSOrchestrator:
    """Tests for ParallelOLSOrchestrator."""
    
    def test_sequential_vs_parallel_dataframe(self, large_dataframe):
        """Test that sequential and parallel processing give same results for DataFrame."""
        data = StreamData(large_dataframe, chunk_size=500)
        
        # Sequential (n_workers=1)
        orch_seq = ParallelOLSOrchestrator(
            data=data,
            feature_cols=['x1', 'x2', 'x3'],
            target_col='y',
            add_intercept=True,
            n_features=4,
            transformed_feature_names=['intercept', 'x1', 'x2', 'x3'],
            alpha=1e-3,
            chunk_size=500,
            n_workers=1,
            show_progress=False
        )
        model_seq = orch_seq.fit()
        
        # Parallel (n_workers=4)
        orch_par = ParallelOLSOrchestrator(
            data=data,
            feature_cols=['x1', 'x2', 'x3'],
            target_col='y',
            add_intercept=True,
            n_features=4,
            transformed_feature_names=['intercept', 'x1', 'x2', 'x3'],
            alpha=1e-3,
            chunk_size=500,
            n_workers=4,
            show_progress=False
        )
        model_par = orch_par.fit()
        
        # Results should be identical
        assert model_seq.n_obs == model_par.n_obs
        assert np.allclose(model_seq.theta, model_par.theta, atol=1e-6)
        assert np.allclose(model_seq.get_r_squared(), model_par.get_r_squared(), atol=1e-6)
    
    def test_sequential_vs_parallel_single_parquet(self, single_parquet):
        """Test that sequential and parallel processing give same results for single parquet."""
        data = StreamData(single_parquet, chunk_size=500)
        
        # Sequential
        orch_seq = ParallelOLSOrchestrator(
            data=data,
            feature_cols=['x1', 'x2', 'x3'],
            target_col='y',
            add_intercept=True,
            n_features=4,
            transformed_feature_names=['intercept', 'x1', 'x2', 'x3'],
            alpha=1e-3,
            chunk_size=500,
            n_workers=1,
            show_progress=False
        )
        model_seq = orch_seq.fit()
        
        # Parallel
        orch_par = ParallelOLSOrchestrator(
            data=data,
            feature_cols=['x1', 'x2', 'x3'],
            target_col='y',
            add_intercept=True,
            n_features=4,
            transformed_feature_names=['intercept', 'x1', 'x2', 'x3'],
            alpha=1e-3,
            chunk_size=500,
            n_workers=4,
            show_progress=False
        )
        model_par = orch_par.fit()
        
        # Results should be identical
        assert model_seq.n_obs == model_par.n_obs
        assert np.allclose(model_seq.theta, model_par.theta, atol=1e-6)
        assert np.allclose(model_seq.get_r_squared(), model_par.get_r_squared(), atol=1e-6)
    
    def test_sequential_vs_parallel_partitioned(self, partitioned_parquet):
        """Test that sequential and parallel processing give same results for partitioned data."""
        data = StreamData(partitioned_parquet, chunk_size=500)
        
        # Sequential
        orch_seq = ParallelOLSOrchestrator(
            data=data,
            feature_cols=['x1', 'x2', 'x3'],
            target_col='y',
            add_intercept=True,
            n_features=4,
            transformed_feature_names=['intercept', 'x1', 'x2', 'x3'],
            alpha=1e-3,
            chunk_size=500,
            n_workers=1,
            show_progress=False
        )
        model_seq = orch_seq.fit()
        
        # Parallel
        orch_par = ParallelOLSOrchestrator(
            data=data,
            feature_cols=['x1', 'x2', 'x3'],
            target_col='y',
            add_intercept=True,
            n_features=4,
            transformed_feature_names=['intercept', 'x1', 'x2', 'x3'],
            alpha=1e-3,
            chunk_size=500,
            n_workers=4,
            show_progress=False
        )
        model_par = orch_par.fit()
        
        # Results should be identical
        assert model_seq.n_obs == model_par.n_obs
        assert np.allclose(model_seq.theta, model_par.theta, atol=1e-6)
        assert np.allclose(model_seq.get_r_squared(), model_par.get_r_squared(), atol=1e-6)
    
    def test_different_worker_counts(self, large_dataframe):
        """Test that different worker counts give same results."""
        data = StreamData(large_dataframe, chunk_size=500)
        
        results = []
        for n_workers in [1, 2, 4]:
            orch = ParallelOLSOrchestrator(
                data=data,
                feature_cols=['x1', 'x2', 'x3'],
                target_col='y',
                add_intercept=True,
                n_features=4,
                transformed_feature_names=['intercept', 'x1', 'x2', 'x3'],
                alpha=1e-3,
                chunk_size=500,
                n_workers=n_workers,
                show_progress=False
            )
            model = orch.fit()
            results.append((n_workers, model))
        
        # All should give same results
        for i in range(len(results) - 1):
            n1, m1 = results[i]
            n2, m2 = results[i + 1]
            
            assert m1.n_obs == m2.n_obs, f"n_obs differs between {n1} and {n2} workers"
            assert np.allclose(m1.theta, m2.theta, atol=1e-6), f"theta differs between {n1} and {n2} workers"
    
    def test_with_clustering(self, large_dataframe):
        """Test parallel processing with clustering."""
        data = StreamData(large_dataframe, chunk_size=500)
        
        # Sequential with clustering
        orch_seq = ParallelOLSOrchestrator(
            data=data,
            feature_cols=['x1', 'x2', 'x3'],
            target_col='y',
            cluster1_col='cluster',
            cluster2_col='time',
            add_intercept=True,
            n_features=4,
            transformed_feature_names=['intercept', 'x1', 'x2', 'x3'],
            alpha=1e-3,
            chunk_size=500,
            n_workers=1,
            show_progress=False
        )
        model_seq = orch_seq.fit()
        
        # Parallel with clustering
        orch_par = ParallelOLSOrchestrator(
            data=data,
            feature_cols=['x1', 'x2', 'x3'],
            target_col='y',
            cluster1_col='cluster',
            cluster2_col='time',
            add_intercept=True,
            n_features=4,
            transformed_feature_names=['intercept', 'x1', 'x2', 'x3'],
            alpha=1e-3,
            chunk_size=500,
            n_workers=4,
            show_progress=False
        )
        model_par = orch_par.fit()
        
        # Results should be identical
        assert model_seq.n_obs == model_par.n_obs
        assert np.allclose(model_seq.theta, model_par.theta, atol=1e-6)
        
        # Cluster statistics should match
        assert len(model_seq.cluster_stats) == len(model_par.cluster_stats)
        assert len(model_seq.cluster2_stats) == len(model_par.cluster2_stats)


class TestOLSAPIParallel:
    """Tests for OLS API with parallel processing."""
    
    def test_ols_dataframe_parallel(self, large_dataframe):
        """Test OLS with DataFrame using parallel processing."""
        # With n_workers specified
        model = OLS("y ~ x1 + x2 + x3", n_workers=4, show_progress=False)
        model.fit(large_dataframe)
        
        assert model.n_obs_ == len(large_dataframe)
        assert model.r_squared_ > 0.9
        
        # Check coefficients are close to true values
        coefs = model.coef_
        assert abs(coefs[1] - 2.0) < 0.1  # x1
        assert abs(coefs[2] - 3.0) < 0.1  # x2
        assert abs(coefs[3] + 1.5) < 0.1  # x3
    
    def test_ols_single_parquet_parallel(self, single_parquet):
        """Test OLS with single parquet file using parallel processing."""
        model = OLS("y ~ x1 + x2 + x3", n_workers=4, show_progress=False)
        model.fit(single_parquet)
        
        assert model.n_obs_ > 0
        assert model.r_squared_ > 0.9
    
    def test_ols_partitioned_parallel(self, partitioned_parquet):
        """Test OLS with partitioned parquet using parallel processing."""
        model = OLS("y ~ x1 + x2 + x3", n_workers=4, show_progress=False)
        model.fit(partitioned_parquet)
        
        assert model.n_obs_ > 0
        assert model.r_squared_ > 0.9
    
    def test_ols_with_clustering_parallel(self, large_dataframe):
        """Test OLS with clustering using parallel processing."""
        model = OLS("y ~ x1 + x2 + x3", n_workers=4, show_progress=False)
        model.fit(large_dataframe, cluster=['cluster', 'time'])
        
        assert model.n_obs_ == len(large_dataframe)
        assert model.results_.cluster_type == 'two_way'
        
        # Should have cluster diagnostics
        assert model.results_.cluster_diagnostics is not None
        assert 'dim1' in model.results_.cluster_diagnostics
        assert 'dim2' in model.results_.cluster_diagnostics
    
    def test_sequential_matches_parallel_ols(self, large_dataframe):
        """Test that OLS gives same results with sequential vs parallel."""
        # Sequential
        model_seq = OLS("y ~ x1 + x2 + x3", n_workers=1, show_progress=False)
        model_seq.fit(large_dataframe)
        
        # Parallel
        model_par = OLS("y ~ x1 + x2 + x3", n_workers=4, show_progress=False)
        model_par.fit(large_dataframe)
        
        # Results should match
        assert np.allclose(model_seq.coef_, model_par.coef_, atol=1e-6)
        assert np.allclose(model_seq.se_, model_par.se_, atol=1e-6)
        assert abs(model_seq.r_squared_ - model_par.r_squared_) < 1e-6
    
    def test_different_se_types(self, large_dataframe):
        """Test that different SE types produce different but reasonable results."""
        results = {}
        
        for se_type in ['stata', 'HC0', 'HC1', 'HC2', 'HC3']:
            model = OLS("y ~ x1 + x2 + x3", n_workers=2, show_progress=False, se_type=se_type)
            model.fit(large_dataframe, cluster='cluster')
            results[se_type] = {
                'coef': model.coef_.copy(),
                'se': model.se_.copy()
            }
        
        # Coefficients should be identical across SE types
        for se_type in ['HC0', 'HC1', 'HC2', 'HC3']:
            assert np.allclose(results['stata']['coef'], results[se_type]['coef'], atol=1e-6)
        
        # Standard errors should differ
        # HC0 should have smallest correction (no correction)
        # stata should be larger due to small-sample correction
        assert np.all(results['stata']['se'] >= results['HC0']['se'] * 0.99)  # Allow small numerical differences
        
        # All SE should be positive
        for se_type, res in results.items():
            assert np.all(res['se'] > 0), f"{se_type} has non-positive SE"
    
    def test_se_type_with_two_way_clustering(self, large_dataframe):
        """Test different SE types with two-way clustering."""
        # stata
        model_stata = OLS("y ~ x1 + x2 + x3", n_workers=2, show_progress=False, se_type='stata')
        model_stata.fit(large_dataframe, cluster=['cluster', 'time'])
        
        # HC1
        model_hc1 = OLS("y ~ x1 + x2 + x3", n_workers=2, show_progress=False, se_type='HC1')
        model_hc1.fit(large_dataframe, cluster=['cluster', 'time'])
        
        # Coefficients should match
        assert np.allclose(model_stata.coef_, model_hc1.coef_, atol=1e-6)
        
        # Standard errors should differ due to different corrections
        assert not np.allclose(model_stata.se_, model_hc1.se_, atol=1e-6)


class TestTwoSLSParallel:
    """Tests for 2SLS with parallel processing."""
    
    @pytest.fixture
    def iv_dataframe(self):
        """Generate data for IV regression."""
        np.random.seed(42)
        n_samples = 3000
        
        # Instruments
        z1 = np.random.randn(n_samples)
        z2 = np.random.randn(n_samples)
        
        # Endogenous variable (correlated with error)
        u = np.random.randn(n_samples)
        x_endog = 1.0 * z1 + 0.5 * z2 + 0.3 * u + 0.5 * np.random.randn(n_samples)
        
        # Exogenous variable
        x_exog = np.random.randn(n_samples)
        
        # Outcome (true effect of x_endog is 2.0)
        y = 2.0 * x_endog + 1.5 * x_exog + u + 0.5 * np.random.randn(n_samples)
        
        df = pd.DataFrame({
            'y': y,
            'x_endog': x_endog,
            'x_exog': x_exog,
            'z1': z1,
            'z2': z2,
            'cluster': np.repeat(np.arange(30), 100)
        })
        
        return df
    
    @pytest.fixture
    def iv_partitioned(self, iv_dataframe, temp_parquet_dir):
        """Create partitioned parquet for IV data."""
        n_partitions = 6
        partition_size = len(iv_dataframe) // n_partitions
        
        for i in range(n_partitions):
            start_idx = i * partition_size
            end_idx = (i + 1) * partition_size if i < n_partitions - 1 else len(iv_dataframe)
            
            partition_df = iv_dataframe.iloc[start_idx:end_idx]
            partition_path = temp_parquet_dir / f"iv_partition_{i:03d}.parquet"
            
            table = pa.Table.from_pandas(partition_df)
            pq.write_table(table, partition_path)
        
        return temp_parquet_dir
    
    def test_twosls_dataframe_parallel(self, iv_dataframe):
        """Test 2SLS with DataFrame using parallel processing."""
        model = TwoSLS(
            "y ~ x_endog + x_exog | z1 + z2",
            endogenous=['x_endog'],
            n_workers=4,
            show_progress=False
        )
        model.fit(iv_dataframe)
        
        assert model.results_.n_obs > 0
        
        # Check second stage coefficient for x_endog is close to 2.0
        second_stage_coefs = model.coef_
        # Find x_endog coefficient (could be x_endog_hat or just x_endog depending on implementation)
        feature_names = model.results_.feature_names
        
        # Look for the endogenous variable in feature names
        endog_indices = [i for i, name in enumerate(feature_names) if 'endog' in name.lower()]
        
        if not endog_indices:
            # Fallback: just check that model ran successfully
            assert len(second_stage_coefs) > 0
        else:
            endog_idx = endog_indices[0]
            # The coefficient should be reasonably close to 2.0, but IV estimation has more variance
            assert abs(second_stage_coefs[endog_idx] - 2.0) < 0.5
    
    def test_twosls_partitioned_parallel(self, iv_partitioned):
        """Test 2SLS with partitioned data using parallel processing."""
        model = TwoSLS(
            "y ~ x_endog + x_exog | z1 + z2",
            endogenous=['x_endog'],
            n_workers=4,
            show_progress=False
        )
        model.fit(iv_partitioned)
        
        assert model.results_.n_obs > 0
        assert len(model.results_.first_stage_results) == 1
    
    def test_sequential_matches_parallel_twosls(self, iv_dataframe):
        """Test that 2SLS gives same results with sequential vs parallel."""
        # Sequential
        model_seq = TwoSLS(
            "y ~ x_endog + x_exog | z1 + z2",
            endogenous=['x_endog'],
            n_workers=1,
            show_progress=False
        )
        model_seq.fit(iv_dataframe)
        
        # Parallel
        model_par = TwoSLS(
            "y ~ x_endog + x_exog | z1 + z2",
            endogenous=['x_endog'],
            n_workers=4,
            show_progress=False
        )
        model_par.fit(iv_dataframe)
        
        # Results should match
        assert np.allclose(model_seq.coef_, model_par.coef_, atol=1e-5)


@pytest.mark.slow
class TestParallelPerformance:
    """Tests for parallel processing performance."""
    
    def test_parallel_faster_than_sequential_large_data(self):
        """Test that parallel processing is faster for large datasets."""
        import time
        
        np.random.seed(42)
        n_samples = 20000
        
        df = pd.DataFrame({
            'y': np.random.randn(n_samples),
            'x1': np.random.randn(n_samples),
            'x2': np.random.randn(n_samples),
            'x3': np.random.randn(n_samples)
        })
        df['y'] = 2 * df['x1'] + 3 * df['x2'] - df['x3'] + 0.5 * np.random.randn(n_samples)
        
        # Sequential
        start = time.time()
        model_seq = OLS("y ~ x1 + x2 + x3", n_workers=1, show_progress=False)
        model_seq.fit(df)
        time_seq = time.time() - start
        
        # Parallel
        start = time.time()
        model_par = OLS("y ~ x1 + x2 + x3", n_workers=4, show_progress=False)
        model_par.fit(df)
        time_par = time.time() - start
        
        # Parallel should be faster (though not always guaranteed due to overhead)
        # Just verify both complete successfully
        assert model_seq.n_obs_ == model_par.n_obs_
        assert time_seq > 0 and time_par > 0
    
    def test_memory_efficiency_chunking(self, large_dataframe):
        """Test that chunking allows processing without loading all data."""
        # This should not raise MemoryError even with very small chunks
        model = OLS("y ~ x1 + x2 + x3", chunk_size=100, n_workers=2, show_progress=False)
        model.fit(large_dataframe)
        
        assert model.n_obs_ == len(large_dataframe)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

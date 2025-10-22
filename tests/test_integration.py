"""
Integration tests for complete workflows.
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile
import shutil

from streamreg.api import OLS, TwoSLS
from streamreg.data import StreamData


class TestOLSIntegration:
    """Integration tests for OLS workflow."""
    
    def test_dataframe_to_results(self):
        """Test complete workflow from DataFrame to results."""
        np.random.seed(42)
        df = pd.DataFrame({
            'y': np.random.randn(100),
            'x1': np.random.randn(100),
            'x2': np.random.randn(100)
        })
        df['y'] = 2 * df['x1'] + 3 * df['x2'] + np.random.randn(100)
        
        # Fit model
        model = OLS("y ~ x1 + x2")
        model.fit(df)
        
        # Check results
        assert model.n_obs_ == 100
        assert model.r_squared_ > 0.5
        
        # Get summary
        summary = model.summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 3  # intercept + 2 features
        
        # Make predictions
        X_new = df[['x1', 'x2']].head(10)
        y_pred = model.predict(X_new)
        assert len(y_pred) == 10
    
    def test_formula_with_transformations(self):
        """Test complete workflow with formula transformations."""
        np.random.seed(42)
        df = pd.DataFrame({
            'y': np.random.randn(200),
            'x1': np.random.randn(200),
            'x2': np.random.randn(200)
        })
        df['y'] = 1 + 2*df['x1'] + 3*df['x2'] + 0.5*df['x1']**2 + df['x1']*df['x2']
        
        # Fit with transformations
        model = OLS("y ~ x1 + x2 + I(x1^2) + x1:x2")
        model.fit(df)
        
        assert model.n_obs_ == 200
        # intercept, x1, x2, x1^2, x1:x2
        assert len(model.coef_) == 5
        assert model.r_squared_ > 0.7
    
    def test_clustering_workflow(self):
        """Test complete workflow with clustering."""
        np.random.seed(42)
        n_samples = 200
        n_clusters = 20
        
        df = pd.DataFrame({
            'y': np.random.randn(n_samples),
            'x1': np.random.randn(n_samples),
            'x2': np.random.randn(n_samples),
            'cluster': np.repeat(np.arange(n_clusters), n_samples // n_clusters)
        })
        
        # Add cluster effects
        cluster_effects = np.random.randn(n_clusters) * 0.5
        df['cluster_effect'] = cluster_effects[df['cluster'].values]
        df['y'] = 2*df['x1'] + 3*df['x2'] + df['cluster_effect'] + 0.1*np.random.randn(n_samples)
        
        # Fit with clustering
        model = OLS("y ~ x1 + x2")
        model.fit(df, cluster='cluster')
        
        assert model.n_obs_ == n_samples
        assert model.results_.cluster_type == 'one_way'
        assert model.results_.cluster_diagnostics is not None


class TestDataIntegration:
    """Integration tests for data loading."""
    
    def test_streamdata_from_dataframe(self):
        """Test StreamData with DataFrame."""
        df = pd.DataFrame({
            'y': np.random.randn(100),
            'x1': np.random.randn(100),
            'x2': np.random.randn(100)
        })
        
        data = StreamData(df, chunk_size=20)
        
        assert data.info.n_rows == 100
        assert data.info.n_cols == 3
        assert 'y' in data.info.columns
        
        # Test iteration
        chunk_count = 0
        total_rows = 0
        for chunk in data.iter_chunks():
            chunk_count += 1
            total_rows += len(chunk)
        
        assert total_rows == 100
        assert chunk_count == 5  # 100 / 20


@pytest.mark.slow
class TestLargeDataIntegration:
    """Integration tests with larger datasets."""
    
    def test_large_dataframe(self):
        """Test with larger in-memory dataset."""
        np.random.seed(42)
        n_samples = 10000
        
        df = pd.DataFrame({
            'y': np.random.randn(n_samples),
            'x1': np.random.randn(n_samples),
            'x2': np.random.randn(n_samples),
            'x3': np.random.randn(n_samples)
        })
        df['y'] = 2*df['x1'] + 3*df['x2'] - 1*df['x3'] + np.random.randn(n_samples)
        
        model = OLS("y ~ x1 + x2 + x3")
        model.fit(df)
        
        assert model.n_obs_ == n_samples
        assert model.r_squared_ > 0.7
    
    def test_chunked_processing(self):
        """Test that chunked processing gives same results as batch."""
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            'y': np.random.randn(n_samples),
            'x1': np.random.randn(n_samples),
            'x2': np.random.randn(n_samples)
        })
        df['y'] = 2*df['x1'] + 3*df['x2']
        
        # Fit with small chunks
        model = OLS("y ~ x1 + x2 - 1", chunk_size=100)
        model.fit(df)
        
        # Coefficients should be close to true values
        assert abs(model.coef_[0] - 2.0) < 0.1
        assert abs(model.coef_[1] - 3.0) < 0.1


class TestErrorHandling:
    """Test error handling in integration scenarios."""
    
    def test_missing_columns_error(self):
        """Test error when columns are missing."""
        df = pd.DataFrame({
            'y': np.random.randn(100),
            'x1': np.random.randn(100)
        })
        
        model = OLS("y ~ x1 + x2")  # x2 doesn't exist
        
        with pytest.raises(ValueError, match="Missing required columns"):
            model.fit(df)
    
    def test_invalid_cluster_specification(self):
        """Test error with invalid cluster specification."""
        df = pd.DataFrame({
            'y': np.random.randn(100),
            'x1': np.random.randn(100)
        })
        
        model = OLS("y ~ x1")
        
        with pytest.raises(ValueError):
            model.fit(df, cluster=['c1', 'c2', 'c3'])  # Too many clusters
    
    def test_formula_parsing_error(self):
        """Test error with invalid formula."""
        with pytest.raises(ValueError):
            model = OLS("invalid formula without tilde")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

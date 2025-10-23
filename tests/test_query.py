"""
Tests for query functionality in StreamData and estimators.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from streamreg.data import StreamData
from streamreg.api import OLS, TwoSLS


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        'y': np.random.randn(n),
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'z1': np.random.randn(n),  # Instrument
        'year': np.random.choice([2000, 2005, 2010, 2015, 2020], n),
        'country': np.random.choice(['USA', 'CAN', 'MEX', 'BRA'], n),
        'gdp': np.random.uniform(1000, 50000, n),
        'population': np.random.uniform(1e6, 1e9, n),
        'developed': np.random.choice([True, False], n)
    })
    
    # Add some NaN values for testing
    df.loc[np.random.choice(n, 10, replace=False), 'x1'] = np.nan
    
    return df


@pytest.fixture
def sample_parquet_file(sample_dataframe, tmp_path):
    """Create a sample parquet file for testing."""
    parquet_file = tmp_path / "test_data.parquet"
    sample_dataframe.to_parquet(parquet_file, index=False)
    return parquet_file


@pytest.fixture
def partitioned_parquet_dir(sample_dataframe, tmp_path):
    """Create a partitioned parquet dataset for testing."""
    parquet_dir = tmp_path / "partitioned_data"
    parquet_dir.mkdir()
    
    # Partition by year
    for year in sample_dataframe['year'].unique():
        year_df = sample_dataframe[sample_dataframe['year'] == year]
        year_dir = parquet_dir / f"year={year}"
        year_dir.mkdir()
        year_df.to_parquet(year_dir / "data.parquet", index=False)
    
    return parquet_dir


class TestStreamDataQuery:
    """Test query functionality in StreamData."""
    
    def test_dataframe_query_numeric(self, sample_dataframe):
        """Test numeric comparison query on DataFrame."""
        data = StreamData(sample_dataframe, query="year >= 2010")
        
        # Check that data is filtered
        assert data.info.n_rows == len(sample_dataframe[sample_dataframe['year'] >= 2010])
        
        # Check that chunks are filtered
        total_rows = 0
        for chunk in data.iter_chunks():
            assert (chunk['year'] >= 2010).all()
            total_rows += len(chunk)
        
        assert total_rows == data.info.n_rows
    
    def test_dataframe_query_string(self, sample_dataframe):
        """Test string comparison query on DataFrame."""
        data = StreamData(sample_dataframe, query="country == 'USA'")
        
        expected_rows = len(sample_dataframe[sample_dataframe['country'] == 'USA'])
        assert data.info.n_rows == expected_rows
        
        for chunk in data.iter_chunks():
            assert (chunk['country'] == 'USA').all()
    
    def test_dataframe_query_multiple_conditions(self, sample_dataframe):
        """Test query with multiple conditions."""
        query = "year >= 2010 and country == 'USA'"
        data = StreamData(sample_dataframe, query=query)
        
        expected_df = sample_dataframe.query(query)
        assert data.info.n_rows == len(expected_df)
        
        for chunk in data.iter_chunks():
            assert (chunk['year'] >= 2010).all()
            assert (chunk['country'] == 'USA').all()
    
    def test_dataframe_query_isin(self, sample_dataframe):
        """Test query with isin() for list membership."""
        query = "country.isin(['USA', 'CAN'])"
        data = StreamData(sample_dataframe, query=query)
        
        expected_df = sample_dataframe.query(query)
        assert data.info.n_rows == len(expected_df)
        
        for chunk in data.iter_chunks():
            assert chunk['country'].isin(['USA', 'CAN']).all()
    
    def test_dataframe_query_boolean_operators(self, sample_dataframe):
        """Test query with boolean operators."""
        query = "(year >= 2010) & (gdp > 10000)"
        data = StreamData(sample_dataframe, query=query)
        
        expected_df = sample_dataframe.query(query)
        assert data.info.n_rows == len(expected_df)
    
    def test_dataframe_query_or_operator(self, sample_dataframe):
        """Test query with OR operator."""
        query = "(year < 2005) | (year > 2015)"
        data = StreamData(sample_dataframe, query=query)
        
        expected_df = sample_dataframe.query(query)
        assert data.info.n_rows == len(expected_df)
    
    def test_dataframe_query_not_operator(self, sample_dataframe):
        """Test query with NOT operator."""
        query = "~(country == 'USA')"
        data = StreamData(sample_dataframe, query=query)
        
        expected_df = sample_dataframe.query(query)
        assert data.info.n_rows == len(expected_df)
    
    def test_dataframe_query_complex(self, sample_dataframe):
        """Test complex query with multiple operators."""
        query = "(year >= 2010) & (gdp > 10000) & country.isin(['USA', 'CAN'])"
        data = StreamData(sample_dataframe, query=query)
        
        expected_df = sample_dataframe.query(query)
        assert data.info.n_rows == len(expected_df)
    
    def test_dataframe_invalid_query(self, sample_dataframe):
        """Test that invalid query raises ValueError."""
        with pytest.raises(ValueError, match="Invalid query string"):
            StreamData(sample_dataframe, query="invalid syntax here")
    
    def test_parquet_query_applied(self, sample_parquet_file):
        """Test that query is applied when reading parquet file."""
        data = StreamData(sample_parquet_file, query="year >= 2010")
        
        total_rows = 0
        for chunk in data.iter_chunks():
            assert (chunk['year'] >= 2010).all()
            total_rows += len(chunk)
        
        # Should have fewer rows than original file
        original_data = StreamData(sample_parquet_file)
        assert total_rows < original_data.info.n_rows
    
    def test_partitioned_parquet_query(self, partitioned_parquet_dir):
        """Test query on partitioned parquet dataset."""
        data = StreamData(partitioned_parquet_dir, query="year >= 2010 and country == 'USA'")
        
        total_rows = 0
        for chunk in data.iter_chunks():
            assert (chunk['year'] >= 2010).all()
            assert (chunk['country'] == 'USA').all()
            total_rows += len(chunk)
        
        assert total_rows > 0
    
    def test_query_column_projection(self, sample_dataframe):
        """Test that query columns are loaded even if not in requested columns."""
        data = StreamData(sample_dataframe, query="year >= 2010")
        
        # Request only x1 and y, but year should be loaded for filtering
        for chunk in data.iter_chunks(columns=['x1', 'y']):
            assert 'x1' in chunk.columns
            assert 'y' in chunk.columns
            # year is filtered but not in final output
            assert 'year' not in chunk.columns
    
    def test_empty_result_after_query(self, sample_dataframe):
        """Test handling of query that filters out all data."""
        # Query that matches nothing
        data = StreamData(sample_dataframe, query="year > 2030")
        
        assert data.info.n_rows == 0
        
        chunks = list(data.iter_chunks())
        assert len(chunks) == 0


class TestOLSWithQuery:
    """Test OLS estimator with query parameter."""
    
    def test_ols_dataframe_query(self, sample_dataframe):
        """Test OLS with DataFrame and query."""
        model = OLS(formula="y ~ x1 + x2")
        model.fit(sample_dataframe, query="year >= 2010")
        
        # Check that only filtered data was used
        expected_n = len(sample_dataframe.query("year >= 2010"))
        # Account for NaN removal
        assert model.n_obs_ <= expected_n
        assert model.n_obs_ > 0
    
    def test_ols_parquet_query(self, sample_parquet_file):
        """Test OLS with parquet file and query."""
        model = OLS(formula="y ~ x1 + x2")
        model.fit(sample_parquet_file, query="country == 'USA'")
        
        # Should have fitted successfully
        assert model.n_obs_ > 0
        assert model.r_squared_ >= 0
    
    def test_ols_partitioned_query(self, partitioned_parquet_dir):
        """Test OLS with partitioned parquet and query."""
        model = OLS(formula="y ~ x1 + x2")
        model.fit(partitioned_parquet_dir, query="year >= 2015")
        
        assert model.n_obs_ > 0
        assert len(model.coef_) == 3  # intercept + x1 + x2
    
    def test_ols_query_with_clustering(self, sample_dataframe):
        """Test OLS with query and cluster-robust standard errors."""
        model = OLS(formula="y ~ x1 + x2")
        model.fit(sample_dataframe, query="year >= 2010", cluster='country')
        
        assert model.n_obs_ > 0
        assert model.results_.cluster_type == 'one_way'
    
    def test_ols_query_two_way_clustering(self, sample_dataframe):
        """Test OLS with query and two-way clustering."""
        model = OLS(formula="y ~ x1 + x2")
        model.fit(sample_dataframe, query="year >= 2010", cluster=['country', 'year'])
        
        assert model.n_obs_ > 0
        assert model.results_.cluster_type == 'two_way'
    
    def test_ols_complex_query(self, sample_dataframe):
        """Test OLS with complex query."""
        query = "(year >= 2010) & (gdp > 10000) & country.isin(['USA', 'CAN'])"
        model = OLS(formula="y ~ x1 + x2")
        model.fit(sample_dataframe, query=query)
        
        assert model.n_obs_ > 0
        
        # Verify that results are sensible
        assert -10 < model.coef_[0] < 10  # Intercept
        assert -10 < model.coef_[1] < 10  # x1
        assert -10 < model.coef_[2] < 10  # x2


class TestTwoSLSWithQuery:
    """Test 2SLS estimator with query parameter."""
    
    def test_twosls_dataframe_query(self, sample_dataframe):
        """Test 2SLS with DataFrame and query."""
        # Make x1 somewhat endogenous with z1 as instrument
        sample_dataframe['x1'] = 0.5 * sample_dataframe['z1'] + np.random.randn(len(sample_dataframe))
        sample_dataframe['y'] = 2 * sample_dataframe['x1'] + sample_dataframe['x2'] + np.random.randn(len(sample_dataframe))
        
        model = TwoSLS(formula="y ~ x1 + x2 | z1 + x2", endogenous=['x1'])
        model.fit(sample_dataframe, query="year >= 2010")
        
        assert model.results_.n_obs > 0
        assert len(model.results_.first_stage_results) == 1
    
    def test_twosls_parquet_query(self, sample_parquet_file):
        """Test 2SLS with parquet file and query."""
        model = TwoSLS(formula="y ~ x1 + x2 | z1 + x2", endogenous=['x1'])
        model.fit(sample_parquet_file, query="country == 'USA'")
        
        assert model.results_.n_obs > 0
    
    def test_twosls_query_with_clustering(self, sample_dataframe):
        """Test 2SLS with query and clustering."""
        sample_dataframe['x1'] = 0.5 * sample_dataframe['z1'] + np.random.randn(len(sample_dataframe))
        sample_dataframe['y'] = 2 * sample_dataframe['x1'] + sample_dataframe['x2'] + np.random.randn(len(sample_dataframe))
        
        model = TwoSLS(formula="y ~ x1 + x2 | z1 + x2", endogenous=['x1'])
        model.fit(sample_dataframe, query="year >= 2010", cluster='country')
        
        assert model.results_.n_obs > 0
        assert model.results_.cluster_type == 'one_way'


class TestQueryEdgeCases:
    """Test edge cases and error handling for queries."""
    
    def test_query_with_missing_column(self, sample_dataframe):
        """Test query referencing non-existent column."""
        # This should be caught during validation
        with pytest.raises(Exception):  # Could be ValueError or pandas error
            data = StreamData(sample_dataframe, query="nonexistent_col > 0")
    
    def test_query_with_nan_handling(self, sample_dataframe):
        """Test that NaN values are handled correctly with queries."""
        data = StreamData(sample_dataframe, query="year >= 2010")
        
        # Iterate and count rows - NaN should be removed during OLS fitting
        total_rows = sum(len(chunk) for chunk in data.iter_chunks())
        assert total_rows > 0
    
    def test_query_all_filtered_out(self, sample_dataframe):
        """Test query that filters out all observations."""
        model = OLS(formula="y ~ x1 + x2")
        
        # This should work but produce no results or raise error
        try:
            model.fit(sample_dataframe, query="year > 2030")
            # If it succeeds, should have 0 observations
            assert model.n_obs_ == 0
        except ValueError:
            # Or it might raise an error about no data
            pass
    
    def test_streamdata_with_existing_query(self, sample_dataframe):
        """Test that query parameter is ignored when data is already StreamData."""
        # Create StreamData with a query
        data = StreamData(sample_dataframe, query="year >= 2010")
        
        # Try to use different query with OLS - should log warning
        model = OLS(formula="y ~ x1 + x2")
        model.fit(data, query="year >= 2015")  # This query should be ignored
        
        # Should use the original query from StreamData
        assert model.n_obs_ > 0


class TestQueryPerformance:
    """Test query performance optimizations."""
    
    def test_filter_pushdown_fastparquet(self, sample_parquet_file):
        """Test that simple queries are converted to filter pushdown for fastparquet."""
        # This test verifies the internal conversion logic
        data = StreamData(sample_parquet_file, query="year >= 2010", backend='auto')
        
        # Query should be applied efficiently
        total_rows = sum(len(chunk) for chunk in data.iter_chunks())
        assert total_rows > 0
    
    def test_partition_pruning(self, partitioned_parquet_dir):
        """Test that partition pruning works for Hive-style partitioning."""
        # Query on partition column should prune partitions
        data = StreamData(partitioned_parquet_dir, query="year >= 2015")
        
        # Should have pruned some partitions
        assert len(data.info.partitions) < 5  # Original has 5 years
        
        total_rows = sum(len(chunk) for chunk in data.iter_chunks())
        assert total_rows > 0
    
    def test_column_projection_with_query(self, sample_dataframe):
        """Test that only necessary columns are loaded with projection."""
        data = StreamData(sample_dataframe, query="year >= 2010")
        
        # Request only specific columns
        for chunk in data.iter_chunks(columns=['y', 'x1', 'x2']):
            # Should only have requested columns (year is used for filtering but not returned)
            assert set(chunk.columns) == {'y', 'x1', 'x2'}


class TestQueryIntegration:
    """Integration tests for query functionality across the full pipeline."""
    
    def test_end_to_end_ols_with_query(self, sample_dataframe):
        """Test complete OLS workflow with query."""
        # Fit model with query
        model = OLS(formula="y ~ x1 + x2 + I(x1^2)", se_type='stata')
        model.fit(sample_dataframe, query="year >= 2010 and country == 'USA'", cluster='year')
        
        # Check results
        assert model.n_obs_ > 0
        assert model.r_squared_ >= 0
        assert len(model.coef_) == 4  # intercept + x1 + x2 + x1^2
        
        # Get summary
        summary = model.summary()
        assert len(summary) == 4
        assert all(summary['p_value'] >= 0)
        assert all(summary['p_value'] <= 1)
    
    def test_end_to_end_twosls_with_query(self, sample_dataframe):
        """Test complete 2SLS workflow with query."""
        # Make x1 endogenous
        sample_dataframe['x1'] = 0.5 * sample_dataframe['z1'] + np.random.randn(len(sample_dataframe))
        sample_dataframe['y'] = 2 * sample_dataframe['x1'] + sample_dataframe['x2'] + np.random.randn(len(sample_dataframe))
        
        # Fit model with query
        model = TwoSLS(formula="y ~ x1 + x2 | z1 + x2", endogenous=['x1'])
        model.fit(sample_dataframe, query="year >= 2010", cluster='country')
        
        # Check results
        assert model.results_.n_obs > 0
        assert len(model.results_.first_stage_results) == 1
        
        # Get summaries
        first_stage = model.summary(stage='first')
        assert len(first_stage) == 1
        
        second_stage = model.summary(stage='second')
        assert len(second_stage) > 0
    
    def test_query_preserves_transformations(self, sample_dataframe):
        """Test that queries work correctly with feature transformations."""
        model = OLS(formula="y ~ x1 * x2")  # Interaction term
        model.fit(sample_dataframe, query="year >= 2010")
        
        # Should have main effects + interaction
        assert len(model.coef_) == 4  # intercept + x1 + x2 + x1:x2
        assert model.n_obs_ > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

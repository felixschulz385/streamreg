import numpy as np
import pandas as pd
import pytest
from scipy import stats
from streamreg.estimators.ols import OnlineRLS, LinAlgHelper
from streamreg.api import OLS


class TestLinAlgHelper:
    """Tests for linear algebra helper functions."""
    
    def test_safe_solve_regular(self):
        """Test safe_solve with well-conditioned matrix."""
        A = np.array([[2, 1], [1, 2]], dtype=float)
        b = np.array([3, 3], dtype=float)
        alpha = 1e-3
        
        x = LinAlgHelper.safe_solve(A, b, alpha)
        
        # Check solution
        assert np.allclose(A @ x, b, atol=1e-6)
    
    def test_safe_solve_singular(self):
        """Test safe_solve with singular matrix (should regularize)."""
        A = np.array([[1, 1], [1, 1]], dtype=float)
        b = np.array([2, 2], dtype=float)
        alpha = 1e-3
        
        # Should not raise, should use regularization
        x = LinAlgHelper.safe_solve(A, b, alpha)
        assert x is not None
        assert x.shape == (2,)
    
    def test_safe_inv_regular(self):
        """Test safe_inv with invertible matrix."""
        A = np.array([[2, 1], [1, 2]], dtype=float)
        
        A_inv = LinAlgHelper.safe_inv(A, use_pinv=False)
        
        # Check inverse
        assert np.allclose(A @ A_inv, np.eye(2), atol=1e-6)
    
    def test_condition_number(self):
        """Test condition number checking."""
        # Well-conditioned
        A = np.eye(3)
        is_good, cond = LinAlgHelper.check_condition_number(A)
        assert is_good
        assert cond < 10
        
        # Ill-conditioned
        A_bad = np.array([[1, 1], [1, 1.0001]])
        is_good, cond = LinAlgHelper.check_condition_number(A_bad)
        # May or may not be well-conditioned depending on threshold
        # NumPy returns np.bool_ which is a subclass of bool
        assert isinstance(is_good, (bool, np.bool_))
        assert cond > 1


class TestOnlineRLS:
    """Tests for OnlineRLS estimator."""
    
    def test_initialization(self):
        """Test OnlineRLS initialization."""
        rls = OnlineRLS(n_features=3, alpha=1e-3)
        
        assert rls.n_features == 3
        assert rls.alpha == 1e-3
        assert rls.n_obs == 0
        assert rls.theta.shape == (3,)
        assert rls.P.shape == (3, 3)
    
    def test_simple_fit(self):
        """Test fitting with simple data."""
        np.random.seed(42)
        n_samples = 500  # Increased from 100
        n_features = 2
        
        # Generate data: y = 2*x1 + 3*x2 + noise
        X = np.random.randn(n_samples, n_features)
        true_theta = np.array([2.0, 3.0])
        y = X @ true_theta + 0.1 * np.random.randn(n_samples)
        
        rls = OnlineRLS(n_features=n_features, alpha=1e-4)
        rls.partial_fit(X, y)
        
        # Check estimates are close to true values
        assert rls.n_obs == n_samples
        assert np.allclose(rls.theta, true_theta, atol=0.2)  # Tighter tolerance
    
    def test_batch_vs_sequential(self):
        """Test that batch and sequential updates give same result."""
        np.random.seed(42)
        n_samples = 200  # Increased from 50
        n_features = 2
        
        X = np.random.randn(n_samples, n_features)
        y = X @ np.array([1.0, 2.0]) + 0.1 * np.random.randn(n_samples)
        
        # Batch
        rls_batch = OnlineRLS(n_features=n_features)
        rls_batch.partial_fit(X, y)
        
        # Sequential
        rls_seq = OnlineRLS(n_features=n_features)
        for i in range(n_samples):
            rls_seq.partial_fit(X[i:i+1], y[i:i+1])
        
        # Should give same results
        assert np.allclose(rls_batch.theta, rls_seq.theta, atol=1e-6)
        assert np.allclose(rls_batch.rss, rls_seq.rss, atol=1e-4)
    
    def test_r_squared(self):
        """Test R-squared calculation."""
        np.random.seed(42)
        n_samples = 500  # Increased from 100
        n_features = 2
        
        X = np.random.randn(n_samples, n_features)
        y = X @ np.array([2.0, 3.0]) + 0.5 * np.random.randn(n_samples)
        
        rls = OnlineRLS(n_features=n_features, alpha=1e-4)
        rls.partial_fit(X, y)
        
        r_squared = rls.get_r_squared()
        
        # R-squared should be between 0 and 1
        assert 0 <= r_squared <= 1
        # With low noise, should be high
        assert r_squared > 0.8  # Stricter with more data
    
    def test_f_statistic(self):
        """Test F-statistic calculation."""
        np.random.seed(42)
        n_samples = 500  # Increased from 100
        n_features = 2
        
        X = np.random.randn(n_samples, n_features)
        y = X @ np.array([2.0, 3.0]) + 0.5 * np.random.randn(n_samples)
        
        rls = OnlineRLS(n_features=n_features, alpha=1e-4, 
                       feature_names=['x1', 'x2'])
        rls.partial_fit(X, y)
        
        f_stat, df_model, df_resid = rls.get_f_statistic()
        
        # Check degrees of freedom
        assert df_model == 2  # No intercept
        assert df_resid == n_samples - n_features
        
        # F-statistic should be positive
        assert f_stat > 0
        
        # With good fit, F should be large
        assert f_stat > 50  # Increased from 10
    
    def test_empty_data_handling(self):
        """Test handling of empty input data."""
        rls = OnlineRLS(n_features=2)
        
        # Empty arrays
        X_empty = np.array([]).reshape(0, 2)
        y_empty = np.array([])
        
        rls.partial_fit(X_empty, y_empty)
        
        # Should not crash and maintain initial state
        assert rls.n_obs == 0
        assert np.allclose(rls.theta, 0)
    
    def test_invalid_data_filtering(self):
        """Test that invalid data (NaN, inf) is filtered out."""
        rls = OnlineRLS(n_features=2)
        
        X = np.array([[1, 2], [np.nan, 4], [5, np.inf], [7, 8]])
        y = np.array([1, 2, 3, 4])
        
        rls.partial_fit(X, y)
        
        # Should only use valid observations (first and last)
        assert rls.n_obs == 2
    
    def test_single_observation(self):
        """Test with single observation."""
        rls = OnlineRLS(n_features=2, alpha=1e-3)
        
        X = np.array([[1, 2]])
        y = np.array([3])
        
        rls.partial_fit(X, y)
        
        assert rls.n_obs == 1
        assert rls.theta.shape == (2,)
    
    def test_numerical_stability_ill_conditioned(self):
        """Test numerical stability with ill-conditioned data."""
        rls = OnlineRLS(n_features=2, alpha=1e-6)
        
        # Create nearly collinear features
        X = np.array([[1, 1.0001], [2, 2.0002], [3, 3.0003]])
        y = np.array([1, 2, 3])
        
        # Should not raise error due to regularization
        rls.partial_fit(X, y)
        assert rls.n_obs == 3
    
    def test_dimension_mismatch_error(self):
        """Test error handling for dimension mismatch."""
        rls = OnlineRLS(n_features=2)
        
        X = np.array([[1, 2, 3]])  # 3 features instead of 2
        y = np.array([1])
        
        with pytest.raises((ValueError, IndexError)):
            rls.partial_fit(X, y)
    
    def test_merge_statistics(self):
        """Test merging statistics from multiple RLS instances."""
        rls1 = OnlineRLS(n_features=2, alpha=1e-4)
        rls2 = OnlineRLS(n_features=2, alpha=1e-4)
        
        X1 = np.array([[1, 2], [3, 4]])
        y1 = np.array([1, 2])
        X2 = np.array([[5, 6], [7, 8]])
        y2 = np.array([3, 4])
        
        rls1.partial_fit(X1, y1)
        rls2.partial_fit(X2, y2)
        
        # Merge rls2 into rls1
        rls1.merge_statistics(rls2)
        
        # Should have combined observations
        assert rls1.n_obs == 4
        
        # Compare with batch fit
        rls_batch = OnlineRLS(n_features=2, alpha=1e-4)
        X_all = np.vstack([X1, X2])
        y_all = np.hstack([y1, y2])
        rls_batch.partial_fit(X_all, y_all)
        
        assert np.allclose(rls1.theta, rls_batch.theta, atol=1e-6)
    
    def test_covariance_matrix_properties(self):
        """Test covariance matrix is positive semi-definite."""
        rls = OnlineRLS(n_features=2, alpha=1e-4)
        
        X = np.random.randn(500, 2)  # Increased from 100
        y = X @ np.array([1, 2]) + np.random.randn(500)
        
        rls.partial_fit(X, y)
        
        cov = rls.get_covariance_matrix()
        
        # Should be symmetric
        assert np.allclose(cov, cov.T)
        
        # Should be positive semi-definite (eigenvalues >= 0)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-10)  # Allow small numerical error


class TestOLSAPI:
    """Tests for OLS API."""
    
    def test_simple_regression(self):
        """Test simple OLS regression via API."""
        np.random.seed(42)
        n_samples = 1000  # Increased from 200
        
        # Generate data
        df = pd.DataFrame({
            'y': np.random.randn(n_samples),
            'x1': np.random.randn(n_samples),
            'x2': np.random.randn(n_samples)
        })
        df['y'] = 2 * df['x1'] + 3 * df['x2'] + 0.5 * np.random.randn(n_samples)
        
        # Fit model
        model = OLS("y ~ x1 + x2")
        model.fit(df)
        
        # Check fit
        assert model.n_obs_ == n_samples
        assert model.r_squared_ > 0.8  # Stricter with more data
        
        # Check coefficients are reasonable
        coefs = model.coef_
        assert len(coefs) == 3  # intercept + 2 features
        assert abs(coefs[1] - 2.0) < 0.2  # Tighter tolerance
        assert abs(coefs[2] - 3.0) < 0.2
    
    def test_formula_with_intercept_removal(self):
        """Test formula with intercept removal."""
        np.random.seed(42)
        df = pd.DataFrame({
            'y': np.random.randn(500),  # Increased from 100
            'x1': np.random.randn(500)
        })
        
        model = OLS("y ~ x1 - 1")
        model.fit(df)
        
        # Should not have intercept
        assert len(model.coef_) == 1
        assert 'intercept' not in model.results_.feature_names
    
    def test_prediction(self):
        """Test prediction functionality."""
        np.random.seed(42)
        df = pd.DataFrame({
            'y': np.random.randn(500),  # Increased from 100
            'x1': np.random.randn(500),
            'x2': np.random.randn(500)
        })
        df['y'] = 2 * df['x1'] + 3 * df['x2']
        
        model = OLS("y ~ x1 + x2 - 1")
        model.fit(df)
        
        # Predict on new data
        X_new = pd.DataFrame({'x1': [1.0], 'x2': [1.0]})
        y_pred = model.predict(X_new)
        
        # Should be approximately 2*1 + 3*1 = 5
        assert np.abs(y_pred[0] - 5.0) < 0.3  # Tighter tolerance
    
    def test_formula_with_interactions(self):
        """Test formula with interaction terms."""
        np.random.seed(42)
        df = pd.DataFrame({
            'y': np.random.randn(500),  # Increased from 100
            'x1': np.random.randn(500),
            'x2': np.random.randn(500)
        })
        df['y'] = 2 * df['x1'] + 3 * df['x2'] + 1.5 * df['x1'] * df['x2']
        
        model = OLS("y ~ x1 * x2")
        model.fit(df)
        
        # Should have intercept, x1, x2, and x1:x2
        assert model.n_obs_ == 500
        assert len(model.coef_) == 4
    
    def test_formula_with_polynomial(self):
        """Test formula with polynomial terms."""
        np.random.seed(42)
        df = pd.DataFrame({
            'y': np.random.randn(500),  # Increased from 100
            'x1': np.random.randn(500)
        })
        df['y'] = 2 * df['x1'] + 3 * df['x1']**2
        
        model = OLS("y ~ x1 + I(x1^2)")
        model.fit(df)
        
        # Should have intercept, x1, x1^2
        assert len(model.coef_) == 3
        assert abs(model.coef_[2] - 3.0) < 0.3  # Tighter tolerance
    
    def test_error_on_missing_columns(self):
        """Test error when required columns are missing."""
        df = pd.DataFrame({
            'y': np.random.randn(500),  # Increased from 100
            'x1': np.random.randn(500)
        })
        
        model = OLS("y ~ x1 + x2")  # x2 doesn't exist
        
        with pytest.raises(ValueError, match="Missing required columns"):
            model.fit(df)
    
    def test_small_sample_warning(self):
        """Test behavior with very small sample."""
        df = pd.DataFrame({
            'y': [1, 2, 3],
            'x1': [1, 2, 3]
        })
        
        model = OLS("y ~ x1")
        model.fit(df)
        
        # Should fit but R-squared might be perfect
        assert model.n_obs_ == 3
        assert 0 <= model.r_squared_ <= 1


@pytest.mark.monte_carlo
def test_monte_carlo_ols():
    """Monte Carlo test: verify OLS recovers true parameters."""
    np.random.seed(42)
    n_simulations = 100  # Increased from 50
    n_samples = 1000  # Increased from 500
    n_features = 3
    true_theta = np.array([1.0, -0.5, 2.0])
    
    estimates = []
    
    for sim in range(n_simulations):
        # Generate data
        X = np.random.randn(n_samples, n_features)
        y = X @ true_theta + np.random.randn(n_samples)
        
        # Fit model
        rls = OnlineRLS(n_features=n_features, alpha=1e-4)
        rls.partial_fit(X, y)
        
        estimates.append(rls.theta)
    
    estimates = np.array(estimates)
    mean_estimates = estimates.mean(axis=0)
    
    # Mean estimate should be close to true value
    assert np.allclose(mean_estimates, true_theta, atol=0.05)  # Tighter tolerance
    
    # Check coverage: 95% CI should contain true value ~95% of times
    for feat_idx in range(n_features):
        ci_lower = np.percentile(estimates[:, feat_idx], 2.5)
        ci_upper = np.percentile(estimates[:, feat_idx], 97.5)
        
        contains_true = ci_lower <= true_theta[feat_idx] <= ci_upper
        # Allow some slack due to random variation
        assert contains_true or abs(true_theta[feat_idx] - mean_estimates[feat_idx]) < 0.1


@pytest.mark.slow
def test_large_dataset_performance():
    """Test performance on larger dataset."""
    np.random.seed(42)
    n_samples = 50000  # Increased from 10000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = X @ np.random.randn(n_features) + np.random.randn(n_samples)
    
    rls = OnlineRLS(n_features=n_features, alpha=1e-4)
    rls.partial_fit(X, y)
    
    assert rls.n_obs == n_samples
    assert 0 <= rls.get_r_squared() <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

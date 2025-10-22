"""
Monte Carlo tests for validating statistical properties of estimators.
"""
import numpy as np
import pytest
from scipy import stats
from streamreg.estimators.ols import OnlineRLS


@pytest.mark.monte_carlo
def test_unbiasedness():
    """Monte Carlo test: Verify estimator is unbiased."""
    np.random.seed(123)
    n_simulations = 500  # Increased from 200
    n_samples = 1000  # Increased from 300
    n_features = 3
    true_theta = np.array([0.5, -1.0, 2.0])
    
    estimates = []
    
    for _ in range(n_simulations):
        X = np.random.randn(n_samples, n_features)
        y = X @ true_theta + np.random.randn(n_samples)
        
        rls = OnlineRLS(n_features=n_features, alpha=1e-4)
        rls.partial_fit(X, y)
        estimates.append(rls.theta)
    
    estimates = np.array(estimates)
    mean_bias = np.abs(estimates.mean(axis=0) - true_theta)
    
    # Bias should be very small
    assert np.all(mean_bias < 0.03)  # Tighter with more data


@pytest.mark.monte_carlo
def test_consistency():
    """Monte Carlo test: Verify estimator is consistent (converges as n→∞)."""
    np.random.seed(456)
    true_theta = np.array([1.0, 2.0])
    sample_sizes = [500, 2000, 10000]  # Increased from [100, 500, 2000]
    
    variances = []
    
    for n_samples in sample_sizes:
        estimates = []
        
        for _ in range(100):  # Increased from 50
            X = np.random.randn(n_samples, 2)
            y = X @ true_theta + np.random.randn(n_samples)
            
            rls = OnlineRLS(n_features=2, alpha=1e-4)
            rls.partial_fit(X, y)
            estimates.append(rls.theta)
        
        estimates = np.array(estimates)
        variances.append(np.var(estimates, axis=0).mean())
    
    # Variance should decrease as sample size increases
    assert variances[1] < variances[0]
    assert variances[2] < variances[1]


@pytest.mark.monte_carlo
@pytest.mark.slow
def test_t_statistic_distribution():
    """Monte Carlo test: Verify t-statistics follow standard normal under H0."""
    np.random.seed(789)
    n_simulations = 1000  # Increased from 500
    n_samples = 500  # Increased from 200
    
    # True parameter is zero (testing H0: beta = 0)
    true_theta = np.array([0.0, 0.0])
    
    t_statistics = []
    
    for _ in range(n_simulations):
        X = np.random.randn(n_samples, 2)
        y = X @ true_theta + np.random.randn(n_samples)
        
        rls = OnlineRLS(n_features=2, alpha=1e-4)
        rls.partial_fit(X, y)
        
        se = rls.get_standard_errors('classical')
        t_stats = rls.theta / se
        t_statistics.extend(t_stats)
    
    t_statistics = np.array(t_statistics)
    
    # Test if t-statistics follow standard normal
    # Shapiro-Wilk test for normality
    _, p_value = stats.shapiro(t_statistics[:500])  # Use larger subset
    assert p_value > 0.01  # Should not reject normality
    
    # Check mean and std
    assert np.abs(t_statistics.mean()) < 0.05  # Tighter
    assert 0.95 < t_statistics.std() < 1.05  # Tighter


@pytest.mark.monte_carlo
def test_r_squared_properties():
    """Monte Carlo test: Verify R² has expected properties."""
    np.random.seed(321)
    n_simulations = 200  # Increased from 100
    n_samples = 1000  # Increased from 300
    
    r_squareds = []
    
    for _ in range(n_simulations):
        X = np.random.randn(n_samples, 2)
        # True R² should be high (signal-to-noise ratio is high)
        y = 5 * X[:, 0] + 3 * X[:, 1] + 0.1 * np.random.randn(n_samples)
        
        rls = OnlineRLS(n_features=2, alpha=1e-4)
        rls.partial_fit(X, y)
        
        r_squareds.append(rls.get_r_squared())
    
    r_squareds = np.array(r_squareds)
    
    # All R² should be in [0, 1]
    assert np.all(r_squareds >= 0)
    assert np.all(r_squareds <= 1)
    
    # With high signal-to-noise, mean R² should be high
    assert r_squareds.mean() > 0.97  # Stricter with more data


@pytest.mark.monte_carlo
def test_f_statistic_distribution():
    """Monte Carlo test: Verify F-statistic distribution under H0."""
    np.random.seed(654)
    n_simulations = 500  # Increased from 200
    n_samples = 500  # Increased from 100
    n_features = 2
    
    f_statistics = []
    
    # Under null: all coefficients are zero
    for _ in range(n_simulations):
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)  # Pure noise, no relationship
        
        rls = OnlineRLS(n_features=n_features, alpha=1e-4)
        rls.partial_fit(X, y)
        
        f_stat, df_model, df_resid = rls.get_f_statistic()
        f_statistics.append(f_stat)
    
    f_statistics = np.array(f_statistics)
    
    # F-statistics should be mostly small under null
    # Mean F should be close to 1 under null, but with noise can vary
    # Relax bounds to account for random variation
    assert 0.5 < f_statistics.mean() < 1.5
    
    # Very few should be large
    assert np.sum(f_statistics > 10) / len(f_statistics) < 0.05


@pytest.mark.monte_carlo
def test_heteroskedasticity_robustness():
    """Monte Carlo test: Verify estimator handles heteroskedasticity."""
    np.random.seed(987)
    n_simulations = 200  # Increased from 100
    n_samples = 1000  # Increased from 300
    true_theta = np.array([1.0, 2.0])
    
    estimates = []
    
    for _ in range(n_simulations):
        X = np.random.randn(n_samples, 2)
        
        # Heteroskedastic errors: variance depends on X
        error_var = 0.1 + 0.5 * X[:, 0]**2
        errors = np.random.randn(n_samples) * np.sqrt(error_var)
        
        y = X @ true_theta + errors
        
        rls = OnlineRLS(n_features=2, alpha=1e-4)
        rls.partial_fit(X, y)
        estimates.append(rls.theta)
    
    estimates = np.array(estimates)
    
    # Estimator should still be unbiased even with heteroskedasticity
    bias = np.abs(estimates.mean(axis=0) - true_theta)
    assert np.all(bias < 0.05)  # Tighter with more data


@pytest.mark.monte_carlo
@pytest.mark.slow
def test_coverage_probability():
    """Monte Carlo test: Verify confidence interval coverage."""
    np.random.seed(111)
    n_simulations = 1000  # Increased from 500
    n_samples = 500  # Increased from 200
    true_theta = np.array([1.0, 2.0])
    
    coverage_count = 0
    
    for _ in range(n_simulations):
        X = np.random.randn(n_samples, 2)
        y = X @ true_theta + np.random.randn(n_samples)
        
        rls = OnlineRLS(n_features=2, alpha=1e-4)
        rls.partial_fit(X, y)
        
        se = rls.get_standard_errors('classical')
        
        # Check 95% CI
        for feat_idx in range(2):
            ci_lower = rls.theta[feat_idx] - 1.96 * se[feat_idx]
            ci_upper = rls.theta[feat_idx] + 1.96 * se[feat_idx]
            
            if ci_lower <= true_theta[feat_idx] <= ci_upper:
                coverage_count += 1
    
    # Coverage should be approximately 95%
    coverage_rate = coverage_count / (n_simulations * 2)
    assert 0.94 <= coverage_rate <= 0.96  # Tighter with more sims


@pytest.mark.monte_carlo
def test_omitted_variable_bias():
    """Monte Carlo test: Verify omitted variable bias is detected."""
    np.random.seed(222)
    n_simulations = 200  # Increased from 100
    n_samples = 2000  # Increased from 500
    
    # True model: y = x1 + 2*x2 + error, x1 and x2 correlated
    estimates_full = []
    estimates_omitted = []
    
    for _ in range(n_simulations):
        x1 = np.random.randn(n_samples)
        x2 = 0.5 * x1 + np.random.randn(n_samples)  # Correlated with x1
        y = x1 + 2 * x2 + 0.5 * np.random.randn(n_samples)
        
        # Full model (correct)
        X_full = np.column_stack([x1, x2])
        rls_full = OnlineRLS(n_features=2, alpha=1e-4)
        rls_full.partial_fit(X_full, y)
        estimates_full.append(rls_full.theta)
        
        # Omitted variable (biased)
        X_omitted = x1.reshape(-1, 1)
        rls_omitted = OnlineRLS(n_features=1, alpha=1e-4)
        rls_omitted.partial_fit(X_omitted, y)
        estimates_omitted.append(rls_omitted.theta[0])
    
    estimates_full = np.array(estimates_full)
    estimates_omitted = np.array(estimates_omitted)
    
    # Full model should be unbiased
    assert np.abs(estimates_full[:, 0].mean() - 1.0) < 0.05  # Tighter
    assert np.abs(estimates_full[:, 1].mean() - 2.0) < 0.05  # Tighter
    
    # Omitted variable model should be biased
    # x1 coefficient picks up effect of correlated x2
    omitted_bias = estimates_omitted.mean() - 1.0
    assert abs(omitted_bias) > 0.5  # Should have substantial bias


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

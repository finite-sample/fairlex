"""Tests for fairlex metrics module."""

import numpy as np
import pytest
from fairlex.metrics import effective_sample_size, design_effect, evaluate_solution


class TestEffectiveSampleSize:
    """Test effective sample size calculation."""
    
    def test_equal_weights(self):
        """Test ESS with equal weights."""
        weights = np.array([1, 1, 1, 1])
        ess = effective_sample_size(weights)
        assert np.isclose(ess, 4.0)  # Should equal sample size
    
    def test_zero_weights(self):
        """Test ESS with zero weights."""
        weights = np.array([0, 0, 0])
        ess = effective_sample_size(weights)
        assert np.isnan(ess)
    
    def test_mixed_weights(self):
        """Test ESS with mixed weights."""
        weights = np.array([2, 1, 1])
        # ESS = (2+1+1)^2 / (4+1+1) = 16/6 = 8/3
        expected = 16.0 / 6.0
        ess = effective_sample_size(weights)
        assert np.isclose(ess, expected)
    
    def test_single_weight(self):
        """Test ESS with single weight."""
        weights = np.array([5])
        ess = effective_sample_size(weights)
        assert np.isclose(ess, 1.0)


class TestDesignEffect:
    """Test design effect calculation."""
    
    def test_equal_weights_deff(self):
        """Test design effect with equal weights."""
        weights = np.array([1, 1, 1, 1])
        deff = design_effect(weights)
        assert np.isclose(deff, 1.0)  # Should be 1 for equal weights
    
    def test_zero_weights_deff(self):
        """Test design effect with zero weights."""
        weights = np.array([0, 0, 0])
        deff = design_effect(weights)
        assert np.isnan(deff)
    
    def test_variable_weights_deff(self):
        """Test design effect with variable weights."""
        weights = np.array([2, 1, 1])
        # n = 3, ESS = 16/6, so deff = 3 / (16/6) = 18/16 = 9/8
        expected = 3.0 / (16.0/6.0)
        deff = design_effect(weights)
        assert np.isclose(deff, expected)


class TestEvaluateSolution:
    """Test solution evaluation function."""
    
    def test_perfect_solution(self):
        """Test evaluation of perfect calibration."""
        A = np.array([[1, 0], [0, 1], [1, 1]])
        b = np.array([2, 3, 5])
        w = np.array([2, 3])
        
        result = evaluate_solution(A, b, w)
        
        assert np.isclose(result['resid_max_abs'], 0.0)
        assert np.isclose(result['resid_median'], 0.0)
        assert np.isclose(result['total_error'], 0.0)
        assert result['ESS'] > 0
        assert result['deff'] > 0
    
    def test_with_base_weights(self):
        """Test evaluation with base weights for relative deviations."""
        A = np.array([[1, 1]])
        b = np.array([5])
        w = np.array([2, 3])
        base_weights = np.array([1, 1])
        
        result = evaluate_solution(A, b, w, base_weights=base_weights)
        
        assert 'max_rel_dev' in result
        assert 'p95_rel_dev' in result
        assert 'median_rel_dev' in result
        
        # Relative deviations: |2-1|/1 = 1, |3-1|/1 = 2
        assert np.isclose(result['max_rel_dev'], 2.0)
    
    def test_zero_base_weights(self):
        """Test evaluation with zero base weights."""
        A = np.array([[1, 1]])
        b = np.array([5])
        w = np.array([2, 3])
        base_weights = np.array([0, 1])  # First base weight is zero
        
        result = evaluate_solution(A, b, w, base_weights=base_weights)
        
        assert 'max_rel_dev' in result
        # Should handle division by zero gracefully
        assert not np.isnan(result['max_rel_dev'])
    
    def test_custom_quantiles(self):
        """Test evaluation with custom quantiles."""
        A = np.array([[1, 1, 1]])
        b = np.array([6])
        w = np.array([1, 2, 3])
        
        result = evaluate_solution(A, b, w, quantiles=(0.9, 0.1))
        
        # Should still return the standard keys
        assert 'weight_p99' in result
        assert 'weight_p95' in result
        assert 'weight_median' in result
    
    def test_residual_calculations(self):
        """Test residual calculations."""
        A = np.array([[1, 0], [0, 1]])
        b = np.array([1, 2])
        w = np.array([1.5, 2.5])  # Residuals: [0.5, 0.5]
        
        result = evaluate_solution(A, b, w)
        
        assert np.isclose(result['resid_max_abs'], 0.5)
        assert np.isclose(result['resid_median'], 0.5)
        assert np.isclose(result['resid_p95'], 0.5)
    
    def test_weight_statistics(self):
        """Test weight distribution statistics."""
        A = np.array([[1, 1, 1, 1]])
        b = np.array([10])
        w = np.array([1, 2, 3, 4])
        
        result = evaluate_solution(A, b, w)
        
        assert np.isclose(result['weight_min'], 1.0)
        assert np.isclose(result['weight_max'], 4.0)
        assert np.isclose(result['weight_median'], 2.5)
        
        # ESS = (1+2+3+4)^2 / (1+4+9+16) = 100/30 = 10/3
        expected_ess = 100.0 / 30.0
        assert np.isclose(result['ESS'], expected_ess)
        
        # deff = n / ESS = 4 / (10/3) = 12/10 = 1.2
        expected_deff = 4.0 / expected_ess
        assert np.isclose(result['deff'], expected_deff)
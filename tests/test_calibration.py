"""Tests for fairlex calibration algorithms."""

import numpy as np
import pytest
from fairlex import leximin_residual, leximin_weight_fair, CalibrationResult


class TestInputValidation:
    """Test input validation."""
    
    def test_inconsistent_shapes(self):
        """Test that inconsistent array shapes raise ValueError."""
        A = np.array([[1, 0], [0, 1]])  # 2x2
        b = np.array([1, 2, 3])         # length 3 (wrong!)
        w0 = np.array([1, 1])           # length 2
        
        with pytest.raises(ValueError, match="b must be of shape"):
            leximin_residual(A, b, w0)
    
    def test_wrong_w0_shape(self):
        """Test that wrong w0 shape raises ValueError."""
        A = np.array([[1, 0], [0, 1]])  # 2x2
        b = np.array([1, 2])            # length 2
        w0 = np.array([1, 1, 1])        # length 3 (wrong!)
        
        with pytest.raises(ValueError, match="w0 must be of shape"):
            leximin_residual(A, b, w0)
    
    def test_1d_array_A(self):
        """Test that 1D array for A raises ValueError."""
        A = np.array([1, 0])  # 1D
        b = np.array([1, 2])
        w0 = np.array([1, 1])
        
        with pytest.raises(ValueError, match="A must be two.*dimensional"):
            leximin_residual(A, b, w0)


class TestSimpleCases:
    """Test simple, known cases."""
    
    def test_single_variable_single_constraint(self):
        """Test single variable, single constraint case."""
        A = np.array([[1.0]])
        b = np.array([2.0])
        w0 = np.array([1.0])
        
        result = leximin_residual(A, b, w0, min_ratio=0.5, max_ratio=3.0)
        
        assert result.status == 0
        assert np.isclose(result.w[0], 2.0)
        assert np.isclose(result.epsilon, 0.0)
    
    def test_exactly_feasible_problem(self):
        """Test problem with exact solution."""
        A = np.array([
            [1, 0],  # x1 
            [0, 1],  # x2
            [1, 1],  # x1 + x2
        ])
        b = np.array([3, 2, 5])
        w0 = np.array([1, 1])
        
        result = leximin_residual(A, b, w0, min_ratio=0.1, max_ratio=10.0)
        
        assert result.status == 0
        assert np.allclose(result.w, [3, 2])
        assert np.isclose(result.epsilon, 0.0)
        assert np.allclose(A @ result.w, b)
    
    def test_infeasible_problem_leximin(self):
        """Test problem with conflicting constraints."""
        A = np.array([
            [1, 0],  # x1
            [1, 0],  # x1 (same constraint, different target)
        ])
        b = np.array([3, 4])  # impossible: x1 cannot be both 3 and 4
        w0 = np.array([1, 1])
        
        result = leximin_residual(A, b, w0, min_ratio=0.1, max_ratio=10.0)
        
        assert result.status == 0
        # Should minimize max absolute residual
        residuals = A @ result.w - b
        max_abs_residual = np.max(np.abs(residuals))
        assert np.isclose(max_abs_residual, 0.5)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_problem(self):
        """Test empty arrays."""
        A = np.array([]).reshape(0, 0)
        b = np.array([])
        w0 = np.array([])
        
        result = leximin_residual(A, b, w0)
        assert result.status == 0
        assert len(result.w) == 0
    
    def test_zero_base_weights(self):
        """Test zero base weights."""
        A = np.array([[1, 1]])
        b = np.array([2])
        w0 = np.array([0, 1])
        
        result = leximin_residual(A, b, w0, min_ratio=0.5, max_ratio=2.0)
        
        assert result.status == 0
        # First weight should remain 0 (bounds are [0,0])
        assert np.isclose(result.w[0], 0)
        assert np.isclose(result.w[1], 2)
    
    def test_negative_base_weights(self):
        """Test negative base weights."""
        A = np.array([[1, 1]])
        b = np.array([1])
        w0 = np.array([-1, 1])
        
        result = leximin_residual(A, b, w0, min_ratio=0.5, max_ratio=2.0)
        
        # This should fail or handle gracefully
        # Bounds for w[0] would be [-2, -0.5] (negative)
        # The LP solver may not handle this well
        assert result.status != 0 or np.any(np.isnan(result.w))


class TestWeightFairLeximin:
    """Test weight-fair leximin variant."""
    
    def test_weight_fair_basic(self):
        """Test basic weight-fair leximin functionality."""
        A = np.array([
            [1, 0],
            [0, 1],
            [1, 1],
        ])
        b = np.array([3, 2, 5])
        w0 = np.array([1, 1])
        
        result = leximin_weight_fair(A, b, w0, min_ratio=0.1, max_ratio=10.0)
        
        assert result.status == 0
        assert np.allclose(result.w, [3, 2])
        assert np.isclose(result.epsilon, 0.0)
        assert result.t is not None
    
    def test_weight_fair_with_slack(self):
        """Test weight-fair with slack parameter."""
        A = np.array([[1, 1]])
        b = np.array([3])
        w0 = np.array([1, 1])
        
        result = leximin_weight_fair(A, b, w0, min_ratio=0.5, max_ratio=2.0, slack=0.1)
        
        assert result.status == 0
        assert result.epsilon is not None
        assert result.t is not None
    
    def test_return_stages(self):
        """Test return_stages parameter."""
        A = np.array([[1, 1]])
        b = np.array([3])
        w0 = np.array([1, 1])
        
        stage1, stage2 = leximin_weight_fair(
            A, b, w0, min_ratio=0.5, max_ratio=2.0, return_stages=True
        )
        
        assert isinstance(stage1, CalibrationResult)
        assert isinstance(stage2, CalibrationResult)
        assert stage1.t is None  # residual stage
        assert stage2.t is not None  # weight-fair stage


class TestNumericalStability:
    """Test numerical stability and precision."""
    
    def test_very_small_weights(self):
        """Test with very small weights."""
        A = np.array([[1, 1]])
        b = np.array([2e-10])
        w0 = np.array([1e-10, 1e-10])
        
        result = leximin_residual(A, b, w0, min_ratio=0.1, max_ratio=10.0)
        
        assert result.status == 0
        assert not np.any(np.isnan(result.w))
    
    def test_large_weights(self):
        """Test with very large weights."""
        A = np.array([[1, 1]])
        b = np.array([2e10])
        w0 = np.array([1e10, 1e10])
        
        result = leximin_residual(A, b, w0, min_ratio=0.1, max_ratio=10.0)
        
        assert result.status == 0
        assert not np.any(np.isnan(result.w))
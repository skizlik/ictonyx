"""Test statistical analysis functions."""
import pytest
import numpy as np
import pandas as pd
from ictonyx.analysis import (
    mann_whitney_test,
    wilcoxon_signed_rank_test,
    compare_two_models,
    cohens_d,
    check_normality,
    StatisticalTestResult
)


class TestStatisticalTests:
    """Test core statistical functions."""
    
    def test_mann_whitney_basic(self):
        """Test Mann-Whitney U test with clear difference."""
        # Two clearly different groups
        group1 = pd.Series([1, 2, 3, 4, 5, 6])
        group2 = pd.Series([7, 8, 9, 10, 11, 12])
        
        result = mann_whitney_test(group1, group2)
        
        assert isinstance(result, StatisticalTestResult)
        assert result.p_value < 0.05  # Should be significant
        assert result.effect_size is not None
        assert 'Mann-Whitney' in result.test_name
    
    def test_mann_whitney_no_difference(self):
        """Test Mann-Whitney with similar groups."""
        np.random.seed(42)
        group1 = pd.Series(np.random.normal(0, 1, 20))
        group2 = pd.Series(np.random.normal(0, 1, 20))
        
        result = mann_whitney_test(group1, group2)
        
        assert result.p_value > 0.05  # Should not be significant
    
    def test_wilcoxon_signed_rank(self):
        """Test Wilcoxon signed-rank test."""
        # Data clearly different from null value
        data = pd.Series([0.6, 0.7, 0.8, 0.75, 0.65, 0.7, 0.8, 0.85])
        
        result = wilcoxon_signed_rank_test(data, null_value=0.5)
        
        assert isinstance(result, StatisticalTestResult)
        assert result.p_value < 0.05  # Should be significant
        assert 'Wilcoxon' in result.test_name
    
    def test_compare_two_models_independent(self):
        """Test model comparison with independent samples."""
        model1 = pd.Series([0.8, 0.82, 0.79, 0.81, 0.83, 0.80])
        model2 = pd.Series([0.75, 0.74, 0.76, 0.73, 0.77, 0.72])
        
        result = compare_two_models(model1, model2, paired=False)
        
        assert isinstance(result, StatisticalTestResult)
        assert hasattr(result, 'p_value')
        assert hasattr(result, 'effect_size')
    
    def test_compare_two_models_paired(self):
        """Test model comparison with paired samples."""
        model1 = pd.Series([0.8, 0.82, 0.79, 0.81, 0.83, 0.80])
        model2 = pd.Series([0.75, 0.77, 0.74, 0.76, 0.78, 0.75])
        
        result = compare_two_models(model1, model2, paired=True)
        
        assert isinstance(result, StatisticalTestResult)
        assert 'Paired' in result.test_name or 'Wilcoxon' in result.test_name


class TestEffectSizes:
    """Test effect size calculations."""
    
    def test_cohens_d(self):
        """Test Cohen's d calculation."""
        group1 = pd.Series([1, 2, 3, 4, 5])
        group2 = pd.Series([3, 4, 5, 6, 7])
        
        d, interpretation = cohens_d(group1, group2)
        
        assert isinstance(d, float)
        assert d < 0  # group1 has lower mean
        assert interpretation in ['negligible', 'small', 'medium', 'large']


class TestAssumptionChecking:
    """Test statistical assumption checks."""
    
    def test_check_normality(self):
        """Test normality checking."""
        # Normal data
        np.random.seed(42)
        normal_data = pd.Series(np.random.normal(0, 1, 100))
        
        is_normal, details = check_normality(normal_data)
        
        assert isinstance(is_normal, bool)
        assert isinstance(details, dict)
        
        # Highly skewed data
        skewed_data = pd.Series(np.random.exponential(1, 100))
        is_normal_skewed, _ = check_normality(skewed_data)
        
        # Normal data should be more likely to pass
        # (not guaranteed due to randomness)

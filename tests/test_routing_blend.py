"""The blend used by scripts/evaluate_routing_blend.py, checked at its two ends and in between."""
import numpy as np

from scripts.evaluate_routing_blend import blend


def test_probability_zero_is_the_standard_model_and_one_the_luxury_model():
    standard, luxury = np.log1p([300_000.0]), np.log1p([2_000_000.0])
    assert blend(standard, luxury, np.array([0.0])) == standard
    assert blend(standard, luxury, np.array([1.0])) == luxury


def test_the_mix_is_in_log_space():
    standard, luxury = np.array([12.0]), np.array([14.0])
    assert blend(standard, luxury, np.array([0.25])) == np.array([12.5])

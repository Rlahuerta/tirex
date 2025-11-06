import unittest
import numpy as np
import pandas as pd

from tirex.utils.filters import ConvolutionFilter


class TestConvolutionFilterFull(unittest.TestCase):
    """Test suite for 'full' (symmetric/non-causal) filter mode."""

    def setUp(self):
        self.input_signal = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
        self.convolution_filter = ConvolutionFilter(adim=len(self.input_signal), window=3, ftype="full")

    def test_init(self):
        """Test filter initialization with correct parameters."""
        self.assertEqual(self.convolution_filter.length, 3)
        self.assertEqual(self.convolution_filter.adim, len(self.input_signal))

    def test_call(self):
        """Test basic filter application returns correct shape."""
        filtered_signal = self.convolution_filter(self.input_signal)
        self.assertIsNotNone(filtered_signal)
        self.assertEqual(len(filtered_signal), len(self.input_signal))

    def test_smoothness(self):
        """Test that filtering reduces noise and smooths the signal."""
        np.random.seed(42)
        input_signal_noisy = self.input_signal + np.random.normal(0, 0.5, len(self.input_signal))
        filtered_signal = self.convolution_filter(input_signal_noisy)
        smoothness_input = np.abs(np.diff(input_signal_noisy)).mean()
        smoothness_output = np.abs(np.diff(filtered_signal)).mean()
        self.assertLess(smoothness_output, smoothness_input)

    def test_impulse_response(self):
        """Test impulse response shows symmetric spreading."""
        spike_signal = np.zeros(20)
        spike_signal[10] = 10.0
        filt = ConvolutionFilter(adim=20, window=5, ftype="full")
        result = filt(spike_signal)
        
        non_zero_idx = np.where(result > 0.01)[0]
        self.assertIn(10, non_zero_idx, "Spike position should have non-zero response")
        self.assertEqual(result[10], np.max(result), "Peak should be at spike position")
        
        # Check symmetry: values should be symmetric around spike
        if len(non_zero_idx) > 2:
            peak_idx = 10
            left_vals = [result[i] for i in non_zero_idx if i < peak_idx]
            right_vals = [result[i] for i in non_zero_idx if i > peak_idx]
            if left_vals and right_vals:
                np.testing.assert_array_almost_equal(
                    left_vals[::-1], right_vals[:len(left_vals)], decimal=5,
                    err_msg="Full filter should be symmetric"
                )

    def test_constant_signal(self):
        """Test that constant signals remain constant after filtering."""
        constant_signal = np.ones(20) * 5.0
        filt = ConvolutionFilter(adim=20, window=5, ftype="full")
        result = filt(constant_signal)
        np.testing.assert_array_almost_equal(result, constant_signal, decimal=10)

    def test_boundary_conditions_start(self):
        """Test filter behavior at the start of the array."""
        signal = np.arange(10, dtype=float)
        filt = ConvolutionFilter(adim=10, window=5, ftype="full")
        result = filt(signal)
        
        # First element should still be filtered (using available neighbors)
        self.assertIsNotNone(result[0])
        self.assertFalse(np.isnan(result[0]))

    def test_boundary_conditions_end(self):
        """Test filter behavior at the end of the array."""
        signal = np.arange(10, dtype=float)
        filt = ConvolutionFilter(adim=10, window=5, ftype="full")
        result = filt(signal)
        
        # Last element should still be filtered
        self.assertIsNotNone(result[-1])
        self.assertFalse(np.isnan(result[-1]))


class TestConvolutionFilterForward(unittest.TestCase):
    """Test suite for 'forward' (causal) filter mode."""

    def test_impulse_response_forward(self):
        """Test that forward filter only looks ahead (causal)."""
        spike_signal = np.zeros(20)
        spike_signal[10] = 10.0
        filt = ConvolutionFilter(adim=20, window=5, ftype="forward")
        result = filt(spike_signal)
        
        non_zero_idx = np.where(result > 0.01)[0]
        # Forward filter: positions before and at spike see it in their future
        self.assertTrue(all(idx <= 10 for idx in non_zero_idx),
                        "Forward filter should only affect positions at or before spike")

    def test_causality_no_future_leakage(self):
        """Test that forward filter doesn't leak future information to past."""
        signal = np.zeros(20)
        signal[15:] = 10.0  # Step function at position 15
        
        filt = ConvolutionFilter(adim=20, window=5, ftype="forward")
        result = filt(signal)
        
        # Positions well before the step (i=10) should see the step ahead
        # But they should have lower values than at the step
        self.assertGreater(result[15], result[10],
                          "Forward filter: position at step should have higher value")

    def test_forward_boundary_end(self):
        """Test forward filter behavior at array end."""
        signal = np.ones(10) * 5.0
        signal[-1] = 10.0  # Spike at end
        
        filt = ConvolutionFilter(adim=10, window=5, ftype="forward")
        result = filt(signal)
        
        # Positions before the end should see the spike
        self.assertGreater(result[-2], 5.0, "Position before spike should be affected")


class TestConvolutionFilterBackward(unittest.TestCase):
    """Test suite for 'backward' (anti-causal) filter mode."""

    def test_impulse_response_backward(self):
        """Test that backward filter only looks behind (anti-causal)."""
        spike_signal = np.zeros(20)
        spike_signal[10] = 10.0
        filt = ConvolutionFilter(adim=20, window=5, ftype="backward")
        result = filt(spike_signal)
        
        non_zero_idx = np.where(result > 0.01)[0]
        # Backward filter: positions after and at spike see it in their past
        self.assertTrue(all(idx >= 10 for idx in non_zero_idx),
                        "Backward filter should only affect positions at or after spike")
        self.assertIn(10, non_zero_idx, "Spike position should have response")

    def test_anti_causality_no_past_leakage(self):
        """Test that backward filter doesn't leak past information to future."""
        signal = np.zeros(20)
        signal[:5] = 10.0  # Step function ending at position 5
        
        filt = ConvolutionFilter(adim=20, window=5, ftype="backward")
        result = filt(signal)
        
        # Positions at and shortly after the step should see it
        self.assertGreater(result[5], 5.0, "Position at step end should see past values")
        # Positions well after should not see it
        self.assertLess(result[15], 1.0, "Backward filter: distant future shouldn't see past")

    def test_backward_boundary_start(self):
        """Test backward filter behavior at array start."""
        signal = np.ones(10) * 5.0
        signal[0] = 10.0  # Spike at start
        
        filt = ConvolutionFilter(adim=10, window=5, ftype="backward")
        result = filt(signal)
        
        # First position should see only itself
        self.assertGreater(result[0], 5.0, "Start position should see the spike")
        # Nearby positions should see it in their past
        self.assertGreater(result[2], 5.0, "Position after spike should be affected")

    def test_backward_not_all_zeros(self):
        """Regression test: backward filter should not produce all zeros."""
        spike_signal = np.zeros(20)
        spike_signal[10] = 10.0
        filt = ConvolutionFilter(adim=20, window=5, ftype="backward")
        result = filt(spike_signal)
        
        self.assertGreater(np.sum(np.abs(result)), 0.1,
                          "Backward filter should not produce all zeros")


class TestConvolutionFilterParameters(unittest.TestCase):
    """Test suite for different parameter configurations."""

    def test_window_size_effect(self):
        """Test that larger windows create smoother outputs."""
        signal = np.sin(np.linspace(0, 4*np.pi, 50))
        
        results = {}
        for window in [3, 7, 11]:
            filt = ConvolutionFilter(adim=50, window=window, ftype="full")
            results[window] = filt(signal)
        
        # Larger windows should have lower variance in differences
        var_3 = np.var(np.diff(results[3]))
        var_11 = np.var(np.diff(results[11]))
        self.assertLess(var_11, var_3, "Larger window should produce smoother output")

    def test_penalty_effect(self):
        """Test that larger penalty creates sharper weight distributions."""
        spike_signal = np.zeros(20)
        spike_signal[10] = 10.0
        
        results = {}
        spreads = {}
        for penal in [0.5, 1.0, 2.0]:
            filt = ConvolutionFilter(adim=20, window=5, penal=penal, ftype="full")
            results[penal] = filt(spike_signal)
            spreads[penal] = np.sum(results[penal] > 0.1)
        
        # Higher penalty should concentrate more weight at center
        self.assertGreater(results[2.0][10], results[0.5][10],
                          "Higher penalty should increase peak value")

    def test_different_window_sizes(self):
        """Test filter works with various window sizes."""
        signal = np.arange(20, dtype=float)
        for window in [3, 5, 7, 11]:
            filt = ConvolutionFilter(adim=20, window=window, ftype="full")
            result = filt(signal)
            self.assertEqual(len(result), 20)
            self.assertFalse(np.any(np.isnan(result)))


class TestConvolutionFilterIntegration(unittest.TestCase):
    """Test suite for integration with Pandas and complex scenarios."""

    def test_pandas_series_input_output(self):
        """Test that Series input produces Series output with preserved index."""
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=['a', 'b', 'c', 'd', 'e'])
        filt = ConvolutionFilter(adim=5, window=3, ftype="full")
        result = filt(data)
        
        self.assertIsInstance(result, pd.Series, "Series input should produce Series output")
        self.assertEqual(list(result.index), list(data.index), "Index should be preserved")
        self.assertEqual(result.name, "filtered", "Result should have name 'filtered'")

    def test_numpy_array_input_output(self):
        """Test that array input produces array output."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        filt = ConvolutionFilter(adim=5, window=3, ftype="full")
        result = filt(data)
        
        self.assertIsInstance(result, np.ndarray, "Array input should produce array output")
        self.assertEqual(len(result), len(data))

    def test_large_signal(self):
        """Test filter works with larger signals."""
        signal = np.random.randn(1000)
        filt = ConvolutionFilter(adim=1000, window=21, ftype="full")
        result = filt(signal)
        
        self.assertEqual(len(result), 1000)
        self.assertFalse(np.any(np.isnan(result)))

    def test_numerical_stability_large_values(self):
        """Test filter handles large values without overflow."""
        signal = np.ones(20) * 1e6
        filt = ConvolutionFilter(adim=20, window=5, ftype="full")
        result = filt(signal)
        
        np.testing.assert_array_almost_equal(result, signal, decimal=5)

    def test_numerical_stability_small_values(self):
        """Test filter handles small values without underflow."""
        signal = np.ones(20) * 1e-6
        filt = ConvolutionFilter(adim=20, window=5, ftype="full")
        result = filt(signal)
        
        np.testing.assert_array_almost_equal(result, signal, decimal=10)


class TestConvolutionFilterEdgeCases(unittest.TestCase):
    """Test suite for error handling and edge cases."""

    def test_dimension_mismatch_raises_error(self):
        """Test that dimension mismatch raises AssertionError."""
        filt = ConvolutionFilter(adim=10, window=3)
        wrong_signal = np.array([1.0, 2.0, 3.0])
        
        with self.assertRaises(AssertionError) as context:
            filt(wrong_signal)
        self.assertIn("Array dim is wrong", str(context.exception))

    def test_invalid_ftype_raises_error(self):
        """Test that invalid filter type raises ValueError."""
        with self.assertRaises(ValueError) as context:
            filt = ConvolutionFilter(adim=10, window=3, ftype="invalid")
        self.assertIn("Invalid filter type", str(context.exception))

    def test_invalid_input_type_raises_error(self):
        """Test that invalid input type raises AttributeError."""
        filt = ConvolutionFilter(adim=5, window=3)
        
        # Lists don't have .shape attribute, so AttributeError is raised
        with self.assertRaises(AttributeError):
            filt([1, 2, 3, 4, 5])  # List instead of array/Series

    def test_small_signal(self):
        """Test filter works with very small signals."""
        signal = np.array([1.0, 2.0, 3.0])
        filt = ConvolutionFilter(adim=3, window=3, ftype="full")
        result = filt(signal)
        
        self.assertEqual(len(result), 3)
        self.assertFalse(np.any(np.isnan(result)))

    def test_single_element_signal(self):
        """Test filter works with single element."""
        signal = np.array([5.0])
        filt = ConvolutionFilter(adim=1, window=3, ftype="full")
        result = filt(signal)
        
        self.assertEqual(len(result), 1)
        np.testing.assert_almost_equal(result[0], 5.0)


if __name__ == '__main__':
    unittest.main()

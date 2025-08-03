import unittest
import numpy as np

from tirex.utils.filters import ConvolutionFilter


class TestConvolutionFilter(unittest.TestCase):

    def setUp(self):
        self.input_signal = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.convolution_filter = ConvolutionFilter(adim=len(self.input_signal), length=3)

    def test_init(self):
        self.assertEqual(self.convolution_filter.length, 3)
        self.assertEqual(self.convolution_filter.adim, len(self.input_signal))

    def test_call(self):
        filtered_signal = self.convolution_filter(self.input_signal)
        self.assertIsNotNone(filtered_signal)
        self.assertEqual(len(filtered_signal), len(self.input_signal))

    def test_smoothness(self):
        input_signal_noisy = self.input_signal + np.random.normal(0, 0.5, len(self.input_signal))
        filtered_signal = self.convolution_filter(input_signal_noisy)
        smoothness_input = np.abs(np.diff(input_signal_noisy)).mean()
        smoothness_output = np.abs(np.diff(filtered_signal)).mean()
        self.assertLess(smoothness_output, smoothness_input)


if __name__ == '__main__':
    unittest.main()

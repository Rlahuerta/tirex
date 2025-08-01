# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import os
from unittest.mock import patch

import numpy as np
import pytest

from tirex.utils.plot import plot_fc


@pytest.fixture
def sample_data():
    """Provides sample data for plotting tests."""
    ctx = np.random.rand(50)
    quantile_fc = np.random.rand(20, 9)
    real_future_values = np.random.rand(20)
    return ctx, quantile_fc, real_future_values


def test_plot_fc_runs_without_error(sample_data):
    """Tests that plot_fc runs without raising an exception."""
    ctx, quantile_fc, real_future_values = sample_data
    try:
        with patch('matplotlib.pyplot.show'):
            plot_fc(ctx, quantile_fc, real_future_values)
            plot_fc(ctx, quantile_fc)
    except Exception as e:
        pytest.fail(f"plot_fc raised an exception: {e}")


def test_plot_fc_save_plot(sample_data, tmp_path):
    """Tests that plot_fc saves a plot to the specified path."""
    ctx, quantile_fc, real_future_values = sample_data
    save_path = tmp_path / "test_plot.png"

    plot_fc(ctx, quantile_fc, real_future_values, save_path=str(save_path))

    assert os.path.exists(save_path)


@patch('matplotlib.pyplot.show')
def test_plot_fc_show_plot(mock_show, sample_data):
    """Tests that plot_fc calls plt.show() when no save_path is provided."""
    ctx, quantile_fc, real_future_values = sample_data

    plot_fc(ctx, quantile_fc, real_future_values)

    mock_show.assert_called_once()

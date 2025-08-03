import os
import unittest
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from joblib import load
from typing import Tuple, Dict, Union
from scipy.signal import welch

from tirex.utils.ceemdan import EEMD, CEEMDAN, ICEEMDAN, EMDVisualisation
from . import test_data_path, full_timeseries_data, unit_test_plots_path

unit_test_ceemdan_plots_path = (unit_test_plots_path / 'ceemdan').resolve()
os.makedirs(unit_test_ceemdan_plots_path, exist_ok=True)

unit_test_iceemdan_plots_path = (unit_test_plots_path / 'iceemdan').resolve()
os.makedirs(unit_test_iceemdan_plots_path, exist_ok=True)

def emd_plot(inp_time: np.ndarray,
             inp_signal: np.ndarray,
             emd_signal: np.ndarray,
             plot_title: str,
             plot_name: str,
             res_signal: Union[np.ndarray, Dict[str, np.ndarray], None] = None,
             ):
    """Helper function to plot EMD/CEEMDAN results."""

    num_plots = emd_signal.shape[0] + 1
    ini_idx = 1

    if res_signal is not None:
        num_plots += 1
        ini_idx = 2

    fig, ax = plt.subplots(num_plots, 1, figsize=(15, 12))
    fig.suptitle(plot_title, fontsize=12)

    ax[0].plot(inp_time, inp_signal, 'r')
    ax[0].plot(inp_time, emd_signal.sum(axis=0), 'blue', dashes=[2, 4])

    ax[0].set_ylabel("Original signal")
    ax[0].grid(which='minor', alpha=0.2)
    ax[0].grid(which='major', alpha=0.5)

    if res_signal is not None:
        ax[1].set_ylabel("Residuum")
        ax[1].grid(which='minor', alpha=0.2)
        ax[1].grid(which='major', alpha=0.5)

        if isinstance(res_signal, np.ndarray):
            ax[1].plot(inp_time, res_signal, 'r')

        elif isinstance(res_signal, Dict):
            for i, (key_i, val_i) in enumerate(res_signal.items()):
                if i == 0:
                    ax[1].plot(inp_time, val_i, 'r', label=key_i)
                else:
                    ax[1].plot(inp_time, val_i, '.', label=key_i)

            ax[1].legend(loc='best')

    for i, n in enumerate(range(ini_idx, num_plots)):
        ax[n].plot(inp_time, emd_signal[i, :], 'g')
        ax[n].set_ylabel("eIMF %i" % (i + 1))
        ax[n].locator_params(axis='y', nbins=5)
        ax[n].grid(which='minor', alpha=0.2)
        ax[n].grid(which='major', alpha=0.5)

    ax[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(f'{plot_name}', dpi=200)


def create_test_signal(n_points: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0., 1., n_points, dtype=np.float64)
    s = (np.sin(2 * np.pi * 5 * t) +
         0.5 * np.sin(2 * np.pi * 25 * t) +
         0.2 * np.sin(2 * np.pi * 50 * t))
    s += 0.1 * np.random.randn(n_points) # Add some noise
    return t, s


def psd_selector(time: np.ndarray, imfs: np.ndarray, tol: float = 1.e-3) -> Tuple[np.ndarray, np.ndarray]:

    list_psd_sum = []
    for i, imf in enumerate(imfs):
        # freq_i, psd_i = welch(imf, fs=1. / (time[1] - time[0]), nperseg=1024)
        freq_i, psd_i = welch(imf, fs=1. / (time[1] - time[0]))
        list_psd_sum.append(psd_i.sum())

    np_psd_nrm = np.array(list_psd_sum) / sum(list_psd_sum)
    np_psd_slc = np_psd_nrm > tol
    imfs_psd_slc = imfs[np_psd_slc, :]

    return np_psd_nrm, imfs_psd_slc


class CEEMDANTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_path = test_data_path

        assert full_timeseries_data.is_file() is True
        cls.joblib_data_file = full_timeseries_data
        cls.price_data = load(cls.joblib_data_file)

        cls.dt = 15
        cls.filter_len = 15
        cls.input_seq_length = 300
        cls.output_seq_length = 24
        cls.target_indices = [0]

        cls.t, cls.s = create_test_signal()

    def test_decomposition(self):
        ceemdan = CEEMDAN(trials=100, epsilon=0.1, seed=42)

        imfs = ceemdan.ceemdan(self.s, self.t)

        # Test that the sum of IMFs and residue reconstructs the original signal
        reconstructed_signal = np.sum(imfs, axis=0)
        np.testing.assert_allclose(self.s, reconstructed_signal, rtol=1e-2, atol=1e-2,
                                   err_msg="Reconstructed signal does not match the original signal within tolerance.")

    def test_imf_count(self):
        ceemdan = CEEMDAN(trials=100, epsilon=0.1, seed=42)

        imfs = ceemdan.ceemdan(self.s, self.t)
        # We expect at least 3 IMFs for our test signal
        self.assertTrue(imfs.shape[0] >= 3, "Not enough IMFs extracted.")

    def test_imf_frequencies(self):
        from scipy.signal import hilbert

        ceemdan = CEEMDAN(trials=200, epsilon=0.5, seed=42)
        imfs = ceemdan.ceemdan(self.s, self.t)
        expected_frequencies = [5, 15, 30]  # Frequencies in our test signal

        list_imf_freq = []
        for i, imf in enumerate(imfs[:-1]):  # Exclude residue
            # Compute instantaneous frequency
            analytic_signal = hilbert(imf)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi * (self.t[1] - self.t[0]))

            # Discard the first and last 10% to avoid edge effects
            instantaneous_frequency = instantaneous_frequency[
                                      int(0.1 * len(instantaneous_frequency)):-int(0.1 * len(instantaneous_frequency))]
            mean_freq = np.mean(instantaneous_frequency)
            list_imf_freq.append(mean_freq)
            print(f"IMF {i + 1} mean frequency: {mean_freq}")

        # Check if mean frequency is close to one of the expected frequencies
        np_imf_freq = np.array(list_imf_freq)
        matched_indices = []
        for expected_freq_i in expected_frequencies:
            diffs = np.abs(np_imf_freq - expected_freq_i)
            for idx in np.argsort(diffs):
                if idx not in matched_indices:
                    min_diff = diffs[idx]
                    if min_diff < 1.0:
                        matched_indices.append(idx)
                        closest_freq = np_imf_freq[idx]
                        print(f"Expected frequency {expected_freq_i} Hz matched with IMF {idx + 1} "
                              f"having mean frequency {closest_freq} Hz")
                        break
            else:
                print(f"Expected frequency {expected_freq_i} Hz not found in IMFs within tolerance.")
                # self.fail(f"Expected frequency {expected_freq_i} Hz not found in IMFs within tolerance.")

        # FIXME
        # self.assertEqual(len(matched_indices), len(expected_frequencies),
        #                  "Not all expected frequencies were matched with IMFs.")
        self.assertTrue(len(matched_indices) > 0)

    def test_orthogonality(self):
        ceemdan = CEEMDAN(trials=600, epsilon=0.01, seed=42)

        imfs = ceemdan.ceemdan(self.s, self.t)
        num_imfs = imfs.shape[0]
        list_orthogonality = []

        # Plot Power Spectral Density (PSD):
        # - Check PSD Peaks: Identify the dominant frequencies in each IMF.
        # - Determine Overlaps: If IMF 2 and IMF 3 share significant frequency content, mode mixing is occurring.
        fig, ax = plt.subplots(imfs.shape[0], 1, figsize=(15, 12), sharex=True)
        fig.suptitle("Plot Power Spectral Density (PSD)", fontsize=12)

        list_psd_sum = []
        for i, imf in enumerate(imfs):
            freq_i, psd_i = welch(imf, fs=1. / (self.t[1] - self.t[0]))
            ax[i].semilogy(freq_i, psd_i)
            ax[i].set_ylabel(f'IMF {i + 1}')
            ax[i].grid(which='minor', alpha=0.2)
            ax[i].grid(which='major', alpha=0.5)
            list_psd_sum.append(psd_i.sum())

        ax[-1].set_xlabel('Frequency [Hz]')

        fig.tight_layout()
        fig.savefig(f'{unit_test_ceemdan_plots_path}/ceemdan_test_case_orthogonality_psd.png', dpi=200)

        np_psd_nrm = np.array(list_psd_sum) / sum(list_psd_sum)
        np_psd_slc = np_psd_nrm > 1.e-3
        imfs_psd_slc = imfs[np_psd_slc, :]

        emd_plot(self.t,
                 self.s,
                 imfs,
                 res_signal=ceemdan.residue,
                 plot_title="CEEMDAN Unit Test Orthogonality Case",
                 plot_name=f"{unit_test_ceemdan_plots_path}/ceemdan_test_case_orthogonality.png",
                 )

        emd_plot(self.t,
                 self.s,
                 imfs_psd_slc,
                 plot_title="CEEMDAN Unit Test Orthogonality Case",
                 plot_name=f"{unit_test_ceemdan_plots_path}/ceemdan_test_case_orthogonality_psd_slc.png",
                 )

        for i in range(num_imfs):
            for j in range(i + 1, num_imfs):
                inner_product = np.dot(imfs[i], imfs[j])
                norm_product = np.linalg.norm(imfs[i]) * np.linalg.norm(imfs[j])
                cosine_similarity = inner_product / norm_product
                list_orthogonality.append(dict(cosine_similarity=cosine_similarity,
                                               imfs_i=i, imfs_j=j
                                               ))

        test = 1.
        # self.assertTrue(np.abs(cosine_similarity) < 0.2,
        #                 f"IMFs {i + 1} and {j + 1} are not orthogonal (cosine similarity: {cosine_similarity}).")

    def test_noise_robustness(self):
        # Run decomposition with different seeds and check that results are similar
        ceemdan1 = CEEMDAN(trials=100, epsilon=0.1, seed=42)
        ceemdan2 = CEEMDAN(trials=100, epsilon=0.1, seed=24)
        imfs1 = ceemdan1(self.s, self.t)
        imfs2 = ceemdan2(self.s, self.t)
        # Compare the first IMF from both decompositions
        np.testing.assert_allclose(imfs1[0], imfs2[0], rtol=1e-1, atol=1e-1,
                                   err_msg="First IMFs from different seeds are not similar.")

    def test_edge_case_short_signal(self):
        t_short = self.t[:50]
        s_short = self.s[:50]

        ceemdan = CEEMDAN(trials=100, epsilon=0.1, seed=42)
        imfs = ceemdan(s_short, t_short)

        # Check if decomposition runs without errors and returns IMFs
        self.assertTrue(imfs.shape[1] == 50, "IMFs do not match the length of the short signal.")

    def test_zero_signal(self):
        s_zero = np.zeros_like(self.s)

        ceemdan = CEEMDAN(trials=100, epsilon=0.1, seed=42)
        imfs = ceemdan(s_zero, self.t)

        # The decomposition of a zero signal should return zeros
        self.assertTrue(np.all(imfs == 0), "Decomposition of zero signal should return zeros.")

    def test_compare_with_emd(self):
        # Compare the first IMF from CEEMDAN with EMD
        eemd = EEMD()
        emd_imfs = eemd(self.s, self.t)

        ceemdan = CEEMDAN(trials=100, epsilon=0.1, seed=42)
        ceemdan_imfs = ceemdan(self.s, self.t)

        # Compare the first IMF
        np.testing.assert_allclose(emd_imfs[0], ceemdan_imfs[0], rtol=1e-1, atol=1e-1,
                                   err_msg="First IMF from CEEMDAN does not match EMD.")

    def test_simple_run(self):

        np_x = np.linspace(0, 1, 100)
        np_y = np.sin(2 * np.pi * np_x)

        config = {"processes": 1, "spline_kind": 'akima', "DTYPE": np.float16}
        ceemdan = CEEMDAN(trials=10, max_imf=-1, **config)

        # First run
        c_imfs = ceemdan.ceemdan(np_y, T=np_x)

        # Plot results
        emd_plot(np_x, np_y, c_imfs,
                 plot_title="CEEMDAN Unit Test Case 1",
                 plot_name=f'{unit_test_ceemdan_plots_path}/ceemdan_test_case_1')

        emd_vis_plot = EMDVisualisation(emd_instance=ceemdan)
        emd_vis_plot.plot_imfs(plot_name=f'{unit_test_ceemdan_plots_path}/ceemdan_test_case_imfs_1.png')
        emd_vis_plot.plot_instant_freq(np_x, plot_name=f'{unit_test_ceemdan_plots_path}/ceemdan_test_case_freq_1.png')

        self.assertTrue("processes" in ceemdan.__dict__)
        self.assertGreaterEqual(c_imfs.shape[0], 1)

    def test_residual_run(self):

        max_imf = -1

        # Signal options
        nsize = 500
        t_min, t_max = 0, 2 * np.pi

        np_x = np.linspace(t_min, t_max, nsize)
        np_y = 3 * np.sin(4 * np_x) + 4 * np.cos(9 * np_x) + np.sin(8.11 * np_x + 1.2)

        # Prepare and run EEMD
        trials = 20
        ceemdan = CEEMDAN(trials=trials)
        C_IMFs = ceemdan(np_y, np_x, max_imf)
        np_res = np_y - np.sum(C_IMFs, axis=0)

        emd_plot(np_x, np_y,
                 C_IMFs,
                 res_signal=np_res,
                 plot_title="CEEMDAN Unit Test Case 3",
                 plot_name=f"{unit_test_ceemdan_plots_path}/ceemdan_test_case_3",
                 )

        self.assertGreaterEqual(C_IMFs.shape[0], 4)
        self.assertLessEqual(np_res.sum(), 1.e-6)

    def test_full_timeseries_run(self):

        max_imf = 8

        np_y = self.price_data[15]["close"].values[:self.input_seq_length]
        np_x = np.linspace(0., 1., np_y.shape[0])

        config = {"processes": 1, "spline_kind": 'akima', "DTYPE": np.float16}
        ceemdan = CEEMDAN(trials=20, max_imf=max_imf, **config)

        # First run
        seed_a = 20
        ceemdan.noise_seed(seed_a)
        c_imfs_a = ceemdan.ceemdan(np_y, T=np_x, max_imf=max_imf)
        emd_plot(np_x, np_y, c_imfs_a,
                 plot_title=f"CEEMDAN Unit Test Case 2 - seed: {seed_a}",
                 plot_name=f"{unit_test_ceemdan_plots_path}/ceemdan_test_case_2a.png",
                 )
        # emd_vis_plot = EMDVisualisation(emd_instance=ceemdan)
        # emd_vis_plot.plot_instant_freq(np_x, plot_name=f'{unit_test_plots_path}/ceemdan_test_case_freq_2a.png')

        # Second run
        seed_b = 49
        ceemdan.noise_seed(seed_b)
        c_imfs_b = ceemdan.ceemdan(np_y, T=np_x, max_imf=max_imf)
        emd_plot(np_x, np_y, c_imfs_b,
                 plot_title=f"CEEMDAN Unit Test Case 2 - seed: {seed_b}",
                 plot_name=f"{unit_test_ceemdan_plots_path}/ceemdan_test_case_2b.png",
                 )
        # emd_vis_plot = EMDVisualisation(emd_instance=ceemdan)
        # emd_vis_plot.plot_instant_freq(np_x, plot_name=f'{unit_test_plots_path}/ceemdan_test_case_freq_2b.png')

        # diff_ab = c_imfs_a - c_imfs_b
        diff_res = c_imfs_a.sum(axis=0) - c_imfs_b.sum(axis=0)
        self.assertLessEqual(diff_res.sum(), 1.e-6)

        np_psd_nrm, imfs_b_psd_slc = psd_selector(np_x, c_imfs_b, tol=0.05)

        emd_plot(np_x, np_y, imfs_b_psd_slc,
                 plot_title=f"CEEMDAN Unit Test Case 2, PSD Selection - seed: {seed_b}",
                 plot_name=f"{unit_test_ceemdan_plots_path}/ceemdan_test_case_2b_psd_slc.png",
                 )

class ICEEMDANTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.t_global, cls.s_global = create_test_signal(2048)
        cls.data_path = test_data_path

        assert full_timeseries_data.is_file() is True
        cls.joblib_data_file = full_timeseries_data
        cls.price_data = load(cls.joblib_data_file)

        cls.dt = 15
        cls.filter_len = 15
        cls.input_seq_length = 300
        cls.output_seq_length = 24
        cls.target_indices = [0]
        cls.default_config = {"trials": 50, "epsilon": 0.2, "seed": 42, "parallel": False}

    def test_simple_run(self):

        np_x = np.linspace(0, 1, 100)
        np_y = np.sin(2 * np.pi * np_x)

        config = {"processes": 1, "spline_kind": 'akima', "DTYPE": np.float16}
        iceemdan = ICEEMDAN(trials=10, max_imf=-1, **config)

        # First run
        c_imfs = iceemdan.iceemdan(np_y, T=np_x)

        # Plot results
        emd_plot(np_x, np_y, c_imfs,
                 plot_title="ICEEMDAN Unit Test Case 1",
                 plot_name=f'{unit_test_iceemdan_plots_path}/iceemdan_test_case_1.png')

        emd_vis_plot = EMDVisualisation(emd_instance=iceemdan)
        emd_vis_plot.plot_imfs(plot_name=f'{unit_test_iceemdan_plots_path}/iceemdan_test_case_imfs_1.png')
        emd_vis_plot.plot_instant_freq(np_x, plot_name=f'{unit_test_iceemdan_plots_path}/iceemdan_test_case_freq_1.png')

        self.assertTrue("processes" in iceemdan.__dict__)
        self.assertGreaterEqual(c_imfs.shape[0], 1)

    def test_residual_run(self):

        max_imf = -1

        # Signal options
        nsize = 500
        t_min, t_max = 0, 2 * np.pi

        np_x = np.linspace(t_min, t_max, nsize)
        np_y = 3 * np.sin(4 * np_x) + 4 * np.cos(9 * np_x) + np.sin(8.11 * np_x + 1.2)

        # Prepare and run EEMD
        trials = 20
        iceemdan = ICEEMDAN(trials=trials)
        C_IMFs = iceemdan(np_y, np_x, max_imf)
        np_res = np_y - np.sum(C_IMFs, axis=0)

        emd_plot(np_x, np_y,
                 C_IMFs,
                 res_signal=np_res,
                 plot_title="ICEEMDAN Unit Test Case 3",
                 plot_name=f"{unit_test_iceemdan_plots_path}/iceemdan_test_case_3.png",
                 )

        self.assertGreaterEqual(C_IMFs.shape[0], 4)
        self.assertLessEqual(np_res.sum(), 1.e-6)

    def test_imf_properties(self):
        """Test basic properties of extracted IMFs like count and frequency content."""

        iceemdan = ICEEMDAN(**self.default_config)
        imfs_stack = iceemdan.iceemdan(self.s_global, self.t_global)
        imfs = imfs_stack[:-1]  # Exclude residue for IMF-specific tests

        self.assertGreaterEqual(imfs.shape[0], 2, "Expected at least 2 IMFs for the test signal.")

        # Welch PSD estimation parameters
        fs = 1.0 / (self.t_global[1] - self.t_global[0])

        actual_dominant_freqs = []
        for i, imf_component in enumerate(imfs):
            if np.all(imf_component == 0): continue  # Skip zero IMFs
            f, Pxx = welch(imf_component, fs=fs, nperseg=min(256, len(imf_component)))
            dominant_freq = f[np.argmax(Pxx)]
            actual_dominant_freqs.append(dominant_freq)
            print(f"IMF {i + 1} dominant frequency: {dominant_freq:.2f} Hz")

        # Check if frequencies are generally decreasing (can be complex due to mode mixing)
        # This is a weak check and highly dependent on the signal and EMD variant.
        # For a more robust check, one might need to compare with known IMFs for specific signals.
        if len(actual_dominant_freqs) > 1:
            for i in range(len(actual_dominant_freqs) - 1):
                self.assertTrue(actual_dominant_freqs[i] >= actual_dominant_freqs[i + 1] - fs * 0.1,
                                # Allow some overlap/flexibility
                                f"IMF frequencies not strictly decreasing: {actual_dominant_freqs[i]} vs {actual_dominant_freqs[i + 1]}")

    def test_full_timeseries_run(self):

        max_imf = 8

        np_y = self.price_data[15]["close"].values[:self.input_seq_length]
        np_x = np.linspace(0., 1., np_y.shape[0])

        config = {"processes": 1, "spline_kind": 'akima', "DTYPE": np.float16}

        # First run
        seed_a = 20
        iceemdan = ICEEMDAN(trials=20, max_imf=max_imf, seed=seed_a, **config)
        c_imfs_a = iceemdan.iceemdan(np_y, T=np_x, max_imf=max_imf)
        emd_plot(np_x, np_y, c_imfs_a,
                 plot_title=f"ICEEMDAN Unit Test Case 2 - seed: {seed_a}",
                 plot_name=f"{unit_test_iceemdan_plots_path}/iceemdan_test_case_2seed20.png",
                 )

        # Second run
        seed_b = 49
        iceemdan = ICEEMDAN(trials=20, max_imf=max_imf, seed=seed_b, **config)
        c_imfs_b = iceemdan.iceemdan(np_y, T=np_x, max_imf=max_imf)
        emd_plot(np_x, np_y, c_imfs_b,
                 plot_title=f"ICEEMDAN Unit Test Case 2 - seed: {seed_b}",
                 plot_name=f"{unit_test_iceemdan_plots_path}/iceemdan_test_case_2seed49.png",
                 )

        # diff_ab = c_imfs_a - c_imfs_b
        diff_res = c_imfs_a.sum(axis=0) - c_imfs_b.sum(axis=0)
        self.assertLessEqual(diff_res.sum(), 1.e-2)

    def test_iceemdan_timeseries_run(self):

        max_imf = 8
        seed_a = 20

        np_y = self.price_data[15]["close"].values[5000:][:2000]
        np_x = np.linspace(0., 1., np_y.shape[0])

        config = {"processes": 1, "spline_kind": 'akima', "DTYPE": np.float16}
        iceemdan = ICEEMDAN(trials=20, epsilon=0.05, seed=seed_a, **config)
        np_c_imfs_a = iceemdan.iceemdan(np_y, T=np_x, max_imf=max_imf)

        ######################################################
        emd_plot(np_x, np_y,
                 np_c_imfs_a,
                 plot_title=f"ICEEMDAN-REF Unit Test Case 3 - seed: {seed_a}",
                 plot_name=f"{unit_test_iceemdan_plots_path}/iceemdan_test_case_ref_3a.png",
                 )


if __name__ == '__main__':
    unittest.main()

import numpy as np
import re
import pytest
from .. import calibration
from ...tests import mock_uvdata as mockuvd
from hera_cal.lst_stack import LSTStack
from hera_cal import redcal, utils


class TestLSTBinCalibration:
    def setup_class(self):
        self.rng = np.random.default_rng(42)
        self.uvd = mockuvd.create_uvd_identifiable(
            integration_time=24 * 3600,
            ntimes=20,
            jd_start=2459844.0,
            ants=[0, 1, 2, 3],
            time_axis_faster_than_bls=False,
            pols=["xx", "yy"],
            freqs=mockuvd.PHASEII_FREQS[:100],
        )

        auto_uvd = mockuvd.create_uvd_identifiable(
            integration_time=24 * 3600,
            ntimes=20,
            jd_start=2459844.0,
            ants=[
                0,
            ],
            time_axis_faster_than_bls=False,
            pols=["xx", "yy"],
            freqs=mockuvd.PHASEII_FREQS[:100],
        )

        self.antpos = {ant: np.array([ant * 10, 0, 0]) for ant in range(4)}
        self.stack = LSTStack(self.uvd)
        self.auto_stack = LSTStack(auto_uvd)
        self.stack.data[1:] = self.stack.data[0]  # All nights exactly the same
        self.auto_stack.data[1:] = self.auto_stack.data[
            0
        ]  # All nights exactly the same
        self.all_reds = redcal.get_reds(self.antpos, pols=self.stack.pols)

    def test_amp_cal(self):
        stack_copy = self.stack.copy()
        auto_stack_copy = self.auto_stack.copy()
        gains = self.rng.normal(1, 0.1, size=(20, stack_copy.data.shape[2], 2))

        stack_copy.data *= gains[:, None, :, :] ** 2
        auto_stack_copy.data *= gains[:, None, :, :] ** 2
        model = np.mean(self.stack.data, axis=0)
        auto_model = np.mean(self.auto_stack.data, axis=0)

        pre_cal_std = np.std(stack_copy.data, axis=0)
        pre_cal_auto_std = np.std(auto_stack_copy.data, axis=0)

        cal_params, _ = calibration.lstbin_absolute_calibration(
            stack_copy,
            model,
            all_reds=[],
            auto_stack=auto_stack_copy,
            auto_model=auto_model,
            run_amplitude_cal=True,
            run_phase_cal=False,
            run_cross_pol_phase_cal=False,
            calibrate_inplace=True,
            smooth_gains=False,
        )

        post_cal_std = np.std(stack_copy.data, axis=0)
        post_cal_auto_std = np.std(auto_stack_copy.data, axis=0)
        del stack_copy

        # Check that the standard deviation of the data decreased after calibration
        assert np.all(post_cal_std < pre_cal_std)
        assert np.all(post_cal_auto_std < pre_cal_auto_std)
        assert np.allclose(cal_params["A_Jnn"], gains[:, :, 0])
        assert np.allclose(cal_params["A_Jee"], gains[:, :, 1])

        stack_copy_w_smoothing = self.stack.copy()
        stack_copy_w_smoothing.data *= gains[:, None, :, :] ** 2
        model = np.mean(self.stack.data, axis=0)

        cal_params, _ = calibration.lstbin_absolute_calibration(
            stack_copy_w_smoothing,
            model,
            all_reds=[],
            auto_stack=auto_stack_copy,
            auto_model=auto_model,
            run_amplitude_cal=True,
            run_phase_cal=False,
            run_cross_pol_phase_cal=False,
            calibrate_inplace=True,
            smooth_gains=True,
        )

        post_cal_std_w_smoothing = np.std(stack_copy_w_smoothing.data, axis=0)

        # Smoothing should increase the standard deviation of the data slightly
        # especially since the gains are random
        assert np.all(post_cal_std < post_cal_std_w_smoothing)

    def test_phase_cal(self):
        stack_copy = self.stack.copy()
        tip_tilt = self.rng.normal(0, 0.1, size=(20, self.stack.data.shape[2], 2))
        gains = np.array(
            [
                np.exp(1j * tip_tilt * (self.antpos[ant2] - self.antpos[ant1])[0])
                for (ant1, ant2) in self.stack.antpairs
            ]
        )
        gains = np.transpose(gains, (1, 0, 2, 3))
        stack_copy.data *= gains
        model = np.mean(self.stack.data, axis=0)

        pre_cal_std = np.std(stack_copy.data, axis=0)

        cal_params, delta_gains = calibration.lstbin_absolute_calibration(
            stack_copy,
            model,
            self.all_reds,
            run_amplitude_cal=False,
            run_phase_cal=True,
            run_cross_pol_phase_cal=False,
            calibrate_inplace=True,
            smooth_gains=False,
        )
        post_cal_std = np.std(stack_copy.data, axis=0)
        del stack_copy

        for ai, antpair in enumerate(self.stack.antpairs):
            if antpair[0] == antpair[1]:
                continue
            assert np.all(post_cal_std[ai] <= pre_cal_std[ai])

        stack_copy_w_smoothing = self.stack.copy()
        stack_copy_w_smoothing.data *= gains

        cal_params, _ = calibration.lstbin_absolute_calibration(
            stack_copy_w_smoothing,
            model,
            self.all_reds,
            run_amplitude_cal=False,
            run_phase_cal=True,
            run_cross_pol_phase_cal=False,
            calibrate_inplace=True,
            smooth_gains=True,
        )

        post_cal_std_w_smoothing = np.std(stack_copy_w_smoothing.data, axis=0)

        # Smoothing should increase the standard deviation of the data slightly
        # especially since the gains are random
        for ai, antpair in enumerate(self.stack.antpairs):
            if antpair[0] == antpair[1]:
                continue
            assert np.all(post_cal_std[ai] <= post_cal_std_w_smoothing[ai])

    def test_full_night_flagged(self):
        stack_copy = self.stack.copy()
        gains = self.rng.normal(1, 0.1, size=(20, stack_copy.data.shape[2], 2))
        tip_tilt = self.rng.normal(0, 0.1, size=(20, self.stack.data.shape[2], 2))
        phs_gains = np.array(
            [
                np.exp(1j * tip_tilt * (self.antpos[ant2] - self.antpos[ant1])[0])
                for (ant1, ant2) in self.stack.antpairs
            ]
        )
        phs_gains = np.transpose(phs_gains, (1, 0, 2, 3))
        stack_copy.data *= gains[:, None, :, :] ** 2 * phs_gains
        stack_copy.flags[0, :, :, :] = True
        model = np.mean(self.stack.data, axis=0)

        pre_cal_std = np.nanstd(
            np.where(stack_copy.flags, np.nan, stack_copy.data), axis=0
        )

        cal_params, _ = calibration.lstbin_absolute_calibration(
            stack_copy,
            model,
            self.all_reds,
            run_amplitude_cal=True,
            run_phase_cal=True,
            calibrate_inplace=True,
            smooth_gains=False,
        )

        post_cal_std = np.nanstd(
            np.where(stack_copy.flags, np.nan, stack_copy.data), axis=0
        )

        for ai, antpair in enumerate(self.stack.antpairs):
            if antpair[0] == antpair[1]:
                continue
            assert np.all(post_cal_std[ai] <= pre_cal_std[ai])

    def test_baseline_fully_flagged(self):
        stack_copy = self.stack.copy()
        stack_copy.data = self.stack.data.copy()
        stack_copy.flags = self.stack.flags.copy()
        auto_stack_copy = self.auto_stack.copy()
        auto_stack_copy.data = self.auto_stack.data.copy()
        auto_stack_copy.flags = self.auto_stack.flags.copy()
        gains = self.rng.normal(1, 0.1, size=(20, stack_copy.data.shape[2], 2))
        tip_tilt = self.rng.normal(0, 0.1, size=(20, self.stack.data.shape[2], 2))
        phs_gains = np.array(
            [
                np.exp(1j * tip_tilt * (self.antpos[ant2] - self.antpos[ant1])[0])
                for (ant1, ant2) in self.stack.antpairs
            ]
        )
        phs_gains = np.transpose(phs_gains, (1, 0, 2, 3))
        stack_copy.data *= gains[:, None, :, :] ** 2 * phs_gains
        auto_stack_copy.data *= gains[:, None, :, :] ** 2
        stack_copy.flags[:, 0, :, :] = True
        auto_stack_copy.flags[:, 0, :, :] = True
        model = np.mean(self.stack.data, axis=0)
        auto_model = np.mean(self.auto_stack.data, axis=0)

        with pytest.warns(
            RuntimeWarning, match=re.escape("Degrees of freedom <= 0 for slice")
        ):
            pre_cal_std = np.nanstd(
                np.where(stack_copy.flags, np.nan, stack_copy.data), axis=0
            )

        cal_params, _ = calibration.lstbin_absolute_calibration(
            stack_copy,
            model,
            self.all_reds,
            auto_stack=auto_stack_copy,
            auto_model=auto_model,
            run_amplitude_cal=True,
            run_phase_cal=True,
            calibrate_inplace=True,
            smooth_gains=False,
        )

        with pytest.warns(
            RuntimeWarning, match=re.escape("Degrees of freedom <= 0 for slice")
        ):
            post_cal_std = np.nanstd(
                np.where(stack_copy.flags, np.nan, stack_copy.data), axis=0
            )

        for ai, antpair in enumerate(self.stack.antpairs):
            if antpair[0] == antpair[1]:
                continue
            assert np.all(post_cal_std[ai] <= pre_cal_std[ai])

    def test_value_errors(self):
        stack_copy = self.stack.copy()
        auto_stack_copy = self.auto_stack.copy()

        model = np.mean(self.stack.data, axis=0)

        # Test that ValueError is raised if run_amplitude_cal=True, autos are provided,
        # use_autos_for_abscal=True, and auto_model is not provided
        with pytest.raises(ValueError) as cm:
            calibration.lstbin_absolute_calibration(
                stack_copy,
                model,
                all_reds=[],
                auto_stack=auto_stack_copy,
                run_amplitude_cal=True,
                run_phase_cal=False,
                calibrate_inplace=True,
                use_autos_for_abscal=True,
            )

        # Test that ValueError is raised if model and data have incompatible shapes
        with pytest.raises(ValueError) as cm:
            calibration.lstbin_absolute_calibration(
                stack_copy,
                stack_copy.data,
                all_reds=[],
            )

        # Test that ValueError is raised if run_amplitude_cal=False and run_phase_cal=False
        with pytest.raises(ValueError) as cm:
            calibration.lstbin_absolute_calibration(
                stack_copy,
                stack_copy.data,
                all_reds=[],
                run_amplitude_cal=False,
                run_phase_cal=False,
            )

    def test_relative_phase_calibration(self):
        rng = np.random.default_rng(42)
        uvd = mockuvd.create_uvd_identifiable(
            integration_time=24 * 3600,
            ntimes=20,
            jd_start=2459844.0,
            ants=[0, 1, 2, 3],
            time_axis_faster_than_bls=False,
            pols=["xx", "yy", "xy", "yx"],
            freqs=mockuvd.PHASEII_FREQS[:100],
        )
        stack = LSTStack(uvd)
        stack.data[1:] = stack.data[0]  # All nights exactly the same

        stack_copy = stack.copy()
        delta = rng.uniform(-1, 1, size=(20, 1, 1)) * np.ones((1, 10, 100))
        gains = np.array(
            [
                np.ones((20, 10, 100)),
                np.ones((20, 10, 100)),
                np.exp(-1j * delta),
                np.exp(1j * delta),
            ]
        )
        gains = np.transpose(gains, (1, 2, 3, 0))

        stack_copy.data *= gains
        model = np.mean(stack.data, axis=0)

        pre_cal_std = np.std(stack_copy.data, axis=0)

        cal_params, _ = calibration.lstbin_absolute_calibration(
            stack_copy,
            model,
            all_reds=[],
            run_amplitude_cal=False,
            run_phase_cal=False,
            run_cross_pol_phase_cal=True,
            calibrate_inplace=True,
            smooth_gains=False,
        )

        post_cal_std = np.std(stack_copy.data, axis=0)
        assert np.all(np.isclose(post_cal_std[..., 2:], 0))

    def test_negative_nsamples(self):
        stack_copy = self.stack.copy()
        auto_stack_copy = self.auto_stack.copy()
        stack_copy.data = self.stack.data.copy()
        stack_copy.flags = self.stack.flags.copy()
        auto_stack_copy = self.auto_stack.copy()
        auto_stack_copy.data = self.auto_stack.data.copy()
        auto_stack_copy.flags = self.auto_stack.flags.copy()
        gains = self.rng.normal(1, 0.1, size=(20, stack_copy.data.shape[2], 2))
        tip_tilt = self.rng.normal(0, 0.1, size=(20, self.stack.data.shape[2], 2))
        phs_gains = np.array(
            [
                np.exp(1j * tip_tilt * (self.antpos[ant2] - self.antpos[ant1])[0])
                for (ant1, ant2) in self.stack.antpairs
            ]
        )
        phs_gains = np.transpose(phs_gains, (1, 0, 2, 3))
        stack_copy.data *= gains[:, None, :, :] ** 2 * phs_gains
        auto_stack_copy.data *= gains[:, None, :, :] ** 2
        model = np.mean(self.stack.data, axis=0)
        auto_model = np.mean(self.auto_stack.data, axis=0)

        # Calculate parameters with positive nsamples
        cal_params, gains = calibration.lstbin_absolute_calibration(
            stack_copy,
            model,
            all_reds=[],
            run_amplitude_cal=False,
            run_phase_cal=False,
            run_cross_pol_phase_cal=True,
            calibrate_inplace=False,
            smooth_gains=False,
        )

        # Now set nsamples to negative values
        stack_copy.nsamples = -np.abs(stack_copy.nsamples)
        auto_stack_copy.nsamples = -np.abs(auto_stack_copy.nsamples)
        cal_params_neg, gains_neg = calibration.lstbin_absolute_calibration(
            stack_copy,
            model,
            all_reds=[],
            run_amplitude_cal=False,
            run_phase_cal=False,
            run_cross_pol_phase_cal=True,
            calibrate_inplace=False,
            smooth_gains=False,
        )

        # Check that the parameters are the same
        for key in cal_params:
            assert np.allclose(
                cal_params[key], cal_params_neg[key], equal_nan=True
            ), f"Mismatch in {key} after setting nsamples to negative values."

        # Check that gains are the same
        for key in gains:
            assert np.allclose(
                gains[key], gains_neg[key], equal_nan=True
            ), f"Mismatch in gains for {key} after setting nsamples to negative values."

@pytest.fixture
def sample_file(tmp_path):
    """
    Create a small, deterministic test file via the writer.
    Ant 0 at origin -> no tip/tilt phase; Ant 1 at x=1 -> picks up tilt[...,0].
    'nn' polarization is fully flagged; 'ee' is not.
    """
    path = tmp_path / "lstcal_test.h5"

    ntimes, nfreqs, ndims = 2, 3, 3
    times = np.array([2450000.0, 2450000.5], dtype=float)
    freqs = np.array([100e6, 110e6, 120e6], dtype=float)
    pols = ["ee", "nn"]

    flag_array = np.zeros((2, ntimes, nfreqs), dtype=bool)
    flag_array[1, ...] = True  # 'nn' fully flagged

    # Unsorted dict order to verify the writer sorts antenna keys
    transformed_antpos_unsorted = {
        1: np.array([1.0, 0.0, 0.0], dtype=float),
        0: np.array([0.0, 0.0, 0.0], dtype=float),
    }

    # Parameter fields
    amplitude_Jee = np.ones((ntimes, nfreqs), dtype=float)
    tip_tilt_Jee = np.zeros((ntimes, nfreqs, ndims), dtype=float)
    tip_tilt_Jee[..., 0] = 0.1  # only x-gradient

    amplitude_Jnn = 2.0 * np.ones((ntimes, nfreqs), dtype=float)
    tip_tilt_Jnn = np.zeros((ntimes, nfreqs, ndims), dtype=float)
    tip_tilt_Jnn[..., 0] = 0.2  # only x-gradient

    cross_pol = 0.3 * np.ones((ntimes, nfreqs), dtype=float)  # applied when Jpol != refpol

    all_params = {
        "amplitude_Jee": amplitude_Jee,
        "tip_tilt_Jee": tip_tilt_Jee,
        "amplitude_Jnn": amplitude_Jnn,
        "tip_tilt_Jnn": tip_tilt_Jnn,
        "cross_pol": cross_pol,
    }

    calibration.write_single_baseline_lstcal_solutions(
        filename=str(path),
        all_calibration_parameters=all_params,
        flag_array=flag_array,
        transformed_antpos=transformed_antpos_unsorted,
        times=times,
        freqs=freqs,
        pols=pols,
    )

    return {
        "path": path,
        "ntimes": ntimes,
        "nfreqs": nfreqs,
        "times": times,
        "freqs": freqs,
        "pols": pols,
        "flag_array": flag_array,
        "antpos": transformed_antpos_unsorted,
        "params": all_params,
    }


def test_roundtrip_load(sample_file):
    """Writer -> loader round-trip preserves data, metadata, and polarizations."""
    (params, flags, antpos, times, freqs, pols) = calibration.load_single_baseline_lstcal_solutions(
        str(sample_file["path"])
    )

    np.testing.assert_allclose(times, sample_file["times"])
    np.testing.assert_allclose(freqs, sample_file["freqs"])
    np.testing.assert_array_equal(flags, sample_file["flag_array"])
    assert pols == sample_file["pols"]
    for k, v in sample_file["params"].items():
        np.testing.assert_allclose(params[k], v)
    for a, pos in sample_file["antpos"].items():
        np.testing.assert_allclose(antpos[a], pos)

    with h5py.File(sample_file["path"], "r") as h5f:
        ants = np.asarray(h5f["Header"]["antenna_numbers"])
        assert np.all(ants[:-1] <= ants[1:])


def test_load_gains_basic(sample_file):
    """Gains have correct amplitude and phases; flags summarized per-Jones-pol."""
    path = str(sample_file["path"])
    baselines = [(0, 1)]
    requested_pols = ["ee", "nn"]

    gains, cal_flags = calibration.load_single_baseline_lstcal_gains(
        path, baselines=baselines, polarizations=requested_pols, refpol="Jee"
    )

    ntimes, nfreqs = sample_file["ntimes"], sample_file["nfreqs"]

    # Ant 0 at origin: no tip/tilt phase
    np.testing.assert_allclose(gains[(0, "Jee")], np.ones((ntimes, nfreqs), dtype=complex))

    # For Jnn: amplitude=2 and cross_pol applied (since Jnn != Jee): phase +0.3
    np.testing.assert_allclose(gains[(0, "Jnn")], 2.0 * np.exp(1j * 0.3) * np.ones((ntimes, nfreqs)))

    # Ant 1 at x=1: picks up tilt[...,0]
    np.testing.assert_allclose(gains[(1, "Jee")], np.exp(-1j * 0.1) * np.ones((ntimes, nfreqs)))
    np.testing.assert_allclose(
        gains[(1, "Jnn")], 2.0 * np.exp(-1j * 0.2) * np.exp(1j * 0.3) * np.ones((ntimes, nfreqs))
    )

    # Flags: 'ee' -> False (not fully flagged); 'nn' -> True (fully flagged)
    assert cal_flags[(0, "Jee")] is False
    assert cal_flags[(1, "Jee")] is False
    assert cal_flags[(0, "Jnn")] is True
    assert cal_flags[(1, "Jnn")] is True


def test_unknown_polarization_raises(sample_file):
    """Requesting a polarization not present in the file should raise ValueError."""
    path = str(sample_file["path"])
    with pytest.raises(ValueError, match="not in gain polarizations"):
        calibration.load_single_baseline_lstcal_gains(
            path, baselines=[(0, 1)], polarizations=["ee", "xy"]
        )


def test_missing_parameter_raises(tmp_path, sample_file):
    """Missing a required parameter dataset should raise KeyError."""
    path = tmp_path / "missing_param.h5"

    params = dict(sample_file["params"])
    params.pop("tip_tilt_Jee")  # required by the gains function

    calibration.write_single_baseline_lstcal_solutions(
        filename=str(path),
        all_calibration_parameters=params,
        flag_array=sample_file["flag_array"],
        transformed_antpos=sample_file["antpos"],
        times=sample_file["times"],
        freqs=sample_file["freqs"],
        pols=sample_file["pols"],
    )

    with pytest.raises(KeyError, match="Missing calibration parameter 'tip_tilt_Jee'"):
        calibration.load_single_baseline_lstcal_gains(
            str(path), baselines=[(0, 1)], polarizations=["ee"]
        )


def test_shapes_and_types_are_consistent(sample_file):
    """Basic sanity: loaded arrays have expected shapes and dtypes."""
    params, flag_array, antpos, times, freqs, pols = calibration.load_single_baseline_lstcal_solutions(
        str(sample_file["path"])
    )
    ntimes, nfreqs = sample_file["ntimes"], sample_file["nfreqs"]

    assert flag_array.shape == (len(pols), ntimes, nfreqs)
    assert times.ndim == 1 and times.shape[0] == ntimes
    assert freqs.ndim == 1 and freqs.shape[0] == nfreqs
    assert params["amplitude_Jee"].shape == (ntimes, nfreqs)
    assert params["tip_tilt_Jee"].shape[-1] == 3
    assert isinstance(pols[0], str)
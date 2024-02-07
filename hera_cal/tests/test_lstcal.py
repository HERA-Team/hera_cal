import pytest
import numpy as np
from copy import deepcopy
from hera_sim.antpos import hex_array

from .. import lstcal, redcal, apply_cal


class TestCalFuncs:
    """ """

    def setup_method(self):
        np.random.seed(0)
        self.nf = 100
        self.ndays = 10
        self.freqs = np.linspace(50e6, 250e6, self.nf)
        self.antpos = hex_array(4, split_core=False, outriggers=0)
        self.reds = redcal.get_reds(self.antpos)
        self.idealized_antpos = redcal.reds_to_antpos(self.reds)
        self.true_vis = {
            k[0]: np.exp(1j * self.freqs / 25e6 + 1j * np.random.uniform(0, 2 * np.pi))[
                None
            ]
            for k in self.reds
        }

        # Generate a set of data, flags, and nsamples
        self.sim_data = {
            k: self.true_vis[k] * np.ones((self.ndays, 1)) for k in self.true_vis
        }
        self.flags = {
            k: np.zeros_like(self.sim_data[k], dtype=bool) for k in self.sim_data
        }
        self.nsamples = {
            k: np.ones_like(self.sim_data[k], dtype=float) for k in self.sim_data
        }
        self.baselines = [bl[:2] for bl in list(self.sim_data.keys())]

    def test_amplitude_calibration(self):
        # Generate a set of degenerate gains
        amp = np.random.uniform(0.9, 1.1, size=(self.ndays, self.nf))
        degen_gains = {(k, "Jnn"): amp for k in self.antpos}

        # Copy the simulation data and apply the degenerate gains
        data = deepcopy(self.sim_data)
        apply_cal.calibrate_in_place(data, degen_gains, gain_convention="multiply")

        # Pack dictionary data into arrays
        data_arr = np.transpose([data[k] for k in data], (1, 0, 2))[..., None]
        flag_arr = np.transpose([self.flags[k] for k in data], (1, 0, 2))[..., None]
        nsamples_arr = np.transpose([self.nsamples[k] for k in data], (1, 0, 2))[
            ..., None
        ]

        # Run amplitude calibration
        gain_dict, _ = lstcal.amplitude_calibration(
            data_arr, flag_arr, nsamples_arr, self.baselines, pols=["nn"]
        )
        lstcal.apply_lstcal_inplace(data_arr, gain_dict, self.baselines, ["nn"])

        # Check that day to day variation in visibilities is small
        assert np.allclose(np.std(data_arr, axis=0), 0)

    def test_delay_slope_calibration(self):
        delay_slope = np.random.uniform(
            -1e-8, 1e-8, size=(self.ndays, len(self.idealized_antpos[0]))
        )
        degen_gains = {
            (k, "Jnn"): np.exp(
                1j * delay_slope.dot(self.idealized_antpos[k])[..., None] * self.freqs
            )
            for k in self.antpos
        }

        # Copy the simulation data and apply the degenerate gains
        data = deepcopy(self.sim_data)
        apply_cal.calibrate_in_place(data, degen_gains, gain_convention="multiply")

        # Pack dictionary data into arrays
        data_arr = np.transpose([data[k] for k in data], (1, 0, 2))[..., None]
        flag_arr = np.transpose([self.flags[k] for k in data], (1, 0, 2))[..., None]
        nsamples_arr = np.transpose([self.nsamples[k] for k in data], (1, 0, 2))[
            ..., None
        ]

        # Run delay slope calibration
        max_phs_iter = 3
        conv_crit = 1e-10
        for i in range(max_phs_iter):
            gains, _ = lstcal.delay_slope_calibration(
                data_arr,
                flag_arr,
                nsamples_arr,
                self.freqs,
                self.baselines,
                self.idealized_antpos,
                ["nn"],
            )
            lstcal.apply_lstcal_inplace(data_arr, gains, self.baselines, ["nn"])
            if (
                np.median(np.linalg.norm([gains[k] - 1 for k in gains], axis=0))
                < conv_crit
            ):
                break

        # Check that day to day variation in visibilities is small
        assert np.allclose(np.std(data_arr, axis=0), 0)

    def test_tip_tilt_calibration(self):
        tip_tilt = np.random.uniform(
            0, 0.01, size=(self.ndays, self.nf, len(self.idealized_antpos[0]))
        )
        degen_gains = {
            (k, "Jnn"): np.exp(1j * tip_tilt.dot(self.idealized_antpos[k]))
            for k in self.antpos
        }

        # Copy the simulation data and apply the degenerate gains
        data = deepcopy(self.sim_data)
        apply_cal.calibrate_in_place(data, degen_gains, gain_convention="multiply")

        # Pack dictionary data into arrays
        data_arr = np.transpose([data[k] for k in data], (1, 0, 2))[..., None]
        flag_arr = np.transpose([self.flags[k] for k in data], (1, 0, 2))[..., None]
        nsamples_arr = np.transpose([self.nsamples[k] for k in data], (1, 0, 2))[
            ..., None
        ]

        gains, _ = lstcal.tip_tilt_calibration(
            data_arr,
            flag_arr,
            nsamples_arr,
            self.baselines,
            self.idealized_antpos,
            ["nn"],
        )
        lstcal.apply_lstcal_inplace(data_arr, gains, self.baselines, ["nn"])

        # Check that day to day variation in visibilities is small
        assert np.allclose(np.std(data_arr, axis=0), 0)

    def test_global_phase_slope_calibration(self):
        slope = np.random.uniform(
            -1e-1, 1e-1, size=(self.ndays, len(self.idealized_antpos[0]))
        )
        degen_gains = {
            (k, "Jnn"): np.exp(1j * slope.dot(self.idealized_antpos[k])[..., None])
            for k in self.antpos
        }

        # Copy the simulation data and apply the degenerate gains
        data = deepcopy(self.sim_data)
        apply_cal.calibrate_in_place(data, degen_gains, gain_convention="multiply")

        # Pack dictionary data into arrays
        data_arr = np.transpose([data[k] for k in data], (1, 0, 2))[..., None]
        flag_arr = np.transpose([self.flags[k] for k in data], (1, 0, 2))[..., None]
        nsamples_arr = np.transpose([self.nsamples[k] for k in data], (1, 0, 2))[
            ..., None
        ]

        gains, _ = lstcal.global_phase_slope_calibration(
            data_arr,
            flag_arr,
            nsamples_arr,
            self.baselines,
            self.idealized_antpos,
            ["nn"],
        )
        lstcal.apply_lstcal_inplace(data_arr, gains, self.baselines, ["nn"])

        # Check that day to day variation in visibilities is small
        assert np.allclose(np.std(data_arr, axis=0), 0)

    def test_calibrate_data(self):
        """ """
        # Copy the simulation data and apply the degenerate gains
        data = deepcopy(self.sim_data)
    
        # Pack dictionary data into arrays
        data_arr = np.transpose([data[k] for k in data], (1, 0, 2))[..., None]
        flag_arr = np.transpose([self.flags[k] for k in data], (1, 0, 2))[..., None]
        nsamples_arr = np.transpose([self.nsamples[k] for k in data], (1, 0, 2))[
            ..., None
        ]

        # Simulate and apply the gains
        amplitude = np.random.normal(1, 0.01, size=(self.ndays, self.nf))
        sim_gains = {(k, 'Jnn'): amplitude for k in self.antpos}
        lstcal.apply_lstcal_inplace(data_arr, sim_gains, self.baselines, ["nn"], gain_convention="multiply")

        # Run LST-calibration - calibration happens in place
        gains = lstcal.calibrate_data(
            data_arr,
            flag_arr,
            nsamples_arr,
            freqs=self.freqs,
            antpairs=self.baselines,
            idealized_antpos=self.idealized_antpos,
            pols=["nn"],
        )

        # Check that day to day variation in visibilities is small
        assert np.allclose(np.std(data_arr, axis=0), 0)

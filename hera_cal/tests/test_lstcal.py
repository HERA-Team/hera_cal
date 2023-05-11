import pytest
import numpy as np
from copy import deepcopy
from hera_sim.antpos import hex_array

from .. import lstcal, redcal, apply_cal

class TestCalFuncs:
    """
    """
    def setup_method(self):
        np.random.seed(0)
        self.nf = 15
        self.ndays = 10
        self.freqs = np.linspace(50e6, 250e6, self.nf)
        self.antpos = hex_array(6, split_core=True, outriggers=0)
        self.reds = redcal.get_reds(self.antpos)
        self.idealized_antpos = redcal.reds_to_antpos(self.reds)
        self.true_vis = {k[0]: np.exp(1j * self.freqs / 25e6 + 1j * np.random.uniform(0, 2 * np.pi))[None] for k in self.reds}

        self.sim_data = {k: self.true_vis[k] * np.ones((self.ndays, 1)) for k in self.true_vis}
        self.flags = {k: np.zeros_like(self.sim_data[k], dtype=bool) for k in self.sim_data}
        self.nsamples = {k: np.ones_like(self.sim_data[k], dtype=float) for k in self.sim_data}
        self.baselines = [bl[:2] for bl in list(self.sim_data.keys())]
        
    def test_amplitude_calibration(self):
        # Generate a set of degenerate gains
        amp = np.random.uniform(0.9, 1.1, size=(self.ndays, self.nf))
        degen_gains = {(k, "Jnn"): amp for k in self.antpos}
        
        # Copy the simulation data and apply the degenerate gains
        data = deepcopy(self.sim_data)
        apply_cal.calibrate_in_place(data, degen_gains, gain_convention='multiply')

        # Pack dictionary data into arrays
        data_arr = np.transpose([data[k] for k in data], (1, 0, 2))[..., None]
        flag_arr = np.transpose([self.flags[k] for k in data], (1, 0, 2))[..., None]
        nsamples_arr = np.transpose([self.nsamples[k] for k in data], (1, 0, 2))[..., None]

        # Run amplitude calibration
        gain_dict, solutions = lstcal.amplitude_calibration(
            data_arr, flag_arr, nsamples_arr, self.freqs, self.baselines, pols=['nn']
        )
        lstcal.apply_lstcal_in_place(data_arr, gain_dict, self.baselines, ["nn"])

        # Check that day to day variation in visibilities is small
        assert np.allclose(np.std(data_arr, axis=0), 0)

    def test_delay_slope_calibration(self):
        delay_slope = np.random.uniform(-1e-8, 1e-8, size=(self.ndays, len(self.idealized_antpos[0])))
        degen_gains = {(k, "Jnn"): np.exp(1j * delay_slope.dot(self.idealized_antpos[k])[..., None] * self.freqs) for k in self.antpos}

        # Copy the simulation data and apply the degenerate gains
        data = deepcopy(self.sim_data)
        apply_cal.calibrate_in_place(data, degen_gains, gain_convention='multiply')

        # Pack dictionary data into arrays
        data_arr = np.transpose([data[k] for k in data], (1, 0, 2))[..., None]
        flag_arr = np.transpose([self.flags[k] for k in data], (1, 0, 2))[..., None]
        nsamples_arr = np.transpose([self.nsamples[k] for k in data], (1, 0, 2))[..., None]

        # Run delay slope calibration
        gains, solutions = lstcal.delay_slope_calibration(
            data_arr, flag_arr, nsamples_arr, self.freqs, self.baselines, self.idealized_antpos, ['nn']
        )
        lstcal.apply_lstcal_in_place(data_arr, gains, self.baselines, ['nn'])

        # Check that day to day variation in visibilities is small
        assert np.allclose(np.std(data_arr, axis=0), 0)
        

        

        
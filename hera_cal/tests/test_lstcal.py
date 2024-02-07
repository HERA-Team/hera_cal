import pytest
import numpy as np
from copy import deepcopy
from hera_sim.antpos import hex_array

from .. import lstcal, redcal, apply_cal, abscal


class TestCalFuncs:
    """ """

    def setup_method(self):
        np.random.seed(0)
        self.nf = 100
        self.ndays = 10
        self.freqs = np.linspace(50e6, 250e6, self.nf)
        self.antpos = hex_array(3, split_core=False, outriggers=0)
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
        # Generate a set of degenerate gains
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
        # TODO: also add phase slope and tip/tilt
        amplitude = np.random.normal(1, 0.01, size=(self.ndays, self.nf)).astype(complex)
        sim_gains = {(k, 'Jnn'): amplitude for k in self.antpos}
        # Generate a set of degenerate gains
        tip_tilt = np.random.uniform(
            0, 0.01, size=(self.ndays, self.nf, len(self.idealized_antpos[0]))
        )
        degen_gains = {
            (k, "Jnn"): np.exp(1j * tip_tilt.dot(self.idealized_antpos[k]))
            for k in self.antpos
        }
        for k in sim_gains:
            sim_gains[k] *= degen_gains[k]

        lstcal.apply_lstcal_inplace(data_arr, sim_gains, self.baselines, ["nn"], gain_convention="multiply")

        data_arr_copy = deepcopy(data_arr) 

        # Run LST-calibration - calibration happens in place
        _ = lstcal.calibrate_data(
            data_arr_copy,
            flag_arr,
            nsamples_arr,
            freqs=self.freqs,
            antpairs=self.baselines,
            idealized_antpos=self.idealized_antpos,
            pols=["nn"],
        )

        # Check that day to day variation in visibilities is small
        assert np.allclose(np.std(data_arr_copy, axis=0), 0)

        data_arr_copy = deepcopy(data_arr) 

        # Test when one day are flagged
        day_flags = np.zeros((self.ndays), dtype=bool)
        day_flags[0] = True

        # Run LST-calibration with day flags
        _ = lstcal.calibrate_data(
            data_arr_copy,
            flag_arr,
            nsamples_arr,
            freqs=self.freqs,
            antpairs=self.baselines,
            idealized_antpos=self.idealized_antpos,
            pols=["nn"],
            day_flags=day_flags
        )

        # Check that day to day variation in visibilities is small
        assert np.allclose(np.std(data_arr_copy[~day_flags], axis=0), 0)

        data_arr_copy = deepcopy(data_arr) 

        # Test when one baseline is flagged
        bls_flags = np.zeros(len(self.baselines), dtype=bool)
        bls_flags[0] = True

        # Run LST-calibration with day flags
        _ = lstcal.calibrate_data(
            data_arr_copy,
            flag_arr,
            nsamples_arr,
            freqs=self.freqs,
            antpairs=self.baselines,
            idealized_antpos=self.idealized_antpos,
            pols=["nn"],
            bls_flags=bls_flags
        )

        # Check that day to day variation in visibilities is small
        assert np.allclose(np.std(data_arr_copy[:, ~bls_flags, ...], axis=0), 0)




class TestCalFuncsMultipleAntpos:
    """ """

    def setup_method(self):
        np.random.seed(0)
        self.nf = 100
        self.ndays = 10
        self.freqs = np.linspace(50e6, 250e6, self.nf)
        antpos1 = hex_array(3, split_core=False, outriggers=0)
        antpos2 = hex_array(3, split_core=False, outriggers=0)
        self.antpos = {}
        for k in antpos1:
            self.antpos[k] = antpos1[k]
            self.antpos[k + len(antpos1)] = antpos2[k] + np.array([50, 0, 0])
        self.reds = redcal.get_reds(self.antpos)
        self.cal_flags = {(k, "Jnn"): False for k in self.antpos}

        bad_day = 0
        second_hex_ants = [k + len(antpos1) for k in antpos2]

        self.idealized_antpos = []
        for i in range(self.ndays):
            for ant in second_hex_ants:
                self.cal_flags[(ant, "Jnn")] = i == bad_day
                
            self.idealized_antpos.append(abscal._get_idealized_antpos(self.cal_flags, self.antpos, ['nn']))

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

        for k in self.flags:
            if k[0] in second_hex_ants or k[1] in second_hex_ants:
                self.flags[k][bad_day] = np.ones((self.nf), dtype=bool)

    def test_delay_slope_calibration(self):
        # Test when antpos is different day-to-day
        degen_gains = {(k, "Jnn"): [] for k in self.antpos}
        for antpos in self.idealized_antpos:
            delay_slope = np.random.uniform(
                    -1e-8, 1e-8, size=(len(antpos[0]))
            )
            for k in antpos:    
                degen_gains[(k, "Jnn")].append(
                    np.exp(1j * delay_slope.dot(antpos[k])[..., None] * self.freqs)
                )


        for k in degen_gains:
            degen_gains[k] = np.array(degen_gains[k])

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

        # Check that day to day variation in calibrated visibilities is small
        assert np.allclose(np.nanstd(np.where(flag_arr, np.nan, data_arr), axis=0), 0)

    def test_tip_tilt_calibration(self):
        # Test when antpos is different day-to-day
        degen_gains = {(k, "Jnn"): [] for k in self.antpos}
        for antpos in self.idealized_antpos:
            tip_tilt = np.random.uniform(
                0, 0.01, size=(self.nf, len(antpos[0]))
            )
            for k in antpos:    
                degen_gains[(k, "Jnn")].append(
                    np.exp(1j * tip_tilt.dot(antpos[k]))
                )


        for k in degen_gains:
            degen_gains[k] = np.array(degen_gains[k])

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

        # Check that day to day variation in calibrated visibilities is small
        assert np.allclose(np.nanstd(np.where(flag_arr, np.nan, data_arr), axis=0), 0)

    def test_global_phase_slope_calibration(self):
        # Compute gains for each set of idealized antpos
        degen_gains = {(k, "Jnn"): [] for k in self.antpos}
        for antpos in self.idealized_antpos:
            slope = np.random.uniform(
                    -1e-1, 1e-1, size=(len(antpos[0]))
            )
            for k in antpos:    
                degen_gains[(k, "Jnn")].append(np.exp(1j * slope.dot(antpos[k]))[..., None])

        for k in degen_gains:
            degen_gains[k] = np.array(degen_gains[k])

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
        assert np.allclose(np.nanstd(np.where(flag_arr, np.nan, data_arr), axis=0), 0)

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
        # TODO: also add phase slope and tip/tilt
        amplitude = np.random.normal(1, 0.01, size=(self.ndays, self.nf)).astype(complex)
        sim_gains = {(k, 'Jnn'): amplitude for k in self.antpos}
        # Generate a set of degenerate gains
        degen_gains = {(k, "Jnn"): [] for k in self.antpos}
        for antpos in self.idealized_antpos:
            slope = np.random.uniform(
                    -1e-1, 1e-1, size=(len(antpos[0]))
            )
            for k in antpos:    
                degen_gains[(k, "Jnn")].append(np.exp(1j * slope.dot(antpos[k]))[..., None])

        for k in degen_gains:
            degen_gains[k] = np.array(degen_gains[k])

        for k in sim_gains:
            sim_gains[k] *= degen_gains[k]

        lstcal.apply_lstcal_inplace(data_arr, sim_gains, self.baselines, ["nn"], gain_convention="multiply")

        # Run LST-calibration - calibration happens in place
        _ = lstcal.calibrate_data(
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

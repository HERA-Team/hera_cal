# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

from __future__ import print_function, division, absolute_import

import unittest
import nose.tools as nt
import numpy as np
from copy import deepcopy
import os
import sys
import shutil
from six.moves import zip
from scipy import stats
from matplotlib import pyplot as plt
from pyuvdata import UVCal, UVData

from hera_cal import io
from hera_cal import reflections
from hera_cal import datacontainer
from hera_cal.data import DATA_PATH


def create_simulated_data(amp=1e-2, delay=155, phase=2):
    # create a simulated dataset
    uvd = UVData()
    uvd.read_miriad(os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA'))

    sim_uvd = uvd.select(inplace=False, antenna_nums=[37, 38, 39])
    sim_uvd.flag_array[:] = False
    sim_uvd.nsample_array[:] = 1.0
    sim_uvd.integration_time[:] = 10.0
    freqs = np.unique(sim_uvd.freq_array)
    Nbls = len(np.unique(sim_uvd.baseline_array))

    def fringe(freqs, bl_len=15.0, theta=np.pi / 3):
        """ theta is radians from horizon, bl_len is meters """
        tau = bl_len * np.cos(theta) / 2.99e8
        return np.exp(-2j * np.pi * tau * freqs)

    def noise(n, sig):
        return stats.norm.rvs(0, sig / np.sqrt(2), n) + 1j * stats.norm.rvs(0, sig / np.sqrt(2), n)

    np.random.seed(0)
    s1 = np.ones((sim_uvd.Ntimes, sim_uvd.Nfreqs), dtype=np.complex128) * 100 * (freqs / 150e6)**-1
    theta = np.pi / 2 - np.pi / 6
    dnu = np.median(np.diff(freqs))

    # get antenna vectors
    antpos, ants = sim_uvd.get_ENU_antpos(center=True, pick_data_ants=True)
    antpos_d = dict(zip(ants, antpos))

    # get antenna signals and noise
    n = dict([(a, noise(s1.size, 1e-1).reshape(s1.shape)) for a in ants])

    # iterate over bls
    for i, bl in enumerate(np.unique(sim_uvd.baseline_array)):
        bl_inds = np.where(sim_uvd.baseline_array == bl)[0]
        antpair = sim_uvd.baseline_to_antnums(bl)

        # get signal
        bl_len = np.linalg.norm(antpos_d[antpair[0]] - antpos_d[antpair[1]])
        s2 = s1 * fringe(freqs, bl_len=bl_len, theta=theta)

        # get noise
        n1, n2 = n[antpair[0]], n[antpair[1]]

        # form signal chain quantities
        v1 = s1 + n1
        v2 = s2 + n2

        # add a reflection term s1 * eps_11
        e = s1 * amp * np.exp(2j * np.pi * freqs * delay * 1e-9 + phase * 1j)
        v1 += e
        v2 += e

        # form visibility
        V = v1 * v2.conj()
        sim_uvd.data_array[bl_inds, 0, :, 0] = V

    return sim_uvd


class Test_ReflectionFitter(unittest.TestCase):
    uvd = create_simulated_data(delay=150.0, phase=2.0, amp=2e-2)
    uvd.flag_array[:, :, 20:22, :] = True

    def test_load_data(self):
        RF = reflections.ReflectionFitter(self.uvd)
        nt.assert_equal(len(RF.data), 6)

        self.uvd.write_miriad("./ex")
        RF = reflections.ReflectionFitter("./ex", filetype='miriad')
        nt.assert_equal(len(RF.data), 6)
        shutil.rmtree("./ex")

    def test_delay_clean(self):
        RF = reflections.ReflectionFitter(self.uvd)
        RF.dly_clean_data(tol=1e-10, maxiter=5000, gain=0.1, skip_wgt=0.1, dly_cut=200.0, edgecut=5,
                          taper='tukey', alpha=0.1, timeavg=True, broadcast_flags=True, time_thresh=0.05,
                          overwrite=True, verbose=True)
        nt.assert_equal(len(RF.data), len(RF.clean_data))
        nt.assert_true(isinstance(RF.clean_data, datacontainer.DataContainer))
        nt.assert_equal(RF.clean_data[(37, 37, 'xx')].shape, (1, 54))
        nt.assert_equal(len(RF.clean_freqs), 54)

        # w/o egdecut
        RF.dly_clean_data(tol=1e-10, maxiter=5000, gain=0.1, skip_wgt=0.1, dly_cut=200.0, edgecut=0,
                          taper='tukey', alpha=0.1, timeavg=True, broadcast_flags=True, time_thresh=0.05,
                          overwrite=True, verbose=True)
        nt.assert_equal(len(RF.data), len(RF.clean_data))
        nt.assert_true(isinstance(RF.clean_data, datacontainer.DataContainer))
        nt.assert_equal(RF.clean_data[(37, 37, 'xx')].shape, (1, 64))
        nt.assert_equal(len(RF.clean_freqs), 64)

    def test_reflection_modeling(self):
        RF = reflections.ReflectionFitter(self.uvd)
        RF.dly_clean_data(tol=1e-12, maxiter=500, gain=1e-2, skip_wgt=0.1, dly_cut=200.0, edgecut=2,
                          taper='hanning', timeavg=True, broadcast_flags=True, time_thresh=0.05,
                          overwrite=True, verbose=True)
        RF.model_reflections((100, 200), taper='hanning', zero_pad=100, overwrite=True, fthin=1, verbose=True)
        nt.assert_true(np.isclose(np.ravel(list(RF.delays.values())), 150.0, atol=2e-1).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.amps.values())), 2e-2, atol=2e-3).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.phs.values())), 2.0, atol=2e-1).all())

        # now reverse delay range
        RF.model_reflections((-200, -100), taper='hanning', zero_pad=100, overwrite=True, fthin=1, verbose=True)
        nt.assert_true(np.isclose(np.ravel(list(RF.delays.values())), -150.0, atol=2e-1).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.amps.values())), 2e-2, atol=2e-3).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.phs.values())), 2 * np.pi - 2.0, atol=2e-1).all())

        # exceptions
        nt.assert_raises(ValueError, RF.model_reflections, (1000, 2000), taper='none', overwrite=True)
        nt.assert_raises(ValueError, RF.model_reflections, (100, 300), overwrite=False)
        nt.assert_raises(AssertionError, RF.model_reflections, (-100, 100), overwrite=True)

        # non-even Nfreqs
        RF = reflections.ReflectionFitter(self.uvd.select(frequencies=np.unique(self.uvd.freq_array)[1:], inplace=False))
        RF.dly_clean_data(tol=1e-12, maxiter=500, gain=1e-2, skip_wgt=0.1, dly_cut=200.0, edgecut=2,
                          taper='hanning', timeavg=True, broadcast_flags=True, time_thresh=0.05,
                          overwrite=True, verbose=True)
        RF.model_reflections((100, 200), taper='hanning', zero_pad=100, overwrite=True, fthin=1, verbose=True)
        nt.assert_true(np.isclose(np.ravel(list(RF.delays.values())), 150.0, atol=5e-1).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.amps.values())), 2e-2, atol=2e-3).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.phs.values())), 2.0, atol=5e-1).all())

    def test_write_reflections(self):
        RF = reflections.ReflectionFitter(self.uvd)
        RF.dly_clean_data(tol=1e-12, maxiter=500, gain=1e-2, skip_wgt=0.1, dly_cut=200.0, edgecut=5,
                          taper='hanning', timeavg=True, broadcast_flags=True, time_thresh=0.05,
                          overwrite=True, verbose=True)
        RF.model_reflections((100, 200), taper='hanning', zero_pad=100, overwrite=True, fthin=1, verbose=True)
        uvc = RF.write_reflections("./ex.calfits", overwrite=True)
        nt.assert_equal(uvc.Ntimes, 1)
        np.testing.assert_array_equal([37, 38, 39], uvc.ant_array)

        # test w/ input calfits
        uvc = RF.write_reflections("./ex.calfits", input_calfits="./ex.calfits", overwrite=True)
        RF.dly_clean_data(keys=[k for k in RF.data.keys() if k[0] == k[1]], tol=1e-10, maxiter=500,
                          gain=1e-1, skip_wgt=0.1, dly_cut=200.0, edgecut=5,
                          taper='hanning', timeavg=False, broadcast_flags=True, time_thresh=0.05,
                          overwrite=True, verbose=True)
        RF.model_reflections((100, 200), taper='hanning', zero_pad=100, overwrite=True, fthin=1, verbose=True)
        uvc = RF.write_reflections("./ex.calfits", input_calfits='./ex.calfits', overwrite=True, add_to_history='testing')
        nt.assert_true('testing' in uvc.history.replace('\n', '').replace(' ', ''))
        nt.assert_true('Thisfilewasproducedbythefunction' in uvc.history.replace('\n', '').replace(' ', ''))
        nt.assert_equal(uvc.Ntimes, 60)
        np.testing.assert_array_equal([37, 38, 39], uvc.ant_array)

        os.remove('./ex.calfits')


if __name__ == '__main__':
    unittest.main()

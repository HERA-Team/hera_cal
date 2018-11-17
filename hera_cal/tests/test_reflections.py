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


def simulate_reflections(camp=1e-2, cdelay=155, cphase=2, add_cable=True,
                         xamp=1e-2, xdelay=300, xphase=0, add_xtalk=False):
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

    def beam(theta):
        """ theta is radians from horizon """
        return np.exp(-((theta - np.pi / 2) / (np.pi / 6))**2)

    np.random.seed(0)
    s1 = np.ones((sim_uvd.Ntimes, sim_uvd.Nfreqs), dtype=np.complex128) * 100 * (freqs / 150e6)**-1
    theta = np.linspace(np.pi / 2 - np.pi / 10, np.pi / 2 + np.pi / 10, uvd.Ntimes)  # pointing
    dnu = np.median(np.diff(freqs))

    # get source fluxes and positions
    src_theta = stats.norm.rvs(0, np.pi / 6, 2)
    src_amp = stats.norm.rvs(1, 0.1, 2)

    # get antenna vectors
    antpos, ants = sim_uvd.get_ENU_antpos(center=True, pick_data_ants=True)
    antpos_d = dict(zip(ants, antpos))

    # get antenna signals and noise
    n = dict([(a, noise(s1.size, 1e-1).reshape(s1.shape)) for a in ants])

    # iterate over bls
    for i, bl in enumerate(np.unique(sim_uvd.baseline_array)):
        bl_inds = np.where(sim_uvd.baseline_array == bl)[0]
        antpair = sim_uvd.baseline_to_antnums(bl)

        # get point source signal
        bl_len = np.linalg.norm(antpos_d[antpair[0]] - antpos_d[antpair[1]])
        s2 = 0
        for src_t, src_a in zip(src_theta, src_amp):
            s2 += s1 * src_a * np.asarray([fringe(freqs, bl_len=bl_len, theta=th + src_t) * beam(th + src_t) for th in theta])

        # get noise
        n1, n2 = n[antpair[0]], n[antpair[1]]

        # form signal chain quantities
        v1 = s1 + n1
        v2 = s2 + n2

        # add a cable reflection term s1 * eps_11
        if add_cable:
            e = s1 * camp * np.exp(2j * np.pi * freqs * cdelay * 1e-9 + cphase * 1j)
            v1 += e
            v2 += e

        # form visibility
        V = v1 * v2.conj()
        sim_uvd.data_array[bl_inds, 0, :, 0] = V

    # add xtalk
    if add_xtalk:
        antpairs = sim_uvd.get_antpairs()
        for ap in antpairs:
            if ap[0] == ap[1]:
                continue
            blt_inds = sim_uvd.antpair2ind(ap, ordered=False)
            blt_inds1 = sim_uvd.antpair2ind((ap[0], ap[0]), ordered=False)
            blt_inds2 = sim_uvd.antpair2ind((ap[1], ap[1]), ordered=False)
            e = (xamp * np.exp(2j * np.pi * freqs * xdelay * 1e-9 + xphase * 1j))
            a1 = sim_uvd.data_array[blt_inds1] * e[:, None]
            a2 = sim_uvd.data_array[blt_inds2] * e[:, None]
            sim_uvd.data_array[blt_inds] += a1 + a2.conj()

    return sim_uvd


class Test_ReflectionFitter_Cables(unittest.TestCase):
    uvd = simulate_reflections(cdelay=150.0, cphase=2.0, camp=2e-1, add_cable=True,
                               xdelay=250.0, xphase=0, xamp=.5, add_xtalk=False)
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
                          taper='hann', alpha=0.1, timeavg=True, broadcast_flags=True, time_thresh=0.05,
                          overwrite=True, verbose=True)
        nt.assert_equal(len(RF.data), len(RF.clean_data))
        nt.assert_true(isinstance(RF.clean_data, datacontainer.DataContainer))
        nt.assert_equal(RF.clean_data[(37, 37, 'xx')].shape, (1, 64))
        nt.assert_equal(len(RF.clean_freqs), 64)
        # check it picks out cable reflection
        RF.fft_data(overwrite=True, taper='hann')
        bl = (38, 38, 'xx')
        nt.assert_almost_equal(RF.clean_dlys[np.argmax(np.abs(RF.dfft[bl][:, :25]))], -150.0)

        # w/o egdecut
        RF.dly_clean_data(tol=1e-10, maxiter=5000, gain=0.1, skip_wgt=0.1, dly_cut=200.0, edgecut=0,
                          taper='hann', alpha=0.1, timeavg=True, broadcast_flags=True, time_thresh=0.05,
                          overwrite=True, verbose=True, keys=[(38, 38, 'xx')])
        nt.assert_equal(len(RF.data), len(RF.clean_data))
        nt.assert_true(isinstance(RF.clean_data, datacontainer.DataContainer))
        nt.assert_equal(RF.clean_data[(37, 37, 'xx')].shape, (1, 64))
        nt.assert_equal(len(RF.clean_freqs), 64)
        # check it picks out cable reflection
        RF.fft_data(overwrite=True, taper='hann')
        bl = (38, 38, 'xx')
        nt.assert_almost_equal(RF.clean_dlys[np.argmax(np.abs(RF.dfft[bl][:, :25]))], -150.0)

    def test_model_auto_reflections(self):
        RF = reflections.ReflectionFitter(self.uvd)
        autokeys = [k for k in RF.data.keys() if k[0] == k[1]]
        edgecut = 0
        RF.dly_clean_data(tol=1e-12, maxiter=500, gain=1e-2, skip_wgt=0.1, dly_cut=200.0, edgecut=edgecut,
                          taper='hann', timeavg=True, broadcast_flags=True, time_thresh=0.05,
                          overwrite=True, verbose=True, keys=autokeys)
        RF.model_auto_reflections((100, 200), taper='hann', zero_pad=100, overwrite=True, fthin=1, edgecut=edgecut, verbose=True)
        nt.assert_true(np.isclose(np.ravel(list(RF.delays.values())), 150.0, atol=2e-1).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.amps.values())), 2e-1, atol=2e-1).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.phs.values())), 2.0, atol=2e-1).all())

        # now reverse delay range
        RF.model_auto_reflections((-200, -100), taper='hanning', zero_pad=100, overwrite=True, fthin=1, edgecut=edgecut, verbose=True)
        nt.assert_true(np.isclose(np.ravel(list(RF.delays.values())), -150.0, atol=2e-1).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.amps.values())), 2e-1, atol=2e-1).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.phs.values())), 2 * np.pi - 2.0, atol=2e-1).all())

        # try with a small edgecut: required precision is less b/c delay resolution is worse
        RF = reflections.ReflectionFitter(self.uvd)
        edgecut = 10
        RF.dly_clean_data(tol=1e-13, maxiter=500, gain=1e-2, skip_wgt=0.1, dly_cut=200.0, edgecut=edgecut,
                          taper='hann', timeavg=True, broadcast_flags=True, time_thresh=0.05,
                          overwrite=True, verbose=True, keys=autokeys)
        RF.model_auto_reflections((100, 200), taper='hann', zero_pad=100, overwrite=True, edgecut=edgecut, fthin=1, verbose=True)
        nt.assert_true(np.isclose(np.ravel(list(RF.delays.values())), 150.0, atol=4e-1).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.amps.values())), 2e-1, atol=4e-1).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.phs.values())), 2.0, atol=4e-1).all())

        # exceptions
        nt.assert_raises(ValueError, RF.model_auto_reflections, (1000, 2000), taper='none', overwrite=True, edgecut=edgecut)
        nt.assert_raises(ValueError, RF.model_auto_reflections, (100, 300), overwrite=False, edgecut=edgecut)
        nt.assert_raises(AssertionError, RF.model_auto_reflections, (-100, 100), overwrite=True, edgecut=edgecut)

        # non-even Nfreqs
        RF = reflections.ReflectionFitter(self.uvd.select(frequencies=np.unique(self.uvd.freq_array)[1:], inplace=False))
        edgecut = 0
        RF.dly_clean_data(tol=1e-12, maxiter=500, gain=1e-2, skip_wgt=0.1, dly_cut=200.0, edgecut=edgecut,
                          taper='hanning', timeavg=True, broadcast_flags=True, time_thresh=0.05,
                          overwrite=True, verbose=True, keys=autokeys)
        RF.model_auto_reflections((100, 200), taper='hanning', zero_pad=100, overwrite=True, fthin=1, edgecut=edgecut, verbose=True)
        nt.assert_true(np.isclose(np.ravel(list(RF.delays.values())), 150.0, atol=2e-1).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.amps.values())), 2e-2, atol=2e-1).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.phs.values())), 2.0, atol=2e-1).all())

    def test_write_auto_reflections(self):
        RF = reflections.ReflectionFitter(self.uvd)
        RF.dly_clean_data(tol=1e-12, maxiter=500, gain=1e-2, skip_wgt=0.1, dly_cut=200.0,
                          taper='hanning', timeavg=True, broadcast_flags=True, time_thresh=0.05,
                          overwrite=True, verbose=True)
        RF.model_auto_reflections((100, 200), taper='hanning', zero_pad=100, overwrite=True, fthin=1, verbose=True)
        uvc = RF.write_auto_reflections("./ex.calfits", overwrite=True)
        nt.assert_equal(uvc.Ntimes, 1)
        np.testing.assert_array_equal([37, 38, 39], uvc.ant_array)

        # test w/ input calfits
        uvc = RF.write_auto_reflections("./ex.calfits", input_calfits="./ex.calfits", overwrite=True)
        RF.dly_clean_data(keys=[k for k in RF.data.keys() if k[0] == k[1]], tol=1e-10, maxiter=500,
                          gain=1e-1, skip_wgt=0.1, dly_cut=200.0,
                          taper='hanning', timeavg=False, broadcast_flags=True, time_thresh=0.05,
                          overwrite=True, verbose=True)
        RF.model_auto_reflections((100, 200), taper='hanning', zero_pad=100, overwrite=True, fthin=1, verbose=True)
        uvc = RF.write_auto_reflections("./ex.calfits", input_calfits='./ex.calfits', overwrite=True)
        nt.assert_equal(uvc.Ntimes, 60)
        np.testing.assert_array_equal([37, 38, 39], uvc.ant_array)

        os.remove('./ex.calfits')


class Test_ReflectionFitter_XTalk(unittest.TestCase):
    uvd = simulate_reflections(cdelay=150.0, cphase=2.0, camp=2e-2, add_cable=False,
                               xdelay=250.0, xphase=0, xamp=.1, add_xtalk=True)
    uvd.flag_array[:, :, 20:21, :] = True

    def test_pca_functions(self):
        RF = reflections.ReflectionFitter(self.uvd)
        RF.dly_clean_data(tol=1e-10, maxiter=1000, gain=5e-2, skip_wgt=0.1, dly_cut=300.0,
                          taper='hann', timeavg=False, broadcast_flags=True, time_thresh=0.05,
                          overwrite=True, verbose=True, keys=[(37, 38, 'xx')])
        # fft data
        RF.fft_data(data=RF.clean_data, taper='tukey', alpha=0.2, overwrite=True)

        # test pca_decomposition
        RF.pca_decomp((180, 330), side='pos', overwrite=True)
        # test containers exist
        nt.assert_true(np.all([hasattr(RF, o) for o in ['umodes', 'vmodes', 'svals', 'uflags', 'pcomps', 'dfft']]))
        # test good information compression
        bl = (37, 38, 'xx')
        nt.assert_true(RF.svals[bl][0] / RF.svals[bl][1] > 40)

        # build a model
        RF.build_model(Nkeep=1, increment=False, overwrite=True)
        # assert its a good fit to the xtalk at 250 ns delay
        Vstd = np.std(RF.dfft[bl][:, 57].real)
        Rstd = np.std(RF.dfft[bl][:, 57].real - RF.pcomp_model[bl][:, 57].real)
        # says that residual is small compared to original array
        nt.assert_true(Rstd / Vstd < 0.01)

        # increment the model 
        RF.pca_decomp((180, 330), side='neg', overwrite=True)
        RF.build_model(Nkeep=1, increment=True)
        # says that the two are similar to each other, which they should be
        nt.assert_true(np.std(RF.dfft[bl][:, 57].real - RF.pcomp_model[bl][:, 7].real) / Vstd < .05)

        # overwrite the model
        RF.pca_decomp((180, 330), side='both', overwrite=True)
        RF.build_model(Nkeep=1, increment=False, overwrite=True)
        # says the residual is small compared to original array
        nt.assert_true(np.std(RF.dfft[bl][:, 57].real - RF.pcomp_model[bl][:, 7].real) / Vstd < .01)
        Rstd = np.std(RF.dfft[bl][:, 57].real - RF.pcomp_model[bl][:, 57].real)
        nt.assert_true(Rstd / Vstd < .01)

        # subtract the model from the data!
        RF.subtract_model(overwrite=True)

        # assert std of difference is smaller than original data, b/c it should be
        Vstd = np.std(RF.data[(37, 38, 'xx')])
        Rstd = np.std(RF.data[(37, 38, 'xx')] - RF.data_pc_sub[(37, 38, 'xx')])
        nt.assert_true(Rstd / Vstd < 0.2)

        # assert difference is near-mean zero
        nt.assert_true(np.abs(np.mean(RF.data[(37, 38, 'xx')] - RF.data_pc_sub[(37, 38, 'xx')])) < 1e-10)

    def test_misc_pca_funcs(self):
        RF = reflections.ReflectionFitter(self.uvd)
        # time average the data
        RF.timeavg_data(30, rephase=False)

        # delay clean as before, but use timeaveraged data
        RF.dly_clean_data(tol=1e-10, maxiter=1000, gain=5e-2, skip_wgt=0.1, dly_cut=300.0,
                          taper='hann', timeavg=False, broadcast_flags=True, time_thresh=0.05,
                          overwrite=True, verbose=True, keys=[(37, 37, 'xx'), (37, 38, 'xx'), (38, 38, 'xx')],
                          data=RF.avg_data, flags=RF.avg_flags, nsamples=RF.avg_nsamples)

        # fft data
        RF.fft_data(data=RF.clean_data, taper='tukey', alpha=0.2, overwrite=True)

        # pca decomp
        RF.pca_decomp((180, 330), side='both', overwrite=True)

        # test interpolation of umodes
        RF.interp_u(overwrite=True, mode='gpr', gp_len=100, gp_nl=0.01, optimizer=None)

        # assert broadcasting to full time resolution worked
        nt.assert_equal(len(RF.umode_interp[(37, 38, 'xx')]), 60)

        # assert that residual between interpolated and non-interpolated is small
        nt.assert_true(np.std(RF.umodes[(37, 38, 'xx')][:, 0] - RF.umode_interp[(37, 38, 'xx')][1::3, 0]) < 0.001)

        # project auto correlation onto umode and assert residual is small
        RF.project_autos_onto_u([(37, 38, 'xx')], [(37, 37, 'xx')], index=0, auto_delay=0, overwrite=True)
        nt.assert_true(np.std(RF.umodes[(37, 38, 'xx')][:, 0] - RF.umode_interp[(37, 38, 'xx')][:, 0]) < 0.0005)

        # test multiple auto projection
        RF.project_autos_onto_u([(37, 38, 'xx')], [[(37, 37, 'xx'), (38, 38, 'xx')]], index=0, auto_delay=0, overwrite=True)
        nt.assert_true(np.std(RF.umodes[(37, 38, 'xx')][:, 0] - RF.umode_interp[(37, 38, 'xx')][:, 0]) < 0.0005)

        # exceptions
        nt.assert_raises(ValueError, RF.interp_u, overwrite=True, mode='foo')


def test_gen_taper():
    for taper in ['none', 'blackmanharris', 'hann', 'tukey', 'blackman']:
        t = reflections._gen_taper(taper, 10)
        nt.assert_equal(t.ndim, 2)
        nt.assert_equal(t.shape, (1, 10))
    nt.assert_raises(ValueError, reflections._gen_taper, 'foo', 10)


if __name__ == '__main__':
    unittest.main()

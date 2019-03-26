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
from pyuvdata import UVCal, UVData
import operator
import functools

from hera_cal import io
from hera_cal import reflections
from hera_cal import datacontainer
from hera_cal.data import DATA_PATH
from hera_cal import apply_cal


def simulate_reflections(camp=1e-2, cdelay=155, cphase=2, add_cable=True, cable_ants=None,
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

    if cable_ants is None:
        cable_ants = sim_uvd.antenna_numbers

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
    s1 = np.sqrt(np.ones((sim_uvd.Ntimes, sim_uvd.Nfreqs), dtype=np.complex128) * 100 * (freqs / 150e6)**-1)
    theta = np.pi / 2  # pointing
    dnu = np.median(np.diff(freqs))

    # get source fluxes and positions
    src_theta = stats.norm.rvs(0, np.pi / 6, 2)
    src_amp = stats.norm.rvs(1, 0.1, 2)

    # get antenna vectors
    antpos, ants = sim_uvd.get_ENU_antpos(center=True, pick_data_ants=True)
    antpos_d = dict(zip(ants, antpos))
    ant_dist = dict(zip(ants, map(np.linalg.norm, antpos)))

    # get antenna signals and noise
    n = dict([(a, noise(s1.size, 1e-8).reshape(s1.shape)) for a in ants])  # set noise to essentially zero
    s = dict([(a, s1 * functools.reduce(operator.add, [src_a * fringe(freqs, bl_len=ant_dist[a], theta=theta + src_t) * beam(theta + src_t) for src_t, src_a in zip(src_theta, src_amp)])) for a in ants])

    # iterate over bls
    for i, bl in enumerate(np.unique(sim_uvd.baseline_array)):
        bl_inds = np.where(sim_uvd.baseline_array == bl)[0]
        antpair = sim_uvd.baseline_to_antnums(bl)

        # get point source signal
        s1, s2 = s[antpair[0]], s[antpair[1]]

        # get noise
        n1, n2 = n[antpair[0]], n[antpair[1]]

        # form signal chain quantities
        v1 = s1 + n1
        v2 = s2 + n2

        # add a cable reflection term s1 * eps_11
        if add_cable:
            if isinstance(cdelay, (float, np.float, int, np.int)):
                cdelay = [cdelay]
            if isinstance(camp, (float, np.float, int, np.int)):
                camp = [camp]
            if isinstance(cphase, (float, np.float, int, np.int)):
                cphase = [cphase]

            g = 1.0
            for ca, cd, cp in zip(camp, cdelay, cphase):
                g *= 1 + ca * np.exp(2j * np.pi * freqs * cd * 1e-9 + cp * 1j)
            if antpair[0] in cable_ants:
                v1 *= g
                #v1 += e
            if antpair[1] in cable_ants:
                v2 *= g
                #v2 += e

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
    uvd_clean = simulate_reflections(add_cable=False, add_xtalk=False)
    uvd = simulate_reflections(cdelay=155.0, cphase=2.0, camp=1e-2, add_cable=True, cable_ants=[37], add_xtalk=False)

    def test_model_auto_reflections(self):
        RF = reflections.ReflectionFitter(self.uvd)
        bl_k = (37, 37, 'xx')
        g_k = (37, 'Jxx')
        RF.model_auto_reflections(RF.data, (100, 200), keys=[bl_k], window='blackmanharris',
                                  zeropad=100, overwrite=True, fthin=1, verbose=True)
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_dly.values())), 155.0, atol=1e-1).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_amp.values())), 1e-2, atol=1e-4).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_phs.values())), 2.0, atol=1e-1).all())

        # try with a small edgecut
        RF = reflections.ReflectionFitter(self.uvd)
        edgecut = 5
        RF.model_auto_reflections(RF.data, (100, 200), keys=[bl_k], window='blackmanharris',
                                  zeropad=100, overwrite=True, fthin=1, verbose=True, edgecut_low=edgecut, edgecut_hi=edgecut)
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_dly.values())), 155.0, atol=1e-1).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_amp.values())), 1e-2, atol=1e-4).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_phs.values())), 2.0, atol=1e-1).all())

        # try a high ref_sig cut
        RF.model_auto_reflections(RF.data, (100, 200), keys=[bl_k], window='blackmanharris',
                                  ref_sig_cut=100, overwrite=True)

        # try filtering the visibilities
        RF.vis_clean(data=RF.data, ax='freq', min_dly=100, overwrite=True, window='blackmanharris', alpha=0.1, tol=1e-8, keys=[bl_k])
        RF.model_auto_reflections(RF.clean_resid, (100, 200), clean_data=RF.clean_data, keys=[bl_k],
                                  window='blackmanharris', zeropad=100, overwrite=True, fthin=1, verbose=True)
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_dly.values())), 155.0, atol=1e-1).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_amp.values())), 1e-2, atol=1e-4).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_phs.values())), 2.0, atol=1e-1).all())

        # try optimization on time-averaged data
        RF.timeavg_data(RF.data, RF.times, RF.lsts, 200, keys=None, overwrite=True)
        RF.model_auto_reflections(RF.avg_data, (100, 200), keys=[bl_k], window='blackmanharris',
                                  zeropad=100, overwrite=True, fthin=1, verbose=True)
        output = RF.refine_auto_reflections(RF.avg_data, (125, 175), RF.ref_amp, RF.ref_dly, RF.ref_phs,
                                            keys=[bl_k], window='blackmanharris', zeropad=100,
                                            maxiter=100, method='BFGS', tol=1e-5)
        ref_amp = output[0]
        ref_dly = output[1]
        ref_phs = output[2]
        # assert equivalence to higher precision
        nt.assert_true(np.isclose(np.ravel(list(ref_dly.values())), 155.0, atol=1e-2).all())
        nt.assert_true(np.isclose(np.ravel(list(ref_amp.values())), 1e-2, atol=1e-5).all())
        nt.assert_true(np.isclose(np.ravel(list(ref_phs.values())), 2.0, atol=1e-2).all())

        # now reverse delay range
        RF.model_auto_reflections(RF.avg_data, (-200, -100), keys=[bl_k], window='blackmanharris',
                                  zeropad=100, overwrite=True, fthin=1, verbose=True)
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_dly.values())), -155.0, atol=1e-1).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_amp.values())), 1e-2, atol=1e-4).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_phs.values())), 2 * np.pi - 2.0, atol=1e-1).all())

        output = RF.refine_auto_reflections(RF.avg_data, (-175, -125), RF.ref_amp, RF.ref_dly, RF.ref_phs,
                                            keys=[bl_k, (39, 39, 'xx')], window='blackmanharris', zeropad=100,
                                            maxiter=100, method='BFGS', tol=1e-5)
        ref_amp = output[0]
        ref_dly = output[1]
        ref_phs = output[2]
        # assert equivalence to higher precision
        nt.assert_true(np.isclose(np.ravel(list(ref_dly.values())), -155.0, atol=1e-2).all())
        nt.assert_true(np.isclose(np.ravel(list(ref_amp.values())), 1e-2, atol=1e-5).all())
        nt.assert_true(np.isclose(np.ravel(list(ref_phs.values())), 2 * np.pi - 2.0, atol=1e-2).all())

        # test flagged data
        _bl = (38, 38, 'xx')
        RF.model_auto_reflections(RF.avg_data, (-200, -100), keys=[_bl], window='blackmanharris',
                                  zeropad=100, overwrite=True, fthin=1, verbose=True)
        RF.avg_flags[_bl][:] = True
        output = RF.refine_auto_reflections(RF.avg_data, (-175, -125), RF.ref_amp, RF.ref_dly, RF.ref_phs,
                                            keys=[_bl], window='blackmanharris', zeropad=100, clean_flags=RF.avg_flags,
                                            maxiter=100, method='BFGS', tol=1e-5)
        nt.assert_false(output[3][(38, 'Jxx')].any())

        # non-even Nfreqs
        RF = reflections.ReflectionFitter(self.uvd.select(frequencies=np.unique(self.uvd.freq_array)[:-1], inplace=False))
        RF.model_auto_reflections(RF.data, (100, 200), keys=[bl_k], window='blackmanharris',
                                  zeropad=100, overwrite=True, fthin=1, verbose=True)
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_dly.values())), 155.0, atol=1e-1).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_amp.values())), 1e-2, atol=1e-4).all())
        nt.assert_true(np.isclose(np.ravel(list(RF.ref_phs.values())), 2.0, atol=1e-1).all())

        # exceptions
        nt.assert_raises(ValueError, RF.model_auto_reflections, RF.data, (1000, 2000), window='none', overwrite=True, edgecut_low=edgecut)

        # try clear
        RF.clear(exclude=['data'])
        nt.assert_equal(len(RF.ref_eps), 0)
        nt.assert_equal(len(RF.ref_gains), 0)
        nt.assert_true(len(RF.data) > 0)

        # try soft copy
        RF2 = RF.soft_copy()
        nt.assert_true(RF2.__class__, reflections.ReflectionFitter)

    def test_write_auto_reflections(self):
        RF = reflections.ReflectionFitter(self.uvd)
        bl_k = (37, 37, 'xx')
        RF.model_auto_reflections(RF.data, (100, 200), window='blackmanharris', zeropad=100, overwrite=True, fthin=1, verbose=True)
        uvc = RF.write_auto_reflections("./ex.calfits", overwrite=True)
        nt.assert_equal(uvc.Ntimes, 60)
        np.testing.assert_array_equal(len(uvc.ant_array), 47)
        nt.assert_true(np.isclose(uvc.gain_array[0], 1.0).all())
        nt.assert_false(np.isclose(uvc.gain_array[uvc.ant_array.tolist().index(37)], 1.0).all())

        # test w/ input calfits
        uvc = RF.write_auto_reflections("./ex.calfits", input_calfits="./ex.calfits", overwrite=True)
        RF.model_auto_reflections(RF.data, (100, 200), window='blackmanharris', zeropad=100, overwrite=True, fthin=1, verbose=True)
        uvc = RF.write_auto_reflections("./ex.calfits", input_calfits='./ex.calfits', overwrite=True)
        nt.assert_equal(uvc.Ntimes, 60)
        np.testing.assert_array_equal(len(uvc.ant_array), 47)

        # test data is corrected by taking ratio w/ clean data
        data = deepcopy(RF.data)
        g = reflections.form_gains(RF.ref_eps)
        apply_cal.calibrate_in_place(data, g, gain_convention='divide')
        r = data[bl_k] / self.uvd_clean.get_data(bl_k)
        nt.assert_true(np.abs(np.mean(r) - 1) < 1e-1)

        os.remove('./ex.calfits')

    def test_auto_reflection_argparser(self):
        sys.argv = [sys.argv[0], 'a', '--output_fname', 'ex.calfits', '--dly_ranges', '10,20', '10,20', '--overwrite']
        parser = reflections.auto_reflection_argparser()
        a = parser.parse_args()
        nt.assert_equal(a.clean_data[0], 'a')
        nt.assert_equal(a.output_fname, 'ex.calfits')
        nt.assert_equal(a.dly_ranges[0], '10,20')
        nt.assert_equal(len(a.dly_ranges), 2)

    def test_auto_reflection_run(self):
        # most of the code tests have been done above, this is just to ensure this wrapper function runs
        uvd = simulate_reflections(cdelay=[150.0, 250.0], cphase=[2.0, 2.0], camp=[1e-2, 1e-2], add_cable=True, cable_ants=[37], add_xtalk=False)
        reflections.auto_reflection_run(uvd, [(100, 200), (200, 300)], "./ex.calfits", time_avg=True, window='blackmanharris', write_npz=True, overwrite=True, ref_sig_cut=1.0)
        nt.assert_true(os.path.exists("./ex.calfits"))
        nt.assert_true(os.path.exists("./ex.npz"))

        # ensure gains have two humps at 150 and 250 ns
        uvc = UVCal()
        uvc.read_calfits('./ex.calfits')
        aind = np.argmin(np.abs(uvc.ant_array - 37))
        g = uvc.gain_array[aind, 0, :, :, 0].T
        delays = np.fft.fftfreq(uvc.Nfreqs, np.diff(uvc.freq_array[0])[0]) * 1e9
        gfft = np.mean(np.abs(np.fft.fft(g, axis=1)), axis=0)

        nt.assert_true(delays[np.argmax(gfft[(delays > 100) & (delays < 200)])], 150)
        nt.assert_true(delays[np.argmax(gfft[(delays > 200) & (delays < 300)])], 250)

        os.remove("./ex.calfits")
        os.remove("./ex.npz")



class Test_ReflectionFitter_XTalk(unittest.TestCase):
    uvd = simulate_reflections(add_cable=False, xdelay=250.0, xphase=0, xamp=.1, add_xtalk=True)

    def test_svd_functions(self):
        RF = reflections.ReflectionFitter(self.uvd)
        bl = (37, 38, 'xx')

        # fft data
        RF.fft_data(data=RF.data, window='blackmanharris', overwrite=True)

        # test sv_decomposition
        wgts = RF.svd_weights(RF.dfft, RF.delays, min_dly=200, max_dly=300, side='pos')
        RF.sv_decomp(RF.dfft, wgts=wgts, keys=[bl], overwrite=True)

        # build a model
        RF.build_pc_model(RF.umodes, RF.vmodes, RF.svals, Nkeep=1, increment=False, overwrite=True)

        # test containers exist
        nt.assert_true(np.all([hasattr(RF, o) for o in ['umodes', 'vmodes', 'svals', 'uflags', 'pcomp_model', 'dfft']]))
        # test good information compression
        nt.assert_true(RF.svals[bl][0] / RF.svals[bl][1] > 20)

        # assert its a good fit to the xtalk at 250 ns delay
        Vrms = np.sqrt(np.mean(RF.dfft[bl][:, 57].real**2))
        Rrms = np.sqrt(np.mean((RF.dfft[bl][:, 57].real - RF.pcomp_model[bl][:, 57].real)**2))
        # says that residual is small compared to original array
        nt.assert_true(Rrms / Vrms < 0.01)

        # increment the model 
        wgts = RF.svd_weights(RF.dfft, RF.delays, min_dly=200, max_dly=300, side='neg')
        RF.sv_decomp(RF.dfft, wgts=wgts, overwrite=True)
        RF.build_pc_model(RF.umodes, RF.vmodes, RF.svals, Nkeep=1, increment=True)
        # says that the two are similar to each other, which they should be
        Vrms = np.sqrt(np.mean(RF.dfft[bl][:, 7].real**2))
        nt.assert_true(np.sqrt(np.mean((RF.dfft[bl][:, 7].real - RF.pcomp_model[bl][:, 7].real)**2)) / Vrms < .01)

        # overwrite the model
        wgts = RF.svd_weights(RF.dfft, RF.delays, min_dly=200, max_dly=300, side='both')
        RF.sv_decomp(RF.dfft, wgts=wgts, overwrite=True)
        RF.build_pc_model(RF.umodes, RF.vmodes, RF.svals, Nkeep=2, increment=False, overwrite=True)
        # says the residual is small compared to original array
        Vrms = np.sqrt(np.mean(RF.dfft[bl][:, 57].real**2))
        Rrms = np.sqrt(np.mean((RF.dfft[bl][:, 57].real - RF.pcomp_model[bl][:, 57].real)**2))
        nt.assert_true(Rrms / Vrms < 0.01)

        # subtract the model from the data
        RF.subtract_model(RF.data, overwrite=True)
        nt.assert_equal(RF.pcomp_model_fft[bl].shape, (60, 64))
        nt.assert_equal(RF.data_pcmodel_resid[bl].shape, (60, 64))

    def test_misc_svd_funcs(self):
        RF = reflections.ReflectionFitter(self.uvd)
        bl = (37, 38, 'xx')
        abl1 = (37, 37, 'xx')
        abl2 = (38, 38, 'xx')
        # time average the data
        RF.timeavg_data(RF.data, RF.times, RF.lsts, 30, rephase=False)

        # fft data
        RF.fft_data(data=RF.avg_data, window='blackmanharris', overwrite=True)

        # sv decomp
        wgts = RF.svd_weights(RF.dfft, RF.delays, min_dly=200, max_dly=300, side='both')
        RF.sv_decomp(RF.dfft, wgts=wgts, keys=[bl, abl1, abl2], overwrite=True)

        # test interpolation of umodes
        RF.interp_u(RF.umodes, RF.avg_times, overwrite=True, mode='gpr', gp_frate=0.4, gp_nl=0.01, optimizer=None)
        nt.assert_true(RF.umode_interp[bl].shape, (20, 20))

        # assert broadcasting to full time resolution worked
        RF.interp_u(RF.umodes, RF.avg_times, full_times=RF.times, overwrite=True, mode='gpr', gp_frate=1.0, gp_nl=0.01, optimizer=None)
        nt.assert_true(RF.umode_interp[bl].shape, (60, 60))

        # exceptions
        nt.assert_raises(ValueError, RF.interp_u, RF.umodes, RF.times, overwrite=True, mode='foo')


if __name__ == '__main__':
    unittest.main()

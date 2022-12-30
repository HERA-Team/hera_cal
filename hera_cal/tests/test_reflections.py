# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import pytest
import numpy as np
from copy import deepcopy
import os
import sys
import shutil
from scipy import stats
from pyuvdata import UVCal, UVData
import operator
import functools
from sklearn.gaussian_process import kernels
import hera_sim as hs
import copy

from .. import apply_cal, datacontainer, io, reflections
from ..data import DATA_PATH


def simulate_reflections(uvd=None, camp=1e-2, cdelay=155, cphase=2, add_cable=True, cable_ants=None,
                         xamp=1e-2, xdelay=300, xphase=0, add_xtalk=False):
    # create a simulated dataset
    if uvd is None:
        uvd = UVData()
        uvd.read(os.path.join(DATA_PATH, 'PyGSM_Jy_downselect.uvh5'),
                 run_check_acceptability=False)
    else:
        if isinstance(uvd, str):
            _uvd = UVData()
            _uvd.read(uvd)
            uvd = _uvd
        elif isinstance(uvd, UVData):
            uvd = deepcopy(uvd)
    uvd.use_future_array_shapes()

    # TODO: use hera_sim.simulate.Simulator
    freqs = np.unique(uvd.freq_array)
    Nbls = len(np.unique(uvd.baseline_array))

    if cable_ants is None:
        cable_ants = uvd.antenna_numbers

    def noise(n, sig):
        return stats.norm.rvs(0, sig / np.sqrt(2), n) + 1j * stats.norm.rvs(0, sig / np.sqrt(2), n)

    np.random.seed(0)

    # get antenna vectors
    antpos, ants = uvd.get_ENU_antpos(center=True, pick_data_ants=True)
    antpos_d = dict(zip(ants, antpos))
    ant_dist = dict(zip(ants, map(np.linalg.norm, antpos)))

    # get autocorr
    autocorr = uvd.get_data(23, 23, 'ee')

    # form cable gains
    if add_cable:
        if isinstance(cdelay, (float, np.floating, int, np.integer)):
            cdelay = [cdelay]
        if isinstance(camp, (float, np.floating, int, np.integer)):
            camp = [camp]
        if isinstance(cphase, (float, np.floating, int, np.integer)):
            cphase = [cphase]

        cable_gains = dict([(k, np.ones((uvd.Ntimes, uvd.Nfreqs), dtype=complex)) for k in uvd.antenna_numbers])

        for ca, cd, cp in zip(camp, cdelay, cphase):
            cg = hs.sigchain.gen_reflection_gains(freqs / 1e9, cable_ants, amp=[ca for a in cable_ants],
                                                  dly=[cd for a in cable_ants], phs=[cp for a in cable_ants])
            for k in cg:
                cable_gains[k] *= cg[k]

    # iterate over bls
    for i, bl in enumerate(np.unique(uvd.baseline_array)):
        bl_inds = np.where(uvd.baseline_array == bl)[0]
        antpair = uvd.baseline_to_antnums(bl)

        # add xtalk
        if add_xtalk:
            if antpair[0] != antpair[1]:
                # add xtalk to both pos and neg delays
                xt = hs.sigchain.gen_cross_coupling_xtalk(freqs / 1e9, autocorr, amp=xamp, dly=xdelay, phs=xphase)
                xt += hs.sigchain.gen_cross_coupling_xtalk(freqs / 1e9, autocorr, amp=xamp, dly=xdelay, phs=xphase, conj=True)
                uvd.data_array[bl_inds] += xt[:, :, None]

        # add a cable reflection term eps_11
        if add_cable:
            gain = cable_gains[antpair[0]] * np.conj(cable_gains[antpair[1]])
            uvd.data_array[bl_inds] *= gain[:, :, None]

    # get fourier modes
    uvd.frates = np.fft.fftshift(np.fft.fftfreq(uvd.Ntimes, np.diff(np.unique(uvd.time_array))[0] * 24 * 3600)) * 1e3
    uvd.delays = np.fft.fftshift(np.fft.fftfreq(uvd.Nfreqs, uvd.channel_width)) * 1e9

    return uvd


@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
@pytest.mark.filterwarnings("ignore:.*dspec.vis_filter will soon be deprecated")
class Test_ReflectionFitter_Cables(object):
    uvd_clean = simulate_reflections(add_cable=False, add_xtalk=False)
    uvd = simulate_reflections(cdelay=255.0, cphase=2.0, camp=1e-2, add_cable=True, cable_ants=[23], add_xtalk=False)

    def test_model_auto_reflections(self):
        RF = reflections.ReflectionFitter(self.uvd)
        bl_k = (23, 23, 'ee')
        g_k = (23, 'Jee')
        RF.fft_data(window='blackmanharris', overwrite=True, ax='freq')  # for inspection

        # basic run through
        RF.model_auto_reflections(RF.data, (200, 300), keys=[bl_k], window='blackmanharris',
                                  zeropad=100, fthin=1, verbose=True)
        assert np.allclose(np.ravel(list(RF.ref_dly.values())), 255.0, atol=1e-1)
        assert np.allclose(np.ravel(list(RF.ref_amp.values())), 1e-2, atol=1e-4)
        assert np.allclose(np.ravel(list(RF.ref_phs.values())), 2.0, atol=1e-1)

        # try with a small edgecut
        RF = reflections.ReflectionFitter(self.uvd)
        edgecut = 5
        RF.model_auto_reflections(RF.data, (200, 300), keys=[bl_k], window='blackmanharris', reject_edges=False,
                                  zeropad=100, fthin=1, verbose=True, edgecut_low=edgecut, edgecut_hi=edgecut)
        assert np.allclose(np.ravel(list(RF.ref_dly.values())), 255.0, atol=1e-1)
        assert np.allclose(np.ravel(list(RF.ref_amp.values())), 1e-2, atol=1e-4)
        assert np.allclose(np.ravel(list(RF.ref_phs.values())), 2.0, atol=1e-1)

        # try a high ref_sig cut: assert ref_flags are True
        RF.model_auto_reflections(RF.data, (200, 300), keys=[bl_k], window='blackmanharris', ref_sig_cut=100)
        assert np.all(RF.ref_flags[g_k])

        # assert refinement uses flags to return zeros
        output = RF.refine_auto_reflections(RF.data, (20, 80), RF.ref_amp, RF.ref_dly, RF.ref_phs,
                                            keys=[bl_k], ref_flags=RF.ref_flags, window='blackmanharris', zeropad=100,
                                            maxiter=100, method='Nelder-Mead', tol=1e-5)
        assert np.allclose(output[0][g_k], 0.0)

        # try filtering the visibilities
        RF.vis_clean(data=RF.data, ax='freq', min_dly=100, overwrite=True, window='blackmanharris', alpha=0.1, tol=1e-8, keys=[bl_k])
        RF.model_auto_reflections(RF.clean_resid, (200, 300), clean_data=RF.clean_data, keys=[bl_k],
                                  window='blackmanharris', zeropad=100, fthin=1, verbose=True)
        assert np.allclose(np.ravel(list(RF.ref_dly.values())), 255.0, atol=1e-1)
        assert np.allclose(np.ravel(list(RF.ref_amp.values())), 1e-2, atol=1e-4)
        assert np.allclose(np.ravel(list(RF.ref_phs.values())), 2.0, atol=1e-1)

        # try optimization on time-averaged data
        RF.timeavg_data(RF.data, RF.times, RF.lsts, 5000, keys=None, overwrite=True)
        RF.model_auto_reflections(RF.avg_data, (200, 300), keys=[bl_k], window='blackmanharris',
                                  zeropad=100, fthin=1, verbose=True)
        output = RF.refine_auto_reflections(RF.avg_data, (20, 80), RF.ref_amp, RF.ref_dly, RF.ref_phs,
                                            keys=[bl_k], window='blackmanharris', zeropad=100,
                                            maxiter=100, method='Nelder-Mead', tol=1e-5)
        ref_amp = output[0]
        ref_dly = output[1]
        ref_phs = output[2]
        # assert equivalence to higher precision
        assert np.allclose(np.ravel(list(ref_dly.values())), 255.0, atol=1e-2)
        assert np.allclose(np.ravel(list(ref_amp.values())), 1e-2, atol=1e-5)
        assert np.allclose(np.ravel(list(ref_phs.values())), 2.0, atol=1e-2)

        # now reverse delay range
        RF.model_auto_reflections(RF.avg_data, (-300, -200), keys=[bl_k], window='blackmanharris',
                                  zeropad=100, fthin=1, verbose=True)
        assert np.allclose(np.ravel(list(RF.ref_dly.values())), -255.0, atol=1e-1)
        assert np.allclose(np.ravel(list(RF.ref_amp.values())), 1e-2, atol=1e-4)
        assert np.allclose(np.ravel(list(RF.ref_phs.values())), 2 * np.pi - 2.0, atol=1e-1)

        output = RF.refine_auto_reflections(RF.avg_data, (80, 20), RF.ref_amp, RF.ref_dly, RF.ref_phs,
                                            keys=[bl_k, (39, 39, 'ee')], window='blackmanharris', zeropad=100,
                                            maxiter=100, method='BFGS', tol=1e-5)
        ref_amp = output[0]
        ref_dly = output[1]
        ref_phs = output[2]
        # assert equivalence to higher precision
        assert np.allclose(np.ravel(list(ref_dly.values())), -255.0, atol=1e-2)
        assert np.allclose(np.ravel(list(ref_amp.values())), 1e-2, atol=1e-5)
        assert np.allclose(np.ravel(list(ref_phs.values())), 2 * np.pi - 2.0, atol=1e-2)

        # test flagged data
        RF.model_auto_reflections(RF.avg_data, (-300, -200), keys=[bl_k], window='blackmanharris',
                                  zeropad=100, fthin=1, verbose=True)
        RF.avg_flags[bl_k][:] = True
        output = RF.refine_auto_reflections(RF.avg_data, (80, 20), RF.ref_amp, RF.ref_dly, RF.ref_phs,
                                            keys=[bl_k], window='blackmanharris', zeropad=100, clean_flags=RF.avg_flags,
                                            maxiter=100, method='BFGS', tol=1e-5)
        assert not np.any(output[3][(23, 'Jee')])
        RF.avg_flags[bl_k][:] = False

        # non-even Nfreqs
        RF = reflections.ReflectionFitter(self.uvd.select(frequencies=np.unique(self.uvd.freq_array)[:-1], inplace=False))
        RF.model_auto_reflections(RF.data, (200, 300), keys=[bl_k], window='blackmanharris',
                                  zeropad=100, fthin=1, verbose=True)
        assert np.allclose(np.ravel(list(RF.ref_dly.values())), 255.0, atol=1e-1)
        assert np.allclose(np.ravel(list(RF.ref_amp.values())), 1e-2, atol=1e-4)
        assert np.allclose(np.ravel(list(RF.ref_phs.values())), 2.0, atol=1e-1)

        # exceptions
        pytest.raises(ValueError, RF.model_auto_reflections, RF.data, (4000, 5000), window='none', edgecut_low=edgecut)

        # test reject_edges: choose dly_range to make max on edge
        # assert peak is in main lobe, not at actual reflection delay
        RF.model_auto_reflections(RF.data, (25, 300), keys=[bl_k], window='blackmanharris', reject_edges=False,
                                  zeropad=100, fthin=1, verbose=True)
        assert np.all(np.ravel(list(RF.ref_dly.values())) < 200)
        # assert peak is correct
        RF.model_auto_reflections(RF.data, (25, 300), keys=[bl_k], window='blackmanharris', reject_edges=True,
                                  zeropad=100, fthin=1, verbose=True)
        assert np.allclose(np.ravel(list(RF.ref_dly.values())), 255.0, atol=1e-1)
        assert np.allclose(np.ravel(list(RF.ref_amp.values())), 1e-2, atol=1e-4)
        assert np.allclose(np.ravel(list(RF.ref_phs.values())), 2.0, atol=1e-1)
        # assert valley results in flagged reflection (make sure zeropad=0)
        RF.model_auto_reflections(RF.data, (25, 225), keys=[bl_k], window='blackmanharris', reject_edges=True,
                                  zeropad=0, fthin=1, verbose=True)
        assert np.all(RF.ref_flags[g_k])

        # try clear
        RF.clear(exclude=['data'])
        assert len(RF.ref_eps) == 0
        assert len(RF.ref_gains) == 0
        assert len(RF.data) > 0

        # try soft copy
        RF2 = RF.soft_copy()
        assert RF2.__class__ == reflections.ReflectionFitter

    def test_write_auto_reflections(self):
        RF = reflections.ReflectionFitter(self.uvd)
        bl_k = (23, 23, 'ee')
        a_k = (23, 'Jee')
        # add a flagged integration
        RF.flags[bl_k][0] = True
        RF._clear_ref()
        RF.model_auto_reflections(RF.data, (200, 300), clean_flags=RF.flags, window='blackmanharris', zeropad=100, fthin=1, verbose=True)
        uvc = RF.write_auto_reflections("./ex.calfits", overwrite=True, write_npz=True)
        assert uvc.Ntimes == 100
        assert len(uvc.ant_array) == 5
        assert not np.allclose(uvc.gain_array[uvc.ant_array.tolist().index(23)], 1.0)
        # assert flag propagation
        assert np.all(uvc.get_flags(a_k)[:, 0])

        # load npz and do some basic checks
        fnpz = np.load('ex.npz', allow_pickle=True)
        assert len(fnpz['times']) == 100
        assert fnpz['flags'].item()[a_k][0, 0]
        assert 'delay' in fnpz and 'amp' in fnpz and 'phase' in fnpz

        # test w/ input calfits
        RF.flags[bl_k][1] = True
        uvc = RF.write_auto_reflections("./ex.calfits", input_calfits="./ex.calfits", overwrite=True)
        RF._clear_ref()
        RF.model_auto_reflections(RF.data, (200, 300), clean_flags=RF.flags, window='blackmanharris', zeropad=100, fthin=1, verbose=True)
        uvc = RF.write_auto_reflections("./ex.calfits", input_calfits='./ex.calfits', overwrite=True)
        assert uvc.Ntimes == 100
        assert len(uvc.ant_array) == 5
        # assert flag propagation
        assert np.all(uvc.get_flags(a_k)[:, :2])

        # test data is corrected by taking ratio w/ clean data
        data = deepcopy(RF.data)
        g = reflections.form_gains(RF.ref_eps)
        apply_cal.calibrate_in_place(data, g, gain_convention='divide')
        r = data[bl_k] / self.uvd_clean.get_data(bl_k)
        assert np.abs(np.mean(r) - 1) < 1e-1

        # test with timeaverage
        full_uvc = copy.deepcopy(uvc)
        RF.timeavg_data(RF.data, RF.times, RF.lsts, 1e10, rephase=False, overwrite=True)
        RF._clear_ref()
        RF.model_auto_reflections(RF.avg_data, (200, 300), window='blackmanharris', zeropad=100, fthin=1, verbose=True)
        assert RF.ref_gains[a_k].shape == (1, 128)
        # test write without input calfits results in Ntimes = 1
        uvc = RF.write_auto_reflections("./ex2.calfits", time_array=RF.avg_times, overwrite=True)
        assert uvc.Ntimes == 1
        # test by feeding full-time calfits that output times are full-time
        uvc = RF.write_auto_reflections("./ex.calfits", time_array=RF.avg_times, input_calfits='./ex.calfits', overwrite=True)
        assert uvc.Ntimes == 100

        # test input calibration with slightly shifted frequencies
        uvc.read_calfits("./ex.calfits")
        uvc.freq_array += 1e-5  # assert this doesn't fail
        uvc.use_future_array_shapes()
        T = reflections.ReflectionFitter(self.uvd, input_cal=uvc)
        assert isinstance(T.hc, io.HERACal)
        uvc.freq_array += 1e2  # now test it fails with a large shift
        pytest.raises(AssertionError, reflections.ReflectionFitter, self.uvd, input_cal=uvc)

        os.remove('./ex.calfits')
        os.remove('./ex2.calfits')
        os.remove('./ex.npz')

    def test_auto_reflection_argparser(self):
        sys.argv = [sys.argv[0], 'a', '--output_fname', 'ex.calfits', '--dly_ranges', '10,20', '10,20', '--overwrite', '--opt_buffer', '25', '75']
        parser = reflections.auto_reflection_argparser()
        a = parser.parse_args()
        assert a.data[0] == 'a'
        assert a.output_fname == 'ex.calfits'
        assert a.dly_ranges[0] == '10,20'
        assert len(a.dly_ranges) == 2
        assert np.allclose(a.opt_buffer, [25, 75])
        assert a.write_each_calfits

        sys.argv = [sys.argv[0], 'a', '--only_write_final_calfits']
        parser = reflections.auto_reflection_argparser()
        a = parser.parse_args()
        assert not a.write_each_calfits

    def test_auto_reflection_run(self):
        # most of the code tests have been done above, this is just to ensure this wrapper function runs
        uvd = simulate_reflections(cdelay=[150.0, 250.0], cphase=[0.0, 0.0], camp=[1e-2, 1e-2], add_cable=True, cable_ants=[23], add_xtalk=False)
        reflections.auto_reflection_run(uvd, [(100, 200), (200, 300)], "./ex.calfits", time_avg=True, compress_tavg_calfits=True,
                                        window='blackmanharris', write_npz=True, overwrite=True, ref_sig_cut=1.0)
        assert os.path.exists("./ex.calfits")
        assert os.path.exists("./ex.npz")
        assert os.path.exists("./ex.ref2.calfits")
        assert os.path.exists("./ex.ref2.npz")

        # ensure gains have two humps at 150 and 250 ns
        uvc = UVCal()
        uvc.read_calfits('./ex.calfits')
        assert uvc.Ntimes == 1  # because time_avg=True
        uvc2 = UVCal()
        uvc2.read_calfits('./ex.ref2.calfits')
        assert uvc2.Ntimes == 1  # because time_avg=True
        uvc.gain_array *= uvc2.gain_array
        aind = np.argmin(np.abs(uvc.ant_array - 23))
        g = uvc.gain_array[aind, 0, :, :, 0].T
        delays = np.fft.fftfreq(uvc.Nfreqs, np.diff(uvc.freq_array[0])[0]) * 1e9
        gfft = np.mean(np.abs(np.fft.fft(g, axis=1)), axis=0)

        assert delays[np.argmax(gfft * ((delays > 100) & (delays < 200)))] == 150
        assert delays[np.argmax(gfft * ((delays > 200) & (delays < 300)))] == 250

        os.remove("./ex.calfits")
        os.remove("./ex.npz")
        os.remove("./ex.ref2.calfits")
        os.remove("./ex.ref2.npz")

        # Try with write_each_calfits = False
        reflections.auto_reflection_run(uvd, [(100, 200), (200, 300)], "./ex.calfits", time_avg=True, compress_tavg_calfits=True,
                                        window='blackmanharris', write_npz=False, overwrite=True, ref_sig_cut=1.0, write_each_calfits=False)
        assert os.path.exists("./ex.calfits")
        assert not os.path.exists("./ex.ref2.calfits")
        uvc3 = UVCal()
        uvc3.read_calfits('./ex.calfits')
        np.testing.assert_array_almost_equal(uvc3.gain_array, uvc.gain_array, 12)
        os.remove("./ex.calfits")


@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
class Test_ReflectionFitter_XTalk(object):
    # simulate
    uvd = simulate_reflections(add_cable=False, xdelay=250.0, xphase=0, xamp=1e-3, add_xtalk=True)

    def test_svd_functions(self):
        RF = reflections.ReflectionFitter(self.uvd)
        bl = (23, 24, 'ee')

        # fft data
        RF.fft_data(data=RF.data, window='blackmanharris', overwrite=True)

        # test sv_decomposition on positive side
        wgts = RF.svd_weights(RF.dfft, RF.delays, min_dly=200, max_dly=300, side='pos')
        RF.sv_decomp(RF.dfft, wgts=wgts, keys=[bl], overwrite=True, sparse_svd=True)

        # build a model
        RF.build_pc_model(RF.umodes, RF.vmodes, RF.svals, Nkeep=1, increment=False, overwrite=True)

        # test containers exist
        assert np.all([hasattr(RF, o) for o in ['umodes', 'vmodes', 'svals', 'uflags', 'pcomp_model', 'dfft']])
        # test good information compression
        assert RF.svals[bl][0] / RF.svals[bl][1] > 20

        # assert its a good fit to the xtalk at 250 ns delay
        ind = np.argmin(np.abs(RF.delays - 250))
        Vrms = np.sqrt(np.mean(RF.dfft[bl][:, ind].real**2))
        Rrms = np.sqrt(np.mean((RF.dfft[bl][:, ind].real - RF.pcomp_model[bl][:, ind].real)**2))
        # says that residual is small compared to original array
        assert Rrms / Vrms < 0.01

        # increment the model
        wgts = RF.svd_weights(RF.dfft, RF.delays, min_dly=200, max_dly=300, side='neg')
        RF.sv_decomp(RF.dfft, wgts=wgts, overwrite=True, sparse_svd=True)
        RF.build_pc_model(RF.umodes, RF.vmodes, RF.svals, Nkeep=1, increment=True)

        # says that the two are similar to each other at -250 ns, which they should be
        ind = np.argmin(np.abs(RF.delays - -250))
        Vrms = np.sqrt(np.mean(RF.dfft[bl][:, ind].real**2))
        Rrms = np.sqrt(np.mean((RF.dfft[bl][:, ind].real - RF.pcomp_model[bl][:, ind].real)**2))
        # says that residual is small compared to original array
        assert Rrms / Vrms < 0.01

        # overwrite the model with double side modeling
        wgts = RF.svd_weights(RF.dfft, RF.delays, min_dly=200, max_dly=300, side='both')
        RF.sv_decomp(RF.dfft, wgts=wgts, overwrite=True, sparse_svd=True)
        RF.build_pc_model(RF.umodes, RF.vmodes, RF.svals, Nkeep=2, increment=False, overwrite=True)
        # says the residual is small compared to original array
        ind = np.argmin(np.abs(RF.delays - 250))
        Vrms = np.sqrt(np.mean(RF.dfft[bl][:, ind].real**2))
        Rrms = np.sqrt(np.mean((RF.dfft[bl][:, ind].real - RF.pcomp_model[bl][:, ind].real)**2))
        assert Rrms / Vrms < 0.01

        # subtract the model from the data
        RF.subtract_model(RF.data, overwrite=True)
        assert RF.pcomp_model_fft[bl].shape == (100, 128)
        assert RF.data_pcmodel_resid[bl].shape == (100, 128)

        # subtract the model from the data: test with edgecut
        RF.subtract_model(RF.data, edgecut_low=1, overwrite=True)
        assert np.isclose(RF.pcomp_model_fft[bl][:, 0], 0).all()  # assert edgecut channels are zeroed

    def test_misc_svd_funcs(self):
        # setup RF object
        RF = reflections.ReflectionFitter(self.uvd)
        # add noise
        np.random.seed(0)
        Namp = 3e0
        for k in RF.data:
            RF.data += stats.norm.rvs(0, Namp, RF.Ntimes * RF.Nfreqs).reshape(RF.Ntimes, RF.Nfreqs) + 1j * stats.norm.rvs(0, Namp, RF.Ntimes * RF.Nfreqs).reshape(RF.Ntimes, RF.Nfreqs)
        bl = (23, 24, 'ee')

        # fft data
        RF.fft_data(data=RF.data, window='blackmanharris', overwrite=True)

        # sparse sv decomp
        svd_wgts = RF.svd_weights(RF.dfft, RF.delays, min_dly=150, max_dly=500, side='both')
        RF.sv_decomp(RF.dfft, wgts=svd_wgts, keys=[bl], overwrite=True, Nkeep=None, sparse_svd=True)
        assert RF.umodes[bl].shape == (100, 98)
        assert RF.vmodes[bl].shape == (98, 128)

        RF.sv_decomp(RF.dfft, wgts=svd_wgts, keys=[bl], overwrite=True, Nkeep=10, sparse_svd=True)
        assert RF.umodes[bl].shape == (100, 10)
        assert RF.vmodes[bl].shape == (10, 128)

        # full svd
        RF.sv_decomp(RF.dfft, wgts=svd_wgts, keys=[bl], overwrite=True, Nkeep=None, sparse_svd=False)
        assert RF.umodes[bl].shape == (100, 100)
        assert RF.vmodes[bl].shape == (100, 128)

        # test interpolation of umodes
        gp_frate = 0.2
        RF.interp_u(RF.umodes, RF.times, overwrite=True, gp_frate=gp_frate, gp_nl=1e-10, optimizer=None, Ninterp=None)
        assert RF.umode_interp[bl].shape == (100, 100)
        RF.interp_u(RF.umodes, RF.times, overwrite=True, gp_frate=gp_frate, gp_nl=1e-10, optimizer=None, Ninterp=10)
        assert RF.umode_interp[bl].shape == (100, 10)

        # get fft and assert a good match within gp_frate
        RF.fft_data(data=RF.umodes, assign='ufft', window='blackmanharris', ax='time', overwrite=True, edgecut_low=5, edgecut_hi=5)
        RF.fft_data(data=RF.umode_interp, assign='uifft', window='blackmanharris', ax='time', overwrite=True, edgecut_low=5, edgecut_hi=5)
        select = np.abs(RF.frates) < gp_frate / 2
        assert np.mean(np.abs(RF.ufft[bl][select, 0] - RF.uifft[bl][select, 0]) / np.abs(RF.ufft[bl][select, 0])) < 0.01
        # plt.plot(RF.frates, np.abs(RF.ufft[bl][:, 0]));plt.plot(RF.frates, np.abs(RF.uifft[bl][:, 0]));plt.yscale('log')

        # test mode projection after interpolation (smoothing)
        umodes = copy.deepcopy(RF.umodes)
        for k in umodes:
            umodes[k][:, :10] = RF.umode_interp[k][:, :10]  # fill in umodes with smoothed components
        vmodes = RF.project_svd_modes(RF.dfft * svd_wgts, umodes=umodes, svals=RF.svals)

        # build systematic models with original vmodes and projected vmodes
        RF.build_pc_model(umodes, RF.vmodes, RF.svals, overwrite=True, Nkeep=10)
        pcomp1 = RF.pcomp_model[bl]
        RF.build_pc_model(umodes, vmodes, RF.svals, overwrite=True, Nkeep=10)
        pcomp2 = RF.pcomp_model[bl]
        # assert pcomp model with projected vmode has less noise in it for a delay with no systematic
        ind = np.argmin(np.abs(RF.delays - 400))  # no systematic at this delay, only noise
        assert np.mean(np.abs(pcomp1[:, ind])) > np.mean(np.abs(pcomp2[:, ind]))

        # test projection of other SVD matrices
        _svals = RF.project_svd_modes(RF.dfft * svd_wgts, umodes=RF.umodes, vmodes=RF.vmodes)
        _umodes = RF.project_svd_modes(RF.dfft * svd_wgts, vmodes=RF.vmodes, svals=RF.svals)

        # assert original is nearly the same as projected
        assert np.allclose(_svals[bl], RF.svals[bl], atol=1e-10)
        assert np.allclose(_umodes[bl][:, 0], RF.umodes[bl][:, 0], atol=1e-10)

        # try with and without Nmirror
        RF.interp_u(RF.umodes, RF.times, overwrite=True, gp_frate=gp_frate, gp_nl=1e-10, optimizer=None, Ninterp=10, Nmirror=0)
        uinterp1 = copy.deepcopy(RF.umode_interp)
        RF.interp_u(RF.umodes, RF.times, overwrite=True, gp_frate=gp_frate, gp_nl=1e-10, optimizer=None, Ninterp=10, Nmirror=25)
        uinterp2 = copy.deepcopy(RF.umode_interp)
        # assert higher order umodes don't diverge as much at time boundaries with Nmirror > 0
        for i in range(3, 10):
            assert np.mean(np.abs(uinterp1[bl][:2, i]) / np.abs(uinterp2[bl][:2, i])) > 1.0
            assert np.mean(np.abs(uinterp1[bl][-2:, i]) / np.abs(uinterp2[bl][-2:, i])) > 1.0

        # test too large Nmirror
        pytest.raises(AssertionError, RF.interp_u, RF.umodes, RF.times, overwrite=True, gp_frate=gp_frate, gp_nl=1e-10, optimizer=None, Ninterp=10, Nmirror=100)

        # assert custom kernel works
        gp_len = 1.0 / (0.4 * 1e-3) / (24.0 * 3600.0)
        kernel = 1**2 * kernels.RBF(gp_len) + kernels.WhiteKernel(1e-10)
        RF.interp_u(RF.umodes, RF.times, overwrite=True, kernels=kernel, optimizer=None)
        assert RF.umode_interp[bl].shape == (100, 100)

        # assert broadcasting to full time resolution worked
        RF.timeavg_data(RF.data, RF.times, RF.lsts, 500, overwrite=True, verbose=False)
        RF.fft_data(data=RF.avg_data, window='blackmanharris', overwrite=True, assign='adfft', dtime=np.diff(RF.avg_times)[0] * 24 * 3600)
        wgts = RF.svd_weights(RF.adfft, RF.delays, min_dly=200, max_dly=300, side='both')
        RF.sv_decomp(RF.adfft, wgts=wgts, keys=[bl], overwrite=True, sparse_svd=True)
        assert RF.umodes[bl].shape == (34, 32)
        RF.interp_u(RF.umodes, RF.avg_times, full_times=RF.times, overwrite=True, gp_frate=1.0, gp_nl=1e-10, optimizer=None)
        assert RF.umode_interp[bl].shape == (100, 32)

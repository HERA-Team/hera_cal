# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import pytest
import os
import shutil
import numpy as np
import sys
from collections import OrderedDict as odict
import copy
import glob
from pyuvdata import UVData
from pyuvdata import utils as uvutils
import unittest
import yaml
from scipy import stats
from scipy import constants
from pyuvdata import UVFlag, UVBeam
from .. import utils
from .. import datacontainer, io, frf
from ..data import DATA_PATH
import warnings

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*Using known values for HERA",
)


@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
def test_timeavg_waterfall():
    fname = os.path.join(DATA_PATH, "zen.2458042.12552.xx.HH.uvXA")

    uvd = UVData()
    uvd.read_miriad(fname, use_future_array_shapes=True)

    d = uvd.get_data(24, 25)
    f = uvd.get_flags(24, 25)
    n = uvd.get_nsamples(24, 25)
    t = np.unique(uvd.time_array)
    fr = uvd.freq_array.squeeze()
    lsts = []
    for _l in uvd.lst_array:
        if _l not in lsts:
            lsts.append(_l)
    lsts = np.array(lsts)
    antpos, ants = uvd.get_ENU_antpos()
    blv = antpos[ants.tolist().index(24)] - antpos[ants.tolist().index(25)]

    # test basic execution
    ad, af, an, al, aea = frf.timeavg_waterfall(d, 25, verbose=False)
    assert ad.shape == (3, 64)
    assert af.shape == (3, 64)
    assert an.shape == (3, 64)
    assert not np.any(af)
    assert np.allclose(an[1, 0], 25.0)
    assert np.allclose(an[2, 0], 10.0)

    # test rephase
    ad, af, an, al, aea = frf.timeavg_waterfall(d, 25, flags=f, rephase=True, lsts=lsts, freqs=fr, bl_vec=blv,
                                                nsamples=n, extra_arrays=dict(times=t), verbose=False)

    assert ad.shape == (3, 64)
    assert af.shape == (3, 64)
    assert an.shape == (3, 64)
    assert np.any(af)
    assert len(al) == 3
    assert len(aea['avg_times']) == 3
    assert np.allclose(an.max(), 25.0)

    # test various Navgs
    ad, af, an, al, aea = frf.timeavg_waterfall(d, 1, flags=f, rephase=True, lsts=lsts, freqs=fr, bl_vec=blv,
                                                nsamples=n, extra_arrays=dict(times=t), verbose=False)

    assert ad.shape == (60, 64)
    ad, af, an, al, aea = frf.timeavg_waterfall(d, 60, flags=f, rephase=True, lsts=lsts, freqs=fr, bl_vec=blv,
                                                nsamples=n, extra_arrays=dict(times=t), verbose=False)
    assert ad.shape == (1, 64)

    # wrap lst
    ad2, af2, an2, al2, aea2 = frf.timeavg_waterfall(d, 60, flags=f, rephase=True, lsts=lsts + 1.52917804, freqs=fr, bl_vec=blv,
                                                     nsamples=n, extra_arrays=dict(times=t), verbose=False)

    assert ad.shape == (1, 64)
    assert np.allclose(ad, ad2)
    assert np.allclose(al, al2 - 1.52917804)

    # Test Error
    with pytest.raises(ValueError):
        frf.timeavg_waterfall(d, 25, verbose=False, wgt_by_nsample=True, wgt_by_favg_nsample=True)

    # test weightings
    d = np.ones((4, 10))
    d[0, :] *= 2
    n = np.ones((4, 10))
    n[0, 0:5] *= 2
    ad, _, _, _, _ = frf.timeavg_waterfall(d, 2, rephase=False, nsamples=n, wgt_by_nsample=True)
    np.testing.assert_array_equal(ad[1, :], 1.0)
    np.testing.assert_array_equal(ad[0, 0:5], 5. / 3)
    np.testing.assert_array_equal(ad[0, 5:10], 1.5)
    ad, _, _, _, _ = frf.timeavg_waterfall(d, 2, rephase=False, nsamples=n, wgt_by_nsample=False, wgt_by_favg_nsample=True)
    np.testing.assert_array_equal(ad[1, :], 1.0)
    np.testing.assert_array_equal(ad[0, :], 1.6)


def test_fir_filtering():
    # convert a high-pass frprofile to an FIR filter
    frbins = np.linspace(-40e-3, 40e-3, 1024)
    frp = np.ones(1024)
    frp[512 - 9:512 + 10] = 0.0
    fir, tbins = frf.frp_to_fir(frp, delta_bin=np.diff(frbins)[0])
    # confirm its purely real
    assert not np.any(np.isclose(np.abs(fir.real), 0.0))
    assert np.allclose(np.abs(fir.imag), 0.0)

    # convert back
    _frp, _frbins = frf.frp_to_fir(fir, delta_bin=np.diff(tbins)[0], undo=True)
    np.testing.assert_array_almost_equal(frp, _frp.real)
    np.testing.assert_array_almost_equal(np.diff(frbins), np.diff(_frbins))
    assert np.allclose(np.abs(_frp.imag), 0.0)

    # test noise averaging properties
    frp = np.zeros(1024)
    frp[512] = 1.0
    t_ratio = frf.fr_tavg(frp)
    assert np.allclose(t_ratio, 1024)


@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
@pytest.mark.filterwarnings("ignore:Fixing auto-correlations")
@pytest.mark.filterwarnings("ignore:The lst_array is not self-consistent")
@pytest.mark.filterwarnings("ignore:Mean of empty slice")  # TODO: probably should remove this
@pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide")  # TODO: probably should remove this
@pytest.mark.filterwarnings("ignore:No new keys provided")
@pytest.mark.filterwarnings("ignore:writing default values for restfreq")
@pytest.mark.filterwarnings("ignore:The uvw_array does not match")
@pytest.mark.filterwarnings("ignore:projected is False, but RA, Dec is off from lst")
class Test_FRFilter:
    def setup_method(self):
        self.fname = os.path.join(DATA_PATH, "zen.2458042.12552.xx.HH.uvXA")
        self.F = frf.FRFilter(self.fname, filetype='miriad')
        self.F.read()

    def test_timeavg_data(self):
        # test basic time average
        self.F.timeavg_data(self.F.data, self.F.times, self.F.lsts, 35, rephase=True, keys=[(24, 25, 'ee')])
        assert self.F.Navg == 3
        assert len(self.F.avg_data) == 1
        assert self.F.avg_data[(24, 25, 'ee')].shape == (20, 64)

        # test full time average and overwrite
        self.F.timeavg_data(self.F.data, self.F.times, self.F.lsts, 1e10, rephase=True, verbose=False, overwrite=False)
        assert self.F.Navg == 60
        assert len(self.F.avg_data) == 28
        assert self.F.avg_data[(24, 25, 'ee')].shape == (20, 64)
        assert self.F.avg_data[(24, 37, 'ee')].shape == (1, 64)

        # test weight by nsample
        F = copy.deepcopy(self.F)
        k = (24, 25, 'ee')
        F.nsamples[k][:3] = 0.0
        F.timeavg_data(F.data, F.times, F.lsts, 35, nsamples=F.nsamples, keys=[k], overwrite=True,
                       wgt_by_nsample=True)
        assert np.all(np.isclose(F.avg_data[k][0], 0.0))  # assert data is zero b/c I zeroed nsample
        assert np.all(np.isclose(F.avg_nsamples[k][0], 0.0))  # assert avg_nsample is also zero
        assert np.all(np.isclose(F.avg_nsamples[k][1:], 3.0))  # assert non-zeroed nsample is 3

        # repeat without nsample wgt
        F.timeavg_data(F.data, F.times, F.lsts, 35, nsamples=F.nsamples, keys=[k], overwrite=True,
                       wgt_by_nsample=False)
        assert not np.any(np.isclose(F.avg_data[k][0, 5:-5], 0.0))  # assert non-edge data is now not zero
        assert np.all(np.isclose(F.avg_nsamples[k][0], 0.0))  # avg_nsample should still be zero

        # exceptions
        pytest.raises(AssertionError, self.F.timeavg_data, self.F.data, self.F.times, self.F.lsts, 1.0)

    def test_filter_data(self):
        # construct high-pass filter
        frates = np.fft.fftshift(np.fft.fftfreq(self.F.Ntimes, self.F.dtime)) * 1e3
        w = np.ones((self.F.Ntimes, self.F.Nfreqs), dtype=float)
        w[np.abs(frates) < 20] = 0.0
        frps = datacontainer.DataContainer(dict([(k, w) for k in self.F.data]))

        # make gaussian random noise
        bl = (24, 25, 'ee')
        window = 'blackmanharris'
        ec = 0
        np.random.seed(0)
        self.F.data[bl] = np.reshape(stats.norm.rvs(0, 1, self.F.Ntimes * self.F.Nfreqs)
                                     + 1j * stats.norm.rvs(0, 1, self.F.Ntimes * self.F.Nfreqs), (self.F.Ntimes, self.F.Nfreqs))
        # fr filter noise
        self.F.filter_data(self.F.data, frps, overwrite=True, verbose=False, axis=0, keys=[bl])

        # check key continue w/ ridiculous edgecut
        self.F.filter_data(self.F.data, frps, overwrite=False, verbose=False, keys=[bl], edgecut_low=100, axis=0)

        # fft
        self.F.fft_data(data=self.F.data, assign='dfft', ax='freq', window=window, edgecut_low=ec, edgecut_hi=ec, overwrite=True)
        self.F.fft_data(data=self.F.filt_data, assign='rfft', ax='freq', window=window, edgecut_low=ec, edgecut_hi=ec, overwrite=True)

        # ensure drop in noise power is reflective of frf_nsamples
        dfft = np.mean(np.abs(self.F.dfft[bl]), axis=0)
        rfft = np.mean(np.abs(self.F.rfft[bl]), axis=0)
        r = np.mean(dfft / rfft)
        assert np.allclose(r, np.sqrt(np.mean(self.F.filt_nsamples[bl])), atol=1e-1)

    def test_write_data(self):
        self.F.timeavg_data(self.F.data, self.F.times, self.F.lsts, 35, rephase=False, verbose=False)
        self.F.write_data(self.F.avg_data, "./out.uv", filetype='miriad', overwrite=True,
                          add_to_history='testing', times=self.F.avg_times, lsts=self.F.avg_lsts)
        assert os.path.exists("./out.uv")
        hd = io.HERAData('./out.uv', filetype='miriad')
        hd.read()
        assert 'testing' in hd.history.replace('\n', '').replace(' ', '')
        assert 'Thisfilewasproducedbythefunction' in hd.history.replace('\n', '').replace(' ', '')
        shutil.rmtree("./out.uv")

        pytest.raises(AssertionError, self.F.write_data, self.F.avg_data, "./out.uv", times=self.F.avg_times)
        pytest.raises(ValueError, self.F.write_data, self.F.data, "hi", filetype='foo')

    @pytest.mark.parametrize("equalize_times", [True, False])
    def test_time_avg_data_and_write(self, tmpdir, equalize_times):
        # time-averaged data written too file will be compared to this.
        tmp_path = tmpdir.strpath
        output = tmp_path + '/test_output.miriad'
        flag_output = tmp_path + '/test_output.flags.h5'
        self.F.timeavg_data(self.F.data, self.F.times, self.F.lsts, 35., rephase=True, overwrite=True,
                            wgt_by_nsample=True, flags=self.F.flags, nsamples=self.F.nsamples)
        frf.time_avg_data_and_write(self.fname, output, t_avg=35., rephase=True, wgt_by_nsample=True,
                                    flag_output=flag_output, filetype='miriad', equalize_interleave_times=equalize_times)
        data_out = frf.FRFilter(output, filetype='miriad')
        data_out.read()
        for k in data_out.data:
            assert np.allclose(data_out.data[k], self.F.avg_data[k])
            assert np.allclose(data_out.flags[k], self.F.avg_flags[k])
            assert np.allclose(data_out.nsamples[k], self.F.avg_nsamples[k])

    @pytest.mark.parametrize(
        "ninterleave, equalize_times", [(2, True), (2, False), (3, True), (3, False),
                                        (4, True), (4, False), (5, True),
                                        (5, False), (6, True), (6, False)])
    def test_time_avg_data_and_write_interleave(self, tmpdir, ninterleave, equalize_times):
        tmp_path = tmpdir.strpath
        input_name = os.path.join(tmp_path, 'test_input.uvh5')
        uvd = UVData()
        uvd.read(self.fname, use_future_array_shapes=True)
        uvd.write_uvh5(input_name)
        output_name = os.path.join(tmp_path, 'test_output.uvh5')
        flag_output = tmp_path + '/test_output.flags.h5'
        frf.time_avg_data_and_write(input_name, output_name, t_avg=35., rephase=True,
                                    wgt_by_nsample=True, flag_output=flag_output,
                                    filetype='uvh5', ninterleave=ninterleave,
                                    equalize_interleave_times=equalize_times)
        # check that the correct number of files exist.
        interleaved_data = {}
        for inum in range(ninterleave):
            iname = os.path.join(os.path.dirname(output_name), os.path.basename(output_name).replace('.uvh5', f'.interleave_{inum}.uvh5'))
            assert os.path.exists(iname)
            hd = io.HERAData(iname)
            hd.read()
            interleaved_data[inum] = hd
            os.remove(iname)
            # if equalize_times was False
            # check that times are complementary (don't overlap and ordered by file correctly).
            # otherwise, check that all times are equal.
            if inum > 0:
                for tn in range(interleaved_data[inum].Ntimes):
                    if not equalize_times:
                        assert interleaved_data[inum].times[tn] > interleaved_data[inum - 1].times[tn]
                    else:
                        assert interleaved_data[inum].times[tn] == interleaved_data[inum - 1].times[tn]
                        assert interleaved_data[inum].lsts[tn] == interleaved_data[inum - 1].lsts[tn]

    def test_time_avg_data_and_write_baseline_list(self, tmpdir):
        # compare time averaging over baseline list versus time averaging
        # without baseline list.
        tmp_path = tmpdir.strpath
        uvh5s = sorted(glob.glob(DATA_PATH + '/zen.2458045.*.uvh5'))
        output_files = []
        for file in uvh5s:
            baseline_list = io.baselines_from_filelist_position(file, uvh5s)
            output = tmp_path + '/' + file.split('/')[-1]
            output_files.append(output)
            output_flags = tmp_path + '/' + file.split('/')[-1].replace('.uvh5', '.flags.h5')
            with pytest.warns(RuntimeWarning):
                frf.time_avg_data_and_write(baseline_list=[], flag_output=output_flags,
                                            input_data_list=uvh5s, rephase=True,
                                            output_data=output, t_avg=35., wgt_by_nsample=True)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message='The uvw_array does not match the expected values')
                frf.time_avg_data_and_write(baseline_list=baseline_list, flag_output=output_flags,
                                            input_data_list=uvh5s, rephase=True,
                                            output_data=output, t_avg=35., wgt_by_nsample=True)

        # now do everything at once:
        output = tmp_path + '/combined.uvh5'
        frf.time_avg_data_and_write(uvh5s, output, t_avg=35., rephase=True, wgt_by_nsample=True)
        data_out = frf.FRFilter(output)
        data_out_bls = frf.FRFilter(output_files)
        data_out.read()
        data_out_bls.read()
        # check that data, flags, nsamples are all close.
        for k in data_out.data:
            assert np.all(np.isclose(data_out.data[k], data_out_bls.data[k]))
            assert np.all(np.isclose(data_out.flags[k], data_out_bls.flags[k]))
            assert np.all(np.isclose(data_out.nsamples[k], data_out_bls.nsamples[k]))

    def test_time_average_argparser_multifile(self):
        sys.argv = [sys.argv[0], "first.uvh5", "second.uvh5", "output.uvh5", "--cornerturnfile", "input.uvh5", "--t_avg", "35.", "--rephase"]
        ap = frf.time_average_argparser()
        args = ap.parse_args()
        assert args.cornerturnfile == "input.uvh5"
        assert args.output_data == "output.uvh5"
        assert args.input_data_list == ['first.uvh5', 'second.uvh5']
        assert args.t_avg == 35.
        assert not args.clobber
        assert not args.verbose
        assert args.flag_output is None
        assert args.filetype == "uvh5"

    @pytest.mark.parametrize(
        "avg_red_bllens, pass_data",
        [(True, False), (False, False), (True, False), (False, True)],
    )
    def test_tophat_frfilter(self, avg_red_bllens, pass_data):
        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        k = (24, 25, 'ee')
        frfil = frf.FRFilter(fname, filetype='miriad')
        frfil.read(bls=[k])
        bl = np.linalg.norm(frfil.antpos[24] - frfil.antpos[25]) / constants.c * 1e9
        sdf = (frfil.freqs[1] - frfil.freqs[0]) / 1e9
        frate_centers, frate_half_widths = frf.select_tophat_frates(case='sky', uvd=frfil.hd, blvecs=frfil.blvecs)
        frfil.tophat_frfilter(tol=1e-2, output_prefix='frfiltered', frate_centers=frate_centers, frate_half_widths=frate_half_widths)
        for k in frfil.data.keys():
            assert frfil.frfiltered_resid[k].shape == (60, 64)
            assert frfil.frfiltered_model[k].shape == (60, 64)
            assert k in frfil.frfiltered_info

        # test skip_wgt imposition of flags
        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        k = (24, 25, 'ee')
        # check successful run when avg_red_bllens is True and when False.
        frfil = frf.FRFilter(fname, filetype='miriad')
        frfil.read(bls=[k])
        if avg_red_bllens:
            frfil.avg_red_baseline_vectors()
        wgts = {k: np.ones_like(frfil.flags[k], dtype=float)}
        wgts[k][:, 0] = 0.0
        if pass_data:
            data_kwargs = {'data': frfil.data}
        else:
            data_kwargs = {}
        for pre_filter in [True, False]:
            frate_centers, frate_half_widths = frf.select_tophat_frates(case='sky', uvd=frfil.hd, blvecs=frfil.blvecs)

            frfil.tophat_frfilter(frate_centers=frate_centers, frate_half_widths=frate_half_widths,
                                  keys=[k], wgts=wgts, tol=1e-5, window='blackman-harris', skip_wgt=0.1,
                                  maxiter=100,
                                  pre_filter_modes_between_lobe_minimum_and_zero=pre_filter, **data_kwargs)
            assert frfil.clean_info[k][(0, frfil.Nfreqs)]['status']['axis_0'][0] == 'skipped'
            np.testing.assert_array_equal(frfil.clean_flags[k][:, 0], np.ones_like(frfil.flags[k][:, 0]))
            np.testing.assert_array_equal(frfil.clean_model[k][:, 0], np.zeros_like(frfil.clean_resid[k][:, 0]))
            np.testing.assert_array_equal(frfil.clean_resid[k][:, 0], np.zeros_like(frfil.clean_resid[k][:, 0]))

    def test_load_tophat_frfilter_and_write_baseline_list(self, tmpdir):
        tmp_path = tmpdir.strpath
        uvh5 = [os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.first.uvh5"),
                os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.second.uvh5")]
        cals = [os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only.part1"),
                os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only.part2")]
        outfilename = os.path.join(tmp_path, 'temp.h5')
        cdir = os.path.join(tmp_path, 'cache_temp')
        # make a cache directory
        if os.path.isdir(cdir):
            shutil.rmtree(cdir)
        os.mkdir(cdir)
        # test graceful exit with baseline list length of zero.
        with pytest.warns(RuntimeWarning):
            frf.load_tophat_frfilter_and_write(datafile_list=uvh5, baseline_list=[],
                                               calfile_list=cals, spw_range=[100, 200], cache_dir=cdir,
                                               read_cache=True, write_cache=True, avg_red_bllens=True,
                                               res_outfilename=outfilename, clobber=True,
                                               mode='dayenu', case='sky')
        for avg_bl in [True, False]:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Antenna 53 not present")
                warnings.filterwarnings("ignore", message='Cannot preserve total_quality_array')
                frf.load_tophat_frfilter_and_write(datafile_list=uvh5, baseline_list=[(53, 54)], polarizations=['ee'],
                                                   calfile_list=cals, spw_range=[100, 200], cache_dir=cdir,
                                                   read_cache=True, write_cache=True, avg_red_bllens=avg_bl,
                                                   res_outfilename=outfilename, clobber=True,
                                                   mode='dayenu', case='sky')
            hd = io.HERAData(outfilename)
            d, f, n = hd.read()
            assert len(list(d.keys())) == 1
            assert d[(53, 54, 'ee')].shape[1] == 100
            assert d[(53, 54, 'ee')].shape[0] == 60
            # now do no spw range and no cal files just to cover those lines.
            frf.load_tophat_frfilter_and_write(datafile_list=uvh5, baseline_list=[(53, 54)], polarizations=['ee'],
                                               cache_dir=cdir,
                                               read_cache=True, write_cache=True, avg_red_bllens=avg_bl,
                                               res_outfilename=outfilename, clobber=True,
                                               mode='dayenu', case='sky')
            hd = io.HERAData(outfilename)
            d, f, n = hd.read()
            assert len(list(d.keys())) == 1
            assert d[(53, 54, 'ee')].shape[1] == 1024
            assert d[(53, 54, 'ee')].shape[0] == 60
        # now test flag factorization and time thresholding.
        # prepare an input files for broadcasting flags
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        input_file = os.path.join(tmp_path, 'temp_special_flags.h5')
        shutil.copy(uvh5, input_file)
        hd = io.HERAData(input_file)
        _, flags, _ = hd.read()
        ntimes_before = hd.Ntimes
        nfreqs_before = hd.Nfreqs
        freqs_before = hd.freqs
        times_before = hd.times
        for bl in flags:
            flags[bl][:] = False
            flags[bl][0, :hd.Nfreqs // 2] = True  # first time has 50% flagged
            flags[bl][-3:, -1] = True  # last channel has flags for three integrations
        hd.update(flags=flags)
        hd.write_uvh5(input_file, clobber=True)
        # this time_threshold will result in
        # entire first integration begin flagged
        # and entire final channel being flagged
        # when flags are broadcasted.
        time_thresh = 2. / hd.Ntimes
        for blnum, bl in enumerate(flags.keys()):
            outfilename = os.path.join(tmp_path, 'bl_chunk_%d.h5' % blnum)
            frf.load_tophat_frfilter_and_write(datafile_list=[input_file], res_outfilename=outfilename,
                                               tol=1e-4, baseline_list=[bl[:2]], polarizations=[bl[-1]],
                                               cache_dir=cdir,
                                               factorize_flags=True, case='sky',
                                               time_thresh=time_thresh, clobber=True)
        # now load all of the outputs in
        output_files = glob.glob(tmp_path + '/bl_chunk_*.h5')
        hd = io.HERAData(output_files)
        d, f, n = hd.read()
        hd_original = io.HERAData(uvh5)
        for bl in hd_original.bls:
            assert bl in d.keys()

        for bl in f:
            assert np.all(f[bl][:, -1])
            assert np.all(f[bl][0, :])

        # test apriori flags and flag_yaml
        flag_yaml = os.path.join(DATA_PATH, 'test_input/a_priori_flags_sample.yaml')
        uvf = UVFlag(hd, mode='flag', copy_flags=True, use_future_array_shapes=True)
        uvf.to_waterfall(keep_pol=False, method='and')
        uvf.flag_array[:] = False
        flagfile = os.path.join(tmp_path, 'test_flag.h5')
        uvf.write(flagfile, clobber=True)
        frf.load_tophat_frfilter_and_write(datafile_list=[input_file], res_outfilename=outfilename,
                                           tol=1e-4, baseline_list=[bl[:2]], polarizations=[bl[-1]],
                                           clobber=True, mode='dayenu', case='sky',
                                           external_flags=flagfile, overwrite_flags=True)
        # test that all flags are False
        hd = io.HERAData(outfilename)
        d, f, n = hd.read()
        for k in f:
            assert np.all(~f[k])
        # now do the external yaml
        frf.load_tophat_frfilter_and_write(datafile_list=[input_file], res_outfilename=outfilename,
                                           tol=1e-4, baseline_list=[bl[:2]], polarizations=[bl[-1]],
                                           clobber=True, mode='dayenu', case='sky',
                                           external_flags=flagfile, overwrite_flags=True,
                                           flag_yaml=flag_yaml)
        # test that all flags are af yaml flags
        hd = io.HERAData(outfilename)
        d, f, n = hd.read()
        for k in f:
            assert np.all(f[k][:, 0])
            assert np.all(f[k][:, 1])
            assert np.all(f[k][:, 10:20])
            assert np.all(f[k][:, 60])
        os.remove(outfilename)
        shutil.rmtree(cdir)

    def test_load_tophat_frfilter_and_write_multifile(self, tmpdir):
        # cover line where baseline-list is None and multiple files are provided.
        uvh5s = sorted(glob.glob(DATA_PATH + '/zen.2458045.*.uvh5'))
        tmp_path = tmpdir.strpath
        outfilename = os.path.join(tmp_path, 'temp_output.uvh5')
        frf.load_tophat_frfilter_and_write(uvh5s, filled_outfilename=outfilename, tol=1e-4, clobber=True, case='sky')
        hd = io.HERAData(uvh5s)
        d, f, n = hd.read()
        hdoutput = io.HERAData(outfilename)
        doutput, foutput, nouput = hdoutput.read()
        for k in doutput:
            assert doutput[k].shape == d[k].shape

    def test_build_fringe_rate_profiles(self):
        test_beam = os.path.join(DATA_PATH, "efield_test_nside16.beamfits")
        test_data = os.path.join(DATA_PATH, "fr_unittest_data_ds.uvh5")
        uvd = UVData()
        uvd.read_uvh5(test_data, use_future_array_shapes=True)
        uvb = UVBeam()
        uvb.read_beamfits(test_beam, use_future_array_shapes=True)
        fr_grid, profiles = frf.build_fringe_rate_profiles(uvd, uvb)
        assert len(fr_grid) == uvd.Ntimes

    def test_get_fringe_rate_limits(self):
        # simulations constructed with the notebook at https://drive.google.com/file/d/1jPPSmL3nqQbp7tTgP77j9KC0802iWyow/view?usp=sharing
        test_beam = os.path.join(DATA_PATH, "fr_unittest_beam.beamfits")
        test_data = os.path.join(DATA_PATH, "fr_unittest_data_ds.uvh5")
        uvd = UVData()
        uvd.read_uvh5(test_data, use_future_array_shapes=True)
        myfrf = frf.FRFilter(uvd)
        sim_c_frates = {}
        sim_w_frates = {}
        uvb = UVBeam()
        uvb.read_beamfits(test_beam, use_future_array_shapes=True)
        c_frs, w_frs = frf.get_fringe_rate_limits(uvd, uvb, percentile_low=10, percentile_high=90)
        for bl in c_frs:
            # fft data
            myfrf.fft_data(data=myfrf.data, ax='time', keys=[bl], overwrite=True, window='bh')
            csum = np.cumsum(np.abs(myfrf.dfft[bl]) ** 2.)
            csum /= csum.max()
            csum = np.hstack([[0], csum, [1.]])
            dfr = np.median(np.diff(myfrf.frates))
            frates = np.hstack([[myfrf.frates.min() - dfr], myfrf.frates, [myfrf.frates.max() + dfr]])
            frlow, frhigh = (frates[np.argmin(np.abs(csum - 0.1))], frates[np.argmin(np.abs(csum - 0.9))])
            sim_c_frates[bl] = .5 * (frlow + frhigh)
            sim_w_frates[bl] = .5 * np.abs(frlow - frhigh)
            sim_w_frates[utils.reverse_bl(bl)] = sim_w_frates[bl]
            sim_c_frates[utils.reverse_bl(bl)] = -sim_c_frates[bl]

        for bl in sim_c_frates:
            assert np.isclose(c_frs[bl], sim_c_frates[bl], atol=0.3, rtol=0.)
            assert np.isclose(w_frs[bl], sim_w_frates[bl], atol=0.3, rtol=0.)
        # test providing fringe-rate profiles.
        fr_grid, profiles = frf.build_fringe_rate_profiles(uvd, uvb)
        c_frs, w_frs = frf.get_fringe_rate_limits(uvd, frate_profiles=profiles)
        for bl in sim_c_frates:
            assert np.isclose(c_frs[bl], sim_c_frates[bl], atol=0.3, rtol=0.)
            assert np.isclose(w_frs[bl], sim_w_frates[bl], atol=0.3, rtol=0.)
        assert pytest.raises(ValueError, frf.get_fringe_rate_limits, uvd, None, None)
        assert pytest.raises(ValueError, frf.get_fringe_rate_limits, uvd, uvb, frate_profiles=profiles)

    @pytest.mark.parametrize(
        "pre_filter",
        [True, False],
    )
    def test_load_tophat_frfilter_and_write_beam_frates(self, tmpdir, pre_filter):
        # simulations constructed with the notebook at https://drive.google.com/file/d/1jPPSmL3nqQbp7tTgP77j9KC0802iWyow/view?usp=sharing
        # load in primary beam model and isotropic noise model of sky.
        test_beam = os.path.join(DATA_PATH, "fr_unittest_beam.beamfits")
        test_data = os.path.join(DATA_PATH, "fr_unittest_data_ds.uvh5")
        tmp_path = tmpdir.strpath
        resid_outfilename = os.path.join(tmp_path, 'resid.uvh5')
        CLEAN_outfilename = os.path.join(tmp_path, 'model.uvh5')
        filled_outfilename = os.path.join(tmp_path, 'filled.uvh5')
        # perform cleaning.
        frf.load_tophat_frfilter_and_write(datafile_list=[test_data], beamfitsfile=test_beam, mode='dpss_leastsq', filled_outfilename=filled_outfilename,
                                           CLEAN_outfilename=CLEAN_outfilename, frate_standoff=0.075, pre_filter_modes_between_lobe_minimum_and_zero=pre_filter,
                                           res_outfilename=resid_outfilename, percentile_high=97.5, percentile_low=2.5, case='uvbeam')
        hd_input = io.HERAData(test_data)
        data, flags, nsamples = hd_input.read()
        hd_resid = io.HERAData(resid_outfilename)
        data_r, flags_r, nsamples_r = hd_resid.read()
        hd_filled = io.HERAData(filled_outfilename)
        data_f, flags_f, nsamples_f = hd_filled.read()

        for bl in data:
            assert np.mean(np.abs(data_r[bl]) ** 2.) <= .2 * np.mean(np.abs(data[bl]) ** 2.)

    def test_sky_frates_minfrate_and_to_filter(self):
        # test edge frates
        V = frf.FRFilter(os.path.join(DATA_PATH, "PyGSM_Jy_downselect.uvh5"))
        V.read()
        for to_filter in [None, list(V.data.keys())[:1]]:
            cfrates, wfrates = frf.sky_frates(V.hd, min_frate_width=1000, blkeys=to_filter)
            # to_filter set to None -> all keys should be present.
            if to_filter is None:
                for k in V.data:
                    assert k in cfrates
                    assert k in wfrates
            # min_frate = 1000 should set all wfrates to 1000
            for k in cfrates:
                assert wfrates[k] == 1000.

    def test_load_tophat_frfilter_and_write(self, tmpdir):
        tmp_path = tmpdir.strpath
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        outfilename = os.path.join(tmp_path, 'temp.h5')
        CLEAN_outfilename = os.path.join(tmp_path, 'temp_clean.h5')
        filled_outfilename = os.path.join(tmp_path, 'temp_filled.h5')
        frf.load_tophat_frfilter_and_write(uvh5, res_outfilename=outfilename, tol=1e-4, clobber=True, Nbls_per_load=1, case='sky')
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
        for bl in d:
            assert not np.all(np.isclose(d[bl], 0.))

        frfil = frf.FRFilter(uvh5, filetype='uvh5')
        frfil.read(bls=[(53, 54, 'ee')])
        frate_centers, frate_half_widths = frf.select_tophat_frates(case='sky', uvd=frfil.hd, blvecs=frfil.blvecs)
        frfil.tophat_frfilter(keys=[(53, 54, 'ee')], tol=1e-4, verbose=True,
                              frate_centers=frate_centers, frate_half_widths=frate_half_widths)
        np.testing.assert_almost_equal(d[(53, 54, 'ee')], frfil.clean_resid[(53, 54, 'ee')], decimal=5)
        np.testing.assert_array_equal(f[(53, 54, 'ee')], frfil.flags[(53, 54, 'ee')])
        # test NotImplementedError
        pytest.raises(NotImplementedError, frf.load_tophat_frfilter_and_write, uvh5, 'sky', res_outfilename=outfilename, tol=1e-4,
                      clobber=True, Nbls_per_load=1, avg_red_bllens=True, baseline_list=[(54, 54)], polarizations=['ee'])

    def test_load_tophat_frfilter_and_write_all_baselines(self, tmpdir):
        tmp_path = tmpdir.strpath
        CLEAN_outfilename = os.path.join(tmp_path, 'temp_clean.h5')
        filled_outfilename = os.path.join(tmp_path, 'temp_filled.h5')
        # test loading and writing all baselines at once.
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        outfilename = os.path.join(tmp_path, 'temp.h5')
        for avg_bl in [True, False]:
            frf.load_tophat_frfilter_and_write(uvh5, res_outfilename=outfilename, tol=1e-4, clobber=True,
                                               Nbls_per_load=None, avg_red_bllens=avg_bl, case='sky')
            hd = io.HERAData(outfilename)
            d, f, n = hd.read(bls=[(53, 54, 'ee')])
            for bl in d:
                assert not np.all(np.isclose(d[bl], 0.))

        frfil = frf.FRFilter(uvh5, filetype='uvh5')
        frfil.read(bls=[(53, 54, 'ee')])
        frate_centers, frate_half_widths = frf.select_tophat_frates(case='sky', uvd=frfil.hd, blvecs=frfil.blvecs)
        frfil.tophat_frfilter(keys=[(53, 54, 'ee')], tol=1e-4, verbose=True, frate_centers=frate_centers, frate_half_widths=frate_half_widths)
        np.testing.assert_almost_equal(d[(53, 54, 'ee')], frfil.clean_resid[(53, 54, 'ee')], decimal=5)
        np.testing.assert_array_equal(f[(53, 54, 'ee')], frfil.flags[(53, 54, 'ee')])

    def test_load_tophat_frfilter_and_write_cal(self, tmpdir):
        tmp_path = tmpdir.strpath
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        outfilename = os.path.join(tmp_path, 'temp.h5')
        CLEAN_outfilename = os.path.join(tmp_path, 'temp_clean.h5')
        filled_outfilename = os.path.join(tmp_path, 'temp_filled.h5')
        cal = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only")
        for avg_bl in [True, False]:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Antenna 53 not present in calibration solution")
                warnings.filterwarnings("ignore", message='Cannot preserve total_quality_array')
                frf.load_tophat_frfilter_and_write(
                    uvh5, calfile_list=cal, tol=1e-4, res_outfilename=outfilename,
                    Nbls_per_load=2, clobber=True, avg_red_bllens=avg_bl, case='sky'
                )
            hd = io.HERAData(outfilename)
            assert 'Thisfilewasproducedbythefunction' in hd.history.replace('\n', '').replace(' ', '')
            d, f, n = hd.read()
            for bl in d:
                if not np.all(f[bl]):
                    assert not np.all(np.isclose(d[bl], 0.))
            np.testing.assert_array_equal(f[(53, 54, 'ee')], True)
            os.remove(outfilename)

    def test_load_tophat_frfilter_and_write_wgt_by_nsample(self, tmpdir):
        tmp_path = tmpdir.strpath
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        outfilename = os.path.join(tmp_path, 'temp.h5')
        CLEAN_outfilename = os.path.join(tmp_path, 'temp_clean.h5')
        filled_outfilename = os.path.join(tmp_path, 'temp_filled.h5')
        # test wgt_by_nsample
        outfilename = os.path.join(tmp_path, 'temp.h5')
        frf.load_tophat_frfilter_and_write(uvh5, res_outfilename=outfilename, tol=1e-4, clobber=True,
                                           Nbls_per_load=None, avg_red_bllens=False, case='sky', wgt_by_nsample=True)
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
        for bl in d:
            assert not np.all(np.isclose(d[bl], 0.))

        frfil = frf.FRFilter(uvh5, filetype='uvh5')
        frfil.read(bls=[(53, 54, 'ee')])

        wgts = datacontainer.DataContainer({k: (~frfil.flags[k]).astype(float) for k in frfil.flags})
        for k in wgts:
            wgts[k] *= frfil.nsamples[k]
        frate_centers, frate_half_widths = frf.select_tophat_frates(case='sky', uvd=frfil.hd, blvecs=frfil.blvecs)
        frfil.tophat_frfilter(keys=[(53, 54, 'ee')], tol=1e-4, verbose=True, frate_centers=frate_centers, frate_half_widths=frate_half_widths, wgts=wgts)
        np.testing.assert_almost_equal(d[(53, 54, 'ee')], frfil.clean_resid[(53, 54, 'ee')], decimal=5)
        np.testing.assert_array_equal(f[(53, 54, 'ee')], frfil.flags[(53, 54, 'ee')])

    def test_load_tophat_frfilter_and_write_lst_blacklist(self, tmpdir):
        # test lst_blacklists
        tmp_path = tmpdir.strpath
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        outfilename = os.path.join(tmp_path, 'temp.h5')
        CLEAN_outfilename = os.path.join(tmp_path, 'temp_clean.h5')
        filled_outfilename = os.path.join(tmp_path, 'temp_filled.h5')
        frf.load_tophat_frfilter_and_write(uvh5, res_outfilename=outfilename, tol=1e-4, clobber=True,
                                           Nbls_per_load=None, avg_red_bllens=False, case='sky', lst_blacklists=[(23, 6.03), (6.18, 7)], blacklist_wgt=1e-3)  # first and last five bins
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
        for bl in d:
            assert not np.all(np.isclose(d[bl], 0.))

        frfil = frf.FRFilter(uvh5, filetype='uvh5')
        frfil.read(bls=[(53, 54, 'ee')])

        wgts = datacontainer.DataContainer({k: (~frfil.flags[k]).astype(float) for k in frfil.flags})
        for k in wgts:
            wgts[k][0:5, :] = 1e-3
            wgts[k][-5:, :] = 1e-3
        frate_centers, frate_half_widths = frf.select_tophat_frates(case='sky', uvd=frfil.hd, blvecs=frfil.blvecs)
        frfil.tophat_frfilter(keys=[(53, 54, 'ee')], tol=1e-4, verbose=True, frate_centers=frate_centers, frate_half_widths=frate_half_widths, wgts=wgts)
        np.testing.assert_almost_equal(d[(53, 54, 'ee')], frfil.clean_resid[(53, 54, 'ee')], decimal=5)
        np.testing.assert_array_equal(f[(53, 54, 'ee')], frfil.flags[(53, 54, 'ee')])

    def test_load_tophat_frfilter_and_write_skip_autos(self, tmpdir):
        tmp_path = tmpdir.strpath
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        outfilename = os.path.join(tmp_path, 'temp.h5')
        CLEAN_outfilename = os.path.join(tmp_path, 'temp_clean.h5')
        filled_outfilename = os.path.join(tmp_path, 'temp_filled.h5')
        # test skip_autos

        frf.load_tophat_frfilter_and_write(uvh5, calfile_list=None, tol=1e-4, res_outfilename=outfilename,
                                           filled_outfilename=filled_outfilename, CLEAN_outfilename=CLEAN_outfilename,
                                           Nbls_per_load=2, clobber=True, skip_autos=True, case='sky')
        hd = io.HERAData(outfilename)
        d, f, n = hd.read()
        hd_original = io.HERAData(uvh5)
        do, fo, no = hd_original.read()
        chd = io.HERAData(CLEAN_outfilename)
        cd, cf, cn = chd.read()
        fhd = io.HERAData(filled_outfilename)
        fd, ff, fn = fhd.read()
        # test that the resids are are equal to original data.
        for bl in do:
            if bl[0] == bl[1]:
                assert np.allclose(do[bl], d[bl])  # check that resid equals original data.
                assert np.allclose(fo[bl], f[bl])
                assert np.allclose(no[bl], n[bl])
                assert np.allclose(cd[bl], np.zeros_like(cd[bl]))  # check that all model values are zero.
                assert np.allclose(fd[bl][~f[bl]], d[bl][~f[bl]])  # check that filled data equals original data.
            else:
                assert not np.allclose(do[bl], d[bl])
                assert np.allclose(no[bl], n[bl])

    def test_load_tophat_frfilter_and_write_broadcast_flags(self, tmpdir):
        # test flag broadcasting with frf.
        tmp_path = tmpdir.strpath
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        outfilename = os.path.join(tmp_path, 'temp.h5')
        CLEAN_outfilename = os.path.join(tmp_path, 'temp_clean.h5')
        filled_outfilename = os.path.join(tmp_path, 'temp_filled.h5')

        # prepare an input file for broadcasting flags
        input_file = os.path.join(tmp_path, 'temp_special_flags.h5')
        shutil.copy(uvh5, input_file)
        hd = io.HERAData(input_file)
        _, flags, _ = hd.read()
        ntimes_before = hd.Ntimes
        nfreqs_before = hd.Nfreqs
        freqs_before = hd.freqs
        times_before = hd.times
        for bl in flags:
            flags[bl][:] = False
            flags[bl][0, :hd.Nfreqs // 2] = True  # first time has 50% flagged
            flags[bl][-3:, -1] = True  # last channel has flags for three integrations
        hd.update(flags=flags)
        hd.write_uvh5(input_file, clobber=True)
        # this time_threshold will result in
        # entire first integration begin flagged
        # and entire final channel being flagged
        # when flags are broadcasted.
        time_thresh = 2. / hd.Ntimes
        frf.load_tophat_frfilter_and_write(input_file, res_outfilename=outfilename, tol=1e-4, case='sky',
                                           factorize_flags=True, time_thresh=time_thresh, clobber=True)
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
        for bl in f:
            assert np.any(f[bl][:, :-1])
            assert np.all(f[bl][0, :])

    def test_load_tophat_frfilter_and_write_partial_io(self, tmpdir):
        tmp_path = tmpdir.strpath
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        outfilename = os.path.join(tmp_path, 'temp.h5')
        CLEAN_outfilename = os.path.join(tmp_path, 'temp_clean.h5')
        filled_outfilename = os.path.join(tmp_path, 'temp_filled.h5')
        # prepare an input file for broadcasting flags
        input_file = os.path.join(tmp_path, 'temp_special_flags.h5')
        shutil.copy(uvh5, input_file)
        hd = io.HERAData(input_file)
        _, flags, _ = hd.read()
        ntimes_before = hd.Ntimes
        nfreqs_before = hd.Nfreqs
        freqs_before = hd.freqs
        times_before = hd.times
        for bl in flags:
            flags[bl][:] = False
            flags[bl][0, :hd.Nfreqs // 2] = True  # first time has 50% flagged
            flags[bl][-3:, -1] = True  # last channel has flags for three integrations
        hd.update(flags=flags)
        hd.write_uvh5(input_file, clobber=True)
        # this time_threshold will result in
        # entire first integration begin flagged
        # and entire final channel being flagged
        # when flags are broadcasted.
        time_thresh = 2. / hd.Ntimes
        # test delay filtering and writing with factorized flags and partial i/o
        time_thresh = 2. / hd.Ntimes

        frf.load_tophat_frfilter_and_write(input_file, res_outfilename=outfilename, tol=1e-4, case='sky',
                                           factorize_flags=True, time_thresh=time_thresh, clobber=True)
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
        for bl in f:
            # check that flags were broadcasted.
            assert np.all(f[bl][0, :])
            assert np.all(f[bl][:, -1])
            assert not np.all(np.isclose(d[bl], 0.))

        frf.load_tophat_frfilter_and_write(input_file, res_outfilename=outfilename, tol=1e-4, Nbls_per_load=1,
                                           factorize_flags=True, time_thresh=time_thresh, clobber=True, case='sky')
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
        for bl in f:
            # check that flags were broadcasted.
            assert np.all(f[bl][0, :])
            assert np.all(f[bl][:, -1])
            assert not np.all(np.isclose(d[bl], 0.))

    def test_load_tophat_frfilter_and_write_yaml(self, tmpdir):
        # test apriori flags and flag_yaml
        tmp_path = tmpdir.strpath
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        outfilename = os.path.join(tmp_path, 'temp.h5')
        CLEAN_outfilename = os.path.join(tmp_path, 'temp_clean.h5')
        filled_outfilename = os.path.join(tmp_path, 'temp_filled.h5')
        hd = io.HERAData(uvh5)
        hd.read()
        flag_yaml = os.path.join(DATA_PATH, 'test_input/a_priori_flags_sample.yaml')
        uvf = UVFlag(hd, mode='flag', copy_flags=True, use_future_array_shapes=True)
        uvf.to_waterfall(keep_pol=False, method='and')
        uvf.flag_array[:] = False
        flagfile = os.path.join(tmp_path, 'test_flag.h5')
        uvf.write(flagfile, clobber=True)
        frf.load_tophat_frfilter_and_write(uvh5, res_outfilename=outfilename,
                                           Nbls_per_load=1, clobber=True, mode='dayenu',
                                           external_flags=flagfile,
                                           overwrite_flags=True, case='sky')
        # test that all flags are False
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
        for k in f:
            assert np.all(~f[k])
        # now without parital io.
        frf.load_tophat_frfilter_and_write(uvh5, res_outfilename=outfilename,
                                           clobber=True, mode='dayenu',
                                           external_flags=flagfile,
                                           overwrite_flags=True, case='sky')
        # test that all flags are False
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
        for k in f:
            assert np.all(~f[k])

    @pytest.mark.parametrize(
        "ninterleave", [2, 3, 4, 5, 6])
    def test_load_tophat_frfilter_and_write_interleave(self, tmpdir, ninterleave):
        tmp_path = tmpdir.strpath
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        outfilename = os.path.join(tmp_path, 'temp.h5')
        frf.load_tophat_frfilter_and_write(uvh5, res_outfilename=outfilename,
                                           Nbls_per_load=1, clobber=True, mode='dpss_leastsq',
                                           overwrite_flags=True, ninterleave=ninterleave,
                                           max_frate_coeffs=[0.025, 0.0],
                                           case='max_frate_coeffs')
        # check that output exists.
        assert os.path.exists(outfilename)
        # check that it has the original number of times.

    def test_sky_frates_minfrate_and_to_filter(self):
        # test edge frates
        V = frf.FRFilter(os.path.join(DATA_PATH, "PyGSM_Jy_downselect.uvh5"))
        V.read()
        for to_filter in [None, list(V.data.keys())[:1]]:
            cfrates, wfrates = frf.sky_frates(uvd=V.hd, min_frate_half_width=1000, keys=to_filter)
            # to_filter set to None -> all keys should be present.
            if to_filter is None:
                for k in V.data:
                    assert k in cfrates
                    assert k in wfrates
            # min_frate = 1000 should set all wfrates to 1000
            for k in cfrates:
                assert wfrates[k] == 1000.

    def test_load_dayenu_filter_and_write(self, tmpdir):
        tmp_path = tmpdir.strpath
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        cdir = os.path.join(tmp_path, 'cache_temp')
        # make a cache directory
        if os.path.isdir(cdir):
            shutil.rmtree(cdir)
        os.mkdir(cdir)
        outfilename = os.path.join(tmp_path, 'temp.h5')
        # run dayenu filter
        avg_bl = True
        frf.load_tophat_frfilter_and_write(uvh5, res_outfilename=outfilename,
                                           cache_dir=cdir, mode='dayenu', case='sky',
                                           Nbls_per_load=1, clobber=True, avg_red_bllens=avg_bl,
                                           spw_range=(0, 32), write_cache=True)
        # generate duplicate cache files to test duplicate key handle for cache load.
        frf.load_tophat_frfilter_and_write(uvh5, res_outfilename=outfilename, cache_dir=cdir,
                                           mode='dayenu', avg_red_bllens=avg_bl, case='sky',
                                           Nbls_per_load=1, clobber=True, read_cache=False,
                                           spw_range=(0, 32), write_cache=True)
        # there should now be six cache files (one per i/o/filter). There are three baselines.
        assert len(glob.glob(cdir + '/*')) == 6
        hd = io.HERAData(outfilename)
        assert 'Thisfilewasproducedbythefunction' in hd.history.replace('\n', '').replace(' ', '')
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
        np.testing.assert_array_equal(f[(53, 54, 'ee')], True)
        os.remove(outfilename)
        shutil.rmtree(cdir)
        os.mkdir(cdir)
        # now do all the baselines at once.
        for avg_bl in [True, False]:
            frf.load_tophat_frfilter_and_write(uvh5, res_outfilename=outfilename,
                                               cache_dir=cdir, mode='dayenu', avg_red_bllens=avg_bl,
                                               Nbls_per_load=None, clobber=True, case='sky',
                                               spw_range=(0, 32), write_cache=True)
            if avg_bl:
                assert len(glob.glob(cdir + '/*')) == 1
            hd = io.HERAData(outfilename)
            assert 'Thisfilewasproducedbythefunction' in hd.history.replace('\n', '').replace(' ', '')
            d, f, n = hd.read(bls=[(53, 54, 'ee')])
            np.testing.assert_array_equal(f[(53, 54, 'ee')], True)
            os.remove(outfilename)
        shutil.rmtree(cdir)
        os.mkdir(cdir)
        # run again using computed cache.
        calfile = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Antenna 53 not present in calibration solution")
            warnings.filterwarnings("ignore", message='Cannot preserve total_quality_array')
            frf.load_tophat_frfilter_and_write(
                uvh5, res_outfilename=outfilename, max_frate_coeffs=[0.0, 0.025],
                cache_dir=cdir, calfile_list=calfile, read_cache=True,
                Nbls_per_load=1, clobber=True, mode='dayenu',
                spw_range=(0, 32), write_cache=True, case='max_frate_coeffs'
            )
        # no new cache files should be generated.
        assert len(glob.glob(cdir + '/*')) == 1
        hd = io.HERAData(outfilename)
        assert 'Thisfilewasproducedbythefunction' in hd.history.replace('\n', '').replace(' ', '')
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
        np.testing.assert_array_equal(f[(53, 54, 'ee')], True)
        os.remove(outfilename)
        shutil.rmtree(cdir)

    @pytest.mark.parametrize("flip", [True, False])
    def test_load_tophat_frfilter_and_write_with_filter_yaml(self, tmpdir, flip):
        tmp_path = tmpdir.strpath
        uvh5 = os.path.join(
            DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5"
        )
        frate_centers = {(53, 54): 0.1}
        frate_half_widths = {(53, 54): 0.05}
        if flip:
            filter_info = dict(
                filter_centers={
                    str(k[::-1]): -v for k, v in frate_centers.items()
                },
                filter_half_widths={
                    str(k[::-1]): v for k, v in frate_half_widths.items()
                },
            )
        else:
            filter_info = dict(
                filter_centers={str(k): v for k, v in frate_centers.items()},
                filter_half_widths={str(k): v for k, v in frate_half_widths.items()},
            )
        frate_centers = {k + ("ee",): v for k, v in frate_centers.items()}
        frate_half_widths = {k + ("ee",): v for k, v in frate_half_widths.items()}

        with open(tmpdir / "filter_info.yaml", "w") as f:
            yaml.dump(filter_info, f)

        outfilename = os.path.join(tmp_path, 'temp.h5')
        frf.load_tophat_frfilter_and_write(
            uvh5,
            res_outfilename=outfilename,
            tol=1e-4,
            clobber=True,
            Nbls_per_load=1,
            param_file=str(tmpdir / "filter_info.yaml"),
            case="param_file",
            skip_autos=True,
        )
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
        for bl in d:
            assert not np.allclose(d[bl], 0)

        frfil = frf.FRFilter(uvh5, filetype='uvh5')
        frfil.read(bls=[(53, 54, 'ee')])
        frfil.tophat_frfilter(
            keys=[(53, 54, 'ee')],
            tol=1e-4,
            verbose=True,
            frate_centers=frate_centers,
            frate_half_widths=frate_half_widths,
        )

        # Check that the filtered data both ways matches.
        np.testing.assert_almost_equal(
            d[(53, 54, 'ee')], frfil.clean_resid[(53, 54, 'ee')], decimal=5
        )

        # Check that the flags match.
        np.testing.assert_array_equal(f[(53, 54, 'ee')], frfil.flags[(53, 54, 'ee')])

    def test_load_tophat_frfilter_and_write_with_bad_filter_yaml(self, tmpdir):
        tmp_path = tmpdir.strpath
        uvh5 = os.path.join(
            DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5"
        )
        param_file = os.path.join(DATA_PATH, "example_filter_params.yaml")

        outfilename = os.path.join(tmp_path, 'temp.h5')
        with pytest.raises(ValueError, match="(53, 54)"):
            frf.load_tophat_frfilter_and_write(
                uvh5,
                res_outfilename=outfilename,
                tol=1e-4,
                clobber=True,
                Nbls_per_load=1,
                param_file=param_file,
                case="param_file",
            )

    def test_tophat_clean_argparser(self):
        sys.argv = [sys.argv[0], 'a', '--clobber', '--window', 'blackmanharris', '--max_frate_coeffs', '0.024', '-0.229']
        parser = frf.tophat_frfilter_argparser()
        a = parser.parse_args()
        assert a.datafilelist == ['a']
        assert a.clobber is True
        assert a.window == 'blackmanharris'
        assert a.max_frate_coeffs[0] == 0.024
        assert a.max_frate_coeffs[1] == -0.229
        assert a.time_thresh == 0.05
        assert not a.factorize_flags

    def test_tophat_linear_argparser(self):
        sys.argv = [sys.argv[0], 'a', '--clobber', '--write_cache', '--cache_dir', '/blah/', '--max_frate_coeffs', '0.024', '-0.229', '--mode', 'dayenu']
        parser = frf.tophat_frfilter_argparser()
        a = parser.parse_args()
        assert a.datafilelist == ['a']
        assert a.clobber is True
        assert a.write_cache is True
        assert a.cache_dir == '/blah/'
        assert a.max_frate_coeffs[0] == 0.024
        assert a.max_frate_coeffs[1] == -0.229
        assert a.time_thresh == 0.05
        assert not a.factorize_flags
        parser = frf.tophat_frfilter_argparser()
        a = parser.parse_args()
        assert a.datafilelist == ['a']
        assert a.clobber is True
        assert a.write_cache is True
        assert a.cache_dir == '/blah/'
        assert a.max_frate_coeffs[0] == 0.024
        assert a.max_frate_coeffs[1] == -0.229
        assert a.time_thresh == 0.05
        assert not a.factorize_flags

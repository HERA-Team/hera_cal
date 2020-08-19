# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import pytest
import numpy as np
from copy import deepcopy
import os
import sys
import shutil
from scipy import constants
from pyuvdata import UVCal, UVData

from .. import io
from .. import xtalk_filter as xf
from ..data import DATA_PATH
import glob
from .. import vis_clean
from .. import utils as utils


class Test_XTalkFilter(object):
    def test_run_xtalk_filter(self):
        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        k = (24, 25, 'ee')
        xfil = xf.XTalkFilter(fname, filetype='miriad')
        xfil.read(bls=[k])
        bl = np.linalg.norm(xfil.antpos[24] - xfil.antpos[25]) / constants.c * 1e9
        sdf = (xfil.freqs[1] - xfil.freqs[0]) / 1e9

        xfil.run_xtalk_filter(to_filter=xfil.data.keys(), tol=1e-2)
        for k in xfil.data.keys():
            assert xfil.clean_resid[k].shape == (60, 64)
            assert xfil.clean_model[k].shape == (60, 64)
            assert k in xfil.clean_info

        # test skip_wgt imposition of flags
        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        k = (24, 25, 'ee')
        # check successful run when round_up_bllens is True and when False.
        for round_up_bllens in [True, False]:
            xfil = xf.XTalkFilter(fname, filetype='miriad', round_up_bllens=round_up_bllens)
            xfil.read(bls=[k])
            wgts = {k: np.ones_like(xfil.flags[k], dtype=np.float)}
            wgts[k][:, 0] = 0.0
            xfil.run_xtalk_filter(to_filter=[k], weight_dict=wgts, tol=1e-5, window='blackman-harris', skip_wgt=0.1, maxiter=100)
            assert xfil.clean_info[k]['status']['axis_0'][0] == 'skipped'
            np.testing.assert_array_equal(xfil.clean_flags[k][:, 0], np.ones_like(xfil.flags[k][:, 0]))
            np.testing.assert_array_equal(xfil.clean_model[k][:, 0], np.zeros_like(xfil.clean_resid[k][:, 0]))
            np.testing.assert_array_equal(xfil.clean_resid[k][:, 0], np.zeros_like(xfil.clean_resid[k][:, 0]))

    def test_load_xtalk_filter_and_write_baseline_list(self, tmpdir):
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
        xf.load_xtalk_filter_and_write_baseline_list(datafile_list=uvh5, baseline_list=[(53, 54, 'ee')],
                                                     calfile_list=cals, spw_range=[100, 200], cache_dir=cdir,
                                                     read_cache=True, write_cache=True,
                                                     res_outfilename=outfilename, clobber=True,
                                                     mode='dayenu')
        hd = io.HERAData(outfilename)
        d, f, n = hd.read()
        assert len(list(d.keys())) == 1
        assert d[(53, 54, 'ee')].shape[1] == 100
        assert d[(53, 54, 'ee')].shape[0] == 60
        # now do no spw range and no cal files just to cover those lines.
        xf.load_xtalk_filter_and_write_baseline_list(datafile_list=uvh5, baseline_list=[(53, 54, 'ee')],
                                                     cache_dir=cdir,
                                                     read_cache=True, write_cache=True,
                                                     res_outfilename=outfilename, clobber=True,
                                                     mode='dayenu')
        hd = io.HERAData(outfilename)
        d, f, n = hd.read()
        assert len(list(d.keys())) == 1
        assert d[(53, 54, 'ee')].shape[1] == 1024
        assert d[(53, 54, 'ee')].shape[0] == 60
        # now test flag factorization and time thresholding.
        # prepare an input files for broadcasting flags and trim_edges.
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
            xf.load_xtalk_filter_and_write_baseline_list(datafile_list=[input_file], res_outfilename=outfilename,
                                                         tol=1e-4, trim_edges=True, baseline_list=[bl],
                                                         cache_dir=cdir,
                                                         factorize_flags=True, time_thresh=time_thresh, clobber=True)
        # now load all of the outputs in
        output_files = glob.glob(tmp_path + '/bl_chunk_*.h5')
        hd = io.HERAData(output_files)
        d, f, n = hd.read()
        hd_original = io.HERAData(uvh5)
        for bl in hd_original.bls:
            assert bl in d.keys()

        assert hd.Ntimes == ntimes_before - 1
        assert hd.Nfreqs == nfreqs_before - 1
        assert np.all(np.isclose(hd.freq_array.squeeze(), freqs_before[:-1]))
        assert np.all(np.isclose(np.unique(hd.time_array), times_before[1:]))
        for bl in f:
            assert not np.any(f[bl])
            assert not np.all(np.isclose(d[bl], 0.))

    def test_load_xtalk_filter_and_write(self, tmpdir):
        tmp_path = tmpdir.strpath
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        outfilename = os.path.join(tmp_path, 'temp.h5')
        xf.load_xtalk_filter_and_write(uvh5, res_outfilename=outfilename, tol=1e-4, clobber=True, Nbls_per_load=1)
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
        for bl in d:
            assert not np.all(np.isclose(d[bl], 0.))

        xfil = xf.XTalkFilter(uvh5, filetype='uvh5')
        xfil.read(bls=[(53, 54, 'ee')])
        xfil.run_xtalk_filter(to_filter=[(53, 54, 'ee')], tol=1e-4, verbose=True)
        np.testing.assert_almost_equal(d[(53, 54, 'ee')], xfil.clean_resid[(53, 54, 'ee')], decimal=5)
        np.testing.assert_array_equal(f[(53, 54, 'ee')], xfil.flags[(53, 54, 'ee')])

        # test loading and writing all baselines at once.
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        outfilename = os.path.join(tmp_path, 'temp.h5')
        xf.load_xtalk_filter_and_write(uvh5, res_outfilename=outfilename, tol=1e-4, clobber=True, Nbls_per_load=None)
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
        for bl in d:
            assert not np.all(np.isclose(d[bl], 0.))

        xfil = xf.XTalkFilter(uvh5, filetype='uvh5')
        xfil.read(bls=[(53, 54, 'ee')])
        xfil.run_xtalk_filter(to_filter=[(53, 54, 'ee')], tol=1e-4, verbose=True)
        np.testing.assert_almost_equal(d[(53, 54, 'ee')], xfil.clean_resid[(53, 54, 'ee')], decimal=5)
        np.testing.assert_array_equal(f[(53, 54, 'ee')], xfil.flags[(53, 54, 'ee')])

        cal = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only")
        outfilename = os.path.join(tmp_path, 'temp.h5')
        os.remove(outfilename)
        xf.load_xtalk_filter_and_write(uvh5, calfile=cal, tol=1e-4, res_outfilename=outfilename,
                                       Nbls_per_load=2, clobber=True)
        hd = io.HERAData(outfilename)
        assert 'Thisfilewasproducedbythefunction' in hd.history.replace('\n', '').replace(' ', '')
        d, f, n = hd.read()
        for bl in d:
            if not np.all(f[bl]):
                assert not np.all(np.isclose(d[bl], 0.))
        np.testing.assert_array_equal(f[(53, 54, 'ee')], True)
        os.remove(outfilename)

        # prepare an input file for broadcasting flags and trim_edges.
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
        xf.load_xtalk_filter_and_write(input_file, res_outfilename=outfilename, tol=1e-4, trim_edges=True,
                                       factorize_flags=True, time_thresh=time_thresh, clobber=True)
        hd = io.HERAData(outfilename)
        assert hd.Ntimes == ntimes_before - 1
        assert hd.Nfreqs == nfreqs_before - 1
        assert np.all(np.isclose(hd.freqs, freqs_before[:-1]))
        assert np.all(np.isclose(hd.times, times_before[1:]))
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
        for bl in f:
            assert not np.any(f[bl])
            assert not np.all(np.isclose(d[bl], 0.))

        # test delay filtering and writing with factorized flags and partial i/o
        xf.load_xtalk_filter_and_write(input_file, res_outfilename=outfilename, tol=1e-4,
                                       factorize_flags=True, time_thresh=time_thresh, clobber=True)
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
        for bl in f:
            # check that flags were broadcasted.
            assert np.all(f[bl][0, :])
            assert np.all(f[bl][:, -1])
            assert not np.all(np.isclose(d[bl], 0.))

        # now test partial i/o not implemented
        pytest.raises(NotImplementedError, xf.load_xtalk_filter_and_write, input_file,
                      res_outfilename=outfilename, trim_edges=True, Nbls_per_load=1)

    def test_load_dayenu_filter_and_write(self, tmpdir):
        tmp_path = tmpdir.strpath
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        cdir = os.getcwd()
        cdir = os.path.join(cdir, 'cache_temp')
        # make a cache directory
        if os.path.isdir(cdir):
            shutil.rmtree(cdir)
        os.mkdir(cdir)
        outfilename = os.path.join(tmp_path, 'temp.h5')
        # run dayenu filter
        xf.load_xtalk_filter_and_write(uvh5, res_outfilename=outfilename,
                                       cache_dir=cdir, mode='dayenu',
                                       Nbls_per_load=1, clobber=True,
                                       spw_range=(0, 32), write_cache=True)
        # generate duplicate cache files to test duplicate key handle for cache load.
        xf.load_xtalk_filter_and_write(uvh5, res_outfilename=outfilename, cache_dir=cdir,
                                       mode='dayenu',
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
        xf.load_xtalk_filter_and_write(uvh5, res_outfilename=outfilename,
                                       cache_dir=cdir, mode='dayenu',
                                       Nbls_per_load=None, clobber=True,
                                       spw_range=(0, 32), write_cache=True)
        assert len(glob.glob(cdir + '/*')) == 1
        hd = io.HERAData(outfilename)
        assert 'Thisfilewasproducedbythefunction' in hd.history.replace('\n', '').replace(' ', '')
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
        np.testing.assert_array_equal(f[(53, 54, 'ee')], True)
        os.remove(outfilename)
        # run again using computed cache.
        calfile = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only")
        xf.load_xtalk_filter_and_write(uvh5, res_outfilename=outfilename,
                                       cache_dir=cdir, calfile=calfile, read_cache=True,
                                       Nbls_per_load=1, clobber=True, mode='dayenu',
                                       spw_range=(0, 32), write_cache=True)
        # now new cache files should be generated.
        assert len(glob.glob(cdir + '/*')) == 1
        hd = io.HERAData(outfilename)
        assert 'Thisfilewasproducedbythefunction' in hd.history.replace('\n', '').replace(' ', '')
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
        np.testing.assert_array_equal(f[(53, 54, 'ee')], True)
        os.remove(outfilename)
        shutil.rmtree(cdir)

    def test_xtalk_clean_argparser(self):
        sys.argv = [sys.argv[0], 'a', '--clobber', '--window', 'blackmanharris', '--max_frate_coeffs', '0.024', '-0.229']
        parser = xf.xtalk_filter_argparser()
        a = parser.parse_args()
        assert a.infilename == 'a'
        assert a.clobber is True
        assert a.window == 'blackmanharris'
        assert a.max_frate_coeffs[0] == 0.024
        assert a.max_frate_coeffs[1] == -0.229
        assert not a.trim_edges
        assert a.time_thresh == 0.05
        assert not a.factorize_flags

    def test_xtalk_linear_argparser(self):
        sys.argv = [sys.argv[0], 'a', '--clobber', '--write_cache', '--cache_dir', '/blah/', '--max_frate_coeffs', '0.024', '-0.229']
        parser = xf.xtalk_filter_argparser(mode='dayenu')
        a = parser.parse_args()
        assert a.infilename == 'a'
        assert a.clobber is True
        assert a.write_cache is True
        assert a.cache_dir == '/blah/'
        assert a.max_frate_coeffs[0] == 0.024
        assert a.max_frate_coeffs[1] == -0.229
        assert not a.trim_edges
        assert a.time_thresh == 0.05
        assert not a.factorize_flags

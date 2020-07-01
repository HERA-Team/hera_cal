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

    def test_load_xtalk_filter_and_write_baseline_list(self):
        uvh5 = [os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.first.uvh5"),
                os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.second.uvh5")]
        cals = [os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only.part1"),
                os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only.part2")]
        outfilename = os.path.join(DATA_PATH, 'test_output/temp.h5')
        cdir = os.getcwd()
        cdir = os.path.join(cdir, 'cache_temp')
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

    def test_load_xtalk_filter_and_write(self):
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        outfilename = os.path.join(DATA_PATH, 'test_output/temp.h5')
        xf.load_xtalk_filter_and_write(uvh5, res_outfilename=outfilename, tol=1e-4, clobber=True, Nbls_per_load=1)
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'ee')])

        xfil = xf.XTalkFilter(uvh5, filetype='uvh5')
        xfil.read(bls=[(53, 54, 'ee')])
        xfil.run_xtalk_filter(to_filter=[(53, 54, 'ee')], tol=1e-4, verbose=True)
        np.testing.assert_almost_equal(d[(53, 54, 'ee')], xfil.clean_resid[(53, 54, 'ee')], decimal=5)
        np.testing.assert_array_equal(f[(53, 54, 'ee')], xfil.flags[(53, 54, 'ee')])

        # test loading and writing all baselines at once.
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        outfilename = os.path.join(DATA_PATH, 'test_output/temp.h5')
        xf.load_xtalk_filter_and_write(uvh5, res_outfilename=outfilename, tol=1e-4, clobber=True, Nbls_per_load=None)
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'ee')])

        xfil = xf.XTalkFilter(uvh5, filetype='uvh5')
        xfil.read(bls=[(53, 54, 'ee')])
        xfil.run_xtalk_filter(to_filter=[(53, 54, 'ee')], tol=1e-4, verbose=True)
        np.testing.assert_almost_equal(d[(53, 54, 'ee')], xfil.clean_resid[(53, 54, 'ee')], decimal=5)
        np.testing.assert_array_equal(f[(53, 54, 'ee')], xfil.flags[(53, 54, 'ee')])

        cal = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only")
        outfilename = os.path.join(DATA_PATH, 'test_output/temp.h5')
        os.remove(outfilename)
        xf.load_xtalk_filter_and_write(uvh5, calfile=cal, tol=1e-4, res_outfilename=outfilename,
                                       Nbls_per_load=2, clobber=True)
        hd = io.HERAData(outfilename)
        assert 'Thisfilewasproducedbythefunction' in hd.history.replace('\n', '').replace(' ', '')
        d, f, n = hd.read()
        np.testing.assert_array_equal(f[(53, 54, 'ee')], True)
        os.remove(outfilename)

    def test_load_dayenu_filter_and_write(self):
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        cdir = os.getcwd()
        cdir = os.path.join(cdir, 'cache_temp')
        # make a cache directory
        if os.path.isdir(cdir):
            shutil.rmtree(cdir)
        os.mkdir(cdir)
        outfilename = os.path.join(DATA_PATH, 'test_output/temp.h5')
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

    def test_reconstitute_xtalk_files_argparser(self):
        sys.argv = [sys.argv[0], 'a', '--clobber', '--fragmentlist', 'a', 'b', 'c', 'd', '--outfilename', 'a.out']
        parser = xf.reconstitute_xtalk_files_argparser()
        a = parser.parse_args()
        assert a.clobber
        for char in ['a', 'b', 'c', 'd']:
            assert char in a.fragmentlist
        assert a.infilename == 'a'
        assert a.outfilename == 'a.out'

    def test_reconstitute_xtalk_files(self, tmp_path):
        # First, construct some cross-talk baseline files.
        datafiles = [os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.first.uvh5"),
                     os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.second.uvh5")]

        cals = [os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only.part1"),
                os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only.part2")]
        # make a cache directory
        cdir = tmp_path / "cache_temp"
        cdir.mkdir()
        # cross-talk filter chunked baselines
        for filenum, file in enumerate(datafiles):
            baselines = io.baselines_from_filelist_position(file, datafiles, polarizations=['ee'])
            fname = 'temp.fragment.part.%d.h5' % filenum
            fragment_filename = tmp_path / fname
            xf.load_xtalk_filter_and_write_baseline_list(datafiles, baseline_list=baselines, calfile_list=cals,
                                                         spw_range=[0, 20], cache_dir=cdir, read_cache=True, write_cache=True,
                                                         res_outfilename=fragment_filename, clobber=True)
            # load in fragment and make sure the number of baselines is equal to the length of the baseline list
            hd_fragment = io.HERAData(str(fragment_filename))
            assert len(hd_fragment.bls) == len(baselines)
            assert hd_fragment.Ntimes == 60
            assert hd_fragment.Nfreqs == 20

        fragments = glob.glob(DATA_PATH + '/test_output/temp.fragment.h5.part*')
        # reconstitute the filtered data
        for filenum, file in enumerate(datafiles):
            # reconstitute
            fname = 'temp.reconstituted.part.%d.h5' % filenum
            xf.reconstitute_xtalk_files(templatefile=file,
                                        fragments=glob.glob(str(tmp_path / 'temp.fragment.part.*.h5')), clobber=True,
                                        outfilename=str(tmp_path / fname))
        # load in the reconstituted files.
        hd_reconstituted = io.HERAData(glob.glob(str(tmp_path / 'temp.reconstituted.part.*.h5')))
        hd_reconstituted.read()
        # compare to xtalk filtering the whole file.
        xf.load_xtalk_filter_and_write(infilename=os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5"),
                                       calfile=os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only"),
                                       res_outfilename=str(tmp_path / 'temp.h5'), clobber=True, spw_range=[0, 20])
        hd = io.HERAData(str(tmp_path / 'temp.h5'))
        hd.read()
        assert np.all(np.isclose(hd.data_array, hd_reconstituted.data_array))

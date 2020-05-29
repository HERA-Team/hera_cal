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
from .. import delay_filter as df
from ..data import DATA_PATH
import glob


@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
@pytest.mark.filterwarnings("ignore:.*dspec.vis_filter will soon be deprecated")
@pytest.mark.filterwarnings("ignore:It seems that the latitude and longitude are in radians")
class Test_DelayFilter(object):
    def test_run_filter(self):
        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        k = (24, 25, 'ee')
        dfil = df.DelayFilter(fname, filetype='miriad')
        dfil.read(bls=[k])
        bl = np.linalg.norm(dfil.antpos[24] - dfil.antpos[25]) / constants.c * 1e9
        sdf = (dfil.freqs[1] - dfil.freqs[0]) / 1e9

        dfil.run_filter(to_filter=dfil.data.keys(), tol=1e-2)
        for k in dfil.data.keys():
            assert dfil.clean_resid[k].shape == (60, 64)
            assert dfil.clean_model[k].shape == (60, 64)
            assert k in dfil.clean_info

        # test skip_wgt imposition of flags
        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        k = (24, 25, 'ee')
        dfil = df.DelayFilter(fname, filetype='miriad')
        dfil.read(bls=[k])
        wgts = {k: np.ones_like(dfil.flags[k], dtype=np.float)}
        wgts[k][0, :] = 0.0
        dfil.run_filter(to_filter=[k], weight_dict=wgts, standoff=0., horizon=1., tol=1e-5, window='blackman-harris', skip_wgt=0.1, maxiter=100)
        assert dfil.clean_info[k]['status']['axis_1'][0] == 'skipped'
        np.testing.assert_array_equal(dfil.clean_flags[k][0, :], np.ones_like(dfil.flags[k][0, :]))
        np.testing.assert_array_equal(dfil.clean_model[k][0, :], np.zeros_like(dfil.clean_resid[k][0, :]))
        np.testing.assert_array_equal(dfil.clean_resid[k][0, :], np.zeros_like(dfil.clean_resid[k][0, :]))

    def test_write_filtered_data(self):
        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        k = (24, 25, 'ee')
        dfil = df.DelayFilter(fname, filetype='miriad')
        dfil.read(bls=[k])

        data = dfil.data
        dfil.run_filter(standoff=0., horizon=1., tol=1e-9, window='blackman-harris', skip_wgt=0.1, maxiter=100, edgecut_low=0, edgecut_hi=0)
        outfilename = os.path.join(DATA_PATH, 'test_output/zen.2458043.12552.xx.HH.filter_test.ORAD.uvh5')
        with pytest.raises(ValueError):
            dfil.write_filtered_data()
        with pytest.raises(NotImplementedError):
            dfil.write_filtered_data(res_outfilename=outfilename, partial_write=True)
        extra_attrs = dict(telescope_name="PAPER")
        dfil.write_filtered_data(res_outfilename=outfilename, add_to_history='Hello_world.', clobber=True, extra_attrs=extra_attrs)

        uvd = UVData()
        uvd.read_uvh5(outfilename)
        assert 'Hello_world.' in uvd.history.replace('\n', '').replace(' ', '')
        assert 'Thisfilewasproducedbythefunction' in uvd.history.replace('\n', '').replace(' ', '')
        assert uvd.telescope_name == 'PAPER'

        filtered_residuals, flags = io.load_vis(uvd)

        dfil.write_filtered_data(CLEAN_outfilename=outfilename, clobber=True)
        clean_model, _flags = io.load_vis(outfilename, filetype='uvh5')

        dfil.write_filtered_data(filled_outfilename=outfilename, clobber=True)
        filled_data, filled_flags = io.load_vis(outfilename, filetype='uvh5')

        for k in data.keys():
            np.testing.assert_array_almost_equal(filled_data[k][~flags[k]], data[k][~flags[k]])
            np.testing.assert_array_almost_equal(dfil.clean_model[k], clean_model[k])
            np.testing.assert_array_almost_equal(dfil.clean_resid[k], filtered_residuals[k])
            np.testing.assert_array_almost_equal(data[k][~flags[k]], (clean_model[k] + filtered_residuals[k])[~flags[k]], 5)
        os.remove(outfilename)

    def test_load_delay_filter_and_write(self):
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        outfilename = os.path.join(DATA_PATH, 'test_output/temp.h5')
        df.load_delay_filter_and_write(uvh5, res_outfilename=outfilename, tol=1e-4, clobber=True, Nbls_per_load=1)
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'ee')])

        dfil = df.DelayFilter(uvh5, filetype='uvh5')
        dfil.read(bls=[(53, 54, 'ee')])
        dfil.run_filter(to_filter=[(53, 54, 'ee')], tol=1e-4, verbose=True)
        np.testing.assert_almost_equal(d[(53, 54, 'ee')], dfil.clean_resid[(53, 54, 'ee')], decimal=5)
        np.testing.assert_array_equal(f[(53, 54, 'ee')], dfil.flags[(53, 54, 'ee')])

        # test loading and writing all baselines at once.
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        outfilename = os.path.join(DATA_PATH, 'test_output/temp.h5')
        df.load_delay_filter_and_write(uvh5, res_outfilename=outfilename, tol=1e-4, clobber=True, Nbls_per_load=None)
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'ee')])

        dfil = df.DelayFilter(uvh5, filetype='uvh5')
        dfil.read(bls=[(53, 54, 'ee')])
        dfil.run_filter(to_filter=[(53, 54, 'ee')], tol=1e-4, verbose=True)
        np.testing.assert_almost_equal(d[(53, 54, 'ee')], dfil.clean_resid[(53, 54, 'ee')], decimal=5)
        np.testing.assert_array_equal(f[(53, 54, 'ee')], dfil.flags[(53, 54, 'ee')])

        cal = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only")
        outfilename = os.path.join(DATA_PATH, 'test_output/temp.h5')
        df.load_delay_filter_and_write(uvh5, calfile=cal, tol=1e-4, res_outfilename=outfilename, Nbls_per_load=2, clobber=True)
        hd = io.HERAData(outfilename)
        assert 'Thisfilewasproducedbythefunction' in hd.history.replace('\n', '').replace(' ', '')
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
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
        df.load_delay_filter_and_write(uvh5, res_outfilename=outfilename,
                                       cache_dir=cdir, mode='dayenu',
                                       Nbls_per_load=1, clobber=True,
                                       spw_range=(0, 32), write_cache=True)
        # generate duplicate cache files to test duplicate key handle for cache load.
        df.load_delay_filter_and_write(uvh5, res_outfilename=outfilename, cache_dir=cdir,
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
        df.load_delay_filter_and_write(uvh5, res_outfilename=outfilename,
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
        df.load_delay_filter_and_write(uvh5, res_outfilename=outfilename,
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

    def test_delay_filter_argparser(self):
        sys.argv = [sys.argv[0], 'a', '--clobber', '--spw_range', '0', '20']
        parser = df.delay_filter_argparser()
        a = parser.parse_args()
        assert a.infilename == 'a'
        assert a.clobber is True
        assert a.spw_range[0] == 0
        assert a.spw_range[1] == 20

    def test_delay_clean_argparser(self):
        sys.argv = [sys.argv[0], 'a', '--clobber', '--window', 'blackmanharris']
        parser = df.delay_clean_argparser()
        a = parser.parse_args()
        assert a.infilename == 'a'
        assert a.clobber is True
        assert a.window == 'blackmanharris'

    def test_delay_linear_argparser(self):
        sys.argv = [sys.argv[0], 'a', '--clobber', '--write_cache', '--cache_dir', '/blah/']
        parser = df.delay_linear_argparser()
        a = parser.parse_args()
        assert a.infilename == 'a'
        assert a.clobber is True
        assert a.write_cache is True
        assert a.cache_dir == '/blah/'

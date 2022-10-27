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
from pyuvdata import UVData, UVFlag

from .. import io
from .. import delay_filter as df
from ..data import DATA_PATH
import glob


@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
@pytest.mark.filterwarnings("ignore:.*dspec.vis_filter will soon be deprecated")
@pytest.mark.filterwarnings("ignore:It seems that the latitude and longitude are in radians")
class Test_DelayFilter(object):
    def test_run_delay_filter(self):
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
        # test that everything runs when baselines lengths are rounded.
        for avg_red_bllens in [True, False]:
            dfil = df.DelayFilter(fname, filetype='miriad')
            dfil.read(bls=[k])
            if avg_red_bllens:
                dfil.avg_red_baseline_vectors()
            wgts = {k: np.ones_like(dfil.flags[k], dtype=float)}
            wgts[k][0, :] = 0.0
            dfil.run_filter(to_filter=[k], weight_dict=wgts, standoff=0., horizon=1., tol=1e-5, window='blackman-harris', skip_wgt=0.1, maxiter=100)
            assert dfil.clean_info[k][(0, dfil.Nfreqs)]['status']['axis_1'][0] == 'skipped'
            np.testing.assert_array_equal(dfil.clean_flags[k][0, :], np.ones_like(dfil.flags[k][0, :]))
            np.testing.assert_array_equal(dfil.clean_model[k][0, :], np.zeros_like(dfil.clean_resid[k][0, :]))
            np.testing.assert_array_equal(dfil.clean_resid[k][0, :], np.zeros_like(dfil.clean_resid[k][0, :]))

    def test_write_filtered_data(self, tmpdir):
        tmp_path = tmpdir.strpath
        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        k = (24, 25, 'ee')
        dfil = df.DelayFilter(fname, filetype='miriad')
        dfil.read(bls=[k])

        data = dfil.data
        dfil.run_delay_filter(standoff=0., horizon=1., tol=1e-9, window='blackman-harris', skip_wgt=0.1, maxiter=100, edgecut_low=0, edgecut_hi=0)
        outfilename = os.path.join(tmp_path, 'zen.2458043.12552.xx.HH.filter_test.ORAD.uvh5')
        with pytest.raises(ValueError):
            dfil.write_filtered_data()
        with pytest.raises(NotImplementedError):
            dfil.write_filtered_data(res_outfilename=outfilename, partial_write=True)
        extra_attrs = dict(telescope_name="PAPER")
        dfil.write_filtered_data(res_outfilename=outfilename, add_to_history='Hello_world.', clobber=True, extra_attrs=extra_attrs)

        uvd = UVData()
        uvd.read_uvh5(outfilename)
        uvd.use_future_array_shapes()
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

    def test_load_delay_filter_and_write(self, tmpdir):
        tmp_path = tmpdir.strpath
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        outfilename = os.path.join(tmp_path, 'temp.h5')
        # test NotImplementedError
        pytest.raises(NotImplementedError, df.load_delay_filter_and_write, uvh5, res_outfilename=outfilename, tol=1e-4,
                      clobber=True, Nbls_per_load=1, avg_red_bllens=True, baseline_list=[(54, 54)], polarizations=['ee'])
        for avg_bl in [True, False]:
            df.load_delay_filter_and_write(uvh5, res_outfilename=outfilename, tol=1e-4, clobber=True, Nbls_per_load=1,
                                           avg_red_bllens=avg_bl)
            hd = io.HERAData(outfilename)
            d, f, n = hd.read(bls=[(53, 54, 'ee')])

            dfil = df.DelayFilter(uvh5, filetype='uvh5')
            dfil.read(bls=[(53, 54, 'ee')])
            dfil.run_delay_filter(to_filter=[(53, 54, 'ee')], tol=1e-4, verbose=True)
            np.testing.assert_almost_equal(d[(53, 54, 'ee')], dfil.clean_resid[(53, 54, 'ee')], decimal=5)
            np.testing.assert_array_equal(f[(53, 54, 'ee')], dfil.flags[(53, 54, 'ee')])

            # test loading and writing all baselines at once.
            uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
            outfilename = os.path.join(tmp_path, 'temp.h5')
            df.load_delay_filter_and_write(uvh5, res_outfilename=outfilename, tol=1e-4, clobber=True,
                                           Nbls_per_load=None, avg_red_bllens=avg_bl)
            hd = io.HERAData(outfilename)
            d, f, n = hd.read(bls=[(53, 54, 'ee')])
        avg_bl = False
        dfil = df.DelayFilter(uvh5, filetype='uvh5')
        dfil.read(bls=[(53, 54, 'ee')])
        dfil.run_delay_filter(to_filter=[(53, 54, 'ee')], tol=1e-4, verbose=True)
        np.testing.assert_almost_equal(d[(53, 54, 'ee')], dfil.clean_resid[(53, 54, 'ee')], decimal=5)
        np.testing.assert_array_equal(f[(53, 54, 'ee')], dfil.flags[(53, 54, 'ee')])

        cal = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only")
        outfilename = os.path.join(tmp_path, 'temp.h5')
        df.load_delay_filter_and_write(uvh5, calfile_list=cal, tol=1e-4, res_outfilename=outfilename, Nbls_per_load=2, clobber=True,
                                       avg_red_bllens=avg_bl)
        hd = io.HERAData(outfilename)
        assert 'Thisfilewasproducedbythefunction' in hd.history.replace('\n', '').replace(' ', '')
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
        np.testing.assert_array_equal(f[(53, 54, 'ee')], True)
        os.remove(outfilename)

        # prepare an input file for broadcasting flags.
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
        df.load_delay_filter_and_write(input_file, res_outfilename=outfilename, tol=1e-4, avg_red_bllens=avg_bl,
                                       factorize_flags=True, time_thresh=time_thresh, clobber=True)
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
        for bl in f:
            assert np.all(f[bl][:, -1])
            assert np.all(f[bl][0, :])

        # test delay filtering and writing with factorized flags and partial i/o
        df.load_delay_filter_and_write(input_file, res_outfilename=outfilename, tol=1e-4, avg_red_bllens=avg_bl,
                                       factorize_flags=True, time_thresh=time_thresh, clobber=True)
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
        for bl in f:
            # check that flags were broadcasted.
            assert np.all(f[bl][0, :])
            assert np.all(f[bl][:, -1])

        df.load_delay_filter_and_write(input_file, res_outfilename=outfilename, tol=1e-4, Nbls_per_load=1, avg_red_bllens=avg_bl,
                                       factorize_flags=True, time_thresh=time_thresh, clobber=True)
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
        for bl in f:
            # check that flags were broadcasted.
            assert np.all(f[bl][0, :])
            assert np.all(f[bl][:, -1])

    def test_load_delay_filter_and_write_baseline_list(self, tmpdir):
        tmp_path = tmpdir.strpath
        uvh5 = [os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.first.uvh5"),
                os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.second.uvh5")]
        cals = [os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only.part1"),
                os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only.part2")]
        outfilename = os.path.join(tmp_path, 'temp.h5')
        cdir = os.path.join(tmp_path, 'cache_temp')
        # test graceful exit with baseline list length of zero.
        with pytest.warns(RuntimeWarning):
            df.load_delay_filter_and_write(datafile_list=uvh5, baseline_list=[],
                                           calfile_list=cals, spw_range=[100, 200], cache_dir=cdir,
                                           read_cache=True, write_cache=True, avg_red_bllens=True,
                                           res_outfilename=outfilename, clobber=True,
                                           mode='dayenu')
        # make a cache directory
        for avg_bl in [True, False]:
            if os.path.isdir(cdir):
                shutil.rmtree(cdir)
            os.mkdir(cdir)
            df.load_delay_filter_and_write(datafile_list=uvh5, baseline_list=[(53, 54)],
                                           calfile_list=cals, spw_range=[100, 200], cache_dir=cdir,
                                           read_cache=True, write_cache=True, avg_red_bllens=avg_bl,
                                           res_outfilename=outfilename, clobber=True,
                                           mode='dayenu')
            hd = io.HERAData(outfilename)
            d, f, n = hd.read()
            assert len(list(d.keys())) == 1
            assert d[(53, 54, 'ee')].shape[1] == 100
            assert d[(53, 54, 'ee')].shape[0] == 60

        # Test baseline_list = None.
        df.load_delay_filter_and_write(datafile_list=uvh5, baseline_list=None,
                                       calfile_list=cals, spw_range=[100, 200], cache_dir=cdir,
                                       read_cache=True, write_cache=True, avg_red_bllens=True,
                                       res_outfilename=outfilename, clobber=True,
                                       mode='dayenu')
        hd = io.HERAData(outfilename)
        d, f, n = hd.read()
        assert d[(53, 54, 'ee')].shape[1] == 100
        assert d[(53, 54, 'ee')].shape[0] == 60
        hdall = io.HERAData(uvh5)
        hdall.read()
        assert np.allclose(hd.baseline_array, hdall.baseline_array)
        assert np.allclose(hd.time_array, hdall.time_array)
        # now do no spw range and no cal files just to cover those lines.
        df.load_delay_filter_and_write(datafile_list=uvh5, baseline_list=[(53, 54)],
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
        # prepare an input files for broadcasting flags.
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
            df.load_delay_filter_and_write(datafile_list=[input_file], res_outfilename=outfilename,
                                           tol=1e-4, baseline_list=[bl[:2]],
                                           cache_dir=cdir,
                                           factorize_flags=True, time_thresh=time_thresh, clobber=True)
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
        uvf = UVFlag(hd, mode='flag', copy_flags=True)
        uvf.use_future_array_shapes()
        uvf.to_waterfall(keep_pol=False, method='and')
        uvf.flag_array[:] = False
        flagfile = os.path.join(tmp_path, 'test_flag.h5')
        uvf.write(flagfile, clobber=True)
        df.load_delay_filter_and_write(datafile_list=[input_file], res_outfilename=outfilename,
                                       tol=1e-4, baseline_list=[bl[:2]],
                                       clobber=True, mode='dayenu',
                                       external_flags=flagfile, overwrite_flags=True)
        # test that all flags are False
        hd = io.HERAData(outfilename)
        d, f, n = hd.read()
        for k in f:
            assert np.all(~f[k])
        # now do the external yaml
        df.load_delay_filter_and_write(datafile_list=[input_file], res_outfilename=outfilename,
                                       tol=1e-4, baseline_list=[bl[:2]],
                                       clobber=True, mode='dayenu',
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
                                       cache_dir=cdir, calfile_list=calfile, read_cache=True,
                                       Nbls_per_load=1, clobber=True, mode='dayenu',
                                       spw_range=(0, 32), write_cache=True)
        # now new cache files should be generated.
        assert len(glob.glob(cdir + '/*')) == 1
        hd = io.HERAData(outfilename)
        assert 'Thisfilewasproducedbythefunction' in hd.history.replace('\n', '').replace(' ', '')
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
        np.testing.assert_array_equal(f[(53, 54, 'ee')], True)

        # test apriori flags and flag_yaml
        hd = io.HERAData(uvh5)
        hd.read()
        flag_yaml = os.path.join(DATA_PATH, 'test_input/a_priori_flags_sample.yaml')
        uvf = UVFlag(hd, mode='flag', copy_flags=True)
        uvf.use_future_array_shapes()
        uvf.to_waterfall(keep_pol=False, method='and')
        uvf.flag_array[:] = False
        flagfile = os.path.join(tmp_path, 'test_flag.h5')
        uvf.write(flagfile, clobber=True)
        df.load_delay_filter_and_write(uvh5, res_outfilename=outfilename,
                                       Nbls_per_load=1, clobber=True, mode='dayenu',
                                       external_flags=flagfile,
                                       overwrite_flags=True)
        # test that all flags are False
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
        for k in f:
            assert np.all(~f[k])
        # now without parital io.
        df.load_delay_filter_and_write(uvh5, res_outfilename=outfilename,
                                       clobber=True, mode='dayenu',
                                       external_flags=flagfile,
                                       overwrite_flags=True)
        # test that all flags are False
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
        for k in f:
            assert np.all(~f[k])

        # now do the external yaml
        df.load_delay_filter_and_write(uvh5, res_outfilename=outfilename,
                                       Nbls_per_load=1, clobber=True, mode='dayenu',
                                       external_flags=flagfile,
                                       overwrite_flags=True, flag_yaml=flag_yaml)
        # test that all flags are af yaml flags
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
        for k in f:
            assert np.all(f[k][:, 0])
            assert np.all(f[k][:, 1])
            assert np.all(f[k][:, 10:20])
            assert np.all(f[k][:, 60])
        os.remove(outfilename)
        shutil.rmtree(cdir)

    def test_delay_clean_argparser(self):
        sys.argv = [sys.argv[0], 'a', '--clobber', '--window', 'blackmanharris', '--mode', 'clean']
        parser = df.delay_filter_argparser()
        a = parser.parse_args()
        assert a.datafilelist == ['a']
        assert a.clobber is True
        assert a.window == 'blackmanharris'

    def test_delay_linear_argparser(self):
        sys.argv = [sys.argv[0], 'a', '--clobber', '--write_cache', '--cache_dir', '/blah/', '--mode', 'dayenu']
        parser = df.delay_filter_argparser()
        a = parser.parse_args()
        assert a.datafilelist == ['a']
        assert a.clobber is True
        assert a.write_cache is True
        assert a.cache_dir == '/blah/'
        sys.argv = [sys.argv[0], 'a', 'b', '--clobber', '--write_cache', '--cache_dir', '/blah/', '--mode', 'dpss_leastsq']
        parser = df.delay_filter_argparser()
        a = parser.parse_args()
        assert a.datafilelist == ['a', 'b']
        assert a.clobber is True
        assert a.write_cache is True
        assert a.cache_dir == '/blah/'

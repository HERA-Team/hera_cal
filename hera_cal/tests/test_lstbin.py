# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License
import pytest
import os
import numpy as np
from collections import OrderedDict as odict
import copy
import glob
import scipy.stats as stats
from pyuvdata import UVCal, UVData
from .. import io, lstbin, utils, redcal, lstbin_simple, apply_cal
from ..datacontainer import DataContainer
from ..data import DATA_PATH
import shutil
from pathlib import Path
from astropy.coordinates import EarthLocation
from itertools import combinations_with_replacement
from pyuvdata import utils as uvutils
from . import mock_uvdata as mockuvd
@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
@pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0 for slice")
@pytest.mark.filterwarnings("ignore:Mean of empty slice")
@pytest.mark.filterwarnings("ignore:invalid value encountered in true_divide")
@pytest.mark.filterwarnings("ignore:invalid value encountered in greater")
class Test_lstbin:
    def setup_method(self):
        # load data
        np.random.seed(0)
        self.data_files = [sorted(glob.glob(DATA_PATH + '/zen.2458043.4*XRAA.uvh5')),
                           sorted(glob.glob(DATA_PATH + '/zen.2458044.4*XRAA.uvh5')),
                           sorted(glob.glob(DATA_PATH + '/zen.2458045.4*XRAA.uvh5'))]
        self.ant_yamls = [DATA_PATH + '/2458043.yaml',
                          DATA_PATH + '/2458044.yaml',
                          DATA_PATH + '/2458045.yaml']

        hd1 = io.HERAData(self.data_files[0])
        hd2 = io.HERAData(self.data_files[1])
        hd3 = io.HERAData(self.data_files[2])
        self.data1, self.flgs1, self.nsmps1 = hd1.read()
        self.ap1, self.freqs1, self.lsts1 = list(hd1.pols.values())[0], list(hd1.freqs.values())[0], np.hstack(list(hd1.lsts.values()))
        self.data2, self.flgs2, self.nsmps2 = hd2.read()
        self.ap2, self.freqs2, self.lsts2 = list(hd2.pols.values())[0], list(hd2.freqs.values())[0], np.hstack(list(hd2.lsts.values()))
        self.data3, self.flgs3, self.nsmps3 = hd3.read()
        self.ap3, self.freqs3, self.lsts3 = list(hd3.pols.values())[0], list(hd3.freqs.values())[0], np.hstack(list(hd3.lsts.values()))

        hd1 = io.HERAData(self.data_files[0])
        hd2 = io.HERAData(self.data_files[1])
        hd3 = io.HERAData(self.data_files[2])
        self.data1, self.flgs1, self.nsmps1 = hd1.read()
        self.ap1, self.freqs1, self.lsts1 = list(hd1.pols.values())[0], list(hd1.freqs.values())[0], np.hstack(list(hd1.lsts.values()))
        self.data2, self.flgs2, self.nsmps2 = hd2.read()
        self.ap2, self.freqs2, self.lsts2 = list(hd2.pols.values())[0], list(hd2.freqs.values())[0], np.hstack(list(hd2.lsts.values()))
        self.data3, self.flgs3, self.nsmps3 = hd3.read()
        self.ap3, self.freqs3, self.lsts3 = list(hd3.pols.values())[0], list(hd3.freqs.values())[0], np.hstack(list(hd3.lsts.values()))

        self.data_list = [self.data1, self.data2, self.data3]
        self.flgs_list = [self.flgs1, self.flgs2, self.flgs3]
        self.lst_list = [self.lsts1, self.lsts2, self.lsts3]
        self.nsmp_list = [self.nsmps1, self.nsmps2, self.nsmps3]
        self.file_ext = "{pol}.{type}.{time:7.5f}.uvh5"
        self.fname_format = "zen.{pol}.{kind}.{lst:7.5f}.uvh5"

    def test_make_lst_grid(self):
        lst_grid = lstbin.make_lst_grid(0.01, begin_lst=None)
        assert len(lst_grid) == 628
        assert np.isclose(lst_grid[0], 0.0050025360725952121)
        lst_grid = lstbin.make_lst_grid(0.01, begin_lst=np.pi)
        assert len(lst_grid) == 628
        assert np.isclose(lst_grid[0], 3.1365901175171982)
        lst_grid = lstbin.make_lst_grid(0.01, begin_lst=-np.pi)
        assert len(lst_grid) == 628
        assert np.isclose(lst_grid[0], 3.1365901175171982)

    def test_config_lst_bin_files(self):
        for data_files in [self.data_files,  # right order
                           [self.data_files[1], self.data_files[0], self.data_files[2]],  # days out of order
                           [self.data_files[0], self.data_files[1][::-1], self.data_files[2]]]:  # single day out of order
            # test that dlst is right
            lst_grid, dlst, file_lsts, begin_lst, lst_arrays, time_arrays = lstbin.config_lst_bin_files(data_files, ntimes_per_file=60)
            np.testing.assert_allclose(dlst, 0.0007830490163485138)
            # test that lst_grid is reasonable
            assert np.isclose(np.median(np.diff(lst_grid)), dlst)
            for fla in file_lsts:
                for fl in fla:
                    assert fl in lst_grid
            # test shape of file_lsts
            assert len(file_lsts) == 4
            for file_lst in file_lsts[0:-1]:
                assert len(file_lst) == 60

    @pytest.mark.filterwarnings("ignore:All-NaN slice encountered")
    @pytest.mark.filterwarnings("ignore:divide by zero encountered in true_divide")
    def test_lstbin(self):
        dlst = 0.0007830490163484
        # test basic execution
        output = lstbin.lst_bin(self.data_list, self.lst_list, flags_list=self.flgs_list, dlst=None,
                                median=True, lst_low=0, lst_hi=np.pi, verbose=False)
        output = lstbin.lst_bin(self.data_list, self.lst_list, flags_list=None, dlst=0.01,
                                verbose=False)

        output = lstbin.lst_bin(self.data_list, self.lst_list, flags_list=self.flgs_list, dlst=dlst,
                                verbose=False)
        # check shape and dtype
        assert output[1][(24, 25, 'ee')].dtype == complex
        assert output[1][(24, 25, 'ee')].shape == (224, 64)
        # check number of points in each bin
        assert np.allclose(output[-1][(24, 25, 'ee')].real[0, 30], 1)
        assert np.allclose(output[-1][(24, 25, 'ee')].real[30, 30], 2)
        assert np.allclose(output[-1][(24, 25, 'ee')].real[100, 30], 3)
        assert np.allclose(output[-1][(24, 25, 'ee')].real[190, 30], 2)
        assert np.allclose(output[-1][(24, 25, 'ee')].real[220, 30], 1)
        # check with large spacing lst_grid
        output = lstbin.lst_bin(self.data_list, self.lst_list, dlst=.01, verbose=False)
        assert np.allclose(output[-1][(24, 25, 'ee')].real[10, 30], 38)
        # check flgs are propagated
        flgs1 = copy.deepcopy(self.flgs1)
        flgs1[(24, 25, 'ee')][:, 32] = True
        flgs2 = copy.deepcopy(self.flgs2)
        flgs2[(24, 25, 'ee')][:, 32] = True
        flgs3 = copy.deepcopy(self.flgs3)
        flgs_list = [flgs1, flgs2, flgs3]
        output = lstbin.lst_bin(self.data_list, self.lst_list, dlst=dlst, flags_list=flgs_list)
        assert np.allclose(output[2][(24, 25, 'ee')][0, 32], True)
        assert np.allclose(output[2][(24, 25, 'ee')][180, 32], False)
        assert np.allclose(output[2][(24, 25, 'ee')][210, 32], False)
        # test return no avg
        output = lstbin.lst_bin(self.data_list, self.lst_list, dlst=dlst, flags_list=self.flgs_list, return_no_avg=True)
        assert len(output[2][list(output[2].keys())[0]][100]) == 3
        assert len(output[2][list(output[2].keys())[0]][100][0]) == 64
        # test switch bl
        conj_data3 = DataContainer(odict([(utils.reverse_bl(k), np.conj(self.data3[k])) for k in self.data3.keys()]))
        data_list = [self.data1, self.data2, conj_data3]
        output = lstbin.lst_bin(data_list, self.lst_list, dlst=dlst)
        assert output[1][(24, 25, 'ee')].shape == (224, 64)
        # test sigma clip
        output = lstbin.lst_bin(self.data_list, self.lst_list, flags_list=None, dlst=0.01,
                                verbose=False, sig_clip=True, min_N=5, sigma=2)
        output = lstbin.lst_bin(self.data_list, self.lst_list, flags_list=None, dlst=0.01,
                                verbose=False, sig_clip=True, min_N=15, flag_below_min_N=True, sigma=2)
        # test wrapping
        lst_list = [(copy.deepcopy(l) + 6) % (2 * np.pi) for l in self.lst_list]
        output = lstbin.lst_bin(self.data_list, lst_list, dlst=0.001, begin_lst=np.pi)
        assert output[0][0] > output[0][-1]
        assert len(output[0]) == 175
        # test appropriate data_count
        output = lstbin.lst_bin(self.data_list, self.lst_list, flags_list=None, dlst=dlst, lst_low=0.25, lst_hi=0.3,
                                verbose=False)
        assert np.allclose(output[4][(24, 25, 'ee')], 3.0)
        # test including additional baselines in bl_list:
        output = lstbin.lst_bin(self.data_list, self.lst_list, nsamples_list=self.nsmp_list,
                                flags_list=self.flgs_list, dlst=dlst, lst_low=0.25, lst_hi=0.3,
                                verbose=False, bl_list=[(512, 512)])
        assert (512, 512, 'ee') in output[4]
        assert np.allclose(output[4][(512, 512, 'ee')], 0.0)
        assert np.all(output[2][(512, 512, 'ee')])
        assert np.allclose(output[1][(512, 512, 'ee')], 1.0)
        # test conjugated flags and nsamples
        flags_list_conj = copy.deepcopy(self.flgs_list)
        nsamp_list_conj = copy.deepcopy(self.nsmp_list)
        data_list_conj = copy.deepcopy(self.data_list)
        # conjugate last night.
        flags_list_conj[-1] = DataContainer({utils.reverse_bl(k): flags_list_conj[-1][k] for k in flags_list_conj[-1]})
        nsamp_list_conj[-1] = DataContainer({utils.reverse_bl(k): nsamp_list_conj[-1][k] for k in nsamp_list_conj[-1]})
        data_list_conj[-1] = DataContainer({utils.reverse_bl(k): np.conj(data_list_conj[-1][k]) for k in data_list_conj[-1]})
        output2 = lstbin.lst_bin(data_list=data_list_conj, lst_list=self.lst_list,
                                 flags_list=flags_list_conj, nsamples_list=nsamp_list_conj,
                                 dlst=dlst, lst_low=0.25, lst_hi=0.3,
                                 verbose=False, bl_list=[(512, 512)])
        # assert outputs are identical, even with conjugations present in the last night.
        for k in output2[4]:
            assert np.all(np.isclose(output[4][k], output2[4][k]))
            assert np.all(np.isclose(output[2][k], output2[2][k]))
            assert np.all(np.isclose(output[1][k], output2[1][k]))

    def test_lstbin_vary_nsamps(self):
        # test execution
        pytest.raises(NotImplementedError, lstbin.lst_bin, self.data_list, self.lst_list, flags_list=self.flgs_list,
                      nsamples_list=self.nsmp_list, dlst=None,
                      median=True, lst_low=0, lst_hi=np.pi, verbose=False)

        lst_output, data_output, flags_output, _, nsamples_output = lstbin.lst_bin(self.data_list, self.lst_list, flags_list=self.flgs_list, dlst=None,
                                                                                   median=False, lst_low=0, lst_hi=np.pi, verbose=False)
        output = lstbin.lst_bin(self.data_list + [data_output], self.lst_list + [lst_output], flags_list=self.flgs_list + [flags_output], dlst=None,
                                nsamples_list=self.nsmp_list + [nsamples_output], median=False, verbose=False)
        # test that nsamples_output are all 3.
        assert np.allclose(output[-1][(24, 25, 'ee')].real[0, 30], 2)
        assert np.allclose(output[-1][(24, 25, 'ee')].real[30, 30], 4)
        assert np.allclose(output[-1][(24, 25, 'ee')].real[100, 30], 6)
        assert np.allclose(output[-1][(24, 25, 'ee')].real[190, 30], 4)
        assert np.allclose(output[-1][(24, 25, 'ee')].real[220, 30], 2)

    def test_lst_align(self):
        # test basic execution
        output = lstbin.lst_align(self.data1, self.lsts1, dlst=None, flags=self.flgs1, flag_extrapolate=True, verbose=False)
        assert output[0][(24, 25, 'ee')].shape == (180, 64)
        assert len(output[2]) == 180
        assert np.allclose(output[2][0], 0.20163512170971379)
        # test flag extrapolate
        assert np.all(output[1][(24, 25, 'ee')][-1])
        # test no dlst
        output = lstbin.lst_align(self.data1, self.lsts1, dlst=None, flags=self.flgs1, flag_extrapolate=True, verbose=False)
        # test wrapped lsts
        lsts = (self.lsts1 + 6) % (2 * np.pi)
        output = lstbin.lst_align(self.data1, lsts, dlst=None, flags=self.flgs1, flag_extrapolate=True, verbose=False)
        assert np.allclose(output[2][150], 0.035628730243852047)

    @pytest.mark.filterwarnings("ignore:The expected shape of the ENU array")
    @pytest.mark.filterwarnings("ignore:antenna_diameters is not set")
    def test_lst_bin_files(self):
        # basic execution
        file_ext = "{pol}.{type}.{time:7.5f}.uvh5"
        lstbin.lst_bin_files(self.data_files, ntimes_per_file=250, outdir="./", overwrite=True,
                             verbose=False, file_ext=file_ext, ignore_flags=True)
        output_lst_file = "./zen.ee.LST.0.20124.uvh5"
        output_std_file = "./zen.ee.STD.0.20124.uvh5"
        assert os.path.exists(output_lst_file)
        assert os.path.exists(output_std_file)
        uv1 = UVData()
        uv1.read(output_lst_file)
        uv1.use_future_array_shapes()
        # assert nsample w.r.t time follows 1-2-3-2-1 pattern
        nsamps = np.mean(uv1.get_nsamples(52, 52, 'ee'), axis=1)
        expectation = np.concatenate([np.ones(22), np.ones(22) * 2, np.ones(136) * 3, np.ones(22) * 2, np.ones(22)]).astype(float)
        assert np.allclose(nsamps[0:len(expectation)], expectation)
        assert np.allclose(nsamps[len(expectation):], 0)
        os.remove(output_lst_file)
        os.remove(output_std_file)

        # test with multiple blgroups
        lstbin.lst_bin_files(self.data_files, ntimes_per_file=250, outdir="./", overwrite=True,
                             verbose=False, file_ext=file_ext, Nblgroups=3, ignore_flags=True)
        assert os.path.exists(output_lst_file)
        assert os.path.exists(output_std_file)
        uv2 = UVData()
        uv2.read(output_lst_file)
        uv2.use_future_array_shapes()
        assert uv1 == uv2
        os.remove(output_lst_file)
        os.remove(output_std_file)

        # test rephase
        lstbin.lst_bin_files(self.data_files, ntimes_per_file=250, outdir="./", overwrite=True,
                             verbose=False, rephase=True, file_ext=file_ext)
        output_lst_file = "./zen.ee.LST.0.20124.uvh5"
        output_std_file = "./zen.ee.STD.0.20124.uvh5"
        assert os.path.exists(output_lst_file)
        assert os.path.exists(output_std_file)
        os.remove(output_lst_file)
        os.remove(output_std_file)

        # test data_list is empty
        data_files = [[sorted(glob.glob(DATA_PATH + '/zen.2458043.*XRAA.uvh5'))[0]],
                      [sorted(glob.glob(DATA_PATH + '/zen.2458045.*XRAA.uvh5'))[-1]]]
        lstbin.lst_bin_files(data_files, ntimes_per_file=30, outdir="./", overwrite=True,
                             verbose=False, file_ext=file_ext)
        output_lst_files = ['./zen.ee.LST.0.20124.uvh5', './zen.ee.LST.0.31870.uvh5', './zen.ee.LST.0.36568.uvh5']
        assert os.path.exists(output_lst_files[0])
        assert os.path.exists(output_lst_files[1])
        assert os.path.exists(output_lst_files[2])
        output_files = np.concatenate([glob.glob("./zen.ee.LST*"),
                                       glob.glob("./zen.ee.STD*")])
        for of in output_files:
            if os.path.exists(of):
                os.remove(of)

        # test smaller ntimes file output, sweeping through f_select
        lstbin.lst_bin_files(self.data_files, ntimes_per_file=80, outdir="./", overwrite=True,
                             verbose=False, vis_units='Jy', file_ext=file_ext)
        output_files = sorted(glob.glob("./zen.ee.LST*") + glob.glob("./zen.ee.STD*"))
        # load a file
        uvd1 = UVData()
        uvd1.read(output_files[1])
        uvd1.use_future_array_shapes()
        assert uvd1.vis_units == 'Jy'
        assert 'Thisfilewasproducedbythefunction' in uvd1.history.replace('\n', '').replace(' ', '')
        assert uvd1.Ntimes == 80
        assert np.isclose(uvd1.nsample_array.max(), 3.0)
        # remove files
        for of in output_files:
            if os.path.exists(of):
                os.remove(of)

        # test output_file_select
        lstbin.lst_bin_files(self.data_files, ntimes_per_file=80, outdir="./", overwrite=True, output_file_select=1,
                             verbose=False, vis_units='Jy', file_ext=file_ext)
        output_files = sorted(glob.glob("./zen.ee.LST*") + glob.glob("./zen.ee.STD*"))
        # load a file
        uvd2 = UVData()
        uvd2.read(output_files[0])
        uvd2.use_future_array_shapes()
        # assert equivalence with previous run
        assert uvd1 == uvd2
        # remove files
        for of in output_files:
            if os.path.exists(of):
                os.remove(of)

        # assert bad output_file_select produces no files
        output_files = sorted(glob.glob("./zen.ee.LST*") + glob.glob("./zen.ee.STD*"))
        for of in output_files:
            if os.path.exists(of):
                os.remove(of)
        lstbin.lst_bin_files(self.data_files, ntimes_per_file=80, outdir="./", overwrite=True, output_file_select=100,
                             verbose=False, file_ext=file_ext)
        output_files = sorted(glob.glob("./zen.ee.LST*") + glob.glob("./zen.ee.STD*"))
        assert len(output_files) == 0

        # test fixed start
        lstbin.lst_bin_files(self.data_files, ntimes_per_file=250, outdir="./", overwrite=True,
                             verbose=False, lst_start=0.18, fixed_lst_start=True, file_ext=file_ext)
        output_lst_file = "./zen.ee.LST.0.17932.uvh5"
        output_std_file = "./zen.ee.STD.0.17932.uvh5"
        assert os.path.exists(output_lst_file)
        assert os.path.exists(output_std_file)
        os.remove(output_lst_file)
        os.remove(output_std_file)
        extra_files = ["zen.ee.LST.0.37508.uvh5", "zen.ee.STD.0.37508.uvh5"]
        for of in extra_files:
            if os.path.exists(of):
                os.remove(of)

        # test input_cal
        uvc = UVCal()
        uvc.read_calfits(os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA.abs.calfits'))
        uvc.use_future_array_shapes()
        uvc.flag_array[uvc.ant_array.tolist().index(24)] = True
        uvc.gain_array[uvc.ant_array.tolist().index(25)] = 1e10
        input_cals = []
        for dfiles in self.data_files:
            input_cals.append([uvc for df in dfiles])

        lstbin.lst_bin_files(self.data_files, ntimes_per_file=250, outdir="./", overwrite=True,
                             verbose=False, input_cals=input_cals, file_ext=file_ext)

        output_lst_file = "./zen.ee.LST.0.20124.uvh5"
        output_std_file = "./zen.ee.STD.0.20124.uvh5"
        assert os.path.exists(output_lst_file)
        assert os.path.exists(output_std_file)
        os.remove(output_lst_file)
        os.remove(output_std_file)

        # test input_cal with only one Ntimes
        input_cals = []
        for dfiles in self.data_files:
            input_cals.append([uvc.select(times=uvc.time_array[:1], inplace=False) for df in dfiles])
        lstbin.lst_bin_files(self.data_files, ntimes_per_file=250, outdir="./", overwrite=True,
                             verbose=False, input_cals=input_cals, file_ext=file_ext)
        assert os.path.exists(output_lst_file)
        assert os.path.exists(output_std_file)

        # assert gains and flags were propagated
        lstb = UVData()
        lstb.read(output_lst_file)
        lstb.use_future_array_shapes()
        assert np.allclose(np.abs(lstb.get_data(25, 37)[~lstb.get_flags(25, 37)]), 0.0)
        assert np.all(lstb.get_flags(24, 25))
        # make sure a is in the data when we dont use the flag yaml.
        for a in [24, 25, 37, 38]:
            assert a in np.unique(np.hstack([lstb.ant_1_array, lstb.ant_2_array]))
        # test removing antennas in a flag yaml
        lstbin.lst_bin_files(self.data_files, ntimes_per_file=250, outdir="./", overwrite=True,
                             verbose=False, input_cals=input_cals, file_ext=file_ext,
                             ex_ant_yaml_files=self.ant_yamls)
        lstb = UVData()
        lstb.read(output_lst_file)
        lstb.use_future_array_shapes()
        for a in [24, 25, 37, 38]:
            assert a not in np.unique(np.hstack([lstb.ant_1_array, lstb.ant_2_array]))

        os.remove(output_lst_file)
        os.remove(output_std_file)

    def test_lstbin_files_inhomogenous_baselines(self, tmpdir):
        tmp_path = tmpdir.strpath
        # now do a test with a more complicated set of files with inhomogenous baselines.
        # between different nights.
        # we want to test that each file that is written has identical ant_1_array
        # and ant_2_array and have baselines that include the union of all the nights.
        os.mkdir(tmp_path + '/lstbin_output/')
        data_lists = [sorted(glob.glob(f'{DATA_PATH}/{jd}/*.uvh5')) for jd in [2459118, 2459119, 2459122, 2459139]]
        lstbin.lst_bin_files(data_lists, outdir=tmp_path + '/lstbin_output/', lst_start=5.178260914725223,
                             dlst=0.0007046864745507975, ntimes_per_file=6)

        bl_union = set()
        for dlist in data_lists:
            hd = UVData()
            hd.read(dlist[-1])
            hd.use_future_array_shapes()
            for bl in hd.get_antpairs():
                bl_union.add(bl)
        output_files = sorted(glob.glob(tmp_path + '/lstbin_output/*LST*.uvh5'))
        lstb = UVData()
        lstb.read(output_files[0])
        lstb.use_future_array_shapes()
        a1arr = lstb.ant_1_array[::lstb.Ntimes]
        a2arr = lstb.ant_2_array[::lstb.Ntimes]
        for of in output_files[1:]:
            lstb = UVData()
            lstb.read(of)
            lstb.use_future_array_shapes()
            assert np.all(lstb.ant_1_array[::lstb.Ntimes] == a1arr)
            assert np.all(lstb.ant_2_array[::lstb.Ntimes] == a2arr)
            aps = set(lstb.get_antpairs())
            for ap in aps:
                assert ap in bl_union or ap[::-1] in bl_union
            for ap in bl_union:
                assert ap in aps or ap[::-1] in aps

        # Do the same test with partial bl loading.
        shutil.rmtree(tmp_path + '/lstbin_output/')
        os.mkdir(tmp_path + '/lstbin_output/')
        data_lists = [sorted(glob.glob(f'{DATA_PATH}/{jd}/*.uvh5')) for jd in [2459118, 2459119, 2459122, 2459139]]
        lstbin.lst_bin_files(data_lists, outdir=tmp_path + '/lstbin_output/', lst_start=5.178260914725223,
                             dlst=0.0007046864745507975, ntimes_per_file=6, Nbls_to_load=1)
        bl_union = set()
        for dlist in data_lists:
            hd = UVData()
            hd.read(dlist[-1])
            hd.use_future_array_shapes()
            for bl in hd.get_antpairs():
                bl_union.add(bl)
        output_files = sorted(glob.glob(tmp_path + '/lstbin_output/*LST*.uvh5'))
        lstb = UVData()
        lstb.read(output_files[0])
        a1arr = lstb.ant_1_array[::lstb.Ntimes]
        a2arr = lstb.ant_2_array[::lstb.Ntimes]
        for of in output_files[1:]:
            lstb = UVData()
            lstb.read(of)
            lstb.use_future_array_shapes()
            # check that all outputs have same baselines
            assert np.all(lstb.ant_1_array[::lstb.Ntimes] == a1arr)
            assert np.all(lstb.ant_2_array[::lstb.Ntimes] == a2arr)
            # check that all outputs have same baselines mod conjugation
            # as the union of all baselines over all nights
            aps = set(lstb.get_antpairs())
            for ap in aps:
                assert ap in bl_union or ap[::-1] in bl_union
            for ap in bl_union:
                assert ap in aps or ap[::-1] in aps
        # test with redundant averaging
        shutil.rmtree(tmp_path + '/lstbin_output/')
        os.mkdir(tmp_path + '/lstbin_output/')
        data_lists = [sorted(glob.glob(f'{DATA_PATH}/{jd}/*.uvh5')) for jd in [2459118, 2459119, 2459122, 2459139]]
        lstbin.lst_bin_files(data_lists, outdir=tmp_path + '/lstbin_output/', lst_start=5.178260914725223,
                             dlst=0.0007046864745507975, ntimes_per_file=6, average_redundant_baselines=True)
        output_files = sorted(glob.glob(tmp_path + '/lstbin_output/*LST*.uvh5'))
        lstb = UVData()
        lstb.read(output_files[0])
        lstb.use_future_array_shapes()
        a1arr = lstb.ant_1_array[::lstb.Ntimes]
        a2arr = lstb.ant_2_array[::lstb.Ntimes]
        for of in output_files[1:]:
            lstb = UVData()
            lstb.read(of)
            lstb.use_future_array_shapes()
            assert np.all(lstb.ant_1_array[::lstb.Ntimes] == a1arr)
            assert np.all(lstb.ant_2_array[::lstb.Ntimes] == a2arr)

    def test_lst_bin_files_redundant_average(self, tmpdir):
        # basic execution
        tmp_path = tmpdir.strpath

        file_ext = "{pol}.{type}.{time:7.5f}.uvh5"
        # generate list of redundantly averaged data files.
        redundantly_averaged_filepaths = []
        redundantly_averaged_data = []
        redundantly_averaged_flags = []
        redundantly_averaged_nsamples = []
        # get reds
        hdt = io.HERAData(self.data_files[0][0])
        hdt.read()
        ants_data = np.unique(np.hstack([hdt.ant_1_array, hdt.ant_2_array]))
        antpos = {a: hdt.antpos[a] for a in ants_data}
        reds_data = redcal.get_reds(antpos, include_autos=True)
        reds_data = [[bl[:2] for bl in grp] for grp in reds_data]
        # build redundantly averaged individual files and write them to disk.
        for fnight, flist in enumerate(self.data_files):
            redundantly_averaged_filepaths.append([])
            for fstr in flist:
                hd = io.HERAData(fstr)
                hd.read()
                if fnight == 1:
                    # conjugate a group to test correctly keying cojugates.
                    reds_average = copy.deepcopy(reds_data)
                    reds_average[1] = [bl[::-1] for bl in reds_average[1]]
                else:
                    reds_to_average = reds_data
                utils.red_average(hd, inplace=True, reds=reds_data)
                out_path = os.path.join(tmp_path, fstr.split('/')[-1].replace('.uvh5', '.red_average.uvh5'))
                hd.write_uvh5(out_path)
                redundantly_averaged_filepaths[-1].append(out_path)
        # test NotImplementedError when we set ignore_flags to True and average_redundant_baselines to True
        pytest.raises(NotImplementedError, lstbin.lst_bin_files, data_files=self.data_files,
                      outdir=tmp_path, overwrite=True, median=False,
                      verbose=False, file_ext=file_ext,
                      ignore_flags=True, average_redundant_baselines=True)
        # get redundantly averaged nsamples.
        lstbin.lst_bin_files(redundantly_averaged_filepaths, outdir=tmp_path, overwrite=True, median=False,
                             verbose=False, file_ext=file_ext, ignore_flags=False, average_redundant_baselines=True)
        output_lst_file = os.path.join(tmp_path, "zen.ee.LST.0.20124.uvh5")
        output_std_file = os.path.join(tmp_path, "zen.ee.STD.0.20124.uvh5")
        assert os.path.exists(output_lst_file)
        assert os.path.exists(output_std_file)
        uv1 = io.HERAData(output_lst_file)
        d1, f1, n1 = uv1.read()
        lstbin.lst_bin_files(self.data_files, outdir=tmp_path, overwrite=True, median=False,
                             verbose=False, file_ext=file_ext, ignore_flags=False, average_redundant_baselines=False)
        uv2 = io.HERAData(output_lst_file)
        uv2.read()
        utils.red_average(uv2, inplace=True, reds=reds_data)
        d2, f2, n2 = uv2.build_datacontainers()
        # assert that all nsamples and data of lstbinned and red averaged equal to red average then lst bin.
        for k in d2:
            assert np.all(np.isclose(d1[k], d2[k]))
            assert np.all(np.isclose(n1[k], n2[k]))
            assert np.all(np.isclose(f1[k], f2[k]))
        lstbin.lst_bin_files(self.data_files, outdir=tmp_path, overwrite=True, median=False,
                             verbose=False, file_ext=file_ext, ignore_flags=False, average_redundant_baselines=True)
        uv3 = io.HERAData(output_lst_file)
        d3, f3, n3 = uv3.read()
        # assert that all nsamples and data of lstbinned and red averaged equal to red average then lst bin.
        for k in d2:
            assert np.all(np.isclose(d3[k], d2[k]))
            assert np.all(np.isclose(n3[k], n2[k]))
            assert np.all(np.isclose(f3[k], f2[k]))

        # next, remove one baseline from a group, including the first
        # baseline which typically acts as the key for the redundant group
        # this tests lstbinners ability to correctly match up red baselines
        # between nights with non-matching keys.
        reds_data_missing = copy.deepcopy(reds_data)
        reds_data_missing[1] = reds_data_missing[1][2:]
        bls_missing_first_group = []
        for grp in reds_data_missing:
            for bl in grp:
                bls_missing_first_group.append(bl)
        # replaced redundantly averaged files in second
        # and third nights with data where
        # first redundant group is missing two baselines.
        for flist in self.data_files[1:]:
            for fstr in flist:
                hd = io.HERAData(fstr)
                hd.read(bls=bls_missing_first_group)
                utils.red_average(hd, inplace=True, reds=reds_data)
                out_path = os.path.join(tmp_path, fstr.split('/')[-1].replace('.uvh5', '.red_average.uvh5'))
                hd.write_uvh5(out_path, clobber=True)

        lstbin.lst_bin_files(redundantly_averaged_filepaths, outdir=tmp_path, overwrite=True, median=False,
                             verbose=False, file_ext=file_ext, ignore_flags=False, average_redundant_baselines=True)

        uv4 = io.HERAData(output_lst_file)
        d4, f4, n4 = uv4.read()
        # assert that all nsamples and data of lstbinned and red averaged equal to red average then lst bin.
        for k in d2:
            if not(k[:2] in reds_data[1] or k[:2][::-1] in reds_data[1]):
                assert np.all(np.isclose(n4[k], n2[k]))
                assert np.all(np.isclose(d4[k], d2[k]))
                assert np.all(np.isclose(f4[k], f2[k]))
        # remove all but a single antenna from data on the first night.
        for fnum, flist in enumerate(self.data_files):
            for fstr in flist:
                hd = UVData()
                hd.read(fstr)
                hd.use_future_array_shapes()
                nants = len(hd.antenna_numbers)
                # on first night, remove all but two antennas.
                if fnum < 2:
                    hd.select(antenna_nums=np.unique(hd.ant_1_array)[:2], keep_all_metadata=False)
                hd.write_uvh5(os.path.join(tmp_path, 'temp.uvh5'), clobber=True)
                hd = io.HERAData(os.path.join(tmp_path, 'temp.uvh5'))
                hd.read()
                if fnum < 2:
                    assert len(hd.antpos) < nants
                utils.red_average(hd, inplace=True, reds=reds_data)
                out_path = os.path.join(tmp_path, fstr.split('/')[-1].replace('.uvh5', '.red_average.uvh5'))
                hd.write_uvh5(out_path, clobber=True)

        lstbin.lst_bin_files(redundantly_averaged_filepaths, outdir=tmp_path, overwrite=True, median=False,
                             verbose=False, file_ext=file_ext, ignore_flags=False, average_redundant_baselines=True)
        # assert that the keys when only two antennas are present on the first night data
        # are identical to the data set when all antennas are present on the first night data
        uv5 = io.HERAData(output_lst_file)
        d5, f5, n5 = uv5.read()
        assert sorted(list(d5.keys())) == sorted(list(d4.keys()))

    def test_lst_bin_arg_parser(self):
        a = lstbin.lst_bin_arg_parser()
        args = a.parse_args(["--dlst", "0.1", "--input_cals", "zen.2458043.12552.HH.uvA.omni.calfits", "zen.2458043.12552.xx.HH.uvORA.abs.calfits",
                             "--overwrite", "zen.2458042.12552.xx.HH.uvXA", "zen.2458043.12552.xx.HH.uvXA"])

        assert np.isclose(args.dlst, 0.1)
        assert len(args.input_cals) == 2
        assert len(args.data_files) == 2

    def test_sigma_clip(self):
        # test basic execution
        np.random.seed(0)
        x = stats.norm.rvs(0, 1, 1000)
        x[10] = 4
        x[11] = -4
        arr = lstbin.sigma_clip(x, sigma=2.0)
        assert np.all(arr[10])
        assert np.all(arr[11])
        # test array performance
        x = np.array([stats.norm.rvs(0, s, 100) for s in np.arange(1, 5.1, 1)])
        x[0, 50] = 100
        x[4, 50] = 5
        arr = lstbin.sigma_clip(x, sigma=2.0)
        assert np.all(arr[0, 50])
        assert not np.any(arr[4, 50])

    def test_gen_nightly_bldicts(self):
        # Test some basic behavior for bl_nightly_dicts.
        hds = [io.HERAData(df[-1]) for df in self.data_files]
        for redundant in [True, False]:
            nightly_bldict_list = lstbin.gen_bl_nightly_dicts(hds, redundant=redundant)
            # baselines all agree over all nights. Make sure their bldicts reflect this.
            for bldict in nightly_bldict_list:
                assert len(bldict) == len(self.data_files)
                assert np.all([bldict[0] == bldict[i] for i in bldict])

    @pytest.mark.filterwarnings("ignore:The expected shape of the ENU array")
    @pytest.mark.filterwarnings("ignore:antenna_diameters is not set")
    @pytest.mark.parametrize('rephase', [True, False])
    def test_simpler_lst_bin_vs_old(self, rephase):
        # basic execution
        file_ext = "{pol}.{type}.{time:7.5f}.uvh5"
        fname_format = self.fname_format
        lstbin_simple.lst_bin_files(
            self.data_files, n_lstbins_per_outfile=250, outdir="./", overwrite=True,
            fname_format=fname_format, ignore_flags=True, rephase=rephase
        )
        output_lst_file = "./zen.ee.LST.0.20124.uvh5"
        output_std_file = "./zen.ee.STD.0.20124.uvh5"
        uv1 = UVData()
        uv1.read(output_lst_file)
        # assert nsample w.r.t time follows 1-2-3-2-1 pattern
        os.remove(output_lst_file)
        os.remove(output_std_file)

        lstbin.lst_bin_files(
            self.data_files, ntimes_per_file=250, outdir="./", overwrite=True,
            file_ext=file_ext, ignore_flags=True, rephase=rephase
        )
        uv2 = UVData()
        uv2.read(output_lst_file)
        os.remove(output_lst_file)
        os.remove(output_std_file)
        # We know the history will be different because they're different functions.
        assert uv1.history != uv2.history
        uv1.history = ""
        uv2.history = ""        

        # We also know the antenna-shaped arrays will be different, because we now only
        # keep the antennas that we need for the baselines.
        uv2.antenna_diameters = uv1.antenna_diameters
        uv2.antenna_names = uv1.antenna_names
        uv2.antenna_numbers = uv1.antenna_numbers
        uv2.antenna_positions = uv1.antenna_positions
        uv2.Nants_telescope = uv1.Nants_telescope
        uv2.uvw_array = uv1.uvw_array
        uv2.extra_keywords = uv1.extra_keywords
        uv2.phase_center_catalog[0]['cat_name'] = 'zenith'
        assert uv1 == uv2


    @pytest.mark.filterwarnings("ignore:The expected shape of the ENU array")
    @pytest.mark.filterwarnings("ignore:antenna_diameters is not set")
    def test_simpler_lst_bin_files(self):
        # basic execution
        file_ext = self.file_ext
        fname_format = self.fname_format
        lstbin_simple.lst_bin_files(
            self.data_files, n_lstbins_per_outfile=250, outdir="./", overwrite=True,
            fname_format=fname_format, ignore_flags=True
        )
        output_lst_file = "./zen.ee.LST.0.20124.uvh5"
        output_std_file = "./zen.ee.STD.0.20124.uvh5"
        assert os.path.exists(output_lst_file)
        assert os.path.exists(output_std_file)
        uv1 = UVData()
        uv1.read(output_lst_file)
        # assert nsample w.r.t time follows 1-2-3-2-1 pattern
        nsamps = np.mean(uv1.get_nsamples(52, 52, 'ee'), axis=1)
        expectation = np.concatenate([np.ones(22), np.ones(22) * 2, np.ones(136) * 3, np.ones(22) * 2, np.ones(22)]).astype(float)
        assert np.allclose(nsamps[0:len(expectation)], expectation)
        assert np.allclose(nsamps[len(expectation):], 0)
        os.remove(output_lst_file)
        os.remove(output_std_file)

        # test with multiple blgroups
        # There are 28 baselines in the files.
        lstbin_simple.lst_bin_files(
            self.data_files, n_lstbins_per_outfile=250, outdir="./", overwrite=True,
            fname_format=fname_format, Nbls_to_load=10, ignore_flags=True
        )
        assert os.path.exists(output_lst_file)
        assert os.path.exists(output_std_file)
        uv2 = UVData()
        uv2.read(output_lst_file)
        assert uv1 == uv2
        os.remove(output_lst_file)
        os.remove(output_std_file)

        # test rephase
        lstbin_simple.lst_bin_files(
            self.data_files, n_lstbins_per_outfile=250, outdir="./", overwrite=True,
            rephase=True, fname_format=fname_format
        )
        output_lst_file = "./zen.ee.LST.0.20124.uvh5"
        output_std_file = "./zen.ee.STD.0.20124.uvh5"
        assert os.path.exists(output_lst_file)
        assert os.path.exists(output_std_file)
        os.remove(output_lst_file)
        os.remove(output_std_file)

        # test data_list is empty
        data_files = [[sorted(glob.glob(DATA_PATH + '/zen.2458043.*XRAA.uvh5'))[0]],
                      [sorted(glob.glob(DATA_PATH + '/zen.2458045.*XRAA.uvh5'))[-1]]]
        lstbin_simple.lst_bin_files(
            data_files, n_lstbins_per_outfile=30, outdir="./", overwrite=True,
            fname_format=fname_format
        )
        output_lst_files = ['./zen.ee.LST.0.20124.uvh5', './zen.ee.LST.0.31870.uvh5', './zen.ee.LST.0.36568.uvh5']
        assert os.path.exists(output_lst_files[0])
        assert os.path.exists(output_lst_files[1])
        assert os.path.exists(output_lst_files[2])
        output_files = np.concatenate([glob.glob("./zen.ee.LST*"),
                                       glob.glob("./zen.ee.STD*")])
        for of in output_files:
            if os.path.exists(of):
                os.remove(of)

        # test smaller ntimes file output, sweeping through f_select
        lstbin_simple.lst_bin_files(
            self.data_files, n_lstbins_per_outfile=80, outdir="./", overwrite=True,
            write_kwargs={'vis_units': 'Jy'}, fname_format=fname_format)
        output_files = sorted(glob.glob("./zen.ee.LST*") + glob.glob("./zen.ee.STD*"))
        # load a file
        uvd1 = UVData()
        uvd1.read(output_files[1])
        assert uvd1.vis_units == 'Jy'
        assert 'Thisfilewasproducedbythefunction' in uvd1.history.replace('\n', '').replace(' ', '')
        assert uvd1.Ntimes == 80
        assert np.isclose(uvd1.nsample_array.max(), 3.0)
        # remove files
        for of in output_files:
            if os.path.exists(of):
                os.remove(of)

        # test output_file_select
        lstbin_simple.lst_bin_files(
            self.data_files,n_lstbins_per_outfile=80, outdir="./", overwrite=True, 
            output_file_select=1, write_kwargs={'vis_units': 'Jy'}, fname_format=fname_format
        )
        output_files = sorted(glob.glob("./zen.ee.LST*") + glob.glob("./zen.ee.STD*"))
        # load a file
        uvd2 = UVData()
        uvd2.read(output_files[0])
        # assert equivalence with previous run
        assert uvd1 == uvd2
        # remove files
        for of in output_files:
            if os.path.exists(of):
                os.remove(of)

        # assert bad output_file_select produces no files
        output_files = sorted(glob.glob("./zen.ee.LST*") + glob.glob("./zen.ee.STD*"))
        for of in output_files:
            if os.path.exists(of):
                os.remove(of)
        lstbin_simple.lst_bin_files(
            self.data_files, n_lstbins_per_outfile=80, outdir="./", overwrite=True, 
            output_file_select=100,
            fname_format=fname_format
        )
        output_files = sorted(glob.glob("./zen.ee.LST*") + glob.glob("./zen.ee.STD*"))
        assert len(output_files) == 0

        # test fixed start
        lstbin_simple.lst_bin_files(
            self.data_files, n_lstbins_per_outfile=250, outdir="./", overwrite=True,
            lst_start=0.18, fname_format=fname_format,
        )
        output_lst_file = "./zen.ee.LST.0.17932.uvh5"
        output_std_file = "./zen.ee.STD.0.17932.uvh5"
        assert os.path.exists(output_lst_file)
        assert os.path.exists(output_std_file)
        os.remove(output_lst_file)
        os.remove(output_std_file)
        extra_files = ["zen.ee.LST.0.37508.uvh5", "zen.ee.STD.0.37508.uvh5"]
        for of in extra_files:
            if os.path.exists(of):
                os.remove(of)


         # test sigma-clip
        lstbin_simple.lst_bin_files(
            self.data_files, n_lstbins_per_outfile=250, outdir="./", overwrite=True,
            lst_start=0.18, fname_format=fname_format, sigma_clip_thresh=4.0, sigma_clip_min_N=4
        )
        output_lst_file = "./zen.ee.LST.0.17932.uvh5"
        output_std_file = "./zen.ee.STD.0.17932.uvh5"
        assert os.path.exists(output_lst_file)
        assert os.path.exists(output_std_file)
        os.remove(output_lst_file)
        os.remove(output_std_file)
        extra_files = ["zen.ee.LST.0.37508.uvh5", "zen.ee.STD.0.37508.uvh5"]
        for of in extra_files:
            if os.path.exists(of):
                os.remove(of)



        # test input_cal
        uvc = UVCal()
        uvc.read_calfits(os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA.abs.calfits'))
        uvc.use_future_array_shapes()
        uvc.flag_array[uvc.ant_array.tolist().index(24)] = True
        uvc.gain_array[uvc.ant_array.tolist().index(25)] = 1e10
        input_cals = []
        for dfiles in self.data_files:
            input_cals.append([uvc] * len(dfiles))

        lstbin_simple.lst_bin_files(
            self.data_files, n_lstbins_per_outfile=250, outdir="./", overwrite=True,
            input_cals=input_cals, fname_format=fname_format)

        output_lst_file = "./zen.ee.LST.0.20124.uvh5"
        output_std_file = "./zen.ee.STD.0.20124.uvh5"
        assert os.path.exists(output_lst_file)
        assert os.path.exists(output_std_file)
        os.remove(output_lst_file)
        os.remove(output_std_file)

        # test input_cal with only one Ntimes
        input_cals = []
        for dfiles in self.data_files:
            input_cals.append([uvc.select(times=uvc.time_array[:1], inplace=False) for df in dfiles])
        lstbin_simple.lst_bin_files(
            self.data_files, n_lstbins_per_outfile=250, outdir="./", overwrite=True,
            input_cals=input_cals, fname_format=fname_format
        )
        assert os.path.exists(output_lst_file)
        assert os.path.exists(output_std_file)

        # assert gains and flags were propagated
        lstb = UVData()
        lstb.read(output_lst_file)
        assert np.allclose(np.abs(lstb.get_data(25, 37)[~lstb.get_flags(25, 37)]), 0.0)
        assert np.all(lstb.get_flags(24, 25))
        # make sure a is in the data when we dont use the flag yaml.
        for a in [24, 25, 37, 38]:
            assert a in np.unique(np.hstack([lstb.ant_1_array, lstb.ant_2_array]))
        # test removing antennas in a flag yaml
        lstbin_simple.lst_bin_files(
            self.data_files, n_lstbins_per_outfile=250, outdir="./", overwrite=True,
            input_cals=input_cals, fname_format=fname_format,
            ex_ant_yaml_files=self.ant_yamls
        )
        lstb = UVData()
        lstb.read(output_lst_file)
        for a in [24, 25, 37, 38]:
            assert a not in np.unique(np.hstack([lstb.ant_1_array, lstb.ant_2_array]))

        os.remove(output_lst_file)
        os.remove(output_std_file)

    def test_lstbin_golden(self):
        # test smaller ntimes file output, sweeping through f_select
        lstbin_simple.lst_bin_files(
            self.data_files, n_lstbins_per_outfile=80, outdir="./", overwrite=True,
            write_kwargs={'vis_units': 'Jy'}, fname_format=self.fname_format)
        output_files = sorted(glob.glob("./zen.ee.LST*") + glob.glob("./zen.ee.STD*"))
        # load a file
        uvd1 = UVData()
        uvd1.read(output_files[1])
        assert uvd1.vis_units == 'Jy'
        assert 'Thisfilewasproducedbythefunction' in uvd1.history.replace('\n', '').replace(' ', '')
        assert uvd1.Ntimes == 80
        assert np.isclose(uvd1.nsample_array.max(), 3.0)
        # remove files
        for of in output_files:
            if os.path.exists(of):
                os.remove(of)


        # test golden_lsts
        lstbin_simple.lst_bin_files(
            self.data_files, n_lstbins_per_outfile=80, outdir="./", overwrite=True, 
            output_file_select=1, write_kwargs={'vis_units': 'Jy'}, 
            fname_format=self.fname_format,
            golden_lsts=(0.265,)
        )
        output_files = sorted(glob.glob("./zen.ee.LST*") + glob.glob("./zen.ee.STD*"))
        # load a file
        uvd2 = UVData()
        uvd2.read(output_files[0])
        # assert equivalence with previous run
        assert uvd1 == uvd2

        golden = glob.glob("./zen.ee.GOLDEN*")
        assert golden

        # remove files
        for of in output_files + golden:
            if os.path.exists(of):
                os.remove(of)

    def test_lstbin_savechans(self):
        # test smaller ntimes file output, sweeping through f_select
        lstbin_simple.lst_bin_files(
            self.data_files, n_lstbins_per_outfile=80, outdir="./", overwrite=True,
            write_kwargs={'vis_units': 'Jy'}, fname_format=self.fname_format)
        output_files = sorted(glob.glob("./zen.ee.LST*") + glob.glob("./zen.ee.STD*"))
        # load a file
        uvd1 = UVData()
        uvd1.read(output_files[1])
        assert uvd1.vis_units == 'Jy'
        assert 'Thisfilewasproducedbythefunction' in uvd1.history.replace('\n', '').replace(' ', '')
        assert uvd1.Ntimes == 80
        assert np.isclose(uvd1.nsample_array.max(), 3.0)
        # remove files
        for of in output_files:
            if os.path.exists(of):
                os.remove(of)


        # test golden_lsts
        lstbin_simple.lst_bin_files(
            self.data_files, n_lstbins_per_outfile=80, outdir="./", overwrite=True, 
            output_file_select=1, write_kwargs={'vis_units': 'Jy'}, 
            fname_format=self.fname_format,
            save_channels=(32,)
        )
        output_files = sorted(glob.glob("./zen.ee.LST*") + glob.glob("./zen.ee.STD*"))
        # load a file
        uvd2 = UVData()
        uvd2.read(output_files[0])
        # assert equivalence with previous run
        assert uvd1 == uvd2

        golden = glob.glob("./zen.ee.REDUCEDCHAN*")
        assert golden

        # remove files
        for of in output_files + golden:
            if os.path.exists(of):
                os.remove(of)

    def tearDown(self):
        output_files = sorted(glob.glob("./zen.ee.LST*") + glob.glob("./zen.ee.STD*"))
        for of in output_files:
            if os.path.exists(of):
                os.remove(of)

def create_small_array_uvd(identifiable: bool = False, **kwargs):
    kwargs.update(
        freqs=np.linspace(150e6, 160e6, 100),
        ants=[0,1,2,127,128],
        antpairs=[(0,0), (0,1), (0,2), (1, 1), (1,2), (2, 2)],
        pols=('xx', 'yy')
    )
    if identifiable:
        return mockuvd.create_uvd_identifiable(**kwargs)
    else:
        return mockuvd.create_uvd_ones(**kwargs)

@pytest.fixture(scope="function")
def uvd():
    return create_small_array_uvd()

@pytest.fixture(scope="function")
def uvc(uvd):
    return UVCal.from_uvdata(
        uvd,
        cal_style = "redundant",
        gain_convention = "multiply",
        jones_array = "linear",
        cal_type = "gain",
        empty=True
    )

@pytest.fixture(scope="function")
def uvd_file(uvd, tmpdir_factory):
    # Write to file, so we can run lst_bin_files
    tmp = tmpdir_factory.mktemp("test_partial_times")
    uvd.write_uvh5(str(tmp / 'mock.uvh5'), clobber=True)
    return str(tmp / 'mock.uvh5')

@pytest.fixture(scope="function")
def uvc_file(uvc, uvd_file):
    # Write to file, so we can run lst_bin_files
    tmp = os.path.dirname(uvd_file)
    fl = f'{tmp}/mock.calfits'
    uvc.write_calfits(fl, clobber=True)
    return fl

class Test_LSTBinSimple:
    def setup_method(self):
        self.ntimes = 10
        self.nants = 4
        self.ants = np.arange(self.nants)
        self.baselines = list(combinations_with_replacement(self.ants, 2))
        self.nbls = len(self.baselines)
        self.nfreqs = 5
        self.npols = 4

        self.shape = (self.ntimes, self.nbls, self.nfreqs, self.npols)

        # Set up some fake random data. Not really important what it is here.
        self.data = (
            np.random.normal(size=self.shape) + 1j*np.random.normal(size=self.shape)
        )

        # LST bins -- each includes one data LST.
        self.lst_bin_edges = np.linspace(
            0, 2*np.pi,
            self.ntimes + 1
        )
        dlst = self.lst_bin_edges[1] - self.lst_bin_edges[0]
        
        self.data_lsts = np.linspace(dlst/2, 2*np.pi - dlst/2, self.ntimes)
        self.ants = np.arange(self.nbls)

        self.freq_array = np.linspace(100e6, 200e6, self.nfreqs)
        self.flags = np.zeros(self.shape, dtype=bool)
        self.nsamples = np.ones(self.shape, dtype=int)
        self.antpos = {ant: np.random.normal(loc=0, scale=30, size=3) for ant in self.ants}

    def simple_lst_bin(self, **kwargs):
        if "data" not in kwargs:
            kwargs["data"] = self.data
        if "data_lsts" not in kwargs:
            kwargs["data_lsts"] = self.data_lsts
        if "lst_bin_edges" not in kwargs:
            kwargs["lst_bin_edges"] = self.lst_bin_edges
        if "baselines" not in kwargs:
            kwargs["baselines"] = self.baselines
        if "freq_array" not in kwargs:
            kwargs["freq_array"] = self.freq_array
        if "antpos" not in kwargs:
            kwargs["antpos"] = self.antpos
        
        return lstbin_simple.simple_lst_bin(**kwargs)
    
    def test_simple_lst_bin_bad_inputs(self):
        # Test that we get the right errors for bad inputs
        data = np.ones((self.ntimes, self.nbls, self.nfreqs, self.npols+1)) 
        with pytest.raises(ValueError, match="data has more than 4 pols"):
            self.simple_lst_bin(data=data)

        with pytest.raises(ValueError, match=f"data should have shape"):
            self.simple_lst_bin(freq_array=self.freq_array[:-1])

        with pytest.raises(ValueError, match=f"flags should have shape"):
            self.simple_lst_bin(flags=self.flags[:-1])

        # Make a wrong-shaped nsample array.
        with pytest.raises(ValueError, match=f"nsamples should have shape"):
            self.simple_lst_bin(nsamples=self.nsamples[:-1])

        # Use only one bin edge
        with pytest.raises(ValueError, match="lst_bin_edges must have at least 2 elements"):
            self.simple_lst_bin(lst_bin_edges=self.lst_bin_edges[:1])

        # Try rephasing without freq_array or antpos
        with pytest.raises(ValueError, match="freq_array and antpos is needed for rephase"):
            self.simple_lst_bin(rephase=True, antpos=None)

    def test_lst_bin_simple_defaults(self):
        bins, d, f, n = self.simple_lst_bin(
            nsamples = self.nsamples,  # nsamples is all ones anyway
            flags=self.flags,          # flags is all false anyway 
            rephase=False,
        )
    
        bins2, d2, f2, n2 = self.simple_lst_bin(
            rephase=False,
        )
        
        assert all(np.all(dd==dd2) for dd, dd2 in zip(d, d2))
        assert all(np.all(dd==dd2) for dd, dd2 in zip(f, f2))
        assert all(np.all(dd==dd2) for dd, dd2 in zip(n, n2))

    def test_lst_bin_simple_happy_path(self):
        bins, d, f, n = self.simple_lst_bin(
            rephase=False
        )
        
        # Check that the bins are what we expect
        assert np.all(bins == self.lst_bin_edges[:-1] + np.diff(self.lst_bin_edges)/2)

        # In this case, we set it up so that each bin should have only one time in it,
        # so that d, f, and n should be the same as self.data, self.flags, and self.nsamples

        d = np.squeeze(np.array(d))
        f = np.squeeze(np.array(f))
        n = np.squeeze(np.array(n))

        # Check that the data is what we expect
        np.testing.assert_allclose(d, self.data)
        np.testing.assert_allclose(f, self.flags)
        np.testing.assert_allclose(n, self.nsamples)

    def test_lst_bin_simple_rephase(self):
        bins, d0, f0, n0 = self.simple_lst_bin(rephase=True,)
        bins, d, f, n = self.simple_lst_bin(rephase=False)
        np.testing.assert_allclose(d, d0)

    def test_lst_bin_simple_lstbinedges(self):
        lst_bin_edges = self.lst_bin_edges.copy()
        lst_bin_edges -= 4*np.pi

        bins, d0, f0, n0 = self.simple_lst_bin(lst_bin_edges=lst_bin_edges)

        lst_bin_edges = self.lst_bin_edges.copy()
        lst_bin_edges += 4*np.pi
    
        bins, d, f, n = self.simple_lst_bin(lst_bin_edges=lst_bin_edges)
        
        np.testing.assert_allclose(d, d0)

        with pytest.raises(ValueError, match="lst_bin_edges must be monotonically increasing."):
            self.simple_lst_bin(lst_bin_edges=lst_bin_edges[::-1])    

        
    def test_reduce_lst_bins_no_out(self):
        bins, d, f, n = self.simple_lst_bin()
        dd, ff, std, nn = lstbin_simple.reduce_lst_bins(d, f, n)

        assert dd.shape == ff.shape == std.shape == nn.shape

        # reduce_data swaps the order of bls/times
        dd = dd.swapaxes(0, 1)
        ff = ff.swapaxes(0, 1)
        nn = nn.swapaxes(0, 1)

        np.testing.assert_allclose(dd, self.data)
        assert np.all(ff == self.flags)
        np.testing.assert_allclose(nn, self.nsamples)

    def test_argparser_returns(self):
        args = lstbin_simple.lst_bin_arg_parser()
        assert args is not None

    def test_apply_calfile_rules(self, tmpdir_factory):
        direc = tmpdir_factory.mktemp("test_apply_calfile_rules")

        datas = [Path(direc / f"data{i}.uvh5") for i in range(3)]
        for d in datas:
            d.touch()

        cals = [Path(direc / f"data{i}.calfile") for i in range(3)]
        for c in cals:
            c.touch()
        
        data_files, calfiles = lstbin_simple.apply_calfile_rules(
            [[str(d) for d in datas]],
            calfile_rules = [('.uvh5', '.calfile')],
            ignore_missing=False
        )
        assert len(data_files[0]) == 3
        assert len(calfiles[0]) == 3

        cals[-1].unlink()
        with pytest.raises(IOError, match="does not exist"):
            lstbin_simple.apply_calfile_rules(
                [[str(d) for d in datas]],
                calfile_rules = [('.uvh5', '.calfile')],
                ignore_missing=False
            )

        data_files, calfiles = lstbin_simple.apply_calfile_rules(
            [[str(d) for d in datas]],
            calfile_rules = [('.uvh5', '.calfile')],
            ignore_missing=True
        )
        assert len(data_files[0]) == 2
        assert len(calfiles[0]) == 2

    def test_lst_bin_files_withcal(self, uvd, uvd_file, uvc_file):
        # Make a mock data file with partial times

        # Make sure that using a calfile with all ones doesn't change the data
        lsts = np.sort(np.unique(uvd.lst_array))
        dlst = lsts[1] - lsts[0]
        nocal_fnames = lstbin_simple.lst_bin_files(
            data_files=[[uvd_file]],
            dlst=dlst,
            n_lstbins_per_outfile=len(uvd.lst_array),
            outdir=os.path.dirname(uvd_file),
            lst_start=uvd.lst_array[0],
        )

        withcal_fnames = lstbin_simple.lst_bin_files(
            data_files=[[uvd_file]],
            input_cals=[[uvc_file]],
            dlst=dlst,
            n_lstbins_per_outfile=len(uvd.lst_array),
            outdir=os.path.dirname(uvd_file),
            lst_start=uvd.lst_array[0],
            overwrite=True
        )

        uvd_nocal = UVData.from_file(nocal_fnames[0]["LST"], read_data=True)
        uvd_withcal = UVData.from_file(withcal_fnames[0]["LST"], read_data=True)

        np.testing.assert_allclose(uvd_nocal.data_array, uvd_withcal.data_array)
        

    def test_lst_average_sigma_clip(self):
        data_n, flg_n, std_n, norm_n = lstbin_simple.lst_average(
            data=self.data,
            nsamples=self.nsamples,
            flags=self.flags,
            sigma_clip_thresh=0.0,
        )

        data, flg, std, norm = lstbin_simple.lst_average(
            data=self.data,
            nsamples=self.nsamples,
            flags=self.flags,
            sigma_clip_thresh=10.0,
        )

        assert data.shape == flg.shape == std.shape == norm.shape == self.data.shape[1:]
        assert np.all(data == data_n)

    def test_lst_bin_files_for_baselines_defaults(self, uvd, uvd_file):
        lstbins, d0, f0, n0, times0 = lstbin_simple.lst_bin_files_for_baselines(
            data_files=[uvd_file],
            lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()+0.01],
            antpairs=uvd.get_antpairs(),
            rephase=False
        )

        lstbins, d, f, n, times = lstbin_simple.lst_bin_files_for_baselines(
            data_files=[uvd_file],
            lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()+0.01],
            antpairs=uvd.get_antpairs(),
            freqs=uvd.freq_array,
            pols=uvd.polarization_array,
            antpos = uvd.antenna_positions,
            time_idx = [np.ones(uvd.Ntimes, dtype=bool)],
            time_arrays=[np.unique(uvd.time_array)],
            lsts = np.unique(uvd.lst_array),
            rephase=False
        )

        np.testing.assert_allclose(d0, d)
        np.testing.assert_allclose(f0, f)
        np.testing.assert_allclose(n0, n)
        np.testing.assert_allclose(times0, times)

    def test_lst_bin_files_for_baselines_empty(self, uvd, uvd_file):
        lstbins, d0, f0, n0, times0 = lstbin_simple.lst_bin_files_for_baselines(
            data_files=[uvd_file],
            lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()],
            antpairs=[(127, 128)],
            rephase=True
        )

        assert np.all(f0)

    def test_lst_bin_files_for_baselines_extra(self, uvd, uvd_file):
        # Providing baselines that don't exist in the file is fine, they're just ignored.
        lstbins, d0, f0, n0, times0 = lstbin_simple.lst_bin_files_for_baselines(
            data_files=[uvd_file],
            lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()],
            antpairs=uvd.get_antpairs() + [(127, 128)],
            rephase=True
        )

        assert np.all(f0[0][:, -1])  # last baseline is the extra one that's all flagged.

    def test_lst_bin_files_for_baselines_straddle_times(self, uvd, uvd_file):
        """This just tests that the code doesn't crash when the lst bins straddle the end of the file.
        
        Since the input uvdata is all ones, the output should be all ones as well, 
        regardless of how many times go into the LST bin.
        """
        dlst = uvd.lst_array.max() - uvd.lst_array.min()

        lstbins, d0, f0, n0, times0 = lstbin_simple.lst_bin_files_for_baselines(
            data_files=[uvd_file],
            lst_bin_edges=[uvd.lst_array.min() - dlst/2, uvd.lst_array.max() + dlst/2],
            antpairs=uvd.get_antpairs(),
            rephase=False
        )

        lstbins1, d01, f01, n01, times01= lstbin_simple.lst_bin_files_for_baselines(
            data_files=[uvd_file],
            lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()],
            antpairs=uvd.get_antpairs(),
            rephase=False
        )
        assert len(d0[0]) == 20
        assert len(d01[0]) == 18
        assert np.allclose(d0, 1+0j)

    def test_lst_bin_files_output_select(self, uvd, uvd_file):
        with pytest.warns(UserWarning, match="One or more indices in output_file_select"):
            # Output file doesn't exist. This just warns instead of erroring, but does
            # exit the function. Not really sure why.
            lsts = np.sort(np.unique(uvd.lst_array))
            dlst = lsts[1] - lsts[0]

            lstbin_simple.lst_bin_files(
                data_files=[[uvd_file]],
                dlst=dlst,
                n_lstbins_per_outfile=len(uvd.lst_array),
                outdir=os.path.dirname(uvd_file),
                lst_start=lsts[0] - dlst/2,
                lst_width=lsts.max() - lsts.min() + dlst,
                output_file_select=6,
            )

        fnames = lstbin_simple.lst_bin_files(
            data_files=[[uvd_file]],
            dlst=dlst,
            n_lstbins_per_outfile=len(np.unique(uvd.lst_array)),
            outdir=os.path.dirname(uvd_file),
            lst_start=lsts.min() - dlst/2,
            lst_width=lsts.max() - lsts.min() + dlst,
            output_file_select=0,
        )

        uvd_nocal = UVData.from_file(fnames[0]['LST'], read_data=True)
        assert uvd_nocal.Ntimes == uvd.Ntimes


    def test_lst_bin_files_calfiles(self, uvd, uvd_file, uvc_file):
        """Test that providing calfiles and using calfile_rules give the same thing."""
        lsts = np.sort(np.unique(uvd.lst_array))
        dlst = lsts[1] - lsts[0]

        fnames = lstbin_simple.lst_bin_files(
            data_files=[[uvd_file]],
            dlst=dlst,
            n_lstbins_per_outfile=len(uvd.lst_array),
            outdir=os.path.dirname(uvd_file),
            lst_start=lsts[0] - dlst/2,
            lst_width=lsts.max() - lsts.min() + dlst,
            input_cals=[[uvc_file]],
        )

        uvd = UVData.from_file(fnames[0]['LST'], read_data=True)

        fnames_rules = lstbin_simple.lst_bin_files(
            data_files=[[uvd_file]],
            dlst=dlst,
            n_lstbins_per_outfile=len(uvd.lst_array),
            outdir=os.path.dirname(uvd_file),
            lst_start=lsts[0] - dlst/2,
            lst_width=lsts.max() - lsts.min() + dlst,
            calfile_rules=[(".uvh5", ".calfits")],
            overwrite=True
        )
        uvd_rules = UVData.from_file(fnames_rules[0]['LST'], read_data=True)

        np.testing.assert_allclose(uvd.data_array, uvd_rules.data_array)

    def test_lst_bin_files_calfiles_freqrange(self, uvd, uvd_file, uvc_file):
        """Test that providing calfiles and using calfile_rules give the same thing."""
        lsts = np.sort(np.unique(uvd.lst_array))
        dlst = lsts[1] - lsts[0]

        fnames = lstbin_simple.lst_bin_files(
            data_files=[[uvd_file]],
            dlst=dlst,
            n_lstbins_per_outfile=len(uvd.lst_array),
            outdir=os.path.dirname(uvd_file),
            lst_start=lsts[0] - dlst/2,
            lst_width=lsts.max() - lsts.min() + dlst,
            input_cals=[[uvc_file]],
            freq_min=153e8,
            freq_max=158e8,
        )

        uvd = UVData.from_file(fnames[0]['LST'], read_data=True)

        fnames_rules = lstbin_simple.lst_bin_files(
            data_files=[[uvd_file]],
            dlst=dlst,
            n_lstbins_per_outfile=len(uvd.lst_array),
            # outdir=os.path.dirname(uvd_file),  # by default it's this dir
            lst_start=lsts[0] - dlst/2,
            lst_width=lsts.max() - lsts.min() + dlst,
            calfile_rules=[(".uvh5", ".calfits")],
            freq_min=153e8,
            freq_max=158e8,
            overwrite=True
        )
        uvd_rules = UVData.from_file(fnames_rules[0]['LST'], read_data=True)

        np.testing.assert_allclose(uvd.data_array, uvd_rules.data_array)

    def test_golden_data(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("lstbin_golden_data")
        uvds = mockuvd.make_dataset(ndays=3, nfiles=4, ntimes=2, identifiable=True, creator=create_small_uvdata)
        data_files = mockuvd.write_files_in_hera_format(uvds, tmp)

        out_files = lstbin_simple.lst_bin_files(
            data_files=data_files,
            n_lstbins_per_outfile=2,
            golden_lsts=uvds[0][1].lst_array.min() + 0.0001
        )

        assert len(out_files) == 4
        assert out_files[1]['GOLDEN']
        assert not out_files[0]["GOLDEN"]
        assert not out_files[2]["GOLDEN"]
        assert not out_files[3]["GOLDEN"]

        # Read the Golden File
        golden_hd = io.HERAData(out_files[1]['GOLDEN'])
        gd, gf, gn = golden_hd.read()
        
        assert gd.shape[0] == 3  # ndays
        assert len(gd.antpairs()) == 6
        assert gd.shape[1] == uvds[0][0].freq_array.size
        assert len(gd.pols()) == 2

        assert len(gd.keys()) == 12

        # Check that autos are all the same
        assert np.all(gd[(0,0,'ee')] == gd[(1, 1,'nn')])
        assert np.all(gd[(0,0,'ee')] == gd[(2, 2,'nn')])

        # Since each day is at exactly the same LST by construction, the golden data
        # over time should be the same.
        for key, data in gd.items():
            for day in data:
                np.testing.assert_allclose(data[0], day)

        assert not np.allclose(gd[(0, 1, 'ee')][0], gd[(0, 2, 'ee')][0])
        assert not np.allclose(gd[(1, 2, 'ee')][0], gd[(0, 2, 'ee')][0])
        assert not np.allclose(gd[(1, 2, 'ee')][0], gd[(0, 1, 'ee')][0])
        

    def test_save_chans(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("lstbin_golden_data")
        uvds = mockuvd.make_dataset(ndays=3, nfiles=4, ntimes=2, identifiable=True, creator=create_small_array_uvd)
        data_files = mockuvd.write_files_in_hera_format(uvds, tmp)

        cfl = tmp / "lstbin_config_file.yaml"
        config_info = lstbin_simple.make_lst_bin_config_file(
            cfl, data_files, ntimes_per_file=2,
        )

        out_files = lstbin_simple.lst_bin_files(
            config_file=cfl, save_channels=[50]
        )

        assert len(out_files) == 4
        # Ensure there's a REDUCEDCHAN file for each output LST
        for fl in out_files:
            assert fl['REDUCEDCHAN']
            
            # Read the Golden File
            hd = io.HERAData(fl['REDUCEDCHAN'])
            gd, gf, gn = hd.read()
            
            assert gd.shape[0] == 3  # ndays
            assert len(gd.antpairs()) == 6
            assert gd.shape[1] == 1  # single frequency
            assert len(gd.pols()) == 2

            assert len(gd.keys()) == 12

            # Check that autos are all the same
            assert np.all(gd[(0,0,'ee')] == gd[(1, 1,'nn')])
            assert np.all(gd[(0,0,'ee')] == gd[(2, 2,'nn')])

            # Since each day is at exactly the same LST by construction, the golden data
            # over time should be the same.
            for key, data in gd.items():
                for day in data:
                    np.testing.assert_allclose(data[0], day, rtol=1e-6)

            assert not np.allclose(gd[(0, 1, 'ee')][0], gd[(0, 2, 'ee')][0])
            assert not np.allclose(gd[(1, 2, 'ee')][0], gd[(0, 2, 'ee')][0])
            assert not np.allclose(gd[(1, 2, 'ee')][0], gd[(0, 1, 'ee')][0])
    
    def test_make_lst_bin_config_file(self, tmp_path_factory):
        tmpdir = tmp_path_factory.mktemp("lstbin_config_file")

        cfl = tmpdir / "lstbin_config_file.yaml"
        uvds = mockuvd.make_dataset(ndays=3, nfiles=4, ntimes=2, identifiable=True, creator=create_small_array_uvd)
        data_files = mockuvd.write_files_in_hera_format(uvds, tmpdir)

        config_info = lstbin_simple.make_lst_bin_config_file(
            cfl, data_files
        )

        assert 'metadata' in config_info

    def test_baseline_chunking(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("baseline_chunking")
        uvds = mockuvd.make_dataset(
            ndays=3, nfiles=4, ntimes=2, 
            creator=mockuvd.create_uvd_identifiable,
            antpairs = [(i,j) for i in range(10) for j in range(i, 10)],  # 55 antpairs
            pols = ['xx', 'yy'],
            freqs=np.linspace(140e6, 180e6, 12),
        )
        data_files = mockuvd.write_files_in_hera_format(uvds, tmp)

        cfl = tmp / "lstbin_config_file.yaml"
        config_info = lstbin_simple.make_lst_bin_config_file(
            cfl, data_files, ntimes_per_file=2,
        )

        out_files = lstbin_simple.lst_bin_files(
            config_file=cfl, fname_format="zen.{kind}.{lst:7.5f}.uvh5",
        )
        out_files_chunked = lstbin_simple.lst_bin_files(
            config_file=cfl, fname_format="zen.{kind}.{lst:7.5f}.chunked.uvh5",
            Nbls_to_load=10,
        )

        for flset, flsetc in zip(out_files, out_files_chunked):
            assert flset['LST'] != flsetc['LST']
            uvdlst = UVData()
            uvdlst.read(flset['LST'])

            uvdlstc = UVData()
            uvdlstc.read(flsetc['LST'])

            assert uvdlst == uvdlstc
            expected = mockuvd.identifiable_data_from_uvd(uvdlst)

            np.testing.assert_allclose(uvdlst.data_array, expected, rtol=1e-4)


    def test_lstbin_compare_nontrivial_cal(
        self, tmp_path_factory
    ):
        tmp = tmp_path_factory.mktemp("nontrivial_cal")
        uvds = mockuvd.make_dataset(
            ndays=3, nfiles=4, ntimes=2, 
            creator=mockuvd.create_uvd_identifiable,
            antpairs = [(i,j) for i in range(7) for j in range(i, 7)],  # 55 antpairs
            pols = ('xx', 'yy'),
            freqs=np.linspace(140e6, 180e6, 3),
        )
        uvcs = [
            [mockuvd.make_uvc_identifiable(d) for d in uvd ] for uvd in uvds
        ]
        
        data_files = mockuvd.write_files_in_hera_format(uvds, tmp)
        cal_files = mockuvd.write_cals_in_hera_format(uvcs, tmp)
        decal_files = [[df.replace(".uvh5", ".decal.uvh5") for df in dfl] for dfl in data_files]        
        
        for flist, clist, ulist in zip(data_files, cal_files, decal_files):
            for df, cf, uf in zip(flist, clist, ulist):
                apply_cal.apply_cal(
                    df, uf, cf,
                    gain_convention='divide',  # go the wrong way
                    clobber=True,
                )

        # First, let's go the other way to check if we get the same thing back directly
        recaled_files = [[df.replace(".uvh5", ".recal.uvh5") for df in dfl] for dfl in data_files]
        for flist, clist, ulist in zip(recaled_files, cal_files, decal_files):
            for df, cf, uf in zip(flist, clist, ulist):
                apply_cal.apply_cal(
                    uf, df, cf,
                    gain_convention='multiply',  # go the wrong way
                    clobber=True,
                )

        for flset, flsetc in zip(data_files, recaled_files):
            for fl, flc in zip(flset, flsetc):
                uvdlst = UVData()
                uvdlst.read(fl)

                uvdlstc = UVData()
                uvdlstc.read(flc)
                np.testing.assert_allclose(uvdlst.data_array, uvdlstc.data_array)

        cfl = tmp / "lstbin_config_file.yaml"
        config_info = lstbin_simple.make_lst_bin_config_file(
            cfl, decal_files, ntimes_per_file=2,
        )
        
        out_files_recal = lstbin_simple.lst_bin_files(
            config_file=cfl, calfile_rules=[(".decal.uvh5", ".calfits")],
            fname_format="zen.{kind}.{lst:7.5f}.recal.uvh5",
            Nbls_to_load=10, 
        )

        config_info = lstbin_simple.make_lst_bin_config_file(
            cfl, data_files, ntimes_per_file=2, clobber=True,
        )
        out_files = lstbin_simple.lst_bin_files(
            config_file=cfl, fname_format="zen.{kind}.{lst:7.5f}.uvh5",
            Nbls_to_load=11, 
        )

        for flset, flsetc in zip(out_files, out_files_recal):
            assert flset['LST'] != flsetc['LST']
            uvdlst = UVData()
            uvdlst.read(flset['LST'])

            uvdlstc = UVData()
            uvdlstc.read(flsetc['LST'])

            # Don't worry about history here, because we know they use different inputs
            expected = mockuvd.identifiable_data_from_uvd(uvdlst)

            strpols = utils.polnum2str(uvdlst.polarization_array)
            for i, ap in enumerate(uvdlst.get_antpairs()):
                for j, pol in enumerate(strpols):
                    print(f"Baseline {ap + (pol,)}")

                    # Unfortunately, we don't have LSTs for the files that exactly align
                    # with bin centres, so some rephasing will happen -- we just have to
                    # live with it and change the tolerance
                    # Furthermore, we only check where the flags are False, because
                    # when we put in flags, we end up setting the data to 1.0 (and 
                    # never using it...)
                    np.testing.assert_allclose(
                        uvdlstc.get_data(ap+(pol,)), 
                        np.where(uvdlst.get_flags(ap+(pol,)), 1.0, expected[i, :, :, j]), 
                        rtol=1e-4
                    )

            uvdlst.history = uvdlstc.history

            if all(fs==0 for fs in flag_strategy):
                # It only makes sense to compare full UVData objects if we're not
                # flagging in the UVCals, since the non-recal files will have
                # all samples, while the recal files will have fewer than that
                assert uvdlst == uvdlstc
    
    


    @pytest.mark.parametrize("random_ants_to_drop", (0, 3))
    @pytest.mark.parametrize("rephase", [True, False])
    @pytest.mark.parametrize("sigma_clip_thresh", [0.0, 3.0])
    @pytest.mark.parametrize("flag_strategy",[(0,0,0), (2,1,3)])
    @pytest.mark.parametrize("pols", [('xx', 'yy'), ('xx', 'yy', 'xy', 'yx')])
    def test_lstbin_with_nontrivial_cal(
        self, tmp_path_factory, random_ants_to_drop: int, rephase: bool, 
        sigma_clip_thresh: float, flag_strategy: tuple[int, int, int],
        pols: tuple[str]
    ):
        tmp = tmp_path_factory.mktemp("nontrivial_cal")
        uvds = mockuvd.make_dataset(
            ndays=3, nfiles=2, ntimes=2, 
            creator=mockuvd.create_uvd_identifiable,
            antpairs = [(i,j) for i in range(7) for j in range(i, 7)],  # 55 antpairs
            pols = pols,
            freqs=np.linspace(140e6, 180e6, 3),
            random_ants_to_drop=random_ants_to_drop,
        )

        uvcs = [
            [mockuvd.make_uvc_identifiable(d, *flag_strategy) for d in uvd ] for uvd in uvds
        ]
        
        data_files = mockuvd.write_files_in_hera_format(uvds, tmp)
        cal_files = mockuvd.write_cals_in_hera_format(uvcs, tmp)
        decal_files = [[df.replace(".uvh5", ".decal.uvh5") for df in dfl] for dfl in data_files]        
        
        for flist, clist, ulist in zip(data_files, cal_files, decal_files):
            for df, cf, uf in zip(flist, clist, ulist):
                apply_cal.apply_cal(
                    df, uf, cf,
                    gain_convention='divide',  # go the wrong way
                    clobber=True,
                )

        cfl = tmp / "lstbin_config_file.yaml"
        config_info = lstbin_simple.make_lst_bin_config_file(
            cfl, data_files, ntimes_per_file=2, clobber=True,
        )
        out_files = lstbin_simple.lst_bin_files(
            config_file=cfl, fname_format="zen.{kind}.{lst:7.5f}.uvh5",
            Nbls_to_load=11, rephase=rephase,
            sigma_clip_thresh=sigma_clip_thresh,
            sigma_clip_min_N=2,
        )
        assert len(out_files) == 2
        for flset in out_files:
            uvdlst = UVData()
            uvdlst.read(flset['LST'])

            # Don't worry about history here, because we know they use different inputs
            expected = mockuvd.identifiable_data_from_uvd(uvdlst, reshape=False)
            
            strpols = utils.polnum2str(uvdlst.polarization_array)
            
            for i, ap in enumerate(uvdlst.get_antpairs()):
                for j, pol in enumerate(strpols):
                    print(f"Baseline {ap + (pol,)}")

                    # Unfortunately, we don't have LSTs for the files that exactly align
                    # with bin centres, so some rephasing will happen -- we just have to
                    # live with it and change the tolerance
                    # Furthermore, we only check where the flags are False, because
                    # when we put in flags, we end up setting the data to 1.0 (and 
                    # never using it...)
                    np.testing.assert_allclose(
                        uvdlst.get_data(ap+(pol,)), 
                        np.where(uvdlst.get_flags(ap+(pol,)), 1.0, expected[i, :, :, j]), 
                        rtol=1e-4 if (not rephase or (ap[0] == ap[1] and pol[0]==pol[1])) else 1e-3
                    )
    
    
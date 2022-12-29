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
from .. import io, lstbin, utils, redcal
from ..datacontainer import DataContainer
from ..data import DATA_PATH
import shutil


@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
@pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0 for slice")
@pytest.mark.filterwarnings("ignore:Mean of empty slice")
@pytest.mark.filterwarnings("ignore:invalid value encountered in true_divide")
@pytest.mark.filterwarnings("ignore:invalid value encountered in greater")
class Test_lstbin(object):
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
        ap, a = hd3.get_ENU_antpos(center=True, pick_data_ants=True)
        t = np.hstack(list(hd3.times.values()))

        hd1 = io.HERAData(self.data_files[0])
        hd2 = io.HERAData(self.data_files[1])
        hd3 = io.HERAData(self.data_files[2])
        self.data1, self.flgs1, self.nsmps1 = hd1.read()
        self.ap1, self.freqs1, self.lsts1 = list(hd1.pols.values())[0], list(hd1.freqs.values())[0], np.hstack(list(hd1.lsts.values()))
        self.data2, self.flgs2, self.nsmps2 = hd2.read()
        self.ap2, self.freqs2, self.lsts2 = list(hd2.pols.values())[0], list(hd2.freqs.values())[0], np.hstack(list(hd2.lsts.values()))
        self.data3, self.flgs3, self.nsmps3 = hd3.read()
        self.ap3, self.freqs3, self.lsts3 = list(hd3.pols.values())[0], list(hd3.freqs.values())[0], np.hstack(list(hd3.lsts.values()))
        ap, a = hd3.get_ENU_antpos(center=True, pick_data_ants=True)
        t = np.hstack(list(hd3.times.values()))

        self.data_list = [self.data1, self.data2, self.data3]
        self.flgs_list = [self.flgs1, self.flgs2, self.flgs3]
        self.lst_list = [self.lsts1, self.lsts2, self.lsts3]
        self.nsmp_list = [self.nsmps1, self.nsmps2, self.nsmps3]

    def test_make_lst_grid(self):
        lst_grid = lstbin.make_lst_grid(0.01, begin_lst=None, verbose=False)
        assert len(lst_grid) == 628
        assert np.isclose(lst_grid[0], 0.0050025360725952121)
        lst_grid = lstbin.make_lst_grid(0.01, begin_lst=np.pi, verbose=False)
        assert len(lst_grid) == 628
        assert np.isclose(lst_grid[0], 3.1365901175171982)
        lst_grid = lstbin.make_lst_grid(0.01, begin_lst=-np.pi, verbose=False)
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
            assert np.median(np.diff(lst_grid)) == dlst
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

    def test_lstbin_filess_inhomogenous_baselines(self, tmpdir):
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

    def tearDown(self):
        output_files = sorted(glob.glob("./zen.ee.LST*") + glob.glob("./zen.ee.STD*"))
        for of in output_files:
            if os.path.exists(of):
                os.remove(of)

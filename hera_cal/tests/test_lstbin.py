'''Tests for lstbin.py'''
import nose.tools as nt
import os
import shutil
import json
import numpy as np
import aipy
import optparse
import sys
from pyuvdata import UVCal, UVData
from pyuvdata import utils as uvutils
import hera_cal as hc
from hera_cal.data import DATA_PATH
from collections import OrderedDict as odict
import copy
from hera_cal.datacontainer import DataContainer
import glob
import scipy.stats as stats


class Test_lstbin:

    def setUp(self):
        # load data
        np.random.seed(0)
        self.data_files = [sorted(glob.glob(DATA_PATH+'/zen.2458042.4*uvA')),
                           sorted(glob.glob(DATA_PATH+'/zen.2458043.4*uvA')),
                           sorted(glob.glob(DATA_PATH+'/zen.2458044.4*uvA'))]

        (self.data1, self.flgs1, self.ap1, a, self.freqs1, t, self.lsts1,
         p) = hc.abscal.UVData2AbsCalDict(self.data_files[0], return_meta=True)
        (self.data2, self.flgs2, ap, a, self.freqs2, t, self.lsts2,
         p) = hc.abscal.UVData2AbsCalDict(self.data_files[1], return_meta=True)
        (self.data3, self.flgs3, ap, a, self.freqs3, t, self.lsts3,
         p) = hc.abscal.UVData2AbsCalDict(self.data_files[2], return_meta=True)
        self.data_list = [self.data1, self.data2, self.data3]
        self.flgs_list = [self.flgs1, self.flgs2, self.flgs3]
        self.lst_list = [self.lsts1, self.lsts2, self.lsts3]

    def test_lstbin(self):
        dlst = 0.0007830490163484
        # test basic execution
        output = hc.lstbin.lst_bin(self.data_list, self.lst_list, flags_list=self.flgs_list, dlst=dlst,
                                   median=True, lst_low=0, lst_hi=np.pi, verbose=False)
        output = hc.lstbin.lst_bin(self.data_list, self.lst_list, flags_list=self.flgs_list, dlst=dlst,
                                   verbose=False)
        # check shape and dtype
        nt.assert_equal(output[0][(24,25,'xx')].dtype, np.complex)
        nt.assert_equal(output[0][(24,25,'xx')].shape, (224, 64))
        # check number of points in each bin
        nt.assert_almost_equal(output[-1][(24, 25, 'xx')].real[0,30], 1)
        nt.assert_almost_equal(output[-1][(24, 25, 'xx')].real[30,30], 2)
        nt.assert_almost_equal(output[-1][(24, 25, 'xx')].real[100,30], 3)
        nt.assert_almost_equal(output[-1][(24, 25, 'xx')].real[190,30], 2)
        nt.assert_almost_equal(output[-1][(24, 25, 'xx')].real[220,30], 1)
        # check with large spacing lst_grid
        output = hc.lstbin.lst_bin(self.data_list, self.lst_list, dlst=.01, verbose=False)
        nt.assert_almost_equal(output[-1][(24, 25, 'xx')].real[10,30], 39)
        # check flgs are propagated
        flgs1 = copy.deepcopy(self.flgs1)
        flgs1[(24, 25, 'xx')][:, 32] = True
        flgs2 = copy.deepcopy(self.flgs2)
        flgs2[(24, 25, 'xx')][:, 32] = True
        flgs3 = copy.deepcopy(self.flgs3)
        flgs_list = [flgs1, flgs2, flgs3]
        output = hc.lstbin.lst_bin(self.data_list, self.lst_list, dlst=dlst, flags_list=flgs_list)
        nt.assert_almost_equal(output[1][(24, 25, 'xx')][0, 32], True)
        nt.assert_almost_equal(output[1][(24, 25, 'xx')][180, 32], False)
        nt.assert_almost_equal(output[1][(24, 25, 'xx')][210, 32], False)
        # test return no avg
        output = hc.lstbin.lst_bin(self.data_list, self.lst_list, dlst=dlst, flags_list=self.flgs_list, return_no_avg=True)
        nt.assert_equal(len(output[0][output[0].keys()[0]][100]), 3)
        nt.assert_equal(len(output[0][output[0].keys()[0]][100][0]), 64)
        # test switch bl
        conj_data3 = DataContainer(odict(map(lambda k: (hc.lstbin.switch_bl(k), np.conj(self.data3[k])), self.data3.keys())))
        data_list = [self.data1, self.data2, conj_data3]
        output = hc.lstbin.lst_bin(data_list, self.lst_list, dlst=dlst)
        nt.assert_equal(output[0][(24,25,'xx')].shape, (224, 64))

    def test_lst_align(self):
        # test basic execution
        output = hc.lstbin.lst_align(self.data1, self.lsts1, dlst=None, flags=self.flgs1, flag_extrapolate=True, verbose=False)
        nt.assert_equal(output[0][(24,25,'xx')].shape, (180, 64))
        nt.assert_equal(len(output[2]), 180)
        nt.assert_almost_equal(output[2][0], 0.1836249943336998)
        # test flag extrapolate
        nt.assert_true(output[1][(24,25,'xx')][0].min())
        # test no dlst
        output = hc.lstbin.lst_align(self.data1, self.lsts1, dlst=None, flags=self.flgs1, flag_extrapolate=True, verbose=False)
        # test wrapped lsts
        lsts = (self.lsts1 + 6) % (2*np.pi)
        output = hc.lstbin.lst_align(self.data1, lsts, dlst=None, flags=self.flgs1, flag_extrapolate=True, verbose=False)
        nt.assert_almost_equal(output[2][150], 0.018401651884187409)

    def test_lst_align_files(self):
        # basic execution
        hc.lstbin.lst_align_files(self.data_files[0][0], outdir="./", overwrite=True)
        nt.assert_true(os.path.exists('./zen.2458042.40141.xx.HH.uvA.L.0.18362'))
        if os.path.exists('./zen.2458042.40141.xx.HH.uvA.L.0.18362'):
            shutil.rmtree('./zen.2458042.40141.xx.HH.uvA.L.0.18362')

    def test_lst_bin_files(self):
        # basic execution
        hc.lstbin.lst_bin_files(self.data_files, ntimes_per_file=250, outdir="./", overwrite=True,
                                verbose=False)
        nt.assert_true(os.path.exists('./zen.xx.LST.0.18441.uv'))
        nt.assert_true(os.path.exists('./zen.xx.STD.0.18441.uv'))
        nt.assert_true(os.path.exists('./zen.xx.NUM.0.18441.uv'))
        shutil.rmtree('./zen.xx.LST.0.18441.uv')
        shutil.rmtree('./zen.xx.STD.0.18441.uv')
        shutil.rmtree('./zen.xx.NUM.0.18441.uv')
        # tst lst_align
        hc.lstbin.lst_bin_files(self.data_files, ntimes_per_file=250, outdir="./", overwrite=True,
                                verbose=False, align=True)
        nt.assert_true(os.path.exists('./zen.xx.LST.0.18441.uv'))
        nt.assert_true(os.path.exists('./zen.xx.STD.0.18441.uv'))
        nt.assert_true(os.path.exists('./zen.xx.NUM.0.18441.uv'))
        shutil.rmtree('./zen.xx.LST.0.18441.uv')
        shutil.rmtree('./zen.xx.STD.0.18441.uv')
        shutil.rmtree('./zen.xx.NUM.0.18441.uv')

        data_files = [sorted(glob.glob(DATA_PATH+'/zen.2458042.*uvA')),
                      sorted(glob.glob(DATA_PATH+'/zen.2458043.*uvA')),
                      sorted(glob.glob(DATA_PATH+'/zen.2458044.*uvA'))]
        # test data_list is empty
        hc.lstbin.lst_bin_files(data_files, ntimes_per_file=30, outdir="./", overwrite=True,
                                verbose=False)
        nt.assert_true(os.path.exists("./zen.xx.LST.0.18441.uv"))
        nt.assert_true(os.path.exists("./zen.xx.LST.4.74175.uv"))
        output_files = np.concatenate([glob.glob("./zen.xx.LST*"),
                                       glob.glob("./zen.xx.STD*"),
                                       glob.glob("./zen.xx.NUM*")])
        for of in output_files:
            if os.path.exists(of):
                shutil.rmtree(of)

        # test smaller ntimes file output, sweeping through f_select
        hc.lstbin.lst_bin_files(self.data_files, ntimes_per_file=120, outdir="./", overwrite=True,
                                verbose=False)
        output_files = np.concatenate([glob.glob("./zen.xx.LST*"),
                                       glob.glob("./zen.xx.STD*"),
                                       glob.glob("./zen.xx.NUM*")])
        for of in output_files:
            if os.path.exists(of):
                shutil.rmtree(of)

    def test_lst_bin_arg_parser(self):
        a = hc.lstbin.lst_bin_arg_parser()

    def test_lst_align_arg_parser(self):
        a = hc.lstbin.lst_align_arg_parser()

    def test_data_to_miriad(self):
        # test basic execution
        hc.lstbin.data_to_miriad("ex.uv", self.data1, self.lsts1, self.freqs1, self.ap1,
                                 flags=self.flgs1, outdir="./", start_jd=2458042)
        # test w/ no flags
        hc.lstbin.data_to_miriad("ex.uv", self.data1, self.lsts1, self.freqs1, self.ap1,
                                 outdir="./", start_jd=2458042, overwrite=True)
        nt.assert_true(os.path.exists('ex.uv'))
        uvd = UVData()
        uvd.read_miriad('ex.uv')
        nt.assert_equal(uvd.get_data(24,25,'xx').shape, (180, 64))
        nt.assert_almost_equal(uvd.get_data(24,25,'xx')[90, 32], (-0.010416029+0.0016994481j))
        shutil.rmtree('ex.uv')
        # test exception
        nt.assert_raises(AttributeError, hc.lstbin.data_to_miriad, "ex.uv", self.data1, self.lsts1,
                         self.freqs1, self.ap1, outdir="./")
        # return uvd
        uvd = hc.lstbin.data_to_miriad("ex.uv", self.data1, self.lsts1, self.freqs1, self.ap1,
                                        outdir="./", start_jd=2458042, overwrite=True, return_uvdata=True)
        nt.assert_equal(type(uvd), UVData)
        shutil.rmtree('ex.uv')

    def test_sigma_clip(self):
        # test basic execution
        np.random.seed(0)
        x = stats.norm.rvs(0, 1, 1000)
        x[10] = 3
        x[11] = -3
        arr = hc.lstbin.sigma_clip(x, sigma=2.0)
        nt.assert_equal(np.isnan(arr[10]), True)
        nt.assert_equal(np.isnan(arr[11]), True)
        # test array performance
        x = np.array(map(lambda s: stats.norm.rvs(0, s, 100), np.arange(1, 5.1, 1)))
        x[0, 50] = 100
        x[4, 50] = 5
        arr = hc.lstbin.sigma_clip(x, sigma=2.0)
        nt.assert_true(np.isnan(arr[0,50]))
        nt.assert_false(np.isnan(arr[4,50]))

    def test_switch_bl(self):
        # test basic execution
        key = (1, 2, 'xx')
        sw_k = hc.lstbin.switch_bl(key)
        nt.assert_equal(sw_k, (2, 1, 'xx'))

    def test_lst_rephase(self):
        # test single dlst
        data = copy.deepcopy(self.data1)
        bls = odict(map(lambda k: (k, self.ap1[k[0]] - self.ap1[k[1]]), self.data1.keys()))
        hc.lstbin.lst_rephase(data, bls, self.freqs1, dlst=.001)
        r = data[(24, 25, 'xx')] / self.data1[(24,25,'xx')]
        # ensure a single delay across time has been applied
        diff = np.diff(np.angle(r[50,10:54]))
        nt.assert_true(np.isclose(diff - np.nanmedian(diff), 0).min())
        nt.assert_true(np.isclose(np.nanmax(np.abs(r)), 1))
        nt.assert_true(np.isclose(np.nanmin(np.abs(r)), 1))
        nt.assert_true(np.isclose(r[10,50], r[11,50]))
        # test multiple dlst
        dlst = np.linspace(.0005, .001, 180)
        data = copy.deepcopy(self.data1)
        hc.lstbin.lst_rephase(data, bls, self.freqs1, dlst=dlst)
        r = data[(24,25,'xx')] / self.data1[(24,25,'xx')]
        # ensure multiple delays across time have been applied
        nt.assert_false(np.isclose(r[10, 50], r[11, 50]))





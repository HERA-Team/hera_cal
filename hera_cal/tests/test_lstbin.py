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

        (self.data1, self.wgts1, self.ap1, a, self.freqs1, t, self.lsts1,
         p) = hc.abscal.UVData2AbsCalDict(self.data_files[0], return_meta=True, return_wgts=True)
        (self.data2, self.wgts2, ap, a, self.freqs2, t, self.lsts2,
         p) = hc.abscal.UVData2AbsCalDict(self.data_files[1], return_meta=True, return_wgts=True)
        (self.data3, self.wgts3, ap, a, self.freqs3, t, self.lsts3,
         p) = hc.abscal.UVData2AbsCalDict(self.data_files[2], return_meta=True, return_wgts=True)
        self.data_list = [self.data1, self.data2, self.data3]
        self.wgts_list = [self.wgts1, self.wgts2, self.wgts3]
        self.lst_list = [self.lsts1, self.lsts2, self.lsts3]

    def test_lstbin(self):
        # test basic execution
        output = hc.lstbin.lst_bin(self.data_list, self.lst_list, wgts_list=self.wgts_list, dlst=0.001, median=True, lst_low=np.pi, lst_hi=2*np.pi)
        output = hc.lstbin.lst_bin(self.data_list, self.lst_list, wgts_list=self.wgts_list)
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
        lst_grid = np.arange(np.pi, 3*np.pi, 0.01)
        output = hc.lstbin.lst_bin(self.data_list, self.lst_list, lst_grid=lst_grid)
        nt.assert_almost_equal(output[-1][(24, 25, 'xx')].real[10,30], 37)
        # check wgts are propagated
        wgts1 = copy.deepcopy(self.wgts1)
        wgts1[(24, 25, 'xx')][:, 32] = 0.0
        wgts2 = copy.deepcopy(self.wgts2)
        wgts2[(24, 25, 'xx')][:, 32] = 0.0
        wgts3 = copy.deepcopy(self.wgts3)
        wgts_list = [wgts1, wgts2, wgts3]
        output = hc.lstbin.lst_bin(self.data_list, self.lst_list, wgts_list=wgts_list)
        nt.assert_almost_equal(output[1][(24, 25, 'xx')][0, 32], 0.0)
        nt.assert_almost_equal(output[1][(24, 25, 'xx')][180, 32], 0.5)
        nt.assert_almost_equal(output[1][(24, 25, 'xx')][210, 32], 1.0)

    def test_lst_align(self):
        # test basic execution
        output = hc.lstbin.lst_align(self.data1, self.lsts1, wgts=self.wgts1)
        nt.assert_equal(output[0][(24,25,'xx')].shape, (179, 64))
        nt.assert_equal(len(output[2]), 179)
        nt.assert_almost_equal(output[2][0], 6.4673213537662111)

    def test_lst_align_files(self):
        # basic execution
        hc.lstbin.lst_align_files(self.data_files[0][0], outdir="./", overwrite=True)
        nt.assert_true(os.path.exists('./zen.2458042.40141.xx.HH.uvA.L.0.18414'))
        if os.path.exists('./zen.2458042.40141.xx.HH.uvA.L.0.18414'):
            shutil.rmtree('./zen.2458042.40141.xx.HH.uvA.L.0.18414')

    def test_lst_bin_files(self):
        # basic execution
        hc.lstbin.lst_bin_files(self.data_files, ntimes_per_file=250, outdir="./", overwrite=True,
                                verbose=False)
        nt.assert_true(os.path.exists('./zen.xx.LST.0.18492.uv'))
        nt.assert_true(os.path.exists('./zen.xx.STD.0.18492.uv'))
        nt.assert_true(os.path.exists('./zen.xx.NUM.0.18492.uv'))
        shutil.rmtree('./zen.xx.LST.0.18492.uv')
        shutil.rmtree('./zen.xx.STD.0.18492.uv')
        shutil.rmtree('./zen.xx.NUM.0.18492.uv')
        # tst lst_align
        hc.lstbin.lst_bin_files(self.data_files, ntimes_per_file=250, outdir="./", overwrite=True,
                                verbose=False, lst_low=6.5, lst_hi=6.6, align=True)
        nt.assert_true(os.path.exists('./zen.xx.LST.0.21702.uv'))
        nt.assert_true(os.path.exists('./zen.xx.STD.0.21702.uv'))
        nt.assert_true(os.path.exists('./zen.xx.NUM.0.21702.uv'))
        shutil.rmtree('./zen.xx.LST.0.21702.uv')
        shutil.rmtree('./zen.xx.STD.0.21702.uv')
        shutil.rmtree('./zen.xx.NUM.0.21702.uv')
        # test skip nightly data
        hc.lstbin.lst_bin_files(self.data_files, ntimes_per_file=250, outdir="./", overwrite=True,
                                verbose=False, lst_low=6.55, lst_hi=6.6)
        nt.assert_true(os.path.exists('./zen.xx.LST.0.26713.uv'))
        nt.assert_true(os.path.exists('./zen.xx.STD.0.26713.uv'))
        nt.assert_true(os.path.exists('./zen.xx.NUM.0.26713.uv'))
        shutil.rmtree('./zen.xx.LST.0.26713.uv')
        shutil.rmtree('./zen.xx.STD.0.26713.uv')
        shutil.rmtree('./zen.xx.NUM.0.26713.uv')
        # test exception
        nt.assert_raises(ValueError, hc.lstbin.lst_bin_files, self.data_files, lst_low=0.21, lst_hi=0.19)

    def test_lst_bin_arg_parser(self):
        a = hc.lstbin.lst_bin_arg_parser()

    def test_data_to_miriad(self):
        # test basic execution
        flags1 = odict(map(lambda k: (k, ~(self.wgts1[k].astype(np.bool))), self.wgts1.keys()))
        flags1 = hc.datacontainer.DataContainer(flags1)
        hc.lstbin.data_to_miriad("ex.uv", self.data1, self.lsts1, self.freqs1, self.ap1,
                                 flags=flags1, outdir="./", start_jd=2458042)
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

    def test_wrap(self):
        # basic execution
        arr = np.arange(np.pi, 3*np.pi, 0.01)
        arr_wrap = hc.lstbin.wrap(arr)
        nt.assert_equal(arr_wrap[-1], 3.138407346410073)
        # starting below zero
        arr = np.arange(-np.pi, np.pi, 0.01)
        arr_wrap = hc.lstbin.wrap(arr)
        nt.assert_equal(arr_wrap[0], 3.141592653589793)
        # starting above 2pi
        arr = np.arange(3*np.pi, 5*np.pi, 0.01)
        arr_wrap = hc.lstbin.wrap(arr)
        nt.assert_equal(arr_wrap[0], 3.141592653589793)
        
    def test_unwrap(self):
        # test basic execution
        arr = np.arange(np.pi, 3*np.pi, 0.01)
        arr_wrap = hc.lstbin.wrap(arr)
        arr_unwrap = hc.lstbin.unwrap(arr_wrap)
        nt.assert_almost_equal(np.abs(arr-arr_unwrap).max(), 0)
        # start below zero
        arr_unwrap = hc.lstbin.unwrap(arr_wrap - 2*np.pi)
        nt.assert_almost_equal(np.abs(arr-arr_unwrap).max(), 0)
        # start above 2pi
        arr_unwrap = hc.lstbin.unwrap(arr_wrap + 2*np.pi)
        nt.assert_almost_equal(np.abs(arr-arr_unwrap + 2*np.pi).max(), 0)

    def test_switch_bl(self):
        # test basic execution
        key = (1, 2, 'xx')
        sw_k = hc.lstbin.switch_bl(key)
        nt.assert_equal(sw_k, (2, 1, 'xx'))



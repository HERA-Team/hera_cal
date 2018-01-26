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

class Test_lstbin:

    def setUp(self):
        # load data
        np.random.seed(0)
        self.data_files = [sorted(glob.glob(DATA_PATH+'/zen.2458042.4*uvA')),
                           sorted(glob.glob(DATA_PATH+'/zen.2458043.4*uvA')),
                           sorted(glob.glob(DATA_PATH+'/zen.2458044.4*uvA'))]

        (self.data1, self.wgts1, ap, a, self.freqs1, t, self.lsts1,
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
        nt.assert_almost_equal(output[-1][(24, 25, 'xx')].real[10,30], 39)
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
        nt.assert_almost_equal(output[2][0], 6.4681043387343315)

    def test_lst_align_files(self):




    def test_lst_bin_files(self):





    def test_lst_align_arg_parser(self):
        a = hc.lstbin.lst_align_arg_parser()


    def test_lst_bin_arg_parser(self):
        a = hc.lstbin.lst_bin_arg_parser()


    def test_data_to_miriad(self):




    def test_sigma_clip(self):





    def test_wrap(self):


    def test_unwrap(self):




    def test_switch_bl(self):

        








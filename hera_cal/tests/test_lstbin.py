# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License
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
from hera_cal import io


class Test_lstbin:

    def setUp(self):
        # load data
        np.random.seed(0)
        self.data_files = [sorted(glob.glob(DATA_PATH + '/zen.2458043.4*uvXRAA')),
                           sorted(glob.glob(DATA_PATH + '/zen.2458044.4*uvXRAA')),
                           sorted(glob.glob(DATA_PATH + '/zen.2458045.4*uvXRAA'))]

        (self.data1, self.flgs1, self.ap1, a, self.freqs1, t, self.lsts1,
         p) = hc.io.load_vis(self.data_files[0], return_meta=True)
        (self.data2, self.flgs2, ap, a, self.freqs2, t, self.lsts2,
         p) = hc.io.load_vis(self.data_files[1], return_meta=True)
        (self.data3, self.flgs3, ap, a, self.freqs3, t, self.lsts3,
         p) = hc.io.load_vis(self.data_files[2], return_meta=True)
        self.data_list = [self.data1, self.data2, self.data3]
        self.flgs_list = [self.flgs1, self.flgs2, self.flgs3]
        self.lst_list = [self.lsts1, self.lsts2, self.lsts3]

    def test_make_lst_grid(self):
        lst_grid = hc.lstbin.make_lst_grid(0.01, lst_start=None, verbose=False)
        nt.assert_equal(len(lst_grid), 628)
        nt.assert_almost_equal(lst_grid[0], 0.0050025360725952121)
        lst_grid = hc.lstbin.make_lst_grid(0.01, lst_start=np.pi, verbose=False)
        nt.assert_equal(len(lst_grid), 628)
        nt.assert_almost_equal(lst_grid[0], 3.1365901175171982)
        lst_grid = hc.lstbin.make_lst_grid(0.01, lst_start=-np.pi, verbose=False)
        nt.assert_equal(len(lst_grid), 628)
        nt.assert_almost_equal(lst_grid[0], 3.1365901175171982)

    def test_lstbin(self):
        dlst = 0.0007830490163484
        # test basic execution
        output = hc.lstbin.lst_bin(self.data_list, self.lst_list, flags_list=self.flgs_list, dlst=None,
                                   median=True, lst_low=0, lst_hi=np.pi, verbose=False)
        output = hc.lstbin.lst_bin(self.data_list, self.lst_list, flags_list=None, dlst=0.01,
                                   verbose=False)
        output = hc.lstbin.lst_bin(self.data_list, self.lst_list, flags_list=self.flgs_list, dlst=dlst,
                                   verbose=False)
        # check shape and dtype
        nt.assert_equal(output[1][(24, 25, 'xx')].dtype, np.complex)
        nt.assert_equal(output[1][(24, 25, 'xx')].shape, (224, 64))
        # check number of points in each bin
        nt.assert_almost_equal(output[-1][(24, 25, 'xx')].real[0, 30], 1)
        nt.assert_almost_equal(output[-1][(24, 25, 'xx')].real[30, 30], 2)
        nt.assert_almost_equal(output[-1][(24, 25, 'xx')].real[100, 30], 3)
        nt.assert_almost_equal(output[-1][(24, 25, 'xx')].real[190, 30], 2)
        nt.assert_almost_equal(output[-1][(24, 25, 'xx')].real[220, 30], 1)
        # check with large spacing lst_grid
        output = hc.lstbin.lst_bin(self.data_list, self.lst_list, dlst=.01, verbose=False)
        nt.assert_almost_equal(output[-1][(24, 25, 'xx')].real[10, 30], 38)
        # check flgs are propagated
        flgs1 = copy.deepcopy(self.flgs1)
        flgs1[(24, 25, 'xx')][:, 32] = True
        flgs2 = copy.deepcopy(self.flgs2)
        flgs2[(24, 25, 'xx')][:, 32] = True
        flgs3 = copy.deepcopy(self.flgs3)
        flgs_list = [flgs1, flgs2, flgs3]
        output = hc.lstbin.lst_bin(self.data_list, self.lst_list, dlst=dlst, flags_list=flgs_list)
        nt.assert_almost_equal(output[2][(24, 25, 'xx')][0, 32], True)
        nt.assert_almost_equal(output[2][(24, 25, 'xx')][180, 32], False)
        nt.assert_almost_equal(output[2][(24, 25, 'xx')][210, 32], False)
        # test return no avg
        output = hc.lstbin.lst_bin(self.data_list, self.lst_list, dlst=dlst, flags_list=self.flgs_list, return_no_avg=True)
        nt.assert_equal(len(output[2][output[2].keys()[0]][100]), 3)
        nt.assert_equal(len(output[2][output[2].keys()[0]][100][0]), 64)
        # test switch bl
        conj_data3 = DataContainer(odict(map(lambda k: (hc.lstbin.switch_bl(k), np.conj(self.data3[k])), self.data3.keys())))
        data_list = [self.data1, self.data2, conj_data3]
        output = hc.lstbin.lst_bin(data_list, self.lst_list, dlst=dlst)
        nt.assert_equal(output[1][(24, 25, 'xx')].shape, (224, 64))
        # test sigma clip
        output = hc.lstbin.lst_bin(self.data_list, self.lst_list, flags_list=None, dlst=0.01,
                                   verbose=False, sig_clip=True, min_N=5, sigma=2)
        # test wrapping
        lst_list = map(lambda l: (copy.deepcopy(l) + 6) % (2 * np.pi), self.lst_list)
        output = hc.lstbin.lst_bin(self.data_list, lst_list, dlst=0.001, lst_start=np.pi)
        nt.assert_true(output[0][0] > output[0][-1])
        nt.assert_equal(len(output[0]), 175)
        # test appropriate data_count
        output = hc.lstbin.lst_bin(self.data_list, self.lst_list, flags_list=None, dlst=dlst, lst_low=0.25, lst_hi=0.3,
                                   verbose=False)
        nt.assert_true(np.isclose(output[4][(24, 25, 'xx')], 3.0).all())

    def test_lst_align(self):
        # test basic execution
        output = hc.lstbin.lst_align(self.data1, self.lsts1, dlst=None, flags=self.flgs1, flag_extrapolate=True, verbose=False)
        nt.assert_equal(output[0][(24, 25, 'xx')].shape, (180, 64))
        nt.assert_equal(len(output[2]), 180)
        nt.assert_almost_equal(output[2][0], 0.20163512170971379)
        # test flag extrapolate
        nt.assert_true(output[1][(24, 25, 'xx')][-1].min())
        # test no dlst
        output = hc.lstbin.lst_align(self.data1, self.lsts1, dlst=None, flags=self.flgs1, flag_extrapolate=True, verbose=False)
        # test wrapped lsts
        lsts = (self.lsts1 + 6) % (2 * np.pi)
        output = hc.lstbin.lst_align(self.data1, lsts, dlst=None, flags=self.flgs1, flag_extrapolate=True, verbose=False)
        nt.assert_almost_equal(output[2][150], 0.035628730243852047)

    def test_lst_align_files(self):
        # basic execution
        hc.lstbin.lst_align_files(self.data_files[0][0], outdir="./", overwrite=True, verbose=False)
        nt.assert_true(os.path.exists('./zen.2458043.40141.xx.HH.uvXRAA.L.0.20124'))
        if os.path.exists('./zen.2458043.40141.xx.HH.uvXRAA.L.0.20124'):
            shutil.rmtree('./zen.2458043.40141.xx.HH.uvXRAA.L.0.20124')

    def test_lst_bin_files(self):
        # basic execution
        hc.lstbin.lst_bin_files(self.data_files, ntimes_per_file=250, outdir="./", overwrite=True,
                                verbose=False)
        output_lst_file = "./zen.xx.LST.0.20124.uv"
        output_std_file = "./zen.xx.STD.0.20124.uv"
        nt.assert_true(os.path.exists(output_lst_file))
        nt.assert_true(os.path.exists(output_std_file))
        shutil.rmtree(output_lst_file)
        shutil.rmtree(output_std_file)
        # test rephase
        hc.lstbin.lst_bin_files(self.data_files, ntimes_per_file=250, outdir="./", overwrite=True,
                                verbose=False, rephase=True)
        output_lst_file = "./zen.xx.LST.0.20124.uv"
        output_std_file = "./zen.xx.STD.0.20124.uv"
        nt.assert_true(os.path.exists(output_lst_file))
        nt.assert_true(os.path.exists(output_std_file))
        shutil.rmtree(output_lst_file)
        shutil.rmtree(output_std_file)

        data_files = [[sorted(glob.glob(DATA_PATH + '/zen.2458043.*uvXRAA'))[0]],
                      [sorted(glob.glob(DATA_PATH + '/zen.2458045.*uvXRAA'))[-1]]]
        # test data_list is empty
        hc.lstbin.lst_bin_files(data_files, ntimes_per_file=30, outdir="./", overwrite=True,
                                verbose=False)
        output_lst_files = ['./zen.xx.LST.0.20124.uv', './zen.xx.LST.0.31870.uv', './zen.xx.LST.0.36568.uv']
        nt.assert_true(os.path.exists(output_lst_files[0]))
        nt.assert_true(os.path.exists(output_lst_files[1]))
        nt.assert_true(os.path.exists(output_lst_files[2]))
        output_files = np.concatenate([glob.glob("./zen.xx.LST*"),
                                       glob.glob("./zen.xx.STD*")])
        for of in output_files:
            if os.path.exists(of):
                shutil.rmtree(of)

        # test smaller ntimes file output, sweeping through f_select
        hc.lstbin.lst_bin_files(self.data_files, ntimes_per_file=80, outdir="./", overwrite=True,
                                verbose=False, vis_units='Jy')
        output_files = sorted(glob.glob("./zen.xx.LST*") + glob.glob("./zen.xx.STD*"))
        # load a file
        uvd1 = UVData()
        uvd1.read_miriad(output_files[1])
        nt.assert_equal(uvd1.vis_units, 'Jy')
        nt.assert_equal(uvd1.Ntimes, 80)
        nt.assert_almost_equal(uvd1.nsample_array.max(), 3.0)
        # remove files
        for of in output_files:
            if os.path.exists(of):
                shutil.rmtree(of)

        # test output_file_select
        hc.lstbin.lst_bin_files(self.data_files, ntimes_per_file=80, outdir="./", overwrite=True, output_file_select=1,
                                verbose=False, vis_units='Jy')
        output_files = sorted(glob.glob("./zen.xx.LST*") + glob.glob("./zen.xx.STD*"))
        # load a file
        uvd2 = UVData()
        uvd2.read_miriad(output_files[0])
        # assert equivalence with previous run
        nt.assert_equal(uvd1, uvd2)
        # remove files
        for of in output_files:
            if os.path.exists(of):
                shutil.rmtree(of)
        # assert bad output_file_select produces no files
        hc.lstbin.lst_bin_files(self.data_files, ntimes_per_file=80, outdir="./", overwrite=True, output_file_select=100,
                                verbose=False)
        output_files = sorted(glob.glob("./zen.xx.LST*") + glob.glob("./zen.xx.STD*"))

        # test fixed start
        hc.lstbin.lst_bin_files(self.data_files, ntimes_per_file=250, outdir="./", overwrite=True,
                                verbose=False, lst_start=0.18, fixed_lst_start=True)
        output_lst_file = "./zen.xx.LST.0.17932.uv"
        output_std_file = "./zen.xx.STD.0.17932.uv"
        nt.assert_true(os.path.exists(output_lst_file))
        nt.assert_true(os.path.exists(output_std_file))
        shutil.rmtree(output_lst_file)
        shutil.rmtree(output_std_file)

    def test_lst_bin_arg_parser(self):
        a = hc.lstbin.lst_bin_arg_parser()

    def test_lst_align_arg_parser(self):
        a = hc.lstbin.lst_align_arg_parser()

    def test_sigma_clip(self):
        # test basic execution
        np.random.seed(0)
        x = stats.norm.rvs(0, 1, 1000)
        x[10] = 4
        x[11] = -4
        arr = hc.lstbin.sigma_clip(x, sigma=2.0)
        nt.assert_true(arr[10])
        nt.assert_true(arr[11])
        # test array performance
        x = np.array(map(lambda s: stats.norm.rvs(0, s, 100), np.arange(1, 5.1, 1)))
        x[0, 50] = 100
        x[4, 50] = 5
        arr = hc.lstbin.sigma_clip(x, sigma=2.0)
        nt.assert_true(arr[0, 50])
        nt.assert_false(arr[4, 50])
        # test flags
        arr = stats.norm.rvs(0, 1, 10).reshape(2, 5)
        flg = np.zeros_like(arr, np.bool)
        flg[0, 3] = True
        out = hc.lstbin.sigma_clip(arr, flags=flg, min_N=5)
        nt.assert_false(out[0, 3])
        out = hc.lstbin.sigma_clip(arr, flags=flg, min_N=1)
        nt.assert_true(out[0, 3])

    def test_switch_bl(self):
        # test basic execution
        key = (1, 2, 'xx')
        sw_k = hc.lstbin.switch_bl(key)
        nt.assert_equal(sw_k, (2, 1, 'xx'))

    def tearDown(self):
        output_files = sorted(glob.glob("./zen.xx.LST*") + glob.glob("./zen.xx.STD*"))
        for of in output_files:
            if os.path.exists(of):
                shutil.rmtree(of)

# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

from __future__ import print_function, division, absolute_import

import nose.tools as nt
import os
import shutil
import json
import numpy as np
import aipy
import optparse
import sys
from collections import OrderedDict as odict
import copy
import glob
from six.moves import map
import scipy.stats as stats
from pyuvdata import UVCal, UVData
from pyuvdata import utils as uvutils

import hera_cal as hc
from hera_cal import io
from hera_cal.datacontainer import DataContainer
from hera_cal.data import DATA_PATH


class Test_lstbin:

    def setUp(self):
        # load data
        np.random.seed(0)
        self.data_files = [sorted(glob.glob(DATA_PATH + '/zen.2458043.4*XRAA.uvh5')),
                           sorted(glob.glob(DATA_PATH + '/zen.2458044.4*XRAA.uvh5')),
                           sorted(glob.glob(DATA_PATH + '/zen.2458045.4*XRAA.uvh5'))]

        (self.data1, self.flgs1, self.ap1, a, self.freqs1, t, self.lsts1,
         p) = hc.io.load_vis(self.data_files[0], return_meta=True, filetype='uvh5')
        (self.data2, self.flgs2, ap, a, self.freqs2, t, self.lsts2,
         p) = hc.io.load_vis(self.data_files[1], return_meta=True, filetype='uvh5')
        (self.data3, self.flgs3, ap, a, self.freqs3, t, self.lsts3,
         p) = hc.io.load_vis(self.data_files[2], return_meta=True, filetype='uvh5')
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
        nt.assert_equal(len(output[2][list(output[2].keys())[0]][100]), 3)
        nt.assert_equal(len(output[2][list(output[2].keys())[0]][100][0]), 64)
        # test switch bl
        conj_data3 = DataContainer(odict(list(map(lambda k: (hc.lstbin.switch_bl(k), np.conj(self.data3[k])), self.data3.keys()))))
        data_list = [self.data1, self.data2, conj_data3]
        output = hc.lstbin.lst_bin(data_list, self.lst_list, dlst=dlst)
        nt.assert_equal(output[1][(24, 25, 'xx')].shape, (224, 64))
        # test sigma clip
        output = hc.lstbin.lst_bin(self.data_list, self.lst_list, flags_list=None, dlst=0.01,
                                   verbose=False, sig_clip=True, min_N=5, sigma=2)
        # test wrapping
        lst_list = list(map(lambda l: (copy.deepcopy(l) + 6) % (2 * np.pi), self.lst_list))
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

    def test_lst_bin_files(self):
        # basic execution
        file_ext = "{pol}.{type}.{time:7.5f}.uvh5"
        hc.lstbin.lst_bin_files(self.data_files, ntimes_per_file=250, outdir="./", overwrite=True,
                                verbose=False, file_ext=file_ext, ignore_flags=True)
        output_lst_file = "./zen.xx.LST.0.20124.uvh5"
        output_std_file = "./zen.xx.STD.0.20124.uvh5"
        nt.assert_true(os.path.exists(output_lst_file))
        nt.assert_true(os.path.exists(output_std_file))
        uv1 = UVData()
        uv1.read(output_lst_file)
        # assert nsample w.r.t time follows 1-2-3-2-1 pattern
        nsamps = np.mean(uv1.get_nsamples(52, 52, 'xx'), axis=1)
        expectation = np.concatenate([np.ones(22), np.ones(22) * 2, np.ones(136) * 3, np.ones(22) * 2, np.ones(21)]).astype(np.float)
        nt.assert_true(np.isclose(nsamps, expectation).all())
        os.remove(output_lst_file)
        os.remove(output_std_file)

        # test with multiple blgroups
        hc.lstbin.lst_bin_files(self.data_files, ntimes_per_file=250, outdir="./", overwrite=True,
                                verbose=False, file_ext=file_ext, Nblgroups=3, ignore_flags=True)
        nt.assert_true(os.path.exists(output_lst_file))
        nt.assert_true(os.path.exists(output_std_file))
        uv2 = UVData()
        uv2.read(output_lst_file)
        nt.assert_equal(uv1, uv2)
        os.remove(output_lst_file)
        os.remove(output_std_file)

        # test rephase
        hc.lstbin.lst_bin_files(self.data_files, ntimes_per_file=250, outdir="./", overwrite=True,
                                verbose=False, rephase=True, file_ext=file_ext)
        output_lst_file = "./zen.xx.LST.0.20124.uvh5"
        output_std_file = "./zen.xx.STD.0.20124.uvh5"
        nt.assert_true(os.path.exists(output_lst_file))
        nt.assert_true(os.path.exists(output_std_file))
        os.remove(output_lst_file)
        os.remove(output_std_file)

        # test data_list is empty
        data_files = [[sorted(glob.glob(DATA_PATH + '/zen.2458043.*XRAA.uvh5'))[0]],
                      [sorted(glob.glob(DATA_PATH + '/zen.2458045.*XRAA.uvh5'))[-1]]]
        hc.lstbin.lst_bin_files(data_files, ntimes_per_file=30, outdir="./", overwrite=True,
                                verbose=False, file_ext=file_ext)
        output_lst_files = ['./zen.xx.LST.0.20124.uvh5', './zen.xx.LST.0.31870.uvh5', './zen.xx.LST.0.36568.uvh5']
        nt.assert_true(os.path.exists(output_lst_files[0]))
        nt.assert_true(os.path.exists(output_lst_files[1]))
        nt.assert_true(os.path.exists(output_lst_files[2]))
        output_files = np.concatenate([glob.glob("./zen.xx.LST*"),
                                       glob.glob("./zen.xx.STD*")])
        for of in output_files:
            if os.path.exists(of):
                os.remove(of)

        # test smaller ntimes file output, sweeping through f_select
        hc.lstbin.lst_bin_files(self.data_files, ntimes_per_file=80, outdir="./", overwrite=True,
                                verbose=False, vis_units='Jy', file_ext=file_ext)
        output_files = sorted(glob.glob("./zen.xx.LST*") + glob.glob("./zen.xx.STD*"))
        # load a file
        uvd1 = UVData()
        uvd1.read(output_files[1])
        nt.assert_equal(uvd1.vis_units, 'Jy')
        nt.assert_true('Thisfilewasproducedbythefunction' in uvd1.history.replace('\n', '').replace(' ', ''))
        nt.assert_equal(uvd1.Ntimes, 80)
        nt.assert_almost_equal(uvd1.nsample_array.max(), 3.0)
        # remove files
        for of in output_files:
            if os.path.exists(of):
                os.remove(of)

        # test output_file_select
        hc.lstbin.lst_bin_files(self.data_files, ntimes_per_file=80, outdir="./", overwrite=True, output_file_select=1,
                                verbose=False, vis_units='Jy', file_ext=file_ext)
        output_files = sorted(glob.glob("./zen.xx.LST*") + glob.glob("./zen.xx.STD*"))
        # load a file
        uvd2 = UVData()
        uvd2.read(output_files[0])
        # assert equivalence with previous run
        nt.assert_equal(uvd1, uvd2)
        # remove files
        for of in output_files:
            if os.path.exists(of):
                os.remove(of)

        # assert bad output_file_select produces no files
        output_files = sorted(glob.glob("./zen.xx.LST*") + glob.glob("./zen.xx.STD*"))
        for of in output_files:
            if os.path.exists(of):
                os.remove(of)
        hc.lstbin.lst_bin_files(self.data_files, ntimes_per_file=80, outdir="./", overwrite=True, output_file_select=100,
                                verbose=False, file_ext=file_ext)
        output_files = sorted(glob.glob("./zen.xx.LST*") + glob.glob("./zen.xx.STD*"))
        nt.assert_equal(len(output_files), 0)

        # test fixed start
        hc.lstbin.lst_bin_files(self.data_files, ntimes_per_file=250, outdir="./", overwrite=True,
                                verbose=False, lst_start=0.18, fixed_lst_start=True, file_ext=file_ext)
        output_lst_file = "./zen.xx.LST.0.17932.uvh5"
        output_std_file = "./zen.xx.STD.0.17932.uvh5"
        nt.assert_true(os.path.exists(output_lst_file))
        nt.assert_true(os.path.exists(output_std_file))
        os.remove(output_lst_file)
        os.remove(output_std_file)

        # test input_cal
        uvc = UVCal()
        uvc.read_calfits(os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA.abs.calfits'))
        uvc.flag_array[uvc.ant_array.tolist().index(24)] = True
        uvc.gain_array[uvc.ant_array.tolist().index(25)] = 1e10
        input_cals = []
        for dfiles in self.data_files:
            input_cals.append([uvc for df in dfiles])

        hc.lstbin.lst_bin_files(self.data_files, ntimes_per_file=250, outdir="./", overwrite=True,
                                verbose=False, input_cals=input_cals, file_ext=file_ext)

        output_lst_file = "./zen.xx.LST.0.20124.uvh5"
        output_std_file = "./zen.xx.STD.0.20124.uvh5"
        nt.assert_true(os.path.exists(output_lst_file))
        nt.assert_true(os.path.exists(output_std_file))
        os.remove(output_lst_file)
        os.remove(output_std_file)

        # test input_cal with only one Ntimes
        input_cals = []
        for dfiles in self.data_files:
            input_cals.append([uvc.select(times=uvc.time_array[:1], inplace=False) for df in dfiles])
        hc.lstbin.lst_bin_files(self.data_files, ntimes_per_file=250, outdir="./", overwrite=True,
                                verbose=False, input_cals=input_cals, file_ext=file_ext)
        nt.assert_true(os.path.exists(output_lst_file))
        nt.assert_true(os.path.exists(output_std_file))

        # assert gains and flags were propagated
        lstb = UVData()
        lstb.read(output_lst_file)
        nt.assert_true(np.isclose(np.abs(lstb.get_data(25, 37)[~lstb.get_flags(25, 37)]), 0.0).all())
        nt.assert_true(lstb.get_flags(24, 25).all())

        os.remove(output_lst_file)
        os.remove(output_std_file)

    def test_lst_bin_arg_parser(self):
        a = hc.lstbin.lst_bin_arg_parser()
        args = a.parse_args(["--dlst", "0.1", "--input_cals", "zen.2458043.12552.HH.uvA.omni.calfits", "zen.2458043.12552.xx.HH.uvORA.abs.calfits",
                             "--overwrite", "zen.2458042.12552.xx.HH.uvXA", "zen.2458042.12552.xx.HH.uvXA"])

        nt.assert_almost_equal(args.dlst, 0.1)
        nt.assert_true(len(args.input_cals), 2)
        nt.assert_true(len(args.data_files), 2)

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
        x = np.array(list(map(lambda s: stats.norm.rvs(0, s, 100), np.arange(1, 5.1, 1))))
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
                os.remove(of)

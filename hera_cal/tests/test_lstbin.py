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
from pyuvdata import UVCal, UVData, UVFlag

from .. import io, lstbin, utils
from ..datacontainer import DataContainer
from ..data import DATA_PATH


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
        self.data_uvd = []
        for dfs in self.data_files:
            _uvds = []
            for df in dfs:
                uvd = UVData()
                uvd.read(df)
                _uvds.append(uvd)
            self.data_uvd.append(_uvds)

        (self.data1, self.flgs1, self.ap1, a, self.freqs1, t, self.lsts1,
         p) = io.load_vis(self.data_uvd[0], return_meta=True, filetype="uvh5")
        (self.data2, self.flgs2, ap, a, self.freqs2, t, self.lsts2,
         p) = io.load_vis(self.data_uvd[1], return_meta=True, filetype="uvh5")
        (self.data3, self.flgs3, ap, a, self.freqs3, t, self.lsts3,
         p) = io.load_vis(self.data_uvd[2], return_meta=True, filetype="uvh5")
        self.data_list = [self.data1, self.data2, self.data3]
        self.flgs_list = [self.flgs1, self.flgs2, self.flgs3]
        self.lst_list = [self.lsts1, self.lsts2, self.lsts3]

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
        assert output[1][(24, 25, 'ee')].dtype == np.complex
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
        conj_data3 = DataContainer(odict(list(map(lambda k: (utils.reverse_bl(k), np.conj(self.data3[k])), self.data3.keys()))))
        data_list = [self.data1, self.data2, conj_data3]
        output = lstbin.lst_bin(data_list, self.lst_list, dlst=dlst)
        assert output[1][(24, 25, 'ee')].shape == (224, 64)
        # test sigma clip
        output = lstbin.lst_bin(self.data_list, self.lst_list, flags_list=None, dlst=0.01,
                                verbose=False, sig_clip=True, min_N=5, sigma=2)
        # test wrapping
        lst_list = list(map(lambda l: (copy.deepcopy(l) + 6) % (2 * np.pi), self.lst_list))
        output = lstbin.lst_bin(self.data_list, lst_list, dlst=0.001, begin_lst=np.pi)
        assert output[0][0] > output[0][-1]
        assert len(output[0]) == 175
        # test appropriate data_count
        output = lstbin.lst_bin(self.data_list, self.lst_list, flags_list=None, dlst=dlst, lst_low=0.25, lst_hi=0.3,
                                verbose=False)
        assert np.allclose(output[4][(24, 25, 'ee')], 3.0)

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
        # generate fake UVFlag files
        flag_files = []
        for dfs, uvds in zip(self.data_files, self.data_uvd):
            _ffiles = []
            for df, uvd in zip(dfs, uvds):
                uvf = UVFlag()
                uvf.from_uvdata(uvd, mode='flag', waterfall=True)
                uvf.flag_array[:, 20] = True
                dfile = df.replace(".uvh5", ".tempflagtest.h5")
                _ffiles.append(dfile)
                uvf.write(dfile, clobber=True)
            flag_files.append(_ffiles)

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
        # assert nsample w.r.t time follows 1-2-3-2-1 pattern
        nsamps = np.mean(uv1.get_nsamples(52, 52, 'ee'), axis=1)
        expectation = np.concatenate([np.ones(22), np.ones(22) * 2, np.ones(136) * 3, np.ones(22) * 2, np.ones(21)]).astype(np.float)
        assert np.allclose(nsamps, expectation)
        # cleanup
        os.remove(output_lst_file)
        os.remove(output_std_file)

        # test with multiple blgroups
        lstbin.lst_bin_files(self.data_files, ntimes_per_file=250, outdir="./", overwrite=True,
                             verbose=False, file_ext=file_ext, Nblgroups=3, ignore_flags=True)
        assert os.path.exists(output_lst_file)
        assert os.path.exists(output_std_file)
        uv2 = UVData()
        uv2.read(output_lst_file)
        assert uv1 == uv2
        # cleanup
        os.remove(output_lst_file)
        os.remove(output_std_file)

        # test rephase and flag_filles
        lstbin.lst_bin_files(self.data_files, flag_files=flag_files, ntimes_per_file=250, outdir="./", overwrite=True,
                             verbose=False, rephase=True, file_ext=file_ext, ignore_flags=False)
        output_lst_file = "./zen.ee.LST.0.20124.uvh5"
        output_std_file = "./zen.ee.STD.0.20124.uvh5"
        assert os.path.exists(output_lst_file)
        assert os.path.exists(output_std_file)
        # check channel flagged in flag files is flagged
        uv1 = UVData()
        uv1.read(output_lst_file)
        assert np.all(uv1.flag_array[:, 0, 20, :])
        # cleanup
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
        # cleanup
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
        assert np.allclose(np.abs(lstb.get_data(25, 37)[~lstb.get_flags(25, 37)]), 0.0)
        assert np.all(lstb.get_flags(24, 25))

        os.remove(output_lst_file)
        os.remove(output_std_file)

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
        x = np.array(list(map(lambda s: stats.norm.rvs(0, s, 100), np.arange(1, 5.1, 1))))
        x[0, 50] = 100
        x[4, 50] = 5
        arr = lstbin.sigma_clip(x, sigma=2.0)
        assert np.all(arr[0, 50])
        assert not np.any(arr[4, 50])
        # test flags
        arr = stats.norm.rvs(0, 1, 10).reshape(2, 5)
        flg = np.zeros_like(arr, np.bool)
        flg[0, 3] = True
        out = lstbin.sigma_clip(arr, flags=flg, min_N=5)
        assert not np.any(out[0, 3])
        out = lstbin.sigma_clip(arr, flags=flg, min_N=1)
        assert np.all(out[0, 3])

    def tearDown(self):
        output_files = sorted(glob.glob("./zen.ee.LST*") + glob.glob("./zen.ee.STD*"))
        for of in output_files:
            if os.path.exists(of):
                os.remove(of)

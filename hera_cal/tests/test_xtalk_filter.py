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
from .. import xtalk_filter as xf
from ..data import DATA_PATH
import glob


class Test_XTalkFilter(object):
    def test_run_xtalk_filter(self):
        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        k = (24, 25, 'ee')
        xfil = xf.XTalkFilter(fname, filetype='miriad')
        xfil.read(bls=[k])
        bl = np.linalg.norm(xfil.antpos[24] - xfil.antpos[25]) / constants.c * 1e9
        sdf = (xfil.freqs[1] - xfil.freqs[0]) / 1e9

        xfil.run_xtalk_filter(to_filter=xfil.data.keys(), tol=1e-2)
        for k in xfil.data.keys():
            assert xfil.clean_resid[k].shape == (60, 64)
            assert xfil.clean_model[k].shape == (60, 64)
            assert k in xfil.clean_info

        # test skip_wgt imposition of flags
        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        k = (24, 25, 'ee')
        xfil = xf.XTalkFilter(fname, filetype='miriad')
        xfil.read(bls=[k])
        wgts = {k: np.ones_like(xfil.flags[k], dtype=np.float)}
        wgts[k][:, 0] = 0.0
        xfil.run_xtalk_filter(to_filter=[k], weight_dict=wgts, tol=1e-5, window='blackman-harris', skip_wgt=0.1, maxiter=100)
        assert xfil.clean_info[k]['status']['axis_0'][0] == 'skipped'
        np.testing.assert_array_equal(xfil.clean_flags[k][:, 0], np.ones_like(xfil.flags[k][:, 0]))
        np.testing.assert_array_equal(xfil.clean_model[k][:, 0], np.zeros_like(xfil.clean_resid[k][:, 0]))
        np.testing.assert_array_equal(xfil.clean_resid[k][:, 0], np.zeros_like(xfil.clean_resid[k][:, 0]))

    def test_load_xtalk_filter_and_write(self):
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        outfilename = os.path.join(DATA_PATH, 'test_output/temp.h5')
        xf.load_xtalk_filter_and_write(uvh5, res_outfilename=outfilename, tol=1e-4, clobber=True, Nbls_per_load=1)
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'ee')])

        xfil = xf.XTalkFilter(uvh5, filetype='uvh5')
        xfil.read(bls=[(53, 54, 'ee')])
        xfil.run_xtalk_filter(to_filter=[(53, 54, 'ee')], tol=1e-4, verbose=True)
        np.testing.assert_almost_equal(d[(53, 54, 'ee')], xfil.clean_resid[(53, 54, 'ee')], decimal=5)
        np.testing.assert_array_equal(f[(53, 54, 'ee')], xfil.flags[(53, 54, 'ee')])

        # test loading and writing all baselines at once.
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        outfilename = os.path.join(DATA_PATH, 'test_output/temp.h5')
        xf.load_xtalk_filter_and_write(uvh5, res_outfilename=outfilename, tol=1e-4, clobber=True, Nbls_per_load=None)
        hd = io.HERAData(outfilename)
        d, f, n = hd.read(bls=[(53, 54, 'ee')])

        xfil = xf.XTalkFilter(uvh5, filetype='uvh5')
        xfil.read(bls=[(53, 54, 'ee')])
        xfil.run_xtalk_filter(to_filter=[(53, 54, 'ee')], tol=1e-4, verbose=True)
        np.testing.assert_almost_equal(d[(53, 54, 'ee')], xfil.clean_resid[(53, 54, 'ee')], decimal=5)
        np.testing.assert_array_equal(f[(53, 54, 'ee')], xfil.flags[(53, 54, 'ee')])

        cal = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only")
        outfilename = os.path.join(DATA_PATH, 'test_output/temp.h5')
        xf.load_xtalk_filter_and_write(uvh5, calfile=cal, tol=1e-4, res_outfilename=outfilename, Nbls_per_load=2, clobber=True)
        hd = io.HERAData(outfilename)
        assert 'Thisfilewasproducedbythefunction' in hd.history.replace('\n', '').replace(' ', '')
        d, f, n = hd.read(bls=[(53, 54, 'ee')])
        np.testing.assert_array_equal(f[(53, 54, 'ee')], True)
        os.remove(outfilename)

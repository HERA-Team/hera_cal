import nose.tools as nt
import os
import shutil
import numpy as np
import sys
from pyuvdata import UVData
from pyuvdata import utils as uvutils
import hera_cal as hc
from uvtools.data import DATA_PATH
from collections import OrderedDict as odict
import copy
import glob
import uvtools as uvt


def test_timeavg_waterfall():
    fname = os.path.join(DATA_PATH, "zen.all.xx.LST.1.06964.uvA")

    uvd = UVData()
    uvd.read_miriad(fname)
    
    d = uvd.get_data(24, 25)
    f = uvd.get_flags(24, 25)
    n = uvd.get_nsamples(24, 25)
    t = np.unique(uvd.time_array)
    fr = uvd.freq_array.squeeze()
    l = []
    for _l in uvd.lst_array:
        if _l not in l:
            l.append(_l)
    l = np.array(l)
    antpos, ants = uvd.get_ENU_antpos()
    blv = antpos[ants.tolist().index(24)] - antpos[ants.tolist().index(25)]

    # test basic execution
    ad, af, an, al, aea = uvt.frf.timeavg_waterfall(d, 4, verbose=False)
    nt.assert_equal(ad.shape, (2, 1024))
    nt.assert_equal(af.shape, (2, 1024))
    nt.assert_equal(an.shape, (2, 1024))
    nt.assert_false(np.any(af))
    nt.assert_almost_equal(an[0, 0], 4.0)
    nt.assert_almost_equal(an[1, 0], 2.0)

    # test rephase
    ad, af, an, al, aea = uvt.frf.timeavg_waterfall(d, 4, flags=f, rephase=True, lsts=l, freqs=fr, bl_vec=blv, 
                                                    nsamples=n, extra_arrays=dict(times=t), verbose=False)
    nt.assert_equal(ad.shape, (2, 1024))
    nt.assert_equal(af.shape, (2, 1024))
    nt.assert_equal(an.shape, (2, 1024))
    nt.assert_true(np.any(af))
    nt.assert_equal(len(al), 2)
    nt.assert_equal(len(aea['avg_times']), 2)
    nt.assert_almost_equal(an.max(), 98.0)


class Test_FRFilter:

    def setUp(self):
        self.fname = os.path.join(DATA_PATH, "zen.all.xx.LST.1.06964.uvA")
        self.F = uvt.frf.FRFilter()
        self.uvd = UVData()
        self.uvd.read_miriad(self.fname)

    def test_load_data(self):
        self.F.load_data(self.fname)
        nt.assert_equal(self.F.inp_uvdata, self.uvd)
        self.F.load_data(self.uvd)
        nt.assert_equal(self.F.inp_uvdata, self.uvd)

    def test_timeavg_data(self):
        self.F.load_data(self.uvd)
        self.F.timeavg_data(600.0, rephase=True)
        nt.assert_equal(self.F.Navg, 3)

        self.F.timeavg_data(1e10, rephase=True, verbose=False)
        nt.assert_equal(self.F.Navg, 6)

        # exceptions
        nt.assert_raises(AssertionError, self.F.timeavg_data, 100.0)

    def test_write_data(self):
        self.F.load_data(self.uvd)
        self.F.timeavg_data(600.0, rephase=False, verbose=False)
        u = self.F.write_data("./out.uv", write_avg=True, filetype='miriad', overwrite=True)
        nt.assert_true(os.path.exists("./out.uv"))
        uv = UVData()
        uv.read_miriad('./out.uv')
        nt.assert_equal(u, uv)

        u = self.F.write_data("./out.uv", overwrite=False)
        nt.assert_equal(u, None)

        u = self.F.write_data("./out.uv", write_avg=False, overwrite=True)
        nt.assert_true(np.isclose(u.data_array, self.uvd.data_array).all())
        if os.path.exists("./out.uv"):
            shutil.rmtree("./out.uv")



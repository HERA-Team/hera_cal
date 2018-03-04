import nose.tools as nt
import numpy as np
import sys
import os
import shutil
from pyuvdata import UVData
from hera_cal import utils, abscal, datacontainer, io
from hera_cal.calibrations import CAL_PATH
from hera_cal.data import DATA_PATH
from hera_cal import io
from pyuvdata import UVCal
import glob
from collections import OrderedDict as odict
import copy


class TestAAFromCalfile(object):
    def setUp(self):
        # define frequencies
        self.freqs = np.array([0.15])

        # add directory with calfile
        if CAL_PATH not in sys.path:
            sys.path.append(CAL_PATH)
        self.calfile = "hera_test_calfile"

    def test_get_aa_from_calfile(self):
        aa = utils.get_aa_from_calfile(self.freqs, self.calfile)
        nt.assert_equal(len(aa), 128)

class TestAAFromUV(object):
    def setUp(self):
        # define test file that is compatible with get_aa_from_uv
        self.test_file = "zen.2457999.76839.xx.HH.uvA"

    def test_get_aa_from_uv(self):
        fn = os.path.join(DATA_PATH, self.test_file)
        uvd = UVData()
        uvd.read_miriad(fn)
        aa = utils.get_aa_from_uv(uvd)
        # like miriad, aipy will pad the aa with non-existent antennas,
        #   because there is no concept of antenna names
        nt.assert_equal(len(aa), 88)

class TestAA(object):
    def setUp(self):
        # define test file that is compatible with get_aa_from_uv
        self.test_file = "zen.2457999.76839.xx.HH.uvA"

    def test_aa_get_params(self):
        # generate aa from file
        fn = os.path.join(DATA_PATH, self.test_file)
        uvd = UVData()
        uvd.read_miriad(fn)
        aa = utils.get_aa_from_uv(uvd)

        # change one antenna position, and read it back in to check it's the same
        antpos = {'x': 0., 'y': 1., 'z': 2.}
        params = aa.get_params()
        for key in antpos.keys():
            params['0'][key] = antpos[key]
        print params['0']
        aa.set_params(params)
        new_params = aa.get_params()
        print new_params['0']
        new_top = [new_params['0'][key] for key in antpos.keys()]
        old_top = [antpos[key] for key in antpos.keys()]
        nt.assert_true(np.allclose(old_top, new_top))


def test_JD2LST():
    # test float execution
    jd = 2458042.
    nt.assert_almost_equal(utils.JD2LST(jd, longitude=21.), 3.930652307266274)
    # test array execution
    jd = np.arange(2458042, 2458046.1, .5)
    lst = utils.JD2LST(jd, longitude=21.)
    nt.assert_equal(len(lst), 9)
    nt.assert_almost_equal(lst[3], 0.81486300218170715)

def test_LST2JD():
    # test basic execution
    lst = np.pi
    jd = utils.LST2JD(lst, start_jd=2458042)
    nt.assert_almost_equal(jd, 2458042.8708433118)
    # test array execution
    lst = np.arange(np.pi, np.pi+1.1, 0.2)
    jd = utils.LST2JD(lst, start_jd=2458042)
    nt.assert_equal(len(jd), 6)
    nt.assert_almost_equal(jd[3], 2458042.9660755517)

def test_JD2RA():
    # test basic execution
    jd = 2458042.5
    ra = utils.JD2RA(jd)
    nt.assert_almost_equal(ra, 46.130897831277629)
    # test array
    jd = np.arange(2458042, 2458043.01, .2)
    ra = utils.JD2RA(jd)
    nt.assert_equal(len(ra), 6)
    nt.assert_almost_equal(ra[3], 82.229459674026003)
    # test exception
    nt.assert_raises(ValueError, utils.JD2RA, jd, epoch='foo')
    # test J2000 epoch
    ra = utils.JD2RA(jd, epoch='J2000')
    nt.assert_almost_equal(ra[0], 225.37671446615548)

def test_get_miriad_times():
    filepaths = sorted(glob.glob(DATA_PATH+"/zen.2458042.*.xx.HH.uvXA"))
    # test execution
    starts, stops, ints = utils.get_miriad_times(filepaths, add_int_buffer=False)
    nt.assert_almost_equal(starts[0], 4.7293432458811866)
    nt.assert_almost_equal(stops[0], 4.7755393587036084)
    nt.assert_almost_equal(ints[0], 0.00078298496309189868)
    nt.assert_equal(len(starts), 2)
    nt.assert_equal(len(stops), 2)
    nt.assert_equal(len(ints), 2)
    # test with integration buffer
    _starts, _stops, _ints = utils.get_miriad_times(filepaths, add_int_buffer=True)
    nt.assert_almost_equal(starts[0], _starts[0])
    nt.assert_almost_equal(stops[-1], _stops[-1])
    nt.assert_not_almost_equal(starts[1], _starts[1])
    nt.assert_not_almost_equal(stops[0], _stops[0])
    nt.assert_almost_equal(_starts[1] + _ints[1], starts[1])
    nt.assert_almost_equal(_stops[0] - _ints[0], stops[0])
    # test if str
    starts, stops, ints = utils.get_miriad_times(filepaths[0])

class Test_Gain(object):
    def setUp(self):
        calfile1 = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA.abs.calfits")
        calfile2 = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA.dly.calfits")
        calfile3 = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORAA.abs.calfits")
        uvfile1 = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.uvXRAA")
        uvfile2 = os.path.join(DATA_PATH, "zen.2458043.40887.xx.HH.uvXRAA")

        self.g1, self.f1 = io.load_cal(calfile1)
        self.g2, self.f2 = io.load_cal(calfile2)
        self.g3, self.f3 = io.load_cal(calfile3)
        self.d1, self.df1 = io.load_vis(uvfile1)
        self.d2, self.df2 = io.load_vis(uvfile2)

    def test_merge_gains(self):
        # test basic execution
        outg, outf = utils.merge_gains([self.g1, self.g2], flags=[self.f1, self.f2], gain_convention='multiply')
        nt.assert_almost_equal(outg[(24, 'jxx')][0,0], self.g1[(24,'jxx')][0,0]*self.g2[(24,'jxx')][0,0])
        nt.assert_false(outf[(24, 'jxx')][0,32])
        nt.assert_true(outf[(53, 'jxx')][0,33])

        # test other gain convention
        outg, outf = utils.merge_gains([self.g1, self.g2], gain_convention='divide')
        nt.assert_almost_equal(outg[(24, 'jxx')][0,0], self.g1[(24,'jxx')][0,0]/self.g2[(24,'jxx')][0,0])

        # test exceptions
        nt.assert_raises(ValueError, utils.merge_gains, [self.g1, self.g2], gain_convention='foo')


    def test_apply_gains(self):
        # test basic execution
        newd, newf = utils.apply_gains(self.g1, self.d1, gain_flags=self.f1, data_flags=self.df1)
        nt.assert_almost_equal(newd[(24, 25, 'xx')][0,0], (self.d1[(24, 25, 'xx')] / (self.g1[(24, 'jxx')]*np.conj(self.g2[(25, 'jxx')])))[0,0])
        nt.assert_false(newf[(52, 53, 'xx')][0, 32])
        nt.assert_true(newf[(52, 53, 'xx')][0, 33])

        # test flag missing and broadcasting across times
        newd, newf = utils.apply_gains(self.g3, self.d1, gain_convention='multiply')
        nt.assert_true(newf[(24,53,'xx')].min())

        # test exceptions
        g = copy.deepcopy(self.g1)
        g[g.keys()[0]] = g[g.keys()[0]][:5, :]
        nt.assert_raises(ValueError, utils.apply_gains, g, self.d1)
        g = copy.deepcopy(self.g1)
        g[g.keys()[0]] = g[g.keys()[0]][:, :5]
        nt.assert_raises(ValueError, utils.apply_gains, g, self.d1)



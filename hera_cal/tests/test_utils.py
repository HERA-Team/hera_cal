import nose.tools as nt
import numpy as np
import sys
import os
import shutil
from pyuvdata import UVData
from hera_cal import utils
from hera_cal.calibrations import CAL_PATH
from hera_cal.data import DATA_PATH
from pyuvdata import UVCal
import glob


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
    nt.assert_almost_equal(utils.JD2LST(2458042., 21.), 3.9306523072662745)
    nt.assert_equal(len(utils.JD2LST(np.arange(2458042, 2458046), 21.)), 4)
def test_LST2JD():
    nt.assert_almost_equal(utils.LST2JD(np.pi, 2458042, 21.), 2458042.8720297855)
    nt.assert_almost_equal(len(utils.LST2JD(np.linspace(2, 4, 4), 2458042, 21.)), 4)
def test_JD2RA():
    nt.assert_almost_equal(utils.JD2RA(2458042.0, 21.), 224.9887083968118)
    nt.assert_almost_equal(len(utils.JD2RA(np.arange(2458042, 2458046), 21.)), 4)


def test_combine_calfits():
    test_file1 = os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA.abs.calfits')
    test_file2 = os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA.dly.calfits')
    # test basic execution
    if os.path.exists('ex.calfits'):
        os.remove('ex.calfits')
    utils.combine_calfits([test_file1, test_file2], 'ex.calfits', outdir='./', overwrite=True, broadcast_flags=True)
    # test it exists
    nt.assert_true(os.path.exists('ex.calfits'))
    # test antenna number
    uvc = UVCal()
    uvc.read_calfits('ex.calfits')
    nt.assert_equal(len(uvc.antenna_numbers), 7)
    # test time number
    nt.assert_equal(uvc.Ntimes, 60)
    # test gain value got properly multiplied
    uvc_dly = UVCal()
    uvc_dly.read_calfits(test_file1)
    uvc_abs = UVCal()
    uvc_abs.read_calfits(test_file2)
    nt.assert_almost_equal(uvc_dly.gain_array[0,0,10,10,0] * uvc_abs.gain_array[0,0,10,10,0], uvc.gain_array[0,0,10,10,0])
    if os.path.exists('ex.calfits'):
        os.remove('ex.calfits')
    utils.combine_calfits([test_file1, test_file2], 'ex.calfits', outdir='./', overwrite=True, broadcast_flags=False)
    nt.assert_true(os.path.exists('ex.calfits'))
    if os.path.exists('ex.calfits'):
        os.remove('ex.calfits')

def test_get_miriad_times():
    filepaths = sorted(glob.glob(DATA_PATH+"/zen.245804?.4*uvA"))
    # test execution
    starts, stops, ints = utils.get_miriad_times(filepaths, add_int_buffer=False)
    nt.assert_almost_equal(starts[0], 0.18438445619743923)
    nt.assert_almost_equal(stops[0], 0.23058056901986126)
    nt.assert_almost_equal(ints[0], 0.00078298)
    nt.assert_equal(len(starts), 9)
    nt.assert_equal(len(stops), 9)
    nt.assert_equal(len(ints), 9)
    # test with integration buffer
    _starts, _stops, _ints = utils.get_miriad_times(filepaths, add_int_buffer=True)
    nt.assert_almost_equal(starts[0], _starts[0])
    nt.assert_almost_equal(stops[-1], _stops[-1])
    nt.assert_not_almost_equal(starts[1], _starts[1])
    nt.assert_not_almost_equal(stops[1], _stops[1])
    nt.assert_almost_equal(_starts[1] + _ints[1], starts[1])
    nt.assert_almost_equal(_stops[1] - _ints[1], stops[1])




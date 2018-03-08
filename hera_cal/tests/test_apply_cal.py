from hera_cal import io
from hera_cal import apply_cal as ac
from hera_cal.datacontainer import DataContainer
import numpy as np
import unittest
from copy import deepcopy
from pyuvdata.utils import check_histories
from pyuvdata import UVCal, UVData
from hera_cal.data import DATA_PATH
import os
import sys
import shutil
from scipy import constants
import warnings

class Test_Update_Cal(unittest.TestCase):

    def test_recalibrate_in_place(self):
        np.random.seed(21)
        vis = np.random.randn(10,10) + 1.0j*np.random.randn(10,10) 
        dc = DataContainer({(0,1,'xx'): deepcopy(vis)})
        f = np.random.randn(10,10) > 0
        flags = DataContainer({(0,1,'xx'): deepcopy(f)})
        g0_new = np.random.randn(10,10) + 1.0j*np.random.randn(10,10) 
        g1_new = np.random.randn(10,10) + 1.0j*np.random.randn(10,10)
        g_new = {(0,'x'): g0_new, (1,'x'): g1_new}
        g0_old = np.random.randn(10,10) + 1.0j*np.random.randn(10,10) 
        g1_old = np.random.randn(10,10) + 1.0j*np.random.randn(10,10)
        g_old = {(0,'x'): g0_old, (1,'x'): g1_old}
        cal_flags = {(0,'x'): np.random.randn(10,10) > 0, (1,'x'): np.random.randn(10,10) > 0}
        # test standard operation
        ac.recalibrate_in_place(dc, flags, g_new, cal_flags, old_gains=g_old, gain_convention='divide')
        for i in range(10):
            for j in range(10):
                if not f[i,j]:
                    self.assertAlmostEqual(dc[(0,1,'xx')][i,j], vis[i,j]*g0_old[i,j]*np.conj(g1_old[i,j])/g0_new[i,j]/np.conj(g1_new[i,j]))
                else:
                    self.assertAlmostEqual(dc[(0,1,'xx')][i,j],vis[i,j])
                if f[i,j] or cal_flags[(0,'x')][i,j] or cal_flags[(1,'x')][i,j]:
                    self.assertTrue(flags[(0,1,'xx')][i,j])
                else:
                    self.assertFalse(flags[(0,1,'xx')][i,j])

        # test without old cal
        dc = DataContainer({(0,1,'xx'): deepcopy(vis)})
        flags = DataContainer({(0,1,'xx'): deepcopy(f)})
        ac.recalibrate_in_place(dc, flags, g_new, cal_flags, gain_convention='divide')
        for i in range(10):
            for j in range(10):
                if not f[i,j]:
                    self.assertAlmostEqual(dc[(0,1,'xx')][i,j], vis[i,j]/g0_new[i,j]/np.conj(g1_new[i,j]))
                else:
                    self.assertAlmostEqual(dc[(0,1,'xx')][i,j],vis[i,j])

        # test multiply
        dc = DataContainer({(0,1,'xx'): deepcopy(vis)})
        flags = DataContainer({(0,1,'xx'): deepcopy(f)})
        ac.recalibrate_in_place(dc, flags, g_new, cal_flags, old_gains=g_old, gain_convention='multiply')
        for i in range(10):
            for j in range(10):
                if not f[i,j]:
                    self.assertAlmostEqual(dc[(0,1,'xx')][i,j], vis[i,j]/g0_old[i,j]/np.conj(g1_old[i,j])*g0_new[i,j]*np.conj(g1_new[i,j]))
                else:
                    self.assertAlmostEqual(dc[(0,1,'xx')][i,j],vis[i,j])

        # test flag propagation when missing antennas in gains
        dc = DataContainer({(0,1,'xx'): deepcopy(vis)})
        flags = DataContainer({(0,1,'xx'): deepcopy(f)})
        ac.recalibrate_in_place(dc, flags, {}, cal_flags, gain_convention='divide')
        np.testing.assert_array_equal(flags[(0,1,'xx')], True)
        dc = DataContainer({(0,1,'xx'): deepcopy(vis)})
        flags = DataContainer({(0,1,'xx'): deepcopy(f)})
        ac.recalibrate_in_place(dc, flags, g_new, cal_flags, old_gains={}, gain_convention='divide')
        np.testing.assert_array_equal(flags[(0,1,'xx')], True)

        # test error
        dc = DataContainer({(0,1,'xx'): deepcopy(vis)})
        flags = DataContainer({(0,1,'xx'): deepcopy(f)})
        with self.assertRaises(KeyError):
            ac.recalibrate_in_place(dc, flags, g_new, cal_flags, old_gains=g_old, gain_convention='blah')

    def test_apply_cal(self):
        fname = os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcA")
        outname = os.path.join(DATA_PATH, "test_output/zen.2457698.40355.xx.HH.applied.uvcA")
        old_cal = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.HH.uvcA.omni.calfits")
        new_cal = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.HH.uvcA.omni.calfits")
        flags_npz = os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcA.fake_flags.npz")
        
        uvd = UVData()
        uvd.read_miriad(fname)
        uvd.flag_array = np.logical_or(uvd.flag_array, np.load(flags_npz)['flag_array'])
        data, data_flags = io.load_vis(uvd)
        new_gains, new_flags = io.load_cal(new_cal)
        uvc_old = UVCal()
        uvc_old.read_calfits(old_cal)
        uvc_old.gain_array *= (3.0 + 4.0j)

        ac.apply_cal(fname, outname, new_cal, old_calibration=uvc_old, gain_convention = 'divide', 
                     flags_npz = flags_npz, filetype = 'miriad', clobber = True)
        new_data, new_flags = io.load_vis(outname)
        for k in new_data.keys():
            for i in range(new_data[k].shape[0]):
                for j in range(new_data[k].shape[1]):
                    if not new_flags[k][i,j]:
                        self.assertAlmostEqual(new_data[k][i,j] / 25.0, data[k][i,j],4)
                    if j < 300 or j > 923:
                        self.assertTrue(new_flags[k][i,j])
                    

        shutil.rmtree(outname)

    def test_apply_cal_argparser(self):
        sys.argv = [sys.argv[0], 'a', 'b', 'c']
        args = ac.apply_cal_argparser()
        self.assertEqual(args.infile, 'a')
        self.assertEqual(args.outfile, 'b')
        self.assertEqual(args.new_cal, 'c')


if __name__ == '__main__':
    unittest.main()

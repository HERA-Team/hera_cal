'''Tests for abscal.py'''
import unittest
import numpy as np
from pyuvdata import UVCal, UVData
from hera_cal.data import DATA_PATH
from collections import OrderedDict as odict
from hera_cal.datacontainer import DataContainer
import hera_cal.io as io
import os

class Test_Visibility_IO(unittest.TestCase):

    def test_load_vis(self):
        self.assertEqual(1,1)

    #TODO: implement this test
    def test_write_vis(self):
        with self.assertRaises(NotImplementedError):
            io.write_vis(None, None, None)

    
    def test_update_vis(self):
        self.assertEqual(1,1)



class Test_Calibration_IO(unittest.TestCase):

    def test_load_cal(self):
        fname = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.xx.HH.uvc.omni.calfits")
        gains, flags = io.load_cal(fname)
        self.assertEqual(len(gains.keys()),18)
        self.assertEqual(len(flags.keys()),18)

        cal = UVCal()
        cal.read_calfits(fname)
        gains, flags = io.load_cal(cal)
        self.assertEqual(len(gains.keys()),18)
        self.assertEqual(len(flags.keys()),18)

        fname_xx = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.xx.HH.uvc.omni.calfits")
        fname_yy = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.yy.HH.uvc.omni.calfits")
        gains, flags, quals, total_qual, ants, freqs, times, pols = io.load_cal([fname_xx,fname_yy], return_meta=True)
        self.assertEqual(len(gains.keys()),36)
        self.assertEqual(len(flags.keys()),36)
        self.assertEqual(len(quals.keys()),36)
        self.assertEqual(freqs.shape, (1, 1024))
        self.assertEqual(times.shape, (3,))
        self.assertEqual(sorted(pols), ['x','y'])

        cal_xx, cal_yy = UVCal(), UVCal()
        cal_xx.read_calfits(fname_xx)
        cal_yy.read_calfits(fname_yy)
        gains, flags, quals, total_qual, ants, freqs, times, pols = io.load_cal([cal_xx,cal_yy], return_meta=True)
        self.assertEqual(len(gains.keys()),36)
        self.assertEqual(len(flags.keys()),36)
        self.assertEqual(len(quals.keys()),36)
        self.assertEqual(freqs.shape, (1, 1024))
        self.assertEqual(times.shape, (3,))
        self.assertEqual(sorted(pols), ['x','y'])


    #TODO: implement this test
    def test_write_cal(self):
        with self.assertRaises(NotImplementedError):
            io.write_cal()


    def test_update_cal(self):
        # load in cal
        fname = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.xx.HH.uvc.omni.calfits")
        outname = os.path.join(DATA_PATH, "test_output/zen.2457698.40355.xx.HH.uvc.modified.calfits.")
        cal = UVCal()
        cal.read_calfits(fname)
        gains, flags, quals, total_qual, ants, freqs, times, pols = io.load_cal(fname, return_meta=True)

        #make some modifications
        new_gains = {key: (2.+1.j)*val for key,val in gains.items()}
        new_flags = {key: np.logical_not(val) for key,val in flags.items()}
        new_quals = {key: 2.*val for key,val in quals.items()}
        io.update_cal(fname, outname, gains=new_gains, flags=new_flags, quals=new_quals,
                      add_to_history='hello world', clobber=True, telescope_name='Super HERA')
        
        #test modifications
        gains, flags, quals, total_qual, ants, freqs, times, pols = io.load_cal(outname, return_meta=True)
        for k in gains.keys():
            self.assertTrue(np.all(new_gains[k] == gains[k]))
            self.assertTrue(np.all(new_flags[k] == flags[k]))
            self.assertTrue(np.all(new_quals[k] == quals[k]))
        cal2 = UVCal()
        cal2.read_calfits(outname)
        self.assertEqual(cal2.history.replace('\n',''), ('hello world' + cal.history).replace('\n',''))
        self.assertEqual(cal2.telescope_name,'Super HERA')
        os.remove(outname)

        #now try the same thing but with a UVCal object instead of path
        io.update_cal(cal, outname, gains=new_gains, flags=new_flags, quals=new_quals,
                      add_to_history='hello world', clobber=True, telescope_name='Super HERA')
        gains, flags, quals, total_qual, ants, freqs, times, pols = io.load_cal(outname, return_meta=True)
        for k in gains.keys():
            self.assertTrue(np.all(new_gains[k] == gains[k]))
            self.assertTrue(np.all(new_flags[k] == flags[k]))
            self.assertTrue(np.all(new_quals[k] == quals[k]))
        cal2 = UVCal()
        cal2.read_calfits(outname)
        self.assertEqual(cal2.history.replace('\n',''), ('hello world' + cal.history).replace('\n',''))
        self.assertEqual(cal2.telescope_name,'Super HERA')
        os.remove(outname)



if __name__ == '__main__':
    unittest.main()

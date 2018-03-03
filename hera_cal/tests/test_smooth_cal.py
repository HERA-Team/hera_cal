from hera_cal import io
from hera_cal import smooth_cal as sc
from hera_cal.datacontainer import DataContainer
import numpy as np
import unittest
from copy import deepcopy
from pyuvdata.utils import check_histories
from pyuvdata import UVCal, UVData
from hera_cal.data import DATA_PATH
import os
import shutil
from scipy import constants
import warnings

class Test_Smooth_Cal_Helper_Functions(unittest.TestCase):

    def test_drop_cross_vis(self):
        dc = DataContainer({(1,1,'xx'): np.ones((10,10)), (1,2,'xx'): np.zeros((10,10)), (1,1,'xy'):np.zeros((10,10))})
        self.assertEqual(len(dc),3)
        dc = sc.drop_cross_vis(dc)
        self.assertEqual(len(dc),1)

    def test_synthesize_ant_flags(self):
        flags = DataContainer({(0,0,'xx'): np.ones((5,5),bool),
                               (0,1,'xx'): np.ones((5,5),bool),
                               (1,2,'xx'): np.zeros((5,5),bool),
                               (2,3,'xx'): np.zeros((5,5),bool)})
        flags[(2,3,'xx')][:,4] = True
        ant_flags = sc.synthesize_ant_flags(flags)
        np.testing.assert_array_equal(ant_flags[(0,'x')], True)
        np.testing.assert_array_equal(ant_flags[(1,'x')], False)
        np.testing.assert_array_equal(ant_flags[(2,'x')][:,0:4], False)
        np.testing.assert_array_equal(ant_flags[(2,'x')][:,4], True)
        np.testing.assert_array_equal(ant_flags[(3,'x')][:,0:4], False)
        np.testing.assert_array_equal(ant_flags[(3,'x')][:,4], True)

    def test_build_weights(self):
        unnorm_chisq_per_ant = np.ones((10,10))
        autocorr = 4.0 * np.ones((10,10))
        autocorr[0,0] = 0
        flags = np.random.randn(10,10) < 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # intentionally ignore divide by zero error, since we put it in
            wgts = sc.build_weights(unnorm_chisq_per_ant, autocorr, flags)
        np.testing.assert_array_equal(wgts[flags], 0.0)
        self.assertEqual(wgts[0,0], 0)
        self.assertAlmostEqual(np.mean(wgts[wgts > 0]), 1.0)
        
    def test_time_kernel(self):
        kernel = sc.time_kernel(100, 10.0, filter_scale=1.0)
        self.assertAlmostEqual(np.sum(kernel), 1.0)
        self.assertEqual(np.max(kernel), kernel[100])
        self.assertEqual(len(kernel), 201)


class Test_Calibration_Smoother(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        super(Test_Calibration_Smoother, self).setUpClass()
        prev,here,subq = '22245','22991', '23737'
        data = os.path.join(DATA_PATH, 'test_input/zen.2458099.' + here + '.xx.HH.uvOR_36x_auto_only')
        prev_data = os.path.join(DATA_PATH, 'test_input/zen.2458099.' + prev + '.xx.HH.uvOR_36x_auto_only')
        next_data = os.path.join(DATA_PATH, 'test_input/zen.2458099.' + subq + '.xx.HH.uvOR_36x_auto_only')
        self.cal = os.path.join(DATA_PATH, 'test_input/zen.2458099.' + here + '.HH.uv.omni.calfits_36x_only')
        prev_cal = os.path.join(DATA_PATH, 'test_input/zen.2458099.' + prev + '.HH.uv.omni.calfits_36x_only')
        next_cal = os.path.join(DATA_PATH, 'test_input/zen.2458099.' + subq + '.HH.uv.omni.calfits_36x_only')
        self.sc = sc.Calibration_Smoother()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # intentionally ignore divide by zero error, since we put it in
            self.sc.load_cal(self.cal, prev_cal=prev_cal, next_cal=next_cal)
            self.sc.load_data(data, prev_data=prev_data, next_data=next_data)
            self.sc.load_cal(self.cal, prev_cal=prev_cal, next_cal=next_cal) #improves test coverage

    def test_check_consistency(self):
        has_cal = self.sc.has_cal
        self.sc.has_cal = False
        with self.assertRaises(AttributeError):
            self.sc.check_consistency()
        self.sc.has_cal = has_cal


    def test_build_weights(self):
        self.assertTrue((36,'x') in self.sc.prev_wgts.keys())
        self.assertTrue((36,'x') in self.sc.wgts.keys())
        self.assertTrue((36,'x') in self.sc.next_wgts.keys())
        np.testing.assert_array_equal(self.sc.wgts[36,'x'] == 0, self.sc.cal_flags[36,'x'])
        np.testing.assert_array_equal(self.sc.wgts[36,'x'] == 0, self.sc.data_ant_flags[36,'x'])
        np.testing.assert_array_equal(self.sc.next_wgts[36,'x'] == 0, self.sc.next_data_ant_flags[36,'x'])
        np.testing.assert_array_equal(self.sc.prev_wgts[36,'x'] == 0, self.sc.prev_data_ant_flags[36,'x'])
        

    def test_load_cal(self):
        self.assertTrue(self.sc.nFreq,1024)
        self.assertTrue(self.sc.nInt,60)
        self.assertAlmostEqual(self.sc.tInt, 10.737781226634979)
        self.assertTrue(self.sc.has_cal)
        self.assertTrue(self.sc.has_next_cal)
        self.assertTrue(self.sc.has_prev_cal)
        self.assertFalse(self.sc.freq_filtered)
        self.assertFalse(self.sc.time_filtered)
        self.assertTrue((36,'x') in self.sc.wgts.keys())
        np.testing.assert_array_equal(self.sc.gains[36,'x'], self.sc.filtered_gains[36,'x'])


    def test_load_data(self):
        self.assertEqual(self.sc.data_filetype, 'miriad')
        self.assertTrue(self.sc.has_data)
        self.assertTrue(self.sc.has_next_data)
        self.assertTrue(self.sc.has_prev_data)
        for d in (self.sc.data, self.sc.prev_data, self.sc.next_data):
            for (i,j,pol) in d.keys():
                self.assertEqual(i,j)
                self.assertEqual(pol[0],pol[1])


    def test_filtering(self):
        
        g = deepcopy(self.sc.filtered_gains[36,'x'])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.sc.freq_filter()
        g2 = deepcopy(self.sc.filtered_gains[36,'x'])
        self.assertFalse(np.all(g == g2))
        self.assertTrue(self.sc.freq_filtered)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.sc.time_filter()
        g3 = deepcopy(self.sc.filtered_gains[36,'x'])
        self.assertFalse(np.all(g == g3))
        self.sc.has_next_cal, self.sc.has_prev_cal = False, False
        self.assertTrue(self.sc.time_filtered)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.sc.time_filter()
        g4 = deepcopy(self.sc.filtered_gains[36,'x'])
        self.assertFalse(np.all(g3 == g4))
        self.sc.has_next_cal, self.sc.has_prev_cal = True, True
        self.assertTrue(self.sc.time_filtered)

        self.sc.filtered_gains[36,'x'] = g
        self.sc.time_filtered, self.sc.freq_filtered = False, False

    def test_write(self):
        outfilename = os.path.join(DATA_PATH, 'test_output/smooth_test.calfits')
        g = deepcopy(self.sc.filtered_gains[36,'x'])
        self.sc.filtered_gains[36,'x'] = np.ones_like(self.sc.filtered_gains[36,'x'])
        self.sc.write_smoothed_cal(outfilename, add_to_history='hello world', clobber=True, telescope_name='PAPER')
        self.sc.filtered_gains[36,'x'] = g

        old_cal, new_cal = UVCal(), UVCal()
        old_cal.read_calfits(self.cal)
        new_cal.read_calfits(outfilename)
        self.assertTrue(check_histories(new_cal.history, old_cal.history + 'hello world'))
        self.assertEqual(new_cal.telescope_name,'PAPER')
        gains, flags = io.load_cal(outfilename)
        np.testing.assert_array_equal(gains[36,'x'], np.ones_like(g))
        os.remove(outfilename)


if __name__ == '__main__':
    unittest.main()

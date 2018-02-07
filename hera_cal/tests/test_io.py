'''Tests for abscal.py'''
import unittest
from pyuvdata import UVCal, UVData

from collections import OrderedDict as odict
from hera_cal.datacontainer import DataContainer
import hera_cal.io as io

class Test_Visibility_IO(unittest.TestCase):

    #TODO: implement this test
    def test_load_vis(self):
        with self.assertRaises(NotImplementedError):
            io.load_vis(None)

    #TODO: implement this test
    def test_write_vis(self):
        with self.assertRaises(NotImplementedError):
            io.write_vis(None, None, None)

    #TODO: implement this test
    def test_update_vis(self):
        self.assertEqual(1,1)



class Test_Calibration_IO(unittest.TestCase):

    #TODO: implement this test
    def test_load_cal(self):
        self.assertEqual(1,1)

    def test_write_cal(self):
        with self.assertRaises(NotImplementedError):
            io.write_cal()

    def test_update_cal(self):
        with self.assertRaises(NotImplementedError):
            io.update_cal()



if __name__ == '__main__':
    unittest.main()

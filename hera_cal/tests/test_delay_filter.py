import hera_cal.delay_filter as df
import numpy as np
import unittest
from copy import deepcopy
from pyuvdata import UVCal, UVData
from hera_cal.data import DATA_PATH


class Test_Delay_Filter(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_init(self):
        dfil = df.Delay_Filter()
        self.assertFalse(dfil.writable)




if __name__ == '__main__':
    unittest.main()

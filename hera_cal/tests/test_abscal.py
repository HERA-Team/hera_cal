'''Tests for abscal.py'''
import nose.tools as nt
import os
import json
import numpy as np
import aipy
import optparse
import sys
from pyuvdata import UVCal, UVData
import hera_cal.abscal as abscal
from hera_cal.omni import compute_reds
from hera_cal.data import DATA_PATH


class Test_AbsCal:

    def setUp(self):
        self.uvd = UVData()
        self.uvd.read_miriad(os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA"))
        self.freq_array

    def noise(size, scale=1.0):
        sig = 1./np.sqrt(2)
        return 1+scale*(np.random.normal(scale=sig, size=size) + 1j*np.random.normal(scale=sig, size=size))

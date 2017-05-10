import heracal.miriad as miriad
from heracal.data import DATA_PATH
import nose.tools as nt
import numpy as np
import os

class TestMethods(object):
    def test_read_files(self):
        files = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        Ntimes = 3
        Nchans = 256
        info, data, flags = miriad.read_files([files], 'cross', 'xx', )
        info_keys = ['lsts', 'times', 'inttime', 'chwidth', 'freqs']
        for k in info.keys():
            nt.assert_true(k in info_keys)
        for k in data.keys():
            nt.assert_equal(data[k].keys(), ['xx'])
            nt.assert_equal(data[k]['xx'].shape, (3, 256))
        for k in flags.keys():
            nt.assert_equal(flags[k].keys(), ['xx'])
            nt.assert_equal(flags[k]['xx'].shape, (3, 256))
        
               

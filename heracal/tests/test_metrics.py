import unittest
from heracal import metrics
import numpy as np

np.random.seed(0)
def noise(size):
    sig = 1./np.sqrt(2)
    return np.random.normal(scale=sig, size=size) + 1j*np.random.normal(scale=sig, size=size)

class TestDataContainer(unittest.TestCase):
    def setUp(self):
        self.bls = [(1,2),(2,3),(3,4),(1,3),(2,4)] # not (3,4)
        self.pols = ['xx','yy']
        self.blpol = {}
        for bl in self.bls:
            self.blpol[bl] = {}
            for pol in self.pols: self.blpol[bl][pol] = 1j
        self.polbl = {}
        for pol in self.pols:
            self.polbl[pol] = {}
            for bl in self.bls: self.polbl[pol][bl] = 1j
        self.both = {}
        for pol in self.pols:
            for bl in self.bls: self.both[bl+(pol,)] = 1j
    def test_init(self):
        dc = metrics.DataContainer(self.blpol)
        for k in dc._data.keys(): self.assertEqual(len(k), 3)
        self.assertEqual(set(self.bls), dc._bls)
        self.assertEqual(set(self.pols), dc._pols)
        dc = metrics.DataContainer(self.polbl)
        for k in dc._data.keys(): self.assertEqual(len(k), 3)
        self.assertEqual(set(self.bls), dc._bls)
        self.assertEqual(set(self.pols), dc._pols)
        dc = metrics.DataContainer(self.both)
        for k in dc._data.keys(): self.assertEqual(len(k), 3)
        self.assertEqual(set(self.bls), dc._bls)
        self.assertEqual(set(self.pols), dc._pols)
        self.assertRaises(AssertionError, metrics.DataContainer, {(1,2,3,4):2})
    def test_bls(self):
        dc = metrics.DataContainer(self.blpol)
        self.assertEqual(set(self.bls), dc.bls())
        self.assertEqual(set(self.bls), dc.bls('xx'))
        self.assertEqual(set(self.bls), dc.bls('yy'))
        dc = metrics.DataContainer(self.polbl)
        self.assertEqual(set(self.bls), dc.bls())
        self.assertEqual(set(self.bls), dc.bls('xx'))
        self.assertEqual(set(self.bls), dc.bls('yy'))
        dc = metrics.DataContainer(self.both)
        self.assertEqual(set(self.bls), dc.bls())
        self.assertEqual(set(self.bls), dc.bls('xx'))
        self.assertEqual(set(self.bls), dc.bls('yy'))
    def test_pols(self):
        dc = metrics.DataContainer(self.blpol)
        self.assertEqual(set(self.pols), dc.pols())
        self.assertEqual(set(self.pols), dc.pols((1,2)))
        dc = metrics.DataContainer(self.polbl)
        self.assertEqual(set(self.pols), dc.pols())
        self.assertEqual(set(self.pols), dc.pols((1,2)))
        dc = metrics.DataContainer(self.both)
        self.assertEqual(set(self.pols), dc.pols())
        self.assertEqual(set(self.pols), dc.pols((1,2)))
    def test_keys(self):
        dc = metrics.DataContainer(self.blpol)
        keys = dc.keys()
        self.assertEqual(len(keys), len(self.pols) * len(self.bls))
        dc = metrics.DataContainer(self.polbl)
        keys = dc.keys()
        self.assertEqual(len(keys), len(self.pols) * len(self.bls))
        dc = metrics.DataContainer(self.both)
        keys = dc.keys()
        self.assertEqual(len(keys), len(self.pols) * len(self.bls))
    def test_getitem(self):
        dc = metrics.DataContainer(self.blpol)
        self.assertEqual(dc[(1,2,'xx')], 1j)
        self.assertEqual(dc[(2,1,'xx')], -1j)
        self.assertEqual(dc[(1,2)], {'xx':1j,'yy':1j})
        self.assertEqual(set(dc['xx'].keys()), set(self.bls))
        dc = metrics.DataContainer(self.polbl)
        self.assertEqual(dc[(1,2,'xx')], 1j)
        self.assertEqual(dc[(2,1,'xx')], -1j)
        self.assertEqual(dc[(1,2)], {'xx':1j,'yy':1j})
        self.assertEqual(set(dc['xx'].keys()), set(self.bls))
        dc = metrics.DataContainer(self.both)
        self.assertEqual(dc[(1,2,'xx')], 1j)
        self.assertEqual(dc[(2,1,'xx')], -1j)
        self.assertEqual(dc[(1,2)], {'xx':1j,'yy':1j})
        self.assertEqual(set(dc['xx'].keys()), set(self.bls))
    def test_has_key(self):
        dc = metrics.DataContainer(self.blpol)
        self.assertTrue(dc.has_key((2,3,'yy')))
        self.assertTrue(dc.has_key((2,3),'yy'))
        self.assertTrue(dc.has_key((3,2),'yy'))
        self.assertFalse(dc.has_key('xy'))
        self.assertFalse(dc.has_key((5,6)))
        self.assertFalse(dc.has_key((1,2,'xy')))
        dc = metrics.DataContainer(self.polbl)
        self.assertTrue(dc.has_key((2,3,'yy')))
        self.assertTrue(dc.has_key((2,3),'yy'))
        self.assertTrue(dc.has_key((3,2),'yy'))
        self.assertFalse(dc.has_key('xy'))
        self.assertFalse(dc.has_key((5,6)))
        self.assertFalse(dc.has_key((1,2,'xy')))
        dc = metrics.DataContainer(self.both)
        self.assertTrue(dc.has_key((2,3,'yy')))
        self.assertTrue(dc.has_key((2,3),'yy'))
        self.assertTrue(dc.has_key((3,2),'yy'))
        self.assertFalse(dc.has_key('xy'))
        self.assertFalse(dc.has_key((5,6)))
        self.assertFalse(dc.has_key((1,2,'xy')))
    def test_has_bl(self):
        dc = metrics.DataContainer(self.blpol)
        self.assertTrue(dc.has_bl((2,3)))
        self.assertFalse(dc.has_bl((0,3)))
        dc = metrics.DataContainer(self.polbl)
        self.assertTrue(dc.has_bl((2,3)))
        self.assertFalse(dc.has_bl((0,3)))
        dc = metrics.DataContainer(self.both)
        self.assertTrue(dc.has_bl((2,3)))
        self.assertFalse(dc.has_bl((0,3)))
    def test_has_pol(self):
        dc = metrics.DataContainer(self.blpol)
        self.assertTrue(dc.has_pol('xx'))
        self.assertFalse(dc.has_pol('xy'))
        dc = metrics.DataContainer(self.polbl)
        self.assertTrue(dc.has_pol('xx'))
        self.assertFalse(dc.has_pol('xy'))
        dc = metrics.DataContainer(self.both)
        self.assertTrue(dc.has_pol('xx'))
        self.assertFalse(dc.has_pol('xy'))
    def test_get(self):
        dc = metrics.DataContainer(self.blpol)
        self.assertEqual(dc.get((1,2),'yy'), 1j)
        self.assertEqual(dc.get((2,1),'yy'), -1j)
        dc = metrics.DataContainer(self.polbl)
        self.assertEqual(dc.get((1,2),'yy'), 1j)
        self.assertEqual(dc.get((2,1),'yy'), -1j)
        dc = metrics.DataContainer(self.both)
        self.assertEqual(dc.get((1,2),'yy'), 1j)
        self.assertEqual(dc.get((2,1),'yy'), -1j)

class TestMethods(unittest.TestCase):
    def setUp(self):
        self.data1 = noise(size=(100,100))
        self.data2 = noise(size=(100,100))
    def test_check_ants(self):
        reds = [[(1,2),(2,3),(3,4),(4,5),(5,6)], 
                [(1,3),(2,4),(3,5),(4,6)], 
                [(1,4),(2,5),(3,6)], [(1,5),(2,6)]]
        data = {'xx':{}}
        for bl in reduce(lambda x,y: x+y, reds):
            data['xx'][bl] = self.data1
        cnts = metrics.check_ants(reds, data)
        for i in [1,2,3,4,5]: self.assertEqual(cnts[(i,'xx')], 0)
        for bl in data['xx']:
            if 3 in bl: data['xx'][bl] = self.data2
        cnts = metrics.check_ants(reds, data)
        for i in [1,2,4,5]:
            self.assertLess(cnts[(i,'xx')], 3)
        self.assertGreater(cnts[(3,'xx')], 3)
        for bl in data['xx']:
            if 3 in bl: data['xx'][bl] = .8*self.data1 + .2 * self.data2
        cnts = metrics.check_ants(reds, data, flag_thresh=.3)
        for i in [1,2,3,4,5]: self.assertEqual(cnts[(i,'xx')], 0)
        for bl in data['xx']:
            if 3 in bl: data['xx'][bl] = .8*self.data2 + .2 * self.data1
        cnts = metrics.check_ants(reds, data, flag_thresh=.3)
        for i in [1,2,4,5]:
            self.assertLess(cnts[(i,'xx')], 3)
        self.assertGreater(cnts[(3,'xx')], 3)
    def test_check_noise_variance(self):
        data = {'xx':{}}
        wgts = {'xx':{}}
        ants = range(10)
        ant_dat = {}
        for i in ants:
            ant_dat[i] = noise(size=(100,100)) + .1 * self.data1
        for i,ai in enumerate(ants):
            for j in ants[i:]:
                data['xx'][(i,j)] = ant_dat[i] * ant_dat[j].conj()
                wgts['xx'][(i,j)] = np.ones((100,100), dtype=float)
        nos = metrics.check_noise_variance(data, wgts, 1., 1.)
        for bl in data['xx']:
            n = nos[bl+('xx',)]
            self.assertEqual(n.shape, (100-1,))
            np.testing.assert_almost_equal(n, np.ones_like(n), 0)
        nos = metrics.check_noise_variance(data, wgts, 1., 10.)
        for bl in data['xx']:
            n = nos[bl+('xx',)]
            self.assertEqual(n.shape, (100-1,))
            np.testing.assert_almost_equal(n, 10*np.ones_like(n), -1)
        nos = metrics.check_noise_variance(data, wgts, 10., 10.)
        for bl in data['xx']:
            n = nos[bl+('xx',)]
            self.assertEqual(n.shape, (100-1,))
            np.testing.assert_almost_equal(n, 106*np.ones_like(n), -2)

if __name__ == '__main__':
    unittest.main()
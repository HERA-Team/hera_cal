# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import pytest
import numpy as np
import os

from .. import abscal, datacontainer, io
from ..data import DATA_PATH


@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
class TestDataContainer(object):
    def setup_method(self):
        self.antpairs = [(1, 2), (2, 3), (3, 4), (1, 3), (2, 4)]  # not (1,4)
        self.pols = ['nn', 'ee']
        self.x_orientation = 'NORTH'
        self.blpol = {}
        for bl in self.antpairs:
            self.blpol[bl] = {}
            for pol in self.pols:
                self.blpol[bl][pol] = 1j
        self.polbl = {}
        for pol in self.pols:
            self.polbl[pol] = {}
            for bl in self.antpairs:
                self.polbl[pol][bl] = 1j
        self.both = {}
        for pol in self.pols:
            for bl in self.antpairs:
                self.both[bl + (pol,)] = 1j
        self.bools = {}
        for pol in self.pols:
            for bl in self.antpairs:
                self.bools[bl + (pol,)] = np.array([True])

    def test_init(self):
        dc = datacontainer.DataContainer(self.blpol, x_orientation=self.x_orientation)
        for k in dc._data.keys():
            assert len(k) == 3
        assert set(self.antpairs) == dc._antpairs
        assert set(self.pols) == dc._pols
        dc = datacontainer.DataContainer(self.polbl, x_orientation=self.x_orientation)
        for k in dc._data.keys():
            assert len(k) == 3
        assert set(self.antpairs) == dc._antpairs
        assert set(self.pols) == dc._pols
        dc = datacontainer.DataContainer(self.both, x_orientation=self.x_orientation)
        for k in dc._data.keys():
            assert len(k) == 3
        assert set(self.antpairs) == dc._antpairs
        assert set(self.pols) == dc._pols
        assert dc.antpos is None
        assert dc.freqs is None
        assert dc.times is None
        assert dc.lsts is None
        assert dc.times_by_bl is None
        assert dc.lsts_by_bl is None
        pytest.raises(KeyError, datacontainer.DataContainer, {(1, 2, 3, 4): 2})

    def test_antpairs(self):
        dc = datacontainer.DataContainer(self.blpol, x_orientation=self.x_orientation)
        assert set(self.antpairs) == dc.antpairs()
        assert set(self.antpairs) == dc.antpairs('nn')
        assert set(self.antpairs) == dc.antpairs('ee')
        dc = datacontainer.DataContainer(self.polbl, x_orientation=self.x_orientation)
        assert set(self.antpairs) == dc.antpairs()
        assert set(self.antpairs) == dc.antpairs('nn')
        assert set(self.antpairs) == dc.antpairs('ee')
        dc = datacontainer.DataContainer(self.both, x_orientation=self.x_orientation)
        assert set(self.antpairs) == dc.antpairs()
        assert set(self.antpairs) == dc.antpairs('nn')
        assert set(self.antpairs) == dc.antpairs('ee')

    def test_bls(self):
        dc = datacontainer.DataContainer(self.blpol, x_orientation=self.x_orientation)
        assert set(dc.keys()) == dc.bls()
        dc = datacontainer.DataContainer(self.polbl, x_orientation=self.x_orientation)
        assert set(dc.keys()) == dc.bls()
        dc = datacontainer.DataContainer(self.both, x_orientation=self.x_orientation)
        assert set(dc.keys()) == dc.bls()

    def test_pols(self):
        dc = datacontainer.DataContainer(self.blpol, x_orientation=self.x_orientation)
        assert set(self.pols) == dc.pols()
        assert set(self.pols) == dc.pols((1, 2))
        dc = datacontainer.DataContainer(self.polbl, x_orientation=self.x_orientation)
        assert set(self.pols) == dc.pols()
        assert set(self.pols) == dc.pols((1, 2))
        dc = datacontainer.DataContainer(self.both, x_orientation=self.x_orientation)
        assert set(self.pols) == dc.pols()
        assert set(self.pols) == dc.pols((1, 2))

    def test_keys(self):
        dc = datacontainer.DataContainer(self.blpol, x_orientation=self.x_orientation)
        keys = dc.keys()
        assert len(keys) == len(self.pols) * len(self.antpairs)
        dc = datacontainer.DataContainer(self.polbl, x_orientation=self.x_orientation)
        keys = dc.keys()
        assert len(keys) == len(self.pols) * len(self.antpairs)
        dc = datacontainer.DataContainer(self.both, x_orientation=self.x_orientation)
        keys = dc.keys()
        assert len(keys) == len(self.pols) * len(self.antpairs)

        for key1, key2 in zip(dc.keys(), dc):
            assert key1 == key2

    def test_values(self):
        dc = datacontainer.DataContainer(self.blpol, x_orientation=self.x_orientation)
        values = list(dc.values())
        assert len(values) == len(self.pols) * len(self.antpairs)
        assert values[0] == 1j
        dc = datacontainer.DataContainer(self.polbl, x_orientation=self.x_orientation)
        values = list(dc.values())
        assert len(values) == len(self.pols) * len(self.antpairs)
        assert values[0] == 1j
        dc = datacontainer.DataContainer(self.both, x_orientation=self.x_orientation)
        values = list(dc.values())
        assert len(values) == len(self.pols) * len(self.antpairs)
        assert values[0] == 1j

    def test_items(self):
        dc = datacontainer.DataContainer(self.blpol, x_orientation=self.x_orientation)
        items = list(dc.items())
        assert len(items) == len(self.pols) * len(self.antpairs)
        assert items[0][0][0:2] in self.antpairs
        assert items[0][0][2] in self.pols
        assert items[0][1] == 1j
        dc = datacontainer.DataContainer(self.polbl, x_orientation=self.x_orientation)
        items = list(dc.items())
        assert items[0][0][0:2] in self.antpairs
        assert items[0][0][2] in self.pols
        assert items[0][1] == 1j
        dc = datacontainer.DataContainer(self.both, x_orientation=self.x_orientation)
        items = list(dc.items())
        assert items[0][0][0:2] in self.antpairs
        assert items[0][0][2] in self.pols
        assert items[0][1] == 1j

    def test_len(self):
        dc = datacontainer.DataContainer(self.blpol, x_orientation=self.x_orientation)
        assert len(dc) == 10
        dc = datacontainer.DataContainer(self.polbl, x_orientation=self.x_orientation)
        assert len(dc) == 10
        dc = datacontainer.DataContainer(self.both, x_orientation=self.x_orientation)
        assert len(dc) == 10

    def test_del(self):
        dc = datacontainer.DataContainer(self.blpol, x_orientation=self.x_orientation)
        assert (1, 2, 'nn') in dc
        assert (1, 2, 'NN') in dc
        del dc[(1, 2, 'nn')]
        assert (1, 2, 'nn') not in dc
        assert 'nn' in dc.pols()
        assert (1, 2) in dc.antpairs()
        del dc[(1, 2, 'ee')]
        assert (1, 2) not in dc.antpairs()
        del dc[(2, 3, 'NN')]
        assert (2, 3, 'nn') not in dc
        assert 'nn' in dc.pols()
        assert (2, 3) in dc.antpairs()

    def test_getitem(self):
        dc = datacontainer.DataContainer(self.blpol, x_orientation=self.x_orientation)
        assert dc[(1, 2, 'nn')] == 1j
        assert dc[(2, 1, 'nn')] == -1j
        assert dc[(1, 2)] == {'nn': 1j, 'ee': 1j}
        assert set(dc['nn'].keys()) == set(self.antpairs)
        assert dc[(1, 2, 'nn')] == dc.get_data((1, 2, 'nn'))
        assert dc[(1, 2, 'nn')] == dc.get_data(1, 2, 'nn')
        dc = datacontainer.DataContainer(self.polbl, x_orientation=self.x_orientation)
        assert dc[(1, 2, 'nn')] == 1j
        assert dc[(2, 1, 'nn')] == -1j
        assert dc[(1, 2)] == {'nn': 1j, 'ee': 1j}
        assert set(dc['nn'].keys()) == set(self.antpairs)
        assert dc[(2, 1, 'nn')] == dc.get_data((2, 1, 'nn'))
        assert dc[(2, 1, 'nn')] == dc.get_data(2, 1, 'nn')
        dc = datacontainer.DataContainer(self.both, x_orientation=self.x_orientation)
        assert dc[(1, 2, 'nn')] == 1j
        assert dc[(2, 1, 'nn')] == -1j
        assert dc[(1, 2)] == {'nn': 1j, 'ee': 1j}
        assert set(dc['nn'].keys()) == set(self.antpairs)
        assert dc[(1, 2)] == dc.get_data((1, 2))
        assert dc[(1, 2)] == dc.get_data(1, 2)
        assert dc[(1, 2, 'XX')] == 1j
        assert dc[(2, 1, 'XX')] == -1j
        assert dc[(2, 1, 'XX')] == dc.get_data(2, 1, 'XX')
        assert dc[(2, 1, 'XX')] == dc.get_data(2, 1, 'nn')
        dc = datacontainer.DataContainer(self.bools)
        assert dc[(1, 2, 'nn')] == np.array([True])
        assert dc[(2, 1, 'nn')] == np.array([True])
        assert dc[(2, 1, 'nn')].dtype == bool
        with pytest.raises(KeyError, match=r".*(10, 1, 'nn').*(1, 10, 'nn).*"):
            dc[(10, 1, 'nn')]

    def test_has_key(self):
        dc = datacontainer.DataContainer(self.blpol, x_orientation=self.x_orientation)
        assert (2, 3, 'ee') in dc
        assert dc.has_key((2, 3), 'ee')
        assert dc.has_key((3, 2), 'ee')
        assert 'ne' not in dc
        assert (5, 6) not in dc
        assert (1, 2, 'ne') not in dc
        dc = datacontainer.DataContainer(self.polbl, x_orientation=self.x_orientation)
        assert (2, 3, 'ee') in dc
        assert dc.has_key((2, 3), 'ee')
        assert dc.has_key((3, 2), 'ee')
        assert 'ne' not in dc
        assert (5, 6) not in dc
        assert (1, 2, 'ne') not in dc
        dc = datacontainer.DataContainer(self.both, x_orientation=self.x_orientation)
        assert (2, 3, 'ee') in dc
        assert dc.has_key((2, 3), 'ee')
        assert dc.has_key((3, 2), 'ee')
        assert 'ne' not in dc
        assert (5, 6) not in dc
        assert (1, 2, 'ne') not in dc

        assert (2, 3, 'EE') in dc
        assert dc.has_key((2, 3), 'EE')
        assert dc.has_key((3, 2), 'EE')
        assert 'EE' not in dc
        assert (5, 6) not in dc
        assert (1, 2, 'NE') not in dc

        # assert switch bl
        dc[(1, 2, 'ne')] = 1j
        assert (2, 1, 'en') in dc

    def test_has_antpair(self):
        dc = datacontainer.DataContainer(self.blpol, x_orientation=self.x_orientation)
        assert dc.has_antpair((2, 3))
        assert dc.has_antpair((3, 2))
        assert not dc.has_antpair((0, 3))
        dc = datacontainer.DataContainer(self.polbl, x_orientation=self.x_orientation)
        assert dc.has_antpair((2, 3))
        assert dc.has_antpair((3, 2))
        assert not dc.has_antpair((0, 3))
        dc = datacontainer.DataContainer(self.both, x_orientation=self.x_orientation)
        assert dc.has_antpair((2, 3))
        assert dc.has_antpair((3, 2))
        assert not dc.has_antpair((0, 3))

    def test_has_pol(self):
        dc = datacontainer.DataContainer(self.blpol, x_orientation=self.x_orientation)
        assert dc.has_pol('nn')
        assert dc.has_pol('NN')
        assert not dc.has_pol('ne')
        assert not dc.has_pol('NE')
        dc = datacontainer.DataContainer(self.polbl, x_orientation=self.x_orientation)
        assert dc.has_pol('nn')
        assert dc.has_pol('NN')
        assert not dc.has_pol('ne')
        assert not dc.has_pol('NE')
        dc = datacontainer.DataContainer(self.both, x_orientation=self.x_orientation)
        assert dc.has_pol('nn')
        assert dc.has_pol('NN')
        assert not dc.has_pol('ne')
        assert not dc.has_pol('NE')

    def test_get(self):
        dc = datacontainer.DataContainer(self.blpol, x_orientation=self.x_orientation)
        assert dc.get((1, 2), 'ee') == 1j
        assert dc.get((2, 1), 'ee') == -1j
        dc = datacontainer.DataContainer(self.polbl, x_orientation=self.x_orientation)
        assert dc.get((1, 2), 'ee') == 1j
        assert dc.get((2, 1), 'ee') == -1j
        dc = datacontainer.DataContainer(self.both, x_orientation=self.x_orientation)
        assert dc.get((1, 2), 'ee') == 1j
        assert dc.get((2, 1), 'ee') == -1j
        assert dc.get((1, 2), 'EE') == 1j
        assert dc.get((2, 1), 'EE') == -1j

    def test_setter(self):
        dc = datacontainer.DataContainer(self.blpol, x_orientation=self.x_orientation)
        # test basic setting
        dc[(100, 101, 'ne')] = np.arange(100) + np.arange(100) * 1j
        assert dc[(100, 101, 'ne')].shape == (100,)
        assert dc[(100, 101, 'ne')].dtype == np.complex
        assert np.allclose(dc[(100, 101, 'ne')][1], (1 + 1j))
        assert np.allclose(dc[(101, 100, 'en')][1], (1 - 1j))
        assert len(dc.keys()) == 11
        assert (100, 101) in dc._antpairs
        assert 'ne' in dc._pols
        # test error
        pytest.raises(ValueError, dc.__setitem__, *((100, 101), 100j))

        dc = datacontainer.DataContainer(self.bools)
        dc[2, 1, 'nn'] = np.array([True])
        assert dc[(1, 2, 'nn')] == np.array([True])
        assert dc[(2, 1, 'nn')] == np.array([True])
        assert dc[(2, 1, 'nn')].dtype == bool


@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
class TestDataContainerWithRealData(object):

    def test_adder(self):
        test_file = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        d, f = io.load_vis(test_file, pop_autos=True)
        d2 = d + d
        assert np.allclose(d2[(24, 25, 'xx')][30, 30], d[(24, 25, 'xx')][30, 30] * 2)
        # test exception
        d2, f2 = io.load_vis(test_file, pop_autos=True)
        d2[list(d2.keys())[0]] = d2[list(d2.keys())[0]][:, :10]
        pytest.raises(ValueError, d.__add__, d2)
        d2[list(d2.keys())[0]] = d2[list(d2.keys())[0]][:10, :]
        pytest.raises(ValueError, d.__add__, d2)
        d2 = d + 1
        assert np.isclose(d2[(24, 25, 'xx')][30, 30], d[(24, 25, 'xx')][30, 30] + 1)

    def test_sub(self):
        test_file = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        d, f = io.load_vis(test_file, pop_autos=True)
        d2 = d - d
        assert np.allclose(d2[(24, 25, 'xx')][30, 30], 0.0)
        # test exception
        d2, f2 = io.load_vis(test_file, pop_autos=True)
        d2[list(d2.keys())[0]] = d2[list(d2.keys())[0]][:, :10]
        pytest.raises(ValueError, d.__sub__, d2)
        d2[list(d2.keys())[0]] = d2[list(d2.keys())[0]][:10, :]
        pytest.raises(ValueError, d.__sub__, d2)
        d2 = d - 1
        assert np.isclose(d2[(24, 25, 'xx')][30, 30], d[(24, 25, 'xx')][30, 30] - 1)

    def test_mul(self):
        test_file = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        d, f = io.load_vis(test_file, pop_autos=True)
        f[(24, 25, 'xx')][:, 0] = False
        f2 = f * f
        assert not np.any(f2[(24, 25, 'xx')][0, 0])
        # test exception
        d2, f2 = io.load_vis(test_file, pop_autos=True)
        d2[list(d2.keys())[0]] = d2[list(d2.keys())[0]][:, :10]
        pytest.raises(ValueError, d.__mul__, d2)
        d2[list(d2.keys())[0]] = d2[list(d2.keys())[0]][:10, :]
        pytest.raises(ValueError, d.__mul__, d2)
        d2 = d * 2
        assert np.isclose(d2[(24, 25, 'xx')][30, 30], d[(24, 25, 'xx')][30, 30] * 2)

    @pytest.mark.filterwarnings("ignore:invalid value encountered in true_divide")
    @pytest.mark.filterwarnings("ignore:invalid value encountered in floor_divide")
    def test_div(self):
        test_file = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        d, f = io.load_vis(test_file, pop_autos=True)
        d2 = d / d
        assert np.allclose(d2[(24, 25, 'xx')][30, 30], 1.0)
        d2 = d / 2.0
        assert np.allclose(d2[(24, 25, 'xx')][30, 30], d[(24, 25, 'xx')][30, 30] / 2.0)
        d2 = d // d
        assert np.allclose(d2[(24, 25, 'xx')][30, 30], 1.0)
        d2 = d // 2.0
        assert np.allclose(d2[(24, 25, 'xx')][30, 30], d[(24, 25, 'xx')][30, 30] // 2.0)

    def test_invert(self):
        test_file = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        d, f = io.load_vis(test_file, pop_autos=True)
        f2 = ~f
        bl = (24, 25, 'xx')
        assert f2[(bl)][0, 0] == ~f[bl][0, 0]
        pytest.raises(TypeError, d.__invert__)

    def test_transpose(self):
        test_file = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        d, f = io.load_vis(test_file, pop_autos=True)
        d2 = d.T
        bl = (24, 25, 'xx')
        assert d2[(bl)][0, 0] == d[bl].T[0, 0]

    def test_neg(self):
        test_file = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        d, f = io.load_vis(test_file, pop_autos=True)
        d2 = -d
        bl = (24, 25, 'xx')
        assert d2[(bl)][0, 0] == -d[(bl)][0, 0]

    def test_concatenate(self):
        test_file = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        d, f = io.load_vis(test_file, pop_autos=True)
        d2 = d.concatenate(d)
        assert d2[(24, 25, 'xx')].shape[0] == d[(24, 25, 'xx')].shape[0] * 2
        d2 = d.concatenate(d, axis=1)
        assert d2[(24, 25, 'xx')].shape[1] == d[(24, 25, 'xx')].shape[1] * 2
        d2 = d.concatenate([d, d], axis=0)
        assert d2[(24, 25, 'xx')].shape[0] == d[(24, 25, 'xx')].shape[0] * 3
        # test exceptions
        d2, f2 = io.load_vis(test_file, pop_autos=True)
        d2[list(d2.keys())[0]] = d2[list(d2.keys())[0]][:10, :10]
        pytest.raises(ValueError, d.concatenate, d2, axis=0)
        pytest.raises(ValueError, d.concatenate, d2, axis=1)

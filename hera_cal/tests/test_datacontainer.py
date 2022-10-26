# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import pytest
import numpy as np
import os
from copy import deepcopy
from hera_sim.antpos import hex_array

from .. import abscal, datacontainer, io, redcal
from ..data import DATA_PATH
from ..utils import reverse_bl


@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
class TestDataContainer(object):
    def setup_method(self):
        self.antpairs = [(1, 2), (2, 3), (3, 4), (1, 3), (2, 4)]  # not (1,4)
        self.pols = ['xx', 'yy']
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
        dc = datacontainer.DataContainer(self.blpol)
        for k in dc._data.keys():
            assert len(k) == 3
        assert set(self.antpairs) == dc._antpairs
        assert set(self.pols) == dc._pols
        dc = datacontainer.DataContainer(self.polbl)
        for k in dc._data.keys():
            assert len(k) == 3
        assert set(self.antpairs) == dc._antpairs
        assert set(self.pols) == dc._pols
        dc = datacontainer.DataContainer(self.both)
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
        dc = datacontainer.DataContainer(self.blpol)
        assert set(self.antpairs) == dc.antpairs()
        assert set(self.antpairs) == dc.antpairs('xx')
        assert set(self.antpairs) == dc.antpairs('yy')
        dc = datacontainer.DataContainer(self.polbl)
        assert set(self.antpairs) == dc.antpairs()
        assert set(self.antpairs) == dc.antpairs('xx')
        assert set(self.antpairs) == dc.antpairs('yy')
        dc = datacontainer.DataContainer(self.both)
        assert set(self.antpairs) == dc.antpairs()
        assert set(self.antpairs) == dc.antpairs('xx')
        assert set(self.antpairs) == dc.antpairs('yy')

    def test_bls(self):
        dc = datacontainer.DataContainer(self.blpol)
        assert set(dc.keys()) == dc.bls()
        dc = datacontainer.DataContainer(self.polbl)
        assert set(dc.keys()) == dc.bls()
        dc = datacontainer.DataContainer(self.both)
        assert set(dc.keys()) == dc.bls()

    def test_pols(self):
        dc = datacontainer.DataContainer(self.blpol)
        assert set(self.pols) == dc.pols()
        assert set(self.pols) == dc.pols((1, 2))
        dc = datacontainer.DataContainer(self.polbl)
        assert set(self.pols) == dc.pols()
        assert set(self.pols) == dc.pols((1, 2))
        dc = datacontainer.DataContainer(self.both)
        assert set(self.pols) == dc.pols()
        assert set(self.pols) == dc.pols((1, 2))

    def test_keys(self):
        dc = datacontainer.DataContainer(self.blpol)
        keys = dc.keys()
        assert len(keys) == len(self.pols) * len(self.antpairs)
        dc = datacontainer.DataContainer(self.polbl)
        keys = dc.keys()
        assert len(keys) == len(self.pols) * len(self.antpairs)
        dc = datacontainer.DataContainer(self.both)
        keys = dc.keys()
        assert len(keys) == len(self.pols) * len(self.antpairs)

        for key1, key2 in zip(dc.keys(), dc):
            assert key1 == key2

    def test_values(self):
        dc = datacontainer.DataContainer(self.blpol)
        values = list(dc.values())
        assert len(values) == len(self.pols) * len(self.antpairs)
        assert values[0] == 1j
        dc = datacontainer.DataContainer(self.polbl)
        values = list(dc.values())
        assert len(values) == len(self.pols) * len(self.antpairs)
        assert values[0] == 1j
        dc = datacontainer.DataContainer(self.both)
        values = list(dc.values())
        assert len(values) == len(self.pols) * len(self.antpairs)
        assert values[0] == 1j

    def test_items(self):
        dc = datacontainer.DataContainer(self.blpol)
        items = list(dc.items())
        assert len(items) == len(self.pols) * len(self.antpairs)
        assert items[0][0][0:2] in self.antpairs
        assert items[0][0][2] in self.pols
        assert items[0][1] == 1j
        dc = datacontainer.DataContainer(self.polbl)
        items = list(dc.items())
        assert items[0][0][0:2] in self.antpairs
        assert items[0][0][2] in self.pols
        assert items[0][1] == 1j
        dc = datacontainer.DataContainer(self.both)
        items = list(dc.items())
        assert items[0][0][0:2] in self.antpairs
        assert items[0][0][2] in self.pols
        assert items[0][1] == 1j

    def test_len(self):
        dc = datacontainer.DataContainer(self.blpol)
        assert len(dc) == 10
        dc = datacontainer.DataContainer(self.polbl)
        assert len(dc) == 10
        dc = datacontainer.DataContainer(self.both)
        assert len(dc) == 10

    def test_del(self):
        dc = datacontainer.DataContainer(self.blpol)
        assert (1, 2, 'xx') in dc
        assert (1, 2, 'XX') in dc
        del dc[(1, 2, 'xx')]
        assert (1, 2, 'xx') not in dc
        assert 'xx' in dc.pols()
        assert (1, 2) in dc.antpairs()
        del dc[(1, 2, 'yy')]
        assert (1, 2) not in dc.antpairs()
        del dc[(2, 3, 'XX')]
        assert (2, 3, 'xx') not in dc
        assert 'xx' in dc.pols()
        assert (2, 3) in dc.antpairs()

        dc = datacontainer.DataContainer(self.blpol)
        del dc[[(1, 2, 'xx'), (1, 2, 'yy')]]
        assert (1, 2, 'xx') not in dc
        assert (1, 2, 'yy') not in dc
        assert (1, 2) not in dc.antpairs()
        assert 'xx' in dc.pols()
        assert 'yy' in dc.pols()

    def test_getitem(self):
        dc = datacontainer.DataContainer(self.blpol)
        assert dc[(1, 2, 'xx')] == 1j
        assert dc[(2, 1, 'xx')] == -1j
        assert dc[(1, 2)] == {'xx': 1j, 'yy': 1j}
        assert set(dc['xx'].keys()) == set(self.antpairs)
        assert dc[(1, 2, 'xx')] == dc.get_data((1, 2, 'xx'))
        assert dc[(1, 2, 'xx')] == dc.get_data(1, 2, 'xx')
        dc = datacontainer.DataContainer(self.polbl)
        assert dc[(1, 2, 'xx')] == 1j
        assert dc[(2, 1, 'xx')] == -1j
        assert dc[(1, 2)] == {'xx': 1j, 'yy': 1j}
        assert set(dc['xx'].keys()) == set(self.antpairs)
        assert dc[(2, 1, 'xx')] == dc.get_data((2, 1, 'xx'))
        assert dc[(2, 1, 'xx')] == dc.get_data(2, 1, 'xx')
        dc = datacontainer.DataContainer(self.both)
        assert dc[(1, 2, 'xx')] == 1j
        assert dc[(2, 1, 'xx')] == -1j
        assert dc[(1, 2)] == {'xx': 1j, 'yy': 1j}
        assert set(dc['xx'].keys()) == set(self.antpairs)
        assert dc[(1, 2)] == dc.get_data((1, 2))
        assert dc[(1, 2)] == dc.get_data(1, 2)
        assert dc[(1, 2, 'XX')] == 1j
        assert dc[(2, 1, 'XX')] == -1j
        assert dc[(2, 1, 'XX')] == dc.get_data(2, 1, 'XX')
        assert dc[(2, 1, 'XX')] == dc.get_data(2, 1, 'xx')
        dc = datacontainer.DataContainer(self.bools)
        assert dc[(1, 2, 'xx')] == np.array([True])
        assert dc[(2, 1, 'xx')] == np.array([True])
        assert dc[(2, 1, 'xx')].dtype == bool
        with pytest.raises(KeyError, match=r".*(10, 1, 'xx').*(1, 10, 'xx).*"):
            dc[(10, 1, 'xx')]

    def test_has_key(self):
        dc = datacontainer.DataContainer(self.blpol)
        assert (2, 3, 'yy') in dc
        assert dc.has_key((2, 3), 'yy')
        assert dc.has_key((3, 2), 'yy')
        assert 'xy' not in dc
        assert (5, 6) not in dc
        assert (1, 2, 'xy') not in dc
        dc = datacontainer.DataContainer(self.polbl)
        assert (2, 3, 'yy') in dc
        assert dc.has_key((2, 3), 'yy')
        assert dc.has_key((3, 2), 'yy')
        assert 'xy' not in dc
        assert (5, 6) not in dc
        assert (1, 2, 'xy') not in dc
        dc = datacontainer.DataContainer(self.both)
        assert (2, 3, 'yy') in dc
        assert dc.has_key((2, 3), 'yy')
        assert dc.has_key((3, 2), 'yy')
        assert 'xy' not in dc
        assert (5, 6) not in dc
        assert (1, 2, 'xy') not in dc

        assert (2, 3, 'YY') in dc
        assert dc.has_key((2, 3), 'YY')
        assert dc.has_key((3, 2), 'YY')
        assert 'XY' not in dc
        assert (5, 6) not in dc
        assert (1, 2, 'XY') not in dc

        # assert switch bl
        dc[(1, 2, 'xy')] = 1j
        assert (2, 1, 'yx') in dc

    def test_has_antpair(self):
        dc = datacontainer.DataContainer(self.blpol)
        assert dc.has_antpair((2, 3))
        assert dc.has_antpair((3, 2))
        assert not dc.has_antpair((0, 3))
        dc = datacontainer.DataContainer(self.polbl)
        assert dc.has_antpair((2, 3))
        assert dc.has_antpair((3, 2))
        assert not dc.has_antpair((0, 3))
        dc = datacontainer.DataContainer(self.both)
        assert dc.has_antpair((2, 3))
        assert dc.has_antpair((3, 2))
        assert not dc.has_antpair((0, 3))

    def test_has_pol(self):
        dc = datacontainer.DataContainer(self.blpol)
        assert dc.has_pol('xx')
        assert dc.has_pol('XX')
        assert not dc.has_pol('xy')
        assert not dc.has_pol('XY')
        dc = datacontainer.DataContainer(self.polbl)
        assert dc.has_pol('xx')
        assert dc.has_pol('XX')
        assert not dc.has_pol('xy')
        assert not dc.has_pol('XY')
        dc = datacontainer.DataContainer(self.both)
        assert dc.has_pol('xx')
        assert dc.has_pol('XX')
        assert not dc.has_pol('xy')
        assert not dc.has_pol('XY')

    def test_get(self):
        dc = datacontainer.DataContainer(self.blpol)
        assert dc.get((1, 2, 'yy')) == 1j
        assert dc.get((2, 1, 'yy'), 10) == -1j
        assert dc.get((1, 200, 'yy'), 10) == 10

    def test_setter(self):
        dc = datacontainer.DataContainer(self.blpol)
        # test basic setting
        dc[(100, 101, 'xy')] = np.arange(100) + np.arange(100) * 1j
        assert dc[(100, 101, 'xy')].shape == (100,)
        assert dc[(100, 101, 'xy')].dtype == complex
        assert np.allclose(dc[(100, 101, 'xy')][1], (1 + 1j))
        assert np.allclose(dc[(101, 100, 'yx')][1], (1 - 1j))
        assert len(dc.keys()) == 11
        assert (100, 101) in dc._antpairs
        assert 'xy' in dc._pols
        # test error
        pytest.raises(ValueError, dc.__setitem__, *((100, 101), 100j))

        dc = datacontainer.DataContainer(self.bools)
        dc[2, 1, 'xx'] = np.array([True])
        assert dc[(1, 2, 'xx')] == np.array([True])
        assert dc[(2, 1, 'xx')] == np.array([True])
        assert dc[(2, 1, 'xx')].dtype == bool


@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
class TestDataContainerWithRealData(object):

    def test_adder(self):
        test_file = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        d, f = io.load_vis(test_file, pop_autos=True)
        d2 = d + d
        assert np.allclose(d2[(24, 25, 'ee')][30, 30], d[(24, 25, 'ee')][30, 30] * 2)
        # test exception
        d2, f2 = io.load_vis(test_file, pop_autos=True)
        d2[list(d2.keys())[0]] = d2[list(d2.keys())[0]][:, :10]
        pytest.raises(ValueError, d.__add__, d2)
        d2[list(d2.keys())[0]] = d2[list(d2.keys())[0]][:10, :]
        pytest.raises(ValueError, d.__add__, d2)
        d2 = d + 1
        assert np.isclose(d2[(24, 25, 'ee')][30, 30], d[(24, 25, 'ee')][30, 30] + 1)

    def test_sub(self):
        test_file = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        d, f = io.load_vis(test_file, pop_autos=True)
        d2 = d - d
        assert np.allclose(d2[(24, 25, 'ee')][30, 30], 0.0)
        # test exception
        d2, f2 = io.load_vis(test_file, pop_autos=True)
        d2[list(d2.keys())[0]] = d2[list(d2.keys())[0]][:, :10]
        pytest.raises(ValueError, d.__sub__, d2)
        d2[list(d2.keys())[0]] = d2[list(d2.keys())[0]][:10, :]
        pytest.raises(ValueError, d.__sub__, d2)
        d2 = d - 1
        assert np.isclose(d2[(24, 25, 'ee')][30, 30], d[(24, 25, 'ee')][30, 30] - 1)

    def test_mul(self):
        test_file = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        d, f = io.load_vis(test_file, pop_autos=True)
        f[(24, 25, 'ee')][:, 0] = False
        f2 = f * f
        assert not np.any(f2[(24, 25, 'ee')][0, 0])
        # test exception
        d2, f2 = io.load_vis(test_file, pop_autos=True)
        d2[list(d2.keys())[0]] = d2[list(d2.keys())[0]][:, :10]
        pytest.raises(ValueError, d.__mul__, d2)
        d2[list(d2.keys())[0]] = d2[list(d2.keys())[0]][:10, :]
        pytest.raises(ValueError, d.__mul__, d2)
        d2 = d * 2
        assert np.isclose(d2[(24, 25, 'ee')][30, 30], d[(24, 25, 'ee')][30, 30] * 2)

    @pytest.mark.filterwarnings("ignore:invalid value encountered in true_divide")
    @pytest.mark.filterwarnings("ignore:invalid value encountered in floor_divide")
    def test_div(self):
        test_file = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        d, f = io.load_vis(test_file, pop_autos=True)
        d2 = d / d
        assert np.allclose(d2[(24, 25, 'ee')][30, 30], 1.0)
        d2 = d / 2.0
        assert np.allclose(d2[(24, 25, 'ee')][30, 30], d[(24, 25, 'ee')][30, 30] / 2.0)
        d2 = d // d
        assert np.allclose(d2[(24, 25, 'ee')][30, 30], 1.0)
        d2 = d // 2.0
        assert np.allclose(d2[(24, 25, 'ee')][30, 30], d[(24, 25, 'ee')][30, 30].real // 2.0 + d[(24, 25, 'ee')][30, 30].imag // 2.0)
        # now convert d to floats and do the same thing.
        for k in d:
            d[k] = d[k].real
        d2 = d // d
        assert np.allclose(d2[(24, 25, 'ee')][30, 30], 1.0)
        d2 = d // 2.0
        assert np.allclose(d2[(24, 25, 'ee')][30, 30], d[(24, 25, 'ee')][30, 30] // 2.0)

    def test_invert(self):
        test_file = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        d, f = io.load_vis(test_file, pop_autos=True)
        f2 = ~f
        bl = (24, 25, 'ee')
        assert f2[(bl)][0, 0] == ~f[bl][0, 0]
        pytest.raises(TypeError, d.__invert__)

    def test_transpose(self):
        test_file = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        d, f = io.load_vis(test_file, pop_autos=True)
        d2 = d.T
        bl = (24, 25, 'ee')
        assert d2[(bl)][0, 0] == d[bl].T[0, 0]

    def test_neg(self):
        test_file = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        d, f = io.load_vis(test_file, pop_autos=True)
        d2 = -d
        bl = (24, 25, 'ee')
        assert d2[(bl)][0, 0] == -d[(bl)][0, 0]

    def test_concatenate(self):
        test_file = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        d, f = io.load_vis(test_file, pop_autos=True)
        d2 = d.concatenate(d)
        assert d2[(24, 25, 'ee')].shape[0] == d[(24, 25, 'ee')].shape[0] * 2
        d2 = d.concatenate(d, axis=1)
        assert d2[(24, 25, 'ee')].shape[1] == d[(24, 25, 'ee')].shape[1] * 2
        d2 = d.concatenate([d, d], axis=0)
        assert d2[(24, 25, 'ee')].shape[0] == d[(24, 25, 'ee')].shape[0] * 3
        # test exceptions
        d2, f2 = io.load_vis(test_file, pop_autos=True)
        d2[list(d2.keys())[0]] = d2[list(d2.keys())[0]][:10, :10]
        pytest.raises(ValueError, d.concatenate, d2, axis=0)
        pytest.raises(ValueError, d.concatenate, d2, axis=1)

    def test_select_or_expand_times(self):
        # try cases that are selections, out of order, or have duplicate entries
        for new_times in [[0], [0, 1, 2], [2, 7, 4], [2, 9, 2, 2], [100]]:

            dc1 = datacontainer.DataContainer({(0, 1, 'ee'): np.arange(10)})
            dc1.times = np.arange(10)
            dc1.times_by_bl = {(0, 1): np.arange(10)}
            dc1.lsts = np.arange(10) * 2 * np.pi / 10
            dc1.lsts_by_bl = {(0, 1): np.arange(10) * 2 * np.pi / 10}

            if new_times == [100]:
                with pytest.raises(ValueError):
                    dc1.select_or_expand_times(new_times)
                continue
            else:
                dc2 = dc1.select_or_expand_times(new_times, in_place=False)
                dc1.select_or_expand_times(new_times, in_place=True)

            for dc in (dc1, dc2):
                assert np.all(dc[(0, 1, 'ee')] == np.arange(10)[new_times])
                assert np.all(dc.times == new_times)
                assert np.all(dc.times_by_bl[0, 1] == new_times)
                assert np.all(dc.lsts == (np.arange(10) * 2 * np.pi / 10)[new_times])
                assert np.all(dc.lsts_by_bl[0, 1] == (np.arange(10) * 2 * np.pi / 10)[new_times])


def test_RedDataContainer():
    # build hexagonal array and data which tells us which baseline it comes from
    antpos = hex_array(4, split_core=False, outriggers=0)
    reds = redcal.get_reds(antpos, pols=['nn', 'ee', 'ne', 'en'])
    data = datacontainer.DataContainer({red[0]: np.ones((10, 10)) * (red[0][0] + 1.0j * red[0][1]) for red in reds})

    # try the several ways of constructing RedDataContainer
    rdata1 = datacontainer.RedDataContainer(deepcopy(data), reds)
    rdata2 = datacontainer.RedDataContainer(deepcopy(data), antpos=antpos)
    data.antpos = antpos
    rdata3 = datacontainer.RedDataContainer(deepcopy(data))
    reverse_reds = [[reverse_bl(bl) for bl in red] for red in reds]
    reverse_data = datacontainer.DataContainer({red[0]: np.ones((10, 10)) * (red[0][0] + 1.0j * red[0][1]) for red in reverse_reds})
    rdata4 = datacontainer.RedDataContainer(deepcopy(reverse_data), reverse_reds)
    rdata5 = datacontainer.RedDataContainer(deepcopy(reverse_data), reds)
    # build an incomplete datacontainer, then finish it
    rdata6 = datacontainer.RedDataContainer(deepcopy(data), reds[:-1])
    rdata6[reds[-1][0]] = deepcopy(data[reds[-1][0]])
    rdata6.build_red_keys(reds)

    # make sure that the data for a redundant group are being accessed from the same place in memory
    for i, rdata in enumerate([rdata1, rdata2, rdata3, rdata4, rdata5, rdata6]):
        for red in rdata.reds:
            for bl in red:
                if rdata != rdata5:
                    assert id(rdata[bl]) == id(rdata[red[0]])
                else:
                    assert id(rdata[reverse_bl(bl)]) == id(rdata[reverse_bl(red[0])])
                assert(rdata.get_ubl_key(bl) in red)
                assert(rdata.has_key(rdata.get_ubl_key(bl)))
                assert set(rdata.get_red(bl)) == set(red)
                assert bl in rdata
            val_here = deepcopy(rdata[red[0]])
            rdata[red[0]] *= 2
            for bl in red:
                np.testing.assert_array_equal(val_here * 2, rdata[bl])

    # test no redundancy information error
    with pytest.raises(ValueError):
        data = datacontainer.DataContainer({red[0]: np.ones((10, 10)) * (red[0][0] + 1.0j * red[0][1]) for red in reds})
        rdata = datacontainer.RedDataContainer(data)

    # test too much redundancy information error
    with pytest.raises(ValueError):
        data = datacontainer.DataContainer({red[0]: np.ones((10, 10)) * (red[0][0] + 1.0j * red[0][1]) for red in reds})
        rdata = datacontainer.RedDataContainer(data, reds=reds, antpos=antpos)

    # test error where the the data has multiple redundant baselines per group
    with pytest.raises(ValueError):
        data = datacontainer.DataContainer({bl: np.ones((10, 10)) * (bl[0] + 1.0j * bl[1]) for red in reds for bl in red})
        rdata = datacontainer.RedDataContainer(data, reds)


def test_RedDataContainerKeyManipulation():
    rdc = datacontainer.RedDataContainer({(0, 1, 'ee'): 10}, reds=[[(0, 1, 'ee'), (1, 2, 'ee')], [(2, 3, 'ee'), (3, 4, 'ee')]])

    # test contains
    assert (0, 1, 'ee') in rdc
    assert (1, 2, 'ee') in rdc
    assert (2, 1, 'ee') in rdc
    assert (2, 3, 'ee') not in rdc

    # basic operation test
    rdc[1, 2, 'ee'] = 11j
    assert rdc[0, 1, 'ee'] == 11j

    # test that uninstantianted red has no data
    with pytest.raises(KeyError):
        rdc[2, 3, 'ee']

    # test that uninstantiated red can still be used late
    rdc[4, 3, 'ee'] = 12j
    assert rdc[2, 3, 'ee'] == -12j

    # test that rekeying eliminates keys
    assert (2, 3, 'ee') in rdc._data or (3, 2, 'ee') in rdc._data
    rdc.build_red_keys([[(0, 1, 'ee'), (1, 2, 'ee')]])
    assert (2, 3, 'ee') not in rdc._data
    assert (3, 2, 'ee') not in rdc._data
    with pytest.raises(KeyError):
        rdc[3, 4, 'ee']

    # test collision after adding keys
    rdc[0, 2, 'ee'] = 20j
    rdc[1, 3, 'ee'] = 21j
    with pytest.raises(ValueError):
        rdc.build_red_keys([[(0, 2, 'ee'), (1, 3, 'ee')]])

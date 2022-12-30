# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

'''Tests for io.py'''

import pytest
import numpy as np
import os
import warnings
import shutil
import copy
from collections import OrderedDict as odict
import pyuvdata
from pyuvdata import UVCal, UVData, UVFlag
from pyuvdata.utils import parse_polstr, parse_jpolstr
import glob
import sys

from .. import io
from ..io import HERACal, HERAData
from ..datacontainer import DataContainer
from ..utils import polnum2str, polstr2num, jnum2str, jstr2num, reverse_bl, split_bl
from ..data import DATA_PATH
from hera_qm.data import DATA_PATH as QM_DATA_PATH


class Test_HERACal(object):
    def setup_method(self):
        self.fname_xx = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.xx.HH.uvc.omni.calfits")
        self.fname_yy = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.yy.HH.uvc.omni.calfits")
        self.fname_2pol = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.HH.omni.calfits")
        self.fname = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.HH.uvcA.omni.calfits")
        self.fname_t0 = os.path.join(DATA_PATH, 'test_input/zen.2458101.44615.xx.HH.uv.abs.calfits_54x_only')
        self.fname_t1 = os.path.join(DATA_PATH, 'test_input/zen.2458101.45361.xx.HH.uv.abs.calfits_54x_only')
        self.fname_t2 = os.path.join(DATA_PATH, 'test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only')

    def test_init(self):
        hc = HERACal(self.fname_xx)
        assert hc.filepaths == [self.fname_xx]
        hc = HERACal([self.fname_xx, self.fname_yy])
        assert hc.filepaths == [self.fname_xx, self.fname_yy]
        hc = HERACal((self.fname_xx, self.fname_yy))
        assert hc.filepaths == [self.fname_xx, self.fname_yy]
        with pytest.raises(TypeError):
            hc = HERACal([0, 1])
        with pytest.raises(ValueError):
            hc = HERACal(None)

    def test_read(self):
        # test one file with both polarizations and a non-None total quality array
        hc = HERACal(self.fname)
        gains, flags, quals, total_qual = hc.read()
        uvc = UVCal()
        uvc.read_calfits(self.fname)
        uvc.use_future_array_shapes()
        np.testing.assert_array_equal(uvc.gain_array[0, :, :, 0].T, gains[9, parse_jpolstr('jxx', x_orientation=hc.x_orientation)])
        np.testing.assert_array_equal(uvc.flag_array[0, :, :, 0].T, flags[9, parse_jpolstr('jxx', x_orientation=hc.x_orientation)])
        np.testing.assert_array_equal(uvc.quality_array[0, :, :, 0].T, quals[9, parse_jpolstr('jxx', x_orientation=hc.x_orientation)])
        np.testing.assert_array_equal(uvc.total_quality_array[:, :, 0].T, total_qual[parse_jpolstr('jxx', x_orientation=hc.x_orientation)])
        np.testing.assert_array_equal(np.unique(uvc.freq_array), hc.freqs)
        np.testing.assert_array_equal(np.unique(uvc.time_array), hc.times)
        assert hc.pols == [parse_jpolstr('jxx', x_orientation=hc.x_orientation), parse_jpolstr('jyy', x_orientation=hc.x_orientation)]
        assert set([ant[0] for ant in hc.ants]) == set(uvc.ant_array)

        # test list loading
        hc = HERACal([self.fname_xx, self.fname_yy])
        gains, flags, quals, total_qual = hc.read()
        assert len(gains.keys()) == 36
        assert len(flags.keys()) == 36
        assert len(quals.keys()) == 36
        assert hc.freqs.shape == (1024,)
        assert hc.times.shape == (3,)
        assert sorted(hc.pols) == [parse_jpolstr('jxx', x_orientation=hc.x_orientation), parse_jpolstr('jyy', x_orientation=hc.x_orientation)]

    def test_read_select(self):
        # test read multiple files and select times
        hc = io.HERACal([self.fname_t0, self.fname_t1, self.fname_t2])
        g, _, _, _ = hc.read()
        g2, _, _, _ = hc.read(times=hc.times[30:90])
        np.testing.assert_array_equal(g2[54, 'Jee'], g[54, 'Jee'][30:90, :])

        # test read multiple files and select freqs/chans
        hc = io.HERACal([self.fname_t0, self.fname_t1, self.fname_t2])
        g, _, _, _ = hc.read()
        g2, _, _, _ = hc.read(frequencies=hc.freqs[0:100])
        g3, _, _, _ = hc.read(freq_chans=np.arange(100))
        np.testing.assert_array_equal(g2[54, 'Jee'], g[54, 'Jee'][:, 0:100])
        np.testing.assert_array_equal(g3[54, 'Jee'], g[54, 'Jee'][:, 0:100])

        # test select on antenna numbers
        hc = io.HERACal([self.fname_xx, self.fname_yy])
        g, _, _, _ = hc.read(antenna_nums=[9, 10])
        hc2 = io.HERACal(self.fname_2pol)
        g2, _, _, _ = hc2.read(antenna_nums=[9, 10])
        for k in g2:
            assert k[0] in [9, 10]
            np.testing.assert_array_equal(g[k], g2[k])

        # test select on pols
        hc = io.HERACal(self.fname_xx)
        g, _, _, _ = hc.read()
        hc2 = io.HERACal(self.fname_2pol)
        g2, _, _, _ = hc2.read(pols=['Jee'])
        for k in g2:
            np.testing.assert_array_equal(g[k], g2[k])

    def test_update(self):
        hc = io.HERACal(self.fname)
        g, f, q, tq = hc.read()
        for ant in hc.ants:
            g[ant] *= (2.0 + 1.0j)
            f[ant] = np.logical_not(f[ant])
            q[ant] += 1
        for pol in hc.pols:
            tq[pol] += 2
        hc.update(gains=g, flags=f, quals=q, total_qual=tq)
        g2, f2, q2, tq2 = hc.build_calcontainers()
        for ant in hc.ants:
            np.testing.assert_array_almost_equal(g[ant], g2[ant])
            np.testing.assert_array_equal(f[ant], f2[ant])
            np.testing.assert_array_equal(q[ant], q2[ant])
        for pol in hc.pols:
            np.testing.assert_array_equal(tq[pol], tq2[pol])

        # test with slicing
        hc = io.HERACal(self.fname)
        g, f, q, tq = hc.read()
        g0, f0, q0, tq0 = hc.read()
        is_updated = np.zeros((hc.Ntimes, hc.Nfreqs), dtype=bool)
        is_updated[slice(0, 11), slice(500, 600)] = True
        for ant in hc.ants:
            g[ant] = g[ant][slice(0, 1), slice(500, 600)]
            f[ant] = f[ant][slice(0, 1), slice(500, 600)]
            q[ant] = q[ant][slice(0, 1), slice(500, 600)]
            g[ant] *= (2.0 + 1.0j)
            f[ant] = np.logical_not(f[ant])
            q[ant] += 1
        for pol in hc.pols:
            tq[pol] = tq[pol][slice(0, 1), slice(500, 600)]
            tq[pol] += 2

        hc.update(gains=g, flags=f, quals=q, total_qual=tq, tSlice=slice(0, 1), fSlice=slice(500, 600))
        g2, f2, q2, tq2 = hc.build_calcontainers()
        for ant in hc.ants:
            np.testing.assert_array_almost_equal(g[ant].flatten(), g2[ant][is_updated])
            np.testing.assert_array_equal(f[ant].flatten(), f2[ant][is_updated])
            np.testing.assert_array_equal(q[ant].flatten(), q2[ant][is_updated])
            np.testing.assert_array_almost_equal(g0[ant][~is_updated], g2[ant][~is_updated])
            np.testing.assert_array_equal(f0[ant][~is_updated], f2[ant][~is_updated])
            np.testing.assert_array_equal(q0[ant][~is_updated], q2[ant][~is_updated])
        for pol in hc.pols:
            np.testing.assert_array_equal(tq[pol].flatten(), tq2[pol][is_updated])
            np.testing.assert_array_equal(tq0[pol][~is_updated], tq2[pol][~is_updated])

    def test_write(self):
        hc = HERACal(self.fname)
        gains, flags, quals, total_qual = hc.read()
        for key in gains.keys():
            gains[key] *= 2.0 + 1.0j
            flags[key] = np.logical_not(flags[key])
            quals[key] *= 2.0
        for key in total_qual.keys():
            total_qual[key] *= 2
        # remove total quality array to test handling absence.
        hc.total_quality_array = None
        hc.update(gains=gains, flags=flags, quals=quals, total_qual=total_qual)
        hc.write_calfits('test.calfits', clobber=True)

        gains_in, flags_in, quals_in, total_qual_in = hc.read()
        hc2 = HERACal('test.calfits')
        gains_out, flags_out, quals_out, total_qual_out = hc2.read()
        for key in gains_in.keys():
            np.testing.assert_array_equal(gains_in[key] * (2.0 + 1.0j), gains_out[key])
            np.testing.assert_array_equal(np.logical_not(flags_in[key]), flags_out[key])
            np.testing.assert_array_equal(quals_in[key] * (2.0), quals_out[key])
        for key in total_qual.keys():
            np.testing.assert_array_equal(total_qual_in[key] * (2.0), total_qual_out[key])

        os.remove('test.calfits')


@pytest.mark.filterwarnings("ignore:It seems that the latitude and longitude are in radians")
@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
@pytest.mark.filterwarnings("ignore:Mean of empty slice")
@pytest.mark.filterwarnings("ignore:invalid value encountered in double_scalars")
class Test_HERAData(object):
    def setup_method(self):
        self.uvh5_1 = os.path.join(DATA_PATH, "zen.2458116.61019.xx.HH.XRS_downselected.uvh5")
        self.uvh5_2 = os.path.join(DATA_PATH, "zen.2458116.61765.xx.HH.XRS_downselected.uvh5")
        self.uvh5_bda = os.path.join(DATA_PATH, "zen.2459122.30030.sum.bda.downsampled.uvh5")
        self.miriad_1 = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        self.miriad_2 = os.path.join(DATA_PATH, "zen.2458043.13298.xx.HH.uvORA")
        self.uvfits = os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvA.vis.uvfits')
        self.four_pol = [os.path.join(DATA_PATH, 'zen.2457698.40355.{}.HH.uvcA'.format(pol))
                         for pol in ['xx', 'yy', 'xy', 'yx']]

    def test_init(self):
        # single uvh5 file
        hd = HERAData(self.uvh5_1)
        assert hd.filepaths == [self.uvh5_1]
        assert not hd.upsample
        assert not hd.downsample
        for meta in hd.HERAData_metas:
            assert getattr(hd, meta) is not None
        assert len(hd.freqs) == 1024
        assert len(hd.bls) == 3
        assert len(hd.times) == 60
        assert len(hd.lsts) == 60
        assert hd._writers == {}

        # multiple uvh5 files
        files = [self.uvh5_1, self.uvh5_2]
        hd = HERAData(files)
        individual_hds = {files[0]: HERAData(files[0]), files[1]: HERAData(files[1])}
        assert hd.filepaths == files
        for meta in hd.HERAData_metas:
            assert getattr(hd, meta) is not None
        for f in files:
            assert len(hd.freqs[f]) == 1024
            np.testing.assert_array_equal(hd.freqs[f], individual_hds[f].freqs)
            assert len(hd.bls[f]) == 3
            np.testing.assert_array_equal(hd.bls[f], individual_hds[f].bls)
            assert len(hd.times[f]) == 60
            np.testing.assert_array_equal(hd.times[f], individual_hds[f].times)
            assert len(hd.lsts[f]) == 60
            np.testing.assert_array_equal(hd.lsts[f], individual_hds[f].lsts)
        assert not hasattr(hd, '_writers')

        # miriad
        hd = HERAData(self.miriad_1, filetype='miriad')
        assert hd.filepaths == [self.miriad_1]
        for meta in hd.HERAData_metas:
            assert getattr(hd, meta) is None

        # uvfits
        hd = HERAData(self.uvfits, filetype='uvfits')
        assert hd.filepaths == [self.uvfits]
        for meta in hd.HERAData_metas:
            assert getattr(hd, meta) is not None

        # bda upsample/downsample
        hd = HERAData(self.uvh5_bda, upsample=True)
        assert hd.upsample
        assert not hd.downsample
        assert hd.shortest_integration == pytest.approx(9.66367642)
        hd = HERAData(self.uvh5_bda, downsample=True)
        assert not hd.upsample
        assert hd.downsample
        assert hd.longest_integration == pytest.approx(77.30941133)

        # test errors
        with pytest.raises(TypeError):
            hd = HERAData([1, 2])
        with pytest.raises(ValueError):
            hd = HERAData(None)
        with pytest.raises(NotImplementedError):
            hd = HERAData(self.uvh5_1, filetype='not a real type')
        with pytest.raises(IOError):
            hd = HERAData('fake path')
        with pytest.raises(ValueError):
            hd = HERAData(self.uvh5_bda, upsample=True, downsample=True)

    def test_add(self):
        hd = HERAData(self.uvh5_1)
        hd.read()

        # test add
        hd2 = copy.deepcopy(hd)
        hd2.polarization_array[0] = -6
        hd3 = hd + hd2
        assert len(hd3._polnum_indices) == 2

    def test_select(self):
        hd = HERAData(self.uvh5_1)
        hd.read()
        hd2 = copy.deepcopy(hd)
        hd2.polarization_array[0] = -6
        hd += hd2

        # blt select
        d1 = hd.get_data(53, 54, 'xx')
        hd.select(bls=[(53, 54)])
        assert len(hd._blt_slices) == 1
        d2 = hd.get_data(53, 54, 'xx')
        np.testing.assert_array_almost_equal(d1, d2)
        hd.select(times=np.unique(hd.time_array)[-5:])
        d3 = hd.get_data(53, 54, 'xx')
        np.testing.assert_array_almost_equal(d2[-5:], d3)

        # pol select
        hd.select(polarizations=['yy'])
        assert len(hd._polnum_indices) == 1

    def test_reset(self):
        hd = HERAData(self.uvh5_1)
        hd.read()
        hd.reset()
        assert hd.data_array is None
        assert hd.flag_array is None
        assert hd.nsample_array is None
        assert hd.filepaths == [self.uvh5_1]
        for meta in hd.HERAData_metas:
            assert getattr(hd, meta) is not None
        assert len(hd.freqs) == 1024
        assert len(hd.bls) == 3
        assert len(hd.times) == 60
        assert len(hd.lsts) == 60
        assert hd._writers == {}

    def test_get_metadata_dict(self):
        hd = HERAData(self.uvh5_1)
        metas = hd.get_metadata_dict()
        for meta in hd.HERAData_metas:
            assert meta in metas
        assert len(metas['freqs']) == 1024
        assert len(metas['bls']) == 3
        assert len(metas['times']) == 60
        assert len(metas['lsts']) == 60
        np.testing.assert_array_equal(metas['times'], np.unique(list(metas['times_by_bl'].values())))
        np.testing.assert_array_equal(metas['lsts'], np.unique(list(metas['lsts_by_bl'].values())))

    def test_determine_blt_slicing(self):
        hd = HERAData(self.uvh5_1)
        for s in hd._blt_slices.values():
            assert isinstance(s, slice)
        for bl, s in hd._blt_slices.items():
            np.testing.assert_array_equal(np.arange(180)[np.logical_and(hd.ant_1_array == bl[0],
                                                                        hd.ant_2_array == bl[1])], np.arange(180)[s])
        # test check for non-regular spacing
        with pytest.raises(NotImplementedError):
            hd.select(blt_inds=[0, 1, 3, 5, 23, 48])
            hd._determine_blt_slicing()

    def test_determine_pol_indexing(self):
        hd = HERAData(self.uvh5_1)
        assert hd._polnum_indices == {-5: 0}
        hd = HERAData(self.four_pol, filetype='miriad')
        hd.read(bls=[(53, 53)], axis='polarization')
        assert hd._polnum_indices == {-8: 3, -7: 2, -6: 1, -5: 0}

    def test_get_slice(self):
        hd = HERAData(self.uvh5_1)
        hd.read()
        for bl in hd.bls:
            np.testing.assert_array_equal(hd._get_slice(hd.data_array, bl), hd.get_data(bl))
        np.testing.assert_array_equal(hd._get_slice(hd.data_array, (54, 53, 'EE')),
                                      hd.get_data((54, 53, 'EE')))
        np.testing.assert_array_equal(hd._get_slice(hd.data_array, (53, 54))[parse_polstr('XX', x_orientation=hd.x_orientation)],
                                      hd.get_data((53, 54, 'EE')))
        np.testing.assert_array_equal(hd._get_slice(hd.data_array, 'EE')[(53, 54)],
                                      hd.get_data((53, 54, 'EE')))
        with pytest.raises(KeyError):
            hd._get_slice(hd.data_array, (1, 2, 3, 4))

        hd = HERAData(self.four_pol, filetype='miriad')
        d, f, n = hd.read(bls=[(80, 81)])
        for p in d.pols():
            np.testing.assert_array_almost_equal(hd._get_slice(hd.data_array, (80, 81, p)),
                                                 hd.get_data((80, 81, p)))
            np.testing.assert_array_almost_equal(hd._get_slice(hd.data_array, (81, 80, p)),
                                                 hd.get_data((81, 80, p)))

    def test_set_slice(self):
        hd = HERAData(self.uvh5_1)
        hd.read()
        np.random.seed(21)

        for bl in hd.bls:
            new_vis = np.random.randn(60, 1024) + np.random.randn(60, 1024) * 1.0j
            hd._set_slice(hd.data_array, bl, new_vis)
            np.testing.assert_array_almost_equal(new_vis, hd.get_data(bl))

        new_vis = np.random.randn(60, 1024) + np.random.randn(60, 1024) * 1.0j
        hd._set_slice(hd.data_array, (54, 53, 'xx'), new_vis)
        np.testing.assert_array_almost_equal(np.conj(new_vis), hd.get_data((53, 54, 'xx')))

        new_vis = np.random.randn(60, 1024) + np.random.randn(60, 1024) * 1.0j
        hd._set_slice(hd.data_array, (53, 54), {'xx': new_vis})
        np.testing.assert_array_almost_equal(new_vis, hd.get_data((53, 54, 'xx')))

        new_vis = np.random.randn(60, 1024) + np.random.randn(60, 1024) * 1.0j
        to_set = {(53, 54): new_vis, (54, 54): 2 * new_vis, (53, 53): 3 * new_vis}
        hd._set_slice(hd.data_array, 'XX', to_set)
        np.testing.assert_array_almost_equal(new_vis, hd.get_data((53, 54, 'xx')))

        with pytest.raises(KeyError):
            hd._set_slice(hd.data_array, (1, 2, 3, 4), None)

    def test_build_datacontainers(self):
        hd = HERAData(self.uvh5_1)
        d, f, n = hd.read()
        for bl in hd.bls:
            np.testing.assert_array_almost_equal(d[bl], hd.get_data(bl))
            np.testing.assert_array_almost_equal(f[bl], hd.get_flags(bl))
            np.testing.assert_array_almost_equal(n[bl], hd.get_nsamples(bl))
        for dc in [d, f, n]:
            assert isinstance(dc, DataContainer)
            for k in dc.antpos.keys():
                assert np.all(dc.antpos[k] == hd.antpos[k])
            assert len(dc.antpos) == 52
            assert len(hd.antpos) == 52
            for k in dc.data_antpos.keys():
                assert np.all(dc.data_antpos[k] == hd.data_antpos[k])
            assert len(dc.data_antpos) == 2
            assert len(hd.data_antpos) == 2
            assert np.all(dc.freqs == hd.freqs)
            assert np.all(dc.times == hd.times)
            assert np.all(dc.lsts == hd.lsts)
            for k in dc.times_by_bl.keys():
                assert np.all(dc.times_by_bl[k] == hd.times_by_bl[k])
                assert np.all(dc.times_by_bl[k] == dc.times_by_bl[(k[1], k[0])])
                assert np.all(dc.lsts_by_bl[k] == hd.lsts_by_bl[k])
                assert np.all(dc.lsts_by_bl[k] == dc.lsts_by_bl[(k[1], k[0])])

    def test_write_read_filter_cache_scratch(self):
        # most of write_filter_cache_scratch and all of read_filter_cache_scratch are covered in
        # test_delay_filter.test_load_dayenu_filter_and_write()
        # This test covers a few odds and ends that are not covered.
        # The case not covered is writing a filter cache with no skip_keys
        # or filter directory.
        io.write_filter_cache_scratch(filter_cache={'crazy': 'universe'})
        # make sure file (and only one file) was written.
        assert len(glob.glob(os.getcwd() + '/*.filter_cache')) == 1
        # make sure read works and read cache is identical to written cache.
        cache = io.read_filter_cache_scratch(os.getcwd())
        assert cache['crazy'] == 'universe'
        assert len(cache) == 1
        # now cleanup cache files.
        cleanup = glob.glob(os.getcwd() + '/*.filter_cache')
        for file in cleanup:
            os.remove(file)

    @pytest.mark.filterwarnings("ignore:miriad does not support partial loading")
    def test_read(self):
        # uvh5
        hd = HERAData(self.uvh5_1)
        d, f, n = hd.read(bls=(53, 54, 'xx'), frequencies=hd.freqs[0:100], times=hd.times[0:10])
        assert hd.last_read_kwargs['bls'] == (53, 54, parse_polstr('XX'))
        assert hd.last_read_kwargs['polarizations'] is None
        for dc in [d, f, n]:
            assert len(dc) == 1
            assert list(dc.keys()) == [(53, 54, parse_polstr('XX', x_orientation=hd.x_orientation))]
            assert dc[53, 54, 'ee'].shape == (10, 100)
        with pytest.raises(ValueError):
            d, f, n = hd.read(polarizations=['xy'])

        # assert return data = False
        o = hd.read(bls=[(53, 53), (53, 54)], return_data=False)
        assert o is None

        # assert __getitem__ isn't a read when key exists
        o = hd[(53, 53, 'ee')]
        assert len(hd._blt_slices) == 2

        # assert __getitem__ is a read when key does not exist
        o = hd[(54, 54, 'ee')]
        assert len(hd._blt_slices) == 1

        # test read with extra UVData kwargs
        hd = HERAData(self.uvh5_1)
        d, f, n = hd.read(bls=hd.bls[:2], frequencies=hd.freqs[:100], multidim_index=True)
        k = list(d.keys())[0]
        assert len(d) == 2
        assert d[k].shape == (hd.Ntimes, 100)

        # test read list
        hd = HERAData([self.uvh5_1, self.uvh5_2])
        d, f, n = hd.read(axis='blt')
        for dc in [d, f, n]:
            assert len(dc) == 3
            assert len(dc.times) == 120
            assert len(dc.lsts) == 120
            assert len(dc.freqs) == 1024
            for i in [53, 54]:
                for j in [53, 54]:
                    assert (i, j, 'ee') in dc
            for bl in dc:
                assert dc[bl].shape == (120, 1024)

        # miriad
        hd = HERAData(self.miriad_1, filetype='miriad')
        d, f, n = hd.read()
        hd = HERAData(self.miriad_1, filetype='miriad')
        d, f, n = hd.read(bls=(52, 53), polarizations=['XX'], frequencies=d.freqs[0:30], times=d.times[0:10])
        assert hd.last_read_kwargs['polarizations'] == ['XX']
        for dc in [d, f, n]:
            assert len(dc) == 1
            assert list(dc.keys()) == [(52, 53, parse_polstr('XX', x_orientation=hd.x_orientation))]
            assert dc[52, 53, 'ee'].shape == (10, 30)
        with pytest.raises(NotImplementedError):
            d, f, n = hd.read(read_data=False)

        # uvfits
        hd = HERAData(self.uvfits, filetype='uvfits')
        hd.read(read_data=False)
        ant_pairs = hd.get_antpairs()
        d, f, n = hd.read(bls=(ant_pairs[0][0], ant_pairs[0][1], 'xx'), freq_chans=list(range(10)))
        assert hd.last_read_kwargs['freq_chans'] == list(range(10))
        for dc in [d, f, n]:
            assert len(dc) == 1
            assert list(dc.keys()) == [
                (ant_pairs[0][0], ant_pairs[0][1], parse_polstr('XX', x_orientation=hd.x_orientation))
            ]
            assert dc[ant_pairs[0][0], ant_pairs[0][1], 'ee'].shape == (60, 10)

    def test_read_bda(self):
        # no upsampling or downsampling
        hd = HERAData(self.uvh5_bda)
        assert np.max(hd.integration_time) == pytest.approx(77.30941133)
        assert np.min(hd.integration_time) == pytest.approx(9.66367642)
        assert len(np.unique(hd.time_array)) > 8
        assert len(hd.times) == 8
        assert len(hd.times_by_bl[117, 118]) == 1
        assert len(hd.times_by_bl[104, 117]) == 8
        d, f, n = hd.read()
        assert len(d.times_by_bl[117, 118]) == 1
        assert len(d.times_by_bl[104, 117]) == 8

    def test_read_bda_upsample(self):
        # show that upsampling works properly even with partial i/o
        hd = HERAData(self.uvh5_bda, upsample=True)
        assert len(hd.times_by_bl[117, 118]) == 8
        d, f, n = hd.read(bls=[(117, 118, 'ee')])
        assert d[(117, 118, 'ee')].shape == (8, len(hd.freqs))
        np.testing.assert_array_almost_equal(hd.integration_time, 9.66367642)
        np.testing.assert_array_almost_equal(np.diff(d.times), 9.66367642 / 24 / 3600)
        assert len(d.times_by_bl[117, 118]) == 8
        np.testing.assert_array_almost_equal(np.diff(d.times_by_bl[117, 118]), 9.66367642 / 24 / 3600)

    def test_read_bda_downsample(self):
        # show that downsampling works properly even with partial i/o
        hd = HERAData(self.uvh5_bda, downsample=True)
        assert len(hd.times_by_bl[104, 117]) == 1
        d, f, n = hd.read(bls=[(104, 117, 'ee')])
        assert d[(104, 117, 'ee')].shape == (1, len(hd.freqs))
        np.testing.assert_array_almost_equal(hd.integration_time, 77.30941133)
        np.testing.assert_array_almost_equal(np.diff(d.times), 77.30941133 / 24 / 3600)
        assert len(d.times_by_bl[104, 117]) == 1
        np.testing.assert_array_almost_equal(np.diff(d.times_by_bl[104, 117]), 77.30941133 / 24 / 3600)

    def test_getitem(self):
        hd = HERAData(self.uvh5_1)
        hd.read()
        for bl in hd.bls:
            np.testing.assert_array_almost_equal(hd[bl], hd.get_data(bl))

    def test_update(self):
        hd = HERAData(self.uvh5_1)
        d, f, n = hd.read()
        for bl in hd.bls:
            d[bl] *= (2.0 + 1.0j)
            f[bl] = np.logical_not(f[bl])
            n[bl] += 1
        hd.update(data=d, flags=f, nsamples=n)
        d2, f2, n2 = hd.build_datacontainers()
        for bl in hd.bls:
            np.testing.assert_array_almost_equal(d[bl], d2[bl])
            np.testing.assert_array_equal(f[bl], f2[bl])
            np.testing.assert_array_equal(n[bl], n2[bl])

        # test with slicing
        hd = HERAData(self.uvh5_1)
        d0, f0, n0 = hd.read()
        d, f, n = hd.read()
        is_updated = np.zeros((hd.Ntimes, hd.Nfreqs), dtype=bool)
        is_updated[slice(3, 10), slice(500, 600)] = True
        for bl in hd.bls:
            d[bl] = d[bl][slice(3, 10), slice(500, 600)]
            f[bl] = f[bl][slice(3, 10), slice(500, 600)]
            n[bl] = n[bl][slice(3, 10), slice(500, 600)]
            d[bl] *= (2.0 + 1.0j)
            f[bl] = np.logical_not(f[bl])
            n[bl] += 1
        hd.update(data=d, flags=f, nsamples=n, tSlice=slice(3, 10), fSlice=slice(500, 600))
        d2, f2, n2 = hd.build_datacontainers()
        for bl in hd.bls:
            np.testing.assert_array_almost_equal(d[bl].flatten(), d2[bl][is_updated])
            np.testing.assert_array_equal(f[bl].flatten(), f2[bl][is_updated])
            np.testing.assert_array_equal(n[bl].flatten(), n2[bl][is_updated])
            np.testing.assert_array_almost_equal(d0[bl][~is_updated], d2[bl][~is_updated])
            np.testing.assert_array_equal(f0[bl][~is_updated], f2[bl][~is_updated])
            np.testing.assert_array_equal(n0[bl][~is_updated], n2[bl][~is_updated])

    def test_partial_write(self):
        hd = HERAData(self.uvh5_1)
        assert hd._writers == {}
        d, f, n = hd.read(bls=hd.bls[0])
        assert hd.last_read_kwargs['bls'] == (53, 53, parse_polstr('XX', x_orientation=hd.x_orientation))
        d[(53, 53, 'EE')] *= 2.0
        hd.partial_write('out.h5', data=d, clobber=True)
        assert 'out.h5' in hd._writers
        assert isinstance(hd._writers['out.h5'], HERAData)
        for meta in hd.HERAData_metas:
            try:
                assert np.all(getattr(hd, meta) == getattr(hd._writers['out.h5'], meta))
            except BaseException:
                for k in getattr(hd, meta).keys():
                    assert np.all(getattr(hd, meta)[k] == getattr(hd._writers['out.h5'], meta)[k])

        d, f, n = hd.read(bls=hd.bls[1])
        assert hd.last_read_kwargs['bls'] == (53, 54, parse_polstr('XX', x_orientation=hd.x_orientation))
        d[(53, 54, 'EE')] *= 2.0
        hd.partial_write('out.h5', data=d, clobber=True)

        d, f, n = hd.read(bls=hd.bls[2])
        assert hd.last_read_kwargs['bls'] == (54, 54, parse_polstr('XX', x_orientation=hd.x_orientation))
        d[(54, 54, 'EE')] *= 2.0
        hd.partial_write('out.h5', data=d, clobber=True, inplace=True)
        d_after, _, _ = hd.build_datacontainers()
        np.testing.assert_array_almost_equal(d[(54, 54, 'EE')], d_after[(54, 54, 'EE')])

        hd = HERAData(self.uvh5_1)
        d, f, n = hd.read()
        hd2 = HERAData('out.h5')
        d2, f2, n2 = hd2.read()
        for bl in hd.bls:
            np.testing.assert_array_almost_equal(d[bl] * 2.0, d2[bl])
            np.testing.assert_array_equal(f[bl], f2[bl])
            np.testing.assert_array_equal(n[bl], n2[bl])
        os.remove('out.h5')

        # test errors
        hd = HERAData(self.miriad_1, filetype='miriad')
        with pytest.raises(NotImplementedError):
            hd.partial_write('out.uv')
        hd = HERAData([self.uvh5_1, self.uvh5_2])
        with pytest.raises(NotImplementedError):
            hd.partial_write('out.h5')
        hd = HERAData(self.uvh5_1)

    def test_iterate_over_bls(self):
        hd = HERAData(self.uvh5_1)
        for (d, f, n) in hd.iterate_over_bls(Nbls=2):
            for dc in (d, f, n):
                assert len(dc.keys()) == 1 or len(dc.keys()) == 2
                assert list(dc.values())[0].shape == (60, 1024)

        hd = HERAData([self.uvh5_1, self.uvh5_2])
        for (d, f, n) in hd.iterate_over_bls():
            for dc in (d, f, n):
                assert len(d.keys()) == 1
                assert list(d.values())[0].shape == (120, 1024)

        # try cover include_autos = False.
        hd = HERAData([self.uvh5_1, self.uvh5_2])
        for (d, f, n) in hd.iterate_over_bls(include_autos=False):
            for dc in (d, f, n):
                assert len(d.keys()) == 1
                bl = list(d.keys())[0]
                # make sure no autos present.
                assert bl[0] != bl[1]
                assert list(d.values())[0].shape == (120, 1024)

        hd = HERAData(self.miriad_1, filetype='miriad')
        d, f, n = next(hd.iterate_over_bls(bls=[(52, 53, 'xx')]))
        assert list(d.keys()) == [(52, 53, parse_polstr('XX', x_orientation=hd.x_orientation))]
        with pytest.raises(NotImplementedError):
            next(hd.iterate_over_bls())

        hd = HERAData(self.uvh5_1)
        for (d, f, n) in hd.iterate_over_bls(chunk_by_redundant_group=True, Nbls=1):
            # check that all baselines in chunk are redundant
            # this will be the case when Nbls = 1
            bl_lens = np.asarray([hd.antpos[bl[0]] - hd.antpos[bl[1]] for bl in d])
            assert np.all(np.isclose(bl_lens - bl_lens[0], 0., atol=1.0))
            for dc in (d, f, n):
                assert list(d.values())[0].shape == (60, 1024)

        hd = HERAData([self.uvh5_1, self.uvh5_2])
        for (d, f, n) in hd.iterate_over_bls(chunk_by_redundant_group=True):
            for dc in (d, f, n):
                assert list(d.values())[0].shape == (120, 1024)

        with pytest.raises(NotImplementedError):
            hd = HERAData(self.miriad_1, filetype='miriad')
            d, f, n = next(hd.iterate_over_bls(bls=[(52, 53, 'xx')], chunk_by_redundant_group=True))

    def test_iterate_over_freqs(self):
        hd = HERAData(self.uvh5_1)
        for (d, f, n) in hd.iterate_over_freqs(Nchans=256):
            for dc in (d, f, n):
                assert len(dc.keys()) == 3
                assert list(dc.values())[0].shape == (60, 256)

        hd = HERAData([self.uvh5_1, self.uvh5_2])
        for (d, f, n) in hd.iterate_over_freqs(Nchans=512):
            for dc in (d, f, n):
                assert len(dc.keys()) == 3
                assert list(dc.values())[0].shape == (120, 512)

        hd = HERAData(self.uvfits, filetype='uvfits')
        d, f, n = hd.read()
        d, f, n = next(hd.iterate_over_freqs(Nchans=2, freqs=d.freqs[0:2]))
        for value in d.values():
            assert value.shape == (60, 2)
        with pytest.raises(NotImplementedError):
            next(hd.iterate_over_bls())

    def test_iterate_over_times(self):
        hd = HERAData(self.uvh5_1)
        for (d, f, n) in hd.iterate_over_times(Nints=20):
            for dc in (d, f, n):
                assert len(dc.keys()) == 3
                assert list(dc.values())[0].shape == (20, 1024)

        hd.read(frequencies=hd.freqs[0:512])
        hd.write_uvh5('out1.h5', clobber=True)
        hd.read(frequencies=hd.freqs[512:])
        hd.write_uvh5('out2.h5', clobber=True)
        hd = HERAData(['out1.h5', 'out2.h5'])
        for (d, f, n) in hd.iterate_over_times(Nints=30):
            for dc in (d, f, n):
                assert len(dc.keys()) == 3
                assert list(dc.values())[0].shape == (30, 1024)
        os.remove('out1.h5')
        os.remove('out2.h5')

        hd = HERAData(self.uvfits, filetype='uvfits')
        d, f, n = hd.read()
        d, f, n = next(hd.iterate_over_times(Nints=2, times=d.times[0:2]))
        for value in d.values():
            assert value.shape == (2, 64)
        with pytest.raises(NotImplementedError):
            next(hd.iterate_over_times())

    def test_uvflag_compatibility(self):
        # Test that UVFlag is able to successfully init from the HERAData object
        uv = UVData()
        uv.read_uvh5(self.uvh5_1)
        uvf1 = UVFlag(uv)
        hd = HERAData(self.uvh5_1)
        hd.read()
        uvf2 = UVFlag(hd)
        assert uvf1 == uvf2

    def init_HERACal(self):
        hd = HERAData(self.uvh5_1)
        hc = hd.init_HERACal(gain_convention='divide', cal_style='redundant')

        # check some metadata
        assert hd.Ntimes == hc.Ntimes
        assert hc.Nants_data == hd.Nants_data
        assert hc.telescope_name == hd.telescope_name

        # check that arrays have been initialized
        assert hc.gain_array is not None
        np.testing.assert_array_equal(hc.gain_array, 1 + 0j)
        assert hc.flag_array is not None
        np.testing.assert_array_equal(hc.flag_array, True)
        assert hc.quality_array is not None
        np.testing.assert_array_equal(hc.quality_array, 0)
        assert hc.total_quality_array is not None
        np.testing.assert_array_equal(hc.total_quality_array, 0)


class Test_ReadHeraHdf5(object):
    def setup_method(self):
        self.uvh5_1 = os.path.join(DATA_PATH, "zen.2458116.61019.xx.HH.XRS_downselected.uvh5")
        self.uvh5_2 = os.path.join(DATA_PATH, "zen.2458116.61765.xx.HH.XRS_downselected.uvh5")
        self.uvh5_pol = os.path.join(DATA_PATH, "zen.2458116.61019.xx.HH.XRS_downselected.uvh5_poltranspose")
        self.uvh5_blt = os.path.join(DATA_PATH, "zen.2459114.60020.sum.downsample_transpose.uvh5")

    def test_basic_read(self):
        for replace in ['.uvh5', '.new_shape.uvh5']:
            rv = io.read_hera_hdf5([self.uvh5_1.replace('.uvh5', replace), self.uvh5_2.replace('.uvh5', replace)],
                                   read_flags=False, read_nsamples=False, verbose=True,
                                   dtype=np.complex128, check=True)
            assert 'info' in rv
            assert 'data' in rv
            assert 'flags' not in rv
            assert 'nsamples' not in rv
            assert len(rv['info']['bls']) * len(rv['info']['pols']) == len(rv['data'])
            assert rv['info']['times'].size == np.unique(rv['info']['times']).size
            for bl, data in rv['data'].items():
                assert data.shape == (rv['info']['times'].size, rv['info']['freqs'].size)
                assert data.dtype == np.complex128

    def test_broken_read(self):
        with pytest.raises(ValueError):
            rv = io.read_hera_hdf5([self.uvh5_1, self.uvh5_2], bls=[(999, 999, 'xx')],
                                   read_flags=False, read_nsamples=False, verbose=True,
                                   dtype=np.complex128, check=True)

    def test_info_only(self):
        rv = io.read_hera_hdf5([self.uvh5_1, self.uvh5_2], verbose=True, check=True,
                               read_data=False, read_flags=False, read_nsamples=False)
        assert 'info' in rv
        assert 'data' not in rv
        assert 'flags' not in rv
        assert 'nsamples' not in rv

    def test_read_all(self):
        rv = io.read_hera_hdf5([self.uvh5_1, self.uvh5_2], verbose=True, check=True,
                               read_flags=True, read_nsamples=True)
        assert 'info' in rv
        assert 'data' in rv
        assert 'flags' in rv
        assert 'nsamples' in rv
        assert len(rv['info']['bls']) * len(rv['info']['pols']) == len(rv['data'])
        assert len(rv['info']['bls']) * len(rv['info']['pols']) == len(rv['flags'])
        assert len(rv['info']['bls']) * len(rv['info']['pols']) == len(rv['nsamples'])
        for bl, data in rv['flags'].items():
            assert data.shape == (rv['info']['times'].size, rv['info']['freqs'].size)
            assert data.dtype == bool
        for bl, data in rv['nsamples'].items():
            assert data.shape == (rv['info']['times'].size, rv['info']['freqs'].size)
            assert data.dtype == np.float32

    def test_read_allbls_poltranspose(self):
        for replace in ['.uvh5', '.new_shape.uvh5']:
            rv = io.read_hera_hdf5([self.uvh5_pol.replace('.uvh5', replace)], dtype=np.complex128, verbose=True, check=True)
            assert 'info' in rv
            assert 'data' in rv
            assert len(rv['info']['bls']) * len(rv['info']['pols']) == len(rv['data'])
            for bl, data in rv['data'].items():
                assert data.shape == (rv['info']['times'].size, rv['info']['freqs'].size)

    def test_read_one_bl(self):
        rv = io.read_hera_hdf5([self.uvh5_1], verbose=True, check=True,
                               read_data=False, read_flags=False, read_nsamples=False)
        bl = list(rv['info']['bls'])[0]
        pol = rv['info']['pols'][0]
        bl = bl + (pol,)
        rv = io.read_hera_hdf5([self.uvh5_1], bls=[bl])
        assert len(rv['data']) == 1
        assert bl in rv['data']

    def test_read_one_bl_poltranpose(self):
        rv = io.read_hera_hdf5(self.uvh5_pol, verbose=True, check=True,
                               read_data=False, read_flags=False, read_nsamples=False)
        bl = list(rv['info']['bls'])[0]
        pol = rv['info']['pols'][0]
        bl = bl + (pol,)
        rv = io.read_hera_hdf5([self.uvh5_1], bls=[bl])
        assert len(rv['data']) == 1
        assert bl in rv['data']

    def test_read_bl_then_time_poltranpose(self):
        for replace in ['.uvh5', '.new_shape.uvh5']:
            rv = io.read_hera_hdf5([self.uvh5_blt.replace('.uvh5', replace)], verbose=True, bls=[(24, 26)], check=True,
                                   read_data=True, read_flags=True, read_nsamples=True)
            assert len(rv['data']) == 4
            assert (24, 26, 'ee') in rv['data']
            assert rv['data'][(24, 26, 'ee')].shape == (2, 1536)
            assert len(rv['info']['times']) == 2


class Test_HERADataFastReader(object):
    def setup_method(self):
        self.uvh5_1 = os.path.join(DATA_PATH, "test_input", "zen.2458042.60288.HH.uvRXLS.uvh5_downselected")
        self.uvh5_2 = os.path.join(DATA_PATH, "test_input", "zen.2458042.61034.HH.uvRXLS.uvh5_downselected")
        self.uvh5_h4c = os.path.join(DATA_PATH, "zen.2459122.30030.sum.single_time.uvh5")

    def test_init(self):
        hd = io.HERADataFastReader(self.uvh5_1)
        assert hd.filepaths == [self.uvh5_1]
        assert hd.antpos is not None

        hd = io.HERADataFastReader(self.uvh5_1, read_metadata=False)
        assert hd.filepaths == [self.uvh5_1]
        assert hd.antpos is None

    def test_read_data(self):
        rv = io.read_hera_hdf5([self.uvh5_1])
        hd = io.HERADataFastReader(self.uvh5_1)
        d, f, n = hd.read(read_flags=False, read_nsamples=False, check=True)
        assert f is None
        assert n is None
        for bl in d:
            if split_bl(bl)[0] != split_bl(bl)[1]:
                np.testing.assert_array_equal(d[bl], rv['data'][bl])
                np.testing.assert_array_equal(d[bl], np.conj(d[reverse_bl(bl)]))
            else:
                np.testing.assert_array_equal(d[bl], np.abs(rv['data'][bl]))

    def test_comp_to_HERAData(self):
        for infile in ([self.uvh5_1], [self.uvh5_1, self.uvh5_2], self.uvh5_h4c):
            hd = io.HERADataFastReader(infile, read_metadata=False)
            d, f, n = hd.read(check=True)
            hd2 = io.HERAData(infile)
            d2, f2, n2 = hd2.read()
            # compare all data and metadata
            for dc1, dc2 in zip([d, f, n], [d2, f2, n2]):
                for bl in dc1:
                    if (split_bl(bl)[0] == split_bl(bl)[1]) and (infile != self.uvh5_h4c):
                        # somehow there are numerical issues at play for H1C data
                        np.testing.assert_allclose(dc1[bl], dc2[bl], rtol=1e-6)
                    else:
                        np.testing.assert_array_equal(dc1[bl], dc2[bl])
                np.testing.assert_array_equal(dc1.freqs, dc2.freqs)
                np.testing.assert_array_equal(dc1.times, dc2.times)
                np.testing.assert_allclose(dc1.lsts, dc2.lsts)
                np.testing.assert_array_equal(dc1.ants, dc2.ants)
                np.testing.assert_array_equal(dc1.data_ants, dc2.data_ants)
                np.testing.assert_array_equal(sorted(dc1.pols()), sorted(dc2.pols()))
                np.testing.assert_array_equal(sorted(dc1.antpairs()), sorted(dc2.antpairs()))
                np.testing.assert_array_equal(sorted(dc1.bls()), sorted(dc2.bls()))
                for ant in dc1.antpos:
                    np.testing.assert_array_almost_equal(dc1.antpos[ant] - dc2.antpos[ant], 0)
                for ant in dc1.data_antpos:
                    np.testing.assert_array_almost_equal(dc1.antpos[ant] - dc2.antpos[ant], 0)
                for ap in dc1.times_by_bl:
                    np.testing.assert_array_equal(dc1.times_by_bl[ap], dc2.times_by_bl[ap])
                for ap in dc1.lsts_by_bl:
                    np.testing.assert_allclose(dc1.lsts_by_bl[ap], dc2.lsts_by_bl[ap])

        # compare metadata stored in hd object
        for infile in ([self.uvh5_1], self.uvh5_h4c):
            hd1 = io.HERADataFastReader(infile)
            d, f, n = hd.read(check=True)
            hd2 = io.HERAData(infile)
            d2, f2, n2 = hd2.read()
            np.testing.assert_array_equal(hd1.freqs, hd2.freqs)
            np.testing.assert_array_equal(hd1.times, hd2.times)
            np.testing.assert_allclose(hd1.lsts, hd2.lsts)
            np.testing.assert_array_equal(hd1.ants, hd2.ants)
            np.testing.assert_array_equal(hd1.data_ants, hd2.data_ants)
            np.testing.assert_array_equal(hd1.pols, hd2.pols)
            np.testing.assert_array_equal(hd1.antpairs, hd2.antpairs)
            np.testing.assert_array_equal(hd1.bls, hd2.bls)
            for ant in hd1.antpos:
                np.testing.assert_array_almost_equal(hd1.antpos[ant] - hd2.antpos[ant], 0)
            for ant in hd1.data_antpos:
                np.testing.assert_array_almost_equal(hd1.antpos[ant] - hd2.antpos[ant], 0)
            for ap in hd1.times_by_bl:
                np.testing.assert_array_equal(hd1.times_by_bl[ap], hd2.times_by_bl[ap])
            for ap in hd1.lsts_by_bl:
                np.testing.assert_allclose(hd1.lsts_by_bl[ap], hd2.lsts_by_bl[ap])

    def test_errors(self):
        hd = io.HERADataFastReader([self.uvh5_1, self.uvh5_2])
        with pytest.raises(NotImplementedError):
            hd.write_uvh5()
        with pytest.raises(NotImplementedError):
            hd.iterate_over_bls('stuff', fake_kwarg=False)


class Test_ReadHeraCalfits(object):
    def setup_method(self):
        self.fname_xx = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.xx.HH.uvc.omni.calfits")
        self.fname_yy = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.yy.HH.uvc.omni.calfits")
        self.fname_2pol = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.HH.omni.calfits")
        self.fname = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.HH.uvcA.omni.calfits")
        self.fname_t0 = os.path.join(DATA_PATH, 'test_input/zen.2458101.44615.xx.HH.uv.abs.calfits_54x_only')
        self.fname_t1 = os.path.join(DATA_PATH, 'test_input/zen.2458101.45361.xx.HH.uv.abs.calfits_54x_only')
        self.fname_t2 = os.path.join(DATA_PATH, 'test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only')

    def test_read_info(self):
        rv = io.read_hera_calfits(self.fname_xx, read_gains=False, read_flags=False,
                                  read_quality=False, read_tot_quality=False, check=True,
                                  verbose=True)
        assert 'info' in rv
        assert len(rv) == 1
        for key in ('ants', 'pols', 'freqs', 'times'):
            assert key in rv['info']

    def test_vs_heracal(self):
        hc = io.HERACal(self.fname_t0)
        g, f, q, tq = hc.read()
        rv = io.read_hera_calfits(self.fname_t0,
                                  read_gains=True, read_flags=True,
                                  read_quality=True, read_tot_quality=True, check=True,
                                  verbose=True)
        assert np.allclose(hc.times, rv['info']['times'])
        assert np.allclose(hc.freqs, rv['info']['freqs'])
        assert hc.x_orientation == rv['info']['x_orientation']
        assert hc.gain_convention == rv['info']['gain_convention']
        for key, gain in g.items():
            assert np.allclose(gain, rv['gains'][key])
        for key, flag in f.items():
            assert np.allclose(flag, rv['flags'][key])
        for key, qual in q.items():
            assert np.allclose(qual, rv['quality'][key])
        for key, total_qual in tq.items():
            assert np.allclose(total_qual, rv['total_quality'][key])

    def test_read(self):
        # test one file with both polarizations and a non-None total quality array
        rv = io.read_hera_calfits(self.fname, read_gains=True, read_flags=True,
                                  read_quality=True, read_tot_quality=True,
                                  dtype=np.complex128, check=True, verbose=True)
        for key in ('info', 'gains', 'flags', 'quality', 'total_quality'):
            assert key in rv
        shape = (rv['info']['times'].size, rv['info']['freqs'].size)
        assert rv['info']['freqs'].size == 1024
        assert rv['info']['times'].size == 1
        for key, gain in rv['gains'].items():
            assert len(key) == 2
            assert gain.dtype == np.complex128
            assert gain.shape == shape
        for key, flag in rv['flags'].items():
            assert len(key) == 2
            assert flag.dtype == bool
            assert flag.shape == shape
        for key, qual in rv['quality'].items():
            assert len(key) == 2
            assert qual.dtype == np.float32
            assert qual.shape == shape
        for key, qual in rv['total_quality'].items():
            assert type(key) == str
            assert qual.dtype == np.float32
            assert qual.shape == shape
        assert rv['info']['pols'] == set(['Jnn', 'Jee'])

        # test list loading
        rv = io.read_hera_calfits([self.fname_xx, self.fname_yy], read_gains=True, read_flags=True,
                                  read_quality=True, read_tot_quality=False,
                                  check=True, verbose=True)
        for key in ('gains', 'flags', 'quality'):
            assert len(rv[key].keys()) == 36

    def test_read_select(self):
        # test read multiple files and select ants
        ants = [(54, 'Jee')]
        rv = io.read_hera_calfits([self.fname_t0, self.fname_t1, self.fname_t2], ants=ants)
        assert len(rv['gains']) == 1
        assert ants[0] in rv['gains']

        # test select on antenna numbers
        rv1 = io.read_hera_calfits([self.fname_xx, self.fname_yy], ants=(9, 10))
        rv2 = io.read_hera_calfits(self.fname_2pol, ants=(9, 10))
        for k in rv2['gains'].keys():
            assert k[0] in [9, 10]
            np.testing.assert_array_equal(rv1['gains'][k], rv2['gains'][k])

        # test select on pols
        rv1 = io.read_hera_calfits(self.fname_xx)
        rv2 = io.read_hera_calfits(self.fname_2pol, pols=['Jee'])
        for k in rv2['gains'].keys():
            assert k[1] == 'Jee'
            assert k in rv1['gains']
            np.testing.assert_array_equal(rv1['gains'][k], rv2['gains'][k])


@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
class Test_Visibility_IO_Legacy(object):
    def test_load_vis(self):
        # inheretied testing from the old abscal_funcs.UVData2AbsCalDict

        # load into pyuvdata object
        self.data_file = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        self.uvd = UVData()
        self.uvd.read_miriad(self.data_file)
        self.uvd.use_future_array_shapes()
        self.freq_array = np.unique(self.uvd.freq_array)
        self.antpos, self.ants = self.uvd.get_ENU_antpos(center=True, pick_data_ants=True)
        self.antpos = odict(list(zip(self.ants, self.antpos)))
        self.time_array = np.unique(self.uvd.time_array)

        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        data, flags = io.load_vis(fname, pop_autos=False)
        assert data[(24, 25, 'ee')].shape == (60, 64)
        assert flags[(24, 25, 'ee')].shape == (60, 64)
        assert (24, 24, parse_polstr('EE', x_orientation=self.uvd.x_orientation)) in data
        data, flags = io.load_vis([fname])
        assert data[(24, 25, 'ee')].shape == (60, 64)

        # test pop autos
        data, flags = io.load_vis(fname, pop_autos=True)
        assert (24, 24, parse_polstr('EE', x_orientation=self.uvd.x_orientation)) not in data

        # test uvd object
        uvd = UVData()
        uvd.read_miriad(fname)
        uvd.use_future_array_shapes()
        data, flags = io.load_vis(uvd)
        assert data[(24, 25, 'ee')].shape == (60, 64)
        data, flags = io.load_vis([uvd])
        assert data[(24, 25, 'ee')].shape == (60, 64)

        # test multiple
        fname2 = os.path.join(DATA_PATH, "zen.2458043.13298.xx.HH.uvORA")
        data, flags = io.load_vis([fname, fname2])
        assert data[(24, 25, 'ee')].shape == (120, 64)
        assert flags[(24, 25, 'ee')].shape == (120, 64)

        # test w/ meta
        d, f, ap, a, f, t, l, p = io.load_vis([fname, fname2], return_meta=True)
        assert len(ap[24]) == 3
        assert len(f) == len(self.freq_array)

        with pytest.raises(NotImplementedError):
            d, f = io.load_vis(fname, filetype='not_a_real_filetype')
        with pytest.raises(NotImplementedError):
            d, f = io.load_vis(['str1', 'str2'], filetype='not_a_real_filetype')

        # test w/ meta pick_data_ants
        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        d, f, ap, a, f, t, l, p = io.load_vis(fname, return_meta=True, pick_data_ants=False)
        assert len(ap[24]) == 3
        assert len(a) == 47
        assert len(f) == len(self.freq_array)

        with pytest.raises(TypeError):
            d, f = io.load_vis(1.0)

    def test_load_vis_nested(self):
        # duplicated testing from firstcal.UVData_to_dict
        filename1 = os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA')
        filename2 = os.path.join(DATA_PATH, 'zen.2458043.13298.xx.HH.uvORA')
        uvd1 = UVData()
        uvd1.read_miriad(filename1)
        uvd1.use_future_array_shapes()

        uvd2 = UVData()
        uvd2.read_miriad(filename2)
        uvd2.use_future_array_shapes()
        if uvd1.phase_type != 'drift':
            uvd1.unphase_to_drift()
        if uvd2.phase_type != 'drift':
            uvd2.unphase_to_drift()
        uvd = uvd1 + uvd2
        d, f = io.load_vis([uvd1, uvd2], nested_dict=True)
        for i, j in d:
            for pol in d[i, j]:
                uvpol = list(uvd1.polarization_array).index(polstr2num(pol, x_orientation=uvd1.x_orientation))
                uvmask = np.all(
                    np.array(list(zip(uvd.ant_1_array, uvd.ant_2_array))) == [i, j], axis=1)
                np.testing.assert_equal(d[i, j][pol], np.resize(
                    uvd.data_array[uvmask][:, :, uvpol], d[i, j][pol].shape))
                np.testing.assert_equal(f[i, j][pol], np.resize(
                    uvd.flag_array[uvmask][:, :, uvpol], f[i, j][pol].shape))

        d, f = io.load_vis([filename1, filename2], nested_dict=True)
        for i, j in d:
            for pol in d[i, j]:
                uvpol = list(uvd.polarization_array).index(polstr2num(pol, x_orientation=uvd.x_orientation))
                uvmask = np.all(
                    np.array(list(zip(uvd.ant_1_array, uvd.ant_2_array))) == [i, j], axis=1)
                np.testing.assert_equal(d[i, j][pol], np.resize(
                    uvd.data_array[uvmask][:, :, uvpol], d[i, j][pol].shape))
                np.testing.assert_equal(f[i, j][pol], np.resize(
                    uvd.flag_array[uvmask][:, :, uvpol], f[i, j][pol].shape))

        uvd = UVData()
        uvd.read_miriad(filename1)
        uvd.use_future_array_shapes()
        assert len(io.load_vis([uvd], nested_dict=True)[0]) == uvd.Nbls
        # reorder baseline array
        uvd.baseline_array = uvd.baseline_array[np.argsort(uvd.baseline_array)]
        assert len(io.load_vis(filename1, nested_dict=True)[0]) == uvd.Nbls

    @pytest.mark.filterwarnings("ignore:The expected shape of the ENU array")
    @pytest.mark.filterwarnings("ignore:antenna_diameters is not set")
    @pytest.mark.filterwarnings("ignore:Unicode equal comparison failed")
    def test_write_vis(self):
        # get data
        uvd = UVData()
        uvd.read_uvh5(os.path.join(DATA_PATH, "zen.2458044.41632.xx.HH.XRAA.uvh5"))
        uvd.use_future_array_shapes()
        data, flgs, ap, a, f, t, l, p = io.load_vis(uvd, return_meta=True)
        nsample = copy.deepcopy(data)
        for k in nsample.keys():
            nsample[k] = np.ones_like(nsample[k], float)

        # test basic execution
        uvd = io.write_vis("ex.uvh5", data, l, f, ap, start_jd=2458044, return_uvd=True, overwrite=True, verbose=True, x_orientation='east', filetype='uvh5')
        uvd.use_future_array_shapes()
        hd = HERAData("ex.uvh5")
        hd.read()
        assert os.path.exists('ex.uvh5')
        assert uvd.data_array.shape == (1680, 64, 1)
        assert hd.data_array.shape == (1680, 64, 1)
        assert np.allclose(data[(24, 25, 'ee')][30, 32], uvd.get_data(24, 25, 'ee')[30, 32])
        assert np.allclose(data[(24, 25, 'ee')][30, 32], hd.get_data(24, 25, 'ee')[30, 32])
        assert hd.x_orientation.lower() == 'east'
        for ant in ap:
            np.testing.assert_array_almost_equal(hd.antpos[ant], ap[ant])
        os.remove("ex.uvh5")

        # test with nsample and flags
        uvd = io.write_vis("ex.uv", data, l, f, ap, start_jd=2458044, flags=flgs, nsamples=nsample, x_orientation='east', return_uvd=True, overwrite=True, verbose=True)
        uvd.use_future_array_shapes()
        assert uvd.nsample_array.shape == (1680, 64, 1)
        assert uvd.flag_array.shape == (1680, 64, 1)
        assert np.allclose(nsample[(24, 25, 'ee')][30, 32], uvd.get_nsamples(24, 25, 'ee')[30, 32])
        assert np.allclose(flgs[(24, 25, 'ee')][30, 32], uvd.get_flags(24, 25, 'ee')[30, 32])
        assert uvd.x_orientation.lower() == 'east'

        # test exceptions
        pytest.raises(AttributeError, io.write_vis, "ex.uv", data, l, f, ap)
        pytest.raises(AttributeError, io.write_vis, "ex.uv", data, l, f, ap, start_jd=2458044, filetype='foo')
        if os.path.exists('ex.uv'):
            shutil.rmtree('ex.uv')

    def test_update_vis(self):
        # load in cal
        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        outname = os.path.join(DATA_PATH, "test_output/zen.2458043.12552.xx.HH.modified.uvORA")
        uvd = UVData()
        uvd.read_miriad(fname)
        uvd.use_future_array_shapes()
        data, flags, antpos, ants, freqs, times, lsts, pols = io.load_vis(fname, return_meta=True)

        # make some modifications
        new_data = {key: (2.) * val for key, val in data.items()}
        new_flags = {key: np.logical_not(val) for key, val in flags.items()}
        io.update_vis(fname, outname, data=new_data, flags=new_flags,
                      add_to_history='hello world', clobber=True, telescope_name='PAPER')

        # test modifications
        data, flags, antpos, ants, freqs, times, lsts, pols = io.load_vis(outname, return_meta=True)
        for k in data.keys():
            assert np.all(new_data[k] == data[k])
            assert np.all(new_flags[k] == flags[k])
        uvd2 = UVData()
        uvd2.read_miriad(outname)
        assert pyuvdata.utils._check_histories(uvd2.history, uvd.history + 'hello world')
        assert uvd2.telescope_name == 'PAPER'
        shutil.rmtree(outname)

        # test writing uvfits instead
        io.update_vis(fname, outname, data=new_data, flags=new_flags, filetype_out='uvfits',
                      add_to_history='hello world', clobber=True, telescope_name='PAPER')
        uvd_fits = UVData()
        uvd_fits.read_uvfits(outname)
        os.remove(outname)

        # Coverage for errors
        with pytest.raises(TypeError):
            io.update_vis(uvd, outname, data=new_data, flags=new_flags, filetype_out='not_a_real_filetype',
                          add_to_history='hello world', clobber=True, telescope_name='PAPER')
        with pytest.raises(NotImplementedError):
            io.update_vis(fname, outname, data=new_data, flags=new_flags, filetype_in='not_a_real_filetype',
                          add_to_history='hello world', clobber=True, telescope_name='PAPER')

        # #now try the same thing but with a UVData object instead of path
        io.update_vis(uvd, outname, data=new_data, flags=new_flags,
                      add_to_history='hello world', clobber=True, telescope_name='PAPER')
        data, flags, antpos, ants, freqs, times, lsts, pols = io.load_vis(outname, return_meta=True)
        for k in data.keys():
            assert np.all(new_data[k] == data[k])
            assert np.all(new_flags[k] == flags[k])
        uvd2 = UVData()
        uvd2.read_miriad(outname)
        assert pyuvdata.utils._check_histories(uvd2.history, uvd.history + 'hello world')
        assert uvd2.telescope_name == 'PAPER'
        shutil.rmtree(outname)


class Test_Calibration_IO_Legacy(object):
    def test_load_cal(self):
        with pytest.raises(TypeError):
            io.load_cal(1.0)

        fname = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.xx.HH.uvc.omni.calfits")
        gains, flags = io.load_cal(fname)
        assert len(gains.keys()) == 18
        assert len(flags.keys()) == 18

        cal = UVCal()
        cal.read_calfits(fname)
        cal.use_future_array_shapes()
        gains, flags = io.load_cal(cal)
        assert len(gains.keys()) == 18
        assert len(flags.keys()) == 18

        with pytest.raises(TypeError):
            io.load_cal([fname, cal])

        fname_xx = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.xx.HH.uvc.omni.calfits")
        fname_yy = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.yy.HH.uvc.omni.calfits")
        gains, flags, quals, total_qual, ants, freqs, times, pols = io.load_cal([fname_xx, fname_yy], return_meta=True)
        assert len(gains.keys()) == 36
        assert len(flags.keys()) == 36
        assert len(quals.keys()) == 36
        assert freqs.shape == (1024,)
        assert times.shape == (3,)
        assert sorted(pols) == [parse_jpolstr('jxx', x_orientation=cal.x_orientation), parse_jpolstr('jyy', x_orientation=cal.x_orientation)]

        cal_xx, cal_yy = UVCal(), UVCal()
        cal_xx.read_calfits(fname_xx)
        cal_yy.read_calfits(fname_yy)
        cal_xx.use_future_array_shapes()
        cal_yy.use_future_array_shapes()
        gains, flags, quals, total_qual, ants, freqs, times, pols = io.load_cal([cal_xx, cal_yy], return_meta=True)
        assert len(gains.keys()) == 36
        assert len(flags.keys()) == 36
        assert len(quals.keys()) == 36
        assert freqs.shape == (1024,)
        assert times.shape == (3,)
        assert sorted(pols) == [parse_jpolstr('jxx', x_orientation=cal_xx.x_orientation), parse_jpolstr('jyy', x_orientation=cal_yy.x_orientation)]

    def test_write_cal(self):
        # create fake data
        ants = np.arange(10)
        pols = np.array(['Jnn'])
        freqs = np.linspace(100e6, 200e6, 64, endpoint=False)
        Nfreqs = len(freqs)
        times = np.linspace(2458043.1, 2458043.6, 100)
        Ntimes = len(times)
        gains = {}
        quality = {}
        flags = {}
        total_qual = {}
        for i, p in enumerate(pols):
            total_qual[p] = np.ones((Ntimes, Nfreqs), float)
            for j, a in enumerate(ants):
                gains[(a, p)] = np.ones((Ntimes, Nfreqs), complex)
                quality[(a, p)] = np.ones((Ntimes, Nfreqs), float) * 2
                flags[(a, p)] = np.zeros((Ntimes, Nfreqs), bool)

        # set some terms to zero
        gains[(5, 'Jnn')][20:30] *= 0

        # test basic execution
        uvc = io.write_cal("ex.calfits", gains, freqs, times, flags=flags, quality=quality,
                           total_qual=total_qual, overwrite=True, return_uvc=True, write_file=True)
        assert os.path.exists("ex.calfits")
        assert uvc.gain_array.shape == (10, 1, 64, 100, 1)
        assert np.allclose(uvc.gain_array[5].min(), 1.0)
        assert np.allclose(uvc.gain_array[0, 0, 0, 0, 0], (1 + 0j))
        assert np.allclose(np.sum(uvc.gain_array), (64000 + 0j))
        assert not np.any(uvc.flag_array[0, 0, 0, 0, 0])
        assert np.sum(uvc.flag_array) == 640
        assert np.allclose(uvc.quality_array[0, 0, 0, 0, 0], 2)
        assert np.allclose(np.sum(uvc.quality_array), 128000.0)
        assert len(uvc.antenna_numbers) == 10
        assert uvc.total_quality_array is not None
        if os.path.exists('ex.calfits'):
            os.remove('ex.calfits')
        # test execution with different parameters
        uvc = io.write_cal("ex.calfits", gains, freqs, times, overwrite=True)
        if os.path.exists('ex.calfits'):
            os.remove('ex.calfits')
        # test single integration write
        gains2 = odict([(k, gains[k][:1]) for k in gains.keys()])
        uvc = io.write_cal("ex.calfits", gains2, freqs, times[:1], return_uvc=True, outdir='./')
        assert np.allclose(uvc.integration_time, 0.0)
        assert uvc.Ntimes == 1
        assert os.path.exists('ex.calfits')
        os.remove('ex.calfits')

        # test multiple pol
        for k in list(gains.keys()):
            gains[(k[0], 'Jyy')] = gains[k].conj()
        uvc = io.write_cal("ex.calfits", gains, freqs, times, return_uvc=True, outdir='./')
        assert uvc.gain_array.shape == (10, 1, 64, 100, 2)
        np.testing.assert_array_almost_equal(uvc.gain_array[0, 0, :, :, 0], uvc.gain_array[0, 0, :, :, 1].conj())
        os.remove('ex.calfits')

        # test zero check
        gains[(0, 'Jnn')][:] = 0.0
        uvc1 = io.write_cal("ex.calfits", gains, freqs, times, return_uvc=True, write_file=False, outdir='./', zero_check=True)
        uvc2 = io.write_cal("ex.calfits", gains, freqs, times, return_uvc=True, write_file=False, outdir='./', zero_check=False)
        assert np.allclose(uvc1.gain_array[0, 0, :, :, 0], 1.0)
        assert np.allclose(uvc2.gain_array[0, 0, :, :, 0], 0.0)

        # test antenna number and names ordering
        antnums2antnames = {a: "THISANT{}".format(a + 1) for a in ants}
        uvc = io.write_cal("ex.calfits", gains, freqs, times, antnums2antnames=antnums2antnames,
                           return_uvc=True, write_file=False)
        assert sorted(uvc.antenna_names) == sorted(antnums2antnames.values())

    def test_update_cal(self):
        # load in cal
        fname = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.xx.HH.uvc.omni.calfits")
        outname = os.path.join(DATA_PATH, "test_output/zen.2457698.40355.xx.HH.uvc.modified.calfits.")
        cal = UVCal()
        cal.read_calfits(fname)
        cal.use_future_array_shapes()
        gains, flags, quals, total_qual, ants, freqs, times, pols = io.load_cal(fname, return_meta=True)

        # make some modifications
        new_gains = {key: (2. + 1.j) * val for key, val in gains.items()}
        new_flags = {key: np.logical_not(val) for key, val in flags.items()}
        new_quals = {key: 2. * val for key, val in quals.items()}
        io.update_cal(fname, outname, gains=new_gains, flags=new_flags, quals=new_quals,
                      add_to_history='hello world', clobber=True, telescope_name='MWA')

        # test modifications
        gains, flags, quals, total_qual, ants, freqs, times, pols = io.load_cal(outname, return_meta=True)
        for k in gains.keys():
            assert np.all(new_gains[k] == gains[k])
            assert np.all(new_flags[k] == flags[k])
            assert np.all(new_quals[k] == quals[k])
        cal2 = UVCal()
        cal2.read_calfits(outname)
        assert pyuvdata.utils._check_histories(cal2.history, cal.history + 'hello world')
        assert cal2.telescope_name == 'MWA'
        os.remove(outname)

        # now try the same thing but with a UVCal object instead of path
        io.update_cal(cal, outname, gains=new_gains, flags=new_flags, quals=new_quals,
                      add_to_history='hello world', clobber=True, telescope_name='MWA')
        gains, flags, quals, total_qual, ants, freqs, times, pols = io.load_cal(outname, return_meta=True)
        for k in gains.keys():
            assert np.all(new_gains[k] == gains[k])
            assert np.all(new_flags[k] == flags[k])
            assert np.all(new_quals[k] == quals[k])
        cal2 = UVCal()
        cal2.read_calfits(outname)
        assert pyuvdata.utils._check_histories(cal2.history, cal.history + 'hello world')
        assert cal2.telescope_name == 'MWA'
        os.remove(outname)


class Test_Flags_IO(object):
    def test_load_flags_npz(self):
        npzfile = os.path.join(DATA_PATH, "test_input/zen.2458101.45361.xx.HH.uvOCR_53x_54x_only.flags.applied.npz")
        flags = io.load_flags(npzfile, filetype='npz')
        assert (53, 54, parse_polstr('XX')) in flags
        for f in flags.values():
            assert f.shape == (60, 1024)
            np.testing.assert_array_equal(f[:, 0:50], True)
            np.testing.assert_array_equal(f[:, -50:], True)
            assert not np.all(f)

        flags, meta = io.load_flags(npzfile, filetype='npz', return_meta=True)
        assert len(meta['freqs']) == 1024
        assert len(meta['times']) == 60
        assert 'history' in meta

    def test_load_flags_h5_baseline(self):
        h5file = os.path.join(QM_DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.testuvflag.flags.h5')
        flags, meta = io.load_flags(h5file, return_meta=True)
        assert len(meta['freqs']) == 256
        assert len(meta['times']) == 3
        assert 'history' in meta
        assert (20, 105, 'xx') in flags
        for k in flags.keys():
            assert len(k) == 3
            assert flags[k].shape == (3, 256)

    def test_load_flags_h5_antenna(self):
        h5file = os.path.join(QM_DATA_PATH, 'antenna_flags.h5')
        flags, meta = io.load_flags(h5file, return_meta=True)
        assert len(meta['freqs']) == 256
        assert len(meta['times']) == 3
        assert 'history' in meta
        assert (20, 'Jxx') in flags
        for k in flags.keys():
            assert len(k) == 2
            assert flags[k].shape == (3, 256)

    def test_load_flags_h5_waterfall(self):
        h5file = os.path.join(QM_DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.omni.calfits.g.flags.h5')
        flags, meta = io.load_flags(h5file, return_meta=True)
        assert len(meta['freqs']) == 256
        assert len(meta['times']) == 3
        assert 'history' in meta
        assert 'Jxx' in flags
        for k in flags.keys():
            assert isinstance(k, str)
            assert flags[k].shape == (3, 256)

    def test_load_flags_errors(self):
        with pytest.raises(ValueError):
            flags = io.load_flags('some/path', filetype='not_a_type')

        h5file = os.path.join(QM_DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.testuvflag.h5')
        with pytest.raises(AssertionError):
            flags = io.load_flags(h5file)


class Test_Meta_IO(object):
    def test_read_write_redcal_meta(self):
        # load file, write it back out, reread it, tests that it all agrees
        meta_path = os.path.join(DATA_PATH, "test_input/zen.2458098.43124.downsample.redcal_meta.hdf5")
        fc_meta, omni_meta, freqs, times, lsts, antpos, history = io.read_redcal_meta(meta_path)

        out_path = os.path.join(DATA_PATH, "test_output/redcal_meta_io_test.hdf5")
        io.save_redcal_meta(out_path, fc_meta, omni_meta, freqs, times, lsts, antpos, history)
        fc_meta2, omni_meta2, freqs2, times2, lsts2, antpos2, history2 = io.read_redcal_meta(out_path)

        for key1 in fc_meta:
            for key2 in fc_meta[key1]:
                np.testing.assert_array_equal(fc_meta[key1][key2], fc_meta2[key1][key2])

        for key1 in omni_meta:
            for key2 in omni_meta[key1]:
                np.testing.assert_array_equal(omni_meta[key1][key2], omni_meta2[key1][key2])

        np.testing.assert_array_equal(freqs, freqs2)
        np.testing.assert_array_equal(times, times2)
        np.testing.assert_array_equal(lsts, lsts2)
        for ant in antpos:
            assert np.all(antpos[ant] == antpos2[ant])
        assert history == history2

        os.remove(out_path)


def test_get_file_times():
    filepaths = sorted(glob.glob(DATA_PATH + "/zen.2458042.*.xx.HH.uvXA"))
    # test execution
    dlsts, dtimes, larrs, tarrs = io.get_file_times(filepaths, filetype='miriad')
    assert np.isclose(larrs[0][0], 4.7293432458811866)
    assert np.isclose(larrs[0][-1], 4.7755393587036084)
    assert np.isclose(dlsts[0], 0.00078298496309189868)
    assert len(dlsts) == 2
    assert len(dtimes) == 2
    assert len(larrs) == 2
    assert len(tarrs) == 2
    assert len(larrs[0]) == 60
    assert len(tarrs[0]) == 60

    # test if fed as a str
    dlsts, dtimes, larrs, tarrs = io.get_file_times(filepaths[0], filetype='miriad')
    assert isinstance(dlsts, (float, np.floating))
    assert isinstance(dtimes, (float, np.floating))
    assert larrs.ndim == 1
    assert tarrs.ndim == 1

    # test uvh5
    fp = os.path.join(DATA_PATH, 'zen.2458098.43124.downsample.uvh5')
    dlsts, dtimes, larrs, tarrs = io.get_file_times(fp, filetype='uvh5')
    assert np.isclose(larrs[0], 1.3356485363481176)
    assert np.isclose(larrs[-1], 1.3669679333582787)
    assert np.isclose(dlsts, 0.015659698505080533)

    # test uvh5 no lsts in header
    fp = os.path.join(DATA_PATH, 'test_input/zen.2458863.28532.HH.no_lsts_in_header.uvh5')
    dlsts, dtimes, larrs, tarrs = io.get_file_times(fp, filetype='uvh5')
    assert np.isclose(larrs[0], 1.00925787)
    assert np.isclose(larrs[1], 1.00996256)

    # exceptions
    pytest.raises(ValueError, io.get_file_times, fp, filetype='foo')


def test_get_file_times_bda():
    fps = [os.path.join(DATA_PATH, 'zen.2459122.30030.sum.bda.downsampled.uvh5'),
           os.path.join(DATA_PATH, 'zen.2459122.30119.sum.bda.downsampled.uvh5')]

    # test single file load
    dlsts, dtimes, larrs, tarrs = io.get_file_times(fps[0], filetype='uvh5')
    hd = io.HERAData(fps[0])
    assert dlsts == np.median(np.diff(hd.lsts))
    assert dtimes == np.median(np.diff(hd.times))
    np.testing.assert_array_equal(larrs, hd.lsts)
    np.testing.assert_array_equal(tarrs, hd.times)

    # test multi-file load
    dlsts, dtimes, larrs, tarrs = io.get_file_times(fps, filetype='uvh5')
    hd = io.HERAData(fps)
    for fp, dlst, dtime, larr, tarr in zip(fps, dlsts, dtimes, larrs, tarrs):
        assert dlst == np.median(np.diff(hd.lsts[fp]))
        assert dtime == np.median(np.diff(hd.times[fp]))
        np.testing.assert_array_equal(larr, hd.lsts[fp])
        np.testing.assert_array_equal(tarr, hd.times[fp])


def test_get_file_times_single_integraiton():
    fp = os.path.join(DATA_PATH, 'zen.2459122.30030.sum.single_time.uvh5')
    dlsts, dtimes, larrs, tarrs = io.get_file_times(fp, filetype='uvh5')
    assert len(larrs) == 1
    assert len(tarrs) == 1

    fp2 = os.path.join(DATA_PATH, 'zen.2459122.30030.sum.bda.downsampled.uvh5')
    dlsts2, dtimes2, larrs2, tarrs2 = io.get_file_times(fp, filetype='uvh5')
    np.testing.assert_array_almost_equal(dlsts, dlsts2)
    np.testing.assert_array_almost_equal(dtimes, dtimes2)
    np.testing.assert_array_equal(larrs[0], larrs2[0])
    np.testing.assert_array_equal(tarrs[0], tarrs2[0])


def test_partial_time_io():
    files = [os.path.join(DATA_PATH, 'zen.2459122.30030.sum.bda.downsampled.uvh5'),
             os.path.join(DATA_PATH, 'zen.2459122.30119.sum.bda.downsampled.uvh5')]

    # single file test
    hd = io.HERAData(files[0])

    # pick out specific times/lsts
    d, f, n = io.partial_time_io(hd, times=hd.times[0:2])
    assert np.min([len(tbb) for tbb in d.times_by_bl.values()]) == 2
    assert np.max([len(tbb) for tbb in d.times_by_bl.values()]) == 2
    d, f, n = io.partial_time_io(hd, lsts=hd.lsts[0:2])
    assert np.min([len(tbb) for tbb in d.times_by_bl.values()]) == 2
    assert np.max([len(tbb) for tbb in d.times_by_bl.values()]) == 2

    # pick out range of times/lsts (this should also get baselines with BDA)
    d, f, n = io.partial_time_io(hd, time_range=hd.times[0:2])
    assert np.min([len(tbb) for tbb in d.times_by_bl.values()]) == 1
    assert np.max([len(tbb) for tbb in d.times_by_bl.values()]) == 2
    d, f, n = io.partial_time_io(hd, lst_range=hd.lsts[0:2])
    assert np.min([len(tbb) for tbb in d.times_by_bl.values()]) == 1
    assert np.max([len(tbb) for tbb in d.times_by_bl.values()]) == 2

    # multi-file test, only taking times out of the first two files
    hd = io.HERAData(files)

    # pick out specific times/lsts
    d, f, n = io.partial_time_io(hd, times=hd.times[hd.filepaths[0]][0:2])
    assert np.min([len(tbb) for tbb in d.times_by_bl.values()]) == 2
    assert np.max([len(tbb) for tbb in d.times_by_bl.values()]) == 2
    d, f, n = io.partial_time_io(hd, lsts=hd.lsts[hd.filepaths[0]][0:2])
    assert np.min([len(tbb) for tbb in d.times_by_bl.values()]) == 2
    assert np.max([len(tbb) for tbb in d.times_by_bl.values()]) == 2

    # pick out range of times/lsts (this should also get baselines with BDA)
    d, f, n = io.partial_time_io(hd, time_range=hd.times[hd.filepaths[0]][0:2])
    assert np.min([len(tbb) for tbb in d.times_by_bl.values()]) == 1
    assert np.max([len(tbb) for tbb in d.times_by_bl.values()]) == 2
    d, f, n = io.partial_time_io(hd, lst_range=hd.lsts[hd.filepaths[0]][0:2])
    assert np.min([len(tbb) for tbb in d.times_by_bl.values()]) == 1
    assert np.max([len(tbb) for tbb in d.times_by_bl.values()]) == 2

    # test upsampling
    hd = io.HERAData(files, upsample=True)
    d, f, n = io.partial_time_io(hd, time_range=hd.times[hd.filepaths[0]][0:2])
    assert np.min([len(tbb) for tbb in d.times_by_bl.values()]) == 2
    assert np.max([len(tbb) for tbb in d.times_by_bl.values()]) == 2
    d, f, n = io.partial_time_io(hd, lst_range=hd.lsts[hd.filepaths[0]][0:2])
    assert np.min([len(tbb) for tbb in d.times_by_bl.values()]) == 2
    assert np.max([len(tbb) for tbb in d.times_by_bl.values()]) == 2

    # test errors
    hd = io.HERAData(files)
    with pytest.raises(ValueError):
        io.partial_time_io(hd, lst_range=[0, 1])
    with pytest.raises(ValueError):
        io.partial_time_io(hd, lst_range=hd.lsts[hd.filepaths[0]][0:2], times=hd.times[hd.filepaths[0]][0:2])


def test_baselines_from_filelist_position(tmpdir):
    tmp_path = tmpdir.strpath
    filelist = [os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.first.uvh5"),
                os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.second.uvh5")]
    # below, we test whether for each file we get a chunk of baselines whose length is either greater then 0
    # or less then then 3 (the number of total baselines in this dataset)
    baselines = []
    for file in filelist:
        baseline_chunk = io.baselines_from_filelist_position(file, filelist)
        assert len(baseline_chunk) > 0 and len(baseline_chunk) < 3
        baselines += baseline_chunk
    # Next, we check whether the total number of chunked baselines equals the original number of baselines
    assert len(baselines) == 3
    # sort baselines by the sum of antenna number
    ant_sum = [bl[0] + bl[1] for bl in baselines]
    sum_indices = np.argsort(ant_sum)
    baselines_sorted = [baselines[m] for m in sum_indices]
    # check that the sorted baselines are all of the original baselines.
    assert baselines_sorted == [(53, 53), (53, 54), (54, 54)]
    # test case when there are less baselines then files
    filelist_1bl = [os.path.join(tmp_path, "zen.2458101.46106.xx.HH.OCR_53x_54x_only.first.uvh5"),
                    os.path.join(tmp_path, "zen.2458101.46106.xx.HH.OCR_53x_54x_only.second.uvh5")]
    # to do this, we first generate single baseline files.
    for fi, fo in zip(filelist, filelist_1bl):
        hd = io.HERAData(fi)
        hd.read(bls=[(53, 54)])
        hd.write_uvh5(fo, clobber=True)
    # then we get baseline chunks
    baseline_chunk = io.baselines_from_filelist_position(filelist_1bl[0], filelist_1bl)
    assert baseline_chunk == [(53, 54)]
    baseline_chunk = io.baselines_from_filelist_position(filelist_1bl[1], filelist_1bl)
    assert baseline_chunk == []


def test_throw_away_flagged_ants_parser():
    sys.argv = [sys.argv[0], 'input', 'output', '--yaml_file', 'test']
    ap = io.throw_away_flagged_ants_parser()
    args = ap.parse_args()
    assert args.infilename == 'input'
    assert args.outfilename == 'output'
    assert not args.clobber
    assert not args.throw_away_fully_flagged_data_baselines
    assert args.yaml_file == 'test'


def test_throw_away_flagged_ants(tmpdir):
    strpath = tmpdir.strpath
    inputfile = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.XRAA.uvh5")
    outputfile = os.path.join(strpath, 'trimmed_output.uvh5')
    yaml_file = os.path.join(DATA_PATH, '2458043.yaml')
    hd = io.HERAData(inputfile)
    hd.read()
    for ant in [24, 25, 37, 38, 52]:
        assert ant in set(hd.ant_1_array).union(set(hd.ant_2_array))
    io.throw_away_flagged_ants(inputfile, outputfile, yaml_file)
    hdo = io.HERAData(outputfile)
    for ant in set(hd.ant_1_array).union(set(hd.ant_2_array)):
        if ant in [24, 25, 37, 38]:
            assert ant not in set(hdo.ant_1_array).union(set(hdo.ant_2_array))
        else:
            assert ant in set(hdo.ant_1_array).union(set(hdo.ant_2_array))

    # now fully flag antenna 11 by setting flags to True.
    hdt = copy.deepcopy(hd)
    dt, ft, nt = hdt.build_datacontainers()
    for k in ft:
        ft[k] = np.zeros_like(ft[k], dtype=bool)
        if k[0] in [52] or k[1] in [52]:
            ft[k] = np.ones_like(ft[k], dtype=bool)
    hdt.update(flags=ft)
    manual_file = os.path.join(strpath, 'manually_flagged.uvh5')
    manual_file_trimmed = os.path.join(strpath, 'manually_flagged_trimmed.uvh5')
    hdt.write_uvh5(manual_file)
    io.throw_away_flagged_ants(manual_file, manual_file_trimmed, yaml_file=yaml_file,
                               throw_away_fully_flagged_data_baselines=True)
    hdo = io.HERAData(manual_file_trimmed)
    for ant in set(hd.ant_1_array).union(set(hd.ant_2_array)):
        if ant in [52, 37, 38, 24, 25]:
            assert ant not in set(hdo.ant_1_array).union(set(hdo.ant_2_array))
        else:
            assert ant in set(hdo.ant_1_array).union(set(hdo.ant_2_array))

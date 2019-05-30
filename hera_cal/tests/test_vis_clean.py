# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

from __future__ import print_function, division, absolute_import

import pytest
import numpy as np
from copy import deepcopy
import os
import sys
import shutil
from scipy import constants
from pyuvdata import UVCal, UVData

from hera_cal import io, datacontainer
from hera_cal import vis_clean
from hera_cal.vis_clean import VisClean
from hera_cal.data import DATA_PATH


@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
@pytest.mark.filterwarnings("ignore:It seems that the latitude and longitude are in radians")
class Test_VisClean(object):

    def test_init(self):
        # test basic init
        fname = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.XRAA.uvh5")
        V = VisClean(fname, filetype='uvh5')
        assert not hasattr(V, 'data')
        V.read(bls=[(24, 25, 'xx')])
        assert hasattr(V, 'data')
        assert hasattr(V, 'antpos')
        assert isinstance(V.hd, io.HERAData)
        assert isinstance(V.hd.data_array, np.ndarray)

        # test basic init w/ uvh5
        fname = os.path.join(DATA_PATH, 'zen.2458098.43124.subband.uvh5')
        V = VisClean(fname, filetype='uvh5')
        assert not hasattr(V, 'data')
        V.read(bls=[(13, 14, 'xx')])
        assert set(V.hd.ant_1_array) == set([13])
        assert isinstance(V.hd, io.HERAData)
        assert isinstance(V.hd.data_array, np.ndarray)

        # test input cal
        fname = os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA')
        uvc = io.HERACal(os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA.abs.calfits'))
        gains, _, _, _ = uvc.read()
        V1 = VisClean(fname, filetype='miriad')
        bl = (52, 53, 'xx')
        V1.read(bls=[bl])
        V2 = VisClean(fname, filetype='miriad', input_cal=uvc)
        V2.read(bls=[bl])
        g = gains[(bl[0], 'Jxx')] * gains[(bl[1], 'Jxx')].conj()
        assert np.allclose((V1.data[bl] / g)[30, 30], V2.data[bl][30, 30])
        V2.apply_calibration(V2.hc, unapply=True)
        assert np.allclose(V1.data[bl][30, 30], V2.data[bl][30, 30], atol=1e-5)

        # test soft copy
        V1.hello = 'hi'
        V1.hello_there = 'bye'
        V1.foo = 'bar'
        V3 = V1.soft_copy(references=["hello*"])
        assert hex(id(V1.data[(52, 53, 'xx')])) == hex(id(V3.data[(52, 53, 'xx')]))
        assert hasattr(V3, 'hello')
        assert hasattr(V3, 'hello_there')
        assert not hasattr(V3, 'foo')
        assert V3.__class__ == VisClean

        # test clear
        V1.clear_containers()
        assert np.all([len(getattr(V1, c)) == 0 for c in ['data', 'flags', 'nsamples']])
        V2.clear_calibration()
        assert not hasattr(V2, 'hc')

    @pytest.mark.filterwarnings("ignore:Selected polarization values are not evenly spaced")
    def test_read_write(self):
        # test read data can be turned off for uvh5
        fname = os.path.join(DATA_PATH, 'zen.2458098.43124.subband.uvh5')
        V = VisClean(fname, filetype='uvh5')
        V.read(read_data=False)
        assert set(V.hd.ant_1_array) == set([1, 11, 12, 13, 14])

        # test read-write-read
        V.read()
        V.write_data(V.data, "./ex.uvh5", overwrite=True, filetype='uvh5', vis_units='Jy')
        V2 = VisClean("./ex.uvh5", filetype='uvh5')
        V2.read()
        assert V2.hd.vis_units == 'Jy'
        assert 'Thisfilewasproducedbythefunction' in V2.hd.history.replace('\n', '').replace(' ', '')
        V.hd.history, V2.hd.history, V2.hd.vis_units = '', '', V.hd.vis_units
        assert V.hd == V2.hd
        os.remove("./ex.uvh5")

        # exceptions
        pytest.raises(ValueError, V.write_data, V.data, 'foo', filetype='what')

        # test write on subset of data
        V.read(read_data=True)
        data = datacontainer.DataContainer(dict([(k, V.data[k]) for k in list(V.data.keys())[:2]]))
        V.write_data(data, "ex.uvh5", overwrite=True, filetype='uvh5')
        assert os.path.exists("ex.uvh5")
        os.remove('ex.uvh5')

    @pytest.mark.filterwarnings("ignore:.*dspec.vis_filter will soon be deprecated")
    def test_vis_clean(self):
        fname = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.XRAA.uvh5")
        V = VisClean(fname, filetype='uvh5')
        V.read()

        # just need to make sure various kwargs run through
        # actual code unit-testing coverage has been done in uvtools.dspec

        # basic freq clean
        V.vis_clean(keys=[(24, 25, 'xx'), (24, 24, 'xx')], ax='freq', overwrite=True)
        assert np.all([i['success'] for i in V.clean_info[(24, 25, 'xx')]])

        # basic time clean
        V.vis_clean(keys=[(24, 25, 'xx'), (24, 24, 'xx')], ax='time', max_frate=10., overwrite=True)
        assert 'skipped' in V.clean_info[(24, 25, 'xx')][0]
        assert 'success' in V.clean_info[(24, 25, 'xx')][3]

        # basic 2d clean
        V.vis_clean(keys=[(24, 25, 'xx'), (24, 24, 'xx')], ax='both', max_frate=10., overwrite=True,
                    filt2d_mode='plus')
        'success' in V.clean_info[(24, 25, 'xx')]

        V.vis_clean(keys=[(24, 25, 'xx'), (24, 24, 'xx')], ax='both', flags=V.flags + True, max_frate=10.,
                    overwrite=True, filt2d_mode='plus')
        assert 'skipped' in V.clean_info[(24, 25, 'xx')]

        # test fft data
        V.vis_clean(keys=[(24, 25, 'xx'), (24, 24, 'xx')], ax='both', max_frate=10., overwrite=True,
                    filt2d_mode='rect')

        # assert foreground peak is at 0 delay bin
        V.fft_data(data=V.clean_model, keys=[(24, 25, 'xx')], ax='freq', window='hann', edgecut_low=10, edgecut_hi=10, overwrite=True)
        assert np.argmax(np.mean(np.abs(V.dfft[(24, 25, 'xx')]), axis=0)) == 32

        # assert foreground peak is at 0 FR bin (just due to FR resolution)
        V.fft_data(data=V.clean_model, keys=[(24, 25, 'xx')], ax='time', window='hann', edgecut_low=10, edgecut_hi=10, overwrite=True)
        assert np.argmax(np.mean(np.abs(V.dfft[(24, 25, 'xx')]), axis=1)) == 30

        # assert foreground peak is at both 0 FR and 0 delay bin
        V.fft_data(data=V.clean_model, keys=[(24, 25, 'xx')], ax='both', window='tukey', alpha=0.5, edgecut_low=10, edgecut_hi=10, overwrite=True)
        assert np.argmax(np.mean(np.abs(V.dfft[(24, 25, 'xx')]), axis=0)) == 32
        assert np.argmax(np.mean(np.abs(V.dfft[(24, 25, 'xx')]), axis=1)) == 30

        # check various kwargs
        V.fft_data(keys=[(24, 25, 'xx')], assign='foo', ifft=True, fftshift=True)
        delays = V.delays
        assert hasattr(V, 'foo')
        V.fft_data(keys=[(24, 25, 'xx')], assign='foo', overwrite=True, ifft=False, fftshift=False)
        np.testing.assert_array_almost_equal(delays, np.fft.fftshift(V.delays))

        # test flag factorization
        flags = V.factorize_flags(inplace=False, time_thresh=0.05)
        assert np.all(flags[(24, 25, 'xx')][45, :])
        assert np.all(flags[(24, 25, 'xx')][:, 5])

    def test_fft_data(self):
        fname = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.XRAA.uvh5")
        V = VisClean(fname, filetype='uvh5')
        V.read()

        # fft
        V.fft_data(zeropad=30, ifft=False)
        assert V.dfft[(24, 25, 'xx')].shape == (60, 124)

        # exceptions
        pytest.raises(ValueError, V.fft_data, ax='foo')
        pytest.raises(ValueError, V.fft_data, keys=[])
        pytest.raises(ValueError, V.fft_data, keys=[('foo')])

    def test_zeropad(self):
        fname = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.XRAA.uvh5")
        V = VisClean(fname, filetype='uvh5')
        V.read()

        # test basic zeropad
        d, _ = vis_clean.zeropad_array(V.data[(24, 25, 'xx')], zeropad=30, axis=-1, undo=False)
        assert d.shape == (60, 124)
        assert np.allclose(d[:, :30], 0.0)
        assert np.allclose(d[:, -30:], 0.0)
        d, _ = vis_clean.zeropad_array(d, zeropad=30, axis=-1, undo=True)
        assert d.shape == (60, 64)

        # test zeropad with bool
        f, _ = vis_clean.zeropad_array(V.flags[(24, 25, 'xx')], zeropad=30, axis=-1, undo=False)
        assert f.shape == (60, 124)
        assert np.all(f[:, :30])
        assert np.all(f[:, -30:])

        # zeropad with binvals
        d, bval = vis_clean.zeropad_array(V.data[(24, 25, 'xx')], zeropad=30, axis=0, binvals=V.times)
        assert np.allclose(np.median(np.diff(V.times)), np.median(np.diff(bval)))
        assert len(bval) == 120

        # 2d zeropad
        d, bval = vis_clean.zeropad_array(V.data[(24, 25, 'xx')], zeropad=(30, 10), axis=(0, 1), binvals=[V.times, V.freqs])
        assert d.shape == (120, 84)
        assert (bval[0].size, bval[1].size) == (120, 84)

        # un-pad with bval
        d, bval = vis_clean.zeropad_array(d, zeropad=(30, 10), axis=(0, 1), binvals=bval, undo=True)
        assert d.shape == (60, 64)
        assert (bval[0].size, bval[1].size) == (60, 64)

        # test VisClean method
        V.zeropad_data(V.data, binvals=V.times, zeropad=10, axis=0, undo=False)
        assert V.data[(24, 25, 'xx')].shape == (80, 64)
        assert V.data.binvals.size == 80

        # exceptions
        pytest.raises(ValueError, vis_clean.zeropad_array, V.data[(24, 25, 'xx')], axis=(0, 1), zeropad=0)
        pytest.raises(ValueError, vis_clean.zeropad_array, V.data[(24, 25, 'xx')], axis=(0, 1), zeropad=(0,))

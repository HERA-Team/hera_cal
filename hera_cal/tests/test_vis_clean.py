# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License
import warnings
import pytest
import numpy as np
from copy import deepcopy
import os
import sys
import shutil
from scipy import constants, interpolate
from pyuvdata import UVCal, UVData
from hera_sim import noise
from uvtools import dspec

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
        V.read(bls=[(24, 25, 'ee')])
        assert hasattr(V, 'data')
        assert hasattr(V, 'antpos')
        assert isinstance(V.hd, io.HERAData)
        assert isinstance(V.hd.data_array, np.ndarray)

        # test basic init w/ uvh5
        fname = os.path.join(DATA_PATH, 'zen.2458098.43124.subband.uvh5')
        V = VisClean(fname, filetype='uvh5')
        assert not hasattr(V, 'data')
        V.read(bls=[(13, 14, 'ee')])
        assert set(V.hd.ant_1_array) == set([13])
        assert isinstance(V.hd, io.HERAData)
        assert isinstance(V.hd.data_array, np.ndarray)

        # test input cal
        fname = os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA')
        uvc = io.HERACal(os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA.abs.calfits'))
        gains, _, _, _ = uvc.read()
        V1 = VisClean(fname, filetype='miriad')
        bl = (52, 53, 'ee')
        V1.read(bls=[bl])
        V2 = VisClean(fname, filetype='miriad', input_cal=uvc)
        V2.read(bls=[bl])
        g = gains[(bl[0], 'Jee')] * gains[(bl[1], 'Jee')].conj()
        assert np.allclose((V1.data[bl] / g)[30, 30], V2.data[bl][30, 30])
        V2.apply_calibration(V2.hc, unapply=True)
        assert np.allclose(V1.data[bl][30, 30], V2.data[bl][30, 30], atol=1e-5)

        # test soft copy
        V1.hello = 'hi'
        V1.hello_there = 'bye'
        V1.foo = 'bar'
        V3 = V1.soft_copy(references=["hello*"])
        assert hex(id(V1.data[(52, 53, 'ee')])) == hex(id(V3.data[(52, 53, 'ee')]))
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
        V.write_data(V.data, "./ex.uvh5", overwrite=True, filetype='uvh5', extra_attrs=dict(vis_units='Jy'))
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

    def test_fourier_filter(self):
        fname = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.XRAA.uvh5")
        V = VisClean(fname, filetype='uvh5')
        V.read()
        # test arg errors
        k = (24, 25, 'ee')
        fc = [0.]
        fw = [100e-9]
        ff = [1e-9]
        fwt = [1e-3]
        assert pytest.raises(ValueError, V.fourier_filter, keys=[k], overwrite=True,
                             filter_centers=fc, filter_half_widths=fw, suppression_factors=ff,
                             ax='height', mode='dayenu', fitting_options=None)
        V.fourier_filter(keys=[k], filter_centers=fc, filter_half_widths=fw, suppression_factors=ff,
                         ax='freq', mode='dayenu', output_prefix='clean', zeropad=10, overwrite=True, max_contiguous_edge_flags=20)
        # this line is repeated to cover the overwrite skip
        V.fourier_filter(keys=[k], filter_centers=fc, filter_half_widths=fw, suppression_factors=ff, max_contiguous_edge_flags=20,
                         ax='freq', mode='dayenu', zeropad=10, output_prefix='clean', overwrite=False)
        assert np.all([V.clean_info[k]['status']['axis_1'][i] == 'success' for i in V.clean_info[k]['status']['axis_1']])
        # now do a time filter
        V.fourier_filter(keys=[k], filter_centers=fc, filter_half_widths=fwt, suppression_factors=ff, overwrite=True,
                         ax='time', mode='dayenu', zeropad=10, max_contiguous_edge_flags=20)
        assert V.clean_info[k]['status']['axis_0'][0] == 'skipped'
        assert V.clean_info[k]['status']['axis_0'][3] == 'success'
        # raise errors.
        assert pytest.raises(ValueError, V.fourier_filter, filter_centers=[fc, fc], ax='both',
                             filter_half_widths=[fwt, fw], suppression_factors=[ff, ff],
                             mode='dayenu', zeropad=0, overwrite=True)
        assert pytest.raises(ValueError, V.fourier_filter, filter_centers=[fc, fc], ax='both',
                             filter_half_widths=[fwt, fw], suppression_factors=[ff, ff], overwrite=True,
                             mode='dayenu', zeropad=['Mathematical Universe', 'Crazy Universe'])
        # check 2d filter.
        V.fourier_filter(filter_centers=[fc, fc],
                         filter_half_widths=[fwt, fw],
                         suppression_factors=[ff, ff],
                         mode='dayenu', overwrite=True,
                         zeropad=[20, 10], ax='both', max_contiguous_edge_flags=100)
        assert V.clean_info[k]['status']['axis_0'][0] == 'skipped'
        assert V.clean_info[k]['status']['axis_0'][3] == 'success'

    @pytest.mark.filterwarnings("ignore:.*dspec.vis_filter will soon be deprecated")
    def test_vis_clean_dayenu(self):
        fname = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.XRAA.uvh5")
        V = VisClean(fname, filetype='uvh5')
        V.read()

        # most coverage is in dspec. Check that args go through here.
        # similar situation for test_vis_clean.
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 25, 'ee')], ax='freq', overwrite=True, mode='dayenu')
        assert np.all([V.clean_info[(24, 25, 'ee')]['status']['axis_1'][i] == 'success' for i in V.clean_info[(24, 25, 'ee')]['status']['axis_1']])

        assert pytest.raises(ValueError, V.vis_clean, keys=[(24, 25, 'ee')], ax='time', mode='dayenu')
        assert pytest.raises(ValueError, V.vis_clean, keys=[(24, 25, 'ee')], ax='time', max_frate='arglebargle', mode='dayenu')

        # cover no overwrite = False skip lines.
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 25, 'ee')], ax='freq', overwrite=False, mode='dayenu')

        V.vis_clean(keys=[(24, 25, 'ee'), (24, 25, 'ee')], ax='time', overwrite=True, max_frate=1.0, mode='dayenu')
        assert V.clean_info[(24, 25, 'ee')]['status']['axis_0'][0] == 'skipped'
        assert V.clean_info[(24, 25, 'ee')]['status']['axis_0'][3] == 'success'

        V.vis_clean(keys=[(24, 25, 'ee'), (24, 25, 'ee')], ax='both', overwrite=True, max_frate=1.0, mode='dayenu')
        assert np.all(['success' == V.clean_info[(24, 25, 'ee')]['status']['axis_1'][i] for i in V.clean_info[(24, 25, 'ee')]['status']['axis_1']])
        assert V.clean_info[(24, 25, 'ee')]['status']['axis_0'][0] == 'skipped'
        assert V.clean_info[(24, 25, 'ee')]['status']['axis_0'][3] == 'success'

        # check whether dayenu filtering axis 1 and then axis 0 is the same as dayenu filtering axis 1 and then filtering the resid.
        # note that filtering axis orders do not commute, we filter axis 1 (foregrounds) before filtering cross-talk.
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 25, 'ee')], ax='both', overwrite=True, max_frate=1.0, mode='dayenu')
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 25, 'ee')], ax='freq', overwrite=True, max_frate=1.0, output_prefix='clean1', mode='dayenu')
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 25, 'ee')], ax='time', overwrite=True, max_frate=1.0, data=V.clean1_resid, output_prefix='clean0', mode='dayenu')
        assert np.all(np.isclose(V.clean_resid[(24, 25, 'ee')], V.clean0_resid[(24, 25, 'ee')]))

    @pytest.mark.filterwarnings("ignore:.*dspec.vis_filter will soon be deprecated")
    def test_vis_clean(self):
        fname = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.XRAA.uvh5")
        V = VisClean(fname, filetype='uvh5')
        V.read()

        # just need to make sure various kwargs run through
        # actual code unit-testing coverage has been done in uvtools.dspec

        # basic freq clean
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='freq', overwrite=True)
        assert np.all([V.clean_info[(24, 25, 'ee')]['status']['axis_1'][i] == 'success' for i in V.clean_info[(24, 25, 'ee')]['status']['axis_1']])

        # basic time clean
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='time', max_frate=10., overwrite=True)
        assert 'skipped' == V.clean_info[(24, 25, 'ee')]['status']['axis_0'][0]
        assert 'success' == V.clean_info[(24, 25, 'ee')]['status']['axis_0'][3]

        # basic 2d clean
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='both', max_frate=10., overwrite=True,
                    filt2d_mode='plus')
        assert np.all(['success' == V.clean_info[(24, 25, 'ee')]['status']['axis_0'][i] for i in V.clean_info[(24, 25, 'ee')]['status']['axis_0']])
        assert np.all(['success' == V.clean_info[(24, 25, 'ee')]['status']['axis_1'][i] for i in V.clean_info[(24, 25, 'ee')]['status']['axis_1']])

        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='both', flags=V.flags + True, max_frate=10.,
                    overwrite=True, filt2d_mode='plus')
        assert np.all([V.clean_info[(24, 25, 'ee')]['status']['axis_1'][i] == 'skipped' for i in V.clean_info[(24, 25, 'ee')]['status']['axis_1']])
        assert np.all([V.clean_info[(24, 25, 'ee')]['status']['axis_0'][i] == 'skipped' for i in V.clean_info[(24, 25, 'ee')]['status']['axis_0']])

        # test fft data
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='both', max_frate=10., overwrite=True,
                    filt2d_mode='rect')

        # assert foreground peak is at 0 delay bin
        V.fft_data(data=V.clean_model, keys=[(24, 25, 'ee')], ax='freq', window='hann', edgecut_low=10, edgecut_hi=10, overwrite=True)
        assert np.argmax(np.mean(np.abs(V.dfft[(24, 25, 'ee')]), axis=0)) == 32

        # assert foreground peak is at 0 FR bin (just due to FR resolution)
        V.fft_data(data=V.clean_model, keys=[(24, 25, 'ee')], ax='time', window='hann', edgecut_low=10, edgecut_hi=10, overwrite=True)
        assert np.argmax(np.mean(np.abs(V.dfft[(24, 25, 'ee')]), axis=1)) == 30

        # assert foreground peak is at both 0 FR and 0 delay bin
        V.fft_data(data=V.clean_model, keys=[(24, 25, 'ee')], ax='both', window='tukey', alpha=0.5, edgecut_low=10, edgecut_hi=10, overwrite=True)
        assert np.argmax(np.mean(np.abs(V.dfft[(24, 25, 'ee')]), axis=0)) == 32
        assert np.argmax(np.mean(np.abs(V.dfft[(24, 25, 'ee')]), axis=1)) == 30

        # check various kwargs
        V.fft_data(keys=[(24, 25, 'ee')], assign='foo', ifft=True, fftshift=True)
        delays = V.delays
        assert hasattr(V, 'foo')
        V.fft_data(keys=[(24, 25, 'ee')], assign='foo', overwrite=True, ifft=False, fftshift=False)
        np.testing.assert_array_almost_equal(delays, np.fft.fftshift(V.delays))

        # test flag factorization
        flags = V.factorize_flags(inplace=False, time_thresh=0.05)
        assert np.all(flags[(24, 25, 'ee')][45, :])
        assert np.all(flags[(24, 25, 'ee')][:, 5])

    def test_fft_data(self):
        fname = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.XRAA.uvh5")
        V = VisClean(fname, filetype='uvh5')
        V.read()

        # fft
        V.fft_data(zeropad=30, ifft=False)
        assert V.dfft[(24, 25, 'ee')].shape == (60, 124)

        # exceptions
        pytest.raises(ValueError, V.fft_data, ax='foo')
        pytest.raises(ValueError, V.fft_data, keys=[])
        pytest.raises(ValueError, V.fft_data, keys=[('foo')])

    # THIS UNIT TEST IS BROKEN!!!
    # See https://github.com/HERA-Team/hera_cal/issues/603
    def test_trim_model(self):
        # load data
        V = VisClean(os.path.join(DATA_PATH, "PyGSM_Jy_downselect.uvh5"))
        V.read(bls=[(23, 23, 'ee'), (23, 24, 'ee')])

        # interpolate to 768 frequencies
        freqs = np.linspace(120e6, 180e6, 768)
        for k in V.data:
            V.data[k] = interpolate.interp1d(V.freqs, V.data[k], axis=1, fill_value='extrapolate', kind='cubic')(freqs)
            V.flags[k] = np.zeros_like(V.data[k], dtype=np.bool)
        V.freqs = freqs
        # the old unit test was using the wrong dnu (for the original frequencies) which means that it was actually cleaning
        # out to 1250 ns. I've fixed this dnu bug and used a larger min_dly below.
        V.Nfreqs = len(V.freqs)
        # dnu should have also been set here to be np.diff(np.median(freqs))
        # but it wasn't Because of this, the old version of vis_clean was cleaning with a
        # delay width = intended delay width x (manually set dnu / original dnu of the attached data)
        np.random.seed(0)
        k = (23, 24, 'ee')
        Op = noise.bm_poly_to_omega_p(V.freqs / 1e9)
        V.data[k] += noise.sky_noise_jy(V.data[(23, 23, 'ee')], V.freqs / 1e9, V.lsts, Op, inttime=50)

        # add lots of random flags
        f = np.zeros(V.Nfreqs, dtype=np.bool)[None, :]
        f[:, 127:156] = True
        f[:, 300:303] = True
        f[:, 450:455] = True
        f[:, 625:630] = True
        V.flags[k] += f
        # Note that the intended delay width of this unit test was 300 ns but because of the dnu bug, the delay width was
        # actuall 300 x V.dnu / np.mean(np.diff(V.freqs))
        # the new vis_clean never explicitly references V.dnu so it doesn't have problems and uses the correct delay width.
        # however, using the correct delay width causes this unit test to fail.
        # so we need to fix it. SEP (Somebody Elses PR).
        V.vis_clean(data=V.data, flags=V.flags, keys=[k], tol=1e-6, min_dly=300. * (V.dnu / np.mean(np.diff(V.freqs))), ax='freq', overwrite=True, window='tukey', alpha=0.2)
        V.fft_data(V.data, window='bh', overwrite=True, assign='dfft1')
        V.fft_data(V.clean_data, window='bh', overwrite=True, assign='dfft2')

        # trim model
        mdl, n = vis_clean.trim_model(V.clean_model, V.clean_resid, V.dnu, noise_thresh=3.0, delay_cut=500,
                                      kernel_size=21, polyfit_deg=None)
        clean_data2 = deepcopy(V.clean_data)
        clean_data2[k][V.flags[k]] = mdl[k][V.flags[k]]
        V.fft_data(clean_data2, window='bh', overwrite=True, assign='dfft3')

        # get averaged spectra
        n1 = vis_clean.noise_eq_bandwidth(dspec.gen_window('bh', V.Nfreqs))
        n2 = vis_clean.noise_eq_bandwidth(dspec.gen_window('bh', V.Nfreqs) * ~V.flags[k][0])
        d1 = np.mean(np.abs(V.dfft1[k]), axis=0) * n1
        d2 = np.mean(np.abs(V.dfft2[k]), axis=0) * n2
        d3 = np.mean(np.abs(V.dfft3[k]), axis=0) * n2

        # confirm that dfft3 and dfft1 match while dfft2 and dfft1 do not near CLEAN boundary
        select = (np.abs(V.delays) < 300) & (np.abs(V.delays) > 100)
        assert np.isclose(np.mean(np.abs(d1)[select]), np.mean(np.abs(d3)[select]), atol=10)
        assert not np.isclose(np.mean(np.abs(d1)[select]), np.mean(np.abs(d2)[select]), atol=10)

        # test that polynomial fitting is a good fit
        _, n1 = vis_clean.trim_model(V.clean_model, V.clean_resid, V.dnu, noise_thresh=3.0, delay_cut=500,
                                     kernel_size=None, polyfit_deg=None)
        _, n2 = vis_clean.trim_model(V.clean_model, V.clean_resid, V.dnu, noise_thresh=3.0, delay_cut=500,
                                     kernel_size=None, polyfit_deg=5)
        assert (np.std(n1[k] - n2[k]) / np.mean(n2[k])) < 0.1  # assert residual is below 10% of fit

        # test well-conditioned check takes effect
        V2 = deepcopy(V)
        V2.clean_resid[k][:-2] = 0.0  # zero all the data except last two integrations
        _, n2 = vis_clean.trim_model(V2.clean_model, V2.clean_resid, V2.dnu, noise_thresh=3.0, delay_cut=500,
                                     kernel_size=None, polyfit_deg=5)
        assert np.all(np.isclose(n2[k][-1], n1[k][-1]))  # assert non-zeroed output are same as n1 (no polyfit)

    def test_neb(self):
        n = vis_clean.noise_eq_bandwidth(dspec.gen_window('blackmanharris', 10000))
        assert np.isclose(n, 1.9689862471203075)

    def test_zeropad(self):
        fname = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.XRAA.uvh5")
        V = VisClean(fname, filetype='uvh5')
        V.read()

        # test basic zeropad
        d, _ = vis_clean.zeropad_array(V.data[(24, 25, 'ee')], zeropad=30, axis=-1, undo=False)
        assert d.shape == (60, 124)
        assert np.allclose(d[:, :30], 0.0)
        assert np.allclose(d[:, -30:], 0.0)
        d, _ = vis_clean.zeropad_array(d, zeropad=30, axis=-1, undo=True)
        assert d.shape == (60, 64)

        # test zeropad with bool
        f, _ = vis_clean.zeropad_array(V.flags[(24, 25, 'ee')], zeropad=30, axis=-1, undo=False)
        assert f.shape == (60, 124)
        assert np.all(f[:, :30])
        assert np.all(f[:, -30:])

        # zeropad with binvals
        d, bval = vis_clean.zeropad_array(V.data[(24, 25, 'ee')], zeropad=30, axis=0, binvals=V.times)
        assert np.allclose(np.median(np.diff(V.times)), np.median(np.diff(bval)))
        assert len(bval) == 120

        # 2d zeropad
        d, bval = vis_clean.zeropad_array(V.data[(24, 25, 'ee')], zeropad=(30, 10), axis=(0, 1), binvals=[V.times, V.freqs])
        assert d.shape == (120, 84)
        assert (bval[0].size, bval[1].size) == (120, 84)

        # un-pad with bval
        d, bval = vis_clean.zeropad_array(d, zeropad=(30, 10), axis=(0, 1), binvals=bval, undo=True)
        assert d.shape == (60, 64)
        assert (bval[0].size, bval[1].size) == (60, 64)

        # test VisClean method
        V.zeropad_data(V.data, binvals=V.times, zeropad=10, axis=0, undo=False)
        assert V.data[(24, 25, 'ee')].shape == (80, 64)
        assert V.data.binvals.size == 80

        # exceptions
        pytest.raises(ValueError, vis_clean.zeropad_array, V.data[(24, 25, 'ee')], axis=(0, 1), zeropad=0)
        pytest.raises(ValueError, vis_clean.zeropad_array, V.data[(24, 25, 'ee')], axis=(0, 1), zeropad=(0,))

    def test_filter_argparser(self):
        sys.argv = [sys.argv[0], 'a', '--clobber', '--spw_range', '0', '20']
        parser = vis_clean._filter_argparser()
        a = parser.parse_args()
        assert a.infilename == 'a'
        assert a.clobber is True
        assert a.spw_range[0] == 0
        assert a.spw_range[1] == 20

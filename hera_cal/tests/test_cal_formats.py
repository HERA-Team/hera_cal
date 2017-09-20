import nose.tools as nt
import os
import numpy as np
from pyuvdata import UVCal
from hera_cal import cal_formats, omni
from hera_cal.data import DATA_PATH


class Test_HERACal(UVCal):
    def test_gainHC(self):
        fn = os.path.join(DATA_PATH, 'test_input', 'zen.2457698.40355.xx.HH.uvc.omni.calfits')
        meta, gains, vis, xtalk = omni.from_fits(fn)
        meta['inttime'] = np.diff(meta['times'])[0] * 60 * 60 * 24
        optional = {'observer': 'heracal'} #because it's easier than changing the fits header
        hc = cal_formats.HERACal(meta, gains, optional=optional)
        uv = UVCal()
        uv.read_calfits(os.path.join(
            DATA_PATH, 'test_input', 'zen.2457698.40355.xx.HH.uvc.omni.calfits'))
        for param in hc:
            if param == '_history':
                continue
            elif param == '_time_range':  # why do we need this?
                nt.assert_equal(np.testing.assert_almost_equal(
                    getattr(hc, param).value, getattr(uv, param).value, 5), None)
            else:
                nt.assert_true(np.all(getattr(hc, param) == getattr(uv, param)))

    def test_delayHC(self):
        # make test data
        meta, gains, vis, xtalk = omni.from_fits(os.path.join(
            DATA_PATH, 'test_input', 'zen.2457698.40355.xx.HH.uvc.first.calfits'), keep_delay=True)
        for pol in gains.keys():
            for k in gains[pol].keys():
                gains[pol][k] = gains[pol][k].reshape(-1, 1)
        meta['inttime'] = np.diff(meta['times'])[0] * 60 * 60 * 24
        meta.pop('chisq9x')
        optional = {'observer': 'Zaki Ali (zakiali@berkeley.edu)'}
        hc = cal_formats.HERACal(meta, gains, optional=optional, DELAY=True)
        uv = UVCal()
        uv.read_calfits(os.path.join(
            DATA_PATH, 'test_input', 'zen.2457698.40355.xx.HH.uvc.first.calfits'))
        for param in hc:
            print param
            print getattr(hc, param).value, getattr(uv, param).value
            if param == '_history':
                continue
            elif param == '_git_hash_cal':
                continue
            elif param == '_git_origin_cal':
                continue
            elif param == '_time_range':  # why do we need this?
                nt.assert_equal(np.testing.assert_almost_equal(
                    getattr(hc, param).value, getattr(uv, param).value, 5), None)
            else:
                nt.assert_true(
                    np.all(getattr(hc, param) == getattr(uv, param)))


class Test_AbsCal(UVCal):
    def test_AbsCal_from_gains_linpol(self):
        '''
        Test generating an AbsCal object for a linearly polarized visibility
        '''
        miriad_fn = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcA')
        abscal_fn = os.path.join(DATA_PATH, 'test_input', 'abscal_gains_x.npz')
        polname = 'xx'
        gain_convention = 'multiply'
        append2hist = "  Absolutely calibrated with the galactic center. "
        ac = cal_formats.AbsCal(miriad_fn, [abscal_fn], polname, gain_convention,
                    append2hist=append2hist)
        nt.assert_equal(ac.Nants_data, 19)

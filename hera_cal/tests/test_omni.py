'''Tests for omni.py'''

import nose.tools as nt
import os
import sys
import numpy as np
import optparse
import shutil
import re
from copy import deepcopy
import aipy
from omnical.calib import RedundantInfo
from pyuvdata import UVCal, UVData, UVFITS
import hera_cal.omni as omni
from hera_cal.data import DATA_PATH
from hera_cal.calibrations import CAL_PATH
import hera_cal.redcal as rc


class AntennaArray(aipy.fit.AntennaArray):

    def __init__(self, *args, **kwargs):
        aipy.fit.AntennaArray.__init__(self, *args, **kwargs)
        self.antpos_ideal = kwargs.pop('antpos_ideal')


def get_aa(freqs, nants=4):
    lat = "45:00"
    lon = "90:00"
    beam = aipy.fit.Beam(freqs)
    ants = []
    for i in range(nants):
        ants.append(aipy.fit.Antenna(0, 50 * i, 0, beam))
    antpos_ideal = np.array([ant.pos for ant in ants])
    aa = AntennaArray((lat, lon), ants, antpos_ideal=antpos_ideal)
    return aa


class TestMethods(object):

    def setUp(self):
        """Set up for basic tests of antenna array to info object."""
        self.freqs = np.linspace(.1, .2, 16)
        self.pols = ['x', 'y']
        self.aa = get_aa(self.freqs)
        self.info = omni.aa_to_info(self.aa, pols=self.pols)
        self.gains = {pol: {ant: np.ones((1, self.freqs.size)) for ant in range(
            self.info.nant)} for pol in self.pols}

    def test_aa_to_info(self):
        info = omni.aa_to_info(self.aa)
        reds = [[(0, 1), (1, 2), (2, 3)], [(0, 2), (1, 3)]]
        nt.assert_true(np.all(info.subsetant == np.array([0, 1, 2, 3])))
        for rb in info.get_reds():
            nt.assert_true(rb in reds)

        info = omni.aa_to_info(self.aa, fcal=True)
        reds = [[(0, 1), (1, 2), (2, 3)], [(0, 2), (1, 3)]]
        nt.assert_true(np.all(info.subsetant == np.array([0, 1, 2, 3])))
        for rb in info.get_reds():
            nt.assert_true(rb in reds)

        aa_antlayout = deepcopy(self.aa)
        del(aa_antlayout.antpos_ideal)
        aa_antlayout.ant_layout = np.array([[0, 1, 2, 3]])
        info = omni.aa_to_info(aa_antlayout)
        reds = [[(0, 1), (1, 2), (2, 3)], [(0, 2), (1, 3)]]
        nt.assert_true(np.all(info.subsetant == np.array([0, 1, 2, 3])))
        for rb in info.get_reds():
            nt.assert_true(rb in reds)

    def test_filter_reds(self):
        # exclude ants
        reds = omni.filter_reds(self.info.get_reds(), ex_ants=[0, 4])
        nt.assert_equal(reds, [[(1, 2), (2, 3)], [(1, 6), (2, 7)], [
                        (5, 2), (6, 3)], [(5, 6), (6, 7)]])
        # include ants
        reds = omni.filter_reds(self.info.get_reds(), ants=[0, 1, 4, 5, 6])
        nt.assert_equal(reds, [[(0, 5), (1, 6)], [(4, 5), (5, 6)]])
        # exclued bls
        reds = omni.filter_reds(self.info.get_reds(), ex_bls=[(0, 2), (1, 2)])
        nt.assert_equal(reds, [[(0, 1), (2, 3)], [(0, 6), (1, 7)], [(0, 5), (1, 6), (2, 7)],
                               [(4, 2), (5, 3)], [
            (4, 1), (5, 2), (6, 3)], [(4, 6), (5, 7)],
            [(4, 5), (5, 6), (6, 7)]])
        # include bls
        reds = omni.filter_reds(self.info.get_reds(), bls=[(0, 2), (1, 2)])
        nt.assert_equal(reds, [])
        # include ubls
        reds = omni.filter_reds(self.info.get_reds(), ubls=[(0, 2), (1, 4)])
        nt.assert_equal(reds, [[(0, 2), (1, 3)], [(4, 1), (5, 2), (6, 3)]])
        # exclude ubls
        reds = omni.filter_reds(self.info.get_reds(), ex_ubls=[
                                (0, 2), (1, 4), (4, 5), (0, 5), (2, 3)])
        nt.assert_equal(
            reds, [[(0, 6), (1, 7)], [(4, 2), (5, 3)], [(4, 6), (5, 7)]])
        # exclude crosspols
        # reds = omni.filter_reds(self.info.get_reds(), ex_crosspols=()

    def test_compute_reds(self):
        reds = omni.compute_reds(
            4, self.pols, self.info.antloc[:self.info.nant])
        for i in reds:
            for k in i:
                for l in k:
                    nt.assert_true(isinstance(l, omni.Antpol))

    def test_reds_for_minimal_V(self):
        reds = omni.compute_reds(
            4, self.pols, self.info.antloc[:self.info.nant])
        mVreds = omni.reds_for_minimal_V(reds)
        # test that the new reds array is shorter by 1/4, as expected
        nt.assert_equal(len(mVreds), len(reds) - len(reds) / 4)
        # test that we haven't invented or lost baselines
        cr, cmv = 0, 0
        for arr in reds:
            cr += len(arr)
        for arr in mVreds:
            cmv += len(arr)
        nt.assert_equal(cr, cmv)
        # test that no crosspols are in linpol red arrays and vice versa
        for arr in mVreds:
            p0 = arr[0][0].pol() + arr[0][1].pol()
            for ap in arr:
                p = ap[0].pol() + ap[1].pol()
                nt.assert_equal(len(set(p)), len(set(p0)))
        # test that every xy has its corresponding yx in the array
        for arr in mVreds:
            p0 = arr[0][0].pol() + arr[0][1].pol()
            if len(set(p0)) == 1:
                continue  # not interested in linpols
            for ap in arr:
                ai, aj = ap
                bi, bj = omni.Antpol(ai.ant(), aj.pol(), self.info.nant), omni.Antpol(
                    aj.ant(), ai.pol(), self.info.nant)
                nt.assert_true(((bi, bj) in arr) or ((bj, bi) in arr))
        # test AssertionError
        _reds = reds[:-1]
        nt.assert_raises(ValueError, omni.reds_for_minimal_V, _reds)

    def test_from_npz(self):
        Ntimes = 3
        Nchans = 1024  # hardcoded for this file
        meta, gains, vis, xtalk = omni.from_npz(os.path.join(
            DATA_PATH, 'test_input', 'zen.2457698.50098.xx.pulledtime.npz'))
        for m in meta.keys():
            if m.startswith('chisq'):
                nt.assert_equal(meta[m].shape, (Ntimes, Nchans))
        nt.assert_equal(len(meta['freqs']), Nchans)
        nt.assert_equal(len(meta['jds']), Ntimes)
        nt.assert_equal(len(meta['lsts']), Ntimes)

        nt.assert_equal(gains.keys(), ['x'])
        for ant in gains['x'].keys():
            nt.assert_equal(gains['x'][ant].dtype, np.complex64)
            nt.assert_equal(gains['x'][ant].shape, (Ntimes, Nchans))

        nt.assert_equal(vis.keys(), ['xx'])
        for bl in vis['xx'].keys():
            nt.assert_equal(vis['xx'][bl].dtype, np.complex64)
            nt.assert_equal(vis['xx'][bl].shape, (Ntimes, Nchans))

        nt.assert_equal(xtalk.keys(), ['xx'])
        for bl in xtalk['xx'].keys():
            nt.assert_equal(xtalk['xx'][bl].dtype, np.complex64)
            nt.assert_equal(xtalk['xx'][bl].shape, (Ntimes, Nchans))
            for time in range(Ntimes):
                nt.assert_true(
                    np.all(xtalk['xx'][bl][0] == xtalk['xx'][bl][time]))

    def test_get_phase(self):
        freqs = np.linspace(.1, .2, 1024).reshape(-1, 1)  # GHz
        tau = 10  # ns
        nt.assert_true(np.all(omni.get_phase(freqs, tau) ==
                              np.exp(-2j * np.pi * freqs * tau)))

    def test_from_fits_gain(self):
        Ntimes = 3 * 2  # need 2 here because reading two files
        Nchans = 1024  # hardcoded for this file
        # read in the same file twice to make sure file concatenation works
        fn = os.path.join(DATA_PATH, 'test_input', 'zen.2457698.40355.xx.HH.uvc.omni.calfits')
        meta, gains, vis, xtalk = omni.from_fits([fn, fn])
        for m in meta.keys():
            if m.startswith('chisq'):
                nt.assert_equal(meta[m].shape, (Ntimes, Nchans))
        nt.assert_equal(len(meta['freqs']), Nchans)
        nt.assert_equal(len(meta['times']), Ntimes)
        nt.assert_equal(type(meta['history']), str)
        nt.assert_equal(meta['gain_conventions'], 'divide')

        nt.assert_equal(gains.keys(), ['x'])
        for ant in gains['x'].keys():
            nt.assert_equal(gains['x'][ant].dtype, np.complex64)
            nt.assert_equal(gains['x'][ant].shape, (Ntimes, Nchans))

        nt.assert_equal(vis.keys(), ['xx'])
        for bl in vis['xx'].keys():
            nt.assert_equal(vis['xx'][bl].dtype, np.complex64)
            nt.assert_equal(vis['xx'][bl].shape, (Ntimes, Nchans))

        nt.assert_equal(xtalk.keys(), ['xx'])
        for bl in xtalk['xx'].keys():
            nt.assert_equal(xtalk['xx'][bl].dtype, np.complex64)
            nt.assert_equal(xtalk['xx'][bl].shape, (Ntimes, Nchans))
            for time in range(Ntimes):
                nt.assert_true(
                    np.all(xtalk['xx'][bl][0] == xtalk['xx'][bl][time]))

        pol2str = {-5: 'x', -6: 'y'}
        uvcal = UVCal()
        uvcal.read_calfits(os.path.join(
            DATA_PATH, 'test_input', 'zen.2457698.40355.xx.HH.uvc.omni.calfits'))
        np.testing.assert_equal(uvcal.freq_array.flatten(), meta['freqs'])
        # need repeat here because reading 2 files.
        np.testing.assert_equal(
            np.resize(uvcal.time_array, (Ntimes,)), meta['times'])
        nt.assert_equal(uvcal.history, meta['history'])
        nt.assert_equal(uvcal.gain_convention, meta['gain_conventions'])
        for ai, ant in enumerate(uvcal.ant_array):
            for ip, pol in enumerate(uvcal.jones_array):
                for nsp in range(uvcal.Nspws):
                    np.testing.assert_equal(np.resize(uvcal.gain_array[
                                            ai, nsp, :, :, ip].T, (Ntimes, Nchans)),  gains[pol2str[pol]][ant])

        str2pol = {'xx': -5, 'yy': -6}
        uvd = UVData()
        fn = os.path.join(DATA_PATH, 'test_input', 'zen.2457698.40355.xx.HH.uvc.vis.uvfits')
        uvd.read_uvfits(fn)
        # from_fits turns data into drift
        uvd.unphase_to_drift()
        for pol in vis:
            for i, j in vis[pol]:
                uvpol = list(uvd.polarization_array).index(str2pol[pol])
                uvmask = np.all(
                    np.array(zip(uvd.ant_1_array, uvd.ant_2_array)) == [i, j], axis=1)
                # need to resize because test is reading in 2 files with
                # from_fits.
                np.testing.assert_equal(vis[pol][i, j], np.resize(
                    uvd.data_array[uvmask][:, 0, :, uvpol], vis[pol][i, j].shape))

        uvd = UVData()
        fn = os.path.join(DATA_PATH, 'test_input', 'zen.2457698.40355.xx.HH.uvc.xtalk.uvfits')
        uvd.read_uvfits(fn)
        # from_fits turns data into drift
        uvd.unphase_to_drift()
        for pol in xtalk:
            for i, j in xtalk[pol]:
                uvpol = list(uvd.polarization_array).index(str2pol[pol])
                uvmask = np.all(
                    np.array(zip(uvd.ant_1_array, uvd.ant_2_array)) == [i, j], axis=1)
                # need to resize because test is reading in 2 files with
                # from_fits.
                np.testing.assert_equal(xtalk[pol][i, j], np.resize(
                    uvd.data_array[uvmask][:, 0, :, uvpol], xtalk[pol][i, j].shape))

    def test_from_fits_delay(self):
        Ntimes = 3 * 2  # need 2 here because reading two files
        Nchans = 1024  # hardcoded for this file
        Ndelay = 1  # number of delays per integration
        # read in the same file twice to make sure file concatenation works
        meta, gains, vis, xtalk = omni.from_fits(
            [os.path.join(DATA_PATH, 'test_input', 'zen.2457698.40355.xx.HH.uvc.first.calfits')] * 2)
        for m in meta.keys():
            if m.startswith('chisq'):
                print(m)
                nt.assert_equal(meta[m].shape, (Ntimes,))
        nt.assert_equal(len(meta['freqs']), Nchans)
        nt.assert_equal(len(meta['times']), Ntimes)
        nt.assert_equal(type(meta['history']), str)
        nt.assert_equal(meta['gain_conventions'], 'divide')

        nt.assert_equal(gains.keys(), ['x'])
        for ant in gains['x'].keys():
            nt.assert_equal(gains['x'][ant].dtype, np.complex128)
            nt.assert_equal(gains['x'][ant].shape, (Ntimes, Nchans))

        nt.assert_equal(vis, {})

        nt.assert_equal(xtalk, {})

        pol2str = {-5: 'x', -6: 'y'}
        uvcal = UVCal()
        uvcal.read_calfits(os.path.join(
            DATA_PATH, 'test_input', 'zen.2457698.40355.xx.HH.uvc.first.calfits'))
        np.testing.assert_equal(uvcal.freq_array.flatten(), meta['freqs'])
        # need repeat here because reading 2 files.
        np.testing.assert_equal(
            np.resize(uvcal.time_array, (Ntimes,)), meta['times'])
        nt.assert_equal(uvcal.history, meta['history'])
        nt.assert_equal(uvcal.gain_convention, meta['gain_conventions'])
        for ai, ant in enumerate(uvcal.ant_array):
            for ip, pol in enumerate(uvcal.jones_array):
                for nsp in range(uvcal.Nspws):
                    np.testing.assert_equal(np.resize(omni.get_phase(uvcal.freq_array, uvcal.delay_array[
                                            ai, nsp, 0, :, ip]).T, (Ntimes, Nchans)),  gains[pol2str[pol]][ant])

        # Now test if we keep delays
        meta, gains, vis, xtalk = omni.from_fits([os.path.join(
            DATA_PATH, 'test_input', 'zen.2457698.40355.xx.HH.uvc.first.calfits')] * 2, keep_delay=True)
        pol2str = {-5: 'x', -6: 'y'}
        uvcal = UVCal()
        uvcal.read_calfits(os.path.join(
            DATA_PATH, 'test_input', 'zen.2457698.40355.xx.HH.uvc.first.calfits'))
        np.testing.assert_equal(uvcal.freq_array.flatten(), meta['freqs'])
        # need repeat here because reading 2 files.
        np.testing.assert_equal(
            np.resize(uvcal.time_array, (Ntimes,)), meta['times'])
        nt.assert_equal(uvcal.history, meta['history'])
        nt.assert_equal(uvcal.gain_convention, meta['gain_conventions'])
        for ai, ant in enumerate(uvcal.ant_array):
            for ip, pol in enumerate(uvcal.jones_array):
                for nsp in range(uvcal.Nspws):
                    np.testing.assert_equal(np.resize(
                        uvcal.delay_array[ai, nsp, 0, :, ip].T, (Ntimes,)),  gains[pol2str[pol]][ant])

    def test_from_fits_gain_select(self):
        Ntimes = 3
        Nchans = 1024  # hardcoded for this file
        # read in the same file twice to make sure file concatenation works
        fn = os.path.join(DATA_PATH, 'test_input', 'zen.2457698.40355.xx.HH.uvc.omni.calfits')
        meta, gains, vis, xtalk = omni.from_fits(fn, antenna_nums=[9, 10, 112, 20, 22])

        pol2str = {-5: 'x', -6: 'y'}
        uvcal = UVCal()
        uvcal.read_calfits(os.path.join(
            DATA_PATH, 'test_input', 'zen.2457698.40355.xx.HH.uvc.omni.calfits'))
        uvcal.select(antenna_nums=[9, 10, 112, 20, 22])
        np.testing.assert_equal(uvcal.freq_array.flatten(), meta['freqs'])
        # need repeat here because reading 2 files.
        np.testing.assert_equal(
            np.resize(uvcal.time_array, (Ntimes,)), meta['times'])
        nt.assert_equal(uvcal.history, meta['history'])
        nt.assert_equal(uvcal.gain_convention, meta['gain_conventions'])
        for ai, ant in enumerate(uvcal.ant_array):
            for ip, pol in enumerate(uvcal.jones_array):
                for nsp in range(uvcal.Nspws):
                    np.testing.assert_equal(np.resize(uvcal.gain_array[
                                            ai, nsp, :, :, ip].T, (Ntimes, Nchans)),  gains[pol2str[pol]][ant])

    def test_from_fits_catch_errors(self):
        # raise error on caltype
        args = [os.path.join(DATA_PATH, 'test_input', 'zen.2457698.40355.xx.HH.uvc.omni.calfits'), os.path.join(
            DATA_PATH, 'test_input', 'zen.2457698.40355.xx.HH.uvc.first.calfits')]
        nt.assert_raises(ValueError, omni.from_fits, args)

        # raise error on gain convention
        uvc = UVCal()
        uvc.read_calfits(args[1])
        uvc.gain_convention = 'multiply'
        uvc.write_calfits(os.path.join(
            DATA_PATH, 'test_output', 'test_from_fits_errors.fits'), clobber=True)
        new_args = [os.path.join(
            DATA_PATH, 'test_output', 'test_from_fits_errors.fits'), args[1]]
        nt.assert_raises(ValueError, omni.from_fits, new_args)

        # raise error on inttime
        uvc = UVCal()
        uvc.read_calfits(args[1])
        uvc.integration_time = 3.145
        uvc.write_calfits(os.path.join(
            DATA_PATH, 'test_output', 'test_from_fits_errors.fits'), clobber=True)
        new_args = [os.path.join(
            DATA_PATH, 'test_output', 'test_from_fits_errors.fits'), args[1]]
        nt.assert_raises(ValueError, omni.from_fits, new_args)

        # raise error on freqs
        uvc = UVCal()
        uvc.read_calfits(args[1])
        uvc.freq_array += 1e4
        uvc.write_calfits(os.path.join(
            DATA_PATH, 'test_output', 'test_from_fits_errors.fits'), clobber=True)
        new_args = [os.path.join(
            DATA_PATH, 'test_output', 'test_from_fits_errors.fits'), args[1]]
        nt.assert_raises(ValueError, omni.from_fits, new_args)

    def test_make_uvdata_vis(self):
        # append data_path to path so we can find calfile.
        if DATA_PATH not in sys.path:
            sys.path.append(DATA_PATH)
        # This aa is specific for the fits file below.
        aa = aipy.cal.get_aa('heratest_calfile', np.array([.15]))
        sys.path[:-1]  # remove last entry from path (DATA_PATH)

        # read in meta, gains, vis, xtalk from file.
        fn = os.path.join(DATA_PATH, 'test_input', 'zen.2457698.40355.xx.HH.uvc.omni.calfits')
        meta, gains, vis, xtalk = omni.from_fits(fn)
        _xtalk = {}
        # overide xtalk to have single visibility. from fits expands to size of
        # vis data.
        for pol in xtalk.keys():
            _xtalk[pol] = {key: xtalk[pol][key][0, :]
                           for key in xtalk[pol].keys()}
        # write to new file for both vis and xtalk
        uv = omni.make_uvdata_vis(aa, meta, vis)
        uv.write_uvfits(os.path.join(DATA_PATH, 'test_output', 'write_vis_test.fits'),
                        force_phase=True, spoof_nonessential=True)
        uv = omni.make_uvdata_vis(aa, meta, _xtalk, xtalk=True)
        uv.write_uvfits(os.path.join(DATA_PATH, 'test_output', 'write_xtalk_test.fits'),
                        force_phase=True, spoof_nonessential=True)

        # read in old and newly written files and check equality.
        uv_vis_in = UVData()
        fn = os.path.join(DATA_PATH, 'test_input', 'zen.2457698.40355.xx.HH.uvc.vis.uvfits')
        uv_vis_in.read_uvfits(fn)
        uv_vis_in.unphase_to_drift()
        # overwrite history because uvdata writes git stuff whenever data is
        # written to a file.
        uv_vis_in.history = 'test_history'

        uv_xtalk_in = UVData()
        fn = os.path.join(DATA_PATH, 'test_input', 'zen.2457698.40355.xx.HH.uvc.xtalk.uvfits')
        uv_xtalk_in.read_uvfits(fn)
        uv_xtalk_in.unphase_to_drift()
        # overwrite history because uvdata writes git stuff whenever data is
        # written to a file.
        uv_xtalk_in.history = 'test_history'

        uv_vis_out = UVData()
        fn = os.path.join(DATA_PATH, 'test_output', 'write_vis_test.fits')
        uv_vis_out.read_uvfits(fn)
        uv_vis_out.unphase_to_drift()
        # overwrite history because uvdata writes git stuff whenever data is
        # written to a file.
        uv_vis_out.history = 'test_history'

        uv_xtalk_out = UVData()
        fn = os.path.join(DATA_PATH, 'test_output', 'write_xtalk_test.fits')
        uv_xtalk_out.read_uvfits(fn)
        uv_xtalk_out.unphase_to_drift()
        # overwrite history because uvdata writes git stuff whenever data is
        # written to a file.
        uv_xtalk_out.history = 'test_history'

        nt.assert_equal(uv_vis_in, uv_vis_out)
        nt.assert_equal(uv_xtalk_in, uv_xtalk_in)

    def test_getPol(self):
        filename = 'zen.2457698.40355.xx.HH.uvcA'
        nt.assert_equal(omni.getPol(filename), 'xx')

    def test_isLinPol(self):
        linpol = 'xx'
        nt.assert_true(omni.isLinPol(linpol))
        crosspol = 'xy'
        nt.assert_false(omni.isLinPol(crosspol))

    def test_file2djd(self):
        filename = 'zen.2457698.40355.xx.HH.uvcA'
        nt.assert_equal(omni.file2djd(filename), '2457698.40355')

    def test_get_optionParser(self):
        method = 'fake_method'
        nt.assert_raises(AssertionError, omni.get_optionParser, method)

    def test_process_ex_ants(self):
        ex_ants = ''
        xants = omni.process_ex_ants(ex_ants)
        nt.assert_equal(xants, [])

        ex_ants = '0,1,2'
        xants = omni.process_ex_ants(ex_ants)
        nt.assert_equal(xants, [0, 1, 2])

        ex_ants = '0,obvious_error'
        nt.assert_raises(AssertionError, omni.process_ex_ants, ex_ants)

        ex_ants = '0,1'
        json_file = os.path.join(DATA_PATH, 'test_input', 'ant_metrics_output.json')
        xants = omni.process_ex_ants(ex_ants, metrics_json=json_file)
        nt.assert_equal(xants, [0, 1, 81])
        return


class Test_Antpol(object):

    def setUp(self):
        self.pols = ['x', 'y']
        antennas = [0]
        self.antpols = []
        for pol in self.pols:
            self.antpols.append(omni.Antpol(antennas[0], pol, 1))

    def test_antpol(self):
        for i, ant in enumerate(self.antpols):
            nt.assert_equal(ant.antpol(), (0, self.pols[i]))
            nt.assert_equal(ant.ant(), 0)
            nt.assert_equal(ant.pol(), self.pols[i])
            nt.assert_equal(int(ant), i)
            nt.assert_equal(str(ant), '{0}{1}'.format(ant.ant(), ant.pol()))
            nt.assert_true(ant == 0)
            nt.assert_equal({ant: None}.keys()[0], ant)


class Test_RedundantInfo(object):

    def setUp(self):
        self.aa = get_aa(np.linspace(.1, .2, 16))
        self.pol = ['x']
        self.info = omni.aa_to_info(self.aa, pols=self.pol)
        self.reds = self.info.get_reds()
        self.nondegenerategains = {}
        self.gains = {}
        self.vis = {}
        self.gains[self.pol[0]] = {gn: np.random.randn(
            1, 16) + 1j * np.random.randn(1, 16) for gn in self.info.subsetant}
        self.nondegenerategains[self.pol[0]] = {gn:  np.random.randn(
            1, 16) + 1j * np.random.randn(1, 16) for gn in self.info.subsetant}
        self.vis[self.pol[0] * 2] = {red[0]: np.random.randn(
            1, 16) + 1j * np.random.randn(1, 16) for red in self.reds}

    def test_bl_order(self):
        self.bl_order = [(omni.Antpol(self.info.subsetant[i], self.info.nant), omni.Antpol(
            self.info.subsetant[j], self.info.nant)) for i, j in self.info.bl2d]
        nt.assert_equal(self.bl_order, self.info.bl_order())

    def test_order_data(self):
        self.data = {}
        for red in self.reds:
            for i, j in red:
                # Randomly swap baseline orientation
                if np.random.randint(2):
                    self.data[i, j] = {
                        self.pol[0] * 2: np.random.randn(1, 16) + 1j * np.random.randn(1, 16)}
                else:
                    self.data[j, i] = {
                        self.pol[0] * 2: np.random.randn(1, 16) + 1j * np.random.randn(1, 16)}

        d = []
        for i, j in self.info.bl_order():
            bl = (i.ant(), j.ant())
            pol = i.pol() + j.pol()
            try:
                d.append(self.data[bl][pol])
            except(KeyError):
                d.append(self.data[bl[::-1]][pol[::-1]].conj())
        nt.assert_equal(np.testing.assert_equal(np.array(d).transpose(
            (1, 2, 0)), self.info.order_data(self.data)), None)

    def test_pack_calpar(self):
        calpar = np.zeros(
            (1, 16, self.info.calpar_size(4, len(self.info.ubl))))
        calpar2 = np.zeros(
            (1, 16, self.info.calpar_size(4, len(self.info.ubl))))

        omni_info = RedundantInfo()
        reds = omni.compute_reds(
            4, self.pol, self.info.antloc[:self.info.nant])
        omni_info.init_from_reds(reds, self.info.antloc)
        _gains = {}
        for pol in self.gains:
            for ant in self.gains[pol]:
                _gains[ant] = self.gains[pol][ant].conj()

        _vis = {}
        for pol in self.vis:
            for i, j in self.vis[pol]:
                _vis[i, j] = self.vis[pol][i, j]
        calpar = omni_info.pack_calpar(calpar, gains=_gains, vis=_vis)
        nt.assert_equal(np.testing.assert_equal(
            self.info.pack_calpar(calpar2, self.gains, self.vis), calpar), None)

        # again with nondegenerate gains
        calpar = np.zeros(
            (1, 16, self.info.calpar_size(4, len(self.info.ubl))))
        calpar2 = np.zeros(
            (1, 16, self.info.calpar_size(4, len(self.info.ubl))))

        omni_info = RedundantInfo()
        reds = omni.compute_reds(
            4, self.pol, self.info.antloc[:self.info.nant])
        omni_info.init_from_reds(reds, self.info.antloc)
        _gains = {}
        for pol in self.gains:
            for ant in self.gains[pol]:
                _gains[ant] = self.gains[pol][
                    ant].conj() / self.nondegenerategains[pol][ant].conj()

        _vis = {}
        for pol in self.vis:
            for i, j in self.vis[pol]:
                _vis[i, j] = self.vis[pol][i, j]
        calpar = omni_info.pack_calpar(calpar, gains=_gains, vis=_vis)
        nt.assert_equal(np.testing.assert_equal(self.info.pack_calpar(
            calpar2, self.gains, self.vis, nondegenerategains=self.nondegenerategains), calpar), None)

        # test not giving gains and vis to calpar
        calpar = np.zeros(
            (1, 16, self.info.calpar_size(4, len(self.info.ubl))))
        calpar_out = omni_info.pack_calpar(calpar)
        nt.assert_equal(np.testing.assert_equal(calpar, calpar_out), None)

    def test_unpack_calpar(self):
        calpar = np.zeros(
            (1, 16, self.info.calpar_size(4, len(self.info.ubl))))
        calpar = self.info.pack_calpar(calpar, self.gains, self.vis)

        m, g, v = self.info.unpack_calpar(calpar)
        for pol in g.keys():
            for ant in g[pol].keys():
                nt.assert_equal(np.testing.assert_almost_equal(
                    g[pol][ant], self.gains[pol][ant]), None)
        nt.assert_equal(np.testing.assert_equal(v, self.vis), None)

        calpar = np.zeros(
            (1, 16, self.info.calpar_size(4, len(self.info.ubl))))
        calpar = self.info.pack_calpar(
            calpar, self.gains, self.vis, nondegenerategains=self.nondegenerategains)

        m, g, v = self.info.unpack_calpar(
            calpar, nondegenerategains=self.nondegenerategains)
        for pol in g.keys():
            for ant in g[pol].keys():
                nt.assert_equal(np.testing.assert_almost_equal(
                    g[pol][ant], self.gains[pol][ant]), None)
        nt.assert_equal(np.testing.assert_equal(v, self.vis), None)


class Test_Redcal_Basics(object):

    def setUp(self):
        self.freqs = np.array([.1, .125, .150, .175, .2])
        self.aa = get_aa(self.freqs)
        self.info = omni.aa_to_info(self.aa)
        self.times = np.arange(3)
        self.pol = ['x']
        self.data = {}
        self.wgts = {}
        for ai, aj in self.info.bl_order():
            self.data[ai.ant(), aj.ant()] = {self.pol[
                0] * 2: np.ones((self.times.size, self.freqs.size), dtype=np.complex64)}
        self.unitgains = {self.pol[0]: {ant: np.ones(
            (self.times.size, self.freqs.size), dtype=np.complex64) for ant in self.info.subsetant}}

    def test_run_omnical(self):
        m, g, v = omni.run_omnical(self.data, self.info, gains0=self.unitgains)
        nt.assert_equal(np.testing.assert_equal(g, self.unitgains), None)

    def test_compute_xtalk(self):
        m, g, v = omni.run_omnical(self.data, self.info, gains0=self.unitgains)
        wgts = {self.pol[0] * 2: {}}
        zeros = {self.pol[0] * 2: {}}
        for ai, aj in self.info.bl_order():
            wgts[self.pol[0] * 2][ai.ant(), aj.ant()] = \
                np.ones_like(m['res'][self.pol[0] * 2][ai.ant(), aj.ant()], dtype=np.bool)
            zeros[self.pol[0] * 2][ai.ant(), aj.ant()] = \
                np.mean(np.zeros_like(m['res'][self.pol[0] * 2][ai.ant(), aj.ant()]), axis=0)
            # need to average over the times
        nt.assert_equal(np.testing.assert_equal(
            omni.compute_xtalk(m['res'], wgts), zeros), None)


class Test_HERACal(UVCal):

    def test_gainHC(self):
        fn = os.path.join(DATA_PATH, 'test_input', 'zen.2457698.40355.xx.HH.uvc.omni.calfits')
        meta, gains, vis, xtalk = omni.from_fits(fn)
        meta['inttime'] = np.diff(meta['times'])[0] * 60 * 60 * 24
        optional = {'observer': 'heracal'} #because it's easier than changing the fits header
        hc = omni.HERACal(meta, gains, optional=optional)
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
        hc = omni.HERACal(meta, gains, optional=optional, DELAY=True)
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




class Test_4pol_remove_degen(object):

    def setUp(self):
        self.freqs = np.linspace(.1, .2, 16)
        self.Ntimes = 3
        if DATA_PATH not in sys.path:
            sys.path.append(DATA_PATH)
        self.aa = aipy.cal.get_aa('heratest_calfile', self.freqs)
        self.antpols = ['x', 'y']
        self.info = omni.aa_to_info(self.aa, pols=self.antpols)
        self.bls = [(i.val,j.val) for (i,j) in self.info.bl_order() if (i.val<128 and j.val<128)]
        antpos = self.info.get_antpos()
        positions = np.array([antpos[ant,0:2] for antpol in self.antpols
                              for ant in self.info.subsetant])
        Rgains = positions
        self.Mgains = np.linalg.pinv(Rgains.T.dot(Rgains)).dot(Rgains.T)


    def test_remove_degen(self):
        pols = ['xx','xy','yx','yy']
        v2 = {pol: {bl: np.random.randn(self.Ntimes, len(self.freqs)) +
                    1.0j * np.random.randn(self.Ntimes, len(self.freqs))
                    for bl in self.bls} for pol in pols}
        g2 = {antpol: {ant: 1.0 + 0.01 * np.random.randn(self.Ntimes, len(self.freqs))
                       + 0.01j * np.random.randn(self.Ntimes, len(self.freqs)) for ant
                       in self.info.subsetant} for antpol in self.antpols}
        gains0 = {antpol: {ant: np.ones((self.Ntimes, len(self.freqs))) for ant
                           in self.info.subsetant} for antpol in self.antpols}

        g3, v3 = omni.remove_degen(self.info, g2, v2, gains0, minV=False)
        gains = np.array([g3[antpol][ant] for antpol in self.antpols
                          for ant in self.info.subsetant])
        gains_x = np.array([g3['x'][ant] for ant in self.info.subsetant])
        gains_y = np.array([g3['y'][ant] for ant in self.info.subsetant])
        meanSqAmplitude = np.mean([np.abs(g3['x'][ant1] * g3['x'][ant2])
                for (ant1,ant2) in v3['xx'].keys()], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, 1.0)
        meanSqAmplitude = np.mean([np.abs(g3['y'][ant1] * g3['y'][ant2])
                for (ant1,ant2) in v3['yy'].keys()], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, 1.0)
        np.testing.assert_almost_equal(np.mean(np.angle(gains_x), axis=0), 0.0)
        np.testing.assert_almost_equal(np.mean(np.angle(gains_y), axis=0), 0.0)
        degenRemoved = np.einsum('ij,jkl',self.Mgains, np.angle(gains))
        np.testing.assert_almost_equal(degenRemoved, 0.0)


    def test_remove_degen_minV(self):
        pols = ['xx','xy','yy']
        v2 = {pol: {bl: np.random.randn(self.Ntimes, len(self.freqs)) +
                    1.0j * np.random.randn(self.Ntimes, len(self.freqs))
                    for bl in self.bls} for pol in pols}
        g2 = {antpol: {ant: 1.0 + 0.01 * np.random.randn(self.Ntimes, len(self.freqs))
                       + 0.01j * np.random.randn(self.Ntimes, len(self.freqs)) for ant
                       in self.info.subsetant} for antpol in self.antpols}
        gains0 = {antpol: {ant: np.ones((self.Ntimes, len(self.freqs))) for ant
                           in self.info.subsetant} for antpol in self.antpols}

        g3, v3 = omni.remove_degen(self.info, g2, v2, gains0, minV=True)
        gains = np.array([g3[antpol][ant] for antpol in self.antpols
                          for ant in self.info.subsetant])
        meanSqAmplitude = np.mean([np.abs(g3['x'][ant1] * g3['x'][ant2])
                for (ant1,ant2) in v3['xx'].keys()], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, 1.0)
        meanSqAmplitude = np.mean([np.abs(g3['y'][ant1] * g3['y'][ant2])
                for (ant1,ant2) in v3['yy'].keys()], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, 1.0)
        np.testing.assert_almost_equal(np.mean(np.angle(gains), axis=0), 0.0)
        degenRemoved = np.einsum('ij,jkl',self.Mgains, np.angle(gains))
        np.testing.assert_almost_equal(degenRemoved, 0.0)
        np.testing.assert_equal(len(g3.keys()),2)
        np.testing.assert_equal(len(v3.keys()),4)



class Test_omni_run(object):

    # single pol tests
    global xx_vis, calfile, xx_fcal
    xx_vis = 'zen.2457698.40355.xx.HH.uvcAA'
    calfile = 'hera_test_calfile'
    xx_fcal = 'zen.2457698.40355.xx.HH.uvcAA.first.calfits'

    # multi pol tests
    global visXX, visXY, visYX, visYY
    global fcalXX, fcalYY
    visXX = 'zen.2457698.40355.xx.HH.uvcA'
    visXY = 'zen.2457698.40355.xy.HH.uvcA'
    visYX = 'zen.2457698.40355.yx.HH.uvcA'
    visYY = 'zen.2457698.40355.yy.HH.uvcA'
    fcalXX = 'zen.2457698.40355.xx.HH.uvcA.first.calfits'
    fcalYY = 'zen.2457698.40355.yy.HH.uvcA.first.calfits'
    testpath = os.path.dirname(os.path.abspath(__file__))
    if CAL_PATH not in sys.path:
        sys.path.append(CAL_PATH)

    def test_empty_fileset_omni_run(self):
        o = omni.get_optionParser('omni_run')
        cmd = "-C %s -p xx --firstcal=%s" % (calfile, xx_fcal)
        opts, files = o.parse_args(cmd.split())
        history = 'history'
        nt.assert_raises(AssertionError, omni.omni_run, files, opts, history)

    def test_minV_without_crosspols_omni_run(self):
        o = omni.get_optionParser('omni_run')
        cmd = "-C %s -p xx --minV --firstcal=%s %s" % (
            calfile, xx_fcal, xx_vis)
        opts, files = o.parse_args(cmd.split())
        history = 'history'
        nt.assert_raises(AssertionError, omni.omni_run, files, opts, history)

    def test_without_firstcal_file_omni_run(self):
        o = omni.get_optionParser('omni_run')
        xx_vis_path = os.path.join(DATA_PATH, xx_vis)
        cmd = "-C %s -p xx %s" % (calfile, xx_vis_path)
        opts, files = o.parse_args(cmd.split())
        history = 'history'
        nt.assert_raises(ValueError, omni.omni_run, files, opts, history)

    def test_single_file_execution_omni_run(self):
        objective_file = os.path.join(
            DATA_PATH, 'test_output', 'zen.2457698.40355.xx.HH.uvcAA.omni.calfits')
        if os.path.exists(objective_file):
            os.remove(objective_file)
        o = omni.get_optionParser('omni_run')
        xx_fcal4real = os.path.join(DATA_PATH, 'test_input', xx_fcal)
        xx_vis4real = os.path.join(DATA_PATH, xx_vis)
        omnipath = os.path.join(DATA_PATH, 'test_output')

        cmd = "-C %s -p xx --firstcal=%s --ex_ants=81 --omnipath=%s %s" % (
            calfile, xx_fcal4real, omnipath, xx_vis4real)
        opts, files = o.parse_args(cmd.split())
        history = 'history'
        omni.omni_run(files, opts, history)
        nt.assert_true(os.path.exists(objective_file))
        os.remove(objective_file)

    def test_single_file_execution_omni_run_overwrite(self):
        objective_file1 = os.path.join(
            DATA_PATH, 'test_output', 'zen.2457698.40355.xx.HH.uvcAA.omni.calfits')
        objective_file2 = os.path.join(
            DATA_PATH, 'test_output', 'zen.2457698.40355.xx.HH.uvcAA.vis.uvfits')
        objective_file3 = os.path.join(
            DATA_PATH, 'test_output', 'zen.2457698.40355.xx.HH.uvcAA.xtalk.uvfits')
        if os.path.exists(objective_file1):
            os.remove(objective_file1)
        if os.path.exists(objective_file2):
            os.remove(objective_file2)
        if os.path.exists(objective_file3):
            os.remove(objective_file3)
        o = omni.get_optionParser('omni_run')
        xx_fcal4real = os.path.join(DATA_PATH, 'test_input', xx_fcal)
        xx_vis4real = os.path.join(DATA_PATH, xx_vis)
        omnipath = os.path.join(DATA_PATH, 'test_output')
        # create blank files
        _ = open(objective_file1, 'a').close()
        _ = open(objective_file2, 'a').close()
        _ = open(objective_file3, 'a').close()
        # run firstcal
        cmd = "-C %s -p xx --firstcal=%s --ex_ants=81 --omnipath=%s --overwrite %s" % (
            calfile, xx_fcal4real, omnipath, xx_vis4real)
        opts, files = o.parse_args(cmd.split())
        history = 'history'
        omni.omni_run(files, opts, history)
        nt.assert_true(os.path.exists(objective_file1))
        nt.assert_true(os.path.exists(objective_file2))
        nt.assert_true(os.path.exists(objective_file3))
            # check the files are calfits and uvfits
        uv1, uv2, uv3 = UVCal(), UVFITS(), UVFITS()
        uv1.read_calfits(objective_file1)
        uv2.read_uvfits(objective_file2)
        uv3.read_uvfits(objective_file3)
        # do a quick meta data check
        nt.assert_equal(uv1.Nants_data, 18)
        nt.assert_equal(uv2.Nants_data, 17)
        nt.assert_equal(uv3.Nants_data, 18)	
        os.remove(objective_file1)
        os.remove(objective_file2)
        os.remove(objective_file3)

    def test_single_file_execution_omni_run_nocalfile(self):
        objective_file = os.path.join(
            DATA_PATH, 'test_output', 'zen.2457999.76839.xx.HH.uvA.omni.calfits')
        if os.path.exists(objective_file):
            os.remove(objective_file)
        o = omni.get_optionParser('omni_run')
        xx_fcal = os.path.join(DATA_PATH, 'test_input',
                               'zen.2457999.76839.xx.HH.uvA.first.calfits')
        xx_vis = os.path.join(DATA_PATH, 'zen.2457999.76839.xx.HH.uvA')
        omnipath = os.path.join(DATA_PATH, 'test_output')
        cmd = "-p xx --firstcal={0} --omnipath={1} {2}".format(xx_fcal, omnipath, xx_vis)
        opts, files = o.parse_args(cmd.split())
        history = 'history'
        omni.omni_run(files, opts, history)
        nt.assert_true(os.path.exists(objective_file))
        os.remove(objective_file)

    def test_single_file_execution_omni_run_with_median(self):
        objective_file = os.path.join(
            DATA_PATH, 'test_output', 'zen.2457698.40355.xx.HH.uvcAA.omni.calfits')
        if os.path.exists(objective_file):
            os.remove(objective_file)
        o = omni.get_optionParser('omni_run')
        xx_fcal4real = os.path.join(DATA_PATH, 'test_input', xx_fcal)
        xx_vis4real = os.path.join(DATA_PATH, xx_vis)
        omnipath = os.path.join(DATA_PATH, 'test_output')

        cmd = "-C %s -p xx --firstcal=%s --ex_ants=81 --omnipath=%s --median %s" % (
            calfile, xx_fcal4real, omnipath, xx_vis4real)
        opts, files = o.parse_args(cmd.split())
        history = 'history'
        omni.omni_run(files, opts, history)
        nt.assert_true(os.path.exists(objective_file))
        os.remove(objective_file)

    def test_execution_omni_run_4pol(self):
        objective_file = os.path.join(
            DATA_PATH, 'zen.2457698.40355.HH.uvcA.omni.calfits')
        if os.path.exists(objective_file):
            os.remove(objective_file)
        o = omni.get_optionParser('omni_run')
        visxx = os.path.join(DATA_PATH, visXX)
        visxy = os.path.join(DATA_PATH, visXY)
        visyx = os.path.join(DATA_PATH, visYX)
        visyy = os.path.join(DATA_PATH, visYY)
        fcalxx = os.path.join(DATA_PATH, 'test_input', fcalXX)
        fcalyy = os.path.join(DATA_PATH, 'test_input', fcalYY)

        cmd = "-C %s -p xx,xy,yx,yy --firstcal=%s,%s --ex_ants=81 --omnipath=%s %s %s %s %s" % (
            calfile, fcalxx, fcalyy, DATA_PATH, visxx, visxy, visyx, visyy)

        opts, files = o.parse_args(cmd.split())
        history = 'history'
        omni.omni_run(files, opts, history)
        nt.assert_true(os.path.exists(objective_file))
        # clean up
        os.remove(objective_file)
        visfile = re.sub('omni\.calfits', 'vis.uvfits', objective_file)
        xtalkfile = re.sub('omni\.calfits', 'xtalk.uvfits', objective_file)
        os.remove(visfile)
        os.remove(xtalkfile)

    def test_execution_omni_run_4polminV(self):
        objective_file = os.path.join(
            DATA_PATH, 'zen.2457698.40355.HH.uvcA.omni.calfits')
        if os.path.exists(objective_file):
            os.remove(objective_file)
        o = omni.get_optionParser('omni_run')
        visxx = os.path.join(DATA_PATH, visXX)
        visxy = os.path.join(DATA_PATH, visXY)
        visyx = os.path.join(DATA_PATH, visYX)
        visyy = os.path.join(DATA_PATH, visYY)
        fcalxx = os.path.join(DATA_PATH, 'test_input', fcalXX)
        fcalyy = os.path.join(DATA_PATH, 'test_input', fcalYY)

        cmd = "-C %s -p xx,xy,yx,yy --minV --firstcal=%s,%s --ex_ants=81 --omnipath=%s %s %s %s %s" % (
            calfile, fcalxx, fcalyy, DATA_PATH, visxx, visxy, visyx, visyy)

        opts, files = o.parse_args(cmd.split())
        history = 'history'
        omni.omni_run(files, opts, history)
        nt.assert_true(os.path.exists(objective_file))
        # clean up
        os.remove(objective_file)
        visfile = re.sub('omni\.calfits', 'vis.uvfits', objective_file)
        xtalkfile = re.sub('omni\.calfits', 'xtalk.uvfits', objective_file)
        os.remove(visfile)
        os.remove(xtalkfile)

class Test_omni_apply(object):

    # single pol tests
    global xx_vis,calfile,xx_fcal,xx_ocal
    xx_vis  = 'zen.2457698.40355.xx.HH.uvcAA'
    xx_fcal = 'zen.2457698.40355.xx.HH.uvcAA.first.calfits'
    xx_ocal = 'zen.2457698.40355.xx.HH.uvcAA.omni.calfits'

    # multi pol tests
    global visXX, visXY, visYX, visYY, fourpol_ocal
    visXX = 'zen.2457698.40355.xx.HH.uvcA'
    visXY = 'zen.2457698.40355.xy.HH.uvcA'
    visYX = 'zen.2457698.40355.yx.HH.uvcA'
    visYY = 'zen.2457698.40355.yy.HH.uvcA'
    fourpol_ocal = 'zen.2457698.40355.HH.uvcA.omni.calfits'

    def test_single_file_execution_omni_apply(self):
        objective_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAAO')
        if os.path.exists(objective_file):
            shutil.rmtree(objective_file)
        o = omni.get_optionParser('omni_apply')
        omni_file = os.path.join(DATA_PATH, 'test_input', xx_ocal)
        vis_file = os.path.join(DATA_PATH,  xx_vis)
        cmd = "-p xx --omnipath={0} --extension=O {1}".format(omni_file, vis_file)

        opts, files = o.parse_args(cmd.split())
        omni.omni_apply(files, opts)
        nt.assert_true(os.path.exists(objective_file))
        # clean up when we're done
        shutil.rmtree(objective_file)

    def test_single_file_execution_omni_apply_overwrite(self):
        objective_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAAO')
        if os.path.exists(objective_file):
            shutil.rmtree(objective_file)
        os.makedirs(objective_file)
        o = omni.get_optionParser('omni_apply')
        omni_file = os.path.join(DATA_PATH, 'test_input', xx_ocal)
        vis_file = os.path.join(DATA_PATH,  xx_vis)
        cmd = "-p xx --omnipath={0} --extension=O --overwrite {1}".format(omni_file, vis_file)

        opts, files = o.parse_args(cmd.split())
        omni.omni_apply(files, opts)
        nt.assert_true(os.path.exists(objective_file))
        # make sure its a miriad file
        uvd = UVData()
        uvd.read_miriad(objective_file)
        # make sure a metadata column is set correctly
        nt.assert_equal(uvd.Nants_data, 19)
        # clean up when we're done
        shutil.rmtree(objective_file)

    def test_single_file_execution_omni_apply_custompath(self):
        objective_file = os.path.join('./', 'zen.2457698.40355.xx.HH.uvcAAO')
        if os.path.exists(objective_file):
            shutil.rmtree(objective_file)
        o = omni.get_optionParser('omni_apply')
        omni_file = os.path.join(DATA_PATH, 'test_input', xx_ocal)
        vis_file = os.path.join(DATA_PATH,  xx_vis)
        cmd = "-p xx --omnipath={0} --extension=O --outpath={1} {2}".format(omni_file, ".", vis_file)

        opts, files = o.parse_args(cmd.split())
        omni.omni_apply(files, opts)
        nt.assert_true(os.path.exists(objective_file))
        # clean up when we're done
        shutil.rmtree(objective_file)

    def test_single_file_firstcal_omni_apply(self):
        objective_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAAF')
        if os.path.exists(objective_file):
            shutil.rmtree(objective_file)
        o = omni.get_optionParser('omni_apply')
        fcal_file = os.path.join(DATA_PATH, 'test_input', xx_fcal)
        vis_file = os.path.join(DATA_PATH,  xx_vis)
        cmd = "-p xx --omnipath={0} --median --firstcal {1}".format(fcal_file, vis_file)

        opts, files = o.parse_args(cmd.split())
        omni.omni_apply(files, opts)
        nt.assert_true(os.path.exists(objective_file))
        # clean up when we're done
        shutil.rmtree(objective_file)

    def test_execution_omni_apply_4pol(self):
        objective_files = [os.path.join(DATA_PATH,f+'O') for f in [visXX, visXY, visYX, visYY]]
        for f in objective_files:
            if os.path.exists(f):
                shutil.rmtree(f)

        fp_ocal = os.path.join(DATA_PATH, 'test_input', fourpol_ocal)
        visxx = os.path.join(DATA_PATH,visXX)
        visxy = os.path.join(DATA_PATH,visXY)
        visyx = os.path.join(DATA_PATH,visYX)
        visyy = os.path.join(DATA_PATH,visYY)

        o = omni.get_optionParser('omni_apply')
        cmd = "-p xx,xy,yx,yy --omnipath={0} --extension=O {1} {2} {3} {4}".format(fp_ocal,visxx,visyy,visyx,visxy)
        opts, files = o.parse_args(cmd.split())
        omni.omni_apply(files, opts)
        for f in objective_files:
            nt.assert_true(os.path.exists(f))

        for f in objective_files:
            if os.path.exists(f):
                shutil.rmtree(f)

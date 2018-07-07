'''Tests for io.py'''
import unittest
import numpy as np
import pyuvdata
from pyuvdata import UVCal, UVData
from hera_cal.data import DATA_PATH
from collections import OrderedDict as odict
from hera_cal.datacontainer import DataContainer
import hera_cal.io as io
from hera_cal.io import HERACal
import os
import shutil
import copy


class Test_HERACal(unittest.TestCase):
    
    def setUp(self):
        self.fname_xx = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.xx.HH.uvc.omni.calfits")
        self.fname_yy = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.yy.HH.uvc.omni.calfits")
        self.fname_both = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.HH.uvcA.omni.calfits")
    
    def test_init(self):
        hc = HERACal(self.fname_xx)
        self.assertEqual(hc.filepaths, [self.fname_xx])
        hc = HERACal([self.fname_xx, self.fname_yy])
        self.assertEqual(hc.filepaths, [self.fname_xx, self.fname_yy])
        hc = HERACal((self.fname_xx, self.fname_yy))
        self.assertEqual(hc.filepaths, [self.fname_xx, self.fname_yy])        
        with self.assertRaises(TypeError):
            hc = HERACal([0,1])
        with self.assertRaises(ValueError):
            hc = HERACal(None)
            
    def test_read(self):
        # test one file with both polarizations and a non-None total quality array
        hc = HERACal(self.fname_both)
        gains, flags, quals, total_qual = hc.read()
        uvc = UVCal()
        uvc.read_calfits(self.fname_both)
        np.testing.assert_array_equal(uvc.gain_array[0, 0, :, :, 0].T, gains[9, 'jxx'])
        np.testing.assert_array_equal(uvc.flag_array[0, 0, :, :, 0].T, flags[9, 'jxx'])        
        np.testing.assert_array_equal(uvc.quality_array[0, 0, :, :, 0].T, quals[9, 'jxx'])                
        np.testing.assert_array_equal(uvc.total_quality_array[0, :, :, 0].T, total_qual['jxx'])
        np.testing.assert_array_equal(np.unique(uvc.freq_array), hc.freqs)
        np.testing.assert_array_equal(np.unique(uvc.time_array), hc.times)        
        self.assertEqual(hc.pols, ['jxx', 'jyy'])
        self.assertEqual(set([ant[0] for ant in hc.ants]), set(uvc.ant_array))
        
        # test list loading
        hc = HERACal([self.fname_xx, self.fname_yy])
        gains, flags, quals, total_qual = hc.read()
        self.assertEqual(len(gains.keys()), 36)
        self.assertEqual(len(flags.keys()), 36)
        self.assertEqual(len(quals.keys()), 36)
        self.assertEqual(hc.freqs.shape, (1024,))
        self.assertEqual(hc.times.shape, (3,))
        self.assertEqual(sorted(hc.pols), ['jxx', 'jyy'])
        
    def test_write(self):
        hc = HERACal(self.fname_both)
        gains, flags, quals, total_qual = hc.read()
        for key in gains.keys():
            gains[key] *= 2.0 + 1.0j
            flags[key] = np.logical_not(flags[key])
            quals[key] *= 2.0
        for key in total_qual.keys():
            total_qual[key] *= 2
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


class Test_Visibility_IO_Legacy(unittest.TestCase):

    def test_load_vis(self):
        # duplicated testing from abscal_funcs.UVData2AbsCalDict

        # load into pyuvdata object
        self.data_file = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        self.uvd = UVData()
        self.uvd.read_miriad(self.data_file)
        self.freq_array = np.unique(self.uvd.freq_array)
        self.antpos, self.ants = self.uvd.get_ENU_antpos(center=True, pick_data_ants=True)
        self.antpos = odict(zip(self.ants, self.antpos))
        self.time_array = np.unique(self.uvd.time_array)

        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        data, flags = io.load_vis(fname, pop_autos=False)
        self.assertEqual(data[(24, 25, 'xx')].shape, (60, 64))
        self.assertEqual(flags[(24, 25, 'xx')].shape, (60, 64))
        self.assertEqual((24, 24, 'xx') in data, True)
        data, flags = io.load_vis([fname])
        self.assertEqual(data[(24, 25, 'xx')].shape, (60, 64))

        # test pop autos
        data, flags = io.load_vis(fname, pop_autos=True)
        self.assertEqual((24, 24, 'xx') in data, False)

        # test uvd object
        uvd = UVData()
        uvd.read_miriad(fname)
        data, flags = io.load_vis(uvd)
        self.assertEqual(data[(24, 25, 'xx')].shape, (60, 64))
        data, flags = io.load_vis([uvd])
        self.assertEqual(data[(24, 25, 'xx')].shape, (60, 64))

        # test multiple
        fname2 = os.path.join(DATA_PATH, "zen.2458043.13298.xx.HH.uvORA")
        data, flags = io.load_vis([fname, fname2])
        self.assertEqual(data[(24, 25, 'xx')].shape, (120, 64))
        self.assertEqual(flags[(24, 25, 'xx')].shape, (120, 64))

        # test w/ meta
        d, f, ap, a, f, t, l, p = io.load_vis([fname, fname2], return_meta=True)
        self.assertEqual(len(ap[24]), 3)
        self.assertEqual(len(f), len(self.freq_array))

        # test uvfits
        fname = os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvA.vis.uvfits')
        with self.assertRaises(NotImplementedError):
            d, f = io.load_vis(fname, filetype='uvfits')
        with self.assertRaises(NotImplementedError):
            d, f = io.load_vis([fname, fname], filetype='uvfits')
        #self.assertEqual(d[(0,1,'xx')].shape, (60,64))

        with self.assertRaises(NotImplementedError):
            d, f = io.load_vis(fname, filetype='not_a_real_filetype')
        with self.assertRaises(NotImplementedError):
            d, f = io.load_vis(['str1', 'str2'], filetype='not_a_real_filetype')
        with self.assertRaises(TypeError):
            d, f = io.load_vis([1, 2], filetype='uvfits')

        # test w/ meta pick_data_ants
        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        d, f, ap, a, f, t, l, p = io.load_vis(fname, return_meta=True, pick_data_ants=False)
        self.assertEqual(len(ap[24]), 3)
        self.assertEqual(len(a), 47)
        self.assertEqual(len(f), len(self.freq_array))

        with self.assertRaises(TypeError):
            d, f = io.load_vis(1.0)

    def test_load_vis_nested(self):
        # duplicated testing from firstcal.UVData_to_dict
        str2pol = io.polstr2num
        filename1 = os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA')
        filename2 = os.path.join(DATA_PATH, 'zen.2458043.13298.xx.HH.uvORA')
        uvd1 = UVData()
        uvd1.read_miriad(filename1)
        uvd2 = UVData()
        uvd2.read_miriad(filename2)
        if uvd1.phase_type != 'drift':
            uvd1.unphase_to_drift()
        if uvd2.phase_type != 'drift':
            uvd2.unphase_to_drift()
        uvd = uvd1 + uvd2
        d, f = io.load_vis([uvd1, uvd2], nested_dict=True)
        for i, j in d:
            for pol in d[i, j]:
                uvpol = list(uvd1.polarization_array).index(str2pol[pol])
                uvmask = np.all(
                    np.array(zip(uvd.ant_1_array, uvd.ant_2_array)) == [i, j], axis=1)
                np.testing.assert_equal(d[i, j][pol], np.resize(
                    uvd.data_array[uvmask][:, 0, :, uvpol], d[i, j][pol].shape))
                np.testing.assert_equal(f[i, j][pol], np.resize(
                    uvd.flag_array[uvmask][:, 0, :, uvpol], f[i, j][pol].shape))

        d, f = io.load_vis([filename1, filename2], nested_dict=True)
        for i, j in d:
            for pol in d[i, j]:
                uvpol = list(uvd.polarization_array).index(str2pol[pol])
                uvmask = np.all(
                    np.array(zip(uvd.ant_1_array, uvd.ant_2_array)) == [i, j], axis=1)
                np.testing.assert_equal(d[i, j][pol], np.resize(
                    uvd.data_array[uvmask][:, 0, :, uvpol], d[i, j][pol].shape))
                np.testing.assert_equal(f[i, j][pol], np.resize(
                    uvd.flag_array[uvmask][:, 0, :, uvpol], f[i, j][pol].shape))

        uvd = UVData()
        uvd.read_miriad(filename1)
        self.assertEqual(len(io.load_vis([uvd], nested_dict=True)[0]), uvd.Nbls)
        # reorder baseline array
        uvd.baseline_array = uvd.baseline_array[np.argsort(uvd.baseline_array)]
        self.assertEqual(len(io.load_vis(filename1, nested_dict=True)[0]), uvd.Nbls)

    def test_write_vis(self):
        # get data
        uvd = UVData()
        uvd.read_miriad(os.path.join(DATA_PATH, "zen.2458044.41632.xx.HH.uvXRAA"))
        data, flgs, ap, a, f, t, l, p = io.load_vis(uvd, return_meta=True)
        nsample = copy.deepcopy(data)
        for k in nsample.keys():
            nsample[k] = np.ones_like(nsample[k], np.float)

        # test basic execution
        uvd = io.write_vis("ex.uv", data, l, f, ap, start_jd=2458044, return_uvd=True, overwrite=True, verbose=True)
        uvd2 = UVData()
        uvd2.read_miriad('ex.uv')
        self.assertTrue(os.path.exists('ex.uv'))
        self.assertEqual(uvd.data_array.shape, (1680, 1, 64, 1))
        self.assertEqual(uvd2.data_array.shape, (1680, 1, 64, 1))
        self.assertAlmostEqual(data[(24, 25, 'xx')][30, 32], uvd.get_data(24, 25, 'xx')[30, 32])
        self.assertAlmostEqual(data[(24, 25, 'xx')][30, 32], uvd2.get_data(24, 25, 'xx')[30, 32])

        # test with nsample and flags
        uvd = io.write_vis("ex.uv", data, l, f, ap, start_jd=2458044, flags=flgs, nsamples=nsample, return_uvd=True, overwrite=True, verbose=True)
        self.assertEqual(uvd.nsample_array.shape, (1680, 1, 64, 1))
        self.assertEqual(uvd.flag_array.shape, (1680, 1, 64, 1))
        self.assertAlmostEqual(nsample[(24, 25, 'xx')][30, 32], uvd.get_nsamples(24, 25, 'xx')[30, 32])
        self.assertAlmostEqual(flgs[(24, 25, 'xx')][30, 32], uvd.get_flags(24, 25, 'xx')[30, 32])

        # test exceptions
        self.assertRaises(AttributeError, io.write_vis, "ex.uv", data, l, f, ap)
        self.assertRaises(AttributeError, io.write_vis, "ex.uv", data, l, f, ap, start_jd=2458044, filetype='foo')
        if os.path.exists('ex.uv'):
            shutil.rmtree('ex.uv')

    def test_update_vis(self):
        # load in cal
        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        outname = os.path.join(DATA_PATH, "test_output/zen.2458043.12552.xx.HH.modified.uvORA")
        uvd = UVData()
        uvd.read_miriad(fname)
        data, flags, antpos, ants, freqs, times, lsts, pols = io.load_vis(fname, return_meta=True)

        # make some modifications
        new_data = {key: (2. + 1.j) * val for key, val in data.items()}
        new_flags = {key: np.logical_not(val) for key, val in flags.items()}
        io.update_vis(fname, outname, data=new_data, flags=new_flags,
                      add_to_history='hello world', clobber=True, telescope_name='PAPER')

        # test modifications
        data, flags, antpos, ants, freqs, times, lsts, pols = io.load_vis(outname, return_meta=True)
        for k in data.keys():
            self.assertTrue(np.all(new_data[k] == data[k]))
            self.assertTrue(np.all(new_flags[k] == flags[k]))
        uvd2 = UVData()
        uvd2.read_miriad(outname)
        self.assertTrue(pyuvdata.utils.check_histories(uvd2.history, uvd.history + 'hello world'))
        self.assertEqual(uvd2.telescope_name, 'PAPER')
        shutil.rmtree(outname)

        # Coverage for errors
        with self.assertRaises(NotImplementedError):
            io.update_vis(uvd, outname, data=new_data, flags=new_flags, filetype_out='uvfits',
                          add_to_history='hello world', clobber=True, telescope_name='PAPER')
        with self.assertRaises(NotImplementedError):
            io.update_vis(fname, outname, data=new_data, flags=new_flags, filetype_in='uvfits',
                          add_to_history='hello world', clobber=True, telescope_name='PAPER')
        with self.assertRaises(TypeError):
            io.update_vis(uvd, outname, data=new_data, flags=new_flags, filetype_out='not_a_real_filetype',
                          add_to_history='hello world', clobber=True, telescope_name='PAPER')
        with self.assertRaises(TypeError):
            io.update_vis(fname, outname, data=new_data, flags=new_flags, filetype_in='not_a_real_filetype',
                          add_to_history='hello world', clobber=True, telescope_name='PAPER')

        # #now try the same thing but with a UVData object instead of path
        io.update_vis(uvd, outname, data=new_data, flags=new_flags,
                      add_to_history='hello world', clobber=True, telescope_name='PAPER')
        data, flags, antpos, ants, freqs, times, lsts, pols = io.load_vis(outname, return_meta=True)
        for k in data.keys():
            self.assertTrue(np.all(new_data[k] == data[k]))
            self.assertTrue(np.all(new_flags[k] == flags[k]))
        uvd2 = UVData()
        uvd2.read_miriad(outname)
        self.assertTrue(pyuvdata.utils.check_histories(uvd2.history, uvd.history + 'hello world'))
        self.assertEqual(uvd2.telescope_name, 'PAPER')
        shutil.rmtree(outname)


class Test_Calibration_IO_Legacy(unittest.TestCase):

    def test_load_cal(self):

        with self.assertRaises(TypeError):
            io.load_cal(1.0)

        fname = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.xx.HH.uvc.omni.calfits")
        gains, flags = io.load_cal(fname)
        self.assertEqual(len(gains.keys()), 18)
        self.assertEqual(len(flags.keys()), 18)

        cal = UVCal()
        cal.read_calfits(fname)
        gains, flags = io.load_cal(cal)
        self.assertEqual(len(gains.keys()), 18)
        self.assertEqual(len(flags.keys()), 18)

        with self.assertRaises(TypeError):
            io.load_cal([fname, cal])

        fname_xx = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.xx.HH.uvc.omni.calfits")
        fname_yy = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.yy.HH.uvc.omni.calfits")
        gains, flags, quals, total_qual, ants, freqs, times, pols = io.load_cal([fname_xx, fname_yy], return_meta=True)
        self.assertEqual(len(gains.keys()), 36)
        self.assertEqual(len(flags.keys()), 36)
        self.assertEqual(len(quals.keys()), 36)
        self.assertEqual(freqs.shape, (1024,))
        self.assertEqual(times.shape, (3,))
        self.assertEqual(sorted(pols), ['jxx', 'jyy'])

        cal_xx, cal_yy = UVCal(), UVCal()
        cal_xx.read_calfits(fname_xx)
        cal_yy.read_calfits(fname_yy)
        gains, flags, quals, total_qual, ants, freqs, times, pols = io.load_cal([cal_xx, cal_yy], return_meta=True)
        self.assertEqual(len(gains.keys()), 36)
        self.assertEqual(len(flags.keys()), 36)
        self.assertEqual(len(quals.keys()), 36)
        self.assertEqual(freqs.shape, (1024,))
        self.assertEqual(times.shape, (3,))
        self.assertEqual(sorted(pols), ['jxx', 'jyy'])

    def test_write_cal(self):
        # create fake data
        ants = np.arange(10)
        pols = np.array(['x'])
        freqs = np.linspace(100e6, 200e6, 64, endpoint=False)
        Nfreqs = len(freqs)
        times = np.linspace(2458043.1, 2458043.6, 100)
        Ntimes = len(times)
        gains = {}
        quality = {}
        flags = {}
        total_qual = {}
        for i, p in enumerate(pols):
            total_qual[p] = np.ones((Ntimes, Nfreqs), np.float)
            for j, a in enumerate(ants):
                gains[(a, p)] = np.ones((Ntimes, Nfreqs), np.complex)
                quality[(a, p)] = np.ones((Ntimes, Nfreqs), np.float) * 2
                flags[(a, p)] = np.zeros((Ntimes, Nfreqs), np.bool)

        # set some terms to zero
        gains[(5, 'x')][20:30] *= 0

        # test basic execution
        uvc = io.write_cal("ex.calfits", gains, freqs, times, flags=flags, quality=quality,
                           total_qual=total_qual, overwrite=True, return_uvc=True, write_file=True)
        self.assertTrue(os.path.exists("ex.calfits"))
        self.assertEqual(uvc.gain_array.shape, (10, 1, 64, 100, 1))
        self.assertAlmostEqual(uvc.gain_array[5].min(), 1.0)
        self.assertAlmostEqual(uvc.gain_array[0, 0, 0, 0, 0], (1 + 0j))
        self.assertAlmostEqual(np.sum(uvc.gain_array), (64000 + 0j))
        self.assertEqual(uvc.flag_array[0, 0, 0, 0, 0], False)
        self.assertEqual(np.sum(uvc.flag_array), 640)
        self.assertAlmostEqual(uvc.quality_array[0, 0, 0, 0, 0], 2)
        self.assertAlmostEqual(np.sum(uvc.quality_array), 128000.0)
        self.assertEqual(len(uvc.antenna_numbers), 10)
        self.assertTrue(uvc.total_quality_array is not None)
        if os.path.exists('ex.calfits'):
            os.remove('ex.calfits')
        # test execution with different parameters
        uvc = io.write_cal("ex.calfits", gains, freqs, times, overwrite=True)
        if os.path.exists('ex.calfits'):
            os.remove('ex.calfits')
        # test single integration write
        gains = odict(map(lambda k: (k, gains[k][:1]), gains.keys()))
        uvc = io.write_cal("ex.calfits", gains, freqs, times[:1], return_uvc=True, outdir='./')
        self.assertAlmostEqual(uvc.integration_time, 0.0)
        self.assertEqual(uvc.Ntimes, 1)
        self.assertTrue(os.path.exists('ex.calfits'))
        os.remove('ex.calfits')

    def test_update_cal(self):
        # load in cal
        fname = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.xx.HH.uvc.omni.calfits")
        outname = os.path.join(DATA_PATH, "test_output/zen.2457698.40355.xx.HH.uvc.modified.calfits.")
        cal = UVCal()
        cal.read_calfits(fname)
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
            self.assertTrue(np.all(new_gains[k] == gains[k]))
            self.assertTrue(np.all(new_flags[k] == flags[k]))
            self.assertTrue(np.all(new_quals[k] == quals[k]))
        cal2 = UVCal()
        cal2.read_calfits(outname)
        self.assertTrue(pyuvdata.utils.check_histories(cal2.history, cal.history + 'hello world'))
        self.assertEqual(cal2.telescope_name, 'MWA')
        os.remove(outname)

        # now try the same thing but with a UVCal object instead of path
        io.update_cal(cal, outname, gains=new_gains, flags=new_flags, quals=new_quals,
                      add_to_history='hello world', clobber=True, telescope_name='MWA')
        gains, flags, quals, total_qual, ants, freqs, times, pols = io.load_cal(outname, return_meta=True)
        for k in gains.keys():
            self.assertTrue(np.all(new_gains[k] == gains[k]))
            self.assertTrue(np.all(new_flags[k] == flags[k]))
            self.assertTrue(np.all(new_quals[k] == quals[k]))
        cal2 = UVCal()
        cal2.read_calfits(outname)
        self.assertTrue(pyuvdata.utils.check_histories(cal2.history, cal.history + 'hello world'))
        self.assertEqual(cal2.telescope_name, 'MWA')
        os.remove(outname)


class Test_Flags_NPZ_IO(unittest.TestCase):

    def test_load_npz_flags(self):
        npzfile = os.path.join(DATA_PATH, "test_input/zen.2458101.45361.xx.HH.uvOCR_53x_54x_only.flags.applied.npz")
        flags = io.load_npz_flags(npzfile)
        self.assertTrue((53, 54, 'xx') in flags)
        for f in flags.values():
            self.assertEqual(f.shape, (60, 1024))
            np.testing.assert_array_equal(f[:, 0:50], True)
            np.testing.assert_array_equal(f[:, -50:], True)
            self.assertFalse(np.all(f))


if __name__ == '__main__':
    unittest.main()

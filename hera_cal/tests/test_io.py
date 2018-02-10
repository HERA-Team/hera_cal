'''Tests for abscal.py'''
import unittest
import numpy as np
from pyuvdata import UVCal, UVData
from hera_cal.data import DATA_PATH
from collections import OrderedDict as odict
from hera_cal.datacontainer import DataContainer
import hera_cal.io as io
import os

class Test_Visibility_IO(unittest.TestCase):

    def test_load_vis(self):
        #duplicated testing from abscal_funcs.UVData2AbsCalDict
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

        # test pol select
        data, flags = io.load_vis(fname, pop_autos=False, pol_select=['xx'])
        self.assertEqual(data[(24, 25, 'xx')].shape, (60, 64))

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
        # fname = os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvA.vis.uvfits')
        # with self.assertRaises(NotImplementedError):
        #     d, f = io.load_vis(fname, format='uvfits')
        # self.assertEqual(d[(0,1,'xx')].shape, (60,64))

        # test w/ meta pick_data_ants
        d, f, ap, a, f, t, l, p = io.load_vis(fname, return_meta=True, pick_data_ants=False)
        self.assertEqual(len(ap[24]), 3)
        self.assertEqual(len(a), 47)
        self.assertEqual(len(f), len(self.freq_array))

    def test_load_vis_nested(self):
        #duplicated testing from firstcal.UVData_to_dict
        str2pol = {'XX': -5, 'YY': -6, 'XY': -7, 'YY': -8}
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
        d, f = io.load_vis([uvd1,uvd2],nested_dict=True)
        for i, j in d:
            for pol in d[i, j]:
                uvpol = list(uvd1.polarization_array).index(str2pol[pol])
                uvmask = np.all(
                    np.array(zip(uvd.ant_1_array, uvd.ant_2_array)) == [i, j], axis=1)
                np.testing.assert_equal(d[i, j][pol], np.resize(
                    uvd.data_array[uvmask][:, 0, :, uvpol], d[i, j][pol].shape))
                np.testing.assert_equal(f[i, j][pol], np.resize(
                    uvd.flag_array[uvmask][:, 0, :, uvpol], f[i, j][pol].shape))

        d, f = io.load_vis([filename1, filename2],nested_dict=True)
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
        self.assertEqual(len(io.load_vis([uvd],nested_dict=True)[0]), uvd.Nbls)
        # reorder baseline array
        uvd.baseline_array = uvd.baseline_array[np.argsort(uvd.baseline_array)]
        self.assertEqual(len(io.load_vis(filename1,nested_dict=True)[0]), uvd.Nbls)

    #TODO: implement this test
    def test_write_vis(self):
        with self.assertRaises(NotImplementedError):
            io.write_vis(None, None, None)

    
    def test_update_vis(self):
        self.assertEqual(1,1)



class Test_Calibration_IO(unittest.TestCase):

    def test_load_cal(self):
        fname = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.xx.HH.uvc.omni.calfits")
        gains, flags = io.load_cal(fname)
        self.assertEqual(len(gains.keys()),18)
        self.assertEqual(len(flags.keys()),18)

        cal = UVCal()
        cal.read_calfits(fname)
        gains, flags = io.load_cal(cal)
        self.assertEqual(len(gains.keys()),18)
        self.assertEqual(len(flags.keys()),18)

        fname_xx = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.xx.HH.uvc.omni.calfits")
        fname_yy = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.yy.HH.uvc.omni.calfits")
        gains, flags, quals, total_qual, ants, freqs, times, pols = io.load_cal([fname_xx,fname_yy], return_meta=True)
        self.assertEqual(len(gains.keys()),36)
        self.assertEqual(len(flags.keys()),36)
        self.assertEqual(len(quals.keys()),36)
        self.assertEqual(freqs.shape, (1, 1024))
        self.assertEqual(times.shape, (3,))
        self.assertEqual(sorted(pols), ['x','y'])

        cal_xx, cal_yy = UVCal(), UVCal()
        cal_xx.read_calfits(fname_xx)
        cal_yy.read_calfits(fname_yy)
        gains, flags, quals, total_qual, ants, freqs, times, pols = io.load_cal([cal_xx,cal_yy], return_meta=True)
        self.assertEqual(len(gains.keys()),36)
        self.assertEqual(len(flags.keys()),36)
        self.assertEqual(len(quals.keys()),36)
        self.assertEqual(freqs.shape, (1, 1024))
        self.assertEqual(times.shape, (3,))
        self.assertEqual(sorted(pols), ['x','y'])


    #TODO: implement this test
    def test_write_cal(self):
        with self.assertRaises(NotImplementedError):
            io.write_cal()


    def test_update_cal(self):
        # load in cal
        fname = os.path.join(DATA_PATH, "test_input/zen.2457698.40355.xx.HH.uvc.omni.calfits")
        outname = os.path.join(DATA_PATH, "test_output/zen.2457698.40355.xx.HH.uvc.modified.calfits.")
        cal = UVCal()
        cal.read_calfits(fname)
        gains, flags, quals, total_qual, ants, freqs, times, pols = io.load_cal(fname, return_meta=True)

        #make some modifications
        new_gains = {key: (2.+1.j)*val for key,val in gains.items()}
        new_flags = {key: np.logical_not(val) for key,val in flags.items()}
        new_quals = {key: 2.*val for key,val in quals.items()}
        io.update_cal(fname, outname, gains=new_gains, flags=new_flags, quals=new_quals,
                      add_to_history='hello world', clobber=True, telescope_name='Super HERA')
        
        #test modifications
        gains, flags, quals, total_qual, ants, freqs, times, pols = io.load_cal(outname, return_meta=True)
        for k in gains.keys():
            self.assertTrue(np.all(new_gains[k] == gains[k]))
            self.assertTrue(np.all(new_flags[k] == flags[k]))
            self.assertTrue(np.all(new_quals[k] == quals[k]))
        cal2 = UVCal()
        cal2.read_calfits(outname)
        self.assertEqual(cal2.history.replace('\n',''), ('hello world' + cal.history).replace('\n',''))
        self.assertEqual(cal2.telescope_name,'Super HERA')
        os.remove(outname)

        #now try the same thing but with a UVCal object instead of path
        io.update_cal(cal, outname, gains=new_gains, flags=new_flags, quals=new_quals,
                      add_to_history='hello world', clobber=True, telescope_name='Super HERA')
        gains, flags, quals, total_qual, ants, freqs, times, pols = io.load_cal(outname, return_meta=True)
        for k in gains.keys():
            self.assertTrue(np.all(new_gains[k] == gains[k]))
            self.assertTrue(np.all(new_flags[k] == flags[k]))
            self.assertTrue(np.all(new_quals[k] == quals[k]))
        cal2 = UVCal()
        cal2.read_calfits(outname)
        self.assertEqual(cal2.history.replace('\n',''), ('hello world' + cal.history).replace('\n',''))
        self.assertEqual(cal2.telescope_name,'Super HERA')
        os.remove(outname)



if __name__ == '__main__':
    unittest.main()

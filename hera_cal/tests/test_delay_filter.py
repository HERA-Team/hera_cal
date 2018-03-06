import hera_cal.delay_filter as df
from hera_cal import io
import numpy as np
import unittest
from copy import deepcopy
from pyuvdata import UVCal, UVData
from hera_cal.data import DATA_PATH
import os
import shutil
from scipy import constants
from uvtools.dspec import delay_filter

class Test_Delay_Filter(unittest.TestCase):

    def test_init(self):
        dfil = df.Delay_Filter()
        self.assertFalse(dfil.writable)

    def test_load_data(self):
        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        dfil = df.Delay_Filter()
        dfil.load_data(fname)
        self.assertEqual(dfil.data[(24, 25, 'xx')].shape, (60, 64))
        self.assertEqual(dfil.flags[(24, 25, 'xx')].shape, (60, 64))
        self.assertEqual(len(dfil.times), 60)
        self.assertEqual(len(dfil.antpos), 7)
        self.assertEqual(type(dfil.antpos[24]), np.ndarray)
        self.assertEqual(len(dfil.antpos[24]), 3)
        self.assertEqual(len(dfil.freqs), 64)
        self.assertTrue(dfil.writable)

        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        uvd = UVData()
        uvd.read_miriad(fname)
        dfil = df.Delay_Filter()
        dfil.load_data(uvd)
        self.assertEqual(dfil.data[(24, 25, 'xx')].shape, (60, 64))
        self.assertEqual(dfil.flags[(24, 25, 'xx')].shape, (60, 64))
        self.assertEqual(len(dfil.times), 60)
        self.assertEqual(len(dfil.antpos), 7)
        self.assertEqual(type(dfil.antpos[24]), np.ndarray)
        self.assertEqual(len(dfil.antpos[24]), 3)
        self.assertEqual(len(dfil.freqs), 64)
        self.assertTrue(dfil.writable)

        filename1 = os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA')
        filename2 = os.path.join(DATA_PATH, 'zen.2458043.13298.xx.HH.uvORA')
        dfil = df.Delay_Filter()
        dfil.load_data([filename1,filename2])
        self.assertFalse(dfil.writable)


    def test_load_data_as_dicts(self):
        dfil = df.Delay_Filter()
        dfil.load_data_as_dicts(None,None,None,None)
        self.assertTrue(hasattr(dfil,'data'))
        self.assertTrue(hasattr(dfil,'flags'))
        self.assertTrue(hasattr(dfil,'freqs'))
        self.assertTrue(hasattr(dfil,'antpos'))
        self.assertFalse(dfil.writable)

    def test_run_filter(self):
        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        k = (24,25,'xx')
        dfil = df.Delay_Filter()
        dfil.load_data(fname)
        bl = np.linalg.norm(dfil.antpos[24] - dfil.antpos[25]) / constants.c * 1e9
        sdf = (dfil.freqs[1]-dfil.freqs[0])/1e9
        
        dfil.run_filter(to_filter=[k], standoff=0., horizon=1., tol=1e-9, window='none', skip_wgt=0.1, maxiter=100)
        d_mdl, d_res, info = delay_filter(dfil.data[k], np.logical_not(dfil.flags[k]), bl, sdf, standoff=0., horizon=1., tol=1e-9, window='none', skip_wgt=0.1, maxiter=100)
        np.testing.assert_almost_equal(d_mdl, dfil.CLEAN_models[k])
        np.testing.assert_almost_equal(d_res, dfil.filtered_residuals[k])

        dfil.run_filter(to_filter=[k], weight_dict={k: np.ones_like(dfil.flags[k])}, standoff=0., horizon=1., tol=1e-9, window='none', skip_wgt=0.1, maxiter=100)
        d_mdl, d_res, info = delay_filter(dfil.data[k], np.ones_like(dfil.flags[k]), bl, sdf, standoff=0., horizon=1., tol=1e-9, window='none', skip_wgt=0.1, maxiter=100)
        np.testing.assert_almost_equal(d_mdl, dfil.CLEAN_models[k])
        np.testing.assert_almost_equal(d_res, dfil.filtered_residuals[k])

        dfil.run_filter()
        for k in dfil.data.keys():
            self.assertEqual(dfil.filtered_residuals[k].shape, (60,64))
            self.assertEqual(dfil.CLEAN_models[k].shape, (60,64))
            self.assertTrue(dfil.info.has_key(k))
        

    def test_write_filtered_data(self):
        dfil = df.Delay_Filter()
        with self.assertRaises(NotImplementedError):
            dfil.write_filtered_data('')

        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        dfil = df.Delay_Filter()
        dfil.load_data(fname)
        data = dfil.data
        dfil.run_filter(standoff=0., horizon=1., tol=1e-9, window='none', skip_wgt=0.1, maxiter=100)
        outfilename = os.path.join(DATA_PATH, 'test_output/zen.2458043.12552.xx.HH.filter_test.uvORAD')
        dfil.write_filtered_data(outfilename, add_to_history='Hello_world.', clobber=True, telescope_name='PAPER')

        uvd = UVData()
        uvd.read_miriad(outfilename)
        self.assertEqual(uvd.history.replace('\n','').replace(' ','')[-12:], 'Hello_world.')
        self.assertEqual(uvd.telescope_name, 'PAPER')
        
        filtered_residuals, flags = io.load_vis(uvd)
        dfil.write_filtered_data(outfilename, write_CLEAN_models=True, clobber=True)
        CLEAN_models, flags = io.load_vis(outfilename)
        for k in data.keys():
            np.testing.assert_array_almost_equal(dfil.CLEAN_models[k], CLEAN_models[k])
            np.testing.assert_array_almost_equal(dfil.filtered_residuals[k], filtered_residuals[k])
            np.testing.assert_array_almost_equal(data[k][~flags[k]], (CLEAN_models[k] + filtered_residuals[k])[~flags[k]], 5)
        shutil.rmtree(outfilename)


if __name__ == '__main__':
    unittest.main()

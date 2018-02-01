import numpy as np
import hera_cal.abscal as abscal
from pyuvdata import UVData
from hera_cal.datacontainer import DataContainer
from collections import OrderedDict as odict
from uvtools.dspec import delay_filter
from copy import deepcopy
from scipy import constants

class Delay_Filter():

    def __init__(self, data, flags=None, freqs=None, antpos=None):
        '''TODO: Document.
        Assumes antpos in meters and freqs in Hz'''
        self.writable = False

    def read_files(self, datafiles, filetype='miriad'):
        '''TODO: Document.'''
        self.writable = True
        self.datafiles, self.filetype = datafiles, filetype
        self.data, self.flags, self.antpos, _, self.freqs, _, _, _ = abscal.UVData2AbsCalDict(datafiles, pop_autos=False, return_meta=True, filetype=filetype)

    def load_UVData(self, uvdata, hold_uvdata=False):
        '''TODO: Document.'''
        if hold_uvdata:
            self.writable = True
            self.uvdata = uvdata
        self.data, self.flags, self.antpos, _, self.freqs, _, _, _ = abscal.UVData2AbsCalDict(uvdata, pop_autos=False, return_meta=True)

    def load_dicts(self, data, flags, freqs, antpos):
        '''TODO: Document.'''
        self.data, self.flags, self.freqs, self.antpos = data, flags, freqs, antpos
        

    def run_filter(self, to_filter=[], weight_dict=None, horizon=1., standoff=0., tol=1e-9, window='none', skip_wgt=0.1, maxiter=100):
        '''TODO: Document.'''
        self.filtered_residuals = deepcopy(self.data)
        self.CLEAN_models = deepcopy(self.data)
        self.info = {}
        if len(to_filter) == 0:
            to_filter = self.data.keys()
        
        for k in to_filter:
            bl_len = np.linalg.norm(self.antpos[k[0]] - self.antpos[k[1]]) / constants.c * 1e9 #in ns
            sdf = np.median(np.diff(self.freqs)) / 1e9 #in GHz
            if weight_dict is not None:
                wgts = weight_dict[k]
            else:
                wgts = np.logical_not(self.flags[k])
            
            d_mdl, d_res, info = delay_filter(self.data[k], wgts, bl_len, sdf, standoff=standoff, horizon=horizon, tol=tol, window=window, skip_wgt=skip_wgt, maxiter=maxiter)
            self.filtered_residuals[k] = d_res
            self.CLEAN_models[k] = d_mdl
            self.info[k] = info

    def write_filtered_data(self, outfilenames, write_CLEAN_models=False):
        '''TODO: Document.'''
        if not self.writable:
            raise NotImplementedError('Writing functionality requires list of input files or a stored UVData object')
        if hasattr(self, 'uvdata'):
            raise NotImplementedError('blah')
        else:
            raise NotImplementedError('blah')



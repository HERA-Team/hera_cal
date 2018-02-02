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
        '''Class for loading data, performing uvtools.dspec.delay_filter, and writing out data using pyuvdata.'''
        self.writable = False

    def read_files(self, datafiles, filetype='miriad'):
        '''Reads in and stores data in miriad or uvfits format for delay filtering. 

        Arguments:
            datafiles: string or list of strings indicating the path to data to load in
            filetype: file format of data. Options: 'miriad' (default)  or 'uvfits'.
        '''
        self.writable = True
        self.datafiles, self.filetype = datafiles, filetype
        self.data, self.flags, self.antpos, _, self.freqs, self.times, _, _ = abscal.UVData2AbsCalDict(datafiles, pop_autos=False, return_meta=True, filetype=filetype)


    def load_UVData(self, uvdata):
        '''Loads in UVData object(s) for performing delay filtering.

        Arguments:
            datafiles: UVData object or list of UVData objects
        '''
        self.writable = True
        self.uvdata = uvdata
        self.data, self.flags, self.antpos, _, self.freqs, self.times, _, _ = abscal.UVData2AbsCalDict(uvdata, pop_autos=False, return_meta=True)


    def load_dicts(self, data, flags, freqs, antpos):
        '''Loads in data manually as a dictionary, an ordered dictionary, or a DataContainer.

        Arguments:
            data: visibility data as a dictionary or DataContainer.
            flags: flags with the same format and keys as data
            freqs: array of frequencies in Hz
            antpos: dictionary mapping antenna index to antenna position in m
        '''
        self.data, self.flags, self.freqs, self.antpos = data, flags, freqs, antpos
        

    def run_filter(self, to_filter=[], weight_dict=None, horizon=1., standoff=0., tol=1e-9, window='none', skip_wgt=0.1, maxiter=100):
        '''Performs uvtools.dspec.Delay_Filter on (a subset of) the data stored in the object.
        Uses stored flags unless explicitly overridden with weight_dict.
    
        Arguments:
            to_filter: list of visibilities to filter in the (i,j,pol) format. 
                If [] (the default), all visibilities are filtered.
            weight_dict: dictionary or DataContainer with all the same keys as self.data. 
                Multiplicative weights to use for the delay filter. Default, use logical not of self.flags
            horizon: proportionality constant for bl_len where 1 is the horizon (full light travel time)
            tol: CLEAN algorithm convergence tolerance (see aipy.deconv.clean)
            window: window function for filtering applied to the filtered axis. 
                See aipy.dsp.gen_window for options.        
            skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt) 
                Only works properly when all weights are all between 0 and 1. 
            maxiter: Maximum number of iterations for aipy.deconv.clean to converge.
        '''
        self.filtered_residuals = deepcopy(self.data)
        self.CLEAN_models = deepcopy(self.data)
        self.info = {}
        if to_filter == []:
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


    def write_filtered_data(self, outfilenames, append_to_history = '', write_CLEAN_models=False):
        '''TODO: Document.'''
        if not self.writable:
            raise NotImplementedError('Writing functionality requires list of input files or a stored UVData object')
        if hasattr(self, 'uvdata'):
            raise NotImplementedError('blah')
        else:
            raise NotImplementedError('blah')



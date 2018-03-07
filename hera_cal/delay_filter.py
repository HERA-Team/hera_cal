import numpy as np
from hera_cal import io
from pyuvdata import UVData
from hera_cal.datacontainer import DataContainer
from collections import OrderedDict as odict
from uvtools.dspec import delay_filter
from copy import deepcopy
from scipy import constants

class Delay_Filter():

    def __init__(self):
        '''Class for loading data, performing uvtools.dspec.delay_filter, and writing out data using pyuvdata.
        To use, run either self.load_data() or self.load_dicts(), then self.run_filter(). If data is loaded with a single 
        string path or a single UVData object, it can be written to a new file using self.write_filtered_data().
        '''
        self.writable = False


    def load_data(self, input_data, filetype='miriad'):
        '''Loads in and stores data for delay filtering. 

        Arguments:
            input_data: data file path, or UVData instance, or list of either strings of data file paths 
                or list of UVData instances to concatenate into a single internal DataContainer
            filetype: file format of data. Default 'miriad.' Ignored if input_data is UVData object(s).
        '''
        if isinstance(input_data, (str,UVData)):
            self.writable = True
            self.input_data, self.filetype = input_data, filetype
        self.data, self.flags, self.antpos, _, self.freqs, self.times, _, _ = io.load_vis(input_data, return_meta=True, filetype=filetype)


    def load_data_as_dicts(self, data, flags, freqs, antpos):
        '''Loads in data manually as a dictionary, an ordered dictionary, or a DataContainer.

        Arguments:
            data: visibility data as a dictionary or DataContainer.
            flags: flags with the same format and keys as data
            freqs: array of frequencies in Hz
            antpos: dictionary mapping antenna index to antenna position in m
        '''
        self.data, self.flags, self.freqs, self.antpos = data, flags, freqs, antpos
        

    def run_filter(self, to_filter=[], weight_dict=None, standoff=0., horizon=1.,tol=1e-9, window='none', skip_wgt=0.1, maxiter=100):
        '''Performs uvtools.dspec.Delay_Filter on (a subset of) the data stored in the object.
        Uses stored flags unless explicitly overridden with weight_dict.
    
        Arguments:
            to_filter: list of visibilities to filter in the (i,j,pol) format. 
                If [] (the default), all visibilities are filtered.
            weight_dict: dictionary or DataContainer with all the same keys as self.data. 
                Linear multiplicative weights to use for the delay filter. Default, use np.logical_not
                of self.flags. uvtools.dspec.Delay_Filter will renormalize to compensate
            standoff: fixed additional delay beyond the horizon (in ns)
            horizon: proportionality constant for bl_len where 1 is the horizon (full light travel time)
            tol: CLEAN algorithm convergence tolerance (see aipy.deconv.clean)
            window: window function for filtering applied to the filtered axis. 
                See aipy.dsp.gen_window for options.        
            skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt) 
                Only works properly when all weights are all between 0 and 1. 
            maxiter: Maximum number of iterations for aipy.deconv.clean to converge.

        Results are stored in:
            self.filtered_residuals: DataContainer formatted like self.data with only high-delay components
            self.CLEAN_models: DataContainer formatted like self.data with only low-delay components 
            self.info: Dictionary of info from uvtools.dspec.delay_filter with the same keys as self.data
        '''
        self.filtered_residuals = deepcopy(self.data)
        self.CLEAN_models = DataContainer({k: np.zeros_like(self.data.values()[0]) for k in self.data.keys()})
        self.info = odict()
        if to_filter == []:
            to_filter = self.data.keys()
        
        for k in to_filter:
            bl_len = np.linalg.norm(self.antpos[k[0]] - self.antpos[k[1]]) / constants.c * 1e9 #in ns
            sdf = np.median(np.diff(self.freqs)) / 1e9 #in GHz
            if weight_dict is not None:
                wgts = weight_dict[k]
            else:
                wgts = np.logical_not(self.flags[k])
            
            d_mdl, d_res, info = delay_filter(self.data[k], wgts, bl_len, sdf, standoff=standoff, horizon=horizon, 
                                              tol=tol, window=window, skip_wgt=skip_wgt, maxiter=maxiter)
            self.filtered_residuals[k] = d_res
            self.CLEAN_models[k] = d_mdl
            self.info[k] = info


    def write_filtered_data(self, outfilename, filetype_out='miriad', add_to_history = '', 
                            clobber = False, write_CLEAN_models=False, **kwargs):
        '''Writes high-pass filtered data to disk, using input (which must be either 
        a single path or a single UVData object) as a template.
        
        Arguments:
            outfilename: filename of the filtered visibility file to be written to disk
            filetype_out: file format of output result. Default 'miriad.'
            add_to_history: string appended to the history of the output file
            clobber: if True, overwrites existing file at outfilename
            write_CLEAN_models: if True, save the low-pass filtered CLEAN model rather 
                than the high-pass filtered residual
            kwargs: addtional keyword arguments update the UVData object before saving. 
                Must be valid UVData object attributes.
        '''
        if not self.writable:
            raise NotImplementedError('Writing functionality requires that the input be a single file path string or a single UVData object.')
        else:
            if write_CLEAN_models:
                data_out = self.CLEAN_models
            else:
                data_out = self.filtered_residuals
            io.update_vis(self.input_data, outfilename, filetype_in=self.filetype, filetype_out=filetype_out, data=data_out, 
                          add_to_history=add_to_history, clobber=clobber, **kwargs)

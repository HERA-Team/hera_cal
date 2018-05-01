import numpy as np
from hera_cal import io
from pyuvdata import UVData
from hera_cal.datacontainer import DataContainer
from collections import OrderedDict as odict
from uvtools.dspec import delay_filter
from copy import deepcopy
from scipy import constants
import argparse
import datetime

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
        self.Nfreqs = len(self.freqs)

    def load_data_as_dicts(self, data, flags, freqs, antpos):
        '''Loads in data manually as a dictionary, an ordered dictionary, or a DataContainer.

        Arguments:
            data: visibility data as a dictionary or DataContainer.
            flags: flags with the same format and keys as data
            freqs: array of frequencies in Hz
            antpos: dictionary mapping antenna index to antenna position in m
        '''
        self.data, self.flags, self.freqs, self.antpos = data, flags, freqs, antpos


    def run_filter(self, to_filter=[], weight_dict=None, standoff=15., horizon=1., tol=1e-9, 
                   window='blackman-harris', skip_wgt=0.1, maxiter=100, verbose=False, 
                   flag_nchan_low=0, flag_nchan_high=0):
        '''Performs uvtools.dspec.Delay_Filter on (a subset of) the data stored in the object.
        Uses stored flags unless explicitly overridden with weight_dict.

        Arguments:
            to_filter: list of visibilities to filter in the (i,j,pol) format.
                If [] (the default), all visibilities are filtered.
            weight_dict: dictionary or DataContainer with all the same keys as self.data.
                Linear multiplicative weights to use for the delay filter. Default, use np.logical_not
                of self.flags. uvtools.dspec.delay_filter will renormalize to compensate
            standoff: fixed additional delay beyond the horizon (in ns)
            horizon: proportionality constant for bl_len where 1 is the horizon (full light travel time)
            tol: CLEAN algorithm convergence tolerance (see aipy.deconv.clean)
            window: window function for filtering applied to the filtered axis.
                See aipy.dsp.gen_window for options.
            skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
                Model is left as 0s, residual is left as data, and info is {'skipped': True} for that 
                time. Skipped channels are then flagged in self.flags.
                Only works properly when all weights are all between 0 and 1.
            maxiter: Maximum number of iterations for aipy.deconv.clean to converge.
            verbose: If True print feedback to stdout
            flag_nchan_low: Integer number of channels to flag on lower band edge before filtering
            flag_nchan_low: Integer number of channels to flag on upper band edge before filtering

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
            if verbose:
                print "\nStarting filter on {} at {}".format(k, str(datetime.datetime.now()))
            bl_len = np.linalg.norm(self.antpos[k[0]] - self.antpos[k[1]]) / constants.c * 1e9 #in ns
            sdf = np.median(np.diff(self.freqs)) / 1e9 #in GHz
            if weight_dict is not None:
                wgts = weight_dict[k]
            else:
                wgts = np.logical_not(self.flags[k])

            # edge flag
            if flag_nchan_low > 0:
                self.flags[k][:, :flag_nchan_low] = True
                wgts[:, :flag_nchan_low] = 0.0
            if flag_nchan_high > 0:
                self.flags[k][:, -flag_nchan_high:] = True
                wgts[:, -flag_nchan_high:] = 0.0                

            d_mdl, d_res, info = delay_filter(self.data[k], wgts, bl_len, sdf, standoff=standoff, horizon=horizon,
                                              tol=tol, window=window, skip_wgt=skip_wgt, maxiter=maxiter)
            self.filtered_residuals[k] = d_res
            self.CLEAN_models[k] = d_mdl
            self.info[k] = info
            # Flag all channels for any time when skip_wgt gets triggered
            for i, info_dict in enumerate(info):
                if info_dict.get('skipped', False):
                    self.flags[k][i,:] = np.ones_like(self.flags[k][i,:])

    def get_filled_data(self):
        """Get original data with flagged pixels filled with CLEAN_models

        Returns
            filled_data: DataContainer with original data and flags filled with CLEAN model
            filled_flgs: DataContainer with flags set to False unless time is skipped
        """
        assert hasattr(self, 'CLEAN_models') and hasattr(self, 'data') and hasattr(self, 'flags'), "self.CLEAN_models, "\
            "self.data and self.flags must all exist to get filled data"
        # construct filled data and filled flags
        filled_data = deepcopy(self.data)
        filled_flgs = deepcopy(self.flags)

        # iterate over filled_data keys
        for k in filled_data.keys():
            # get flags
            f = filled_flgs[k].copy()
            # if flagged across all freqs, "unflag" this f
            f[np.sum(f, axis=1) / float(self.Nfreqs) > 0.99999] = False
            # replace data_out with CLEAN_models at f == True
            filled_data[k][f] = self.CLEAN_models[k][f]
            # unflag at f == True
            filled_flgs[k][f] = False

        return filled_data, filled_flgs

    def write_filtered_data(self, outfilename, filetype_out='miriad', add_to_history='',
                            clobber = False, write_CLEAN_models=False, write_filled_data=False, 
                            **kwargs):
        '''Writes high-pass filtered data to disk, using input (which must be either
        a single path or a single UVData object) as a template.

        Arguments:
            outfilename: filename of the filtered visibility file to be written to disk
            filetype_out: file format of output result. Default 'miriad.'
            add_to_history: string appended to the history of the output file
            clobber: if True, overwrites existing file at outfilename
            write_CLEAN_models: if True, save the low-pass filtered CLEAN model rather
                than the high-pass filtered residual
            write_filled_data: if True, save the original data with its flags filled in
                with the CLEAN model rather writing the high-pass filtered residual.
            kwargs: addtional keyword arguments update the UVData object before saving.
                Must be valid UVData object attributes.
        '''
        if not self.writable:
            raise NotImplementedError('Writing functionality requires that the input be a single file path string or a single UVData object.')
        else:
            if write_CLEAN_models:
                assert not write_filled_data, "cannot choose both write_CLEAN_models and write_filled_data"
                data_out = self.CLEAN_models
                flags_out = self.flags
            elif write_filled_data:
                assert not write_CLEAN_models, "cannot choose both write_CLEAN_models and write_filled_data"
                # construct filled data and filled flags
                data_out, flags_out = self.get_filled_data()
            else:
                data_out = self.filtered_residuals
                flags_out = self.flags
            io.update_vis(self.input_data, outfilename, filetype_in=self.filetype, filetype_out=filetype_out, 
                          data=data_out, flags=flags_out, add_to_history=add_to_history, clobber=clobber, 
                          **kwargs)


def delay_filter_argparser():
    '''Arg parser for commandline operation of hera_cal.delay_filter.'''
    a = argparse.ArgumentParser(description="Perform delay filter of visibility data.")
    a.add_argument("infile", type=str, help="path to visibility data file to delay filter")
    a.add_argument("outfile", nargs='?', default=None, type=str, help="path to new visibility results file. "
                   "Note that if outfile is not supplied, the parser defaults to None, but delay_filter_run.py "
                   "will override this default, and instead sets outfile to infile + 'D'.")
    a.add_argument("--filetype", type=str, default='miriad', help='filetype of input and output data files (default "miriad")')
    a.add_argument("--write_model", default=False, action="store_true", help="Write the low-pass filtered CLEAN model rather"\
                   "than the high-pass filtered residual or the flag-filled data.")
    a.add_argument("--write_filled", default=False, action='store_true', help='Write the original data with its flags filled with '\
                    "the CLEAN model instead of residual or CLEAN model.")
    a.add_argument("--write_all", default=False, action='store_true', help="Write the high-pass residual, the low-pass "\
                    "CLEAN model and the flag-filled original data. The residual gets the outfile name (or default 'D' extension), " \
                    "and the model and original data get the 'M' and 'F' extension on top of the residual filename respectively.")
    a.add_argument("--flag_nchan_low", default=0, type=int, help="Number of channels to flag on lower band edge before filtering.")
    a.add_argument("--flag_nchan_high", default=0, type=int, help="Number of channels to flag on upper band edge before filtering.")
    a.add_argument("--clobber", default=False, action="store_true", help='overwrites existing file at outfile')

    filt_options = a.add_argument_group(title='Options for the delay filter')
    filt_options.add_argument("--standoff", type=float, default=15.0, help='fixed additional delay beyond the horizon (default 15 ns)')
    filt_options.add_argument("--horizon", type=float, default=1.0, help='proportionality constant for bl_len where 1.0 (default) is the horizon\
                              (full light travel time)')
    filt_options.add_argument("--tol", type=float, default=1e-9, help='CLEAN algorithm convergence tolerance (default 1e-9)')
    filt_options.add_argument("--window", type=str, default='blackman-harris', help='window function for frequency filtering (default "blackman-harris",\
                              see aipy.dsp.gen_window for options')
    filt_options.add_argument("--skip_wgt", type=float, default=0.1, help='skips filtering and flags times with unflagged fraction ~< skip_wgt (default 0.1)')
    filt_options.add_argument("--maxiter", type=int, default=100, help='maximum iterations for aipy.deconv.clean to converge (default 100)')

    return a

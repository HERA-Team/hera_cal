# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

"""Module for delay filtering data and related operations."""

from __future__ import absolute_import, division, print_function
import numpy as np
from hera_cal import io, apply_cal
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
        To use, run either self.load_data() or self.load_dicts(), then self.run_filter(). If data is loaded from
        disk or with a UVData/HERAData object(s), it can be written to a new file using self.write_filtered_data().
        '''
        self.writable = False

    def load_data(self, input_data, filetype='uvh5', input_cal=None, **read_kwargs):
        '''Loads in and stores data for delay filtering.

        Arguments:
            input_data: data file path, or UVData/HERAData instance, or list of either strings of data file
                paths or list of UVData/HERAData instances to concatenate into a single internal DataContainer
            filetype: file format of data. Default 'uvh5.' Ignored if input_data is UVData/HERAData object(s).
            input_cal: calibration to apply to data before delay filtering. Could be a string path to a calfits file,
                a UVCal/HERACal object, or a list of either.
            read_kwargs: kwargs to be passed into HERAData.read() for partial data loading
        '''
        # load data into data containers
        self.hd = io.to_HERAData(input_data, filetype=filetype)
        if self.hd.data_array is not None:
            self.data, self.flags, _ = self.hd.build_datacontainers()
        else:
            self.data, self.flags, _ = self.hd.read(**read_kwargs)

        # optionally apply calibration and calibration flags
        if input_cal is not None:
            g, f = io.load_cal(input_cal)
            apply_cal.recalibrate_in_place(self.data, self.flags, g, f)

        # save metadata
        self.antpos = deepcopy(self.data.antpos)
        self.freqs = deepcopy(self.data.freqs)
        self.Nfreqs = len(self.freqs)
        self.writable = True

    def load_data_as_dicts(self, data, flags, freqs, antpos):
        '''Loads in data manually as a dictionary, an ordered dictionary, or a DataContainer.

        Arguments:
            data: visibility data as a dictionary or DataContainer.
            flags: flags with the same format and keys as data
            freqs: array of frequencies in Hz
            antpos: dictionary mapping antenna index to antenna position in m
        '''
        self.data, self.flags, self.freqs, self.antpos = data, flags, freqs, antpos

    def run_filter(self, to_filter=[], weight_dict=None, standoff=15., horizon=1., min_dly=0.0,
                   tol=1e-9, window='blackman-harris', skip_wgt=0.1, maxiter=100, verbose=False,
                   flag_nchan_low=0, flag_nchan_high=0, gain=0.1, **win_kwargs):
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
            min_dly: minimum delay used for cleaning [ns]: if bl_len * horizon + standoff < min_dly, use min_dly.
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
            gain: The fraction of a residual used in each iteration. If this is too low, clean takes
                unnecessarily long. If it is too high, clean does a poor job of deconvolving.
            win_kwargs : keyword arguments to feed aipy.dsp.gen_window()

        Results are stored in:
            self.filtered_residuals: DataContainer formatted like self.data with only high-delay components
            self.CLEAN_models: DataContainer formatted like self.data with only low-delay components
            self.info: Dictionary of info from uvtools.dspec.delay_filter with the same keys as self.data
        '''
        self.filtered_residuals = deepcopy(self.data)
        self.CLEAN_models = DataContainer({k: np.zeros_like(list(self.data.values())[0]) for k in self.data.keys()})
        self.info = odict()
        if to_filter == []:
            to_filter = self.data.keys()

        for k in to_filter:
            if verbose:
                print("\nStarting filter on {} at {}".format(k, str(datetime.datetime.now())))
            bl_len = np.linalg.norm(self.antpos[k[0]] - self.antpos[k[1]]) / constants.c * 1e9  # in ns
            sdf = np.median(np.diff(self.freqs)) / 1e9  # in GHz
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

            d_mdl, d_res, info = delay_filter(self.data[k], wgts, bl_len, sdf, standoff=standoff, horizon=horizon, min_dly=min_dly,
                                              tol=tol, window=window, skip_wgt=skip_wgt, maxiter=maxiter, gain=gain, **win_kwargs)
            self.filtered_residuals[k] = d_res
            self.CLEAN_models[k] = d_mdl
            self.info[k] = info
            # Flag all channels for any time when skip_wgt gets triggered
            for i, info_dict in enumerate(info):
                if info_dict.get('skipped', False):
                    self.flags[k][i, :] = np.ones_like(self.flags[k][i, :])

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

    def write_filtered_data(self, res_outfilename=None, CLEAN_outfilename=None, filled_outfilename=None, filetype='uvh5',
                            partial_write=False, clobber=False, add_to_history='', **kwargs):
        '''Method for writing filtered residuals, CLEAN models, and/or original data with flags filled
        by CLEAN models where possible. Uses input_data from Delay_Filter.load_data() as a template.

        Arguments:
            res_outfilename: path for writing the filtered visibilities with flags 
            CLEAN_outfilename: path for writing the CLEAN model visibilities (with the same flags)
            filled_outfilename: path for writing the original data but with flags unflagged and replaced
                with CLEAN models wherever possible
            filetype: file format of output result. Default 'uvh5.' Also supports 'miriad' and 'uvfits'.
            partial_write: use uvh5 partial writing capability (only works when going from uvh5 to uvh5)
            clobber: if True, overwrites existing file at the outfilename
            add_to_history: string appended to the history of the output file
            kwargs: addtional UVData keyword arguments update the before saving.
                Must be valid UVData object attributes.
        '''
        if not self.writable:
            raise ValueError('Writing functionality only enabled by running Delay_Filter.load_data()')
        if (res_outfilename is None) and (CLEAN_outfilename is None) and (filled_outfilename is None):
            raise ValueError('You must specifiy at least one outfilename.')
        else:
            # loop over the three output modes if a corresponding outfilename is supplied
            for mode, outfilename in zip(['residual', 'CLEAN', 'filled'], 
                                         [res_outfilename, CLEAN_outfilename, filled_outfilename]):
                if outfilename is not None:
                    if mode == 'residual':
                        data_out, flags_out = self.filtered_residuals, self.flags
                    elif mode == 'CLEAN':
                        data_out, flags_out = self.CLEAN_models, self.flags
                    elif mode == 'filled':
                        data_out, flags_out = self.get_filled_data()
                    if partial_write:
                        if not ((filetype == 'uvh5') and (getattr(self.hd, 'filetype', None) == 'uvh5')):
                            raise NotImplementedError('Partial writing requires input and output types to be "uvh5".')
                        hd.partial_write(outfilename, data=data_out, flags=flags_out, clobber=clobber, 
                                         add_to_history=add_to_history, **kwargs)
                    else:
                        io.update_vis(self.hd, outfilename, filetype_out=filetype, data=data_out, flags=flags_out, 
                                      add_to_history=add_to_history, clobber=clobber, **kwargs)


def partial_load_delay_filter_and_write(self, infilename, calfile=None, Nbls=1,
                                        res_outfilename=None, CLEAN_outfilename=None, filled_outfilename=None,
                                        clobber=False, add_to_history='', **filter_kwargs):
    '''Function using partial data loading and writing to perform delay filtering.

    Arguments:
        infilename: string path to data to uvh5 file to load
        cal: optional string path to calibration file to apply to data before delay filtering
        Nbls: the number of baselines to load at once.
        res_outfilename: path for writing the filtered visibilities with flags 
        CLEAN_outfilename: path for writing the CLEAN model visibilities (with the same flags)
        filled_outfilename: path for writing the original data but with flags unflagged and replaced
            with CLEAN models wherever possible
        clobber: if True, overwrites existing file at the outfilename
        add_to_history: string appended to the history of the output file
        filter_kwargs: additional keyword arguments to be passed to Delay_Filter.run_filter()
    '''
    hd = HERAData(infilename, filetype='uvh5')
    if calfile is not None:
        calfile = HERACal(calfile)
        calfile.read()
    # loop over all baselines in increments of Nbls
    for i in range(0, len(hd.bls), Nbls):
        df = Delay_Filter()
        df.load_data(hd, input_cal=calfile, bls=bls[i:i + Nbls])
        df.run_filter(**filter_kwargs)
        df.write_filtered_data(res_outfilename=res_outfilename, CLEAN_outfilename=CLEAN_outfilename,
                               filled_outfilename=filled_outfilename, partial_write=True,
                               clobber=clobber, add_to_history=add_to_history)
        del df.hd.data_array  # this forces a reload in the next loop


def delay_filter_argparser():
    '''Arg parser for commandline operation of hera_cal.delay_filter.'''
    a = argparse.ArgumentParser(description="Perform delay filter of visibility data.")
    a.add_argument("infilename", type=str, help="path to visibility data file to delay filter")
    a.add_argument("--filetype_in", type=str, default='uvh5', help='filetype of input data files (default "uvh5")')
    a.add_argument("--filetype_out", type=str, default='uvh5', help='filetype for output data files (default "uvh5")')
    a.add_argument("--calfile", default=None, type=str, help="optional string path to calibration file to apply to data before delay filtering")
    a.add_argument("--partial_load_Nbls", default=None, type=int, help="the number of baselines to load at once (default None means load full data")
    a.add_argument("--res_outfilename", default=None, type=str, help="path for writing the filtered visibilities with flags")
    a.add_argument("--CLEAN_outfilename", default=None, type=str, help="path for writing the CLEAN model visibilities (with the same flags)")
    a.add_argument("--filled_outfilename", default=None, type=str, help="path for writing the original data but with flags unflagged and replaced with CLEAN models wherever possible")
    a.add_argument("--clobber", default=False, action="store_true", help='overwrites existing file at outfile')

    filt_options = a.add_argument_group(title='Options for the delay filter')
    filt_options.add_argument("--standoff", type=float, default=15.0, help='fixed additional delay beyond the horizon (default 15 ns)')
    filt_options.add_argument("--horizon", type=float, default=1.0, help='proportionality constant for bl_len where 1.0 (default) is the horizon\
                              (full light travel time)')
    filt_options.add_argument("--min_dly", type=float, default=0.0, help="A minimum delay threshold [ns] used for cleaning.")
    filt_options.add_argument("--tol", type=float, default=1e-9, help='CLEAN algorithm convergence tolerance (default 1e-9)')
    filt_options.add_argument("--window", type=str, default='blackman-harris', help='window function for frequency filtering (default "blackman-harris",\
                              see aipy.dsp.gen_window for options')
    filt_options.add_argument("--skip_wgt", type=float, default=0.1, help='skips filtering and flags times with unflagged fraction ~< skip_wgt (default 0.1)')
    filt_options.add_argument("--maxiter", type=int, default=100, help='maximum iterations for aipy.deconv.clean to converge (default 100)')
    filt_options.add_argument("--flag_nchan_low", default=0, type=int, help="Number of channels to flag on lower band edge before filtering.")
    filt_options.add_argument("--flag_nchan_high", default=0, type=int, help="Number of channels to flag on upper band edge before filtering.")
    filt_options.add_argument("--gain", type=float, default=0.1, help="Fraction of residual to use in each iteration.")
    filt_options.add_argument("--alpha", type=float, default=.5, help="If window='tukey', use this alpha parameter (default .5).")

    return a

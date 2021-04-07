# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"""Module for xtalk filtering data and related operations."""

import numpy as np

from . import io
from . import version
from .frf import FRFilter
from . import vis_clean

import pickle
import random
import glob
import os
import warnings
from copy import deepcopy
from pyuvdata import UVCal
import argparse


class TophatFRFilter(FRFilter):
    """
    FRFilter object with methods for applying tophat filters in fringe-rate space.

    Used for Tophat fringe-rate CLEANing and filtering.
    See vis_clean.VisClean for details on time-domain filtering.
    """

    def run_tophat_frfilter(self, to_filter=None, weight_dict=None, mode='clean',
                            frate_standoff=0.0, frac_frate_sky_max=1.0, min_frate=0.025,
                            max_frate_coeffs=None,
                            skip_wgt=0.1, tol=1e-9, verbose=False, cache_dir=None, read_cache=False,
                            write_cache=False,
                            data=None, flags=None, **filter_kwargs):
        '''
        Interpolate / filter data in time using the physical fringe-rates of the sky. (or constant frate)
        Arguments:
          to_filter: list of visibilities to filter in the (i,j,pol) format.
              If None (the default), all visibilities are filtered.
          weight_dict: dictionary or DataContainer with all the same keys as self.data.
              Linear multiplicative weights to use for the delay filter. Default, use np.logical_not
              of self.flags. uvtools.dspec.fourier_filter will renormalize to compensate.
          mode: string specifying filtering mode. See fourier_filter or uvtools.dspec.fourier_filter for supported modes.
          frate_standoff: float, optional
              Additional fringe-rate standoff in mHz to add to Omega_E b_{EW} nu/c for fringe-rate inpainting.
              default = 0.0.
          frac_frate_sky_max: float, optional
             fraction of horizon to fringe-rate filter.
          min_frate: float, optional
             minimum fringe-rate to filter, regardless of baseline length in mHz.
             Default is 0.025
          max_frate_coeffs, 2-tuple float
            Maximum fringe-rate coefficients for the model max_frate [mHz] = x1 * EW_bl_len [ m ] + x2."
            Providing these overrides the sky-based fringe-rate determination! Default is None.
          skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
              Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
              time. Skipped channels are then flagged in self.flags.
              Only works properly when all weights are all between 0 and 1.
          tol : float, optional. To what level are foregrounds subtracted.
          verbose: If True print feedback to stdout
          cache_dir: string, optional, path to cache file that contains pre-computed dayenu matrices.
                      see uvtools.dspec.dayenu_filter for key formats.
          read_cache: bool, If true, read existing cache files in cache_dir before running.
          write_cache: bool. If true, create new cache file with precomputed matrices
                             that were not in previously loaded cache files.
          cache: dictionary containing pre-computed filter products.
          skip_flagged_edges : bool, if true do not include edge times in filtering region (filter over sub-region).
          verbose: bool, optional, lots of outputs!
          filter_kwargs: see fourier_filter for a full list of filter_specific arguments.

        Results are stored in:
          self.clean_resid: DataContainer formatted like self.data with only high-fringe-rate components
          self.clean_model: DataContainer formatted like self.data with only low-fringe-rate components
          self.clean_info: Dictionary of info from uvtools.dspec.fourier_filter with the same keys as self.data
        '''
        if to_filter is None:
            to_filter = list(self.data.keys())
        # read in cache
        if not mode == 'clean':
            if read_cache:
                filter_cache = io.read_filter_cache_scratch(cache_dir)
            else:
                filter_cache = {}
            keys_before = list(filter_cache.keys())
        else:
            filter_cache = None
        if max_frate_coeffs is None:
            # compute maximum fringe rate dict based on baseline lengths.
            blcosines = {k: self.blvecs[k[:2]][0] / np.linalg.norm(self.blvecs[k[:2]]) for k in to_filter}
            frateamps = {k: 1. / (24. * 3.6) * self.freqs.max() / 3e8 * 2 * np.pi * np.linalg.norm(self.blvecs[k[:2]]) for k in to_filter}
            # set autocorrs to have blcose of 0.0
            for k in blcosines:
                if np.isnan(blcosines[k]):
                    blcosines[k] = 0.0
            sinlat = np.sin(np.abs(self.hd.telescope_location_lat_lon_alt[0]))
            max_frates = io.DataContainer({})
            min_frates = io.DataContainer({})
            center_frates = io.DataContainer({})
            width_frates = io.DataContainer({})
            # calculate min/max center fringerates.
            # these depend on the sign of the blcosine.
            for k in to_filter:
                if blcosines[k] >= 0:
                    max_frates[k] = frateamps[k] * np.sqrt(sinlat ** 2. + blcosines[k] ** 2. * (1 - sinlat ** 2.))
                    min_frates[k] = -frateamps[k] * sinlat
                else:
                    min_frates[k] = -frateamps[k] * np.sqrt(sinlat ** 2. + blcosines[k] ** 2. * (1 - sinlat ** 2.))
                    max_frates[k] = frateamps[k] * sinlat
                center_frates[k] = (max_frates[k] + min_frates[k]) / 2.
                width_frates[k] = np.abs(max_frates[k] - min_frates[k]) / 2. * frac_frate_sky_max + frate_standoff
            # divide by center fringe rate to take advantage of Fourier shift theorem and use a
            # zero centered filter even if the center fringe rate is not at zero.
            for k in to_filter:
                self.data[k] /= np.exp(2j * np.pi * self.times[:, None] * 3.6 * 24. * center_frates[k])
                width_frates[k] = np.max([width_frates[k], min_frate])
            # perform vis_clean.
        else:
            width_frates = io.DataContainer({k: np.max([max_frate_coeffs[0] * self.blvecs[k[:2]][0] + max_frate_coeffs[1], 0.0]) for k in self.data})
        self.vis_clean(keys=to_filter, data=self.data, flags=self.flags, wgts=weight_dict,
                       ax='time', x=(self.times - np.mean(self.times)) * 24. * 3600.,
                       cache=filter_cache, mode=mode, tol=tol, skip_wgt=skip_wgt, max_frate=width_frates,
                       overwrite=True, verbose=verbose, **filter_kwargs)
        if 'output_prefix' in filter_kwargs:
            filtered_data = getattr(self, filter_kwargs['output_prefix'] + '_data')
            filtered_model = getattr(self, filter_kwargs['output_prefix'] + '_model')
            filtered_resid = getattr(self, filter_kwargs['output_prefix'] + '_resid')
        else:
            filtered_data = self.clean_data
            filtered_model = self.clean_model
            filtered_resid = self.clean_resid
        if max_frate_coeffs is None:
            for k in to_filter:
                filtered_data[k] *= np.exp(2j * np.pi * self.times[:, None] * 3.6 * 24. * center_frates[k])
                filtered_model[k] *= np.exp(2j * np.pi * self.times[:, None] * 3.6 * 24. * center_frates[k])
                filtered_resid[k] *= np.exp(2j * np.pi * self.times[:, None] * 3.6 * 24. * center_frates[k])
                self.data[k] *= np.exp(2j * np.pi * self.times[:, None] * 3.6 * 24. * center_frates[k])
        if not mode == 'clean':
            if write_cache:
                filter_cache = io.write_filter_cache_scratch(filter_cache, cache_dir, skip_keys=keys_before)


def tophat_frfilter_argparser(mode='clean'):
    '''Arg parser for commandline operation of tophat fr-filters.

    Parameters
    ----------
    mode : string, optional.
        Determines sets of arguments to load.
        Can be 'clean', 'dayenu', or 'dpss_leastsq'.

    Returns
    -------
    argparser
        argparser for tophat fringe-rate (time-domain) filtering for specified filtering mode

    '''
    if mode == 'clean':
        ap = vis_clean._clean_argparser()
    elif mode == 'dayenu':
        ap = vis_clean._linear_argparser()
    elif mode == 'dpss_leastsq':
        ap = vis_clean._dpss_argparser()
    filt_options = ap.add_argument_group(title='Options for the fr-filter')
    ap.add_argument("--skip_if_flag_within_edge_distance", type=int, default=0, help="skip integrations channels if there is a flag within this integer distance of edge.")
    ap.add_argument("--frac_frate_sky_max", type=float, default=1.0, help="Fraction of maximum sky-fringe-rate to interpolate / filter.")
    ap.add_argument("--frate_standoff", type=float, default=0.0, help="Standoff in fringe-rate to filter [mHz].")
    ap.add_argument("--min_frate", type=float, default=0.025, help="Minimum fringe-rate to filter [mHz].")
    ap.add_argument("--max_frate_coeffs", type=float, default=None, nargs=2, help="Maximum fringe-rate coefficients for the model max_frate [mHz] = x1 * EW_bl_len [ m ] + x2."
                                                                                  "Providing these overrides the sky-based fringe-rate determination! Default is None.")
    return ap


def load_tophat_frfilter_and_write(datafile_list, baseline_list=None, calfile_list=None,
                                   Nbls_per_load=None, spw_range=None, cache_dir=None,
                                   read_cache=False, write_cache=False, external_flags=None,
                                   factorize_flags=False, time_thresh=0.05,
                                   res_outfilename=None, CLEAN_outfilename=None, filled_outfilename=None,
                                   clobber=False, add_to_history='', avg_red_bllens=False, polarizations=None,
                                   skip_flagged_edges=False, overwrite_flags=False,
                                   flag_yaml=None,
                                   clean_flags_in_resid_flags=True, **filter_kwargs):
    '''
    A tophat fr-filtering method that only simultaneously loads and writes user-provided
    list of baselines. This is to support parallelization over baseline (rather then time) if baseline_list is specified.

    Arguments:
        datafile_list: list of data files to perform cross-talk filtering on
        baseline_list: list of antenna-pair-pol triplets to filter and write out from the datafile_list.
                       If None, load all baselines in files. Default is None.
        calfile_list: optional list of calibration files to apply to data before fr filtering
        Nbls_per_load: int, the number of baselines to load at once.
            If None, load all baselines at once. default : None.
        spw_range: 2-tuple or 2-list, spw_range of data to filter.
        cache_dir: string, optional, path to cache file that contains pre-computed dayenu matrices.
            see uvtools.dspec.dayenu_filter for key formats.
        read_cache: bool, If true, read existing cache files in cache_dir before running.
        write_cache: bool. If true, create new cache file with precomputed matrices
            that were not in previously loaded cache files.
        factorize_flags: bool, optional
            If True, factorize flags before running delay filter. See vis_clean.factorize_flags.
        time_thresh : float, optional
            Fractional threshold of flagged pixels across time needed to flag all times
            per freq channel. It is not recommend to set this greater than 0.5.
            Fully flagged integrations do not count towards triggering time_thresh.
        res_outfilename: path for writing the filtered visibilities with flags
        CLEAN_outfilename: path for writing the CLEAN model visibilities (with the same flags)
        filled_outfilename: path for writing the original data but with flags unflagged and replaced
            with CLEAN models wherever possible
        clobber: if True, overwrites existing file at the outfilename
        add_to_history: string appended to the history of the output file
        avg_red_bllens: bool, if True, round baseline lengths to redundant average. Default is False.
        polarizations : list of polarizations to process (and write out). Default None operates on all polarizations in data.
        skip_flagged_edges : bool, if true do not include edge times in filtering region (filter over sub-region).
        overwrite_flags : bool, if true reset data flags to False except for flagged antennas.
        flag_yaml: path to manual flagging text file.
        clean_flags_in_resid_flags: bool, optional. If true, include clean flags in residual flags that get written.
                                    default is True.
        filter_kwargs: additional keyword arguments to be passed to TophatFRFilter.run_tophat_frfilter()
    '''
    if baseline_list is not None and Nbls_per_load is not None:
        raise NotImplementedError("baseline loading and partial i/o not yet implemented.")
    hd = io.HERAData(datafile_list, filetype='uvh5', axis='blt')
    if baseline_list is None:
        baseline_list = hd.bls
    if spw_range is None:
        spw_range = [0, hd.Nfreqs]
    freqs = hd.freq_array.flatten()[spw_range[0]:spw_range[1]]
    baseline_antennas = []
    for blpolpair in baseline_list:
        baseline_antennas += list(blpolpair[:2])
    baseline_antennas = np.unique(baseline_antennas).astype(int)
    if calfile_list is not None:
        cals = io.HERACal(calfile_list)
        cals.read(antenna_nums=baseline_antennas, frequencies=freqs)
    else:
        cals = None
    if polarizations is None:
        if len(hd.filepaths) > 1:
            polarizations = list(hd.pols.values())[0]
        else:
            polarizations = hd.pols
    baseline_list = [bl for bl in baseline_list if bl[-1] in polarizations or len(bl) == 2]
    if Nbls_per_load is None:
        Nbls_per_load = len(baseline_list)
    for i in range(0, len(baseline_list), Nbls_per_load):
        tfrfil = TophatFRFilter(hd, input_cal=cals, axis='blt')
        tfrfil.read(bls=baseline_list[i:i + Nbls_per_load], frequencies=freqs)
        if avg_red_bllens:
            tfrfil.avg_red_baseline_vectors()
        if external_flags is not None:
            tfrfil.apply_flags(external_flags, overwrite_flags=overwrite_flags)
        if flag_yaml is not None:
            tfrfil.apply_flags(flag_yaml, overwrite_flags=overwrite_flags, filetype='yaml')
        if factorize_flags:
            tfrfil.factorize_flags(time_thresh=time_thresh, inplace=True)
        tfrfil.run_tophat_frfilter(cache_dir=cache_dir, read_cache=read_cache, write_cache=write_cache,
                                   skip_flagged_edges=skip_flagged_edges, **filter_kwargs)
        tfrfil.write_filtered_data(res_outfilename=res_outfilename, CLEAN_outfilename=CLEAN_outfilename,
                               filled_outfilename=filled_outfilename, partial_write=Nbls_per_load < len(baseline_list),
                               clobber=clobber, add_to_history=add_to_history,
                               extra_attrs={'Nfreqs': tfrfil.hd.Nfreqs, 'freq_array': tfrfil.hd.freq_array})
        tfrfil.hd.data_array = None  # this forces a reload in the next loop

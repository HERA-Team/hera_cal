# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"""Module for applying calibration solutions to visibility data, both in memory and on disk."""

import numpy as np
import argparse
import copy
import warnings
from . import io
from . import utils
from . import redcal
import pyuvdata.utils as uvutils
from pyuvdata import UVData


def _check_polarization_consistency(data, gains):
    '''This fucntion raises an error if all the gain keys are cardinal but none of the data keys are cardinal
    (e/n rather than x/y), or vice versa. In the mixed case, or if one is empty, no errors are raised.'''
    if (len(data) > 0) and (len(gains) > 0):
        data_keys_cardinal = [utils._is_cardinal(bl[2]) for bl in data.keys()]
        gain_keys_cardinal = [utils._is_cardinal(ant[1]) for ant in gains.keys()]
        if np.all(data_keys_cardinal) and not np.any(gain_keys_cardinal):
            raise KeyError("All the data keys are cardinal (e.g. 'nn' or 'ee'), but none of the gain keys are.")
        elif np.all(gain_keys_cardinal) and not np.any(data_keys_cardinal):
            raise KeyError("All the gain keys are cardinal (e.g. 'Jnn' or 'Jee'), but none of the data keys are.")


def calibrate_redundant_solution(data, data_flags, new_gains, new_flags, all_reds,
                                 old_gains=None, old_flags=None, gain_convention='divide'):
    '''Update the calibration of a redundant visibility solution (or redundantly averaged visibilities).
    This function averages together all gain ratios (old/new) within a redundant group (which should
    ideally all be the same) to figure out the proper gain to apply/unapply to the visibilities. If all
    gain ratios are flagged for a given time/frequency within a redundant group, the data_flags are
    updated. Typical use is to use absolute/smooth_calibrated gains as new_gains, omnical gains as
    old_gains, and omnical visibility solutions as data. NOTE: BDA not supported; gain and data shapes must match.

    Arguments:
        data: DataContainer containing baseline-pol complex visibility data. This is modified in place.
        data_flags: DataContainer containing data flags. They are updated based on the flags of the
            calibration solutions.
        new_gains: Dictionary of complex calibration gains to apply with keys like (1,'Jnn')
        new_flags: Dictionary with keys like (1,'Jnn') of per-antenna boolean flags to update data_flags
            if either antenna in a visibility is flagged. Must have all keys in new_gains.
        all_reds: list of lists of redundant baseline tuples, e.g. (0,1,'nn'). Must be a superset of
            the reds used for producing cal
        old_gains: Dictionary of complex calibration gains to take out with keys like (1,'Jnn').
            Default of None implies means that the "old" gains are all 1s. Must be either None or
            have all the same keys as new_gains.
        old_flags: Dictionary with keys like (1,'Jnn') of per-antenna boolean flags to update data_flags
            if either antenna in a visibility is flagged. Default of None all old_gains are unflagged.
            Must be either None or have all the same keys as new_flags.
        gain_convention: str, either 'divide' or 'multiply'. 'divide' means V_obs = gi gj* V_true,
            'multiply' means V_true = gi gj* V_obs. Assumed to be the same for new_gains and old_gains.
    '''

    _check_polarization_consistency(data, new_gains)
    _check_polarization_consistency(data_flags, new_flags)
    exponent = {'divide': 1, 'multiply': -1}[gain_convention]
    if old_gains is None:
        old_gains = {ant: np.ones_like(new_gains[ant]) for ant in new_gains}
    if old_flags is None:
        old_flags = {ant: np.zeros_like(new_flags[ant]) for ant in new_flags}

    # assert that all antennas in new_gains are also in new_flags, old_gains, and old_flags
    assert np.all([ant in new_flags for ant in new_gains])
    assert np.all([ant in old_gains for ant in new_gains])
    assert np.all([ant in old_flags for ant in new_gains])

    for red in all_reds:
        # skip if there's nothing to calibrate
        if np.all([bl not in data for bl in red]):
            continue

        # Fill in missing antennas with flags
        for bl in red:
            for ant in utils.split_bl(bl):
                if ant not in new_gains:
                    new_gains[ant] = np.ones_like(list(new_gains.values())[0])
                    new_flags[ant] = np.ones_like(list(new_flags.values())[0])
                if ant not in old_gains:
                    old_gains[ant] = np.ones_like(list(old_gains.values())[0])
                    old_flags[ant] = np.ones_like(list(old_flags.values())[0])

        # Compute all gain ratios within a redundant baseline, ensuring autocorrelations say real
        gain_ratios = [old_gains[i, utils.split_pol(pol)[0]] * np.conj(old_gains[j, utils.split_pol(pol)[1]])
                       / new_gains[i, utils.split_pol(pol)[0]] / np.conj(new_gains[j, utils.split_pol(pol)[1]])
                       if not ((i == j) and (utils.split_pol(pol)[0] == utils.split_pol(pol)[1]))
                       else np.abs(old_gains[i, utils.split_pol(pol)[0]])**2 / np.abs(new_gains[i, utils.split_pol(pol)[0]])**2
                       for (i, j, pol) in red]

        # Set flagged values to np.nan for those gain rations
        for n, bl in enumerate(red):
            ant1, ant2 = utils.split_bl(bl)
            gain_ratios[n][new_flags[ant1] | new_flags[ant2] | old_flags[ant1] | old_flags[ant2]] = np.nan

        # Average gain ratios using np.nanmean
        avg_gains = np.nanmean(gain_ratios, axis=0)
        avg_flags = ~np.isfinite(avg_gains)
        avg_gains[avg_flags] = 1

        # Apply average gains ratios and update flags
        for bl in red:
            if bl in data:
                data_flags[bl] |= avg_flags
                data[bl] *= avg_gains**exponent


def build_gains_by_cadences(data, gains, cal_flags=None, flags_are_wgts=False):
    ''' Builds dictionaries that map gains to the various cadences in potentially BDA data.
        As necessary, will upsample gains/flags by duplication and downsample gains/flags by
        (weighted) averaging. When downsampling, flags are ORed and weights are averaged.
        Assumes that the all cadences in the data are a power-of-two multiple of the slowest cadence.

    Arguments:
        data: DataContainer containing baseline-pol complex visibility data. Only used
            to figure out the various waterfall shapes.
        gains: Dictionary mapping antenna tuples to complex gains to upsample/downsample as needed.
        cal_flags: Dictionary mapping antenna tuples to boolean flags (or float weights).
        flags_are_wgts: if True, treat data_flags as weights where 0s represent flags and
            non-zero weights are unflagged data.

    Returns:
        gains_by_Nt: dictionary mapping numbers of integration to gain dictionaries
        cal_flags_by_Nt: dictionary mapping numbers of integration to flag/weight dictionaries.
            If cal_flags is None, this will be None as well.
    '''
    # get all cadences (unique shapes of the time dimension in the data)
    data_Nts = sorted(list(set([wf.shape[0] for wf in data.values()])))

    # Warn the user if the data doesn't conform to the expectation that all BDA is by a power of 2
    for Nt in data_Nts:
        power_of_2 = np.log(Nt / np.min(data_Nts)) / np.log(2)
        if not np.isclose(power_of_2, np.round(power_of_2)):
            warnings.warn(f'Data with {Nt} integrations is inconsistent with BDA by powers of 2 '
                          f'when the slowest cadence has {np.min(data_Nts)} integrations.')

    # initialize results dictionaries, handling the case where there are None and/or empty dicts
    # and also the case where gains/flags are scalars, which then get recast as 2D arrays
    if gains == {}:
        gains_by_Nt = {Nt: {} for Nt in data_Nts}
    else:
        if np.isscalar(list(gains.values())[0]):
            gains_by_Nt = {1: {ant: np.array([[gain]]) for ant, gain in gains.items()}}
        else:
            gains_by_Nt = {list(gains.values())[0].shape[0]: gains}
    cal_flags_by_Nt = None
    if cal_flags is not None:
        if cal_flags == {}:
            cal_flags_by_Nt = {Nt: {} for Nt in data_Nts}
        else:
            if np.isscalar(list(cal_flags.values())[0]):
                cal_flags_by_Nt = {1: {ant: np.array([[cf]]) for ant, cf in cal_flags.items()}}
            else:
                cal_flags_by_Nt = {list(cal_flags.values())[0].shape[0]: cal_flags}

    # Handle the case where gains/flags have a single integration (and are thus trivially broadcastable)
    if 1 in gains_by_Nt:
        for Nt in data_Nts:
            gains_by_Nt[Nt] = gains_by_Nt[1]
    if cal_flags_by_Nt is not None and 1 in cal_flags_by_Nt:
        for Nt in data_Nts:
            cal_flags_by_Nt[Nt] = cal_flags_by_Nt[1]

    # If necessary, upsample gains (and flags) by repeating them
    while True:
        max_gain_Nt = np.max(list(gains_by_Nt.keys()))
        if max_gain_Nt >= np.max(list(data_Nts)):
            break
        gains_by_Nt[max_gain_Nt * 2] = {ant: gains_by_Nt[max_gain_Nt][ant].repeat(2, axis=0)
                                        for ant in gains_by_Nt[max_gain_Nt]}
        if cal_flags_by_Nt is not None:
            cal_flags_by_Nt[max_gain_Nt * 2] = {ant: cal_flags_by_Nt[max_gain_Nt][ant].repeat(2, axis=0)
                                                for ant in cal_flags_by_Nt[max_gain_Nt]}

    # If necessary, downsample gains (and flags) by (flag-weigted) averaging (ORing) them
    while True:
        min_gain_Nt = np.min(list(gains_by_Nt.keys()))
        if min_gain_Nt <= np.min(list(data_Nts)):
            break
        gains_by_Nt[min_gain_Nt // 2] = {}
        if cal_flags_by_Nt is not None:
            cal_flags_by_Nt[min_gain_Nt // 2] = {}
        for ant, gain in gains_by_Nt[min_gain_Nt].items():
            # break gains and flags into even and odd times to average together
            even_gains = gain[0::2, :]
            odd_gains = gain[1::2, :]
            if cal_flags_by_Nt is not None:
                # use flags/weights to perform a weighted average
                even_flags = cal_flags_by_Nt[min_gain_Nt][ant][0::2, :]
                odd_flags = cal_flags_by_Nt[min_gain_Nt][ant][1::2, :]
                if flags_are_wgts:
                    weights = [even_flags, odd_flags]
                    # average weights
                    cal_flags_by_Nt[min_gain_Nt // 2][ant] = (even_flags + odd_flags) / 2
                else:
                    weights = [(~even_flags).astype(float), (~odd_flags).astype(float)]
                    # OR flags
                    cal_flags_by_Nt[min_gain_Nt // 2][ant] = even_flags | odd_flags
                # average with mask array to robustly handle case where weights sum to 0
                gains_by_Nt[min_gain_Nt // 2][ant] = np.ma.average([even_gains, odd_gains], axis=0, weights=weights).data
            else:
                # just do a straight average
                gains_by_Nt[min_gain_Nt // 2][ant] = np.average([even_gains, odd_gains], axis=0)

    # Warn if there cadences in the data that are missing that still aren't in gains_by_Nt
    for Nt in data_Nts:
        if Nt not in gains_by_Nt:
            warnings.warn(f'Data with {Nt} integrations cannot be calibrated with any of gain cadences: {list(gains_by_Nt.keys())}')

    return gains_by_Nt, cal_flags_by_Nt


def calibrate_in_place(data, new_gains, data_flags=None, cal_flags=None, old_gains=None,
                       gain_convention='divide', flags_are_wgts=False):
    '''Update data and data_flags in place, taking out old calibration solutions, putting in new calibration
    solutions, and updating flags from those calibration solutions. Previously flagged data is modified, but
    left flagged. Missing antennas from either the new gains, the cal_flags, or (if it's not None) the old
    gains are automatically flagged in the data's visibilities that involves those antennas. Data and gain
    shapes should always match in the frequency direction. Can apply Ntimes=1 gains by broadcasting. Can
    also up/downsample gains with Ntimes differing from those in the data by a power of 2, which is useful
    when the data is BDA and has Ntimes of multiple different powers of 2.

    Arguments:
        data: DataContainer containing baseline-pol complex visibility data. This is modified in place.
        new_gains: Dictionary of complex calibration gains to apply with keys like (1,'Jnn')
        data_flags: DataContainer containing data flags. This is modified in place if its not None.
        cal_flags: Dictionary with keys like (1,'Jnn') of per-antenna boolean flags to update data_flags
            if either antenna in a visibility is flagged. Any missing antennas are assumed to be totally
            flagged, so leaving this as None will result in input data_flags becoming totally flagged.
        old_gains: Dictionary of complex calibration gains to take out with keys like (1,'Jnn').
            Default of None implies that the data is raw (i.e. uncalibrated).
        gain_convention: str, either 'divide' or 'multiply'. 'divide' means V_obs = gi gj* V_true,
            'multiply' means V_true = gi gj* V_obs. Assumed to be the same for new_gains and old_gains.
        flags_are_weights: bool, if True, treat data_flags as weights where 0s represent flags and
            non-zero weights are unflagged data.
    '''

    _check_polarization_consistency(data, new_gains)
    exponent = {'divide': 1, 'multiply': -1}[gain_convention]

    # build dictionary of all necessary gain shapes to account for calibration of BDA data
    new_gains_by_Nt, cal_flags_by_Nt = build_gains_by_cadences(data, new_gains, cal_flags=cal_flags, flags_are_wgts=flags_are_wgts)
    if old_gains is not None:
        old_gains_by_Nt, _ = build_gains_by_cadences(data, old_gains)

    # loop over baselines in data
    for (i, j, pol) in data.keys():

        ap1, ap2 = utils.split_pol(pol)
        flag_all = False

        # get relevant shaped gains for this data waterfall
        Nt = data[(i, j, pol)].shape[0]
        try:
            new_gains_here = new_gains_by_Nt[Nt]
        except KeyError:
            raise ValueError(f'new_gains with {list(new_gains.values())[0].shape[0]} integrations are incompatible with data with {Nt} integrations.')
        cal_flags_here = None
        if cal_flags_by_Nt is not None:
            try:
                cal_flags_here = cal_flags_by_Nt[Nt]
            except KeyError:
                raise ValueError(f'cal_flags with {list(cal_flags.values())[0].shape[0]} integrations are incompatible with data with {Nt} integrations.')
        old_gains_here = None
        if old_gains is not None:
            try:
                old_gains_here = old_gains_by_Nt[Nt]
            except KeyError:
                raise ValueError(f'old_gains with {list(old_gains.values())[0].shape[0]} integrations are incompatible with data with {Nt} integrations.')

        # handle autocorrelations separately to keep them real
        if (i == j) & (ap1 == ap2):
            try:
                data[(i, j, pol)] /= (np.abs(new_gains_here[(i, ap1)])**2)**exponent
            except KeyError:
                flag_all = True
            if old_gains is not None:
                try:
                    data[(i, j, pol)] *= (np.abs(old_gains_here[(i, ap1)])**2)**exponent
                except KeyError:
                    flag_all = True
        else:
            # apply new gains for antennas i and j. If either is missing, flag the whole baseline
            try:
                data[(i, j, pol)] /= (new_gains_here[(i, ap1)])**exponent
            except KeyError:
                flag_all = True
            try:
                data[(i, j, pol)] /= np.conj(new_gains_here[(j, ap2)])**exponent
            except KeyError:
                flag_all = True
            # unapply old gains for antennas i and j. If either is missing, flag the whole baseline
            if old_gains is not None:
                try:
                    data[(i, j, pol)] *= (old_gains_here[(i, ap1)])**exponent
                except KeyError:
                    flag_all = True
                try:
                    data[(i, j, pol)] *= np.conj(old_gains_here[(j, ap2)])**exponent
                except KeyError:
                    flag_all = True

        if data_flags is not None:
            if cal_flags is None:
                # when data_flags is provided but cal_flags is not, flag everything
                flag_all = True
            else:
                # update data_flags in the case where flags are weights, flag all if cal_flags are missing
                if flags_are_wgts:
                    try:
                        data_flags[(i, j, pol)] *= (~cal_flags_here[(i, ap1)]).astype(float)
                        data_flags[(i, j, pol)] *= (~cal_flags_here[(j, ap2)]).astype(float)
                    except KeyError:
                        flag_all = True
                # update data_flags in the case where flags are booleans, flag all if cal_flags are missing
                else:
                    try:
                        data_flags[(i, j, pol)] += cal_flags_here[(i, ap1)]
                        data_flags[(i, j, pol)] += cal_flags_here[(j, ap2)]
                    except KeyError:
                        flag_all = True

            # if the flag object is given, update it for this baseline to be totally flagged
            if flag_all:
                if flags_are_wgts:
                    data_flags[(i, j, pol)] = np.zeros_like(data[(i, j, pol)], dtype=float)
                else:
                    data_flags[(i, j, pol)] = np.ones_like(data[(i, j, pol)], dtype=bool)


def apply_cal(data_infilename, data_outfilename, new_calibration, old_calibration=None, flag_file=None,
              flag_filetype='h5', a_priori_flags_yaml=None, flag_nchan_low=0, flag_nchan_high=0, filetype_in='uvh5', filetype_out='uvh5',
              nbl_per_load=None, gain_convention='divide', upsample=False, downsample=False, redundant_solution=False, bl_error_tol=1.0,
              add_to_history='', clobber=False, redundant_average=False, redundant_weights=None,
              freq_atol=1., redundant_groups=1, dont_red_average_flagged_data=False, spw_range=None,
              exclude_from_redundant_mode="data", vis_units=None, **kwargs):
    '''Update the calibration solution and flags on the data, writing to a new file. Takes out old calibration
    and puts in new calibration solution, including its flags. Also enables appending to history.

    Arguments:
        data_infilename: filename of the data to be calibrated.
        data_outfilename: filename of the resultant data file with the new calibration and flags.
        new_calibration: filename of the calfits file (or a list of filenames) for the calibration
            to be applied, along with its new flags (if any).
        old_calibration: filename of the calfits file (or a list of filenames) for the calibration
            to be unapplied. Default None means that the input data is raw (i.e. uncalibrated).
        flag_file: optional path to file containing flags to be ORed with flags in input data. Must have
            the same shape as the data.
        flag_filetype: filetype of flag_file to pass into io.load_flags. Either 'h5' (default) or legacy 'npz'.
        a_priori_flags_yaml : path to YAML with antenna frequency and time flags in the YAML.
            Flags are combined with ant_metrics's xants and ex_ants. If any
            polarization is flagged for an antenna, all polarizations are flagged.
            see hera_qm.metrics_io.read_a_priori_chan_flags (for freq flag format),
            hera_qm.metrics_io.read_a_priori_int_flags (for time flag format),
            hera_qm.metrics_io.read_a_priori_ant_flags (for antenna flag format).
        flag_nchan_low: integer number of channels at the low frequency end of the band to always flag (default 0)
        flag_nchan_high: integer number of channels at the high frequency end of the band to always flag (default 0)
        filetype_in: type of data infile. Supports 'miriad', 'uvfits', and 'uvh5'.
        filetype_out: type of data outfile. Supports 'miriad', 'uvfits', and 'uvh5'.
        nbl_per_load: maximum number of baselines to load at once. Default (None) is to load the whole file at once.
            Enables partial reading and writing, but only for uvh5 to uvh5.
            nbl_per_load is only supported if filetype_in is .uvh5.
        gain_convention: str, either 'divide' or 'multiply'. 'divide' means V_obs = gi gj* V_true,
            'multiply' means V_true = gi gj* V_obs. Assumed to be the same for new_gains and old_gains.
        upsample: if True, upsample baseline-dependent-averaged data file to the highest temporal resolution
        downsample: if True, downsample baseline-dependent-averaged data file to the lowest temporal resolution
        redundant_solution: If True, average gain ratios in redundant groups to recalibrate e.g. redcal solutions.
            NOTE: BDA data is not supported in this mode. Gain shapes must be made to match data samples using upsample/downsample.
        bl_error_tol: the largest allowable difference between baselines in a redundant group
            (in the same units as antpos). Normally, this is up to 4x the largest antenna position error.
        add_to_history: appends a string to the history of the output file. This will preceed combined histories
            of flag_file (if applicable), new_calibration and, old_calibration (if applicable).
        clobber: if True, overwrites existing file at outfilename
        redundant_average : bool, optional
            If True, redundantly average calibrated data and save to <data_outfilename>.red_avg.<filetype_out>
        redundant_weights : datacontainer, optional.
            Datacontainer containing weights to use in redundant averaging.
            only used if redundant_average is True.
            Default is None. If None is passed, then nsamples are used as the redundant weights.
        tol_factor: float, optional
            Float specifying the tolerance (as a fraction of channel width) within which cal frequencies must be matched in calibration solution to apply.
        redundant_groups : int, optional.
            Integer specifying how many different subsets of each redundant group to write to an independent file.
            If more then one redundant subgroup is specified, then output files will have label .uvh5 -> .n.uvh5
            redundant_groups>1 not supported with partial I/O yet.
        dont_red_average_flagged_data : bool, optional.
            If True, baselines within a redundant group with all pols flagged do not count towards the number of baselines
            in that group above the number of groups to output. This lets us throw away groups that in principal have greater
            then the minimum number of baselines to allow for a split into different output groups but could result in one of
            the subgroups being entirely flagged. This option is only used when redundant_groups > 1.
            Not supported for partial I/O.
        spw_range : 2-tuple specifying range of channels to select and redundantly average.
        exclude_from_redundant_mode: str, optional
            specify whether to use entirely flagged data, 'data', or ex_ants from an external yaml file 'yaml' to determine
            baselines to exclude from redundant average.
        vis_units : str, optional
            string specifying units of calibrated visibility. Overrides gain_scale in calibration file.
            Default is None -> calibration gain_scale is used to set vis_units in calibrated file.
        kwargs: dictionary mapping updated UVData attributes to their new values.
            See pyuvdata.UVData documentation for more info.
    '''
    # UPDATE CAL FLAGS WITH EX_ANTS INSTEAD OF FILTERING BASELINES.
    # optionally load external flags
    if exclude_from_redundant_mode not in ['yaml', 'data']:
        raise ValueError("exclude_from_redundant_mode must be 'yaml' or 'data'.")
    if flag_file is not None:
        ext_flags, flag_meta = io.load_flags(flag_file, filetype=flag_filetype, return_meta=True)
        add_to_history += '\nFLAGS_HISTORY: ' + str(flag_meta['history']) + '\n'

    # load new calibration solution
    hc = io.HERACal(new_calibration)
    new_gains, new_flags, _, _ = hc.read()
    if a_priori_flags_yaml is not None:
        from hera_qm.utils import apply_yaml_flags
        from hera_qm.metrics_io import read_a_priori_ant_flags
        # flag hc
        hc = apply_yaml_flags(hc, a_priori_flags_yaml,
                              ant_indices_only=True)
        # and rebuild data containers.
        new_gains, new_flags, _, _ = hc.build_calcontainers()
        ex_ants = read_a_priori_ant_flags(a_priori_flags_yaml, ant_indices_only=True)
    else:
        ex_ants = None
    add_to_history += '\nNEW_CALFITS_HISTORY: ' + hc.history + '\n'

    # load old calibration solution
    if old_calibration is not None:
        old_hc = io.HERACal(old_calibration)
        old_hc.read()
        # determine frequencies to load in old_hc that are close to hc
        freqs_to_load = []
        for f in old_hc.freqs:
            # set atol to be 1/10th of a channel
            if np.any(np.isclose(hc.freqs, f, rtol=0., atol=freq_atol)):
                freqs_to_load.append(f)
        if spw_range is not None:
            freqs_to_load = freqs_to_load[spw_range[0]:spw_range[1]]
        old_hc.select(frequencies=np.asarray(freqs_to_load))  # match up frequencies with hc.freqs
        old_gains, old_flags, _, _ = old_hc.build_calcontainers()
        add_to_history += '\nOLD_CALFITS_HISTORY: ' + old_hc.history + '\n'
    else:
        old_gains, old_flags = None, None
    hd = io.HERAData(data_infilename, filetype=filetype_in, upsample=upsample, downsample=downsample)
    if spw_range is None:
        spw_range = (0, hd.Nfreqs)
    else:
        if filetype_in != 'uvh5':
            raise NotImplementedError("spw only implemented for uvh5 files.")
    if filetype_in == 'uvh5':
        freqs_to_load = []
        for f in hd.freqs[spw_range[0]:spw_range[1]]:
            if np.any(np.isclose(hc.freqs, f, rtol=0., atol=freq_atol)):
                freqs_to_load.append(f)
    else:
        freqs_to_load = None
    # reselect cals to match hd freqs_to_load
    if freqs_to_load is not None:
        calfreqs = []
        calfreqsold = []
        for f in hc.freqs:
            if np.any(np.isclose(freqs_to_load, f)):
                calfreqs.append(f)
            if old_calibration is not None and np.any(np.isclose(old_hc.freqs, f)):
                calfreqsold.append(f)
        hc.select(frequencies=calfreqs)
        new_gains, new_flags, _, _ = hc.build_calcontainers()
        if old_calibration is not None:
            old_hc.select(frequencies=calfreqsold)
            old_gains, old_flags, _, _ = old_hc.build_calcontainers()

    add_to_history = utils.history_string(add_to_history)
    no_red_weights = redundant_weights is None
    if nbl_per_load is not None:
        if not ((filetype_in == 'uvh5') and (filetype_out == 'uvh5')):
            raise NotImplementedError('Partial writing is not implemented for non-uvh5 I/O.')
        if not redundant_groups == 1:
            raise NotImplementedError("Splitting redundant groups into subgroups is not yet implemented for partial I/O!")
        if dont_red_average_flagged_data:
            raise NotImplementedError("Completely skipping flagged data in redundantly averaged data not implemented for partial I/O!")
        for attribute, value in kwargs.items():
            hd.__setattr__(attribute, value)
        if redundant_average or redundant_solution:
            all_reds = redcal.get_reds(hd.data_antpos, pols=hd.pols, bl_error_tol=bl_error_tol, include_autos=True)
        else:
            all_reds = []
        if redundant_average:
            # initialize a redunantly averaged HERAData on disk
            # first copy the original HERAData
            all_red_antpairs = [[bl[:2] for bl in grp] for grp in all_reds if grp[-1][-1] == hd.pols[0]]
            hd_red = io.HERAData(data_infilename, upsample=upsample, downsample=downsample)
            # go through all redundant groups and remove the groups that do not
            # have baselines in the data. Each group is still labeled by the
            # first baseline of each group regardless if that baseline is in
            # the data file.
            reds_data = redcal.filter_reds(all_reds, bls=hd.bls)
            reds_data_bls = []
            for grp in reds_data:
                reds_data_bls.append(grp[0])
            # couldn't get a system working where we just read in the outputs one at a time.
            # so unfortunately, we have to load one baseline per redundant group.
            hd_red.read(bls=reds_data_bls, frequencies=freqs_to_load)

        # consider calucate reds here instead and pass in (to avoid computing it multiple times)
        # I'll look into generators and whether the reds calc is being repeated.
        for data, data_flags, data_nsamples in hd.iterate_over_bls(Nbls=nbl_per_load, chunk_by_redundant_group=redundant_average,
                                                                   reds=all_reds, frequencies=freqs_to_load):
            for bl in data_flags.keys():
                # apply band edge flags
                data_flags[bl][:, 0:flag_nchan_low] = True
                data_flags[bl][:, data_flags[bl].shape[1] - flag_nchan_high:] = True
                # apply external flags
                if flag_file is not None:
                    data_flags[bl] = np.logical_or(data_flags[bl], ext_flags[bl])
            if redundant_solution:
                calibrate_redundant_solution(data, data_flags, new_gains, new_flags, all_reds, old_gains=old_gains,
                                             old_flags=old_flags, gain_convention=gain_convention)
            else:
                calibrate_in_place(data, new_gains, data_flags=data_flags, cal_flags=new_flags,
                                   old_gains=old_gains, gain_convention=gain_convention)
            hd.update(data=data, flags=data_flags)

            if redundant_average:
                # by default, weight by nsamples (but not flags). This prevents spectral structure from being introduced
                # and also allows us to compute redundant averaged vis in flagged channels (in case flags are spurious).
                if no_red_weights:
                    redundant_weights = copy.deepcopy(data_nsamples)
                    for bl in data_flags:
                        if exclude_from_redundant_mode == 'data':
                            if np.all(data_flags[bl]):
                                redundant_weights[bl][:] = 0.
                        elif exclude_from_redundant_mode == 'yaml' and ex_ants is not None:
                            if bl[0] in ex_ants or bl[1] in ex_ants:
                                redundant_weights[bl][:] = 0.
                # redundantly average
                utils.red_average(data=data, flags=data_flags, nsamples=data_nsamples,
                                  reds=all_red_antpairs, wgts=redundant_weights, inplace=True,
                                  propagate_flags=True)
                # update redundant data. Don't partial write.
                hd_red.update(nsamples=data_nsamples, flags=data_flags, data=data)
            else:
                if vis_units is None:
                    if hasattr(hc, 'gain_scale') and hc.gain_scale is not None:
                        if hd.vis_units is not None and hc.gain_scale.lower() != "uncalib" and hd.vis_units.lower() != hc.gain_scale.lower():
                            warnings.warn(f"Replacing original data vis_units of {hd.vis_units}"
                                          f" with calibration vis_units of {hc.gain_scale}", RuntimeWarning)
                        vis_units = hc.gain_scale
                    else:
                        vis_units = hd.vis_units
                    # partial write works for no redundant averaging.
                hd.partial_write(data_outfilename, inplace=True, clobber=clobber, add_to_history=add_to_history, vis_units=vis_units, **kwargs)

        if redundant_average:
            # if we did redundant averaging, just write the redundant dataset out in the end at once.
            if hasattr(hc, 'gain_scale') and hc.gain_scale is not None:
                if hd.vis_units is not None and hc.gain_scale.lower() != "uncalib" and hd.vis_units.lower() != hc.gain_scale.lower():
                    warnings.warn(f"Replacing original data vis_units of {hd.vis_units}"
                                  f" with calibration vis_units of {hc.gain_scale}", RuntimeWarning)
                hd_red.vis_units = hc.gain_scale
            if vis_units is not None:
                hd_red.vis_units = vis_units
            hd_red.write_uvh5(data_outfilename, clobber=clobber)
    # full data loading and writing
    else:
        data, data_flags, data_nsamples = hd.read(frequencies=freqs_to_load)
        data_antpos = hd.get_metadata_dict()['data_antpos']
        pols = hd.get_metadata_dict()['pols']
        if redundant_average or redundant_solution:
            all_reds = redcal.get_reds(data_antpos, pols=pols, bl_error_tol=bl_error_tol, include_autos=True)
        else:
            all_reds = []
        if redundant_average:
            all_red_antpairs = [[bl[:2] for bl in grp] for grp in all_reds if grp[-1][-1] == pols[0]]
            data_antpairs = hd.get_antpairs()
            reds_data = [[bl for bl in blg if bl in data_antpairs] for blg in all_red_antpairs]
            reds_data = [blg for blg in reds_data if len(blg) > 0]
        for bl in data_flags.keys():
            # apply band edge flags
            data_flags[bl][:, 0:flag_nchan_low] = True
            data_flags[bl][:, data_flags[bl].shape[1] - flag_nchan_high:] = True
            # apply external flags
            if flag_file is not None:
                data_flags[bl] = np.logical_or(data_flags[bl], ext_flags[bl])
        if redundant_solution:
            calibrate_redundant_solution(data, data_flags, new_gains, new_flags, all_reds, old_gains=old_gains,
                                         old_flags=old_flags, gain_convention=gain_convention)
        else:
            calibrate_in_place(data, new_gains, data_flags=data_flags, cal_flags=new_flags,
                               old_gains=old_gains, gain_convention=gain_convention)
        if not redundant_average:
            if vis_units is None:
                if hasattr(hc, 'gain_scale') and hc.gain_scale is not None:
                    if hd.vis_units is not None and hd.vis_units.lower() != "uncalib" and hd.vis_units.lower() != hc.gain_scale.lower():
                        warnings.warn(f"Replacing original data vis_units of {hd.vis_units}"
                                      " with calibration vis_units of {hc.gain_scale}", RuntimeWarning)
                    vis_units = hc.gain_scale
            if vis_units is not None:
                kwargs['vis_units'] = vis_units
            io.update_uvdata(hd, data=data, flags=data_flags, add_to_history=add_to_history, **kwargs)
            io._write_HERAData_to_filetype(hd, data_outfilename, filetype_out=filetype_out, clobber=clobber)

        else:
            all_red_antpairs = [[bl[:2] for bl in grp] for grp in all_reds if grp[-1][-1] == hd.pols[0]]
            hd.update(data=data, flags=data_flags, nsamples=data_nsamples, **kwargs)
            # by default, weight by nsamples (but not flags). This prevents spectral structure from being introduced
            # and also allows us to compute redundant averaged vis in flagged channels (in case flags are spurious).
            if no_red_weights:
                redundant_weights = copy.deepcopy(data_nsamples)
                for bl in data_flags:
                    if np.all(data_flags[bl]):
                        if exclude_from_redundant_mode == 'data':
                            if np.all(data_flags[bl]):
                                redundant_weights[bl][:] = 0.
                        elif exclude_from_redundant_mode == 'yaml' and ex_ants is not None:
                            if bl[0] in ex_ants or bl[1] in ex_ants:
                                redundant_weights[bl][:] = 0.
            for red_chunk in range(redundant_groups):
                red_antpairs = []
                reds_data_bls = []
                for grp in reds_data:
                    # trim group to only include baselines with redundant weights not equal to zero.
                    grp0 = grp[0]
                    if dont_red_average_flagged_data and redundant_groups > 1:
                        grp = [ap for ap in grp if np.any(np.asarray([~np.isclose(redundant_weights[ap + (pol,)], 0.0) for pol in data_flags.pols()]))]
                    # only include groups with more elements then redundant groups!
                    if len(grp) >= redundant_groups:
                        red_antpairs.append(grp[red_chunk:: redundant_groups])
                        reds_data_bls.append(grp0)
                data_red, flags_red, nsamples_red = utils.red_average(data=data, flags=data_flags, nsamples=data_nsamples,
                                                                      reds=red_antpairs, red_bl_keys=reds_data_bls, wgts=redundant_weights, inplace=False,
                                                                      propagate_flags=True)
                # update redundant data. Don't partial write.
                hd_red = io.HERAData(data_infilename, upsample=upsample, downsample=downsample)
                if len(reds_data_bls) > 0:
                    hd_red.read(bls=reds_data_bls, frequencies=freqs_to_load)
                    # update redundant data. Don't partial write.
                    hd_red.update(nsamples=nsamples_red, flags=flags_red, data=data_red)
                    hd_red.update(nsamples=nsamples_red, flags=flags_red, data=data_red)
                    if redundant_groups > 1:
                        outfile = data_outfilename.replace('.uvh5', f'.{red_chunk}.uvh5')
                    else:
                        outfile = data_outfilename
                    if filetype_out == 'uvh5':
                        if hasattr(hc, 'gain_scale') and hc.gain_scale is not None:
                            if hd_red.vis_units is not None and hd_red.vis_units.lower() != "uncalib" and hd_red.vis_units.lower() != hc.gain_scale.lower():
                                warnings.warn(f"Replacing original data vis_units of {hd.vis_units}"
                                              " with calibration vis_units of {hc.gain_scale}", RuntimeWarning)
                            hd_red.vis_units = hc.gain_scale
                        if vis_units is not None:
                            hd_red.vis_units = vis_units
                        hd_red.write_uvh5(outfile, clobber=clobber)
                    else:
                        raise NotImplementedError("redundant averaging only supported for uvh5 outputs.")
                else:
                    warnings.warn("No unflagged data so no calibration or outputs produced.")


def apply_cal_argparser():
    '''Arg parser for commandline operation of apply_cal.'''
    a = argparse.ArgumentParser(description="Apply (and optionally, also unapply) a calfits file to visibility file.")
    a.add_argument("infilename", type=str, help="path to visibility data file to calibrate")
    a.add_argument("outfilename", type=str, help="path to new visibility results file")
    a.add_argument("--new_cal", type=str, default=None, nargs="+", help="path to new calibration calfits file (or files for cross-pol)")
    a.add_argument("--old_cal", type=str, default=None, nargs="+", help="path to old calibration calfits file to unapply (or files for cross-pol)")
    a.add_argument("--flag_file", type=str, default=None, help="path to file of flags to OR with data flags")
    a.add_argument("--flag_filetype", type=str, default='h5', help="filetype of flag_file (either 'h5' or legacy 'npz'")
    a.add_argument("--flag_nchan_low", type=int, default=0, help="integer number of channels at the low frequency end of the band to always flag (default 0)")
    a.add_argument("--flag_nchan_high", type=int, default=0, help="integer number of channels at the high frequency end of the band to always flag (default 0)")
    a.add_argument("--filetype_in", type=str, default='uvh5', help='filetype of input data files')
    a.add_argument("--filetype_out", type=str, default='uvh5', help='filetype of output data files')
    a.add_argument("--nbl_per_load", type=str, default=None, help="Maximum number of baselines to load at once. uvh5 to uvh5 only."
                                                                  "Default loads the whole file. If 'none' is provided, also loads whole file.")
    a.add_argument("--redundant_groups", type=int, default=1, help="Number of subgroups to split each redundant baseline into for cross power spectra. ")
    a.add_argument("--gain_convention", type=str, default='divide',
                   help="'divide' means V_obs = gi gj* V_true, 'multiply' means V_true = gi gj* V_obs.")
    a.add_argument("--upsample", default=False, action="store_true", help="Upsample BDA files to the highest temporal resolution.")
    a.add_argument("--downsample", default=False, action="store_true", help="Downsample BDA files to the highest temporal resolution.")
    a.add_argument("--redundant_solution", default=False, action="store_true",
                   help="If True, average gain ratios in redundant groups to recalibrate e.g. redcal solutions.")
    a.add_argument("--clobber", default=False, action="store_true", help='overwrites existing file at outfile')
    a.add_argument("--vis_units", default=None, type=str, help="String to insert into vis_units attribute of output visibility file.")
    a.add_argument("--redundant_average", default=False, action="store_true", help="Redundantly average calibrated data.")
    a.add_argument("--dont_red_average_flagged_data", default=False, action="store_true", help="Do not include flagged data in redundant averages. Prevents redundant groups where one subgroup is flagged.")
    a.add_argument("--spw_range", default=None, type=int, nargs=2, help="specify spw range to load.")
    a.add_argument("--exclude_from_redundant_mode", default='data', type=str, help="exclude visibilities from redundant average based on whether entire waterfall is flagged ,'data'"
                                                                                   ", or whether its antennas are present in a yaml file.")
    a.add_argument("--a_priori_flags_yaml", type=str, default=None, help="path to yaml file to use in apriori flags.")
    return a

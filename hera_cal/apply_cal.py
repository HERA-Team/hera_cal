# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"""Module for applying calibration solutions to visibility data, both in memory and on disk."""

import numpy as np
import argparse
import copy

from . import io
from . import version
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
    '''Update the calibrtion of a redundant visibility solution (or redundantly averaged visibilities).
    This function averages together all gain ratios (old/new) within a redundant group (which should
    ideally all be the same) to figure out the proper gain to apply/unapply to the visibilities. If all
    gain ratios are flagged for a given time/frequency within a redundant group, the data_flags are
    updated. Typical use is to use absolute/smooth_calibrated gains as new_gains, omnical gains as
    old_gains, and omnical visibility solutions as data.

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

        # Compute all gain ratios within a redundant baseline
        gain_ratios = [old_gains[i, utils.split_pol(pol)[0]] * np.conj(old_gains[j, utils.split_pol(pol)[1]])
                       / new_gains[i, utils.split_pol(pol)[0]] / np.conj(new_gains[j, utils.split_pol(pol)[1]])
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


def calibrate_in_place(data, new_gains, data_flags=None, cal_flags=None, old_gains=None,
                       gain_convention='divide', flags_are_wgts=False):
    '''Update data and data_flags in place, taking out old calibration solutions, putting in new calibration
    solutions, and updating flags from those calibration solutions. Previously flagged data is modified, but
    left flagged. Missing antennas from either the new gains, the cal_flags, or (if it's not None) the old
    gains are automatically flagged in the data's visibilities that involves those antennas.

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
    # loop over baselines in data
    for (i, j, pol) in data.keys():
        ap1, ap2 = utils.split_pol(pol)
        flag_all = False

        # apply new gains for antennas i and j. If either is missing, flag the whole baseline
        try:
            data[(i, j, pol)] /= (new_gains[(i, ap1)])**exponent
        except KeyError:
            flag_all = True
        try:
            data[(i, j, pol)] /= np.conj(new_gains[(j, ap2)])**exponent
        except KeyError:
            flag_all = True
        # unapply old gains for antennas i and j. If either is missing, flag the whole baseline
        if old_gains is not None:
            try:
                data[(i, j, pol)] *= (old_gains[(i, ap1)])**exponent
            except KeyError:
                flag_all = True
            try:
                data[(i, j, pol)] *= np.conj(old_gains[(j, ap2)])**exponent
            except KeyError:
                flag_all = True

        if data_flags is not None:
            # update data_flags in the case where flags are weights, flag all if cal_flags are missing
            if flags_are_wgts:
                try:
                    data_flags[(i, j, pol)] *= (~cal_flags[(i, ap1)]).astype(np.float)
                    data_flags[(i, j, pol)] *= (~cal_flags[(j, ap2)]).astype(np.float)
                except KeyError:
                    flag_all = True
            # update data_flags in the case where flags are booleans, flag all if cal_flags are missing
            else:
                try:
                    data_flags[(i, j, pol)] += cal_flags[(i, ap1)]
                    data_flags[(i, j, pol)] += cal_flags[(j, ap2)]
                except KeyError:
                    flag_all = True

            # if the flag object is given, update it for this baseline to be totally flagged
            if flag_all:
                if flags_are_wgts:
                    data_flags[(i, j, pol)] = np.zeros_like(data[(i, j, pol)], dtype=np.float)
                else:
                    data_flags[(i, j, pol)] = np.ones_like(data[(i, j, pol)], dtype=np.bool)

def sum_diff_2_even_odd(sum_infilename, diff_infilname, even_outfilename, odd_outfilename,
                        nbl_per_load=None, filetype_in='uvh5'):
    """Generate even and odd data sets from sum and diff

    Arguments:
        sum_infilname: str
            filename for sum file.
        diff_infilename: str
            filename for diff file.
        even_outfilename: str
            filename to write even.
        odd_outfilename: str
            filename to write odd.
        nbl_per_load: int, optional
            number of baselines to load simultaneously
            default, None results in all baselines loaded.
    """
    hd_sum = io.HERAData(sum_infilename, filetype=filetype_in)
    hd_diff = io.HERAData(diff_infilname, filetype=filetype_in)
    if nbl_per_load is not None:
        if not ((filetype_in == 'uvh5') and (filetype_out == 'uvh5')):
            raise NotImplementedError('Partial writing is not implemented for non-uvh5 I/O.')
        for sum, sum_flags, sum_nsamples in hd.iterate_over_bls(Nbls=nbl_per_load):
            diff, diff_flags, diff_nsamples = hd_diff.load(bls=list(sum.keys()))
            sum = (sum + diff) / 2.
            diff = sum - diff
            for k in sum_flags:
                sum_flags[k] = sum_flags[k] | diff_flags[k]
                diff_flags[k] = sum_flags[k]
                diff_nsamples[k] = (sum_nsamples[k] + diff_nsamples[k]) / 2.
                sum_nsamples[k] = diff_nsamples[k]
            sum.update(data=sum, flags=sum_flags, nsamples=sum_nsamples)
            diff.update(data=diff, flags=diff_flags, nsamples=diff_nsamples)
            sum.partial_write(even_outfilename, inplace=True, clobber=clobber)
            diff.partial_write(odd_outfilename, inplace=True, clobber=clobber)
    else:
        sum, sum_flags, sum_nsamples = hd.read()
        diff, diff_flags, diff_nsamples = hd.read()
        sum = (sum + diff) / 2.
        diff = sum - diff
        for k in sum_flags:
            sum_flags[k] = sum_flags[k] | diff_flags[k]
            diff_flags[k] = sum_flags[k]
            diff_nsamples[k] = (sum_nsamples[k] + diff_nsamples[k]) / 2.
            sum_nsamples[k] = diff_nsamples[k]
        sum.update(data=sum, flags=sum_flags, nsamples=sum_nsamples)
        diff.update(data=diff, flags=diff_flags, nsamples=diff_nsamples)
        sum.write_uvh5(even_outfilename, inplace=True, clobber=clobber)
        diff.write_uvh5(odd_outfilename, inplace=True, clobber=clobber)

def apply_cal(data_infilename, data_outfilename, new_calibration, old_calibration=None, flag_file=None,
              flag_filetype='h5', a_priori_flags_yaml=None, flag_nchan_low=0, flag_nchan_high=0, filetype_in='uvh5', filetype_out='uvh5',
              nbl_per_load=None, gain_convention='divide', redundant_solution=False, bl_error_tol=1.0,
              add_to_history='', clobber=False, redundant_average=False, redundant_weights=None, **kwargs):
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
        gain_convention: str, either 'divide' or 'multiply'. 'divide' means V_obs = gi gj* V_true,
            'multiply' means V_true = gi gj* V_obs. Assumed to be the same for new_gains and old_gains.
        redundant_solution: If True, average gain ratios in redundant groups to recalibrate e.g. redcal solutions.
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
        kwargs: dictionary mapping updated UVData attributes to their new values.
            See pyuvdata.UVData documentation for more info.
    '''
    # UPDATE CAL FLAGS WITH EX_ANTS INSTEAD OF FILTERING BASELINES.
    # optionally load external flags
    if flag_file is not None:
        ext_flags, flag_meta = io.load_flags(flag_file, filetype=flag_filetype, return_meta=True)
        add_to_history += '\nFLAGS_HISTORY: ' + str(flag_meta['history']) + '\n'

    # load new calibration solution
    hc = io.HERACal(new_calibration)
    new_gains, new_flags, _, _ = hc.read()
    if a_priori_flags_yaml is not None:
        from hera_qm.utils import apply_yaml_flags
        # flag hc
        hc = apply_yaml_flags(hc, a_priori_flags_yaml,
                              ant_indices_only=True)
        # and rebuild data containers.
        new_gains, new_flags, _, _ = hc.build_calcontainers()
    add_to_history += '\nNEW_CALFITS_HISTORY: ' + hc.history + '\n'

    # load old calibration solution
    if old_calibration is not None:
        old_hc = io.HERACal(old_calibration)
        old_hc.read()
        # determine frequencies to load in old_hc that are close to hc
        freqs_to_load = []
        for f in old_hc.freqs:
            if np.any(np.isclose(hc.freqs, f)):
                freqs_to_load.append(f)
        old_hc.select(frequencies=np.asarray(freqs_to_load)) # match up frequencies with hc.freqs
        old_gains, old_flags, _, _ = old_hc.build_calcontainers()
        add_to_history += '\nOLD_CALFITS_HISTORY: ' + old_hc.history + '\n'
    else:
        old_gains, old_flags = None, None
    hd = io.HERAData(data_infilename, filetype=filetype_in)
    freqs_to_load = []
    for f in hd.freqs:
        if np.any(np.isclose(hc.freqs, f)):
            freqs_to_load.append(f)
    add_to_history = version.history_string(add_to_history)
    no_red_weights = redundant_weights is None
    # partial loading and writing using uvh5
    if nbl_per_load is not None:
        if not ((filetype_in == 'uvh5') and (filetype_out == 'uvh5')):
            raise NotImplementedError('Partial writing is not implemented for non-uvh5 I/O.')
        for attribute, value in kwargs.items():
            hd.__setattr__(attribute, value)
        if redundant_average or redundant_solution:
            all_reds = redcal.get_reds(hd.antpos, pols=hd.pols, bl_error_tol=bl_error_tol, include_autos=True)
        else:
            all_reds = []
        if redundant_average:
            # initialize a redunantly averaged HERAData on disk
            # first copy the original HERAData
            all_red_antpairs = [[bl[:2] for bl in grp] for grp in all_reds if grp[-1][-1] == hd.pols[0]]
            hd_red = io.HERAData(data_infilename)
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
                                                                   reds=all_reds, freqs_to_load=freqs_to_load):
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
                        if np.all(data_flags[bl]):
                            redundant_weights[bl][:] = 0.
                # redundantly average
                utils.red_average(data=data, flags=data_flags, nsamples=data_nsamples,
                                  reds=all_red_antpairs, wgts=redundant_weights, inplace=True,
                                  propagate_flags=True)
                # update redundant data. Don't partial write.
                hd_red.update(nsamples=data_nsamples, flags=data_flags, data=data)
            else:
                # partial write works for no redundant averaging.
                hd.partial_write(data_outfilename, inplace=True, clobber=clobber, add_to_history=add_to_history, **kwargs)
        if redundant_average:
            # if we did redundant averaging, just write the redundant dataset out in the end at once.
            hd_red.write_uvh5(data_outfilename, clobber=clobber)
    # full data loading and writing
    else:
        data, data_flags, data_nsamples = hd.read(frequencies=freqs_to_load)
        all_reds = redcal.get_reds(data.antpos, pols=data.pols(), bl_error_tol=bl_error_tol, include_autos=True)
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
            io.update_vis(data_infilename, data_outfilename, filetype_in=filetype_in, filetype_out=filetype_out,
                          data=data, flags=data_flags, add_to_history=add_to_history, clobber=clobber, **kwargs)
        else:
            all_red_antpairs = [[bl[:2] for bl in grp] for grp in all_reds if grp[-1][-1] == hd.pols[0]]
            hd.update(data=data, flags=data_flags, **kwargs)
            # by default, weight by nsamples (but not flags). This prevents spectral structure from being introduced
            # and also allows us to compute redundant averaged vis in flagged channels (in case flags are spurious).
            if no_red_weights:
                redundant_weights = copy.deepcopy(data_nsamples)
                for bl in data_flags:
                    if np.all(data_flags[bl]):
                        redundant_weights[bl][:] = 0.

            utils.red_average(hd, reds=all_red_antpairs, inplace=True, wgts=redundant_weights,
                              propagate_flags=True)
            if filetype_out == 'uvh5':
                # overwrite original outfile with
                hd.write_uvh5(data_outfilename, clobber=clobber)
            else:
                raise NotImplementedError("redundant averaging only supported for uvh5 outputs.")

def sum_diff_2_even_odd_argparser():
    '''Arg parser for even/odd to sum/diff function.'''
    a = argparse.ArgumentParser(description="Convert a sum and a diff file to an even and an odd file.")
    a.add_argument("sumfilename", type=str, help="name of sum file.")
    a.add_argument("difffilename", type=str, help="name of diff file.")
    a.add_argument("evenfilename", type=str, help="name of even file.")
    a.add_argument("oddfilename", type=str, help="name of odd file.")
    a.add_argument("--nbl_per_load", type=int, default=None, help="Maximum number of baselines to load at once. uvh5 to uvh5 only.")
    return a

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
    a.add_argument("--nbl_per_load", type=int, default=None, help="Maximum number of baselines to load at once. uvh5 to uvh5 only."
                                                                  "Default loads the whole file. If 0 is provided, also loads whole file.")
    a.add_argument("--gain_convention", type=str, default='divide',
                   help="'divide' means V_obs = gi gj* V_true, 'multiply' means V_true = gi gj* V_obs.")
    a.add_argument("--redundant_solution", default=False, action="store_true",
                   help="If True, average gain ratios in redundant groups to recalibrate e.g. redcal solutions.")
    a.add_argument("--clobber", default=False, action="store_true", help='overwrites existing file at outfile')
    a.add_argument("--redundant_average", default=False, action="store_true", help="Redundantly average calibrated data.")
    return a

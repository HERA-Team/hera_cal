# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

"""Module for applying calibration solutions to visibility data, both in memory and on disk."""

from __future__ import absolute_import, division, print_function
import numpy as np
from hera_cal import io
from pyuvdata import UVCal, UVData
import argparse
from hera_cal.datacontainer import DataContainer
from hera_cal import utils


def recalibrate_in_place(data, data_flags, new_gains, cal_flags, old_gains=None, gain_convention='divide'):
    '''Update data and data_flags in place, taking out old calibration solutions, putting in
    new calibration solutions, and updating flags from those calibration solutions. Previously
    flagged data is modified, but left flagged. Missing antennas from either the new gains or (if it's not None),
    the old gains are automatically flagged in the data's visibilities that involves those antennas.

    Arguments:
        data: DataContainer containing baseline-pol complex visibility data. This is modified in place.
        data_flags: DataContainer containing data flags. This is modified in place. Can also be fed as a
            data weights dictionary with float dtype. In this case, wgts of 0 are treated as flagged
            data and non-zero wgts are unflagged data.
        new_gains: Dictionary of complex calibration gains to apply with keys like (1,'x')
        cal_flags: Dictionary with keys like (1,'x') of per-antenna boolean flags to update data_flags
            if either antenna in a visibility is flagged.
        old_gains: Dictionary of complex calibration gains to take out with keys like (1,'x').
            Default of None implies that the data is raw (i.e. uncalibrated).
        gain_convention: str, either 'divide' or 'multiply'. 'divide' means V_obs = gi gj* V_true,
            'multiply' means V_true = gi gj* V_obs. Assumed to be the same for new_gains and old_gains.
    '''
    # get datatype of data_flags to determine if flags or wgts
    if np.all([(df.dtype == np.bool) for df in data_flags.values()]):
        bool_flags = True
    elif np.all([(df.dtype == np.float) for df in data_flags.values()]):
        bool_flags = False
        wgts = data_flags
        data_flags = DataContainer(dict(map(lambda k: (k, ~wgts[k].astype(np.bool)), wgts.keys())))
    else:
        raise ValueError("didn't recognize dtype of data_flags")

    # loop over keys
    for (i, j, pol) in data.keys():
        ap1, ap2 = utils.split_pol(pol)
        # Check to see that all necessary antennas are present in the gains
        if (i, ap1) in new_gains and (j, ap2) in new_gains and (old_gains is None
                                                                or ((i, ap1) in old_gains and (j, ap2) in old_gains)):
            gigj_new = new_gains[(i, ap1)] * np.conj(new_gains[(j, ap2)])
            if old_gains is not None:
                gigj_old = old_gains[(i, ap1)] * np.conj(old_gains[(j, ap2)])
            else:
                gigj_old = np.ones_like(gigj_new)
            # update all the data, even if it was flagged
            if gain_convention == 'divide':
                data[(i, j, pol)] *= (gigj_old / gigj_new)
            elif gain_convention == 'multiply':
                data[(i, j, pol)] *= (gigj_new / gigj_old)
            else:
                raise KeyError("gain_convention must be either 'divide' or 'multiply'.")
            # update data flags
            if bool_flags:
                # treat as flags
                data_flags[(i, j, pol)][cal_flags[(i, ap1)]] = True
                data_flags[(i, j, pol)][cal_flags[(j, ap2)]] = True
            else:
                # treat as data weights
                wgts[(i, j, pol)][cal_flags[(i, ap1)]] = 0.0
                wgts[(i, j, pol)][cal_flags[(j, ap2)]] = 0.0
        else:
            # If any antenna is missing from the gains, the data is flagged
            if bool_flags:
                data_flags[(i, j, pol)] = np.ones_like(data_flags[(i, j, pol)], dtype=np.bool)
            else:
                wgts[(i, j, pol)] = np.zeros_like(wgts[(i, j, pol)], dtype=np.float)


def apply_cal(data_infilename, data_outfilename, new_calibration, old_calibration=None, flags_npz=None,
              flag_nchan_low=0, flag_nchan_high=0, filetype_in='uvh5', filetype_out='uvh5',
              nbl_per_load=None, gain_convention='divide', add_to_history='', clobber=False, **kwargs):
    '''Update the calibration solution and flags on the data, writing to a new file. Takes out old calibration
    and puts in new calibration solution, including its flags. Also enables appending to history.

    Arguments:
        data_infilename: filename of the data to be calibrated.
        data_outfilename: filename of the resultant data file with the new calibration and flags.
        new_calibration: filename of the calfits file (or a list of filenames) for the calibration
            to be applied, along with its new flags (if any).
        old_calibration: filename of the calfits file (or a list of filenames) for the calibration
            to be unapplied. Default None means that the input data is raw (i.e. uncalibrated).
        flags_npz: optional path to npz file containing just flags to be ORed with flags in input data
        flag_nchan_low: integer number of channels at the low frequency end of the band to always flag (default 0)
        flag_nchan_high: integer number of channels at the high frequency end of the band to always flag (default 0)
        filetype_in: type of data infile. Supports 'miriad', 'uvfits', and 'uvh5'.
        filetype_out: type of data outfile. Supports 'miriad', 'uvfits', and 'uvh5'.
        nbl_per_load: maximum number of baselines to load at once. Default (None) is to load the whole file at once.
            Enables partial reading and writing, but only for uvh5 to uvh5.
        gain_convention: str, either 'divide' or 'multiply'. 'divide' means V_obs = gi gj* V_true,
            'multiply' means V_true = gi gj* V_obs. Assumed to be the same for new_gains and old_gains.
        add_to_history: appends a string to the history of the output file. This will preceed combined histories
            of flags_npz (if applicable), new_calibration and, old_calibration (if applicable).
        clobber: if True, overwrites existing file at outfilename
        kwargs: dictionary mapping updated UVData attributes to their new values.
            See pyuvdata.UVData documentation for more info.
    '''

    # optionally load npz flags
    if flags_npz is not None:
        npz_flags = np.load(flags_npz)
        add_to_history += ' FLAGS_NPZ_HISTORY: ' + str(npz_flags['history']) + '\n'
        npz_flag_dc = io.load_npz_flags(flags_npz)

    # load new calibration solution
    hc = io.HERACal(new_calibration)
    new_gains, new_flags, _, _ = hc.read()
    add_to_history += ' NEW_CALFITS_HISTORY: ' + hc.history + '\n'

    # load old calibration solution
    if old_calibration is not None:
        old_hc = io.HERACal(old_calibration)
        old_calibration, _, _, _ = old_hc.read()
        add_to_history += ' OLD_CALFITS_HISTORY: ' + old_hc.history + '\n'

    # partial loading and writing using uvh5
    if nbl_per_load is not None:
        if not (filetype_in is 'uvh5') or not (filetype_out is 'uvh5'):
            raise NotImplementedError('Partial writing is not implemented for non-uvh5 I/O.')
        hd = io.HERAData(data_infilename, filetype='uvh5')
        for attribute, value in kwargs.items():
            hd.__setattr__(attribute, value)
        for data, data_flags, _ in hd.iterate_over_bls(Nbls=nbl_per_load):
            for bl in data_flags.keys():
                # apply band edge flags
                data_flags[bl][:, 0:flag_nchan_low] = True
                data_flags[bl][:, data_flags[bl].shape[1] - flag_nchan_high:] = True
                # apply npz flags
                if flags_npz is not None:
                    data_flags[bl] = np.logical_or(data_flags[bl], npz_flag_dc[bl])
            recalibrate_in_place(data, data_flags, new_gains, new_flags,
                                 old_gains=old_calibration, gain_convention=gain_convention)
            hd.partial_write(data_outfilename, data=data, flags=data_flags,
                             inplace=True, clobber=clobber, add_to_history=add_to_history)

    # full data loading and writing
    else:
        hd = io.HERAData(data_infilename, filetype=filetype_in)
        hd.read()
        # apply npz flags
        if flags_npz is not None:
            hd.flag_array = np.logical_or(npz_flags['flag_array'], hd.flag_array)
        data, data_flags, _ = hd.build_datacontainers()
        for bl in data_flags.keys():
            # apply band edge flags
            data_flags[bl][:, 0:flag_nchan_low] = True
            data_flags[bl][:, data_flags[bl].shape[1] - flag_nchan_high:] = True
            # apply npz flags
            if flags_npz is not None:
                data_flags[bl] = np.logical_or(data_flags[bl], npz_flag_dc[bl])
        recalibrate_in_place(data, data_flags, new_gains, new_flags, old_gains=old_calibration, gain_convention=gain_convention)
        io.update_vis(data_infilename, data_outfilename, filetype_in=filetype_in, filetype_out=filetype_out,
                      data=data, flags=data_flags, add_to_history=add_to_history, clobber=clobber, **kwargs)


def apply_cal_argparser():
    '''Arg parser for commandline operation of apply_cal.'''
    a = argparse.ArgumentParser(description="Apply (and optionally, also unapply) a calfits file to visibility file.")
    a.add_argument("infilename", type=str, help="path to visibility data file to calibrate")
    a.add_argument("outfilename", type=str, help="path to new visibility results file")
    a.add_argument("--new_cal", type=str, default=None, nargs="+", help="path to new calibration calfits file (or files for cross-pol)")
    a.add_argument("--old_cal", type=str, default=None, nargs="+", help="path to old calibration calfits file to unapply (or files for cross-pol)")
    a.add_argument("--flags_npz", type=str, default=None, help="path to npz file of flags to OR with data flags")
    a.add_argument("--flag_nchan_low", type=int, default=0, help="integer number of channels at the low frequency end of the band to always flag (default 0)")
    a.add_argument("--flag_nchan_high", type=int, default=0, help="integer number of channels at the high frequency end of the band to always flag (default 0)")
    a.add_argument("--filetype_in", type=str, default='miriad', help='filetype of input data files')
    a.add_argument("--filetype_out", type=str, default='miriad', help='filetype of output data files')
    a.add_argument("--gain_convention", type=str, default='divide',
                   help="'divide' means V_obs = gi gj* V_true, 'multiply' means V_true = gi gj* V_obs.")
    a.add_argument("--clobber", default=False, action="store_true", help='overwrites existing file at outfile')
    a.add_argument("--vis_units", default=None, type=str, help="String to insert into vis_units attribute of output visibility file.")
    return a
# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

import numpy as np
import hera_cal.io as io
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
              flag_nchan_low=0, flag_nchan_high=0, filetype='miriad', gain_convention='divide',
              add_to_history='', clobber=False, **kwargs):
    '''Update the calibration solution and flags on the data, writing to a new file. Takes out old calibration
    and puts in new calibration solution, including its flags. Also enables appending to history.

    Arguments:
        data_infilename: filename (or UVData object) of the data file to be updated.
        data_outfilename: filename of the resultant data file with the new calibration and flags.
        new_calibration: filename of the calfits file (or a list of filenames) or UVCal object for the calibration
            to be applied, along with its new flags (if any).
        old_calibration: filename of the calfits file (or a list of filenames) or UVCal object for the calibration
            to be unapplied. Default None means that the input data is raw (i.e. uncalibrated).
        flags_npz: optional path to npz file containing just flags to be ORed with flags in input data
        flag_chan_low: integer number of channels at the low frequency end of the band to always flag (default 0)
        flag_chan_high: integer number of channels at the high frequency end of the band to always flag (default 0)
        filetype: filename for the new file, either 'miriad' or 'uvfits'
        gain_convention: str, either 'divide' or 'multiply'. 'divide' means V_obs = gi gj* V_true,
            'multiply' means V_true = gi gj* V_obs. Assumed to be the same for new_gains and old_gains.
        add_to_history: appends a string to the history of the output file. This will preceed combined histories
            of flags_npz (if applicable), new_calibration and, old_calibration (if applicable).
        clobber: if True, overwrites existing file at outfilename
        kwargs: dictionary mapping updated attributes to their new values.
            See pyuvdata.UVData documentation for more info.
    '''
    # load data, flags, and (optionally) npz flags
    uvd = UVData()
    if filetype == 'miriad':
        uvd.read_miriad(data_infilename)
    else:
        raise NotImplementedError('This function has not been implemented yet.')
    if flags_npz is not None:
        npz_flags = np.load(flags_npz)
        uvd.flag_array = np.logical_or(npz_flags['flag_array'], uvd.flag_array)
        add_to_history += ' FLAGS_NPZ_HISTORY: ' + str(npz_flags['history']) + '\n'
    data, data_flags = io.load_vis(uvd)

    # apply band edge flags
    for bl in data_flags.keys():
        data_flags[bl][:, 0:flag_nchan_low] = True
        data_flags[bl][:, data_flags[bl].shape[1] - flag_nchan_high:] = True

    # load new calibration solution
    if new_calibration is None:
        raise ValueError('Must provide a calibration solution to apply.')
    if isinstance(new_calibration, UVCal):
        uvc = new_calibration
    else:
        uvc = UVCal()
        uvc.read_calfits(new_calibration)
    add_to_history += ' NEW_CALFITS_HISTORY: ' + uvc.history + '\n'
    new_gains, new_flags = io.load_cal(uvc)

    # load old calibration solution
    if old_calibration is not None:
        if isinstance(old_calibration, UVCal):
            old_uvc = old_calibration
        else:
            old_uvc = UVCal()
            old_uvc.read_calfits(old_calibration)
        add_to_history += ' OLD_CALFITS_HISTORY: ' + old_uvc.history + '\n'
        old_calibration, _ = io.load_cal(old_uvc)

    recalibrate_in_place(data, data_flags, new_gains, new_flags, old_gains=old_calibration, gain_convention=gain_convention)
    io.update_vis(data_infilename, data_outfilename, filetype_in=filetype, filetype_out=filetype, data=data,
                  flags=data_flags, add_to_history=add_to_history, clobber=clobber, **kwargs)


def apply_cal_argparser():
    '''Arg parser for commandline operation of apply_cal.'''
    a = argparse.ArgumentParser(description="Apply (and optionally, also unapply) a calfits file to visibility file.")
    a.add_argument("infile", type=str, help="path to visibility data file to calibrate")
    a.add_argument("outfile", type=str, help="path to new visibility results file")
    a.add_argument("--new_cal", type=str, default=None, nargs="+", help="path to new calibration calfits file (or files for cross-pol)")
    a.add_argument("--old_cal", type=str, default=None, nargs="+", help="path to old calibration calfits file to unapply (or files for cross-pol)")
    a.add_argument("--flags_npz", type=str, default=None, help="path to npz file of flags to OR with data flags")
    a.add_argument("--flag_nchan_low", type=int, default=0, help="integer number of channels at the low frequency end of the band to always flag (default 0)")
    a.add_argument("--flag_nchan_high", type=int, default=0, help="integer number of channels at the high frequency end of the band to always flag (default 0)")
    a.add_argument("--filetype", type=str, default='miriad', help='filetype of input and output data files')
    a.add_argument("--gain_convention", type=str, default='divide',
                   help="'divide' means V_obs = gi gj* V_true, 'multiply' means V_true = gi gj* V_obs.")
    a.add_argument("--clobber", default=False, action="store_true", help='overwrites existing file at outfile')
    a.add_argument("--vis_units", default=None, type=str, help="String to insert into vis_units attribute of output visibility file.")
    return a

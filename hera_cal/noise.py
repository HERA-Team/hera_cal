# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

"""Module for predicting noise on visibilties using autocorrelations and for producing noise standard deviation data files."""

from __future__ import print_function, division, absolute_import

import numpy as np
import argparse

from . import io
from .utils import split_pol, predict_noise_variance_from_autos
from .apply_cal import calibrate_in_place
from .datacontainer import DataContainer


def per_antenna_noise_std(autos, dt=None, df=None):
    '''Predicts per-antenna noise using autocorrelation data. The result is a noise standard deviation. To predict
    the noise variance on e.g. (1, 2, 'xy'), one would use noise[1, 1, 'xx'] * noise[2, 2, 'yy'].
    
    Arguments:
        autos: DataContainer containing autocorrelation data
        dt: integration time in seconds. If None, will try infer this from the times stored in the DataContainer
        df: channel width in Hz. If None, will try to infer this from the frequencies stored in the DataContainer

    Returns:
        noise: DataContainer mapping autocorrelation baselines like (1, 1, 'xx') to predictions for the standard
            deviation of noise for that antenna.
    '''
    noise = {}
    for bl in autos:
        if (bl[0] == bl[1]) and (split_pol(bl[2])[0] == split_pol(bl[2])[1]):
            noise[bl] = np.sqrt(predict_noise_variance_from_autos(bl, autos, dt=dt, df=df))
    return DataContainer(noise)


def write_per_antenna_noise_std_from_autos(infile, outfile, calfile=None, gain_convention='divide', add_to_history='', clobber=False):
    '''Loads autocorrelations and uses them to predict the per-antenna noise standard deviation on visibilities (e.g. to predict
    the noise variance on e.g. (1, 2, 'xy'), one would use noise[1, 1, 'xx'] * noise[2, 2, 'yy']). Optionally applies calibration
    solutions with associated flags. Only reads and writes to .uhv5 files.

    Arguments:
        infile: string path to .uvh5 visibility data file from which to extract autocorrelations
        outfile: string path to .uvh5 output data file of noise standard deviations (saved as if they were autocorrelations).
        calfile: optional string path to .calfits calibration file to apply to autocorrelations before computing noise.
            Can also take a list of paths if the .calfits files can be combined into a single UVCal object.
        gain_convention: str, either 'divide' or 'multiply'. 'divide' means V_obs = gi gj* V_true,
            'multiply' means V_true = gi gj* V_obs.
        add_to_history: appends a string to the history of the infile when writing the outfile
        clobber: if True, overwrites existing file at outfile
    '''
    hd = io.HERAData(infile)
    auto_bls = [bl for bl in hd.bls if (bl[0] == bl[1] and split_pol(bl[2])[0] == split_pol(bl[2])[1])]
    autos, auto_flags, _ = hd.read(bls=auto_bls)
    if calfile is not None:
        hc = io.HERACal(calfile)
        gains, cal_flags, _, _ = hc.read()
        calibrate_in_place(autos, gains, data_flags=auto_flags, cal_flags=cal_flags, gain_convention=gain_convention)
    noise = per_antenna_noise_std(autos)
    hd.update(data=data, flags=auto_flags)
    hd.history += add_to_history
    hd.write_uvh5(outfile, clobber=clobber)


def noise_std_argparser():
    '''Arg parser for commandline operation of noise_from_autos.py'''
    a = argparse.ArgumentParser(description="Read autocorrelations from a .uvh5 file and write predictions for noise standard \
                                             deviations to disk as a .uvh5 file, optionally calibrating")
    a.add_argument("infile", type=str, help="path to .uvh5 visibility data file from which to extract autocorrelations")
    a.add_argument("outfile", type=str, help="path to .uvh5 output data file of noise standard deviations")
    a.add_argument("--calfile", type=str, default=None, nargs="+", help="optional path to new calibration calfits file (or files) to apply")
    a.add_argument("--gain_convention", type=str, default='divide',
                   help="'divide' means V_obs = gi gj* V_true, 'multiply' means V_true = gi gj* V_obs.")
    a.add_argument("--clobber", default=False, action="store_true", help='overwrites existing file at outfile')
    return a

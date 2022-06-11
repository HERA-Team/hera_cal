# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"""Module for extracting autocorrelations, with the option to calibrate before writing to disk."""

import argparse
from . import io
from . import utils
from .utils import split_pol
from .apply_cal import calibrate_in_place


def read_and_write_autocorrelations(infile, outfile, calfile=None, gain_convention='divide', add_to_history='', clobber=False):
    '''Extracts autocorrelations from visibility file on disk, optionally applies a calibration solution,
    then writes the result to disk. Only reads and writes .uvh5 files.

    Arguments:
        infile: string path to .uvh5 visibility data file from which to extract autocorrelations
        outfile: string path to .uvh5 output data file containing only autocorrelations
        calfile: optional string path to .calfits calibration file to apply to autocorrelations before writing.
            Can also take a list of paths if the .calfits files can be combined into a single UVCal object.
        gain_convention: str, either 'divide' or 'multiply'. 'divide' means V_obs = gi gj* V_true,
            'multiply' means V_true = gi gj* V_obs.
        add_to_history: appends a string to the history of the infile when writing the outfile
        clobber: if True, overwrites existing file at outfile
    '''
    hd = io.HERAData(infile)
    # remove cross correlations (bl[0] != bl[1]) and cross-polarizations ('xy' or 'yx')
    auto_bls = [bl for bl in hd.bls if (bl[0] == bl[1] and split_pol(bl[2])[0] == split_pol(bl[2])[1])]
    if calfile is not None:
        data, data_flags, _ = hd.read(bls=auto_bls)
        hc = io.HERACal(calfile)
        gains, cal_flags, _, _ = hc.read()
        calibrate_in_place(data, gains, data_flags=data_flags, cal_flags=cal_flags, gain_convention=gain_convention)
        hd.update(data=data, flags=data_flags)
    else:
        hd.read(bls=auto_bls, return_data=False)
    hd.history += utils.history_string(add_to_history)
    hd.write_uvh5(outfile, clobber=clobber)


def extract_autos_argparser():
    '''Arg parser for commandline operation of extract_autos.py.'''
    a = argparse.ArgumentParser(description="Extract autocorrelations from a .uvh5 file and write to disk as a .uvh5 file, optionally calibrating")
    a.add_argument("infile", type=str, help="path to .uvh5 visibility data file from which to extract autocorrelations")
    a.add_argument("outfile", type=str, help="path to .uvh5 output data file containing only autocorrelations")
    a.add_argument("--calfile", type=str, default=None, nargs="+", help="optional path to new calibration calfits file (or files) to apply")
    a.add_argument("--gain_convention", type=str, default='divide',
                   help="'divide' means V_obs = gi gj* V_true, 'multiply' means V_true = gi gj* V_obs.")
    a.add_argument("--clobber", default=False, action="store_true", help='overwrites existing file at outfile')
    return a

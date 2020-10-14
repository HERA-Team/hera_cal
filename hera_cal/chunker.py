#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License


from .io import HERAData
import argparse



def chunk_files(filenames, outputfile, filetype='uvh5', polarizations=None, spw=None, throw_away_flagged_bls=False):
    """A file chunker

    Parameters
    ----------
    filenames: list of strings
        list of filenames to chunk. Should be homogenous in blt.
    outpufile: str
        name of outputfile to write time-concatenated data too.
    filetype: str, optional
        file type to write out.
        Default is uvh5.
    polarizations: list of strs, optional
        Limit output to polarizations listed.
        Default None selects all polarizations.
    spw: 2-list or 2-tuple
        optional lower and upper channel range to select
    throw_away_flagged_bls: bool, optional
        if true, throw away baselines that are fully flagged.
        default is False.
    Returns
    -------
    Concatenated HERAData object.

    """
    hd = io.HERAData(filenames)
    read_args = {}
    if polarizations is not None:
        read_args['polarizations'] = polarizations
    if spw is not None:
        read_args['channels'] = np.arange(spw[0], spw[1]).astype(int)
    data, flags, nsamples = hd.read(axis='blt', **read_args)
    # throw away fully flagged baselines.
    if throw_away_flagged_bls:
        bls2keep = []
        for bl in data:
            if not np.all(flags[bl]):
                bls2keep.append(bl)
        # Throw away unflagged antennas.
        hd.select(bls=bls2keep)
    if filetype == 'uvh5':
        hd.write_uvh5(outputfile)
    elif file_type == 'miriad':
        hd.write_miriad(outputfile)
    elif file_type == 'uvfits':
        hd.write_uvfit(outputfile)

    return hd



def chunk_parser():
    """
    An argument parser for chunking.

    Parameters
    ----------
    N/A

    Returns
    -------
    Argument parser.
    """
    a = argparse.ArgumentParser(description="Chunk visibility files.")
    a.add_argument("filenames", type=str, nargs="+", help="list of filenames to chunk together.")
    a.add_argument("outputfile", type='str', help="Name of output file.")
    a.add_argument("--filetype", type=str, help="Type of output file. Default is uvh5" default="uvh5")
    a.add_argument("--polarizations", type=str, nargs="+", default=None, help="optional list of polarizations to select.")
    a.add_argument("--spw", type=str, nargs=2, defaults=None, help="optional 2-tuple of frequency channels to select.")
    a.add_argument("--throw_away_flagged_bls", default=False, action="store_true", help="Throw away baselines that are fully flagged.")

    return a

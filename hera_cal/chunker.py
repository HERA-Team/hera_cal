#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 the HERA Project
# Licensed under the MIT License


from . import io
import argparse
import numpy as np
import sys
import warnings
from pyuvdata import utils as uvutils


def chunk_files(filenames, inputfile, outputfile, chunk_size, type="data",
                polarizations=None, spw_range=None, throw_away_flagged_ants=False,
                clobber=False, ant_flag_yaml=None, **read_kwargs):
    """Chunk a list of data or cal files together into a single file.

    Parameters
    ----------
    filenames: list of strings
        list of filenames to chunk. Should be homogenous in blt.
    inputfile: string,
        name of the file within filenames to use for the start of the chunk.
        data between the index of input file and the index of inputfile + chunk_size
        will be chunked together.
    outpufile: str
        name of outputfile to write time-concatenated data too.
    chunk_size: int
        number of files to chunk after the index of the input file.
    type : str
        specify whether "data", "gains"
    polarizations: list of strs, optional
        Limit output to polarizations listed.
        Default None selects all polarizations.
    spw_range: 2-list or 2-tuple of integers
        optional lower and upper channel range to select
    throw_away_flagged_ants: bool, optional
        if true, throw away baselines that are fully flagged.
        default is False.
    clobber: bool, optional
        if true, overwrite any preexisting output files.
        defualt is false.
    flag_yaml : str, optional
        yaml file with list of antennas to flag and throw away if throw_away_flagged_ants is True
    read_kwargs: kwargs dict
        additional kwargs for for io.HERAData.read()
    Returns
    -------
    None

    """
    filenames = sorted(filenames)
    start = filenames.index(inputfile)
    end = start + chunk_size
    if type == 'data':
        chunked_files = io.HERAData(filenames[start:end])
    elif type == 'gains':
        chunked_files = io.HERACal(filenames[start:end])
    else:
        raise ValueError("Invalid type provided. Must be in ['data', 'gains']")
    read_args = {}
    if type == 'data':
        if polarizations is None:
            if len(chunked_files.filepaths) > 1:
                polarizations = list(chunked_files.pols.values())[0]
            else:
                polarizations = chunked_files.pols
        if spw_range is None:
            spw_range = (0, chunked_files.Nfreqs)
        data, flags, nsamples = chunked_files.read(polarizations=polarizations,
                                                   freq_chans=range(spw_range[0], spw_range[1]), **read_kwargs)
    elif type == 'gains':
        chunked_files.read()
        if polarizations is None:
            polarizations = [pol[1:] for pol in chunked_files.pols]
        if spw_range is None:
            spw_range = (0, chunked_files.Nfreqs)
        # convert polarizations to jones integers.
        jones = [uvutils.polstr2num(pol, x_orientation=chunked_files.x_orientation) for pol in polarizations]
        chunked_files.select(freq_chans=np.arange(spw_range[0], spw_range[1]).astype(int), jones=jones)
    # throw away fully flagged baselines.
    if throw_away_flagged_ants:
        from hera_qm.utils import apply_yaml_flags
        chunked_files = apply_yaml_flags(chunked_files, ant_flag_yaml, flag_freqs=False, flag_times=False,
                                         flag_ants=True, ant_indices_only=True, throw_away_flagged_ants=True)
    if type == 'data':
        chunked_files.write_uvh5(outputfile, clobber=clobber)
    elif type == 'gains':
        chunked_files.write_calfits(outputfile, clobber=clobber)


def chunk_parser():
    """
    An argument parser for chunking.

    Parameters
    ----------
    None

    Returns
    -------
    ap: argparse.ArgumentParser object.
        An argument parser.
    """
    ap = argparse.ArgumentParser(description="Chunk visibility files.")
    ap.add_argument("filenames", type=str, nargs="+", help="list of filenames to chunk together.")
    ap.add_argument("inputfile", type=str, help="name of input file to start chunk at.")
    ap.add_argument("outputfile", type=str, help="Name of output file.")
    ap.add_argument("chunk_size", type=int, help="Number of files after filenames to chunk.")
    ap.add_argument("--type", type=str, help="Specify whether you are chunking 'gains' or 'data'", default="data")
    ap.add_argument("--polarizations", type=str, nargs="+", default=None, help="optional list of polarizations to select.")
    ap.add_argument("--spw_range", type=int, nargs=2, default=None, help="optional 2-tuple of frequency channels to select.")
    ap.add_argument("--clobber", default=False, action="store_true", help="overwrite output if it exists.")
    ap.add_argument("--throw_away_flagged_ants", default=False, action="store_true", help="throw away flagged baselines.")
    ap.add_argument("--ant_flag_yaml", default=None, help="path to yaml file with flagged data.")
    return ap

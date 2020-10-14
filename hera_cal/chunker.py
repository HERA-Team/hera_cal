#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License


from .io import HERAData, HERACal
import argparse



def chunk_data_files(filenames, inputfile, outputfile, chunk_size, filetype='uvh5',
                     polarizations=None, spw_range=None, throw_away_flagged_bls=False):
    """A data file chunker

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
    spw_range: 2-list or 2-tuple
        optional lower and upper channel range to select
    throw_away_flagged_bls: bool, optional
        if true, throw away baselines that are fully flagged.
        default is False.
    Returns
    -------
    Concatenated HERAData object.

    """
    start = filenames.index(inputfile)
    end = start + chunk_size
    hd = io.HERAData(filenames[start:end])
    read_args = {}
    if polarizations is not None:
        read_args['polarizations'] = polarizations
    if spw_range is not None:
        read_args['channels'] = np.arange(spw_range[0], spw_range[1]).astype(int)
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



def chunk_data_parser():
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
    a.add_argument("inputfile", type=str, help="name of input file to start chunk at.")
    a.add_argument("outputfile", type='str', help="Name of output file.")
    a.add_argument("chunk_size", type=int, help="Number of files after filenames to chunk.")
    a.add_argument("--filetype", type=str, help="Type of output file. Default is uvh5" default="uvh5")
    a.add_argument("--polarizations", type=str, nargs="+", default=None, help="optional list of polarizations to select.")
    a.add_argument("--spw_range", type=str, nargs=2, defaults=None, help="optional 2-tuple of frequency channels to select.")
    a.add_argument("--throw_away_flagged_bls", default=False, action="store_true", help="Throw away baselines that are fully flagged.")
    return a



def chunk_cal_files(filenames, inputfile, outputfile, chunk_size, spw=None):
    """A calibration file chunker

    Parameters
    ----------
    filenames: list of strings
        list of filenames to chunk. Should be homogenous in blt.
    inputfile: str
        name of intput file to start chunk at.
    outpufile: str
        name of outputfile to write time-concatenated data too.
    chunk_size: int
        number of files after inputfile to include in chunk
    spw: 2-tuple or 2-list of ints
        channels to keep. Default, None, keeps all channels.
    Returns
    -------
    Concatenated HERACal object.

    """
    start = filenames.index(inputfile)
    end = start + chunk_size
    hc = io.HERACal(filenames[start:end])
    hc.read()
    # throw away fully flagged baselines.
    if spw_range is not None:
        hc.select(channels=np.arange(spw_range[0], spw_range[1]).astype(int))
    hc.write_calfits(outputfile)
    return hc

def chunk_cal_parser():
    """
    An argument parser for cal chunking.

    Parameters
    ----------
    N/A

    Returns
    -------
    Argument parser.
    """
    a = argparse.ArgumentParser(description="Chunk calibration files.")
    a.add_argument("filenames", type=str, nargs="+", help="list of filenames to chunk together.")
    a.add_argument("outputfile", type='str', help="Name of output file.")
    a.add_argument("inputfile", type='str', help='name of input file to start chunk at')
    a.add_argument("chunk_size", type=int, help="number of files after inputfile to include in chunk")
    a.add_argument("--spw_range", type=int, nargs=2, help="min and max channel index to save in chunk.", default=None)
    return a

def chunk_data_parser():
    """
    An argument parser for data chunking.

    Parameters
    ----------
    N/A

    Returns
    -------
    Argument parser.
    """
    a = argparse.ArgumentParser(description="Chunk visibility files.")
    a.add_argument("filenames", type=str, nargs="+", help="list of filenames to chunk together.")
    a.add_argument("inputfile", type=str, help="name of input file to start chunk at.")
    a.add_argument("outputfile", type='str', help="Name of output file.")
    a.add_argument("chunk_size", type=int, help="Number of files after filenames to chunk.")
    a.add_argument("--filetype", type=str, help="Type of output file. Default is uvh5" default="uvh5")
    a.add_argument("--polarizations", type=str, nargs="+", default=None, help="optional list of polarizations to select.")
    a.add_argument("--spw_range", type=int, nargs=2, defaults=None, help="optional 2-tuple of frequency channels to select.")
    a.add_argument("--throw_away_flagged_bls", default=False, action="store_true", help="Throw away baselines that are fully flagged.")
    return a

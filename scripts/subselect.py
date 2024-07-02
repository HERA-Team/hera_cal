#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 the HERA Project
# Licensed under the MIT License

"""Command-line interface for subselecting parts of a data file and writing a new file out.

This is useful in that it essentially supports the full range of functionality
of the UVData select() method, but provides useful ways of specifying the selection
criteria via command-line arguments.
"""

from hera_cal import io
from hera_cal._cli_tools import parse_args, run_with_profiling, filter_kwargs
from argparse import ArgumentParser
import os
import numpy as np
from pyuvdata import UVData
import logging
logger = logging.getLogger("hera_cal.subselect")

ap = ArgumentParser(
    description="Sub-select parts of a data file and write out the results."
)

ap.add_argument("infile", type=str, help="Path to input file.")
ap.add_argument("outfile", type=str, help="Path to output file.")

freq_grp = ap.add_argument_group("Frequency Selection")
freq_grp.add_argument("--freq-min", type=float, help="Minimum frequency to select.", default=None)
freq_grp.add_argument("--freq-max", type=float, help="Maximum frequency to select.", default=None)
freq_grp.add_argument("--freq-spws", type=str, help="Comma-separated list of frequency spws to select, eg '0~100,200~300'.", default=None)

time_grp = ap.add_argument_group("Time Selection")
time_grp.add_argument("--time-min", type=float, help="Minimum time to select, JD", default=None)
time_grp.add_argument("--time-max", type=float, help="Maximum time to select, JD", default=None)
time_grp.add_argument("--time-idxs", type=str, help="Comma-separated list of time indices to select, eg '0~100,200~300'.", default=None)
time_grp.add_argument("--lst-min", type=float, help="Minimum LST to select", default=None)
time_grp.add_argument("--lst-max", type=float, help="Maximum time to select", default=None)
time_grp.add_argument("--lst-in-hours", action="store_true", help="whether LST bounds are in hours (otherwise radians)")

pol_grp = ap.add_argument_group("Polarization Selection")
pol_grp.add_argument("--pols", type=str, help="Comma-separated list of polarizations to select.", default=None)

ant_grp = ap.add_argument_group("Antenna Selection")
ant_grp.add_argument("--antennas", type=str, help="Comma-separated list of antenna ranges to select.", default=None)
ant_grp.add_argument("--calfile", type=str, help="Path to calfile to use for antenna selection.", default=None)

bl_grp = ap.add_argument_group("Baseline Selection")
bl_grp.add_argument("--bls", type=str, help="Comma-separated list of baselines to select, eg '3:5,15:27'", default=None)
bl_grp.add_argument("--only-autos", action="store_true", help="Only select auto-correlations.")
bl_grp.add_argument("--only-cross", action="store_true", help="Only select cross-correlations.")
bl_grp.add_argument("--min-bl-length", type=float, help="Minimum baseline length to select, metres", default=None)
bl_grp.add_argument("--max-bl-length", type=float, help="Maximum baseline length to select, metres", default=None)
bl_grp.add_argument("--min-ew-length", type=float, help="Minimum EW baseline length to select, metres", default=None)
bl_grp.add_argument("--max-ew-length", type=float, help="Maximum EW baseline length to select, metres", default=None)

io_grp = ap.add_argument_group("I/O")
io_grp.add_argument("--clobber", action="store_true", help="Overwrite output file if it exists.")

check_grp = ap.add_argument_group("Checking Arguments")
check_grp.add_argument("--check", action="store_true", help="Check output file for consistency with input file.")
check_grp.add_argument("--check-acceptability", action="store_true", help="Check output file for consistency with input file.")
check_grp.add_argument("--check-uvw-strict", action="store_true", help="Check whether UVW's are strictly consistent with antpos.")
check_grp.add_argument("--check-autos", action="store_true", help="Check that autos are real")
check_grp.add_argument("--fix-autos", action="store_true", help="Fix autos if they are not real")

args = parse_args(ap)
kw = filter_kwargs(vars(args))


def select(
    infile: str, outfile: str,
    freq_min=None, freq_max=None, freq_spws=None, time_min=None, time_max=None,
    time_idxs=None, lst_min=None, lst_max=None, lst_in_hours=False, pols=None,
    antennas=None, calfile=None, bls=None, only_autos=False, only_cross=False,
    min_bl_length=None, max_bl_length=None, min_ew_length=None, max_ew_length=None,
    clobber=False, check=False, check_acceptability=False, check_uvw_strict=False,
    check_autos=False, fix_autos=False
):
    if not os.path.exists(infile):
        raise FileNotFoundError(f"File {ap} does not exist.")
    if os.path.exists(outfile) and not clobber:
        raise FileExistsError(f"File {outfile} exists and clobber is False.")

    hd = UVData()
    logger.info(f"Reading metadata from file {infile}")
    hd.read(infile, read_data=False)
    hd.use_future_array_shapes()

    # Get frequencies
    if freq_min is not None or freq_max is not None or freq_spws is not None:
        logger.info("Getting frequencies to read.")
        freqs = hd.freq_array.copy()
        if freq_min is not None:
            freqs = freqs[freqs >= freq_min]
        if freq_max is not None:
            freqs = freqs[freqs <= freq_max]
        if freq_spws is not None:
            freq_spws = [tuple(map(int, spw.split("~"))) for spw in freq_spws.split(",")]
            freqs = freqs[np.concatenate([np.arange(*spw) for spw in freq_spws])]
    else:
        freqs = None

    # Get times
    if time_min is not None or time_max is not None or time_idxs is not None:
        logger.info("Getting times to read.")
        times = np.unique(hd.time_array)

        if time_idxs is not None:
            time_bools = np.zeros_like(times, dtype=bool)
            time_idxs = [tuple(map(int, idx.split("~"))) for idx in time_idxs.split(",")]

            for idx in time_idxs:
                time_bools[idx[0]:idx[1]] = True
        else:
            time_bools = np.ones_like(times, dtype=bool)

        if time_min is not None:
            time_bools[times < time_min] = False
        if time_max is not None:
            time_bools[times > time_max] = False
        if lst_min is not None or lst_max is not None:
            lsts = np.unique(hd.lsts)
            if lst_in_hours:
                lsts *= 2 * np.pi / 24
            if lst_min is not None:
                time_bools[lsts < lst_min] = False
            if lst_max is not None:
                time_bools[lsts > lst_max] = False

        times = times[time_bools]
    else:
        times = None

    # Get polarizations
    if pols is not None:
        logger.info("Getting polarizations to read.")
        pols = [pol.upper() for pol in pols.split(",")]
    else:
        pols = None

    # Get antennas
    if antennas is not None or calfile is not None:
        logger.info("Getting antennas to read.")
        if antennas is not None:
            antennas = [tuple(map(int, ant.split("~"))) for ant in antennas.split(",")]
            antennas = np.concatenate([np.arange(*ant) for ant in antennas])
        else:
            antennas = hd.get_ants()

        if calfile is not None:
            hc = io.HERACal(calfile)
            gains, flags, quals, total_qual = hc.read()
            bad_ants = [ant for ant, flg in flags.items() if np.all(flg)]
            antennas = [ant for ant in antennas if ant not in bad_ants]
    else:
        antennas = None

    # Get baselines
    if (
        bls is not None or only_autos or only_cross or min_bl_length is not None or
        max_bl_length is not None or min_ew_length is not None or
        max_ew_length is not None
        or antennas is not None
    ):
        logger.info("Getting baselines to read.")
        if bls is not None:
            bls = [tuple(map(int, bl.split(":"))) for bl in bls.split(",")]
        else:
            bls = hd.get_antpairs()
            if antennas is not None:
                bls = [bl for bl in bls if bl[0] in antennas and bl[1] in antennas]
            if only_autos:
                bls = [bl for bl in bls if bl[0] == bl[1]]
            if only_cross:
                bls = [bl for bl in bls if bl[0] != bl[1]]
            if (
                min_bl_length is not None or max_bl_length is not None or
                min_ew_length is not None or max_ew_length is not None
            ):
                antpos, ants = hd.get_ENU_antpos()
                antpos = {ant: pos for ant, pos in zip(ants, antpos)}

                if min_bl_length is not None or max_bl_length is not None:
                    min_bl_length = min_bl_length or 0
                    max_bl_length = max_bl_length or np.inf

                    def bl_length(bl):
                        return np.sqrt(np.sum(np.square(antpos[bl[0]] - antpos[bl[1]])))

                    # bllens = np.sqrt(np.sum(np.square(hd.uvw_array), axis=1))
                    bls = [bl for bl in bls if min_bl_length <= bl_length(bl) <= max_bl_length]

                if min_ew_length is not None or max_ew_length is not None:
                    min_ew_length = min_ew_length or 0
                    max_ew_length = max_ew_length or np.inf

                    def ew_length(bl):
                        return np.abs(antpos[bl[0]][0] - antpos[bl[1]][0])

                    bls = [bl for bl in bls if min_ew_length <= ew_length(bl) <= max_ew_length]
    else:
        bls = None

    logger.info("Reading data.")
    hd.read(
        infile,
        bls=bls, times=times, frequencies=freqs, polarizations=pols,
        run_check=check,
        run_check_acceptability=check_acceptability,
        strict_uvw_antpos_check=check_uvw_strict,
        check_autos=check_autos,
        fix_autos=fix_autos,
    )

    logger.info(f"Writing data to {outfile}")
    hd.write_uvh5(
        outfile,
        run_check=check,
        run_check_acceptability=check_acceptability,
        strict_uvw_antpos_check=check_uvw_strict,
        check_autos=check_autos,
        fix_autos=fix_autos,
        clobber=clobber
    )


run_with_profiling(
    select,
    args,
    **kw
)

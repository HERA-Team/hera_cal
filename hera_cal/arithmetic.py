#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

from . import io
import argparse
import numpy as np
import sys
import warnings


def sum_files(file_list, outfilename, flag_mode="and", nsamples_mode="average", clobber=False):
    """Add the contents of different pyuvdata files together.

    Arguments:
        file_list: list
            list of strings specifying paths of files to be added together.
        outfilename: str
            string specifying outfile to write to.
        flag_mode: str, optional
            string specifying out how to combine flags. Options are "and" and "or".
            default is "and".
        nsamples_mode: str, optional
            specify how to combine nsamples. Options are "sum": add together or
            "average", average them.
        clobber: bool, optional
            If true, ovewrite output if it exists.
            Default is False.

    Returns
    -------
    summed HERAData.
    """
    hd = io.HERAData(file_list[0])
    d, f, n = hd.read()
    if nsample_mode == "average":
        n /= len(file_list)
    for dfile in file_list:
        hdt = io.HERAData(dfile)
        dt, ft, nt = hdt.read()
        d = d + dt
        for k in f:
            if flag_mode == "and":
                f[k] = f[k] & ft[k]
            elif flag_mode  == "or":
                f[k] = f[k] | ft[k]
            else:
                raise ValueError(f"{flag_mode} is an invalid flag_mode. valid modes are ('or', 'and')")
            if nsample_mode == "sum":
                n[k] = n[k] + nt[k]
            elif nsample_mode  == "average":
                n[k] = n[k] + nt[k] / len(file_list)
            else:
                raise ValueError(f"{nsample_mode} is an invalid nsample_mode. valid modes are ('sum', 'average')")
    hd.update(data=d, nsamples=n, flags=f)
    hd.write_uvh5(outfilename)
    return hd


def sum_diff_2_even_odd(sum_infilename, diff_infilename, even_outfilename, odd_outfilename,
                        nbl_per_load=None, filetype_in='uvh5', external_flags=None,
                        overwrite_data_flags=False, clobber=False, polarizations=None):
    """Generate even and odd data sets from sum and diff

    Arguments:
        sum_infilename: str
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
        filetype_in: str, optional
            file_type
        external_flags: str, optional
            Name of external flags to apply.
        overwrite_data_flags: bool, optional
            if true, ovewrite flags of non-fully flagged baselines with external flags.
        clobber: bool, optional
            clobber output files if they already exist.
        polarizations: list of strs, optional
            list of string pols to include in output. If no provided

    Returns
    -------
    sum, diff HERAData objects.
    """
    hd_sum = io.HERAData(sum_infilename, filetype=filetype_in)
    hd_diff = io.HERAData(diff_infilename, filetype=filetype_in)
    # set tolerance to one tenth of an integration time.
    if external_flags is not None:
        external_flags = UVFlag(external_flags)
        atol = np.mean(np.diff(external_flags.time_array)) / 10.
        times_select = np.unique([t for t in external_flags.time_array if np.any(np.isclose(t, hd_sum.times, rtol=0, atol=atol))])
        atol = np.mean(np.diff(external_flags.freq_array.squeeze())) / 10.
        freqs_select = np.unique([f for f in external_flags.freq_array if np.any(np.isclose(f, hd_sum.freqs,rtol=0, atol=atol))])
        external_flags.select(times=times_select, frequencies=freqs_select)
    # select external flags by time and frequency
    if polarizations is None:
        if filetype_in == 'uvh5':
            polarizations = hd_sum.pols
        else:
            raise NotImplementedError("Must specify pols if operating on non uvh5 data.")
    if nbl_per_load is not None:
        if not ((filetype_in == 'uvh5') and (filetype_out == 'uvh5')):
            raise NotImplementedError('Partial writing is not implemented for non-uvh5 I/O.')
        for sum, sum_flags, sum_nsamples in hd_sum.iterate_over_bls(Nbls=nbl_per_load, pols_to_load=polarizations):
            diff, diff_flags, diff_nsamples = hd_diff.load(bls=list(sum.keys()))
            sum = (sum + diff)
            diff = sum - diff - diff
            for k in sum_flags:
                sum_flags[k] = sum_flags[k]
                diff_flags[k] = sum_flags[k]
                diff_nsamples[k] = sum_nsamples[k]
                sum_nsamples[k] = diff_nsamples[k]
                if overwrite_data_flags and not np.all(sum_flags[k]) and external_flags is not None:
                    sum_flags[k][:] = False
                    diff_flags[k][:] = False
            hd_sum.update(data=sum, flags=sum_flags, nsamples=sum_nsamples)
            hd_diff.update(data=diff, flags=diff_flags, nsamples=diff_nsamples)
            # set time array to avoid floating point mismatches.
            if external_flags is not None:
                external_flags.time_array = np.unique(hd_sum.time_array)
                from hera_qm.xrfi import flag_apply
                flag_apply(external_flags, hd_sum, force_pol=True, keep_existing=True)
                flag_apply(external_flags, hd_diff, force_pol=True, keep_existing=True)
            hd_sum.partial_write(even_outfilename, inplace=True, clobber=clobber)
            hd_diff.partial_write(odd_outfilename, inplace=True, clobber=clobber)
    else:
        sum, sum_flags, sum_nsamples = hd_sum.read(polarizations=polarizations)
        diff, diff_flags, diff_nsamples = hd_diff.read(polarizations=polarizations)
        sum = (sum + diff)
        diff = sum - diff - diff
        for k in sum_flags:
            sum_flags[k] = sum_flags[k]
            diff_flags[k] = sum_flags[k]
            diff_nsamples[k] = sum_nsamples[k]
            sum_nsamples[k] = diff_nsamples[k]
            if overwrite_data_flags and not np.all(sum_flags[k]) and external_flags is not None:
                sum_flags[k][:] = False
                diff_flags[k][:] = False
        hd_sum.update(data=sum, flags=sum_flags, nsamples=sum_nsamples)
        hd_diff.update(data=diff, flags=diff_flags, nsamples=diff_nsamples)
        # set time array to avoid floating point mismatches.
        if external_flags is not None:
            external_flags.time_array = np.unique(hd_sum.time_array)
            from hera_qm.xrfi import flag_apply
            flag_apply(external_flags, hd_sum, force_pol=True, keep_existing=True)
            flag_apply(external_flags, hd_diff, force_pol=True, keep_existing=True)
        hd_sum.write_uvh5(even_outfilename, clobber=clobber)
        hd_diff.write_uvh5(odd_outfilename, clobber=clobber)
    return hd_sum, hd_diff

def sum_files_argparser():
    a = argparse.ArgumentParser(description="Add together the data from a list of files.")
    a.add_argument("file_list", type=str, nargs="+", help="list of files to add together.")
    a.add_argument("outfilename", type=str, help="name of output file to store sum in.")
    a.add_argument("--flag_mode", type=str, default="and", help="mode to combine flags. Options are 'and' and 'or'.")
    a.add_argument("--nsample_mode", type=str, default="average", help="mode to combine nsamples. Options are 'sum' and 'average'.")
    a.add_argument("--clobber", default=False, action="store_true", help="overwrite output if it already exists.")
    return a

def sum_diff_2_even_odd_argparser():
    '''Arg parser for even/odd to sum/diff function.'''
    a = argparse.ArgumentParser(description="Convert a sum and a diff file to an even and an odd file.")
    a.add_argument("sumfilename", type=str, help="name of sum file.")
    a.add_argument("difffilename", type=str, help="name of diff file.")
    a.add_argument("evenfilename", type=str, help="name of even file.")
    a.add_argument("oddfilename", type=str, help="name of odd file.")
    a.add_argument("--nbl_per_load", type=int, default=None, help="Maximum number of baselines to load at once. uvh5 to uvh5 only.")
    a.add_argument("--clobber", default=False, action="store_true")
    a.add_argument("--external_flags", default=None, type=str, help="name of external flag file(s) to apply.")
    a.add_argument("--overwrite_data_flags", default=False, action="store_true", help="Overwrite existing data flags with external flags\
                                                                                       for antennas that are not entirely flagged.")
    a.add_argument("--polarizations", default=None, type=str, nargs="+", help="polarizations to include in output.")
    return a

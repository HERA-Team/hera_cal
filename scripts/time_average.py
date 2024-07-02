#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 the HERA Project
# Licensed under the MIT License

from hera_cal import frf
from hera_cal import io
from hera_cal._cli_tools import parse_args, run_with_profiling

a = frf.time_average_argparser()
args = parse_args(a)

# only use baseline_list if cornerturn requested.
if args.cornerturnfile is not None:
    baseline_list = io.baselines_from_filelist_position(filename=args.cornerturnfile,
                                                        filelist=args.input_data_list)
else:
    baseline_list = None

run_with_profiling(
    frf.time_avg_data_and_write,
    args,
    flag_output=args.flag_output,
    input_data_list=args.input_data_list,
    baseline_list=baseline_list,
    output_data=args.output_data,
    t_avg=args.t_avg, rephase=args.rephase,
    # wgt_by_nsample is default True in frf.time_avg_data_and_write
    # for this reason, the argparser requests the negation.
    filetype=args.filetype,
    wgt_by_nsample=not (args.dont_wgt_by_nsample),
    wgt_by_favg_nsample=args.wgt_by_favg_nsample,
    clobber=args.clobber, verbose=args.verbose, ninterleave=args.ninterleave,
    equalize_interleave_times=args.equalize_interleave_times,
    equalize_interleave_ntimes=args.equalize_interleave_ntimes)

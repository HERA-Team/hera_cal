#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

from hera_cal import frf
from hera_cal import io

a = frf.time_average_argparser(multifile=True)
args = a.parse_args()

baseline_list = io.baselines_from_filelist_position(filename=args.input_data,
                                                    filelist=args.input_data_list)
print(baseline_list[0])
for f in args.input_data_list:
    hd = io.HERAData(f)
if len(baseline_list) > 0:
    frf.time_avg_data_and_write_baseline_list(
                                flag_output=args.flag_output,
                                input_data_list=args.input_data_list,
                                interleaved_input_data_list=args.interleaved_input_data_list,
                                interleaved_output_data=args.interleaved_output_data,
                                interleaved_diff_odd=args.interleaved_diff_odd,
                                interleaved_diff_even=args.interleaved_diff_even,
                                baseline_list=baseline_list,
                                output_data=args.output_data,
                                t_avg=args.t_avg, rephase=args.rephase,
                                wgt_by_nsample=not(args.dont_wgt_by_nsample),
                                clobber=args.clobber, verbose=args.verbose)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

from hera_cal import frf

a = frf.time_average_argparser()
args = a.parse_args()

frf.time_avg_data_and_write(input_data=args.input_data,
                            output_data=args.output_data,
                            t_avg=args.t_avg, rephase=args.rephase,
                            wgt_by_nsample=not(args.dont_wgt_by_nsample),
                            clobber=args.clobber, verbose=args.verbose,
                            flag_output=args.flag_output,
                            interleaved_input_data=args.interleaved_input_data,
                            interleaved_output_data=args.interleaved_output_data,
                            interleaved_diff_odd=args.interleaved_diff_odd,
                            interleaved_diff_even=args.interleaved_diff_even,)

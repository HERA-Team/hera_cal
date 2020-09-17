#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

from hera_cal import apply_cal

a = apply_cal.sum_diff_2_even_odd_argparser()
args = a.parse_args()

if args.nbl_per_load == "none":
    args.nbl_per_load = None

apply_cal.sum_diff_2_even_odd(args.sumfilename, args.difffilename,
                              args.evenfilename, args.oddfilename,
                              nbl_per_load=args.nbl_per_load)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

from hera_cal import arithmetic

a = arithmetic.sum_files_argparser()
args = a.parse_args()

arithmetic.sum_files(file_list=args.file_list,
                     outfilename=args.outfilename,
                     flag_mode=args.flag_mode,
                     nsample_mode=args.nsample_mode,
                     clobber=args.clobber)

#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"Command-line drive script for hera_cal.xtalk_filter. Only performs DAYENU Filtering"

from hera_cal import xtalk_filter

parser = xtalk_filter.reconstitute_xtalk_files_argparser()

a = parser.parse_args()

# Run Xtalk Filter
xtalk_filter.reconstitute_xtalk_files(templatefile=a.infilename,
                                      fragments=a.fragmentlist,
                                      outfilename=a.outfilename)

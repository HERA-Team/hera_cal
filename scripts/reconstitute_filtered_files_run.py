#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2020 the HERA Project
# Licensed under the MIT License

"Command line driver for changing from files chunking by baseline to files chunking by time.""

from hera_cal import xtalk_filter

parser = xtalk_filter.reconstitute_files_argparser()

a = parser.parse_args()

# Run Xtalk Filter
xtalk_filter.reconstitute_files(templatefile=a.infilename,
                                      fragments=a.fragmentlist,
                                      outfilename=a.outfilename)

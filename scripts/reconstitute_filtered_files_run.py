#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2020 the HERA Project
# Licensed under the MIT License

"Command line driver for changing from files chunking by baseline to files chunking by time."

from hera_cal import vis_clean

parser = vis_clean.reconstitute_files_argparser()

a = parser.parse_args()

# Run Xtalk Filter
vis_clean.reconstitute_files(templatefile=a.infilename,
                             fragments=a.fragmentlist,
                             outfilename=a.outfilename,
                             clobber=a.clobber)

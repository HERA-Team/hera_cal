#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2020 the HERA Project
# Licensed under the MIT License

"Command line driver for changing from files chunking by baseline to files chunking by time."

from hera_cal import vis_clean

parser = vis_clean.time_chunk_from_baseline_chunks_argparser()

a = parser.parse_args()

vis_clean.time_chunk_from_baseline_chunks(time_chunk_template=a.time_chunk_template,
                                          baseline_chunk_files=a.baseline_chunk_files, clobber=a.clobber,
                                          outfilename=a.outfilename, time_bounds=a.time_bounds)

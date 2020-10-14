#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License


""" Command Line Driver for data file chunker."""

from hera_cal import chunker

a = chunker.chunk_data_parser()
args = a.parse_args()

chunker.chunk_data_files(filenames=a.filenames, outputfile=a.outputfile,
                         inputfile=a.inputfile, filetype=a.filetype, polarizations=a.polarizations,
                         spw_range=a.spw_range, throw_away_flagged_bls=a.throw_away_flagged_bls)

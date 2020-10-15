#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License


""" Command Line Driver for data file chunker."""

from hera_cal import chunker

a = chunker.chunk_data_parser()
args = a.parse_args()

chunker.chunk_data_files(filenames=args.filenames, outputfile=args.outputfile,
                         chunk_size=args.chunk_size, clobber=args.clobber,
                         inputfile=args.inputfile, filetype=args.filetype, polarizations=args.polarizations,
                         spw_range=args.spw_range, throw_away_flagged_bls=args.throw_away_flagged_bls)

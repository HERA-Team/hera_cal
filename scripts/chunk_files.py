#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 the HERA Project
# Licensed under the MIT License


""" Command Line Driver for data file chunker."""

from hera_cal import chunker

ap = chunker.chunk_parser()
args = ap.parse_args()

chunker.chunk_files(filenames=args.filenames, outputfile=args.outputfile,
                    chunk_size=args.chunk_size, clobber=args.clobber, ant_flag_yaml=args.ant_flag_yaml,
                    inputfile=args.inputfile, type=args.type, polarizations=args.polarizations,
                    spw_range=args.spw_range, throw_away_flagged_ants=args.throw_away_flagged_ants)

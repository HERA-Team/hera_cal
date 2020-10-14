#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License


""" Command Line Driver for file chunker."""

from hera_cal import chunker

a = chunker.chunk_parser()
args = a.parse_args()

chunker.chunk_files(filenames=a.filenames, outputfile=a.outputfile,
                    filetype=a.filetype, polarizations=a.polarizations,
                    spw=a.spw, throw_away_flagged_bls=a.throw_away_flagged_bls)

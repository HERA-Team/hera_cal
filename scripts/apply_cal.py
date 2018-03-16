#!/usr/bin/env python2.7
"""Command-line drive script for hera_cal.apply_cal"""
import argparse
from hera_cal import apply_cal as ac
import sys

args = ac.apply_cal_argparser()
ac.apply_cal(args.infile, args.outfile, args.new_cal, old_calibration=args.old_cal,
             flags_npz=args.flags_npz, filetype=args.filetype, gain_convention=args.gain_convention,
             add_to_history=' '.join(sys.argv), clobber=args.clobber)

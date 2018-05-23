#!/usr/bin/env python2.7
"""Command-line drive script for hera_cal.apply_cal"""
import argparse
from hera_cal import apply_cal as ac
import sys

a = ac.apply_cal_argparser()
args = a.parse_args()

kwargs = {}
if args.vis_units is not None:
    kwargs['vis_units'] = args.vis_units

ac.apply_cal(args.infile, args.outfile, args.new_cal, old_calibration=args.old_cal, flags_npz=args.flags_npz, 
             flag_nchan_low=args.flag_nchan_low, flag_nchan_high=args.flag_nchan_high, filetype=args.filetype, 
             gain_convention=args.gain_convention, add_to_history = ' '.join(sys.argv), clobber=args.clobber,
             **kwargs)

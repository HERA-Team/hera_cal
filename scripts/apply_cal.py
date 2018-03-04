#! /usr/bin/env python
"""
apply *.calfits gain solution to visibility file(s).
"""
from hera_cal import io
import sys
import os

a = io.apply_cal_arg_parser()
args = a.parse_args()
history = ' '.join(sys.argv)

kwargs = dict(vars(args))
kwargs.pop('uvfiles')
kwargs.pop('apply_gains')
kwargs.pop('unapply_gains')
kwargs.pop('silence')
kwargs['history'] = history
flag_missing = args.noflag_missing == False
kwargs.pop('noflag_missing')
verbose = args.silence is False

io.apply_cal(args.uvfiles, apply_gain_files=args.apply_gains, unapply_gain_files=args.unapply_gains, 
             verbose=verbose, flag_missing=flag_missing, **kwargs)

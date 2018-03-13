#! /usr/bin/env python
"""
abscal.AbsCal drive script assuming data is already omnicalibrated
"""
from hera_cal import abscal
import sys
import os

a = abscal.omni_abscal_arg_parser()
args = a.parse_args()
history = ' '.join(sys.argv)

kwargs = dict(vars(args))
kwargs.pop('data_files')
kwargs.pop('model_files')
kwargs.pop('silence')
verbose = args.silence is False
kwargs.pop('data_is_omni_solution')
if args.data_is_omni_solution:
    kwargs['reweight'] = True
    kwargs['match_red_bls'] = True

abscal.abscal_run(args.data_files, args.model_files, verbose=verbose, history=history, **kwargs)


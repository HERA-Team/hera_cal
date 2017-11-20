#! /usr/bin/env python

from hera_cal import abscal
import sys
import os

a = abscal.abscal_arg_parser()
args = a.parse_args()
history = ' '.join(sys.argv)

abscal.run_abscal(args.data_files, args.model_files,
                  calfits_fname=args.calfits_fname,
                  overwrite=args.overwrite,
                  verbose=args.silence is False,
                  save=True)


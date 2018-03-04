#! /usr/bin/env python
"""
multiply together multiple *.calfits files
"""
from hera_cal import io
import sys
import os

a = io.combine_calfits_arg_parser()
args = a.parse_args()
history = ' '.join(sys.argv)

kwargs = dict(vars(args))
del kwargs['files']
del kwargs['fname']
verbose = kwargs['silence'] == False
del kwargs['silence']
broadcast_flags = kwargs['no_broadcast'] == False
del kwargs['no_broadcast']

io.combine_calfits(args.files, args.fname, broadcast_flags=broadcast_flags, verbose=verbose, **kwargs)

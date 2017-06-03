#! /usr/bin/env python
import optparse
import sys
from heracal.firstcal import firstcal_run

o = optparse.OptionParser()
o.set_usage("omni_run.py -C [calfile] -p [pol] [options] *.uvc")
a.scripting.add_standard_options(o, cal=True, pol=True)
o.add_option('--ubls', default='',
             help='Unique baselines to use, separated by commas (ex: 1_4,64_49).')
o.add_option('--ex_ants', default='',
             help='Antennas to exclude, separated by commas (ex: 1,4,64,49).')
o.add_option('--outpath', default=None,
             help='Output path of solution npz files. Default will be the same directory as the data files.')
o.add_option('--verbose', action='store_true',
             default=False, help='Turn on verbose.')
o.add_option('--finetune', action='store_false',
             default=True, help='Fine tune the delay fit.')
o.add_option('--average', action='store_true', default=False,
             help='Average all data before finding delays.')
o.add_option('--observer', default='Observer',
             help='optional observer input to fits file')
o.add_option('--git_hash_cal', default='None',
             help='optionally add the git hash of the cal repo')
o.add_option('--git_origin_cal', default='None',
             help='optionally add the git origin of the cal repo')
opts, files = o.parse_args(sys.argv[1:])

history = ' '.join(sys.argv)

firstcal_run(files, opts, history)

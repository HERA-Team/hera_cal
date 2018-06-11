#! /usr/bin/env python

import numpy
import optparse
from hera_cal import omni
from hera_qm import vis_metrics
from pyuvdata import UVData
import aipy as a
import sys

o = optparse.OptionParser()
o.set_usage("omni_run.py -C [calfile] [options] *.uvc")
a.scripting.add_standard_options(o, cal=True)
o.add_option('--frac', default=.3,
             help='Fraction of total number of antennas to flag as bad and write out')
o.add_option('--write', action='store_true',
             help='write out simple txt file of bad antennas/bls')
o.add_option('--ex_ants',
             help='list of known bad antennas to exclude from metrics.')
opts, args = o.parse_args(sys.argv[1:])

# read in miriad file to get frequency info for aa
uv = a.miriad.UV(args[0])
fqs = a.cal.get_freqs(uv['sdf'], uv['sfreq'], uv['nchan'])
del(uv)
# create antenna array
aa = a.cal.get_aa(opts.cal, fqs)
info = omni.aa_to_info(aa)  # we have no info here
reds = info.get_reds()

# parse ex_ants
ex_ants = []
if opts.ex_ants:
    for ant in opts.ex_ants.split(','):
        try:
            ex_ants.append(int(ant))
        except BaseException:
            pass


for filename in args:
    uvd = UVData()
    uvd.read_miriad(filename)
    if uvd.phase_type != 'drift':
        uvd.unphase_to_drift()
    data, flags = omni.UVData_to_dict([uvd])
    bad_ants = metrics.check_ants(reds, data, skip_ants=ex_ants)
    total_ba = ex_ants  # start string with known bad ants
    for ba in bad_ants:
        # check if bad ants count is larger than some number of antennas.
        if bad_ants[ba] > opts.frac * len(info.subsetant):
            # check if antenna
            if isinstance(ba[-1], str):
                ret_ba = ba[0]  # get antenna number of bad ant
            # else it's a baseline. Don't support this now
            else:
                pass
            total_ba.append(ret_ba)
    if opts.write:
        print 'Writing {0} to file'.format(total_ba)
        writefile = open(filename + '.badants.txt', 'w')
        writefile.write(','.join(map(str, total_ba)))

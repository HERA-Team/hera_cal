#! /usr/bin/env python

import numpy, optparse
from heracal import metrics, omni
from pyuvdata import UVData
import aipy as a
import sys

o = optparse.OptionParser()
a.scripting.add_standard_options(o, cal=True)
o.add_option('--frac', default=.3, 
             help='Fraction of total number of antennas to flag as bad and write out')
o.add_option('--write', action='store_true',
             help='write out simple txt file of bad antennas/bls')
opts,args = o.parse_args(sys.argv[1:])

# read in miriad file to get frequency info for aa
uv = a.miriad.UV(args[0])
fqs = a.cal.get_freqs(uv['sdf'], uv['sfreq'], uv['nchan'])
del(uv)
# create antenna array
aa = a.cal.get_aa(opts.cal, fqs)
info = omni.aa_to_info(aa)  # we have no info here
reds = info.get_reds()

for filename in args:
    uvd = UVData()
    uvd.read_miriad(filename)
    if uvd.phase_type != 'drift':
        uvd.unphase_to_drift()
    data, flags = omni.UVData_to_dict([uvd])
    bad_ants = metrics.check_ants(reds, data)
    total_ba_string = ''
    for ba in bad_ants:
        # check if bad ants count is larger than some number of antennas.
        if bad_ants[ba] > opts.frac*len(info.subsetant):
            print ba
            # check if antenna
            if type(ba[-1]) is str:
                ret_ba = ''.join(map(str,ba))[:-1]  # only get 1 pol
            # else it's a baseline
            else: 
                ret_ba = '('+','.join(map(str,ba))+')'
            total_ba_string += ret_ba + ','
    if opts.write:
        print 'Writing {0} to file'.format(total_ba_string)
        writefile = open(filename+'.badants.txt', 'w')
        writefile.write(total_ba_string)

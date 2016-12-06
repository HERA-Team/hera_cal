#! /usr/bin/env python
'''Takes raw firstcal files and rewrites to give the median gain solution.'''

import numpy as np, glob, optparse, sys
from omni import save_gains_fc

def load_dmv(files, offsets=False, verbose=False):
    delays = {}
    means = {}
    medians = {}
    vars = {}
    if offsets: ss = 'o'
    else: ss = 'd'
    for f in files:
        if verbose:
            print 'Reading %s'%f    
        npz = np.load(f)
        for key in npz.files:
            if key.startswith(ss):
                if key not in delays.keys(): 
                    delays[key] = npz[key]
                    continue 
                delays[key] = np.hstack((delays[key],npz[key]))
    
    for k in delays.keys():
        delays[k] = np.array(delays[k]).flatten()
        means[k] = np.mean(delays[k])
        medians[k] = np.median(delays[k])
        vars[k] = np.var(delays[k])
        
    return delays,means,medians,vars


o = optparse.OptionParser()
o.add_option('--mean', action='store_true', 
    help='Use the mean value of antenna delays for gain solutions.')
o.add_option('--median', action='store_true',
    help='Use the median value of antenna delays for gain solutions.')
opts,args = o.parse_args(sys.argv[1:])

if opts.mean and opts.median:
    raise ValueError('Must choose between mean and median values.')
    sys.exit(1)

fqs = np.load(args[0])['freqs']
_,means,medians,_ = load_dmv(args)
do = {}
for ant in medians.keys():
    ant = ant[1:]
    do[int(ant)] = [medians['d'+ant]]
for f in args:
    fname = '.'.join(f.split('.')[:-2] + ['median'])
    pol = fname.split('.')[3]
    save_gains_fc(do,fqs,pol[0],fname)



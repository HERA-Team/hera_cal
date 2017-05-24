#! /usr/bin/env python
import sys, os, math, optparse
o = optparse.OptionParser()
o.add_option('-w', '--wrap', dest='wrap', action='store_true',
    help='Instead of pulling nothing for indices off the end of the list, wrap around and repull arguments from the beginning.')
opts,args = o.parse_args(sys.argv[1:])
try:
    n = int(os.environ['SGE_TASK_FIRST'])
    m = int(os.environ['SGE_TASK_LAST'])
    i = int(os.environ['SGE_TASK_ID']) - 1
    if (m-n) <= len(args) or not opts.wrap:
        num = int(math.ceil(float(len(args)) / (m - n + 1)))
        print ' '.join(args[num*i:num*(i+1)])
    else:
        print args[i % len(args)]
except(KeyError,ValueError): print ' '.join(args)

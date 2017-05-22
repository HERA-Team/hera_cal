#! /usr/bin/env python
import numpy as np, optparse, sys
import aipy as a
from heracal import omni
from heracal import firstcal
from pyuvdata import UVData

o = optparse.OptionParser()
a.scripting.add_standard_options(o,cal=True,pol=True)
o.add_option('--ubls', default='', help='Unique baselines to use, separated by commas (ex: 1_4,64_49).')
o.add_option('--ex_ants', default='', help='Antennas to exclude, separated by commas (ex: 1,4,64,49).')
o.add_option('--outpath', default=None,help='Output path of solution npz files. Default will be the same directory as the data files.')
o.add_option('--verbose', action='store_true', default=False, help='Turn on verbose.')
o.add_option('--finetune', action='store_false', default=True, help='Fine tune the delay fit.')
o.add_option('--average', action='store_true', default=False, help='Average all data before finding delays.')
o.add_option('--observer', default='Observer', help='optional observer input to fits file')
o.add_option('--git_hash_cal', default='None', help='optionally add the git hash of the cal repo')
o.add_option('--git_origin_cal', default='None', help='optionally add the git origin of the cal repo')
opts,args = o.parse_args(sys.argv[1:])

def flatten_reds(reds):
    freds = []
    for r in reds:
        freds += r
    return freds
#get frequencies
uv = a.miriad.UV(args[0])
fqs = a.cal.get_freqs(uv['sdf'], uv['sfreq'], uv['nchan'])
del(uv)

#hera info assuming a hex of 19 and 128 antennas
aa = a.cal.get_aa(opts.cal, fqs)
ex_ants = []
ubls = []
for ant in opts.ex_ants.split(','):
    try: ex_ants.append(int(ant))
    except: pass
for bl in opts.ubls.split(','):
    try:
        i,j = bl.split('_')
        ubls.append((int(i),int(j)))
    except: pass
print 'Excluding Antennas:',ex_ants
if len(ubls) != None: print 'Using Unique Baselines:',ubls
info = omni.aa_to_info(aa, fcal=True, ubls=ubls, ex_ants=ex_ants)
bls = flatten_reds(info.get_reds())
print 'Number of redundant baselines:',len(bls)
#Read in data here.

for filename in args:
    uv_in = UVData()
    uv_in.read_miriad(filename)
    if uv_in.phase_type != 'drift':
        print("Setting phase type to drift")
        uv_in.unphase_to_drift()
    datapack, wgtpack = omni.UVData_to_dict([uv_in])
    wgtpack = {k : { p : np.logical_not(wgtpack[k][p]) for p in wgtpack[k]} for k in wgtpack}  # logical_not of wgtpack

    #gets phase solutions per frequency.
    fc = firstcal.FirstCal(datapack,wgtpack,fqs,info)
    sols = fc.run(finetune=opts.finetune,verbose=opts.verbose,average=opts.average,window='none')

    #Converting solutions to a type that heracal can use to write uvfits files.
    meta = {}
    meta['lsts'] = uv_in.lst_array.reshape(uv_in.Ntimes, uv_in.Nbls)[:,0]
    meta['times'] = uv_in.time_array.reshape(uv_in.Ntimes, uv_in.Nbls)[:,0]
    meta['freqs'] = uv_in.freq_array[0]  # in Hz
    meta['inttime'] = uv_in.integration_time  # in sec
    meta['chwidth'] = uv_in.channel_width  # in Hz

    delays = {}
    antflags = {}
    for pol in opts.pol.split(','):
        pol = pol[0]
        delays[pol] = {}
        antflags[pol] = {}
        for ant in sols.keys():
            delays[ant.pol()][ant.val] = sols[ant].T
            antflags[ant.pol()][ant.val] = np.zeros(shape=(len(meta['lsts']), len(meta['freqs'])))
            #generate chisq per antenna/pol.
            meta['chisq{0}'.format(str(ant))] = np.ones(shape=(uv_in.Ntimes, 1))
    #overall chisq. This is a required parameter for uvcal.
    meta['chisq'] = np.ones_like(sols[ant].T)

    #Save solutions
    filename=args[0]+'.firstcal.fits'
    if not opts.outpath is None:
        outname='%s/%s'%(opts.outpath,filename.split('/')[-1])
    else:
        outname='%s'%filename

    optional = {'observer': opts.observer,
                'git_origin_cal': opts.git_origin_cal,
                'git_hash_cal':  opts.git_hash_cal}
                
    hc = omni.HERACal(meta, delays, flags=antflags, ex_ants=ex_ants, DELAY=True, appendhist=' '.join(sys.argv), optional=optional)
    print('     Saving {0}'.format(outname))
    hc.write_calfits(outname)

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
    data = uv_in.data_array.reshape(uv_in.Ntimes, uv_in.Nbls, uv_in.Nspws, uv_in.Nfreqs, uv_in.Npols)
    flags = uv_in.flag_array.reshape(uv_in.Ntimes, uv_in.Nbls, uv_in.Nspws, uv_in.Nfreqs, uv_in.Npols)
     
    datapack,wgtpack = {},{}
    for ip, pol in enumerate(uv_in.polarization_array):
        pol = a.miriad.pol2str[pol]
        if pol != opts.pol: 
            continue
        for nbl, (i,j) in enumerate(map(uv_in.baseline_to_antnums, uv_in.baseline_array[:uv_in.Nbls])):
            if not (i, j) in bls and not (j, i) in bls:
                continue
            datapack[(i,j)] = data[:, nbl, 0, :, ip]
            wgtpack[(i,j)] = np.logical_not( flags[:, nbl, 0, :, ip] )
    

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
        delays[pol[0]] = {}
        antflags[pol[0]] = {}
        for ant in sols.keys():
            delays[pol[0]][ant] = sols[ant].T / 1e9  # get into units of seconds
            antflags[pol[0]][ant] = np.zeros(shape=(len(meta['lsts']), len(meta['freqs'])))
            #generate chisq per antenna/pol.
            meta['chisq{0}{1}'.format(ant,pol[0])] = np.ones(shape=(uv_in.Ntimes, 1))
    #overall chisq. This is a required parameter for uvcal.
    meta['chisq'] = np.ones_like(sols[ant].T)

    #Save solutions
    filename=args[0]+'.firstcal.fits'
    if not opts.outpath is None:
        outname='%s/%s'%(opts.outpath,filename.split('/')[-1])
    else:
        outname='%s'%filename

    optional = {'observer': 'Zaki Ali (zakiali@berkeley.edu)',
                'git_origin_cal': 'None',
                'git_hash_cal': 'None'}
                
    hc = omni.HERACal(meta, delays, flags=antflags, ex_ants=ex_ants, DELAY=True, appendhist=' '.join(sys.argv), optional=optional)
    print('     Saving {0}'.format(outname))
    hc.write_calfits(outname)

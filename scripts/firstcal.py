#! /usr/bin/env python
import numpy as np, optparse, sys
import aipy as a
from heracal import omni
from heracal.miriad import read_files

o = optparse.OptionParser()
a.scripting.add_standard_options(o,cal=True,pol=True)
o.add_option('--ubls', default='', help='Unique baselines to use, separated by commas (ex: 1_4,64_49).')
o.add_option('--ex_ants', default='', help='Antennas to exclude, separated by commas (ex: 1,4,64,49).')
o.add_option('--outpath', default=None,help='Output path of solution npz files. Default will be the same directory as the data files.')
o.add_option('--plot', action='store_true', default=False, help='Turn on plotting in firstcal class.')
o.add_option('--verbose', action='store_true', default=False, help='Turn on verbose.')
o.add_option('--finetune', action='store_false', default=True, help='Fine tune the delay fit.')
o.add_option('--clean', action='store_true', default=False, help='Run clean on delay transform.')
o.add_option('--offset', action='store_true', default=False, help='Solve for offset along with delay.')
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
for a in opts.ex_ants.split(','):
    try: ex_ants.append(int(a))
    except: pass
for bl in opts.ubls.split(','):
    try:
        i,j = bl.split('_')
        ubls.append((int(i),int(j)))
    except: pass
print 'Excluding Antennas:',ex_ants
if len(ubls) != None: print 'Using Unique Baselines:',ubls
info = omni.aa_to_info(aa, fcal=True, ubls=ubls, ex_ants=ex_ants)
reds = flatten_reds(info.get_reds())

print 'Number of redundant baselines:',len(reds)
#Read in data here.
ant_string =','.join(map(str,info.subsetant))
bl_string = ','.join(['_'.join(map(str,k)) for k in reds])

for filename in args:
    times, data, flags = read_files([filename], bl_string, opts.pol, verbose=True)
    datapack,wgtpack = {},{}
    for (i,j) in data.keys():
        datapack[(i,j)] = data[(i,j)][opts.pol]
        wgtpack[(i,j)] = np.logical_not(flags[(i,j)][opts.pol])
    dlys = np.fft.fftshift(np.fft.fftfreq(fqs.size, np.diff(fqs)[0]))

    #gets phase solutions per frequency.
    fc = omni.FirstCal(datapack,wgtpack,fqs,info)
    sols = fc.run(finetune=opts.finetune,verbose=opts.verbose,plot=opts.plot,noclean= not opts.clean,offset=opts.offset,average=opts.average,window='none')

    #Converting solutions to a type that heracal can use to write uvfits files.
    meta = {}
    meta['lsts'] = times['lsts']
    meta['jds'] = times['times']
    meta['freqs'] = fqs
    meta['inttime'] = times['inttime']
    meta['chwidth'] = times['chwidth']

    delays = {}
    antflags = {}
    for pol in opts.pol.split(','):
        delays[pol[0]] = {}
        antflags[pol[0]] = {}
        for ant in sols.keys():
            delays[pol[0]][ant] = sols[ant].T
            antflags[pol[0]][ant] = np.zeros(shape=(len(meta['lsts']), len(meta['freqs'])))
            #generate chisq per antenna/pol.
            meta['chisq{0}{1}'.format(ant,pol[0])] = np.ones(shape=(len(times['times']), 1))
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

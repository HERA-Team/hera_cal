#! /usr/bin/env python
import numpy as np, optparse, sys
import aipy as a
from miriad import read_files
from omni import aa_to_info, FirstCal, save_gains_fc
from heracal import HERACal

o = optparse.OptionParser()
a.scripting.add_standard_options(o,cal=True,pol=True)
o.add_option('--ubls', default='', help='Unique baselines to use, separated by commas (ex: 1_4,64_49).')
o.add_option('--ex_ants', default='', help='Antennas to exclude, separated by commas (ex: 1,4,64,49).')
o.add_option('--outpath', default=None,help='Output path of solution npz files. Default will be the same directory as the data files.')
o.add_option('--plot', action='store_true', default=False, help='Turn on plotting in firstcal class.')
o.add_option('--verbose', action='store_true', default=False, help='Turn on verbose.')
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
info = aa_to_info(aa, fcal=True, ubls=ubls, ex_ants=ex_ants)
reds = flatten_reds(info.get_reds())

print 'Number of redundant baselines:',len(reds)
#Read in data here.
ant_string =','.join(map(str,info.subsetant))
bl_string = ','.join(['_'.join(map(str,k)) for k in reds])
times, data, flags = read_files(args, bl_string, opts.pol, verbose=True)
datapack,wgtpack = {},{}
for (i,j) in data.keys():
    datapack[(i,j)] = data[(i,j)][opts.pol]
    wgtpack[(i,j)] = np.logical_not(flags[(i,j)][opts.pol])
dlys = np.fft.fftshift(np.fft.fftfreq(fqs.size, np.diff(fqs)[0]))

#gets phase solutions per frequency.
fc = FirstCal(datapack,wgtpack,fqs,info)
sols = fc.run(finetune=True,verbose=opts.verbose,plot=opts.plot,noclean=True,offset=False,average=False,window='none')

#Converting solutions to a type that heracal can use to write uvfits files.
meta = {}
meta['lsts'] = times['lsts']
meta['jds'] = times['times']
meta['freqs'] = fqs

delays = {}
for pol in opts.pol.split(','):
    delays[pol[0]] = {}
    for ant in sols.keys():
        delays[pol[0]][ant] = sols[ant].T
        #generate chisq per antenna/pol.
        meta['chisq{0}{1}'.format(ant,pol[0])] = np.ones(shape=(len(times['times']), 1))
#overall chisq
meta['chisq'] = np.ones_like(sols[ant].T)

#Save solutions
if len(args)==1: filename=args[0]+'.fits'
else: filename='fcgains.%s.fits'%opts.pol #if averaging a bunch together of files together
if not opts.outpath is None:
    outname='%s/%s'%(opts.outpath,filename.split('/')[-1])
else:
    outname='%s'%filename
hc = HERACal(meta, delays, ex_ants=ex_ants, DELAY=True, appendhist='Testcal')
hc.write_calfits(outname)

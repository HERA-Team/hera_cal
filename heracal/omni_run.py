#! /usr/bin/env python

import omnical, aipy, numpy
from omni import from_npz, aa_to_info, redcal, compute_xtalk, to_npz
from miriad import read_files
import pickle, optparse, os, sys
from heracal import HERACal
import pyuvdata 

o = optparse.OptionParser()
o.set_usage('omni_run.py [options] *uvcRRE')
o.set_description(__doc__)
aipy.scripting.add_standard_options(o,cal=True,pol=True)
o.add_option('--calpar',dest='calpar',type='string',
            help='Path and name of POL.p ("xx.p") calpar file from firstcal v1.')
o.add_option('--redinfo',dest='redinfo',type='string',default='',
            help='Path and name of .bin redundant info file from firstcal v1. If not given, it will be created.')
o.add_option('--omnipath',dest='omnipath',default='',type='string',
            help='Path to save .npz files. Include final / in path.')
o.add_option('--ba',dest='ba',default=None,
            help='Antennas to exclude, separated by commas.')
o.add_option('--fc2',dest='fc2',type='string',
            help='Path and name of POL.npz file outputted by firstcal v2.')
opts,args = o.parse_args(sys.argv[1:])

def write_uvdata_vis(filename, aa, m, v, xtalk=False, returnuv=True):

    pols = v.keys()
    bls = v[pols[0]].keys()
                
    uv = pyuvdata.UVData()
    if xtalk:
        uv.Ntimes = 1
    else:
        uv.Ntimes = len(m['jds']) 
    uv.Npols = len(pols)
    uv.Nbls = len(v[pols[0]].keys()) 
    uv.Nblts = uv.Nbls * uv.Ntimes
    uv.Nfreqs = len(m['freqs'])
    data = {}
    for p in v.keys():
        if not p in data.keys(): data[p] = []
        for bl in v[p].keys():
            data[p].append(v[p][bl])
        data[p] = numpy.array(data[p]).reshape(uv.Nblts,uv.Nfreqs)

    uv.data_array = numpy.expand_dims(numpy.array([data[p] for p in pols]).T.swapaxes(0,1), axis=1) # turn into correct shpe
    uv.vis_units='uncalib'
    uv.nsample_array = numpy.ones_like(uv.data_array,dtype=numpy.float)
    uv.flag_array = numpy.zeros_like(uv.data_array,dtype=numpy.bool)
    uv.Nspws = 1 # this is always 1 for paper and hera(currently)
    uv.spw_array = numpy.array([uv.Nspws])
#    uvw_array = aa.
    blts = numpy.array([bl for bl in bls for i in range(uv.Ntimes)])
    if xtalk:
        uv.time_array = numpy.array(list(m['jds'][:1]) * uv.Nbls)
        uv.lst_array = numpy.array(list(m['lsts'][:1]) * uv.Nbls)
    else:
        uv.time_array = numpy.array(list(m['jds']) * uv.Nbls)
        uv.lst_array = numpy.array(list(m['lsts']) * uv.Nbls)

    #generate uvw
    uvw = []
    for t in range(uv.Ntimes):
        for bl in bls:
            uvw.append(aa.gen_uvw(*bl, src='z').reshape(3,-1))
    uv.uvw_array = numpy.array(uvw).reshape(-1,3)

    uv.ant_1_array = blts[:,0]
    uv.ant_2_array = blts[:,1]
    uv.baseline_array = uv.antnums_to_baseline(uv.ant_1_array, 
                                               uv.ant_2_array)

    uv.freq_array = m['freqs'].reshape(1,-1)
    poldict = {'xx':-5,'yy':-6,'xy':-7,'yx':-8}
    uv.polarization_array = numpy.array([poldict[p] for p in pols])
    if xtalk:
        uv.integration_time = m2['inttime'] * len(m2['jds']) # xtalk integration time is averaged over the whole file
    else:
        uv.integration_time = m2['inttime'] 
    uv.channel_width = numpy.diff(uv.freq_array[0])[0]

    #observation parameters 
    uv.object_name = 'zenith'
    uv.telescope_name = 'HERA'
    uv.instrument = 'HERA'
    tobj = pyuvdata.uvtel.get_telescope(uv.telescope_name)
    uv.telescope_location = tobj.telescope_location
    uv.history = m2['history']# + os. get command line
    
    #phasing information
    uv.phase_type = 'drift'
    uv.zenith_ra = uv.lst_array
    uv.zenith_dec = numpy.array([aa.lat]*uv.Nblts)
    
    #antenna information 
    uv.Nants_telescope = 128
    uv.antenna_numbers = numpy.arange(uv.Nants_telescope, dtype=int)
    uv.Nants_data = len(numpy.unique(numpy.concatenate([uv.ant_1_array, uv.ant_2_array]).flatten()))
    uv.antenna_names = numpy.array(['ant{0}'.format(ant) for ant in uv.antenna_numbers])
    antpos = []
    for k in aa:
        antpos.append(k.pos)
          
    uv.antenna_positions = numpy.array(antpos)

    if returnuv:
        return uv
    else:
        uv.write_uvfits(filename, force_phase=True, spoof_nonessential=True)
    


#Dictionary of calpar gains and files
pols = opts.pol.split(',')
files = {}
g0 = {} #firstcal gains
for pp,p in enumerate(pols):
    #dictionary of calpars per pol
    g0[p[0]] = {} #indexing by one pol letter instead of two
    if opts.calpar != None: #if calpar is given
        if p in opts.calpar: #your supplied calpar matches a pol
            print 'Reading', opts.calpar
            cp = pickle.load(open(opts.calpar,'rb'))
            for i in xrange(cp[p].shape[1]): #loop over antennas
                g0[p[0]][i] = numpy.conj(cp[p][:,i] / numpy.abs(cp[p][:,i]))
        else: #looks for a calpar you haven't stated in the call
            new_cp = opts.calpar.split('.p')[0][:-2]+p+'.p' #XXX assumes calpar naming is *pol.p
            if os.path.exists(new_cp): #if it exists, use it
                print 'Reading', new_cp
                cp = pickle.load(open(new_cp,'rb'))
                for i in xrange(cp[p].shape[1]): #loop over antennas
                    g0[p[0]][i] = numpy.conj(cp[p][:,i] / numpy.abs(cp[p][:,i]))
            elif len(list(set(p))) > 1: #if the crosspol first_cal is missing, don't worry
                #print '%s not found, but that is OK'%new_cp
                continue
            else: #if the linpol first_cal is missing, do worry
                raise IOError('Missing first_cal file %s'%new_cp)
    if opts.fc2 != None: #if fc2 file is given
        if p in opts.fc2:
            print 'Reading', opts.fc2
            _,_g0,_,_ = from_npz(opts.fc2)
            for i in _g0[p[0]].keys():
                print i
                g0[p[0]][i] = _g0[p[0]][i][:,:] / numpy.abs(_g0[p[0]][i][:,:])
        else:
            raise IOError("Please provide a first cal file") 
for filename in args:
    files[filename] = {}
    for p in pols:
        fn = filename.split('.')
        fn[3] = p
        files[filename][p] = '.'.join(fn)

#Create info
if opts.redinfo != '': #reading redinfo file
    print 'Reading',opts.redinfo
    info = omnical.info.RedundantInfoLegacy()
    print '   Getting reds from redundantinfo'
    info.fromfile(opts.redinfo)
else: #generate reds from calfile
    aa = aipy.cal.get_aa(opts.cal,numpy.array([.15]))
    print 'Getting reds from calfile'
    if opts.ba: #XXX assumes exclusion of the same antennas for every pol
        ex_ants = []
        for a in opts.ba.split(','):
            ex_ants.append(int(a))
        print '   Excluding antennas:',sorted(ex_ants)
    else: ex_ants = []
    info = aa_to_info(aa, pols=list(set(''.join(pols))), ex_ants=ex_ants, crosspols=pols)
reds = info.get_reds()
#import IPython;IPython.embed()

### Omnical-ing! Loop Through Compressed Files ###
for f,filename in enumerate(args):
    file_group = files[filename] #dictionary with pol indexed files
    print 'Reading:'
    for key in file_group.keys(): print '   '+file_group[key]

    #pol = filename.split('.')[-2] #XXX assumes 1 pol per file
    
    if len(pols)>1: #zen.jd.npz
        npzb = 3
    else: #zen.jd.pol.npz
        npzb = 4 
    fitsname = opts.omnipath+'.'.join(filename.split('/')[-1].split('.')[0:npzb])+'.fitsA'
    if os.path.exists(fitsname):
        print '   %s exists. Skipping...' % fitsname
        continue

    timeinfo,d,f = read_files([file_group[key] for key in file_group.keys()], antstr='cross', polstr=opts.pol, decimate=20)
    t_jd = timeinfo['times']
    t_lst = timeinfo['lsts']
    freqs = numpy.arange(.1,.2,.1/len(d[d.keys()[0]][pols[0]][0]))
    SH = d.values()[0].values()[0].shape #shape of file data (ex: (19,203))
    data,wgts,xtalk = {}, {}, {}
    m2,g2,v2 = {}, {}, {}
    for p in g0.keys():
        for i in g0[p]: 
            if g0[p][i].shape != (len(t_jd),len(freqs)):
                g0[p][i] = numpy.resize(g0[p][i],SH) #resize gains like data
            else: continue
    data = d #indexed by bl and then pol (backwards from everything else)
    for p in pols:
        wgts[p] = {} #weights dictionary by pol
        for bl in f: 
            i,j = bl
            wgts[p][(j,i)] = wgts[p][(i,j)] = numpy.logical_not(f[bl][p]).astype(numpy.int)
    print '   Logcal-ing' 
    m1,g1,v1 = redcal(data,info,gains=g0, removedegen=False) #SAK CHANGE REMOVEDEGEN
    print '   Lincal-ing'
    m2,g2,v2 = redcal(data, info, gains=g1, vis=v1, uselogcal=False, removedegen=False)
    xtalk = compute_xtalk(m2['res'], wgts) #xtalk is time-average of residual
    m2['history'] = 'OMNI_RUN: '+''.join(sys.argv) + '\n'
    m2['jds'] = t_jd
    m2['lsts'] = t_lst
    m2['freqs'] = freqs
    m2['inttime'] = timeinfo['inttime']
    
    print '   Saving %s'%fitsname 
    hc = HERACal(m2, g2)
    hc.write_calfits(fitsname)
    print('   Saving {0}'.format('.'.join(fitsname.split('.')[:-1]) + '.vis.fits'))
    write_uvdata_vis('.'.join(fitsname.split('.')[:-1]) + '.vis.fits', aa, m2, v2)
    print('   Saving {0}'.format('.'.join(fitsname.split('.')[:-1]) + '.xtalk.fits'))
    write_uvdata_vis('.'.join(fitsname.split('.')[:-1]) + '.xtalk.fits', aa, m2, xtalk, xtalk=True)

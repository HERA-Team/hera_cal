import aipy as a, numpy as np

def read_files(filenames, antstr, polstr, decimate=1, decphs=0, verbose=False, recast_as_array=True):
    '''Read in miriad uv files.
       Parameters
       ---------
       filenames : list of files
       antstr    : string
            list of antennas and or baselines. e.g. 9_10,5_3,...etc.
       polstr    : string
            polarization to extract.

       Returns 
       -------
       info      : dict. 
            the lsts and jd's of the data
       dat       : dict
            the data in dictionary format. dat[bl(in tuple format)][pol(in string)]  
       flg       : dict
            corresponding flags to data. Same format. 
    '''
    info = {'lsts':[], 'times':[]}
    ts = {}
    dat, flg = {}, {}
    if type(filenames) == 'str': filenames = [filenames]
    for filename in filenames:
        if verbose: print '   Reading', filename
        uv = a.miriad.UV(filename)
        a.scripting.uv_selector(uv, antstr, polstr)
        if decimate > 1: uv.select('decimate', decimate, decphs)
        for (crd,t,(i,j)),d,f in uv.all(raw=True):
            if not ts.has_key(t):
                info['times'].append(t)
                info['lsts'].append(uv['lst'])
                ts[t] = None
            bl = (i,j)
            if not dat.has_key(bl): dat[bl],flg[bl] = {},{}
            pol = a.miriad.pol2str[uv['pol']]
            if not dat[bl].has_key(pol):
                dat[bl][pol],flg[bl][pol] = [],[]
            chans = a.scripting.parse_chans(chan_str, len(d))
            dat[bl][pol].append(d[chans])
            flg[bl][pol].append(f[chans])
    info['freqs'] = a.cal.get_freqs(uv['sdf'], uv['sfreq'], uv['nchan'])
    if recast_as_array:
        # This option helps reduce memory footprint, but it shouldn't
        # be necessary: the replace below should free RAM as quickly
        # as it is allocated.  Unfortunately, it doesn't seem to...
        for bl in dat.keys():
          for pol in dat[bl].keys():
            dat[bl][pol] = np.array(dat[bl][pol])
            flg[bl][pol] = np.array(flg[bl][pol])
        info['lsts'] = np.array(info['lsts'])
        info['times'] = np.array(info['times'])
    return info, dat, flg

def read_files_dict(filenames, antstr, polstr, chanbunch='all', decimate=1, decphs=0, verbose=False, recast_as_array=True):
    '''Read in miriad uv files and return a dictionary with the following format. 
       dict[chanbunch][pol][baseline]

       Parameters
       ---------
       filenames : list of files
       antstr    : string
            list of antennas and or baselines. e.g. 9_10,5_3,...etc.
       polstr    : string
            polarization to extract.
       chanbunch : int
            Groups of channels to be returned for each dictionary.

       Returns 
       -------
       info      : dict. 
            the lsts and jd's of the data
       dat       : dict
            the data in dictionary format. dat[bl(in tuple format)][pol(in string)]  
       flg       : dict
            corresponding flags to data. Same format. 
    '''
    info = {'lsts':[], 'times':[]}
    ts = {}
    dat, flg = {}, {}
    if type(filenames) == 'str': filenames = [filenames]
    for filename in filenames:
        if verbose: print '   Reading', filename
        uv = a.miriad.UV(filename)
        a.scripting.uv_selector(uv, antstr, polstr)
        if decimate > 1: uv.select('decimate', decimate, decphs)
        for (crd,t,(i,j)),d,f in uv.all(raw=True):
            if not ts.has_key(t):
                info['times'].append(t)
                info['lsts'].append(uv['lst'])
                ts[t] = None
            bl = (i,j)
            for chans in xrange(0,len(d), chanbunch):
                if not dat.has_key(chans): dat[chans],flg[chans] = {},{}
                if not dat[chans].has_key(bl): dat[chans][bl],flg[chans][bl] = {},{}
                pol = a.miriad.pol2str[uv['pol']]
                if not dat[chans][bl].has_key(pol):
                    dat[chans][bl][pol],flg[chans][bl][pol] = [],[]
                dat[chans][bl][pol].append(d[chans:chans+chanbunch])
                flg[chans][bl][pol].append(f[chans:chans+chanbunch])
    info['freqs'] = a.cal.get_freqs(uv['sdf'], uv['sfreq'], uv['nchan'])
    if recast_as_array:
        # This option helps reduce memory footprint, but it shouldn't
        # be necessary: the replace below should free RAM as quickly
        # as it is allocated.  Unfortunately, it doesn't seem to...
        for chan in dat.keys():
            for bl in dat[chan].keys():
              for pol in dat[chan][bl].keys():
                dat[chan][bl][pol] = np.array(dat[chan][bl][pol])
                flg[chan][bl][pol] = np.array(flg[chan][bl][pol])
        info['lsts'] = np.array(info['lsts'])
        info['times'] = np.array(info['times'])
        info['chanbunch'] = chanbunch
    return info, dat, flg


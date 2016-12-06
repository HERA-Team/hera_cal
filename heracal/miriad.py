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
            dat[bl][pol].append(d)
            flg[bl][pol].append(f)
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


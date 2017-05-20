import numpy as np

class DataContainer:
    """Object that abstracts away the pol/ant pair ordering of data dict's."""
    def __init__(self, data):
        """
        Args:
            data (dict): dictionary of visibilities with keywords of pol/ant pair 
                in any order.
        """
        self._data = {}
        if type(data.keys()[0]) is str: # Nested POL:{bls}
            for pol in data.keys():
                for bl in data[pol]:
                    self._data[self.mk_key(bl,pol)] = data[pol][bl]
        elif len(data.keys()[0]) == 2: # Nested bl:{POL}
            for bl in data.keys():
                for pol in data[bl]:
                    self._data[self.mk_key(bl,pol)] = data[bl][pol]
        else:
            assert(len(data.keys()[0]) == 3)
            self._data = data
        self._bls = set([k[:2] for k in self._data.keys()])
        self._pols = set([k[-1] for k in self._data.keys()])
    def mk_key(self, bl, pol): return bl + (pol,)
    def _switch_bl(self, key):
        if len(key) == 3: return (key[1], key[0], key[2])
        else: return (key[1], key[0])
    def bls(self, pol=None):
        if pol is None: return self._bls.copy()
        else: return set([bl for bl in self._bls if self.has_key(bl,pol)])
    def pols(self, bl=None):
        if bl is None: return self._pols.copy()
        else: return set([pol for pol in self._pols if self.has_key(bl,pol)])
    def keys(self): return self._data.keys()
    def __getitem__(self, key):
        if type(key) is str: # asking for a pol
            return dict(zip(self._bls, [self[self.mk_key(bl,key)] for bl in self._bls]))
        elif len(key) == 2: # asking for a bl
            return dict(zip(self._pols, [self[self.mk_key(key,pol)] for pol in self._pols]))
        else:
            try: return self._data[key]
            except(KeyError): return np.conj(self._data[self._switch_bl(key)])
    def has_key(self, *args):
        if len(args) == 1: return self._data.has_key(args[0]) or self._data.has_key(self._switch_bl(args[0]))
        else: return self.has_key(self.mk_key(args[0],args[1]))
    def has_bl(self, bl): return bl in self._bls
    def has_pol(self, pol): return pol in self._pols
    def get(self, bl, pol): return self[self.mk_key(bl,pol)]
            
        

def _count_flags(m, labels):
    '''
    Count flags in an "inside-out" way that doesn't penalize well-performing
    indices (e.g. bls or ants) pairings with poor-performing ones.
        
    Args:
        m : boolean array of masks
        labels: labels for the masks (bls, ants, etc...)

    Returns:
        dict: dictionary of flag counts
    '''
    order = np.argsort(np.sum(m, axis=1))[::-1]
    m = m[order][:,order]
    labels = [labels[i] for i in order]
    cnts = {}
    for i,label in enumerate(labels): cnts[label] = np.sum(m[i,i:])
    return cnts

def check_ants(reds, data, flag_thresh=.2, skip_ants=[]):
    '''Correlate bls within redundant groups to find counts of poorly correlated (broken) data per antennas.

    Args:
        reds: list of list of redundant baselines in antenna pair tuple format.
        data: data dictionary with pol and blpair keys (in any order)
        flag_thresh: float, threshold to flag bad antennas.
        skip_ants: list of antennas to not include in calculation. i.e. predetermined bad antennas.

    Returns:
        dict: dictionary of counts with antennas as indices and number of bad data involved with antenna as values.
             
    '''
    def exclude_ants(bls, skip_ants): 
        return [bl for bl in bls if bl[0] not in skip_ants and bl[1] not in skip_ants]
    dc = DataContainer(data)
    reds = [exclude_ants(bls, skip_ants) for bls in reds]
    all_bls = reduce(lambda x,y: x+y, reds)
    cnts = {}
    for pol in dc.pols():
        auto_pwr, ant2col = {}, {}
        for bl in dc.bls(pol).intersection(all_bls):
            d = dc.get(bl,pol)
            auto_pwr[bl] = np.median(np.sum(np.abs(d)**2, axis=0))
            ant2col[bl[0]] = ant2col.get(bl[0],len(ant2col))
            ant2col[bl[1]] = ant2col.get(bl[1],len(ant2col))
        col2ant = {}
        for i in ant2col: col2ant[ant2col[i]] = i
        nants = len(ant2col)
        Fant = np.zeros((nants,nants), dtype=np.int)
        for bls in reds:
            bls = list(dc.bls(pol).intersection(bls))
            C = np.zeros((len(bls),len(bls)), dtype=np.float)
            for i,bl1 in enumerate(bls):
                d1 = dc.get(bl1,pol)
                for j,bl2 in enumerate(bls[i:]):
                    j += i
                    d2 = dc.get(bl2,pol)
                    pwr12 = np.median(np.abs(np.sum(d1*d2.conj(), axis=0)))
                    C[i,j] = C[j,i] = pwr12 / np.sqrt(auto_pwr[bl1] * auto_pwr[bl2])
            cnts = _count_flags(np.where(C < flag_thresh, 1, 0), bls)
            for i,j in cnts: Fant[ant2col[i],ant2col[j]] = Fant[ant2col[j],ant2col[i]] = cnts[(i,j)]
        cnt = _count_flags(np.where(Fant >= 1, 1, 0), [col2ant[i] for i in xrange(nants)])
        for i in cnt: cnts[(i,pol)] = cnt[i]
    return cnts

def check_noise_variance(data, wgts, bandwidth, inttime):
    """XXX"""
    dc = DataContainer(data)
    wc = DataContainer(wgts)
    Cij = {}
    for pol in dc.pols():
        for bl in dc.bls(pol):
            i,j = bl
            d = dc.get(bl,pol)
            w = wc.get(bl,pol)
            ai, aj = dc.get((i,i),pol).real, dc.get((j,j),pol).real
            ww = w[1:,1:] * w[1:,:-1] * w[:-1,1:] * w[:-1,:-1]
            dd = ((d[:-1,:-1] - d[:-1,1:]) - (d[1:,:-1] - d[1:,1:])) * ww / np.sqrt(4)
            dai = ((ai[:-1,:-1] + ai[:-1,1:]) + (ai[1:,:-1] + ai[1:,1:])) * ww / 4
            daj = ((aj[:-1,:-1] + aj[:-1,1:]) + (aj[1:,:-1] + aj[1:,1:])) * ww / 4
            Cij[bl+(pol,)] = np.sum(np.abs(dd)**2,axis=0)/np.sum(dai*daj,axis=0) * (bandwidth * inttime)
    return Cij

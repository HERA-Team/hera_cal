import numpy as np
import omnical
from copy import deepcopy
import numpy.linalg as la
from pyuvdata import UVCal, UVData, uvtel
from heracal.firstcal import FirstCalRedundantInfo
import warnings, os
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import scipy.sparse as sps

POL_TYPES = 'xylrabne'
# XXX this can't support restarts or changing # pols between runs
POLNUM = {}  # factor to multiply ant index for internal ordering
NUMPOL = {}


def add_pol(p):
    '''
        Add's pols to the global POLNUM and NUMPOL dictionaries.
        Used for creating Antpol objects.
    '''
    global NUMPOL
    assert(p in POL_TYPES)
    POLNUM[p] = len(POLNUM)
    NUMPOL = dict(zip(POLNUM.values(), POLNUM.keys()))


class Antpol:
    '''Defines an Antpol object that encodes an antenna number and polarization value.'''
    def __init__(self, *args):
        '''
            Creates an Antpol object.
                Args:
                    ant: antenna number, integer.
                    pol: polarization string. e.g. 'x', or 'y'
                    nant: total number of antennas.
        '''
        try:
            ant, pol, nant = args
            if pol not in POLNUM: add_pol(pol)
            self.val, self.nant = POLNUM[pol] * nant + ant, nant
        except(ValueError): self.val, self.nant = args
    def antpol(self): return self.val % self.nant, NUMPOL[self.val / self.nant]
    def ant(self): return self.antpol()[0]
    def pol(self): return self.antpol()[1]
    def __int__(self): return self.val
    def __hash__(self): return self.ant()
    def __str__(self): return ''.join(map(str, self.antpol()))
    def __eq__(self, v): return self.ant() == v
    def __repr__(self): return str(self)


# XXX filter_reds w/ pol support should probably be in omnical
def filter_reds(reds, bls=None, ex_bls=None, ants=None, ex_ants=None, ubls=None, ex_ubls=None, crosspols=None, ex_crosspols=None):
    '''
        Filter redundancies to include/exclude the specified bls, antennas, unique bl groups and polarizations.
        Assumes reds indices are Antpol objects.
        Args:
            reds: list of lists of redundant baselines as antenna pair tuples. e.g. [[(1,2),(2,3)], [(1,3)]]
        Kwargs:
            bls: list of baselines as antenna pair tuples to include in reds.
            ex_bls: list of baselines as antenna pair tuples to exclude in reds.
            ants: list of antenna numbers (as int's) to include in reds.
            ex_ants: list of antenna numbers (as int's) to exclude in reds.
            ubls: list of baselines representing their redundant group to include in reds.
            ex_ubls: list of baselines representing their redundant group to exclude in reds.
            crosspols: cross polarizations to include in reds. e.g. 'xy' or 'yx'.
            ex_crosspols: cross polarizations to exclude in reds. e.g. 'xy' or 'yx'.
        Return:
            reds: list of lists of redundant baselines as antenna pair tuples.
    '''
    def pol(bl): return bl[0].pol() + bl[1].pol()
    if crosspols: reds = [r for r in reds if pol(r[0]) in crosspols]
    if ex_crosspols: reds = [r for r in reds if not pol(r[0]) in ex_crosspols]
    return omnical.arrayinfo.filter_reds(reds, bls=bls, ex_bls=ex_bls, ants=ants, ex_ants=ex_ants, ubls=ubls, ex_ubls=ex_ubls)


class RedundantInfo(omnical.calib.RedundantInfo):
    '''RedundantInfo object to interface with omnical. Includes support for Antpol objects.'''
    def __init__(self, nant, filename=None):
        '''
            Args:
                nant: number of antennas.
                filename: filename (str) for legacy info objects.'''
        omnical.info.RedundantInfo.__init__(self, filename=filename)
        self.nant = nant

    def bl_order(self):
        '''
            Return:
                (i,j) baseline tuples in the order that they should appear in data.
                Antenna indicies are in real-world order
                (as opposed to the internal ordering used in subsetant).
        '''
        return [(Antpol(self.subsetant[i], self.nant), Antpol(self.subsetant[j], self.nant)) for (i, j) in self.bl2d]

    def order_data(self, dd):
        '''
            Create a data array ordered for use in _omnical.redcal.
            Args:
                dd: dictionary whose keys are (i,j) antenna tuples; antennas i,j should be ordered to reflect
                    the conjugation convention of the provided data.  'dd' values are 2D arrays of (time,freq) data.
            Return:
                array: array whose ordering reflects the internal ordering of omnical. Used to pass into pack_calpar
        '''
        d = []
        for i, j in self.bl_order():
            bl = (i.ant(), j.ant())
            pol = i.pol() + j.pol()
            try: d.append(dd[bl][pol])
            except(KeyError): d.append(dd[bl[::-1]][pol[::-1]].conj())
        return np.array(d).transpose((1, 2, 0))

    def pack_calpar(self, calpar, gains=None, vis=None, **kwargs):
        '''
            Note that this function includes polarization support by wrapping
            into calpar format.
            Args:
                calpar: array whose size is given by self.calpar_size. Usually initialized to zeros.
            Kwargs:
                gains: dictionary of starting gains for omnical run. dict[pol][antenna]
                vis: dictionary of starting visibilities (for a redundant group) for omnical run. dict[pols][bl],
            Extra Kwargs:
                nondegenerategains: gains that don't have a degeneracy component to them (e.g. firstcal gains).
                                    The gains get divided out before handing off calpar to omnical.

            Returns:
                calpar: The populated calpar array.
        '''
        nondegenerategains = kwargs.pop('nondegenerategains', None)
        if gains:
            _gains = {}
            for pol in gains:
                for i in gains[pol]:
                    ai = Antpol(i, pol, self.nant)
                    if nondegenerategains is not None:
                        _gains[int(ai)] = gains[pol][i].conj()/nondegenerategains[pol][i].conj()  # This conj is necessary to conform to omnical conj conv.
                    else:
                        _gains[int(ai)] = gains[pol][i].conj()  # This conj is necessary to conform to omnical conj conv.
        else:
            _gains = gains

        if vis:
            _vis = {}
            for pol in vis:
                for i, j in vis[pol]:
                    ai, aj = Antpol(i, pol[0], self.nant), Antpol(j, pol[1], self.nant)
                    _vis[(int(ai), int(aj))] = vis[pol][(i, j)]
        else:
            _vis = vis

        calpar = omnical.calib.RedundantInfo.pack_calpar(self, calpar, gains=_gains, vis=_vis)

        return calpar

    def unpack_calpar(self, calpar, **kwargs):
        '''
            Unpack the solved for calibration parameters and repack to antpol format
            Args:
                calpar: calpar array output from omnical.
            Kwargs:
                nondegenerategains: the nondegenerategains that were divided out in pack_calpar.
                                    These are multiplied back into calpar here. gain dictionary format.

            Return:
                meta: dictionary of meta information from omnical. e.g. chisq, iters, etc
                gains: dictionary of gains solved for by omnical. gains[pol][ant]
                vis: dictionary of model visibilities solved for by omnical. vis[pols][blpair]
        '''
        nondegenerategains = kwargs.pop('nondegenerategains', None)
        meta, gains, vis = omnical.calib.RedundantInfo.unpack_calpar(self, calpar, **kwargs)

        def mk_ap(a): return Antpol(a, self.nant)
        if 'res' in meta:
            for i, j in meta['res'].keys():
                api, apj = mk_ap(i), mk_ap(j)
                pol = api.pol() + apj.pol()
                bl = (api.ant(), apj.ant())
                if not meta['res'].has_key(pol): meta['res'][pol] = {}
                meta['res'][pol][bl] = meta['res'].pop((i, j))
        # XXX make chisq a nested dict, with individual antpol keys?
        for k in [k for k in meta.keys() if k.startswith('chisq')]:
            try:
                ant = int(k.split('chisq')[1])
                meta['chisq' + str(mk_ap(ant))] = meta.pop(k)
            except(ValueError): pass
        for i in gains.keys():
            ap = mk_ap(i)
            if not gains.has_key(ap.pol()): gains[ap.pol()] = {}
            gains[ap.pol()][ap.ant()] = gains.pop(i).conj()
            if nondegenerategains:
                gains[ap.pol()][ap.ant()]*= nondegenerategains[ap.pol()][ap.ant()]
        for i, j in vis.keys():
            api, apj = mk_ap(i), mk_ap(j)
            pol = api.pol() + apj.pol()
            bl = (api.ant(), apj.ant())
            if not vis.has_key(pol): vis[pol] = {}
            vis[pol][bl] = vis.pop((i, j))
        return meta, gains, vis


def compute_reds(nant, pols, *args, **kwargs):
    '''
        Compute the redundancies given antenna_positions and wrap into Antpol format.
        Args:
            nant: number of antennas
            pols: polarization labels, e.g. pols=['x']
            *args: args to be passed to omnical.arrayinfo.compute_reds, spcifically
                   antpos: array of antenna positions in order of subsetant.

        Return:
            reds: list of list of baselines as antenna tuples
       '''
    _reds = omnical.arrayinfo.compute_reds(*args, **kwargs)
    reds = []
    for pi in pols:
        for pj in pols:
            reds += [[(Antpol(i, pi, nant), Antpol(j, pj, nant)) for i, j in gp] for gp in _reds]
    return reds

def reds_for_minimal_V(reds):
    '''
    Manipulate redundancy array to combine crosspols
    into a single redundancy array - imposing that
    Stokes V = 0.

    This works in the simple way that it does because
    of the way the reds arrays are constructed in
    aa_to_info when 4 polarizations are present: it
    reprsents them as 4 co-located arrays in (NS,EW) and
    displaced in z, with the cross-combinations (e.g. 
    polarization xy and yx) _always_ in the middle.
    
    Args:
        reds: list of list of redundant baselines as antenna tuples
        
    Return:
        _reds: the adjusted array of redundant baseline sets.
    '''
    _reds = []
    n = len(reds)
    if n%4 != 0:
        raise ValueError('Expected number of redundant baseline types to be a multiple of 4')
    _reds += reds[:n/4]
    xpols = reds[n/4:3*n/4]
    _xpols = []
    for i in range(n/4):
        _xpols.append(xpols[i] + xpols[i+n/4])
    _reds+=_xpols
    _reds+=reds[3*n/4:]
    return _reds

def aa_to_info(aa, pols=['x'], fcal=False, minV=False, **kwargs):
    '''
        Generate set of redundancies given an antenna array with idealized antenna positions.
        Args:
            aa: aipy antenna array object. Must have antpos_ideal or ant_layout attributes.

            The remaining arguments are passed to omnical.arrayinfo.filter_reds()
        Kwargs:
            pols: list of antenna polarizations to include. default is ['x'].
            fcal: toggle for using FirstCalRedundantInfo.
            minV: toggle pseudo-Stokes V minimization.

        Return:
            info: omnical info object. e.g. RedundantInfo or FirstCalRedundantInfo

    '''
    nant = len(aa)
    try:
        antpos_ideal = aa.antpos_ideal
        xs, ys, zs = antpos_ideal.T
        layout = np.arange(len(xs))
    except(AttributeError):
        layout = aa.ant_layout
        xs, ys = np.indices(layout.shape)
    antpos = -np.ones((nant * len(pols), 3))  # remake antpos with pol information. -1 to flag
    for ant, x, y in zip(layout.flatten(), xs.flatten(), ys.flatten()):
        for z, pol in enumerate(pols):
            z = 2**z # exponential ensures diff xpols aren't redundant w/ each other
            i = Antpol(ant, pol, len(aa))
            antpos[int(i), 0], antpos[int(i), 1], antpos[int(i), 2] = x, y, z
    reds = compute_reds(nant, pols, antpos[:nant], tol=.1)
    ex_ants = [Antpol(i, nant).ant() for i in range(antpos.shape[0]) if antpos[i, 0] == -1]
    kwargs['ex_ants'] = kwargs.get('ex_ants', []) + ex_ants
    reds = filter_reds(reds, **kwargs)
    if minV:
        reds = reds_for_minimal_V(reds)
    if fcal:
        info = FirstCalRedundantInfo(nant)
    else:
        info = RedundantInfo(nant)
    info.init_from_reds(reds, antpos)
    return info


def run_omnical(data, info, gains0=None, xtalk=None, maxiter=50,
            conv=1e-3, stepsize=.3, trust_period=1):
    '''
        Run a full run through of omnical: Logcal, lincal, and removing degeneracies.
        Args:
            data: dictionary of data with pol and blpair keys
            info: RedundantInfo object that can parse data
        Kwargs:
            gains0: dictionary (with pol, ant keys) used as the starting point for omnical.
            xtalk: input xtalk dictionary (similar to data). Used to remove an additive offset
                   before running omnical. This is usually left as None.
            maxiter: Maximum number of iterations to run in lincal.
            conv: convergence criterion for lincal.
            stepsize: size of steps to take in lincal.
            trust_period: This is the number of iterations to trust in lincal. If > 1, uses the
                         previous solution as starting point of lincal's next iteration. This
                         should always be 1!
    '''
    m1, g1, v1 = omnical.calib.logcal(data, info, xtalk=xtalk, gains=gains0,
                                      maxiter=maxiter, conv=conv, stepsize=stepsize,
                                      trust_period=trust_period)

    m2, g2, v2 = omnical.calib.lincal(data, info, gains=g1, vis=v1, xtalk=xtalk,
                                      conv=conv, stepsize=stepsize,
                                      trust_period=trust_period, maxiter=maxiter)

    _, g3, v3 = omnical.calib.removedegen(data, info, g2, v2, nondegenerategains=gains0)

    return m2, g3, v3


def compute_xtalk(res, wgts):
    '''
        Estimate xtalk as time-average of omnical residuals.
        Args:
            res: omnical residuals.
            wgts: dictionary of weights to use in xtalk generation.
    '''
    xtalk = {}
    for pol in res.keys():
        xtalk[pol] = {}
        for key in res[pol]:
            r, w = np.where(wgts[pol][key] > 0, res[pol][key], 0), wgts[pol][key].sum(axis=0)
            w = np.where(w == 0, 1, w)
            xtalk[pol][key] = (r.sum(axis=0) / w).astype(res[pol][key].dtype)  # avg over time
    return xtalk


def from_npz(filename, pols=None, bls=None, ants=None, verbose=False):
    '''
        ##Deprecated and only used for legacy purposes##
        Reconstitute results from to_npz, returns meta, gains, vismdl, xtalk, each
        keyed first by polarization, and then by bl/ant/keyword.
        Optional variables:
        pols: list of polarizations. default: None, return all
        bls: list of baselines. default: None, return all
        ants: list of antennas for gain. default: None, return all
    '''
    if type(filename) is str: filename = [filename]
    if type(pols) is str: pols = [pols]
    if type(bls) is tuple and type(bls[0]) is int: bls = [bls]
    if type(ants) is int: ants = [ants]
    #filename = np.array(filename)
    meta, gains, vismdl, xtalk = {}, {}, {}, {}
    def parse_key(k):
        bl,pol = k.split()
        bl = tuple(map(int,bl[1:-1].split(',')))
        return pol,bl
    for f in filename:
        if verbose: print 'Reading', f
        npz = np.load(f)
        for k in npz.files:
            if k[0].isdigit():
                pol,ant = k[-1:],int(k[:-1])
                if (pols==None or pol in pols) and (ants==None or ant in ants):
                    if not gains.has_key(pol): gains[pol] = {}
                    gains[pol][ant] = gains[pol].get(ant,[]) + [np.copy(npz[k])]
            try: pol,bl = parse_key(k)
            except(ValueError): continue
            if (pols is not None) and (pol not in pols): continue
            if (bls is not None) and (bl not in bls): continue
            if k.startswith('<'):
                if not vismdl.has_key(pol): vismdl[pol] = {}
                vismdl[pol][bl] = vismdl[pol].get(bl,[]) + [np.copy(npz[k])]
            elif k.startswith('('):
                if not xtalk.has_key(pol): xtalk[pol] = {}
                try:
                    dat = np.resize(np.copy(npz[k]),vismdl[pol][vismdl[pol].keys()[0]][0].shape) #resize xtalk to be like vismdl (with a time dimension too)
                except(KeyError):
                    for tempkey in npz.files:
                        if tempkey.startswith('<'): break
                    dat = np.resize(np.copy(npz[k]),npz[tempkey].shape) #resize xtalk to be like vismdl (with a time dimension too)
                if xtalk[pol].get(bl) is None: #no bl key yet
                    xtalk[pol][bl] = dat
                else: #append to array
                    xtalk[pol][bl] = np.vstack((xtalk[pol].get(bl),dat))
        # for k in [f for f in npz.files if f.startswith('<')]:
        #     pol,bl = parse_key(k)
        #     if not vismdl.has_key(pol): vismdl[pol] = {}
        #     vismdl[pol][bl] = vismdl[pol].get(bl,[]) + [np.copy(npz[k])]
        # for k in [f for f in npz.files if f.startswith('(')]:
        #     pol,bl = parse_key(k)
        #     if not xtalk.has_key(pol): xtalk[pol] = {}
        #     dat = np.resize(np.copy(npz[k]),vismdl[pol][vismdl[pol].keys()[0]][0].shape) #resize xtalk to be like vismdl (with a time dimension too)
        #     if xtalk[pol].get(bl) is None: #no bl key yet
        #         xtalk[pol][bl] = dat
        #     else: #append to array
        #         xtalk[pol][bl] = np.vstack((xtalk[pol].get(bl),dat))
        # for k in [f for f in npz.files if f[0].isdigit()]:
        #     pol,ant = k[-1:],int(k[:-1])
        #     if not gains.has_key(pol): gains[pol] = {}
        #     gains[pol][ant] = gains[pol].get(ant,[]) + [np.copy(npz[k])]
        kws = ['chi','hist','j','l','f']
        for kw in kws:
            for k in [f for f in npz.files if f.startswith(kw)]:
                meta[k] = meta.get(k,[]) + [np.copy(npz[k])]
    #for pol in xtalk: #this is already done above now
        #for bl in xtalk[pol]: xtalk[pol][bl] = np.concatenate(xtalk[pol][bl])
    for pol in vismdl:
        for bl in vismdl[pol]: vismdl[pol][bl] = np.concatenate(vismdl[pol][bl])
    for pol in gains:
        for bl in gains[pol]: gains[pol][bl] = np.concatenate(gains[pol][bl])
    for k in meta:
        try: meta[k] = np.concatenate(meta[k])
        except(ValueError): pass
    return meta, gains, vismdl, xtalk


def get_phase(freqs, tau):
    '''
        Turn a delay into a phase.
        Args:
           freqs: array of frequencies in Hz (or GHz)
           tau: delay in seconds (or ns)
        Returns:
            array: of complex phases the size of freqs
    '''
    freqs = freqs.reshape(-1,1)
    return np.exp(-2j*np.pi*freqs*tau)

def from_fits(filename, pols=None, bls=None, ants=None, verbose=False):
    """
    Read a calibration fits file (pyuvdata format). This also finds the model
    visibilities and the xtalkfile. 

    Args:
        filename: Name of calfits file storing omnical solutions.
                  There should also be corresponding files for the visibilities
                  and crosstalk. These filenames should have be *vis{xtalk}.fits.
    Kwargs:
        pols: Specify polarizations (xx,yy,xy,yx, etc...) to read.
        bls: Specify bls to read. list of bls
        ants: Specify ants to read. list of antenna numbers
        verbose: Be verbose.

    Returns meta, gains, vis, xtalk in old format (see from_npz)
    """
    if type(filename) is str: filename = [filename]
    if type(pols) is str: pols = [pols]
    if type(bls) is tuple and type(bls[0]) is int: bls = [bls]
    if type(ants) is int: ants = [ants]
    meta, gains = {}, {}
    poldict = {-5: 'xx', -6: 'yy', -7:'xy', -8:'yx'}
    if not pols is None: jones_params = set(x for pol in pols for x in pol)
    
    firstcal = filename[0].split('.')[-2] == 'firstcal'

    cal = UVCal()
    # filename loop
    for f in filename:
        cal.read_calfits(f)
        if ants is not None:
            cal.select(antenna_nums=ants, jones=jones_params)
        # number of spectral windows loop
        for nspw in xrange(cal.Nspws):
            # polarization loop
            for k, p in enumerate(cal.jones_array):
                pol = poldict[p][0]
                if pol not in gains.keys(): gains[pol] = {}
                # antenna loop
                for i, ant in enumerate(cal.antenna_numbers):
                    # if the cal_type is gain, create or concatenate gain_array
                    if cal.cal_type == 'gain':
                        if ant not in gains[pol].keys():
                            gains[pol][ant] = cal.gain_array[i, nspw, :, :, k].T
                        else:
                            gains[pol][ant] = np.concatenate([gains[pol][ant], cal.gain_array[i, nspw, :, :, k].T])
                        if not 'chisq{0}{1}'.format(ant, pol) in meta.keys():
                            meta['chisq{0}{1}'.format(ant, pol)] = cal.quality_array[i, nspw, :, :, k].T
                        else:
                            meta['chisq{0}{1}'.format(ant, pol)] = np.concatenate([meta['chisq{0}{1}'.format(ant, pol)], cal.quality_array[i, nspw, :, :, k].T])
                    # if the cal_type is delay, create or concatenate delay_array
                    elif cal.cal_type == 'delay':
                        if ant not in gains[pol].keys():
                            gains[pol][ant] = get_phase(cal.freq_array, cal.delay_array[i, nspw, :, k]).T
                        else:
                            gains[pol][ant] = np.concatenate([gains[pol][ant], get_phase(cal.freq_array, cal.delay_array[i, nspw, :, k]).T])
                        if not 'chisq{0}{1}'.format(ant, pol) in meta.keys():
                            meta['chisq{0}{1}'.format(ant, pol)] = cal.quality_array[i, nspw, :, k].T
                        else:
                            meta['chisq{0}{1}'.format(ant, pol)] = np.concatenate([meta['chisq{0}{1}'.format(ant, pol)], cal.quality_array[i, nspw, :, k].T])
                    else:
                        print 'Not a recognized file type'
        if not 'times' in meta.keys():
            meta['times'] = cal.time_array
        else:
            meta['times'] = np.concatenate([meta['times'], cal.time_array])

        meta['history'] = cal.history  # only taking history of the last file
        # checks to see if all files have the same cal_types
        if meta.has_key('caltype'):
            if cal.cal_type == meta['caltype']: pass
            else: raise ValueError("All caltypes are not the same across files")
        else: meta['caltype'] = cal.cal_type

        # checks to see if all files have the same gain conventions
        if meta.has_key('gain_conventions'):
            if cal.gain_convention == meta['gain_conventions']: pass
            else: raise ValueError("All gain conventions for calibration solutions is not the same across files.")
        else: meta['gain_conventions'] = cal.gain_convention

        # checks to see if all files have the same gain conventions
        if meta.has_key('inttime'):
            if cal.integration_time == meta['inttime']: pass
            else: raise ValueError("All integration times for calibration solutions is not the same across files.")
        else: meta['inttime'] = cal.integration_time


        # checks to see if all files have the same frequencies
        if meta.has_key('freqs'):
            if np.all(cal.freq_array.flatten() == meta['freqs']): pass
            else: raise ValueError("All files don't have the same frequencies")
        else: meta['freqs'] = cal.freq_array.flatten()



    v = {}
    x = {}
    # if these are omnical solutions, there vis.fits and xtalk.fits were created.
    if not firstcal:
        visfile = ['.'.join(fitsname.split('.')[:-1]) + '.vis.fits' for fitsname in filename]
        xtalkfile = ['.'.join(fitsname.split('.')[:-1]) + '.xtalk.fits' for fitsname in filename]

        vis = UVData()
        xtalk = UVData()
        for f1, f2 in zip(visfile, xtalkfile):
            if os.path.exists(f1) and os.path.exists(f2):
                vis.read_uvfits(f1)
                vis.unphase_to_drift()  # need to do this since all uvfits files are phased! PAPER/HERA miriad files are drift.
                xtalk.read_uvfits(f2)
                xtalk.unphase_to_drift()  # need to do this since all uvfits files are phased! PAPER/HERA miriad files are drift.
                if not ants is None:
                    vis.select(antenna_nums=ants)
                    xtalk.select(antenna_nums=ants)
                if not bls is None:
                    vis.select(ant_pair_nums=bls)
                    xtalk.select(ant_pair_nums=bls)
                for p, pol in enumerate(vis.polarization_array):
                    pol = poldict[pol]
                    if pol not in v.keys(): v[pol] = {}
                    for bl, k in zip(*np.unique(vis.baseline_array, return_index=True)):
                        # note we reverse baseline here b/c of conventions
                        if not vis.baseline_to_antnums(bl)[::-1] in v[pol].keys():
                            v[pol][vis.baseline_to_antnums(bl)[::-1]] = vis.data_array[k:k + vis.Ntimes, 0, :, p]
                        else:
                            v[pol][vis.baseline_to_antnums(bl)[::-1]] = np.concatenate([v[pol][vis.baseline_to_antnums(bl)[::-1]], vis.data_array[k:k + vis.Ntimes, 0, :, p]])

                DATA_SHAPE = (vis.Ntimes, vis.Nfreqs)
                for p, pol in enumerate(xtalk.polarization_array):
                    pol = poldict[pol]
                    if pol not in x.keys(): x[pol] = {}
                    for bl, k in zip(*np.unique(xtalk.baseline_array, return_index=True)):
                        if not xtalk.baseline_to_antnums(bl)[::-1] in x[pol].keys():
                            x[pol][xtalk.baseline_to_antnums(bl)[::-1]] = np.resize(xtalk.data_array[k:k + xtalk.Ntimes, 0, :, p], DATA_SHAPE)
                        else:
                            x[pol][xtalk.baseline_to_antnums(bl)[::-1]] = np.concatenate([x[pol][xtalk.baseline_to_antnums(bl)[::-1]], np.resize(xtalk.data_array[k:k + xtalk.Ntimes, 0, :, p], DATA_SHAPE)])
        # use vis to get lst array
        if not 'lsts' in meta.keys():
            meta['lsts'] = vis.lst_array[:vis.Ntimes]
        else:
            meta['lsts'] = np.concatenate([meta['lsts'], vis.lst_array[:vis.Ntimes]])
                            
    return meta, gains, v, x

def make_uvdata_vis(aa, m, v, xtalk=False):
    '''
    Given meta information and visibilities (from omnical), write out a uvfits file.
    aa      : aipy antenna array object (object)
    m       : dictionary of information (dict)
    v       : dictionary of visibilities with keys antenna pair and pol (dict)
    xtalk   : visibilities given are xtalk visibilities. (bool)
    '''

    pols = v.keys()
    antnums = np.array(v[pols[0]].keys()).T

    uv = UVData()
    # purposefully indexed with 1, 0 becuase of bug in pyuvdata. Make sure it is fixed when it gets fixed in pyuvdata
    bls = sorted(map(uv.antnums_to_baseline, antnums[1], antnums[0])) # XXX sort the baselines
    if xtalk:
        uv.Ntimes = 1
    else:
        uv.Ntimes = len(m['times'])
    uv.Npols = len(pols)
    uv.Nbls = len(v[pols[0]].keys())
    uv.Nblts = uv.Nbls * uv.Ntimes
    uv.Nfreqs = len(m['freqs'])
    data = {}
    for p in v.keys():
        if p not in data.keys():
            data[p] = []
        for bl in bls:  # crucial to loop over bls here and not v[p].keys()
            data[p].append(v[p][uv.baseline_to_antnums(bl)])
        data[p] = np.array(data[p]).reshape(uv.Nblts, uv.Nfreqs)

    uv.data_array = np.expand_dims(np.array([data[p] for p in pols]).T.swapaxes(0, 1), axis=1)
    uv.vis_units = 'uncalib'
    uv.nsample_array = np.ones_like(uv.data_array, dtype=np.float)
    uv.flag_array = np.zeros_like(uv.data_array, dtype=np.bool)
    uv.Nspws = 1  # this is always 1 for paper and hera(currently)
    uv.spw_array = np.array([uv.Nspws])
    blts = np.array([bl for bl in bls for i in range(uv.Ntimes)])
    if xtalk:
        uv.time_array = np.array(list(m['times'][:1]) * uv.Nbls)
        uv.lst_array = np.array(list(m['lsts'][:1]) * uv.Nbls)
    else:
        uv.time_array = np.array(list(m['times']) * uv.Nbls)
        uv.lst_array = np.array(list(m['lsts']) * uv.Nbls)

    # generate uvw
    uvw = []
    for t in range(uv.Ntimes):
        for bl in bls:
            uvw.append(aa.gen_uvw(*uv.baseline_to_antnums(bl), src='z').reshape(3, -1))
    uv.uvw_array = np.array(uvw).reshape(-1, 3)

    uv.ant_1_array = uv.baseline_to_antnums(blts)[0]
    uv.ant_2_array = uv.baseline_to_antnums(blts)[1]
    uv.baseline_array = uv.antnums_to_baseline(uv.ant_1_array,
                                               uv.ant_2_array)

    uv.freq_array = m['freqs'].reshape(1, -1)
    poldict = {'xx': -5, 'yy': -6, 'xy': -7, 'yx': -8}
    uv.polarization_array = np.array([poldict[p] for p in pols])
    if xtalk:
        # xtalk integration time is averaged over the whole file
        uv.integration_time = m['inttime'] * len(m['times'])
    else:
        uv.integration_time = m['inttime']
    uv.channel_width = np.float(np.diff(uv.freq_array[0])[0])

    # observation parameters
    uv.object_name = 'zenith'
    uv.telescope_name = 'HERA'
    uv.instrument = 'HERA'
    tobj = uvtel.get_telescope(uv.telescope_name)
    uv.telescope_location = tobj.telescope_location
    uv.history = m['history']

    # phasing information
    uv.phase_type = 'drift'
    uv.zenith_ra = uv.lst_array
    uv.zenith_dec = np.array([aa.lat] * uv.Nblts)

    # antenna information
    uv.Nants_telescope = len(aa)
    uv.antenna_numbers = np.arange(uv.Nants_telescope, dtype=int)
    uv.Nants_data = len(np.unique(np.concatenate([uv.ant_1_array, uv.ant_2_array]).flatten()))
    uv.antenna_names = ['ant{0}'.format(ant) for ant in uv.antenna_numbers]
    antpos = []
    for k in aa:
        antpos.append(k.pos)

    uv.antenna_positions = np.array(antpos)

    return uv

class HERACal(UVCal):
    '''
       Class that loads in hera omnical data into a pyuvdata calfits object.
       This can then be saved to a file, plotted, etc.
    '''
    def __init__(self, meta, gains, flags=None, DELAY=False, ex_ants=[], appendhist='', optional={}):
        '''
            Initialize a UVCal object.
            Args:
                meta: meta information dictionary. As returned by from_fits or from_npz.
                gains: dictionary of complex gain solutions or delays.
            Kwargs:
                flags: Optional input flags for gains.
                DELAY: toggle if calibration solutions in gains are delays.
                ex_ants: antennas that are excluded from gains.
                appendhist: string to append to history
                optional: dictionary of optional parameters to be passed to UVCal object.
        '''

        super(HERACal, self).__init__()

        # helpful dictionaries for antenna polarization of gains
        str2pol = {'x': -5, 'y': -4}
        pol2str = {-5: 'x', -4: 'y'}

        chisqdict = {}
        datadict = {}
        flagdict = {}
        ants = []
        # generate data (gain) and flag dictionaries.
        for pol in gains:
            for ant in np.sort(gains[pol].keys()):
                datadict['%d%s' % (ant, pol)] = gains[pol][ant]
                if flags:
                    flagdict['%d%s' % (ant, pol)] = flags[pol][ant]
                if ant not in ants:
                    ants.append(ant)

        # drop antennas that are not solved for.
        allants = ants + ex_ants  # total number of antennas
        ants = np.sort(ants)
        allants = np.sort(allants)
        antnames = ['ant' + str(ant) for ant in ants]
        time = meta['times']
        freq = meta['freqs']  # this is in Hz (should be anyways)
        pols = [str2pol[p] for p in gains.keys()]  # all of the polarizations

        # get sizes of things
        npol = len(pols)
        ntimes = time.shape[0]
        nfreqs = freq.shape[0]
        nants = len(ants)

        datarray = []
        flgarray = []
        for ii in range(npol):
            dd = []
            fl = []
            for ant in ants:
                try:
                    dd.append(datadict[str(ant) + pol2str[pols[ii]]])
                    if flags:
                        fl.append(flagdict[str(ant) + pol2str[pols[ii]]])
                    else:
                        fl.append(np.zeros_like(dd[-1], dtype=bool))
                # if antenna not in data dict (aka, a bad antenna)
                except(KeyError):
                    print "Can't find antenna {0}".format(ant)
            datarray.append(dd)
            flgarray.append(fl)

        datarray = np.array(datarray)
        datarray = datarray.swapaxes(0, 3).swapaxes(0, 1)

        flgarray = np.array(flgarray)
        flgarray = flgarray.swapaxes(0, 3).swapaxes(0, 1)

        tarray = time
        parray = np.array(pols)
        farray = np.array(freq)
        numarray = list(map(int, ants))
        namarray = antnames

        chisqarray = []
        for ii in range(npol):
            ch = []
            for ant in ants:
                try:
                    ch.append(meta['chisq'+str(ant)+pol2str[pols[ii]]])
                except:
                    ch.append(np.ones_like(dd[-1]))  # array of ones
            chisqarray.append(ch)
        chisqarray = np.array(chisqarray).swapaxes(0, 3).swapaxes(0, 1)

        # set the optional attributes to UVCal class.
        for key in optional:
            setattr(self, key, optional[key])
        self.telescope_name = 'HERA'
        self.Nfreqs = nfreqs
        self.Njones = len(pols)
        self.Ntimes = ntimes
        try:
            self.history = meta['history'].replace('\n', ' ') + appendhist
        except KeyError:
            self.history = appendhist
        self.Nants_data = len(ants)  # only ants with data
        self.antenna_names = namarray[:self.Nants_data]
        self.antenna_numbers = numarray[:self.Nants_data]
        self.ant_array = np.array(numarray[:self.Nants_data])
        self.Nants_telescope = nants  # total number of antennas
        self.Nspws = 1  # This is by default 1. No support for > 1 in pyuvdata.

        self.freq_array = farray[:self.Nfreqs].reshape(self.Nspws, -1)
        self.channel_width = np.diff(self.freq_array)[0][0]
        self.jones_array = parray[:self.Njones]
        self.time_array = tarray[:self.Ntimes]
        self.integration_time = meta['inttime']
        self.gain_convention = 'divide'
        self.x_orientation = 'east'
        self.time_range = [self.time_array[0], self.time_array[-1]]
        self.freq_range = [self.freq_array[0][0], self.freq_array[0][-1]]
        if DELAY:
            self.set_delay()
            self.delay_array = datarray  # units of seconds
            self.quality_array = chisqarray  
            self.flag_array = flgarray.astype(np.bool)[:, np.newaxis,  :, :, :]
        else:
            self.set_gain()
            # adding new axis for the spectral window axis. This is default to 1.
            # This needs to change when support for Nspws>1 in pyuvdata.
            self.gain_array = datarray[:, np.newaxis, :, :, :]
            self.quality_array = chisqarray[:, np.newaxis, :, :, :]
            self.flag_array = flgarray.astype(np.bool)[:, np.newaxis, :, :, :]

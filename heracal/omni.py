import numpy as np
import omnical
from red import redundant_bl_cal_simple
from copy import deepcopy
import numpy.linalg as la
from pyuvdata import UVCal, UVData
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import scipy.sparse as sps

POL_TYPES = 'xylrab'
# XXX this can't support restarts or changing # pols between runs
POLNUM = {}  # factor to multiply ant index for internal ordering
NUMPOL = {}


def add_pol(p):
    global NUMPOL
    assert(p in POL_TYPES)
    POLNUM[p] = len(POLNUM)
    NUMPOL = dict(zip(POLNUM.values(), POLNUM.keys()))


class Antpol:
    def __init__(self, *args):
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
    '''Filter redundancies to include/exclude the specified bls, antennas, and unique bl groups and polarizations.
    Assumes reds indices are Antpol objects.'''
    def pol(bl): return bl[0].pol() + bl[1].pol()
    if crosspols: reds = [r for r in reds if pol(r[0]) in crosspols]
    if ex_crosspols: reds = [r for r in reds if not pol(r[0]) in ex_crosspols]
    return omnical.arrayinfo.filter_reds(reds, bls=bls, ex_bls=ex_bls, ants=ants, ex_ants=ex_ants, ubls=ubls, ex_ubls=ex_ubls)


class RedundantInfo(omnical.calib.RedundantInfo):
    def __init__(self, nant, filename=None):
        omnical.info.RedundantInfo.__init__(self, filename=filename)
        self.nant = nant

    def bl_order(self):
        '''Return (i,j) baseline tuples in the order that they should appear in data.  Antenna indicies
        are in real-world order (as opposed to the internal ordering used in subsetant).'''
        return [(Antpol(self.subsetant[i], self.nant), Antpol(self.subsetant[j], self.nant)) for (i, j) in self.bl2d]

    def order_data(self, dd):
        '''Create a data array ordered for use in _omnical.redcal.  'dd' is
        a dict whose keys are (i,j) antenna tuples; antennas i,j should be ordered to reflect
        the conjugation convention of the provided data.  'dd' values are 2D arrays
        of (time,freq) data.'''
        d = []
        for i, j in self.bl_order():
            bl = (i.ant(), j.ant())
            pol = i.pol() + j.pol()
            try: d.append(dd[bl][pol])
            except(KeyError): d.append(dd[bl[::-1]][pol[::-1]].conj())
        return np.array(d).transpose((1, 2, 0))

    def pack_calpar(self, calpar, gains=None, vis=None, **kwargs):
        '''Take the pol/antenna/bl formatted gains and visibilities and
           wrap them to antpol format. Call RedundantInfo pack_calpar to
           generate calpar for omnical format.'''
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
        '''Unpack the solved for calibration parameters and repack
           those to antpol format'''
        nondegenerategains = kwargs.pop('nondegenerategains', None)
        meta, gains, vis = omnical.calib.RedundantInfo.unpack_calpar(self, calpar, **kwargs)

        def mk_ap(a): return Antpol(a, self.nant)
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


class FirstCalRedundantInfo(omnical.info.FirstCalRedundantInfo):
    def __init__(self, nant):
        omnical.info.FirstCalRedundantInfo.__init__(self)
        self.nant = nant
        print 'Loading FirstCalRedundantInfo class'

    def order_data(self, dd):
        '''Create a data array ordered for use in _omnical.redcal.  'dd' is
        a dict whose keys are (i,j) antenna tuples; antennas i,j should be ordered to reflect
        the conjugation convention of the provided data.  'dd' values are 2D arrays
        of (time,freq) data.''' # XXX does time/freq ordering matter.  should data be 2D instead?
        return np.array([dd[bl] if dd.has_key(bl) else dd[bl[::-1]].conj()
            for bl in self.bl_order()]).transpose((1,2,0))

def compute_reds(nant, pols, *args, **kwargs):
    _reds = omnical.arrayinfo.compute_reds(*args, **kwargs)
    reds = []
    for pi in pols:
        for pj in pols:
            reds += [[(Antpol(i, pi, nant), Antpol(j, pj, nant)) for i, j in gp] for gp in _reds]
    return reds


def aa_to_info(aa, pols=['x'], fcal=False, **kwargs):
    '''Use aa.ant_layout to generate redundances based on ideal placement.
        The remaining arguments are passed to omnical.arrayinfo.filter_reds()'''
    nant = len(aa)
    try:
        antpos_ideal = aa.antpos_ideal
        xs, ys, zs = antpos_ideal.T
        layout = np.arange(len(xs))
        # antpos = np.concatenat([antpos_ideal for i in len(pols)])
    except(AttributeError):
        layout = aa.ant_layout
        xs, ys = np.indices(layout.shape)
    antpos = -np.ones((nant * len(pols), 3))  # remake antpos with pol information. -1 to flag
    for ant, x, y in zip(layout.flatten(), xs.flatten(), ys.flatten()):
        for z, pol in enumerate(pols):
            z = 2**z
            i = Antpol(ant, pol, len(aa))
            antpos[i, 0], antpos[i, 1], antpos[i, 2] = x, y, z
    reds = compute_reds(nant, pols, antpos[:nant], tol=.1)
    ex_ants = [Antpol(i, nant).ant() for i in range(antpos.shape[0]) if antpos[i, 0] == -1]
    kwargs['ex_ants'] = kwargs.get('ex_ants', []) + ex_ants
    reds = filter_reds(reds, **kwargs)
    if fcal:
        info = FirstCalRedundantInfo(nant)
    else:
        info = RedundantInfo(nant)
    info.init_from_reds(reds, antpos)
    return info


def run_omnical(data, info, gains0=None, xtalk=None, maxiter=50,
            conv=1e-3, stepsize=.3, trust_period=1):
    '''Run a full run through of omnical: Logcal, lincal, and removing degeneracies.'''


    m1, g1, v1 = omnical.calib.logcal(data, info, xtalk=xtalk, gains=gains0,
                                      maxiter=maxiter, conv=conv, stepsize=stepsize,
                                      trust_period=trust_period)

    m2, g2, v2 = omnical.calib.lincal(data, info, gains=g1, vis=v1, xtalk=xtalk,
                                      conv=conv, stepsize=stepsize,
                                      trust_period=trust_period, maxiter=maxiter)

    _, g3, v3 = omnical.calib.removedegen(data, info, g2, v2, nondegenerategains=gains0)

    return m2, g3, v3


def compute_xtalk(res, wgts):
    '''Estimate xtalk as time-average of omnical residuals.'''
    xtalk = {}
    for pol in res.keys():
        xtalk[pol] = {}
        for key in res[pol]:
            r, w = np.where(wgts[pol][key] > 0, res[pol][key], 0), wgts[pol][key].sum(axis=0)
            w = np.where(w == 0, 1, w)
            xtalk[pol][key] = (r.sum(axis=0) / w).astype(res[pol][key].dtype)  # avg over time
    return xtalk


def to_npz(filename, meta, gains, vismdl, xtalk):
    '''Write results from omnical.calib.redcal (meta,gains,vismdl,xtalk) to npz file.
    Each of these is assumed to be a dict keyed by pol, and then by bl/ant/keyword'''
    d = {}
    metakeys = ['jds', 'lsts', 'freqs', 'history']  # ,chisq]
    for key in meta:
        if key.startswith('chisq'): d[key] = meta[key]  # separate if statements  pending changes to chisqs
        for k in metakeys:
            if key.startswith(k): d[key] = meta[key]
    for pol in gains:
        for ant in gains[pol]:
            d['%d%s' % (ant, pol)] = gains[pol][ant]
    for pol in vismdl:
        for bl in vismdl[pol]:
            d['<%d,%d> %s' % (bl[0], bl[1], pol)] = vismdl[pol][bl]
    for pol in xtalk:
        for bl in xtalk[pol]:
            d['(%d,%d) %s' % (bl[0], bl[1], pol)] = xtalk[pol][bl]
    np.savez(filename, **d)


def from_npz(filename, pols=None, bls=None, ants=None, verbose=False):
    '''Reconstitute results from to_npz, returns meta, gains, vismdl, xtalk, each
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




class HERACal(UVCal):
    '''
       Class that loads in hera omnical data into a pyuvdata calfits object.
       This can then be saved to a file, plotted, etc.
    '''
    def __init__(self, meta, gains, flags=None, DELAY=False, ex_ants=[], appendhist='', optional={}):
        '''Given meta and gain dictionary after running omnical (run_omnical), 
           populate a UVCal class upon creation.'''

        super(HERACal, self).__init__()

        str2pol = {'x': -5, 'y': -4}
        pol2str = {-5: 'x', -4: 'y'}

        chisqdict = {}
        datadict = {}
        flagdict = {}
        ants = []
        for pol in gains:
            for ant in np.sort(gains[pol].keys()):
                datadict['%d%s' % (ant, pol)] = gains[pol][ant]
                if flags:
                    flagdict['%d%s' % (ant, pol)] = flags[pol][ant]
                if ant not in ants:
                    ants.append(ant)

        # drop antennas that are not solved for.
        allants = ants + ex_ants
        ants = np.sort(ants)
        allants = np.sort(allants)
        time = meta['jds']
        freq = meta['freqs'] * 1e9
        pols = [str2pol[p] for p in gains.keys()]
        npol = len(pols)
        ntimes = time.shape[0]
        nfreqs = freq.shape[0]
        nants = len(ants)
        antnames = ['ant' + str(ant) for ant in ants]
        datarray = []
        flgarray = []
        # import IPython; IPython.embed()
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
        numarray = np.array(list(ants))
        namarray = np.array(antnames)

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
        for key in opional:
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
        self.Nants_telescope = nants  # total number of antennas
        self.Nspws = 1

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
            self.delay_array = datarray
            self.quality_array = chisqarray
            self.flag_array = flgarray.astype(np.bool)

        else:
            self.set_gain()
            self.gain_array = datarray
            self.quality_array = chisqarray
            self.flag_array = flgarray.astype(np.bool)

    def from_fits(filename, pols=None, bls=None, ants=None, verbose=False):
        """
        Read a calibration fits file (pyuvdata format). This also finds the model
        visibilities and the xtalkfile. 

        filename: Name of calfits file storing omnical solutions. 
                  There should also be corresponding files for the visibilities 
                  and crosstalk. These filenames should have be *vis{xtalk}.fits.
        pols: Specify polarization to read.
        bls: Specify bls to read. 
        ants: Specify ants to read.
        verbose: Be verbose.
    
        Returns meta, gains, vis, xtalk in old format (see from_npz)
        """
        if type(filename) is str: filename = [filename]
        if type(pols) is str: pols = [pols]
        if type(bls) is tuple and type(bls[0]) is int: bls = [bls]
        if type(ants) is int: ants = [ants]
        meta, gains = {}, {}
        poldict = {-5: 'x', -4: 'y'}
        visfile = ['.'.join(fitsname.split('.')[:-1]) + '.vis.fits' for fitsname in filename]
        xtalkfile = ['.'.join(fitsname.split('.')[:-1]) + '.xtalk.fits' for fitsname in filename]

        cal = UVCal()
        for f in filename:
            cal.read_calfits(f)
            for k, p in enumerate(cal.polarization_array):
                pol = poldict[p]
                if pol not in gains.keys(): gains[pol] = {}
                for i, ant in enumerate(cal.antenna_numbers):
                    if cal.cal_type == 'gain':
                        if ant not in gains[pol].keys():
                            gains[pol][ant] = cal.gain_array[i, :, :, k].T
                        else:
                            gains[pol][ant] = np.concatenate([gains[pol][ant], cal.gain_array[i, :, : k].T])
                        meta['chisq{0}{1}'.format(ant, pol)] = cal.quality_array[i, :, :, k].T
                    else:
                        if ant not in gains[pol].keys():
                            gains[pol][ant] = cal.gain_array[i, :, :, k].T
                        else:
                            gains[pol][ant] = np.concatenate([gains[pol][ant], cal.gain_array[i, :, :k].T])

        vis = UVData()
        v = {}
        for f in visfile:
            vis.read_uvfits(f)
            for p, pol in enumerate(vis.polarization_array):
                pol = poldict[pol] * 2
                if pol not in v.keys(): v[pol] = {}
                for bl, k in zip(*np.unique(vis.baseline_array, return_index=True)):
                    # note we reverse baseline here b/c of conventions
                    v[pol][vis.baseline_to_antnums(bl)[::-1]] = vis.data_array[k:k + vis.Ntimes, 0, :, p]

        xtalk = UVData()
        x = {}
        for f in xtalkfile:
            xtalk.read_uvfits(f)
            for p, pol in enumerate(xtalk.polarization_array):
                pol = poldict[pol] * 2
                if pol not in x.keys(): x[pol] = {}
                for bl, k in zip(*np.unique(xtalk.baseline_array, return_index=True)):
                    x[pol][xtalk.baseline_to_antnums(bl)[::-1]] = xtalk.data_array[k:k + xtalk.Ntimes, 0, :, p]

        meta['times'] = cal.time_array
        meta['freqs'] = cal.freq_array
        meta['history'] = cal.history
        meta['caltype'] = cal.cal_type
        meta['gain_conventions'] = cal.gain_convention

        return meta, gains, v, x


class FirstCal(object):
    def __init__(self, data, wgts, fqs, info):
        self.data = data
        self.fqs = fqs
        self.info = info
        self.wgts = wgts

    def data_to_delays(self, **kwargs):
        '''data = dictionary of visibilities.
           info = FirstCalRedundantInfo class
           can give it kwargs:
                supports 'window': window function for fourier transform. default is none
                         'tune'  : to fit and remove a linear slope to phase.
                         'plot'  : Low level plotting in the red.redundant_bl_cal_simple script.
                         'clean' : Clean level when deconvolving sampling function out.
           Returns 2 dictionaries:
                1. baseline pair : delays
                2. baseline pari : offset
        '''
        verbose = kwargs.get('verbose', False)
        blpair2delay = {}
        blpair2offset = {}
        dd = self.info.order_data(self.data)
        ww = self.info.order_data(self.wgts)
        for (bl1, bl2) in self.info.bl_pairs:
            if verbose:
                print (bl1, bl2)
            d1 = dd[:, :, self.info.bl_index(bl1)]
            w1 = ww[:, :, self.info.bl_index(bl1)]
            d2 = dd[:, :, self.info.bl_index(bl2)]
            w2 = ww[:, :, self.info.bl_index(bl2)]
            delay, offset = redundant_bl_cal_simple(d1, w1, d2, w2, self.fqs, **kwargs)
            blpair2delay[(bl1, bl2)] = delay
            blpair2offset[(bl1, bl2)] = offset
        return blpair2delay, blpair2offset

    def get_N(self, nblpairs):
        return sps.eye(nblpairs)

    def get_M(self, **kwargs):
        blpair2delay, blpair2offset = self.data_to_delays(**kwargs)
        sz = len(blpair2delay[blpair2delay.keys()[0]])
        M = np.zeros((len(self.info.bl_pairs), sz))
        O = np.zeros((len(self.info.bl_pairs), sz))
        for pair in blpair2delay:
            M[self.info.blpair_index(pair), :] = blpair2delay[pair]
            O[self.info.blpair_index(pair), :] = blpair2offset[pair]
        return M, O

    def run(self, **kwargs):
        verbose = kwargs.get('verbose', False)
        offset = kwargs.get('offset', False)
        # make measurement matrix
        print "Geting M,O matrix"
        self.M, self.O = self.get_M(**kwargs)
        print "Geting N matrix"
        N = self.get_N(len(self.info.bl_pairs))
        # XXX This needs to be addressed. If actually do invers, slows code way down.
        # self._N = np.linalg.inv(N)
        self._N = N  # since just using identity now

        # get coefficients matrix,A
        self.A = sps.csr_matrix(self.info.A)
        print 'Shape of coefficient matrix: ', self.A.shape

#        solve for delays
        print "Inverting A.T*N^{-1}*A matrix"
        invert = self.A.T.dot(self._N.dot(self.A)).todense()  # make it dense for pinv
        dontinvert = self.A.T.dot(self._N.dot(self.M))  # converts it all to a dense matrix
#        definitely want to use pinv here and not solve since invert is probably singular.
        self.xhat = np.dot(np.linalg.pinv(invert), dontinvert)
        # solve for offset
        if offset:
            print "Inverting A.T*N^{-1}*A matrix"
            invert = self.A.T.dot(self._N.dot(self.A)).todense()
            dontinvert = self.A.T.dot(self._N.dot(self.O))
            self.ohat = np.dot(np.linalg.pinv(invert), dontinvert)
            # turn solutions into dictionary
            return dict(zip(self.info.subsetant, zip(self.xhat, self.ohat)))
        else:
            # turn solutions into dictionary
            return dict(zip(self.info.subsetant, self.xhat))

    def get_solved_delay(self):
        solved_delays = []
        for pair in self.info.bl_pairs:
            ant_indexes = self.info.blpair2antind(pair)
            dlys = self.xhat[ant_indexes]
            solved_delays.append(dlys[0] - dlys[1] - dlys[2] + dlys[3])
        self.solved_delays = np.array(solved_delays)

# PYUV
def get_phase(fqs,tau, offset=False):
    fqs = fqs.reshape(-1,1) #need the extra axis
    if offset:
        delay = tau[0]
        offset = tau[1]
        return np.exp(-1j*(2*np.pi*fqs*delay) - offset)
    else:
        return np.exp(-2j*np.pi*fqs*tau)

# PYUV
def save_gains_fc(s,fqs,pol,filename,ubls=None,ex_ants=None,verbose=False):
    """
    s: solutions
    fqs: frequencies
    pol: polarization of single antenna i.e. 'x', or 'y'.
    filename: if a specific file was used (instead of many), change output name
    ubls: unique baselines used to solve for s'
    ex_ants: antennae excluded to solve for s'
    """
#    if isinstance(filename, list): filename=filename[0] #XXX this is evil. why?
    NBINS = len(fqs)
    s2 = {}
    for k,i in s.iteritems():
        #len > 1 means that one is using the "tune" parameter in omni.firstcal
        if len(i)>1:
            s2[str(k)+pol] = get_phase(fqs,i,offset=True).T
            s2['d'+str(k)] = i[0]
            s2['o'+str(k)] = i[1]
            if verbose: print 'dly=%f , off=%f'%i
        else:
            s2[str(k)+pol] = get_phase(fqs,i).T #reshape plays well with omni apply
            s2['d'+str(k)] = i
            if verbose: print 'dly=%f'%i
    if not ubls is None: s2['ubls']=ubls
    if not ex_ants is None: s2['ex_ants']=ex_ants
    s2['freqs']=fqs#in GHz
    outname='%s.fc.npz'%filename
    import sys
    s2['cmd'] = ' '.join(sys.argv)
    print 'Saving fcgains to %s'%outname
    np.savez(outname,**s2)

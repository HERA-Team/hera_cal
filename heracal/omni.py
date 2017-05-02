import numpy as np
import omnical
from copy import deepcopy
import numpy.linalg as la
from pyuvdata import UVCal, UVData
from heracal.firstcal import FirstCalRedundantInfo
import warnings, os
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
            antpos[int(i), 0], antpos[int(i), 1], antpos[int(i), 2] = x, y, z
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


def get_phase(freqs, tau):
    '''
       Turn a delay into a phase.
       freqs: array of frequencies in GHz.
       tau: delay in nanoseconds.
    '''
    freqs = freqs.reshape(-1,1)
    return np.exp(-2j*np.pi*freqs*tau)

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
        for k, p in enumerate(cal.jones_array):
            pol = poldict[p]
            if pol not in gains.keys(): gains[pol] = {}
            for i, ant in enumerate(cal.antenna_numbers):
                if cal.cal_type == 'gain':
                    if ant not in gains[pol].keys():
                        gains[pol][ant] = cal.gain_array[i, :, :, k].T
                    else:
                        gains[pol][ant] = np.concatenate([gains[pol][ant], cal.gain_array[i, :, : k].T])
                    meta['chisq{0}{1}'.format(ant, pol)] = cal.quality_array[i, :, :, k].T
                elif cal.cal_type == 'delay':
                    if ant not in gains[pol].keys():
                        gains[pol][ant] = get_phase(cal.freq_array/1e9, cal.delay_array[i, :, k]).T
                    else:
                        gains[pol][ant] = np.concatenate([gains[pol][ant], get_phase(cal.freq_array/1e9, cal.gain_array[i, :, :k]).T])
                else:
                    print 'Not a recognized file type'
     
    vis = UVData()
    v = {}
    for f in visfile:
        if os.path.exists(f):
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
        if os.path.exists(f):
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
        self.ant_array = numarray[:self.Nants_data]
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

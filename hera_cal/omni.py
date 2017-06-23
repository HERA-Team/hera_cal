from __future__ import print_function, division, absolute_import
import numpy as np
import omnical
from copy import deepcopy
import numpy.linalg as la
from pyuvdata import UVCal, UVData, uvtel
import aipy
from aipy.miriad import pol2str
import warnings
import os
import glob
import re
import optparse
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import scipy.sparse as sps

POL_TYPES = 'xylrabne'
# XXX this can't support restarts or changing # pols between runs
POLNUM = {}  # factor to multiply ant index for internal ordering
NUMPOL = {}

# dict for converting to polarizations
jonesLookup = {
    -5: (-5, -5),
    -6: (-6, -6),
    -7: (-5, -6),
    -8: (-6, -5)
}


def add_pol(p):
    '''Add's pols to the global POLNUM and NUMPOL dictionaries; used for creating Antpol objects.'''
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
            ant (int): antenna number
            pol (str): polarization string. e.g. 'x', or 'y'
            nant(int): total number of antennas.
        '''
        try:
            ant, pol, nant = args
            if pol not in POLNUM:
                add_pol(pol)
            self.val, self.nant = POLNUM[pol] * nant + ant, nant
        except(ValueError):
            self.val, self.nant = args

    def antpol(self):
        return self.val % self.nant, NUMPOL[self.val // self.nant]

    def ant(self):
        return self.antpol()[0]

    def pol(self):
        return self.antpol()[1]

    def __int__(self):
        return self.val

    def __hash__(self):
        return self.ant()

    def __str__(self):
        return ''.join(map(str, self.antpol()))

    def __eq__(self, v):
        return self.ant() == v

    def __repr__(self):
        return str(self)


# XXX filter_reds w/ pol support should probably be in omnical
def filter_reds(reds, bls=None, ex_bls=None, ants=None, ex_ants=None, ubls=None, ex_ubls=None, crosspols=None, ex_crosspols=None):
    '''
    Filter redundancies to include/exclude the specified bls, antennas, unique bl groups and polarizations.
    Assumes reds indices are Antpol objects.
    Args:
        reds: list of lists of redundant baselines as antenna pair tuples. e.g. [[(1,2),(2,3)], [(1,3)]]
        bls (optional): list of baselines as antenna pair tuples to include in reds.
        ex_bls (optional): list of baselines as antenna pair tuples to exclude in reds.
        ants (optional): list of antenna numbers (as int's) to include in reds.
        ex_ants (optional): list of antenna numbers (as int's) to exclude in reds.
        ubls (optional): list of baselines representing their redundant group to include in reds.
        ex_ubls (optional): list of baselines representing their redundant group to exclude in reds.
        crosspols (optional): cross polarizations to include in reds. e.g. 'xy' or 'yx'.
        ex_crosspols (optional): cross polarizations to exclude in reds. e.g. 'xy' or 'yx'.

    Return:
        reds: list of lists of redundant baselines as antenna pair tuples.
    '''
    def pol(bl): return bl[0].pol() + bl[1].pol()
    if crosspols:
        reds = [r for r in reds if pol(r[0]) in crosspols]
    if ex_crosspols:
        reds = [r for r in reds if not pol(r[0]) in ex_crosspols]
    return omnical.arrayinfo.filter_reds(reds, bls=bls, ex_bls=ex_bls, ants=ants, ex_ants=ex_ants, ubls=ubls, ex_ubls=ex_ubls)


class RedundantInfo(omnical.calib.RedundantInfo):
    '''RedundantInfo object to interface with omnical. Includes support for Antpol objects.'''

    def __init__(self, nant, filename=None):
        '''Initialize with base clas __init__ and number of antennas.

        Args:
            nant (int): number of antennas.
            filename (str): filename (str) for legacy info objects.
        '''
        omnical.info.RedundantInfo.__init__(self, filename=filename)
        self.nant = nant

    def bl_order(self):
        '''Returns expected order of baselines.

        Return:
            (i,j) baseline tuples in the order that they should appear in data.
            Antenna indicies are in real-world order
            (as opposed to the internal ordering used in subsetant).
        '''
        return [(Antpol(self.subsetant[i], self.nant), Antpol(self.subsetant[j], self.nant)) for (i, j) in self.bl2d]

    def order_data(self, dd):
        """Create a data array ordered for use in _omnical.redcal.

        Args:
            dd (dict): dictionary whose keys are (i,j) antenna tuples; antennas i,j should be ordered to reflect
                       the conjugation convention of the provided data.  'dd' values are 2D arrays of (time,freq) data.
        Return:
            array: array whose ordering reflects the internal ordering of omnical. Used to pass into pack_calpar
        """
        d = []
        for i, j in self.bl_order():
            bl = (i.ant(), j.ant())
            pol = i.pol() + j.pol()
            try:
                d.append(dd[bl][pol])
            except(KeyError):
                d.append(dd[bl[::-1]][pol[::-1]].conj())
        return np.array(d).transpose((1, 2, 0))

    def pack_calpar(self, calpar, gains=None, vis=None, **kwargs):
        ''' Pack a calpar array for use in omnical.

        Note that this function includes polarization support by wrapping
        into calpar format.

        Args:
            calpar (array): array whose size is given by self.calpar_size. Usually initialized to zeros.
            gains (dict): dictionary of starting gains for omnical run. dict[pol][antenna]
            vis (dict): dictionary of starting visibilities (for a redundant group) for omnical run. dict[pols][bl],
            nondegenerategains dict(): gains that don't have a degeneracy component to them (e.g. firstcal gains).
                                       The gains get divided out before handing off calpar to omnical.

        Returns:
            calpar (array): The populated calpar array.
        '''
        nondegenerategains = kwargs.pop('nondegenerategains', None)
        if gains:
            _gains = {}
            for pol in gains:
                for i in gains[pol]:
                    ai = Antpol(i, pol, self.nant)
                    if nondegenerategains is not None:
                        # This conj is necessary to conform to omnical conj
                        # conv.
                        _gains[int(ai)] = gains[pol][i].conj() / nondegenerategains[pol][i].conj()
                    else:
                        # This conj is necessary to conform to omnical conj
                        # conv.
                        _gains[int(ai)] = gains[pol][i].conj()
        else:
            _gains = gains

        if vis:
            _vis = {}
            for pol in vis:
                for i, j in vis[pol]:
                    ai, aj = Antpol(i, pol[0], self.nant), Antpol(
                        j, pol[1], self.nant)
                    _vis[(int(ai), int(aj))] = vis[pol][(i, j)]
        else:
            _vis = vis

        calpar = omnical.calib.RedundantInfo.pack_calpar(
            self, calpar, gains=_gains, vis=_vis)

        return calpar

    def unpack_calpar(self, calpar, **kwargs):
        '''Unpack the solved for calibration parameters and repack to antpol format

        Args:
            calpar (array): calpar array output from omnical.
            nondegenerategains (dict, optional): The nondegenerategains that were divided out in pack_calpar.
                These are multiplied back into calpar here. gain dictionary format.

        Return:
            meta (dict): dictionary of meta information from omnical. e.g. chisq, iters, etc
            gains (dict): dictionary of gains solved for by omnical. gains[pol][ant]
            vis (dict): dictionary of model visibilities solved for by omnical. vis[pols][blpair]
    '''
        nondegenerategains = kwargs.pop('nondegenerategains', None)
        meta, gains, vis = omnical.calib.RedundantInfo.unpack_calpar(
            self, calpar, **kwargs)

        def mk_ap(a): return Antpol(a, self.nant)
        if 'res' in meta:
            for i, j in meta['res'].keys():
                api, apj = mk_ap(i), mk_ap(j)
                pol = api.pol() + apj.pol()
                bl = (api.ant(), apj.ant())
                if not meta['res'].has_key(pol):
                    meta['res'][pol] = {}
                meta['res'][pol][bl] = meta['res'].pop((i, j))
        # XXX make chisq a nested dict, with individual antpol keys?
        for k in [k for k in meta.keys() if k.startswith('chisq')]:
            try:
                ant = int(k.split('chisq')[1])
                meta['chisq' + str(mk_ap(ant))] = meta.pop(k)
            except(ValueError):
                pass
        for i in gains.keys():
            ap = mk_ap(i)
            if not gains.has_key(ap.pol()):
                gains[ap.pol()] = {}
            gains[ap.pol()][ap.ant()] = gains.pop(i).conj()
            if nondegenerategains:
                gains[ap.pol()][ap.ant()] *= nondegenerategains[ap.pol()][ap.ant()]
        for i, j in vis.keys():
            api, apj = mk_ap(i), mk_ap(j)
            pol = api.pol() + apj.pol()
            bl = (api.ant(), apj.ant())
            if not vis.has_key(pol):
                vis[pol] = {}
            vis[pol][bl] = vis.pop((i, j))
        return meta, gains, vis


def compute_reds(nant, pols, *args, **kwargs):
    '''Compute the redundancies given antenna_positions and wrap into Antpol format.

    Args:
        nant: number of antennas
        pols: polarization labels, e.g. pols=['x']
        *args: args to be passed to omnical.arrayinfo.compute_reds, specifically
               antpos: array of antenna positions in order of subsetant.
        **kwargs: extra keyword arguments

    Return:
        reds: list of list of baselines as antenna tuples
       '''
    _reds = omnical.arrayinfo.compute_reds(*args, **kwargs)
    reds = []
    for pi in pols:
        for pj in pols:
            reds += [[(Antpol(i, pi, nant), Antpol(j, pj, nant))
                      for i, j in gp] for gp in _reds]
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
    if n % 4 != 0:
        raise ValueError(
            'Expected number of redundant baseline types to be a multiple of 4')
    _reds += reds[:n // 4]
    xpols = reds[n // 4:3 * n // 4]
    _xpols = []
    for i in range(n // 4):
        _xpols.append(xpols[i] + xpols[i + n // 4])
    _reds += _xpols
    _reds += reds[3 * n // 4:]
    return _reds


def aa_to_info(aa, pols=['x'], fcal=False, minV=False, **kwargs):
    '''Generate set of redundancies given an antenna array with idealized antenna positions.

    Args:
        aa: aipy antenna array object. Must have antpos_ideal or ant_layout attributes.

        (The remaining arguments are passed to omnical.arrayinfo.filter_reds())
        pols (optional): list of antenna polarizations to include. default is ['x'].
        fcal (optional): toggle for using FirstCalRedundantInfo.
        minV (optional): toggle pseudo-Stokes V minimization.

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
    # remake antpos with pol information. -1 to flag
    antpos = -np.ones((nant * len(pols), 3))
    for ant, x, y in zip(layout.flatten(), xs.flatten(), ys.flatten()):
        for z, pol in enumerate(pols):
            z = 2**z  # exponential ensures diff xpols aren't redundant w/ each other
            i = Antpol(ant, pol, len(aa))
            antpos[int(i), 0], antpos[int(i), 1], antpos[int(i), 2] = x, y, z
    reds = compute_reds(nant, pols, antpos[:nant], tol=.1)
    ex_ants = [Antpol(i, nant).ant()
               for i in range(antpos.shape[0]) if antpos[i, 0] == -1]
    kwargs['ex_ants'] = kwargs.get('ex_ants', []) + ex_ants
    reds = filter_reds(reds, **kwargs)
    if minV:
        reds = reds_for_minimal_V(reds)
    if fcal:
        from hera_cal.firstcal import FirstCalRedundantInfo
        info = FirstCalRedundantInfo(nant)
    else:
        info = RedundantInfo(nant)
    info.init_from_reds(reds, antpos)
    return info


def run_omnical(data, info, gains0=None, xtalk=None, maxiter=50,
                conv=1e-3, stepsize=.3, trust_period=1):
    '''Run a full run through of omnical: Logcal, lincal, and removing degeneracies.

    Args:
        data: dictionary of data with pol and blpair keys
        info: RedundantInfo object that can parse data
        gains0 (dict, optional): dictionary (with pol, ant keys) used as the starting point for omnical.
        xtalk (optional): input xtalk dictionary (similar to data). Used to remove an additive offset
            before running omnical. This is usually left as None.
        maxiter (optional): Maximum number of iterations to run in lincal.
        conv (optional): convergence criterion for lincal.
        stepsize (optional): size of steps to take in lincal.
        trust_period (optional): This is the number of iterations to trust in lincal. If > 1, uses the
                     previous solution as starting point of lincal's next iteration. This
                     should always be 1!

    Returns:
        m2 (dict): dictionary of meta information.
        g3 (dict): dictionary of gain solutions.
        v3 (dict): dictionary of model visibilites.
    '''
    m1, g1, v1 = omnical.calib.logcal(data, info, xtalk=xtalk, gains=gains0,
                                      maxiter=maxiter, conv=conv, stepsize=stepsize,
                                      trust_period=trust_period)

    m2, g2, v2 = omnical.calib.lincal(data, info, gains=g1, vis=v1, xtalk=xtalk,
                                      conv=conv, stepsize=stepsize,
                                      trust_period=trust_period, maxiter=maxiter)

    _, g3, v3 = omnical.calib.removedegen(
        data, info, g2, v2, nondegenerategains=gains0)

    return m2, g3, v3


def compute_xtalk(res, wgts):
    '''Estimate xtalk as time-average of omnical residuals.

    Args:
        res: omnical residuals.
        wgts: dictionary of weights to use in xtalk generation.

    Returns:
        xtalk (dict): dictionary of visibilities.
    '''
    xtalk = {}
    for pol in res.keys():
        xtalk[pol] = {}
        for key in res[pol]:
            r, w = np.where(wgts[pol][key] > 0, res[pol][key], 0), wgts[
                pol][key].sum(axis=0)
            w = np.where(w == 0, 1, w)
            xtalk[pol][key] = (r.sum(axis=0) / w).astype(res[pol]
                                                         [key].dtype)  # avg over time
    return xtalk


def from_npz(filename, pols=None, bls=None, ants=None, verbose=False):
    '''##Deprecated and only used for legacy purposes##

    Args:
        filename (list): list of npz files to read.
        pols (optional): list of polarizations. default: None, return all
        bls (optional): list of baselines. default: None, return all
        ants (optional): list of antennas for gain. default: None, return all

    Returns:
        meta (dict): dictionary of meta information
        gains (dict): dictionary of gains
        xtalk (dict): dictionary of xtalk
    '''
    if type(filename) is str:
        filename = [filename]
    if type(pols) is str:
        pols = [pols]
    if type(bls) is tuple and type(bls[0]) is int:
        bls = [bls]
    if type(ants) is int:
        ants = [ants]
    #filename = np.array(filename)
    meta, gains, vismdl, xtalk = {}, {}, {}, {}

    def parse_key(k):
        bl, pol = k.split()
        bl = tuple(map(int, bl[1:-1].split(',')))
        return pol, bl
    for f in filename:
        if verbose:
            print('Reading', f)
        npz = np.load(f)
        for k in npz.files:
            if k[0].isdigit():
                pol, ant = k[-1:], int(k[:-1])
                if (pols == None or pol in pols) and (ants == None or ant in ants):
                    if not gains.has_key(pol):
                        gains[pol] = {}
                    gains[pol][ant] = gains[pol].get(
                        ant, []) + [np.copy(npz[k])]
            try:
                pol, bl = parse_key(k)
            except(ValueError):
                continue
            if (pols is not None) and (pol not in pols):
                continue
            if (bls is not None) and (bl not in bls):
                continue
            if k.startswith('<'):
                if not vismdl.has_key(pol):
                    vismdl[pol] = {}
                vismdl[pol][bl] = vismdl[pol].get(bl, []) + [np.copy(npz[k])]
            elif k.startswith('('):
                if not xtalk.has_key(pol):
                    xtalk[pol] = {}
                try:
                    # resize xtalk to be like vismdl (with a time dimension too)
                    dat = np.resize(np.copy(npz[k]), vismdl[pol][
                                    vismdl[pol].keys()[0]][0].shape)
                except(KeyError):
                    for tempkey in npz.files:
                        if tempkey.startswith('<'):
                            break
                    # resize xtalk to be like vismdl (with a time dimension too)
                    dat = np.resize(np.copy(npz[k]), npz[tempkey].shape)
                if xtalk[pol].get(bl) is None:  # no bl key yet
                    xtalk[pol][bl] = dat
                else:  # append to array
                    xtalk[pol][bl] = np.vstack((xtalk[pol].get(bl), dat))
        kws = ['chi', 'hist', 'j', 'l', 'f']
        for kw in kws:
            for k in [f for f in npz.files if f.startswith(kw)]:
                meta[k] = meta.get(k, []) + [np.copy(npz[k])]

    for pol in vismdl:
        for bl in vismdl[pol]:
            vismdl[pol][bl] = np.concatenate(vismdl[pol][bl])
    for pol in gains:
        for bl in gains[pol]:
            gains[pol][bl] = np.concatenate(gains[pol][bl])
    for k in meta:
        try:
            meta[k] = np.concatenate(meta[k])
        except(ValueError):
            pass
    return meta, gains, vismdl, xtalk


def get_phase(freqs, tau):
    '''Turn a delay into a phase.

    Args:
        freqs: array of frequencies in Hz (or GHz)
        tau: delay in seconds (or ns)
    Returns:
        array: of complex phases the size of freqs
    '''
    freqs = freqs.reshape(-1, 1)
    return np.exp(-2j * np.pi * freqs * tau)


def from_fits(filename, keep_delay=False, **kwargs):
    """
    Read a calibration fits file (pyuvdata format). This also finds the model
    visibilities and the xtalkfile. 

    Args:
        filename: Name of calfits file storing omnical solutions.
            There should also be corresponding files for the visibilities
            and crosstalk. These filenames should have be *vis{xtalk}.fits.
        **kwargs : extra keywords that are passed into the select function 
            for the UVCal object and UVData object. Refer to pyuvdata.UVCal.select
            and pyuvdata.UVData.select for use.
    Returns:
        meta (dict): dictionary of meta information
        gains (dict): dictionary of gains
        xtalk (dict): dictionary of xtalk
    """
    if type(filename) is str:
        filename = [filename]
    meta, gains = {}, {}
    poldict = {-5: 'xx', -6: 'yy', -7: 'xy', -8: 'yx'}

    firstcal = filename[0].split('.')[-2] == 'first'

    cal = UVCal()
    # filename loop
    for f in filename:
        cal.read_calfits(f)
        if len(kwargs) != 0:
            cal.select(**kwargs)

        print(f)

        #######error checks#######
        # checks to see if all files have the same cal_types
        if meta.has_key('caltype'):
            if cal.cal_type == meta['caltype']:
                pass
            else:
                raise ValueError("All caltypes are not the same across files")
        else:
            meta['caltype'] = cal.cal_type

        # checks to see if all files have the same gain conventions
        if meta.has_key('gain_conventions'):
            if cal.gain_convention == meta['gain_conventions']:
                pass
            else:
                raise ValueError(
                    "All gain conventions for calibration solutions is not the same across files.")
        else:
            meta['gain_conventions'] = cal.gain_convention

        # checks to see if all files have the same gain conventions
        if meta.has_key('inttime'):
            if cal.integration_time == meta['inttime']:
                pass
            else:
                raise ValueError(
                    "All integration times for calibration solutions is not the same across files.")
        else:
            meta['inttime'] = cal.integration_time

        # checks to see if all files have the same frequencies
        if meta.has_key('freqs'):
            if np.all(cal.freq_array.flatten() == meta['freqs']):
                pass
            else:
                raise ValueError("All files don't have the same frequencies")
        else:
            meta['freqs'] = cal.freq_array.flatten()

        # number of spectral windows loop
        for nspw in xrange(cal.Nspws):
            # polarization loop
            for k, p in enumerate(cal.jones_array):
                pol = poldict[p][0]
                if pol not in gains.keys():
                    gains[pol] = {}
                # antenna loop
                for i, ant in enumerate(cal.ant_array):
                    # if the cal_type is gain, create or concatenate gain_array
                    if cal.cal_type == 'gain':
                        if ant not in gains[pol].keys():
                            gains[pol][ant] = cal.gain_array[
                                i, nspw, :, :, k].T
                        else:
                            gains[pol][ant] = np.concatenate(
                                [gains[pol][ant], cal.gain_array[i, nspw, :, :, k].T])
                        if not 'chisq{0}{1}'.format(ant, pol) in meta.keys():
                            meta['chisq{0}{1}'.format(ant, pol)] = cal.quality_array[
                                i, nspw, :, :, k].T
                        else:
                            meta['chisq{0}{1}'.format(ant, pol)] = np.concatenate(
                                [meta['chisq{0}{1}'.format(ant, pol)], cal.quality_array[i, nspw, :, :, k].T])
                    # if the cal_type is delay, create or concatenate
                    # delay_array
                    elif cal.cal_type == 'delay':
                        if ant not in gains[pol].keys():
                            if keep_delay:
                                gains[pol][ant] = cal.delay_array[
                                    i, nspw, 0, :, k].T
                            else:
                                gains[pol][ant] = get_phase(
                                    cal.freq_array, cal.delay_array[i, nspw, 0, :, k]).T
                        else:
                            if keep_delay:
                                gains[pol][ant] = np.concatenate(
                                    [gains[pol][ant], cal.delay_array[i, nspw, 0, :, k].T])
                            else:
                                gains[pol][ant] = np.concatenate([gains[pol][ant], get_phase(
                                    cal.freq_array, cal.delay_array[i, nspw, 0, :, k]).T])
                        if not 'chisq{0}{1}'.format(ant, pol) in meta.keys():
                            meta['chisq{0}{1}'.format(ant, pol)] = cal.quality_array[
                                i, nspw, 0, :, k].T
                        else:
                            meta['chisq{0}{1}'.format(ant, pol)] = np.concatenate(
                                [meta['chisq{0}{1}'.format(ant, pol)], cal.quality_array[i, nspw, 0, :, k].T])
                    else:
                        raise ValueError("Not a recognized file type.")

        if not 'times' in meta.keys():
            meta['times'] = cal.time_array
        else:
            meta['times'] = np.concatenate([meta['times'], cal.time_array])

        meta['history'] = cal.history  # only taking history of the last file

    v = {}
    x = {}
    # if these are omnical solutions, there vis.fits and xtalk.fits were
    # created.
    if not firstcal:
        visfile = ['.'.join(fitsname.split('.')[:-2]) +
                   '.vis.uvfits' for fitsname in filename]
        xtalkfile = ['.'.join(fitsname.split('.')[:-2]) +
                     '.xtalk.uvfits' for fitsname in filename]

        vis = UVData()
        xtalk = UVData()
        for f1, f2 in zip(visfile, xtalkfile):
            if os.path.exists(f1) and os.path.exists(f2):
                vis.read_uvfits(f1)
                # need to do this since all uvfits files are phased! PAPER/HERA
                # miriad files are drift.
                vis.unphase_to_drift()
                xtalk.read_uvfits(f2)
                # need to do this since all uvfits files are phased! PAPER/HERA
                # miriad files are drift.
                xtalk.unphase_to_drift()
                if len(kwargs) != 0:
                    vis.select(**kwargs)
                    xtalk.select(**kwargs)
                for p, pol in enumerate(vis.polarization_array):
                    pol = poldict[pol]
                    if pol not in v.keys():
                        v[pol] = {}
                    for bl, k in zip(*np.unique(vis.baseline_array, return_index=True)):
                        # note we reverse baseline here b/c of conventions
                        if not vis.baseline_to_antnums(bl) in v[pol].keys():
                            v[pol][vis.baseline_to_antnums(bl)] = vis.data_array[
                                k:k + vis.Ntimes, 0, :, p]
                        else:
                            v[pol][vis.baseline_to_antnums(bl)] = np.concatenate(
                                [v[pol][vis.baseline_to_antnums(bl)], vis.data_array[k:k + vis.Ntimes, 0, :, p]])

                DATA_SHAPE = (vis.Ntimes, vis.Nfreqs)
                for p, pol in enumerate(xtalk.polarization_array):
                    pol = poldict[pol]
                    if pol not in x.keys():
                        x[pol] = {}
                    for bl, k in zip(*np.unique(xtalk.baseline_array, return_index=True)):
                        if not xtalk.baseline_to_antnums(bl) in x[pol].keys():
                            x[pol][xtalk.baseline_to_antnums(bl)] = np.resize(
                                xtalk.data_array[k:k + xtalk.Ntimes, 0, :, p], DATA_SHAPE)
                        else:
                            x[pol][xtalk.baseline_to_antnums(bl)] = np.concatenate([x[pol][xtalk.baseline_to_antnums(
                                bl)], np.resize(xtalk.data_array[k:k + xtalk.Ntimes, 0, :, p], DATA_SHAPE)])
        # use vis to get lst array
        if not 'lsts' in meta.keys():
            meta['lsts'] = vis.lst_array[:vis.Ntimes]
        else:
            meta['lsts'] = np.concatenate(
                [meta['lsts'], vis.lst_array[:vis.Ntimes]])

    return meta, gains, v, x


def make_uvdata_vis(aa, m, v, xtalk=False):
    '''Given meta information and visibilities (from omnical), return a UVData object.

    Args:
        aa: aipy antenna array object (object)
        m: dictionary of information (dict)
        v: dictionary of visibilities with keys antenna pair and pol (dict)
        xtalk (optional): visibilities given are xtalk visibilities. (bool)

    Returns:
        uv: UVData object
    '''

    pols = v.keys()
    antnums = np.array(v[pols[0]].keys()).T

    uv = UVData()
    bls = sorted(map(uv.antnums_to_baseline, antnums[0], antnums[1]))
    if xtalk:
        uv.Ntimes = 1
    else:
        uv.Ntimes = len(m['times'])
    uv.Npols = len(pols)
    uv.Nbls = len(bls)
    uv.Nblts = uv.Nbls * uv.Ntimes
    uv.Nfreqs = len(m['freqs'])
    data = {}
    for p in pols:
        if p not in data.keys():
            data[p] = []
        for bl in bls:  # crucial to loop over bls here and not v[p].keys()
            data[p].append(v[p][uv.baseline_to_antnums(bl)])
        data[p] = np.array(data[p]).reshape(uv.Nblts, uv.Nfreqs)

    uv.data_array = np.expand_dims(
        np.array([data[p] for p in pols]).T.swapaxes(0, 1), axis=1)
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
            uvw.append(aa.gen_uvw(
                *uv.baseline_to_antnums(bl), src='z').reshape(3, -1))
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
    uv.Nants_data = len(np.unique(np.concatenate(
        [uv.ant_1_array, uv.ant_2_array]).flatten()))
    uv.antenna_names = ['ant{0}'.format(ant) for ant in uv.antenna_numbers]
    antpos = []
    for k in aa:
        antpos.append(k.pos)

    uv.antenna_positions = np.array(antpos)

    return uv

# XXX Eventually this may belong in pyuvdata


def concatenate_UVCal_on_pol(calfitsList):
    '''
    Joins UVCal files of different polarizations along 
    the polarization axis of the delay_array, flag_array,
    gain_array, and quality_array.

    Args:
        calfitsList: list of calfits filenames
            type: list of strings

    Returns:
        a single cal file, with relevant arrays
        concatenated along the polarization axis
            type: pyuvdata.UVCal()
    '''
    # XXX these could be more flexible if we wanted to have it as optional
    constProperties = ['antenna_names', 'antenna_numbers', 'cal_type', 'channel_width',  'freq_range', 'gain_convention',
                       'integration_time', 'Nants_data', 'Nants_telescope', 'Nfreqs', 'Njones', 'Nspws', 'Ntimes',  'time_range', 'x_orientation']
    constPropertiesArrays = ['ant_array', 'freq_array', 'time_array']

    # check that constProperties match between files
    calname0 = calfitsList[0]
    cal0 = UVCal()
    cal0.read_calfits(calname0)

    if cal0.Njones != 1:
        raise ValueError(
            'Njones!=1; cannot concantenate > 1 polarization at a time')
    for calname1 in calfitsList[1:]:
        cal1 = UVCal()
        cal1.read_calfits(calname1)
        for prp in constProperties:
            if not getattr(cal0, prp) == getattr(cal1, prp):
                raise ValueError('%s of %s does not match %s' %
                                 (prp, calname0, calname1))
        for prp in constPropertiesArrays:
            if not (getattr(cal0, prp) == getattr(cal1, prp)).all():
                raise ValueError('%s of %s does not match %s' %
                                 (prp, calname0, calname1))
        if not cal1.Njones == 1:
            raise ValueError(
                'Njones!=1; cannot concantenate > 1 polarization at a time')
        if cal1.jones_array[0] in cal0.jones_array:
            raise ValueError(
                'Cannot concatenate calfits files of identical polarization')

        cal0.Njones += 1
        cal0.jones_array = np.concatenate((cal0.jones_array, cal1.jones_array))
        if cal0.delay_array is not None:
            cal0.delay_array = np.concatenate((cal0.delay_array, cal1.delay_array), axis=4)
        if cal0.gain_array is not None:
            cal0.gain_array = np.concatenate((cal0.gain_array, cal1.gain_array), axis=4)
        cal0.flag_array = np.concatenate((cal0.flag_array, cal1.flag_array), axis=4)
        cal0.quality_array = np.concatenate((cal0.quality_array, cal1.quality_array), axis=4)
    return cal0

class HERACal(UVCal):
    '''
       Class that loads in hera omnical data into a pyuvdata calfits object.
       This can then be saved to a file, plotted, etc.
    '''

    def __init__(self, meta, gains, flags=None, DELAY=False, ex_ants=[], appendhist='', optional={}):
        '''Initialize a UVCal object.

            Args:
                meta: meta information dictionary. As returned by from_fits or from_npz.
                gains: dictionary of complex gain solutions or delays.
                flags (optional): Optional input flags for gains.
                DELAY (optional): toggle if calibration solutions in gains are delays.
                ex_ants (optional): antennas that are excluded from gains.
                appendhist (optional): string to append to history
                optional (optional): dictionary of optional parameters to be passed to UVCal object.
        '''

        super(HERACal, self).__init__()

        # helpful dictionaries for antenna polarization of gains
        str2pol = {'x': -5, 'y': -6}
        pol2str = {-5: 'x', -6: 'y'}

        chisqdict = {}
        datadict = {}
        flagdict = {}

        # drop antennas that are not solved for. Since we are feeding in omnical/firstcal solutions into this,
        # if we provided an ex_ants those antennas will not have a key in gains. Need to provide ex_ants list
        # to hera_cal object.
        # create set to get unique antennas from both pol
        ants = list(set([ant for pol in gains.keys() for ant in gains[pol].keys()]))
        allants = np.sort(ants + ex_ants)  # total number of antennas
        ants = np.sort(ants)
        # antenna names for all antennas
        antnames = ['ant' + str(ant) for ant in allants]
        time = meta['times']
        freq = meta['freqs']  # this is in Hz (should be anyways)
        pols = [str2pol[p] for p in gains.keys()]  # all of the polarizations

        # get sizes of things
        nspw = 1  # This is by default 1. No support for > 1 in pyuvdata.
        npol = len(pols)
        ntimes = time.shape[0]
        nfreqs = freq.shape[0]

        datarray = np.array([[gains[pol2str[pol]][ant] for ant in ants]
                             for pol in pols]).swapaxes(0, 3).swapaxes(0, 1)
        if flags:
            flgarray = np.array([[flags[pol2str[pol]][ant] for ant in ants]
                                 for pol in pols]).swapaxes(0, 3).swapaxes(0, 1)
        else:
            if DELAY:
                flgarray = np.zeros((len(ants), nfreqs, ntimes, npol), dtype=bool)
            else:
                # dont need to swap since datarray alread same shape
                flgarray = np.zeros(datarray.shape, dtype=bool)
        # do the same for the chisquare, which is the same shape as the data
        try:
            chisqarray = np.array([[meta['chisq' + str(ant) + pol2str[pol]]
                                    for ant in ants] for pol in pols]).swapaxes(0, 3).swapaxes(0, 1)
        except:
            chisqarray = np.ones(datarray.shape, dtype=bool)
        # get the array-wide chisq, which does not have separate axes for
        # antennas or polarization
        try:
            totchisqarray = np.array(meta['chisq']).swapaxes(0, 1)
            # add a polarization axis until this is fixed properly
            totchisqarray = totchisqarray[:, :, np.newaxis]
            # repeat polarization axis npol times for proper shape
            np.repeat(totchisqarray, npol, axis=-1)
        except:
            # leave it empty
            totchisqarray = None

        tarray = time
        parray = np.array(pols)
        farray = np.array(freq)
        antarray = list(map(int, ants))
        numarray = list(map(int, allants))
        namarray = antnames

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
        self.Nants_telescope = len(allants)  # all antennas in telescope
        self.antenna_names = namarray[:self.Nants_telescope]
        self.antenna_numbers = numarray[:self.Nants_telescope]
        self.ant_array = np.array(antarray[:self.Nants_data])
        self.Nspws = nspw
        # XXX: needs to change when we support more than 1 spw!
        self.spw_array = np.array([0])
        self.freq_array = farray[:self.Nfreqs].reshape(self.Nspws, -1)
        self.channel_width = np.diff(self.freq_array)[0][0]
        self.jones_array = parray[:self.Njones]
        self.time_array = tarray[:self.Ntimes]
        self.integration_time = meta['inttime']
        self.gain_convention = 'divide'
        self.x_orientation = 'east'
        self.time_range = [self.time_array[0], self.time_array[-1]]
        self.freq_range = [self.freq_array[0][0], self.freq_array[0][-1]]
        # adding new axis for the spectral window axis. This is default to 1.
        # This needs to change when support for Nspws>1 in pyuvdata.
        self.quality_array = chisqarray[:, np.newaxis, :, :, :]
        self.flag_array = flgarray.astype(np.bool)[:, np.newaxis, :, :, :]
        if DELAY:
            self.set_delay()
            self.delay_array = datarray[:, np.newaxis, :, :, :]  # units of seconds
        else:
            self.set_gain()
            self.gain_array = datarray[:, np.newaxis, :, :, :]
        if totchisqarray is not None:
            self.total_quality_array = totchisqarray[np.newaxis, :, :, :]

# omni_run and omni_apply helper functions
def getPol(fname):
    # XXX assumes file naming format
    # extract just the filename if we're passed a path with periods in it
    fn = re.findall('zen\.\d{7}\.\d{5}\..*', fname)[0]
    return fn.split('.')[3]


def isLinPol(polstr):
    return len(list(set(polstr))) == 1


def file2djd(fname):
    return re.findall("\d{7}\.\d{5}", fname)[0]


def get_optionParser(methodName):
    methods = ['omni_run', 'omni_apply']  # XXX TODO: include "firstcal_run"
    try:
        assert(methodName in methods)
    except:
        raise AssertionError('methodName must be one of %s' %
                             (','.join(methods)))

    o = optparse.OptionParser()

    if methodName == 'omni_run':
        cal = True
        median_help_string = 'Take the median over time of the starting calibration gains (e.g. firstcal).'
        o.set_usage(
            "omni_run.py -C [calfile] -p [pol] --firstcal=[firstcal path] [options] *.uvc")
    elif methodName == 'omni_apply':
        cal = False
        median_help_string = 'Take the median in time before applying solution. Applicable only in delay.'
        o.set_usage(
            "omni_apply.py -p [pol] --omnipath=[/path/to/omni.calfits] --extension=[extension] [options] *.uvc")

    aipy.scripting.add_standard_options(o, cal=cal, pol=True)
    o.add_option('--omnipath', dest='omnipath', default='.',
                 type='string', help='Path to/for omnical solutions.')
    o.add_option('--median', action='store_true', help=median_help_string)

    if methodName == 'omni_run':
        o.add_option('--ex_ants', dest='ex_ants', default=None,
                     help='Antennas to exclude, separated by commas.')
        o.add_option('--firstcal', dest='firstcal', type='string',
                     help='Path and name of firstcal file. Can pass in wildcards.')
        o.add_option('--minV', action='store_true',
                     help='Toggle V minimization capability. This only makes sense in the case of 4-pol cal, which will set crosspols (xy & yx) equal to each other')

    elif methodName == 'omni_apply':
        o.add_option('--firstcal', action='store_true',
                     help='Applying firstcal solutions.')
        o.add_option('--extension', dest='extension', default='O', type='string',
                     help='Filename extension to be appended to the input filename')

    return o


def omni_run(files, opts, history):
    pols = opts.pol.split(',')

    if len(files) == 0:
        raise AssertionError('Please provide visibility files.')
    if opts.minV and len(list(set(''.join(pols)))) == 1:
        raise AssertionError(
            'Stokes V minimization requires crosspols in the "-p" option.')

    linear_pol_keys = []
    for pp in pols:
        if isLinPol(pp):
            linear_pol_keys.append(pp)

    # Create info
    # generate reds from calfile
    aa = aipy.cal.get_aa(opts.cal, np.array([.15]))
    print('Getting reds from calfile')
    if opts.ex_ants:
        # assumes exclusion of the same antennas for every pol
        ex_ants = map(int, opts.ex_ants.split(','))
        print('   Excluding antennas:', sorted(ex_ants))
    else:
        ex_ants = []
    info = aa_to_info(aa, pols=list(set(''.join(pols))),
                      ex_ants=ex_ants, crosspols=pols, minV=opts.minV)
    reds = info.get_reds()
    bls = [bl for red in reds for bl in red]

    ### Collect all firstcal files ###
    firstcal_files = {}
    if not opts.firstcal:
        raise ValueError('Please provide a firstcal file. Exiting...')
    # XXX this requires a firstcal file for any implementation

    Nf = 0
    for pp in pols:
        if isLinPol(pp):
            # we cannot use cross-pols to firstcal
            if '*' in opts.firstcal or '?' in opts.firstcal:
                flist = glob.glob(opts.firstcal)
            elif ',' in opts.firstcal:
                flist = opts.firstcal.split(',')
            else:
                flist = [str(opts.firstcal)]
            firstcal_files[pp] = sorted([s for s in flist if pp in s])
            Nf += len(firstcal_files[pp])

    ### Match firstcal files according to mode of calibration ###
    filesByPol = {}
    for pp in pols:
        filesByPol[pp] = []
    file2firstcal = {}

    for f, filename in enumerate(files):
        if Nf == len(files) * len(pols):
            fi = f  # atomic firstcal application
        else:
            fi = 0  # one firstcal file serves all visibility files
        pp = getPol(filename)
        if isLinPol(pp):
            file2firstcal[filename] = [firstcal_files[pp][fi]]
        else:
            file2firstcal[filename] = [firstcal_files[lpk][fi]
                                       for lpk in linear_pol_keys]
        filesByPol[pp].append(filename)

    # XXX can these be combined into one loop?

    ### Execute Omnical stages ###
    for filenumber in range(len(files) // len(pols)):
        file_group = {}  # there is one file_group per djd
        for pp in pols:
            file_group[pp] = filesByPol[pp][filenumber]
        if len(pols) == 1:
            bname = os.path.basename(file_group[pols[0]])
        else:
            bname = os.path.basename(
                file_group[pols[0]]).replace('.%s' % pols[0], '')
        fitsname = '%s/%s.omni.calfits' % (opts.omnipath, bname)

        if os.path.exists(fitsname):
            print('   %s exists. Skipping...' % fitsname)
            continue

        # get correct firstcal files
        # XXX not a fan of the way this is done, open to suggestions
        fcalfile = None
        if len(pols) == 1:  # single pol
            fcalfile = file2firstcal[file_group[pols[0]]]
        else:
            for pp in pols:  # 4 pol
                if pp not in linear_pol_keys:
                    fcalfile = file2firstcal[file_group[pp]]
                    break
        if not fcalfile:  # 2 pol
            fcalfile = [file2firstcal[file_group[pp]][0]
                        for pp in linear_pol_keys]

        _, g0, _, _ = from_fits(fcalfile)

        #uvd = pyuvdata.UVData()
        #uvd.read_miriad([file_group[pp] for pp in pols])
        # XXX This will become much simpler when pyuvdata can read multiple MIRIAD
        # files at once.

        # collect metadata -- should be the same for each file
        f0 = file_group[pols[0]]
        uvd = UVData()
        uvd.read_miriad(f0)
        if uvd.phase_type != 'drift':
            uvd.phase_to_drift()
        t_jd = uvd.time_array.reshape(uvd.Ntimes, uvd.Nbls)[:, 0]
        t_lst = uvd.lst_array.reshape(uvd.Ntimes, uvd.Nbls)[:, 0]
        t_int = uvd.integration_time
        freqs = uvd.freq_array[0]
        # shape of file data (ex: (19,203))
        SH = (uvd.Ntimes, uvd.Nfreqs)

        uvd_dict = {}
        for pp in pols:
            uvd = UVData()
            uvd.read_miriad(file_group[pp])
            if uvd.phase_type != 'drift':
                uvd.unphase_to_drift()
            uvd_dict[pp] = uvd

        # format g0 for application to data
        if opts.median:
            for p in g0.keys():
                for i in g0[p]:
                    # take median along time axis and resize to shape of data.
                    g0[p][i] = np.resize(np.median(g0[p][i], axis=0), SH)

        # read data into dictionaries
        d, f = {}, {}
        for ip, pp in enumerate(pols):
            uvdp = uvd_dict[pp]

            if ip == 0:
                for nbl, (i, j) in enumerate(map(uvdp.baseline_to_antnums, uvdp.baseline_array[:uvdp.Nbls])):
                    if not (i, j) in bls and not (j, i) in bls:
                        continue
                    d[(i, j)] = {}
                    f[(i, j)] = {}

            # XXX I *really* don't like looping again, but I'm not sure how better
            # to do it
            for nbl, (i, j) in enumerate(map(uvdp.baseline_to_antnums, uvdp.baseline_array[:uvdp.Nbls])):
                if not (i, j) in bls and not (j, i) in bls:
                    continue
                d[(i, j)][pp] = uvdp.data_array.reshape(uvdp.Ntimes, uvdp.Nbls,
                                                        uvdp.Nspws, uvdp.Nfreqs, uvdp.Npols)[:, nbl, 0, :, 0]
                f[(i, j)][pp] = np.logical_not(uvdp.flag_array.reshape(
                    uvdp.Ntimes, uvdp.Nbls, uvdp.Nspws, uvdp.Nfreqs, uvdp.Npols)[:, nbl, 0, :, 0])

        # Finally prepared to run omnical
        print('   Running Omnical')
        m2, g3, v3 = run_omnical(d, info, gains0=g0)

        # Collect weights for xtalk
        wgts, xtalk = {}, {}
        for pp in pols:
            wgts[pp] = {}  # weights dictionary by pol
            for i, j in f:
                if (i, j) in bls:
                    wgts[pp][(i, j)] = np.logical_not(
                        f[i, j][pp]).astype(np.int)
                else:  # conjugate
                    wgts[pp][(j, i)] = np.logical_not(
                        f[i, j][pp]).astype(np.int)
        # xtalk is time-average of residual: data - omnical model
        xtalk = compute_xtalk(m2['res'], wgts)

        # Append metadata parameters
        m2['history'] = 'OMNI_RUN: ' + history + '\n'
        m2['times'] = t_jd
        m2['lsts'] = t_lst
        m2['freqs'] = freqs
        m2['inttime'] = t_int
        optional = {'observer': 'hera_cal'}

        print('   Saving %s' % fitsname)
        hc = HERACal(m2, g3, ex_ants=ex_ants,  optional=optional)
        hc.write_calfits(fitsname)
        fsj = '.'.join(fitsname.split('.')[:-2])
        uv_vis = make_uvdata_vis(aa, m2, v3)
        uv_vis.reorder_pols()
        uv_vis.write_uvfits('%s.vis.uvfits' %
                            fsj, force_phase=True, spoof_nonessential=True)
        uv_xtalk = make_uvdata_vis(aa, m2, xtalk, xtalk=True)
        uv_xtalk.reorder_pols()
        uv_xtalk.write_uvfits('%s.xtalk.uvfits' %
                              fsj, force_phase=True, spoof_nonessential=True)

    return


def omni_apply(files, opts):
    pols = opts.pol.split(',')
    linear_pol_keys = []
    for pp in pols:
        if isLinPol(pp):
            linear_pol_keys.append(pp)

    filedict = {}
    solution_files = sorted(glob.glob(opts.omnipath))

    if opts.firstcal:
        firstcal_files = {}
        nf = 0
        for pp in pols:
            if isLinPol(pp):
                firstcal_files[pp] = sorted(
                    [s for s in glob.glob(opts.omnipath) if pp in s])
                nf += len(firstcal_files[pp])

    for i, f in enumerate(files):
        pp = getPol(f)
        djd = file2djd(f)
        if not opts.firstcal:
            if len(pols) == 1:
                # atomic solution application
                # XXX this is fragile
                fexpected = solution_files[i]
            else:
                # one solution file per djd
                fexpected = next((s for s in solution_files if djd in s), None)
            try:
                ind = solution_files.index(fexpected)
                filedict[f] = str(solution_files[ind])
            except ValueError:
                raise Exception(
                    'Solution file %s expected; not found.' % fexpected)

        else:
            if nf == len(solution_files) * len(pols):  # atomic firstcal application
                filedict[f] = solution_files[i]  # XXX this is fragile
            else:  # one firstcal file for many data files
                if isLinPol(pp):
                    filedict[f] = [firstcal_files[pp][0]]
                else:
                    filedict[f] = [firstcal_files[lpk][0]
                                   for lpk in linear_pol_keys]

    for f in files:
        mir = UVData()
        print("  Reading {0}".format(f))
        mir.read_miriad(f)
        if mir.phase_type != 'drift':
            mir.unphase_to_drift()
        cal = UVCal()
        print("  Reading calibration : {0}".format(filedict[f]))
        if len(pols) == 1 or not opts.firstcal:
            cal.read_calfits(filedict[f])
        else:
            if isLinPol(getPol(f)):
                cal.read_calfits(filedict[f][0])
            else:
                cal = concatenate_UVCal_on_pol(filedict[f])

        print("  Calibrating...")
        antenna_index = dict(zip(*(cal.ant_array, range(cal.Nants_data))))
        for p, pol in enumerate(mir.polarization_array):
            # XXX could replace with numpy function instead of casting to list
            p1, p2 = [list(cal.jones_array).index(pk)
                      for pk in jonesLookup[pol]]
            for bl, k in zip(*np.unique(mir.baseline_array, return_index=True)):
                blmask = np.where(mir.baseline_array == bl)[0]
                ai, aj = mir.baseline_to_antnums(bl)
                if not ai in cal.ant_array or not aj in cal.ant_array:
                    continue
                for nsp, nspws in enumerate(mir.spw_array):
                    if cal.cal_type == 'gain' and cal.gain_convention == 'multiply':
                        mir.data_array[blmask, nsp, :, p] = \
                            mir.data_array[blmask, nsp, :, p] * \
                            cal.gain_array[antenna_index[ai], nsp, :, :, p1].T * \
                            np.conj(cal.gain_array[antenna_index[aj], nsp, :, :, p2].T)

                    if cal.cal_type == 'gain' and cal.gain_convention == 'divide':
                        mir.data_array[blmask, nsp, :, p] =  \
                            mir.data_array[blmask, nsp, :, p] / \
                            cal.gain_array[antenna_index[ai], nsp, :, :, p1].T / \
                            np.conj(cal.gain_array[antenna_index[aj], nsp, :, :, p2].T)

                    if cal.cal_type == 'delay' and cal.gain_convention == 'multiply':
                        if opts.median:
                            mir.data_array[blmask, nsp, :, p] =  \
                                mir.data_array[blmask, nsp, :, p] * \
                                get_phase(cal.freq_array, np.median(cal.delay_array[antenna_index[ai], nsp, 0, :, p1])).reshape(1, -1) * \
                                np.conj(get_phase(cal.freq_array, np.median(
                                    cal.delay_array[antenna_index[aj], nsp, 0, :, p2])).reshape(1, -1))
                        else:
                            mir.data_array[blmask, nsp, :, p] =  \
                                mir.data_array[blmask, nsp, :, p] * \
                                get_phase(cal.freq_array, cal.delay_array[antenna_index[ai], nsp, 0, :, p1]) * \
                                np.conj(get_phase(
                                    cal.freq_array, cal.delay_array[antenna_index[aj], nsp, 0, :, p2]))

                    if cal.cal_type == 'delay' and cal.gain_convention == 'divide':
                        if opts.median:
                            mir.data_array[blmask, nsp, :, p] =  \
                                mir.data_array[blmask, nsp, :, p] / \
                                get_phase(cal.freq_array, np.median(cal.delay_array[antenna_index[ai], nsp, 0, :, p1])).reshape(1, -1) / \
                                np.conj(get_phase(cal.freq_array, np.median(
                                    cal.delay_array[antenna_index[aj], nsp, 0, :, p2])).reshape(1, -1))
                        else:
                            mir.data_array[blmask, nsp, :, p] =  \
                                mir.data_array[blmask, nsp, :, p] / \
                                get_phase(cal.freq_array, cal.delay_array[antenna_index[ai], nsp, 0, :, p1]).T / \
                                np.conj(get_phase(
                                    cal.freq_array, cal.delay_array[antenna_index[aj], nsp, 0, :, p2]).T)

                    # Update miriad flags array
                    mir.flag_array[blmask, nsp, :, p] = np.logical_or(
                        mir.flag_array[blmask, nsp, :, p],
                        np.logical_or(cal.flag_array[antenna_index[ai], nsp, :, :, p1].T,
                                      cal.flag_array[antenna_index[aj], nsp, :, :, p2].T))

        if opts.firstcal:
            print(" Writing {0}".format(f + 'F'))
            mir.write_miriad(f + 'F')
        else:
            print(" Writing {0}".format(f + opts.extension))
            mir.write_miriad(f + opts.extension)

    return

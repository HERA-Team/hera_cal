# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import numpy as np
import os
import copy
import astropy.constants as const
from astropy.time import Time
from astropy import coordinates as crd
from astropy import units as unt
from scipy import signal, interpolate
import pyuvdata.utils as uvutils
from pyuvdata import UVCal, UVData
from pyuvdata.utils import polnum2str, polstr2num, jnum2str, jstr2num, conj_pol
from pyuvdata.utils import POL_STR2NUM_DICT, JONES_STR2NUM_DICT, JONES_NUM2STR_DICT, _x_orientation_rep_dict
import sklearn.gaussian_process as gp
import warnings
import argparse
import inspect
from . import __version__

try:
    AIPY = True
    import aipy
except ImportError:
    AIPY = False


# Taken from a H1C data file and confirmed to match a H3C data file
HERA_TELESCOPE_LOCATION = np.array([5109325.855210627429187297821044921875,
                                    2005235.091429826803505420684814453125,
                                    -3239928.424753960222005844116210937500])

# Defines characters to look for to see if the polarization string is in east/north format. Nominally {'e', 'n'}.
_KEY_CARDINAL_CHARS = set([c.lower() for c in _x_orientation_rep_dict('north').values()])
# Define characters that appear in all Jones polarization strings. Nominally {'j'}.
_KEY_JONES_CHARS = set([c.lower() for val in JONES_NUM2STR_DICT.values() for c in val
                       if np.all([c in v for v in JONES_NUM2STR_DICT.values()])])
# set of characters that appear in either _KEY_CARDINAL_CHARS or _KEY_JONES_CHARS
_KEY_POL_CHARS = _KEY_CARDINAL_CHARS | _KEY_JONES_CHARS

# Get set of standard visibility polarization strings. Nominally {'rr', 'xy', 'ne', 'nn', 'lr', 'en', 'yx', 'xx', 'ee', 'rl', 'll', 'yy'}
_VISPOLS = set([pol for pol in list(POL_STR2NUM_DICT.keys()) if polstr2num(pol) < 0 and pol == polnum2str(polstr2num(pol))])
# Add east/north polarizations to _VISPOLS while relying only on pyuvdata definitions
for pol in copy.deepcopy(_VISPOLS):
    try:
        _VISPOLS.add(polnum2str(polstr2num(pol), x_orientation='north'))
    except KeyError:
        pass

# Get set of standard antenna polarization strings. Nominally {'Jee', 'Jen', 'Jll', 'Jlr', 'Jne', 'Jnn', 'Jrl', 'Jrr', 'Jxx', 'Jxy', 'Jyx', 'Jyy'}
_ANTPOLS = set([antpol for antpol in list(JONES_STR2NUM_DICT.keys()) if jstr2num(antpol) < 0 and antpol == jnum2str(jstr2num(antpol))])
# Add east/north polarizations to _ANTPOLS while relying only on pyuvdata definitions
for antpol in copy.deepcopy(_ANTPOLS):
    try:
        _ANTPOLS.add(jnum2str(jstr2num(antpol), x_orientation='north'))
    except KeyError:
        pass


def _is_cardinal(polstr):
    '''Returns true if all characters in polstr match those in _KEY_CARDINAL_CHARS or _KEY_JONES_CHARS.'''
    return set(polstr.lower()).issubset(_KEY_POL_CHARS)


def _comply_antpol(antpol):
    '''Maps an input antenna polarization string onto a string compliant with pyuvdata
    and hera_cal.'''
    if antpol in _ANTPOLS:
        return antpol
    if _is_cardinal(antpol):
        return jnum2str(jstr2num(antpol, x_orientation='north'), x_orientation='north')
    else:
        return jnum2str(jstr2num(antpol))


def _comply_vispol(pol):
    '''Maps an input visibility polarization string onto a string compliant with pyuvdata
    and hera_cal.'''
    if pol in _VISPOLS:
        return pol
    elif _is_cardinal(pol):
        return polnum2str(polstr2num(pol, x_orientation='north'), x_orientation='north')
    else:
        return polnum2str(polstr2num(pol))


SPLIT_POL = {pol: (_comply_antpol(pol[0]), _comply_antpol(pol[1])) for pol in _VISPOLS}


def split_pol(pol):
    '''Splits visibility polarization string (pyuvdata's polstr) into
    antenna polarization strings (pyuvdata's jstr).'''
    return SPLIT_POL[_comply_vispol(pol)]


JOIN_POL = {v: k for k, v in SPLIT_POL.items()}


def join_pol(p1, p2):
    '''Joins antenna polarization strings (pyuvdata's jstr) into
    visibility polarization string (pyuvdata's polstr).'''
    return JOIN_POL[(_comply_antpol(p1),
                     _comply_antpol(p2))]


def comply_pol(pol):
    '''Maps an input (visibility or antenna) polarization string onto a string
    compliant with pyuvdata and hera_cal.'''
    try:
        return _comply_vispol(pol)
    except(KeyError):  # happens if we have an antpol, not vispol
        return _comply_antpol(pol)


def split_bl(bl):
    '''Splits a (i,j,pol) baseline key into ((i,pi),(j,pj)), where pol=pi+pj.'''
    pi, pj = split_pol(bl[2])
    return ((bl[0], pi), (bl[1], pj))


def join_bl(ai, aj):
    '''Joins two (i,pi) antenna keys to make a (i,j,pol) baseline key.'''
    return (ai[0], aj[0], join_pol(ai[1], aj[1]))


def reverse_bl(bl):
    '''Reverses a (i,j) or (i,j,pol) baseline key to make (j,i)
    or (j,i,pol[::-1]), respectively.'''
    i, j = bl[:2]
    if len(bl) == 2:
        return (j, i)
    else:
        return (j, i, conj_pol(_comply_vispol(bl[2])))


def comply_bl(bl):
    '''Translates an input (i,j,pol) baseline to ensure pol is compliant with
    pyuvdata and hera_cal. Inputs of length 2, e.g. (i,j) are unmodified.'''
    if len(bl) == 2:
        return bl
    else:
        i, j, p = bl
        return (i, j, _comply_vispol(p))


def make_bl(*args):
    '''Create an (i,j,pol) baseline key that is compliant with pyuvdata
    and hera_cal.  Accepts (bl, pol) or (i, j, pol) as input.'''
    if len(args) == 1:
        args = args[0]  # this handles the case where the input is a tuple
    if len(args) == 2:
        (i, j), pol = args
    else:
        i, j, pol = args
    return (i, j, _comply_vispol(pol))


def filter_bls(bls, ants=None, ex_ants=None, pols=None, antpos=None, min_bl_cut=None, max_bl_cut=None):
    '''Filter a list of baseline tuples on one of more of the following criteria.

    Arguments:
        bls: list of baseline tuples like (0, 1, 'ee')
        ants: list of integer antenna indices or antenna tuples like (0, 'Jee') to keep. None means no filter.
        ex_ants: list of integer antenna indices or antenna tuples like (0, 'Jee') to exclude. None means no filter.
        pols: list of visibility polarizations like 'ee' to keep. None means no filter.
        antpos: dictionary mapping antenna indicies to ENU position arrays. Required if min_bl_cut or max_bl_cut
            is specified.
        min_bl_cut: minimum length of baselines to keep in the same units as antenna postions. None means no filter.
        max_bl_cut: maximum length of baselines to keep in the same units as antenna postions. None means no filter.

    Returns:
        filtered_bls: list of baseline tuples like (0, 1, 'ee'), a subset of bls
    '''
    filtered_bls = []

    if (min_bl_cut is not None) or (max_bl_cut is not None):
        assert antpos is not None, 'antpos must be passed in if min_bl_cut or max_bl_cut is specified.'

    for bl in bls:
        ant1, ant2 = split_bl(bl)
        # filter on antennas to keep
        if (ants is not None) and (ant1 not in ants) and (ant1[0] not in ants):
            continue
        if (ants is not None) and (ant2 not in ants) and (ant2[0] not in ants):
            continue

        # filter on antennas to exclude
        if (ex_ants is not None) and ((ant1 in ex_ants) or (ant1[0] in ex_ants)):
            continue
        if (ex_ants is not None) and ((ant2 in ex_ants) or (ant2[0] in ex_ants)):
            continue

        # filter on polarizations
        if (pols is not None) and (bl[2] not in pols):
            continue

        # filter on baseline length
        if antpos is not None:
            bl_length = np.linalg.norm(antpos[ant2[0]] - antpos[ant1[0]])
            if (min_bl_cut is not None) and (bl_length < min_bl_cut):
                continue
            if (max_bl_cut is not None) and (bl_length > max_bl_cut):
                continue

        filtered_bls.append(bl)

    return filtered_bls


def history_string(notes=""):
    """
    Creates a standardized history string that all functions that write to
    disk can use. Optionally add notes.
    """
    notes = f"""\n\nNotes:\n{notes}""" if notes else ""

    stack = inspect.stack()[1]
    history = f"""
    ------------
    This file was produced by the function {stack[3]}() in {stack[1]} using version {__version__}
    {notes}
    ------------
    """
    return history


# XXX change df/ f0 for freqs directly
def fft_dly(data, df, wgts=None, f0=0.0, medfilt=False, kernel=(1, 11), edge_cut=0):
    """Get delay of visibility across band using FFT and Quinn's Second Method to fit the delay and phase offset.
    Arguments:
        data : ndarray of complex data (e.g. gains or visibilities) of shape (Ntimes, Nfreqs)
        df : frequency channel width in Hz
        wgts : multiplicative wgts of the same shape as the data
        f0 : float lowest frequency channel. Optional parameter used in getting the offset correct.
        medfilt : boolean, median filter data before fft
        kernel : size of median filter kernel along (time, freq) axes
        edge_cut : int, number of channels to exclude at each band edge of data in FFT window
    Returns:
        dlys : (Ntimes, 1) ndarray containing delay for each integration
        offset : (Ntimes, 1) ndarray containing estimated frequency-independent phases
    """
    # setup
    Ntimes, Nfreqs = data.shape
    # XXX prefer saving compute by not using wgts at all
    if wgts is None:
        wgts = np.ones_like(data, dtype=np.float32)

    # smooth via median filter
    # XXX is this mode ever used?
    if medfilt:
        data = copy.deepcopy(data)  # this prevents filtering of the original input data
        data.real = signal.medfilt(data.real, kernel_size=kernel)
        data.imag = signal.medfilt(data.imag, kernel_size=kernel)

    # fft w/ wgts
    dw = data * wgts
    # XXX is edge_cut ever used?
    if edge_cut > 0:
        assert 2 * edge_cut < Nfreqs - 1, "edge_cut cannot be >= Nfreqs/2 - 1"
        dw = dw[:, edge_cut:(-edge_cut + 1)]
    dw[np.isnan(dw)] = 0
    fftfreqs = np.fft.fftfreq(dw.shape[1], df)
    dtau = fftfreqs[1] - fftfreqs[0]
    vfft = np.fft.fft(dw, axis=1)

    # get interpolated peak and indices
    inds, bin_shifts, peaks, interp_peaks = interp_peak(vfft)
    dlys = (fftfreqs[inds] + bin_shifts * dtau).reshape(-1, 1)

    # XXX if not using phase offset, should have mode to not compute it?
    # Now that we know the slope, estimate the remaining phase offset
    freqs = np.arange(Nfreqs, dtype=data.dtype) * df + f0
    fSlice = slice(edge_cut, len(freqs) - edge_cut)
    offset = np.angle(
        np.sum(
            wgts[:, fSlice] * data[:, fSlice] * np.exp(
                -np.complex64(2j * np.pi) * dlys * freqs[fSlice].reshape(1, -1)
            ),
            axis=1, keepdims=True
        ) / np.sum(wgts[:, fSlice], axis=1, keepdims=True)
    )

    return dlys, offset


def quinn_tau(x):
    '''Quinn subgrid interpolation parameter (see https://ieeexplore.ieee.org/document/558515)'''
    t = .25 * np.log(3 * x**2 + 6 * x + 1)
    t -= 6**.5 / 24 * np.log((x + 1 - (2 / 3)**.5) / (x + 1 + (2 / 3)**.5))
    return t


def interp_peak(data, method='quinn', reject_edges=False):
    """
    Spectral interpolation for finding peak and amplitude of data along last axis.

    Args:
        data : complex 2d ndarray in Fourier space.
            If fed as 1d array will reshape into [1, N] array.
            Quinn's method usually operates on complex data (eg. fft'ed data) while the
            quadratic method operates on real-valued data (generally absolute values).
        method : either 'quinn' (see https://ieeexplore.ieee.org/document/558515) or 'quadratic'
            (see https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html).
        reject_edges : bool, if True, reject solution if it isn't a true "peak", in other words
            if it is along the axis edges

    Returns:
        indices : index array holding argmax of data along last axis
        bin_shifts : estimated peak bin shift value [-1, 1] from indices
        peaks : argmax of data corresponding to indices
        new_peaks : estimated peak value at indices + bin_shifts
    """
    # get properties
    if data.ndim == 1:
        data = data[None, :]
    N1, N2 = data.shape

    # get abs
    dabs = np.abs(data)

    # ensure edge cases are handled is requested
    if reject_edges:
        # scroll through diffs and set monotonically decreasing edges to zero
        forw_diff = dabs[:, 1:] - dabs[:, :-1]
        for i, fd in enumerate(forw_diff):
            ncut = np.argmax(fd > 0)
            if ncut > 0:
                dabs[i, :ncut] = 0.0
            ncut = N2 - np.argmax(fd[::-1] < 0)
            if ncut > 0:
                dabs[i, ncut:] = 0.0

    # get argmaxes along last axis
    if method not in ('quinn', 'quadratic'):
        raise ValueError("'{}' is not a recognized peak interpolation method.".format(method))

    indices = np.argmax(dabs, axis=-1)
    peaks = data[range(N1), indices]

    # calculate shifted peak for sub-bin resolution
    k0 = data[range(N1), indices - 1]
    k1 = data[range(N1), indices]
    k2 = data[range(N1), (indices + 1) % N2]

    if method == 'quinn':
        alpha1 = (k0 / k1).real
        alpha2 = (k2 / k1).real
        delta1 = alpha1 / (1 - alpha1)
        delta2 = -alpha2 / (1 - alpha2)
        d = (delta1 + delta2) / 2 + quinn_tau(delta1 ** 2) - quinn_tau(delta2 ** 2)
        d[~np.isfinite(d)] = 0.

        numerator_ck = np.exp(2.0j * np.pi * d) - 1
        ck = np.array([np.true_divide(numerator_ck, 2.0j * np.pi * (d - k),
                                      where=~(d == 0)) for k in [-1, 0, 1]])
        rho = np.abs(k0 * ck[0] + k1 * ck[1] + k2 * ck[2]) / np.abs(np.sum(ck ** 2))
        rho[d == 0] = np.abs(k1[d == 0])
        return indices, d, np.abs(peaks), rho

    elif method == 'quadratic':
        denom = (k0 - 2 * k1 + k2)
        bin_shifts = 0.5 * np.true_divide((k0 - k2), denom, where=~np.isclose(denom, 0.0))
        new_peaks = k1 - 0.25 * (k0 - k2) * bin_shifts
        return indices, bin_shifts, peaks, new_peaks


def echo(message, type=0, verbose=True):
    if verbose:
        if type == 0:
            print(message)
        elif type == 1:
            print('')
            print(message)
            print("-" * 40)


if AIPY:
    class AntennaArray(aipy.pol.AntennaArray):
        def __init__(self, *args, **kwargs):
            aipy.pol.AntennaArray.__init__(self, *args, **kwargs)
            self.antpos_ideal = kwargs.pop('antpos_ideal')
            # yes, this is a thing. cm per meter
            self.cm_p_m = 100.

        def update(self):
            aipy.pol.AntennaArray.update(self)

        def get_params(self, ant_prms={'*': '*'}):
            try:
                prms = aipy.pol.AntennaArray.get_params(self, ant_prms)
            except(IndexError):
                return {}
            return prms

        def set_params(self, prms):
            changed = aipy.pol.AntennaArray.set_params(self, prms)
            for i, ant in enumerate(self):
                ant_changed = False
                top_pos = np.dot(self._eq2zen, ant.pos)
                try:
                    top_pos[0] = prms[str(i)]['top_x']
                    ant_changed = True
                except(KeyError):
                    pass
                try:
                    top_pos[1] = prms[str(i)]['top_y']
                    ant_changed = True
                except(KeyError):
                    pass
                try:
                    top_pos[2] = prms[str(i)]['top_z']
                    ant_changed = True
                except(KeyError):
                    pass
                if ant_changed:
                    # rotate from zenith to equatorial, convert from meters to ns
                    ant.pos = np.dot(np.linalg.inv(self._eq2zen), top_pos) / aipy.const.len_ns * self.cm_p_m
                changed |= ant_changed
            if changed:
                self.update()
            return changed


def get_aa_from_uv(uvd, freqs=[0.15]):
    '''
    Generate an AntennaArray object from a pyuvdata UVData object.

    This function creates an AntennaArray object from the metadata
    contained in a UVData object. It assumes that the antenna_positions
    array in the UVData object is in earth-centered, earth-fixed (ECEF)
    coordinates, relative to the center of the array, also given in
    ECEF coordinates. We must add these together, and then rotate so that
    the x-axis is aligned with the local meridian (rotECEF). rotECEF is the
    coordinate system for Antenna objects in the AntennaArray object (which
    inherits this behavior from MIRIAD). It is also expected that distances
    are given in nanoseconds, rather than meters, also because of the
    default behavior in MIRIAD.

    Arguments
    =================
    uvd: a pyuvdata UVData object containing the data.
    freqs (optional): list of frequencies to pass to aa object. Defaults to single frequency
        (150 MHz), suitable for computing redundancy and uvw info.

    Returns
    ====================
    aa: AntennaArray object that can be used to calculate redundancies from
       antenna positions.
    '''
    assert AIPY, "you need aipy to run this function"
    # center of array values from file
    cofa_lat, cofa_lon, cofa_alt = uvd.telescope_location_lat_lon_alt
    location = (cofa_lat, cofa_lon, cofa_alt)

    # get antenna positions from file
    antpos = {}
    for i, antnum in enumerate(uvd.antenna_numbers):
        # we need to add the CofA location to the relative coordinates
        pos = uvd.antenna_positions[i, :] + uvd.telescope_location
        # convert from meters to nanoseconds
        c_ns = const.c.to('m/ns').value
        pos = pos / c_ns

        # rotate from ECEF -> rotECEF
        rotECEF = uvutils.rotECEF_from_ECEF(pos, cofa_lon)

        # make a dict for parameter-setting purposes later
        antpos[antnum] = {'x': rotECEF[0], 'y': rotECEF[1], 'z': rotECEF[2]}

    # make antpos_ideal array
    nants = np.max(list(antpos.keys())) + 1
    antpos_ideal = np.zeros(shape=(nants, 3), dtype=float) - 1
    # unpack from dict -> numpy array
    for k in list(antpos.keys()):
        antpos_ideal[k, :] = np.array([antpos[k]['x'], antpos[k]['y'], antpos[k]['z']])
    freqs = np.asarray(freqs)
    # Make list of antennas.
    # These are values for a zenith-pointing antenna, with a dummy Gaussian beam.
    antennas = []
    for i in range(nants):
        beam = aipy.fit.Beam(freqs)
        phsoff = {'x': [0., 0.], 'y': [0., 0.]}
        amp = 1.
        amp = {'x': amp, 'y': amp}
        bp_r = [1.]
        bp_r = {'x': bp_r, 'y': bp_r}
        bp_i = [0., 0., 0.]
        bp_i = {'x': bp_i, 'y': bp_i}
        twist = 0.
        antennas.append(aipy.pol.Antenna(0., 0., 0., beam, phsoff=phsoff,
                                         amp=amp, bp_r=bp_r, bp_i=bp_i, pointing=(0., np.pi / 2, twist)))

    # Make the AntennaArray and set position parameters
    aa = AntennaArray(location, antennas, antpos_ideal=antpos_ideal)
    pos_prms = {}
    for i in list(antpos.keys()):
        pos_prms[str(i)] = antpos[i]
    aa.set_params(pos_prms)
    return aa


def JD2LST(JD, latitude=-30.721526120689507, longitude=21.428303826863015, altitude=1051.690000018105):
    """
    Input:
    ------
    JD : type=float or list of floats containing Julian Date(s) of an observation

    latitude : type=float, degrees North of observer, default=HERA latitude

    longitude : type=float, degrees East of observer, default=HERA longitude

    altitude : type=float, altitude in meters, default=HERA altitude

    Output:
    -------
    Local Apparent Sidreal Time [radians]

    Notes:
    ------
    The Local Apparent Sidereal Time is *defined* as the right ascension in the current epoch.
    Wrapper around pyuvdata's get_lst_for_time().
    """
    # get JD type
    if isinstance(JD, list) or isinstance(JD, np.ndarray):
        _array = True
    else:
        _array = False
        JD = [JD]

    # use pyuvdata
    LST = uvutils.get_lst_for_time(np.array(JD), latitude, longitude, altitude)

    if _array:
        return LST
    else:
        return LST[0]


def LST2JD(LST, start_jd, allow_other_jd=False, lst_branch_cut=0.0, latitude=-30.721526120689507,
           longitude=21.428303826863015, altitude=1051.690000018105):
    """
    Convert Local Apparent Sidereal Time -> Julian Date via a linear fit
    at the 'start_JD' anchor point.

    Input:
    ------
    LST : type=float or array-like, local apparent sidereal time [radians]

    start_jd : type=int, integer julian day to use as starting point for LST2JD conversion

    allow_other_jd : type=bool, ensure that the lst_branch_cut falls on start_jd but allow
                     LSTs to correspond to previous or subsequent days if necessary

    lst_branch_cut : type=float, LST that must fall during start_jd even when allow_other_jd
                     is True. Used as the starting point starting point where LSTs below this
                     value map to later JDs than this LST [radians]

    latitude : type=float, degrees North of observer, default=HERA latitude

    longitude : type=float, degrees East of observer, default=HERA longitude

    altitude : type=float, altitude in meters, default=HERA altitude

    Output:
    -------
    JD : type=type(LST), Julian Date(s). accurate to ~1 milliseconds
    """
    # get LST type
    if isinstance(LST, list) or isinstance(LST, np.ndarray):
        LST = np.array(LST)
        _array = True
    else:
        LST = np.array([LST])
        _array = False

    # increase LSTs so no values fall below the branch cut (avoiding wraps in interpolation)
    while np.any(LST < lst_branch_cut):
        LST[LST < lst_branch_cut] += 2 * np.pi

    # create interpolator for a given start date that puts the lst_branch_cut on start_jd
    jd_grid = start_jd + np.linspace(-1, 2, 31)  # include previous and next days
    while True:
        lst_grid = (JD2LST(jd_grid, latitude=latitude, longitude=longitude, altitude=altitude))
        interpolator = interpolate.interp1d(np.unwrap(lst_grid - 2 * np.pi), jd_grid,
                                            kind='linear', fill_value='extrapolate')
        if np.floor(interpolator(lst_branch_cut)) > np.floor(start_jd):
            jd_grid -= unt.sday.to(unt.day)

        elif np.floor(interpolator(lst_branch_cut)) < np.floor(start_jd):
            jd_grid += unt.sday.to(unt.day)
        else:
            break

    # interpolate
    jd_array = interpolator(LST)

    # Enforce single JD if desired
    if not allow_other_jd:
        while True:
            lt_indices = np.floor(jd_array) < np.floor(start_jd)
            gt_indices = np.floor(jd_array) > np.floor(start_jd)
            # if any of the resultant days fall on a JD less than start_jd, recalculate them 2pi higher
            if np.any(lt_indices):
                LST[lt_indices] += 2 * np.pi
                jd_array[lt_indices] = interpolator(LST[lt_indices])
            # if any of the resultant days fall on a JD greater than start_jd, recalculate them 2pi lower
            elif np.any(gt_indices):
                LST[gt_indices] -= 2 * np.pi
                jd_array[gt_indices] = interpolator(LST[gt_indices])
            else:  # all resultant JDs fall on start_jd
                break

    if _array:
        return jd_array
    else:
        return jd_array[0]


def JD2RA(JD, latitude=-30.721526120689507, longitude=21.428303826863015, epoch='current'):
    """
    Convert from Julian date to Equatorial Right Ascension at zenith
    during a specified epoch.

    Parameters:
    -----------
    JD : type=float, a float or an array of Julian Dates

    longitude : type=float, longitude of observer in degrees east, default=HERA longitude

    latitude : type=float, latitude of observer in degrees north, default=HERA latitutde
               This only matters when using epoch="J2000"

    epoch : type=str, epoch for RA calculation. options=['current', 'J2000'].
            The 'current' epoch is the epoch at JD. Note that
            LST is defined as the zenith RA in the current epoch. Note that
            epoch='J2000' corresponds to the ICRS standard.

    Output:
    -------
    RA : type=float, right ascension [degrees] at zenith JD times
         in the specified epoch.
    """
    # get JD type
    if isinstance(JD, list) or isinstance(JD, np.ndarray):
        _array = True
    else:
        _array = False
        JD = [JD]

    # setup RA list
    RA = []

    # iterate over jd
    for jd in JD:

        # use current epoch calculation
        if epoch == 'current':
            ra = JD2LST(jd, longitude=longitude) * 180 / np.pi
            RA.append(ra)

        # use J2000 epoch
        elif epoch == 'J2000':
            loc = crd.EarthLocation(lat=latitude * unt.deg, lon=longitude * unt.deg)
            t = Time(jd, format='jd', scale='utc')
            zen = crd.SkyCoord(frame='altaz', alt=90 * unt.deg, az=0 * unt.deg, obstime=t, location=loc)
            RA.append(zen.icrs.ra.degree)

        else:
            raise ValueError("didn't recognize {} epoch".format(epoch))

    RA = np.array(RA)

    if _array:
        return RA
    else:
        return RA[0]


def get_sun_alt(jds, latitude=-30.721526120689507, longitude=21.428303826863015):
    """
    Given longitude and latitude, get the Solar alittude at a given time.

    Parameters
    ----------
    jds : float or ndarray of floats
        Array of Julian Dates

    longitude : float
        Longitude of observer in degrees East

    latitude : float
        Latitude of observer in degrees North

    Returns
    -------
    alts : float or ndarray
        Array of altitudes [degrees] of the Sun
    """
    # type check
    array = True
    if isinstance(jds, (float, np.floating, np.float64, int, np.integer, np.int32)):
        jds = [jds]
        array = False

    # get earth location
    e = crd.EarthLocation(lat=latitude * unt.deg, lon=longitude * unt.deg)

    # get AltAz frame
    a = crd.AltAz(location=e)

    # get Sun locations
    alts = np.array([crd.get_sun(Time(t, format='jd')).transform_to(a).alt.value for t in jds])

    if array:
        return alts
    else:
        return alts[0]


def combine_calfits(files, fname, outdir=None, overwrite=False, broadcast_flags=True, verbose=True):
    """
    multiply together multiple calfits gain solutions (overlapping in time and frequency)

    Parameters:
    -----------
    files : type=list, dtype=str, list of files to multiply together

    fname : type=str, path to output filename

    outdir : type=str, path to output directory

    overwrite : type=bool, overwrite output file

    broadcast_flags : type=bool, if True, broadcast flags from each calfits to final solution
    """
    # get io params
    if outdir is None:
        outdir = "./"

    output_fname = os.path.join(outdir, fname)
    if os.path.exists(fname) and overwrite is False:
        raise IOError("{} exists, not overwriting".format(output_fname))

    # iterate over files
    for i, f in enumerate(files):
        if i == 0:
            echo("...loading {}".format(f), verbose=verbose)
            uvc = UVCal()
            uvc.read_calfits(f)
            f1 = copy.copy(f)

            # set flagged data to unity
            uvc.gain_array[uvc.flag_array] /= uvc.gain_array[uvc.flag_array]

        else:
            uvc2 = UVCal()
            uvc2.read_calfits(f)

            # set flagged data to unity
            gain_array = uvc2.gain_array
            gain_array[uvc2.flag_array] /= gain_array[uvc2.flag_array]

            # multiply gain solutions in
            uvc.gain_array *= uvc2.gain_array

            # pass flags
            if broadcast_flags:
                uvc.flag_array += uvc2.flag_array
            else:
                uvc.flag_array = uvc.flag_array * uvc2.flag_array

    # write to file
    echo("...saving {}".format(output_fname), verbose=verbose)
    uvc.write_calfits(output_fname, clobber=True)


def lst_rephase(data, bls, freqs, dlst, lat=-30.721526120689507, inplace=True, array=False):
    """
    Shift phase center of each integration in data by amount dlst [radians] along right ascension axis.
    If inplace == True, this function directly edits the arrays in 'data' in memory, so as not to
    make a copy of data.

    Parameters:
    -----------
    data : type=DataContainer, holding 2D visibility data, with [0] axis time and [1] axis frequency

    bls : type=dictionary, same keys as data, values are 3D float arrays holding baseline vector
                            in ENU frame in meters

    freqs : type=ndarray, frequency array of data [Hz]

    dlst : type=ndarray or float, delta-LST to rephase by [radians]. If a float, shift all integrations
                by dlst, elif an ndarray, shift each integration by different amount w/ shape=(Ntimes)

    lat : type=float, latitude of observer in degrees North

    inplace : type=bool, if True edit arrays in data in memory, else make a copy and return

    array : type=bool, if True, treat data as a visibility ndarray and bls as a baseline vector

    Notes:
    ------
    The rephasing uses top2eq_m and eq2top_m matrices (borrowed from pyuvdata and aipy) to convert from
    array TOPO frame to Equatorial frame, induces time rotation, converts back to TOPO frame,
    calculates new pointing vector s_prime and inserts a delay plane into the data for rephasing.

    This method of rephasing follows Eqn. 21 & 22 of Zhang, Y. et al. 2018 "Unlocking Sensitivity..."
    """
    # check format of dlst
    if isinstance(dlst, list):
        lat = np.ones_like(dlst) * lat
        dlst = np.array(dlst)
        zero = np.zeros_like(dlst)
    elif isinstance(dlst, np.ndarray):
        lat = np.ones_like(dlst) * lat
        zero = np.zeros_like(dlst)
    else:
        zero = 0

    # get top2eq matrix
    top2eq = top2eq_m(zero, lat * np.pi / 180)

    # get eq2top matrix
    eq2top = eq2top_m(-dlst, lat * np.pi / 180)

    # get full rotation matrix
    rot = np.einsum("...jk,...kl->...jl", eq2top, top2eq)

    # make copy of data if desired
    if not inplace:
        data = copy.deepcopy(data)

    # turn array into dict
    if array:
        inplace = False
        data = {'data': data}
        bls = {'data': bls}

    # iterate over data keys
    for i, k in enumerate(data.keys()):

        # get new s-hat vector
        s_prime = np.einsum("...ij,j->...i", rot, np.array([0.0, 0.0, 1.0]))
        s_diff = s_prime - np.array([0., 0., 1.0])

        # get baseline vector
        bl = bls[k]

        # dot bl with difference of pointing vectors to get new u: Zhang, Y. et al. 2018 (Eqn. 22)
        u = np.einsum("...i,i->...", s_diff, bl)

        # get delay
        tau = u / const.c.value

        # reshape tau
        if isinstance(tau, np.ndarray):
            pass
        else:
            tau = np.array([tau])

        # get phasor
        phs = np.exp(-2j * np.pi * freqs[None, :] * tau[:, None])

        # multiply into data
        data[k] *= phs

    if array:
        data = data['data']

    if not inplace:
        return data


def chisq(data, model, data_wgts=None, gains=None, gain_flags=None, split_by_antpol=False,
          reds=None, chisq=None, nObs=None, chisq_per_ant=None, nObs_per_ant=None):
    """Computes chi^2 defined as:

    chi^2 = sum_ij(|data_ij - model_ij * g_i conj(g_j)|^2 * wgts_ij)

    and also a chisq_per_antenna which is the same sum but with fixed i. Also keeps track of the
    number of unflagged observations that go into each chi^2 waterfall, both overall and per-antenna.

    Arguments:
        data: dictionary or DataContainer mapping baseline-pol keys like (1,2,'nn') to complex 2D
            visibility data, with [0] axis time and [1] axis frequency.
        model: dictionary or DataContainer mapping baseline-pol keys like (1,2,'nn') to complex 2D
            visibility data, with [0] axis time and [1] axis frequency. Gains are multiplied by model
            before comparing them to the data.
        data_wgts: dictionary or DataContainer mapping baseline-pol keys like (1,2,'nn') to real
            weights with [0] axis time and [1] axis frequency. Weights are interpeted as 1/sigma^2
            where sigma is the noise on the data (but not necessarily the model if gains are provided).
            Flags are be expressed as data_wgts equal to 0, so (~flags) produces binary weights.
            If None, assumed to be all 1.0s with the same keys as data.
        gains: optional dictionary mapping ant-pol keys like (1,'Jnn') to a waterfall of complex gains to
            be multiplied into the model (or, equivalently, divided out from the data). Default: None,
            which is interpreted as all gains are 1.0 (ie..e the data is already calibrated)
        gain_flags: optional dictionary mapping ant-pol keys like (1,'Jnn') to a boolean flags waterfall
            with the same shape as the data. Default: None, which means no per-antenna flagging.
        split_by_antpol: if True, chisq and nObs are dictionaries mapping antenna polarizations to numpy
            arrays. Additionally, if split_by_antpol is True, cross-polarized visibilities are ignored.
        reds: list of lists of redundant baseline tuples, e.g. (ind1,ind2,pol). Requires that the model
            contains visibilities for the first baseline in each redundant group. Any other baselines
            in those redundant groups are overwritten in the model, which is copied not modified.
        chisq: optional chisq to update (see below)
        nObs: optional nObs to update (see below). Must be specified if chisq is specified and must be
            left as None if chisq is left as None.
        chisq_per_ant: optional chisq_per_ant to update (see below)
        nObs_per_ant: optional nObs_per_ant to update (see below). Must have same keys as chisq_per_ant.

    Returns:
        chisq: numpy array with the same shape each visibility of chi^2 calculated as above. If chisq
            is provided, this is the sum of the input chisq and the calculated chisq from all unflagged
            data-to-model comparisons possible given their overlapping baselines. If split_by_antpol is
            True, instead returns a dictionary that maps antenna polarization strings to these numpy arrays.
        nObs: numpy array with the integer number of unflagged data-to-model comparisons that go into
            each time and frequency of the chisq calculation. If nObs is specified, this updates that
            with a count of any new unflagged data-to-model comparisons. If split_by_antpol is True,
            instead returns a dictionary that maps antenna polarization strings to these numpy arrays.
        chisq_per_ant: dictionary mapping ant-pol keys like (1,'Jnn') to chisq per antenna, computed as
            above but keeping i fixed and varying only j. If chisq_per_ant is specified, this adds in
            new chisq calculations that include this antenna
        nObs_per_ant: dictionary mapping ant-pol keys like (1,'Jnn') to the integer number of unflagged
            data-to-model comparisons that go into each time and frequency of the per-antenna chisq
            calculation. If nObs_per_ant, this is updated to include all new unflagged observations.
    """
    # build containers for chisq and nObs if not supplied
    if chisq is None and nObs is None:
        if split_by_antpol:
            chisq = {}
            nObs = {}
        else:
            chisq = np.zeros(list(data.values())[0].shape, dtype=float)
            nObs = np.zeros(list(data.values())[0].shape, dtype=int)
    elif (chisq is None) ^ (nObs is None):
        raise ValueError('Both chisq and nObs must be specified or nor neither can be.')

    # build containers for chisq_per_ant and nObs_per_ant if not supplied
    if chisq_per_ant is None and nObs_per_ant is None:
        chisq_per_ant = {}
        nObs_per_ant = {}
    elif (chisq_per_ant is None) ^ (nObs_per_ant is None):
        raise ValueError('Both chisq_per_ant and nObs_per_ant must be specified or nor neither can be.')

    # if data_wgts is unspecified, make it 1.0
    if data_wgts is None:
        data_wgts = {bl: 1.0 for bl in data.keys()}

    # Expand model to include all bl in reds, assuming that model has the first bl in the redundant group
    if reds is not None:
        model = copy.deepcopy(model)
        for red in reds:
            if np.any([bl in data for bl in red]):
                for bl in red:
                    model[bl] = model[red[0]]

    for bl in data.keys():
        ap1, ap2 = split_pol(bl[2])
        # make that if split_by_antpol is true, the baseline is not cross-polarized
        if bl in model and bl in data_wgts and (not split_by_antpol or ap1 == ap2):
            ant1, ant2 = (bl[0], ap1), (bl[1], ap2)

            # multiply model by gains if they are supplied
            if gains is not None:
                model_here = model[bl] * gains[ant1] * np.conj(gains[ant2])
            else:
                model_here = copy.deepcopy(model[bl])

            # include gain flags in data weights
            assert np.isrealobj(data_wgts[bl])
            if gain_flags is not None:
                wgts = data_wgts[bl] * ~(gain_flags[ant1]) * ~(gain_flags[ant2])
            else:
                wgts = data_wgts[bl]

            # convert wgts into nObs, handling the case where wgts is a scaler because data_wgts is None
            if np.isscalar(wgts):
                wgts = wgts * np.ones(data[bl].shape, dtype=float)
            nObs_from_wgts = np.array(wgts > 0, dtype=int)

            # calculate chi^2
            chisq_here = np.asarray(np.abs(model_here - data[bl]) ** 2 * wgts, dtype=np.float64)
            if split_by_antpol:
                if ap1 in chisq:
                    assert ap1 in nObs
                    chisq[ap1] = chisq[ap1] + chisq_here
                    nObs[ap1] += nObs[ap1] + nObs_from_wgts
                else:
                    assert ap1 not in nObs
                    chisq[ap1] = copy.deepcopy(chisq_here)
                    nObs[ap1] = nObs_from_wgts
            else:
                chisq += chisq_here
                nObs += (wgts > 0)

            # assign chisq and observations to both chisq_per_ant and nObs_per_ant
            for ant in [ant1, ant2]:
                if ant in chisq_per_ant:
                    assert ant in nObs_per_ant
                    chisq_per_ant[ant] = chisq_per_ant[ant] + chisq_here
                    nObs_per_ant[ant] = nObs_per_ant[ant] + nObs_from_wgts
                else:
                    assert ant not in nObs_per_ant
                    chisq_per_ant[ant] = copy.deepcopy(chisq_here)
                    nObs_per_ant[ant] = nObs_from_wgts

    return chisq, nObs, chisq_per_ant, nObs_per_ant


def per_antenna_modified_z_scores(metric):
    """Compute modified Z-Score over antennas for each antenna polarization.
    This function computes the per-pol modified z-score of the given metric
    dictionary for each antenna.
    The modified Z-score is defined as:
        0.6745 * (metric - median(all_metrics))/ median_absoulte_deviation

    Parameters:
        metric : dict
            Dictionary of metric data to compute z-score. Keys are expected to
            have the form: (ant, antpol)
    Returns:
        zscores : dict
            Dictionary of z-scores for the given data.
    """
    zscores = {}
    antpols = set([key[1] for key in metric])
    for antpol in antpols:
        values = np.array([val for (key, val) in metric.items()
                           if key[1] == antpol])
        median = np.nanmedian(values)
        medAbsDev = np.nanmedian(np.abs(values - median))
        for (key, val) in metric.items():
            if key[1] == antpol:
                # this factor makes it comparable to a
                # standard z-score for gaussian data
                zscores[key] = 0.6745 * (val - median) / medAbsDev
    return zscores


def gp_interp1d(x, y, x_eval=None, flags=None, length_scale=1.0, nl=1e-10,
                kernel=None, Nmirror=0, optimizer=None, xthin=None):
    """
    Gaussian Process interpolation.

    Interpolate or smooth a series of datavectors y with a Gaussian Process
    along its zeroth axis.
    See sklearn.gaussian_process for more details on the methodology.

    Args:
        x : real ndarray
            Independent variable 1-d array of shape (Nvalues,)
        y : ndarray
            Dependent variable 2-d array of shape (Nvalues, Nvectors)
        x_eval : real ndarray
            A 1-d array holding model evaluation x-values.
            Default is x.
        flags : ndarray
            A boolean array of y flags of shape (Nvalues, Nvectors)
        length_scale : float
            Length scale for RBF kernel in units of x if input kernel is None.
        nl : float
            Noise level for WhiteNoise kernel if input kernel is None.
            Recommended to keep this near 1e-10 for non-expert user.
            This is normalized to a unity-variance process.
        kernel : sklearn Kernel object
            Custom kernel to use. This supercedes length_scale and nl choices.
        Nmirror : int
            Number of x values to mirror about either end before interpolation.
            This can minimize impact of boundary effects on interpolation.
        optimizer : str
            Hyperparameter optimization method. Default is no optimization.
        xthin : int
            Thinning factor along prediction x-axis of unflagged data.
            Default is no thinning.

    Returns:
        ndarray
            Interpolated y-vector of shape (len(x_eval), Nvectors)
    """
    # type checks
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if x_eval is None:
        x_eval = x.reshape(-1, 1)
    else:
        if x_eval.ndim == 1:
            x_eval = x_eval.reshape(-1, 1)

    # setup kernel
    if kernel is None:
        kernel = 1**2 * gp.kernels.RBF(length_scale=length_scale) + gp.kernels.WhiteKernel(noise_level=nl)

    # initialize GP
    GP = gp.GaussianProcessRegressor(kernel=kernel, optimizer=optimizer, normalize_y=False, copy_X_train=False)

    # get flags
    if flags is None:
        flags = np.zeros_like(y, dtype=bool)

    # thin x-axis if desired
    if xthin is not None:
        assert xthin < x.size // 2, "Can't thin x-axis by more then len(x) // 2"
        x = x[::xthin]
        y = y[::xthin, :]
        flags = flags[::xthin, :]

    # mirror if desired
    if Nmirror > 0:
        assert Nmirror < x.size, "Nmirror can't be equal or larger than x"
        x = np.pad(x, Nmirror, mode='reflect', reflect_type='odd')
        y = np.concatenate([y[1:Nmirror + 1, :][::-1, :], y, y[-Nmirror - 1:-1, :][::-1, :]], axis=0)
        flags = np.concatenate([flags[1:Nmirror + 1, :][::-1, :], flags, flags[-Nmirror - 1:-1, :][::-1, :]], axis=0)

    # setup training values
    X = x.reshape(-1, 1)

    # get unique flagging patterns
    flag_patterns = []
    flag_hashes = []
    flag_indices = {}
    for i in range(flags.shape[1]):
        # hash the flag pattern
        h = hash(flags[:, i].tobytes())
        # append to list
        if h not in flag_hashes:
            flag_hashes.append(h)
            flag_patterns.append(flags[:, i])
            flag_indices[h] = [i]
        else:
            flag_indices[h].append(i)

    # do real and imag separately if complex
    iscomplex = np.iscomplexobj(y)
    if iscomplex:
        y_iter = [y.real, y.imag]
    else:
        y_iter = [y]
    ypredict = []
    for _y in y_iter:
        # shift by median and normalize by MAD
        ymed = np.median(_y, axis=0, keepdims=True)
        ymad = np.median(np.abs(_y - ymed), axis=0, keepdims=True) * 1.4826
        # in rare case that ymad == 0, set it to 1.0 to prevent divbyzero error
        ymad[np.isclose(ymad, 0.0)] = 1.0
        _y = (_y - ymed) / ymad

        # iterate over flagging patterns
        ypred = np.zeros((len(x_eval), _y.shape[1]), float)
        for i, fh in enumerate(flag_hashes):
            # get 1st axis indices for this flagging pattern
            inds = flag_indices[fh]
            # get 0th axis indices given flags
            select = ~flag_patterns[i]
            # interpolate if not completely flagged
            if np.any(select):
                # pick out unflagged data for training
                GP.fit(X[select, :], _y[select, :][:, inds])
                # insert predicted data at x_eval points into output vector
                ypred[:, inds] = GP.predict(x_eval).reshape(x_eval.shape[0], len(inds)) * ymad[:, inds] + ymed[:, inds]

        # append
        ypredict.append(ypred)

    # cast back into complex domain if necessary
    if iscomplex:
        ypredict = ypredict[0] + 1j * ypredict[1]
    else:
        ypredict = ypredict[0]

    return ypredict


def gain_relative_difference(old_gains, new_gains, flags, denom=None):
    """Compuate relative gain differences between two sets of calibration solutions
    (e.g. abscal and smooth_cal), as well as antenna-averaged relative gain differences.

    Arguments:
        old_gains: dictionary mapping keys like (0, 'Jnn') to waterfalls of complex gains.
            Must contain all keys in new_gains.
        new_gains: dictionary mapping keys like (0, 'Jnn') to waterfalls of complex gains
        flags: dictionary mapping keys like (0, 'Jnn') to boolean flag waterfalls. Must
            contain all keys in new_gains.
        denom: gain dictionary to use to normalize the relative difference. Default None
            uses old_gains. Anywhere denom is 0 must also be flagged.

    Returns:
        relative_diff: dictionary mapping keys like (0, 'Jnn') to waterfalls of relative
            differences between old and new gains
        avg_relative_diff: dictionary mapping antpols (e.g. 'Jnn') to waterfalls. Flagged
            gains are excluded from the average; completely flagged times and channels are
            replaced by 0s to match the convention of chi^2 above

    """
    if denom is None:
        denom = old_gains
    relative_diff = {}
    for ant in new_gains:
        assert ~np.any(denom[ant] == 0) or np.all(flags[np.isclose(np.abs(denom[ant]), 0.0)])
        relative_diff[ant] = np.true_divide(np.abs(old_gains[ant] - new_gains[ant]), np.abs(denom[ant]),
                                            where=~np.isclose(np.abs(denom[ant]), 0))

    # compute average relative diff over all antennas for each polarizations separately
    avg_relative_diff = {}
    pols = set([ant[1] for ant in new_gains])
    for pol in pols:
        diffs = {ant: copy.deepcopy(relative_diff[ant]) for ant in new_gains if ant[1] == pol}
        for ant in diffs:
            diffs[ant][flags[ant]] = np.nan
        avg_relative_diff[pol] = np.nanmean(list(diffs.values()), axis=0)
        avg_relative_diff[pol][~np.isfinite(avg_relative_diff[pol])] = 0.0  # if completely flagged

    return relative_diff, avg_relative_diff


def red_average(data, reds=None, bl_tol=1.0, inplace=False,
                wgts=None, flags=None, nsamples=None,
                red_bl_keys=None,
                propagate_flags=False):
    """
    Redundantly average visibilities in a DataContainer, HERAData or UVData object.
    Average is weighted by integration_time * nsamples * ~flags unless wgts are fed.

    Args:
        data : DataContainer, HERAData or UVData object
            Object to redundantly average
        reds : list, optional
            Nested lists of antpair tuples to redundantly average.
            E.g. [ [(1, 2), (2, 3)], [(1, 3), (2, 4)], ...]
            If None, will calculate these from the metadata
        bl_tol : float
            Baseline redundancy tolerance in meters. Only used if reds is None.
        inplace : bool
            Perform average and downselect inplace, otherwise returns a deepcopy.
            The first baseline in each reds sublist is kept.
        wgts : DataContainer
            Manual weights to use in redundant average. This supercedes flags and nsamples as weights
            if provided, and will also be used if input data is a UVData or a subclass of it.
        flags : DataContainer
            If data is a DataContainer, these are its flags. Default (None) is no flags.
            If data is a UVData, then data.flag_array is used regardless of flags input.
        nsamples : DataContainer
            If data is a DataContainer, these are its nsamples. Default (None) is 1.0 for all pixels.
            Furthermore, if data is a DataContainer, integration_time is 1.0 for all pixels.
            If data is a UVData, then data.nsample_array is used regardless of nsamples input.
        red_bl_keys : list, optional
            Optional list of keys to use for each redundantly averaged group. If None,
            use the first key from each group.
            Default is None.
        propagate_flags : bool, optional
            If True, propagate input flags to the average flag, even if wgts are provided.
            Note, if wgts are provided, the input flags are NOT used for weighting, but
            are propagated to the output flags. Default = False.
    Returns:
        if fed a DataContainer:
            Dict, averaged data
            Dict, averaged flags
            Dict, summed nsamples
        elif fed a HERAData or UVData:
            HERAData or UVData object, averaged data

    Notes:
        1. Different polarizations are assumed to be non-redundant.
        2. Default weighting is nsamples * integration_time * ~flags.
    """
    from hera_cal import redcal, datacontainer

    # type checks
    if not (isinstance(data, datacontainer.DataContainer) or isinstance(data, UVData)):
        raise ValueError("data must be a DataContainer or a UVData or its subclass")
    fed_container = isinstance(data, datacontainer.DataContainer)

    # fill DataContainers if necessary
    if fed_container:
        if not inplace:
            flags = copy.deepcopy(flags)
            nsamples = copy.deepcopy(nsamples)
        if flags is None:
            flags = datacontainer.DataContainer({k: np.zeros_like(data[k], bool) for k in data})
        if nsamples is None:
            nsamples = datacontainer.DataContainer({k: np.ones_like(data[k], float) for k in data})

    # get weights: if wgts are not fed, then use flags and nsamples
    if wgts is None:
        if fed_container:
            wgts = datacontainer.DataContainer({k: nsamples[k] * ~flags[k] for k in data})
        else:
            wgts = datacontainer.DataContainer({k: data.get_nsamples(k) * ~data.get_flags(k) for k in data.get_antpairpols()})

    # deepcopy
    if not inplace:
        data = copy.deepcopy(data)

    # get metadata
    if fed_container:
        pols = sorted(data.pols())
    else:
        pols = [polnum2str(pol, x_orientation=data.x_orientation) for pol in data.polarization_array]

    # get redundant groups
    if reds is None:
        # if DataContainer, check for antpos
        if fed_container:
            if not hasattr(data, 'antpos') or data.antpos is None:
                raise ValueError("DataContainer must have antpos dictionary to calculate reds")
            antposd = data.antpos
        else:
            antpos, ants = data.get_ENU_antpos()
            antposd = dict(zip(ants, antpos))
        reds = redcal.get_pos_reds(antposd, bl_error_tol=bl_tol)

    # eliminate baselines not in data
    if fed_container:
        antpairs = sorted(data.antpairs())
    else:
        antpairs = data.get_antpairs()
    reds = [[bl for bl in blg if bl in antpairs or bl[::-1] in antpairs] for blg in reds]
    reds = [blg for blg in reds if len(blg) > 0]

    # iterate over redundant groups and polarizations
    if red_bl_keys is None:
        red_bl_keys = [blg[0] for blg in reds]
    for pol in pols:
        for blg, blk in zip(reds, red_bl_keys):
            # get data and weighting for this pol-blgroup
            if fed_container:
                d = np.asarray([data[bl + (pol,)] for bl in blg])
                f = np.asarray([(~flags[bl + (pol,)]).astype(float) for bl in blg])
                n = np.asarray([nsamples[bl + (pol,)] for bl in blg])
                # DataContainer can't track integration time, so no tint here
                tint = np.array([1.0])
                w = np.asarray([wgts[bl + (pol,)] for bl in blg])

            else:
                d = np.asarray([data.get_data(bl + (pol,)) for bl in blg])
                f = np.asarray([(~data.get_flags(bl + (pol,))).astype(float) for bl in blg])
                n = np.asarray([data.get_nsamples(bl + (pol,)) for bl in blg])
                tint = []
                for bl in blg:
                    blinds = data.antpair2ind(bl + (pol,))
                    if len(blinds) == 0:
                        blinds = data.antpair2ind(reverse_bl(bl + (pol,)))
                    tint.append(data.integration_time[blinds])
                tint = np.asarray(tint)[:, :, None]
                w = np.asarray([wgts[bl + (pol,)] for bl in blg]) * tint
            # take the weighted average
            wsum = np.sum(w, axis=0).clip(1e-10, np.inf)  # this is the normalization
            davg = np.sum(d * w, axis=0) / wsum  # weighted average
            # set flagged data to 1.0+0.0j
            flagged_f = np.all(f == 0, axis=0)
            davg[flagged_f] = 1.0
            fmax = np.max(f, axis=2)  # collapse along freq: marks any fully flagged integrations
            if tint.squeeze().ndim == 1:  # tint.squeeze() can be one-dimensional if we have one time sample in our data.
                iavg = np.sum(tint.squeeze()[:, None] * fmax, axis=0) / np.sum(fmax, axis=0).clip(1e-10, np.inf)
            else:
                iavg = np.sum(tint.squeeze() * fmax, axis=0) / np.sum(fmax, axis=0).clip(1e-10, np.inf)

            binary_wgts = (~np.isclose(w, 0)).astype(float)  # binary weights.
            navg = np.sum(n * binary_wgts, axis=0)
            if propagate_flags:
                favg = np.all(np.isclose(w * f, 0), axis=0)
            else:
                favg = np.all(np.isclose(w, 0), axis=0)

            # replace with new data
            if fed_container:
                blkey = blk + (pol,)
                data[blkey] = davg
                flags[blkey] = favg
                nsamples[blkey] = navg

            else:
                blinds = data.antpair2ind(blk)
                polind = pols.index(pol)
                if len(blinds) == 0:
                    blinds = data.antpair2ind(reverse_bl(blk))
                    davg = np.conj(davg)
                data.data_array[blinds, :, polind] = davg
                data.flag_array[blinds, :, polind] = favg
                data.nsample_array[blinds, :, polind] = navg
                data.integration_time[blinds] = iavg

    # select out averaged bls
    bls = [blk + (pol,) for pol in pols for blk in red_bl_keys]
    if fed_container:
        to_del = [bl for bl in data.keys() if bl not in bls and reverse_bl(bl) not in bls]
        del data[to_del], flags[to_del], nsamples[to_del]
    else:
        data.select(bls=bls)

    if not inplace:
        if fed_container:
            return data, flags, nsamples
        else:
            return data


def eq2top_m(ha, dec):
    """Return the 3x3 matrix converting equatorial coordinates to topocentric
    at the given hour angle (ha) and declination (dec).
    Borrowed from pyuvdata which borrowed from aipy"""
    sin_H, cos_H = np.sin(ha), np.cos(ha)
    sin_d, cos_d = np.sin(dec), np.cos(dec)
    mat = np.array([[sin_H, cos_H, np.zeros_like(ha)],
                    [-sin_d * cos_H, sin_d * sin_H, cos_d],
                    [cos_d * cos_H, -cos_d * sin_H, sin_d]])
    if len(mat.shape) == 3:
        mat = mat.transpose([2, 0, 1])
    return mat


def top2eq_m(ha, dec):
    """Return the 3x3 matrix converting topocentric coordinates to equatorial
    at the given hour angle (ha) and declination (dec).
    Slightly changed from aipy to simply write the matrix instead of inverting.
    Borrowed from pyuvdata."""
    sin_H, cos_H = np.sin(ha), np.cos(ha)
    sin_d, cos_d = np.sin(dec), np.cos(dec)
    mat = np.array([[sin_H, -cos_H * sin_d, cos_d * cos_H],
                    [cos_H, sin_d * sin_H, -cos_d * sin_H],
                    [np.zeros_like(ha), cos_d, sin_d]])
    if len(mat.shape) == 3:
        mat = mat.transpose([2, 0, 1])
    return mat


def chunk_baselines_by_redundant_groups(reds, max_chunk_size):
    """Chunk list of baselines by redundant group constrained by number of baselines.

    This method calculates chunks of baselines to load simultaneously constrained by the conditions that
    all baselines simultaneously loaded completely fill out redundant groups and the number of baselines
    loaded is less then or equal to max_chunk_size.
    If a redundant group exceeds max_chunk_size it will be loaded in its entirety anyways
    and warning will be raised.

    Parameters
    ----------
    reds : list
        list of lists with redundant groups of antpairpols
    max_chunk_size :
        maximum number of baselines to group in each chunk
                    while filling out redundant groups.

    Returns
    -------

    chunked_baselines : list
        list of lists of antpairpols where each chunk containes entire redundant groups
        and (if possible) includes a number of baselines less then max_chunk_size.
    """
    baseline_chunks = []
    group_index = 0
    # iterate through redundant groups
    for grp in reds:
        if len(grp) > max_chunk_size:
            # if red group is larger then the chunk size.
            # then give a warning and treate the red group as a chunk anyways.
            warnings.warn("Warning: baseline group of length %d encountered with number"
                          " of baselines exceeding max_chunk_size=%d."
                          " First baseline is %s"
                          " Loading group anyways." % (len(reds[group_index]), max_chunk_size, str(reds[group_index][0])))
            baseline_chunks.append(grp)
        else:
            # if baseline_chunks is empty, initialize the first chunk.
            if len(baseline_chunks) == 0:
                baseline_chunks.append([])
            if len(baseline_chunks[-1]) + len(grp) <= max_chunk_size:
                baseline_chunks[-1].extend(grp)
            else:
                baseline_chunks.append(grp)
    return baseline_chunks


def select_spw_ranges(inputfilename, outputfilename, spw_ranges=None, clobber=False):
    """Utility function for selecting spw_ranges and writing out

    Parameters
    ----------
    inputfilename: str
        name if inputfile to read in
    outputfilename: str
        name of outputfile to write spw selected data too.
    spw_ranges: list, optional
        list of 2-tuples specifying spw-ranges to select.
        default is None -> select all frequencies in inputfile.
    clobber: bool, optional
    """
    hd = UVData()
    hd.read_uvh5(inputfilename, read_data=False)
    if spw_ranges is None:
        spw_ranges = [(0, hd.Nfreqs)]
    # read in selected spw_ranges
    hd.read(inputfilename, freq_chans=np.hstack([np.arange(spw[0], spw[1]).astype(int) for spw in spw_ranges]))
    hd.write_uvh5(outputfilename, clobber=clobber)


def select_spw_ranges_argparser():
    '''Arg parser for commandline operation of select_spw_ranges.'''
    def list_of_int_tuples(v):
        if '~' in v:
            v = [tuple([int(_x) for _x in x.split('~')]) for x in v.split(",")]
        else:
            v = [tuple([int(_x) for _x in x.split()]) for x in v.split(",")]
        return v
    ap = argparse.ArgumentParser(description="Select spw-ranges from a file.")
    ap.add_argument("inputfilename", type=str, help="path to visibility data to select spw_ranges from.")
    ap.add_argument("outputfilename", type=str, help="path to spw selected visibility file to write out.")
    ap.add_argument("--spw_ranges", default=None, type=list_of_int_tuples, help="List of spw channel selections. Two acceptable formats are "
                                                                                "Ex1: '200~300,500~650' --> [(200, 300), (500, 650), ...] and "
                                                                                "Ex2: '200 300, 500 650' --> [(200, 300), (500, 650), ...]")
    ap.add_argument("--clobber", default=False, action="store_true", help='overwrites existing file at outfile')
    return ap

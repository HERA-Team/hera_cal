# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import numpy as np
from copy import deepcopy
import argparse
import os
import linsolve
from itertools import chain

from . import utils
from .noise import predict_noise_variance_from_autos, infer_dt
from .datacontainer import DataContainer, RedDataContainer
from .utils import split_pol, conj_pol, split_bl, reverse_bl, join_bl, join_pol, comply_pol, per_antenna_modified_z_scores
from .io import HERAData, HERACal, write_cal, save_redcal_meta
from .apply_cal import calibrate_in_place


SEC_PER_DAY = 86400.
IDEALIZED_BL_TOL = 1e-8  # bl_error_tol for redcal.get_reds when using antenna positions calculated from reds


def get_pos_reds(antpos, bl_error_tol=1.0, include_autos=False):
    """ Figure out and return list of lists of redundant baseline pairs. Ordered by length. All baselines
        in a group have the same orientation with a preference for positive b_y and, when b_y==0, positive
        b_x where b((i,j)) = pos(j) - pos(i).

        Args:
            antpos: dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}. 1D and 2D also OK.
            bl_error_tol: the largest allowable difference between baselines in a redundant group
                (in the same units as antpos). Normally, this is up to 4x the largest antenna position error.
            include_autos: bool, optional
                if True, include autos in the list of pos_reds. default is False
        Returns:
            reds: list (sorted by baseline legnth) of lists of redundant tuples of antenna indices (no polarizations),
            sorted by index with the first index of the first baseline the lowest in the group.
    """
    keys = list(antpos.keys())
    reds = {}
    assert np.all([len(pos) <= 3 for pos in antpos.values()]), 'Get_pos_reds only works in up to 3 dimensions.'
    ap = {ant: np.pad(pos, (0, 3 - len(pos)), mode='constant') for ant, pos in antpos.items()}  # increase dimensionality
    array_is_flat = np.all(np.abs(np.array(list(ap.values()))[:, 2] - np.mean(list(ap.values()), axis=0)[2]) < bl_error_tol / 4.0)
    p_or_m = (0, -1, 1)
    if array_is_flat:
        epsilons = [[dx, dy, 0] for dx in p_or_m for dy in p_or_m]
    else:
        epsilons = [[dx, dy, dz] for dx in p_or_m for dy in p_or_m for dz in p_or_m]

    def check_neighbors(delta):  # Check to make sure reds doesn't have the key plus or minus rounding error
        for epsilon in epsilons:
            newKey = (delta[0] + epsilon[0], delta[1] + epsilon[1], delta[2] + epsilon[2])
            if newKey in reds:
                return newKey
        return

    for i, ant1 in enumerate(keys):
        if include_autos:
            start_ind = i
        else:
            start_ind = i + 1
        for ant2 in keys[start_ind:]:
            bl_pair = (ant1, ant2)
            delta = tuple(np.round(1.0 * (np.array(ap[ant2]) - np.array(ap[ant1])) / bl_error_tol).astype(int))
            new_key = check_neighbors(delta)
            if new_key is None:  # forward baseline has no matches
                new_key = check_neighbors(tuple([-d for d in delta]))
                if new_key is not None:  # reverse baseline does have a match
                    bl_pair = (ant2, ant1)
            if new_key is not None:  # either the forward or reverse baseline has a match
                reds[new_key].append(bl_pair)
            else:  # this baseline is entirely new
                if delta[0] <= 0 or (delta[0] == 0 and delta[1] <= 0) or (delta[0] == 0 and delta[1] == 0 and delta[2] <= 0):
                    delta = tuple([-d for d in delta])
                    bl_pair = (ant2, ant1)
                reds[delta] = [bl_pair]

    # sort reds by length and each red to make sure the first antenna of the first bl in each group is the lowest antenna number
    orderedDeltas = [delta for (length, delta) in sorted(zip([np.linalg.norm(delta) for delta in reds.keys()], reds.keys()))]
    return [sorted(reds[delta]) if sorted(reds[delta])[0][0] == np.min(reds[delta])
            else sorted([reverse_bl(bl) for bl in reds[delta]]) for delta in orderedDeltas]


def add_pol_reds(reds, pols=['nn'], pol_mode='1pol'):
    """ Takes positonal reds (antenna indices only, no polarizations) and converts them
    into baseline tuples with polarization, depending on pols and pol_mode specified.

    Args:
        reds: list of list of antenna index tuples considered redundant
        pols: a list of polarizations e.g. ['nn', 'ne', 'en', 'ee']
        pol_mode: polarization mode of calibration
            '1pol': 1 antpol and 1 vispol (e.g. 'Jnn' and 'nn'). Default.
            '2pol': 2 antpols, no cross-vispols (e.g. 'Jnn','Jee' and 'nn','ee')
            '4pol': 2 antpols, 4 vispols (e.g. 'Jnn','Jee' and 'nn','ne','en','ee')
            '4pol_minV': 2 antpols, 4 vispols in data but assuming V_ne = V_en in model

    Returns:
        reds: list of lists of redundant baseline tuples, e.g. (ind1,ind2,pol)
    """
    # pre-process to ensure pols complies w/ hera_cal polarization convention
    pols = [comply_pol(p) for p in pols]

    redsWithPols, didBothCrossPolsForMinV = [], False
    for pol in pols:
        if pol_mode != '4pol_minV' or pol[0] == pol[1]:
            redsWithPols += [[bl + (pol,) for bl in bls] for bls in reds]
        elif pol_mode == '4pol_minV' and not didBothCrossPolsForMinV:
            # Combine together e.g. 'ne' and 'en' visibilities as redundant
            redsWithPols += [([bl + (pol,) for bl in bls]
                              + [bl + (conj_pol(pol),) for bl in bls]) for bls in reds]
            didBothCrossPolsForMinV = True
    return redsWithPols


def get_reds(antpos, pols=['nn'], pol_mode='1pol', bl_error_tol=1.0, include_autos=False):
    """ Combines redcal.get_pos_reds() and redcal.add_pol_reds(). See their documentation.

    Args:
        antpos: dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}.
        pols: a list of polarizations e.g. ['nn', 'ne', 'en', 'ee']
        pol_mode: polarization mode of calibration
            '1pol': 1 antpol and 1 vispol (e.g. 'Jnn' and 'nn'). Default.
            '2pol': 2 antpols, no cross-vispols (e.g. 'Jnn','Jee' and 'nn','ee')
            '4pol': 2 antpols, 4 vispols (e.g. 'Jnn','Jee' and 'nn','ne','en','ee')
            '4pol_minV': 2 antpols, 4 vispols in data but assuming V_ne = V_en in model
        bl_error_tol: the largest allowable difference between baselines in a redundant group
            (in the same units as antpos). Normally, this is up to 4x the largest antenna position error.
        include_autos: bool, optional
            if true, include autocorr redundant group
            Default is false.

    Returns:
        reds: list (sorted by baseline length) of lists of redundant baseline tuples, e.g. (ind1,ind2,pol).
            Each interior list is sorted so that the lowest index is first in the first baseline.

    """
    pos_reds = get_pos_reds(antpos, bl_error_tol=bl_error_tol, include_autos=include_autos)
    return add_pol_reds(pos_reds, pols=pols, pol_mode=pol_mode)


def filter_reds(reds, bls=None, ex_bls=None, ants=None, ex_ants=None, ubls=None, ex_ubls=None,
                pols=None, ex_pols=None, antpos=None, min_bl_cut=None, max_bl_cut=None,
                max_dims=None, min_dim_size=1):
    '''
    Filter redundancies to include/exclude the specified bls, antennas, unique bl groups and polarizations.
    Also allows filtering reds by removing antennas so that the number of generalized tip/tilt degeneracies
    is no more than max_dims. Arguments are evaluated, in order of increasing precedence: (pols, ex_pols,
    ubls, ex_ubls, bls, ex_bls, ants, ex_ants, min_bl_cut, max_bl_cut, max_dims).

    Args:
        reds: list of lists of redundant (i,j,pol) baseline tuples, e.g. the output of get_reds().
            Not modified in place.
        bls (optional): baselines to include. Baselines of the form (i,j,pol) include that specific
            visibility.  Baselines of the form (i,j) are broadcast across all polarizations present in reds.
        ex_bls (optional): same as bls, but excludes baselines.
        ants (optional): antennas to include. Only baselines where both antenna indices are in ants
            are included.  Antennas of the form (i,pol) include that antenna/pol. Antennas of the form i are
            broadcast across all polarizations present in reds.
        ex_ants (optional): same as ants, but excludes antennas.
        ubls (optional): redundant (unique baseline) groups to include. Each baseline in ubls is taken to
            represent the redundant group containing it. Baselines of the form (i,j) are broadcast across all
            polarizations, otherwise (i,j,pol) selects a specific redundant group.
        ex_ubls (optional): same as ubls, but excludes groups.
        pols (optional): polarizations to include in reds. e.g. ['nn','ee','ne','en']. Default includes all
            polarizations in reds.
        ex_pols (optional): same as pols, but excludes polarizations.
        antpos: dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}. 1D and 2D also OK.
        min_bl_cut: cut redundant groups with average baseline lengths shorter than this. Same units as antpos
            which must be specified.
        max_bl_cut: cut redundant groups with average baselines lengths longer than this. Same units as antpos
            which must be specified.
        max_dims: maximum number of dimensions required to specify antenna positions (up to some arbitary shear).
            This is equivalent to the number of generalized tip/tilt phase degeneracies of redcal that are fixed
            with remove_degen() and must be later abscaled. 2 is a classically "redundantly calibratable" planar
            array. More than 2 usually arises with subarrays of redundant baselines. None means no filtering.
        min_dim_size: minimum number of atennnas allowed with non-zero positions in a given dimension. This
            allows filtering out of antennas where only a few are responsible for adding a dimension. Ignored
            if max_dims is None. Default 1 means no further filtering based on the number of anntenas in that dim.

    Return:
        reds: list of lists of redundant baselines in the same form as input reds.
    '''
    # pre-processing step to ensure that reds complies with hera_cal polarization conventions
    reds = [[(i, j, comply_pol(p)) for (i, j, p) in gp] for gp in reds]
    if pols is None:  # if no pols are provided, deduce them from the red
        pols = set(gp[0][2] for gp in reds)
    if ex_pols:
        pols = set(p for p in pols if p not in ex_pols)
    reds = [gp for gp in reds if gp[0][2] in pols]

    def expand_bls(gp):
        gp3 = [(g[0], g[1], p) for g in gp if len(g) == 2 for p in pols]
        return set(gp3 + [g for g in gp if len(g) == 3])
    antpols = set(sum([list(split_pol(p)) for p in pols], []))

    def expand_ants(gp):
        gp2 = [(g, p) for g in gp if not hasattr(g, '__len__') for p in antpols]
        return set(gp2 + [g for g in gp if hasattr(g, '__len__')])

    def split_bls(bls):
        return set(split_bl(bl) for bl in bls)
    if ubls or ex_ubls:
        bl2gp = {}
        for i, gp in enumerate(reds):
            for key in gp:
                bl2gp[key] = bl2gp[reverse_bl(key)] = i
        if ubls:
            ubls = expand_bls(ubls)
            ubls = set(bl2gp[key] for key in ubls if key in bl2gp)
        else:
            ubls = set(range(len(reds)))
        if ex_ubls:
            ex_ubls = expand_bls(ex_ubls)
            ex_ubls = set(bl2gp[bl] for bl in ex_ubls if bl in bl2gp)
        else:
            ex_ubls = set()
        reds = [gp for i, gp in enumerate(reds) if i in ubls and i not in ex_ubls]
    if bls:
        bls = expand_bls(bls)
    else:  # default to set of all baselines
        bls = set(key for gp in reds for key in gp)
    if ex_bls:
        ex_bls = expand_bls(ex_bls)
        ex_bls |= set(reverse_bl(k) for k in ex_bls)  # put in reverse baselines
        bls = set(k for k in bls if k not in ex_bls)
    if ants:
        ants = expand_ants(ants)
        bls = set(join_bl(i, j) for i, j in split_bls(bls) if i in ants and j in ants)
    if ex_ants:
        ex_ants = expand_ants(ex_ants)
        bls = set(join_bl(i, j) for i, j in split_bls(bls) if i not in ex_ants and j not in ex_ants)
    bls |= set(reverse_bl(k) for k in bls)  # put in reverse baselines
    reds = [[key for key in gp if key in bls] for gp in reds]
    reds = [gp for gp in reds if len(gp) > 0]

    if min_bl_cut is not None or max_bl_cut is not None:
        assert antpos is not None, 'antpos must be passed in if min_bl_cut or max_bl_cut is specified.'
        lengths = [np.mean([np.linalg.norm(antpos[bl[1]] - antpos[bl[0]]) for bl in gp]) for gp in reds]
        reds = [gp for gp, l in zip(reds, lengths) if ((min_bl_cut is None or l > min_bl_cut)
                                                       and (max_bl_cut is None or l < max_bl_cut))]

    if max_dims is not None:
        while True:
            # Compute idealized antenna positions from redundancies. Given the reds (a list of list of
            # redundant baselines), these positions will be coordinates in a vector space that reproduce
            # the ideal antenna positions with a set of unknown basis vectors. The dimensionality of
            # idealized_antpos is determined in reds_to_antpos by first assigning each antenna its own
            # dimension and then inferring how many of those are simply linear combinations of others
            # using the redundancies. The number of dimensions is equivalent to the number of generalized
            # tip/tilt degeneracies of redundant calibration.
            idealized_antpos = reds_to_antpos(reds, tol=IDEALIZED_BL_TOL)
            ia_array = np.array(list(idealized_antpos.values()))

            # if we've removed all antennas, break
            if len(ia_array) == 0:
                break
            # if we're down to 1 dimension, the mode finding below won't work. Just check Nants >= min_dim_size.
            if len(ia_array[0]) <= 1:
                if len(ia_array) >= min_dim_size:
                    break

            # Find dimension with the most common mode idealized coordinate value. This is supposed to look
            # for outlier antennas off the redundant grid small sub-arrays that cannot be redundantly
            # calibrated without adding more degeneracies than desired.
            mode_count, mode_value, mode_dim = 0, 0, 0
            for dim, coords in enumerate(np.array(list(idealized_antpos.values())).T):
                rounded_coords = coords.round(decimals=int(np.floor(-np.log10(IDEALIZED_BL_TOL))))
                unique, counts = np.unique(rounded_coords, return_counts=True)
                if np.max(counts) > mode_count:
                    mode_count = np.max(counts)
                    mode_value = unique[counts == mode_count][0]
                    mode_dim = dim

            # Cut all antennas not part of that mode to reduce the dimensionality of idealized_antpos
            new_ex_ants = [ant for ant in idealized_antpos if
                           np.abs(idealized_antpos[ant][mode_dim] - mode_value) > IDEALIZED_BL_TOL]

            # If we're down to the reqested number of dimensions and if the next filtering would
            # eliminate more antennas than min_dim_size, then break instead of filtering.
            if len(ia_array[0]) <= max_dims:
                if (len(new_ex_ants) >= min_dim_size):
                    break

            reds = filter_reds(reds, ex_ants=new_ex_ants)

    return reds


def combine_reds(reds1, reds2, unfiltered_reds=None):
    '''Combine the groups in two separate lists of redundancies into one which
    does not contain repeats.

    Arguments:
        reds1: list of list or redundant baseline tuples to combine
        reds2: another list of list or redundant baseline tuples to combine
        unfiltered_reds: optional list of list of redundant baselines. Used to combine
            non-overlapping but redundant groups to get the most accurate answers.

    Returns:
        combined_reds: list of list of redundant baselines, combining reds1 and reds2
            as much as possible
     '''
    if unfiltered_reds is not None:
        bls_to_use = set([bl for reds in [reds1, reds2] for red in reds for bl in red])
        combined_reds = filter_reds(unfiltered_reds, bls=bls_to_use)
    else:
        # if unfilterd reds is not provided, try to combine the groups as much as possible.
        # N.B. this can still give wrong answers if there are baselines redundant with each
        # other but unique to reds1 and reds2 respectively
        reds1_sets = [set(red) for red in reds1]
        reds1_map = {bl: n for n, red1_set in enumerate(reds1_sets) for bl in red1_set}
        for red2 in reds2:
            # figure out if any baseline in this group corresponds to a baseline in reds1
            matched_group = None
            for bl in red2:
                if bl in reds1_map:
                    matched_group = reds1_map[bl]
            if matched_group is not None:
                # if there's a match, take the union of the two groups
                reds1_sets[matched_group] |= set(red2)
            else:
                # otherwise, make a new group
                reds1_sets.append(set(red2))
        combined_reds = [list(red) for red in reds1_sets]

    # sort result in a useful way
    combined_reds = [sorted(red, key=lambda x: x[0]) for red in sorted(combined_reds, key=len, reverse=True)]
    return combined_reds


def reds_to_antpos(reds, tol=1e-10):
    '''Computes a set of antenna positions consistent with the given redundancies.
    Useful for projecting out phase slope degeneracies, see https://arxiv.org/abs/1712.07212
    Arguments:
        reds: list of lists of redundant baseline tuples, either (i,j,pol) or (i,j)
        tol: level for two vectors to be considered equal (enabling dimensionality reduction)
    Returns:
        antpos: dictionary of antenna positions in the form {ant_index: np.ndarray}.
            These positions may differ from the true positions of antennas by an arbitrary
            linear transformation. The dimensionality of the positions will be the minimum
            necessary to describe all redundancies (non-redundancy introduces extra dimensions),
            though the most used dimensions will come first.
    '''
    ants = set([ant for red in reds for bl in red for ant in bl[:2]])
    # start with all antennas (except the first) having their own dimension, then reduce the dimensionality
    antpos = {ant: np.array([1. if d + 1 == i else 0. for d in range(len(ants) - 1)])
              for i, ant in enumerate(ants)}
    for red in reds:
        for bl in red:
            # look for vectors in the higher dimensional space that are equal to 0
            delta = (antpos[bl[1]] - antpos[bl[0]]) - (antpos[red[0][1]] - antpos[red[0][0]])
            if np.linalg.norm(delta) > tol:  # this baseline can help us reduce the dimensionality
                dim_to_elim = np.max(np.arange(len(delta))[np.abs(delta) > tol])
                antpos = {ant: np.delete(pos - pos[dim_to_elim] / delta[dim_to_elim] * delta, dim_to_elim)
                          for ant, pos in antpos.items()}
    # remove any all-zero dimensions
    dim_to_elim = np.argwhere(np.sum(np.abs(list(antpos.values())), axis=0) == 0).flatten()
    antpos = {ant: np.delete(pos, dim_to_elim) for ant, pos in antpos.items()}

    # sort dims so that most-used dimensions come first
    dim_usage = np.sum(np.abs(list(antpos.values())) > tol, axis=0)
    antpos = {ant: pos[np.argsort(dim_usage)[::-1]] for ant, pos in antpos.items()}
    return antpos


def make_sol_finite(sol):
    '''Replaces nans and infs in solutions, which are usually the result of visibilities that are
    identically equal to 0. Modifies sol (which is a dictionary with gains and visibilities) in place,
    replacing visibilities with 0.0s and gains with 1.0s'''
    for k, v in sol.items():
        not_finite = ~np.isfinite(v)
        if len(k) == 3:  # visibilities
            sol[k][not_finite] = np.zeros_like(v[not_finite])
        elif len(k) == 2:  # gains
            sol[k][not_finite] = np.ones_like(v[not_finite])


def remove_degen_gains(reds, gains, degen_gains=None, mode='phase', pol_mode='1pol'):
    """ Removes degeneracies from gains (or replaces them with those in gains).  This
    function is nominally intended for use with firstcal, which returns (phase/delay) solutions
    for antennas only.

    Args:
        gains: dictionary that contains gain solutions in the {(index,antpol): np.array} format.
        degen_gains: Optional dictionary in the same format as gains. Gain amplitudes and phases
            in degen_sol replace the values of sol in the degenerate subspace of redcal. If
            left as None, average gain amplitudes will be 1 and average phase terms will be 0.
            For logcal/lincal/omnical, putting firstcal solutions in here can help avoid structure
            associated with phase-wrapping issues.
        mode: 'phase' or 'complex', indicating whether the gains are passed as phases (e.g. delay
            or phi in e^(i*phi)), or as the complex number itself.  If 'phase', only phase degeneracies
            removed.  If 'complex', both phase and amplitude degeneracies are removed.
        pol_mode: polarization mode of redundancies. Can be '1pol', '2pol', '4pol', or '4pol_minV'.
    Returns:
        new_gains: gains with degeneracy removal/replacement performed
    """
    # Check supported pol modes
    assert pol_mode in ['1pol', '2pol', '4pol', '4pol_minV'], f'Unrecognized pol_mode: {pol_mode}'
    assert mode in ('phase', 'complex'), 'Unrecognized mode: %s' % mode
    ants = list(set(ant for gp in reds for bl in gp for ant in split_bl(bl) if ant in gains))
    gainPols = np.array([ant[1] for ant in ants])  # gainPols is list of antpols, one per antenna
    antpols = list(set(gainPols))

    # if mode is 2pol, run as two 1pol remove degens
    if pol_mode == '2pol':
        pol0_gains = {k: v for k, v in gains.items() if k[1] == antpols[0]}
        pol1_gains = {k: v for k, v in gains.items() if k[1] == antpols[1]}
        reds0 = [gp for gp in reds if gp[0][-1] in join_pol(antpols[0], antpols[0])]
        reds1 = [gp for gp in reds if gp[0][-1] in join_pol(antpols[1], antpols[1])]
        new_gains = remove_degen_gains(reds0, pol0_gains, degen_gains=degen_gains, mode=mode, pol_mode='1pol')
        new_gains.update(remove_degen_gains(reds1, pol1_gains, degen_gains=degen_gains, mode=mode, pol_mode='1pol'))
        return new_gains

    # Extract gains and degenerate gains and put into numpy arrays
    gainSols = np.array([gains[ant] for ant in ants])
    if degen_gains is None:
        if mode == 'phase':
            degenGains = np.array([np.zeros_like(gains[ant]) for ant in ants])
        else:  # complex
            degenGains = np.array([np.ones_like(gains[ant]) for ant in ants])
    else:
        degenGains = np.array([degen_gains[ant] for ant in ants])

    # Build matrices for projecting gain degeneracies
    antpos = reds_to_antpos(reds)
    positions = np.array([antpos[ant[0]] for ant in ants])
    if pol_mode == '1pol' or pol_mode == '4pol_minV':
        # In 1pol and 4pol_minV, the phase degeneracies are 1 overall phase and 2 tip-tilt terms
        # Rgains maps gain phases to degenerate parameters (either average phases or phase slopes)
        Rgains = np.hstack((positions, np.ones((positions.shape[0], 1))))
    else:  # pol_mode is '4pol'
        # two columns give sums for two different polarizations
        phasePols = np.vstack((gainPols == antpols[0], gainPols == antpols[1])).T
        Rgains = np.hstack((positions, phasePols))
    # Mgains is like (AtA)^-1 At in linear estimator formalism. It's a normalized estimator of degeneracies
    Mgains = np.linalg.pinv(Rgains.T.dot(Rgains)).dot(Rgains.T)

    # degenToRemove is the amount we need to move in the degenerate subspace
    if mode == 'phase':
        # Fix phase terms only
        degenToRemove = np.einsum('ij,jkl', Mgains, gainSols - degenGains)
        gainSols -= np.einsum('ij,jkl', Rgains, degenToRemove)
    else:  # working on complex data
        # Fix phase terms
        degenToRemove = np.einsum('ij,jkl', Mgains, np.angle(gainSols * np.conj(degenGains)))
        gainSols *= np.exp(np.complex64(-1j) * np.einsum('ij,jkl', Rgains, degenToRemove))
        # Fix abs terms: fixes the mean abs product of gains (as they appear in visibilities)
        for pol in antpols:
            meanSqAmplitude = np.mean([np.abs(g1 * g2) for (a1, p1), g1 in gains.items()
                                       for (a2, p2), g2 in gains.items()
                                       if p1 == pol and p2 == pol and a1 != a2], axis=0)
            degenMeanSqAmplitude = np.mean([(np.ones_like(gains[k1]) if degen_gains is None
                                             else np.abs(degen_gains[k1] * degen_gains[k2]))
                                            for k1 in gains.keys() for k2 in gains.keys()
                                            if k1[1] == pol and k2[1] == pol and k1[0] != k2[0]], axis=0)
            gainSols[gainPols == pol] *= (degenMeanSqAmplitude / meanSqAmplitude)**.5

    # Create new solutions dictionary
    new_gains = {ant: gainSol for ant, gainSol in zip(ants, gainSols)}
    return new_gains


class RedSol():
    '''Object for containing solutions to redundant calibraton, namely gains and
    unique-baseline visibilities, along with a variety of convenience methods.'''
    def __init__(self, reds, gains={}, vis={}, sol_dict={}):
        '''Initializes RedSol object.

        Arguments:
            reds: list of lists of redundant baseline tuples, e.g. (0, 1, 'ee')
            gains: optional dictionary. Maps keys like (1, 'Jee') to complex
                numpy arrays of gains of size (Ntimes, Nfreqs).
            vis: optional dictionary or DataContainer. Maps keys like (0, 1, 'ee')
                to complex numpy arrays of visibilities of size (Ntimes, Nfreqs).
                May only contain at most one visibility per unique baseline group.
            sol_dict: optional dictionary. Maps both gain keys and visibilitity keys
                to numpy arrays. Must be empty if gains or vis is not.
        '''
        if len(sol_dict) > 0:
            if (len(gains) > 0) or (len(vis) > 0):
                raise ValueError('If sol_dict is not empty, both gains and vis must be.')
            self.gains = {key: val for key, val in sol_dict.items() if len(key) == 2}
            vis = {key: val for key, val in sol_dict.items() if len(key) == 3}
        else:
            self.gains = gains
        self.reds = reds
        self.vis = RedDataContainer(vis, reds=self.reds)

    def __getitem__(self, key):
        '''Get underlying gain or visibility, depending on the length of the key.'''
        if len(key) == 3:  # visibility key
            return self.vis[key]
        elif len(key) == 2:  # antenna-pol key
            return self.gains[key]
        else:
            raise KeyError('RedSol keys should be length-2 (for gains) or length-3 (for visibilities).')

    def __setitem__(self, key, value):
        '''Set underlying gain or visibility, depending on the length of the key.'''
        if len(key) == 3:  # visibility key
            self.vis[key] = value
        elif len(key) == 2:  # antenna-pol key
            self.gains[key] = value
        else:
            raise KeyError('RedSol keys should be length-2 (for gains) or length-3 (for visibilities).')

    def __contains__(self, key):
        '''Returns True if key is a gain key or a redundant visbility key, False otherwise.'''
        return (key in self.gains) or (key in self.vis)

    def __iter__(self):
        '''Iterate over gain keys, then iterate over visibility keys.'''
        return chain(self.gains, self.vis)

    def __len__(self):
        '''Returns the total number of entries in self.gains or self.vis.'''
        return len(self.gains) + len(self.vis)

    def keys(self):
        '''Iterate over gain keys, then iterate over visibility keys.'''
        return self.__iter__()

    def values(self):
        '''Iterate over gain values, then iterate over visibility values.'''
        return chain(self.gains.values(), self.vis.values())

    def items(self):
        '''Returns the keys and values of the gains, then over those of the visibilities.'''
        return chain(self.gains.items(), self.vis.items())

    def get(self, key, default=None):
        '''Returns value associated with key, but default if key is not found.'''
        if key in self:
            return self[key]
        else:
            return default

    def make_sol_finite(self):
        '''Replaces nans and infs in this object, see redcal.make_sol_finite() for details.'''
        make_sol_finite(self)

    def remove_degen(self, degen_sol=None, inplace=True):
        """ Removes degeneracies from solutions (or replaces them with those in degen_sol).

        Arguments:
            sol: dictionary (or RedSol) that contains both visibility and gain solutions in the
                {(ind1,ind2,pol): np.array} and {(index,antpol): np.array} formats respectively
            degen_sol: Optional dictionary or RedSol, formatted like sol. Gain amplitudes and phases
                in degen_sol replace the values of sol in the degenerate subspace of redcal. If
                left as None, average gain amplitudes will be 1 and average phase terms will be 0.
                Visibilties in degen_sol are ignored, so this can also be a dictionary of gains.
            inplace: If True, replaces self.vis and self.gains. If False, returns a new RedSol object.
        Returns:
            new_sol: if not inplace, RedSol with degeneracy removal/replacement performed
        """
        old_gains = self.gains
        new_gains = remove_degen_gains(self.reds, old_gains, degen_gains=degen_sol, mode='complex',
                                       pol_mode=parse_pol_mode(self.reds))
        if inplace:
            calibrate_in_place(self.vis, new_gains, old_gains=old_gains)
            self.gains = new_gains
        else:
            new_vis = deepcopy(self.vis)
            calibrate_in_place(new_vis, new_gains, old_gains=old_gains)
            return RedSol(self.reds, gains=new_gains, vis=new_vis)

    def gain_bl(self, bl):
        '''Return gain for baseline bl = (ai, aj).

        Arguments:
            bl: tuple, baseline to be split into antennas indexing gain.

        Returns:
            gain: gi * conj(gj)
        '''
        ai, aj = split_bl(bl)
        return self.gains[ai] * np.conj(self.gains[aj])

    def model_bl(self, bl):
        '''Return visibility data model (gain * vissol) for baseline bl

        Arguments:
            bl: tuple, baseline to return model for

        Returns:
            vis: gi * conj(gj) * vis[bl]
        '''
        return self.gain_bl(bl) * self.vis[bl]

    def calibrate_bl(self, bl, data, copy=True):
        '''Return calibrated data for baseline bl

        Arguments:
            bl: tuple, baseline from which to divide out gains
            data: numpy array of data to calibrate
            copy: if False, apply calibration to data in place

        Returns:
            vis: data / (gi * conj(gj))
        '''
        gij = self.gain_bl(bl)
        if copy:
            return np.divide(data, gij, where=(gij != 0))
        else:
            np.divide(data, gij, out=data, where=(gij != 0))
            return data

    def update_vis_from_data(self, data, wgts={}, reds_to_update=None):
        '''Performs redundant averaging of data using reds and gains stored in this RedSol object and
           stores the result as the redundant solution.

        Arguments:
            data: DataContainer containing visibilities to redundantly average.
            wgts: optional DataContainer weighting visibilities in averaging.
                If not provided, it is assumed that all data are uniformly weighted.
            reds_to_update: list of reds to update, otherwise update all.

        Returns:
            None
        '''
        if reds_to_update is None:
            reds_to_update = self.reds
        else:
            self.vis.build_red_keys(combine_reds(self.reds, reds_to_update))
            self.reds = self.vis.reds
        for grp in reds_to_update:
            self.vis[grp[0]] = np.average([self.calibrate_bl(bl, data[bl]) for bl in grp], axis=0,
                                          weights=([wgts.get(bl, 1) for bl in grp] if len(wgts) > 0 else None))

    def extend_vis(self, data, wgts={}, reds_to_solve=None):
        '''Performs redundant averaging of ubls not already solved for in RedSol.vis
        and adds them to RedSol.vis

        Arguments:
            data: DataContainer containing visibilities to redundantly average.
            wgts: optional DataContainer weighting visibilities in averaging.
                If not provided, it is assumed that all data are uniformly weighted.
            reds_to_solve: subset of reds to update, otherwise update all

        Returns:
            None
        '''
        if reds_to_solve is None:
            unsolved_reds = [gp for gp in self.reds if not gp[0] in self.vis]
            reds_to_solve = filter_reds(unsolved_reds, ants=self.gains.keys())
        self.update_vis_from_data(data, wgts=wgts, reds_to_update=reds_to_solve)

    def extend_gains(self, data, wgts={}, extended_reds=None):
        '''Extend redundant solutions to antennas gains not already solved for
        using redundant baseline solutions in RedSol.vis, adding them to RedSol.gains.

        Arguments:
            data: DataContainer containing visibilities to redundantly average.
            wgts: optional DataContainer weighting visibilities in averaging.
                If not provided, it is assumed that all data are uniformly weighted.
            extended_reds: Broader list of reds to update, otherwise use existing reds.

        Returns:
            None
        '''
        if extended_reds is None:
            extended_reds = self.reds
        gsum = {}
        gwgt = {}
        for grp in extended_reds:
            try:
                u = self.vis[grp[0]]  # RedDataContainer will take care of mapping.
            except(KeyError):
                # no redundant visibility solution for this group, so skip
                continue
            # loop through baselines and select ones that have one solved antenna
            # and one unsolved to solve for.
            for bl in grp:
                a_i, a_j = split_bl(bl)
                if a_i not in self.gains:
                    if a_j not in self.gains:
                        # no solution for either antenna in this baseline, so skip
                        continue
                    _gsum = data[bl] * (u.conj() * self[a_j])
                    _gwgt = np.abs(u)**2 * np.abs(self[a_j])**2
                    if len(wgts) > 0:
                        _gsum *= wgts[bl]
                        _gwgt *= wgts[bl]
                    gsum[a_i] = gsum.get(a_i, 0) + _gsum
                    gwgt[a_i] = gwgt.get(a_i, 0) + _gwgt
                elif a_j not in self.gains:
                    _gsum = data[bl].conj() * (u * self[a_i])
                    _gwgt = np.abs(u)**2 * np.abs(self[a_i])**2
                    if len(wgts) > 0:
                        _gsum *= wgts[bl]
                        _gwgt *= wgts[bl]
                    gsum[a_j] = gsum.get(a_j, 0) + _gsum
                    gwgt[a_j] = gwgt.get(a_j, 0) + _gwgt
        for k in gsum.keys():
            self[k] = np.divide(gsum[k], gwgt[k], where=(gwgt[k] > 0))

    def chisq(self, data, data_wgts, gain_flags=None):
        """Computes chi^2 defined as: chi^2 = sum_ij(|data_ij - model_ij * g_i conj(g_j)|^2 * wgts_ij)
        and also a chisq_per_antenna which is the same sum but with fixed i.

        Arguments:
            data: DataContainer mapping baseline-pol tuples like (0,1,'nn') to complex data of shape (Nt, Nf).
            data_wgts: multiplicative weights with which to combine chisq per visibility. Usually
                equal to (visibility noise variance)**-1.
            gain_flags: optional dictionary mapping ant-pol keys like (1,'Jnn') to a boolean flags waterfall
                with the same shape as the data. Default: None, which means no per-antenna flagging.

        Returns:
            chisq: numpy array with the same shape each visibility of chi^2 calculated as above. If the
                inferred pol_mode from reds (see redcal.parse_pol_mode) is '1pol' or '2pol', this is a
                dictionary mapping antenna polarization (e.g. 'Jnn') to chi^2. Otherwise, there is a single
                chisq (because polarizations mix) and this is a numpy array.
            chisq_per_ant: dictionary mapping ant-pol keys like (1,'Jnn') to chisq per antenna, computed as
                above but keeping i fixed and varying only j.
        """
        split_by_antpol = parse_pol_mode(self.reds) in ['1pol', '2pol']
        chisq, _, chisq_per_ant, _ = utils.chisq(data, self.vis, data_wgts=data_wgts,
                                                 gains=self.gains, gain_flags=gain_flags,
                                                 reds=self.reds, split_by_antpol=split_by_antpol)
        return chisq, chisq_per_ant

    def normalized_chisq(self, data, data_wgts):
        '''Computes chi^2 and chi^2 per antenna with proper normalization per DoF.

        Arguments:
            data: DataContainer mapping baseline-pol tuples like (0,1,'nn') to complex data of shape (Nt, Nf).
            data_wgts: multiplicative weights with which to combine chisq per visibility. Usually
                equal to (visibility noise variance)**-1.

        Returns:
            chisq: chi^2 per degree of freedom for the calibration solution. If the inferred pol_mode from
                reds (see redcal.parse_pol_mode) is '1pol' or '2pol', this is a dictionary mapping antenna
                polarization (e.g. 'Jnn') to chi^2. Otherwise, there is a single chisq (because polarizations
                mix) and this is a numpy array.
            chisq_per_ant: dictionary mapping ant-pol tuples like (1,'Jnn') to the sum of all chisqs for
                visibilities that an antenna participates in, DoF normalized using predict_chisq_per_ant
    '''
        chisq, chisq_per_ant = normalized_chisq(data, data_wgts, self.reds, self.vis, self.gains)
        return chisq, chisq_per_ant


def _check_polLists_minV(polLists):
    """Given a list of unique visibility polarizations (e.g. for each red group), returns whether
    they are all either single identical polarizations (e.g. 'nn') or both cross polarizations
    (e.g. ['ne','en']) so that the 4pol_minV can be assumed."""

    for polList in polLists:
        if len(polList) == 1:
            if split_pol(polList[0])[0] != split_pol(polList[0])[1]:
                return False
        elif len(polList) == 2:
            if polList[0] != conj_pol(polList[1]) or split_pol(polList[0])[0] == split_pol(polList[0])[1]:
                return False
        else:
            return False
    return True


def parse_pol_mode(reds):
    """Based on reds, figures out the pol_mode.

    Args:
        reds: list of list of baselines (with polarizations) considered redundant

    Returns:
        pol_mode: polarization mode of calibration
            '1pol': 1 antpol and 1 vispol (e.g. 'Jnn' and 'nn'). Default.
            '2pol': 2 antpols, no cross-vispols (e.g. 'Jnn','Jee' and 'nn','ee')
            '4pol': 2 antpols, 4 vispols (e.g. 'Jnn','Jee' and 'nn','ne','en','ee')
            '4pol_minV': 2 antpols, 4 vispols in data but assuming V_ne = V_en in model
            'unrecognized_pol_mode': something else
    """

    pols = list(set([bl[2] for bls in reds for bl in bls]))
    antpols = list(set([antpol for pol in pols for antpol in split_pol(pol)]))
    if len(pols) == 1 and len(antpols) == 1:
        return '1pol'
    elif len(pols) == 2 and np.all([split_pol(pol)[0] == split_pol(pol)[1] for pol in pols]):
        return '2pol'
    elif len(pols) == 4 and len(antpols) == 2:
        polLists = [list(set([bl[2] for bl in bls])) for bls in reds]
        polListLens = np.array([len(polList) for polList in polLists])
        if np.all(polListLens == 1) and len(pols) == 4 and len(antpols) == 2:
            return '4pol'
        elif _check_polLists_minV(polLists) and len(pols) == 4 and len(antpols) == 2:
            return '4pol_minV'
        else:
            return 'unrecognized_pol_mode'
    else:
        return 'unrecognized_pol_mode'


class OmnicalSolver(linsolve.LinProductSolver):
    def __init__(self, data, sol0, wgts={}, gain=.3, **kwargs):
        """Set up a nonlinear system of equations of the form g_i * g_j.conj() * V_mdl = V_ij
        to linearize via the Omnical algorithm described in HERA Memo 50
        (scripts/notebook/omnical_convergence.ipynb).

        Args:
            data: Dictionary that maps nonlinear product equations, written as valid python-interpetable
                strings that include the variables in question, to (complex) numbers or numpy arrarys.
                Variables with trailing underscores '_' are interpreted as complex conjugates (e.g. x*y_
                parses as x * y.conj()).
            sol0: Dictionary mapping all variables (as keyword strings) to their starting guess values.
                This is the point that is Taylor expanded around, so it must be relatively close to the
                true chi^2 minimizing solution. In the same format as that produced by
                linsolve.LogProductSolver.solve() or linsolve.LinProductSolver.solve().
            wgts: Dictionary that maps equation strings from data to real weights to apply to each
                equation. Weights are treated as 1/sigma^2. All equations in the data must have a weight
                if wgts is not the default, {}, which means all 1.0s.
            gain: The fractional step made toward the new solution each iteration.  Default is 0.3.
                Values in the range 0.1 to 0.5 are generally safe.  Increasing values trade speed
                for stability.
            **kwargs: keyword arguments of constants (python variables in keys of data that
                are not to be solved for) which are passed to linsolve.LinProductSolver.
        """
        linsolve.LinProductSolver.__init__(self, data, sol0, wgts=wgts, **kwargs)
        self.gain = np.float32(gain)  # float32 to avoid accidentally promoting data to doubles.

    def _get_ans0(self, sol, keys=None):
        '''Evaluate the system of equations given input sol.
        Specify keys to evaluate only a subset of the equations.'''
        if keys is None:
            keys = self.keys
        _sol = {k + '_': v.conj() for k, v in sol.items() if k.startswith('g')}
        _sol.update(sol)
        return {k: eval(k, _sol) for k in keys}

    def solve_iteratively(self, conv_crit=1e-10, maxiter=50, check_every=4, check_after=1,
                          wgt_func=lambda x: 1., verbose=False):
        """Repeatedly solves and updates solution until convergence or maxiter is reached.
        Returns a meta-data about the solution and the solution itself.

        Args:
            conv_crit: A convergence criterion (default 1e-10) below which to stop iterating.
                Converegence is measured L2-norm of the change in the solution of all the variables
                divided by the L2-norm of the solution itself.
            maxiter: An integer maximum number of iterations to perform before quitting. Default 50.
            check_every: Compute convergence and updates weights every Nth iteration (saves computation). Default 4.
            check_after: Start computing convergence and updating weights after the first N iterations.  Default 1.
            wgt_func: a function f(abs^2 * wgt) operating on weighted absolute differences between
                data and model that returns an additional data weighting to apply to when calculating
                chisq and updating parameters. Example: lambda x: np.where(x>0, 5*np.tanh(x/5)/x, 1)
                clamps deviations to 5 sigma. Default is no additional weighting (lambda x: 1.).

        Returns: meta, sol
            meta: a dictionary with metadata about the solution, including
                iter: the number of iterations taken to reach convergence (or maxiter), with dimensions of the data.
                chisq: the chi^2 of the solution produced by the final iteration, with dimensions of the data.
                conv_crit: the convergence criterion evaluated at the final iteration, with dimensions of the data.
            sol: a dictionary of complex solutions with variables as keys, with dimensions of the data.
        """
        sol = self.sol0
        terms = [(linsolve.get_name(gi), linsolve.get_name(gj), linsolve.get_name(uij))
                 for term in self.all_terms for (gi, gj, uij) in term]
        dmdl_u = self._get_ans0(sol)
        abs2_u = {k: np.abs(self.data[k] - dmdl_u[k])**2 * self.wgts[k] for k in self.keys}
        chisq = sum([v * wgt_func(v) for v in abs2_u.values()])
        update = np.where(chisq > 0)
        abs2_u = {k: v[update] for k, v in abs2_u.items()}
        # variables with '_u' are flattened and only include pixels that need updating
        dmdl_u = {k: v[update].flatten() for k, v in dmdl_u.items()}
        # wgts_u hold the wgts the user provides
        wgts_u = {k: (v * np.ones(chisq.shape, dtype=np.float32))[update].flatten()
                  for k, v in self.wgts.items()}
        # clamp_wgts_u adds additional sigma clamping done by wgt_func.
        # abs2_u holds abs(data - mdl)**2 * wgt (i.e. noise-weighted deviations), which is
        # passed to wgt_func to determine any additional weighting (to, e.g., clamp outliers).
        clamp_wgts_u = {k: v * wgt_func(abs2_u[k]) for k, v in wgts_u.items()}
        sol_u = {k: v[update].flatten() for k, v in sol.items()}
        iters = np.zeros(chisq.shape, dtype=int)
        conv = np.ones_like(chisq)
        for i in range(1, maxiter + 1):
            if verbose:
                print('Beginning iteration %d/%d' % (i, maxiter))
            if (i % check_every) == 1:
                # compute data wgts: dwgts = sum(V_mdl^2 / n^2) = sum(V_mdl^2 * wgts)
                # don't need to update data weighting with every iteration
                # clamped weighting is passed to dwgts_u, which is used to update parameters
                dwgts_u = {k: dmdl_u[k] * dmdl_u[k].conj() * clamp_wgts_u[k] for k in self.keys}
                sol_wgt_u = {k: 0 for k in sol.keys()}
                for k, (gi, gj, uij) in zip(self.keys, terms):
                    w = dwgts_u[k]
                    sol_wgt_u[gi] += w
                    sol_wgt_u[gj] += w
                    sol_wgt_u[uij] += w
                dw_u = {k: v[update] * dwgts_u[k] for k, v in self.data.items()}
            sol_sum_u = {k: 0 for k in sol_u.keys()}
            for k, (gi, gj, uij) in zip(self.keys, terms):
                # compute sum(wgts * V_meas / V_mdl)
                numerator = dw_u[k] / dmdl_u[k]
                sol_sum_u[gi] += numerator
                sol_sum_u[gj] += numerator.conj()
                sol_sum_u[uij] += numerator
            new_sol_u = {k: v * ((1 - self.gain) + self.gain * sol_sum_u[k] / sol_wgt_u[k])
                         for k, v in sol_u.items()}
            dmdl_u = self._get_ans0(new_sol_u)
            # check if i % check_every is 0, which is purposely one less than the '1' up at the top of the loop
            if i < maxiter and (i < check_after or (i % check_every) != 0):
                # Fast branch when we aren't expensively computing convergence/chisq
                sol_u = new_sol_u
            else:
                # Slow branch when we compute convergence/chisq
                abs2_u = {k: np.abs(v[update] - dmdl_u[k])**2 * wgts_u[k] for k, v in self.data.items()}
                new_chisq_u = sum([v * wgt_func(v) for v in abs2_u.values()])
                chisq_u = chisq[update]
                gotbetter_u = (chisq_u > new_chisq_u)
                where_gotbetter_u = np.where(gotbetter_u)
                update_where = tuple(u[where_gotbetter_u] for u in update)
                chisq[update_where] = new_chisq_u[where_gotbetter_u]
                iters[update_where] = i
                new_sol_u = {k: np.where(gotbetter_u, v, sol_u[k]) for k, v in new_sol_u.items()}
                deltas_u = [v - sol_u[k] for k, v in new_sol_u.items()]
                conv_u = np.sqrt(sum([(v * v.conj()).real for v in deltas_u])
                                 / sum([(v * v.conj()).real for v in new_sol_u.values()]))
                conv[update_where] = conv_u[where_gotbetter_u]
                for k, v in new_sol_u.items():
                    sol[k][update] = v
                update_u = np.where((conv_u > conv_crit) & gotbetter_u)
                if update_u[0].size == 0 or i == maxiter:
                    meta = {'iter': iters, 'chisq': chisq, 'conv_crit': conv}
                    return meta, sol
                dmdl_u = {k: v[update_u] for k, v in dmdl_u.items()}
                wgts_u = {k: v[update_u] for k, v in wgts_u.items()}
                sol_u = {k: v[update_u] for k, v in new_sol_u.items()}
                abs2_u = {k: v[update_u] for k, v in abs2_u.items()}
                clamp_wgts_u = {k: v * wgt_func(abs2_u[k]) for k, v in wgts_u.items()}
                update = tuple(u[update_u] for u in update)
            if verbose:
                print('    <CHISQ> = %f, <CONV> = %f, CNT = %d', (np.mean(chisq), np.mean(conv), update[0].size))


def _wrap_phs(phs, wrap_pnt=(np.pi / 2)):
    '''Adjust phase wrap point to be [-wrap_pnt, 2pi-wrap_pnt)'''
    return (phs + wrap_pnt) % (2 * np.pi) - wrap_pnt


def _flip_frac(offsets, flipped=set(), flip_pnt=(np.pi / 2)):
    '''Calculate the fraction of (bl1, bl2) pairings an antenna is involved
    in which have large phase offsets.'''
    cnt = {}
    tot = {}
    for (bl1, bl2), off in offsets.items():
        ijmn = split_bl(bl1) + split_bl(bl2)
        num_in_flipped = sum([int(ant in flipped) for ant in ijmn])
        for ant in ijmn:
            tot[ant] = tot.get(ant, 0) + 1
            if off > flip_pnt and num_in_flipped % 2 == 0:
                cnt[ant] = cnt.get(ant, 0) + 1
    flip_frac = [(k, v / tot[k]) for k, v in cnt.items()]
    return flip_frac


def _find_flipped(offsets, flip_pnt=(np.pi / 2), maxiter=100):
    '''Given a dict of (bl1, bl2) keys and phase offset vals, identify
    antennas which are likely to have a np.pi phase offset.'''
    flipped = set()
    for i in range(maxiter):
        flip_frac = _flip_frac(offsets, flipped=flipped, flip_pnt=flip_pnt)
        changed = False
        for (ant, frac) in flip_frac:
            if frac > 0.5:
                changed = True
                if ant in flipped:
                    flipped.remove(ant)
                else:
                    flipped.add(ant)
        if not changed:
            break
    return flipped


def _firstcal_align_bls(bls, freqs, data, norm=True, wrap_pnt=(np.pi / 2)):
    '''Given a redundant group of bls, find per-baseline dly/off params that
    bring them into phase alignment using hierarchical pairing.'''
    fftfreqs = np.fft.fftfreq(freqs.shape[-1], np.median(np.diff(freqs)))
    dtau = fftfreqs[1] - fftfreqs[0]
    grps = [(bl,) for bl in bls]  # start with each bl in its own group
    _data = {bl: data[bl[0]] for bl in grps}
    Ntimes, Nfreqs = data[bls[0]].shape
    times = np.arange(Ntimes)
    dly_off_gps = {}

    def process_pair(gp1, gp2):
        '''Phase-align two groups, recording dly/off in dly_off_gps for gp2
        and the phase-aligned sum in _data. Returns gp1 + gp2, which
        keys the _data dict and represents group for next iteration.'''
        d12 = _data[gp1] * np.conj(_data[gp2])
        if norm:
            ad12 = np.abs(d12)
            np.divide(d12, ad12, out=d12, where=(ad12 != 0))
        vfft = np.fft.fft(d12, axis=1)

        # get interpolated peak and indices
        inds = np.argmax(np.abs(vfft), axis=-1)

        # calculate shifted peak for sub-bin resolution
        k0 = vfft[times, (inds - 1) % Nfreqs]
        k1 = vfft[times, inds]
        k1 = np.where(k1 == 0, 1, k1)  # prevents nans
        k2 = vfft[times, (inds + 1) % Nfreqs]

        alpha1 = (k0 / k1).real
        alpha2 = (k2 / k1).real
        delta1 = alpha1 / (1 - alpha1)
        delta2 = -alpha2 / (1 - alpha2)
        bin_shifts = (delta1 + delta2) / 2 + utils.quinn_tau(delta1 ** 2) - utils.quinn_tau(delta2 ** 2)

        dly = (fftfreqs[inds] + bin_shifts * dtau).reshape(-1, 1)
        phasor = np.exp(np.complex64(2j * np.pi) * dly * freqs)
        off = np.angle(np.sum(d12 / phasor, axis=1, keepdims=True))

        # Now that we know the slope, estimate the remaining phase offset
        dly_off_gps[gp2] = dly, off
        _data[gp1 + gp2] = _data[gp1] + _data[gp2] * phasor * np.exp(np.complex64(1j) * off)
        return gp1 + gp2

    # Main N log N loop
    while len(grps) > 1:
        new_grps = []
        for gp1, gp2 in zip(grps[::2], grps[1::2]):
            new_grps.append(process_pair(gp1, gp2))
        # deal with stragglers
        if len(grps) % 2 == 1:
            new_grps = new_grps[:-1] + [process_pair(new_grps[-1], grps[-1])]
        grps = new_grps
    bl0 = bls[0]  # everything is effectively phase referenced off first bl
    dly_offs = {}
    for gp, (dly, off) in dly_off_gps.items():
        for bl in gp:
            dly0, off0 = dly_offs.get((bl0, bl), (0, 0))
            dly_offs[(bl0, bl)] = (dly0 + dly, off0 + off)
    dly_offs = {k: (v[0], _wrap_phs(v[1], wrap_pnt=wrap_pnt)) for k, v in dly_offs.items()}
    return dly_offs


class RedundantCalibrator:

    def __init__(self, reds, check_redundancy=False):
        """Initialization of a class object for performing redundant calibration with logcal
        and lincal, both utilizing linsolve, and also degeneracy removal.

        Args:
            reds: list of lists of redundant baseline tuples, e.g. (ind1,ind2,pol). The first
                item in each list will be treated as the key for the unique baseline
            check_redundancy: if True, raise an error if the array is not redundantly calibratable,
                even when allowing for an arbitrary number of phase slope degeneracies.
        """
        self._set_reds(reds)
        self.pol_mode = parse_pol_mode(self.reds)

        if check_redundancy:
            nDegens = self.count_degens(assume_redundant=False)
            nDegensExpected = self.count_degens()
            if nDegens != nDegensExpected:
                nPhaseSlopes = len(list(reds_to_antpos(self.reds).values())[0])
                raise ValueError('{} degeneracies found, but {} '.format(nDegens, nDegensExpected)
                                 + 'degeneracies expected (assuming {} phase slopes).'.format(nPhaseSlopes))

    def _set_reds(self, reds):
        '''Sets reds interally, updating self._ubl_to_reds_index and self._ants_in_reds.'''
        self.reds = reds
        self._ubl_to_reds_index = {red[0]: i for i, red in enumerate(self.reds)}
        self._ants_in_reds = set([ant for red in self.reds for bl in red for ant in split_bl(bl)])

    def build_eqs(self, dc=None):
        """Function for generating linsolve equation strings. Optionally takes in a DataContainer to check
        whether baselines in self.reds (or their complex conjugates) occur in the data. Returns a dictionary
        that maps linsolve string to (ant1, ant2, pol) for all visibilities."""
        eqs = {}
        for ubl_index, blgrp in enumerate(self.reds):
            for ant_i, ant_j, pol in blgrp:
                if dc is not None and (ant_i, ant_j, pol) not in dc:
                    raise KeyError('Baseline {} not in provided DataContainer'.format((ant_i, ant_j, pol)))
                params = (ant_i, split_pol(pol)[0], ant_j, split_pol(pol)[1], ubl_index, blgrp[0][2])
                eqs['g_%d_%s * g_%d_%s_ * u_%d_%s' % params] = (ant_i, ant_j, pol)
        return eqs

    def _solver(self, solver, data, wgts={}, detrend_phs=False, **kwargs):
        """Instantiates a linsolve solver for performing redcal.

        Args:
            solver: linsolve solver (e.g. linsolve.LogProductSolver or linsolve.LinProductSolver)
            data: visibility data in the dictionary format {(ant1,ant2,pol): np.array}
            wgts: dictionary of linear weights in the same format as data. Defaults to equal wgts.
            detrend_phs: takes out average phase, useful for logcal
            **kwargs: other keyword arguments passed into the solver for use by linsolve, e.g.
                sparse (use sparse matrices to represent system of equations).

        Returns:
            solver: instantiated solver with redcal equations and weights
        """
        dtype = list(data.values())[0].dtype
        dc = DataContainer(data)
        eqs = self.build_eqs(dc)
        self.phs_avg = {}  # detrend phases within redundant group, used for logcal to avoid phase wraps
        if detrend_phs:
            for grp in self.reds:
                self.phs_avg[grp[0]] = np.exp(-np.complex64(1j) * np.median(np.unwrap([np.log(dc[bl]).imag for bl in grp], axis=0), axis=0))
                for bl in grp:
                    self.phs_avg[bl] = self.phs_avg[grp[0]].astype(dc[bl].dtype)
        d_ls, w_ls = {}, {}
        for eq, key in eqs.items():
            d_ls[eq] = dc[key] * self.phs_avg.get(key, np.float32(1))
        if len(wgts) > 0:
            wc = DataContainer(wgts)
            for eq, key in eqs.items():
                w_ls[eq] = wc[key]
        return solver(data=d_ls, wgts=w_ls, **kwargs)

    def unpack_sol_key(self, k):
        """Turn linsolve's internal variable string into antenna or baseline tuple (with polarization)."""

        if k.startswith('g'):  # 'g' = gain solution
            return (int(k.split('_')[1]), k.split('_')[2])
        else:  # 'u' = unique baseline solution
            return self.reds[int(k.split('_')[1])][0]

    def pack_sol_key(self, k):
        """Turn an antenna or baseline tuple (with polarization) into linsolve's internal variable string."""

        if len(k) == 2:  # 'g' = gain solution
            return 'g_%d_%s' % k
        else:  # 'u' = unique baseline solution
            return 'u_%d_%s' % (self._ubl_to_reds_index[k], k[-1])

    # XXX remove functionality that lives in RedSol
    def compute_ubls(self, data, gains):
        """Given a set of guess gain solutions, return a dictionary of calibrated visbilities
        averged over a redundant group. Not strictly necessary for typical operation."""

        dc = DataContainer(deepcopy(data))
        calibrate_in_place(dc, gains, gain_convention='divide')
        ubl_sols = {}
        for ubl, blgrp in enumerate(self.reds):
            d_gp = [dc[bl] for bl in blgrp]
            ubl_sols[blgrp[0]] = np.average(d_gp, axis=0)
        return ubl_sols

    def firstcal(self, data, freqs, maxiter=100, sparse=False, mode='default', flip_pnt=(np.pi / 2)):
        """Solve for a calibration solution parameterized by a single delay and phase offset
        per antenna using the phase difference between nominally redundant measurements.
        Delays are solved in a single iteration, but phase offsets are solved for
        iteratively to account for phase wraps.

        Args:
            data: visibility data in the dictionary format {(ant1,ant2,pol): np.array}
            freqs: numpy array of frequencies in the data
            maxiter: maximum number of iterations for finding flipped antennas
            sparse: represent the A matrix (visibilities to parameters) sparsely in linsolve
            mode: solving mode passed to the linsolve linear solver ('default', 'lsqr', 'pinv', or 'solve')
                Suggest using 'default' unless solver is having stability (convergence) problems.
                More documentation of modes in linsolve.LinearSolver.solve().
            flip_pnt: cutoff median phase to assign baselines the "majority" polarity group.
                (pi - max_rel_angle() is the cutoff for "minority" group. Must be between 0 and pi/2.

        Returns:
            meta: dictionary of metadata (including delays and suspected antenna flips for each integration)
            sol: RedSol with Ntimes x Nfreqs per-antenna gains solutions of the form
                 np.exp(2j * np.pi * delay * freqs + 1j * offset), as well as visibility
                 solutions formed from redundantly averaged first-caled data.
        """
        Ntimes, Nfreqs = data[self.reds[0][0]].shape
        dlys_offs = {}

        for bls in self.reds:
            if len(bls) < 2:
                continue
            _dly_off = _firstcal_align_bls(bls, freqs, data)
            dlys_offs.update(_dly_off)

        # offsets often have phase wraps and need some finesse around np.pi
        avg_offsets = {k: np.mean(v[1]) for k, v in dlys_offs.items()}  # XXX maybe do per-integration
        flipped = _find_flipped(avg_offsets, flip_pnt=flip_pnt, maxiter=maxiter)

        d_ls = {}
        for (bl1, bl2), (dly, off) in dlys_offs.items():
            ai, aj = split_bl(bl1)
            am, an = split_bl(bl2)
            i, j, m, n = (self.pack_sol_key(k) for k in (ai, aj, am, an))
            eq_key = '%s-%s-%s+%s' % (i, j, m, n)
            n_flipped = sum([int(ant in flipped) for ant in (ai, aj, am, an)])
            if n_flipped % 2 == 0:
                d_ls[eq_key] = np.array((dly, off))
            else:
                d_ls[eq_key] = np.array((dly, _wrap_phs(off + np.pi)))
        ls = linsolve.LinearSolver(d_ls, sparse=sparse)
        sol = ls.solve(mode=mode)
        dlys = {self.unpack_sol_key(k): v[0] for k, v in sol.items()}
        offs = {self.unpack_sol_key(k): v[1] for k, v in sol.items()}
        # add back in antennas in reds but not in the system of equations
        ants = set([ant for red in self.reds for bl in red for ant in utils.split_bl(bl)])
        dlys = {ant: dlys.get(ant, (np.zeros_like(list(dlys.values())[0]))) for ant in ants}
        offs = {ant: offs.get(ant, (np.zeros_like(list(offs.values())[0]))) for ant in ants}

        for ant in flipped:
            offs[ant] = _wrap_phs(offs[ant] + np.pi)

        dtype = np.find_common_type([d.dtype for d in data.values()], [])
        meta = {'dlys': {ant: dly.flatten() for ant, dly in dlys.items()},
                'offs': {ant: off.flatten() for ant, off in offs.items()},
                'polarity_flips': {ant: np.ones(Ntimes, dtype=bool) * bool(ant in flipped) for ant in ants}}
        gains = {ant: np.exp(2j * np.pi * dly * freqs + 1j * offs[ant]).astype(dtype) for ant, dly in dlys.items()}
        sol = RedSol(self.reds, gains=gains)
        sol.update_vis_from_data(data)  # compute vis sols for completeness (though not strictly necessary)
        return meta, sol

    def logcal(self, data, sol0=None, wgts={}, sparse=False, mode='default'):
        """Takes the log to linearize redcal equations and minimizes chi^2.

        Args:
            data: visibility data in the dictionary format {(ant1,ant2,pol): np.array}
            sol0: dictionary that includes all starting (e.g. firstcal) gains in the
                {(ant,antpol): np.array} format. These are divided out of the data before
                logcal and then multiplied back into the returned gains in the solution.
                Missing gains are treated as 1.0s.
            wgts: dictionary of linear weights in the same format as data. Defaults to equal wgts.
            sparse: represent the A matrix (visibilities to parameters) sparsely in linsolve
            mode: solving mode passed to the linsolve linear solver ('default', 'lsqr', 'pinv', or 'solve')
                Suggest using 'default' unless solver is having stability (convergence) problems.
                More documentation of modes in linsolve.LinearSolver.solve().

        Returns:
            meta: empty dictionary (to maintain consistency with related functions)
            sol: dictionary of gain and visibility solutions in the {(index,antpol): np.array}
                and {(ind1,ind2,pol): np.array} formats respectively
        """
        cal_data = {bl: data[bl] for gp in self.reds for bl in gp}
        if sol0 is not None:
            cal_data = {bl: sol0.calibrate_bl(bl, data[bl]) for bl in cal_data}
        ls = self._solver(linsolve.LogProductSolver, cal_data, wgts=wgts, detrend_phs=True, sparse=sparse)
        prms = ls.solve(mode=mode)
        prms = {self.unpack_sol_key(k): v for k, v in prms.items()}
        sol = RedSol(self.reds, sol_dict=prms)
        # put back in phase trend that was taken out with detrend_phs=True
        for ubl_key in sol.vis:
            sol[ubl_key] *= self.phs_avg[ubl_key].conj()
        if sol0 is not None:
            # put back in sol0 gains that were divided out
            for ant in sol.gains:
                sol.gains[ant] *= sol0.gains.get(ant, 1.0)
        return {}, sol

    def lincal(self, data, sol0, wgts={}, sparse=False, mode='default', conv_crit=1e-10, maxiter=50, verbose=False):
        """Taylor expands to linearize redcal equations and iteratively minimizes chi^2.

        Args:
            data: visibility data in the dictionary format {(ant1,ant2,pol): np.array}
            sol0: dictionary of guess gains and unique model visibilities, keyed by antenna tuples
                like (ant,antpol) or baseline tuples like. Gains should include firstcal gains.
            wgts: dictionary of linear weights in the same format as data. Defaults to equal wgts.
            sparse: represent the A matrix (visibilities to parameters) sparsely in linsolve
            conv_crit: maximum allowed relative change in solutions to be considered converged
            maxiter: maximum number of lincal iterations allowed before it gives up
            verbose: print stuff
            mode: solving mode passed to the linsolve linear solver ('default', 'lsqr', 'pinv', or 'solve')
                Suggest using 'default' unless solver is having stability (convergence) problems.
                More documentation of modes in linsolve.LinearSolver.solve().

        Returns:
            meta: dictionary of information about the convergence and chi^2 of the solution
            sol: dictionary of gain and visibility solutions in the {(index,antpol): np.array}
                and {(ind1,ind2,pol): np.array} formats respectively
        """
        sol0pack = {self.pack_sol_key(ant): sol0.gains[ant] for ant in self._ants_in_reds}
        for ubl in self._ubl_to_reds_index.keys():
            sol0pack[self.pack_sol_key(ubl)] = sol0[ubl]
        ls = self._solver(linsolve.LinProductSolver, data, sol0=sol0pack, wgts=wgts, sparse=sparse)
        meta, prms = ls.solve_iteratively(conv_crit=conv_crit, maxiter=maxiter, verbose=verbose, mode=mode)
        prms = {self.unpack_sol_key(k): v for k, v in prms.items()}
        sol = RedSol(self.reds, sol_dict=prms)
        return meta, sol

    def omnical(self, data, sol0, wgts={}, gain=.3, conv_crit=1e-10, maxiter=50, check_every=4, check_after=1, wgt_func=lambda x: 1.):
        """Use the Liu et al 2010 Omnical algorithm to linearize equations and iteratively minimize chi^2.

        Args:
            data: visibility data in the dictionary format {(ant1,ant2,pol): np.array}
            sol0: dictionary of guess gains and unique model visibilities, keyed by antenna tuples
                like (ant,antpol) or baseline tuples like. Gains should include firstcal gains.
            wgts: dictionary of linear weights in the same format as data. Defaults to equal wgts.
            conv_crit: maximum allowed relative change in solutions to be considered converged
            maxiter: maximum number of omnical iterations allowed before it gives up
            check_every: Compute convergence every Nth iteration (saves computation).  Default 4.
            check_after: Start computing convergence only after N iterations.  Default 1.
            gain: The fractional step made toward the new solution each iteration.  Default is 0.3.
                Values in the range 0.1 to 0.5 are generally safe.  Increasing values trade speed
                for stability.
            wgt_func: a function f(abs^2 * wgt) operating on weighted absolute differences between
                data and model that returns an additional data weighting to apply to when calculating
                chisq and updating parameters. Example: lambda x: np.where(x>0, 5*np.tanh(x/5)/x, 1)
                clamps deviations to 5 sigma. Default is no additional weighting (lambda x: 1.).

        Returns:
            meta: dictionary of information about the convergence and chi^2 of the solution
            sol: dictionary of gain and visibility solutions in the {(index,antpol): np.array}
                and {(ind1,ind2,pol): np.array} formats respectively
        """
        sol0pack = {self.pack_sol_key(ant): sol0.gains[ant] for ant in self._ants_in_reds}
        for ubl in self._ubl_to_reds_index.keys():
            sol0pack[self.pack_sol_key(ubl)] = sol0[ubl]
        ls = self._solver(OmnicalSolver, data, sol0=sol0pack, wgts=wgts, gain=gain)
        meta, prms = ls.solve_iteratively(conv_crit=conv_crit, maxiter=maxiter, check_every=check_every, check_after=check_after, wgt_func=wgt_func)
        prms = {self.unpack_sol_key(k): v for k, v in prms.items()}
        sol = RedSol(self.reds, sol_dict=prms)
        return meta, sol

    def count_degens(self, assume_redundant=True):
        """Count the number of degeneracies in this redundant calibrator, given the redundancies and the pol_mode.
        Does not assume coplanarity and instead introduces additional phase slope degeneracies to compensate.

        Args:
            assume_redundant: if True, assume the the array is "redundantly calibrtable" and the only way to get
                extra degneracies is through additional phase slopes (typically 2 per pol for a coplanar array).
                False is slower for large arrays because it has to compute a matrix rank.

        Returns:
            nDegens: the integer number of degeneracies of redundant calibration given the array configuration.
        """
        if assume_redundant:
            nPhaseSlopes = len(list(reds_to_antpos(self.reds).values())[0])  # number of phase slope degeneracies
            if self.pol_mode == '1pol':
                return 1 + 1 + nPhaseSlopes  # 1 amplitude degen, 1 phase degen, N phase slopes
            elif self.pol_mode == '2pol':
                return 2 + 2 + 2 * nPhaseSlopes  # 2 amplitude degens, 2 phase degens, 2N phase slopes
            elif self.pol_mode == '4pol':
                return 2 + 2 + nPhaseSlopes  # 4pol ties phase slopes together, so just N phase slopes
            else:  # '4pol_minV'
                return 2 + 1 + nPhaseSlopes  # 4pol_minV ties overall phase together, so just 1 overall phase
        else:
            dummy_data = DataContainer({bl: np.ones((1, 1), dtype=complex) for red in self.reds for bl in red})
            solver = self._solver(linsolve.LogProductSolver, dummy_data)
            return np.sum([A.shape[1] - np.linalg.matrix_rank(np.dot(np.squeeze(A).T, np.squeeze(A)))
                           for A in [solver.ls_amp.get_A(), solver.ls_phs.get_A()]])


def is_redundantly_calibratable(antpos, bl_error_tol=1.0, require_coplanarity=True):
    """Figures out whether an array is redundantly calibratable.

    Args:
        antpos: dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}.
        bl_error_tol: the largest allowable difference between baselines in a redundant group
            (in the same units as antpos). Normally, this is up to 4x the largest antenna position error.
        require_coplanarity: if True, require that the array have 2 or fewer phase slope degeneracies
            (i.e. that it is a classic monolithic coplanar redundant array)

    Returns:
        boolean: true if the number of 1pol degeneracies is <=4 and thus the array is redundantly calibratable
    """
    reds = get_reds(antpos, pol_mode='1pol', bl_error_tol=bl_error_tol)
    rc = RedundantCalibrator(reds)
    if require_coplanarity:
        if len(list(reds_to_antpos(reds).values())[0]) > 2:  # not a monolithic, coplanar array
            return False
    return (rc.count_degens() == rc.count_degens(assume_redundant=False))


def predict_chisq_per_bl(reds, pol_separable=None):
    '''Predict the expected value of chi^2 for each baselines (equivalently, the
    effective number of degrees of freedom). This is calculated from the logcal
    A and B matrices and their respective data resolution matrices.

    Arguments:
        reds: list of list of baselines (with polarizations) considered redundant
        pol_separable: whether polarizations are separable, if known. If None, will
            be computed.

    Returns:
        predicted_chisq_per_bl: dictionary mapping baseline tuples to the expected
            value of chi^2 = |Vij - gigj*Vi-j|^2/sigmaij^2.
    '''
    if pol_separable is None:
        pol_separable = (parse_pol_mode(reds) == '2pol')
    if pol_separable:
        # pols are separable and can be solved independently for significant speedup
        reds_by_pol = {}
        for gp in reds:
            pol = gp[0][-1]
            reds_by_pol[pol] = reds_by_pol.get(pol, []) + [gp]
        predicted_chisq_per_bl = {}
        for pol, polreds in reds_by_pol.items():
            predicted_chisq_per_bl.update(predict_chisq_per_bl(polreds, pol_separable=False))
        return predicted_chisq_per_bl
    else:
        # pols are not further separable and we need to build full equations
        # doing a quicker breakdown of logcal equations w/o linsolve to
        # avoid spending a lot of time parsing strings
        bls = [bl for red in reds for bl in red]
        ants = {}
        for bl in bls:
            for ant in split_bl(bl):
                ants[ant] = ants.get(ant, len(ants))
        # eqinds = (antpol, antpol, red-group-number) for each bl
        eqinds = [split_bl(bl) + (u,)
                  for u, gp in enumerate(reds) for bl in gp]
        # eqinds = (ant-prm-index, ant-prm-index, red-gp-prm-index) for each bl
        eqinds = [(ants[ai], ants[aj], len(ants) + ug)
                  for ai, aj, ug in eqinds]
        # eqinds = (AtA x/y indices = outer product of prm indices) for each bl
        eqinds = [(np.array([ai, ai, ai, aj, aj, aj, ug, ug, ug]),
                   np.array([ai, aj, ug, ai, aj, ug, ai, aj, ug]))
                  for ai, aj, ug in eqinds]
        nprms = len(ants) + len(reds)
        diag_sums = []
        # loop over amplitude and phase (respectively) "logcal" terms
        for ci, cj, cu in ((1, 1, 1), (1, -1, 1)):
            # coeffs = (AtA x/y wgts = outer product of prm coeffs) for each bl
            coeffs = np.array([ci * ci, ci * cj, ci * cu,
                               cj * ci, cj * cj, cj * cu,
                               cu * ci, cu * cj, cu * cu])
            # build A.T dot A from sparse representation of equations in eqinds
            AtA = np.zeros((nprms, nprms), dtype=float)
            for x, y in eqinds:
                AtA[x, y] += coeffs
            AtAi = np.linalg.pinv(AtA, rcond=1e-12, hermitian=True)
            # compute sum(A.T * AtA^-1 dot A.T dot A, axis=0), using that for each eq
            # in A, eq * (AtA^-1) * eq corresponds to summing over the same x/y
            # indices we built above
            diag_sum = np.array([np.sum(coeffs * AtAi[x, y])
                                for x, y in eqinds])
            diag_sums.append(diag_sum)
        predicted_chisq_per_bl = 1.0 - sum(diag_sums) / 2.0
        return {bl: dof for bl, dof in zip(bls, predicted_chisq_per_bl)}


def predict_chisq_per_red(reds):
    '''Predict the expected value of chi^2 for each redundant baselines group
    (equivalently, the effective number of degrees of freedom).

    Arguments:
        reds: list of list of baselines (with polarizations) considered redundant

    Returns:
        predicted_chisq_per_bl: dictionary mapping unique baseline tuples to the
            expected sum(|Vij - gigj*Vi-j|^2/sigmaij^2) over baselines in a group
    '''
    predicted_chisq_per_bl = predict_chisq_per_bl(reds)
    return {red[0]: np.sum([predicted_chisq_per_bl[bl] for bl in red]) for red in reds}


def predict_chisq_per_ant(reds):
    '''Predict the expected value of chi^2 per antenna (equivalently, the effective
    number of degrees of freedom). The sum over all antennas will twice the total
    DoF, since each baseline has two antennas.

    Arguments:
        reds: list of list of baselines (with polarizations) considered redundant

    Returns:
        predicted_chisq_per_ant: dictionary mapping antenna-pol tuples to the expected
        sum(|Vij - gigj*Vi-j|^2/sigmaij^2) over all baselines including that antenna
    '''
    predicted_chisq_per_bl = predict_chisq_per_bl(reds)
    predicted_chisq_per_ant = {}
    for bl, chisq in predicted_chisq_per_bl.items():
        for ant in split_bl(bl):
            predicted_chisq_per_ant[ant] = predicted_chisq_per_ant.get(ant, 0) + chisq
    return predicted_chisq_per_ant


def normalized_chisq(data, data_wgts, reds, vis_sols, gains):
    '''Computes chi^2 and chi^2 per antenna with proper normalization per DoF. When the only
    source of non-redundancy is noise, these quantities should have expectation values of 1.

    Arguments:
        data: DataContainer mapping baseline-pol tuples like (0,1,'nn') to complex data of
            shape (Nt, Nf).
        data_wgts: multiplicative weights with which to combine chisq per visibility. Usually
            equal to (visibility noise variance)**-1.
        reds: list of lists of redundant baseline tuples, e.g. (0,1,'nn'). The first
            item in each list will be treated as the key for the unique baseline.
        vis_sols: omnical visibility solutions dictionary with baseline-pol tuple keys that are the
            first elements in each of the sub-lists of reds.
        gains: gain dictionary keyed by ant-pol tuples like (1,'Jnn')

    Returns:
        chisq: chi^2 per degree of freedom for the calibration solution. Normalized using noise derived
            from autocorrelations and a number of DoF derived from the reds using predict_chisq_per_ant.
            If the inferred pol_mode from reds (see redcal.parse_pol_mode) is '1pol' or '2pol', this
            is a dictionary mapping antenna polarization (e.g. 'Jnn') to chi^2. Otherwise, there is a
            single chisq (because polarizations mix) and this is a numpy array.
        chisq_per_ant: dictionary mapping ant-pol tuples like (1,'Jnn') to the sum of all chisqs for
            visibilities that an antenna participates in, DoF normalized using predict_chisq_per_ant
    '''
    pol_mode = parse_pol_mode(reds)
    chisq, _, chisq_per_ant, _ = utils.chisq(data, vis_sols, data_wgts=data_wgts, gains=gains,
                                             reds=reds, split_by_antpol=(pol_mode in ['1pol', '2pol']))
    predicted_chisq_per_ant = predict_chisq_per_ant(reds)
    chisq_per_ant = {ant: cs / predicted_chisq_per_ant[ant] for ant, cs in chisq_per_ant.items()}
    if pol_mode in ['1pol', '2pol']:  # in this case, chisq is split by antpol
        for antpol in chisq.keys():
            chisq[antpol] /= np.sum([cspa / 2.0 for ant, cspa in predicted_chisq_per_ant.items()
                                     if antpol in ant], axis=0)
    else:
        chisq /= np.sum(list(predicted_chisq_per_ant.values())) / 2.0
    return chisq, chisq_per_ant


def _get_pol_load_list(pols, pol_mode='1pol'):
    '''Get a list of lists of polarizations to load simultaneously, depending on the polarizations
    in the data and the pol_mode (which can be 1pol, 2pol, 4pol, or 4pol_minV)'''
    if pol_mode in ['1pol', '2pol']:
        pol_load_list = [[pol] for pol in pols if split_pol(pol)[0] == split_pol(pol)[1]]
    elif pol_mode in ['4pol', '4pol_minV']:
        assert len(pols) == 4, 'For 4pol calibration, there must be four polarizations in the data file.'
        pol_load_list = [pols]
    else:
        raise ValueError('Unrecognized pol_mode: {}'.format(pol_mode))
    return pol_load_list


def redundantly_calibrate(data, reds, sol0=None, run_logcal=True, run_omnical=True,
                          remove_degen=True, compute_chisq=True, freqs=None, times_by_bl=None,
                          oc_conv_crit=1e-10, oc_maxiter=500, check_every=10, check_after=50,
                          gain=.4, max_dims=2, use_gpu=False):
    '''Performs all three steps of redundant calibration: firstcal, logcal, and omnical.

    Arguments:
        data: dictionary or DataContainer mapping baseline-pol tuples like (0,1,'nn') to
            complex data of shape. Asummed to have no flags.
        reds: list of lists of redundant baseline tuples, e.g. (0,1,'nn'). The first
            item in each list will be treated as the key for the unique baseline.
        sol0: Optional RedSol. If not default None,
            skips performing firstcal and substitutes this for 'g_firstcal' in the returned meta dictionary.
        run_logcal: Perform logcal before omnical. Default True. If False, use firstcal's sol or sol_0.
        run_omnical: Perform omnical. Default True. If False, will return logcal or firstcal's sol.
        remove_degen: Project out degeneracies, replacing them with sol0 (or internally from firstcal)
        compute_chisq: Add normalized chisq to returned meta dictionary
        freqs: 1D numpy array frequencies in Hz. Optional if inferable from data DataContainer,
            but must be provided if data is a dictionary, if it doesn't have .freqs, or if the
            length of data.freqs is 1.
        times_by_bl: dictionary mapping antenna pairs like (0,1) to float Julian Date. Optional if
            inferable from data DataContainer, but must be provided if data is a dictionary,
            if it doesn't have .times_by_bl, or if the length of any list of times is 1.
        oc_conv_crit: maximum allowed relative change in omnical solutions for convergence
        oc_maxiter: maximum number of omnical iterations allowed before it gives up
        check_every: compute omnical convergence every Nth iteration (saves computation).
        check_after: start computing omnical convergence only after N iterations (saves computation).
        gain: The fractional step made toward the new solution each omnical iteration. Values in the
            range 0.1 to 0.5 are generally safe. Increasing values trade speed for stability.
        max_dims: maximum allowed generalized tip/tilt phase degeneracies of redcal that are fixed
            with remove_degen() and must be later abscaled. None is no limit. 2 is a classically
            "redundantly calibratable" planar array.  More than 2 usually arises with subarrays of
            redundant baselines. Antennas will be excluded from reds to satisfy this.
        use_gpu: Bool default False. If True, use GPU to run omnical. Requires hera_gpu.

    Returns:
        meta: a dictionary of results with the following keywords:
            'filtered_reds': the filtered redundancies (based on max_dims) used internally
            'fc_meta' : dictionary that includes delays and identifies flipped antennas
            'fc_gains': firstcal gains in dictionary keyed by ant-pol tuples like (1,'Jnn').
                Gains are Ntimes x Nfreqs gains but fully described by a per-antenna delay.
            'omni_meta': dictionary of information about the omnical convergence and chi^2 of the solution
            'chisq': chi^2 per degree of freedom for the omnical solution. Normalized using noise derived
                from autocorrelations. If the inferred pol_mode from reds (see redcal.parse_pol_mode) is
                '1pol' or '2pol', this is a dictionary mapping antenna polarization (e.g. 'Jnn') to chi^2.
                Otherwise, there is a single chisq (because polarizations mix) and this is a numpy array.
            'chisq_per_ant': dictionary mapping ant-pol tuples like (1,'Jnn') to the average chisq
                for all visibilities that an antenna participates in.
        sol: a RedSol object containing the final calibration solution
    '''
    meta = {}  # dictionary of metadata
    meta['filtered_reds'] = filter_reds(reds, max_dims=max_dims)
    if use_gpu:
        from hera_gpu.redcal import RedundantCalibratorGPU
        rc = RedundantCalibratorGPU(meta['filtered_reds'])
    else:
        rc = RedundantCalibrator(meta['filtered_reds'])
    if freqs is None:
        freqs = data.freqs
    if times_by_bl is None:
        times_by_bl = data.times_by_bl
    red_bls = [bl for red in reds for bl in red if bl in data]

    # perform firstcal if no sol0 is provided
    if sol0 is None:
        meta['fc_meta'], sol0 = rc.firstcal(data, freqs)
    else:
        meta['fc_meta'] = None
    meta['fc_gains'] = sol0.gains

    # perform logcal
    if run_logcal:
        _, sol = rc.logcal(data, sol0=sol0)
    else:
        sol = sol0

    # calculate data_wgts for omnical or calculating chisq
    if run_omnical or compute_chisq:
        dts_by_bl = DataContainer({bl: infer_dt(times_by_bl, bl, default_dt=SEC_PER_DAY**-1) * SEC_PER_DAY for bl in red_bls})
        data_wgts = DataContainer({bl: predict_noise_variance_from_autos(bl, data, dt=dts_by_bl[bl])**-1 for bl in red_bls})

    # perform omnical
    if run_omnical:
        meta['omni_meta'], sol = rc.omnical(data, sol, wgts=data_wgts, conv_crit=oc_conv_crit, maxiter=oc_maxiter,
                                            check_every=check_every, check_after=check_after, gain=gain)

    # remove degneracies using firstcal or sol0
    if remove_degen:
        sol.remove_degen(degen_sol=sol0, inplace=True)

    # compute chisqs
    if compute_chisq:
        meta['chisq'], meta['chisq_per_ant'] = sol.normalized_chisq(data, data_wgts)
    return meta, sol


def expand_omni_vis(sol, expanded_reds, data, nsamples=None, chisq=None, chisq_per_ant=None):
    '''Updates sol by solving for unique visibilities not in sol.vis but for which there is at
    least one baseline with both antennas in sol.gains. Constructs inverse-variance weights from
    noise inferred from autocorrelations (and nsamples, if provided).

    Arguments:
        sol: RedSol with vis and gains, which are used to update its vis.
            Both sol.vis.reds and sol.reds will be updated.
        expanded_reds: List of list of redundant baseline tuples, of which sol.reds is a subset.
        data: DataContainer mapping baseline-pol tuples like (0,1,'nn') to complex visibility data waterfalls.
        nsamples: Optional DataContainer mapping baseline-pol tuples like (0,1,'nn') to waterfalls of number of samples.
            If None, treat all data as having nsamples = 1.
        chisq: chi^2 to update, if desired. See normalized_chisq() for more info.
        chisq_per_ant: per-antenna chi^2 to update, if desired. See normalized_chisq() for more info.
    '''
    # figure out which reds are solvable using the baselines for which we have gains for both antennas
    solved_ants = set(sol.gains.keys())
    good_ants_reds = filter_reds(expanded_reds, ants=solved_ants)
    reds_to_solve = [red for red in good_ants_reds if not np.any([bl in sol.vis for bl in red])]
    data_bls_to_use = [bl for red in reds_to_solve for bl in red
                       if (split_bl(bl)[0] in solved_ants) and (split_bl(bl)[1] in solved_ants)]

    if len(reds_to_solve) > 0:
        # figure out weights for only the absolutely necessary baselines (to save memory)
        if nsamples is None:
            nsamples = DataContainer({bl: 1.0 for bl in data_bls_to_use})
        dts_by_bl = DataContainer({bl: infer_dt(data.times_by_bl, bl, default_dt=SEC_PER_DAY**-1) * SEC_PER_DAY for bl in data_bls_to_use})
        data_wgts = DataContainer({bl: predict_noise_variance_from_autos(bl, data, dt=dts_by_bl[bl])**-1 * nsamples[bl] for bl in data_bls_to_use})

        # extend sol and re-solve for chi^2 if desired
        sol.extend_vis(data, wgts=data_wgts, reds_to_solve=reds_to_solve)
        if (chisq is not None) or (chisq_per_ant is not None):
            chisq, chisq_per_ant = normalized_chisq(data, data_wgts, good_ants_reds, sol.vis, sol.gains)

    # update keys
    sol.vis.build_red_keys(expanded_reds)


def expand_omni_gains(sol, expanded_reds, data, nsamples=None, chisq_per_ant=None):
    '''Updates sol by solving for gains involved in visibilities with at least one solved-for gain.
    This is performed iteratively until all antennas that can be solved for have been. Constructs
    inverse-variance weights from noise inferred from autocorrelations (and nsamples, if provided).

    Arguments:
        sol: RedSol with vis and gains, which are used to update its vis.
            Both sol.vis.reds and sol.reds will be updated.
        expanded_reds: List of list of redundant baseline tuples, of which sol.reds is a subset.
        data: DataContainer mapping baseline-pol tuples like (0,1,'nn') to complex visibility data waterfalls.
        nsamples: Optional DataContainer mapping baseline-pol tuples like (0,1,'nn') to waterfalls of number of samples.
            If None, treat all data as having nsamples = 1.
        chisq_per_ant: per-antenna chi^2 to update, if desired. Only newly-solved-for antennas will be updated.
            See normalized_chisq() for more info.
    '''
    while True:
        bls_to_use = set([bl for red in expanded_reds for bl in red if ((red[0] in sol.vis)
                          and ((split_bl(bl)[0] not in sol.gains) ^ (split_bl(bl)[1] not in sol.gains)))])
        if len(bls_to_use) == 0:
            break  # iterate to also solve for ants only found in bls with other ex_ants

        # create data subset and get data_wgts for just that subset of data to save memory
        if nsamples is None:
            nsamples_here = DataContainer({bl: 1.0 for bl in bls_to_use})
        else:
            nsamples_here = nsamples
        data_subset = DataContainer({bl: data[bl] for bl in bls_to_use})
        dts_by_bl = DataContainer({bl: infer_dt(data.times_by_bl, bl, default_dt=SEC_PER_DAY**-1) * SEC_PER_DAY for bl in bls_to_use})
        data_wgts = DataContainer({bl: predict_noise_variance_from_autos(bl, data, dt=dts_by_bl[bl])**-1 * nsamples_here[bl] for bl in bls_to_use})

        # preliminaries for updating chi^2 using only baselines where at least one antenna is good
        if chisq_per_ant is not None:
            bls_for_chisq = [bl for red in expanded_reds for bl in red if ((red[0] in sol.vis)
                             and ((split_bl(bl)[0] in sol.gains) | (split_bl(bl)[1] in sol.gains)))]
            reds_for_chisq = filter_reds(expanded_reds, bls=bls_for_chisq)
            predicted_chisq_per_ant = predict_chisq_per_ant(reds_for_chisq)

        # expand gains
        sol.extend_gains(data_subset, wgts=data_wgts, extended_reds=filter_reds(expanded_reds, bls=bls_to_use))

        # update chi^2 if desired
        if chisq_per_ant is not None:
            _, _, cspa, _ = utils.chisq(data_subset, sol.vis, data_wgts=data_wgts, gains=sol.gains, reds=expanded_reds)
            for ant, cs in cspa.items():
                if ant not in chisq_per_ant:
                    chisq_per_ant[ant] = cs / predicted_chisq_per_ant[ant]
                    chisq_per_ant[ant][~np.isfinite(cs)] = np.zeros_like(cs[~np.isfinite(cs)])


def _init_redcal_meta_dict(nTimes, nFreqs, ants, pol_load_list):
    '''Helper for redcal_iteration. Creates dictionary to contain firstcal and omnical metas for the full file.'''
    redcal_meta = {}
    redcal_meta['fc_meta'] = {'dlys': {ant: np.full(nTimes, np.nan) for ant in ants}}
    redcal_meta['fc_meta']['polarity_flips'] = {ant: np.full(nTimes, np.nan) for ant in ants}
    redcal_meta['omni_meta'] = {'chisq': {str(pols): np.zeros((nTimes, nFreqs), dtype=float) for pols in pol_load_list}}
    redcal_meta['omni_meta']['iter'] = {str(pols): np.zeros((nTimes, nFreqs), dtype=int) for pols in pol_load_list}
    redcal_meta['omni_meta']['conv_crit'] = {str(pols): np.zeros((nTimes, nFreqs), dtype=float) for pols in pol_load_list}
    return redcal_meta


def _update_redcal_meta(redcal_meta, meta_slice, tSlice, fSlice, pols):
    '''Helper for redcal_iteration. Updates a subset of redcal_meta using meta_slice.'''
    # update firstcal metadata
    for ant in meta_slice['fc_meta']['dlys']:
        redcal_meta['fc_meta']['dlys'][ant][tSlice] = meta_slice['fc_meta']['dlys'][ant]
        redcal_meta['fc_meta']['polarity_flips'][ant][tSlice] = meta_slice['fc_meta']['polarity_flips'][ant]

    # update omnical metadata
    redcal_meta['omni_meta']['chisq'][str(pols)][tSlice, fSlice] = meta_slice['omni_meta']['chisq']
    redcal_meta['omni_meta']['iter'][str(pols)][tSlice, fSlice] = meta_slice['omni_meta']['iter']
    redcal_meta['omni_meta']['conv_crit'][str(pols)][tSlice, fSlice] = meta_slice['omni_meta']['conv_crit']


def count_redundant_nsamples(nsamples, reds, good_ants=None):
    '''Computes a nsamples RedDataContainer from a full nsamples DataContainer and a set of reds.

    Arguments:
        nsamples: DataContainer mapping baseline tuples to nsamples waterfalls.
        reds: List of list of redundant baseline tuples
        good_ants: optional list (or iterable that supports "in") of ant-pol tuples where both
            antennas in a baseline must be good for that nsamples to be included. If None (default),
            all baselines will be included.

    Returns:
        red_nsamples: RedDataContainer of result with red_nsamples.reds set to the input reds
    '''
    red_nsamples = {}
    for red in reds:
        good_bls = [bl for bl in red if (good_ants is None) or ((split_bl(bl)[0] in good_ants) and (split_bl(bl)[1] in good_ants))]
        red_nsamples[red[0]] = np.sum([nsamples[bl] for bl in good_bls], axis=0)
    return RedDataContainer(red_nsamples, reds=reds)


def redcal_iteration(hd, nInt_to_load=None, pol_mode='2pol', bl_error_tol=1.0, ex_ants=[],
                     solar_horizon=0.0, flag_nchan_low=0, flag_nchan_high=0,
                     oc_conv_crit=1e-10, oc_maxiter=500, check_every=10, check_after=50,
                     gain=.4, max_dims=2, verbose=False, **filter_reds_kwargs):
    '''Perform redundant calibration (firstcal, logcal, and omnical) an entire HERAData object, loading only
    nInt_to_load integrations at a time and skipping and flagging times when the sun is above solar_horizon.

    Arguments:
        hd: HERAData object, instantiated with the datafile or files to calibrate. Must be loaded using uvh5.
            Assumed to have no prior flags.
        nInt_to_load: number of integrations to load and calibrate simultaneously. Default None loads all integrations.
            Lower numbers save memory, but incur a CPU overhead.
        pol_mode: polarization mode of redundancies. Can be '1pol', '2pol', '4pol', or '4pol_minV'.
            See recal.get_reds for more information.
        bl_error_tol: the largest allowable difference between baselines in a redundant group
            (in the same units as antpos). Normally, this is up to 4x the largest antenna position error.
        ex_ants: list of antennas to exclude from calibration and flag. Can be either antenna numbers or
            antenna-polarization tuples. In the former case, all pols for an antenna will be excluded.
        solar_horizon: float, Solar altitude flagging threshold [degrees]. When the Sun is above
            this altitude, calibration is skipped and the integrations are flagged.
        flag_nchan_low: integer number of channels at the low frequency end of the band to always flag (default 0)
        flag_nchan_high: integer number of channels at the high frequency end of the band to always flag (default 0)
        oc_conv_crit: maximum allowed relative change in omnical solutions for convergence
        oc_maxiter: maximum number of omnical iterations allowed before it gives up
        check_every: compute omnical convergence every Nth iteration (saves computation).
        check_after: start computing omnical convergence only after N iterations (saves computation).
        gain: The fractional step made toward the new solution each omnical iteration. Values in the
            range 0.1 to 0.5 are generally safe. Increasing values trade speed for stability.
        max_dims: maximum allowed generalized tip/tilt phase degeneracies of redcal that are fixed
            with remove_degen() and must be later abscaled. None is no limit. 2 is a classically
            "redundantly calibratable" planar array.  More than 2 usually arises with subarrays of
            redundant baselines. Antennas will be excluded from reds to satisfy this.
        verbose: print calibration progress updates
        filter_reds_kwargs: additional filters for the redundancies (see redcal.filter_reds for documentation)

    Returns:
        redcal_meta: dictionary of 'fc_meta' and 'omni_meta' dictionaries, which contain informaiton about about
            firstcal delays, polarity flips, omnical chisq, omnical iterations, and omnical convergence
        hc_first: HERACal object containing the results of firstcal. Flags are all False for antennas calibrated.
            Qualities (usually chisq_per_ant) are all zeros. Total qualities (usually chisq) is None.
        hc_omni: HERACal object containing the results of omnical. Flags arise from NaNs in log/omnical or ex_ants.
            Qualities are chisq_per_ant and total qualities are chisq.
        hd_vissol: HERAData object containing omnical visibility solutions. DataContainers (data, flags, nsamples)
            can be extracted using hd_vissol.build_datacontainers().
    '''
    if nInt_to_load is None:
        if hd.data_array is None:  # if data loading hasn't happened yet, load the whole file
            hd.read()
        if hd.times is None:  # load metadata into HERAData object if necessary
            for key, value in hd.get_metadata_dict().items():
                setattr(hd, key, value)

    # get basic antenna, polarization, and observation info
    nTimes, nFreqs = len(hd.times), len(hd.freqs)
    fSlice = slice(flag_nchan_low, nFreqs - flag_nchan_high)
    antpols = list(set([ap for pol in hd.pols for ap in split_pol(pol)]))
    ant_nums = np.unique(np.append(hd.ant_1_array, hd.ant_2_array))
    ants = [(ant, antpol) for ant in ant_nums for antpol in antpols]
    pol_load_list = _get_pol_load_list(hd.pols, pol_mode=pol_mode)

    # get full set of reds and filtered reds to calibrate
    all_reds = get_reds({ant: hd.antpos[ant] for ant in ant_nums}, bl_error_tol=bl_error_tol,
                        pol_mode=pol_mode, pols=set([pol for pols in pol_load_list for pol in pols]))
    filtered_reds = filter_reds(all_reds, ex_ants=ex_ants, antpos=hd.antpos, **filter_reds_kwargs)

    # initialize HERAData and HERACal to contain the full set of data and metadata
    hd_vissol = HERAData(hd.filepaths, upsample=hd.upsample, downsample=hd.downsample)
    hc_first = hd_vissol.init_HERACal()
    hc_omni = hd_vissol.init_HERACal()
    hd_vissol.read(bls=[red[0] for red in all_reds], return_data=False)
    hd_vissol.empty_arrays()

    # setup metadata dictionaries
    redcal_meta = _init_redcal_meta_dict(nTimes, nFreqs, ants, pol_load_list)

    # solar flagging
    lat, lon, alt = hd.telescope_location_lat_lon_alt_degrees
    solar_alts = utils.get_sun_alt(hd.times, latitude=lat, longitude=lon)
    solar_flagged = solar_alts > solar_horizon
    if verbose and np.any(solar_flagged):
        print(len(hd.times[solar_flagged]), 'integrations flagged due to sun above', solar_horizon, 'degrees.')
    if np.all(solar_flagged):
        return redcal_meta, hc_first, hc_omni, hd_vissol

    # loop over polarizations and times, performing partial loading if desired
    for pols in pol_load_list:
        if verbose:
            print('Now calibrating', pols, 'polarization(s)...')
        reds = filter_reds(filtered_reds, ex_ants=ex_ants, pols=pols)

        # figure out time slices to load
        if nInt_to_load is None:
            nInt_to_load = nTimes
        start_ind = np.min(np.arange(nTimes)[~solar_flagged])
        stop_ind = np.max(np.arange(nTimes)[~solar_flagged])
        tSlices = [slice(start_ind + i, min(start_ind + i + nInt_to_load, stop_ind + 1)) for i in range(0, nTimes, nInt_to_load)]

        for tSlice in tSlices:
            # get DataContainers, performing i/o if necessary
            if verbose:
                print('    Now calibrating times', hd.times[tSlice][0], 'through', hd.times[tSlice][-1], '...')
            if nInt_to_load == nTimes:  # don't perform partial I/O
                data, _, nsamples = hd.build_datacontainers()  # this may contain unused polarizations, but that's OK
                for bl in data:
                    data[bl] = data[bl][tSlice, fSlice]  # cut down size of DataContainers to match unflagged indices
                    nsamples[bl] = nsamples[bl][tSlice, fSlice]
            else:  # perform partial i/o
                data, _, nsamples = hd.read(time_range=(hd.times[tSlice][0], hd.times[tSlice][-1]), frequencies=hd.freqs[fSlice], polarizations=pols)

            # run redundant calibration
            meta, sol = redundantly_calibrate(data, reds, freqs=hd.freqs[fSlice], times_by_bl=hd.times_by_bl,
                                              run_logcal=True, run_omnical=True,
                                              oc_conv_crit=oc_conv_crit, oc_maxiter=oc_maxiter,
                                              check_every=check_every, check_after=check_after,
                                              max_dims=max_dims, gain=gain)

            # solve for visibilities excluded from reds but with baselines with both gains in sol. These are not flagged.
            all_reds_this_pol = filter_reds(all_reds, pols=pols)
            expand_omni_vis(sol, all_reds_this_pol, data, nsamples, chisq=meta['chisq'], chisq_per_ant=meta['chisq_per_ant'])

            # rekey sol.vis so that iterating over it will use red[0] in all_reds
            sol.vis = RedDataContainer({red[0]: sol.vis[red[0]] for red in all_reds_this_pol if red[0] in sol.vis}, reds=all_reds_this_pol)

            # make dicts and data containers not in meta or sol. Everything not yet solved for stays flagged.
            first_flags = {ant: np.zeros_like(g, dtype=bool) for ant, g in meta['fc_gains'].items()}
            omni_flags = {ant: ~np.isfinite(g) for ant, g in sol.gains.items()}
            vissol_flags = RedDataContainer({bl: ~np.isfinite(v) for bl, v in sol.vis.items()}, reds=sol.vis.reds)
            vissol_nsamples = count_redundant_nsamples(nsamples, all_reds_this_pol, good_ants=sol.gains)
            sol.make_sol_finite()

            # try to solve for gains on antennas excluded from calibration, but keep them flagged
            expand_omni_gains(sol, all_reds_this_pol, data, nsamples, chisq_per_ant=meta['chisq_per_ant'])

            # try one more time to expand visibilities, keeping new ones flagged, but don't update chisq or chisq_per_ant
            expand_omni_vis(sol, all_reds_this_pol, data, nsamples)
            sol.make_sol_finite()

            # update various objects containing information about the full file
            hc_first.update(gains=meta['fc_gains'], flags=first_flags, tSlice=tSlice, fSlice=fSlice)
            total_qual = meta['chisq']
            if pol_mode in ['4pol', '4pol_minV']:  # in which case meta['chisq'] is a numpy array
                total_qual = {ap: meta['chisq'] for ap in hc_omni.pols}
            hc_omni.update(gains=sol.gains, flags=omni_flags, quals=meta['chisq_per_ant'], total_qual=total_qual, tSlice=tSlice, fSlice=fSlice)
            hd_vissol.update(data=sol.vis, flags=vissol_flags, nsamples=vissol_nsamples, tSlice=tSlice, fSlice=fSlice)
            _update_redcal_meta(redcal_meta, meta, tSlice, fSlice, pols)

    return redcal_meta, hc_first, hc_omni, hd_vissol


def redcal_run(input_data, firstcal_ext='.first.calfits', omnical_ext='.omni.calfits',
               omnivis_ext='.omni_vis.uvh5', meta_ext='.redcal_meta.hdf5', iter0_prefix='', outdir=None,
               metrics_files=[], a_priori_ex_ants_yaml=None, clobber=False, nInt_to_load=None,
               upsample=False, downsample=False, pol_mode='2pol', bl_error_tol=1.0, ex_ants=[],
               ant_z_thresh=4.0, max_rerun=5, solar_horizon=0.0, flag_nchan_low=0, flag_nchan_high=0,
               oc_conv_crit=1e-10, oc_maxiter=500, check_every=10,
               check_after=50, gain=.4, max_dims=2, add_to_history='',
               verbose=False, **filter_reds_kwargs):
    '''Perform redundant calibration (firstcal, logcal, and omnical) an uvh5 data file, saving firstcal and omnical
    results to calfits and uvh5. Uses partial io if desired, performs solar flagging, and iteratively removes antennas
    with high chi^2, rerunning calibration as necessary.

    Arguments:
        input_data: path to visibility data file to calibrate or HERAData object
        firstcal_ext: string to replace file extension of input_data for saving firstcal calfits
        omnical_ext: string to replace file extension of input_data for saving omnical calfits
        omnivis_ext: string to replace file extension of input_data for saving omnical visibilities as uvh5
        meta_ext: string to replace file extension of input_data for saving metadata as hdf5
        iter0_prefix: if not '', save the omnical results with this prefix appended to each file after the 0th
            iteration, but only if redcal has found any antennas to exclude and re-run without
        outdir: folder to save data products. If None, will be the same as the folder containing input_data
        metrics_files: path or list of paths to file(s) containing ant_metrics or auto_metrics readable by
            hera_qm.metrics_io.load_metric_file. Used for finding ex_ants and is combined with antennas
            excluded via ex_ants.
        a_priori_ex_ants_yaml : path to YAML with antenna flagging information parsable by
            hera_qm.metrics_io.read_a_priori_ant_flags(). Frequency and time flags in the YAML
            are ignored. Flags are combined with ant_metrics's xants and ex_ants. If any
            polarization is flagged for an antenna, all polarizations are flagged.
        clobber: if True, overwrites existing files for the firstcal and omnical results
        nInt_to_load: number of integrations to load and calibrate simultaneously. Default None loads all integrations.
            Lower numbers save memory, but incur a CPU overhead.
        upsample: if True, upsample baseline-dependent-averaged data file to highest temporal resolution
        downsample: if True, downsample baseline-dependent-averaged data file to lowest temporal resolution
        pol_mode: polarization mode of redundancies. Can be '1pol', '2pol', '4pol', or '4pol_minV'.
            See recal.get_reds for more information.
        bl_error_tol: the largest allowable difference between baselines in a redundant group
            (in the same units as antpos). Normally, this is up to 4x the largest antenna position error.
        ex_ants: list of antennas to exclude from calibration and flag. Can be either antenna numbers or
            antenna-polarization tuples. In the former case, all pols for an antenna will be excluded.
        ant_z_thresh: threshold of modified z-score (like number of sigmas but with medians) for chi^2 per
            antenna above which antennas are thrown away and calibration is re-run iteratively. Z-scores are
            computed independently for each antenna polarization, but either polarization being excluded
            triggers the entire antenna to get flagged (when multiple polarizations are calibrated)
        max_rerun: maximum number of times to run redundant calibration
        solar_horizon: float, Solar altitude flagging threshold [degrees]. When the Sun is above
            this altitude, calibration is skipped and the integrations are flagged.
        flag_nchan_low: integer number of channels at the low frequency end of the band to always flag (default 0)
        flag_nchan_high: integer number of channels at the high frequency end of the band to always flag (default 0)
        oc_conv_crit: maximum allowed relative change in omnical solutions for convergence
        oc_maxiter: maximum number of omnical iterations allowed before it gives up
        check_every: compute omnical convergence every Nth iteration (saves computation).
        check_after: start computing omnical convergence only after N iterations (saves computation).
        gain: The fractional step made toward the new solution each omnical iteration. Values in the
            range 0.1 to 0.5 are generally safe. Increasing values trade speed for stability.
        max_dims: maximum allowed generalized tip/tilt phase degeneracies of redcal that are fixed
            with remove_degen() and must be later abscaled. None is no limit. 2 is a classically
            "redundantly calibratable" planar array.  More than 2 usually arises with subarrays of
            redundant baselines. Antennas will be excluded from reds to satisfy this.
        add_to_history: string to add to history of output firstcal and omnical files
        verbose: print calibration progress updates
        filter_reds_kwargs: additional filters for the redundancies (see redcal.filter_reds for documentation)

    Returns:
        redcal_meta: dictionary of 'fc_meta' and 'omni_meta' dictionaries, which contain informaiton about about
            firstcal delays, polarity flips, omnical chisq, omnical iterations, and omnical convergence
        hc_first: HERACal object containing the results of firstcal. Flags are all False for antennas calibrated.
            Qualities (usually chisq_per_ant) are all zeros. Total qualities (usually chisq) is None.
        hc_omni: HERACal object containing the results of omnical. Flags arise from NaNs in log/omnical or ex_ants.
            Qualities are chisq_per_ant and total qualities are chisq.
        hd_vissol: HERAData object containing omnical visibility solutions. DataContainers (data, flags, nsamples)
            can be extracted using hd_vissol.build_datacontainers().
    '''
    if isinstance(input_data, str):
        hd = HERAData(input_data, upsample=upsample, downsample=downsample)
        if nInt_to_load is None:
            hd.read()
    elif isinstance(input_data, HERAData):
        hd = input_data
        hd.upsample = upsample
        hd.downsample = downsample
        input_data = hd.filepaths[0]
    else:
        raise TypeError('input_data must be a single string path to a visibility data file or a HERAData object')

    # parse ex_ants from function, metrics_files, and apriori yamls
    ex_ants = set(ex_ants)
    if metrics_files is not None:
        if isinstance(metrics_files, str):
            metrics_files = [metrics_files]
        if len(metrics_files) > 0:
            from hera_qm.metrics_io import load_metric_file
            for mf in metrics_files:
                metrics = load_metric_file(mf)
                # load from an ant_metrics file
                if 'xants' in metrics:
                    for ant in metrics['xants']:
                        ex_ants.add(ant[0])  # Just take the antenna number, flagging both polarizations
                # load from an auto_metrics file
                elif 'ex_ants' in metrics and 'r2_ex_ants' in metrics['ex_ants']:
                    for ant in metrics['ex_ants']['r2_ex_ants']:
                        ex_ants.add(ant)  # Auto metrics reports just antenna numbers
    if a_priori_ex_ants_yaml is not None:
        from hera_qm.metrics_io import read_a_priori_ant_flags
        ex_ants = ex_ants.union(set(read_a_priori_ant_flags(a_priori_ex_ants_yaml, ant_indices_only=True)))
    high_z_ant_hist = ''

    # setup output
    filename_no_ext = os.path.splitext(os.path.basename(input_data))[0]
    if outdir is None:
        outdir = os.path.dirname(input_data)

    # loop over calibration, removing bad antennas and re-running if necessary
    run_number = 0
    while True:
        # Run redundant calibration
        if verbose:
            print('\nNow running redundant calibration without antennas', list(ex_ants), '...')
        redcal_meta, hc_first, hc_omni, hd_vissol = redcal_iteration(hd, nInt_to_load=nInt_to_load, pol_mode=pol_mode, bl_error_tol=bl_error_tol,
                                                                     ex_ants=ex_ants, solar_horizon=solar_horizon, flag_nchan_low=flag_nchan_low,
                                                                     flag_nchan_high=flag_nchan_high, oc_conv_crit=oc_conv_crit, oc_maxiter=oc_maxiter,
                                                                     check_every=check_every, check_after=check_after, max_dims=max_dims, gain=gain,
                                                                     verbose=verbose, **filter_reds_kwargs)

        # Determine whether to add additional antennas to exclude
        _, _, chisq_per_ant, _ = hc_omni.build_calcontainers()
        z_scores = per_antenna_modified_z_scores({ant: np.nanmedian(cspa) for ant, cspa in chisq_per_ant.items()
                                                  if (ant[0] not in ex_ants) and not np.all(cspa == 0)})
        n_ex = len(ex_ants)
        for ant, score in z_scores.items():
            if (score >= ant_z_thresh):
                ex_ants.add(ant[0])
                bad_ant_str = 'Throwing out antenna ' + str(ant[0]) + ' for a z-score of ' + str(score) + ' on polarization ' + str(ant[1]) + '.\n'
                high_z_ant_hist += bad_ant_str
                if verbose:
                    print(bad_ant_str)
        run_number += 1
        if len(ex_ants) == n_ex or run_number >= max_rerun:
            break

        # If there is going to be a re-run and if iter0_prefix is not the empty string, then save the iter0 results.
        if run_number == 1 and len(iter0_prefix) > 0:
            out_prefix = os.path.join(outdir, filename_no_ext + iter0_prefix)
            _add_to_history = utils.history_string(add_to_history + '\n' + 'Iteration 0 Results.\n')
            for h in [hc_first, hc_omni, hd_vissol]:
                h.history += _add_to_history
            hc_first.write_calfits(out_prefix + firstcal_ext, clobber=clobber)
            hc_omni.write_calfits(out_prefix + omnical_ext, clobber=clobber)
            hd_vissol.write_uvh5(out_prefix + omnivis_ext, clobber=clobber)
            save_redcal_meta(out_prefix + meta_ext, redcal_meta['fc_meta'], redcal_meta['omni_meta'], hd.freqs,
                             hd.times, hd.lsts, hd.antpos, hd.history + _add_to_history, clobber=clobber)

    # output results files
    out_prefix = os.path.join(outdir, filename_no_ext)
    _add_to_history = utils.history_string(add_to_history + '\n' + high_z_ant_hist)
    for h in [hc_first, hc_omni, hd_vissol]:
        h.history += _add_to_history
    hc_first.write_calfits(out_prefix + firstcal_ext, clobber=clobber)
    hc_omni.write_calfits(out_prefix + omnical_ext, clobber=clobber)
    hd_vissol.write_uvh5(out_prefix + omnivis_ext, clobber=clobber)
    save_redcal_meta(out_prefix + meta_ext, redcal_meta['fc_meta'], redcal_meta['omni_meta'], hd.freqs,
                     hd.times, hd.lsts, hd.antpos, hd.history + _add_to_history, clobber=clobber)

    return redcal_meta, hc_first, hc_omni, hd_vissol


def redcal_argparser():
    '''Arg parser for commandline operation of redcal_run'''
    a = argparse.ArgumentParser(description="Redundantly calibrate a file using hera_cal.redcal. This includes firstcal, logcal, and omnical. \
                                Iteratively re-runs by flagging antennas with large chi^2. Saves the result to calfits and uvh5 files.")
    a.add_argument("input_data", type=str, help="path to uvh5 visibility data file to calibrate.")
    a.add_argument("--firstcal_ext", default='.first.calfits', type=str, help="string to replace file extension of input_data for saving firstcal calfits")
    a.add_argument("--omnical_ext", default='.omni.calfits', type=str, help="string to replace file extension of input_data for saving omnical calfits")
    a.add_argument("--omnivis_ext", default='.omni_vis.uvh5', type=str, help="string to replace file extension of input_data for saving omnical visibilities as uvh5")
    a.add_argument("--meta_ext", default='.redcal_meta.hdf5', type=str, help="string to replace file extension of input_data for saving metadata as hdf5")
    a.add_argument("--outdir", default=None, type=str, help="folder to save data products. Default is the same as the folder containing input_data")
    a.add_argument("--iter0_prefix", default='', type=str, help="if not default '', save the omnical results with this prefix appended to each file after the 0th iteration, \
                   but only if redcal has found any antennas to exclude and re-run without.")
    a.add_argument("--clobber", default=False, action="store_true", help="overwrites existing files for the firstcal and omnical results")
    a.add_argument("--verbose", default=False, action="store_true", help="print calibration progress updates")

    redcal_opts = a.add_argument_group(title='Runtime Options for Redcal')
    redcal_opts.add_argument("--metrics_files", type=str, nargs='*', default=[], help="path to file containing ant_metrics or auto_metrics readable by hera_qm.metrics_io.load_metric_file. \
                             Used for finding ex_ants and is combined with antennas excluded via ex_ants.")
    redcal_opts.add_argument("--ex_ants", type=int, nargs='*', default=[], help='space-delimited list of antennas to exclude from calibration and flag. All pols for an antenna will be excluded.')
    redcal_opts.add_argument("--a_priori_ex_ants_yaml", type=str, default=None, help='path to YAML file containing a priori ex_ants parsable by hera_qm.metrics_io.read_a_priori_ant_flags()')
    redcal_opts.add_argument("--ant_z_thresh", type=float, default=4.0, help="Threshold of modified z-score for chi^2 per antenna above which antennas are thrown away and calibration is re-run iteratively.")
    redcal_opts.add_argument("--max_rerun", type=int, default=5, help="Maximum number of times to re-run redundant calibration.")
    redcal_opts.add_argument("--solar_horizon", type=float, default=0.0, help="When the Sun is above this altitude in degrees, calibration is skipped and the integrations are flagged.")
    redcal_opts.add_argument("--flag_nchan_low", type=int, default=0, help="integer number of channels at the low frequency end of the band to always flag (default 0)")
    redcal_opts.add_argument("--flag_nchan_high", type=int, default=0, help="integer number of channels at the high frequency end of the band to always flag (default 0)")
    redcal_opts.add_argument("--nInt_to_load", type=int, default=None, help="number of integrations to load and calibrate simultaneously. Lower numbers save memory, but incur a CPU overhead. \
                             Default None loads all integrations.")
    redcal_opts.add_argument("--upsample", default=False, action="store_true", help="Upsample BDA files to the highest temporal resolution.")
    redcal_opts.add_argument("--downsample", default=False, action="store_true", help="Downsample BDA files to the highest temporal resolution.")
    redcal_opts.add_argument("--pol_mode", type=str, default='2pol', help="polarization mode of redundancies. Can be '1pol', '2pol', '4pol', or '4pol_minV'. See recal.get_reds documentation.")
    redcal_opts.add_argument("--bl_error_tol", type=float, default=1.0, help="the largest allowable difference between baselines in a redundant group")
    redcal_opts.add_argument("--min_bl_cut", type=float, default=None, help="cut redundant groups with average baseline lengths shorter than this length in meters")
    redcal_opts.add_argument("--max_bl_cut", type=float, default=None, help="cut redundant groups with average baseline lengths longer than this length in meters")
    redcal_opts.add_argument("--max_dims", type=float, default=2, help='maximum allowed tip/tilt phase degeneracies of redcal. None is no limit. Default 2 is a classically \
                                                                        "redundantly calibratable" planar array. Antennas may be flagged to satisfy this criterion. See redcal.filter_reds() for details.')

    omni_opts = a.add_argument_group(title='Firstcal and Omnical-Specific Options')
    omni_opts.add_argument("--oc_conv_crit", type=float, default=1e-10, help="maximum allowed relative change in omnical solutions for convergence")
    omni_opts.add_argument("--oc_maxiter", type=int, default=500, help="maximum number of omnical iterations allowed before it gives up")
    omni_opts.add_argument("--check_every", type=int, default=10, help="compute omnical convergence every Nth iteration (saves computation).")
    omni_opts.add_argument("--check_after", type=int, default=50, help="start computing omnical convergence only after N iterations (saves computation).")
    omni_opts.add_argument("--gain", type=float, default=.4, help="The fractional step made toward the new solution each omnical iteration. Values in the range 0.1 to 0.5 are generally safe.")

    args = a.parse_args()
    return args

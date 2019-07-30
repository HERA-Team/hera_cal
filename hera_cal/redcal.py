# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

from __future__ import absolute_import, division, print_function

import numpy as np
from copy import deepcopy
import argparse
import os
from six.moves import range, zip
import linsolve

from . import utils
from . import version
from .noise import predict_noise_variance_from_autos
from .datacontainer import DataContainer
from .utils import split_pol, conj_pol, split_bl, reverse_bl, join_bl, join_pol, comply_pol
from .io import HERAData, HERACal, write_cal, write_vis
from .apply_cal import calibrate_in_place


SEC_PER_DAY = 86400.


def get_pos_reds(antpos, bl_error_tol=1.0):
    """ Figure out and return list of lists of redundant baseline pairs. Ordered by length. All baselines
        in a group have the same orientation with a preference for positive b_y and, when b_y==0, positive
        b_x where b((i,j)) = pos(j) - pos(i).

        Args:
            antpos: dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}. 1D and 2D also OK.
            bl_error_tol: the largest allowable difference between baselines in a redundant group
                (in the same units as antpos). Normally, this is up to 4x the largest antenna position error.

        Returns:
            reds: list (sorted by baseline legnth) of lists of redundant tuples of antenna indices (no polarizations),
            sorted by index with the first index of the first baseline the lowest in the group.
    """
    keys = list(antpos.keys())
    reds = {}
    assert np.all([len(pos) <= 3 for pos in antpos.values()]), 'Get_pos_reds only works in up to 3 dimensions.'
    ap = {ant: np.pad(pos, (0, 3 - len(pos)), mode='constant') for ant, pos in antpos.items()}  # increase dimensionality
    array_is_flat = np.all(np.abs(np.array(list(ap.values()))[:, 2] - np.mean(list(ap.values()), axis=0)[2]) < bl_error_tol / 4.0)
    for i, ant1 in enumerate(keys):
        for ant2 in keys[i + 1:]:
            delta = tuple(np.round(1.0 * (np.array(ap[ant2]) - np.array(ap[ant1])) / bl_error_tol).astype(int))
            if delta[0] > 0 or (delta[0] == 0 and delta[1] > 0) or (delta[0] == 0 and delta[1] == 0 and delta[2] > 0):
                bl_pair = (ant1, ant2)
            else:
                delta = tuple([-d for d in delta])
                bl_pair = (ant2, ant1)
            # Check to make sure reds doesn't have the key plus or minus rounding error
            p_or_m = (0, -1, 1)
            if array_is_flat:
                epsilons = [[dx, dy, 0] for dx in p_or_m for dy in p_or_m]
            else:
                epsilons = [[dx, dy, dz] for dx in p_or_m for dy in p_or_m for dz in p_or_m]
            for epsilon in epsilons:
                newKey = (delta[0] + epsilon[0], delta[1] + epsilon[1], delta[2] + epsilon[2])
                if newKey in reds:
                    reds[newKey].append(bl_pair)
                    break
            if newKey not in reds:
                reds[delta] = [bl_pair]

    # sort reds by length and each red to make sure the first antenna of the first bl in each group is the lowest antenna number
    orderedDeltas = [delta for (length, delta) in sorted(zip([np.linalg.norm(delta) for delta in reds.keys()], reds.keys()))]
    return [sorted(reds[delta]) if sorted(reds[delta])[0][0] == np.min(reds[delta])
            else sorted([reverse_bl(bl) for bl in reds[delta]]) for delta in orderedDeltas]


def add_pol_reds(reds, pols=['xx'], pol_mode='1pol'):
    """ Takes positonal reds (antenna indices only, no polarizations) and converts them
    into baseline tuples with polarization, depending on pols and pol_mode specified.

    Args:
        reds: list of list of antenna index tuples considered redundant
        pols: a list of polarizations e.g. ['xx', 'xy', 'yx', 'yy']
        pol_mode: polarization mode of calibration
            '1pol': 1 antpol and 1 vispol (e.g. 'jxx' and 'xx'). Default.
            '2pol': 2 antpols, no cross-vispols (e.g. 'jxx','jyy' and 'xx','yy')
            '4pol': 2 antpols, 4 vispols (e.g. 'jxx','jyy' and 'xx','xy','yx','yy')
            '4pol_minV': 2 antpols, 4 vispols in data but assuming V_xy = V_yx in model

    Returns:
        reds: list of lists of redundant baseline tuples, e.g. (ind1,ind2,pol)
    """
    # pre-process to ensure pols complies w/ hera_cal polarization convention
    pols = [comply_pol(p) for p in pols]

    redsWithPols, didBothCrossPolsForMinV = [], False
    for pol in pols:
        if pol_mode is not '4pol_minV' or pol[0] == pol[1]:
            redsWithPols += [[bl + (pol,) for bl in bls] for bls in reds]
        elif pol_mode is '4pol_minV' and not didBothCrossPolsForMinV:
            # Combine together e.g. 'xy' and 'yx' visibilities as redundant
            redsWithPols += [([bl + (pol,) for bl in bls]
                              + [bl + (conj_pol(pol),) for bl in bls]) for bls in reds]
            didBothCrossPolsForMinV = True
    return redsWithPols


def get_reds(antpos, pols=['xx'], pol_mode='1pol', bl_error_tol=1.0):
    """ Combines redcal.get_pos_reds() and redcal.add_pol_reds(). See their documentation.

    Args:
        antpos: dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}.
        pols: a list of polarizations e.g. ['xx', 'xy', 'yx', 'yy']
        pol_mode: polarization mode of calibration
            '1pol': 1 antpol and 1 vispol (e.g. 'jxx' and 'xx'). Default.
            '2pol': 2 antpols, no cross-vispols (e.g. 'jxx','jyy' and 'xx','yy')
            '4pol': 2 antpols, 4 vispols (e.g. 'jxx','jyy' and 'xx','xy','yx','yy')
            '4pol_minV': 2 antpols, 4 vispols in data but assuming V_xy = V_yx in model
        bl_error_tol: the largest allowable difference between baselines in a redundant group
            (in the same units as antpos). Normally, this is up to 4x the largest antenna position error.

    Returns:
        reds: list (sorted by baseline length) of lists of redundant baseline tuples, e.g. (ind1,ind2,pol).
            Each interior list is sorted so that the lowest index is first in the first baseline.

    """
    pos_reds = get_pos_reds(antpos, bl_error_tol=bl_error_tol)
    return add_pol_reds(pos_reds, pols=pols, pol_mode=pol_mode)


def filter_reds(reds, bls=None, ex_bls=None, ants=None, ex_ants=None, ubls=None, ex_ubls=None, 
                pols=None, ex_pols=None, antpos=None, min_bl_cut=None, max_bl_cut=None):
    '''
    Filter redundancies to include/exclude the specified bls, antennas, unique bl groups and polarizations.
    Arguments are evaluated, in order of increasing precedence: (pols, ex_pols, ubls, ex_ubls, bls, ex_bls,
    ants, ex_ants).
    Args:
        reds: list of lists of redundant (i,j,pol) baseline tuples, e.g. the output of get_reds()
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
        pols (optional): polarizations to include in reds. e.g. ['xx','yy','xy','yx']. Default includes all
            polarizations in reds.
        ex_pols (optional): same as pols, but excludes polarizations.
        antpos: dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}. 1D and 2D also OK.
        min_bl_cut: cut redundant groups with average baseline lengths shorter than this. Same units as antpos
            which must be specified.
        max_bl_cut: cut redundant groups with average baselines lengths longer than this. Same units as antpos
            which must be specified.
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
        return gp3 + [g for g in gp if len(g) == 3]
    antpols = set(sum([list(split_pol(p)) for p in pols], []))

    def expand_ants(gp):
        gp2 = [(g, p) for g in gp if not hasattr(g, '__len__') for p in antpols]
        return gp2 + [g for g in gp if hasattr(g, '__len__')]

    def split_bls(bls):
        return [split_bl(bl) for bl in bls]
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
        bls = set(expand_bls(bls))
    else:  # default to set of all baselines
        bls = set(key for gp in reds for key in gp)
    if ex_bls:
        ex_bls = expand_bls(ex_bls)
        bls = set(k for k in bls if k not in ex_bls and reverse_bl(k) not in ex_bls)
    if ants:
        ants = expand_ants(ants)
        bls = set(join_bl(i, j) for i, j in split_bls(bls) if i in ants and j in ants)
    if ex_ants:
        ex_ants = expand_ants(ex_ants)
        bls = set(join_bl(i, j) for i, j in split_bls(bls) if i not in ex_ants and j not in ex_ants)
    bls.union(set(reverse_bl(k) for k in bls))  # put in reverse baselines, just in case
    reds = [[key for key in gp if key in bls] for gp in reds]
    reds = [gp for gp in reds if len(gp) > 0]

    if min_bl_cut is not None or max_bl_cut is not None:
        assert antpos is not None, 'antpos must be passed in if min_bl_cut or max_bl_cut is specified.'
        lengths = [np.mean([np.linalg.norm(antpos[bl[1]] - antpos[bl[0]]) for bl in gp]) for gp in reds]
        reds = [gp for gp, l in zip(reds, lengths) if ((min_bl_cut is None or l > min_bl_cut)
                                                       and (max_bl_cut is None or l < max_bl_cut))]
    return reds


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
            necessary to describe all redundancies (non-redundancy introduces extra dimensions.)
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
    return antpos


def _check_polLists_minV(polLists):
    """Given a list of unique visibility polarizations (e.g. for each red group), returns whether
    they are all either single identical polarizations (e.g. 'xx') or both cross polarizations
    (e.g. ['xy','yx']) so that the 4pol_minV can be assumed."""

    for polList in polLists:
        ps = list()
        if len(polList) is 1:
            if split_pol(polList[0])[0] != split_pol(polList[0])[1]:
                return False
        elif len(polList) is 2:
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
            '1pol': 1 antpol and 1 vispol (e.g. 'jxx' and 'xx'). Default.
            '2pol': 2 antpols, no cross-vispols (e.g. 'jxx','jyy' and 'xx','yy')
            '4pol': 2 antpols, 4 vispols (e.g. 'jxx','jyy' and 'xx','xy','yx','yy')
            '4pol_minV': 2 antpols, 4 vispols in data but assuming V_xy = V_yx in model
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


def get_gains_and_vis_from_sol(sol):
    """Splits a sol dictionary into len(key)==2 entries, taken to be gains,
    and len(key)==3 entries, taken to be model visibilities."""

    g = {key: val for key, val in sol.items() if len(key) == 2}
    v = {key: val for key, val in sol.items() if len(key) == 3}
    return g, v


def make_sol_finite(sol):
    '''Replaces nans and infs in solutions, which are usually the result of visibilities that are
    identically equal to 0. Modifies sol (which is a dictionary with gains and visibilities) in place,
    replacing visibilities with 0.0s and gains with 1.0s'''
    for k in sol.keys():
        if len(k) == 3:  # visibilities
            sol[k][~np.isfinite(sol[k])] = np.zeros_like(sol[k][~np.isfinite(sol[k])])
        elif len(k) == 2:  # gains
            sol[k][~np.isfinite(sol[k])] = np.ones_like(sol[k][~np.isfinite(sol[k])])


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

    def solve_iteratively(self, conv_crit=1e-10, maxiter=50, check_every=4, check_after=1, verbose=False):
        """Repeatedly solves and updates solution until convergence or maxiter is reached.
        Returns a meta-data about the solution and the solution itself.

        Args:
            conv_crit: A convergence criterion (default 1e-10) below which to stop iterating.
                Converegence is measured L2-norm of the change in the solution of all the variables
                divided by the L2-norm of the solution itself.
            maxiter: An integer maximum number of iterations to perform before quitting. Default 50.
            check_every: Compute convergence and updates weights every Nth iteration (saves computation). Default 4.
            check_after: Start computing convergence and updating weights after the first N iterations.  Default 1.

        Returns: meta, sol
            meta: a dictionary with metadata about the solution, including
                iter: the number of iterations taken to reach convergence (or maxiter), with dimensions of the data.
                chisq: the chi^2 of the solution produced by the final iteration, with dimensions of the data.
                conv_crit: the convergence criterion evaluated at the final iteration, with dimensions of the data.
            sol: a dictionary of complex solutions with variables as keys, with dimensions of the data.
        """
        assert maxiter > 0, 'Omnical must have at least 1 iteration.'
        sol = self.sol0
        terms = [(linsolve.get_name(gi), linsolve.get_name(gj), linsolve.get_name(uij))
                 for term in self.all_terms for (gi, gj, uij) in term]
        dmdl_u = self._get_ans0(sol)
        chisq = sum([np.abs(self.data[k] - dmdl_u[k])**2 * self.wgts[k] for k in self.keys])
        update = np.where(chisq > 0)
        # variables with '_u' are flattened and only include pixels that need updating
        dmdl_u = {k: v[update].flatten() for k, v in dmdl_u.items()}
        # wgts_u hold the wgts the user provides.  dwgts_u is what is actually used to wgt the data
        wgts_u = {k: (v * np.ones(chisq.shape, dtype=np.float32))[update].flatten() for k, v in self.wgts.items()}
        sol_u = {k: v[update].flatten() for k, v in sol.items()}
        iters = np.zeros(chisq.shape, dtype=np.int)
        conv = np.ones_like(chisq)
        for i in range(1, maxiter + 1):
            if verbose:
                print('Beginning iteration %d/%d' % (i, maxiter))
            if (i % check_every) == 1:
                # compute data wgts: dwgts = sum(V_mdl^2 / n^2) = sum(V_mdl^2 * wgts)
                # don't need to update data weighting with every iteration
                dwgts_u = {k: dmdl_u[k] * dmdl_u[k].conj() * wgts_u[k] for k in self.keys}
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
                new_chisq_u = sum([np.abs(v[update] - dmdl_u[k])**2 * wgts_u[k] for k, v in self.data.items()])
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
                update = tuple(u[update_u] for u in update)
            if verbose:
                print('    <CHISQ> = %f, <CONV> = %f, CNT = %d', (np.mean(chisq), np.mean(conv), update[0].size))


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

        self.reds = reds
        self.pol_mode = parse_pol_mode(self.reds)

        if check_redundancy:
            nDegens = self.count_degens(assume_redundant=False)
            nDegensExpected = self.count_degens()
            if nDegens != nDegensExpected:
                nPhaseSlopes = len(list(reds_to_antpos(self.reds).values())[0])
                raise ValueError('{} degeneracies found, but {} '.format(nDegens, nDegensExpected)
                                 + 'degeneracies expected (assuming {} phase slopes).'.format(nPhaseSlopes))

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
            for blgrp in self.reds:
                self.phs_avg[blgrp[0]] = np.exp(-np.complex64(1j) * np.median(np.unwrap([np.log(dc[bl]).imag for bl in blgrp], axis=0), axis=0))
                for bl in blgrp:
                    self.phs_avg[bl] = self.phs_avg[blgrp[0]].astype(dc[bl].dtype)
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
            ubl_num = [cnt for cnt, blgrp in enumerate(self.reds) if blgrp[0] == k][0]
            return 'u_%d_%s' % (ubl_num, k[-1])

    def compute_ubls(self, data, gain_sols):
        """Given a set of guess gain solutions, return a dictionary of calibrated visbilities
        averged over a redundant group. Not strictly necessary for typical operation."""

        dc = DataContainer(data)
        ubl_sols = {}
        for ubl, blgrp in enumerate(self.reds):
            d_gp = [dc[bl] for bl in blgrp]
            ubl_sols[blgrp[0]] = np.average(d_gp, axis=0)  # XXX add option for median here?
        return ubl_sols

    def _firstcal_iteration(self, data, df, f0, wgts={}, offsets_only=False,
                            sparse=False, mode='default', norm=True, medfilt=False, kernel=(1, 11)):
        '''Runs a single iteration of firstcal, which uses phase differences between nominally
        redundant meausrements to solve for delays and phase offsets that produce gains of the
        form: np.exp(2j * np.pi * delay * freqs + 1j * offset).
        
        Arguments:
            df: frequency change between data bins, scales returned delays by 1/df.
            f0: frequency of the first channel in the data
            offsets_only: only solve for phase offsets, dly_sol will be {}
            For all other arguments, see RedundantCalibrator.firstcal()

        Returns:
            dly_sol: dictionary of per-antenna delay solutions in the {(index,antpol): np.array}
                format.  All delays are multiplied by 1/df, so use that to set physical scale.
            off_sol: dictionary of per antenna phase offsets (in radians) in the same format.
        '''
        Nfreqs = data[next(iter(data))].shape[1]  # hardcode freq is axis 1 (time is axis 0)
        if len(wgts) == 0:
            wgts = {k: np.ones_like(data[k], dtype=np.float32) for k in data}
        wgts = DataContainer(wgts)
        taus_offs, twgts = {}, {}
        for bls in self.reds:
            for i, bl1 in enumerate(bls):
                d1, w1 = data[bl1], wgts[bl1]
                for bl2 in bls[i + 1:]:
                    d12 = d1 * np.conj(data[bl2])
                    if norm:
                        ad12 = np.abs(d12)
                        d12 /= np.where(ad12 == 0, np.float32(1), ad12)
                    w12 = w1 * wgts[bl2]
                    taus_offs[(bl1, bl2)] = utils.fft_dly(d12, df, f0=f0, wgts=w12, medfilt=medfilt, kernel=kernel)
                    twgts[(bl1, bl2)] = np.sum(w12)
        d_ls, w_ls = {}, {}
        for (bl1, bl2), tau_off_ij in taus_offs.items():
            ai, aj = split_bl(bl1)
            am, an = split_bl(bl2)
            i, j, m, n = (self.pack_sol_key(k) for k in (ai, aj, am, an))
            eq_key = '%s-%s-%s+%s' % (i, j, m, n)
            d_ls[eq_key] = np.array(tau_off_ij)
            w_ls[eq_key] = twgts[(bl1, bl2)]
        ls = linsolve.LinearSolver(d_ls, wgts=w_ls, sparse=sparse)
        sol = ls.solve(mode=mode)
        dly_sol = {self.unpack_sol_key(k): v[0] for k, v in sol.items()}
        off_sol = {self.unpack_sol_key(k): v[1] for k, v in sol.items()}
        return dly_sol, off_sol

    def firstcal(self, data, freqs, wgts={}, maxiter=25, conv_crit=1e-6,
                 sparse=False, mode='default', norm=True, medfilt=False, kernel=(1, 11)):
        """Solve for a calibration solution parameterized by a single delay and phase offset
        per antenna using the phase difference between nominally redundant measurements. 
        Delays are solved in a single iteration, but phase offsets are solved for 
        iteratively to account for phase wraps.

        Args:
            data: visibility data in the dictionary format {(ant1,ant2,pol): np.array}
            freqs: numpy array of frequencies in the data
            wgts: dictionary of linear weights in the same format as data. Defaults to equal wgts.
            maxiter: maximum number of phase offset solver iterations
            conv_crit: convergence criterion for iterative offset solver, defined as the L2 norm
                of the changes in phase (in radians) over all times and antennas
            sparse: represent the A matrix (visibilities to parameters) sparsely in linsolve
            mode: solving mode passed to the linsolve linear solver ('default', 'lsqr', 'pinv', or 'solve')
                Suggest using 'default' unless solver is having stability (convergence) problems.
                More documentation of modes in linsolve.LinearSolver.solve().
            norm: calculate delays from just the phase information (not the amplitude) of the data.
                This is a pretty effective way to get reliable delay even in the presence of RFI.
            medfilt : boolean, median filter data before fft.  This can work for data containing
                unflagged RFI, but tends to be less effective in practice than 'norm'.  Default False.
            kernel : size of median filter kernel along (time, freq) axes

        Returns:
            g_fc: dictionary of Ntimes x Nfreqs per-antenna gains solutions in the 
                {(index, antpol): np.exp(2j * np.pi * delay * freqs + 1j * offset)} format.
        """
        df = np.median(np.ediff1d(freqs))
        dtype = np.find_common_type([d.dtype for d in data.values()], [])

        # iteratively solve for offsets to account for phase wrapping
        assert maxiter > 0, 'Firstcal must have at least 1 iteration.'
        for i in range(maxiter):
            dlys, delta_off = self._firstcal_iteration(data, df=df, f0=freqs[0], wgts=wgts, 
                                                       offsets_only=(i > 0), sparse=sparse, mode=mode, 
                                                       norm=norm, medfilt=medfilt, kernel=kernel)
            if i == 0:  # only solve for delays on the first iteration
                g_fc = {ant: np.array(np.exp(2j * np.pi * np.outer(dly, freqs)),
                                      dtype=dtype) for ant, dly in dlys.items()}
                calibrate_in_place(data, g_fc, gain_convention='divide')  # applies calibration
            
            if np.linalg.norm(list(delta_off.values())) < conv_crit:
                break
            delta_gains = {ant: np.array(np.ones_like(g_fc[ant]) * np.exp(1.0j * delta_off[ant]),
                                         dtype=dtype) for ant in g_fc.keys()}
            calibrate_in_place(data, delta_gains, gain_convention='divide')  # update calibration
            g_fc = {ant: g_fc[ant] * delta_gains[ant] for ant in g_fc}

        calibrate_in_place(data, g_fc, gain_convention='multiply')  # unapply calibration
        return g_fc

    def logcal(self, data, sol0={}, wgts={}, sparse=False, mode='default'):
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
            sol: dictionary of gain and visibility solutions in the {(index,antpol): np.array}
                and {(ind1,ind2,pol): np.array} formats respectively
        """
        fc_data = deepcopy(data)
        calibrate_in_place(fc_data, sol0)
        ls = self._solver(linsolve.LogProductSolver, fc_data, wgts=wgts, detrend_phs=True, sparse=sparse)
        sol = ls.solve(mode=mode)
        sol = {self.unpack_sol_key(k): sol[k] for k in sol.keys()}
        for ubl_key in [k for k in sol.keys() if len(k) == 3]:
            sol[ubl_key] = sol[ubl_key] * self.phs_avg[ubl_key].conj()
        sol_with_fc = {key: (sol[key] * sol0[key] if (key in sol0 and len(key) == 2) else sol[key]) for key in sol.keys()}
        return sol_with_fc

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

        sol0 = {self.pack_sol_key(k): sol0[k] for k in sol0.keys()}
        ls = self._solver(linsolve.LinProductSolver, data, sol0=sol0, wgts=wgts, sparse=sparse)
        meta, sol = ls.solve_iteratively(conv_crit=conv_crit, maxiter=maxiter, verbose=verbose, mode=mode)
        sol = {self.unpack_sol_key(k): sol[k] for k in sol.keys()}
        return meta, sol

    def omnical(self, data, sol0, wgts={}, gain=.3, conv_crit=1e-10, maxiter=50, check_every=4, check_after=1):
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

        Returns:
            meta: dictionary of information about the convergence and chi^2 of the solution
            sol: dictionary of gain and visibility solutions in the {(index,antpol): np.array}
                and {(ind1,ind2,pol): np.array} formats respectively
        """

        sol0 = {self.pack_sol_key(k): sol0[k] for k in sol0.keys()}
        ls = self._solver(OmnicalSolver, data, sol0=sol0, wgts=wgts, gain=gain)
        meta, sol = ls.solve_iteratively(conv_crit=conv_crit, maxiter=maxiter, check_every=check_every, check_after=check_after)
        sol = {self.unpack_sol_key(k): sol[k] for k in sol.keys()}
        return meta, sol

    def remove_degen_gains(self, gains, degen_gains=None, mode='phase'):
        """ Removes degeneracies from solutions (or replaces them with those in degen_sol).  This
        function in nominally intended for use with firstcal, which returns (phase/delay) solutions
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
        Returns:
            new_gains: gains with degeneracy removal/replacement performed
        """

        # Check supported pol modes
        assert self.pol_mode in ['1pol', '2pol', '4pol', '4pol_minV'], 'Unrecognized pol_mode: %s' % self.pol_mode
        assert mode in ('phase', 'complex'), 'Unrecognized mode: %s' % mode
        if degen_gains is None:
            if mode == 'phase':
                degen_gains = {key: np.zeros_like(val) for key, val in gains.items()}
            else:  # complex
                degen_gains = {key: np.ones_like(val) for key, val in gains.items()}
        ants = gains.keys()
        gainPols = np.array([ant[1] for ant in gains])  # gainPols is list of antpols, one per antenna
        antpols = list(set(gainPols))

        # if mode is 2pol, run as two 1pol remove degens
        if self.pol_mode is '2pol':
            self.pol_mode = '1pol'
            pol0_gains = {k: v for k, v in gains.items() if k[1] == antpols[0]}
            pol1_gains = {k: v for k, v in gains.items() if k[1] == antpols[1]}
            new_gains = self.remove_degen_gains(pol0_gains, degen_gains=degen_gains, mode=mode)
            new_gains.update(self.remove_degen_gains(pol1_gains, degen_gains=degen_gains, mode=mode))
            self.pol_mode = '2pol'
            return new_gains

        # Extract gain and model visibiltiy solutions
        gainSols = np.array([gains[ant] for ant in ants])
        degenGains = np.array([degen_gains[ant] for ant in ants])

        # Build matrices for projecting gain degeneracies
        antpos = reds_to_antpos(self.reds)
        positions = np.array([antpos[ant[0]] for ant in gains])
        if self.pol_mode is '1pol' or self.pol_mode is '4pol_minV':
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
                degenMeanSqAmplitude = np.mean([np.abs(degen_gains[k1] * degen_gains[k2]) for k1 in gains.keys()
                                                for k2 in gains.keys()
                                                if k1[1] == pol and k2[1] == pol and k1[0] != k2[0]], axis=0)
                gainSols[gainPols == pol] *= (degenMeanSqAmplitude / meanSqAmplitude)**.5

        # Create new solutions dictionary
        new_gains = {ant: gainSol for ant, gainSol in zip(ants, gainSols)}
        return new_gains

    def remove_degen(self, sol, degen_sol=None):
        """ Removes degeneracies from solutions (or replaces them with those in degen_sol).  This
        function is nominally intended for use with solutions from logcal, omnical, or lincal, which
        return complex solutions for antennas and visibilities.

        Args:
            sol: dictionary that contains both visibility and gain solutions in the
                {(ind1,ind2,pol): np.array} and {(index,antpol): np.array} formats respectively
            degen_sol: Optional dictionary in the same format as sol. Gain amplitudes and phases
                in degen_sol replace the values of sol in the degenerate subspace of redcal. If
                left as None, average gain amplitudes will be 1 and average phase terms will be 0.
                Visibilties in degen_sol are ignored.  For logcal/lincal/omnical, putting firstcal
                solutions in here can help avoid structure associated with phase-wrapping issues.
        Returns:
            new_sol: sol with degeneracy removal/replacement performed
        """
        gains, vis = get_gains_and_vis_from_sol(sol)
        if degen_sol is None:
            degen_sol = {key: np.ones_like(val) for key, val in gains.items()}
        new_gains = self.remove_degen_gains(gains, degen_gains=degen_sol, mode='complex')
        new_sol = deepcopy(vis)
        calibrate_in_place(new_sol, new_gains, old_gains=gains)
        new_sol.update(new_gains)
        return new_sol

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
            dummy_data = DataContainer({bl: np.ones((1, 1), dtype=np.complex) for red in self.reds for bl in red})
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


def expand_omni_vis(cal, all_reds, data, flags, nsamples):
    '''This function expands and harmonizes a calibration solution produced by redcal.redundantly_calibrate
    to a set of un-filtered redundancies. This only affects omnical visibility solutions. It has three effects:
        1) Visibility solutions are now keyed by the first entry in each red in all_reds, even if they were
            originally keyed by a different entry.
        2) Unique baselines that were exluded from the redundant calibration are filled in by averaging
            unflagged calibrated visibilities.
        3) cal gets a new entry, cal['vns_omnical'] which is a nsamples data container of the number of 
            visibilites that went into each unique baseline solution/average. 

    Arguments:
        cal: dictionary of redundant calibration solutions produced by redcal.redundantly_calibrate. 
            Modified in place, including adding an entry with key 'vns_omnical' that gives a number of
            samples that went into each unique baseline visibility solution
        all_reds: list of lists of redundant baseline tuples, e.g. (0,1,'xx'). The first
            item in each list will be treated as the key for the unique baseline. Must be a superset of
            the reds used for producing cal
        data: DataContainer mapping baseline-pol tuples like (0,1,'xx') to complex data of 
            shape (Nt, Nf). Calibrated in place using cal['g_omnical'] and cal['gf_omnical']
        flags: DataContainer mapping baseline-pol tuples like (0,1,'xx') to boolean flags of
            shape (Nt, Nf). Modified in place using cal['gf_omnical']
        nsamples: DataContainer mapping baseline-pol tuples like (0,1,'xx') to float number of samples.
            Used for counting the number of non-flagged visibilities that went into each redundant group.
    '''
    calibrate_in_place(data, cal['g_omnical'], data_flags=flags, cal_flags=cal['gf_omnical'])
    cal['vns_omnical'] = {}

    for red in all_reds:
        omni_keys = [bl for bl in red if bl in cal['v_omnical']]
        assert len(omni_keys) <= 1, "The input calibration's 'v_omnical' entry can have at most visibility per unique baseline group."
        cal['vns_omnical'][red[0]] = np.sum([nsamples[bl] * (1.0 - flags[bl]) for bl in red], axis=0).astype(np.float32)

        if len(omni_keys) == 0:  # the omnical solution doesn't have this baseline, so compute it by averaging the calibrated data
            cal['v_omnical'][red[0]] = np.sum([data[bl] * (1.0 - flags[bl]) * nsamples[bl] for bl in red], axis=0).astype(np.complex64)
            cal['v_omnical'][red[0]] /= cal['vns_omnical'][red[0]]
            cal['vf_omnical'][red[0]] = np.logical_or(cal['vns_omnical'][red[0]] == 0, ~np.isfinite(cal['v_omnical'][red[0]]))
            cal['v_omnical'][red[0]][~np.isfinite(cal['v_omnical'][red[0]])] = np.complex64(1.)
        elif omni_keys[0] != red[0]:  # the omnical solution has the baseline, but it's keyed by something other than red[0]
            cal['v_omnical'][red[0]] = deepcopy(cal['v_omnical'][omni_keys[0]])
            cal['vf_omnical'][red[0]] = deepcopy(cal['vf_omnical'][omni_keys[0]])
            del cal['v_omnical'][omni_keys[0]], cal['vf_omnical'][omni_keys[0]]
    
    cal['vns_omnical'] = DataContainer(cal['vns_omnical'])


def redundantly_calibrate(data, reds, freqs=None, times_by_bl=None, wgts={}, prior_firstcal=False,
                          fc_conv_crit=1e-6, fc_maxiter=50, oc_conv_crit=1e-10, oc_maxiter=500, 
                          check_every=10, check_after=50, gain=.4):
    '''Performs all three steps of redundant calibration: firstcal, logcal, and omnical.

    Arguments:
        data: dictionary or DataContainer mapping baseline-pol tuples like (0,1,'xx') to
            complex data of shape. Asummed to have no flags.
        reds: list of lists of redundant baseline tuples, e.g. (0,1,'xx'). The first
            item in each list will be treated as the key for the unique baseline.
        freqs: 1D numpy array frequencies in Hz. Optional if inferable from data DataContainer,
            but must be provided if data is a dictionary, if it doesn't have .freqs, or if the
            length of data.freqs is 1.
        times_by_bl: dictionary mapping antenna pairs like (0,1) to float Julian Date. Optional if
            inferable from data DataContainer, but must be provided if data is a dictionary,
            if it doesn't have .times_by_bl, or if the length of any list of times is 1.
        wgts : dictionary or DataContainer of data weights matching data shape, to be applied linearly
            to the data. All baselines in reds must be in wgts if it is no the default, {}. In the default
            case, this uses identical weights for firstcal and logcal and autocorrelations to estimate
            noise weighting for omnical. Regardless, auto-based noise estimates normalize chi^2.
        prior_firstcal: either bool or dictionary:
            If False (default): the data is raw and firstcal is unknown. Run firstcal normally.
            If True: the data has been pre-calibrated. Assumes firstcal is all unity gains.
            If a dictionary: the data is raw, but instead of using the firstcal algorithm, 
                use these gains instead. Must be in the same format 
        fc_conv_crit: maximum allowed changed in firstcal phases for convergence
        fc_maxiter: maximum number of firstcal iterations allowed for finding per-antenna phases
        oc_conv_crit: maximum allowed relative change in omnical solutions for convergence
        oc_maxiter: maximum number of omnical iterations allowed before it gives up
        check_every: compute omnical convergence every Nth iteration (saves computation).
        check_after: start computing omnical convergence only after N iterations (saves computation).
        gain: The fractional step made toward the new solution each omnical iteration. Values in the
            range 0.1 to 0.5 are generally safe. Increasing values trade speed for stability.

    Returns a dictionary of results with the following keywords:
        'g_firstcal': firstcal gains in dictionary keyed by ant-pol tuples like (1,'Jxx').
            Gains are Ntimes x Nfreqs gains but fully described by a per-antenna delay.
            Note that ants/bls in data but not in reds will not appear in returned gains/vis sols.
        'gf_firstcal': firstcal gain flags in the same format as 'g_firstcal'. Will be all False.
        'g_omnical': full omnical gain dictionary (which include firstcal gains) in the same format.
            Flagged gains will be 1.0s.
        'gf_omnical': omnical flag dictionary in the same format. Flags arise from NaNs in log/omnical.
        'v_omnical': omnical visibility solutions dictionary with baseline-pol tuple keys that are the
            first elements in each of the sub-lists of reds. Flagged visibilities will be 0.0s.
        'vf_omnical': omnical visibility flag dictionary in the same format. Flags arise from NaNs.
        'chisq': chi^2 per degree of freedom for the omnical solution. Normalized using noise derived
            from autocorrelations. If the inferred pol_mode from reds (see redcal.parse_pol_mode) is
            '1pol' or '2pol', this is a dictionary mapping antenna polarization (e.g. 'Jxx') to chi^2.
            Otherwise, there is a single chisq (because polarizations mix) and this is a numpy array.
        'chisq_per_ant': dictionary mapping ant-pol tuples like (1,'Jxx') to the average chisq
            for all visibilities that an antenna participates in.
        'omni_meta': dictionary of information about the omnical convergence and chi^2 of the solution
    '''
    rv = {}  # dictionary of return values
    rc = RedundantCalibrator(reds)
    if freqs is None:
        freqs = data.freqs
    if times_by_bl is None:
        times_by_bl = data.times_by_bl
    ants = sorted(set([ant for red in reds for bl in red for ant in utils.split_bl(bl)]))

    # perform firstcal if required
    if prior_firstcal is False:
        # perform firstcal normally on raw data
        rv['g_firstcal'] = rc.firstcal(data, freqs, maxiter=fc_maxiter, conv_crit=fc_conv_crit, wgts=wgts)
    elif prior_firstcal is True:
        # assume the data is pre-calibrated, so all firstcal gains are 1.0 + 0.0j
        rv['g_firstcal'] = {ant: np.ones_like(list(data.values())[0]) for ant in ant}
    else:
        # assume the data is raw, but use an external set of gains for firstcal
        assert isinstance(prior_firstcal, dict), 'prior_firstcal must be a boolean or a dictionary of gains.'
        assert np.all([ant in prior_firstcal for ant in ants]), \
               'if prior_firstcal is a dict, it must have gains for all antennas that appear in reds'
        rv['g_firstcal'] = prior_firstcal
    rv['gf_firstcal'] = {ant: np.zeros_like(g, dtype=bool) for ant, g in rv['g_firstcal'].items()}

    # perform logcal and omnical. Use noise_wgts if wgts if None
    log_sol = rc.logcal(data, sol0=rv['g_firstcal'], wgts=wgts)
    make_sol_finite(log_sol)
    noise_wgts = {bl: predict_noise_variance_from_autos(bl, data, dt=(np.median(np.ediff1d(times_by_bl[bl[:2]]))
                                                                      * SEC_PER_DAY))**-1 for bl in data.keys()}
    if len(wgts) == 0:
        wgts = noise_wgts
    rv['omni_meta'], omni_sol = rc.omnical(data, log_sol, wgts=wgts, conv_crit=oc_conv_crit, maxiter=oc_maxiter,
                                           check_every=check_every, check_after=check_after, gain=gain)

    # update omnical flags and then remove degeneracies
    rv['g_omnical'], rv['v_omnical'] = get_gains_and_vis_from_sol(omni_sol)
    rv['gf_omnical'] = {ant: ~np.isfinite(g) for ant, g in rv['g_omnical'].items()}
    rv['vf_omnical'] = DataContainer({bl: ~np.isfinite(v) for bl, v in rv['v_omnical'].items()})
    make_sol_finite(omni_sol)
    rd_sol = rc.remove_degen(omni_sol, degen_sol=rv['g_firstcal'])
    rv['g_omnical'], rv['v_omnical'] = get_gains_and_vis_from_sol(rd_sol)
    rv['v_omnical'] = DataContainer(rv['v_omnical'])
    rv['g_omnical'] = {ant: g * ~rv['gf_omnical'][ant] + rv['gf_omnical'][ant] for ant, g in rv['g_omnical'].items()}

    # compute chisqs
    rv['chisq'], nObs, rv['chisq_per_ant'], nObs_per_ant = utils.chisq(data, rv['v_omnical'], data_wgts=noise_wgts,
                                                                       gains=rv['g_omnical'], reds=reds,
                                                                       split_by_antpol=(rc.pol_mode in ['1pol', '2pol']))
    rv['chisq_per_ant'] = {ant: cs / nObs_per_ant[ant] for ant, cs in rv['chisq_per_ant'].items()}
    nDegen = rc.count_degens()  # need to add back in nDegen/2 complex degrees of freedom
    if rc.pol_mode in ['1pol', '2pol']:  # in this case, chisq is split by antpol
        for antpol in rv['chisq'].keys():
            Ngains = len([ant for ant in rv['g_omnical'].keys() if ant[1] == antpol])
            Nvis = len([bl for bl in rv['v_omnical'].keys() if bl[2] == join_pol(antpol, antpol)])
            rv['chisq'][antpol] /= (nObs[antpol] - Ngains - Nvis + nDegen / {'1pol': 2.0, '2pol': 4.0}[rc.pol_mode])  
    elif rc.pol_mode == '4pol':
        rv['chisq'] /= (nObs - len(rv['g_omnical']) - len(rv['v_omnical']) + nDegen / 2.0)
    else:  # 4pol_minV
        rv['chisq'] /= (nObs - len(rv['g_omnical']) - len(rv['v_omnical']) + nDegen / 2.0)
    return rv


def redcal_iteration(hd, nInt_to_load=None, pol_mode='2pol', bl_error_tol=1.0, ex_ants=[], solar_horizon=0.0,
                     flag_nchan_low=0, flag_nchan_high=0, fc_conv_crit=1e-6, fc_maxiter=50, oc_conv_crit=1e-10, 
                     oc_maxiter=500, check_every=10, check_after=50, gain=.4, verbose=False, **filter_reds_kwargs):
    '''Perform redundant calibration (firstcal, logcal, and omnical) an entire HERAData object, loading only
    nInt_to_load integrations at a time and skipping and flagging times when the sun is above solar_horizon.

    Arguments:
        hd: HERAData object, instantiated with the datafile or files to calibrate. Must be loaded using uvh5.
            Assumed to have no prior flags.
        nInt_to_load: number of integrations to load and calibrate simultaneously. Default None loads all integrations.
            Partial io requires 'uvh5' filetype for hd. Lower numbers save memory, but incur a CPU overhead.
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
        fc_conv_crit: maximum allowed changed in firstcal phases for convergence
        fc_maxiter: maximum number of firstcal iterations allowed for finding per-antenna phases
        oc_conv_crit: maximum allowed relative change in omnical solutions for convergence
        oc_maxiter: maximum number of omnical iterations allowed before it gives up
        check_every: compute omnical convergence every Nth iteration (saves computation).
        check_after: start computing omnical convergence only after N iterations (saves computation).
        gain: The fractional step made toward the new solution each omnical iteration. Values in the
            range 0.1 to 0.5 are generally safe. Increasing values trade speed for stability.
        verbose: print calibration progress updates
        filter_reds_kwargs: additional filters for the redundancies (see redcal.filter_reds for documentation)

    Returns a dictionary of results with the following keywords:
        'g_firstcal': firstcal gains in dictionary keyed by ant-pol tuples like (1,'Jxx').
            Gains are Ntimes x Nfreqs gains but fully described by a per-antenna delay.
        'gf_firstcal': firstcal gain flags in the same format as 'g_firstcal'. Will be all False.
        'g_omnical': full omnical gain dictionary (which include firstcal gains) in the same format.
            Flagged gains will be 1.0s.
        'gf_omnical': omnical flag dictionary in the same format. Flags arise from NaNs in log/omnical.
        'v_omnical': omnical visibility solutions dictionary with baseline-pol tuple keys that are the
            first elements in each of the sub-lists of reds. Flagged visibilities will be 0.0s.
        'vf_omnical': omnical visibility flag dictionary in the same format. Flags arise from NaNs.
        'vns_omnical': omnical visibility nsample dictionary that counts the number of unflagged redundancies.
        'chisq': chi^2 per degree of freedom for the omnical solution. Normalized using noise derived
            from autocorrelations. If the inferred pol_mode from reds (see redcal.parse_pol_mode) is
            '1pol' or '2pol', this is a dictionary mapping antenna polarization (e.g. 'Jxx') to chi^2.
            Otherwise, there is a single chisq (because polarizations mix) and this is a numpy array.
        'chisq_per_ant': dictionary mapping ant-pol tuples like (1,'Jxx') to the average chisq
            for all visibilities that an antenna participates in.
    '''
    if nInt_to_load is not None:
        assert hd.filetype == 'uvh5', 'Partial loading only available for uvh5 filetype.'
    else:
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

    # initialize gains to 1s, gain flags to True, and chisq to 0s
    rv = {}  # dictionary of return values
    rv['g_firstcal'] = {ant: np.ones((nTimes, nFreqs), dtype=np.complex64) for ant in ants}
    rv['gf_firstcal'] = {ant: np.ones((nTimes, nFreqs), dtype=bool) for ant in ants}
    rv['g_omnical'] = {ant: np.ones((nTimes, nFreqs), dtype=np.complex64) for ant in ants}
    rv['gf_omnical'] = {ant: np.ones((nTimes, nFreqs), dtype=bool) for ant in ants}
    rv['chisq'] = {antpol: np.zeros((nTimes, nFreqs), dtype=np.float32) for antpol in antpols}
    rv['chisq_per_ant'] = {ant: np.zeros((nTimes, nFreqs), dtype=np.float32) for ant in ants}

    # get reds and then intitialize omnical visibility solutions to all 1s and all flagged
    all_reds = get_reds({ant: hd.antpos[ant] for ant in ant_nums}, bl_error_tol=bl_error_tol,
                        pol_mode=pol_mode, pols=set([pol for pols in pol_load_list for pol in pols]))
    rv['v_omnical'] = DataContainer({red[0]: np.ones((nTimes, nFreqs), dtype=np.complex64) for red in all_reds})
    rv['vf_omnical'] = DataContainer({red[0]: np.ones((nTimes, nFreqs), dtype=bool) for red in all_reds})
    rv['vns_omnical'] = DataContainer({red[0]: np.zeros((nTimes, nFreqs), dtype=np.float32) for red in all_reds})
    filtered_reds = filter_reds(all_reds, ex_ants=ex_ants, antpos=hd.antpos, **filter_reds_kwargs)

    # solar flagging
    lat, lon, alt = hd.telescope_location_lat_lon_alt_degrees
    solar_alts = utils.get_sun_alt(hd.times, latitude=lat, longitude=lon)
    solar_flagged = solar_alts > solar_horizon
    if verbose and np.any(solar_flagged):
        print(len(hd.times[solar_flagged]), 'integrations flagged due to sun above', solar_horizon, 'degrees.')

    # loop over polarizations and times, performing partial loading if desired
    for pols in pol_load_list:
        if verbose:
            print('Now calibrating', pols, 'polarization(s)...')
        reds = filter_reds(filtered_reds, ex_ants=ex_ants, pols=pols)
        if nInt_to_load is not None:  # split up the integrations to load nInt_to_load at a time
            tind_groups = np.split(np.arange(nTimes)[~solar_flagged],
                                   np.arange(nInt_to_load, len(hd.times[~solar_flagged]), nInt_to_load))
        else:
            tind_groups = [np.arange(nTimes)[~solar_flagged]]  # just load a single group
        for tinds in tind_groups:
            if len(tinds) > 0:
                if verbose:
                    print('    Now calibrating times', hd.times[tinds[0]], 'through', hd.times[tinds[-1]], '...')
                if nInt_to_load is None:  # don't perform partial I/O
                    data, flags, nsamples = hd.build_datacontainers()  # this may contain unused polarizations, but that's OK
                    for bl in data:
                        data[bl] = data[bl][tinds, fSlice]  # cut down size of DataContainers to match unflagged indices
                        flags[bl] = flags[bl][tinds, fSlice]
                        nsamples[bl] = nsamples[bl][tinds, fSlice] 
                else:  # perform partial i/o
                    data, flags, nsamples = hd.read(times=hd.times[tinds], frequencies=hd.freqs[fSlice], polarizations=pols)
                cal = redundantly_calibrate(data, reds, freqs=hd.freqs[fSlice], times_by_bl=hd.times_by_bl,
                                            fc_conv_crit=fc_conv_crit, fc_maxiter=fc_maxiter, oc_conv_crit=oc_conv_crit, 
                                            oc_maxiter=oc_maxiter, check_every=check_every, check_after=check_after, gain=gain)
                expand_omni_vis(cal, filter_reds(all_reds, pols=pols), data, flags, nsamples)
                
                # gather results
                for ant in cal['g_omnical'].keys():
                    rv['g_firstcal'][ant][tinds, fSlice] = cal['g_firstcal'][ant]
                    rv['gf_firstcal'][ant][tinds, fSlice] = cal['gf_firstcal'][ant]
                    rv['g_omnical'][ant][tinds, fSlice] = cal['g_omnical'][ant]
                    rv['gf_omnical'][ant][tinds, fSlice] = cal['gf_omnical'][ant]
                    rv['chisq_per_ant'][ant][tinds, fSlice] = cal['chisq_per_ant'][ant]
                for bl in cal['v_omnical'].keys():
                    rv['v_omnical'][bl][tinds, fSlice] = cal['v_omnical'][bl]
                    rv['vf_omnical'][bl][tinds, fSlice] = cal['vf_omnical'][bl]
                    rv['vns_omnical'][bl][tinds, fSlice] = cal['vns_omnical'][bl]
                if pol_mode in ['1pol', '2pol']:
                    for antpol in cal['chisq'].keys():
                        rv['chisq'][antpol][tinds, fSlice] = cal['chisq'][antpol]
                else:  # duplicate chi^2 into both antenna polarizations
                    for antpol in rv['chisq'].keys():
                        rv['chisq'][antpol][tinds, fSlice] = cal['chisq']

    return rv


def redcal_run(input_data, filetype='uvh5', firstcal_ext='.first.calfits', omnical_ext='.omni.calfits', 
               omnivis_ext='.omni_vis.uvh5', outdir=None, ant_metrics_file=None, clobber=False, 
               nInt_to_load=None, pol_mode='2pol', bl_error_tol=1.0, ex_ants=[], ant_z_thresh=4.0, 
               max_rerun=5, solar_horizon=0.0, flag_nchan_low=0, flag_nchan_high=0, fc_conv_crit=1e-6, 
               fc_maxiter=50, oc_conv_crit=1e-10, oc_maxiter=500, check_every=10, check_after=50, gain=.4, 
               add_to_history='', verbose=False, **filter_reds_kwargs):
    '''Perform redundant calibration (firstcal, logcal, and omnical) an uvh5 data file, saving firstcal and omnical
    results to calfits and uvh5. Uses partial io if desired, performs solar flagging, and iteratively removes antennas
    with high chi^2, rerunning calibration as necessary.

    Arguments:
        input_data: path to visibility data file to calibrate or HERAData object
        filetype: filetype of input_data (if it's a path). Supports 'uvh5' (defualt), 'miriad', 'uvfits'
        firstcal_ext: string to replace file extension of input_data for saving firstcal calfits
        omnical_ext: string to replace file extension of input_data for saving omnical calfits
        omnivis_ext: string to replace file extension of input_data for saving omnical visibilities as uvh5
        outdir: folder to save data products. If None, will be the same as the folder containing input_data
        ant_metrics_file: path to file containing ant_metrics readable by hera_qm.metrics_io.load_metric_file.
            Used for finding ex_ants and is combined with antennas excluded via ex_ants.
        clobber: if True, overwrites existing files for the firstcal and omnical results
        nInt_to_load: number of integrations to load and calibrate simultaneously. Default None loads all integrations.
            Partial io requires 'uvh5' filetype. Lower numbers save memory, but incur a CPU overhead.
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
        fc_conv_crit: maximum allowed changed in firstcal phases for convergence
        fc_maxiter: maximum number of firstcal iterations allowed for finding per-antenna phases
        oc_conv_crit: maximum allowed relative change in omnical solutions for convergence
        oc_maxiter: maximum number of omnical iterations allowed before it gives up
        check_every: compute omnical convergence every Nth iteration (saves computation).
        check_after: start computing omnical convergence only after N iterations (saves computation).
        gain: The fractional step made toward the new solution each omnical iteration. Values in the
            range 0.1 to 0.5 are generally safe. Increasing values trade speed for stability.
        add_to_history: string to add to history of output firstcal and omnical files
        verbose: print calibration progress updates
        filter_reds_kwargs: additional filters for the redundancies (see redcal.filter_reds for documentation)

    Returns:
        cal: the dictionary result of the final run of redcal_iteration (see above for details)
    '''
    if isinstance(input_data, str):
        hd = HERAData(input_data, filetype=filetype)
        if filetype != 'uvh5' or nInt_to_load is None:
            hd.read()

    elif isinstance(input_data, HERAData):
        hd = input_data
        input_data = hd.filepaths[0]
    else:
        raise TypeError('input_data must be a single string path to a visibility data file or a HERAData object')

    ex_ants = set(ex_ants)
    from hera_qm.metrics_io import load_metric_file
    if ant_metrics_file is not None:
        for ant in load_metric_file(ant_metrics_file)['xants']:
            ex_ants.add(ant[0])  # Just take the antenna number, flagging both polarizations
    high_z_ant_hist = ''

    # loop over calibration, removing bad antennas and re-running if necessary
    from hera_qm.ant_metrics import per_antenna_modified_z_scores
    run_number = 0
    while True:
        # Run redundant calibration
        if verbose:
            print('\nNow running redundant calibration without antennas', list(ex_ants), '...')
        cal = redcal_iteration(hd, nInt_to_load=nInt_to_load, pol_mode=pol_mode, bl_error_tol=bl_error_tol, ex_ants=ex_ants, 
                               solar_horizon=solar_horizon, flag_nchan_low=flag_nchan_low, flag_nchan_high=flag_nchan_high, 
                               fc_conv_crit=fc_conv_crit, fc_maxiter=fc_maxiter, oc_conv_crit=oc_conv_crit, oc_maxiter=oc_maxiter, 
                               check_every=check_every, check_after=check_after, gain=gain, verbose=verbose, **filter_reds_kwargs)

        # Determine whether to add additional antennas to exclude
        z_scores = per_antenna_modified_z_scores({ant: np.nanmedian(cspa) for ant, cspa in cal['chisq_per_ant'].items()
                                                  if not np.all(cspa == 0)})
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

    # output results files
    filename_no_ext = os.path.splitext(os.path.basename(input_data))[0]
    if outdir is None:
        outdir = os.path.dirname(input_data)

    if verbose:
        print('\nNow saving firstcal gains to', os.path.join(outdir, filename_no_ext + firstcal_ext))
    write_cal(filename_no_ext + firstcal_ext, cal['g_firstcal'], hd.freqs, hd.times,
              flags=cal['gf_firstcal'], outdir=outdir, overwrite=clobber, 
              history=version.history_string(add_to_history))

    if verbose:
        print('Now saving omnical gains to', os.path.join(outdir, filename_no_ext + omnical_ext))
    write_cal(filename_no_ext + omnical_ext, cal['g_omnical'], hd.freqs, hd.times, flags=cal['gf_omnical'],
              quality=cal['chisq_per_ant'], total_qual=cal['chisq'], outdir=outdir, overwrite=clobber,
              history=version.history_string(add_to_history + '\n' + high_z_ant_hist))

    if verbose:
        print('Now saving omnical visibilities to', os.path.join(outdir, filename_no_ext + omnivis_ext))
    hd.read(bls=cal['v_omnical'].keys())
    hd.update(data=cal['v_omnical'], flags=cal['vf_omnical'], nsamples=cal['vns_omnical'])
    hd.history += version.history_string(add_to_history + '\n' + high_z_ant_hist)
    hd.write_uvh5(os.path.join(outdir, filename_no_ext + omnivis_ext), clobber=True)

    return cal


def redcal_argparser():
    '''Arg parser for commandline operation of redcal_run'''
    a = argparse.ArgumentParser(description="Redundantly calibrate a file using hera_cal.redcal. This includes firstcal, logcal, and omnical. \
                                Iteratively re-runs by flagging antennas with large chi^2. Saves the result to calfits and uvh5 files.")
    a.add_argument("input_data", type=str, help="path to uvh5 visibility data file to calibrate.")
    a.add_argument("--firstcal_ext", default='.first.calfits', type=str, help="string to replace file extension of input_data for saving firstcal calfits")
    a.add_argument("--omnical_ext", default='.omni.calfits', type=str, help="string to replace file extension of input_data for saving omnical calfits")
    a.add_argument("--omnivis_ext", default='.omni_vis.uvh5', type=str, help="string to replace file extension of input_data for saving omnical visibilities as uvh5")
    a.add_argument("--outdir", default=None, type=str, help="folder to save data products. Default is the same as the folder containing input_data")
    a.add_argument("--clobber", default=False, action="store_true", help="overwrites existing files for the firstcal and omnical results")
    a.add_argument("--verbose", default=False, action="store_true", help="print calibration progress updates")

    redcal_opts = a.add_argument_group(title='Runtime Options for Redcal')
    redcal_opts.add_argument("--ant_metrics_file", type=str, default=None, help="path to file containing ant_metrics readable by hera_qm.metrics_io.load_metric_file. \
                             Used for finding ex_ants and is combined with antennas excluded via ex_ants.")
    redcal_opts.add_argument("--ex_ants", type=int, nargs='*', default=[], help='space-delimited list of antennas to exclude from calibration and flag. All pols for an antenna will be excluded.')
    redcal_opts.add_argument("--ant_z_thresh", type=float, default=4.0, help="Threshold of modified z-score for chi^2 per antenna above which antennas are thrown away and calibration is re-run iteratively.")
    redcal_opts.add_argument("--max_rerun", type=int, default=5, help="Maximum number of times to re-run redundant calibration.")
    redcal_opts.add_argument("--solar_horizon", type=float, default=0.0, help="When the Sun is above this altitude in degrees, calibration is skipped and the integrations are flagged.")
    redcal_opts.add_argument("--flag_nchan_low", type=int, default=0, help="integer number of channels at the low frequency end of the band to always flag (default 0)")
    redcal_opts.add_argument("--flag_nchan_high", type=int, default=0, help="integer number of channels at the high frequency end of the band to always flag (default 0)")
    redcal_opts.add_argument("--nInt_to_load", type=int, default=None, help="number of integrations to load and calibrate simultaneously. Lower numbers save memory, but incur a CPU overhead. \
                             Default None loads all integrations.")
    redcal_opts.add_argument("--pol_mode", type=str, default='2pol', help="polarization mode of redundancies. Can be '1pol', '2pol', '4pol', or '4pol_minV'. See recal.get_reds documentation.")
    redcal_opts.add_argument("--bl_error_tol", type=float, default=1.0, help="the largest allowable difference between baselines in a redundant group")
    redcal_opts.add_argument("--min_bl_cut", type=float, default=None, help="cut redundant groups with average baseline lengths shorter than this length in meters")
    redcal_opts.add_argument("--max_bl_cut", type=float, default=None, help="cut redundant groups with average baseline lengths longer than this length in meters")

    omni_opts = a.add_argument_group(title='Firstcal and Omnical-Specific Options')
    omni_opts.add_argument("--fc_conv_crit", type=float, default=1e-6, help="maximum allowed changed in firstcal phases for convergence")
    omni_opts.add_argument("--fc_maxiter", type=int, default=50, help="maximum number of firstcal iterations allowed for finding per-antenna phases")
    omni_opts.add_argument("--oc_conv_crit", type=float, default=1e-10, help="maximum allowed relative change in omnical solutions for convergence")
    omni_opts.add_argument("--oc_maxiter", type=int, default=500, help="maximum number of omnical iterations allowed before it gives up")
    omni_opts.add_argument("--check_every", type=int, default=10, help="compute omnical convergence every Nth iteration (saves computation).")
    omni_opts.add_argument("--check_after", type=int, default=50, help="start computing omnical convergence only after N iterations (saves computation).")
    omni_opts.add_argument("--gain", type=float, default=.4, help="The fractional step made toward the new solution each omnical iteration. Values in the range 0.1 to 0.5 are generally safe.")

    args = a.parse_args()
    return args

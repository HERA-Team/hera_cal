# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

import numpy as np
import linsolve
from copy import deepcopy
from hera_cal.datacontainer import DataContainer
from hera_cal.utils import split_pol, conj_pol, split_bl, reverse_bl, join_bl, comply_pol, fft_dly
from hera_cal.apply_cal import calibrate_in_place
import six


def noise(size):
    """Return complex random gaussian array with given size and variance = 1."""
    # XXX I think this should be superceded by hera_sim
    sig = 1. / np.sqrt(2)
    return np.random.normal(scale=sig, size=size) + 1j * np.random.normal(scale=sig, size=size)


def sim_red_data(reds, gains=None, shape=(10, 10), gain_scatter=.1):
    """ Simulate noise-free random but redundant (up to differing gains) visibilities.
        # XXX I think this should be superceded by hera_sim

        Args:
            reds: list of lists of baseline-pol tuples where each sublist has only
                redundant pairs
            gains: pre-specify base gains to then scatter on top of in the
                {(index,antpol): np.array} format. Default gives all ones.
            shape: tuple of (Ntimes, Nfreqs). Default is (10,10).
            gain_scatter: Relative amplitude of per-antenna complex gain scatter. Default is 0.1.

        Returns:
            gains: true gains used in the simulation in the {(index,antpol): np.array} format
            true_vis: true underlying visibilities in the {(ind1,ind2,pol): np.array} format
            data: simulated visibilities in the {(ind1,ind2,pol): np.array} format
    """

    data, true_vis = {}, {}
    ants = list(set([ant for bls in reds for bl in bls for ant in
                    [(bl[0], split_pol(bl[2])[0]), (bl[1], split_pol(bl[2])[1])]]))
    if gains is None:
        gains = {}
    else:
        gains = deepcopy(gains)
    for ant in ants:
        gains[ant] = gains.get(ant, 1 + gain_scatter * noise((1,))) * np.ones(shape, dtype=np.complex)
    for bls in reds:
        true_vis[bls[0]] = noise(shape)
        for (i, j, pol) in bls:
            data[(i, j, pol)] = true_vis[bls[0]] * gains[(i, split_pol(pol)[0])] * gains[(j, split_pol(pol)[1])].conj()
    return gains, DataContainer(true_vis), DataContainer(data)


def get_pos_reds(antpos, bl_error_tol=1.0, low_hi=False):
    """ Figure out and return list of lists of redundant baseline pairs. Ordered by length. All baselines
        in a group have the same orientation with a preference for positive b_y and, when b_y==0, positive
        b_x where b((i,j)) = pos(j) - pos(i). This yields HERA baselines in i < j order.

        Args:
            antpos: dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}.
            bl_error_tol: the largest allowable difference between baselines in a redundant group
                (in the same units as antpos). Normally, this is up to 4x the largest antenna position error.
            low_hi: Check to make sure the first bl (i,j) in each bl group has i < j, otherwise conjugate all bls
                in the group.

        Returns:
            reds: list of lists of redundant tuples of antenna indices (no polarizations).
    """
    keys = antpos.keys()
    reds = {}
    array_is_flat = np.all(np.abs(np.array(antpos.values())[:, 2] - np.mean(antpos.values(), axis=0)[2]) < bl_error_tol / 4.0)
    for i, ant1 in enumerate(keys):
        for ant2 in keys[i + 1:]:
            delta = tuple(np.round(1.0 * (np.array(antpos[ant2]) - np.array(antpos[ant1])) / bl_error_tol).astype(int))
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

    orderedDeltas = [delta for (length, delta) in sorted(zip([np.linalg.norm(delta) for delta in reds.keys()], reds.keys()))]
    if low_hi:  # sort each group after sorting each
        return [sorted([tuple(sorted(bl)) for bl in reds[delta]]) for delta in orderedDeltas]
    else:  # sort each red and make sure the first antenna of the first bl in each group is the lowest antenna number
        return [sorted(reds[delta]) if sorted(reds[delta])[0][0] == np.min(reds[delta])
                else sorted([reverse_bl(bl) for bl in reds[delta]]) for delta in orderedDeltas]


def add_pol_reds(reds, pols=['xx'], pol_mode='1pol', ex_ants=[]):
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
        ex_ants: list of antennas to exclude in the [(1,'jxx'),(10,'jyy')] format

    Returns:
        reds: list of lists of redundant baseline tuples, e.g. (ind1,ind2,pol)
    """
    # pre-process to ensure pols complies w/ hera_cal polarization convention
    pols = [comply_pol(p) for p in pols]

    def excluded(bl, pol):
        return ((bl[0], split_pol(pol)[0]) in ex_ants) or ((bl[1], split_pol(pol)[1]) in ex_ants)

    redsWithPols, didBothCrossPolsForMinV = [], False
    for pol in pols:
        if pol_mode is not '4pol_minV' or pol[0] == pol[1]:
            redsWithPols += [[bl + (pol,) for bl in bls if not excluded(bl, pol)] for bls in reds]
        elif pol_mode is '4pol_minV' and not didBothCrossPolsForMinV:
            # Combine together e.g. 'xy' and 'yx' visibilities as redundant
            redsWithPols += [([bl + (pol,) for bl in bls if not excluded(bl, pol)]
                              + [bl + (conj_pol(pol),) for bl in bls if not excluded(bl, conj_pol(pol))]) for bls in reds]
            didBothCrossPolsForMinV = True
    return redsWithPols


def get_reds(antpos, pols=['xx'], pol_mode='1pol', ex_ants=[], bl_error_tol=1.0, low_hi=False):
    """ Combines redcal.get_pos_reds() and redcal.add_pol_reds(). See their documentation.

    Args:
        antpos: dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}.
        pols: a list of polarizations e.g. ['xx', 'xy', 'yx', 'yy']
        pol_mode: polarization mode of calibration
            '1pol': 1 antpol and 1 vispol (e.g. 'jxx' and 'xx'). Default.
            '2pol': 2 antpols, no cross-vispols (e.g. 'jxx','jyy' and 'xx','yy')
            '4pol': 2 antpols, 4 vispols (e.g. 'jxx','jyy' and 'xx','xy','yx','yy')
            '4pol_minV': 2 antpols, 4 vispols in data but assuming V_xy = V_yx in model
        ex_ants: list of antennas to exclude in the [(1,'jxx'),(10,'jyy')] format
        bl_error_tol: the largest allowable difference between baselines in a redundant group
            (in the same units as antpos). Normally, this is up to 4x the largest antenna position error.
        low_hi: For all returned baseline index tuples (i,j) to have i < j

    Returns:
        reds: list of lists of redundant baseline tuples, e.g. (ind1,ind2,pol)

    """
    pos_reds = get_pos_reds(antpos, bl_error_tol=bl_error_tol, low_hi=low_hi)
    return add_pol_reds(pos_reds, pols=pols, pol_mode=pol_mode, ex_ants=ex_ants)


def filter_reds(reds, bls=None, ex_bls=None, ants=None, ex_ants=None, ubls=None, ex_ubls=None, pols=None, ex_pols=None):
    '''
    Filter redundancies to include/exclude the specified bls, antennas, unique bl groups and polarizations.
    Arguments are evaluated, in order of increasing precedence: (pols, ex_pols, ubls, ex_ubls, bls, ex_bls,
    ants, ex_ants).
    Args:
        reds: list of lists of redundant (i,j,pol) baseline tuples, e.g. the output of get_reds()
        bls (optional): baselines to include.  Baselines of the form (i,j,pol) include that specific
            visibility.  Baselines of the form (i,j) are broadcast across all polarizations present in reds.
        ex_bls (optional): same as bls, but excludes baselines. 
        ants (optional): antennas to include.  Only baselines where both antenna indices are in ants
            are included.  Antennas of the form (i,pol) include that antenna/pol.  Antennas of the form i are
            broadcast across all polarizations present in reds.
        ex_ants (optional): same as ants, but excludes antennas.
        ubls (optional): redundant (unique baseline) groups to include.  Each baseline in ubls is taken to
            represent the redundant group containing it.  Baselines of the form (i,j) are broadcast across all
            polarizations, otherwise (i,j,pol) selects a specific redundant group.
        ex_ubls (optional): same as ubls, but excludes groups.
        pols (optional): polarizations to include in reds. e.g. ['XX','YY','XY','YX'].  Default includes all
            polarizations in reds.
        ex_pols (optional): same as pols, but excludes polarizations.
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
    return [gp for gp in reds if len(gp) > 0]


def check_polLists_minV(polLists):
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
        elif check_polLists_minV(polLists) and len(pols) == 4 and len(antpols) == 2:
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
        sol = self.sol0
        terms = [(linsolve.get_name(gi), linsolve.get_name(gj), linsolve.get_name(uij)) 
                 for term in self.all_terms for (gi, gj, uij) in term]
        dmdl_u = self._get_ans0(sol)
        chisq = sum([np.abs(self.data[k] - dmdl_u[k])**2 * self.wgts[k] for k in self.keys])
        # variables with '_u' are flattened and only include pixels that need updating
        dmdl_u = {k: v.flatten() for k, v in dmdl_u.items()}
        # wgts_u hold the wgts the user provides.  dwgts_u is what is actually used to wgt the data
        wgts_u = {k: (v * np.ones(chisq.shape, dtype=np.float32)).flatten() for k, v in self.wgts.items()}
        sol_u = {k: v.flatten() for k, v in sol.items()}
        iters = np.zeros(chisq.shape, dtype=np.int)
        conv = np.ones_like(chisq)
        update = np.where(chisq > 0)
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

    def __init__(self, reds):
        """Initialization of a class object for performing redundant calibration with logcal
        and lincal, both utilizing linsolve, and also degeneracy removal.

        Args:
            reds: list of lists of redundant baseline tuples, e.g. (ind1,ind2,pol). The first
                item in each list will be treated as the key for the unique baseline
        """

        self.reds = reds
        self.pol_mode = parse_pol_mode(self.reds)

    def build_eqs(self, dc):
        """Function for generating linsolve equation strings. Takes in a DataContainer to check whether 
        baselines in self.reds (or their complex conjugates) occur in the data. Returns a dictionary 
        that maps linsolve string to (ant1, ant2, pol) for all visibilities."""

        eqs = {}
        for ubl_index, blgrp in enumerate(self.reds):
            for ant_i, ant_j, pol in blgrp:
                if (ant_i, ant_j, pol) in dc:
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
        # XXX ARP: concerned about detrend_phs.  Why is it necessary?
        dtype = data.values()[0].dtype  # TODO: fix this for python 3
        dc = DataContainer(data)
        eqs = self.build_eqs(dc)
        self.phs_avg = {}  # detrend phases within redundant group, used for logcal to avoid phase wraps
        if detrend_phs:
            for blgrp in self.reds:
                self.phs_avg[blgrp[0]] = np.exp(-1j * np.median(np.unwrap([np.log(dc[bl]).imag for bl in blgrp], axis=0), axis=0))
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

    def firstcal(self, data, df, wgts={}, sparse=False, mode='default', norm=True, medfilt=False, kernel=(1, 11)):
        """Solves for a per-antenna delay by fitting a line to the phase difference between
        nominally redundant measurements.  To turn these delays into gains, you need to do:
        np.exp(2j * np.pi * delay * freqs)

        Args:
            data: visibility data in the dictionary format {(ant1,ant2,pol): np.array}
            df: frequency change between data bins, scales returned delays by 1/df.
            wgts: dictionary of linear weights in the same format as data. Defaults to equal wgts.
            sparse: represent the A matrix (visibilities to parameters) sparsely in linsolve
            norm: calculate delays from just the phase information (not the amplitude) of the data.
                This is a pretty effective way to get reliable delay even in the presence of RFI.
            medfilt : boolean, median filter data before fft.  This can work for data containing
                unflagged RFI, but tends to be less effective in practice than 'norm'.  Default False.
            kernel : size of median filter kernel along (time, freq) axes

        Returns:
            sol: dictionary of per-antenna delay solutions in the {(index,antpol): np.array}
                format.  All delays are multiplied by 1/df, so use that to set physical scale.
        """
        Nfreqs = six.next(six.itervalues(data)).shape[1]  # hardcode freq is axis 1 (time is axis 0)
        if len(wgts) == 0:
            wgts = {k: np.float32(1) for k in data}
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
                    taus_offs[(bl1, bl2)] = fft_dly(d12, df, wgts=w12, medfilt=medfilt, kernel=kernel)
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
        sol = {self.unpack_sol_key(k): v[0] for k, v in sol.items()}  # ignoring offset
        return sol

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
            max_iter: maximum number of lincal iterations allowed before it gives up
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
            max_iter: maximum number of lincal iterations allowed before it gives up
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

    def remove_degen_gains(self, antpos, gains, degen_gains=None, mode='phase'):
        """ Removes degeneracies from solutions (or replaces them with those in degen_sol).  This
        function in nominally intended for use with firstcal, which returns (phase/delay) solutions
        for antennas only.

        Args:
            antpos: dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}.
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
            new_gains = self.remove_degen_gains(antpos, pol0_gains, degen_gains=degen_gains, mode=mode)
            new_gains.update(self.remove_degen_gains(antpos, pol1_gains, degen_gains=degen_gains, mode=mode))
            self.pol_mode = '2pol'
            return new_gains

        # Extract gain and model visibiltiy solutions
        gainSols = np.array([gains[ant] for ant in ants])
        degenGains = np.array([degen_gains[ant] for ant in ants])

        # Build matrices for projecting gain degeneracies
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

    def remove_degen(self, antpos, sol, degen_sol=None):
        """ Removes degeneracies from solutions (or replaces them with those in degen_sol).  This
        function is nominally intended for use with solutions from logcal, omnical, or lincal, which
        return complex solutions for antennas and visibilities. 

        Args:
            antpos: dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}.
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
        new_gains = self.remove_degen_gains(antpos, gains, degen_gains=degen_sol, mode='complex')
        new_sol = deepcopy(vis)
        calibrate_in_place(new_sol, new_gains, old_gains=gains)
        new_sol.update(new_gains)
        return new_sol


def count_redcal_degeneracies(antpos, bl_error_tol=1.0):
    """Figures out whether an array is redundantly calibratable.

    Args:
        antpos: dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}.
        bl_error_tol: the largest allowable difference between baselines in a redundant group
            (in the same units as antpos). Normally, this is up to 4x the largest antenna position error.

    Returns:
        int: the number of 1-pol redundant baseline calibration degeneracies (4 means redundantly calibratable)
    """

    reds = get_reds(antpos, bl_error_tol=bl_error_tol)
    gains, true_vis, data = sim_red_data(reds, shape=(1, 1))
    cal = RedundantCalibrator(reds)
    ls = cal._solver(linsolve.LogProductSolver, data, wgts={})

    A, B = ls.ls_amp.get_A()[:, :, 0], ls.ls_phs.get_A()[:, :, 0]
    AtA, BtB = np.conj(A.T).dot(A), np.conj(B.T).dot(B)
    return len(AtA) + len(BtB) - np.linalg.matrix_rank(AtA) - np.linalg.matrix_rank(BtB)


def is_redundantly_calibratable(antpos, bl_error_tol=1.0):
    """Figures out whether an array is redundantly calibratable.

    Args:
        antpos: dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}.
        bl_error_tol: the largest allowable difference between baselines in a redundant group
            (in the same units as antpos). Normally, this is up to 4x the largest antenna position error.

    Returns:
        boolean: true if the number of 1pol degeneracies is 4 and thus the array is redundantly calibratable
    """

    return count_redcal_degeneracies(antpos, bl_error_tol=bl_error_tol) == 4

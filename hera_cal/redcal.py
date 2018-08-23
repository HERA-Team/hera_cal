import numpy as np
import linsolve
from copy import deepcopy
from hera_cal.datacontainer import DataContainer
from hera_cal.utils import split_pol, split_bl, join_bl, reverse_bl


# XXX I think this should be superceded by hera_sim
def noise(size):
    """Return complex random gaussian array with given size and variance = 1."""

    sig = 1. / np.sqrt(2)
    return np.random.normal(scale=sig, size=size) + 1j * np.random.normal(scale=sig, size=size)


# XXX I think this should be superceded by hera_sim
def sim_red_data(reds, gains=None, shape=(10, 10), gain_scatter=.1):
    """ Simulate noise-free random but redundant (up to differing gains) visibilities.

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
    ants = list(set([ant for bls in reds for bl in bls for ant in [(bl[0], bl[2][0]), (bl[1], bl[2][1])]]))
    if gains is None:
        gains = {}
    else:
        gains = deepcopy(gains)
    for ant in ants:
        gains[ant] = gains.get(ant, 1 + gain_scatter * noise((1,))) * np.ones(shape, dtype=np.complex)
    for bls in reds:
        true_vis[bls[0]] = noise(shape)
        for (i, j, pol) in bls:
            data[(i, j, pol)] = true_vis[bls[0]] * gains[(i, pol[0])] * gains[(j, pol[1])].conj()
    return gains, true_vis, data


def get_pos_reds(antpos, bl_error_tol=1.0, low_hi=False):
    """ Figure out and return list of lists of redundant baseline pairs. Ordered by length. All baselines
        in a group have the same orientation with a preference for positive b_y and, when b_y==0, positive
        b_x where b((i,j)) = pos(j) - pos(i). This yields HERA baselines in i < j order.

        Args:
            antpos: dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}.
            bl_error_tol: the largest allowable difference between baselines in a redundant group
                (in the same units as antpos). Normally, this is up to 4x the largest antenna position error.
            low_hi: For all returned baseline index tuples (i,j) to have i < j.

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
    if low_hi:
        return [[tuple(sorted(bl)) for bl in reds[delta]] for delta in orderedDeltas]
    else:
        return [reds[delta] for delta in orderedDeltas]


def add_pol_reds(reds, pols=['xx'], pol_mode='1pol', ex_ants=[]):
    """ Takes positonal reds (antenna indices only, no polarizations) and converts them
    into baseline tuples with polarization, depending on pols and pol_mode specified.

    Args:
        reds: list of list of antenna index tuples considered redundant
        pols: a list of polarizations e.g. ['xx', 'xy', 'yx', 'yy']
        pol_mode: polarization mode of calibration
            '1pol': 1 antpol and 1 vispol (e.g. 'x' and 'xx'). Default.
            '2pol': 2 antpols, no cross-vispols (e.g. 'x','y' and 'xx','yy')
            '4pol': 2 antpols, 4 vispols (e.g. 'x','y' and 'xx','xy','yx','yy')
            '4pol_minV': 2 antpols, 4 vispols in data but assuming V_xy = V_yx in model
        ex_ants: list of antennas to exclude in the [(1,'x'),(10,'y')] format

    Returns:
        reds: list of lists of redundant baseline tuples, e.g. (ind1,ind2,pol)
    """

    def excluded(bl, pol):
        return ((bl[0], pol[0]) in ex_ants) or ((bl[1], pol[1]) in ex_ants)

    redsWithPols, didBothCrossPolsForMinV = [], False
    for pol in pols:
        if pol_mode is not '4pol_minV' or pol[0] == pol[1]:
            redsWithPols += [[bl + (pol,) for bl in bls if not excluded(bl, pol)] for bls in reds]
        elif pol_mode is '4pol_minV' and not didBothCrossPolsForMinV:
            # Combine together e.g. 'xy' and 'yx' visibilities as redundant
            redsWithPols += [([bl + (pol,) for bl in bls if not excluded(bl, pol)]
                              + [bl + (pol[::-1],) for bl in bls if not excluded(bl, pol[::-1])]) for bls in reds]
            didBothCrossPolsForMinV = True
    return redsWithPols


def get_reds(antpos, pols=['xx'], pol_mode='1pol', ex_ants=[], bl_error_tol=1.0, low_hi=False):
    """ Combines redcal.get_pos_reds() and redcal.add_pol_reds(). See their documentation.

    Args:
        antpos: dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}.
        pols: a list of polarizations e.g. ['xx', 'xy', 'yx', 'yy']
        pol_mode: polarization mode of calibration
            '1pol': 1 antpol and 1 vispol (e.g. 'x' and 'xx'). Default.
            '2pol': 2 antpols, no cross-vispols (e.g. 'x','y' and 'xx','yy')
            '4pol': 2 antpols, 4 vispols (e.g. 'x','y' and 'xx','xy','yx','yy')
            '4pol_minV': 2 antpols, 4 vispols in data but assuming V_xy = V_yx in model
        ex_ants: list of antennas to exclude in the [(1,'x'),(10,'y')] format
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
    Assumes reds indices are Antpol objects.
    Args:
        reds: list of lists of redundant baseline tuples, e.g. (ind1,ind2,pol)
        bls (optional): baselines as antenna pair tuples (i,j) to include in reds.
        ex_bls (optional): baselines as antenna pair tuples (i,j) to exclude in reds.
        ants (optional): antenna numbers (as int's) to include in reds.
        ex_ants (optional): antenna numbers (as int's) to exclude in reds.
        ubls (optional): baselines representing their redundant group to include in reds.
        ex_ubls (optional): baselines representing their redundant group to exclude in reds.
        pols (optional): polarizations to include in reds. e.g. 'xy' or 'yx'.
        ex_pols (optional): polarizations to exclude in reds. e.g. 'xy' or 'yx'.
    Return:
        reds: list of lists of redundant baselines as antenna pair tuples.
    '''
    if pols is None: # if no pols are provided, deduce them from the reds
        pols = set(gp[0][2] for gp in reds)
    if ex_pols:
        pols = set(p for p in pols if p not in ex_pols)
    reds = [gp for gp in reds if gp[0][2] in pols]
    def expand_bls(gp):
        gp3 = [(g[0],g[1],p) for g in gp if len(g) == 2 for p in pols]
        return gp3 + [g for g in gp if len(g) == 3]
    antpols = set(sum([list(split_pol(p)) for p in pols],[]))
    def expand_ants(gp):
        gp2 = [(g,p) for g in gp if not hasattr(g,'__len__') for p in antpols]
        return gp2 + [g for g in gp if hasattr(g,'__len__')]
    def split_bls(bls):
        return [split_bl(bl) for bl in bls]
    if ubls or ex_ubls:
        bl2gp = {}
        for i,gp in enumerate(reds):
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
        reds = [gp for i,gp in enumerate(reds) if i in ubls and i not in ex_ubls]
    if bls:
        bls = set(expand_bls(bls))
    else: # default to set of all baselines
        bls = set(key for gp in reds for key in gp)
    if ex_bls:
        ex_bls = expand_bls(ex_bls)
        bls = set(k for k in bls if k not in ex_bls and reverse_bl(k) not in ex_bls)
    if ants:
        ants = expand_ants(ants)
        bls = set(join_bl(i,j) for i,j in split_bls(bls) if i in ants and j in ants)
    if ex_ants:
        ex_ants = expand_ants(ex_ants)
        bls = set(join_bl(i,j) for i,j in split_bls(bls) if i not in ex_ants and j not in ex_ants)
    bls.union(set(reverse_bl(k) for k in bls)) # put in reverse baselines, just in case
    reds = [[key for key in gp if key in bls] for gp in reds]
    return [gp for gp in reds if len(gp) > 1] # XXX do we want to filter off length one reds?


def check_polLists_minV(polLists):
    """Given a list of unique visibility polarizations (e.g. for each red group), returns whether
    they are all either single identical polarizations (e.g. 'xx') or both cross polarizations
    (e.g. ['xy','yx']) so that the 4pol_minV can be assumed."""

    for polList in polLists:
        ps = list()
        if len(polList) is 1:
            if polList[0][0] != polList[0][1]:
                return False
        elif len(polList) is 2:
            if polList[0] != polList[1][::-1] or polList[0][0] == polList[0][1]:
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
            '1pol': 1 antpol and 1 vispol (e.g. 'x' and 'xx'). Default.
            '2pol': 2 antpols, no cross-vispols (e.g. 'x','y' and 'xx','yy')
            '4pol': 2 antpols, 4 vispols (e.g. 'x','y' and 'xx','xy','yx','yy')
            '4pol_minV': 2 antpols, 4 vispols in data but assuming V_xy = V_yx in model
            'unrecognized_pol_mode': something else
    """

    pols = list(set([bl[2] for bls in reds for bl in bls]))
    antpols = list(set(''.join(pols)))
    if len(pols) == 1 and len(antpols) == 1:
        return '1pol'
    elif len(antpols) == 2 and set(pols) == set([2 * antpol for antpol in antpols]):
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


def _apply_gains(target, gains, operation, target_type):
    """Helper function designed to be used with divide_by_gains and multiply_by_gains."""

    assert(target_type in ['vis', 'gain'])
    output = deepcopy(target)
    if target_type is 'vis':
        # loop over len=3 keys in target
        for (ant1, ant2, pol) in [key for key in output.keys() if len(key) == 3]:
            try:
                output[(ant1, ant2, pol)] = operation(output[(ant1, ant2, pol)], gains[(ant1, pol[0])])
            except KeyError:
                pass
            try:
                output[(ant1, ant2, pol)] = operation(output[(ant1, ant2, pol)], np.conj(gains[(ant2, pol[1])]))
            except KeyError:
                pass
    elif target_type is 'gain':
        # loop over len=2 keys in target
        for ant in [key for key in output.keys() if len(key) == 2]:
            try:
                output[ant] = operation(output[ant], gains[ant])
            except KeyError:
                pass
    return output


def divide_by_gains(target, gains, target_type='vis'):
    """Helper function for applying gains to visibilities or other gains, e.g. for firstcal.

    Args:
        target: dictionary of gains in the {(ant,antpol): np.array} or visibilities in the
            {(ant1,ant2,pol): np.array} format. Target is copied and the original is untouched.
        gains: dictionary of gains in the {(ant,antpol): np.array} to apply . It can be a full
            'sol' dictionary with both gains and visibilities, but only the gains are used.
        target_type: either 'vis' (default) or 'gain'. For 'vis', only len=3 keys in the target
            are modified. For 'gain', only len=2 keys are modified.

    Returns:
        output: copy of target with gains divided out.
    """

    return _apply_gains(target, gains, (lambda x, y: x / y), target_type)


def multiply_by_gains(target, gains, target_type='vis'):
    """Helper function for removing gains from visibilities or other gains, e.g. for firstcal.

    Args:
        target: dictionary of gains in the {(ant,antpol): np.array} or visibilities in the
            {(ant1,ant2,pol): np.array} format. Target is copied and the original is untouched.
        gains: dictionary of gains in the {(ant,antpol): np.array} to remove. It can be a full
            'sol' dictionary with both gains and visibilities, but only the gains are used.
        target_type: either 'vis' (default) or 'gain'. For 'vis', only len=3 keys in the target
            are modified. For 'gain', only len=2 keys are modified.

    Returns:
        output: copy of target with gains multiplied back in.
    """

    return _apply_gains(target, gains, (lambda x, y: x * y), target_type)

class OmnicalSolver(linsolve.LinProductSolver):
    def __init__(self, data, sol0, wgts={}, gain=.3, **kwargs):
        """Set up a nonlinear system of equations of the form g_i * g_j.conj() * V_mdl = V_ij
        to linearize via the Omnical algorithm from Liu et al. 2010.

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
            gain: The fractional step made toward the new solution each iteration.
            **kwargs: keyword arguments of constants (python variables in keys of data that 
                are not to be solved for)

        Returns:
            None
        """
        linsolve.LinProductSolver.__init__(self, data, sol0, wgts=wgts, **kwargs)
        self.gain = np.float32(gain) # float32 to avoid accidentally promoting data to doubles.

    def _get_ans0(self, sol, keys=None):
        '''Evaluate the system of equations given input sol. 
        Specify keys to evaluate only a subset of the equations.'''
        if keys is None:
            keys = self.keys
        _sol = {k+'_':v.conj() for k,v in sol.items() if k.startswith('g')}
        _sol.update(sol)
        return {k: eval(k, _sol) for k in keys}

    def solve_iteratively(self, conv_crit=1e-10, maxiter=50, check_every=4, check_after=1, verbose=False):
        """Repeatedly solves and updates solution until convergence or maxiter is reached. 
        Returns a meta object containing the number of iterations, chisq, and convergence criterion.

        Args:
            conv_crit: A convergence criterion (default 1e-10) below which to stop iterating. 
                Converegence is measured L2-norm of the change in the solution of all the variables
                divided by the L2-norm of the solution itself.
            maxiter: An integer maximum number of iterations to perform before quitting. Default 50.
            check_every: Compute convergence every Nth iteration (saves computation).  Default 4.
            check_after: Start computing convergence only after N iterations.  Default 1.

        Returns: meta, sol
            meta: a dictionary with metadata about the solution, including
                iter: the number of iterations taken to reach convergence (or maxiter)
                chisq: the chi^2 of the solution produced by the final iteration
                conv_crit: the convergence criterion evaluated at the final iteration
            sol: a dictionary of complex solutions with variables as keys
        """
        sol = self.sol0
        terms = [(linsolve.get_name(gi),linsolve.get_name(gj),linsolve.get_name(uij)) 
            for term in self.all_terms for (gi,gj,uij) in term]
        dmdl_u = self._get_ans0(sol)
        chisq = sum([np.abs(self.data[k]-dmdl_u[k])**2 * self.wgts[k] for k in self.keys])
        # variables with '_u' are flattened and only include pixels that need updating
        dmdl_u = {k:v.flatten() for k,v in dmdl_u.items()}
        # wgts_u hold the wgts the user provides.  dwgts_u is what is actually used to wgt the data
        wgts_u = {k: (v * np.ones(chisq.shape, dtype=np.float32)).flatten() for k,v in self.wgts.items()}
        sol_u = {k:v.flatten() for k,v in sol.items()}
        iters = np.zeros(chisq.shape, dtype=np.int)
        conv = np.ones_like(chisq)
        update = np.where(chisq > 0)
        for i in range(1,maxiter+1):
            if verbose: print('Beginning iteration %d/%d' % (i,maxiter))
            if (i % check_every) == 1:
                # compute data wgts: dwgts = sum(V_mdl^2 / n^2) = sum(V_mdl^2 * wgts)
                # don't need to update data weighting with every iteration
                dwgts_u = {k: dmdl_u[k] * dmdl_u[k].conj() * wgts_u[k] for k in self.keys}
                sol_wgt_u = {k:0 for k in sol.keys()}
                for k,(gi,gj,uij) in zip(self.keys, terms):
                    w = dwgts_u[k]
                    sol_wgt_u[gi] += w
                    sol_wgt_u[gj] += w
                    sol_wgt_u[uij] += w
                dw_u = {k:v[update] * dwgts_u[k] for k,v in self.data.items()}
            sol_sum_u = {k:0 for k in sol_u.keys()}
            for k,(gi,gj,uij) in zip(self.keys, terms):
                # compute sum(wgts * V_meas / V_mdl)
                numerator = dw_u[k] / dmdl_u[k]
                sol_sum_u[gi] += numerator
                sol_sum_u[gj] += numerator.conj()
                sol_sum_u[uij] += numerator
            new_sol_u = {k: v * ((1 - self.gain) + self.gain * sol_sum_u[k]/sol_wgt_u[k]) 
                            for k,v in sol_u.items()}
            dmdl_u = self._get_ans0(new_sol_u)
            # check if i % check_every is 0, which is purposely one less than the '1' up at the top of the loop
            if i < maxiter and (i < check_after or (i % check_every) != 0):
                # Fast branch when we aren't expensively computing convergence/chisq
                sol_u = new_sol_u
            else:
                # Slow branch when we compute convergence/chisq
                new_chisq_u = sum([np.abs(v[update]-dmdl_u[k])**2 * wgts_u[k] for k,v in self.data.items()])
                chisq_u = chisq[update]
                gotbetter_u = (chisq_u > new_chisq_u)
                where_gotbetter_u = np.where(gotbetter_u)
                update_where = tuple(u[where_gotbetter_u] for u in update)
                chisq[update_where] = new_chisq_u[where_gotbetter_u]
                iters[update_where] = i
                new_sol_u = {k: np.where(gotbetter_u, v, sol_u[k]) for k,v in new_sol_u.items()}
                deltas_u = [v-sol_u[k] for k,v in new_sol_u.items()]
                conv_u = np.sqrt(sum([(v*v.conj()).real for v in deltas_u]) \
                            / sum([(v*v.conj()).real for v in new_sol_u.values()]))
                conv[update_where] = conv_u[where_gotbetter_u]
                for k,v in new_sol_u.items():
                    sol[k][update] = v
                update_u = np.where((conv_u > conv_crit) & gotbetter_u)
                if update_u[0].size == 0 or i == maxiter:
                    meta = {'iter': iters, 'chisq': chisq, 'conv_crit': conv}
                    return meta, sol
                dmdl_u = {k:v[update_u] for k,v in dmdl_u.items()}
                wgts_u = {k:v[update_u] for k,v in wgts_u.items()}
                sol_u = {k: v[update_u] for k,v in new_sol_u.items()}
                update = tuple(u[update_u] for u in update)
            if verbose: print('    <CHISQ> = %f, <CONV> = %f, CNT = %d', (np.mean(chisq), np.mean(conv), update[0].size))


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

    def build_eqs(self, bls_in_data):
        """Function for generating linsolve equation strings. Takes in a list of baselines that
        occur in the data in the (ant1,ant2,pol) format and returns a dictionary that maps
        linsolve string to (ant1, ant2, pol) for all visibilities."""

        eqs = {}
        for ubl_index, blgrp in enumerate(self.reds):
            for ant_i, ant_j, pol in blgrp:
                if (ant_i, ant_j, pol) in bls_in_data:
                    params = (ant_i, pol[0], ant_j, pol[1], ubl_index, blgrp[0][2])
                    eqs['g%d%s * g%d%s_ * u%d%s' % params] = (ant_i, ant_j, pol)
        return eqs

    def _solver(self, solver, data, wgts={}, detrend_phs=False, **kwargs):
        """Instantiates a linsolve solver for performing redcal.

        Args:
            solver: linsolve solver (e.g. linsolve.LogProductSolver or linsolve.LinProductSolver)
            data: visibility data in the dictionary format {(ant1,ant2,pol): np.array}
            wgts: dictionary of linear weights in the same format as data. Defaults to equal wgts.
            detrend_phs: takes out average phase, useful for logcal
            **kwargs: other keyword arguments passed into the solver for use by linsolve

        Returns:
            solver: instantiated solver with redcal equations and weights
        """
        # XXX ARP: concerned about detrend_phs.  Why is it necessary?
        dtype = data.values()[0].dtype
        dc = DataContainer(data)
        eqs = self.build_eqs(dc.keys())
        self.phs_avg = {}  # detrend phases within redundant group, used for logcal to avoid phase wraps
        if detrend_phs:
            for blgrp in self.reds:
                self.phs_avg[blgrp[0]] = np.exp(-np.complex64(1j) * np.median(np.unwrap([np.log(dc[bl]).imag for bl in blgrp], axis=0), axis=0))
                for bl in blgrp:
                    self.phs_avg[bl] = self.phs_avg[blgrp[0]]
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
            return (int(k[1:-1]), k[-1])
        else:  # 'u' = unique baseline solution
            return self.reds[int(k[1:-2])][0]

    def pack_sol_key(self, k):
        """Turn an antenna or baseline tuple (with polarization) into linsolve's internal variable string."""

        if len(k) == 2:  # 'g' = gain solution
            return 'g%d%s' % k
        else:  # 'u' = unique baseline solution
            ubl_num = [cnt for cnt, blgrp in enumerate(self.reds) if blgrp[0] == k][0]
            return 'u%d%s' % (ubl_num, k[-1])

    def compute_ubls(self, data, gain_sols):
        """Given a set of guess gain solutions, return a dictionary of calibrated visbilities
        averged over a redundant group. Not strictly necessary for typical operation."""

        dc = DataContainer(data)
        ubl_sols = {}
        for ubl, blgrp in enumerate(self.reds):
            d_gp = [dc[bl] for bl in blgrp]
            ubl_sols[blgrp[0]] = np.average(d_gp, axis=0)  # XXX add option for median here?
        return ubl_sols

    def logcal(self, data, sol0={}, wgts={}, sparse=False, mode='default'):
        """Takes the log to linearize redcal equations and minimizes chi^2.

        Args:
            data: visibility data in the dictionary format {(ant1,ant2,pol): np.array}
            sol0: dictionary that includes all starting (e.g. firstcal) gains in the
                {(ant,antpol): np.array} format. These are divided out of the data before
                logcal and then multiplied back into the returned gains in the solution.
                Default empty dictionary does nothing.
            wgts: dictionary of linear weights in the same format as data. Defaults to equal wgts.
            sparse: represent the A matrix (visibilities to parameters) sparsely in linsolve

        Returns:
            sol: dictionary of gain and visibility solutions in the {(index,antpol): np.array}
                and {(ind1,ind2,pol): np.array} formats respectively
        """

        fc_data = divide_by_gains(data, sol0, target_type='vis')
        ls = self._solver(linsolve.LogProductSolver, fc_data, wgts=wgts, detrend_phs=True, sparse=sparse)
        sol = ls.solve(mode=mode)
        sol = {self.unpack_sol_key(k): sol[k] for k in sol.keys()}
        for ubl_key in [k for k in sol.keys() if len(k) == 3]:
            sol[ubl_key] = sol[ubl_key] * self.phs_avg[ubl_key].conj()
        sol_with_fc = multiply_by_gains(sol, sol0, target_type='gain')
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

    def remove_degen(self, antpos, sol, degen_sol=None):
        """ Removes degeneracies from solutions (or replaces them with those in degen_sol).

        Args:
            antpos: dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}.
            sol: dictionary that contains both visibility and gain solutions in the
                {(ind1,ind2,pol): np.array} and {(index,antpol): np.array} formats respectively
            degen_sol: Optional dictionary in the same format as sol. Gain amplitudes and phases
                in degen_sol replace the values of sol in the degenerate subspace of redcal. If
                left as None, average gain amplitudes will be 1 and average phase terms will be 0.
                Visibilties in degen_sol are ignored. Putting in firstcal solutions here can
                help avoid phasewrapping issues.
        Returns:
            newSol: sol with degeneracy removal/replacement performed
        """

        g, v = get_gains_and_vis_from_sol(sol)
        if degen_sol is None:
            degen_sol = {key: np.ones_like(val) for key, val in g.items()}
        ants = g.keys()

        gainPols = np.array([ant[1] for ant in ants])
        # gainPols is list of antpols, one per antenna
        antpols = list(set(gainPols))
        positions = np.array([antpos[ant[0]] for ant in ants])
        bl_pairs = v.keys()
        visPols = np.array([[bl[2][0], bl[2][1]] for bl in bl_pairs])
        # visPols is list of pol, one per baseline
        bl_vecs = np.array([antpos[bl_pair[0]] - antpos[bl_pair[1]] for bl_pair in bl_pairs])
        if self.pol_mode not in ['1pol', '2pol', '4pol', '4pol_minV']:
            raise ValueError('Remove_degen cannot operate on pol_mode determined from reds')

        # if mode is 2pol, run as two 1pol remove degens
        if self.pol_mode is '2pol':
            self.pol_mode = '1pol'
            newSol = self.remove_degen(antpos, {key: val for key, val in sol.items()
                                                if antpols[0] in key[-1]}, degen_sol=degen_sol)
            newSol.update(self.remove_degen(antpos, {key: val for key, val in sol.items()
                                                     if antpols[1] in key[-1]}, degen_sol=degen_sol))
            self.pol_mode = '2pol'
            return newSol

        # Extract gain and model visibiltiy solutions
        gainSols = np.array([sol[ant] for ant in ants])
        visSols = np.array([sol[bl_pair] for bl_pair in bl_pairs])
        degenGains = np.array([degen_sol[ant] for ant in ants])

        # Amplitude renormalization: fixes the mean abs product of gains (as they appear in visibilities)
        for antpol in antpols:
            meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                       for key2 in g.keys() if key1[1] == antpol and key2[1] == antpol and key1[0] != key2[0]], axis=0)
            degenMeanSqAmplitude = np.mean([np.abs(degen_sol[key1] * degen_sol[key2]) for key1 in g.keys()
                                            for key2 in g.keys() if key1[1] == antpol and key2[1] == antpol and key1[0] != key2[0]], axis=0)
            gainSols[gainPols == antpol] *= (degenMeanSqAmplitude / meanSqAmplitude)**.5
            visSols[visPols[:, 0] == antpol] *= (meanSqAmplitude / degenMeanSqAmplitude)**.5
            visSols[visPols[:, 1] == antpol] *= (meanSqAmplitude / degenMeanSqAmplitude)**.5

        # Fix phase terms
        if self.pol_mode is '1pol' or self.pol_mode is '4pol_minV':
            # In 1pol and 4pol_minV, the phase degeneracies are 1 overall phase and 2 tip-tilt terms
            # Rgains maps gain phases to degenerate parameters (either average phases or phase slopes)
            Rgains = np.hstack((positions, np.ones((positions.shape[0], 1))))
            # Rvis maps visibility phases to the same set of degenerate parameters, keeping chi^2 constant
            Rvis = np.hstack((-bl_vecs, np.zeros((len(bl_vecs), 1))))
        else:  # pole_mode is '4pol'
            # two columns give sums for two different polarizations
            phasePols = np.vstack((gainPols == antpols[0], gainPols == antpols[1])).T
            Rgains = np.hstack((positions, phasePols))
            # These terms detect cross terms only, which pick up overall phase terms in 4pol (see HERA memo #30)
            is_ab = np.array((visPols[:, 0] == antpols[0]) * (visPols[:, 1] == antpols[1]), dtype=float)
            is_ba = np.array((visPols[:, 0] == antpols[1]) * (visPols[:, 1] == antpols[0]), dtype=float)
            visPhaseSigns = np.vstack((is_ab - is_ba, is_ba - is_ab)).T
            Rvis = np.hstack((-bl_vecs, -visPhaseSigns))
        # Mgains is like (AtA)^-1 At in linear estimator formalism. It's a normalized estimator of degeneracies
        Mgains = np.linalg.pinv(Rgains.T.dot(Rgains)).dot(Rgains.T)
        # degenToRemove is the amount we need to move in the degenerate subspace
        degenToRemove = np.einsum('ij,jkl', Mgains, np.angle(gainSols * np.conj(degenGains)))
        # Now correct gains and visibilities while preserving chi^2
        gainSols *= np.exp(np.complex64(-1j) * np.einsum('ij,jkl', Rgains, degenToRemove))
        visSols *= np.exp(np.complex64(-1j) * np.einsum('ij,jkl', Rvis, degenToRemove))

        # Create new solutions dictionary
        newSol = {ant: gainSol for ant, gainSol in zip(ants, gainSols)}
        newSol.update({bl_pair: visSol for bl_pair, visSol in zip(bl_pairs, visSols)})

        return newSol


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


# XXX return_extra_degens is unused
def is_redundantly_calibratable(antpos, bl_error_tol=1.0, return_extra_degens=False):
    """Figures out whether an array is redundantly calibratable.

    Args:
        antpos: dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}.
        bl_error_tol: the largest allowable difference between baselines in a redundant group
            (in the same units as antpos). Normally, this is up to 4x the largest antenna position error.

    Returns:
        boolean: true if the number of 1pol degeneracies is 4 and thus the array is redundantly calibratable
    """

    return count_redcal_degeneracies(antpos, bl_error_tol=bl_error_tol) == 4

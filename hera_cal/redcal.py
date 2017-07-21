import numpy as np
from copy import deepcopy


def noise(size):
    """Return complex random gaussian array with given size and variance = 1."""

    sig = 1./np.sqrt(2)
    return np.random.normal(scale=sig, size=size) + 1j*np.random.normal(scale=sig, size=size)


def sim_red_data(reds, gains=None, shape=(10,10), gain_scatter=.1):
    """ Simulate noise-free random but redundant (up to differing gains) visibilities.

        Args:
            reds: list of lists of baseline (or bl-pol) tuples where each sublist has only 
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
    ants = list(set([ant for bls in reds for bl in bls for ant in [(bl[0],bl[2][0]), (bl[1],bl[2][1])]]))
    if gains is None: gains = {}
    else: gains = deepcopy(gains)
    for ant in ants:
        gains[ant] = gains.get(ant, 1+gain_scatter*noise((1,))) * np.ones(shape,dtype=np.complex)
    for bls in reds:
        true_vis[bls[0]] = noise(shape)
        for (i,j,pol) in bls:
            data[(i,j,pol)] = true_vis[bls[0]] * gains[(i,pol[0])] * gains[(j,pol[1])].conj()
    return gains, true_vis, data


def get_pos_reds(antpos, precisionFactor=1e6):
    """ Figure out and return list of lists of redundant baseline pairs. Ordered by length. 
        All baselines have the same orientation with a preference for positive b_y and, 
        when b_y==0, positive b_x where b((i,j)) = pos(i) - pos(j).

        Args:
            antpos: dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}.
            precisionFactor: factor that when multiplied by different baseline vectors and rounded
                to integer values, gives unique integer tuples for unique baselines

        Returns:
            reds: list of lists of redundant tuples of antenna indices (no polarizations)
    """

    keys = antpos.keys()
    reds = {}
    array_is_2D = np.all(np.all(np.array(antpos.values())[:,2]==0))
    for i,ant1 in enumerate(keys):
        for ant2 in keys[i+1:]:
            delta = tuple((precisionFactor*2.0 * (np.array(antpos[ant1]) - np.array(antpos[ant2]))).astype(int))
            # Multiply by 2.0 because rounding errors can mimic changes below the grid spacing
            if delta[0] > 0 or (delta[0]==0 and delta[1] > 0) or (delta[0]==0 and delta[1]==0 and delta[2] > 0):
                bl_pair = (ant1,ant2)
            else:
                delta = tuple([-d for d in delta])
                bl_pair = (ant2,ant1)
            # Check to make sure reds doesn't have the key plus or minus rounding error
            p_or_m = (0,-1,1)
            if array_is_2D:
                epsilons = [[dx,dy,0] for dx in p_or_m for dy in p_or_m]
            else:
                epsilons = [[dx,dy,dz] for dx in p_or_m for dy in p_or_m for dz in p_or_m]
            for epsilon in epsilons:
                newKey = (delta[0]+epsilon[0], delta[1]+epsilon[1], delta[2]+epsilon[2])
                if reds.has_key(newKey):
                    reds[newKey].append(bl_pair)
                    break
            if not reds.has_key(newKey):
                reds[delta] = [bl_pair]
    orderedDeltas = [delta for (length,delta) in sorted(zip([np.linalg.norm(delta) for delta in reds.keys()],reds.keys()))]   
    return [reds[delta] for delta in orderedDeltas]


def add_pol_reds(reds, pols=['xx'], pol_mode='1pol'):
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

    Returns:
        reds: list of lists of redundant baseline tuples, e.g. (ind1,ind2,pol)
    """

    redsWithPols, didBothCrossPolsForMinV = [], False
    for pol in pols:
        if pol_mode is not '4pol_minV' or pol[0] == pol[1]:
            redsWithPols += [[bl + (pol,) for bl in bls] for bls in reds]
        elif pol_mode is '4pol_minV' and not didBothCrossPolsForMinV:
            #Combine together e.g. 'xy' and 'yx' visibilities as redundant
            redsWithPols += [([bl + (pol,) for bl in bls] + [bl + (pol[::-1],) for bl in bls]) for bls in reds]
            didBothCrossPolsForMinV = True
    return redsWithPols


def get_reds(antpos, pols=['xx'], pol_mode='1pol', precisionFactor=1e6):
    """ Combines redcal.get_pos_reds() and redcal.add_pol_reds(). 
    
    Args:
        antpos: dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}.
        pols: a list of polarizations e.g. ['xx', 'xy', 'yx', 'yy']
        pol_mode: polarization mode of calibration
            '1pol': 1 antpol and 1 vispol (e.g. 'x' and 'xx'). Default.
            '2pol': 2 antpols, no cross-vispols (e.g. 'x','y' and 'xx','yy')
            '4pol': 2 antpols, 4 vispols (e.g. 'x','y' and 'xx','xy','yx','yy')
            '4pol_minV': 2 antpols, 4 vispols in data but assuming V_xy = V_yx in model
        precisionFactor: factor that when multiplied by different baseline vectors and rounded
            to integer values, gives unique integer tuples for unique baselines

    Returns:
        reds: list of lists of redundant baseline tuples, e.g. (ind1,ind2,pol)

    """
    pos_reds = get_pos_reds(antpos, precisionFactor=precisionFactor)
    return add_pol_reds(pos_reds, pols=pols, pol_mode=pol_mode)


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
    elif len(antpols) == 2 and set(pols) == set([2*antpol for antpol in antpols]):
        return '2pol'
    elif len(pols) == 4 and len(antpols) == 2:
        polLists = [list(set([bl[2] for bl in bls ])) for bls in reds]
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

        g = {key: val for key,val in sol.items() if len(key)==2}
        v = {key: val for key,val in sol.items() if len(key)==3}
        return g, v


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
            for ant_i,ant_j,pol in blgrp:
                if (ant_i,ant_j,pol) in bls_in_data:
                    params = (ant_i, pol[0], ant_j, pol[1], ubl_index, blgrp[0][2])
                    eqs['g%d%s * g%d%s_ * u%d%s' % params] = (ant_i,ant_j,pol)
        return eqs


    def _solver(self, solver, data, wgts={}, detrend_phs=False, sparse=False, **kwargs):
        """Instantiates a linsolve solver for performing redcal.

        Args:
            solver: linsolve solver (e.g. linsolve.LogProductSolver or linsolve.LinProductSolver)
            data: visibility data in the dictionary format {(ant1,ant2,pol): np.array}
            wgts: dictionary of linear weights in the same format as data. Defaults to equal wgts.
            detrend_phs: takes out average phase, useful for logcal
            sparse: represent the A matrix (visibilities to parameters) sparsely in linsolve
            **kwargs: other keyword arguments passed into the solver for use by linsolve

        Returns:
            solver: instantiated solver with redcal equations and weights
        """
        
        from hera_qm.datacontainer import DataContainer
        dc = DataContainer(data)
        eqs = self.build_eqs(dc.keys())
        self.phs_avg = {} # detrend phases within redundant group, used for logcal to avoid phase wraps
        if detrend_phs:
            for blgrp in self.reds:
                self.phs_avg[blgrp[0]] = np.exp(-1j*np.median(np.unwrap([np.log(dc[bl]).imag for bl in blgrp],axis=0), axis=0))
                for bl in blgrp: 
                    self.phs_avg[bl] = self.phs_avg[blgrp[0]]
        d_ls,w_ls = {}, {}
        for eq,key in eqs.items():
            d_ls[eq] = dc[key] * self.phs_avg.get(key,1)
        if len(wgts) > 0:
            wc = DataContainer(wgts)
            for eq,key in eqs.items(): w_ls[eq] = wc[key]
        return solver(data=d_ls, wgts=w_ls, sparse=sparse, **kwargs)

    def unpack_sol_key(self, k):
        """Turn linsolve's internal variable string into antenna or baseline tuple (with polarization)."""

        if k.startswith('g'): # 'g' = gain solution
            return (int(k[1:-1]),k[-1])
        else: # 'u' = unique baseline solution
            return self.reds[int(k[1:-2])][0]


    def pack_sol_key(self, k):
        """Turn an antenna or baseline tuple (with polarization) into linsolve's internal variable string."""

        if len(k) == 2: # 'g' = gain solution
            return 'g%d%s' % k
        else: # 'u' = unique baseline solution
            ubl_num = [cnt for cnt,blgrp in enumerate(self.reds) if blgrp[0] == k][0]
            return 'u%d%s' % (ubl_num, k[-1])


    def compute_ubls(self, data, gain_sols):
        """Given a set of guess gain solutions, return a dictionary of calibrated visbilities 
        averged over a redundant group. Not strictly necessary for typical operation."""

        from hera_qm.datacontainer import DataContainer
        dc = DataContainer(data)
        ubl_sols = {}
        for ubl, blgrp in enumerate(self.reds):
            d_gp = [dc[bl] for bl in blgrp]
            ubl_sols[blgrp[0]] = np.average(d_gp, axis=0) # XXX add option for median here?
        return ubl_sols

    

    def logcal(self, data, sol0={}, wgts={}, sparse=False):
        """Takes the log to linearize redcal equations and minimizes chi^2.

        Args:
            data: visibility data in the dictionary format {(ant1,ant2,pol): np.array}
            sol0: placeholder. TODO: figure out what we're supposed to do with firstcal solutions
            wgts: dictionary of linear weights in the same format as data. Defaults to equal wgts.
            sparse: represent the A matrix (visibilities to parameters) sparsely in linsolve

        Returns:
            sol: dictionary of gain and visibility solutions in the {(index,antpol): np.array}
                and {(ind1,ind2,pol): np.array} formats respectively
        """
        import linsolve
        ls = self._solver(linsolve.LogProductSolver, data, wgts=wgts, detrend_phs=True, sparse=sparse)
        sol = ls.solve()
        sol = {self.unpack_sol_key(k): sol[k] for k in sol.keys()}
        for ubl_key in [k for k in sol.keys() if len(k) == 3]:
            sol[ubl_key] = sol[ubl_key] * self.phs_avg[ubl_key].conj()
        return sol


    def lincal(self, data, sol0, wgts={}, sparse=False, conv_crit=1e-10, maxiter=50):
        """Taylor expands to linearize redcal equations and iteratively minimizes chi^2.

        Args:
            data: visibility data in the dictionary format {(ant1,ant2,pol): np.array}
            sol0: dictionary of guess gains and unique model visibilities, keyed by antenna tuples
                like (ant,antpol) or baseline tuples like 
            wgts: dictionary of linear weights in the same format as data. Defaults to equal wgts.
            sparse: represent the A matrix (visibilities to parameters) sparsely in linsolve
            conv_crit: maximum allowed relative change in solutions to be considered converged
            max_iter: maximum number of lincal iterations allowed before it gives up

        Returns:
            meta: dictionary of information about the convergence and chi^2 of the solution
            sol: dictionary of gain and visibility solutions in the {(index,antpol): np.array}
                and {(ind1,ind2,pol): np.array} formats respectively
        """
        
        import linsolve
        sol0 = {self.pack_sol_key(k):sol0[k] for k in sol0.keys()}
        ls = self._solver(linsolve.LinProductSolver, data, sol0=sol0, wgts=wgts, sparse=sparse)
        meta, sol = ls.solve_iteratively(conv_crit=conv_crit, maxiter=maxiter)
        return meta, {self.unpack_sol_key(k):sol[k] for k in sol.keys()}
    

    def remove_degen(self, antpos, sol, degen_sol=None):
        """ Removes degeneracies from solutions (or replaces them with those in degen_sol).

        Args:
            antpos: dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}.
            sol: dictionary that contains both visibility and gain solutions in the 
                {(ind1,ind2,pol): np.array} and {(index,antpol): np.array} formats respectively
            degen_sol: Optional dictionary in the same format as sol. Gain amplitudes and phases
                in degen_sol replace the values of sol in the degenerate subspace of redcal. If
                left as None, average gain amplitudes will be 1 and average phase terms will be 0.
        Returns:
            newSol: sol with degeneracy removal/replacement performed
    """

        g, v = get_gains_and_vis_from_sol(sol)
        if degen_sol is None: 
            degen_sol = {key: np.ones_like(val) for key,val in g.items()}
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
            raise ValueError, 'Remove_degen cannot operate on pol_mode determined from reds'

        #if mode is 2pol, run as two 1pol remove degens
        if self.pol_mode is '2pol':
            self.pol_mode = '1pol'
            newSol = self.remove_degen(antpos, {key: val for key,val in sol.items() 
                     if antpols[0] in key[-1]}, degen_sol=degen_sol)
            newSol.update(self.remove_degen(antpos, {key: val for key,val in sol.items() 
                     if antpols[1] in key[-1]}, degen_sol=degen_sol))
            self.pol_mode = '2pol'
            return newSol

        #Extract gain and model visibiltiy solutions
        gainSols = np.array([sol[ant] for ant in ants])
        visSols = np.array([sol[bl_pair] for bl_pair in bl_pairs])
        degenGains = np.array([degen_sol[ant] for ant in ants])

        #Amplitude renormalization: fixes the mean abs product of gains (as they appear in visibilities)
        for antpol in antpols:
            meanSqAmplitude = np.mean([np.abs(g[(ant1,pol[0])] * g[(ant2,pol[1])]) 
                for (ant1,ant2,pol) in bl_pairs if pol == 2*antpol], axis=0)
            degenMeanSqAmplitude = np.mean([np.abs(degen_sol[(ant1,pol[0])] * degen_sol[(ant2,pol[1])]) 
                for (ant1,ant2,pol) in bl_pairs if pol == 2*antpol], axis=0)
            gainSols[gainPols == antpol] *= (degenMeanSqAmplitude / meanSqAmplitude)**.5
            visSols[visPols[:,0] == antpol] *= (meanSqAmplitude / degenMeanSqAmplitude)**.5
            visSols[visPols[:,1] == antpol] *= (meanSqAmplitude / degenMeanSqAmplitude)**.5

        # Fix phase terms
        if self.pol_mode is '1pol' or self.pol_mode is '4pol_minV':
            # In 1pol and 4pol_minV, the phase degeneracies are 1 overall phase and 2 tip-tilt terms
            # Rgains maps gain phases to degenerate parameters (either average phases or phase slopes)
            Rgains = np.hstack((positions, np.ones((positions.shape[0],1))))
            # Rvis maps visibility phases to the same set of degenerate parameters, keeping chi^2 constant
            Rvis = np.hstack((-bl_vecs, np.zeros((len(bl_vecs),1))))
        else: # pole_mode is '4pol'
            # two columns give sums for two different polarizations
            phasePols = np.vstack((gainPols==antpols[0], gainPols==antpols[1])).T
            Rgains = np.hstack((positions, phasePols))
            # These terms detect cross terms only, which pick up overall phase terms in 4pol (see HERA memo #30)
            is_ab = np.array((visPols[:,0] == antpols[0]) * (visPols[:,1] == antpols[1]),dtype=float)
            is_ba = np.array((visPols[:,0] == antpols[1]) * (visPols[:,1] == antpols[0]),dtype=float)
            visPhaseSigns = np.vstack((is_ab-is_ba, is_ba-is_ab)).T
            Rvis = np.hstack((-bl_vecs, -visPhaseSigns))
        # Mgains is like (AtA)^-1 At in linear estimator formalism. It's a normalized estimator of degeneracies
        Mgains = np.linalg.pinv(Rgains.T.dot(Rgains)).dot(Rgains.T)
        # degenToRemove is the amount we need to move in the degenerate subspace
        degenToRemove = np.einsum('ij,jkl', Mgains, np.angle(gainSols)-np.angle(degenGains))
        # Now correct gains and visibilities while preserving chi^2
        gainSols *= np.exp(-1.0j * np.einsum('ij,jkl',Rgains,degenToRemove)) 
        visSols *= np.exp(-1.0j * np.einsum('ij,jkl',Rvis,degenToRemove)) 

        #Create new solutions dictionary
        newSol = {ant: gainSol for ant,gainSol in zip(ants,gainSols)}
        newSol.update({bl_pair: visSol for bl_pair,visSol in zip(bl_pairs,visSols)})
        return newSol
        

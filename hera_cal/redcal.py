from hera_qm.datacontainer import DataContainer
import linsolve
import numpy as np
from copy import deepcopy


def check_pol_mode(pol_mode, pols, antpols=None, pols_from_sols=False):
    """Ensure that the chosen pol_mode is consistent with the available visibility
    pols and antenna polarizations. If pols come from sols rather than data, 
    then it's OK for there to be just 3 in the 4pol_minV case."""

    if antpols is None:
        antpols = list(set([antpol for pol in pols for antpol in pol]))
    elif set([char for pol in pols for char in pol]) != set(antpols):
        raise ValueError, 'Visibility pols and antpols are inconsistent.'
    if pol_mode not in ['1pol', '2pol', '4pol', '4pol_minV']:
        raise ValueError, 'Unrecognized pol_mode: ' + str(pol_mode)
    elif (pol_mode is '1pol' and not (len(antpols) == 1 and len(pols) == 1)) \
          or (pol_mode is '2pol' and not (len(antpols) == 2 and \
              set(pols) == set([2*antpol for antpol in antpols]))) \
          or (pol_mode is '4pol' and not (len(antpols) == 2 and (len(pols) == 4))) \
          or (pol_mode is '4pol_minV' and not (len(antpols) == 2 \
              and (len(pols) == 4 or (pols_from_sols and len(pols) == 3)))):
        raise ValueError, 'Pols and antpols inconsistent with pol_mode.'


def noise(size):
    """Return complex random gaussian array with given size and variance = 1."""

    sig = 1./np.sqrt(2)
    return np.random.normal(scale=sig, size=size) + 1j*np.random.normal(scale=sig, size=size)


def sim_red_data(reds, pols, pol_mode, gains=None, shape=(10,10), gain_scatter=.1):
    """ Simulate noise-free random but redundant (up to differing gains) visibilities.

        Args:
            reds: list of lists of tuples of antenna indices where each sublist has only 
                redundant pairs
            pols: a list of polarizations e.g. ['xx', 'xy', 'yx', 'yy']
            pol_mode: polarization mode of calibration
                '1pol': 1 antpol and 1 vispol (e.g. 'x' and 'xx')
                '2pol': 2 antpols, no cross-vispols (e.g. 'x','y' and 'xx','yy')
                '4pol': 2 antpols, 4 vispols (e.g. 'x','y' and 'xx','xy','yx','yy')
                '4pol_minV': 2 antpols, 4 vispols in data but assuming V_xy = V_yx in model
            gains: pre-specify base gains to then scatter on top of in the 
                {(index,antpol): np.array} format. Default gives all ones.
            shape: tuple of (Ntimes, Nfreqs). Default is (10,10).
            gain_scatter: Relative amplitude of per-antenna complex gain scatter. Default is 0.1.

        Returns:
            gains: true gains used in the simulation in the {(index,antpol): np.array} format
            true_vis: true underlying visibilities in the {(ind1,ind2,pol): np.array} format
            data: simulated visibilities in the {(ind1,ind2,pol): np.array} format
    """

    check_pol_mode(pol_mode, pols)
    data, true_vis = {}, {}
    bls = reduce(lambda x,y: x+y, reds)
    ants = set(reduce(lambda x,y: x+y, bls))
    if gains is None: gains = {}
    else: gains = deepcopy(gains)
    for ai in ants:
        for pol in pols:
            gains[(ai,pol[0])] = gains.get((ai,pol[0]), 1+gain_scatter*noise((1,))) * np.ones(shape,dtype=np.complex)
            gains[(ai,pol[1])] = gains.get((ai,pol[1]), 1+gain_scatter*noise((1,))) * np.ones(shape,dtype=np.complex)
    for bls in reds:
        for pol in pols:
            if pol_mode is '4pol_minV':
                if not true_vis.has_key(bls[0]+(pol,)):
                    true_vis[bls[0]+(pol,)] = noise(shape)
                    true_vis[bls[0]+(pol[::-1],)] = np.array(true_vis[bls[0]+(pol,)])
            else:
                true_vis[bls[0]+(pol,)] = noise(shape)
    for bls in reds:
        for pol in pols:
            for bl in bls:
                data[bl+(pol,)] = true_vis[bls[0]+(pol,)] * gains[(bl[0],pol[0])] * gains[(bl[1],pol[1])].conj()
    return gains, true_vis, data

def get_reds(antpos, precisionFactor=1e6):
    """ Figure out and return list of lists of redundant baseline pairs. Ordered by length. 
        All baselines have the same orientation with a preference for positive b_y and, 
        when b_y==0, positive b_x where b((i,j)) = pos(i) - pos(j).

        Args:
            antpos: dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}.
            precisionFactor: factor that when multiplied by different baseline vectors and rounded
                to integer values, gives unique integer tuples for unique baselines

        Returns:
            reds: list of lists of redundant tuples of antenna indices
    """

    keys = antpos.keys()
    reds = {}
    for i,ant1 in enumerate(keys):
        for ant2 in keys[i+1:]:
            delta = tuple((precisionFactor * (np.array(antpos[ant1]) - np.array(antpos[ant2]))).astype(int))
            if delta[1] > 0 or (delta[1]==0 and delta[0] > 0):
                if reds.has_key(delta): reds[delta] += [(ant1,ant2)]
                else: reds[delta] = [(ant1,ant2)]
            else:
                delta = tuple([-d for d in delta])
                if reds.has_key(delta): reds[delta] += [(ant2,ant1)]
                else: reds[delta] = [(ant2,ant1)]
    orderedDeltas = [delta for (delta,length) in sorted(zip(reds.keys(), [np.linalg.norm(delta) for delta in reds.keys()]))]
    return [reds[delta] for delta in orderedDeltas]


class RedundantCalibrator:

    def __init__(self, reds, antpos, pol_mode):
        """Initialization of a class object for performing redundant calibration with logcal
        and lincal, both utilizing linsolve, and also degeneracy removal.

        Args:
            reds: list of lists of tuples of antenna indices where each sublist has only 
                redundant pairs
            antpos: dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}.
            pol_mode: polarization mode of calibration
                '1pol': 1 antpol and 1 vispol (e.g. 'x' and 'xx')
                '2pol': 2 antpols, no cross-vispols (e.g. 'x','y' and 'xx','yy')
                '4pol': 2 antpols, 4 vispols (e.g. 'x','y' and 'xx','xy','yx','yy')
                '4pol_minV': 2 antpols, 4 vispols in data but assuming V_xy = V_yx in model
        """

        self.reds, self.antpos, self.pol_mode = reds, antpos, pol_mode


    def build_eqs(self, bls, pols):
        """TODO: document"""
        eqs = {}
        for ubl, blgrp in enumerate(self.reds):
            blgrp = set(blgrp) 
            for i,j in blgrp.intersection(bls):
                for pi,pj in pols:
                    eqs[self.pack_eqs_key(i,pi,j,pj,ubl)] = (i,j,pi+pj)
        return eqs


    def _solver(self, solver, data, wgts={}, detrend_phs=False, sparse=False, **kwargs):
        """TODO: document"""
        dc = DataContainer(data)
        eqs = self.build_eqs(dc.bls(), dc.pols())
        self.phs_avg = {} # detrend phases within redundant group, used for logcal to avoid phase wraps
        if detrend_phs:
            for ubl, blgrp in enumerate(self.reds):
                for pol in dc.pols():
                    self.phs_avg[(blgrp[0],pol)] = np.exp(-1j*np.median(np.unwrap([np.log(dc[bl+(pol,)]).imag for bl in blgrp],axis=0), axis=0))
                    for bl in blgrp: self.phs_avg[bl+(pol,)] = self.phs_avg[(blgrp[0],pol)]
        d_ls,w_ls = {}, {}
        for eq,key in eqs.items():
            d_ls[eq] = dc[key] * self.phs_avg.get(key,1)
        if len(wgts) > 0:
            wc = DataContainer(wgts)
            for eq,key in eqs.items(): w_ls[eq] = wc[key]
        return solver(data=d_ls, wgts=w_ls, sparse=sparse, **kwargs)


    def pack_eqs_key(self, ant_i, pol_i, ant_j, pol_j, ubl_num):
        """TODO: document"""
        if self.pol_mode is '4pol_minV':
            # make xy and yx the same
            pol = ''.join(sorted([pol_i,pol_j])) 
        else:
            pol = pol_i + pol_j
        return 'g%d%s * g%d%s_ * u%d%s' % (ant_i,pol_i,ant_j,pol_j,ubl_num,pol)


    def unpack_sol_key(self, k):
        """TODO: document"""
        if k.startswith('g'): # 'g' = gain solution
            return (int(k[1:-1]),k[-1])
        else: # 'u' = unique baseline solution
            return self.reds[int(k[1:-2])][0]+ (k[-2:],) 


    def pack_sol_key(self, k):
        """TODO: document"""
        if len(k) == 2: # 'g' = gain solution
            return 'g%d%s' % k
        else: # 'u' = unique baseline solution
            ubl_num = [cnt for cnt,blgrp in enumerate(self.reds) if blgrp[0] == k[:2]][0]
            return 'u%d%s' % (ubl_num, k[-1])


    def compute_ubls(self, data, gain_sols):
        """TODO: document"""
        dc = DataContainer(data)
        ubl_sols = {}
        for ubl, blgrp in enumerate(self.reds):
            for pol in dc.pols():
                #d_gp = [dc[bl+(pol,)] / (gain_sols[(bl[0],pol[0])] * gain_sols[(bl[1],pol[1])].conj()) for bl in blgrp]
                d_gp = [dc[bl+(pol,)] for bl in blgrp]
                ubl_sols[blgrp[0]+(pol,)] = np.average(d_gp, axis=0) # XXX add option for median here?
        return ubl_sols


    def logcal(self, data, wgts={}, sparse=False):
        """TODO: document"""
        pols = list(set([key[2] for key in data.keys()]))
        check_pol_mode(self.pol_mode, pols)
        
        ls = self._solver(linsolve.LogProductSolver, data, wgts=wgts, detrend_phs=True, sparse=sparse)
        sol = ls.solve()
        sol = {self.unpack_sol_key(k): sol[k] for k in sol.keys()}
        for ubl_key in [k for k in sol.keys() if len(k) == 3]:
            sol[ubl_key] = sol[ubl_key] * self.phs_avg[ubl_key].conj()
        return sol


    def lincal(self, data, sol0, wgts={}, sparse=False, conv_crit=1e-10, maxiter=50): # XXX use itersolve eventually
        """TODO: document"""
        pols = list(set([key[2] for key in data.keys()]))
        antpols = list(set([key[1] for key in sol0.keys() if len(key)==2]))
        check_pol_mode(self.pol_mode, pols, antpols)
        
        sol0 = {self.pack_sol_key(k):sol0[k] for k in sol0.keys()}
        ls = self._solver(linsolve.LinProductSolver, data, sol0=sol0, wgts=wgts, sparse=sparse)
        meta, sol = ls.solve_iteratively(conv_crit=conv_crit, maxiter=maxiter)
        return meta, {self.unpack_sol_key(k):sol[k] for k in sol.keys()}
    

    def remove_degen(self, sol, degen_sol=None):
        """ Removes degeneracies from solutions (or replaces them with those in degen_sol).s

        Args:
            sol: dictionary that contains both visibility and gain solutions in the 
                {(ind1,ind2,pol): np.array} and {(index,antpol): np.array} formats respectively
            degen_sol: Optional dictionary in the same format as sol. Gain amplitudes and phases
                in degen_sol replace the values of sol in the degenerate subspace of redcal. If
                left as None, average gain amplitudes will be 1 and average phase terms will be 0.
        Returns:
            newSol: sol with degeneracy removal/replacement performed
    """

        ants = [key for key in sol.keys() if len(key)==2]
        gainPols = np.array([ant[1] for ant in ants])
        antpols = list(set(gainPols))

        positions = np.array([self.antpos[ant[0]] for ant in ants])
        bl_pairs = [key for key in sol.keys() if len(key)==3]
        visPols = np.array([[bl[2][0], bl[2][1]] for bl in bl_pairs])
        pols = list(set([bl[2] for bl in bl_pairs]))
        bl_vecs = np.array([self.antpos[bl_pair[0]] - self.antpos[bl_pair[1]] for bl_pair in bl_pairs])
        check_pol_mode(self.pol_mode, pols, antpols, pols_from_sols=True)

        #if mode is 2pol, run as two 1pol remove degens
        if self.pol_mode is '2pol':
            self.pol_mode = '1pol'
            newSol = self.remove_degen({key: val for key,val in sol.items() if antpols[0] in key[-1]}, degen_sol=degen_sol)
            newSol.update(self.remove_degen({key: val for key,val in sol.items() if antpols[1] in key[-1]}, degen_sol=degen_sol))
            self.pol_mode = '2pol'
            return newSol

        #Extract gain and model visibiltiy solutions
        gainSols = np.array([sol[ant] for ant in ants])
        visSols = np.array([sol[bl_pair] for bl_pair in bl_pairs])
        if degen_sol is None: 
            degenGains = np.ones_like(gainSols)
        else:
            degenGains = np.array([degen_sol[ant] for ant in ants])

        #Amplitude renormalization
        timesModified = np.zeros_like(visSols)
        for antpol in antpols:
            meanAmplitude = np.mean(np.abs(gainSols[gainPols == antpol]),axis=0)
            degenMeanAmplitude = np.mean(np.abs(degenGains[gainPols == antpol]),axis=0)
            gainSols[gainPols == antpol] *= (degenMeanAmplitude / meanAmplitude)
            visSols[visPols[:,0] == antpol] *= (meanAmplitude / degenMeanAmplitude)
            visSols[visPols[:,1] == antpol] *= (meanAmplitude / degenMeanAmplitude)

        # Fix phase terms
        if self.pol_mode is '1pol' or self.pol_mode is '4pol_minV':
            Rgains = np.hstack((positions, np.ones((positions.shape[0],1))))
            Rvis = np.hstack((-bl_vecs, np.zeros((len(bl_vecs),1))))
        else: # pole_mode is '4pol'
            phasePols = np.vstack((gainPols==antpols[0], gainPols==antpols[1])).T
            Rgains = np.hstack((positions, phasePols))
            is_ab = np.array((visPols[:,0] == antpols[0]) * (visPols[:,1] == antpols[1]),dtype=float)
            is_ba = np.array((visPols[:,0] == antpols[1]) * (visPols[:,1] == antpols[0]),dtype=float)
            visPhaseSigns = np.vstack((is_ab-is_ba, is_ba-is_ab)).T
            Rvis = np.hstack((-bl_vecs, -visPhaseSigns))
        Mgains = np.linalg.pinv(Rgains.T.dot(Rgains)).dot(Rgains.T)
        degenRemoved = np.einsum('ij,jkl',Mgains, np.angle(gainSols)-np.angle(degenGains))
        gainSols *= np.exp(-1.0j * np.einsum('ij,jkl',Rgains,degenRemoved)) 
        visSols *= np.exp(-1.0j * np.einsum('ij,jkl',Rvis,degenRemoved)) 

        #Create new solutions dictionary
        newSol = {ant: gainSol for ant,gainSol in zip(ants,gainSols)}
        newSol.update({bl_pair: visSol for bl_pair,visSol in zip(bl_pairs,visSols)})
        return newSol
        

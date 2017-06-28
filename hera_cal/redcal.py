from hera_qm.datacontainer import DataContainer
import linsolve
import numpy as np
from copy import deepcopy

def noise(size):
    sig = 1./np.sqrt(2)
    return np.random.normal(scale=sig, size=size) + 1j*np.random.normal(scale=sig, size=size)

def sim_red_data(reds, pols, gains=None, stokes_v_indep=True, shape=(10,10), gain_scatter=.1):
    data, true_vis = {}, {}
    if not stokes_v_indep: assert('xy' in pols and 'yx' in pols)
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
            vis = noise(shape)
            true_vis[bls[0]+(pol,)] = vis
            for bl in bls:
                data[bl+(pol,)] = vis * gains[(bl[0],pol[0])] * gains[(bl[1],pol[1])].conj()
                if not stokes_v_indep:
                    data[bl+(pol[::-1],)] = vis * gains[(bl[0],pol[1])] * gains[(bl[1],pol[0])].conj()
    return gains, true_vis, data

def get_reds(antpos, precisionFactor=1000000):
    """Returns an list of lists of tuples representing redundancies. Ordered by length. All baselines have the same 
    orientation with a preference for positive b_y and, when b_y==0, positive b_x where b((i,j)) = pos(i) - pos(j)."""
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
    def __init__(self, reds, antpos, stokes_v_indep=True):
        """Initializes based on list of lists of tuples of antenna indices in the order so that each sublist has only redundant pairs.
        Also takes a dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}."""
        self.reds, self.antpos, self.stokes_v_indep = reds, antpos, stokes_v_indep
    
    def build_eqs(self, bls, pols):
        eqs = {}
        for ubl, blgrp in enumerate(self.reds):
            blgrp = set(blgrp) 
            for i,j in blgrp.intersection(bls):
                for pi,pj in pols:
                    eqs[self.pack_eqs_key(i,pi,j,pj,ubl)] = (i,j,pi+pj)
        return eqs
    
    def _solver(self, solver, data, wgts={}, detrend_phs=False, sparse=False, **kwargs):
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
        if self.stokes_v_indep: pol = pol_i + pol_j
        else: pol = ''.join(sorted([pol_i,pol_j])) # make xy and yx the same
        return 'g%d%s * g%d%s_ * u%d%s' % (ant_i,pol_i,ant_j,pol_j,ubl_num,pol)
    
    def unpack_sol_key(self, k):
        if k.startswith('g'): # 'g' = gain solution
            return (int(k[1:-1]),k[-1])
        else: # 'u' = unique baseline solution
            return self.reds[int(k[1:-2])][0]+ (k[-2:],) 
    
    def pack_sol_key(self, k):
        if len(k) == 2: # 'g' = gain solution
            return 'g%d%s' % k
        else: # 'u' = unique baseline solution
            ubl_num = [cnt for cnt,blgrp in enumerate(self.reds) if blgrp[0] == k[:2]][0]
            return 'u%d%s' % (ubl_num, k[-1])
    
    def compute_ubls(self, data, gain_sols):
        dc = DataContainer(data)
        ubl_sols = {}
        for ubl, blgrp in enumerate(self.reds):
            for pol in dc.pols():
                #d_gp = [dc[bl+(pol,)] / (gain_sols[(bl[0],pol[0])] * gain_sols[(bl[1],pol[1])].conj()) for bl in blgrp]
                d_gp = [dc[bl+(pol,)] for bl in blgrp]
                ubl_sols[blgrp[0]+(pol,)] = np.average(d_gp, axis=0) # XXX add option for median here?
        return ubl_sols
    
    def logcal(self, data, wgts={}, sparse=False):
        ls = self._solver(linsolve.LogProductSolver, data, wgts=wgts, detrend_phs=True, sparse=sparse)
        sol = ls.solve()
        sol = {self.unpack_sol_key(k): sol[k] for k in sol.keys()}
        for ubl_key in [k for k in sol.keys() if len(k) == 3]:
            sol[ubl_key] = sol[ubl_key] * self.phs_avg[ubl_key].conj()
        return sol
    
    def lincal(self, data, sol0, wgts={}, sparse=False, conv_crit=1e-10, maxiter=50): # XXX use itersolve eventually
        #sol0 = dict(zip([self.pack_sol_key(k) for k in sol0.keys()],sol0.values()))
        sol0 = {self.pack_sol_key(k):sol0[k] for k in sol0.keys()}
        ls = self._solver(linsolve.LinProductSolver, data, sol0=sol0, wgts=wgts, sparse=sparse)
        meta, sol = ls.solve_iteratively(conv_crit=conv_crit, maxiter=maxiter)
        return meta, {self.unpack_sol_key(k):sol[k] for k in sol.keys()}
    
    def remove_degen(self, sol, degen_sol=None):
        """This function removes the omnical degeneracies in sol. If degen_sol is not specified, the amplitude terms are set to 1.0 
        and the phase terms are set to 0.0. Otherwise, the degneraices are set to match those in degen_sol, which must have all the 
        same keys as sol. Only resolves amplitude, phase, tip, and tilt degeneracies, which means that it will only work properly in 
        the single polarization case or when Stokes V is artificially minimized by assuming Vxy = Vyx*."""
        ants = [key for key in sol.keys() if len(key)==2]
        positions = [self.antpos[ant[0]] for ant in ants]
        bl_pairs = [key for key in sol.keys() if len(key)==3]
        bl_vecs = np.array([self.antpos[bl_pair[0]] - self.antpos[bl_pair[1]] for bl_pair in bl_pairs])
        
        #Extract gain and model visibiltiy solutions
        gainSols = np.array([sol[ant] for ant in ants])
        visSols = np.array([sol[bl_pair] for bl_pair in bl_pairs])
        if degen_sol is None: degenGains = np.ones_like(gainSols)
        else: degenGains = np.array([degen_sol[ant] for ant in ants])
        
        #Amplitude renormalization
        ampRenorm = (np.mean(np.abs(degenGains), axis=0) / np.mean(np.abs(gainSols), axis=0))
        newGainSols = gainSols * ampRenorm
        newVisSols = visSols / ampRenorm**2

        #Fix phase degeneracies
        Rgains = np.array([np.append(pos,1.0) for pos in positions])
        Mgains = np.linalg.pinv(Rgains.T.dot(Rgains)).dot(Rgains.T)
        Rvis = np.hstack((-bl_vecs, np.zeros((len(bl_pairs),1))))
        degenRemoved = np.einsum('ij,jkl',Mgains, np.angle(newGainSols)-np.angle(degenGains))
        newGainSols *= np.exp(-1.0j * np.einsum('ij,jkl',Rgains,degenRemoved)) 
        newVisSols *= np.exp(-1.0j * np.einsum('ij,jkl',Rvis,degenRemoved)) 
        
        #Create new solutions dictionary
        newSol = {}
        for ant,gainSol in zip(ants,newGainSols): newSol[ant] = gainSol
        for bl_pair,visSol in zip(bl_pairs,newVisSols): newSol[bl_pair] = visSol
        return newSol
        

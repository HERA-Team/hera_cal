import pytest 
import numpy as np
from copy import deepcopy
from hera_sim.antpos import linear_array

from .. import redcal
from .. import nucal

class TestRadialRedundantGroup:
    def setup(self):
        self.antpos = linear_array(6)
        self.pols = ['nn']
        self.reds = redcal.get_reds(self.antpos, pols=self.pols)
        
    def test_init(self):
        # Select group from reds
        radial_group = nucal.RadialRedundantGroup(self.reds[0], self.antpos)
        assert radial_group.pol == self.pols[0]
        radial_group = nucal.RadialRedundantGroup(self.reds[0], self.antpos, pol='nn')
        assert radial_group.pol == self.pols[0]
        radial_group = nucal.RadialRedundantGroup(self.reds[0], self.antpos, blvec=np.array([1, 0, 0]))
        assert np.allclose(radial_group.blvec, np.array([1, 0, 0]))
        
        for bls in radial_group:
            ant1, ant2, pol = bls
            blvec = (self.antpos[ant2] - self.antpos[ant1]) / np.linalg.norm(self.antpos[ant2] - self.antpos[ant1])
            assert np.allclose(radial_group.blvec, blvec)
            
        assert len(radial_group) == len(self.reds[0])
        
        baseline_lengths = []
        for bls in self.reds[0]:
            ant1, ant2, pol = bls
            baseline_lengths.append(np.linalg.norm(self.antpos[ant2] - self.antpos[ant1]))
            
        assert np.allclose(radial_group.baseline_lengths, np.sort(baseline_lengths))
        
        antpos = linear_array(4)
        pols = ['nn', 'ee']
        reds = redcal.get_reds(antpos, pols=pols)
        baselines = sum(reds, [])
        pytest.raises(ValueError, nucal.RadialRedundantGroup, baselines, antpos)
        
    def test_get_u_bounds(self):
        freqs = np.linspace(50e6, 250e6, 10)
        umodes = []
        for bls in self.reds[0]:
            ant1, ant2, pol = bls
            umodes.append(freqs * np.linalg.norm(self.antpos[ant2] - self.antpos[ant1]) / 2.998e8)
            
        umin, umax = np.min(umodes), np.max(umodes)
        radial_group = nucal.RadialRedundantGroup(self.reds[0], self.antpos)
        _umin, _umax = radial_group.get_u_bounds(freqs)
        assert np.isclose(umin, _umin)
        assert np.isclose(umax, _umax)
        
    def test_filter_groups(self):
        radial_group = nucal.RadialRedundantGroup(self.reds[0], self.antpos)
        
        # Minimum baseline length cut
        rg = deepcopy(radial_group)
        rg.filter_group(min_bl_cut=20)
        for blmag in rg.baseline_lengths:
            assert blmag > 20
        
        # Max baseline length cut
        rg = deepcopy(radial_group)
        rg.filter_group(max_bl_cut=50)
        for blmag in rg.baseline_lengths:
            assert blmag < 50
            
        # Filter by antenna number
        rg = deepcopy(radial_group)
        rg.filter_group(ex_ants=(0, 'Jnn'))
        for bls in rg:
            assert 0 not in bls
            
        # Filter all baselines by min_bl_cut
        rg = deepcopy(radial_group)
        rg.filter_group(min_bl_cut=1000)
        assert len(rg) == 0
        assert len(rg.baseline_lengths) == 0
        
        # Filter all baselines by ex_ants
        rg = deepcopy(radial_group)
        rg.filter_group(ex_ants=[(i, 'Jnn') for i in range(6)])
        assert len(rg) == 0
        assert len(rg.baseline_lengths) == 0

    def test_get_item(self):
        radial_group = nucal.RadialRedundantGroup(self.reds[0], self.antpos)
        for i in range(len(radial_group)):
            assert isinstance(radial_group[i], tuple)
            assert len(radial_group[i]) == 3



def test_is_frequency_redundant():
    antpos = {i: np.array([i, 0, 0]) for i in range(3)}
    freqs = np.linspace(1, 2, 10)
    
    # Two baselines in a linear array should be redundant
    bl1 = (0, 1, 'nn')
    bl2 = (0, 2, 'nn')
    assert nucal.is_frequency_redundant(bl1, bl2, freqs, antpos)
    
    # One baseline should technically be frequency redundant with itself
    assert nucal.is_frequency_redundant(bl1, bl1, freqs, antpos)
    
    # Narrowing the bandwidth should make the baselines not frequency redundant
    freqs = np.linspace(0.5, 0.6, 10)
    assert not nucal.is_frequency_redundant(bl1, bl2, freqs, antpos)
    
    # Orthogonal baselines should not be frequency redundant
    bl1 = (0, 1, 'nn')
    bl2 = (0, 2, 'nn')
    antpos = {0: np.array([0, 0, 0]), 1: np.array([1, 0, 0]), 2: np.array([0, 1, 0])}
    assert not nucal.is_frequency_redundant(bl1, bl2, freqs, antpos)
import pytest
import numpy as np
from copy import deepcopy
from hera_sim.antpos import linear_array, hex_array

from .. import redcal
from .. import nucal
from .. import utils


def test_is_frequency_redundant():
    antpos = {i: np.array([i, 0, 0]) for i in range(3)}
    freqs = np.linspace(1, 2, 10)

    # Two baselines in a linear array should be redundant
    bl1 = (0, 1, "nn")
    bl2 = (0, 2, "nn")
    assert nucal.is_frequency_redundant(bl1, bl2, freqs, antpos)

    # One baseline should technically be frequency redundant with itself
    assert nucal.is_frequency_redundant(bl1, bl1, freqs, antpos)

    # Narrowing the bandwidth should make the baselines not frequency redundant
    freqs = np.linspace(0.5, 0.6, 10)
    assert not nucal.is_frequency_redundant(bl1, bl2, freqs, antpos)

    # Orthogonal baselines should not be frequency redundant
    bl1 = (0, 1, "nn")
    bl2 = (0, 2, "nn")
    antpos = {0: np.array([0, 0, 0]), 1: np.array([1, 0, 0]), 2: np.array([0, 1, 0])}
    assert not nucal.is_frequency_redundant(bl1, bl2, freqs, antpos)


def test_get_unique_orientations():
    antpos = linear_array(7)
    radial_groups = nucal.get_unique_orientations(antpos)

    # Linear arrays should only have one unique orientation
    assert len(radial_groups) == 1

    # Give spatial redundancies to function and confirm result is the same
    reds = redcal.get_reds(antpos)
    _radial_groups = nucal.get_unique_orientations(antpos, reds)
    assert len(radial_groups) == len(_radial_groups)
    assert len(radial_groups[0]) == len(_radial_groups[0])

    # Check to see if function identifies flipped orientations
    reds[1] = [utils.reverse_bl(bls) for bls in reds[1]]
    radial_groups = nucal.get_unique_orientations(antpos, reds)
    assert len(radial_groups) == 1

    # If multiple polarizations are requested, list should be size 2
    radial_groups = nucal.get_unique_orientations(antpos, pols=["nn", "ee"])
    assert len(radial_groups) == 2

    # Filter by minimum number of baselines in unique orientation
    antpos = hex_array(4, outriggers=0, split_core=False)
    radial_groups = nucal.get_unique_orientations(
        antpos, pols=["nn"], min_ubl_per_orient=5
    )
    for group in radial_groups:
        assert len(group) >= 5


class TestRadialRedundantGroup:
    def setup(self):
        self.antpos = linear_array(6)
        self.pols = ["nn"]
        self.reds = redcal.get_reds(self.antpos, pols=self.pols)

    def test_init(self):
        # Select group from reds
        radial_group = nucal.RadialRedundantGroup(self.reds[0], self.antpos)
        assert radial_group.pol == self.pols[0]
        radial_group = nucal.RadialRedundantGroup(self.reds[0], self.antpos, pol="nn")
        assert radial_group.pol == self.pols[0]
        radial_group = nucal.RadialRedundantGroup(
            self.reds[0], self.antpos, blvec=np.array([1, 0, 0])
        )
        assert np.allclose(radial_group.blvec, np.array([1, 0, 0]))

        for bls in radial_group:
            ant1, ant2, pol = bls
            blvec = (self.antpos[ant2] - self.antpos[ant1]) / np.linalg.norm(
                self.antpos[ant2] - self.antpos[ant1]
            )
            assert np.allclose(radial_group.blvec, blvec)

        assert len(radial_group) == len(self.reds[0])

        baseline_lengths = []
        for bls in self.reds[0]:
            ant1, ant2, pol = bls
            baseline_lengths.append(
                np.linalg.norm(self.antpos[ant2] - self.antpos[ant1])
            )

        assert np.allclose(radial_group.baseline_lengths, np.sort(baseline_lengths))

        antpos = linear_array(4)
        pols = ["nn", "ee"]
        reds = redcal.get_reds(antpos, pols=pols)
        baselines = sum(reds, [])
        pytest.raises(ValueError, nucal.RadialRedundantGroup, baselines, antpos)

    def test_get_u_bounds(self):
        freqs = np.linspace(50e6, 250e6, 10)
        umodes = []
        for bls in self.reds[0]:
            ant1, ant2, pol = bls
            umodes.append(
                freqs
                * np.linalg.norm(self.antpos[ant2] - self.antpos[ant1])
                / nucal.SPEED_OF_LIGHT
            )

        umin, umax = np.min(umodes), np.max(umodes)
        radial_group = nucal.RadialRedundantGroup(self.reds[0], self.antpos)
        _umin, _umax = radial_group.get_u_bounds(freqs)
        assert np.isclose(umin, _umin)
        assert np.isclose(umax, _umax)

    def test_filter_groups(self):
        radial_group = nucal.RadialRedundantGroup(
            [red[0] for red in self.reds], self.antpos
        )

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
        rg.filter_group(ex_ants=(0, "Jnn"))
        for bls in rg:
            assert 0 not in bls

        # Filter all baselines by min_bl_cut
        rg = deepcopy(radial_group)
        rg.filter_group(min_bl_cut=1000)
        assert len(rg) == 0
        assert len(rg.baseline_lengths) == 0

        # Filter all baselines by ex_ants
        rg = deepcopy(radial_group)
        rg.filter_group(ex_ants=[(i, "Jnn") for i in range(6)])
        assert len(rg) == 0
        assert len(rg.baseline_lengths) == 0

    def test_get_item(self):
        radial_group = nucal.RadialRedundantGroup(self.reds[0], self.antpos)
        for i in range(len(radial_group)):
            assert isinstance(radial_group[i], tuple)
            assert len(radial_group[i]) == 3


class TestFrequencyRedundancy:
    def setup(self):
        self.antpos = hex_array(4, outriggers=0, split_core=False)
        self.freq_reds = nucal.FrequencyRedundancy(self.antpos)

    def test_init(self):
        pass

    def test_filter_groups(self):
        freq_reds = deepcopy(self.freq_reds)

        # Make sure all groups have at least 4 baselines
        freq_reds.filter_radial_groups(min_nbls=2)
        for group in freq_reds:
            assert len(group) >= 2

        # Filter out antenna (0, "Jnn")
        freq_reds = deepcopy(self.freq_reds)
        freq_reds.filter_radial_groups(ex_ants=[(0, "Jnn")])
        for group in freq_reds:
            for bls in group:
                assert 0 not in bls

    def test_get_item(self):
        """ """
        # Check indexing
        groups = [self.freq_reds[i] for i in range(len(self.freq_reds))]
        for gi, grp in enumerate(self.freq_reds):
            assert groups[gi] == grp

    def test_get_pol(self):
        freq_reds = nucal.FrequencyRedundancy(self.antpos, pols=["nn", "ee"])
        for group in self.freq_reds.get_pol("nn"):
            assert group.pol == "nn"

    def test_get_radial_group(self):
        reds = redcal.get_reds(self.antpos)
        for red in reds:
            group = self.freq_reds.get_radial_group(red[0])
            assert red[0] in group

    def test_get_redundant_group(self):
        # Loop through the redundancies and confirm that we can grab spatially redundant groups
        reds = redcal.get_reds(self.antpos)
        for red in reds:
            for bls in red:
                group = self.freq_reds.get_redundant_group(bls)
                assert bls in group

        # Make sure KeyError is raised if incorrect key not given
        pytest.raises(KeyError, self.freq_reds.get_redundant_group, (-1, -1, "nn"))

        # Check flipped baseline
        bls = (1, 0, "nn")
        group = self.freq_reds.get_redundant_group(bls)
        assert bls in group

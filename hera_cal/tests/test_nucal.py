import pytest
import numpy as np
from copy import deepcopy
from hera_sim.antpos import linear_array, hex_array

from .. import redcal
from .. import nucal
from .. import utils
from .. import abscal
from ..datacontainer import DataContainer

def test_get_u_bounds():
    antpos = {i: np.array([i, 0, 0]) for i in range(7)}
    freqs = np.linspace(50e6, 250e6, 10)
    radial_reds = nucal.RadialRedundancy(antpos)
    u_bounds = nucal.get_u_bounds(radial_reds, antpos, freqs)
    
    # List of u-bounds should be the same length as radial reds
    assert len(u_bounds) == len(radial_reds)

    baseline_lengths = [radial_reds.baseline_lengths[bl] for bl in radial_reds[0]]

    # Check the minimum and maximum u-bounds are what we would expect
    assert np.isclose(u_bounds[0][0], np.min(baseline_lengths) * freqs[0] / nucal.SPEED_OF_LIGHT)
    assert np.isclose(u_bounds[0][1], np.max(baseline_lengths) * freqs[-1] / nucal.SPEED_OF_LIGHT)

def test_is_same_orientation():
    antpos = {i: np.array([i, 0, 0]) for i in range(3)}
    
    # Add extra orthogonal baseline
    antpos[3] = np.array([0, 1, 0])
    bl1 = (0, 1, 'nn')
    bl2 = (0, 2, 'nn')
    bl3 = (0, 3, 'nn')

    # These baselines should have the same orientation
    assert nucal.is_same_orientation(bl1, bl2, antpos)

    # These baselines should not
    assert not nucal.is_same_orientation(bl1, bl3, antpos)

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

    # Try baselines with different polarizations
    antpos = {i: np.array([i, 0, 0]) for i in range(3)}
    bl1 = (0, 1, "en")
    bl2 = (0, 2, "nn")
    assert not nucal.is_frequency_redundant(bl1, bl2, freqs, antpos)

def test_get_unique_orientations():
    antpos = linear_array(7)

    # Give spatial redundancies to function and confirm result is the same
    reds = redcal.get_reds(antpos)
    radial_groups = nucal.get_unique_orientations(antpos, reds)
    assert len(radial_groups) == 1

    # Check to see if function identifies flipped orientations
    reds[1] = [utils.reverse_bl(bls) for bls in reds[1]]
    radial_groups = nucal.get_unique_orientations(antpos, reds)
    assert len(radial_groups) == 1

    # Filter by minimum number of baselines in unique orientation
    antpos = hex_array(4, outriggers=0, split_core=False)
    reds = redcal.get_reds(antpos)
    radial_groups = nucal.get_unique_orientations(antpos, reds, min_ubl_per_orient=5)
    for group in radial_groups:
        assert len(group) >= 5

class TestRadialRedundancy:
    def setup(self):
        self.antpos = hex_array(4, outriggers=0, split_core=False)
        self.radial_reds = nucal.RadialRedundancy(self.antpos)

    def test_init(self):
        reds = redcal.get_reds(self.antpos)
        radial_reds = nucal.RadialRedundancy(self.antpos, reds=reds)
        assert len(radial_reds.reds) == len(self.radial_reds.reds)

    def test_filter_groups(self):
        radial_reds = deepcopy(self.radial_reds)

        # Make sure all groups have at least 4 baselines
        radial_reds.filter_radial_groups(min_nbls=2)
        for group in radial_reds:
            assert len(group) >= 2

        # Filter out baseline lengths less than 20 meters
        radial_reds = deepcopy(self.radial_reds)
        radial_reds.filter_radial_groups(min_bl_cut=20)
        for group in radial_reds:
            for bl in group:
                assert radial_reds.baseline_lengths[bl] > 20

        # Filter out baseline lengths greater than 20 meters
        radial_reds = deepcopy(self.radial_reds)
        radial_reds.filter_radial_groups(max_bl_cut=20)
        for group in radial_reds:
            for bl in group:
                assert radial_reds.baseline_lengths[bl] < 20

    def test_get_item(self):
        """ """
        # Check indexing
        groups = [self.radial_reds[i] for i in range(len(self.radial_reds))]
        for gi, grp in enumerate(self.radial_reds):
            assert groups[gi] == grp

    def test_get_pol(self):
        radial_reds = nucal.RadialRedundancy(self.antpos, pols=["nn", "ee"])
        for group in self.radial_reds.get_pol("nn"):
            assert group[0][-1] == "nn"

    def test_get_radial_group(self):
        reds = redcal.get_reds(self.antpos)
        for red in reds:
            group = self.radial_reds.get_radial_group(red[0])
            assert red[0] in group

        # Try to get baseline not in data
        pytest.raises(KeyError, self.radial_reds.get_radial_group, (1, 1, 'nn'))

        # Try to get flipped baseline
        group1 = self.radial_reds.get_radial_group((1, 0, 'nn'))
        group2 = self.radial_reds.get_radial_group((0, 1, 'nn'))

        for bi in range(len(group1)):
            assert group1[bi] == utils.reverse_bl(group2[bi])

    def test_get_redundant_group(self):
        # Loop through the redundancies and confirm that we can grab spatially redundant groups
        reds = redcal.get_reds(self.antpos)
        for red in reds:
            for bls in red:
                group = self.radial_reds.get_redundant_group(bls)
                assert bls in group

        # Make sure KeyError is raised if incorrect key not given
        pytest.raises(KeyError, self.radial_reds.get_redundant_group, (-1, -1, "nn"))

        # Check flipped baseline
        bls = (1, 0, "nn")
        group = self.radial_reds.get_redundant_group(bls)
        assert bls in group

    def test_set_item(self):
        radial_reds = deepcopy(self.radial_reds)
        # Find radial group with 1 baseline
        radial_reds.sort(reverse=False)
        group = deepcopy(radial_reds[0])

        # Filter all groups with fewer than 3 baselines
        radial_reds.filter_radial_groups(min_nbls=3)
        radial_reds[0] = group

        # Add baseline group with same heading as existing heading
        antpos = linear_array(10)
        radial_reds = nucal.RadialRedundancy(antpos)
        radial_reds.filter_radial_groups(max_bl_cut=40)

        bls = []
        for i in range(10):
            if np.linalg.norm(antpos[i] - antpos[0]) > 40:
                bls.append((0, i, 'nn'))

        pytest.raises(ValueError, radial_reds.__setitem__, 0, bls)

    def test_append(self):
        radial_reds = deepcopy(self.radial_reds)
        # Find radial group with 1 baseline
        radial_reds.sort(reverse=False)
        group = deepcopy(radial_reds[0])

        # Filter all groups with fewer than 10 baselines
        radial_reds.filter_radial_groups(min_nbls=10)

        # Append group
        radial_reds.append(group)

        # Try appending something else
        pytest.raises(TypeError, radial_reds.append, "Test string")

        # Add baseline group with same heading as existing heading
        antpos = linear_array(10)
        radial_reds = nucal.RadialRedundancy(antpos)
        radial_reds.filter_radial_groups(max_bl_cut=40)

        bls = []
        for i in range(10):
            if np.linalg.norm(antpos[i] - antpos[0]) > 40:
                bls.append((0, i, 'nn'))

        pytest.raises(ValueError, radial_reds.append, bls)


    def test_add_radial_group(self):
        radial_reds = deepcopy(self.radial_reds)
        # Find radial group with 1 baseline
        radial_reds.sort(reverse=False)
        group1 = deepcopy(radial_reds[0])

        # Find group with more baselines
        for group in radial_reds:
            if len(group) == 3:
                group2 = deepcopy(group)
                break

        # Filter all groups with fewer than 10 baselines
        radial_reds.filter_radial_groups(min_nbls=10)
        
        # Add group with a single baseline
        radial_reds.add_radial_group(group1)

        # Add group with more baselines
        radial_reds.add_radial_group(group2)

        # Try to add a group with baselines that are not radially redundant
        pytest.raises(ValueError, radial_reds.add_radial_group, group1 + group2)

        # Same as above but with different polarizations
        group3 = [(0, 1, 'nn'), (0, 2, 'ee')]
        pytest.raises(ValueError, radial_reds.add_radial_group, group3)

        # Add baseline group with same heading as existing heading
        antpos = linear_array(10)
        radial_reds = nucal.RadialRedundancy(antpos)
        radial_reds.filter_radial_groups(max_bl_cut=40)

        bls = []
        for i in range(10):
            if np.linalg.norm(antpos[i] - antpos[0]) > 40:
                bls.append((0, i, 'nn'))

        radial_reds.add_radial_group(bls)


    def test_sort(self):
        radial_reds = deepcopy(self.radial_reds)
        radial_reds.sort(reverse=False)
        assert len(radial_reds[0]) < len(radial_reds[-1])
        radial_reds.sort(reverse=True)
        assert len(radial_reds[0]) > len(radial_reds[-1])

def test_compute_spatial_filters():
    # Generate a mock array for generating filters
    antpos = hex_array(3, split_core=False, outriggers=0)
    radial_reds = nucal.RadialRedundancy(antpos)
    radial_reds.filter_radial_groups(min_nbls=3)
    freqs = np.linspace(50e6, 250e6, 200)

    spatial_filters = nucal.compute_spatial_filters(radial_reds, freqs)

    # Number of filters should equal number of baselines in radial_reds
    assert len(spatial_filters) == sum(map(len, radial_reds))

    # First index of filter should equal number of frequencies
    # Second index is number of modeling components, should be less than or 
    # equal to number of frequencies
    for bl in spatial_filters:
        assert spatial_filters[bl].shape[0] == freqs.shape[0]
        assert spatial_filters[bl].shape[1] <= freqs.shape[0]

    # All filters within the same group should have the same size
    for rdgrp in radial_reds:
        filter_shape = spatial_filters[rdgrp[0]].shape
        for bl in rdgrp:
            assert filter_shape == spatial_filters[bl].shape

    # Show that filters can be used to model a common u-plane with 
    # uneven sampling
    antpos = linear_array(6, sep=5)
    radial_reds = nucal.RadialRedundancy(antpos)
    spatial_filters = nucal.compute_spatial_filters(radial_reds, freqs)
    data = []
    design_matrix = []
    for rdgrp in radial_reds:
        for bl in rdgrp:
            blmag = np.linalg.norm(antpos[bl[1]] - antpos[bl[0]])
            data.append(np.sin(freqs * blmag / 2.998e8))
            design_matrix.append(spatial_filters[bl])
            
    # Fit PSWF to mock data
    design_matrix = np.array(design_matrix)
    XTXinv = np.linalg.pinv(np.einsum('afm,afn->mn', design_matrix, design_matrix))
    Xy = np.einsum('afm,af->m', design_matrix, data)
    model = design_matrix @ (XTXinv @ Xy)
    np.allclose(model, data, atol=1e-6)


def test_build_nucal_wgts():
    bls = [(0, 1, 'ee'), (0, 2, 'ee'), (1, 2, 'ee')]
    auto_bls = [(0, 0, 'ee'), (1, 1, 'ee'), (2, 2, 'ee')]
    data_flags = DataContainer({bl: np.zeros((3, 4), dtype=bool) for bl in bls})
    data_flags.times_by_bl = {bl[:2]: np.arange(3) / 86400 for bl in bls}
    data_flags.freqs = np.arange(4)
    data_flags.antpos = {0: np.array([0, 0, 0]), 1: np.array([10, 0, 0]), 2: np.array([20, 0, 0])}
    data_flags.data_antpos = {0: np.array([0, 0, 0]), 1: np.array([10, 0, 0]), 2: np.array([20, 0, 0])}
    data_nsamples = DataContainer({bl: np.ones((3, 4), dtype=float) for bl in bls})
    data_nsamples[(0, 1, 'ee')][1, 1] = 2
    model_flags = data_flags
    autocorrs = DataContainer({bl: np.ones((3, 4), dtype=complex) for bl in auto_bls})
    autocorrs[(1, 1, 'ee')][2, 2] = 3
    auto_flags = DataContainer({bl: np.zeros((3, 4), dtype=bool) for bl in auto_bls})

    radial_reds = nucal.RadialRedundancy(data_flags.data_antpos, pols=['ee'])
    freqs = np.linspace(100e6, 200e6, 4)

    #  Set weights for low end of the frequency band to zeros
    wgts = nucal.build_nucal_wgts(data_flags, data_nsamples, autocorrs, auto_flags, radial_reds, freqs, min_freq_cut=130e6)
    for key in wgts:
        assert np.allclose(wgts[key][:, 0], 0)

    # Set weights for high end of the frequency band to zeros
    wgts = nucal.build_nucal_wgts(data_flags, data_nsamples, autocorrs, auto_flags, radial_reds, freqs, max_freq_cut=180e6)
    for key in wgts:
        assert np.allclose(wgts[key][:, -1], 0)

    # Set weights for samples below a certain u-magnitude to zero
    wgts = nucal.build_nucal_wgts(data_flags, data_nsamples, autocorrs, auto_flags, radial_reds, freqs, min_u_cut=10)
    for key in wgts:
        bl = radial_reds.baseline_lengths[radial_reds._bl_to_red_key[key]]
        umodes = freqs * bl / 2.998e8
        assert np.allclose(wgts[key][:, umodes < 10], 0)

    # Set weights for samples above a certain u-magnitude to zero
    wgts = nucal.build_nucal_wgts(data_flags, data_nsamples, autocorrs, auto_flags, radial_reds, freqs, max_u_cut=10)
    for key in wgts:
        bl = radial_reds.baseline_lengths[radial_reds._bl_to_red_key[key]]
        umodes = freqs * bl / 2.998e8
        assert np.allclose(wgts[key][:, umodes > 10], 0)

    # Set weights for samples above a certain u-magnitude to zero
    wgts = nucal.build_nucal_wgts(data_flags, data_nsamples, autocorrs, auto_flags, radial_reds, freqs, spw_range_flags=[(120e6, 180e6)])
    for key in wgts:
        assert np.allclose(wgts[key][:, [1, 2]], 0)

    # Assert that weights are the same in the case when there are no model flags or cuts in u-magnitude or frequency
    abscal_wgts = abscal.build_data_wgts(data_flags, data_nsamples, model_flags, autocorrs, auto_flags)
    nucal_wgts = nucal.build_nucal_wgts(data_flags, data_nsamples, autocorrs, auto_flags, radial_reds, freqs)
    for key in abscal_wgts:
        assert np.allclose(abscal_wgts[key], nucal_wgts[key])

def test_project_u_model_comps_on_spec_axis():
    # Test that the projection of the u-model components on the spectral axis
    # is the same as the projection of the data on the spectral axis
    antpos = linear_array(6, sep=5)
    radial_reds = nucal.RadialRedundancy(antpos)
    spatial_filters = nucal.compute_spatial_filters(radial_reds, freqs)
    data = []
    design_matrix = []
    for rdgrp in radial_reds:
        for bl in rdgrp:
            blmag = np.linalg.norm(antpos[bl[1]] - antpos[bl[0]])
            data.append(np.sin(freqs * blmag / 2.998e8))
            design_matrix.append(spatial_filters[bl])
    data = np.array(data)
    design_matrix = np.array(design_matrix)
    XTXinv = np.linalg.pinv(np.einsum('afm,afn->mn', design_matrix, design_matrix))
    Xy = np.einsum('afm,af->m', design_matrix, data)
    model = design_matrix @ (XTXinv @ Xy)
    model = model.reshape(len(radial_reds), 4)
    model_proj = nucal.project_u_model_comps_on_spec_axis(model, freqs)
    data_proj = np.einsum('af,af->f', data, design_matrix)
    np.allclose(model_proj, data_proj, atol=1e-6)
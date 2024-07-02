import pytest
import numpy as np
from copy import deepcopy
from hera_filters import dspec
from hera_sim.antpos import linear_array, hex_array

from .. import redcal, nucal, utils, abscal, apply_cal
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
    def setup_method(self):
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

    # Show that filters with a u-max cutoff are set to zero
    umax = 15
    antpos = linear_array(6, sep=5)
    radial_reds = nucal.RadialRedundancy(antpos)
    spatial_filters = nucal.compute_spatial_filters(radial_reds, freqs, umax=umax)

    for rdgrp in radial_reds:
        for bl in rdgrp:
            umodes = radial_reds.baseline_lengths[bl] * freqs / 2.998e8
            assert np.allclose(spatial_filters[bl][umodes > umax], 0)


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
    freqs = np.linspace(50e6, 250e6, 200)
    spatial_filters = nucal.compute_spatial_filters(radial_reds, freqs)
    spectral_filters = dspec.dpss_operator(freqs, [0], [1e-9], eigenval_cutoff=[1e-9])[0].real
    data = {}
    wgts = {}
    for rdgrp in radial_reds:
        for bl in rdgrp:
            blmag = np.linalg.norm(antpos[bl[1]] - antpos[bl[0]])
            data[bl] = np.sin(freqs * blmag / 2.998e8)[None, :]
            wgts[bl] = np.ones_like(data[bl])

    # Get model components for a u-dependent model
    model_comps = nucal.fit_nucal_foreground_model(data, wgts, radial_reds, spatial_filters, solver='solve', return_model_comps=True)

    # Project the model components on the spectral axis
    model_proj = nucal.project_u_model_comps_on_spec_axis(model_comps, spectral_filters)

    # Get model for both cases
    model1 = nucal.evaluate_foreground_model(radial_reds, model_comps, spatial_filters)
    model2 = nucal.evaluate_foreground_model(radial_reds, model_proj, spatial_filters, spectral_filters)

    # Check that the two models are relatively close
    for bl in data:
        assert np.sqrt(np.square(model1[bl] - model2[bl]).mean()) < 1e-4


def test_linear_fit():
    # Create a set of mock data to fit
    freqs = np.linspace(50e6, 250e6, 200)
    y = np.sin(freqs * 100e-9)

    # Create a design matrix
    X = dspec.dpss_operator(freqs, [0], [100e-9], eigenval_cutoff=[1e-13])[0].real

    # Compute XTX and Xy
    XTX = np.dot(X.T, X)
    Xy = np.dot(X.T, y)

    # Test different modes
    b1, cached_input = nucal._linear_fit(XTX, Xy, solver='lu_solve')
    assert cached_input.get('LU') is not None
    b1_cached, _ = nucal._linear_fit(XTX, Xy, solver='lu_solve', cached_input=cached_input)
    # Show that the cached result is the same as the original
    np.testing.assert_allclose(b1, b1_cached)
    b2, _ = nucal._linear_fit(XTX, Xy, solver='solve')
    b3, _ = nucal._linear_fit(XTX, Xy, solver='lstsq')
    b4, cached_input = nucal._linear_fit(XTX, Xy, solver='pinv')
    assert cached_input.get('XTXinv') is not None

    # Show that all modes give the same result
    np.testing.assert_allclose(b1, b2, atol=1e-6)
    np.testing.assert_allclose(b1, b3, atol=1e-6)
    np.testing.assert_allclose(b1, b4, atol=1e-6)

    # Test that the fit is correct
    model = np.dot(X, b4)
    np.testing.assert_allclose(model, y, atol=1e-6)

    # Test that an error is raised if the solver is not defined
    with pytest.raises(AssertionError):
        b = nucal._linear_fit(XTX, Xy, solver='undefined_solver')

    # Test that an error is raised if the tolerance is negative
    with pytest.raises(AssertionError):
        b = nucal._linear_fit(XTX, Xy, alpha=-1)


def test_compute_spectral_filters():
    # Create a set of mock data to fit
    freqs = np.linspace(50e6, 250e6, 200)
    y = np.sin(freqs * 100e-9)

    # Create a design matrix
    spectral_filters = nucal.compute_spectral_filters(freqs, spectral_filter_half_width=100e-9, eigenval_cutoff=1e-13)

    # Compute XTX and Xy
    XTX = np.dot(spectral_filters.T, spectral_filters)
    Xy = np.dot(spectral_filters.T, y)
    model = spectral_filters @ nucal._linear_fit(XTX, Xy, solver='solve')[0]

    # Test that the spectral filters are correct
    np.testing.assert_allclose(y, model, atol=1e-6)


def test_evaluate_foreground_model():
    antpos = linear_array(6, sep=5)
    radial_reds = nucal.RadialRedundancy(antpos)
    freqs = np.linspace(50e6, 250e6, 200)
    spatial_filters = nucal.compute_spatial_filters(radial_reds, freqs)
    spectral_filters = dspec.dpss_operator(freqs, [0], [1e-9], eigenval_cutoff=[1e-9])[0].real
    data = {}
    data_wgts = {}

    # Generate mock data
    for rdgrp in radial_reds:
        for bl in rdgrp:
            blmag = np.linalg.norm(antpos[bl[1]] - antpos[bl[0]])
            data[bl] = np.sin(freqs * blmag / 2.998e8)[None, :]
            data_wgts[bl] = np.ones_like(data[bl])

    # Compute the model
    model_comps = nucal.fit_nucal_foreground_model(data, data_wgts, radial_reds, spatial_filters, return_model_comps=True)

    # Evaluate the model
    model = nucal.evaluate_foreground_model(radial_reds, model_comps, spatial_filters)

    # Check that the model has the same shape as the data
    for k in model:
        assert model[k].shape == data[k].shape

    # Check that function raises AssertionError if spectral filters are provided
    # but the model components are not projected on the spectral axis
    with pytest.raises(AssertionError):
        model = nucal.evaluate_foreground_model(radial_reds, model_comps, spatial_filters, spectral_filters=spectral_filters)

    # Check that function raises AssertionError if spatial filters are the wrong shape
    _spatial_filters = {k: spatial_filters[k][:, :-3] for k in spatial_filters}
    with pytest.raises(AssertionError):
        model = nucal.evaluate_foreground_model(radial_reds, model_comps, _spatial_filters)


def test_fit_nucal_foreground_model():
    antpos = linear_array(6, sep=5)
    radial_reds = nucal.RadialRedundancy(antpos)
    freqs = np.linspace(50e6, 250e6, 200)
    spatial_filters = nucal.compute_spatial_filters(radial_reds, freqs)
    spectral_filters = dspec.dpss_operator(freqs, [0], [5e-9], eigenval_cutoff=[1e-9])[0].real
    data = {}
    data_wgts = {}

    # Generate mock data
    for rdgrp in radial_reds:
        for bl in rdgrp:
            blmag = np.linalg.norm(antpos[bl[1]] - antpos[bl[0]])
            data[bl] = np.sin(freqs * blmag / 2.998e8)[None, :]
            data_wgts[bl] = np.ones_like(data[bl])

    # Compute the model
    model = nucal.fit_nucal_foreground_model(data, data_wgts, radial_reds, spatial_filters, spectral_filters, return_model_comps=False)

    # Compute the model - no spectral filters
    u_model = nucal.fit_nucal_foreground_model(data, data_wgts, radial_reds, spatial_filters, return_model_comps=False)

    # Check that the model has the same shape as the data and is close to the data
    for k in model:
        assert model[k].shape == data[k].shape
        assert u_model[k].shape == data[k].shape
        assert np.sqrt(np.square(data[k] - model[k]).mean()) < 1e-5
        assert np.sqrt(np.square(data[k] - u_model[k]).mean()) < 1e-5

    # Generate mock data for multiple times
    ntimes = 5
    for rdgrp in radial_reds:
        for bl in rdgrp:
            blmag = np.linalg.norm(antpos[bl[1]] - antpos[bl[0]])
            data[bl] = np.sin(freqs * blmag / 2.998e8) * np.ones((ntimes, 1))
            data_wgts[bl] = np.ones_like(data[bl])

    # Compute a foreground model shared across time
    model = nucal.fit_nucal_foreground_model(data, data_wgts, radial_reds, spatial_filters, spectral_filters, return_model_comps=False, share_fg_model=True)
    u_model = nucal.fit_nucal_foreground_model(data, data_wgts, radial_reds, spatial_filters, return_model_comps=False, share_fg_model=True)

    # Check that the model has the same shape as the data and is close to the data
    for k in model:
        assert model[k].shape == (1, 200)
        assert u_model[k].shape == (1, 200)
        assert np.sqrt(np.square(data[k] - model[k]).mean()) < 1e-5
        assert np.sqrt(np.square(data[k] - u_model[k]).mean()) < 1e-5

    # Compute a model not shared across time
    model = nucal.fit_nucal_foreground_model(data, data_wgts, radial_reds, spatial_filters, spectral_filters, return_model_comps=False, share_fg_model=False)
    u_model = nucal.fit_nucal_foreground_model(data, data_wgts, radial_reds, spatial_filters, return_model_comps=False, share_fg_model=False)

    # Check that the model has the same shape as the data and is close to the data
    for k in model:
        assert model[k].shape == data[k].shape
        assert u_model[k].shape == data[k].shape
        assert np.sqrt(np.square(data[k] - model[k]).mean()) < 1e-5
        assert np.sqrt(np.square(data[k] - u_model[k]).mean()) < 1e-5

    # Return model components
    model_comps = nucal.fit_nucal_foreground_model(data, data_wgts, radial_reds, spatial_filters, spectral_filters, return_model_comps=True, share_fg_model=False)
    u_model_comps = nucal.fit_nucal_foreground_model(data, data_wgts, radial_reds, spatial_filters, return_model_comps=True, share_fg_model=False)

    for k in model_comps:
        assert model_comps[k].shape[0] == ntimes
        assert u_model_comps[k].shape[0] == ntimes
        assert model_comps[k].shape[1:] == (spectral_filters.shape[-1], spatial_filters[k].shape[-1])
        assert u_model_comps[k].shape[1:] == (spatial_filters[k].shape[-1],)


class TestGradientDescent:
    def setup_method(self):
        self.freqs = np.linspace(50e6, 250e6, 400)
        self.antpos = linear_array(10, sep=2)

        ns_antpos = linear_array(10, sep=2)
        for ant in ns_antpos:
            if ant > 0:
                self.antpos[ant + 10] = np.array([ns_antpos[ant][1], ns_antpos[ant][0], 0])

        self.radial_reds = nucal.RadialRedundancy(self.antpos)
        self.radial_reds.filter_radial_groups(min_nbls=8)

        # Create a u-dependent model
        self.data = {
            bl: np.sin(2 * np.pi * self.radial_reds.baseline_lengths[bl] * self.freqs / 2.998e8 * 0.25)[None] * np.ones((2, 1))
            for rdgrp in self.radial_reds for bl in rdgrp
        }
        self.data = DataContainer(self.data)
        self.data_wgts = DataContainer({k: np.ones(self.data[k].shape) for k in self.data})
        self.data.freqs = self.freqs
        self.frc = nucal.SpectrallyRedundantCalibrator(self.radial_reds)

        # Compute the filters
        self.frc._compute_filters(self.freqs, 10e-9)

        self.init_model_comps = nucal.fit_nucal_foreground_model(
            self.data, self.data_wgts, self.radial_reds, self.frc.spatial_filters, share_fg_model=True,
            return_model_comps=True
        )
        self.init_model_comps = nucal.project_u_model_comps_on_spec_axis(self.init_model_comps, self.frc.spectral_filters)

        # Calculate baseline vectors
        self.blvecs = np.array([(self.antpos[bl[1]] - self.antpos[bl[0]])[:2] for rdgrp in self.radial_reds for bl in rdgrp])  # [:, None]

        self.model_parameters = {
            "tip_tilt": np.zeros((2, 2, 400)),
            "amplitude": np.ones((2, 400)),
            "fg_r": [self.init_model_comps[rdgrp[0]].real for rdgrp in self.radial_reds],
            "fg_i": [self.init_model_comps[rdgrp[0]].imag for rdgrp in self.radial_reds],
        }

        self.spatial_filters = [np.array([self.frc.spatial_filters[blkey] for blkey in rdgrp]) for rdgrp in self.radial_reds]

    def test_foreground_model(self):
        params = {
            "fg_r": [np.random.normal(0, 1, (self.data.shape[0], self.frc.spectral_filters.shape[-1], f.shape[-1])) for f in self.spatial_filters],
            "fg_i": [np.random.normal(0, 1, (self.data.shape[0], self.frc.spectral_filters.shape[-1], f.shape[-1])) for f in self.spatial_filters],
        }

        model_r, model_i = nucal._foreground_model(params, self.frc.spectral_filters, self.spatial_filters)

        # Check that the model has the correct shape
        assert model_r.shape == (len(self.data), self.data.shape[0], self.data.shape[1])
        assert model_i.shape == (len(self.data), self.data.shape[0], self.data.shape[1])

    def test_mean_squared_error(self):

        # Create a mock model that's the same as the data
        model = deepcopy(self.data)

        # Separate the real and imaginary components
        model_r = np.array([model[bl].real for rdgrp in self.radial_reds for bl in rdgrp])
        model_i = np.array([model[bl].imag for rdgrp in self.radial_reds for bl in rdgrp])

        # Separate the real and imaginary components
        data_r = np.array([self.data[bl].real for rdgrp in self.radial_reds for bl in rdgrp])
        data_i = np.array([self.data[bl].imag for rdgrp in self.radial_reds for bl in rdgrp])
        wgts = np.ones_like(data_r)

        # Compute the mean squared error
        mse = nucal._mean_squared_error(self.model_parameters, data_r, data_i, wgts, model_r, model_i, self.blvecs)

        # Check that the mse is zero
        assert np.isclose(mse, 0)

    def test_calibration_loss_function(self):
        # Separate the real and imaginary components
        data_r = np.array([self.data[bl].real for rdgrp in self.radial_reds for bl in rdgrp])
        data_i = np.array([self.data[bl].imag for rdgrp in self.radial_reds for bl in rdgrp])
        wgts = np.ones_like(data_r)

        mse = nucal._calibration_loss_function(self.model_parameters, data_r, data_i, wgts, self.frc.spectral_filters, self.spatial_filters, self.blvecs)

        # Check that the mse is zero
        assert np.isclose(mse, 0)

    def test_nucal_post_redcal(self):
        amp = np.random.normal(1, 0.01, size=self.data[list(self.data.keys())[0]].shape)

        # Separate the real and imaginary components
        data_r = np.array([amp * self.data[bl].real for rdgrp in self.radial_reds for bl in rdgrp])
        data_i = np.array([amp * self.data[bl].imag for rdgrp in self.radial_reds for bl in rdgrp])
        wgts = np.ones_like(data_r)

        # Make optimizer
        optimizer = nucal.OPTIMIZERS['novograd'](learning_rate=1e-3)

        # Run gradient descent
        _, metadata = nucal._nucal_post_redcal(
            data_r, data_i, wgts, deepcopy(self.model_parameters), spectral_filters=self.frc.spectral_filters,
            spatial_filters=self.spatial_filters, idealized_blvecs=self.blvecs, optimizer=optimizer, major_cycle_maxiter=10,
        )

        # Check that the gradient descent decreases the value of the loss function
        assert metadata['loss_history'][0] > metadata['loss_history'][-1]

        # Run gradient descent w/ minor cycle
        _, _metadata = nucal._nucal_post_redcal(
            data_r, data_i, wgts, deepcopy(self.model_parameters), spectral_filters=self.frc.spectral_filters,
            spatial_filters=self.spatial_filters, idealized_blvecs=self.blvecs, optimizer=optimizer, major_cycle_maxiter=10,
            minor_cycle_maxiter=3
        )

        # Check that the gradient descent decreases the value of the loss function
        assert _metadata['loss_history'][0] > _metadata['loss_history'][-1]

        # Check that final loss with minor cycle is less than without
        assert metadata['loss_history'][-1] > _metadata['loss_history'][-1]


class TestSpectrallyRedundantCalibrator:
    def setup_method(self):
        self.freqs = np.linspace(50e6, 250e6, 400)
        self.antpos = linear_array(10, sep=2)
        self.radial_reds = nucal.RadialRedundancy(self.antpos)

        # Create a u-dependent model
        self.data = {
            bl: 1 + 0.2 * np.sin(2 * np.pi * self.radial_reds.baseline_lengths[bl] * self.freqs / 2.998e8 * 0.25)[None] * np.ones((2, 1), dtype="complex128")
            for rdgrp in self.radial_reds for bl in rdgrp
        }
        self.data = DataContainer(self.data)
        self.data_wgts = DataContainer({k: np.ones(self.data[k].shape) for k in self.data})
        self.data.freqs = self.freqs
        self.frc = nucal.SpectrallyRedundantCalibrator(self.radial_reds)

    def test_compute_filters(self):
        assert not self.frc._filters_computed

        # Compute filters
        self.frc._compute_filters(self.freqs, 20e-9)
        assert self.frc._filters_computed

        # Trigger check that filters are already computed
        sf = deepcopy(self.frc.spectral_filters)
        self.frc._compute_filters(self.freqs, 20e-9)
        assert np.allclose(self.frc.spectral_filters, sf)

        # Recompute with different values
        self.frc._compute_filters(self.freqs, 30e-9)
        assert sf.shape != self.frc.spectral_filters.shape

        # Check that filters are the correct shape
        assert len(self.frc.spatial_filters) == len(self.radial_reds[0])

    def test_estimate_degeneracies(self):
        # Copy data
        data = deepcopy(self.data)

        # Compute filters
        self.frc._compute_filters(self.freqs, 20e-9)

        # Fit model
        model = nucal.fit_nucal_foreground_model(
            data, self.data_wgts, self.radial_reds, self.frc.spatial_filters,
            return_model_comps=False, share_fg_model=True
        )

        # Estimate degeneracies
        amplitude, tip_tilt = self.frc._estimate_degeneracies(data, model, self.data_wgts)

        # Since the data are already "perfectly calibrated" and the model should match the data to high precision,
        # the estimated tip-tilt should be zero and the amplitude should be 1
        assert np.isclose(
            np.sqrt(np.mean(np.square(amplitude['nn'] - 1))), 0, atol=1e-5
        )
        assert np.isclose(
            np.sqrt(np.mean(np.square(tip_tilt['nn']))), 0, atol=1e-5
        )

        # Shape of amplitude and tip-tilt should be (ndims, ntimes, nfreqs)
        assert tip_tilt["nn"].shape == (1, 2, 400)

    def test_post_redcal_nucal(self):
        """
        """
        # Set random seed
        np.random.seed(42)

        # Set gains
        amp = np.random.normal(1, 0.01, size=(2, 400))
        gains = {(k, "Jnn"): amp for k in self.antpos}
        dc = deepcopy(self.data)
        apply_cal.calibrate_in_place(dc, gains, gain_convention='multiply')

        fit_gains, _model_params, _meta, _ = self.frc.post_redcal_nucal(
            dc, self.data_wgts, spatial_estimate_only=True, minor_cycle_maxiter=3,
            share_fg_model=True, major_cycle_maxiter=250, spectral_filter_half_width=5e-9,
            estimate_degeneracies=True, return_model=True
        )

        # Apply gains to data
        apply_cal.calibrate_in_place(dc, fit_gains, gain_convention='divide')

        # Recompute model with the new gains
        model = nucal.fit_nucal_foreground_model(
            dc, self.data_wgts, self.radial_reds, self.frc.spatial_filters, spectral_filters=self.frc.spectral_filters, alpha=1e-12,
            share_fg_model=True
        )

        # Check that the model is close to the data within a certain tolerance
        assert np.sqrt(np.mean([np.square(model[k] - dc[k]) for k in model])) < 1e-5

        dc = deepcopy(self.data)
        apply_cal.calibrate_in_place(dc, gains, gain_convention='multiply')

        # Calibrate without estimating degeneracies
        fit_gains, _model_params, _meta = self.frc.post_redcal_nucal(
            dc, self.data_wgts, spatial_estimate_only=True, minor_cycle_maxiter=3,
            share_fg_model=True, major_cycle_maxiter=250, spectral_filter_half_width=5e-9,
            estimate_degeneracies=False
        )

        # Apply gains to data
        apply_cal.calibrate_in_place(dc, fit_gains, gain_convention='divide')

        # Recompute model with the new gains
        model = nucal.fit_nucal_foreground_model(
            dc, self.data_wgts, self.radial_reds, self.frc.spatial_filters, spectral_filters=self.frc.spectral_filters, alpha=1e-12,
            share_fg_model=True
        )

        # Check that the model is close to the data within a certain tolerance
        assert np.sqrt(np.mean([np.square(model[k] - dc[k]) for k in model])) < 1e-5

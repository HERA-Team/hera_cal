"""
Tests for hera_cal/polfilt.py

Covers:
    - radec_to_lmn
    - estimate_polarized_source_delay
    - estimate_freq_from_polarized_source_delay
    - unpack_data_containers
    - _fit_rotation_measure
    - _fit_polarized_source_position
    - iteratively_fit_polarized_source_params
"""

import numpy as np
import pytest
from astropy.coordinates import EarthLocation
import astropy.units as u
from astropy.time import Time

import hera_cal.polfilt as pf
from hera_cal.datacontainer import DataContainer


# ---------------------------------------------------------------------------
# Shared constants and helpers
# ---------------------------------------------------------------------------

HERA_LOCATION = EarthLocation(
    lat=-30.721527 * u.deg,
    lon=21.428305 * u.deg,
    height=1073.0 * u.m,
)
C = pf.SPEED_OF_LIGHT


def _make_dcs(vis, freqs, times, antpos, all_flagged=False):
    """
    Build data, flags, and nsamples DataContainers from ``vis``, a dict mapping
    ``(ant1, ant2, pol)`` to a (n_times, n_freqs) array. DataContainer returns the
    conjugate for reversed baselines, as ``unpack_data_containers`` expects.
    """
    data = DataContainer(vis)
    flags = DataContainer({k: np.full(v.shape, all_flagged) for k, v in vis.items()})
    nsamples = DataContainer({k: np.ones(v.shape) for k, v in vis.items()})
    for dc in (data, flags, nsamples):
        dc.freqs, dc.times, dc.antpos = freqs, times, antpos
    return data, flags, nsamples


def _point_source_vis(uvw, ra, dec, rm, times, freqs, real_sky=False):
    """Noiseless point source at (ra, dec) with rotation measure ``rm``:
    V = exp(2πi · uvw·lmn(t)) · exp(−2i·λ²·rm), for uvw of shape (..., 3, n_freqs).

    With ``real_sky=True``, the source spectrum is instead a real Stokes Q, cos(2·λ²·rm),
    which is what makes V(−b) = conj(V(b)) consistent with the same model, as
    ``unpack_data_containers`` assumes when it adds each baseline's conjugate."""
    lmn = pf.radec_to_lmn(ra, dec, times, HERA_LOCATION)
    phase = np.einsum("...cf,ct->...tf", uvw, lmn)
    lambda_sq = (C / freqs) ** 2
    spectrum = np.cos(2 * lambda_sq * rm) if real_sky else np.exp(-2j * lambda_sq * rm)
    return np.exp(2j * np.pi * phase) * spectrum


# ---------------------------------------------------------------------------
# radec_to_lmn
# ---------------------------------------------------------------------------

class TestRadecToLmn:

    @pytest.mark.parametrize("times", [
        np.array([2459000.0]),
        np.linspace(2459000.0, 2459000.01, 7),
        Time([2459000.0, 2459000.1], format="jd"),
    ])
    def test_output_shape(self, times):
        assert pf.radec_to_lmn(45.0, -30.0, times, HERA_LOCATION).shape == (3, len(times))

    def test_unit_vector(self):
        """l² + m² + n² must equal 1 at every time step."""
        times = np.linspace(2459000.0, 2459000.01, 10)
        lmn = pf.radec_to_lmn(10.0, -20.0, times, HERA_LOCATION)
        np.testing.assert_allclose(np.sum(lmn**2, axis=0), 1.0, atol=1e-12)

    def test_below_horizon_n_negative(self):
        """The North Pole is always below the horizon at HERA."""
        times = np.linspace(2459000.0, 2459000.01, 5)
        assert np.all(pf.radec_to_lmn(0.0, 90.0, times, HERA_LOCATION)[2] < 0)


# ---------------------------------------------------------------------------
# estimate_polarized_source_delay and estimate_freq_from_polarized_source_delay
# ---------------------------------------------------------------------------

class TestPolarizedSourceDelay:

    def test_scalar_value(self):
        freq, rm = 150e6, 10.0
        expected = 2.0 * (C**2 / freq**3) * rm / np.pi
        np.testing.assert_allclose(pf.estimate_polarized_source_delay(freq, rm), expected, rtol=1e-12)

    def test_monotonicity_and_special_cases(self):
        """Delay increases with RM, decreases with freq, and is zero/negative for zero/negative RM."""
        delay = pf.estimate_polarized_source_delay
        assert delay(150e6, 100.0) > delay(150e6, 1.0)
        assert delay(100e6, 10.0) > delay(200e6, 10.0)
        assert delay(150e6, 0.0) == 0.0
        assert delay(150e6, -10.0) < 0.0

    def test_array_broadcast(self):
        tau_f = pf.estimate_polarized_source_delay(np.array([100e6, 150e6, 200e6]), 5.0)
        assert tau_f.shape == (3,) and np.all(tau_f > 0)

        tau_rm = pf.estimate_polarized_source_delay(150e6, np.array([1.0, 10.0, 100.0]))
        assert tau_rm.shape == (3,)
        np.testing.assert_allclose(tau_rm[1] / tau_rm[0], 10.0, rtol=1e-10)

    @pytest.mark.parametrize("freqs,rm", [
        (150e6, 10.0),
        (np.linspace(100e6, 200e6, 20), 25.0),
    ])
    def test_freq_from_delay_round_trip(self, freqs, rm):
        tau = pf.estimate_polarized_source_delay(freqs, rm)
        np.testing.assert_allclose(pf.estimate_freq_from_polarized_source_delay(tau, rm), freqs, rtol=1e-10)

    def test_freq_from_delay_higher_delay_lower_freq(self):
        rm = 10.0
        assert (pf.estimate_freq_from_polarized_source_delay(1e-8, rm)
                > pf.estimate_freq_from_polarized_source_delay(1e-7, rm))
        assert pf.estimate_freq_from_polarized_source_delay(np.ones(5) * 1e-8, rm).shape == (5,)


# ---------------------------------------------------------------------------
# unpack_data_containers
# ---------------------------------------------------------------------------

class TestUnpackDataContainers:

    ANTPOS = {0: np.array([0.0, 0.0, 0.0]), 1: np.array([14.6, 0.0, 0.0]), 2: np.array([0.0, 14.6, 0.0])}
    FREQS = np.linspace(100e6, 200e6, 16)
    TIMES = np.linspace(2459000.0, 2459000.1, 8)

    def _unpack(self, antpairs=((0, 1), (0, 2), (1, 2)), all_flagged=False, **kwargs):
        rng = np.random.default_rng(42)
        shape = (len(self.TIMES), len(self.FREQS))
        vis = {ap + ("ee",): rng.standard_normal(shape) + 1j * rng.standard_normal(shape) for ap in antpairs}
        dcs = _make_dcs(vis, self.FREQS, self.TIMES, self.ANTPOS, all_flagged=all_flagged)
        return pf.unpack_data_containers(*dcs, antpos=self.ANTPOS, freqs=self.FREQS, **kwargs)

    def test_output_shapes_full(self):
        vis, weights, uvw, t_out, f_out = self._unpack()
        assert vis.shape == weights.shape == (6, 8, 16)  # each baseline and its conjugate
        assert uvw.shape == (6, 3, 16)
        np.testing.assert_array_equal(t_out, self.TIMES)
        np.testing.assert_array_equal(f_out, self.FREQS)

    def test_slices_applied(self):
        vis, _, _, t_out, f_out = self._unpack(time_slice=slice(2, 6), freq_slice=slice(4, 12))
        assert vis.shape == (6, 4, 8)
        np.testing.assert_array_equal(t_out, self.TIMES[2:6])
        np.testing.assert_array_equal(f_out, self.FREQS[4:12])

    def test_weights(self):
        """Flagged samples get zero weight, and weights are binary if not weighting by nsamples."""
        _, weights, *_ = self._unpack(antpairs=[(0, 1)], all_flagged=True)
        assert np.all(weights == 0.0)
        _, weights, *_ = self._unpack(antpairs=[(0, 1)], weight_by_nsamples=False)
        assert set(np.unique(weights)) <= {0.0, 1.0}

    def test_uvw_conjugate_pair_negated(self):
        vis, _, uvw, *_ = self._unpack(antpairs=[(0, 1)])
        np.testing.assert_allclose(uvw[0], -uvw[1])
        np.testing.assert_allclose(vis[0], vis[1].conj())


# ---------------------------------------------------------------------------
# _fit_rotation_measure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rm_true", [15.0, -20.0])
def test_fit_rotation_measure_recovers_known_rm(rm_true):
    """Nearly-noiseless RM signal with zero UVW (so the position phase is zero)."""
    rng = np.random.default_rng(0)
    freqs = np.linspace(120e6, 180e6, 32)
    times = np.linspace(2459000.0, 2459000.01, 6)
    uvw = np.zeros((4, 3, len(freqs)))
    vis = _point_source_vis(uvw, 0.0, 0.0, rm_true, times, freqs)
    vis = vis + 0.01 * (rng.standard_normal(vis.shape) + 1j * rng.standard_normal(vis.shape))

    rm_fit = pf._fit_rotation_measure(
        vis, np.ones(vis.shape), uvw, times, freqs, ra=0.0, dec=0.0, start_rm=rm_true,
        location=HERA_LOCATION, drm=20.0, dtest=1000,
    )
    assert isinstance(rm_fit, float)
    assert abs(rm_fit - rm_true) < 1.0, f"Expected RM ~ {rm_true}, got {rm_fit:.3f}"


# ---------------------------------------------------------------------------
# _fit_polarized_source_position
# ---------------------------------------------------------------------------

class TestFitPolarizedSourcePosition:
    """
    Uses noiseless visibilities for a point source (see ``_point_source_vis``). At zero
    offset, the phased visibilities are a uniform constant A, forcing x=[A, 0, 0] in the
    normal equations, so (ra_fit, dec_fit) = (ra0, dec0) to floating-point precision.
    """

    FREQS = np.linspace(120e6, 180e6, 32)
    RA0, DEC0, RM0 = 45.0, -30.0, 10.0

    def _make_vis(self, ra, dec, rm, n_times=1, n_bls=6):
        rng = np.random.default_rng(7)
        blvecs = rng.standard_normal((n_bls, 3)) * 100.0
        blvecs[:, 2] = 0.0
        uvw = blvecs[:, :, None] * self.FREQS[None, None, :] / C
        times = np.linspace(2459000.0, 2459000.0 + 5e-4 * n_times, n_times)
        vis = _point_source_vis(uvw, ra, dec, rm, times, self.FREQS)
        return vis, np.ones(vis.shape), uvw, times

    def _fit(self, vis, weights, uvw, times, ra=RA0, dec=DEC0, rm=RM0):
        return pf._fit_polarized_source_position(vis, weights, uvw, ra, dec, rm, times, self.FREQS, HERA_LOCATION)

    @pytest.mark.parametrize("n_times", [1, 6])
    @pytest.mark.parametrize("rm", [10.0, 0.0, -20.0])
    def test_exact_position_zero_offset(self, rm, n_times):
        ra_fit, dec_fit = self._fit(*self._make_vis(self.RA0, self.DEC0, rm, n_times=n_times), rm=rm)
        assert isinstance(ra_fit, float) and isinstance(dec_fit, float)
        np.testing.assert_allclose(ra_fit, self.RA0, atol=1e-8)
        np.testing.assert_allclose(dec_fit, self.DEC0, atol=1e-8)

    @pytest.mark.parametrize("dra, ddec", [(0.01, 0.0), (0.0, 0.01), (0.008, 0.006)])
    def test_small_offset_recovered(self, dra, ddec):
        ra_true, dec_true = self.RA0 + dra, self.DEC0 + ddec
        vis, w, uvw, t = self._make_vis(ra_true, dec_true, self.RM0)

        # Iterate the fitter a few times so the linear approximation improves as we get closer.
        ra_fit, dec_fit = self.RA0, self.DEC0
        for _ in range(5):
            ra_fit, dec_fit = self._fit(vis, w, uvw, t, ra=ra_fit, dec=dec_fit)
        np.testing.assert_allclose(ra_fit, ra_true, atol=1e-3)
        np.testing.assert_allclose(dec_fit, dec_true, atol=1e-3)

    def test_weights(self):
        """Uniformly rescaling the weights doesn't matter, and zero-weight baselines are ignored."""
        vis, w, uvw, t = self._make_vis(self.RA0, self.DEC0, self.RM0, n_bls=8)
        ref = self._fit(vis, w, uvw, t)
        np.testing.assert_allclose(self._fit(vis, w * 100.0, uvw, t), ref, atol=1e-10)

        w[:2] = 0.0
        np.testing.assert_allclose(self._fit(vis, w, uvw, t), ref, atol=1e-6)

    def test_correct_rm_beats_wrong_rm(self):
        """Faraday de-rotation matters: fitting with the true RM gets closer to the true RA."""
        ra_true, rm_true = self.RA0 + 0.01, 50.0
        vis, w, uvw, t = self._make_vis(ra_true, self.DEC0, rm=rm_true)
        ra_correct, _ = self._fit(vis, w, uvw, t, rm=rm_true)
        ra_wrong, _ = self._fit(vis, w, uvw, t, rm=0.0)
        assert abs(ra_correct - ra_true) < abs(ra_wrong - ra_true)


# ---------------------------------------------------------------------------
# iteratively_fit_polarized_source_params
# ---------------------------------------------------------------------------

class TestIterativelyFitPolarizedSourceParams:
    """
    End-to-end tests on noiseless visibilities of a point source with a real Stokes Q
    spectrum (see ``_point_source_vis``), so the true position is an exact fixed point of
    the fitter. The array is 45 baselines from 10 randomly placed (but seeded)
    antennas, and the source is at Dec ≈ HERA's latitude, so it transits nearly overhead
    and stays above the horizon throughout the 10-minute observation.
    """

    RA_TRUE, DEC_TRUE, RM_TRUE = 45.0, -30.0, 45.0  # degrees, degrees, rad/m²
    FREQS = np.linspace(180e6, 200e6, 128)
    TIMES = np.linspace(2459000.0, 2459000.0 + 10 / 1440.0, 12)  # 12 integrations over 10 minutes
    ANTPOS = {i: np.append(pos, 0.0) for i, pos in enumerate(np.random.default_rng(1).uniform(0, 300, (10, 2)))}
    ANTPAIRS = [(i, j) for i in range(10) for j in range(i + 1, 10)]
    # Fit tolerances. A real Q spectrum also has a mirror component at -RM, which pulls the
    # fit RM by ~0.08 rad/m² for this band.
    ATOL = {"ra": 1e-4, "dec": 1e-4, "rm": 0.1}  # degrees, degrees, rad/m²

    def _source_dcs(self, pol="pQ", all_flagged=False):
        vis = {}
        for ap in self.ANTPAIRS:
            uvw = (self.ANTPOS[ap[1]] - self.ANTPOS[ap[0]])[:, None] * self.FREQS[None, :] / C
            vis[ap + (pol,)] = _point_source_vis(uvw, self.RA_TRUE, self.DEC_TRUE, self.RM_TRUE, self.TIMES, self.FREQS,
                                                 real_sky=True)
        return _make_dcs(vis, self.FREQS, self.TIMES, self.ANTPOS, all_flagged=all_flagged)

    def _fit(self, ra=RA_TRUE, dec=DEC_TRUE, rm=RM_TRUE, all_flagged=False, **kwargs):
        return pf.iteratively_fit_polarized_source_params(
            *self._source_dcs(all_flagged=all_flagged), right_ascension=ra, declination=dec,
            rotation_measure=rm, location=HERA_LOCATION, **kwargs,
        )

    @pytest.mark.parametrize("dra, ddec, drm", [(0.0, 0.0, 0.0), (0.01, 0.01, 1.0)])
    def test_recovers_truth(self, dra, ddec, drm):
        """Starting at (or with small offsets from) the truth, the fitter converges to the truth."""
        fit = self._fit(ra=self.RA_TRUE + dra, dec=self.DEC_TRUE + ddec, rm=self.RM_TRUE + drm,
                        drm=3.0, dtest=500, maxiter=20, verbose=True)
        for param, value, truth in zip(("ra", "dec", "rm"), fit, (self.RA_TRUE, self.DEC_TRUE, self.RM_TRUE)):
            np.testing.assert_allclose(value, truth, atol=self.ATOL[param], err_msg=f"{param} did not converge to truth")

    @pytest.mark.parametrize("param, offset", [("ra", 0.02), ("dec", 0.02), ("rm", 3.0)])
    def test_corrects_single_offset(self, param, offset):
        """An incorrect initial RA, Dec, or RM is corrected."""
        truth = {"ra": self.RA_TRUE, "dec": self.DEC_TRUE, "rm": self.RM_TRUE}
        start = {**truth, param: truth[param] + offset}
        drm = offset + 1.0 if param == "rm" else 2.0  # the RM search window must contain the truth
        fit = dict(zip(("ra", "dec", "rm"), self._fit(**start, drm=drm, dtest=500)))
        np.testing.assert_allclose(fit[param], truth[param], atol=self.ATOL[param])

    @pytest.mark.parametrize("kwargs", [dict(all_flagged=True), dict(maxiter=0)])
    def test_returns_starting_params_without_fitting(self, kwargs):
        start = (10.0, -30.0, 5.0)
        assert self._fit(*start, **kwargs) == start


# ---------------------------------------------------------------------------
# subtract_polarized_source_model
# ---------------------------------------------------------------------------

class TestSubtractPolarizedSourceModel:
    """A point source's visibilities on a real baseline, built with the module's own uvw
    convention (``unpack_data_containers``, b = antpos[ant2] - antpos[ant1]), are removed
    exactly by a model that is unity at the phase center: the subtraction is the inverse of
    the phasing the source filtering uses to build its models."""

    @pytest.fixture
    def setup(self, tmp_path):
        from hera_cal import io
        from hera_cal.data import DATA_PATH
        import os
        hd = io.HERAData(os.path.join(DATA_PATH, "zen.2458098.43124.downsample.uvh5"))
        antpair = next(ap for ap in hd.get_antpairs() if ap[0] != ap[1])
        data, flags, nsamples = hd.read(bls=[antpair])
        # a source 20 degrees east of the meridian and 15 degrees north of zenith at the file's LSTs
        ra = (np.degrees(np.mean(hd.lsts)) + 20.0) % 360
        dec = hd.telescope.location.lat.deg + 15.0
        # the model file: unity at the phase center, unflagged, tagged with the source's position
        hd.update(data={k: np.ones_like(v) for k, v in data.items()},
                  flags={k: np.zeros_like(v) for k, v in flags.items()})
        hd.extra_keywords["SOURCE_RA"], hd.extra_keywords["SOURCE_DEC"] = ra, dec
        model_file = str(tmp_path / "model.uvh5")
        hd.write_uvh5(model_file, clobber=True)
        # the data: that source on this baseline, in the module's own convention
        _, _, uvw, times, freqs = pf.unpack_data_containers(data, flags, nsamples, pol=antpair_pol(data, antpair),
                                                           antpairs=[antpair], weight_by_nsamples=False)
        source = _point_source_vis(uvw[0], ra, dec, 0.0, times, freqs)
        for k in data:
            data[k][:] = source
            flags[k][:] = False
        return data, flags, antpair, model_file, source

    def test_exact_removal(self, setup):
        data, flags, antpair, model_file, source = setup
        assert np.any(np.abs(np.angle(source)) > 0.5)  # a non-trivial fringe to remove
        # residuals are at the file's complex64 precision; a wrong sign would leave them of order 2
        pf.subtract_polarized_source_model(data, flags, model_file)
        for k in data:
            np.testing.assert_allclose(data[k], 0, atol=1e-5)

    def test_flags_and_extra_flags(self, setup):
        data, flags, antpair, model_file, source = setup
        k = next(iter(data))
        flags[k][1, 5:10] = True
        extra = np.zeros(source.shape, dtype=bool)
        extra[0, :3] = True
        pf.subtract_polarized_source_model(data, flags, model_file, extra_flags=extra)
        np.testing.assert_allclose(data[k][1, 5:10], source[1, 5:10])  # untouched where flagged
        np.testing.assert_allclose(data[k][0, :3], source[0, :3])
        np.testing.assert_allclose(data[k][2:], 0, atol=1e-5)

    def test_freq_range(self, setup):
        data, flags, antpair, model_file, source = setup
        k = next(iter(data))
        f_min = np.median(data.freqs)
        pf.subtract_polarized_source_model(data, flags, model_file, freq_range=(f_min, np.inf))
        in_range = data.freqs >= f_min
        np.testing.assert_allclose(data[k][:, in_range], 0, atol=1e-5)
        np.testing.assert_allclose(data[k][:, ~in_range], source[:, ~in_range])

    def test_baseline_selection(self, setup):
        data, flags, antpair, model_file, source = setup
        k = next(iter(data))
        other = (antpair[0] + 1000, antpair[1] + 1000, k[2])
        data[other], flags[other] = source.copy(), np.zeros(source.shape, dtype=bool)
        data.antpos[antpair[0] + 1000], data.antpos[antpair[1] + 1000] = data.antpos[antpair[0]], data.antpos[antpair[1]]
        with pytest.raises(ValueError, match="baseline must be given"):
            pf.subtract_polarized_source_model(data, flags, model_file)
        with pytest.raises(ValueError, match="not stored"):
            pf.subtract_polarized_source_model(data, flags, model_file, baseline=antpair[::-1])
        pf.subtract_polarized_source_model(data, flags, model_file, baseline=antpair)
        np.testing.assert_allclose(data[k], 0, atol=1e-5)
        np.testing.assert_allclose(data[other], source)

    def test_grid_mismatch(self, setup):
        data, flags, antpair, model_file, source = setup
        data.freqs = data.freqs + 1e3  # a kilohertz off the model's grid
        with pytest.raises(ValueError, match="grid"):
            pf.subtract_polarized_source_model(data, flags, model_file)


def antpair_pol(data, antpair):
    """The polarization of the one key in data for this antpair."""
    return next(k[2] for k in data if k[:2] == tuple(antpair))

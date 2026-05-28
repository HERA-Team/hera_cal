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
from unittest.mock import MagicMock
from astropy.coordinates import EarthLocation
import astropy.units as u
from astropy.time import Time

import hera_cal.polfilt as pf
from hera_cal import datacontainer


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

HERA_LOCATION = EarthLocation(
    lat=-30.721527 * u.deg,
    lon=21.428305 * u.deg,
    height=1073.0 * u.m,
)

FREQS = np.linspace(100e6, 200e6, 64)  # Hz
C = pf.SPEED_OF_LIGHT


# ---------------------------------------------------------------------------
# Shared mock-data helpers
# ---------------------------------------------------------------------------

def _make_mock_dc(n_times=8, n_freqs=16, antpairs=None, pol="ee", seed=42,
                  all_flagged=False):
    """
    Build lightweight mock DataContainer objects for testing
    ``unpack_data_containers``.  Pass ``all_flagged=True`` to flag every sample.
    """
    rng = np.random.default_rng(seed)
    if antpairs is None:
        antpairs = [(0, 1), (0, 2), (1, 2)]

    antpos = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([14.6, 0.0, 0.0]),
        2: np.array([0.0, 14.6, 0.0]),
    }
    freqs = np.linspace(100e6, 200e6, n_freqs)
    times = np.linspace(2459000.0, 2459000.1, n_times)

    data, flags, nsamples = {}, {}, {}
    for ap in antpairs:
        key = ap + (pol,)
        rev_key = (ap[1], ap[0], pol)
        vis = (rng.standard_normal((n_times, n_freqs))
               + 1j * rng.standard_normal((n_times, n_freqs)))
        data[key] = vis
        data[rev_key] = vis.conj()
        flags[key] = np.ones((n_times, n_freqs), dtype=bool) if all_flagged \
                         else np.zeros((n_times, n_freqs), dtype=bool)
        flags[rev_key] = flags[key].copy()
        nsamples[key] = np.ones((n_times, n_freqs))
        nsamples[rev_key] = np.ones((n_times, n_freqs))

    def _mock(d):
        dc = MagicMock(spec=datacontainer.DataContainer)
        dc.__getitem__ = lambda self, k: d[k]
        dc.antpairs = MagicMock(return_value=antpairs)
        dc.freqs = freqs
        dc.times = times
        dc.antpos = antpos
        return dc

    return _mock(data), _mock(flags), _mock(nsamples), freqs, times, antpos, antpairs


# ---------------------------------------------------------------------------
# radec_to_lmn
# ---------------------------------------------------------------------------

class TestRadecToLmn:

    @pytest.mark.parametrize("n", [1, 7])
    def test_output_shape(self, n):
        times = np.linspace(2459000.0, 2459000.01, n)
        result = pf.radec_to_lmn(0.0, 0.0, times, HERA_LOCATION)
        assert result.shape == (3, n)

    def test_unit_vector(self):
        """l² + m² + n² must equal 1 at every time step."""
        times = np.linspace(2459000.0, 2459000.01, 10)
        lmn = pf.radec_to_lmn(10.0, -20.0, times, HERA_LOCATION)
        np.testing.assert_allclose(np.sum(lmn**2, axis=0), 1.0, atol=1e-12)

    def test_below_horizon_n_negative(self):
        """The North Pole is always below the horizon at HERA."""
        times = np.linspace(2459000.0, 2459000.01, 5)
        lmn = pf.radec_to_lmn(0.0, 90.0, times, HERA_LOCATION)
        assert np.all(lmn[2] < 0)

    def test_astropy_time_input(self):
        t = Time([2459000.0, 2459000.1], format="jd")
        assert pf.radec_to_lmn(45.0, -30.0, t, HERA_LOCATION).shape == (3, 2)


# ---------------------------------------------------------------------------
# estimate_polarized_source_delay
# ---------------------------------------------------------------------------

class TestEstimatePolarizedSourceDelay:

    def test_scalar_value(self):
        freq, rm = 150e6, 10.0
        expected = 2.0 * (C**2 / freq**3) * rm / np.pi
        np.testing.assert_allclose(
            pf.estimate_polarized_source_delay(freq, rm), expected, rtol=1e-12
        )

    def test_monotonicity_and_special_cases(self):
        """Delay increases with |RM|, decreases with freq, zero/negative RM."""
        f = 150e6
        assert pf.estimate_polarized_source_delay(f, 100.0) > \
               pf.estimate_polarized_source_delay(f, 1.0), "should increase with RM"
        assert pf.estimate_polarized_source_delay(100e6, 10.0) > \
               pf.estimate_polarized_source_delay(200e6, 10.0), "should decrease with freq"
        assert pf.estimate_polarized_source_delay(f, 0.0) == 0.0
        assert pf.estimate_polarized_source_delay(f, -10.0) < 0.0

    def test_array_broadcast(self):
        """Array freq, array RM, and mixed broadcasting all produce correct shapes."""
        tau_f = pf.estimate_polarized_source_delay(np.array([100e6, 150e6, 200e6]), 5.0)
        assert tau_f.shape == (3,) and np.all(tau_f > 0)

        rms = np.array([1.0, 10.0, 100.0])
        tau_rm = pf.estimate_polarized_source_delay(150e6, rms)
        assert tau_rm.shape == (3,)
        np.testing.assert_allclose(tau_rm[1] / tau_rm[0], 10.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# estimate_freq_from_polarized_source_delay
# ---------------------------------------------------------------------------

class TestEstimateFreqFromDelay:

    @pytest.mark.parametrize("freqs,rm", [
        (150e6, 10.0),
        (np.linspace(100e6, 200e6, 20), 25.0),
    ])
    def test_round_trip(self, freqs, rm):
        tau = pf.estimate_polarized_source_delay(freqs, rm)
        np.testing.assert_allclose(
            pf.estimate_freq_from_polarized_source_delay(tau, rm), freqs, rtol=1e-10
        )

    def test_higher_delay_lower_freq_and_shape(self):
        rm = 10.0
        f_lo = pf.estimate_freq_from_polarized_source_delay(1e-7, rm)
        f_hi = pf.estimate_freq_from_polarized_source_delay(1e-8, rm)
        assert f_hi > f_lo

        out = pf.estimate_freq_from_polarized_source_delay(np.ones(5) * 1e-8, rm)
        assert out.shape == (5,)


# ---------------------------------------------------------------------------
# unpack_data_containers
# ---------------------------------------------------------------------------

class TestUnpackDataContainers:

    def test_output_shapes_full(self):
        n_times, n_freqs, n_bl = 8, 16, 3
        dc_d, dc_f, dc_n, freqs, times, antpos, _ = _make_mock_dc(n_times, n_freqs)
        vis, weights, uvw, t_out, f_out = pf.unpack_data_containers(
            dc_d, dc_f, dc_n, antpos=antpos, freqs=freqs
        )
        assert vis.shape == (2 * n_bl, n_times, n_freqs)
        assert weights.shape == (2 * n_bl, n_times, n_freqs)
        assert uvw.shape == (2 * n_bl, 3, n_freqs)
        assert t_out.shape == (n_times,)
        assert f_out.shape == (n_freqs,)

    @pytest.mark.parametrize("axis,slc,axis_idx", [
        ("freq", slice(4, 12), -1),
        ("time", slice(2, 6), 1),
    ])
    def test_slices_applied(self, axis, slc, axis_idx):
        dc_d, dc_f, dc_n, freqs, times, antpos, _ = _make_mock_dc(8, 16)
        kwargs = dict(antpos=antpos, freqs=freqs)
        kwargs["freq_slice" if axis == "freq" else "time_slice"] = slc
        vis, _, _, t_out, f_out = pf.unpack_data_containers(dc_d, dc_f, dc_n, **kwargs)
        expected = len(range(*slc.indices(16 if axis == "freq" else 8)))
        assert vis.shape[axis_idx] == expected
        if axis == "freq":
            np.testing.assert_array_equal(f_out, freqs[slc])

    def test_weights_zero_for_flagged_samples(self):
        dc_d, dc_f, dc_n, freqs, *_, antpos, _ = _make_mock_dc(
            4, 8, antpairs=[(0, 1)], all_flagged=True
        )
        _, weights, *_ = pf.unpack_data_containers(dc_d, dc_f, dc_n, antpos=antpos, freqs=freqs)
        assert np.all(weights == 0.0)

    def test_binary_weights_without_nsamples(self):
        dc_d, dc_f, dc_n, freqs, *_, antpos, _ = _make_mock_dc(4, 8, antpairs=[(0, 1)])
        _, weights, *_ = pf.unpack_data_containers(
            dc_d, dc_f, dc_n, antpos=antpos, freqs=freqs, weight_by_nsamples=False
        )
        assert set(np.unique(weights)) <= {0.0, 1.0}

    def test_uvw_conjugate_pair_negated(self):
        dc_d, dc_f, dc_n, freqs, *_, antpos, _ = _make_mock_dc(4, 8, antpairs=[(0, 1)])
        _, _, uvw, *_ = pf.unpack_data_containers(dc_d, dc_f, dc_n, antpos=antpos, freqs=freqs)
        np.testing.assert_allclose(uvw[0], -uvw[1])


# ---------------------------------------------------------------------------
# _fit_rotation_measure
# ---------------------------------------------------------------------------

class TestFitRotationMeasure:

    def _make_synthetic_data(self, rm_true=10.0, n_bls=4, n_times=6, n_freqs=32):
        """Noiseless RM signal with zero UVW (position phase = 0)."""
        rng = np.random.default_rng(0)
        freqs = np.linspace(120e6, 180e6, n_freqs)
        lsq = (C / freqs) ** 2
        spec = np.exp(-2j * lsq * rm_true)
        vis = np.broadcast_to(spec[None, None, :], (n_bls, n_times, n_freqs)).copy()
        vis += 0.01 * (rng.standard_normal((n_bls, n_times, n_freqs))
                        + 1j * rng.standard_normal((n_bls, n_times, n_freqs)))
        weights = np.ones_like(vis, dtype=float)
        uvw = np.zeros((n_bls, 3, n_freqs))
        times = np.linspace(2459000.0, 2459000.01, n_times)
        return vis, weights, uvw, times, freqs

    @pytest.mark.parametrize("rm_true", [15.0, -20.0])
    def test_recovers_known_rm(self, rm_true):
        vis, weights, uvw, times, freqs = self._make_synthetic_data(rm_true=rm_true)
        rm_fit = pf._fit_rotation_measure(
            vis, weights, uvw, times, freqs,
            ra=0.0, dec=0.0, start_rm=rm_true,
            location=HERA_LOCATION, drm=20.0, dtest=1000,
        )
        assert abs(rm_fit - rm_true) < 1.0, f"Expected RM ~ {rm_true}, got {rm_fit:.3f}"

    def test_returns_float(self):
        vis, weights, uvw, times, freqs = self._make_synthetic_data()
        rm_fit = pf._fit_rotation_measure(
            vis, weights, uvw, times, freqs,
            ra=0.0, dec=0.0, start_rm=10.0, location=HERA_LOCATION,
        )
        assert isinstance(rm_fit, float)


# ---------------------------------------------------------------------------
# _fit_polarized_source_position
# ---------------------------------------------------------------------------

class TestFitPolarizedSourcePosition:
    """
    Synthetic-data strategy
    -----------------------
    Noiseless visibilities for a point source at (ra, dec, rm):

        V[b,t,f] = A · exp(2πi · uvw·lmn) · exp(−2i·λ²·rm)

    With n_times=1, the fitter's linear model is exact.  At zero offset the
    residual is a uniform constant A, forcing x=[A,0,0] in the normal
    equations, so (ra_fit, dec_fit) = (ra0, dec0) to floating-point precision.
    """

    FREQS = np.linspace(120e6, 180e6, 32)
    RA0, DEC0, RM0 = 45.0, -30.0, 10.0

    def _make_uvw(self, n_bls=6, seed=7):
        rng = np.random.default_rng(seed)
        blvecs = rng.standard_normal((n_bls, 3)) * 100.0
        blvecs[:, 2] = 0.0
        return blvecs[:, :, None] * self.FREQS[None, None, :] / C

    def _make_vis(self, ra, dec, rm, n_times=1, n_bls=6, amplitude=1.0):
        times = np.linspace(2459000.0, 2459000.0 + 5e-4 * n_times, n_times)
        uvw = self._make_uvw(n_bls=n_bls)
        lmn = pf.radec_to_lmn(ra, dec, times, HERA_LOCATION)
        lsq = (C / self.FREQS) ** 2
        phase = np.einsum("bcf,ct->btf", uvw, lmn)
        vis = amplitude * (np.exp(2j * np.pi * phase)
                               * np.exp(-2j * lsq[None, None, :] * rm))
        return vis, np.ones_like(vis, dtype=float), uvw, times

    def _fit(self, vis, weights, uvw, times, ra=None, dec=None, rm=None):
        return pf._fit_polarized_source_position(
            vis, weights, uvw,
            ra if ra is not None else self.RA0,
            dec if dec is not None else self.DEC0,
            rm if rm is not None else self.RM0,
            times, self.FREQS, HERA_LOCATION,
        )

    # --- return-type / basic sanity ---

    def test_returns_finite_floats(self):
        vis, w, uvw, t = self._make_vis(self.RA0, self.DEC0, self.RM0)
        ra_fit, dec_fit = self._fit(vis, w, uvw, t)
        assert isinstance(ra_fit, float) and isinstance(dec_fit, float)
        assert np.isfinite(ra_fit) and np.isfinite(dec_fit)

    # --- zero-offset exactness (parametrised over RM sign/value) ---

    @pytest.mark.parametrize("rm", [10.0, 0.0, -20.0])
    def test_exact_position_zero_offset(self, rm):
        vis, w, uvw, t = self._make_vis(self.RA0, self.DEC0, rm)
        ra_fit, dec_fit = self._fit(vis, w, uvw, t, rm=rm)
        np.testing.assert_allclose(ra_fit, self.RA0, atol=1e-8)
        np.testing.assert_allclose(dec_fit, self.DEC0, atol=1e-8)

    # --- offset recovery ---

    @pytest.mark.parametrize("dra, ddec", [
        (0.01, 0.0),
        (0.0, 0.01),
        (0.008, 0.006),
    ])
    def test_small_offset_recovered(self, dra, ddec):
        ra_true, dec_true = self.RA0 + dra, self.DEC0 + ddec
        vis, w, uvw, t = self._make_vis(ra_true, dec_true, self.RM0)
        ra_fit, dec_fit = self.RA0, self.DEC0

        # Iterate the fitter a few times to allow the linear approximation to improve as we get closer.
        for _ in range(5):
            ra_fit, dec_fit = self._fit(vis, w, uvw, t, ra=ra_fit, dec=dec_fit)

        assert abs(ra_fit - ra_true) < abs(self.RA0 - ra_true) or dra == 0.0
        assert abs(dec_fit - dec_true) < abs(self.DEC0 - dec_true) or ddec == 0.0
        np.testing.assert_allclose(ra_fit, ra_true, atol=1e-3)
        np.testing.assert_allclose(dec_fit, dec_true, atol=1e-3)

    def test_uniform_weight_scaling_invariant(self):
        vis, w, uvw, t = self._make_vis(self.RA0, self.DEC0, self.RM0)
        ra1, dec1 = self._fit(vis, w, uvw, t)
        ra2, dec2 = self._fit(vis, w * 100.0, uvw, t)
        np.testing.assert_allclose(ra1, ra2, atol=1e-10)
        np.testing.assert_allclose(dec1, dec2, atol=1e-10)

    def test_zero_weight_baselines_ignored(self):
        vis, w, uvw, t = self._make_vis(self.RA0, self.DEC0, self.RM0, n_bls=8)
        ra_ref, dec_ref = self._fit(vis, w, uvw, t)

        w_partial = w.copy()
        w_partial[:2] = 0.0
        ra_fit, dec_fit = self._fit(vis, w_partial, uvw, t)
        np.testing.assert_allclose(ra_fit, ra_ref, atol=1e-6)
        np.testing.assert_allclose(dec_fit, dec_ref, atol=1e-6)

    # --- RM de-rotation is load-bearing ---

    def test_correct_rm_beats_wrong_rm(self):
        ra_true, rm_true = self.RA0 + 0.01, 50.0
        vis, w, uvw, t = self._make_vis(ra_true, self.DEC0, rm=rm_true)
        ra_correct, _ = self._fit(vis, w, uvw, t, rm=rm_true)
        ra_wrong, _ = self._fit(vis, w, uvw, t, rm=0.0)
        assert abs(ra_correct - ra_true) < abs(ra_wrong - ra_true)

    # --- multi-time-step handling ---

    @pytest.mark.parametrize("n_times", [1, 6])
    def test_n_times_exact_position(self, n_times):
        vis, w, uvw, t = self._make_vis(self.RA0, self.DEC0, self.RM0, n_times=n_times)
        ra_fit, dec_fit = self._fit(vis, w, uvw, t)
        assert np.isfinite(ra_fit) and np.isfinite(dec_fit)
        if n_times == 1:
            np.testing.assert_allclose(ra_fit, self.RA0, atol=1e-8)
            np.testing.assert_allclose(dec_fit, self.DEC0, atol=1e-8)


# ---------------------------------------------------------------------------
# iteratively_fit_polarized_source_params
# ---------------------------------------------------------------------------

class TestIterativelyFitAllFlagged:

    def test_all_flagged_returns_original_params(self):
        dc_d, dc_f, dc_n, *_ = _make_mock_dc(
            4, 8, antpairs=[(0, 1)], pol="pQ", all_flagged=True
        )
        ra0, dec0, rm0 = 10.0, -30.0, 5.0
        ra, dec, rm = pf.iteratively_fit_polarized_source_params(
            dc_d, dc_f, dc_n,
            right_ascension=ra0, declination=dec0, rotation_measure=rm0,
            location=HERA_LOCATION,
        )
        assert ra == ra0 and dec == dec0 and rm == rm0


class TestIterativelyFitPolarizedSourceParams:
    """
    End-to-end tests for ``iteratively_fit_polarized_source_params``.

    Synthetic-data strategy
    -----------------------
    Visibilities are constructed directly from the point-source model::

        V[(a1,a2,pol)][t,f] = exp(2πi · uvw·lmn(t)) · exp(−2i·λ²·rm)

    where ``lmn(t)`` is computed with :func:`radec_to_lmn` at the *true*
    source position.  Because the forward model matches the fitter's assumed
    signal model exactly, the true parameter values are the unique global
    maximum of the likelihood.  This lets us verify convergence without
    resorting to per-test tolerance tuning.

    Array geometry
    --------------
    Six baselines with good spread in East–West and North–South, chosen to
    give the position fitter a well-conditioned normal matrix.  The source is
    at Dec ≈ HERA's latitude so it transits nearly overhead and remains above
    the horizon throughout the 10-minute observing window.
    """

    # True source parameters used across tests
    RA_TRUE = 45.0   # degrees
    DEC_TRUE = -30.0  # degrees — near HERA latitude, always above horizon
    RM_TRUE = 45.0   # rad/m²

    FREQS = np.linspace(180e6, 200e6, 128)

    ANTPOS = {
        i: np.array([
            np.random.uniform(0, 300),
            np.random.uniform(0, 300),
            0.0
        ])
        for i in range(10)
    }
    ANTPAIRS = [
        (i, j)
        for i in range(10)
        for j in range(i + 1, 10)
    ]
    # 12 integrations spanning 10 minutes — enough time diversity for the
    # position fitter without heavy cost.
    TIMES = np.linspace(2459000.0, 2459000.0 + 10 / 1440.0, 12)

    def _make_dc(
        self,
        ra=None, dec=None, rm=None,
        pol="pQ",
        noise_level=0.0,
        seed=0,
    ):
        """
        Build mock DataContainers whose visibilities follow the exact
        point-source model for the given (ra, dec, rm).

        Conjugate baselines (a2, a1) are stored as the complex conjugate of
        (a1, a2), consistent with the Hermitian symmetry expected by
        ``unpack_data_containers``.
        """
        ra = self.RA_TRUE if ra is None else ra
        dec = self.DEC_TRUE if dec is None else dec
        rm = self.RM_TRUE if rm is None else rm

        rng = np.random.default_rng(seed)
        freqs = self.FREQS
        times = self.TIMES
        antpos = self.ANTPOS
        antpairs = self.ANTPAIRS
        n_times = len(times)
        n_freqs = len(freqs)

        lambda_sq = (C / freqs) ** 2
        lmn = pf.radec_to_lmn(ra, dec, times, HERA_LOCATION)  # (3, n_times)
        rm_phasor = np.exp(-2j * lambda_sq * rm)                      # (n_freqs,)

        data, flags, nsamples = {}, {}, {}
        for ap in antpairs:
            blvec = antpos[ap[1]] - antpos[ap[0]]
            uvw_bl = blvec[:, None] * freqs[None, :] / C   # (3, n_freqs)

            # phase[t,f] = u·l(t) + v·m(t) + w·n(t)
            phase = np.einsum("cf,ct->tf", uvw_bl, lmn)    # (n_times, n_freqs)
            vis = np.exp(2j * np.pi * phase) * rm_phasor[None, :]

            if noise_level > 0.0:
                vis += noise_level * (
                    rng.standard_normal((n_times, n_freqs))
                    + 1j * rng.standard_normal((n_times, n_freqs))
                )

            key = ap + (pol,)
            rev_key = (ap[1], ap[0], pol)
            data[key] = vis
            data[rev_key] = vis.conj()
            flag_arr = np.zeros((n_times, n_freqs), dtype=bool)
            flags[key] = flag_arr
            flags[rev_key] = flag_arr.copy()
            nsamples[key] = np.ones((n_times, n_freqs))
            nsamples[rev_key] = np.ones((n_times, n_freqs))

        def _mock(d):
            dc = MagicMock(spec=datacontainer.DataContainer)
            dc.__getitem__ = lambda self, k: d[k]
            dc.antpairs = MagicMock(return_value=antpairs)
            dc.freqs = freqs
            dc.times = times
            dc.antpos = antpos
            return dc

        return _mock(data), _mock(flags), _mock(nsamples)

    def _fit(self, dc_d, dc_f, dc_n, ra=None, dec=None, rm=None, **kwargs):
        """Thin wrapper that supplies defaults for the initial guess."""
        return pf.iteratively_fit_polarized_source_params(
            dc_d, dc_f, dc_n,
            right_ascension=self.RA_TRUE if ra is None else ra,
            declination=self.DEC_TRUE if dec is None else dec,
            rotation_measure=self.RM_TRUE if rm is None else rm,
            location=HERA_LOCATION,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Initialised at truth — should stay there
    # ------------------------------------------------------------------

    def test_initialized_at_truth_stays_near_truth(self):
        """
        When the initial guess equals the true position and RM, every
        iteration is a no-op and all three outputs must remain within the
        fitter's convergence tolerances of the truth.
        """
        dc_d, dc_f, dc_n = self._make_dc()
        ra, dec, rm = self._fit(dc_d, dc_f, dc_n, drm=2.0, dtest=500)
        np.testing.assert_allclose(ra, self.RA_TRUE, atol=1e-3)
        np.testing.assert_allclose(dec, self.DEC_TRUE, atol=1e-3)
        np.testing.assert_allclose(rm, self.RM_TRUE, atol=0.1)

    # ------------------------------------------------------------------
    # Each parameter corrected independently
    # ------------------------------------------------------------------

    def test_corrects_ra_offset(self):
        """An incorrect initial RA is pulled toward truth."""
        dra = 0.02
        dc_d, dc_f, dc_n = self._make_dc()
        ra, dec, rm = self._fit(dc_d, dc_f, dc_n, ra=self.RA_TRUE + dra,
                                drm=2.0, dtest=500)
        assert abs(ra - self.RA_TRUE) < dra, (
            f"RA should improve: initial error {dra:.4f} deg, "
            f"final error {abs(ra - self.RA_TRUE):.4f} deg"
        )

    def test_corrects_dec_offset(self):
        """An incorrect initial Dec is pulled toward truth."""
        ddec = 0.02
        dc_d, dc_f, dc_n = self._make_dc()
        ra, dec, rm = self._fit(dc_d, dc_f, dc_n, dec=self.DEC_TRUE + ddec,
                                drm=2.0, dtest=500)
        assert abs(dec - self.DEC_TRUE) < ddec, (
            f"Dec should improve: initial error {ddec:.4f} deg, "
            f"final error {abs(dec - self.DEC_TRUE):.4f} deg"
        )

    def test_corrects_rm_offset(self):
        """An incorrect initial RM is pulled toward truth."""
        drm_offset = 3.0
        dc_d, dc_f, dc_n = self._make_dc()
        ra, dec, rm = self._fit(
            dc_d, dc_f, dc_n, rm=self.RM_TRUE + drm_offset,
            drm=drm_offset + 1.0,   # search window must contain the truth
            dtest=500,
        )
        assert abs(rm - self.RM_TRUE) < drm_offset, (
            f"RM should improve: initial error {drm_offset:.2f} rad/m², "
            f"final error {abs(rm - self.RM_TRUE):.2f} rad/m²"
        )

    # ------------------------------------------------------------------
    # Joint end-to-end recovery
    # ------------------------------------------------------------------

    def test_end_to_end_joint_recovery(self):
        """
        Starting with small offsets in all three parameters simultaneously,
        the fitter must converge to the truth within tight tolerances.
        """
        dc_d, dc_f, dc_n = self._make_dc()
        ra, dec, rm = self._fit(
            dc_d, dc_f, dc_n,
            ra=self.RA_TRUE + 0.01,
            dec=self.DEC_TRUE + 0.01,
            rm=self.RM_TRUE + 1.0,
            drm=3.0,
            dtest=500,
            maxiter=20,
        )
        np.testing.assert_allclose(ra, self.RA_TRUE, atol=1e-3,
                                   err_msg="RA did not converge to truth")
        np.testing.assert_allclose(dec, self.DEC_TRUE, atol=1e-3,
                                   err_msg="Dec did not converge to truth")
        np.testing.assert_allclose(rm, self.RM_TRUE, atol=0.1,
                                   err_msg="RM did not converge to truth")

import numpy as np
from .. import flagging
import pytest


class TestThresholdFlags:
    def test_inplace(self):
        flags = np.zeros((10, 12), dtype=bool)  # shape = (ntime, ...)
        flags[:8, 0] = True

        new = flagging.threshold_flags(flags, flag_thresh=0.7, inplace=True)

        assert np.all(flags[:, 0])
        assert np.all(new == flags)

    def test_not_inplace(self):
        flags = np.zeros((10, 12), dtype=bool)  # shape = (ntime, ...)
        flags[:8, 0] = True

        new = flagging.threshold_flags(flags, flag_thresh=0.7, inplace=False)

        assert not np.all(flags[:, 0])
        assert np.all(new[:, 0])


# Define a fixture for common array shapes used in the tests
@pytest.fixture(params=[(10, 5, 5, 2), (8, 10, 10, 4), (15, 3, 7, 1)])
def array_shape(request):
    return request.param


class TestSigmaClip:
    # Happy path tests with various realistic test values
    @pytest.mark.parametrize(
        "threshold, min_N, median_axis, threshold_axis, clip_type, flag_bands",
        [
            (4.0, 4, 0, 0, 'direct', None),
            (3.0, 4, 1, 1, 'mean', [(1, 3), (4, 5)]),
            (5.0, 4, 2, 2, 'median', [(2, 4)]),
        ]
    )
    def test_sigma_clip_happy_path(
        self, array_shape, threshold, min_N, median_axis, threshold_axis, clip_type, flag_bands
    ):
        # Arrange
        np.random.seed(42)
        array = np.random.normal(size=array_shape)

        print(
            f"threshold={threshold}, min_N={min_N}, median_axis={median_axis}, "
            f"threshold_axis={threshold_axis}, array_shape={array_shape}, "
            f"clip_type={clip_type}, flag_bands={flag_bands}"
        )
        # Act
        clip_flags = flagging.sigma_clip(
            array=array,
            threshold=threshold,
            min_N=min_N,
            median_axis=median_axis,
            threshold_axis=threshold_axis,
            clip_type=clip_type,
            flag_bands=flag_bands
        )

        # Assert
        assert isinstance(clip_flags, np.ndarray), "The output should be an ndarray."
        assert clip_flags.dtype == bool, "The output array should be of boolean type."
        assert clip_flags.shape == array.shape, "The output flags should have the same shape as the input array."

    # Edge cases
    @pytest.mark.parametrize("array, threshold, min_N, median_axis, threshold_axis, clip_type, flag_bands, test_id", [
        (np.array([]), 4.0, 4, 0, 0, 'direct', None, 'edge_empty_array'),
        (np.array([np.nan, np.nan]), 4.0, 4, 0, 0, 'direct', None, 'edge_all_nan'),
        (np.random.normal(size=(5, 5)), 4.0, 6, 0, 0, 'direct', None, 'edge_below_min_N'),
    ])
    def test_sigma_clip_edge_cases(self, array, threshold, min_N, median_axis, threshold_axis, clip_type, flag_bands, test_id):
        # Act
        clip_flags = flagging.sigma_clip(
            array, threshold, min_N, median_axis, threshold_axis, clip_type, flag_bands
        )

        # Assert
        assert isinstance(clip_flags, np.ndarray), f"Test ID: {test_id} - The output should be an ndarray."
        assert clip_flags.dtype == bool, f"Test ID: {test_id} - The output array should be of boolean type."
        assert clip_flags.shape == array.shape, f"Test ID: {test_id} - The output flags should have the same shape as the input array."

    # Error cases
    def test_sigma_clip_error_cases(self):
        # Act / Assert
        array = np.array([1 + 1j, 2 + 2j])
        with pytest.raises(ValueError, match=".*must be real.*"):
            flagging.sigma_clip(array)

        array = np.array([1, 2])
        with pytest.raises(ValueError, match=".*clip_type.*"):
            flagging.sigma_clip(array, min_N=0, clip_type='wrong_clip_type')

    @pytest.mark.parametrize(
        "clip_type", ['direct', 'mean', 'median']
    )
    @pytest.mark.parametrize(
        "flag_bands", [None, [(0, 12)], [(0, 1), (1, 12)]]
    )
    def test_constant_data_no_flags(self, clip_type, flag_bands):
        """Test an array with a reasonable expected shape, but constant values.

        Expect that nothing is flagged in this case.
        """
        array = np.ones((10, 8, 12, 4))  # nnights, nbls, nfreqs, npols
        clip_flags = flagging.sigma_clip(
            array,
            clip_type=clip_type,
            flag_bands=flag_bands,
            median_axis=0,
            threshold_axis=2,
            min_N=0,
            threshold=3.0,
        )
        assert not np.any(clip_flags)

    def test_spiky_data_direct_clip(self):
        """Test an array with a reasonable expected shape with spiky outliers.

        We expect that JUST the outliers are flagged.
        """
        array = np.random.normal(size=(100, 8, 12, 4))  # nnights, nbls, nfreqs, npols

        array[1, 0, 0, 0] = 1000
        array[0, 1, 0, 0] = 1000
        array[0, 0, 1, 0] = 1000
        array[0, 0, 0, 1] = 1000

        clip_flags = flagging.sigma_clip(
            array,
            clip_type='direct',
            median_axis=0,
            threshold_axis=2,
            min_N=0,
            threshold=10.0,
        )

        assert np.sum(clip_flags) == 4
        assert clip_flags[1, 0, 0, 0]
        assert clip_flags[0, 1, 0, 0]
        assert clip_flags[0, 0, 1, 0]
        assert clip_flags[0, 0, 0, 1]

    @pytest.mark.parametrize(
        "flag_bands", [[(0, 12)], [(0, 2), (2, 12)]]
    )
    @pytest.mark.parametrize(
        "clip_type", ['mean', 'median']
    )
    def test_spiky_data_mean_clip(self, flag_bands, clip_type):
        """Test an array with a reasonable expected shape with spiky outliers.

        In this case, we aggregate over the frequency axis by taking the mean,
        and we also flag in particular bands.
        """
        array = np.random.normal(size=(100, 8, 12, 4))

        array[0, 0, flag_bands[0][0]:flag_bands[0][1], 0] = 1000

        clip_flags = flagging.sigma_clip(
            array,
            clip_type=clip_type,
            median_axis=0,
            threshold_axis=2,
            min_N=0,
            threshold=6.0,
            flag_bands=flag_bands,
        )

        # The entire first band should be clipped, but nothing else.
        assert np.sum(clip_flags) == flag_bands[0][1] - flag_bands[0][0]
        assert np.all(clip_flags[0, 0, flag_bands[0][0]:flag_bands[0][1], 0])

    def test_passing_list(self):
        """Test that passing a list of arrays works."""
        array = np.random.normal(size=(8, 12, 4))
        clip_flags = flagging.sigma_clip(
            [array, array],
            clip_type='direct',
            median_axis=0,
            threshold_axis=2,
            min_N=0,
            threshold=3.0,
        )

        assert clip_flags[0].shape == array.shape
        assert clip_flags[1].shape == array.shape

    @pytest.mark.parametrize(
        "scale", [
            np.random.normal(size=(8, 12, 4)),  # no median_axis
            np.random.normal(size=(1, 8, 12, 4)),  # dummy median axis
            np.random.normal(size=(10, 8, 12, 4)),  # full median axis
        ]
    )
    def test_passing_good_scale(self, scale):
        """Test that passing a scale works."""
        array = np.random.normal(size=(10, 8, 12, 4))
        clip_flags = flagging.sigma_clip(
            array,
            clip_type='direct',
            median_axis=0,
            threshold_axis=2,
            min_N=0,
            threshold=3.0,
            scale=scale,
        )

        assert clip_flags.shape == array.shape

    @pytest.mark.parametrize(
        "scale", [
            np.random.normal(size=(12, 4)),  # no median_axis
            np.random.normal(size=(11, 8, 12, 4)),  # dummy median axis
        ]
    )
    def test_passing_bad_scale(self, scale):
        """Test that passing a scale works."""
        array = np.random.normal(size=(10, 8, 12, 4))
        with pytest.raises(ValueError, match='scale must have same shape as array'):
            clip_flags = flagging.sigma_clip(
                array,
                clip_type='direct',
                median_axis=0,
                threshold_axis=2,
                min_N=0,
                threshold=3.0,
                scale=scale,
            )

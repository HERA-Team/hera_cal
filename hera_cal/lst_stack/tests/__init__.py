import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore:The uvw_array does not match the expected values given the antenna positions.",
    "ignore:.*Using known values for HERA",
)

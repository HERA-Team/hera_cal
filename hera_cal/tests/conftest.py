# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"""Testing environment setup and teardown for pytest."""
from __future__ import absolute_import, division, print_function

import os
import pytest
import six.moves.urllib as urllib
from astropy.utils import iers
from astropy.time import Time
import warnings
from pyuvdata.telescopes import ignore_telescope_param_update_warnings_for

# Update warnings related to updating params for the HERA telescope
ignore_telescope_param_update_warnings_for("HERA")


@pytest.fixture(autouse=True, scope="session")
def setup_and_teardown_package():
    # Do a calculation that requires a current IERS table. This will trigger
    # automatic downloading of the IERS table if needed, including trying the
    # mirror site in python 3 (but won't redownload if a current one exists).
    # If there's not a current IERS table and it can't be downloaded, turn off
    # auto downloading for the tests and turn it back on once all tests are
    # completed (done by extending auto_max_age).
    # Also, the checkWarnings function will ignore IERS-related warnings.
    try:
        t1 = Time.now()
        t1.ut1
    except (Exception):
        iers.conf.auto_max_age = None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Fixing auto-correlations to be be real-only")
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="The uvw_array does not match the expected values given the antenna position")
        warnings.filterwarnings("ignore", message="telescope_location is not set")
        yield

    iers.conf.auto_max_age = 30

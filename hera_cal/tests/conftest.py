# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"""Testing environment setup and teardown for pytest."""
from __future__ import absolute_import, division, print_function

import os
import pytest
import six.moves.urllib as urllib
from astropy.utils import iers


@pytest.fixture(autouse=True, scope="session")
def setup_and_teardown_package():
    # try to download the iers table. If it fails, turn off auto downloading for the tests
    # and turn it back on in teardown_package (done by extending auto_max_age)
    try:
        iers_a = iers.IERS_A.open(iers.IERS_A_URL)
    except(urllib.error.URLError):
        iers.conf.auto_max_age = None

    yield

    iers.conf.auto_max_age = 30

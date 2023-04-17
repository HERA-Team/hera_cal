from hera_cal.red_groups import RedundantGroups
import pytest
import numpy as np


class TestRedundantGroups:
    # Tests creating a RedundantGroups object from an antpos dictionary with default 
    # arguments. 
    def test_create_redundant_groups(self):
        antpos = {
            0: np.array([0, 0, 0]), 
            1: np.array([14, 0, 0]), 
            2: np.array([0, 14, 0]), 
            3: np.array([0, 0, 14])
        }
        rg = RedundantGroups.from_antpos(antpos)
        assert len(rg) == 7
        assert (0, 0) in rg
        assert (0, 1) in rg
        assert (0, 2) in rg
        assert (1, 2) in rg
        assert (0, 3) in rg
        assert (1, 3) in rg
        assert (2, 3) in rg

    # Tests filtering baselines out of a RedundantGroups object. 
    def test_filter_reds(self):
        antpos = {
            0: np.array([0, 0, 0]), 
            1: np.array([14, 0, 0]), 
            2: np.array([0, 14, 0]), 
            3: np.array([0, 0, 14])
        }
        rg = RedundantGroups.from_antpos(antpos, include_autos=False)
        rg_filtered = rg.filter_reds(bls=[(0, 1), (0, 2), (1, 2)])
        assert len(rg_filtered) == 3
        assert (0, 1) in rg_filtered
        assert (0, 2) in rg_filtered
        assert (1, 2) in rg_filtered

        rg.filter_reds(ants=[0, 1, 2])
        assert len(rg) == 3
        assert (0, 1) in rg
        assert (0, 2) in rg
        assert (1, 2) in rg

    # Tests inserting a new redundant group at the beginning or end of the list of 
    # redundant groups. 
    def test_insert_redundant_group(self):
        antpos = {0: np.array([0, 0, 0]), 1: np.array([14, 0, 0]), 2: np.array([0, 14, 0]), 3: np.array([0, 0, 14])}
        rg = RedundantGroups.from_antpos(antpos)
        rg.insert(0, [(0, 1), (0, 2)])
        assert len(rg) == 7
        assert rg[0] == [(0, 1), (0, 2)]
        rg.insert(-1, [(1, 2), (1, 3)])
        assert len(rg) == 8
        assert rg[-1] == [(1, 2), (1, 3)]

    # Tests adding two RedundantGroups objects together.  
    def test_add_redundant_groups(self):
        antpos1 = {0: np.array([0, 0, 0]), 1: np.array([1, 0, 0]), 2: np.array([0, 1, 0])}
        antpos2 = {3: np.array([1, 1, 0]), 4: np.array([0, 0, 1]), 5: np.array([1, 0, 1])}
        rg1 = RedundantGroups.from_antpos(antpos1)
        rg2 = RedundantGroups.from_antpos(antpos2)
        rg3 = rg1 + rg2
        assert len(rg3) == len(rg1) + len(rg2)
        assert set(rg3.data_ants) == set(antpos1.keys()) | set(antpos2.keys())

    # Tests getting a redundant group for a key that is not in the data.  
    def test_get_red_not_in_data(self):
        antpos = {0: np.array([0, 0, 0]), 1: np.array([1, 0, 0]), 2: np.array([0, 1, 0])}
        rg = RedundantGroups.from_antpos(antpos)
        with pytest.raises(KeyError):
            rg.get_red((0, 3))

    # Tests setting a redundant group for a key that is not in the data.  
    def test_set_red_not_in_data(self):
        antpos = {0: np.array([0, 0, 0]), 1: np.array([1, 0, 0]), 2: np.array([0, 1, 0])}
        rg = RedundantGroups.from_antpos(antpos)
        with pytest.raises(ValueError):
            rg[(0, 3)] = [(0, 1), (1, 3)]
from hera_cal.red_groups import RedundantGroups
import pytest
import numpy as np
import copy


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
        rg = RedundantGroups.from_antpos(antpos, pols=('nn',), include_autos=False)
        rg_filtered = rg.filter_reds(ants=[0, 1, 2])
        assert len(rg_filtered) == 3
        assert (0, 3, 'nn') not in rg_filtered

    # Tests inserting a new redundant group at the beginning or end of the list of
    # redundant groups.
    def test_insert_redundant_group(self):
        antpos = {
            0: np.array([0, 0, 0]),
            1: np.array([14, 0, 0]),
            2: np.array([0, 14, 0]),
            3: np.array([-14, 0, 0])
        }
        rg = RedundantGroups([[(0, 1), (3, 0)], [(0, 2)]], antpos=antpos)
        rg.insert(0, [(1, 2)])
        assert len(rg) == 3
        assert rg[0] == [(1, 2)]
        rg.append([(2, 3)])
        assert len(rg) == 4
        assert rg[-1] == [(2, 3)]

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

    def test_init_with_multiple_types(self):
        with pytest.raises(TypeError, match="All baselines must have the same type, got an AntPair and a Baseline"):
            RedundantGroups(
                [[(0, 1)], [(0, 2, 'nn')]],
            )

    def test_bad_add(self):
        with pytest.raises(TypeError, match="can only add RedundantGroups object to RedundantGroups object"):
            RedundantGroups([[(0, 1)]]) + 3
        with pytest.raises(ValueError, match="can't add RedundantGroups objects with one having antpos"):
            RedundantGroups([[(0, 1)]], antpos={0: np.array([0, 0, 0])}) + RedundantGroups([[(0, 1)]])

    def test_add_no_antpos(self):
        rg1 = RedundantGroups([[(0, 1)]])
        rg2 = RedundantGroups([[(0, 2)]])
        rg3 = rg1 + rg2
        assert len(rg3) == 2
        assert set(rg3.data_ants) == set([0, 1, 2])

    def test_add_antpos(self):
        rg1 = RedundantGroups.from_antpos(antpos={0: np.array([0, 0, 0]), 1: np.array([10, 0, 0])})
        rg2 = RedundantGroups.from_antpos(antpos={0: np.array([0, 0, 0]), 2: np.array([0, 10, 0])})
        rg3 = rg1 + rg2
        print(rg3)
        assert len(rg3) == 3
        assert set(rg3.data_ants) == set([0, 1, 2])

    def test_append_with_bad_ant(self):
        rg = RedundantGroups([[(0, 1)]], antpos={0: np.array([0, 0, 0]), 1: np.array([1, 0, 0])})

        with pytest.raises(ValueError, match="not in antpos"):
            rg.append([(0, 2)])

        with pytest.raises(ValueError, match="not in antpos"):
            rg.append([(2, 0)])

    def test_append_existing_different(self):
        rg = RedundantGroups(
            [[(0, 1)], [(0, 2), (0, 3)]],
        )

        with pytest.raises(ValueError, match="Attempting to add a new redundant group where some baselines"):
            rg.append([(0, 1), (0, 2)])

    def test_append_existing_same(self):
        rg = RedundantGroups(
            [[(0, 1), (0, 2), (0, 3)]],
        )

        rg.append([(0, 1), (0, 2), (0, 3)])

        assert len(rg) == 1
        assert rg[0] == [(0, 1), (0, 2), (0, 3)]

        new = rg.append([(0, 1), (0, 2), (0, 3)], inplace=False)
        assert len(new) == 1
        assert new[0] == [(0, 1), (0, 2), (0, 3)]

    def test_append_some_exist(self):
        rg = RedundantGroups(
            [[(0, 1), (0, 2), (0, 3)]],
        )

        rg.append([(0, 1), (0, 2), (0, 3), (0, 4)])

        assert len(rg) == 1
        assert rg[0] == [(0, 1), (0, 2), (0, 3), (0, 4)]

    def test_getitem_keyint(self):
        rg = RedundantGroups(
            [[(0, 1)], [(0, 2), (0, 3)]],
        )

        assert rg[0] == [(0, 1)]
        assert rg[1] == [(0, 2), (0, 3)]
        assert rg[(0, 1)] == [(0, 1)]
        assert rg[(0, 2)] == [(0, 2), (0, 3)]

    def test_setitem(self):
        rg = RedundantGroups(
            [[(0, 1)], [(0, 2), (0, 3)]],
        )

        rg[0] = [(0, 1), (0, 4)]
        assert rg[0] == [(0, 1), (0, 4)]

        rg[(0, 1)] = [(0, 1), (0, 4), (0, 5)]
        assert rg[0] == [(0, 1), (0, 4), (0, 5)]

        rg[(0, 2)] = [(0, 1), (0, 2)]
        assert rg[1] == [(0, 1), (0, 2)]

    def test_indexing(self):
        rg = RedundantGroups(
            [[(0, 1)], [(0, 2), (0, 3)]],
        )
        rg.index((0, 1)) == 0
        rg.index((0, 2)) == 1
        rg.index((0, 3)) == 1

    def test_filter_reds_inplace(self):
        rg = RedundantGroups.from_antpos(
            antpos={
                0: np.array([0, 0, 0]),
                1: np.array([10, 0, 0]),
                2: np.array([0, 10, 0]),
                3: np.array([-10, 0, 0])
            },
            include_autos=True,
            pols=('ee',)
        )
        rg.filter_reds(ants=(0, 1), inplace=True)
        assert rg.data_ants == {0, 1}
        print(rg)
        assert len(rg) == 2

    def test_sort(self):
        rg = RedundantGroups(
            [[(0, 2), (0, 3)], [(0, 1)]],
        )
        rg.sort()
        assert rg[0] == [(0, 1)]
        assert rg[1] == [(0, 2), (0, 3)]

    def test_get_full_redundancies(self):
        rg = RedundantGroups.from_antpos(
            antpos={
                0: np.array([0, 0, 0]),
                1: np.array([1, 0, 0]),
                2: np.array([0, 1, 0]),
                3: np.array([-1, 0, 0])
            },
            include_autos=True
        )
        new = copy.deepcopy(rg)
        del new[(0, 1)]
        newrg = new.get_full_redundancies()
        assert len(newrg) == len(rg) != new

        rg = RedundantGroups.from_antpos(
            antpos={
                0: np.array([0, 0, 0]),
                1: np.array([1, 0, 0]),
                2: np.array([0, 1, 0]),
                3: np.array([-1, 0, 0])
            },
            include_autos=True,
            pols=('ee',)
        )
        new = rg.filter_reds(ants=(0, 1), inplace=False)
        newrg = new.get_full_redundancies()
        assert len(newrg) == len(rg) != new

    def test_keyed_on_bls(self):
        rg = RedundantGroups(
            [[(0, 1)], [(0, 2), (0, 3)]],
        )
        new = rg.keyed_on_bls(bls=[(0, 3)])
        assert (0, 2) in new
        assert new.get_ubl_key((0, 2)) == (0, 3)
        assert new.get_ubl_key((0, 1)) == (0, 1)

        new.keyed_on_bls(bls=[(0, 2)], inplace=True)
        assert new.get_ubl_key((0, 2)) == (0, 2)

    def test_delitem(self):
        rg = RedundantGroups(
            [[(0, 1)], [(0, 2), (0, 3)]],
        )
        del rg[(0, 1)]
        assert len(rg) == 1
        assert (0, 1) not in rg
        assert (0, 2) in rg
        assert (0, 3) in rg
        del rg[0]
        assert len(rg) == 0

    def test_clear_before_instantiated(self):
        rg = RedundantGroups([])
        rg.clear_cache()
        assert len(rg) == 0

    def test_data_bls(self):
        rg = RedundantGroups(
            [[(0, 1)], [(0, 2), (0, 3)]],
        )
        assert len(rg.data_bls) == 3


class TestRedundantGroupsGetRedsInBlSet:
    def setup_method(self, method):
        self.rg = RedundantGroups(
            [[(0, 1)], [(0, 2), (0, 3)]],
        )

    def test_default(self):
        assert self.rg.get_reds_in_bl_set((0, 1), {(0, 1), (0, 2)}) == {(0, 1)}
        assert self.rg.get_reds_in_bl_set((0, 1), {(0, 2)}) == set()
        assert self.rg.get_reds_in_bl_set((0, 2), {(0, 2)}) == {(0, 2)}
        assert self.rg.get_reds_in_bl_set((0, 2), {(0, 2), (0, 3)}) == {(0, 2), (0, 3)}

    def test_conj_in_set(self):
        """Test that passing a baseline that exists and only having the conj in the set returns nothing."""
        assert self.rg.get_reds_in_bl_set((0, 1), {(1, 0), (0, 2)}) == set()
        assert self.rg.get_reds_in_bl_set((0, 2), {(2, 0)}) == set()
        assert self.rg.get_reds_in_bl_set((0, 2), {(2, 0), (0, 3)}) == {(0, 3)}

        # However, if we *give* it the conjugate, it should work
        assert self.rg.get_reds_in_bl_set((1, 0), {(1, 0), (0, 2)}) == {(1, 0)}

    def test_bad_key(self):
        with pytest.raises(KeyError):
            # The baseline (0, 4) doesn't exist
            self.rg.get_reds_in_bl_set((0, 4), {})

        with pytest.raises(KeyError):
            self.rg.get_reds_in_bl_set((0, 1, 'ee'), {})

    def test_conj_in_set_include_conj_match_bl(self):
        # The following test that at least the conjugates are returned
        assert self.rg.get_reds_in_bl_set((0, 1), {(1, 0), (0, 2)}, include_conj=True) == {(0, 1)}
        assert self.rg.get_reds_in_bl_set((0, 2), {(2, 0)}, include_conj=True) == {(0, 2)}
        assert self.rg.get_reds_in_bl_set((0, 2), {(2, 0), (0, 3)}, include_conj=True) == {(0, 2), (0, 3)}

        # If both the original and conjugate are in the set, only the original is returned, unless match_conj_to_set is True
        assert self.rg.get_reds_in_bl_set((0, 1), {(1, 0), (0, 2), (0, 1)}, include_conj=True) == {(0, 1)}
        assert self.rg.get_reds_in_bl_set((0, 2), {(2, 0), (0, 2)}, include_conj=True) == {(0, 2)}
        assert self.rg.get_reds_in_bl_set((0, 2), {(2, 0), (0, 3), (0, 2)}, include_conj=True) == {(0, 2), (0, 3)}

    def test_conj_in_set_include_conj_match_set(self):
        # If both the original and conjugate are in the set, both returned if match_conj_to_set is True
        kw = dict(include_conj=True, match_conj_to_set=True)
        assert self.rg.get_reds_in_bl_set((0, 1), {(1, 0), (0, 2), (0, 1)}, **kw) == {(0, 1), (1, 0)}
        assert self.rg.get_reds_in_bl_set((0, 2), {(2, 0), (0, 2)}, **kw) == {(0, 2), (2, 0)}
        assert self.rg.get_reds_in_bl_set((0, 2), {(2, 0), (0, 3), (0, 2)}, **kw) == {(0, 2), (2, 0), (0, 3)}

    def test_conj_in_set_only_if_missing(self):
        # If the original is in the set, don't return the conjugate
        kw = dict(include_conj=True, include_conj_only_if_missing=True, match_conj_to_set=True)
        assert self.rg.get_reds_in_bl_set((0, 1), {(0, 1), (1, 0)}, **kw) == {(0, 1)}
        assert self.rg.get_reds_in_bl_set((0, 2), {(2, 0), (3, 0)}, **kw) == {(2, 0), (3, 0)}
        assert self.rg.get_reds_in_bl_set((0, 2), {(0, 2), (3, 0)}, **kw) == {(0, 2), (3, 0)}

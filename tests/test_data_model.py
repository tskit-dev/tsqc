import logging

import msprime
import numpy as np
import numpy.testing as nt
import pytest
import tskit

from tsqc import model


def single_tree_example_ts():
    # 2.00┊    6    ┊
    #     ┊  ┏━┻━┓  ┊
    # 1.00┊  4   5  ┊
    #     ┊ ┏┻┓ ┏┻┓ ┊
    # 0.00┊ 0 1 2 3 ┊
    #     0         10
    ts = tskit.Tree.generate_balanced(4, span=10).tree_sequence
    tables = ts.dump_tables()
    for j in range(6):
        tables.sites.add_row(position=j + 1, ancestral_state="A")
        tables.mutations.add_row(site=j, derived_state="T", node=j)
    tables.sites.add_row(position=7, ancestral_state="FOOBAR")
    tables.mutations.add_row(site=6, derived_state="FOOBARD", node=6)
    return tables.tree_sequence()


def single_tree_recurrent_mutation_example_ts():
    # 2.00 ┊                    6                    ┊
    #      ┊            ┏━━━━━━━┻━━━━━━━┓            ┊
    #      ┊      4:A→T x               x 5:A→T      ┊
    #      ┊            |               x 6:A→G      ┊
    # 1.00 ┊            4               5            ┊
    #      ┊       ┏━━━━┻━━━━┓     ┏━━━━┻━━━━┓       ┊
    #      ┊ 0:A→T x   1:A→T x     x 2:A→T   x 3:A→T ┊
    #      ┊       |         |     |         |       ┊
    # 0.00 ┊       0         1     2         3       ┊
    #      0                                        10
    ts = tskit.Tree.generate_balanced(4, span=10).tree_sequence
    tables = ts.dump_tables()
    for j in range(6):
        tables.sites.add_row(position=j + 1, ancestral_state="A")
        tables.mutations.add_row(site=j, derived_state="T", node=j)
    tables.mutations.add_row(site=j, derived_state="G", node=j, parent=j)
    ts = tables.tree_sequence()
    return tables.tree_sequence()


def multiple_trees_example_ts():
    # 2.00┊   4   ┊   4   ┊
    #     ┊ ┏━┻┓  ┊  ┏┻━┓ ┊
    # 1.00┊ ┃  3  ┊  3  ┃ ┊
    #     ┊ ┃ ┏┻┓ ┊ ┏┻┓ ┃ ┊
    # 0.00┊ 0 1 2 ┊ 0 1 2 ┊
    #     0       5      10
    ts = tskit.Tree.generate_balanced(3, span=10).tree_sequence
    tables = ts.dump_tables()
    tables.edges[1] = tables.edges[1].replace(right=5)
    tables.edges[2] = tables.edges[2].replace(right=5)
    tables.edges.add_row(5, 10, 3, 0)
    tables.edges.add_row(5, 10, 4, 2)
    tables.sort()
    return tables.tree_sequence()


def single_tree_with_polytomies_example_ts():
    # 3.00┊         8         ┊
    #     ┊  ┏━━━━━━╋━━━━━━━┓ ┊
    # 2.00┊  ┃      7       ┃ ┊
    #     ┊  ┃  ┏━━━╋━━━━┓  ┃ ┊
    # 1.00┊  5  ┃   6    ┃  ┃ ┊
    #     ┊ ┏┻┓ ┃ ┏━╋━━┓ ┃  ┃ ┊
    # 0.00┊ 0 1 2 3 4 11 9 10 ┊
    #     0                  10
    ts = tskit.Tree.generate_balanced(5, span=10).tree_sequence
    tables = ts.dump_tables()
    tables.nodes.add_row(flags=1, time=0)
    tables.edges.add_row(0, 10, 7, 9)
    tables.nodes.add_row(flags=1, time=0)
    tables.edges.add_row(0, 10, 8, 10)
    tables.nodes.add_row(flags=1, time=0)
    tables.edges.add_row(0, 10, 6, 11)
    tables.sort()
    return tables.tree_sequence()


def multi_tree_with_polytomies_example_ts():
    # 3.00┊     8       ┊     8       ┊
    #     ┊  ┏━━┻━┓     ┊  ┏━━┻━━┓    ┊
    # 2.00┊  ┃    7     ┊  ┃     7    ┊
    #     ┊  ┃  ┏━┻━┓   ┊  ┃  ┏━━╋━━┓ ┊
    # 1.00┊  5  ┃   6   ┊  5  ┃  6  ┃ ┊
    #     ┊ ┏┻┓ ┃ ┏━╋━┓ ┊ ┏┻┓ ┃ ┏┻┓ ┃ ┊
    # 0.00┊ 0 1 2 3 4 9 ┊ 0 1 2 3 4 9 ┊
    #     0             5            10
    ts = tskit.Tree.generate_balanced(5, span=10).tree_sequence
    tables = ts.dump_tables()
    tables.nodes.add_row(flags=1, time=0)
    tables.edges.add_row(0, 5, 6, 9)
    tables.edges.add_row(5, 10, 7, 9)
    tables.sort()
    return tables.tree_sequence()


class TestMutationDataTable:
    def test_single_tree_example(self):
        ts = single_tree_example_ts()
        tsm = model.TSModel(ts)
        df = tsm.mutations_df
        assert len(df) == 7
        nt.assert_array_equal(df.id, list(range(7)))
        nt.assert_array_equal(df.node, list(range(7)))
        nt.assert_array_equal(df.position, list(range(1, 8)))
        nt.assert_array_equal(df.time, [0, 0, 0, 0, 1, 1, 2])
        nt.assert_array_equal(df.derived_state, ["T"] * 6 + ["FOOBARD"])
        nt.assert_array_equal(df.inherited_state, ["A"] * 6 + ["FOOBAR"])
        nt.assert_array_equal(df.num_parents, [0] * 7)
        nt.assert_array_equal(df.num_descendants, [1] * 4 + [2] * 2 + [4])
        nt.assert_array_equal(df.num_inheritors, [1] * 4 + [2] * 2 + [4])

    def test_single_tree_recurrent_mutation_example(self):
        ts = single_tree_recurrent_mutation_example_ts()
        tsm = model.TSModel(ts)
        df = tsm.mutations_df
        assert len(df) == 7
        nt.assert_array_equal(df.id, list(range(7)))
        nt.assert_array_equal(df.node, [0, 1, 2, 3, 4, 5, 5])
        nt.assert_array_equal(df.position, [1, 2, 3, 4, 5, 6, 6])
        nt.assert_array_equal(df.time, [0, 0, 0, 0, 1, 1, 1])
        nt.assert_array_equal(df.derived_state, ["T"] * 6 + ["G"])
        nt.assert_array_equal(df.inherited_state, ["A"] * 6 + ["T"])
        nt.assert_array_equal(df.num_parents, [0] * 6 + [1])
        nt.assert_array_equal(df.num_descendants, [1] * 4 + [2] * 3)
        nt.assert_array_equal(df.num_inheritors, [1] * 4 + [2, 0, 2])


class TestEdgeDataTable:
    def test_single_tree_example(self):
        ts = single_tree_example_ts()
        tsm = model.TSModel(ts)
        df = tsm.edges_df
        assert len(df) == 6
        nt.assert_array_equal(df.left, [0, 0, 0, 0, 0, 0])
        nt.assert_array_equal(df.right, [10, 10, 10, 10, 10, 10])
        nt.assert_array_equal(df.parent, [4, 4, 5, 5, 6, 6])
        nt.assert_array_equal(df.child, [0, 1, 2, 3, 4, 5])
        nt.assert_array_equal(df.child_time, [0, 0, 0, 0, 1, 1])
        nt.assert_array_equal(df.parent_time, [1, 1, 1, 1, 2, 2])

    def test_multiple_trees_example(self):
        ts = multiple_trees_example_ts()
        tsm = model.TSModel(ts)
        df = tsm.edges_df
        assert len(df) == 6
        nt.assert_array_equal(df.left, [5, 0, 0, 0, 5, 0])
        nt.assert_array_equal(df.right, [10, 10, 5, 5, 10, 10])
        nt.assert_array_equal(df.parent, [3, 3, 3, 4, 4, 4])
        nt.assert_array_equal(df.child, [0, 1, 2, 0, 2, 3])
        nt.assert_array_equal(df.child_time, [0, 0, 0, 0, 0, 1])
        nt.assert_array_equal(df.parent_time, [1, 1, 1, 2, 2, 2])


class TestNodeDataTable:
    def test_single_tree_example(self):
        ts = single_tree_example_ts()
        tsm = model.TSModel(ts)
        df = tsm.nodes_df
        assert len(df) == 7
        nt.assert_array_equal(df.time, [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0])
        nt.assert_array_equal(df.num_mutations, [1, 1, 1, 1, 1, 1, 1])
        nt.assert_array_equal(df.ancestors_span, [10, 10, 10, 10, 10, 10, -np.inf])
        nt.assert_array_equal(df.node_flags, [1, 1, 1, 1, 0, 0, 0])

    def test_multiple_tree_example(self):
        ts = multiple_trees_example_ts()
        tsm = model.TSModel(ts)
        df = tsm.nodes_df
        assert len(df) == 5
        nt.assert_array_equal(df.time, [0.0, 0.0, 0.0, 1.0, 2.0])
        nt.assert_array_equal(df.num_mutations, [0, 0, 0, 0, 0])
        nt.assert_array_equal(df.ancestors_span, [10, 10, 10, 10, -np.inf])
        nt.assert_array_equal(df.node_flags, [1, 1, 1, 0, 0])


def compute_mutation_counts(ts):
    pop_mutation_count = np.zeros((ts.num_populations, ts.num_mutations), dtype=int)
    for pop in ts.populations():
        for tree in ts.trees(tracked_samples=ts.samples(population=pop.id)):
            for mut in tree.mutations():
                count = tree.num_tracked_samples(mut.node)
                pop_mutation_count[pop.id, mut.id] = count
    return pop_mutation_count


class TestMutationFrequencies:
    def example_ts(self):
        demography = msprime.Demography()
        demography.add_population(name="A", initial_size=10_000)
        demography.add_population(name="B", initial_size=5_000)
        demography.add_population(name="C", initial_size=1_000)
        demography.add_population_split(time=1000, derived=["A", "B"], ancestral="C")
        return msprime.sim_ancestry(
            samples={"A": 1, "B": 1},
            demography=demography,
            random_seed=12,
            sequence_length=10_000,
        )

    def check_ts(self, ts):
        C1 = compute_mutation_counts(ts)
        C2 = model.compute_population_mutation_counts(ts)
        nt.assert_array_equal(C1, C2)
        tsm = model.TSModel(ts, calc_population_frequencies=True)
        df = tsm.mutations_df
        nt.assert_array_equal(df["pop_A_freq"], C1[0] / ts.num_samples)
        nt.assert_array_equal(df["pop_B_freq"], C1[1] / ts.num_samples)
        nt.assert_array_equal(df["pop_C_freq"], C1[2] / ts.num_samples)

    def test_all_nodes(self):
        ts = self.example_ts()
        tables = ts.dump_tables()
        for u in range(ts.num_nodes - 1):
            site_id = tables.sites.add_row(u, "A")
            tables.mutations.add_row(site=site_id, node=u, derived_state="T")
        ts = tables.tree_sequence()
        self.check_ts(ts)

    @pytest.mark.parametrize("seed", range(1, 7))
    def test_simulated_mutations(self, seed):
        ts = msprime.sim_mutations(self.example_ts(), rate=1e-6, random_seed=seed)
        assert ts.num_mutations > 0
        self.check_ts(ts)

    def test_no_metadata_schema(self):
        ts = msprime.sim_mutations(self.example_ts(), rate=1e-6, random_seed=43)
        assert ts.num_mutations > 0
        tables = ts.dump_tables()
        tables.populations.metadata_schema = tskit.MetadataSchema(None)
        self.check_ts(tables.tree_sequence())

    def test_no_populations(self):
        tables = single_tree_example_ts().dump_tables()
        tables.populations.add_row(b"{}")
        tsm = model.TSModel(tables.tree_sequence(), calc_population_frequencies=True)
        with pytest.raises(ValueError, match="must be assigned to populations"):
            tsm.mutations_df


class TestNodeIsSample:
    def test_simple_example(self):
        ts = single_tree_example_ts()
        is_sample = model.node_is_sample(ts)
        for node in ts.nodes():
            assert node.is_sample() == is_sample[node.id]

    @pytest.mark.parametrize("bit", [1, 2, 17, 31])
    def test_sample_and_other_flags(self, bit):
        tables = single_tree_example_ts().dump_tables()
        flags = tables.nodes.flags
        tables.nodes.flags = flags | (1 << bit)
        ts = tables.tree_sequence()
        is_sample = model.node_is_sample(ts)
        for node in ts.nodes():
            assert node.is_sample() == is_sample[node.id]
            assert (node.flags & (1 << bit)) != 0


class TestTreesDataTable:
    def test_single_tree_example(self):
        ts = single_tree_example_ts()
        tsm = model.TSModel(ts)
        df = tsm.trees_df
        assert len(df) == 1
        nt.assert_array_equal(df.left, 0)
        nt.assert_array_equal(df.right, 10)
        nt.assert_array_equal(df.total_branch_length, 6.0)
        # nt.assert_array_equal(df.mean_internal_arity, 2.0)
        nt.assert_array_equal(df.max_internal_arity, 2.0)

    def test_single_tree_with_polytomies_example(self):
        ts = single_tree_with_polytomies_example_ts()
        tsm = model.TSModel(ts)
        df = tsm.trees_df
        assert len(df) == 1
        nt.assert_array_equal(df.left, 0)
        nt.assert_array_equal(df.right, 10)
        nt.assert_array_equal(df.total_branch_length, 16.0)
        # nt.assert_array_equal(df.mean_internal_arity, 2.75)
        nt.assert_array_equal(df.max_internal_arity, 3.0)

    def test_multi_tree_with_polytomies_example(self):
        ts = multi_tree_with_polytomies_example_ts()
        tsm = model.TSModel(ts)
        df = tsm.trees_df
        assert len(df) == 2
        nt.assert_array_equal(df.left, [0, 5])
        nt.assert_array_equal(df.right, [5, 10])
        nt.assert_array_equal(df.total_branch_length, [11.0, 12.0])
        # nt.assert_array_equal(df.mean_internal_arity, [2.25, 2.25])
        nt.assert_array_equal(df.max_internal_arity, [3.0, 3.0])


def test_cache(caplog, tmpdir):
    caplog.set_level(logging.INFO)
    ts = multiple_trees_example_ts()
    tsm = model.TSModel(ts)
    # Use the logging out put to determine cache usage
    t1 = tsm.trees_df
    t2 = tsm.trees_df
    assert t1.equals(t2)
    assert "No uuid, not caching trees_df" in caplog.text

    ts.dump(tmpdir / "cache.trees")
    ts = tskit.load(tmpdir / "cache.trees")
    tsm = model.TSModel(ts)
    # Use the logging out put to determine cache usage
    caplog.clear()
    t1 = tsm.trees_df
    assert "Calculating" in caplog.text
    caplog.clear()

    ts2 = tskit.load(tmpdir / "cache.trees")
    tsm2 = model.TSModel(ts2)
    t2 = tsm2.trees_df
    assert "Fetching" in caplog.text
    assert t1.equals(t2)

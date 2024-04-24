import dataclasses
import json
from functools import cached_property

import daiquiri
import numba
import numpy as np
import pandas as pd
import tskit

from .cache import disk_cache


logger = daiquiri.getLogger("tsqc")

spec = [
    ("num_edges", numba.int64),
    ("sequence_length", numba.float64),
    ("edges_left", numba.float64[:]),
    ("edges_right", numba.float64[:]),
    ("edge_insertion_order", numba.int32[:]),
    ("edge_removal_order", numba.int32[:]),
    ("edge_insertion_index", numba.int64),
    ("edge_removal_index", numba.int64),
    ("interval", numba.float64[:]),
    ("in_range", numba.int64[:]),
    ("out_range", numba.int64[:]),
]


@numba.experimental.jitclass(spec)
class TreePosition:
    def __init__(
        self,
        num_edges,
        sequence_length,
        edges_left,
        edges_right,
        edge_insertion_order,
        edge_removal_order,
    ):
        self.num_edges = num_edges
        self.sequence_length = sequence_length
        self.edges_left = edges_left
        self.edges_right = edges_right
        self.edge_insertion_order = edge_insertion_order
        self.edge_removal_order = edge_removal_order
        self.edge_insertion_index = 0
        self.edge_removal_index = 0
        self.interval = np.zeros(2)
        self.in_range = np.zeros(2, dtype=np.int64)
        self.out_range = np.zeros(2, dtype=np.int64)

    def next(self):  # noqa
        left = self.interval[1]
        j = self.in_range[1]
        k = self.out_range[1]
        self.in_range[0] = j
        self.out_range[0] = k
        M = self.num_edges
        edges_left = self.edges_left
        edges_right = self.edges_right
        out_order = self.edge_removal_order
        in_order = self.edge_insertion_order

        while k < M and edges_right[out_order[k]] == left:
            k += 1
        while j < M and edges_left[in_order[j]] == left:
            j += 1
        self.out_range[1] = k
        self.in_range[1] = j

        right = self.sequence_length
        if j < M:
            right = min(right, edges_left[in_order[j]])
        if k < M:
            right = min(right, edges_right[out_order[k]])
        self.interval[:] = [left, right]
        return j < M or left < self.sequence_length


# Helper function to make it easier to communicate with the numba class
def alloc_tree_position(ts):
    return TreePosition(
        num_edges=ts.num_edges,
        sequence_length=ts.sequence_length,
        edges_left=ts.edges_left,
        edges_right=ts.edges_right,
        edge_insertion_order=ts.indexes_edge_insertion_order,
        edge_removal_order=ts.indexes_edge_removal_order,
    )


@numba.njit
def _compute_per_tree_stats(
    tree_pos, num_trees, num_nodes, nodes_time, edges_parent, edges_child
):
    tbl = np.zeros(num_trees)
    num_internal_nodes = np.zeros(num_trees)
    max_arity = np.zeros(num_trees, dtype=np.int32)
    num_children = np.zeros(num_nodes, dtype=np.int32)
    nodes_with_arity = np.zeros(num_nodes, dtype=np.int32)

    current_tbl = 0
    tree_index = 0
    current_num_internal_nodes = 0
    current_max_arity = 0
    while tree_pos.next():
        for j in range(tree_pos.out_range[0], tree_pos.out_range[1]):
            e = tree_pos.edge_removal_order[j]
            p = edges_parent[e]
            nodes_with_arity[num_children[p]] -= 1
            if (
                num_children[p] == current_max_arity
                and nodes_with_arity[num_children[p]] == 1
            ):
                current_max_arity -= 1

            num_children[p] -= 1
            if num_children[p] == 0:
                current_num_internal_nodes -= 1
            else:
                nodes_with_arity[num_children[p]] += 1
            c = edges_child[e]
            branch_length = nodes_time[p] - nodes_time[c]
            current_tbl -= branch_length

        for j in range(tree_pos.in_range[0], tree_pos.in_range[1]):
            e = tree_pos.edge_insertion_order[j]
            p = edges_parent[e]
            if num_children[p] == 0:
                current_num_internal_nodes += 1
            else:
                nodes_with_arity[num_children[p]] -= 1
            num_children[p] += 1
            nodes_with_arity[num_children[p]] += 1
            if num_children[p] > current_max_arity:
                current_max_arity = num_children[p]
            c = edges_child[e]
            branch_length = nodes_time[p] - nodes_time[c]
            current_tbl += branch_length
        tbl[tree_index] = current_tbl
        num_internal_nodes[tree_index] = current_num_internal_nodes
        max_arity[tree_index] = current_max_arity
        tree_index += 1
        # print("tree", tree_index, nodes_with_arity)

    return tbl, num_internal_nodes, max_arity


def compute_per_tree_stats(ts):
    """
    Returns the per-tree statistics
    """
    tree_pos = alloc_tree_position(ts)
    return _compute_per_tree_stats(
        tree_pos,
        ts.num_trees,
        ts.num_nodes,
        ts.nodes_time,
        ts.edges_parent,
        ts.edges_child,
    )


@numba.njit
def _compute_mutation_parent_counts(mutations_parent):
    N = mutations_parent.shape[0]
    num_parents = np.zeros(N, dtype=np.int32)

    for j in range(N):
        u = j
        while mutations_parent[u] != -1:
            num_parents[j] += 1
            u = mutations_parent[u]
    return num_parents


@numba.njit
def _compute_mutation_inheritance_counts(
    tree_pos,
    num_nodes,
    num_mutations,
    edges_parent,
    edges_child,
    samples,
    mutations_position,
    mutations_node,
    mutations_parent,
):
    parent = np.zeros(num_nodes, dtype=np.int32) - 1
    num_samples = np.zeros(num_nodes, dtype=np.int32)
    num_samples[samples] = 1
    mutations_num_descendants = np.zeros(num_mutations, dtype=np.int32)
    mutations_num_inheritors = np.zeros(num_mutations, dtype=np.int32)

    mut_id = 0

    while tree_pos.next():
        for j in range(tree_pos.out_range[0], tree_pos.out_range[1]):
            e = tree_pos.edge_removal_order[j]
            c = edges_child[e]
            p = edges_parent[e]
            parent[c] = -1
            u = p
            while u != -1:
                num_samples[u] -= num_samples[c]
                u = parent[u]

        for j in range(tree_pos.in_range[0], tree_pos.in_range[1]):
            e = tree_pos.edge_insertion_order[j]
            p = edges_parent[e]
            c = edges_child[e]
            parent[c] = p
            u = p
            while u != -1:
                num_samples[u] += num_samples[c]
                u = parent[u]
        left, right = tree_pos.interval
        while mut_id < num_mutations and mutations_position[mut_id] < right:
            assert mutations_position[mut_id] >= left
            mutation_node = mutations_node[mut_id]
            descendants = num_samples[mutation_node]
            mutations_num_descendants[mut_id] = descendants
            mutations_num_inheritors[mut_id] = descendants
            # Subtract this number of descendants from the parent mutation. We are
            # guaranteed to list parents mutations before their children
            mut_parent = mutations_parent[mut_id]
            if mut_parent != -1:
                mutations_num_inheritors[mut_parent] -= descendants
            mut_id += 1

    return mutations_num_descendants, mutations_num_inheritors


@dataclasses.dataclass
class MutationCounts:
    num_parents: np.ndarray
    num_inheritors: np.ndarray
    num_descendants: np.ndarray


def compute_mutation_counts(ts):
    logger.info("Computing mutation inheritance counts")
    tree_pos = alloc_tree_position(ts)
    mutations_position = ts.sites_position[ts.mutations_site].astype(int)
    num_descendants, num_inheritors = _compute_mutation_inheritance_counts(
        tree_pos,
        ts.num_nodes,
        ts.num_mutations,
        ts.edges_parent,
        ts.edges_child,
        ts.samples(),
        mutations_position,
        ts.mutations_node,
        ts.mutations_parent,
    )
    num_parents = _compute_mutation_parent_counts(ts.mutations_parent)
    return MutationCounts(num_parents, num_inheritors, num_descendants)


@numba.njit
def _compute_population_mutation_counts(
    tree_pos,
    num_nodes,
    num_mutations,
    num_populations,
    edges_parent,
    edges_child,
    nodes_is_sample,
    nodes_population,
    mutations_position,
    mutations_node,
    mutations_parent,
):
    num_pop_samples = np.zeros((num_nodes, num_populations), dtype=np.int32)

    pop_mutation_count = np.zeros((num_populations, num_mutations), dtype=np.int32)
    parent = np.zeros(num_nodes, dtype=np.int32) - 1

    for u in range(num_nodes):
        if nodes_is_sample[u]:
            num_pop_samples[u, nodes_population[u]] = 1

    mut_id = 0
    while tree_pos.next():
        for j in range(tree_pos.out_range[0], tree_pos.out_range[1]):
            e = tree_pos.edge_removal_order[j]
            c = edges_child[e]
            p = edges_parent[e]
            parent[c] = -1
            u = p
            while u != -1:
                for k in range(num_populations):
                    num_pop_samples[u, k] -= num_pop_samples[c, k]
                u = parent[u]

        for j in range(tree_pos.in_range[0], tree_pos.in_range[1]):
            e = tree_pos.edge_insertion_order[j]
            p = edges_parent[e]
            c = edges_child[e]
            parent[c] = p
            u = p
            while u != -1:
                for k in range(num_populations):
                    num_pop_samples[u, k] += num_pop_samples[c, k]
                u = parent[u]

        left, right = tree_pos.interval
        while mut_id < num_mutations and mutations_position[mut_id] < right:
            assert mutations_position[mut_id] >= left
            mutation_node = mutations_node[mut_id]
            for pop in range(num_populations):
                pop_mutation_count[pop, mut_id] = num_pop_samples[mutation_node, pop]
            mut_id += 1

    return pop_mutation_count


def node_is_sample(ts):
    sample_flag = np.full_like(ts.nodes_flags, tskit.NODE_IS_SAMPLE)
    return np.bitwise_and(ts.nodes_flags, sample_flag) != 0


def compute_population_mutation_counts(ts):
    """
    Return a (num_populations, num_mutations) array that gives the frequency
    of each mutation in each of the populations in the specified tree sequence.
    """
    logger.info(
        f"Computing mutation frequencies within {ts.num_populations} populations"
    )
    mutations_position = ts.sites_position[ts.mutations_site].astype(int)

    if np.any(ts.nodes_population[ts.samples()] == -1):
        raise ValueError("Sample nodes must be assigned to populations")

    return _compute_population_mutation_counts(
        alloc_tree_position(ts),
        ts.num_nodes,
        ts.num_mutations,
        ts.num_populations,
        ts.edges_parent,
        ts.edges_child,
        node_is_sample(ts),
        ts.nodes_population,
        mutations_position,
        ts.mutations_node,
        ts.mutations_parent,
    )


class TSModel:
    """
    A wrapper around a tskit.TreeSequence object that provides some
    convenience methods for analysing the tree sequence.
    """

    def __init__(self, ts, calc_population_frequencies=False, name=None):
        self.ts = ts
        self.name = name

        self.sites_num_mutations = np.bincount(
            self.ts.mutations_site, minlength=self.ts.num_sites
        )
        self.nodes_num_mutations = np.bincount(
            self.ts.mutations_node, minlength=self.ts.num_nodes
        )
        self.calc_population_frequencies = calc_population_frequencies

    @property
    def file_uuid(self):
        return self.ts.file_uuid

    @cached_property
    @disk_cache("v1")
    def summary_df(self):
        nodes_with_zero_muts = np.sum(self.nodes_num_mutations == 0)
        sites_with_zero_muts = np.sum(self.sites_num_mutations == 0)

        data = [
            ("samples", self.ts.num_samples),
            ("nodes", self.ts.num_nodes),
            ("mutations", self.ts.num_mutations),
            ("nodes_with_zero_muts", nodes_with_zero_muts),
            ("sites_with_zero_muts", sites_with_zero_muts),
            ("max_mutations_per_site", np.max(self.sites_num_mutations)),
            ("mean_mutations_per_site", np.mean(self.sites_num_mutations)),
            ("median_mutations_per_site", np.median(self.sites_num_mutations)),
            ("max_mutations_per_node", np.max(self.nodes_num_mutations)),
        ]
        df = pd.DataFrame(
            {"property": [d[0] for d in data], "value": [d[1] for d in data]}
        )
        logger.info("Computed summary dataframe")
        return df.set_index("property")

    def _repr_html_(self):
        return self.summary_df._repr_html_()

    @staticmethod
    @numba.njit
    def child_bounds(num_nodes, edges_left, edges_right, edges_child):
        num_edges = edges_left.shape[0]
        child_left = np.zeros(num_nodes, dtype=np.float64) + np.inf
        child_right = np.zeros(num_nodes, dtype=np.float64)

        for e in range(num_edges):
            u = edges_child[e]
            if edges_left[e] < child_left[u]:
                child_left[u] = edges_left[e]
            if edges_right[e] > child_right[u]:
                child_right[u] = edges_right[e]
        return child_left, child_right

    @cached_property
    @disk_cache("v2")
    def mutations_df(self):
        # FIXME use tskit's impute mutations time
        ts = self.ts
        mutations_time = ts.mutations_time.copy()
        mutations_node = ts.mutations_node.copy()
        unknown = tskit.is_unknown_time(mutations_time)
        mutations_time[unknown] = self.ts.nodes_time[mutations_node[unknown]]

        position = ts.sites_position[ts.mutations_site]

        tables = self.ts.tables
        derived_state = tables.mutations.derived_state
        offsets = tables.mutations.derived_state_offset
        derived_state = np.array(
            [
                derived_state[s].tobytes().decode("utf-8")
                for s in (
                    slice(start, end) for start, end in zip(offsets[:-1], offsets[1:])
                )
            ]
        )
        ancestral_state = tables.sites.ancestral_state
        offsets = tables.sites.ancestral_state_offset
        ancestral_state = np.array(
            [
                ancestral_state[s].tobytes().decode("utf-8")
                for s in (
                    slice(start, end) for start, end in zip(offsets[:-1], offsets[1:])
                )
            ]
        )
        del tables
        inherited_state = ancestral_state[ts.mutations_site]
        mutations_with_parent = ts.mutations_parent != -1

        parent = ts.mutations_parent[mutations_with_parent]
        assert np.all(parent >= 0)
        inherited_state[mutations_with_parent] = derived_state[parent]
        self.mutations_derived_state = derived_state
        self.mutations_inherited_state = inherited_state

        population_data = {}
        if ts.num_populations > 0 and self.calc_population_frequencies:
            pop_mutation_count = compute_population_mutation_counts(ts)
            for pop in ts.populations():
                name = f"pop{pop.id}"
                if isinstance(pop.metadata, bytes):
                    metadata_dict = json.loads(pop.metadata.decode("utf-8"))
                else:
                    metadata_dict = pop.metadata
                if "name" in metadata_dict:
                    name = metadata_dict["name"]
                col_name = f"pop_{name}_freq"
                population_data[col_name] = pop_mutation_count[pop.id] / ts.num_samples

        counts = compute_mutation_counts(ts)
        df = pd.DataFrame(
            {
                "id": np.arange(ts.num_mutations),
                "position": position,
                "node": ts.mutations_node,
                "time": mutations_time,
                "derived_state": self.mutations_derived_state,
                "inherited_state": self.mutations_inherited_state,
                "num_descendants": counts.num_descendants,
                "num_inheritors": counts.num_inheritors,
                "num_parents": counts.num_parents,
                **population_data,
            }
        )
        logger.info("Computed mutations dataframe")
        return df.astype(
            {
                "id": "int",
                "position": "float64",
                "node": "int",
                "time": "float64",
                "derived_state": "str",
                "inherited_state": "str",
                "num_descendants": "int",
                "num_inheritors": "int",
                "num_parents": "int",
            }
        )

    @cached_property
    @disk_cache("v1")
    def edges_df(self):
        ts = self.ts
        left = ts.edges_left
        right = ts.edges_right
        edges_parent = ts.edges_parent
        edges_child = ts.edges_child
        nodes_time = ts.nodes_time
        parent_time = nodes_time[edges_parent]
        child_time = nodes_time[edges_child]
        branch_length = parent_time - child_time
        span = right - left

        df = pd.DataFrame(
            {
                "left": left,
                "right": right,
                "parent": edges_parent,
                "child": edges_child,
                "parent_time": parent_time,
                "child_time": child_time,
                "branch_length": branch_length,
                "span": span,
            }
        )

        logger.info("Computed edges dataframe")
        return df.astype(
            {
                "left": "float64",
                "right": "float64",
                "parent": "int",
                "child": "int",
                "parent_time": "float64",
                "child_time": "float64",
                "branch_length": "float64",
                "span": "float64",
            }
        )

    @cached_property
    @disk_cache("v2")
    def nodes_df(self):
        ts = self.ts
        child_left, child_right = self.child_bounds(
            ts.num_nodes, ts.edges_left, ts.edges_right, ts.edges_child
        )
        df = pd.DataFrame(
            {
                "time": ts.nodes_time,
                "num_mutations": self.nodes_num_mutations,
                "ancestors_span": child_right - child_left,
                "node_flags": ts.nodes_flags,
            }
        )
        logger.info("Computed nodes dataframe")
        return df.astype(
            {
                "time": "float64",
                "num_mutations": "int",
                "ancestors_span": "float64",
                "node_flags": "int",
            }
        )

    @cached_property
    @disk_cache("v1")
    def trees_df(self):
        ts = self.ts
        num_trees = ts.num_trees

        total_branch_length, num_internal_nodes, max_arity = compute_per_tree_stats(ts)

        # FIXME - need to add this to the computation above
        mean_internal_arity = np.zeros(num_trees)

        site_tree_index = self.calc_site_tree_index()
        unique_values, counts = np.unique(site_tree_index, return_counts=True)
        sites_per_tree = np.zeros(ts.num_trees, dtype=np.int64)
        sites_per_tree[unique_values] = counts
        breakpoints = ts.breakpoints(as_array=True)
        df = pd.DataFrame(
            {
                "left": breakpoints[:-1],
                "right": breakpoints[1:],
                "total_branch_length": total_branch_length,
                "mean_internal_arity": mean_internal_arity,
                "max_internal_arity": max_arity,
                "num_sites": sites_per_tree,
                "num_mutations": self.calc_mutations_per_tree(),
            }
        )

        logger.info("Computed trees dataframe")
        return df.astype(
            {
                "left": "int",
                "right": "int",
                "total_branch_length": "float64",
                "mean_internal_arity": "float64",
                "max_internal_arity": "float64",
                "num_sites": "int",
                "num_mutations": "int",
            }
        )

    def genes_df(self, genes_file):
        genes_df = pd.read_csv(genes_file, sep=";")
        # TODO file checks!
        genes_df.columns = ["chr", "position", "end", "strand", "id", "name"]
        genes_df = genes_df[
            (genes_df["position"] >= self.ts.first().interval.left)
            & (genes_df["end"] <= self.ts.last().interval.right)
        ]
        logger.info("Computed genes dataframe")
        return genes_df

    def calc_polytomy_fractions(self):
        """
        Calculates the fraction of polytomies for each tree in the
        tree sequence
        """
        assert self.ts.num_samples > 2
        polytomy_fractions = []
        for tree in self.ts.trees():
            if tree.num_edges == 0:
                polytomy_fractions.append(None)
            else:
                polytomy_fractions.append(
                    float(
                        (tree.num_edges - self.ts.num_samples)
                        / (self.ts.num_samples - 2)
                    )
                )
        return polytomy_fractions

    def map_stats_to_genome(self, to_map):
        """
        Converts a list of tree-based stats to genomic coordinates
        """
        mapped = np.zeros(int(self.ts.sequence_length))
        for i, tree in enumerate(self.ts.trees()):
            left, right = map(int, tree.interval)
            mapped[left:right] = to_map[i]
        return mapped

    def make_sliding_windows(self, iterable, size, overlap=0):
        start = 0
        assert overlap < size, "overlap must be smaller then window size"
        end = size
        step = size - overlap

        length = len(iterable)
        while end < length:
            yield iterable[start:end]
            start += step
            end += step
        yield iterable[start:]

    def calc_mean_node_arity(self):
        span_sums = np.bincount(
            self.ts.edges_parent,
            weights=self.ts.edges_right - self.ts.edges_left,
            minlength=self.ts.num_nodes,
        )
        node_spans = self.ts.sample_count_stat(
            [self.ts.samples()],
            lambda x: (x > 0),
            1,
            polarised=True,
            span_normalise=False,
            strict=False,
            mode="node",
        )[:, 0]
        return span_sums / node_spans

    def calc_site_tree_index(self):
        return (
            np.searchsorted(
                self.ts.breakpoints(as_array=True), self.ts.sites_position, side="right"
            )
            - 1
        )

    def calc_sites_per_tree(self):
        site_tree_index = self.calc_site_tree_index()
        unique_values, counts = np.unique(site_tree_index, return_counts=True)
        sites_per_tree = np.zeros(self.ts.num_trees, dtype=np.int64)
        sites_per_tree[unique_values] = counts
        return sites_per_tree

    def calc_mutations_per_tree(self):
        site_tree_index = self.calc_site_tree_index()
        mutation_tree_index = site_tree_index[self.ts.mutations_site]
        unique_values, counts = np.unique(mutation_tree_index, return_counts=True)
        mutations_per_tree = np.zeros(self.ts.num_trees, dtype=np.int64)
        mutations_per_tree[unique_values] = counts
        return mutations_per_tree

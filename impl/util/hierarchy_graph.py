from typing import Optional
import networkx as nx
import util
import impl.util.nlp as nlp_util
import impl.util.hypernymy as hypernymy_util
from impl.util.base_graph import BaseGraph
from collections import defaultdict
import copy


class HierarchyGraph(BaseGraph):
    """An extension of the graph with methods to add, remove, and merge nodes.

    Existing nodes can be merged into a new node. The existing nodes then become its 'parts'.
    """

    # initialisations
    def __init__(self, graph: nx.DiGraph, root_node: str = None):
        super().__init__(graph, root_node)
        self._node_by_name = None
        self._nodes_by_part = defaultdict(set)

    # node attribute definitions
    ATTRIBUTE_NAME = 'attribute_name'
    ATTRIBUTE_PARTS = 'attribute_parts'

    def copy(self):
        new_self = super().copy()
        new_self._nodes_by_part = copy.deepcopy(self._nodes_by_part)
        return new_self

    def _check_node_exists(self, node: str):
        if not self.has_node(node):
            raise Exception(f'Node {node} not in graph.')

    def _reset_node_indices(self):
        self._node_by_name = None

    def get_name(self, node: str) -> str:
        self._check_node_exists(node)
        return self._get_attr(node, self.ATTRIBUTE_NAME)

    def _set_name(self, node: str, name: str):
        self._check_node_exists(node)
        self._set_attr(node, self.ATTRIBUTE_NAME, name)
        self._node_by_name = None  # reset name-to-node index due to changes

    def get_node_by_name(self, name: str) -> Optional[str]:
        if self._node_by_name is None:  # initialise name-to-node index if not existing
            self._node_by_name = {self.get_name(node): node for node in self.nodes}
        return self._node_by_name[name] if name in self._node_by_name else None

    # graph connectivity

    def remove_unconnected(self):
        """Remove all nodes that are not connected to the root node."""
        valid_nodes = {self.root_node} | self.descendants(self.root_node)
        self._remove_all_nodes_except(valid_nodes)
        return self

    def append_unconnected(self):
        """Make all unconnected nodes children of the root node."""
        unconnected_root_nodes = {node for node in self.content_nodes if not self.parents(node)}
        self._add_edges([(self.root_node, node) for node in unconnected_root_nodes])
        return self

    def resolve_cycles(self):
        """Resolve cycles by removing cycle edges that point from a node with a higher depth to a node with a lower depth."""
        util.get_logger().debug('HierarchyGraph: Looking for cycles to resolve..')
        num_edges = len(self.edges)
        # remove all edges N1-->N2 of a cycle with depth(N1) > depth(N2)
        self._remove_cycle_edges_by_node_depth(lambda x, y: x > y)
        # remove all edges N1-->N2 of a cycle with depth(N1) >= depth(N2)
        self._remove_cycle_edges_by_node_depth(lambda x, y: x >= y)
        util.get_logger().debug(f'HierarchyGraph: Removed {num_edges - len(self.edges)} to resolve cycles.')
        return self

    def _remove_cycle_edges_by_node_depth(self, comparator):
        edges_to_remove = set()
        for cycle in nx.simple_cycles(self.graph):
            node_depths = {node: self.depth(node) for node in cycle}
            for i in range(len(cycle)):
                current_edge = (cycle[i], cycle[(i+1) % len(cycle)])
                if comparator(node_depths[current_edge[0]], node_depths[current_edge[1]]):
                    edges_to_remove.add(current_edge)
        self._remove_edges(edges_to_remove)

    # semantic connectivity

    def remove_unrelated_edges(self):
        """Remove edges that connect nodes which have head nouns that are neither synonyms nor hypernyms."""
        node_names = [self.get_name(node) for node in self.nodes]
        headlemmas = dict(zip(self.nodes, nlp_util.get_head_lemmas(node_names)))
        valid_edges = {(p, c) for p, c in self.edges if self._is_hierarchical_edge(headlemmas[p], headlemmas[c])}
        self._remove_all_edges_except(valid_edges)
        return self

    def _is_hierarchical_edge(self, parent_lemmas: set, child_lemmas: set) -> bool:
        return any(hypernymy_util.is_hypernym(pl, cl) for pl in parent_lemmas for cl in child_lemmas)

    # compound nodes

    def get_nodes_for_part(self, part: str) -> set:
        # be sure not to return outdated nodes that are not in the graph anymore
        return {n for n in self._nodes_by_part[part] if self.has_node(n)}

    def get_parts(self, node: str) -> set:
        self._check_node_exists(node)
        return self._get_attr(node, self.ATTRIBUTE_PARTS)

    def _set_parts(self, node: str, parts: set):
        self._check_node_exists(node)
        self._set_attr(node, self.ATTRIBUTE_PARTS, parts)
        for part in parts:
            self._nodes_by_part[part].add(node)

    def merge_nodes(self):
        """Merges any two nodes that have the same canonical name.

        A canonical name of a node is its name without any postfixes that Wikipedia appends for organisational purposes.
        E.g., we remove by-phrases like in "Authors by name", and we remove alphabetical splits like in "Authors: A-C".
        """
        nodes_containing_by = {node for node in self.nodes if '_by_' in node}
        nodes_canonical_names = {}
        for node in nodes_containing_by:
            node_name = self.get_name(node)
            canonical_name = nlp_util.get_canonical_name(node_name)
            if node_name != canonical_name:
                nodes_canonical_names[node] = canonical_name
        remaining_nodes_to_merge = set(nodes_canonical_names)
        util.get_logger().debug(f'Found {len(remaining_nodes_to_merge)} nodes to merge.')

        # 1) compute direct merge and synonym merge
        direct_merges = defaultdict(set)

        nodes_important_words = {node: nlp_util.without_stopwords(canonical_name) for node, canonical_name in nodes_canonical_names.items()}
        for node in remaining_nodes_to_merge:
            node_important_words = nodes_important_words[node]
            for parent in self.parents(node):
                if parent not in nodes_important_words:
                    nodes_important_words[parent] = nlp_util.without_stopwords(nlp_util.get_canonical_name(self.get_name(parent)))
                parent_important_words = nodes_important_words[parent]

                if all(any(hypernymy_util.is_synonym(niw, piw) for piw in parent_important_words) for niw in node_important_words):
                    direct_merges[node].add(parent)
        util.get_logger().debug(f'Found {len(direct_merges)} nodes to merge directly.')

        # 2) compute category set merge
        catset_merges = defaultdict(set)
        remaining_nodes_to_merge = remaining_nodes_to_merge.difference(set(direct_merges))
        for node in remaining_nodes_to_merge:
            node_canonical_name = nodes_canonical_names[node]
            for parent in self.parents(node):
                if parent == self.root_node:
                    continue
                similar_children_count = len({child for child in self.children(parent) if child in nodes_canonical_names and nodes_canonical_names[child] == node_canonical_name})
                if similar_children_count > 1:
                    catset_merges[node].add(parent)
        util.get_logger().debug(f'Found {len(catset_merges)} nodes to merge via category sets.')

        remaining_nodes_to_merge = remaining_nodes_to_merge.difference(set(catset_merges))
        util.get_logger().debug(f'The {len(remaining_nodes_to_merge)} remaining nodes will not be merged.')

        # 3) conduct merge
        nodes_to_merge = set(direct_merges) | set(catset_merges)
        all_merges = {node: direct_merges[node] | catset_merges[node] for node in nodes_to_merge}

        iteration = 0
        while all_merges:
            independent_nodes = set(all_merges).difference({merge_target for mts in all_merges.values() for merge_target in mts})
            util.get_logger().debug(f'Merge iteration {iteration}: Merging {len(independent_nodes)} of remaining {len(all_merges)} nodes.')
            for node_to_merge in independent_nodes:
                merge_targets = all_merges[node_to_merge]
                del all_merges[node_to_merge]

                for mt in merge_targets:
                    self._set_parts(mt, self.get_parts(mt) | self.get_parts(node_to_merge))

                parents = self.parents(node_to_merge)
                children = self.children(node_to_merge)
                edges_to_add = {(p, c) for p in parents for c in children}
                self._add_edges(edges_to_add)
                self._remove_nodes({node_to_merge})
            iteration += 1

        return self

    def remove_transitive_edges(self):
        """Removes all transitive edges from the graph."""
        self._remove_all_edges_except(set(nx.transitive_reduction(self.graph).edges))
        return self

from typing import Set
import networkx as nx
from utils import get_logger
import impl.util.nlp as nlp_util
import impl.util.hypernymy as hypernymy_util
from impl.util.base_graph import BaseGraph
from collections import defaultdict
import copy
from impl.util.rdf import RdfResource


class HierarchyGraph(BaseGraph):
    """An extension of the graph with methods to add, remove, and merge nodes.

    Existing nodes can be merged into a new node. The existing nodes then become its 'parts'.
    """

    # initialisations
    def __init__(self, graph: nx.DiGraph, root_node: str = None):
        super().__init__(graph, root_node)
        self._nodes_by_part = defaultdict(set)

    # node attribute definitions
    ATTRIBUTE_LABEL = 'attribute_label'
    ATTRIBUTE_PARTS = 'attribute_parts'

    def copy(self):
        new_self = super().copy()
        new_self._nodes_by_part = copy.deepcopy(self._nodes_by_part)
        return new_self

    def _check_node_exists(self, node: str):
        if not self.has_node(node):
            raise Exception(f'Node {node} not in graph.')

    def get_label(self, node: str) -> str:
        self._check_node_exists(node)
        return self._get_attr(node, self.ATTRIBUTE_LABEL)

    def _set_label(self, node: str, label: str):
        self._check_node_exists(node)
        self._set_attr(node, self.ATTRIBUTE_LABEL, label)

    def get_node_LHS(self) -> dict:
        nodes = list(self.content_nodes)
        node_labels = [self.get_label(n) for n in nodes]
        node_LHS = dict(zip(nodes, nlp_util.get_lexhead_subjects(node_labels)))
        return defaultdict(set, node_LHS)

    def get_node_LH(self) -> dict:
        nodes = list(self.content_nodes)
        node_labels = [self.get_label(n) for n in nodes]
        node_LH = dict(zip(nodes, nlp_util.get_lexhead_remainder(node_labels)))
        return defaultdict(set, node_LH)

    def get_node_NH(self) -> dict:
        nodes = list(self.content_nodes)
        node_labels = [self.get_label(n) for n in nodes]
        node_NH = dict(zip(nodes, nlp_util.get_nonlexhead_part(node_labels)))
        return defaultdict(set, node_NH)

    # graph connectivity

    def append_unconnected(self, aggressive=True):
        """Make all unconnected nodes children of the next closest parent node or the root node.

        For example, the category 'Israeli speculative fiction writers' has no parents and should be connected to other nodes.
        We look for similar nodes based on their lexical head, i.e. we connect it to 'Israeli writers' and 'Speculative fiction writers'.
        """
        self._resolve_cycles()  # make sure the graph is cycle-free before making any changes to the hierarchy

        if aggressive:
            # attach node to closest node by lexical head
            unconnected_head_nodes = {node for node in self.content_nodes if not self.parents(node)}
            nodes_to_parents_mapping = self.find_parents_by_headlemma_match(unconnected_head_nodes, self)
            nodes_to_parents_mapping = {n: parents.difference(self.descendants(n)) for n, parents in nodes_to_parents_mapping.items()}
            self._add_edges([(parent, node) for node, parents in nodes_to_parents_mapping.items() for parent in parents])

        # set root_node as parent for nodes without any valid parents
        remaining_unconnected_root_nodes = {node for node in self.content_nodes if not self.parents(node)}
        self._add_edges([(self.root_node, node) for node in remaining_unconnected_root_nodes])

        return self

    def find_parents_by_headlemma_match(self, unconnected_nodes: set, source_graph) -> dict:
        """For every node in unconnected_nodes, find a set of parents that matches the lexical head best."""
        connected_nodes = self.descendants(self.root_node)
        target_LHS = self.get_node_LHS()
        target_LH = self.get_node_LH()
        target_NH = self.get_node_NH()
        if source_graph == self:
            source_LHS = target_LHS
            source_LH = target_LH
            source_NH = target_NH
        else:
            source_LHS = source_graph.get_node_LHS()
            source_LH = source_graph.get_node_LH()
            source_NH = source_graph.get_node_NH()
        # compute mapping from head lemmas to nodes in graph
        lhs_to_node_mapping = defaultdict(set)
        for graph_node, lhs in target_LHS.items():
            for lemma in lhs:
                lhs_to_node_mapping[lemma].add(graph_node)

        nodes_to_parents_mapping = {}
        for node in unconnected_nodes:
            LHS = source_LHS[node]
            NH = source_NH[node]
            exact_parent_node_candidates = {n for subject_lemma in LHS for n in lhs_to_node_mapping[subject_lemma] if NH == target_NH[n]}.intersection(connected_nodes)
            best_parents = self._find_parents_for_node_in_candidates(source_LH[node], exact_parent_node_candidates, target_LH)
            if not best_parents and len(NH) > 0:
                # if we do not find any best parents, we try to find candidates without NH
                nonhead_parent_node_candidates = {n for subject_lemma in LHS for n in lhs_to_node_mapping[subject_lemma] if len(target_NH[n]) == 0}.intersection(connected_nodes)
                best_parents = self._find_parents_for_node_in_candidates(source_LH[node], nonhead_parent_node_candidates, target_LH)
            nodes_to_parents_mapping[node] = best_parents

        return nodes_to_parents_mapping

    @staticmethod
    def _find_parents_for_node_in_candidates(node_LH, candidates, target_LH):
        # get rid of candidates that contain information which is not contained in node
        candidates = {cand for cand in candidates if not target_LH[cand].difference(node_LH)}
        # find candidates that have the highest overlap in lexical head
        lemma_matches = {cand: len(target_LH[cand].intersection(node_LH)) for cand in candidates}
        highest_match_score = max(lemma_matches.values(), default=0)
        if highest_match_score > 0:
            # select the nodes that have the best matching lexical head
            return {cand for cand, score in lemma_matches.items() if score == highest_match_score}
        else:
            # if no related nodes are found, use the most generic node (if available)
            return {cand for cand in candidates if len(target_LH[cand]) == 0}

    def _resolve_cycles(self):
        """Resolve cycles by removing cycle edges that point from a node with a higher depth to a node with a lower depth."""
        num_edges = len(self.edges)
        # remove all edges N1-->N2 of a cycle with depth(N1) > depth(N2)
        self._remove_cycle_edges_by_node_depth(lambda x, y: x > y)
        # remove all edges N1-->N2 of a cycle with depth(N1) >= depth(N2)
        self._remove_cycle_edges_by_node_depth(lambda x, y: x >= y)
        get_logger().debug(f'Removed {num_edges - len(self.edges)} edges to resolve cycles.')
        return self

    def _remove_cycle_edges_by_node_depth(self, comparator):
        edges_to_remove = set()
        node_depths = self.depths()
        for cycle in nx.simple_cycles(self.graph):
            for i in range(len(cycle)):
                current_edge = (cycle[i], cycle[(i+1) % len(cycle)])
                if comparator(node_depths[current_edge[0]], node_depths[current_edge[1]]):
                    edges_to_remove.add(current_edge)
        self._remove_edges(edges_to_remove)

    # semantic connectivity

    def remove_unrelated_edges(self):
        """Remove edges that connect nodes which have head nouns that are neither synonyms nor hypernyms."""
        lexhead_subjects = self.get_node_LHS()
        valid_edges = {(p, c) for p, c in self.edges if self._is_hierarchical_edge(lexhead_subjects[p], lexhead_subjects[c])}
        self._remove_all_edges_except(valid_edges)
        self.append_unconnected()
        return self

    @staticmethod
    def _is_hierarchical_edge(parent_lemmas: set, child_lemmas: set) -> bool:
        return any(hypernymy_util.is_hypernym(pl, cl) for pl in parent_lemmas for cl in child_lemmas)

    # compound nodes

    def get_nodes_for_part(self, part: RdfResource) -> Set[str]:
        # be sure not to return outdated nodes that are not in the graph anymore
        return {n for n in self._nodes_by_part[part] if self.has_node(n)}

    def get_parts(self, node: str) -> Set[RdfResource]:
        self._check_node_exists(node)
        return self._get_attr(node, self.ATTRIBUTE_PARTS)

    def _set_parts(self, node: str, parts: Set[RdfResource]):
        self._check_node_exists(node)
        self._set_attr(node, self.ATTRIBUTE_PARTS, parts)
        for part in parts:
            self._nodes_by_part[part].add(node)

    def merge_nodes(self):
        """Merges any two nodes that have the same canonical label.

        A canonical label of a node is its label without any postfixes that Wikipedia appends for organisational purposes.
        E.g., we remove by-phrases like in "Authors by name", and we remove alphabetical splits like in "Authors: A-C".
        """
        get_logger().debug('Merging nodes with the same label..')
        canonical_labels = {}
        for node in self.content_nodes:
            node_label = self.get_label(node)
            canonical_label = nlp_util.get_canonical_label(node_label)
            if node_label != canonical_label:
                canonical_labels[node] = canonical_label
        remaining_nodes_to_merge = set(canonical_labels)
        get_logger().debug(f'Found {len(remaining_nodes_to_merge)} nodes to merge.')

        # 1) compute direct merge and synonym merge
        direct_merges = defaultdict(set)
        important_words = {node: nlp_util.without_stopwords(canonical_label) for node, canonical_label in canonical_labels.items()}
        for node in remaining_nodes_to_merge:
            # first compute important words for all parents
            for parent in self.parents(node):
                if parent not in important_words:
                    important_words[parent] = nlp_util.without_stopwords(nlp_util.get_canonical_label(self.get_label(parent)))
            # find exact matches
            exact_matches = {p for p in self.parents(node) if important_words[node] == important_words[p]}
            if exact_matches:
                direct_merges[node] = exact_matches
            else:
                # find synonym matches if no exact matches
                for parent in self.parents(node):
                    if all(any(hypernymy_util.is_synonym(niw, piw) for piw in important_words[parent]) for niw in important_words[node]):
                        direct_merges[node].add(parent)
        get_logger().debug(f'Merging {len(direct_merges)} nodes directly.')

        # 2) compute category set merge
        catset_merges = defaultdict(set)
        remaining_nodes_to_merge = remaining_nodes_to_merge.difference(set(direct_merges))
        for node in remaining_nodes_to_merge:
            for parent in self.parents(node).difference({self.root_node}):
                similar_children_count = len({child for child in self.children(parent) if child in canonical_labels and canonical_labels[child] == canonical_labels[node]})
                if similar_children_count > 1:
                    catset_merges[node].add(parent)
        get_logger().debug(f'Merging {len(catset_merges)} nodes via category sets.')

        remaining_nodes_to_merge = remaining_nodes_to_merge.difference(set(catset_merges))
        get_logger().debug(f'The {len(remaining_nodes_to_merge)} remaining nodes will not be merged.')

        # 3) conduct merge
        nodes_to_merge = set(direct_merges) | set(catset_merges)
        all_merges = {node: direct_merges[node] | catset_merges[node] for node in nodes_to_merge}

        while all_merges:
            independent_nodes = set(all_merges).difference({merge_target for mts in all_merges.values() for merge_target in mts})
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

        return self

    def remove_transitive_edges(self):
        """Removes all transitive edges from the graph."""
        self._remove_all_edges_except(set(nx.transitive_reduction(self.graph).edges))
        return self

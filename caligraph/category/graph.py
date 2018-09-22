import networkx as nx
from . import store as cat_store
from . import nlp as cat_nlp
import caligraph.util.nlp as nlp_util
import util


class CategoryGraph:
    def __init__(self, graph: nx.DiGraph, root_node: str):
        self.graph = graph
        self.root_node = root_node

    @property
    def statistics(self) -> str:
        return '\n'.join([
            '{:^40}'.format('CATEGORY GRAPH'),
            '=' * 40,
            '{:>18} | {:>19}'.format('nodes', self.graph.number_of_nodes()),
            '{:>18} | {:>19}'.format('edges', self.graph.number_of_edges()),
            '{:>18} | {:>19}'.format('in-degree', self.graph.in_degree),
            '{:>18} | {:>19}'.format('out-degree', self.graph.out_degree)
        ])

    def predecessors(self, node: str) -> set:
        return set(self.graph.predecessors(node))

    def successors(self, node: str) -> set:
        return set(self.graph.successors(node))

    def remove_unconnected(self):
        valid_nodes = set(nx.bfs_tree(self.graph, self.root_node))
        self._remove_all_nodes_except(valid_nodes)
        return self

    def append_unconnected(self):
        unconnected_root_nodes = {node for node in self.graph.nodes if len(self.predecessors(node)) == 0 and node != self.root_node}
        self.graph.add_edges_from([(self.root_node, node) for node in unconnected_root_nodes])
        return self

    def make_conceptual(self):
        categories = set(self.graph.nodes)
        # filtering maintenance categories
        categories = categories.difference(cat_store.get_maintenance_cats())
        # filtering administrative categories
        categories = {cat for cat in categories if not cat.endswith(('templates', 'navigational boxes'))}
        # filtering non-conceptual categories
        categories = {cat for cat in categories if cat_nlp.is_conceptual(cat)}
        # persisting spacy cache so that parsed categories are cached
        nlp_util.persist_cache()

        self._remove_all_nodes_except(categories | {self.root_node})
        return self

    def _remove_all_nodes_except(self, valid_nodes: set):
        invalid_nodes = set(self.graph.nodes).difference(valid_nodes)
        self.graph.remove_nodes_from(invalid_nodes)

    @classmethod
    def create_from_dbpedia(cls, root_node=None):
        edges = [(node, child) for node in cat_store.get_all_cats() for child in cat_store.get_children(node) if node != child]
        root_node = root_node if root_node else util.get_config('caligraph.category.root_node')
        return CategoryGraph(nx.DiGraph(incoming_graph_data=edges), root_node)

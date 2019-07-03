from .graph import CategoryGraph
import util


# TODO: make sure that there are no cycles in wikitaxonomy graph


def get_conceptual_category_graph() -> CategoryGraph:
    global __CONCEPTUAL_CATEGORY_GRAPH__
    if '__CONCEPTUAL_CATEGORY_GRAPH__' not in globals():
        initializer = lambda: CategoryGraph.create_from_dbpedia().remove_unconnected().make_conceptual()
        __CONCEPTUAL_CATEGORY_GRAPH__ = util.load_or_create_cache('catgraph_conceptual', initializer)
    return __CONCEPTUAL_CATEGORY_GRAPH__


def get_cycle_free_category_graph() -> CategoryGraph:
    global __CYCLEFREE_CATEGORY_GRAPH__
    if '__CYCLEFREE_CATEGORY_GRAPH__' not in globals():
        initializer = lambda: get_conceptual_category_graph().resolve_cycles()
        __CYCLEFREE_CATEGORY_GRAPH__ = util.load_or_create_cache('catgraph_cyclefree', initializer)
    return __CYCLEFREE_CATEGORY_GRAPH__


def get_wikitaxonomy_graph() -> CategoryGraph:
    global __WIKITAXONOMY_CATEGORY_GRAPH__
    if '__WIKITAXONOMY_CATEGORY_GRAPH__' not in globals():
        __WIKITAXONOMY_CATEGORY_GRAPH__ = util.load_or_create_cache('catgraph_wikitaxonomy', CategoryGraph.create_from_wikitaxonomy)
    return __WIKITAXONOMY_CATEGORY_GRAPH__

import traceback
import utils
from utils import get_logger
import mailer
from impl import wikipedia
from impl.dbpedia.category import DbpCategoryStore
import impl.category.cat2ax as cat_axioms
import impl.util.hypernymy as hypernymy_util
from impl import category
from impl import caligraph


def _setup_hypernyms():
    """Initialisation of hypernyms that are extracted from Wikipedia categories using Cat2Ax axioms."""
    if utils.load_cache('wikitaxonomy_hypernyms') is not None:
        return  # only compute hypernyms if they are not existing already
    dbc = DbpCategoryStore.instance()
    ccg = category.get_conceptual_category_graph()
    valid_categories = {dbc.get_category_by_name(node) for node in ccg.content_nodes}
    # initialise cat2ax axioms
    cat2ax_axioms = cat_axioms.extract_category_axioms(valid_categories)
    utils.update_cache('cat2ax_axioms', cat2ax_axioms)
    # initialise wikitaxonomy hypernyms
    wikitaxonomy_hypernyms = hypernymy_util.compute_hypernyms(ccg)
    utils.update_cache('wikitaxonomy_hypernyms', wikitaxonomy_hypernyms)


if __name__ == '__main__':
    try:
        get_logger().info('Starting extraction of CaLiGraph..')

        # prepare resources like type lexicalisations from hearst patterns and wikitaxonomy hypernyms
        get_logger().info('Preparing resources..')
        wikipedia.extract_wiki_corpus_resources()
        _setup_hypernyms()

        # run the complete extraction cycle and end with serializing CaLiGraph
        get_logger().info('Running extraction..')
        caligraph.extract_and_serialize()

        success_msg = 'Finished extraction of CaLiGraph.'
        mailer.send_success(success_msg)
        get_logger().info(success_msg)
    except Exception as e:
        error_msg = traceback.format_exc()
        mailer.send_error(error_msg)
        get_logger().error(error_msg)
        raise e

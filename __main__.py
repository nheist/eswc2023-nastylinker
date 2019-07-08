import traceback
import util
import mailer
import impl.category.base as cat_base
import impl.category.cat2ax as cat_axioms
import impl.category.wikitaxonomy as cat_wikitax
import impl.list.store as list_store
import impl.list.hierarchy as list_hierarchy
import impl.list.base as list_base
import impl.list.parser as list_parser
#import impl.dbpedia.heuristics as dbp_heur
#import impl.category.cat2ax as cat_axioms
import impl.util.nlp as nlp_util


def setup():
    category_graph = cat_base.get_conceptual_category_graph()

    # initialise cat2ax axioms
    cat2ax_confidence = util.get_config('cat2ax.pattern_confidence')
    cat2ax_axioms = cat_axioms.extract_category_axioms(category_graph, cat2ax_confidence)
    util.update_cache('cat2ax_axioms', cat2ax_axioms)

    # initialise wikitaxonomy hypernyms
    wikitaxonomy_hypernyms = cat_wikitax.compute_hypernyms(category_graph)
    util.update_cache('wikitaxonomy_hypernyms', wikitaxonomy_hypernyms)


if __name__ == '__main__':
    try:
        util.get_logger().info('Starting CaLiGraph extraction..')

        #list_parser.get_parsed_listpages()

        #entity_data = list_base.get_listpage_entity_data()
        #nlp_util.persist_cache()
        #entity_data.to_csv('data_caligraph/entities_train_v6.csv', index=False)

        #edges_found = len(cat_wikitax.get_valid_edges())

        list_hierarchy.get_child_listpages('')


        mailer.send_success(f'FINISHED list hierarchy extraction')
        util.get_logger().info('Finished CaLiGraph extraction.')
    except Exception as e:
        error_msg = traceback.format_exc()
        mailer.send_error(error_msg)
        util.get_logger().error(error_msg)
        raise e

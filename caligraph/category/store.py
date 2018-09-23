from . import util as cat_util
import caligraph.util.rdf as rdf_util
import caligraph.dbpedia.store as dbp_store
import util
from collections import Counter


def get_resource_type_distribution(category: str) -> dict:
    resources = get_resources(category)
    type_counts = Counter()
    for resource in resources:
        type_counts += Counter(dbp_store.get_transitive_types(resource))
    return {t: count / len(resources) for t, count in type_counts.items()}


def get_all_cats() -> set:
    if '__CATEGORIES__' not in globals():
        initializer = lambda: set(rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.categories')], rdf_util.PREDICATE_TYPE))
        global __CATEGORIES__
        __CATEGORIES__ = util.load_or_create_cache('dbpedia_categories', initializer)

    return __CATEGORIES__


def get_label(category: str) -> str:
    if '__LABELS__' not in globals():
        global __LABELS__
        __LABELS__ = rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.categories')], rdf_util.PREDICATE_LABEL)

    return __LABELS__[category] if category in __LABELS__ else cat_util.category2name(category)


def get_resources(category: str) -> set:
    if '__RESOURCES__' not in globals():
        initializer = lambda: rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.article_categories')], rdf_util.PREDICATE_SUBJECT, reverse_key=True)
        global __RESOURCES__
        __RESOURCES__ = util.load_or_create_cache('dbpedia_category_resources', initializer)

    return __RESOURCES__[category]


def get_resource_to_cats_mapping() -> dict:
    if '__RESOURCE_CATEGORIES__' not in globals():
        initializer = lambda: rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.article_categories')], rdf_util.PREDICATE_SUBJECT)
        global __RESOURCE_CATEGORIES__
        __RESOURCE_CATEGORIES__ = util.load_or_create_cache('dbpedia_resource_categories', initializer)

    return __RESOURCE_CATEGORIES__


def get_parents(category: str) -> set:
    if '__PARENTS__' not in globals():
        initializer = lambda: rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.categories')], rdf_util.PREDICATE_BROADER)
        global __PARENTS__
        __PARENTS__ = util.load_or_create_cache('dbpedia_category_parents', initializer)

    return __PARENTS__[category]


def get_transitive_parents(category: str) -> set:
    if '__TRANSITIVE_PARENTS__' not in globals():
        global __TRANSITIVE_PARENTS__
        __TRANSITIVE_PARENTS__ = dict()
    if category not in __TRANSITIVE_PARENTS__:
        parents = get_parents(category)
        __TRANSITIVE_PARENTS__[category] = parents | {tp for p in parents for tp in get_transitive_parents(p)}

    return __TRANSITIVE_PARENTS__[category]


def get_children(category: str) -> set:
    if '__CHILDREN__' not in globals():
        initializer = lambda: rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.categories')], rdf_util.PREDICATE_BROADER, reverse_key=True)
        global __CHILDREN__
        __CHILDREN__ = util.load_or_create_cache('dbpedia_category_children', initializer)

    return __CHILDREN__[category]


def get_transitive_children(category: str) -> set:
    if '__TRANSITIVE_CHILDREN__' not in globals():
        global __TRANSITIVE_CHILDREN__
        __TRANSITIVE_CHILDREN__ = dict()
    if category not in __TRANSITIVE_CHILDREN__:
        children = get_children(category)
        __TRANSITIVE_CHILDREN__[category] = children | {tc for c in children for tc in get_transitive_children(c)}

    return __TRANSITIVE_PARENTS__[category]


def get_redirects(category: str) -> set:
    if '__REDIRECTS__' not in globals():
        global __REDIRECTS__
        __REDIRECTS__ = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.category_redirects')], rdf_util.PREDICATE_REDIRECTS)

    return __REDIRECTS__[category]


def get_topics(category: str) -> set:
    if '__TOPICS__' not in globals():
        global __TOPICS__
        __TOPICS__ = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.topical_concepts')], rdf_util.PREDICATE_SUBJECT)

    return __TOPICS__[category]


def get_maintenance_cats() -> set:
    if '__MAINTENANCE_CATS__' not in globals():
        global __MAINTENANCE_CATS__
        __MAINTENANCE_CATS__ = set(rdf_util.create_single_val_dict_from_rdf([util.get_data_file('files.dbpedia.maintenance_categories')], rdf_util.PREDICATE_TYPE))

    return __MAINTENANCE_CATS__

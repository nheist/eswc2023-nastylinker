import util
import caligraph.util.rdf as rdf_util


def get_independent_types(dbp_types: set) -> set:
    """Returns only types that are independent, i.e. there are no two types T, T' with T transitiveSupertypeOf T'"""
    return dbp_types.difference({st for t in dbp_types for st in get_transitive_supertypes(t)})


def get_transitive_types(dbp_resource: str) -> set:
    types = get_types(dbp_resource)
    return types | {st for t in types for st in get_transitive_supertypes(t)}


def get_types(dbp_resource: str) -> set:
    if '__RESOURCE_TYPE_MAPPING__' not in globals():
        type_files = [util.get_data_file('files.dbpedia.instance_types'), util.get_data_file('files.dbpedia.transitive_instance_types')]
        initializer = lambda: rdf_util.create_multi_val_dict_from_rdf(type_files, rdf_util.PREDICATE_TYPE)
        global __RESOURCE_TYPE_MAPPING__
        __RESOURCE_TYPE_MAPPING__ = util.load_or_create_cache('dbpedia_resource_type_mapping', initializer)

    return __RESOURCE_TYPE_MAPPING__[dbp_resource]


def get_supertypes(dbp_type: str) -> set:
    if '__SUPERTYPE_MAPPING__' not in globals():
        global __SUPERTYPE_MAPPING__
        __SUPERTYPE_MAPPING__ = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_SUBCLASS_OF)
        # completing the supertypes of each type with the supertypes of its equivalent types
        for t in __SUPERTYPE_MAPPING__:
            __SUPERTYPE_MAPPING__[t] = {st for et in get_equivalent_types(t) for st in __SUPERTYPE_MAPPING__[et]}

    return __SUPERTYPE_MAPPING__[dbp_type]


def get_transitive_supertypes(dbp_type: str) -> set:
    if '__TRANSITIVE_SUPERTYPE_MAPPING__' not in globals():
        global __TRANSITIVE_SUPERTYPE_MAPPING__
        __TRANSITIVE_SUPERTYPE_MAPPING__ = dict()
    if dbp_type not in __TRANSITIVE_SUPERTYPE_MAPPING__:
        direct_supertypes = get_supertypes(dbp_type)
        indirect_supertypes = {tst for dst in direct_supertypes for tst in get_transitive_supertypes(dst)}
        __TRANSITIVE_SUPERTYPE_MAPPING__[dbp_type] = direct_supertypes | indirect_supertypes

    return __TRANSITIVE_SUPERTYPE_MAPPING__[dbp_type]


def get_subtypes(dbp_type: str) -> set:
    if '__SUBTYPE_MAPPING__' not in globals():
        global __SUBTYPE_MAPPING__
        __SUBTYPE_MAPPING__ = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_SUBCLASS_OF, reverse_key=True)
        # completing the subtypes of each type with the subtypes of its equivalent types
        for t in __SUBTYPE_MAPPING__:
            __SUBTYPE_MAPPING__[t] = {st for et in get_equivalent_types(t) for st in __SUBTYPE_MAPPING__[et]}

    return __SUBTYPE_MAPPING__[dbp_type]


def get_transitive_subtypes(dbp_type: str) -> set:
    if '__TRANSITIVE_SUBTYPE_MAPPING__' not in globals():
        global __TRANSITIVE_SUBTYPE_MAPPING__
        __TRANSITIVE_SUBTYPE_MAPPING__ = dict()
    if dbp_type not in __TRANSITIVE_SUBTYPE_MAPPING__:
        subtypes = get_subtypes(dbp_type)
        __TRANSITIVE_SUBTYPE_MAPPING__[dbp_type] = subtypes | {tst for dst in subtypes for tst in get_transitive_subtypes(dst)}

    return __TRANSITIVE_SUBTYPE_MAPPING__[dbp_type]


def get_equivalent_types(dbp_type: str) -> set:
    if '__EQUIVALENT_TYPE_MAPPING__' not in globals():
        global __EQUIVALENT_TYPE_MAPPING__
        __EQUIVALENT_TYPE_MAPPING__ = rdf_util.create_multi_val_dict_from_rdf([util.get_data_file('files.dbpedia.taxonomy')], rdf_util.PREDICATE_EQUIVALENT_CLASS, reflexive=True)

    return {dbp_type} | __EQUIVALENT_TYPE_MAPPING__[dbp_type]

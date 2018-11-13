import util
from typing import Optional
import caligraph.dbpedia.store as dbp_store
from collections import defaultdict
import math

# implementing heuristics from Töpper et al. 2012 - DBpedia Ontology Enrichment for Inconsistency Detection
DOMAIN_THRESHOLD = .96
DISJOINT_THRESHOLD = .17


def get_domain(dbp_predicate) -> Optional[str]:
    global __DOMAINS__
    if '__DOMAINS__' not in globals():
        __DOMAINS__ = util.load_or_create_cache('dbpedia_heuristic_domains', _compute_domains)
    return __DOMAINS__[dbp_predicate] if dbp_predicate in __DOMAINS__ else None


def _compute_domains() -> dict:
    predicate_type_distribution = defaultdict(lambda: defaultdict(int))

    resource_property_mapping = dbp_store.get_resource_property_mapping()
    for r in resource_property_mapping:
        for pred in resource_property_mapping[r]:
            triple_count = len(resource_property_mapping[r][pred])
            predicate_type_distribution[pred]['_sum'] += triple_count
            for t in dbp_store.get_transitive_types(r):
                predicate_type_distribution[pred][t] += triple_count

    predicate_domains = {}
    for pred in predicate_type_distribution:
        t_sum = predicate_type_distribution[pred]['_sum']
        t_scores = {t: t_count / t_sum for t, t_count in predicate_type_distribution[pred].items() if t != '_sum'}
        if t_scores:
            t_score_max = max(t_scores.values())
            if t_score_max >= DOMAIN_THRESHOLD:
                valid_domains = {t for t, t_score in t_scores.items() if t_score == t_score_max}
                if len(valid_domains) > 1:
                    valid_domains = {t for t in valid_domains if not valid_domains.intersection(dbp_store.get_transitive_subtypes(t))}

                if len(valid_domains) == 1 or dbp_store.are_equivalent_types(valid_domains):
                    predicate_domains[pred] = valid_domains.pop()

    return predicate_domains


def get_disjoint_types(dbp_type) -> set:
    global __DISJOINT_TYPES__
    if '__DISJOINT_TYPES__' not in globals():
        __DISJOINT_TYPES__ = util.load_or_create_cache('dbpedia_heuristic_disjoint_types', _compute_disjoint_types)
    return __DISJOINT_TYPES__[dbp_type]


def _compute_disjoint_types() -> dict:
    disjoint_types = defaultdict(set)

    type_property_weights = _compute_type_property_weights()
    dbp_types = {t for types in dbp_store._get_resource_type_mapping().values() for t in types}
    util.get_logger().debug('computing type similarities..')
    while len(dbp_types) > 0:
        dbp_type = dbp_types.pop()
        for other_dbp_type in dbp_types:
            if _compute_type_similarity(dbp_type, other_dbp_type, type_property_weights) <= DISJOINT_THRESHOLD:
                disjoint_types[dbp_type].add(other_dbp_type)
                disjoint_types[other_dbp_type].add(dbp_type)

    # remove subtypes from disjoint types
    disjoint_types = {t: {dt for dt in dts if not dbp_store.get_transitive_supertypes(dt).intersection(dts)} for t, dts in disjoint_types.items()}
    util.get_logger().debug('computed type similarities.')
    return disjoint_types


def _compute_type_property_weights() -> dict:
    util.get_logger().debug('computing type property weights..')
    type_property_weights = defaultdict(lambda: defaultdict(float))

    property_frequencies = _compute_property_frequencies()
    inverse_type_frequencies = _compute_inverse_type_frequencies()
    for dbp_type in dbp_store.get_all_types():
        for dbp_pred in inverse_type_frequencies:
            type_property_weights[dbp_type][dbp_pred] = property_frequencies[dbp_type][dbp_pred] * inverse_type_frequencies[dbp_pred]
    util.get_logger().debug('computed type property weights.')
    return type_property_weights


def _compute_property_frequencies() -> dict:
    util.get_logger().debug('computing property frequencies..')
    property_frequencies = defaultdict(lambda: defaultdict(int))
    for r in dbp_store.get_resources():
        types = dbp_store.get_transitive_types(r)
        for pred, values in dbp_store.get_properties(r).items():
            for t in types:
                property_frequencies[t][pred] += len(values)

    util.get_logger().debug('computed property frequencies.')
    return defaultdict(lambda: defaultdict(float), {t: defaultdict(float, {pred: (1 + math.log2(count) if count > 0 else 0) for pred, count in property_frequencies[t].items()}) for t in property_frequencies})


def _compute_inverse_type_frequencies() -> dict:
    util.get_logger().debug('computing inverse type frequencies..')
    predicate_types = defaultdict(set)
    for r in dbp_store.get_resources():
        for pred in dbp_store.get_properties(r):
            predicate_types[pred].update(dbp_store.get_transitive_types(r))

    overall_type_count = len(dbp_store.get_all_types())
    util.get_logger().debug('computed inverse type frequencies.')
    return {pred: math.log2(overall_type_count / (len(predicate_types[pred]) + 1)) for pred in dbp_store.get_all_predicates()}


def _compute_type_similarity(type_a: str, type_b: str, type_property_weights: dict) -> float:
    numerator = sum(type_property_weights[type_a][pred] * type_property_weights[type_b][pred] for pred in type_property_weights[type_a])
    denominator_a = math.sqrt(sum([type_property_weights[type_a][pred] ** 2 for pred in type_property_weights[type_a]]))
    denominator_b = math.sqrt(sum([type_property_weights[type_b][pred] ** 2 for pred in type_property_weights[type_b]]))
    return numerator / (denominator_a * denominator_b) if denominator_a * denominator_b > 0 else 0


"""Compute hypernyms with methods similar to Ponzetto & Strube: Deriving a large scale taxonomy from Wikipedia."""

from collections import defaultdict, Counter
import utils
import bz2
import pickle
import impl.util.words as words_util
from polyleven import levenshtein


def is_hypernym(hyper_word: str, hypo_word: str) -> bool:
    """Returns True, if `hyper_word` and `hypo_word` are synonyms or if the former is a hypernym of the latter."""
    global __WIKITAXONOMY_HYPERNYMS__
    if '__WIKITAXONOMY_HYPERNYMS__' not in globals():
        __WIKITAXONOMY_HYPERNYMS__ = utils.load_cache('wikitaxonomy_hypernyms')
        if not __WIKITAXONOMY_HYPERNYMS__:
            raise ValueError('wikitaxonomy_hypernyms not initialised. Run hypernym extraction once to create the necessary cache!')

    if is_synonym(hyper_word, hypo_word):
        return True
    return hyper_word.lower() in __WIKITAXONOMY_HYPERNYMS__[hypo_word.lower()]


def phrases_are_equal(phrase_a: set, phrase_b: set) -> bool:
    if len(phrase_a) != len(phrase_b):
        return False
    return {w.lower() for w in phrase_a} == {w.lower() for w in phrase_b}


def phrases_are_synonymous(phrase_a: set, phrase_b: set) -> bool:
    """Returns True, if the phrases consist of synonymous pairs of words (i.e. there is a synonym for every word)."""
    if len(phrase_a) != len(phrase_b):
        return False
    return all(any(is_synonym(a, b) for a in phrase_a) for b in phrase_b) and all(any(is_synonym(b, a) for b in phrase_b) for a in phrase_a)


def is_synonym(word: str, another_word: str) -> bool:
    """Returns True, if the words are synonyms."""
    if not word or not another_word:
        return False
    if word == another_word:
        return True
    if (word.istitle() or another_word.istitle()) and (word.lower().startswith(another_word.lower()) or another_word.lower().startswith(word.lower())):
        # better recognition of inflectional forms of countries (e.g. recognise 'Simbabwe' and 'Simbabwean' as synonyms)
        return True
    return any(word in s for s in words_util.get_synonyms(another_word)) or any(another_word in s for s in words_util.get_synonyms(word))


def get_variations(text: str) -> set:
    """Returns all synonyms of a text having an edit-distance of 2 or less."""
    text = text.replace(' ', '_')
    return {s.replace('_', ' ') for s in words_util.get_synonyms(text) if levenshtein(s, text, 2) <= 2}


# thresholds of individual hypernym sources
THRESHOLD_AXIOM = 10
THRESHOLD_WIKI = 100
THRESHOLD_WEBISALOD = .4


def compute_hypernyms(category_graph) -> dict:
    """Retrieves all hypernym relationships from the three sources (Wiki corpus, WebIsALOD, Category axioms)."""
    hypernyms = defaultdict(set)

    # collect hypernyms from axiom matches between Wikipedia categories
    cat_headlemmas = category_graph.get_node_LHS()
    axiom_hypernyms = defaultdict(Counter)
    for parent, child in category_graph.get_axiom_edges():
        for cl in cat_headlemmas[child]:
            for pl in cat_headlemmas[parent]:
                axiom_hypernyms[cl.lower()][pl.lower()] += 1

    # load remaining hypernyms
    wiki_hypernyms = utils.load_cache('wikipedia_hypernyms')
    webisalod_data = pickle.load(bz2.open(utils.get_data_file('files.dbpedia.webisalod_hypernyms'), mode='rb'))
    webisalod_hypernyms = defaultdict(dict)
    for parent, child, conf in webisalod_data:
        webisalod_hypernyms[child][parent] = conf

    # merge hypernyms
    candidates = set(axiom_hypernyms) | set(wiki_hypernyms) | set(webisalod_hypernyms)
    for candidate in candidates:
        hyper_count = Counter()
        if candidate in axiom_hypernyms:
            for word, count in axiom_hypernyms[candidate].items():
                if count >= THRESHOLD_AXIOM:
                    hyper_count[word] += 2
        if candidate in wiki_hypernyms:
            for word, count in wiki_hypernyms[candidate].items():
                if count >= THRESHOLD_WIKI:
                    hyper_count[word] += 1
        if candidate in webisalod_hypernyms:
            for word, conf in webisalod_hypernyms[candidate].items():
                if conf >= THRESHOLD_WEBISALOD:
                    hyper_count[word] += 1
        hypernyms[candidate] = {word for word, count in hyper_count.items() if count > 1}

    return hypernyms

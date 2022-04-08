from typing import Iterator, List, Tuple
from tqdm import tqdm
from collections import defaultdict
import utils
import spacy
from spacy.tokens import Span
from impl.util.spacy.components import LEXICAL_HEAD, LEXICAL_HEAD_SUBJECT, LEXICAL_HEAD_SUBJECT_PLURAL, BY_PHRASE
import impl.util.spacy.hearst_matcher as hearst_matcher


BATCH_SIZE = 20000
N_PROCESSES = utils.get_config('max_cpus')
__SET_DOCUMENT_CACHE__ = defaultdict(lambda: None)


def parse_sets(taxonomic_sets: list) -> Iterator:
    """Parse potential set structures of a taxonomy, e.g. Wikipedia categories or list pages."""
    global __SET_PARSER__
    if '__SET_PARSER__' not in globals():
        __SET_PARSER__ = spacy.load('en_core_web_lg')
        __SET_PARSER__.remove_pipe('ner')

        __SET_PARSER__.vocab.strings.add(LEXICAL_HEAD)
        __SET_PARSER__.vocab.strings.add(LEXICAL_HEAD_SUBJECT)
        __SET_PARSER__.vocab.strings.add(LEXICAL_HEAD_SUBJECT_PLURAL)
        __SET_PARSER__.vocab.strings.add(BY_PHRASE)

        __SET_PARSER__.add_pipe('tag_lexical_head')
        __SET_PARSER__.add_pipe('tag_lexical_head_subjects')
        __SET_PARSER__.add_pipe('tag_by_phrase')

    unknown_sets = [s for s in taxonomic_sets if s and s not in __SET_DOCUMENT_CACHE__]
    if len(unknown_sets) <= BATCH_SIZE * N_PROCESSES:
        for s in unknown_sets:
            __SET_DOCUMENT_CACHE__[s] = __SET_PARSER__(s)
    else:
        set_tuples = [(s, s) for s in unknown_sets]
        for doc, s in tqdm(__SET_PARSER__.pipe(set_tuples, as_tuples=True, batch_size=BATCH_SIZE, n_process=N_PROCESSES), total=len(unknown_sets), desc='Parsing sets with spaCy'):
            __SET_DOCUMENT_CACHE__[s] = doc
    return iter([__SET_DOCUMENT_CACHE__[s] for s in taxonomic_sets])


def parse_texts(texts: list) -> Iterator:
    """Parse plain texts like the content of a Wikipedia page or a listing item."""
    bp = _get_base_parser()
    if len(texts) <= BATCH_SIZE:
        return iter([bp(t) for t in texts])
    return iter(tqdm(bp.pipe(texts, batch_size=BATCH_SIZE, n_process=N_PROCESSES), total=len(texts), desc='Parsing texts with spaCy'))


def get_hearst_pairs(text: str) -> List[Tuple[Span, Span]]:
    """Parse text and retrieve (sub, obj) pairs for every occurrence of a hearst pattern."""
    bp = _get_base_parser()
    return hearst_matcher.get_hearst_matches(text, bp)


def _get_base_parser():
    global __BASE_PARSER__
    if '__BASE_PARSER__' not in globals():
        __BASE_PARSER__ = spacy.load('en_core_web_lg')
    return __BASE_PARSER__

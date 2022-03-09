import impl.listpage.store as list_store
import utils
from tqdm import tqdm
from . import combine, extract
from .preprocess.word_tokenize import WordTokenizer
from .preprocess.pos_label import map_entities_to_pos_labels
from impl.listpage.subject_entity import find_subject_entities_for_listpage
from impl import wikipedia
import torch


def get_page_subject_entities(graph) -> dict:
    """Retrieve the extracted entities per page with context."""
    # TODO: merge retrieval and combine step => add disambiguation of *ALL* entities during retrieval
    return combine.match_entities_with_uris(_get_subject_entity_predictions(graph))


def _get_subject_entity_predictions(graph) -> dict:
    initializer = lambda: _make_subject_entity_predictions(graph)
    return utils.load_or_create_cache('subject_entity_predictions', initializer)


def _make_subject_entity_predictions(graph) -> dict:
    tokenizer, model = extract.get_tagging_tokenizer_and_model(lambda: _get_training_data(graph))
    predictions = {p: extract.extract_subject_entities(chunks, tokenizer, model)[0] for p, chunks in tqdm(_get_page_data().items(), desc='Predicting subject entities')}
    torch.cuda.empty_cache()  # flush GPU cache to free GPU for other purposes
    return predictions


def _get_training_data(graph) -> tuple:
    # retrieve or extract page-wise training data
    initializer = lambda: _get_tokenized_list_pages_with_entity_labels(graph)
    training_data = utils.load_or_create_cache('subject_entity_training_data', initializer)
    # flatten training data into chunks and replace entities with their POS tags
    tokens, labels = [], []
    for token_chunks, entity_chunks in training_data.values():
        tokens.extend(token_chunks)
        labels.extend(map_entities_to_pos_labels(entity_chunks))
    return tokens, labels


def _get_tokenized_list_pages_with_entity_labels(graph) -> dict:
    listpages = list_store.get_parsed_listpages()
    entity_labels = {lp_uri: find_subject_entities_for_listpage(lp_uri, lp_data, graph) for lp_uri, lp_data in listpages.items()}
    return WordTokenizer()(listpages, entity_labels=entity_labels)


def _get_page_data() -> dict:
    initializer = lambda: WordTokenizer()(wikipedia.get_parsed_articles())
    return utils.load_or_create_cache('subject_entity_page_data', initializer)

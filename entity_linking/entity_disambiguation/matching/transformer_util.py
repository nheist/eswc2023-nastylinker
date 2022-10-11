from typing import List, Tuple, Dict, Union
from itertools import cycle, islice
import torch.nn as nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, CrossEncoder, losses, InputExample
import utils
from impl.util.string import alternate_iters_to_string
from impl.util.transformer import SpecialToken
from impl.wikipedia.page_parser import WikiListing, WikiTable, WikiListingItem, WikiEnumEntry
from impl.dbpedia.resource import DbpResourceStore
from impl.caligraph.entity import ClgEntity
from entity_linking.entity_disambiguation.data import DataCorpus


def add_special_tokens(model: Union[SentenceTransformer, CrossEncoder]):
    if isinstance(model, SentenceTransformer):
        word_embedding_model = model._first_module()
        tokenizer = word_embedding_model.tokenizer
        transformer = word_embedding_model.auto_model
    elif isinstance(model, CrossEncoder):
        tokenizer = model.tokenizer
        transformer = model.model
    else:
        raise ValueError(f'Invalid type for model: {type(model)}')
    tokenizer.add_tokens(list(SpecialToken.all_tokens()), special_tokens=True)
    transformer.resize_token_embeddings(len(tokenizer))


def get_loss_function(loss: str, model) -> nn.Module:
    if loss == 'COS':
        return losses.CosineSimilarityLoss(model=model)
    elif loss == 'RL':
        return losses.MultipleNegativesRankingLoss(model=model)
    elif loss == 'SRL':
        return losses.MultipleNegativesSymmetricRankingLoss(model=model)
    raise ValueError(f'Unknown loss identifier: {loss}')


def generate_training_data(training_set: DataCorpus, negatives: set, batch_size: int, add_page_context: bool, add_listing_entities: bool, add_entity_abstract: bool, add_kg_info: bool) -> DataLoader:
    source_input = prepare_listing_items(training_set.source, add_page_context, add_listing_entities)
    target_input = source_input if training_set.target is None else prepare_entities(training_set.target, add_entity_abstract, add_kg_info)
    input_examples = [InputExample(texts=[source_input[source_id], target_input[target_id]], label=1) for source_id, target_id in training_set.alignment]
    input_examples.extend([InputExample(texts=[source_input[source_id], target_input[target_id]], label=0) for source_id, target_id in negatives])
    return DataLoader(input_examples, shuffle=True, batch_size=batch_size)


def prepare_listing_items(listings: List[WikiListing], add_page_context: bool, add_listing_entities: bool) -> Dict[Tuple[int, int, int], str]:
    utils.get_logger().debug('Preparing listing items..')
    if not add_page_context and not add_listing_entities:
        return {(l.page_idx, l.idx, i.idx): i.subject_entity.label for l in listings for i in l.get_items() if i.subject_entity is not None}
    result = {}
    for listing in listings:
        prepared_context = _prepare_listing_context(listing)
        prepared_items = [_prepare_listing_item(item) for item in listing.get_items()]
        for idx, item in enumerate(listing.get_items()):
            item_se = item.subject_entity
            if item_se is None:
                continue
            item_id = (listing.page_idx, listing.idx, item.idx)
            # add subject entity, its type, and page context
            item_content = f' {SpecialToken.CONTEXT_SEP.value} '.join([item_se.label, item_se.entity_type, prepared_context])
            # add item and `add_listing_entities` subsequent items (add items from start if no subsequent items left)
            item_content += ''.join(islice(cycle(prepared_items), idx, idx + add_listing_entities + 1))
            result[item_id] = item_content
        return result


def _prepare_listing_context(listing: WikiListing) -> str:
    page_resource = DbpResourceStore.instance().get_resource_by_idx(listing.page_idx)
    ctx = [page_resource.get_label(), listing.topsection.title, listing.section.title]
    if isinstance(listing, WikiTable):
        ctx.append(_prepare_listing_item(listing.header))
    return f' {SpecialToken.CONTEXT_SEP.value} '.join(ctx) + f' {SpecialToken.CONTEXT_END.value} '


def _prepare_listing_item(item: WikiListingItem) -> str:
    if isinstance(item, WikiEnumEntry):
        tokens = [SpecialToken.get_entry_by_depth(item.depth)] + item.tokens
        whitespaces = [' '] + item.whitespaces[:-1] + [' ']
    else:  # WikiTableRow
        tokens, whitespaces = [], []
        for cell_tokens, cell_whitespaces in zip(item.tokens, item.whitespaces):
            tokens += [SpecialToken.TABLE_COL.value] + cell_tokens
            whitespaces += [' '] + cell_whitespaces[:-1] + [' ']
        tokens[0] = SpecialToken.TABLE_ROW.value  # special indicator for start of table row
    return alternate_iters_to_string(tokens, whitespaces)


# TODO: potential caching of prepared entities w.r.t add_entity_abstract and add_kg_info
def prepare_entities(entities: List[ClgEntity], add_entity_abstract: bool, add_kg_info: int) -> Dict[int, str]:
    utils.get_logger().debug('Preparing entities..')
    result = {}
    for e in entities:
        ent_description = [e.get_label(), e.get_type_label().name]
        if add_entity_abstract:
            ent_description.append((e.get_abstract() or '')[:200])
        if add_kg_info:
            kg_info = [('type', t) for t in e.get_types()]
            prop_count = max(0, add_kg_info - len(kg_info))
            if prop_count > 0:
                kg_info += list(e.get_properties(as_tuple=True))[:prop_count]
        result[e.idx] = f' {SpecialToken.CONTEXT_SEP.value} '.join(ent_description)
    return result

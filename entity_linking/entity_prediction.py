from typing import Dict, List, Tuple, Set, Iterable
import random
import utils
from impl.util.rdf import EntityIndex
from impl import subject_entity
from impl.subject_entity import combine
from impl.subject_entity.preprocess.word_tokenize import WordTokenizer, WordTokenizerSpecialToken
from transformers import Trainer, IntervalStrategy, TrainingArguments, AutoTokenizer, AutoModel
from impl.dbpedia.resource import DbpResource, DbpResourceStore
from entity_linking.preprocessing.embeddings import EntityIndexToEmbeddingMapper
from entity_linking.model.entity_prediction import TransformerForEntityPrediction
from entity_linking.data.entity_prediction import prepare_dataset
from entity_linking.evaluation.entity_prediction import EntityPredictionEvaluator


def run_prediction(model_name: str, sample: int, epochs: int, batch_size: int, loss: str, learning_rate: float, warmup_steps: int, weight_decay: float, num_ents: int, ent_dim: int, items_per_chunk: int, cls_predictor: bool):
    run_id = f'{model_name}_s-{sample}_ipc-{items_per_chunk}_ne-{num_ents}_cp-{cls_predictor}_ed-{ent_dim}_e-{epochs}_bs-{batch_size}_loss-{loss}_lr-{learning_rate}_ws-{warmup_steps}_wd-{weight_decay}'
    # prepare tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, additional_special_tokens=list(WordTokenizerSpecialToken.all_tokens()))
    encoder = AutoModel.from_pretrained(model_name)
    encoder.resize_token_embeddings(len(tokenizer))
    ent_idx2emb = EntityIndexToEmbeddingMapper(ent_dim)
    model = TransformerForEntityPrediction(encoder, cls_predictor, ent_idx2emb, ent_dim, loss)
    # load data
    dataset_version = f'ep-s{sample}-ipc{items_per_chunk}-ne{num_ents}'
    train_data, val_data = utils.load_or_create_cache('vector_prediction_training_data', lambda: _load_train_and_val_datasets(tokenizer, sample, items_per_chunk, num_ents), version=dataset_version)
    # run evaluation
    training_args = TrainingArguments(
        seed=42,
        save_strategy=IntervalStrategy.NO,
        output_dir=f'./vp_eval/output/{run_id}',
        logging_strategy=IntervalStrategy.STEPS,
        logging_dir=f'./vp_eval/logs/{run_id}',
        logging_steps=1000,
        evaluation_strategy=IntervalStrategy.STEPS,
        eval_steps=5000,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=EntityPredictionEvaluator(ent_idx2emb, batch_size).evaluate
    )
    trainer.train()


def _load_train_and_val_datasets(tokenizer, sample: int, items_per_chunk: int, num_ents: int) -> Tuple[Dict[DbpResource, tuple], Dict[DbpResource, tuple]]:
    subject_entity_pages = combine.get_subject_entity_page_content(subject_entity._get_subject_entity_predictions())
    # split into train and validation
    sample_fraction = sample / 100  # sample is given as a percentage
    all_pages = random.sample(list(subject_entity_pages), int(len(subject_entity_pages) * sample_fraction))  # only use `sample` % of overall data

    train_pages = set(random.sample(all_pages, int(len(all_pages) * .9)))  # 90% of pages for train, 10% for val
    train_data = _create_vector_prediction_data({res: content for res, content in subject_entity_pages.items() if res in train_pages}, items_per_chunk, False)
    train_data = _prepare_data(train_data.values(), tokenizer, num_ents)

    val_pages = set(all_pages).difference(train_pages)
    val_data = _create_vector_prediction_data({res: content for res, content in subject_entity_pages.items() if res in val_pages}, items_per_chunk, True)
    val_data = _prepare_data(val_data.values(), tokenizer, num_ents)
    return train_data, val_data


def _create_vector_prediction_data(subject_entity_pages: Dict[DbpResource, dict], items_per_chunk: int, include_new_entities: bool) -> Dict[DbpResource, Tuple[List[List[str]], List[List[str]], List[List[int]]]]:
    entity_labels = _get_subject_entity_labels(subject_entity_pages, include_new_entities)
    return WordTokenizer(max_items_per_chunk=items_per_chunk)(subject_entity_pages, entity_labels=entity_labels)


def _get_subject_entity_labels(subject_entity_pages: Dict[DbpResource, dict], include_new_entities: bool) -> Dict[DbpResource, Tuple[Set[int], Set[int]]]:
    valid_entity_indices = set(DbpResourceStore.instance().get_embedding_vectors())
    entity_labels = {}
    for res, page_content in subject_entity_pages.items():
        # collect all subject entity labels
        subject_entity_indices = set()
        subject_entity_indices.update({ent['idx'] for s in page_content['sections'] for enum in s['enums'] for entry in enum for ent in entry['entities']})
        subject_entity_indices.update({ent['idx'] for s in page_content['sections'] for table in s['tables'] for row in table['data'] for cell in row for ent in cell['entities']})
        # get rid of non-entities and entities without RDF2vec embeddings (as we can't use them for training/eval)
        subject_entity_indices = subject_entity_indices.intersection(valid_entity_indices)
        if include_new_entities:
            subject_entity_indices.add(EntityIndex.NEW_ENTITY.value)
        entity_labels[res] = (subject_entity_indices, set())
    return entity_labels


def _prepare_data(page_data: Iterable[Tuple[List[List[str]], List[List[int]]]], tokenizer, num_ents: int):
    tokens, labels = [], []
    for token_chunks, _, entity_chunks in page_data:
        tokens.extend(token_chunks)
        labels.extend(entity_chunks)
    return prepare_dataset(tokens, labels, tokenizer, num_ents)
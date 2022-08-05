from typing import Tuple, List
from copy import deepcopy
from collections import namedtuple, Counter
import numpy as np
import utils
from transformers import Trainer, IntervalStrategy, TrainingArguments, AutoTokenizer, AutoModelForTokenClassification, EvalPrediction
from impl.util.transformer import SpecialToken, EntityIndex
from impl.subject_entity.preprocess.pos_label import POSLabel
from impl.subject_entity.preprocess.word_tokenize import ListingType
from entity_linking.data import prepare
from entity_linking.data.mention_detection import prepare_dataset, MentionDetectionDataset
from entity_linking.model.mention_detection import TransformerForMentionDetectionAndTypePrediction


def run_evaluation(model_name: str, epochs: int, batch_size: int, learning_rate: float, warmup_steps: int, weight_decay: float, ignore_tags: bool, predict_single_tag: bool, negative_sample_size: float):
    run_id = f'{model_name}_it-{ignore_tags}_st-{predict_single_tag}_nss-{negative_sample_size}_e-{epochs}_lr-{learning_rate}_ws-{warmup_steps}_wd-{weight_decay}'
    # prepare tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, additional_special_tokens=list(SpecialToken.all_tokens()))
    number_of_labels = 2 if ignore_tags else len(POSLabel)
    if predict_single_tag:
        model = TransformerForMentionDetectionAndTypePrediction(model_name, len(tokenizer), number_of_labels)
    else:
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=number_of_labels)
        model.resize_token_embeddings(len(tokenizer))
    # load data
    train_dataset, val_dataset = _load_train_and_val_datasets(tokenizer, ignore_tags, predict_single_tag, negative_sample_size)
    # run evaluation
    training_args = TrainingArguments(
        seed=42,
        save_strategy=IntervalStrategy.NO,
        output_dir=f'./entity_linking/MD/output/{run_id}',
        logging_strategy=IntervalStrategy.STEPS,
        logging_dir=f'./entity_linking/MD/logs/{run_id}',
        logging_steps=500,
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
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda eval_prediction: SETagsEvaluator(eval_prediction, val_dataset.listing_types, predict_single_tag).evaluate()
    )
    trainer.train()


def _load_train_and_val_datasets(tokenizer, ignore_tags: bool, predict_single_tag: bool, negative_sample_size: float) -> Tuple[MentionDetectionDataset, MentionDetectionDataset]:
    train_data, val_data = utils.load_or_create_cache('MD_data', prepare.get_md_train_and_val_data)
    # prepare data
    train_dataset = prepare_dataset(train_data, tokenizer, ignore_tags, predict_single_tag, negative_sample_size)
    val_dataset = prepare_dataset(val_data, tokenizer, ignore_tags, predict_single_tag)
    return train_dataset, val_dataset


Entity = namedtuple("Entity", "e_type start_offset end_offset")


class SETagsEvaluator:
    def __init__(self, eval_prediction: EvalPrediction, listing_types: List[str], predict_single_tag: bool):
        if predict_single_tag:
            # with mention logits we only predict whether there is a subject entity in this position (1 or 0)
            # so we multiply with type_id to "convert" it back to the notion where we predict types per position
            mention_logits, type_logits = eval_prediction.predictions
            # TODO: DEBUG START
            print(type(eval_prediction.predictions))
            print(type(mention_logits))
            print(mention_logits)
            print(type(type_logits))
            print(type_logits)
            # TODO: DEBUG END
            type_ids = np.expand_dims(type_logits.argmax(-1), -1)
            self.mentions = mention_logits.argmax(-1) * type_ids
            # same for labels
            mention_labels = eval_prediction.label_ids[:, 0, :]
            type_labels = np.expand_dims(eval_prediction.label_ids[:, 1, 0], -1)
            self.labels = mention_labels * type_labels
            self.masks = mention_labels != EntityIndex.IGNORE.value
        else:
            self.mentions = eval_prediction.predictions.argmax(-1)
            self.labels = eval_prediction.label_ids
            self.masks = self.labels != EntityIndex.IGNORE.value

        self.listing_types = listing_types

        result_schema = {
            'strict': Counter({'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 0}),
            'exact': Counter({'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 0}),
            'partial': Counter({'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 0}),
            'ent_type': Counter({'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 0}),
        }
        self.results = {
            'overall': deepcopy(result_schema),
            ListingType.ENUMERATION.value: deepcopy(result_schema),
            ListingType.TABLE.value: deepcopy(result_schema)
        }

    def evaluate(self) -> dict:
        for mention_ids, true_ids, mask, listing_type in zip(self.mentions, self.labels, self.masks, self.listing_types):
            # remove invalid preds/labels
            mention_ids = mention_ids[mask]
            true_ids = true_ids[mask]
            # run computation
            self.compute_metrics(self._collect_named_entities(mention_ids), self._collect_named_entities(true_ids), listing_type)

        # compute overall stats for listing types
        for metric in self.results['overall']:
            self.results['overall'][metric] = self.results[ListingType.ENUMERATION.value][metric] + self.results[ListingType.TABLE.value][metric]

        return self._compute_precision_recall_wrapper()

    @classmethod
    def _collect_named_entities(cls, mention_ids) -> set:
        named_entities = set()
        start_offset = None
        ent_type = None

        for offset, mention_id in enumerate(mention_ids):
            if mention_id == 0:
                if ent_type is not None and start_offset is not None:
                    named_entities.add(Entity(ent_type, start_offset, offset))
                    start_offset = None
                    ent_type = None
            elif ent_type is None:
                ent_type = mention_id
                start_offset = offset
        # catches an entity that goes up until the last token
        if ent_type is not None and start_offset is not None:
            named_entities.add(Entity(ent_type, start_offset, len(mention_ids)))
        return named_entities

    def compute_metrics(self, pred_named_entities: set, true_named_entities: set, listing_type: str):
        # keep track of entities that overlapped
        true_which_overlapped_with_pred = set()

        for pred in pred_named_entities:
            # Check each of the potential scenarios in turn. For scenario explanation see
            # http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/

            # Scenario I: Exact match between true and pred
            if pred in true_named_entities:
                true_which_overlapped_with_pred.add(pred)
                self.results[listing_type]['strict']['correct'] += 1
                self.results[listing_type]['ent_type']['correct'] += 1
                self.results[listing_type]['exact']['correct'] += 1
                self.results[listing_type]['partial']['correct'] += 1

            else:
                # check for overlaps with any of the true entities
                found_overlap = False
                for true in true_named_entities:
                    pred_range = set(range(pred.start_offset, pred.end_offset))
                    true_range = set(range(true.start_offset, true.end_offset))

                    # Scenario IV: Offsets match, but entity type is wrong
                    if true.start_offset == pred.start_offset and true.end_offset == pred.end_offset and true.e_type != pred.e_type:
                        self.results[listing_type]['strict']['incorrect'] += 1
                        self.results[listing_type]['ent_type']['incorrect'] += 1
                        self.results[listing_type]['partial']['correct'] += 1
                        self.results[listing_type]['exact']['correct'] += 1
                        true_which_overlapped_with_pred.add(true)
                        found_overlap = True
                        break

                    # check for an overlap i.e. not exact boundary match, with true entities
                    elif true_range.intersection(pred_range):
                        true_which_overlapped_with_pred.add(true)

                        # Scenario V: There is an overlap (but offsets do not match exactly), and the entity type is the same.
                        if pred.e_type == true.e_type:
                            self.results[listing_type]['strict']['incorrect'] += 1
                            self.results[listing_type]['ent_type']['correct'] += 1
                            self.results[listing_type]['partial']['partial'] += 1
                            self.results[listing_type]['exact']['incorrect'] += 1
                            found_overlap = True
                            break

                        # Scenario VI: Entities overlap, but the entity type is different.
                        else:
                            self.results[listing_type]['strict']['incorrect'] += 1
                            self.results[listing_type]['ent_type']['incorrect'] += 1
                            self.results[listing_type]['partial']['partial'] += 1
                            self.results[listing_type]['exact']['incorrect'] += 1
                            found_overlap = True
                            break

                # Scenario II: Entities are spurious (i.e., over-generated).
                if not found_overlap:
                    self.results[listing_type]['strict']['spurious'] += 1
                    self.results[listing_type]['ent_type']['spurious'] += 1
                    self.results[listing_type]['partial']['spurious'] += 1
                    self.results[listing_type]['exact']['spurious'] += 1

        # Scenario III: Entity was missed entirely.
        missed_entities = len(true_named_entities.difference(true_which_overlapped_with_pred))
        self.results[listing_type]['strict']['missed'] += missed_entities
        self.results[listing_type]['ent_type']['missed'] += missed_entities
        self.results[listing_type]['partial']['missed'] += missed_entities
        self.results[listing_type]['exact']['missed'] += missed_entities

    def _compute_precision_recall_wrapper(self):
        final_metrics = {}
        for lt, stats in self.results.items():
            for k, v in stats.items():
                for metric_key, metric_value in self._compute_precision_recall(lt, k, v).items():
                    final_metrics[metric_key] = metric_value
        return final_metrics

    @classmethod
    def _compute_precision_recall(cls, listing_type, eval_schema, eval_data):
        correct = eval_data['correct']
        incorrect = eval_data['incorrect']
        partial = eval_data['partial']
        missed = eval_data['missed']
        spurious = eval_data['spurious']
        actual = correct + incorrect + partial + spurious  # number of annotations produced by the NER system
        possible = correct + incorrect + partial + missed  # number of annotations in the gold-standard which contribute to the final score

        if eval_schema in ['partial', 'ent_type']:
            precision = (correct + 0.5 * partial) / actual if actual > 0 else 0
            recall = (correct + 0.5 * partial) / possible if possible > 0 else 0
        else:
            precision = correct / actual if actual > 0 else 0
            recall = correct / possible if possible > 0 else 0

        return {f'{listing_type}-COUNT': possible, f'{listing_type}-P-{eval_schema}': precision, f'{listing_type}-R-{eval_schema}': recall}
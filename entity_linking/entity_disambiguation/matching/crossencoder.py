from typing import Set, List, Optional, Dict
from collections import defaultdict
import random
from sentence_transformers import CrossEncoder
import utils
from impl.caligraph.entity import ClgEntity
from impl.wikipedia.page_parser import WikiListing
from entity_linking.entity_disambiguation.data import Pair, DataCorpus
from entity_linking.entity_disambiguation.matching.util import MatchingScenario
from entity_linking.entity_disambiguation.matching.matcher import MatcherWithCandidates
from entity_linking.entity_disambiguation.matching import transformer_util


class CrossEncoderMatcher(MatcherWithCandidates):
    def __init__(self, scenario: MatchingScenario, params: dict):
        super().__init__(scenario, params)
        # model params
        self.base_model = params['base_model']
        self.confidence_threshold = params['confidence_threshold']
        self.add_page_context = params['add_page_context']
        self.add_listing_entities = params['add_listing_entities']
        self.add_entity_abstract = params['add_entity_abstract']
        self.add_kg_info = params['add_kg_info']
        # training params
        self.batch_size = params['batch_size']
        self.epochs = params['epochs']
        self.warmup_steps = params['warmup_steps']
        # prepare Cross-Encoder
        self.model = CrossEncoder(self.base_model, num_labels=1)
        transformer_util.add_special_tokens(self.model)

    def _get_param_dict(self) -> dict:
        params = {
            'bm': self.base_model,
            'ct': self.confidence_threshold,
            'apc': self.add_page_context,
            'ale': self.add_listing_entities,
            'aea': self.add_entity_abstract,
            'aki': self.add_kg_info,
            'bs': self.batch_size,
            'e': self.epochs,
            'ws': self.warmup_steps,
        }
        return super()._get_param_dict() | params

    def train(self, training_set: DataCorpus, eval_set: DataCorpus, save_alignment: bool) -> Dict[str, Set[Pair]]:
        utils.get_logger().info('Training matcher..')
        self._train_model(training_set, eval_set)
        return {}  # never predict on the training set with the cross-encoder

    def _train_model(self, training_set: DataCorpus, eval_set: DataCorpus):
        if self.epochs == 0:
            return  # skip training
        utils.get_logger().debug('Preparing training data..')
        negatives = list(self.candidates[self.MODE_TRAIN].difference(training_set.alignment))
        negatives_sample = random.sample(negatives, min([len(negatives), len(training_set.alignment) * 2]))
        train_dataloader = transformer_util.generate_training_data(training_set, negatives_sample, self.batch_size, self.add_page_context, self.add_listing_entities, self.add_entity_abstract, self.add_kg_info)
        utils.release_gpu()
        utils.get_logger().debug('Starting training..')
        self.model.fit(train_dataloader=train_dataloader, epochs=self.epochs, warmup_steps=self.warmup_steps, save_best_model=False)

    def predict(self, prefix: str, source: List[WikiListing], target: Optional[List[ClgEntity]]) -> Set[Pair]:
        source_input = transformer_util.prepare_listing_items(source, self.add_page_context, self.add_listing_entities)
        if self.scenario == MatchingScenario.MENTION_MENTION:
            target_input = source_input
        else:
            target_input = transformer_util.prepare_entities(target, self.add_entity_abstract, self.add_kg_info)
        candidates = self.candidates[prefix]
        model_input = [[source_input[source_id], target_input[target_id]] for source_id, target_id, _ in candidates]
        candidate_scores = self.model.predict(model_input, batch_size=self.batch_size, show_progress_bar=True)
        if self.scenario == MatchingScenario.MENTION_MENTION:
            # take all matches that are higher than threshold
            alignment = [(cand[0], cand[1], score) for cand, score in zip(candidates, candidate_scores) if score > self.confidence_threshold]
        else:  # MENTION_ENTITY
            # take only the most likely match for an item
            item_entity_scores = defaultdict(set)
            for (item_id, entity_id, _), score in zip(candidates, candidate_scores):
                if score > self.confidence_threshold:
                    item_entity_scores[item_id].add((entity_id, score))
            alignment = [(i, *max(js, key=lambda x: x[1])) for i, js in item_entity_scores.items()]
        return {Pair(source, target, score) for source, target, score in alignment}
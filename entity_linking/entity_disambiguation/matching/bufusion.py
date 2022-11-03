from typing import Set, Dict, Optional, Tuple, List, Union
from collections import defaultdict
import random
import itertools
import numpy as np
import utils
from impl.wikipedia.page_parser import MentionId
from entity_linking.entity_disambiguation.data import Pair, DataCorpus
from entity_linking.entity_disambiguation.matching.util import MatchingScenario
from entity_linking.entity_disambiguation.matching.crossencoder import CrossEncoderMatcher


class FusionCluster:
    idx: int
    mentions: Set[MentionId]
    candidates: Dict[MentionId, float]
    entity: Optional[int]

    def __init__(self, idx: int, mentions: Set[MentionId], candidates: Dict[MentionId, float], entity: Optional[int] = None):
        self.idx = idx
        self.mentions = mentions
        self.candidates = candidates
        self.entity = entity


class BottomUpFusionMatcher(CrossEncoderMatcher):
    def __init__(self, scenario: MatchingScenario, params: dict):
        super().__init__(scenario, params)
        # model params
        self.cluster_comparisons = params['cluster_comparisons']
        self.cluster_threshold = params['cluster_threshold']

    def _get_param_dict(self) -> dict:
        return super()._get_param_dict() | {'cc': self.cluster_comparisons, 'ct': self.cluster_threshold}

    def predict(self, eval_mode: str, data_corpus: DataCorpus) -> Set[Pair]:
        # prepare inputs
        mention_input, _ = data_corpus.get_mention_input(self.add_page_context, self.add_text_context)
        entity_input = data_corpus.get_entity_input(self.add_entity_abstract, self.add_kg_info)
        # retrieve entities for mentions and form clusters based on best entities; init remaining mentions as 1-clusters
        predictions = self._make_me_predictions(eval_mode, mention_input, entity_input)
        clusters, cluster_by_mid = self._init_clusters(eval_mode, predictions)
        # merge clusters
        iteration = 0
        while True:  # repeat as long as we have candidate matches
            iteration += 1
            if all(len(cluster.candidates) == 0 for cluster in clusters):
                break
            cluster_merge_candidates = set()
            for cluster in clusters:
                if cluster.entity is not None:
                    # discard candidates that are in a cluster with another entity
                    cluster.candidates = defaultdict(float, {cand: score for cand, score in cluster.candidates.items() if cluster_by_mid[cand].entity is None})
                if not cluster.candidates:
                    # ignore cluster without further candidates
                    continue
                # try merge with most promising candidate
                mention_merge_candidate = max(cluster.candidates.items(), key=lambda x: x[1])[0]
                cluster_merge_candidate = cluster_by_mid[mention_merge_candidate].idx
                cluster_merge_candidates.add(tuple(sorted([cluster.idx, cluster_merge_candidate])))
            # sample and evaluate MM/ME matches for cluster-pairs; aggregate by cluster
            cluster_by_id = {cluster.idx: cluster for cluster in clusters}
            mention_candidates, candidate_clusters = self._sample_candidates_for_cluster_merges(cluster_by_id, cluster_merge_candidates)
            mention_candidate_scores = self._compute_scores_for_mention_candidates(mention_candidates, mention_input, entity_input)
            cluster_merge_scores = self._compute_cluster_merge_scores(mention_candidate_scores, candidate_clusters)
            # merge clusters starting with highest left cluster index (in the case of a merge, we keep the left index)
            merge_conducted, merge_discarded = 0, 0
            for (cluster_a_id, cluster_b_id), score in sorted(cluster_merge_scores.items(), key=lambda x: x[0][0], reverse=True):
                cluster_a, cluster_b = cluster_by_id[cluster_a_id], cluster_by_id[cluster_b_id]
                if cluster_a == cluster_b:
                    continue
                if score > self.cluster_threshold:  # merge clusters and update indices
                    cluster_a.mentions |= cluster_b.mentions
                    merged_candidates = (set(cluster_a.candidates) | set(cluster_b.candidates)).difference(cluster_a.mentions)
                    cluster_a.candidates = defaultdict(float, {cand: max(cluster_a.candidates[cand], cluster_b.candidates[cand]) for cand in merged_candidates})
                    try:
                        clusters.remove(cluster_b)
                    except ValueError:
                        pass  # cluster already removed in another merge of this iteration
                    cluster_by_id[cluster_b_id] = cluster_a
                    for m_id in cluster_b.mentions:
                        cluster_by_mid[m_id] = cluster_a
                    merge_conducted += 1
                else:  # make sure clusters are not considered for merge again (by deleting candidates in other cluster)
                    cluster_a.candidates = defaultdict(float, {cand: score for cand, score in cluster_a.candidates.items() if cand not in cluster_b.mentions})
                    cluster_b.candidates = defaultdict(float, {cand: score for cand, score in cluster_b.candidates.items() if cand not in cluster_a.mentions})
                    merge_discarded += 1
            utils.get_logger().debug(f'BUF: Iteration {iteration}; Clusters: {len(clusters)}; Cluster Candidates: {len(cluster_merge_candidates)}; Mention Candidates: {len(mention_candidates)}; Merged: {merge_conducted}; Discarded: {merge_discarded}')
        # compute final alignment
        alignment = set()
        for cluster in clusters:
            alignment.update({Pair(m_id, cluster.entity, 1) for m_id in cluster.mentions})
            alignment.update({Pair(*sorted(item_pair), 1) for item_pair in itertools.combinations(cluster.mentions, 2)})
        return alignment

    def _make_me_predictions(self, eval_mode: str, mention_input: Dict[MentionId, str], entity_input: Dict[int, str]) -> Dict[MentionId, Dict[int, float]]:
        candidates = [cand[:2] for cand in self.me_candidates[eval_mode]]
        candidate_scores = self._compute_scores_for_mention_candidates(candidates, mention_input, entity_input)
        predictions = defaultdict(dict)
        for (mention_id, entity_id), score in zip(candidates, candidate_scores):
            predictions[mention_id][entity_id] = score
        return predictions

    def _compute_scores_for_mention_candidates(self, candidates: List[Tuple[MentionId, Union[MentionId, int]]], mention_input: Dict[MentionId, str], entity_input: Dict[int, str]) -> List[float]:
        model_input = []
        for cand in candidates:
            if isinstance(cand[1], MentionId):
                mention_a, mention_b = cand
                model_input.append([mention_input[mention_a], mention_input[mention_b]])
            else:
                mention_id, entity_id = cand
                model_input.append([mention_input[mention_id], entity_input[entity_id]])
        utils.release_gpu()
        return self.model.predict(model_input, batch_size=self.batch_size, show_progress_bar=True)

    def _init_clusters(self, eval_mode: str, predictions: Dict[MentionId, Dict[int, float]]) -> Tuple[List[FusionCluster], Dict[MentionId, FusionCluster]]:
        # group mentions by matching entity
        me_mapping = {m_id: max(ents.items(), key=lambda x: x[1]) for m_id, ents in predictions.items()}
        me_mapping = {m_id: ent_id for m_id, (ent_id, score) in me_mapping.items() if score > self.me_threshold}
        em_mapping = defaultdict(set)
        for m_id, e_id in me_mapping.items():
            em_mapping[e_id].add(m_id)
        # group candidates by mention
        mention_candidates = defaultdict(dict)
        for mention_a, mention_b, score in self.mm_candidates[eval_mode]:
            mention_candidates[mention_a][mention_b] = score
            mention_candidates[mention_b][mention_a] = score
        # form clusters with known entities
        clusters = []
        cluster_by_mid = {}
        cluster_id = 0
        for e_id, mention_ids in em_mapping.items():
            # collect candidates
            cluster_candidates = defaultdict(float)
            for m_id in mention_ids:
                for cand_id, score in mention_candidates[m_id].items():
                    if cand_id in mention_ids:
                        continue  # discard candidates in same cluster
                    cluster_candidates[cand_id] = max(cluster_candidates[cand_id], score)
            cluster = FusionCluster(cluster_id, mention_ids, cluster_candidates, e_id)
            clusters.append(cluster)
            for m_id in mention_ids:
                cluster_by_mid[m_id] = cluster
            cluster_id += 1
        # form clusters with remaining mentions
        for m_id, candidates in mention_candidates.items():
            if m_id in cluster_by_mid:
                continue
            cluster = FusionCluster(cluster_id, {m_id}, defaultdict(float, candidates))
            clusters.append(cluster)
            cluster_by_mid[m_id] = cluster
            cluster_id += 1
        return clusters, cluster_by_mid

    def _sample_candidates_for_cluster_merges(self, cluster_by_id: Dict[int, FusionCluster], cluster_merge_candidates: Set[Tuple[int, int]]) -> Tuple[List[Tuple[MentionId, Union[MentionId, int]]], List[Tuple[int, int]]]:
        mention_candidates = []
        candidate_clusters = []
        for cluster_a_id, cluster_b_id in cluster_merge_candidates:
            cluster_1, cluster_2 = cluster_by_id[cluster_a_id], cluster_by_id[cluster_b_id]
            if cluster_1.entity is not None:  # make sure that cluster with entity is always cluster_2
                cluster_1, cluster_2 = cluster_2, cluster_1
            cluster_1_candidates = random.sample(list(cluster_1.mentions), min(len(cluster_1.mentions), self.cluster_comparisons))
            cluster_2_candidates = [cluster_2.entity] if cluster_2.entity else []
            cluster_2_mentions_to_add = min(len(cluster_2.mentions), self.cluster_comparisons - len(cluster_2_candidates))
            cluster_2_candidates += random.sample(list(cluster_2.mentions), cluster_2_mentions_to_add)
            candidates = list(itertools.product(cluster_1_candidates, cluster_2_candidates))
            mention_candidates.extend(candidates)
            candidate_clusters.extend([(cluster_a_id, cluster_b_id)] * len(candidates))
        return mention_candidates, candidate_clusters

    def _compute_cluster_merge_scores(self, mention_candidate_scores: List[float], candidate_clusters: List[Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
        cluster_merge_scores = defaultdict(list)
        for cluster_merge_id, score in zip(candidate_clusters, mention_candidate_scores):
            cluster_merge_scores[cluster_merge_id].append(score)
        return {cluster_merge_id: np.mean(scores) for cluster_merge_id, scores in cluster_merge_scores.items()}
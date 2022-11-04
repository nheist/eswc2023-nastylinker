from typing import Dict, Set
from enum import Enum
import os
import pickle
import utils
from entity_linking.entity_disambiguation.data import Pair


class MatchingScenario(Enum):
    MENTION_MENTION = 'MM'
    MENTION_ENTITY = 'ME'
    FUSION = 'F'

    def is_MM(self) -> bool:
        return self in [self.MENTION_MENTION, self.FUSION]

    def is_ME(self) -> bool:
        return self in [self.MENTION_ENTITY, self.FUSION]


class MatchingApproach(Enum):
    EXACT = 'exact'
    WORD = 'word'
    BM25 = 'bm25'
    BIENCODER = 'biencoder'
    CROSSENCODER = 'crossencoder'
    POPULARITY = 'popularity'  # ME only!
    TOP_DOWN_FUSION = 'tdf'
    BOTTOM_UP_FUSION = 'buf'


def store_candidates(approach_name: str, candidates: Dict[str, Dict[MatchingScenario, Set[Pair]]]):
    utils.get_logger().debug(f'Storing candidates for matcher with name "{approach_name}"..')
    with open(get_model_path(approach_name) + '.p', mode='wb') as f:
        return pickle.dump(candidates, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_candidates(approach_id: str, scenario: MatchingScenario) -> Dict[str, Set[Pair]]:
    utils.get_logger().debug(f'Loading candidates from matcher with id "{approach_id}"..')
    with open(_get_approach_path_by_id(approach_id), mode='rb') as f:
        candidates = pickle.load(f)
        candidates = {eval_mode: candidates_by_scenario[scenario] for eval_mode, candidates_by_scenario in candidates.items()}
        return candidates


def get_model_path(approach_name: str) -> str:
    return os.path.join(utils._get_root_path(), 'entity_linking', 'data', approach_name)


def _get_approach_path_by_id(approach_id: str) -> str:
    data_dir = os.path.join(utils._get_root_path(), 'entity_linking', 'data')
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if os.path.isfile(filepath) and filename.startswith(approach_id):
            return filepath
    raise FileNotFoundError(f'Could not find file for approach with ID {approach_id}.')

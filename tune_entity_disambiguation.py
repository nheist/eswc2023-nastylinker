import os
os.environ['DISABLE_SPACY_CACHE'] = '1'  # make sure that spaCy cache is disabled
import configargparse
import random
import numpy as np
import pandas as pd
import torch
import itertools
import multiprocessing as mp
import utils
from tqdm import tqdm
from collections import namedtuple
from typing import Tuple
from impl.subject_entity.entity_disambiguation.matching import MatchingScenario, MatchingApproach, Matcher
from impl.subject_entity.entity_disambiguation.matching.io import get_cache_path
from impl.subject_entity.entity_disambiguation.data import CorpusType, DataCorpus, get_data_corpora
from impl.subject_entity.entity_disambiguation.matching.greedy_clustering import NastyLinker, EdinMatcher
from impl.subject_entity.entity_disambiguation.evaluation import MetricsCalculator, AlignmentComparison


VERSION = 1


AlignmentCorpus = namedtuple('AlignmentCorpus', ['alignment'])


def _eval_matcher(approach: MatchingApproach, corpus_type: CorpusType, sample_size: int, params: dict, num_processes: int):
    # load corpus and use only alignment for tuning
    _, _, data_corpus = get_data_corpora(corpus_type, sample_size)
    alignment_corpus = AlignmentCorpus(data_corpus.alignment)
    # compute results
    inputs = [(approach, alignment_corpus, tuple(params.keys()), param_config) for param_config in itertools.product(*params.values())]
    with mp.Pool(processes=num_processes) as pool:
        df = pd.DataFrame([res for res in pool.imap_unordered(_eval_param_config, tqdm(inputs))])
    # add macro-metrics
    df['P'] = (df['P_k'] + df['P_u']) / 2
    df['R'] = (df['R_k'] + df['R_u']) / 2
    df['F1'] = (df['F1_k'] + df['F1_u']) / 2
    df['C'] = df['C_k'] + df['C_u']
    df['eNMI'] = (df['eNMI_k'] + df['eNMI_u']) / 2
    df['mNMI'] = (df['mNMI_k'] + df['mNMI_u']) / 2
    # store
    df.to_csv(get_cache_path(f'hyperparams_{approach.name}_v{VERSION}.csv'), sep=';')


def _eval_param_config(args: Tuple[MatchingApproach, DataCorpus, tuple, tuple]):
    approach, data_corpus, param_names, param_config = args
    param_dict = dict(zip(param_names, param_config))
    matcher = _init_matcher(approach, param_dict)
    ca = matcher.predict(matcher.MODE_TEST, data_corpus)
    evaluator = MetricsCalculator('', True)
    alignment_comparison = AlignmentComparison(ca, data_corpus.alignment, True)
    metrics_known = evaluator._compute_metrics_for_partition(alignment_comparison, 0, False)
    metrics_unknown = evaluator._compute_metrics_for_partition(alignment_comparison, 0, True)
    return dict(zip(param_names, param_config)) | {
        'P_k': metrics_known['me-P'],
        'R_k': metrics_known['me-R'],
        'F1_k': metrics_known['me-F1'],
        'C_k': metrics_known['clusters'],
        'eNMI_k': metrics_known['eNMI'],
        'mNMI_k': metrics_known['mNMI'],
        'P_u': metrics_unknown['me-P'],
        'R_u': metrics_unknown['me-R'],
        'F1_u': metrics_unknown['me-F1'],
        'C_u': metrics_unknown['clusters'],
        'eNMI_u': metrics_unknown['eNMI'],
        'mNMI_u': metrics_unknown['mNMI'],
    }


def _init_matcher(approach: MatchingApproach, params: dict) -> Matcher:
    matcher_params = {'id': None} | params
    if approach == MatchingApproach.NASTY_LINKER:
        return NastyLinker(MatchingScenario.FULL, matcher_params)
    elif approach == MatchingApproach.EDIN:
        return EdinMatcher(MatchingScenario.FULL, matcher_params)
    raise ValueError(f'Unsupported approach: {approach.name}')


if __name__ == '__main__':
    parser = configargparse.ArgumentParser(description='Run a grid search over entity disambiguation approaches.')
    parser.add_argument('config_file', is_config_file=True, help='Path to hyperparameter config file')
    parser.add_argument('--approach', type=str, help='Approach used for matching')
    parser.add_argument('--corpus', type=str, choices=['LIST', 'NILK'], help='Data corpus to use for the experiments')
    parser.add_argument('-ss', '--sample_size', type=int, choices=list(range(5, 101, 5)), default=5, help='Percentage of dataset to use')
    parser.add_argument('--mm_approach', action='append', type=str, help='Mention-mention approach (ID) used for candidate generation')
    parser.add_argument('--me_approach', action='append', type=str, help='Mention-entity approach (ID) used for candidate generation')
    parser.add_argument('--mm_threshold', action='append', type=float, help="Confidence threshold to filter MM predictions.")
    parser.add_argument('--me_threshold', action='append', type=float, help="Confidence threshold to filter ME predictions.")
    parser.add_argument('--path_threshold', action='append', type=float, help="Confidence threshold to filter graph paths (NastyLinker only).")
    parser.add_argument('--me_cluster_threshold', action='append', type=float, help="Confidence threshold to filter graph paths (EDIN only).")
    parser.add_argument('--cpus', type=int, default=40, help='Number of CPUs to use')
    args = parser.parse_args()
    # fix all seeds
    SEED = 310
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # disable debug logging
    utils.get_logger().setLevel('INFO')
    # collect params
    param_names = ['mm_approach', 'me_approach', 'mm_threshold', 'me_threshold', 'path_threshold', 'me_cluster_threshold']
    params = {n: getattr(args, n) for n in param_names if getattr(args, n) is not None}
    # run tuning
    _eval_matcher(MatchingApproach(args.approach), CorpusType(args.corpus), args.sample_size, params, args.cpus)

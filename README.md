# ESWC 2023 - NASTyLinker: NIL-Aware Scalable Transformer-based Entity Linking

The code for running NASTyLinker has been integrated into the CaLiGraph extraction framework,
as this is the easiest way to access the LISTING dataset. The modifications implemented by the authors
can be found in `impl.subject_entity.entity_disambiguation` (contains datasets, matchers, evaluators).
In addition to that, the notebook `ESWC2023-NASTyLinker.ipynb` contains the code for generating the
figures of the paper and the results of the qualitative analysis.

For running the experiments, the authors provide the two scripts `evaluate_entity_disambiguation.py`
(for running single matchers like NASTyLinker) and `tune_entity_disambiguation.py` (for finding the best
hyperparameters for the clustering matchers). Results are logged under `logs/ED/` and can best be viewed
with Tensorboard (contained in dev-dependencies). Matchers results have to be created in the correct order
(i.e., when intending to run NASTyLinker, results from a Bi-Encoder and potentially Cross-Encoder need
to be generated first). The first run of the framework will take a considerable amount of time,
as caches for CaLiGraph are initialized and the LISTING dataset is prepared (see the original README
from CaLiGraph below for further instructions on how to setup the system).

## LISTING Extraction Results

The predictions for the LISTING dataset can be accessed [here](TODO). The file is in json format and contains a list of entities (which may be known or NIL) with the mentions that have been identified for them. The fields are as follows:

```
idx: Index of the entity
name: Name of the entity (None, if NIL-entity)
is_nil: Whether the entity is a NIL-entity
mentions: (list)
  page: Name of the page where the mention is found
  listing: Index of the listing where the mention is found
  item: Index of the item within the listing where the mention is found
  text: Mention text
```

## Configurations of the Paper Results

| *NILK*     |$\tau_m$|$\tau_e$|$\tau_a$|
|------------|--------|--------|--------|
|*No Reranking*|  -   |    -   |    -   |
|Bottom-Up   |   0.9  |  0.9   |    -   |
|Majority    |   0.85 |  0.8   |    -   |
|NASTyLinker |   0.8  |  0.85  |  0.75  |
|*Mention Reranking*|-|    -   |    -   |
|Bottom-Up   |   0.8  |  0.9   |    -   |
|Majority    |   0.85 |  0.8   |    -   |
|NASTyLinker |   0.8  |  0.8   |  0.75  |
|*Entity Reranking*|- |    -   |    -   |
|Bottom-Up   |   0.9  |  0.9   |    -   |
|Majority    |   0.85 |  0.75  |    -   |
|NASTyLinker |   0.8  |  0.9   |  0.75  |
|*Full Reranking*| -  |    -   |    -   |
|Bottom-Up   |  0.8   |  0.9   |    -   |
|Majority    |  0.85  |  0.75  |    -   |
|NASTyLinker |  0.8   |  0.9   |  0.75  |


| *LISTING*  |$\tau_m$|$\tau_e$|$\tau_a$|
|------------|--------|--------|--------|
|*No Reranking*|  -   |    -   |    -   |
|Bottom-Up   |  0.9   |  0.9   |    -   |
|Majority    |  0.85  |  0.8   |    -   |
|NASTyLinker |  0.8   |  0.85  |  0.75  |
|*Entity Reranking*|- |    -   |    -   |
|Bottom-Up   |  0.9   |  0.85  |    -   |
|Majority    |  0.9   |  0.8   |    -   |
|NASTyLinker |  0.9   |  0.85  |  0.8   |


# CaLiGraph
**A Large Semantic Knowledge Graph from Wikipedia Categories and Listings**

For information about the general idea, extraction statistics, and resources of CaLiGraph, visit the [CaLiGraph website](http://caligraph.org).

## Configuration
### System Requirements
- At least 300 GB of RAM as we load most of DBpedia in memory to speed up the extraction
- At least one GPU to run [transformers](https://huggingface.co/transformers/)
- During the first execution of an extraction you need a stable internet connection as the required DBpedia files are downloaded automatically 

### Prerequisites
- Environment manager: [conda](https://docs.continuum.io/anaconda/install/)
- Dependency manager: [poetry](https://python-poetry.org/docs/#installation)

### Setup
- In the project root, create a conda environment with: `conda env create -f environment.yaml`

- Activate the environment with `conda activate caligraph`

- Install dependencies with `poetry install`
- Install PyTorch for your specific cuda version with `poetry run poe autoinstall-torch-cuda`

- If you have not downloaded them already, you have to fetch the latest corpora for spaCy and nltk (run in terminal):
```
# download the most recent corpus of spaCy
python -m spacy download en_core_web_lg
# download wordnet & words corpora of nltk
python -c 'import nltk; nltk.download("wordnet"); nltk.download("words"); nltk.download("omw-1.4")'
```

### Basic Configuration Options

You can configure the application-specific parameters as well as logging- and file-related parameters in `config.yaml`. 

## Usage

Make sure that the virtual environment `caligraph` is activated. Then you can run the extraction in the project root folder with `python .`

All the required resources, like DBpedia files, will be downloaded automatically during execution.
CaLiGraph is serialized in N-Triple format. The resulting files are placed in the `results` folder.

### Evaluations
#### Subject Entity Detection

Use the script `evaluate_mention_detection.py` to evaluate a specific configuration for subject entity detection.

Make sure that there is a free GPU on your system and that the environment `caligraph` is activated. Then you can run an evaluation as follows:
```
python evaluate_mention_detection.py <GPU-ID> <HUGGINGFACE-MODEL> <OPTIONAL-CONFIG-PARAMS>
```
Have a look at the evaluation script for a description of the optional configuration parameters.

## Tests

In the project root, run tests with `pytest`

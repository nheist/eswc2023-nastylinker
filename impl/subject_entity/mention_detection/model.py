import utils
from datetime import datetime
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForTokenClassification, IntervalStrategy
from impl.util.transformer import SpecialToken
from impl.util.nlp import EntityTypeLabel
from impl.subject_entity.mention_detection.data import get_listpage_training_dataset, get_page_training_dataset


LISTPAGE_MODEL = 'transformer_for_listpage_mention_detection'
PAGE_MODEL = 'transformer_for_page_mention_detection'


def model_exists(model_id: str) -> bool:
    path_to_model = utils._get_cache_path(model_id)
    return path_to_model.is_dir()


def load_tokenizer_and_model(model_id: str):
    path_to_model = utils._get_cache_path(model_id)
    tokenizer = AutoTokenizer.from_pretrained(path_to_model)
    model = AutoModelForTokenClassification.from_pretrained(path_to_model, output_hidden_states=True)
    return tokenizer, model


def train_tokenizer_and_model(model_id: str, base_model_id: str = None):
    if base_model_id is None:
        base_model = utils.get_config('subject_entity.model_mention_detection')
        tokenizer = AutoTokenizer.from_pretrained(base_model, add_prefix_space=True, additional_special_tokens=list(SpecialToken.all_tokens()))
        model = AutoModelForTokenClassification.from_pretrained(base_model, num_labels=len(EntityTypeLabel))
        model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer, model = load_tokenizer_and_model(base_model_id)

    if model_id == LISTPAGE_MODEL:
        dataset = get_listpage_training_dataset(tokenizer)
        num_epochs = 2
    elif model_id == PAGE_MODEL:
        dataset = get_page_training_dataset(tokenizer)
        num_epochs = 1
    else:
        raise ValueError(f'Invalid value for model_id: {model_id}')

    run_id = '{}_{}'.format(datetime.now().strftime('%Y%m%d-%H%M%S'), utils.get_config('logging.filename'))
    training_args = TrainingArguments(
        save_strategy=IntervalStrategy.NO,
        output_dir=f'/tmp',
        logging_dir=f'./logs/transformers/MD_{run_id}',
        logging_steps=500,
        per_device_train_batch_size=64,
        num_train_epochs=num_epochs,
        learning_rate=5e-5,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    path_to_model = utils._get_cache_path(model_id)
    model.save_pretrained(path_to_model)
    tokenizer.save_pretrained(path_to_model)

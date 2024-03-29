from typing import Iterable
import re

def capitalize(text: str) -> str:
    return text[0].upper() + text[1:] if len(text) > 1 else text.upper()


def regularize_spaces(text: str) -> str:
    return ' '.join(text.split())


def alternate_iters_to_string(iter_a: Iterable, iter_b: Iterable) -> str:
    return ''.join(a + b for a, b in zip(iter_a, iter_b))


def transfer_word_casing(source_word: str, target_word: str) -> str:
    new_target_word = []
    for idx, c in enumerate(target_word):
        if idx >= len(source_word):
            new_target_word.append(c)
        elif source_word[idx].isupper():
            new_target_word.append(c.upper())
        else:
            new_target_word.append(c.lower())
    return ''.join(new_target_word)


def normalize_separators(text: str) -> str:
    return text.replace('-', '_')


def make_alphanumeric(text: str) -> str:
    text_alphanum = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return text_alphanum if len(text_alphanum) > 2 else text

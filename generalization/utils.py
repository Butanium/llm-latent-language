from transformers import StoppingCriteria
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES, MODEL_ALIASES
from transformer_lens import HookedTransformerKeyValueCache as KeyValueCache

from nnsight import LanguageModel
from nnsight.models.UnifiedTransformer import UnifiedTransformer
from display_utils import *  # todo: remove this legacy import


def save_yaml(content, path):
    with open(path, 'w') as f:
        yaml.dump(content, f, default_flow_style=False)


def load_yaml(path):
    with open(path, 'r') as f:
        content = yaml.safe_load(f)
    return content


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_tokens):
        """
        Args:
            stop_tokens (int or List[int]): The token(s) to stop generation at.
        """
        if isinstance(stop_tokens, int):
            stop_tokens = [stop_tokens]
        self.stop_tokens = stop_tokens

    def __call__(self, input_ids, _scores, **_kwargs):
        if input_ids[0][-1] in self.stop_tokens:
            return True  # Stop generation
        return False  # Continue generation

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

    def from_string(string, tokenizer):
        """
        Initialize the stop tokens as all the tokens that start or end with the given string.
        """
        stop_tokens = [
            i
            for i in range(tokenizer.vocab_size)
            if tokenizer.decode(i).startswith(string)
            or tokenizer.decode(i).endswith(string)
            or string in tokenizer.decode(i)
        ]
        return StopOnTokens(stop_tokens)


class StopOnSequence(StoppingCriteria):
    def __init__(self, stop_sequence):
        """
        Args:
            stop_sequence (List[int]): The sequence to stop generation at.
        """
        self.stop_sequence = stop_sequence
        self.state = 0

    def __call__(self, input_ids, _scores, **_kwargs):
        if input_ids[0][-1] == self.stop_sequence[self.state]:
            self.state += 1
            if self.state == len(self.stop_sequence):
                return True
        else:
            self.state = 0
        return False

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

    def from_string(string, tokenizer):
        """
        Initialize the stop tokens as all the tokens that start or end with the given string.
        """
        stop_sequence = [tokenizer(string, add_special_tokens=False)]
        return StopOnSequence(stop_sequence)


def add_model_to_transformer_lens(official_name, alias=None):
    """
    Hacky way to add a model to transformer_lens even if it's not in the official list.
    """
    if alias is None:
        alias = official_name
    if official_name not in OFFICIAL_MODEL_NAMES:
        OFFICIAL_MODEL_NAMES.append(official_name)
        MODEL_ALIASES[official_name] = [alias]
    else:
        print(f"Model {official_name} already in the official transformer lens models.")


def expend_tl_cache(cache: KeyValueCache, batch_size: int):
    """
    Expend the cache to the given batch size.
    """
    for entry in cache:
        entry.past_keys = entry.past_keys.expand(batch_size, *entry.past_keys.shape[1:])
        entry.past_values = entry.past_values.expand(
            batch_size, *entry.past_values.shape[1:]
        )
    cache.previous_attention_mask = cache.previous_attention_mask.expand(
        batch_size, *cache.previous_attention_mask.shape[1:]
    )
    return cache


def ulist(lst):
    """
    Returns a list with unique elements from the input list.
    """
    return list(dict.fromkeys(lst))

def lfilter(lst, f):
    """
    Returns a list with elements from the input list that satisfy the condition.
    """
    return list(filter(f, lst))


def get_tokenizer(model_or_tokenizer):
    """
    Returns the tokenizer of the given model or the given tokenizer.
    """
    if isinstance(model_or_tokenizer, LanguageModel) or isinstance(
        model_or_tokenizer, UnifiedTransformer
    ):
        return model_or_tokenizer.tokenizer
    return model_or_tokenizer


def str_or_list_to_list(s):
    """
    Returns a list of the given string or list.
    """
    if isinstance(s, str):
        return [s]
    return s

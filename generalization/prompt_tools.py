import pandas as pd
from random import sample
from typing import Optional, Callable
from dataclasses import dataclass
from utils import get_tokenizer, ulist
import torch as th
import re
import sys

sys.path.append("../src/")
from association_prompt import LoadAssociations

SPACE_TOKENS = ["▁", "Ġ", " "]


def token_prefixes(token_str: str):
    n = len(token_str)
    tokens = [token_str[:i] for i in range(1, n + 1)]
    return tokens


def add_spaces(tokens):
    return sum([[s + token for token in tokens] for s in SPACE_TOKENS], []) + tokens


# TODO?: Add capitalization


def byte_string_to_list(input_string):
    # Find all parts of the string: either substrings or hex codes
    parts = re.split(r"(\\x[0-9a-fA-F]{2})", input_string)
    result = []
    for part in parts:
        if re.match(r"\\x[0-9a-fA-F]{2}", part):
            # Convert hex code to integer
            result.append(int(part[2:], 16))
        else:
            if part:  # Skip empty strings
                result.append(part)
    return result


def unicode_prefixes(tok_str):
    encoded = str(tok_str.encode())[2:-1]
    if "\\x" not in encoded:
        return []  # No bytes in the string
    chr_list = byte_string_to_list(encoded)
    if isinstance(chr_list[0], int):
        first_byte_token = [
            f"<{hex(chr_list[0]).upper()}>"
        ]  # For llama2 like tokenizer, this is how bytes are represented
    else:
        first_byte_token = []
    # We need to convert back to latin1 to get the character
    for i, b in enumerate(chr_list):
        # those bytes are not valid latin1 characters and are shifted by 162 in Llama3 and Qwen
        if isinstance(b, str):
            continue
        if b >= 127 and b <= 160:
            chr_list[i] += 162
        chr_list[i] = chr(chr_list[i])
    # Convert back to string
    vocab_str = "".join(
        chr_list
    )  # This is the string that will be in the tokenizer vocab for Qwen and Llama3
    return first_byte_token + token_prefixes(vocab_str)


def process_tokens(words: str | list[str], tok_vocab):
    if isinstance(words, str):
        words = [words]
    final_tokens = []
    for word in words:
        with_prefixes = token_prefixes(word) + unicode_prefixes(word)
        with_spaces = add_spaces(with_prefixes)
        for word in with_spaces:
            if word in tok_vocab:
                final_tokens.append(tok_vocab[word])
    return ulist(final_tokens)


def process_tokens_with_tokenization(words: str | list[str], tokenizer):
    if isinstance(words, str):
        words = [words]
    final_tokens = []
    for word in words:
        # If you get the value error even with add_prefix_space=False,
        # you can use the following hacky code to get the token without the prefix
        # hacky_token = tokenizer("🍐", add_special_tokens=False).input_ids
        # length = len(hacky_token)
        # tokens = tokenizer("🍐" + word, add_special_tokens=False).input_ids
        # assert (
        #     tokens[:length] == hacky_token
        # ), "I didn't expect this to happen, please check this code"
        # if len(tokens) > length:
        #     final_tokens.append(tokens[length])

        token = tokenizer(word, add_special_tokens=False).input_ids[0]
        token_with_start_of_word = tokenizer(
            " " + word, add_special_tokens=False
        ).input_ids[0]
        if token == token_with_start_of_word:
            raise ValueError(
                "Seems like you use a tokenizer that wasn't initialized with add_prefix_space=False. Not good :("
            )
        final_tokens.append(token)
        if (
            token_with_start_of_word
            != tokenizer(" ", add_special_tokens=False).input_ids[0]
        ):
            final_tokens.append(token_with_start_of_word)
    return ulist(final_tokens)


@dataclass
class Prompt:
    prompt: str
    target_tokens: list[int]
    latent_tokens: dict[str, list[int]]
    target_strings: str
    latent_strings: dict[str, str | list[str]]

    @classmethod
    def from_strings(
        cls, prompt, target_strings, latent_strings, tokenizer, augment_token=False
    ):
        tok_vocab = tokenizer.get_vocab()
        process_toks = (
            (lambda s: process_tokens(s, tok_vocab))
            if augment_token
            else (lambda s: process_tokens_with_tokenization(s, tokenizer))
        )
        tokenizer = get_tokenizer(tokenizer)
        target_tokens = process_toks(target_strings)
        latent_tokens = {
            lang: process_toks(words) for lang, words in latent_strings.items()
        }
        return cls(
            target_tokens=target_tokens,
            latent_tokens=latent_tokens,
            target_strings=target_strings,
            latent_strings=latent_strings,
            prompt=prompt,
        )

    def get_target_probs(self, probs):
        target_probs = probs[:, :, self.target_tokens].sum(dim=2)
        return target_probs.cpu()

    def get_latent_probs(self, probs, layer=None):
        latent_probs = {
            lang: probs[:, :, tokens].sum(dim=2).cpu()
            for lang, tokens in self.latent_tokens.items()
        }
        if layer is not None:
            latent_probs = {
                lang: probs_[:, layer] for lang, probs_ in latent_probs.items()
            }
        return latent_probs

    @th.no_grad
    def run(self, nn_model, get_probs: Callable):
        """
        Run the prompt through the model and return the probabilities of the next token for both the target and latent languages.
        """
        probs = get_probs(nn_model, self.prompt)
        return self.get_target_probs(probs), self.get_latent_probs(probs)

    def has_no_collisions(self, ignore_langs: Optional[str | list[str]] = None):
        tokens = self.target_tokens[:]  # Copy the list
        if isinstance(ignore_langs, str):
            ignore_langs = [ignore_langs]
        if ignore_langs is None:
            ignore_langs = []
        for lang, lang_tokens in self.latent_tokens.items():
            if lang in ignore_langs:
                continue
            tokens += lang_tokens
        return len(tokens) == len(set(tokens))


lang2name = {
    "fr": "Français",
    "de": "Deutsch",
    "ru": "Русский",
    "en": "English",
    "zh": "中文",
    "es": "Español",
    "ko": "한국어",
    "ja": "日本語",
    "it": "Italiano",
    "nl": "Nederlands",
    "et": "Eesti",
    "fi": "Suomi",
    "hi": "हिन्दी",
    "A": "A",
    "B": "B",
}


def prompts_from_df(
    input_lang: str,
    target_lang: str,
    df: pd.DataFrame,
    n: int = 5,
    input_lang_name=None,
    target_lang_name=None,
    cut_at_obj=False,
):
    prompts = []
    pref_input = (
        input_lang_name if input_lang_name is not None else lang2name[input_lang]
    )
    pref_target = (
        target_lang_name if target_lang_name is not None else lang2name[target_lang]
    )
    if pref_input:
        pref_input += ": "
    if pref_target:
        pref_target += ": "
    for idx, row in df.iterrows():
        idxs = df.index.tolist()
        idxs.remove(idx)
        fs_idxs = sample(idxs, n)
        prompt = ""
        for fs_idx in fs_idxs:
            fs_row = df.loc[fs_idx]
            in_word = fs_row[input_lang]
            target_word = fs_row[target_lang]
            if isinstance(in_word, list):
                in_word = in_word[0]
            if isinstance(target_word, list):
                target_word = target_word[0]
            prompt += f'{pref_input}"{in_word}" - {pref_target}"{target_word}"\n'
        in_word = row[input_lang]
        if isinstance(in_word, list):
            in_word = in_word[0]
        prompt += f'{pref_input}"{in_word}'
        if not cut_at_obj:
            prompt += '" - {pref_target}"'
        prompts.append(prompt)
    return prompts


def translation_prompts(
    df,
    tokenizer,
    input_lang: str,
    target_lang: str,
    latent_langs: str | list[str] | None = None,
    n=5,
    only_best=False,
    augment_tokens=True,
    input_lang_name=None,
    target_lang_name=None,
    cut_at_obj=False,
) -> list[Prompt]:
    """
    Get a translation prompt from input_lang to target_lang for each row in the dataframe.

    Args:
        df: DataFrame containing translations
        tokenizer: Huggingface tokenizer
        input_lang: Language to translate from
        target_lang: Language to translate to
        n: Number of few-shot examples for each translation
        only_best: If True, only use the best translation for each row
        augment_tokens: If True, take the subwords, _word for each word

    Returns:
        List of Prompt objects
    """
    tok_vocab = tokenizer.get_vocab()
    if isinstance(latent_langs, str):
        latent_langs = [latent_langs]
    if latent_langs is None:
        latent_langs = []
    assert (
        len(df) > n
    ), f"Not enough translations from {input_lang} to {target_lang} for n={n}"
    prompts = []
    prompts_str = prompts_from_df(
        input_lang,
        target_lang,
        df,
        n=n,
        input_lang_name=input_lang_name,
        target_lang_name=target_lang_name,
        cut_at_obj=cut_at_obj,
    )
    for prompt, (_, row) in zip(prompts_str, df.iterrows()):
        target_words = row[target_lang]
        if only_best and isinstance(target_words, list):
            target_words = target_words[0]
        if augment_tokens:
            target_tokens = process_tokens(target_words, tok_vocab)
        else:
            target_tokens = process_tokens_with_tokenization(target_words, tokenizer)
        latent_tokens = {}
        latent_words = {}
        for lang in latent_langs:
            l_words = row[lang]
            if only_best and isinstance(l_words, list):
                l_words = l_words[0]
            latent_words[lang] = l_words
            if augment_tokens:
                latent_tokens[lang] = process_tokens(l_words, tok_vocab)
            else:
                latent_tokens[lang] = process_tokens_with_tokenization(
                    l_words, tokenizer
                )
        if len(target_tokens) and all(
            len(latent_tokens_) for latent_tokens_ in latent_tokens.values()
        ):
            prompts.append(
                Prompt(
                    prompt,
                    target_tokens,
                    latent_tokens,
                    target_words,
                    latent_words,
                )
            )
    return prompts


def cloze_prompts(df, tokenizer, lang, latent_langs=None, **kwargs):
    if latent_langs is None:
        latent_langs = []
    return translation_prompts(
        df,
        tokenizer,
        f"definitions_wo_ref_{lang}",
        f"senses_{lang}",
        [f"senses_{l}" for l in latent_langs],
        input_lang_name="",
        target_lang_name="",
        **kwargs,
    )


def color_prompts(tokenizer, lang, n=4):
    associations = LoadAssociations(lang)
    prompts = associations.generate_all_prompts(n)
    en_colors = associations.en_colors
    lang_colors = associations.other_colors
    color_prompts = []
    for prompt, target_string in prompts:
        latent_strings = {f"{color}_en": color for color in en_colors}
        lang_latent_strings = {
            f"{en_color}_{lang}": color
            for en_color, color in zip(en_colors, lang_colors)
        }
        latent_strings.update(lang_latent_strings)
        color_prompts.append(
            Prompt.from_strings(
                prompt, target_string, latent_strings, tokenizer, augment_token=True
            )
        )
    return color_prompts


def get_obj_id(sample_prompt, tokenizer):
    """
    For a prompt with the format '..."object" - X: "', return the index of the last token of the object.
    """
    split = sample_prompt.split('"')
    start = '"'.join(split[:-2])
    end = '"' + '"'.join(split[-2:])
    tok_start = tokenizer.encode(start, add_special_tokens=False)
    tok_end = tokenizer.encode(end, add_special_tokens=False)
    full = tokenizer.encode(sample_prompt, add_special_tokens=False)
    if tok_start + tok_end != full:
        raise ValueError("This is weird, check code")
    idx = -len(tok_end) - 1
    return idx


def lang_few_shot_prompts(
    df,
    tokenizer,
    langs,
    target_lang,
    latent_langs=None,
    lang_per_prompt=None,
    n_per_lang=1,
    num_prompts=200,
):
    if lang_per_prompt is None:
        lang_per_prompt = len(langs)
    if latent_langs is None:
        latent_langs = []
    prompts = []
    for _ in range(num_prompts):
        lang_sample = sample(langs, lang_per_prompt)
        concepts = df.sample(n_per_lang * lang_per_prompt)
        prompt = ""
        for i, lang in enumerate(lang_sample):
            for j in range(n_per_lang):
                row = concepts.iloc[i * n_per_lang + j]
                obj = row[f"senses_{lang}"][0]
                prompt += f"{obj}: {lang}\n"
        prompt += "_:"
        if len(tokenizer.encode("_:", add_special_tokens=False)) != 2:
            raise ValueError(
                "Weird tokenization going on, patchscope index might be wrong"
            )
        prompts.append(
            Prompt.from_strings(
                prompt, target_lang, {l: l for l in latent_langs}, tokenizer
            )
        )
    return prompts

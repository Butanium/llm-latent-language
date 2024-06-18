from __future__ import annotations

from typing import Optional
from exp_tools import (
    Prompt,
    load_lang,
    process_tokens,
    DATA_PATH,
    process_tokens_with_tokenization,
)
import pandas as pd
from warnings import warn
from random import sample
from tqdm.auto import tqdm
from munch import munchify
from cache_decorator import Cache
from collections import defaultdict
import ast
import regex as re
from emoji import emoji_count

from wrpy import WordReference
import babelnet as bn
from babelnet.sense import BabelLemmaType, BabelSense
from babelnet import BabelSynset
from utils import ulist
from babelnet.api import BabelAPIType, _api_type
from pathlib import Path


def BabelCache(**kwargs):
    def deco(func):
        cached_func = Cache(**kwargs)(func)

        def wrapper(*args, **kwargs):
            result = cached_func(*args, **kwargs)
            if result == []:
                res_type = list
            else:
                res_type = type(result[0]) if isinstance(result, list) else type(result)
            if "Online" in str(res_type) and _api_type == BabelAPIType.RPC:
                # If we are using the RPC API, we can delete the cache and save the offline data instead
                path = Path(Cache.compute_path(cached_func, *args, **kwargs))
                path.unlink()
            return cached_func(*args, **kwargs)

        return wrapper

    return deco


def get_gpt4_dataset(source_lang, target_langs, num_words=None):
    """
    Load the translations generated by GPT-4 for the given source and target languages.
    """
    langs = [source_lang] + target_langs
    dfs = [load_lang(lang) for lang in langs]
    dfs_dict = {f"{lang}": df for lang, df in zip(langs, dfs)}
    merged_df = pd.DataFrame(
        {
            name: df.set_index("word_original")["word_translation"]
            for name, df in dfs_dict.items()
        }
    )
    if "en" not in langs:
        merged_df = (
            merged_df.dropna(how="any")
            .reset_index()
            .rename(columns={"word_original": "en"})
        )
    else:
        merged_df = merged_df.dropna(how="any").reset_index(drop=True)
    print(f"Found {len(merged_df)} translations")
    if num_words is not None:
        merged_df = merged_df.sample(num_words)
    return merged_df


@Cache()
def cached_wr(source_lang, target_lang, word):
    return WordReference(source_lang, target_lang).translate(word)


def wr_translate(source_lang, target_lang, word, single_word=False, noun_only=True):
    response = munchify(cached_wr(source_lang, target_lang, word))
    translations = []
    # only take the principal and secondary translations
    if len(response.translations) == 0:
        return []
    entries = response.translations[0].entries
    if (
        len(response.translations) > 1
        and response.translations[1].title == "Additional Translations"
    ):
        entries.extend(response.translations[1].entries)
    for ent in entries:
        for trans in ent.to_word:
            trans = trans.meaning.replace("，", ", ")
            if noun_only and (
                ent.from_word.grammar == "" or ent.from_word.grammar[0] != "n"
            ):
                continue
            if target_lang != "zh":
                for w in trans.split(","):
                    w = w.strip()
                    if len(w.split(" ")) > 1 and single_word:
                        continue
                    if w.isnumeric():
                        continue
                    translations.append(w)
            else:
                if "TCTraditional" in trans:
                    split = trans.split("TCTraditional")
                    assert len(split) == 2
                    trans = split[0]
                translations.extend(re.findall(r"\p{Han}+", trans))

    translations = list(dict.fromkeys(translations))  # remove duplicates
    return translations


def get_wr_dataset(input_lang, output_langs, num_words=None):
    """
    Load translations from multiple languages and filter by token type if necessary
    """
    words = load_lang(input_lang)["word_translation"].values
    dic = {input_lang: words}
    for lang in output_langs:
        dic[lang] = [wr_translate(input_lang, lang, word) for word in tqdm(words)]
    df = pd.DataFrame(dic)
    df = df[df.map(lambda x: x != []).all(axis=1)]
    print(f"Found {len(df)} translations")
    if num_words is not None:
        df = df.sample(num_words)
    return df


id_to_bn_lang = {
    "en": bn.Language("English"),
    "es": bn.Language("Spanish"),
    "fr": bn.Language("French"),
    "de": bn.Language("German"),
    "it": bn.Language("Italian"),
    "ja": bn.Language("Japanese"),
    "ru": bn.Language("Russian"),
    "zh": bn.Language("Chinese"),
    "ko": bn.Language("Korean"),
    "nl": bn.Language("Dutch"),
    "et": bn.Language("Estonian"),
    "fi": bn.Language("Finnish"),
    "hi": bn.Language("Hindi"),
}
lang_to_id = {v: k for k, v in id_to_bn_lang.items()}


@BabelCache()
def cached_synset_from_id(synset_id, to_langs=None):
    id = bn.BabelSynsetID(synset_id)
    return bn.get_synset(
        id, to_langs=to_langs and [id_to_bn_lang[to_lang] for to_lang in to_langs]
    )


@BabelCache()
def cached_bn_synsets(word, from_langs, poses=None, to_langs=None):
    return bn.get_synsets(
        word,
        from_langs=[id_to_bn_lang[from_lang] for from_lang in from_langs],
        to_langs=[id_to_bn_lang[to_lang] for to_lang in to_langs],
        poses=poses,
    )


@BabelCache(
    args_to_ignore=("sense_filters", "synset_filters"),
)
def cached_bn_senses(
    word, from_langs, to_langs, sense_filters=None, synset_filters=None, poses=None
):
    return bn.get_senses(
        word,
        sense_filters=sense_filters,
        synset_filters=synset_filters,
        from_langs=[id_to_bn_lang[from_lang] for from_lang in from_langs],
        to_langs=[id_to_bn_lang[to_lang] for to_lang in to_langs],
        poses=poses,
    )


def get_synset_from_id(synset_id, to_langs=None):
    if to_langs is not None:
        to_langs = sorted(to_langs)
    return cached_synset_from_id(synset_id, to_langs=to_langs)


def get_bn_synsets(word, from_langs, poses=None, to_langs=None):
    if isinstance(from_langs, str):
        from_langs = [from_langs]
    from_langs = sorted(from_langs)
    if to_langs is None:
        to_langs = from_langs
    else:
        to_langs = sorted(to_langs)
    return cached_bn_synsets(word, from_langs, poses=poses, to_langs=to_langs)


def get_bn_senses(
    word, from_langs, to_langs, sense_filters=None, synset_filters=None, poses=None
):
    if isinstance(from_langs, str):
        from_langs = [from_langs]
    if isinstance(to_langs, str):
        to_langs = [to_langs]
    from_langs = sorted(from_langs)
    to_langs = sorted(to_langs)
    return cached_bn_senses(
        word,
        from_langs,
        to_langs,
        sense_filters=sense_filters,
        synset_filters=synset_filters,
        poses=poses,
    )


def _synset_filter(synset: BabelSynset):
    if not synset.is_key_concept:
        return False  # Removes albums, movies, etc.
    return True


def filter_synsets(synsets):
    return [synset for synset in synsets if _synset_filter(synset)]


def bn_translate(word, source_lang, target_langs, noun_only=True):
    def sense_filter(sense: BabelSense):
        lemma = sense.lemma.lemma
        if (
            sense._lemma.lemma_type
            != BabelLemmaType.HIGH_QUALITY  # Removes low-quality translations
            or lemma.lower() == word.lower()  # Removes the original word
            or emoji_count(lemma) > 0  # Removes emojis
            or any(char.isdigit() for char in lemma)  # Remove numbers
        ):
            return False  # Removes low-quality translations
        return True

    if isinstance(target_langs, str):
        target_langs = [target_langs]

    kwargs = dict(
        from_langs=[source_lang],
        to_langs=target_langs,
    )
    if noun_only:
        kwargs["poses"] = [bn.POS.NOUN]
    senses = get_bn_senses(word, **kwargs)
    senses = [sense for sense in senses if sense_filter(sense)]
    max_degree = max([sense.synset.synset_degree for sense in senses], default=0)
    best_senses = [
        sense for sense in senses if sense.synset.synset_degree == max_degree
    ]
    filtered_senses = [sense for sense in senses if _synset_filter(sense.synset)]
    if filtered_senses == []:
        warn(f"Didn't find any key concept for {word}")
    filtered_senses += best_senses
    translations = {lang: [] for lang in target_langs}
    for sense in senses:
        translations[lang_to_id[sense.language]].append(sense)
    for lang in target_langs:
        # Sorting like this puts the most common translations first :
        # "In practice, this connectivity measure weights a sense as more appropriate if it has a high degree"
        sort = sorted(
            translations[lang],
            key=lambda s: s.synset.synset_degree,
            reverse=True,
        )
        translations[lang] = ulist(
            [sense.lemma.lemma.replace("_", " ") for sense in sort]
        )
    return translations


def generate_bn_dataset(
    source_lang, target_langs, source_words=None, num_words=None, prune_empty=True
):
    """
    Load the translations generated by BabelNet for the given source and target languages.
    """
    original_df = None
    if source_words is None:
        original_df = load_lang(source_lang)
        source_words = list(original_df["word_translation"].values)
    if num_words is not None:
        source_words = sample(list(source_words), num_words)
    if isinstance(target_langs, str):
        target_langs = [target_langs]
    translations = defaultdict(list)
    for word in source_words:  # todo: tqdm
        try:
            tr = bn_translate(word, source_lang, target_langs)
            for lang in target_langs:
                translations[lang].append(tr[lang])
        except RuntimeError as e:
            print(f"Error with {word}: {e}")
            if "babelnet" not in str(e):
                raise e

    if source_lang in target_langs:
        for i, bn_source_words in enumerate(translations[source_lang]):
            if source_words[i] in bn_source_words:
                bn_source_words.remove(source_words[i])
            bn_source_words.insert(0, source_words[i])
    else:
        translations[source_lang] = source_words
    df = pd.DataFrame(translations)
    if original_df is not None:
        df["word_original"] = original_df["word_original"]
    if prune_empty:
        df = df[df.map(lambda x: x != []).all(axis=1)]
    print(f"Found {len(df)} translations")
    return df


def generate_bn_closure_dataset(lang, source_df):
    source_words = list(source_df["word_translation"].values)
    word_original = list(source_df["word_original"].values)

    def sense_filter(sense: BabelSense):
        keep_syn = _synset_filter(sense.synset)
        lemma = sense.lemma.lemma
        if (
            sense._lemma.lemma_type != BabelLemmaType.HIGH_QUALITY
            or emoji_count(lemma) > 0
            or any(char.isdigit() for char in lemma)  # Removes emojis
        ):  # Remove numbers
            return False
        return keep_syn

    dataset = {
        "word_original": [],
        "word": [],
        "senses": [],
        "closure": [],
        "definition": [],
    }
    for word, wo in zip(source_words, word_original):
        synsets = list(
            filter(
                _synset_filter,
                get_bn_synsets(word, from_langs=lang, poses=[bn.POS.NOUN]),
            )
        )
        synsets.sort(key=lambda s: s.synset_degree, reverse=True)
        regex = re.compile(r"\b" + word + r"\b", flags=re.IGNORECASE)
        for synset in synsets:
            closures = []
            defs = []
            glosses = synset.glosses()
            senses = ulist(
                [
                    str(s.lemma).replace("_", " ")
                    for s in synset.senses()
                    if sense_filter(s)
                ]
            )
            for gloss in glosses:
                gloss = str(gloss)
                sub_gloss = re.sub(regex, "_", gloss)
                if sub_gloss != gloss:
                    closures.append(sub_gloss)
                defs.append(gloss)
            if closures:
                break
        if closures or defs:
            dataset["word_original"].append(wo)
            dataset[f"word_original_{lang}"].append(word)
            senses = ulist(
                [
                    str(s.lemma).replace("_", " ")
                    for s in synset.senses()
                    if sense_filter(s)
                ]
            )
            dataset["senses"].append(senses)
            dataset["closure"].append(closures)
            dataset["definition"].append(defs)
    return pd.DataFrame(dataset)


def get_bn_dataset(
    source_lang: str,
    target_langs: str | list[str],
    num_words: Optional[int] = None,
    do_sample=True,
):
    if isinstance(target_langs, str):
        target_langs = [target_langs]
    df = pd.read_csv(DATA_PATH / "langs" / source_lang / "babelnet.csv")
    if num_words is not None:
        if do_sample:
            df = df.sample(num_words)
        else:
            df = df.head(num_words)

    def map_entry(entry):
        return ast.literal_eval(entry)

    out_df = pd.DataFrame()
    for lang in target_langs:
        out_df[lang] = df[lang].map(map_entry)
    out_df[source_lang] = df[source_lang].map(map_entry)
    assert (
        out_df.map(lambda x: x != []).all(axis=1)
    ).all(), "Some translations are empty"
    out_df["word_original"] = df["word_original"]
    return out_df


def filter_translations(
    translation_df, tok_vocab=None, multi_token_only=False, single_token_only=False
):
    assert not (
        multi_token_only and single_token_only
    ), "Cannot have both multi_token_only and single_token_only"
    assert tok_vocab is not None or not (
        multi_token_only or single_token_only
    ), "Cannot filter tokens without a tokenizer"
    if not multi_token_only and not single_token_only:
        return translation_df
    for idx, row in translation_df.iterrows():
        for lang in translation_df.columns:
            if row[lang] in tok_vocab or "▁" + row[lang] in tok_vocab:
                if multi_token_only:
                    translation_df.drop(idx, inplace=True)
                    break
            elif single_token_only:
                translation_df.drop(idx, inplace=True)
                break
    print(f"Filtered to {len(translation_df)} translations")
    return translation_df


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


def prompts_from_df(input_lang: str, target_lang: str, df: pd.DataFrame, n: int = 5):
    prompts = []
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
            prompt += f'{lang2name[input_lang]}: "{in_word}" - {lang2name[target_lang]}: "{target_word}"\n'
        in_word = row[input_lang]
        if isinstance(in_word, list):
            in_word = in_word[0]
        prompt += f'{lang2name[input_lang]}: "{in_word}" - {lang2name[target_lang]}: "'
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
    prompts_str = prompts_from_df(input_lang, target_lang, df, n=n)
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

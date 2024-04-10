from __future__ import annotations
from exp_tools import Prompt, load_lang, process_tokens
import pandas as pd
from random import sample
from tqdm.auto import tqdm
from utils import convert_to_munch
from cache_decorator import Cache




def get_gpt4_dataset(source_lang, target_langs):
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


from wrpy import WordReference

from generalization.exp_tools import Prompt, process_tokens


@Cache()
def cached_wr(source_lang, target_lang, word):
    return WordReference(source_lang, target_lang).translate(word)


import regex


def wr_translate(source_lang, target_lang, word, single_word=False, noun_only=True):
    response = convert_to_munch(cached_wr(source_lang, target_lang, word))
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
                translations.extend(regex.findall(r"\p{Han}+", trans))

    translations = list(dict.fromkeys(translations))  # remove duplicates
    return translations


def get_wr_dataset(input_lang, output_langs):
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
    return df


import babelnet as bn
from babelnet.sense import BabelLemmaType, BabelSense
from babelnet import BabelSynset

id_to_lang = {
    "en": bn.Language("English"),
    "es": bn.Language("Spanish"),
    "fr": bn.Language("French"),
    "de": bn.Language("German"),
    "it": bn.Language("Italian"),
    "ja": bn.Language("Japanese"),
    "ru": bn.Language("Russian"),
    "zh": bn.Language("Simplified Chinese"),
    "ko": bn.Language("Korean"),
}
lang_to_id = {v: k for k, v in id_to_lang.items()}


@Cache()
def bn_translate(source_lang, target_langs, word, noun_only=True):
    bn_source_lang = id_to_lang[source_lang]
    bn_target_langs = [id_to_lang[lang] for lang in target_langs]

    def sense_filter(sense: BabelSense):
        if sense._lemma.lemma_type != BabelLemmaType.HIGH_QUALITY:
            return False  # Removes low-quality translations
        return True

    def synset_filter(synset: BabelSynset):
        if not synset.is_key_concept:
            return False  # Removes albums, movies, etc.
        return True

    kwargs = dict(
        from_langs=[bn_source_lang],
        to_langs=bn_target_langs,
        sense_filters=[sense_filter],
        synset_filters=[synset_filter],
    )
    if noun_only:
        kwargs["poses"] = [bn.POS.NOUN]
    senses = bn.get_senses(word, **kwargs)
    translations = {lang: [] for lang in target_langs}
    for sense in senses:
        translations[lang_to_id[sense.language]].append(sense.lemma)
    for lang in target_langs:
        translations[lang] = sorted(list(dict.fromkeys(translations[lang])), key=lambda s : s.synset.synset_degree, reverse=True)
        # Sorting like this puts the most common translations first : 
        # "In practice, this connectivity measure weights a sense as more appropriate if it has a high degree"
    return translations


def get_bn_dataset(source_lang, target_langs, extend_source=False):
    """
    Load the translations generated by BabelNet for the given source and target languages.

    @param extend_source: If True, add synonyms of the source words to the dataset
    """
    source_words = load_lang(source_lang)["word_translation"].values
    translations = {source_lang: source_words}
    for lang in target_langs:
        translations[lang] = [
            bn_translate(source_lang, [lang], word) for word in tqdm(source_words)
        ]
    if extend_source:
        translations[source_lang] = [
            bn_translate(source_lang, [], word) for word in tqdm(source_words)
        ]
        for i, source_words in enumerate(translations[source_lang]):
            if source_words[i] in translations[source_lang][i]:
                translations[source_lang][i].remove(source_words[i])
            translations[source_lang][i].insert(0, source_words[i])
    df = pd.DataFrame(translations)
    df = df[df.map(lambda x: x != []).all(axis=1)]
    print(f"Found {len(df)} translations")
    return df

        


def filter_translations(
    translation_df, tokenizer=None, multi_token_only=False, single_token_only=False
):
    assert not (
        multi_token_only and single_token_only
    ), "Cannot have both multi_token_only and single_token_only"
    assert tokenizer is not None or not (
        multi_token_only or single_token_only
    ), "Cannot filter tokens without a tokenizer"
    if not multi_token_only and not single_token_only:
        return translation_df
    for idx, row in translation_df.iterrows():
        for lang in translation_df.columns:
            if (
                row[lang] in tokenizer.get_vocab()
                or "▁" + row[lang] in tokenizer.get_vocab()
            ):
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
}


def translation_prompts(
    df,
    tokenizer,
    input_lang,
    target_lang,
    latent_langs: str | list[str],
    n=5,
    only_best=False,
):
    """
    Get a translation prompt from input_lang to target_lang for each row in the dataframe.

    Args:
        df: DataFrame containing translations
        tokenizer: Huggingface tokenizer
        input_lang: Language to translate from
        target_lang: Language to translate to
        n: Number of few-shot examples for each translation
        only_best: If True, only use the best translation for each row

    Returns:
        List of Prompt objects
    """
    if isinstance(latent_langs, str):
        latent_langs = [latent_langs]
    assert (
        len(df) > n
    ), f"Not enough translations from {input_lang} to {target_lang} for n={n}"
    prompts = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        idxs = df.index.tolist()
        idxs.remove(idx)
        fs_idxs = sample(idxs, n)
        prompt = ""
        for fs_idx in fs_idxs:
            fs_row = df.loc[fs_idx]
            in_word = fs_row[input_lang]
            target_word = fs_row[target_lang]
            if isinstance(target_word, list):
                target_word = target_word[0]
            prompt += f'{lang2name[input_lang]}: "{in_word}" - {lang2name[target_lang]}: "{target_word}"\n'
        in_word = row[input_lang]
        prompt += f'{lang2name[input_lang]}: "{in_word}" - {lang2name[target_lang]}: "'
        target_words = row[target_lang]
        if only_best and isinstance(target_words, list):
            target_words = target_words[0]
        target_tokens = process_tokens(target_words, tokenizer, lang=target_lang)
        latent_tokens = {}
        latent_words = {}
        for lang in latent_langs:
            l_words = row[lang]
            if only_best and isinstance(l_words, list):
                l_words = l_words[0]
            latent_words[lang] = l_words
            latent_tokens[lang] = process_tokens(l_words, tokenizer, lang=lang)
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
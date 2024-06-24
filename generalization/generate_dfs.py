import sys

sys.path.append("..")
import json


from concurrent.futures import ThreadPoolExecutor
from random import sample
from collections import defaultdict
from babelnet.sense import BabelSense, BabelLemmaType
from translation_tools import (
    load_lang,
    _synset_filter,
    get_bn_synsets,
    filter_synsets,
    filter_senses,
    bn_translate,
    id_to_bn_lang,
    get_synset_from_id,
    DATA_PATH,
    prompts_from_df,
)
from emoji import emoji_count
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
import babelnet as bn
from utils import ulist
import re


def generate_bn_dataset(
    source_lang,
    target_langs,
    source_words=None,
    num_words=None,
    prune_empty=True,
    keep_original_word=False,
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
            tr = bn_translate(
                word, source_lang, target_langs, keep_original_word=keep_original_word
            )
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


def generate_synset_dataset(lang, source_df):
    synset_ids = list(source_df["synset"].values)
    word_original = list(source_df["word_original"].values)

    def sense_filter(sense: BabelSense):
        lemma = sense.lemma.lemma
        if (
            sense._lemma.lemma_type != BabelLemmaType.HIGH_QUALITY
            or emoji_count(lemma) > 0
            or any(char.isdigit() for char in lemma)  # Removes emojis
        ):  # Remove numbers
            return False
        return True

    dataset = {
        "word_original": [],
        "synset": [],
        "senses": [],
        "definitions": [],
    }
    for id, wo in zip(synset_ids, word_original):
        synset = get_synset_from_id(id, to_langs=lang)
        senses = ulist(
            [str(s.lemma).replace("_", " ") for s in synset.senses() if sense_filter(s)]
        )
        defs = [str(g) for g in synset.glosses()]
        if defs and senses:
            dataset["word_original"].append(wo)
            dataset["synset"].append(id)
            dataset["senses"].append(senses)
            dataset["definitions"].append(defs)
    df_ = pd.DataFrame(dataset)
    df = df_.drop_duplicates(subset=["synset"])
    if len(df_) != len(df):
        print(f"Found {len(df_) - len(df)} words associated with the same synset")
    return df


def generate_bn_cloze_dataset(lang, placeholder="___"):
    synset_df = pd.read_csv(DATA_PATH / f"langs/{lang}/synset_dataset.csv")
    all_clozes = []
    all_clozes_sow = []
    all_definitions_wo_ref = []
    for i, row in synset_df.iterrows():
        glosses = eval(row["definitions"])
        glosses = sorted(glosses, key=len, reverse=True)
        senses = eval(row["senses"])
        senses = sorted(senses, key=len, reverse=True)
        clozes = []
        clozes_sow = []
        definitions_wo_ref = []
        for gloss in glosses:
            found = False
            for main_sense in senses:
                cloze = gloss
                cloze_sow = gloss
                acceptable_senses = [main_sense]
                acceptable_senses_sow = [main_sense]
                regex_sow = re.compile(r"\b" + main_sense, flags=re.IGNORECASE)
                if re.search(regex_sow, cloze_sow):
                    cloze_sow = re.sub(regex_sow, placeholder, cloze_sow)
                    cloze = re.sub(
                        r"\b" + main_sense + r"\b",
                        placeholder,
                        cloze,
                        flags=re.IGNORECASE,
                    )
                    found = True
                else:
                    continue
                for sense in senses:
                    if sense == main_sense:
                        continue
                    if not re.search(r"\b" + sense + r"\b", gloss, flags=re.IGNORECASE):
                        acceptable_senses.append(sense)
                    if not re.search(r"\b" + sense, gloss, flags=re.IGNORECASE):
                        acceptable_senses_sow.append(sense)
                if cloze != gloss:
                    clozes.append((cloze, tuple(acceptable_senses)))
                clozes_sow.append((cloze_sow, tuple(acceptable_senses_sow)))
            if not found:
                definitions_wo_ref.append(gloss)
        all_clozes.append(ulist(clozes))
        all_clozes_sow.append(ulist(clozes_sow))
        all_definitions_wo_ref.append(ulist(definitions_wo_ref))
    synset_df["clozes"] = all_clozes
    synset_df["clozes_with_start_of_word"] = all_clozes_sow
    synset_df = synset_df.rename(columns={"definitions": "original_definitions"})
    synset_df["definitions_wo_ref"] = all_definitions_wo_ref
    return synset_df


def build_bn_dataset(input_lang, out_langs, expand=False, use_tqdm=False):
    """
    Patchscope with source hidden from:
    index -1 and Prompt = source_input_lang: A -> source_target_lang:
    Into target prompt:
    into index = -1, prompt = input_lang: A -> target_lang:
    Then plot with latent_langs, target_lang, source_target_lang
    """
    print(f"{input_lang} -> {out_langs}")
    df = generate_bn_dataset(input_lang, out_langs)
    print(f"{input_lang} -> {out_langs}: Got {len(df)} translations")
    # save it
    if expand:
        original_df = pd.read_csv(f"../data/langs/{input_lang}/babelnet.csv")
        # merge the two using word_original as the key
        cols_to_use = df.columns.difference(original_df.columns)
        cols_to_use = cols_to_use.insert(0, "word_original")
        df = df[cols_to_use]
        df = pd.merge(original_df, df, on="word_original", how="inner")
        df.to_csv(DATA_PATH / f"langs/{input_lang}/babelnet.csv", index=False)
    else:
        df.to_csv(DATA_PATH / f"langs/{input_lang}/babelnet.csv", index=False)
    for target_lang in out_langs:
        if target_lang == input_lang:
            continue
        prompts: list = prompts_from_df(
            input_lang,
            target_lang,
            df,
        )
        json_dic = {}
        wrapper = tqdm if use_tqdm else lambda x: x
        for prompt, (_, row) in wrapper(zip(prompts, df.iterrows())):
            json_dic[str(row[input_lang])] = {
                "prompt": prompt,
                "target": str(row[target_lang]),
                "word original": str(row["word_original"]),
            }
        with open(
            DATA_PATH / f"langs/{input_lang}/{target_lang}_prompts.json", "w"
        ) as f:
            json.dump(json_dic, f, indent=4)
    print(f"Done {input_lang}")


def build_synset_dataset(words, lang, use_tqdm=False):
    """
    Create a dataset of synset from a list of words
    """
    df = {"word_original": [], "synset": [], "glosses": []}
    wrapper = tqdm if use_tqdm else lambda x: x
    for word in wrapper(words):
        synsets = filter_synsets(
            get_bn_synsets(word, from_langs=lang, poses=[bn.POS.NOUN])
        )
        key_synsets = filter_synsets(synsets, key_concept_only=True)
        if not synsets:
            print(f"No synsets for {word}")
            continue
        if not key_synsets:
            key_synsets = synsets
        synset = max(key_synsets, key=lambda s: s.synset_degree)
        df["word_original"].append(word)
        df["synset"].append(synset.id)
        df["glosses"].append([str(g) for g in synset.glosses(id_to_bn_lang["en"])])
    return pd.DataFrame(df)


def main_translation_dataset(args):
    parser = ArgumentParser()
    langs = ["fr", "de", "ru", "en", "zh", "es"]
    out_langs = langs + ["ja", "ko", "et", "fi", "nl", "hi", "it"]
    parser.add_argument("--in-langs", "-i", nargs="+", default=langs)
    parser.add_argument("--out-langs", "-o", nargs="+", default=out_langs)
    parser.add_argument("--expand", "-e", action="store_true")
    parser.add_argument("--tqdm", "-t", action="store_true")
    args = parser.parse_args(args)
    langs = args.in_langs
    out_langs = args.out_langs
    expand = args.expand

    def process_item(input_lang):
        build_bn_dataset(input_lang, out_langs, expand, use_tqdm=args.tqdm)

    with ThreadPoolExecutor(max_workers=30) as executor:
        results = list(executor.map(process_item, langs))


def main_synset_dataset(args):
    parser = ArgumentParser()
    langs = ["fr", "de", "ru", "en", "zh", "es"] + [
        "ja",
        "ko",
        "et",
        "fi",
        "nl",
        "hi",
        "it",
    ]
    parser.add_argument("--langs", "-l", nargs="+", default=langs)
    args = parser.parse_args(args)

    def process_item(lang):
        synsets_df = pd.read_csv(DATA_PATH / f"basic_english_synset.csv")
        df = generate_synset_dataset(
            lang,
            synsets_df,
        )
        path = DATA_PATH / "langs" / lang / "synset_dataset.csv"
        path.parent.mkdir(exist_ok=True)
        df.to_csv(path, index=False)

    with ThreadPoolExecutor(max_workers=30) as executor:
        _ = list(executor.map(process_item, args.langs))


def main_cloze_dataset(args):
    parser = ArgumentParser()
    langs = ["fr", "de", "ru", "en", "zh", "es"] + [
        "ja",
        "ko",
        "et",
        "fi",
        "nl",
        "hi",
        "it",
    ]
    parser.add_argument("--langs", "-l", nargs="+", default=langs)
    args = parser.parse_args(args)

    def process_item(lang):
        print(f"Starting {lang}")
        df = generate_bn_cloze_dataset(lang)
        path = DATA_PATH / "langs" / lang / "cloze_dataset.csv"
        path.parent.mkdir(exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Done {lang}")

    with ThreadPoolExecutor(max_workers=30) as executor:
        _ = list(executor.map(process_item, args.langs))


def main_synset_dataset_base():
    build_synset_dataset(
        pd.read_csv(DATA_PATH / "basic_english_picturable_words.csv")["word"], "en"
    ).to_csv(DATA_PATH / "synset_dataset.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--translation", action="store_true", help="Generate translation dataset"
    )
    group.add_argument("--synset", action="store_true", help="Generate synset dataset")
    group.add_argument("--cloze", action="store_true", help="Generate cloze dataset")
    args, unknown = parser.parse_known_args()

    if args.translation:
        main_translation_dataset(unknown)
    elif args.synset:
        main_synset_dataset(unknown)
    else:
        main_cloze_dataset(unknown)

    print("Done")

import os
import sys

sys.path.append("..")
import torch as th
import numpy as np
import pandas as pd
import json
from tqdm.auto import tqdm
from pathlib import Path
import itertools

langs = ["fr", "de", "ru", "en", "zh", "es"]
out_langs = langs + ["ja", "ko", "et", "fi", "nl", "hi"]
from translation_tools import prompts_from_df

from translation_tools import get_bn_dataset as get_translations


def build_bn_dataset(input_lang):
    """
    Patchscope with source hidden from:
    index -1 and Prompt = source_input_lang: A -> source_target_lang:
    Into target prompt:
    into index = -1, prompt = input_lang: A -> target_lang:
    Then plot with latent_langs, target_lang, source_target_lang
    """
    print(f"{input_lang} -> {out_langs}")
    df = get_translations(input_lang, out_langs)
    print(f"{input_lang} -> {out_langs}: Got {len(df)} translations")
    # save it
    df.to_csv(f"../data/langs/{input_lang}/babelnet.csv", index=False)
    for target_lang in out_langs:
        if target_lang == input_lang:
            continue
        prompts: list = prompts_from_df(
            input_lang,
            target_lang,
            df,
        )
        json_dic = {}
        for prompt, (_, row) in zip(prompts, df.iterrows()):
            json_dic[str(row[input_lang])] = prompt
        with open(f"../data/langs/{input_lang}/{target_lang}_prompts.json", "w") as f:
            json.dump(json_dic, f, indent=4)
    print(f"Done {input_lang}")


from concurrent.futures import ThreadPoolExecutor


def process_item(input_lang):
    build_bn_dataset(input_lang)


with ThreadPoolExecutor(max_workers=30) as executor:
    results = list(executor.map(process_item, langs))

print("Done")

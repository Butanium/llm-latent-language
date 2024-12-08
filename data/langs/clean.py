import os
import json


def format_json_file(filepath):
    with open(filepath, "r+") as f:
        data = json.load(f)
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()


def fromat_json():
    for dirName, subdirList, fileList in os.walk("."):
        for fname in fileList:
            if fname.endswith(".json"):
                format_json_file(os.path.join(dirName, fname))


import pandas as pd
import json


def add_word_orignal_column():
    langs = ["de", "en", "es", "fr", "ru", "zh"]
    for lang in langs:
        or_df = pd.read_csv(f"{lang}/clean.csv")
        bn_df = pd.read_csv(f"{lang}/babelnet.csv")
        word_original = []
        for words in bn_df[lang]:
            # look for the row in or_df such that it's word_translation is contained in words
            # and get the word_original
            words = json.loads(words.replace("'", '"'))
            word = or_df[or_df["word_translation"].isin(words)]["word_original"].values
            assert (
                len(word) == 1
            ), f"Expected 1 word, got {len(word)} for {words} in {lang}. {word}"
            word_original.append(word[0])


from pathlib import Path


def map_file_in_all_langs(file_name, func):
    # get all langs directories
    dirs = [d for d in Path(".").iterdir() if d.is_dir()]
    for dir in dirs:
        file_path = dir / file_name
        if file_path.exists():
            func(file_path)
        else:
            print(f"{file_path} does not exist")


def remove_file_in_all_langs(file_name):
    map_file_in_all_langs(file_name, lambda file_path: file_path.unlink())

def edit_cloze():
    def fun(file_path):
        df = pd.read_csv(file_path)
        df = df[["word_original", "synset", "senses", "definitions"]]
        df.to_csv(file_path.parent / "synset_dataset.csv", index=False)
        # file_path.unlink()
    map_file_in_all_langs("cloze_dataset.csv", fun)

def remove_duplicates(file, subset=None):
    def fun(file_path):
        df = pd.read_csv(file_path)
        df2 = df.drop_duplicates(subset=subset)
        print(f"Removed {len(df) - len(df2)} duplicates from {file_path}")
        df2.to_csv(file_path, index=False)
    map_file_in_all_langs(file, fun)

if __name__ == "__main__":
    # fromat_json()
    # add_word_orignal_column()
    # remove_file_in_all_langs("closure_dataset.csv")
    # edit_cloze()
    # remove_file_in_all_langs("cloze_dataset.csv")
    remove_duplicates("cloze_dataset.csv", subset=["synset"])
    remove_duplicates("synset_dataset.csv", subset=["synset"])
    
